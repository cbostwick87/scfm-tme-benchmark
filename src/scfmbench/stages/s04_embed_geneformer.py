"""Zero-shot Geneformer cell embeddings, chunked and resumable per shard.

Design constraints, all measured rather than assumed:
  * HOST RAM (15 GiB) is the binding constraint, not VRAM. Cells are streamed
    from backed h5ad in shards; no dataset matrix is ever held whole.
  * fp16 autocast only. bf16 is REPORTED as supported on this Turing GPU and
    runs without error, but is emulated at 2.45 vs 26.3 TFLOP/s -- a
    capability-flag check would silently select the 10x slower path.
  * Each shard is written as its own .npz, so an interrupted run resumes from
    the last completed shard. Every cell is embedded ONCE per model.
  * The encoding is verified before any shard is embedded; the stage refuses to
    run if verification fails.

Cell embedding = mean of final hidden states over non-padding tokens. The BERT
pooler is NOT used: this checkpoint's pooler weights are newly initialised.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import pathlib
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch

from scfmbench import config, provenance
from scfmbench.models.geneformer_embed import GeneformerEncoder, verify_encoding


def find_snapshot(root: str) -> tuple[str, str]:
    """Locate the pinned checkpoint and dictionary directories."""
    ck = sorted(glob.glob(f"{root}/models--ctheodoris--Geneformer/snapshots/*/Geneformer-V2-104M"))
    dd = sorted(glob.glob(f"{root}/models--ctheodoris--Geneformer/snapshots/*/geneformer"))
    if not ck or not dd:
        raise FileNotFoundError(
            f"Geneformer checkpoint not found under {root}. Run the model-fetch step first."
        )
    return ck[0], dd[0]


def _csr_row_block(h5file, lo: int, hi: int, n_var: int) -> sp.csr_matrix:
    """Read rows [lo, hi) of a CSR-encoded h5ad /X group with h5py directly.

    anndata 0.10.8's backed sparse slicing is broken against the scipy in this
    environment (`backed_csr_matrix` has no `_validate_indices`), so the row block
    is assembled from the raw CSR arrays instead. This is not a workaround for
    convenience: it is also strictly leaner, reading only the indptr window and the
    exact data slice rather than instantiating a backed matrix object per shard.
    """
    g = h5file["X"]
    indptr = g["indptr"][lo:hi + 1]
    start, end = int(indptr[0]), int(indptr[-1])
    data = g["data"][start:end]
    indices = g["indices"][start:end]
    return sp.csr_matrix((data, indices, indptr - indptr[0]),
                         shape=(hi - lo, n_var))


def embed_dataset(name: str, h5: pathlib.Path, out_dir: pathlib.Path,
                  enc: GeneformerEncoder, model, device: str,
                  shard_cells: int, batch_tokens: int, max_len: int) -> dict:
    """Embed one dataset in shards. Returns per-dataset stats."""
    import h5py
    a = ad.read_h5ad(h5, backed="r")
    n = a.n_obs
    n_var = a.n_vars
    syms = list(a.var_names)
    obs_names = np.asarray(a.obs_names, dtype=object)
    a.file.close()
    cols, toks, meds, gstats = enc.map_genes(syms)
    if len(cols) == 0:
        raise ValueError(f"{name}: no genes map to the Geneformer vocabulary")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = int(np.ceil(n / shard_cells))
    t0 = time.time()
    done = 0
    for s in range(n_shards):
        f = out_dir / f"{name}__shard{s:04d}.npz"
        if f.exists():
            try:                                    # trust a cache only if it LOADS
                with np.load(f) as z:
                    if "emb" in z and "cell_id" in z and len(z["cell_id"]) > 0:
                        done += len(z["cell_id"]); continue
            except Exception:
                f.unlink(missing_ok=True)           # truncated shard -> redo
        lo, hi = s * shard_cells, min((s + 1) * shard_cells, n)
        with h5py.File(h5, "r") as hf:
            X = _csr_row_block(hf, lo, hi, n_var)
        ids = obs_names[lo:hi]

        seqs = enc.encode_rows(X, cols, toks, meds)
        del X
        embs = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)

        # Dynamic batching by TOKEN count, not cell count: sequence lengths vary
        # by an order of magnitude across cells, so a fixed cell batch either
        # wastes VRAM on short cells or OOMs on long ones.
        order = np.argsort([len(s_) for s_ in seqs])   # group similar lengths
        i = 0
        while i < len(order):
            batch, tok_budget = [], 0
            while i < len(order):
                L = max(len(seqs[order[i]]), 1)
                if batch and tok_budget + L > batch_tokens:
                    break
                batch.append(order[i]); tok_budget += L; i += 1
            if not batch:
                break
            L = max(max(len(seqs[b]) for b in batch), 1)
            L = min(L, max_len)
            ii = np.full((len(batch), L), enc.pad_token, dtype=np.int64)
            am = np.zeros((len(batch), L), dtype=np.int64)
            for k, b in enumerate(batch):
                s_ = seqs[b][:L]
                if len(s_):
                    ii[k, :len(s_)] = s_; am[k, :len(s_)] = 1
                else:
                    am[k, 0] = 1                    # all-zero cell: 1 pad token attended
            ii_t = torch.from_numpy(ii).to(device)
            am_t = torch.from_numpy(am).to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                h = model(input_ids=ii_t, attention_mask=am_t).last_hidden_state
                m = am_t.unsqueeze(-1).to(h.dtype)
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)   # mean over real tokens
            out = pooled.float().cpu().numpy()
            if not np.isfinite(out).all():
                raise FloatingPointError(
                    f"{name} shard {s}: non-finite embedding under fp16. Refusing to "
                    f"persist -- silently saving NaNs would corrupt every downstream result."
                )
            for k, b in enumerate(batch):
                embs[b] = out[k]
            del ii_t, am_t, h, pooled

        np.savez_compressed(f, emb=embs, cell_id=np.array([f"{name}|{c}" for c in ids]),
                            n_tokens=np.array([len(x) for x in seqs], dtype=np.int32))
        done += len(ids)
        del seqs, embs
        gc.collect()
        print(f"  [{name}] shard {s+1}/{n_shards} ({done}/{n} cells) "
              f"{time.time()-t0:.0f}s", flush=True)
    return {"dataset": name, "cells": int(n), "shards": n_shards,
            "seconds": round(time.time() - t0, 1), **gstats}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of dataset names (pilot / timing runs)")
    ap.add_argument("--shard-cells", type=int, default=2000)
    ap.add_argument("--batch-tokens", type=int, default=8192)
    args = ap.parse_args(argv)
    cfg = config.load(args.config)

    from transformers import AutoModel, AutoConfig
    root = cfg["models"]["hf_cache"]
    ck, dd = find_snapshot(root)
    max_len = int(cfg["models"]["geneformer"]["input_size"])

    enc = GeneformerEncoder(dd, model_input_size=max_len)
    checks = verify_encoding(enc)          # refuses to proceed on failure
    print("encoding verified:", json.dumps(checks), flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; refusing to embed on CPU (would take days)")

    # Host-memory guard. NOTE: `ulimit -v` must NOT be used to bound a CUDA process
    # -- CUDA initialisation alone reserves ~8.1 GiB of VIRTUAL address space (mostly
    # unbacked mappings for unified addressing), so an address-space cap that looks
    # generous starves the driver and surfaces as "CUDA driver error: out of memory"
    # while nvidia-smi reports the GPU completely free. Bound RESIDENT memory instead,
    # which is what actually competes for the 15 GiB host and what the OOM killer
    # watches.
    import resource
    rss_cap_gb = float(cfg["run"].get("max_worker_rss_gb", 3.5))
    resource.setrlimit(resource.RLIMIT_RSS,
                       (int(rss_cap_gb * 1024**3), resource.RLIM_INFINITY))

    def _rss_gb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    print(f"host RSS guard: {rss_cap_gb} GiB (RLIMIT_RSS); virtual address space left "
          f"unbounded because CUDA requires it", flush=True)
    mcfg = AutoConfig.from_pretrained(ck)
    model = AutoModel.from_pretrained(ck).eval().to("cuda")

    proc = pathlib.Path(cfg["data"]["processed"])
    out = pathlib.Path(cfg["data"]["embeddings"]) / "geneformer"
    res = pathlib.Path(cfg["data"]["results"])
    names = [d["name"] for d in cfg["corpus"]["datasets"]]
    if args.datasets:
        names = [n for n in names if n in args.datasets]

    rows, timings = [], []
    for nm in names:
        with provenance.timed(f"embed_geneformer:{nm}", timings):
            rows.append(embed_dataset(nm, proc / f"{nm}.h5ad", out, enc, model, "cuda",
                                      args.shard_cells, args.batch_tokens, max_len))
        print(f"[done] {nm}: {rows[-1]}", flush=True)
        (out / "_stats").mkdir(parents=True, exist_ok=True)
        (out / "_stats" / f"{nm}.json").write_text(json.dumps(rows[-1]))

    # assemble stats from sidecars so a resumed run cannot truncate the table
    allr = [json.loads(p.read_text()) for p in sorted((out / "_stats").glob("*.json"))]
    import pandas as pd
    pd.DataFrame(allr).to_csv(res / "embed_geneformer_stats.csv", index=False)
    json.dump(timings, open(res / "timings_embed_geneformer.json", "w"), indent=2)
    print(json.dumps({"datasets": len(allr),
                      "peak_vram_mb": int(torch.cuda.max_memory_allocated() / 1e6),
                      "peak_host_rss_gb": round(_rss_gb(), 2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
