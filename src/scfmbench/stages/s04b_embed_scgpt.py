"""Zero-shot scGPT cell embeddings, per-dataset and resumable.

scGPT ships its own embedding entry point (`scgpt.tasks.embed_data`), so unlike
Geneformer nothing here reimplements the model's preprocessing -- the released
code path is used as published, which is the safer choice when it is available.

`use_fast_transformer=False` is mandatory on this hardware: the default resolves
to a FlashAttention path that does not build on a Turing GPU. The brief expected
this to be the phase's casualty; passing the flag avoids the fight entirely.

NOTE ON NORMALISATION: scGPT returns L2-normalised embeddings (measured row norm
1.00 +/- 0.00), while Geneformer's mean-pooled states are unnormalised (norm
~16.4 +/- 2.1). Both are passed through the SAME train-fit standardiser before
the classifier, so neither model is advantaged by its output convention.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time

import anndata as ad
import numpy as np

from scfmbench import config, provenance


def embed_dataset(name: str, h5: pathlib.Path, out_dir: pathlib.Path,
                  model_dir: str, shard_cells: int, batch_size: int) -> dict:
    from scgpt.tasks import embed_data
    a = ad.read_h5ad(h5)
    n = a.n_obs
    out_dir.mkdir(parents=True, exist_ok=True)
    n_shards = int(np.ceil(n / shard_cells))
    t0, done, matched = time.time(), 0, None

    for s in range(n_shards):
        f = out_dir / f"{name}__shard{s:04d}.npz"
        if f.exists():
            try:
                with np.load(f, allow_pickle=True) as z:
                    if "emb" in z and len(z["cell_id"]) > 0:
                        done += len(z["cell_id"]); continue
            except Exception:
                f.unlink(missing_ok=True)
        lo, hi = s * shard_cells, min((s + 1) * shard_cells, n)
        sub = a[lo:hi].copy()
        sub.var["gene_name"] = sub.var_names.astype(str)
        out = embed_data(sub, model_dir=model_dir, gene_col="gene_name",
                         use_fast_transformer=False, batch_size=batch_size,
                         return_new_adata=True)
        E = np.asarray(out.obsm["X_scGPT"] if "X_scGPT" in out.obsm else out.X,
                       dtype=np.float32)
        if not np.isfinite(E).all():
            raise FloatingPointError(
                f"{name} shard {s}: non-finite scGPT embedding; refusing to persist.")
        ids = np.array([f"{name}|{c}" for c in sub.obs_names])
        np.savez_compressed(f, emb=E, cell_id=ids)
        done += len(ids)
        del sub, out, E
        gc.collect()
        print(f"  [{name}] shard {s+1}/{n_shards} ({done}/{n}) {time.time()-t0:.0f}s",
              flush=True)
    del a
    gc.collect()
    return {"dataset": name, "cells": int(n), "shards": n_shards,
            "seconds": round(time.time() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--shard-cells", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    cfg = config.load(args.config)

    import resource
    cap = float(cfg["run"].get("max_worker_rss_gb", 3.5))
    resource.setrlimit(resource.RLIMIT_RSS, (int(cap * 1024**3), resource.RLIM_INFINITY))

    model_dir = cfg["models"]["scgpt"]["checkpoint_dir"]
    if not pathlib.Path(model_dir, "best_model.pt").exists():
        raise FileNotFoundError(f"scGPT checkpoint missing at {model_dir}")

    proc = pathlib.Path(cfg["data"]["processed"])
    out = pathlib.Path(cfg["data"]["embeddings"]) / "scgpt"
    res = pathlib.Path(cfg["data"]["results"])
    names = [d["name"] for d in cfg["corpus"]["datasets"]]
    if args.datasets:
        names = [n for n in names if n in args.datasets]

    timings = []
    for nm in names:
        with provenance.timed(f"embed_scgpt:{nm}", timings):
            r = embed_dataset(nm, proc / f"{nm}.h5ad", out, model_dir,
                              args.shard_cells, args.batch_size)
        (out / "_stats").mkdir(parents=True, exist_ok=True)
        (out / "_stats" / f"{nm}.json").write_text(json.dumps(r))
        print(f"[done] {nm}: {r}", flush=True)

    import pandas as pd
    allr = [json.loads(p.read_text()) for p in sorted((out / "_stats").glob("*.json"))]
    pd.DataFrame(allr).to_csv(res / "embed_scgpt_stats.csv", index=False)
    json.dump(timings, open(res / "timings_embed_scgpt.json", "w"), indent=2)
    print(json.dumps({"datasets": len(allr)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
