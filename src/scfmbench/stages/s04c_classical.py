"""Classical representations, fit per split on the TRAINING partition only.

This is the structural difference between the classical arm and the scFM arm,
and it is not incidental to the study -- it is the point. A zero-shot foundation
model embeds every cell once, independent of any partition, so its embedding can
be cached globally. HVG+PCA, scVI and Harmony are FIT, so each must be refit for
every split, using only that split's training cells, and then applied unchanged
to the test cells.

Caching a single global PCA and reusing it across splits would be faster and
would be leakage: the components would carry information from cells that later
appear in test. This stage therefore produces one representation per split file,
not one per corpus.

Cost consequence, stated plainly: the classical arm is more expensive to
evaluate than the scFM arm here, which is the opposite of the usual framing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfmbench import config, provenance
from scfmbench.models import classical, deep_classical


def load_corpus(cfg) -> tuple[sp.csr_matrix, pd.DataFrame, np.ndarray]:
    """Concatenate the processed corpus on the INTERSECTION of gene symbols.

    The intersection, not the union: a union would pad each dataset with zero
    columns for genes it never measured, and a zero in an unmeasured gene is not
    an observation of absence. PCA fit on union-padded data would model the
    assay's gene panel rather than biology, and the resulting batch structure
    would flatter integration methods for the wrong reason.
    """
    import h5py
    proc = pathlib.Path(cfg["data"]["processed"])
    names = [d["name"] for d in cfg["corpus"]["datasets"]]
    gene_sets, obs_list = [], []
    for nm in names:
        a = ad.read_h5ad(proc / f"{nm}.h5ad", backed="r")
        gene_sets.append(set(a.var_names.astype(str)))
        o = a.obs[["dataset", "group", "donor", "label"]].copy()
        o.index = [f"{nm}|{c}" for c in a.obs_names]
        obs_list.append(o)
        a.file.close()
    common = sorted(set.intersection(*gene_sets))
    if len(common) < 1000:
        raise ValueError(f"only {len(common)} genes common to all datasets; refusing "
                         f"to build a corpus that thin")
    blocks = []
    for nm in names:
        a = ad.read_h5ad(proc / f"{nm}.h5ad", backed="r")
        cols = pd.Index(a.var_names.astype(str)).get_indexer(pd.Index(common))
        n, nv = a.n_obs, a.n_vars
        a.file.close()
        parts = []
        with h5py.File(proc / f"{nm}.h5ad", "r") as hf:
            for lo in range(0, n, 5000):
                hi = min(lo + 5000, n)
                g = hf["X"]
                ip = g["indptr"][lo:hi + 1]
                s, e = int(ip[0]), int(ip[-1])
                blk = sp.csr_matrix((g["data"][s:e], g["indices"][s:e], ip - ip[0]),
                                    shape=(hi - lo, nv))
                parts.append(blk[:, cols])
        blocks.append(sp.vstack(parts).tocsr())
    X = sp.vstack(blocks).tocsr()
    obs = pd.concat(obs_list)
    return X, obs, np.array(common)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="split file names; default = every split")
    ap.add_argument("--methods", nargs="*", default=["hvg_pca"])
    args = ap.parse_args(argv)
    cfg = config.load(args.config)

    import resource
    cap = float(cfg["run"].get("max_worker_rss_gb", 6.0))
    resource.setrlimit(resource.RLIMIT_RSS, (int(cap * 1024**3), resource.RLIM_INFINITY))

    split_dir = pathlib.Path(cfg["data"]["splits"])
    out_root = pathlib.Path(cfg["data"]["embeddings"])
    res = pathlib.Path(cfg["data"]["results"])

    X, obs, genes = load_corpus(cfg)
    idx = pd.read_parquet(split_dir / "cell_index.parquet")
    order = pd.Index(obs.index).get_indexer(pd.Index(idx["cell_id"]))
    if (order < 0).any():
        raise ValueError("corpus rows do not cover the cell index")
    X = X[order]
    print(f"corpus {X.shape} on {len(genes)} common genes, "
          f"{X.nnz/1e6:.1f}M nonzeros", flush=True)

    Xn = classical.normalise_log1p(X)
    # scVI needs the ORIGINAL matrix (TISCH2 log1p(CP10K)), from which
    # pseudo-counts are reconstructed; every other arm uses the renormalised Xn.
    X_raw = X

    files = ([split_dir / f for f in args.splits] if args.splits
             else sorted(p for p in split_dir.glob("*.parquet")
                         if p.name != "cell_index.parquet"))
    # HVG count and PC count are GRIDS in the config, tuned by inner CV on train
    # only (guardrail 2: never on test). The sweep stage performs that selection; a
    # representation is materialised for each grid point so the sweep can choose
    # among them without refitting.
    hp = cfg["embeddings"]["hvg_pca"]
    hvg_grid = list(hp["n_hvg_grid"])
    pc_grid = list(hp["n_pcs_grid"])
    timings, rows = [], []
    for f in files:
        part = pd.read_parquet(f)["partition"].to_numpy()
        train = part == "train"
        for method in args.methods:
            d = out_root / method
            d.mkdir(parents=True, exist_ok=True)
            o = d / f"{f.stem}.npz"
            if o.exists():
                try:
                    with np.load(o) as z:
                        if "emb" in z and len(z["cell_id"]) == len(idx):
                            continue
                except Exception:
                    o.unlink(missing_ok=True)
            t0 = time.time()
            with provenance.timed(f"{method}:{f.stem}", timings):
                extra, hv, model = {}, None, None
                if method == "hvg_pca":
                    hv = classical.select_hvg(Xn, train, max(hvg_grid))  # TRAIN only
                    tf, model = classical.fit_pca(Xn[:, hv], train, max(pc_grid),
                                                  seed=0)               # TRAIN only
                    Z = tf(Xn[:, hv])

                elif method == "harmony":
                    # PCA is fit on TRAIN only; the Harmony step is transductive
                    # because harmonypy has no out-of-sample projection. That is a
                    # DECLARED deviation and it advantages this baseline -- see
                    # models/deep_classical.py for the full argument.
                    hv = classical.select_hvg(Xn, train, max(hvg_grid))
                    tf, model = classical.fit_pca(Xn[:, hv], train, max(pc_grid), seed=0)
                    Zp = tf(Xn[:, hv])
                    bkey = cfg["embeddings"]["harmony"].get("batch_key", ["dataset"])
                    bkey = bkey if isinstance(bkey, list) else [bkey]
                    batch = idx[bkey[0]].astype(str)
                    for k in bkey[1:]:
                        if k in idx.columns:
                            batch = batch + "|" + idx[k].astype(str)
                    Z, extra = deep_classical.fit_harmony(Zp, batch.to_numpy(), train, seed=0)

                elif method == "scvi":
                    # scVI IS inductive: trained on train cells, then encodes all.
                    # Input is depth-normalised pseudo-counts because TISCH2 ships
                    # log1p(CP10K), not UMIs -- see deep_classical.py.
                    sc_cfg = cfg["embeddings"]["scvi"]
                    # HVGs selected on TRAIN ONLY, same selector as every other
                    # arm. Passing the full gene space made a single split exceed
                    # 55 min and swap on a 15 GiB host.
                    hv = classical.select_hvg(Xn, train,
                                              int(sc_cfg.get("n_hvg", 2000)))
                    Z, extra = deep_classical.fit_scvi(
                        X_raw, train, idx["dataset"].to_numpy(),
                        n_latent=int(sc_cfg["n_latent"]),
                        max_epochs=int(sc_cfg["max_epochs"]),
                        early_stopping=bool(sc_cfg.get("early_stopping", True)),
                        seed=0, hvg_mask=hv)
                else:
                    raise NotImplementedError(method)
            np.savez_compressed(o, emb=Z.astype(np.float32),
                                cell_id=idx["cell_id"].to_numpy(),
                                n_hvg=np.array([len(hv) if hv is not None else 0]),
                                pc_grid=np.array(pc_grid))
            rows.append({"method": method, "split": f.stem,
                         "n_hvg": int(len(hv)) if hv is not None else 0,
                         "n_components": int(Z.shape[1]),
                         "explained_variance": (float(model.explained_variance_ratio_.sum())
                                                if model is not None else float("nan")),
                         **{f"info_{k}": v for k, v in extra.items()},
                         "seconds": round(time.time() - t0, 1)})
            print(f"  {method} {f.stem}: {Z.shape} "
                  f"evr={rows[-1]['explained_variance']:.3f} "
                  f"{rows[-1]['seconds']}s", flush=True)
            (d / "_stats").mkdir(exist_ok=True)
            (d / "_stats" / f"{f.stem}.json").write_text(json.dumps(rows[-1]))

    allr = []
    for method in args.methods:
        allr += [json.loads(p.read_text())
                 for p in sorted((out_root / method / "_stats").glob("*.json"))]
    pd.DataFrame(allr).to_csv(res / "classical_fit_stats.csv", index=False)
    print(json.dumps({"fits": len(allr)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
