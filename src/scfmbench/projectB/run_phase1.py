"""Project B Phase 1: the main conformal sweep (B1, B3, B5).

Sharded by (representation, split) and resumable: each shard writes its own parquet
under results/projectB/shards/ and is skipped if already present, so a crash loses at
most one shard. This is A's pattern and the brief requires it.

The schedule is DERIVED FROM A's T2, not declared here: for every (representation,
split, budget) that A actually ran, B reuses A's realised seed values and A's recorded
C_selected / n_dims_selected. A cell A never ran is not invented, and B cannot silently
run a different grid from the one it claims to compare against.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/fm-bench")
REPO = ROOT / "scfm-tme-benchmark"
OUT = REPO / "results" / "projectB"
SHARDS = OUT / "shards"
BUDGETS = ["10", "100", "all"]


def build_schedule() -> pd.DataFrame:
    t2 = pd.read_csv(REPO / "results" / "T2_results_long.csv")
    runs = t2.drop_duplicates(["representation", "split_file", "budget", "seed"])
    s = runs[runs.budget.isin(BUDGETS)][
        ["representation", "split_file", "budget", "seed", "C_selected", "n_dims_selected"]
    ].copy()
    s["split"] = s.split_file.str.replace(".parquet", "", regex=False)
    s["scheme"] = s.split.str.split("__").str[0]
    return s.sort_values(["representation", "split", "budget", "seed"]).reset_index(drop=True)


def run_shard(rep: str, split: str, sched: pd.DataFrame, idx: pd.DataFrame,
              force: bool = False) -> tuple[str, int, float]:
    from ..stages.s05_sweep import load_embedding
    from . import sweep as SW

    tag = f"{rep}__{split}"
    fr, fc = SHARDS / f"{tag}__results.parquet", SHARDS / f"{tag}__calib.parquet"
    if fr.exists() and fc.exists() and not force:
        return tag, -1, 0.0

    t0 = time.time()
    E, cids = load_embedding(ROOT / "embeddings", rep, split)
    order = pd.Series(np.arange(len(cids)), index=cids).reindex(idx.cell_id).to_numpy()
    if np.isnan(order).any():
        raise ValueError(f"{tag}: embedding does not cover the cell index")
    E = E[order.astype(int)]
    part = pd.read_parquet(ROOT / "splits" / f"{split}.parquet")
    if not np.array_equal(part.cell_id.to_numpy(), idx.cell_id.to_numpy()):
        raise ValueError(f"{tag}: split file is not aligned to the cell index")
    p = part.partition.to_numpy()
    y = idx.label.to_numpy()
    ds = idx.dataset.to_numpy()
    tr_all = np.flatnonzero(p == "train")
    cal, te = p == "calibration", p == "test"

    rows, calrows = [], []
    for r in sched.itertuples(index=False):
        sub = SW.budget_indices(y[tr_all], None if r.budget == "all" else int(r.budget), r.seed)
        clf, std, Ez = SW.fit_head(E, y, tr_all[sub], r.C_selected, int(r.n_dims_selected))
        meta = dict(representation=rep, split=split, scheme=r.scheme, budget=r.budget,
                    seed=int(r.seed), n_train=int(len(sub)),
                    C_selected=float(r.C_selected), n_dims=int(r.n_dims_selected))
        a, b = SW.conformal_rows(clf, std, Ez, y, cal, te, ds, meta)
        rows += a
        calrows += b

    SHARDS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(fr, index=False)
    pd.DataFrame(calrows).to_parquet(fc, index=False)
    return tag, len(rows), time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="geneformer,scgpt,hvg_pca,scvi,harmony")
    ap.add_argument("--schemes", default="S1_within_dataset,S2_leave_donor_out,S3_leave_dataset_out")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N shards only (pilot)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    idx = pd.read_parquet(ROOT / "splits" / "cell_index.parquet")
    sched = build_schedule()
    reps = a.reps.split(",")
    schemes = a.schemes.split(",")
    sched = sched[sched.representation.isin(reps) & sched.scheme.isin(schemes)]
    keys = sched.groupby(["representation", "split"]).size().reset_index(name="n_runs")
    if a.limit:
        keys = keys.head(a.limit)
    print(f"{len(keys)} shards, {int(keys.n_runs.sum())} head fits", flush=True)

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(
        delayed(run_shard)(k.representation, k.split,
                           sched[(sched.representation == k.representation)
                                 & (sched.split == k.split)], idx, a.force)
        for k in keys.itertuples(index=False))
    done = [r for r in res if r[1] >= 0]
    print(f"\n{len(done)} shards written, {len(res)-len(done)} already present")
    if done:
        print(f"rows: {sum(r[1] for r in done)} | slowest shard {max(r[2] for r in done):.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
