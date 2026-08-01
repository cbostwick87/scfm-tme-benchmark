"""Build and persist the three-way leakage-controlled splits.

Emits, for every (scheme, seed) and for S3 additionally every holdout group:
  splits/{scheme}__seed{K}[__holdout-{GROUP}].parquet   cell_id -> partition

The CALIBRATION partition is written but is NEVER read by this project: it is
reserved untouched for the downstream conformal-prediction work. Project A uses
TRAIN and TEST only, enforced at read time by `assert_calibration_untouched`.

Every emitted split is checked by `assert_no_leakage` before it is written, so a
split that violates its own scheme cannot reach disk.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import anndata as ad
import pandas as pd

from scfmbench import config, provenance, splits


def build_index(cfg: dict) -> pd.DataFrame:
    """Assemble the cell index (cell_id, dataset, group, donor, label) from the
    per-dataset processed h5ad, reading obs only -- never the matrices."""
    proc = pathlib.Path(cfg["data"]["processed"])
    frames = []
    for d in cfg["corpus"]["datasets"]:
        f = proc / f"{d['name']}.h5ad"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing; run stage.harmonise first")
        a = ad.read_h5ad(f, backed="r")
        obs = a.obs[["dataset", "group", "donor", "label"]].copy()
        obs.index.name = "cell_id"
        a.file.close()
        frames.append(obs.reset_index())
    idx = pd.concat(frames, ignore_index=True)
    # cell_id must be globally unique: TISCH2 barcodes repeat across datasets,
    # so a bare barcode would silently collide and put the same key in two
    # partitions. Qualify it by dataset.
    idx["cell_id"] = idx["dataset"] + "|" + idx["cell_id"].astype(str)
    if idx["cell_id"].duplicated().any():
        raise ValueError("duplicate cell_id after dataset qualification")
    return idx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = config.load(args.config)

    out = pathlib.Path(cfg["data"]["splits"]); out.mkdir(parents=True, exist_ok=True)
    res = pathlib.Path(cfg["data"]["results"]); res.mkdir(parents=True, exist_ok=True)

    idx = build_index(cfg)
    idx.to_parquet(out / "cell_index.parquet", index=False)
    fracs = cfg["splits"]["fractions"]
    rows, timings = [], []

    for scheme in cfg["splits"]["scheme_ids"]:
        for seed in cfg["run"]["seeds"]:
            targets = [None]
            if scheme == "S3_leave_dataset_out":
                targets = sorted(idx["group"].unique())
            for holdout in targets:
                tag = f"{scheme}__seed{seed}" + (f"__holdout-{holdout}" if holdout else "")
                f = out / f"{tag}.parquet"
                if f.exists() and not args.force:
                    continue
                with provenance.timed(f"split:{tag}", timings):
                    part = splits.make_splits(idx, scheme, fracs, seed=seed,
                                              holdout_group=holdout)
                    # Refuse to persist a split that violates its own scheme.
                    splits.assert_no_leakage(idx, part, scheme)
                    df = idx[["cell_id"]].copy()
                    df["partition"] = part.to_numpy()
                    df.to_parquet(f, index=False)

                d = idx.assign(partition=part.to_numpy())
                counts = d["partition"].value_counts()
                row = {"scheme": scheme, "seed": seed, "holdout_group": holdout or "",
                       "file": f.name}
                for p in splits.PARTITIONS:
                    row[f"n_{p}"] = int(counts.get(p, 0))
                    sub = d[d.partition == p]
                    row[f"classes_{p}"] = int(sub["label"].nunique())
                    row[f"datasets_{p}"] = int(sub["dataset"].nunique())
                # classes usable by the PRIMARY metric: present in train AND test
                tr = set(d[d.partition == "train"]["label"])
                te = set(d[d.partition == "test"]["label"])
                row["classes_learnable"] = len(tr & te)
                row["classes_test_only"] = len(te - tr)
                rows.append(row)

    man = pd.DataFrame(rows)
    man.to_csv(res / "split_manifest.csv", index=False)
    json.dump(timings, open(res / "timings_splits.json", "w"), indent=2)
    print(json.dumps({"splits_written": len(rows), "cells": len(idx)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
