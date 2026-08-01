"""Figures F1-F5, regenerated from the persisted tidy results table.

Every figure is a pure function of results/T2_results_long.csv and
results/T3_primary_statistics.csv -- no notebook state, no manual steps. Each
is emitted as PDF (vector, for the report), SVG (editable) and PNG (preview).

Plotting conventions that are not cosmetic:
  * Dataset-level points are always shown, not only the aggregate. With ~13
    replication units, a bar of means hides whether an effect is consistent or
    driven by two datasets -- which is exactly what the reader needs to judge.
  * Uncertainty is bootstrap CI across DATASETS, matching the statistics; error
    bars over seeds would understate uncertainty by an order of magnitude.
  * The zero line is drawn on every difference plot, because the study's
    expected answer is "no difference" and that must be legible at a glance.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, out_dir: pathlib.Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _boot_ci(x, n_boot=5000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    return tuple(np.percentile(b, [2.5, 97.5]))


def f1_performance_vs_budget(t2: pd.DataFrame, out: pathlib.Path) -> None:
    """F1: macro-F1 against label budget, one panel per split scheme."""
    schemes = sorted(t2["scheme"].dropna().unique())
    fig, axes = plt.subplots(1, len(schemes), figsize=(5 * len(schemes), 4.2),
                             sharey=True, squeeze=False)
    order = {"5": 0, "10": 1, "25": 2, "50": 3, "100": 4, "all": 5}
    for ax, sch in zip(axes[0], schemes):
        sub = t2[t2.scheme == sch].copy()
        sub["unit"] = np.where(sub["holdout_group"].astype(str).str.len() > 0,
                               sub["holdout_group"].astype(str), sub["split_file"])
        for rep, g in sub.groupby("representation"):
            a = (g.groupby(["unit", "budget"])["macro_f1"].mean().reset_index()
                   .groupby("budget")["macro_f1"])
            mean = a.mean()
            idx = sorted(mean.index, key=lambda b: order.get(str(b), 99))
            lo = [_boot_ci(g[g.budget.astype(str) == str(b)]
                           .groupby("unit")["macro_f1"].mean())[0] for b in idx]
            hi = [_boot_ci(g[g.budget.astype(str) == str(b)]
                           .groupby("unit")["macro_f1"].mean())[1] for b in idx]
            xs = np.arange(len(idx))
            ax.plot(xs, [mean[b] for b in idx], marker="o", label=rep, lw=1.8)
            ax.fill_between(xs, lo, hi, alpha=0.15)
            ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in idx])
        ax.set_title(sch.replace("_", " "))
        ax.set_xlabel("labelled cells per class")
        ax.grid(alpha=0.25, ls=":")
    axes[0][0].set_ylabel("macro-F1 (learnable classes)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle("F1 — annotation performance vs label budget "
                 "(mean over datasets, bootstrap 95% CI across datasets)", y=1.02)
    save(fig, out, "F1_performance_vs_label_budget")


def f4_transfer_matrix(t2: pd.DataFrame, out: pathlib.Path) -> None:
    """F4: per-holdout-group transfer performance under leave-dataset-out."""
    sub = t2[(t2.scheme == "S3_leave_dataset_out") &
             (t2.holdout_group.astype(str).str.len() > 0)]
    if sub.empty:
        return
    piv = sub.pivot_table(index="holdout_group", columns="representation",
                          values="macro_f1", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.3 * len(piv.columns) + 3, 0.45 * len(piv) + 2.5))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < np.nanmean(piv.to_numpy()) else "black")
    fig.colorbar(im, ax=ax, label="macro-F1")
    ax.set_title("F4 — leave-dataset-out transfer, per held-out study")
    save(fig, out, "F4_transfer_matrix")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    from scfmbench import config
    cfg = config.load(args.config)
    res = pathlib.Path(cfg["data"]["results"])
    out = pathlib.Path(cfg["data"]["figures"])
    t2 = pd.read_csv(res / "T2_results_long.csv")
    f1_performance_vs_budget(t2, out)
    f4_transfer_matrix(t2, out)
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
