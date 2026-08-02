"""Primary statistics: paired contrasts, effect sizes, CIs, BH-FDR correction.

THE UNIT OF REPLICATION IS THE DATASET, NOT THE RUN (guardrail 5).
Seeds are aggregated FIRST (mean per dataset x condition), and the test is then
across datasets. Treating 5 seeds x 13 datasets as n=65 would inflate the
apparent sample size by an order of magnitude and manufacture significance from
seed noise; seeds measure estimator variance, datasets measure generalisation.

NO SIGNIFICANCE WITHOUT EFFECT SIZE AND CI (guardrail 9). Every contrast reports
the paired mean difference in macro-F1, its bootstrap confidence interval, the
raw p-value and the BH-FDR-adjusted p-value. A delta of 0.005 that reaches
p < 0.05 is reported as what it is: statistically detectable and scientifically
negligible.

Hypotheses:
  H1  in-distribution (S1): scFM vs each classical representation
  H2  under shift (S2, S3): does any scFM advantage appear or grow?
  H3  rarity: does the contrast depend on class prevalence in training?
  H4  gene overlap: does transfer degrade with feature-space divergence?
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats


def paired_contrast(df: pd.DataFrame, a: str, b: str, metric: str,
                    n_boot: int = 10000, seed: int = 0) -> dict:
    """Paired contrast a - b across DATASETS (seeds already aggregated).

    Wilcoxon signed-rank is the primary test: with ~13 datasets, normality of
    the difference distribution cannot be established, and a paired t-test's
    assumptions would be doing unearned work. The t-test is reported alongside
    for transparency, not as the decision rule.
    """
    piv = df.pivot_table(index="dataset", columns="representation",
                         values=metric, aggfunc="mean")
    if a not in piv.columns or b not in piv.columns:
        return {"error": f"missing representation {a!r} or {b!r}"}
    pair = piv[[a, b]].dropna()
    d = (pair[a] - pair[b]).to_numpy()
    n = len(d)
    if n < 3:
        return {"n_datasets": n, "error": "fewer than 3 paired datasets"}

    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    try:
        w_stat, w_p = stats.wilcoxon(d, alternative="two-sided",
                                     zero_method="wilcox", mode="auto")
    except ValueError:                      # all differences identical/zero
        w_stat, w_p = float("nan"), 1.0
    t_stat, t_p = stats.ttest_rel(pair[a], pair[b])

    # Cohen's dz for paired designs; NaN when the differences have no spread.
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else float("nan")

    return {"contrast": f"{a} - {b}", "metric": metric, "n_datasets": int(n),
            "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "cohens_dz": dz,
            "wilcoxon_stat": float(w_stat), "p_raw": float(w_p),
            "ttest_p": float(t_p),
            "n_datasets_favouring_a": int((d > 0).sum()),
            "datasets": ";".join(pair.index.astype(str))}


def aggregate_seeds(t2: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Collapse seeds to one value per (dataset, representation, condition).

    THE UNIT OF REPLICATION IS THE DATASET (guardrail 5), and it is read from an
    explicit `dataset` column produced by per-dataset scoring in the sweep.

    There is deliberately NO fallback. An earlier version fell back to the split
    file when no dataset column existed, which under S1/S2 -- where one split
    spans every dataset -- silently made the SPLIT SEED the replication unit, so
    the paired test would have run across 5 resamplings of the same corpus
    presented as 5 independent units. That is the error guardrail 5 exists to
    prevent, and a fallback that produces a plausible number is worse than a
    hard failure, because nothing downstream can tell the difference.
    """
    if "dataset" not in t2.columns:
        raise ValueError(
            "T2 has no `dataset` column, so the unit of replication cannot be the "
            "dataset. Re-run the sweep: it must score each run per dataset "
            "(metrics.evaluate_per_dataset). Refusing to substitute the split file, "
            "which would test across split seeds and inflate the apparent "
            "independence of the sample.")
    if t2["dataset"].isna().any():
        raise ValueError("T2 contains rows with a null dataset; refusing to aggregate.")
    keys = ["dataset", "representation", "scheme", "budget"]
    agg = (t2.groupby(keys, dropna=False)[metric]
             .agg(["mean", "std", "count"]).reset_index()
             .rename(columns={"mean": metric, "std": f"{metric}_seed_sd",
                              "count": "n_seeds"}))
    return agg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--results", default=None)
    args = ap.parse_args(argv)
    from scfmbench import config
    cfg = config.load(args.config)
    res = pathlib.Path(cfg["data"]["results"])
    t2 = pd.read_csv(args.results or res / "T2_results_long.csv")

    enabled = [k for k, v in cfg["embeddings"].items()
               if isinstance(v, dict) and v.get("enabled")]
    scfms = [r for r in enabled if r in ("geneformer", "scgpt")]
    classical = [r for r in enabled if r not in ("geneformer", "scgpt")]
    metric = "macro_f1"

    rows = []
    for scheme in sorted(t2["scheme"].dropna().unique()):
        for budget in sorted(t2["budget"].astype(str).unique()):
            sub = t2[(t2.scheme == scheme) & (t2.budget.astype(str) == budget)]
            if sub.empty:
                continue
            agg = aggregate_seeds(sub, metric)
            for fm in scfms:
                for cl in classical:
                    r = paired_contrast(agg, fm, cl, metric)
                    if "error" in r and "n_datasets" not in r:
                        continue
                    r.update({"scheme": scheme, "budget": budget,
                              "hypothesis": "H1" if scheme == "S1_within_dataset" else "H2"})
                    rows.append(r)

    t3 = pd.DataFrame(rows)
    if not t3.empty and "p_raw" in t3:
        # BH-FDR across the whole family of primary contrasts, not per scheme:
        # correcting within subgroups would understate the multiplicity that is
        # actually being incurred.
        from statsmodels.stats.multitest import multipletests
        ok = t3["p_raw"].notna()
        t3.loc[ok, "p_fdr"] = multipletests(t3.loc[ok, "p_raw"], method="fdr_bh")[1]
        t3["significant_fdr_005"] = t3["p_fdr"] < 0.05
        # An effect can be significant and negligible. Flag it explicitly so a
        # reader never has to infer it from the delta column.
        thr = float(cfg["statistics"].get("negligible_delta", 0.02))
        t3["negligible_effect"] = t3["delta_mean"].abs() < thr
        t3["verdict"] = np.where(
            ~t3["significant_fdr_005"], "no detectable difference",
            np.where(t3["negligible_effect"],
                     f"detectable but negligible (|delta| < {thr})",
                     np.where(t3["delta_mean"] > 0, "scFM better", "classical better")))
    t3.to_csv(res / "T3_primary_statistics.csv", index=False)
    print(json.dumps({"contrasts": len(t3)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
