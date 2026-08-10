"""Project B statistics: B1 (coverage under shift), B3 (set size at matched
coverage), B5 (class-conditional failure).

Two rules from the brief govern every function here:

REPLICATION UNIT IS THE DATASET (n=13). Seeds -- both split seeds and label seeds --
are resampling noise and are averaged away BEFORE any test. A's DECISIONS entry 48
records nearly presenting five split seeds as five units; the aggregation step exists
so that cannot happen. Under S3 a run contributes one row (the test partition IS the
held-out dataset), so the dataset axis is the holdout identity.

COVERAGE AND SET SIZE ARE REPORTED TOGETHER (guardrail 7). Perfect coverage with
useless sets is not a success and small sets with broken coverage are dangerous, so
every contrast function returns both and the table keeps them adjacent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def aggregate_seeds(df: pd.DataFrame, metrics=("coverage", "mean_set_size", "coverage_gap",
                                               "empty_rate", "singleton_rate")) -> pd.DataFrame:
    """Mean over seeds within (representation, scheme, budget, alpha, variant, score,
    dataset). n_runs counts split-partitions x label-seeds, not seeds -- the name
    matters because under S1 five split files x five label seeds is 25, and a reader
    seeing n_seeds=25 would think the design ran 25 seeds."""
    keys = ["representation", "scheme", "budget", "alpha", "nominal", "variant", "score", "dataset"]
    g = df.groupby(keys, observed=True)
    out = g[list(metrics)].mean()
    out["n_runs"] = g.size()
    return out.reset_index()


def paired_across_datasets(agg: pd.DataFrame, metric: str, rep_a: str, rep_b: str,
                           **subset) -> dict:
    """Paired contrast of rep_a against rep_b over datasets, with effect size and CI.

    Never reports significance without both (guardrail 11). Uses the exact Wilcoxon
    signed-rank when n is small enough for it, and reports the attainable p-floor so a
    boundary result cannot be mistaken for strong evidence.
    """
    m = agg.copy()
    for k, v in subset.items():
        m = m[m[k] == v]
    a = m[m.representation == rep_a].set_index("dataset")[metric]
    b = m[m.representation == rep_b].set_index("dataset")[metric]
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 3:
        return {"n_datasets": len(common), "insufficient": True}
    d = (a.loc[common] - b.loc[common]).to_numpy()
    n = len(d)
    # exact signed-rank when no ties/zeros, else the normal approximation
    try:
        w = stats.wilcoxon(d, alternative="two-sided", method="exact")
    except Exception:
        w = stats.wilcoxon(d, alternative="two-sided")
    floor = 2.0 / 2 ** n
    # bootstrap CI on the mean paired difference (dataset is the resampling unit)
    rng = np.random.default_rng(20260810)
    boot = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(10000)])
    sd = d.std(ddof=1)
    return {"metric": metric, "rep_a": rep_a, "rep_b": rep_b,
            "n_datasets": n, "mean_diff": float(d.mean()),
            "median_diff": float(np.median(d)),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "dz": float(d.mean() / sd) if sd > 0 else np.nan,
            "wilcoxon_p": float(w.pvalue), "p_floor_exact": float(floor),
            "at_floor": bool(abs(w.pvalue - floor) < 1e-12),
            "n_favouring_a": int((d > 0).sum()), "n_favouring_b": int((d < 0).sum()),
            "insufficient": False, **subset}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[o[i]] * n / (i + 1))
        q[o[i]] = prev
    return np.clip(q, 0, 1)
