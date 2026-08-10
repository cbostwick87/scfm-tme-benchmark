"""Project B conformal sweep: one row per
(representation x split x budget x seed x coverage x variant x score x dataset).

COST STRUCTURE, which differs fundamentally from FM-BENCH-A's sweep and is why this
is cheap where A's was expensive:

  * NO INNER CV. A recorded C_selected and n_dims_selected per run in T2, so the head
    is refit at the already-selected value. A's DECISIONS entry 58 measured inner CV
    at 99.3% of a run's cost against 0.4% for the final refit, so removing it removes
    essentially all of the per-run expense -- and it is also MORE correct, because it
    reproduces A's exact classifier rather than re-selecting from a different
    partition. Nothing is selected here, so this stage cannot introduce hyperparameter
    selection on test.
  * ONE FIT SERVES THE WHOLE GRID. Coverage levels, conformal variants and score
    families are all downstream of the fitted probabilities, so a single head fit
    yields every (alpha x variant x score) cell at once. The conformal grid is
    therefore nearly free relative to the fit.

Two leakage controls are enforced here rather than assumed:
  * The head is fit on TRAIN only and the standardiser is fit on TRAIN only.
  * The CALIBRATION partition reaches nothing but the quantile. Project A never read
    it; this is the stage that first does, and it must be used for nothing else.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..models.classical import fit_standardiser
from . import conformal as CF

ALPHAS = (0.20, 0.10, 0.05)
VARIANTS = ("marginal", "mondrian")
SCORES = ("inverse_softmax", "aps")


def budget_indices(y_train: np.ndarray, budget, seed: int) -> np.ndarray:
    """Label-budget subsample. Identical rule and seed semantics to A's sweep, so a
    B run consumes the same labelled cells as the A run it corresponds to."""
    if budget is None:
        return np.arange(len(y_train))
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y_train):
        i = np.flatnonzero(y_train == c)
        keep.append(rng.permutation(i)[:int(budget)])
    return np.sort(np.concatenate(keep))


def fit_head(E: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
             C: float, n_dims: int, max_iter: int = 2000):
    """Refit A's head. TRUNCATE FIRST: hvg_pca's PC count is chosen per run by inner
    CV (30/50/100 observed) while every other arm is fixed-width. A's DECISIONS entry
    94 records that ignoring n_dims_selected silently refit a higher-dimensional model
    than the run being reconstructed, corrupting 1,966 runs before its guard caught it.
    """
    if n_dims > E.shape[1]:
        raise ValueError(f"n_dims_selected={n_dims} exceeds stored width {E.shape[1]}")
    Ez = E[:, :n_dims]
    std = fit_standardiser(Ez[train_idx])          # TRAIN-only fit
    clf = LogisticRegression(C=float(C), max_iter=max_iter)
    clf.fit(std(Ez[train_idx]), y[train_idx])
    if clf.coef_.shape[0] != len(clf.classes_):
        raise ValueError("head is not multinomial")
    return clf, std, Ez


def conformal_rows(clf, std, Ez, y, calib_mask, test_mask, dataset,
                   meta: dict, alphas=ALPHAS, variants=VARIANTS, scores=SCORES,
                   unseen_mask: np.ndarray | None = None) -> tuple[list[dict], list[dict]]:
    """All conformal cells for one fitted head.

    Returns (result_rows, calibration_count_rows). `unseen_mask` marks cells of an S4
    held-out type, which are EXCLUDED from coverage (no true class exists in the
    label space) and scored separately for B4.
    """
    cls = list(clf.classes_)
    cidx = {c: i for i, c in enumerate(cls)}
    Pc = clf.predict_proba(std(Ez[calib_mask]))
    Pt = clf.predict_proba(std(Ez[test_mask]))
    yc, yt = y[calib_mask], y[test_mask]
    # Calibration cells whose class is outside the head's label space cannot form a
    # nonconformity score; under S4 this is exactly the excluded type and must be 0.
    keep_c = np.array([c in cidx for c in yc])
    yc_i = np.array([cidx[c] for c in yc[keep_c]])
    seen_t = np.array([c in cidx for c in yt])
    yt_i = np.full(len(yt), -1)
    yt_i[seen_t] = [cidx[c] for c in yt[seen_t]]

    ds_t = dataset[test_mask]
    rows, calrows = [], []
    for score in scores:
        sc = CF.SCORES[score](Pc[keep_c], yc_i)
        for alpha in alphas:
            thr_by_variant = {}
            qm, minfo = CF.marginal_threshold(sc, alpha)
            thr_by_variant["marginal"] = (qm, [dict(minfo, class_idx=-1,
                                                    class_name="__marginal__")])
            qv, vinfo = CF.mondrian_thresholds(sc, yc_i, len(cls), alpha)
            for d in vinfo:
                d["class_name"] = cls[d["class_idx"]]
            thr_by_variant["mondrian"] = (qv, vinfo)

            for variant in variants:
                thr, info = thr_by_variant[variant]
                sets = CF.prediction_sets(Pt, thr, score)
                for d in info:
                    calrows.append({**meta, "score": score, "alpha": alpha,
                                    "variant": variant, **d})
                # per DATASET -- the unit of replication (guardrail 5)
                for ds in np.unique(ds_t):
                    m = (ds_t == ds) & seen_t
                    if unseen_mask is not None:
                        m &= ~unseen_mask[test_mask]
                    if not m.any():
                        continue
                    s = CF.summarise(sets[m], yt_i[m], len(cls))
                    per_class = s.pop("per_class")
                    rows.append({**meta, "dataset": ds, "score": score,
                                 "alpha": alpha, "nominal": 1 - alpha,
                                 "variant": variant,
                                 "coverage_gap": (1 - alpha) - s["coverage"],
                                 **s,
                                 # keyed by CLASS NAME, not index: the index is only
                                 # meaningful alongside this run's clf.classes_, and B5
                                 # aggregates per-class coverage ACROSS runs whose label
                                 # spaces differ (S4 drops one class, S3 holdouts vary).
                                 "per_class_json": json.dumps(
                                     {cls[k]: v for k, v in per_class.items()})})
    return rows, calrows
