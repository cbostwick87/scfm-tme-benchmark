"""Split (inductive) conformal prediction for FM-BENCH-B.

Implements the two mandated variants -- MARGINAL (one global quantile) and
MONDRIAN (one quantile per class) -- over two nonconformity scores, INVERSE_SOFTMAX
(primary) and APS (secondary).

Three implementation details decide whether the coverage guarantee actually holds,
and each is the kind of error that produces plausible-looking numbers rather than
an exception:

1. FINITE-SAMPLE QUANTILE LEVEL. The split-conformal guarantee is
   P(Y in C(X)) >= 1-alpha only if the calibration quantile is taken at
   ceil((n+1)(1-alpha))/n, NOT at the empirical (1-alpha) quantile. With n=100 and
   alpha=0.1 the naive choice takes the 90th percentile where the correct one takes
   the 91st; coverage then sits just below nominal and looks like a real finding
   about the representation. `np.quantile(..., method="higher")` on level
   ceil((n+1)(1-alpha))/n reproduces the standard estimator exactly.

2. VALIDITY FLOOR. That level exceeds 1 only when n < ceil(1/alpha) - 1, in which
   case NO finite threshold attains the coverage and the honest answer is an
   infinite quantile (every label admitted). This is recorded per class rather
   than silently replaced by the marginal quantile -- brief requirement U4.

3. TIE HANDLING IN APS. The randomised variant of APS achieves exact coverage; the
   deterministic variant is conservative. We use the DETERMINISTIC form
   consistently across every representation, so the comparison between arms is
   unaffected, and state it rather than leaving the reader to infer which.

Scores are oriented so that LARGER = MORE NONCONFORMING throughout.
"""
from __future__ import annotations

import numpy as np

INF = np.inf


# --------------------------------------------------------------------------- #
# nonconformity scores
# --------------------------------------------------------------------------- #
def score_inverse_softmax(proba: np.ndarray, class_idx: np.ndarray) -> np.ndarray:
    """1 - p(y | x) for the nominated class of each row."""
    return 1.0 - proba[np.arange(len(proba)), class_idx]


def score_aps(proba: np.ndarray, class_idx: np.ndarray) -> np.ndarray:
    """Adaptive Prediction Sets: cumulative probability mass down to the nominated
    class, sorting classes by descending probability. Deterministic variant."""
    order = np.argsort(-proba, axis=1, kind="stable")
    csum = np.cumsum(np.take_along_axis(proba, order, axis=1), axis=1)
    rank = np.argmax(order == class_idx[:, None], axis=1)
    return csum[np.arange(len(proba)), rank]


SCORES = {"inverse_softmax": score_inverse_softmax, "aps": score_aps}


def all_class_scores(proba: np.ndarray, score: str) -> np.ndarray:
    """Nonconformity of EVERY candidate class, shape (n_cells, n_classes).

    Needed on test, where the true label must not be consulted: a prediction set is
    formed by admitting every class whose score falls at or below the threshold.
    """
    n_cls = proba.shape[1]
    if score == "inverse_softmax":
        return 1.0 - proba
    if score == "aps":
        order = np.argsort(-proba, axis=1, kind="stable")
        csum = np.cumsum(np.take_along_axis(proba, order, axis=1), axis=1)
        out = np.empty_like(proba)
        np.put_along_axis(out, order, csum, axis=1)
        return out
    raise ValueError(f"unknown score {score!r}")


# --------------------------------------------------------------------------- #
# quantiles
# --------------------------------------------------------------------------- #
def min_calibration_n(alpha: float) -> int:
    """Smallest calibration count admitting a finite conformal quantile."""
    return int(np.ceil(1.0 / alpha)) - 1


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal threshold: the k-th smallest calibration score,
    k = ceil((n+1)(1-alpha)). Returns +inf when k > n, i.e. when no finite threshold
    can attain the requested coverage.

    Selected BY INDEX on the sorted scores rather than through np.quantile. The
    quantile route is where this function was wrong: `np.quantile(..., method="higher")`
    interpolates over n-1 intervals, so at level k/n it returns the (k+1)-th order
    statistic, not the k-th. Every prediction set was then built from a threshold one
    order statistic too high -- valid, but conservative, and increasingly so as the
    calibration count fell. That is a silent bias in exactly the direction that would
    have manufactured a B5 result: rare classes have the fewest calibration cells, so
    they would have been the most over-covered under Mondrian, and the marginal-versus-
    Mondrian gap B5 measures would have been inflated by an implementation artefact.
    Caught by the mandatory synthetic gate (validate_conformal.py) comparing empirical
    coverage against the exact discrete value k/(n+1); it is the reason that gate is
    non-negotiable.

    Coverage of the returned threshold is exactly k/(n+1) for continuous scores.
    """
    n = len(cal_scores)
    if n == 0:
        return INF
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return INF
    return float(np.partition(cal_scores, k - 1)[k - 1])


def marginal_threshold(cal_scores: np.ndarray, alpha: float) -> tuple[float, dict]:
    q = conformal_quantile(cal_scores, alpha)
    return q, {"n_calibration": int(len(cal_scores)),
               "insufficient": bool(len(cal_scores) < min_calibration_n(alpha))}


def mondrian_thresholds(cal_scores: np.ndarray, cal_class_idx: np.ndarray,
                        n_classes: int, alpha: float) -> tuple[np.ndarray, list[dict]]:
    """One threshold per class, each from that class's calibration cells only.

    A class with too few calibration examples gets +inf (its label is always
    admitted) and is FLAGGED. It is never silently given the marginal quantile:
    doing so would quietly convert Mondrian into marginal for exactly the rare
    classes B5 is about.
    """
    need = min_calibration_n(alpha)
    q = np.full(n_classes, INF)
    info = []
    for k in range(n_classes):
        s = cal_scores[cal_class_idx == k]
        q[k] = conformal_quantile(s, alpha)
        info.append({"class_idx": k, "n_calibration": int(len(s)),
                     "min_required": need,
                     "insufficient": bool(len(s) < need),
                     "threshold": float(q[k])})
    return q, info


# --------------------------------------------------------------------------- #
# prediction sets
# --------------------------------------------------------------------------- #
def prediction_sets(test_proba: np.ndarray, threshold, score: str) -> np.ndarray:
    """Boolean (n_cells, n_classes) membership matrix.

    `threshold` is a scalar (marginal) or a per-class vector (Mondrian). A class is
    admitted where its nonconformity is <= the threshold that governs it.
    """
    s = all_class_scores(test_proba, score)
    thr = np.asarray(threshold, dtype=float)
    return s <= (thr[None, :] if thr.ndim == 1 else thr)


def summarise(sets: np.ndarray, y_idx: np.ndarray, n_classes: int) -> dict:
    """Coverage and set size, always together (brief guardrail 7)."""
    covered = sets[np.arange(len(sets)), y_idx]
    size = sets.sum(axis=1)
    per_class = {}
    for k in range(n_classes):
        m = y_idx == k
        if m.any():
            per_class[k] = {"n": int(m.sum()),
                            "coverage": float(covered[m].mean()),
                            "mean_set_size": float(size[m].mean())}
    return {"coverage": float(covered.mean()),
            "mean_set_size": float(size.mean()),
            "median_set_size": float(np.median(size)),
            "empty_rate": float((size == 0).mean()),
            "singleton_rate": float((size == 1).mean()),
            "n_test": int(len(sets)),
            "per_class": per_class}
