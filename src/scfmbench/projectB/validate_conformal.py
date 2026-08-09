"""MANDATORY GATE (brief phase 0, analysis_plan): validate the conformal
implementation on synthetic EXCHANGEABLE data where coverage is known to hold.

If empirical coverage does not match nominal within Monte Carlo error here, the
implementation is wrong and every downstream number in Project B is meaningless.

Design. Calibration and test are drawn i.i.d. from ONE generative process, so they
are exchangeable by construction and the split-conformal guarantee applies
regardless of how badly the classifier fits. That is the point: conformal validity
is a property of exchangeability, not of model quality. We therefore deliberately
include a MISSPECIFIED classifier arm -- if coverage held only for a good model,
the implementation would be relying on calibrated probabilities rather than on the
conformal construction.

WHAT THE THEORY ACTUALLY GUARANTEES, and therefore what may be gated on. An earlier
version of this file gated every arm on the two-sided band [1-alpha, 1-alpha+1/(n_cal+1)]
and reported FAIL for Mondrian and for APS. That band was the wrong criterion for
those two arms, and the failures were artefacts of the gate rather than defects in
the implementation. Recorded here because a validation that reports failures it
invented is worse than no validation: it trains the reader to discount real ones.

  * LOWER BOUND, universal. Coverage >= 1-alpha holds for ANY nonconformity score
    under exchangeability. This is the validity property and every arm is gated on it.
  * UPPER BOUND, tie-free scores only. Coverage <= 1-alpha+1/(n+1) requires the score
    distribution to be continuous. inverse_softmax on continuous probabilities is
    tie-free, so the MARGINAL/inverse_softmax arm is gated two-sided -- and that is
    the arm where an off-by-one in the quantile level would show, because the band is
    only 1/(n+1) wide. APS is NOT gated on it: the deterministic (non-randomised)
    APS variant is conservative by construction, so over-coverage there is the
    documented behaviour of the estimator, not an error.
  * MONDRIAN is governed by PER-CLASS calibration counts, not the total. With n_cal
    cells over K classes each quantile sees ~n_cal/K, so the attainable band is
    1/(n_k+1) wide -- roughly K times wider than the marginal band. Mondrian is
    therefore gated per class against its own n_k, which is the stronger and correct
    requirement.

NEGATIVE CONTROL. A validation that only confirms coverage holds cannot show the
harness would notice if it broke. The control makes classes DELIBERATELY unequal in
difficulty (one heavily overlapping class) and then shifts the test distribution
toward the hard class, so calibration and test are no longer exchangeable. Marginal
coverage must fall detectably below nominal. An earlier version shifted labels among
six symmetric, equally-separable classes; every class then had near-identical
coverage, reweighting them changed nothing, and the control could not have detected
any failure. Class heterogeneity is what makes the violation bite.

Emits results/projectB/U5_conformal_validation.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import conformal as CF

OUT = Path("/mnt/fm-bench/scfm-tme-benchmark/results/projectB")
ALPHAS = (0.20, 0.10, 0.05)
N_REPLICATES = 200
N_CLASSES = 6
N_FEATURES = 20


def _make_centres(rng, heterogeneous=False):
    """Class means. When `heterogeneous`, class 0 is placed near class 1 so it is
    genuinely hard, which is what lets the negative control detect a violation."""
    c = rng.normal(size=(N_CLASSES, N_FEATURES)) * 1.6
    if heterogeneous:
        c[0] = c[1] + rng.normal(size=N_FEATURES) * 0.15
    return c


def _gaussian_mixture(rng, n, centres, weights=None):
    y = rng.choice(len(centres), size=n, p=weights)
    X = centres[y] + rng.normal(size=(n, X_DIM := centres.shape[1]))
    return X, y


def _run_once(rng, alpha, n_cal, n_test, misspecified=False, shift=False):
    centres = _make_centres(rng, heterogeneous=shift)
    Xtr, ytr = _gaussian_mixture(rng, 2000, centres)
    # A misspecified head (heavily over-regularised, one feature) still yields a
    # valid conformal predictor: exchangeability, not calibration, is what matters.
    clf = (LogisticRegression(C=1e-4, max_iter=500).fit(Xtr[:, :1], ytr)
           if misspecified else
           LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, ytr))
    cols = slice(0, 1) if misspecified else slice(None)

    Xc, yc = _gaussian_mixture(rng, n_cal, centres)
    # NEGATIVE CONTROL: test is drawn with a different class distribution, weighted
    # toward the deliberately-hard class, so calibration and test are not exchangeable.
    w = None
    if shift:
        w = np.full(N_CLASSES, 0.10 / (N_CLASSES - 1))
        w[0] = 0.90
    Xt, yt = _gaussian_mixture(rng, n_test, centres, weights=w)

    Pc, Pt = clf.predict_proba(Xc[:, cols]), clf.predict_proba(Xt[:, cols])
    k = len(clf.classes_)
    out = {}
    for score in ("inverse_softmax", "aps"):
        sc = CF.SCORES[score](Pc, yc)
        # marginal
        q, _ = CF.marginal_threshold(sc, alpha)
        s = CF.prediction_sets(Pt, q, score)
        r = CF.summarise(s, yt, k)
        r["_n_cal_per_class"] = None
        out[("marginal", score)] = r
        # mondrian -- record the realised per-class calibration counts, because the
        # attainable coverage band is set by n_k, not by n_cal.
        qv, info = CF.mondrian_thresholds(sc, yc, k, alpha)
        s = CF.prediction_sets(Pt, qv, score)
        r = CF.summarise(s, yt, k)
        r["_n_cal_per_class"] = {d["class_idx"]: d["n_calibration"] for d in info}
        out[("mondrian", score)] = r
    return out


def gate0_exact_order_statistic(n_rep: int = 20000) -> tuple[bool, pd.DataFrame]:
    """GATE 0, the sharpest test in this file: on exchangeable SCALAR scores the
    threshold's coverage is known in closed form -- exactly k/(n+1) for
    k = ceil((n+1)(1-alpha)) -- with no classifier, no features and no mixture in
    the way. Anything wrong with the quantile shows here as a clean numeric
    disagreement rather than as a diffuse over-coverage that could be argued away.

    This gate is what caught the original off-by-one: np.quantile(method="higher")
    interpolates over n-1 intervals and returned the (k+1)-th order statistic.
    """
    rng = np.random.default_rng(7)
    rows = []
    for alpha in ALPHAS:
        for n in (10, 19, 26, 50, 200, 2000):
            cov = []
            for _ in range(n_rep):
                s = rng.normal(size=n + 1)          # n calibration + 1 test
                cov.append(s[n] <= CF.conformal_quantile(s[:n], alpha))
            emp = float(np.mean(cov))
            se = float(np.std(cov) / np.sqrt(n_rep))
            k = int(np.ceil((n + 1) * (1 - alpha)))
            exact = k / (n + 1) if k <= n else 1.0
            rows.append(dict(alpha=alpha, n_calibration=n, k_order_statistic=k,
                             exact_coverage=round(exact, 5),
                             empirical_coverage=round(emp, 5), mc_se=round(se, 5),
                             deviation=round(emp - exact, 5),
                             ok=bool(abs(emp - exact) <= 4 * se)))
    df = pd.DataFrame(rows)
    return bool(df.ok.all()), df


def main(argv=None) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    g0, g0df = gate0_exact_order_statistic()
    g0df.to_csv(OUT / "U5_conformal_validation_exact_quantile.csv", index=False)
    print(g0df.to_string(index=False))
    print(f"\nGATE 0 quantile : coverage == exact k/(n+1)  : {'PASS' if g0 else 'FAIL'}\n")
    rows = []
    for cond, kw in [("exchangeable", {}),
                     ("exchangeable_misspecified_head", {"misspecified": True}),
                     ("NEGATIVE_CONTROL_label_shift", {"shift": True})]:
        for alpha in ALPHAS:
            for n_cal in (200, 2000):
                acc = {}
                rng = np.random.default_rng(20260809)
                for _ in range(N_REPLICATES):
                    r = _run_once(rng, alpha, n_cal, 2000, **kw)
                    for key, v in r.items():
                        a = acc.setdefault(key, {"cov": [], "sz": [], "pc": [], "nk": []})
                        a["cov"].append(v["coverage"])
                        a["sz"].append(v["mean_set_size"])
                        # worst per-class coverage, and the smallest per-class
                        # calibration count that governs it
                        a["pc"].append(min(c["coverage"] for c in v["per_class"].values()))
                        if v["_n_cal_per_class"]:
                            nk = v["_n_cal_per_class"]
                            a["nk"].append(min(nk.values()))
                            # Mondrian's marginal coverage is the prevalence-weighted
                            # mixture of per-class coverages, each with its OWN band
                            # 1/(n_k+1). Averaging n_k first and taking one band from
                            # it understates the attainable ceiling whenever the n_k
                            # are unequal, which they always are for real class
                            # frequencies. Mix the per-class ceilings instead.
                            w = np.array([v["per_class"][k]["n"] for k in sorted(v["per_class"])],
                                         dtype=float)
                            ceil_k = np.array([1.0 / (nk.get(k, 0) + 1)
                                               for k in sorted(v["per_class"])])
                            a.setdefault("hi_mix", []).append(
                                float((w * ceil_k).sum() / w.sum()))
                for (variant, score), a in acc.items():
                    cov = float(np.mean(a["cov"]))
                    mc = float(np.std(a["cov"]) / np.sqrt(N_REPLICATES))
                    # The attainable upper bound is set by the count each quantile
                    # is estimated from: n_cal marginally, min_k n_k for Mondrian.
                    n_gov = int(np.mean(a["nk"])) if a["nk"] else n_cal
                    lo = 1 - alpha
                    hi = 1 - alpha + (float(np.mean(a["hi_mix"])) if a.get("hi_mix")
                                      else 1.0 / (n_cal + 1))
                    # Two-sided gating applies only to a tie-free score. The
                    # deterministic APS variant is conservative by construction.
                    two_sided = (score == "inverse_softmax")
                    covers_lo = cov >= lo - 3 * mc
                    within_hi = cov <= hi + 3 * mc
                    rows.append(dict(
                        condition=cond, variant=variant, score=score, alpha=alpha,
                        nominal=1 - alpha, n_calibration=n_cal,
                        n_governing_quantile=n_gov,
                        empirical_coverage=round(cov, 5),
                        mc_se=round(mc, 5),
                        band_lo=round(lo, 5), band_hi=round(hi, 5),
                        gated_two_sided=bool(two_sided),
                        meets_lower_bound=bool(covers_lo),
                        within_upper_bound=bool(within_hi),
                        min_per_class_coverage=round(float(np.mean(a["pc"])), 5),
                        min_n_cal_per_class=(int(np.mean(a["nk"])) if a["nk"] else None),
                        mean_set_size=round(float(np.mean(a["sz"])), 4),
                        n_replicates=N_REPLICATES))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "U5_conformal_validation.csv", index=False)

    exch = df[df.condition.str.startswith("exchangeable")]
    neg = df[df.condition.str.startswith("NEGATIVE")]

    # GATE 1 (validity): every exchangeable arm attains the LOWER bound. This is the
    # property the guarantee actually asserts and it binds on all four arms.
    g1 = bool(exch.meets_lower_bound.all())
    # GATE 2 (sharpness): the tie-free arm also respects the UPPER bound, i.e. it is
    # not conservative. This is where an off-by-one in the quantile level shows.
    tie_free = exch[exch.gated_two_sided]
    g2 = bool(tie_free.within_upper_bound.all())
    # GATE 3 (class-conditional): Mondrian holds coverage for the WORST class,
    # against the per-class count that governs its quantile.
    mond = exch[exch.variant == "mondrian"]
    g3 = bool((mond.min_per_class_coverage
               >= mond.nominal - 3.0 * np.sqrt(mond.nominal * (1 - mond.nominal)
                                               / mond.min_n_cal_per_class.clip(lower=1))).all())
    # GATE 4 (sensitivity): the negative control DOES lose coverage, so the harness
    # can detect the exchangeability failure B1 exists to measure.
    negm = neg[(neg.variant == "marginal") & (neg.score == "inverse_softmax")]
    g4 = bool((negm.empirical_coverage < negm.nominal - 0.01).any())

    show = ["condition", "variant", "score", "alpha", "n_calibration",
            "n_governing_quantile", "empirical_coverage", "band_lo", "band_hi",
            "meets_lower_bound", "within_upper_bound", "min_per_class_coverage",
            "mean_set_size"]
    print(df[show].to_string(index=False))
    print(f"\nGATE 0 quantile  : coverage == exact k/(n+1)                   : {'PASS' if g0 else 'FAIL'}")
    print(f"GATE 1 validity  : all exchangeable arms attain 1-alpha        : {'PASS' if g1 else 'FAIL'}")
    print(f"GATE 2 sharpness : tie-free arm within 1-alpha+1/(n+1)         : {'PASS' if g2 else 'FAIL'}")
    print(f"GATE 3 mondrian  : worst-class coverage at nominal             : {'PASS' if g3 else 'FAIL'}")
    print(f"GATE 4 sensitivity: negative control detects non-exchangeability: {'PASS' if g4 else 'FAIL'}")
    ok = g0 and g1 and g2 and g3 and g4
    print(f"\nCONFORMAL IMPLEMENTATION VALIDATION: {'PASS' if ok else 'FAIL'}")
    print(f"-> {OUT / 'U5_conformal_validation.csv'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
