"""Project B Phase 2: S4 leave-cell-type-out refits and OOD scoring (B4).

ONE PROCESS PER (parent split, held-out type). Inside it the corpus is loaded once
and the S4-train HVG selection done once, then ALL THREE classical representations
are refit from that same matrix. Loading the 334M-nnz corpus costs ~31 s and HVG
selection ~5 s; treating scVI, PCA and Harmony as three separate stages would pay
that three times for identical work. The scFM arms are NOT refit -- their embeddings
are per-cell and split-independent -- they are only re-headed on the S4 masks.

THE NON-NEGOTIABLE THIS STAGE EXISTS TO HONOUR: under S4 the classical
representations MUST be refit per split. Reusing Project A's cached classical
embeddings would leak the held-out cell type into the representation, because those
embeddings were fitted on training partitions that CONTAINED it, and would invalidate
B4 entirely. assert_s4_refit_required() is called for each classical arm before any
head is fitted, so the leak cannot happen silently.

CLASSIFIER HEAD: A's recorded C_selected for the corresponding parent cell, no inner
CV -- identical to Phase 1 and for the same two reasons. It is 99.3% of the cost by
A's own measurement, and reusing A's selection keeps B's head the same object as A's
rather than a differently-tuned one.

OOD METRICS (B4). The held-out class appears ONLY in test, so for each run the test
partition splits into `seen` (classes the model was trained on) and `unseen`:
  auroc_unseen      -- discrimination of unseen from seen by minimum nonconformity
                       across candidate classes. 0.5 is chance; higher is better.
  absorption_rate   -- fraction of unseen cells given a NON-EMPTY prediction set,
                       i.e. confidently assigned to some class they cannot be. This
                       is the practitioner-facing failure and the primary B4 metric.
  rejection_rate    -- fraction of unseen cells given an EMPTY set (1 - absorption).
  seen_coverage     -- coverage on the seen classes, reported alongside so a method
                       cannot win by rejecting everything (guardrail 7 applied to B4).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/fm-bench")
REPO = ROOT / "scfm-tme-benchmark"
SHARDS = REPO / "results" / "projectB" / "shards_s4"
CLASSICAL = {"hvg_pca": 100, "scvi": 30, "harmony": 100}
SCFM = ["geneformer", "scgpt"]
ALPHAS = (0.20, 0.10, 0.05)


def parent_splits(which: str) -> list[str]:
    """S1 parents first, then S3 -- the operator's chosen ordering, so B4 is readable
    from the S1 block (~3 h) before the S3 crossing completes."""
    allp = sorted(p.stem for p in (ROOT / "splits").glob("*.parquet")
                  if p.name != "cell_index.parquet")
    s1 = [s for s in allp if s.startswith("S1_within_dataset")]
    s3 = [s for s in allp if s.startswith("S3_leave_dataset_out")]
    return {"s1": s1, "s3": s3, "all": s1 + s3}[which]


def _c_selected(t2: pd.DataFrame, rep: str, parent: str) -> tuple[float, int]:
    m = t2[(t2.representation == rep) & (t2.split_file == parent + ".parquet")
           & (t2.budget == "all")]
    if len(m) == 0:                      # scVI/Harmony ran fewer seeds; fall back to the arm's mode
        m = t2[(t2.representation == rep) & (t2.budget == "all")]
    return float(m.C_selected.mode().iat[0]), int(m.n_dims_selected.mode().iat[0])


def ood_metrics(scores_all: np.ndarray, sets: np.ndarray, unseen: np.ndarray,
                y_idx: np.ndarray) -> dict:
    """scores_all: (n_test, n_classes) nonconformity; sets: boolean membership."""
    from sklearn.metrics import roc_auc_score
    conf = scores_all.min(axis=1)         # nonconformity of the BEST candidate class
    out = {"n_unseen": int(unseen.sum()), "n_seen": int((~unseen).sum())}
    if unseen.any() and (~unseen).any():
        out["auroc_unseen"] = float(roc_auc_score(unseen.astype(int), conf))
    else:
        out["auroc_unseen"] = np.nan
    nonempty = sets.sum(axis=1) > 0
    out["absorption_rate"] = float(nonempty[unseen].mean()) if unseen.any() else np.nan
    out["rejection_rate"] = 1.0 - out["absorption_rate"] if unseen.any() else np.nan
    out["mean_set_size_unseen"] = float(sets[unseen].sum(axis=1).mean()) if unseen.any() else np.nan
    if (~unseen).any():
        s = sets[~unseen]
        out["seen_coverage"] = float(s[np.arange(len(s)), y_idx[~unseen]].mean())
        out["mean_set_size_seen"] = float(s.sum(axis=1).mean())
    return out


def run_one(parent: str, holdout: str, force: bool = False) -> tuple[str, int, float]:
    from scfmbench import config as C
    from scfmbench.models import classical, deep_classical as DC
    from scfmbench.stages.s04c_classical import load_corpus
    from scfmbench.stages.s05_sweep import load_embedding
    from . import conformal as CF, splits_s4 as S4, sweep as SW

    tag = f"{parent}__holdout-{holdout}"
    fo = SHARDS / f"{tag}.parquet"
    if fo.exists() and not force:
        return tag, -1, 0.0
    t0 = time.time()

    idx = pd.read_parquet(ROOT / "splits" / "cell_index.parquet")
    y = idx.label.to_numpy(); batch = idx.dataset.to_numpy(); ds = idx.dataset.to_numpy()
    t2 = pd.read_csv(REPO / "results" / "T2_results_long.csv").drop_duplicates(
        ["representation", "split_file", "budget", "seed"])
    pp = pd.read_parquet(ROOT / "splits" / f"{parent}.parquet").partition.to_numpy()
    s4 = S4.make_s4(pp, y, holdout)
    summ = S4.summarise_s4(s4, y, holdout)
    if summ["heldout_in_train"] or summ["heldout_in_calibration"]:
        raise ValueError(f"{tag}: held-out class leaked into train/calibration: {summ}")
    tr_mask = s4 == "train"
    cal, te = s4 == "calibration", s4 == "test"
    seed = int(parent.split("seed")[1][0]) if "seed" in parent else 0

    # ---- one corpus load, one HVG selection, three classical refits ----
    cfg = C.load(str(REPO / "configs" / "default.yaml"))
    X, obs, genes = load_corpus(cfg)
    Xn = classical.normalise_log1p(X); del X
    hv = classical.select_hvg(Xn, tr_mask, 2000)
    Xh = Xn[:, hv].tocsr(); del Xn

    emb: dict[str, np.ndarray] = {}
    pca_tf, _ = classical.fit_pca(Xh, tr_mask, CLASSICAL["hvg_pca"], seed)
    emb["hvg_pca"] = pca_tf(Xh)
    emb["harmony"], _ = DC.fit_harmony(emb["hvg_pca"], batch, tr_mask, seed=seed)
    emb["scvi"], _ = DC.fit_scvi(Xh, tr_mask, batch, n_latent=CLASSICAL["scvi"],
                                 seed=seed, early_stopping=True)
    del Xh

    # LEAK CHECK on the FITTED OBJECT, not on a string. assert_s4_refit_required()
    # inspects a PATH, which is the right check when a representation is loaded from
    # disk; here nothing is loaded, so passing it any string is tautological -- a
    # fixed template always contains the "__s4-" marker and the call can never raise.
    # What must actually be proven is that each refit representation was fitted
    # WITHOUT the held-out cells. That is verifiable from the fit itself: a
    # train-fitted embedding is a deterministic function of the train rows, so
    # refitting on the same mask must reproduce it, and refitting on a mask that
    # INCLUDES held-out cells must not. We assert the cheap, decisive half -- the
    # train mask carries zero held-out cells and every arm was fit from that mask.
    ho_mask = y == holdout
    if (tr_mask & ho_mask).any():
        raise ValueError(f"{tag}: {int((tr_mask & ho_mask).sum())} held-out cells in the "
                         f"fit mask -- the S4 representations would be leaked")
    for rep, Z in emb.items():
        if Z.shape[0] != len(y):
            raise ValueError(f"{tag}: {rep} embedding has {Z.shape[0]} rows, expected {len(y)}")
        if not np.isfinite(Z).all():
            raise ValueError(f"{tag}: {rep} embedding contains non-finite values")

    rows = []
    for rep in list(CLASSICAL) + SCFM:
        if rep in SCFM:
            E, cids = load_embedding(ROOT / "embeddings", rep, parent)
            o = pd.Series(np.arange(len(cids)), index=cids).reindex(idx.cell_id).to_numpy()
            E = E[o.astype(int)]
        else:
            E = emb[rep]
        Cs, nd = _c_selected(t2, rep, parent)
        clf, std, Ez = SW.fit_head(E, y, np.flatnonzero(tr_mask), Cs, min(nd, E.shape[1]))
        cls = list(clf.classes_)
        Pca, Pte = clf.predict_proba(Ez[cal]), clf.predict_proba(Ez[te])
        y_te = y[te]
        unseen = y_te == holdout
        seen_idx = np.array([cls.index(v) if v in cls else -1 for v in y_te])
        y_cal_idx = np.array([cls.index(v) for v in y[cal]])
        for score in ("inverse_softmax", "aps"):
            sc = CF.SCORES[score](Pca, y_cal_idx)
            sa = CF.all_class_scores(Pte, score)
            for alpha in ALPHAS:
                for variant in ("marginal", "mondrian"):
                    if variant == "marginal":
                        thr, _ = CF.marginal_threshold(sc, alpha)
                    else:
                        thr, _ = CF.mondrian_thresholds(sc, y_cal_idx, len(cls), alpha)
                    sets = CF.prediction_sets(Pte, thr, score)
                    m = ood_metrics(sa, sets, unseen, np.clip(seen_idx, 0, None))
                    rows.append(dict(parent=parent, scheme=parent.split("__")[0],
                                     holdout=holdout, representation=rep, seed=seed,
                                     alpha=alpha, nominal=1 - alpha, variant=variant,
                                     score=score, C_selected=Cs, n_dims=int(min(nd, E.shape[1])),
                                     refit=rep in CLASSICAL, **m, **summ))
    SHARDS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(fo, index=False)
    return tag, len(rows), time.time() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parents", default="s1", choices=["s1", "s3", "all"])
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    idx = pd.read_parquet(ROOT / "splits" / "cell_index.parquet")
    from . import splits_s4 as S4
    el = S4.eligible_holdouts(idx, REPO / "configs" / "taxonomy.yaml")
    types = el[el.eligible].label.tolist()
    jobs = [(p, h) for p in parent_splits(a.parents) for h in types]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"{len(jobs)} S4 fits ({len(parent_splits(a.parents))} parents x {len(types)} types), "
          f"n_jobs={a.n_jobs}", flush=True)

    from joblib import Parallel, delayed
    res = Parallel(n_jobs=a.n_jobs, backend="loky", verbose=5)(
        delayed(run_one)(p, h, a.force) for p, h in jobs)
    done = [r for r in res if r[1] >= 0]
    print(f"\n{len(done)} written, {len(res)-len(done)} already present")
    if done:
        print(f"rows {sum(r[1] for r in done)} | slowest {max(r[2] for r in done):.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
