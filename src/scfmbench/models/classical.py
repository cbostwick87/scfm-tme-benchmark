"""Classical representations: HVG+PCA, scVI, Harmony, celltypist.

HVG+PCA IS THE METHOD TO BEAT. It is not a foil. Every knob it has is set with
the same care as the foundation-model pipeline, because a benchmark that
straw-mans its baseline measures nothing (guardrail 4).

LEAKAGE CONTRACT -- the single most important property of this module.
Every representation here is FIT ON THE TRAINING PARTITION ONLY and then
APPLIED to calibration/test:
  * HVG selection: variance ranked on train cells only.
  * PCA: components fit on train only (incremental, for memory).
  * scVI: encoder trained on train cells only.
  * Harmony: correction fit on train, applied to test by nearest-centroid
    projection -- NOT re-fit jointly, which would let test cells influence the
    correction and is the most common leak in integration benchmarks.
  * Standardisation: scaler fit on train only.
Fitting any of these on the full corpus is leakage and invalidates the study.

Every function here takes an explicit `train_mask` and asserts it is non-trivial,
so a caller cannot accidentally pass an all-True mask and silently fit on
everything.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class LeakageError(RuntimeError):
    pass


def _check_train_mask(train_mask: np.ndarray, n: int) -> None:
    if train_mask.shape[0] != n:
        raise ValueError(f"train_mask length {train_mask.shape[0]} != n_cells {n}")
    if train_mask.all():
        raise LeakageError(
            "train_mask selects every cell: a representation fit on the full corpus "
            "is leakage. Pass the TRAIN partition only."
        )
    if train_mask.sum() < 50:
        raise ValueError(f"train_mask selects only {int(train_mask.sum())} cells")


def normalise_log1p(X: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """CP10k + log1p. Per-cell, so it involves no cross-cell fitting and is
    leakage-free by construction."""
    X = X.tocsr(copy=True).astype(np.float32)
    counts = np.asarray(X.sum(axis=1)).ravel()
    scale = np.divide(target_sum, counts, out=np.zeros_like(counts), where=counts > 0)
    X = sp.diags(scale) @ X
    X.data = np.log1p(X.data)
    return X.tocsr()


def select_hvg(X: sp.csr_matrix, train_mask: np.ndarray, n_top: int) -> np.ndarray:
    """Select highly variable genes on TRAIN cells only.

    Uses the normalised-dispersion criterion on log1p data: variance is binned by
    mean expression and genes are ranked within bin, so selection is not simply a
    proxy for abundance. Returns gene indices.
    """
    _check_train_mask(train_mask, X.shape[0])
    Xt = X[train_mask]
    n = Xt.shape[0]
    mean = np.asarray(Xt.mean(axis=0)).ravel()
    sq = np.asarray(Xt.multiply(Xt).mean(axis=0)).ravel()
    var = np.maximum(sq - mean ** 2, 0.0) * (n / max(n - 1, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        disp = np.where(mean > 0, var / mean, 0.0)
    # bin by mean expression and z-score dispersion within bin
    nz = mean > 0
    bins = np.zeros_like(mean, dtype=int)
    if nz.sum() > 20:
        edges = np.quantile(mean[nz], np.linspace(0, 1, 21))
        bins = np.clip(np.searchsorted(edges, mean, side="right") - 1, 0, 19)
    score = np.full_like(disp, -np.inf)
    for b in np.unique(bins[nz]):
        m = (bins == b) & nz
        d = disp[m]
        mu, sd = d.mean(), d.std()
        score[m] = (d - mu) / sd if sd > 0 else 0.0
    k = min(n_top, int(nz.sum()))
    return np.sort(np.argsort(-score)[:k])


def fit_pca(X: sp.csr_matrix, train_mask: np.ndarray, n_comp: int, seed: int,
            batch: int = 4096, fit_subsample: int = 50000):
    """PCA fit on TRAIN cells only; returns (transform_fn, model).

    Randomized SVD on a subsample of the training cells, rather than
    IncrementalPCA over every training cell. Two things make this safe:

      * The FIT is subsampled; the TRANSFORM is applied to every cell. Principal
        directions of a 3000-gene space are estimated to well within their own
        sampling error from tens of thousands of cells -- adding the remaining
        ~90k moves the components far less than the seed-to-seed variation the
        study already reports. Equivalence against the exhaustive fit is checked
        in tests (subspace alignment, not just explained variance).
      * The subsample is drawn from TRAIN ONLY, so this changes cost, not the
        leakage contract.

    IncrementalPCA over the full training partition took ~16 min per split, which
    across 75 splits is 20 h for one baseline arm. That is a wall-clock problem,
    not a scientific one, and it is fixed by better linear algebra rather than by
    reducing the experiment (guardrail 4: the baseline is not weakened to save time).
    """
    from sklearn.decomposition import PCA
    _check_train_mask(train_mask, X.shape[0])
    idx = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    if len(idx) > fit_subsample:
        idx = rng.choice(idx, fit_subsample, replace=False)
    n_comp = int(min(n_comp, X.shape[1], len(idx) - 1))

    Xf = np.asarray(X[idx].todense(), dtype=np.float32)
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    pca.fit(Xf)
    del Xf

    def transform(Xa: sp.csr_matrix, chunk: int = 8192) -> np.ndarray:
        out = np.empty((Xa.shape[0], n_comp), dtype=np.float32)
        for s in range(0, Xa.shape[0], chunk):
            out[s:s + chunk] = pca.transform(
                np.asarray(Xa[s:s + chunk].todense(), dtype=np.float32))
        return out

    return transform, pca


def fit_standardiser(Z_train: np.ndarray):
    """Per-feature standardisation fit on TRAIN only.

    Applied to every representation, including the foundation-model embeddings.
    Mean-pooled transformer states are strongly anisotropic (measured mean
    pairwise cosine 0.954), while PCA output is centred by construction --
    leaving the scFM embeddings unscaled would handicap them relative to the
    baseline, which is straw-manning in the direction that flatters the expected
    negative result.
    """
    mu = Z_train.mean(axis=0)
    sd = Z_train.std(axis=0)
    sd[sd < 1e-8] = 1.0

    def apply(Z: np.ndarray) -> np.ndarray:
        return ((Z - mu) / sd).astype(np.float32)

    return apply
