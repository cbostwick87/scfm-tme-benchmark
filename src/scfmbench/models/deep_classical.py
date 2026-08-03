"""scVI and Harmony representations, fit per split on the TRAINING partition.

These are the two baselines the brief marks MANDATORY alongside HVG+PCA, and
H2 is stated as scFMs "match or exceed scVI and Harmony". They were missing
from the first pass; an external review caught it.

------------------------------------------------------------------------
COUNTS: TISCH2 DOES NOT DISTRIBUTE RAW UMI COUNTS, AND scVI NEEDS THEM
------------------------------------------------------------------------
Measured, not assumed: every TISCH2 matrix is log1p(CP10K) -- per-cell
sum(expm1(x)) is exactly 10,000 to floating point. scVI's negative-binomial
likelihood requires integer counts; feeding it log-normalised data is silent
garbage rather than an error.

Testing whether counts are invertible gave a split answer. For 9 of the 13
datasets every value in a cell is an exact integer multiple of that cell's
smallest nonzero value, so true UMI counts ARE exactly recoverable as
round(expm1(x) / min_nonzero) with implied library sizes of 1.3k-4.1k. For
the other 4 they are not: implied library sizes of 337k, 297k and 49M mark
these as plate-based TPM, where integer counts do not exist to recover.

So a uniform construction is used for ALL datasets:

    pseudo_counts = round(expm1(x))          # counts per 10,000

Why uniform rather than true counts where available: the alternative gives
scVI genuine counts on 9 datasets and degraded input on 4, which would
handicap scVI on those 4 specifically. Since the dataset is the unit of
replication, that would put a per-dataset artefact directly into the paired
contrast. Treating every dataset identically keeps the confound out of the
comparison at the cost of a known, declared approximation.

CONSEQUENCE, stated plainly for the report: under this construction every
cell has library size ~10,000 by construction, so scVI's library-size latent
carries no information. scVI is therefore evaluated on depth-normalised
pseudo-counts, not raw counts, and its numbers should be read as a lower
bound on what scVI would achieve with true counts. This is a limitation of
the DATA SOURCE, not of scVI.

------------------------------------------------------------------------
HARMONY IS TRANSDUCTIVE AND CANNOT HONOUR THE LEAKAGE CONTRACT AS WRITTEN
------------------------------------------------------------------------
The brief requires Harmony be "fit on the TRAINING partition only, then
applied to calibration and test". harmonypy 2.0 exposes no transform or
project method (verified: its Harmony class has K, R, Y, Z_corr, Z_cos,
Z_orig and no projection API). Harmony corrects the cells it is given; there
is no out-of-sample extension in the package.

Three options and why the third was taken:
  1. Fit on train, leave test uncorrected -- train and test would then live
     in different spaces, crippling Harmony for a reason that has nothing to
     do with its merits. That is straw-manning a mandatory baseline.
  2. Implement a Symphony-style projection (soft-assign test cells to the
     learned centroids and apply the per-cluster batch correction). Correct
     in principle, but harmonypy does not expose the per-cluster correction
     terms, so they would have to be recovered by regression -- a bespoke
     reimplementation whose subtle errors would look like results.
  3. Run Harmony TRANSDUCTIVELY on the PCA embedding of train and test
     together, using only the batch key and NEVER any label.

Option 3 is what is implemented, and it is a DECLARED DEVIATION from the
leakage control, not an oversight. Two things must be said about it in the
report:

  * It ADVANTAGES Harmony, because Harmony sees the test cells' expression
    (not their labels) while every other arm does not. Harmony is a
    BASELINE, so this biases toward "classical methods win" -- which is the
    study's own prior expectation, and therefore the most dangerous
    direction for an artefact to point. It must not be read as evidence for
    H1.
  * Under S3 the held-out dataset is a batch Harmony has never seen in any
    inductive sense. Transductive fitting is the only way Harmony can
    correct it at all, and this is precisely the practical situation Harmony
    is used in. That makes the comparison realistic rather than merely
    permissive.

PCA is still fit on the training partition only; only the Harmony correction
step sees test coordinates.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .classical import LeakageError, _check_train_mask


def to_pseudo_counts(X) -> sp.csr_matrix:
    """log1p(CP10K) -> integer pseudo-counts per 10,000. See module docstring.

    Applied identically to every dataset so no dataset is differentially
    advantaged, which matters because the dataset is the unit of replication.
    """
    X = sp.csr_matrix(X)
    out = X.copy().astype(np.float64)
    out.data = np.rint(np.expm1(out.data))
    out.eliminate_zeros()
    return sp.csr_matrix(out.astype(np.float32))


def fit_scvi(X, train_mask: np.ndarray, batch_labels, n_latent: int = 30,
             max_epochs: int = 200, seed: int = 0, early_stopping: bool = True):
    """scVI latent space. TRAINED ON TRAIN CELLS ONLY, then encodes all cells.

    scVI is genuinely inductive: the encoder is a function of expression, so
    held-out cells are ENCODED rather than refit. That satisfies the leakage
    control exactly -- unlike Harmony, no compromise is needed here.

    Returns (Z_all, info).
    """
    _check_train_mask(train_mask, X.shape[0])
    import anndata as ad
    import pandas as pd
    import scvi
    import torch

    scvi.settings.seed = seed
    counts = to_pseudo_counts(X)

    obs = pd.DataFrame({"batch": np.asarray(batch_labels, dtype=object)},
                       index=[f"c{i}" for i in range(X.shape[0])])
    adata = ad.AnnData(X=counts, obs=obs)
    adata.layers["counts"] = adata.X.copy()

    tr = adata[train_mask].copy()
    # A batch present in test but absent from train cannot be a scVI batch
    # covariate -- under S3 the held-out dataset is exactly that. Encoding
    # such cells requires a batch the model never saw, so the batch covariate
    # is dropped whenever train does not cover every batch. Silently mapping
    # them to an arbitrary seen batch would fabricate a correction.
    train_batches = set(tr.obs["batch"].unique())
    all_batches = set(adata.obs["batch"].unique())
    use_batch = train_batches >= all_batches and len(train_batches) > 1
    bkey = "batch" if use_batch else None

    scvi.model.SCVI.setup_anndata(tr, layer="counts", batch_key=bkey)
    model = scvi.model.SCVI(tr, n_latent=n_latent)
    accel = "gpu" if torch.cuda.is_available() else "cpu"
    model.train(max_epochs=max_epochs, early_stopping=early_stopping,
                accelerator=accel, devices=1, enable_progress_bar=False)

    # encode EVERY cell with the train-fitted model
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=bkey)
    Z = model.get_latent_representation(adata).astype(np.float32)

    return Z, {"n_latent": int(n_latent), "batch_covariate_used": bool(use_batch),
               "n_train_cells": int(train_mask.sum()),
               "epochs_run": int(model.trainer.current_epoch),
               "accelerator": accel}


def fit_harmony(Z_pca: np.ndarray, batch_labels, train_mask: np.ndarray,
                seed: int = 0, max_iter: int = 10):
    """Harmony correction of a train-fitted PCA embedding.

    TRANSDUCTIVE BY NECESSITY -- see the module docstring. `train_mask` is
    taken and recorded but Harmony is run over all rows, because harmonypy
    has no out-of-sample projection. The flag in the returned info dict is
    what the report cites; it is not decoration.
    """
    _check_train_mask(train_mask, Z_pca.shape[0])
    import pandas as pd
    from harmonypy import run_harmony

    b = np.asarray(batch_labels, dtype=object)
    if len(np.unique(b)) < 2:
        # one batch: Harmony has nothing to correct and would divide by zero
        return Z_pca.astype(np.float32), {
            "transductive": True, "n_batches": 1, "skipped": True,
            "reason": "single batch in this split; correction is undefined"}

    meta = pd.DataFrame({"batch": b})
    np.random.seed(seed)
    ho = run_harmony(Z_pca.astype(np.float64), meta, ["batch"],
                     max_iter_harmony=max_iter)
    # harmonypy 2.0 returns Z_corr as (n_components, n_cells) -- the TRANSPOSE of
    # its input. Asserted rather than assumed: if a future version returns
    # (n_cells, n_components) instead, a silent transpose would swap cells with
    # components and produce a plausible-looking matrix of nonsense.
    Z = np.asarray(ho.Z_corr)
    if Z.shape == (Z_pca.shape[1], Z_pca.shape[0]):
        Z = Z.T
    elif Z.shape != Z_pca.shape:
        raise RuntimeError(
            f"harmonypy returned Z_corr with shape {Z.shape}; expected "
            f"{Z_pca.shape} or its transpose. Refusing to guess an orientation.")
    Z = np.ascontiguousarray(Z).astype(np.float32)
    if Z.shape != Z_pca.shape:
        raise RuntimeError(f"harmony output {Z.shape} != input {Z_pca.shape}")
    return Z, {"transductive": True, "n_batches": int(len(np.unique(b))),
               "skipped": False,
               "n_train_cells": int(train_mask.sum()),
               "leakage_note": "Harmony saw test-cell EXPRESSION (never labels); "
                               "declared deviation from the train-only control, "
                               "and it advantages this baseline"}
