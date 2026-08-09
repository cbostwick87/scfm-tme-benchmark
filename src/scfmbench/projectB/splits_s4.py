"""S4 leave-cell-type-out splits -- NEW IN PROJECT B (brief ood_arm).

One immune cell type is removed from BOTH train and calibration; it appears ONLY in
test. Everything else follows the parent scheme, so S4 is a MODIFIER on an existing
split rather than a new partitioning of the corpus:

    S4(S1) -- unseen type, no dataset shift        (clean OOD detection)
    S4(S3) -- unseen type AND held-out dataset     (realistic deployment case)

Building S4 by modifying A's persisted splits, rather than re-partitioning, is what
keeps B's cells corresponding exactly to A's: every seen cell keeps the partition
assignment A gave it, so a B result is attributable to the same underlying split.

ELIGIBILITY (brief): present in >= 5 datasets, and removal leaves >= 6 classes in the
training taxonomy. Restricted to the IMMUNE compartment -- the brief's arm is about
unseen immune cell types, and the non-immune context classes (Malignant, Fibroblast,
Endothelial) are background rather than annotation targets.

LEAKAGE (brief critical_implementation_warning): the foundation-model embeddings are
per-cell and split-independent, so they are reused directly under S4. HVG+PCA, scVI
and Harmony are FIT to a training partition and MUST be refit against the S4 training
mask. Reusing A's classical embeddings here would leak the held-out cell type into
the representation and invalidate B4. `assert_s4_refit_required` exists to make that
substitution fail loudly rather than silently.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MIN_DATASETS = 5
MIN_REMAINING_CLASSES = 6


def immune_classes(taxonomy_path: Path | None = None) -> list[str]:
    """The immune WORKING classes, taken from labels.IMMUNE_CLASSES.

    Deliberately NOT read from taxonomy.yaml's `compartment` field. That file lists
    the pre-harmonisation target names, where Monocyte and Macrophage appear
    separately; labels.py merges them into the single working class `Mono_Macro`,
    which therefore has no entry in taxonomy.yaml and read as non-immune by name
    matching. Mono_Macro is 29,649 cells across 10 datasets and is eligible for S4 --
    excluding it on a naming artefact would have silently narrowed B4's held-out set
    from 7 types to 6, weakening the arm with no scientific reason and no error.
    labels.IMMUNE_CLASSES is the definition A actually harmonised to and is the only
    authoritative one.
    """
    from scfmbench.labels import IMMUNE_CLASSES
    return list(IMMUNE_CLASSES)


def eligible_holdouts(index: pd.DataFrame, taxonomy_path: Path) -> pd.DataFrame:
    """Every working class with its eligibility verdict and the reason."""
    imm = set(immune_classes(taxonomy_path))
    n_classes = index["label"].nunique()
    rows = []
    for lab, g in index.groupby("label", observed=True):
        nds = g["dataset"].nunique()
        is_imm = lab in imm
        remaining = n_classes - 1
        reasons = []
        if not is_imm:
            reasons.append("non-immune context class")
        if nds < MIN_DATASETS:
            reasons.append(f"present in {nds} datasets (< {MIN_DATASETS})")
        if remaining < MIN_REMAINING_CLASSES:
            reasons.append(f"only {remaining} classes would remain")
        rows.append(dict(label=lab, compartment="immune" if is_imm else "non_immune",
                         n_datasets=nds, n_cells=len(g),
                         prevalence=len(g) / len(index),
                         eligible=not reasons,
                         reason="; ".join(reasons) if reasons else "eligible"))
    return pd.DataFrame(rows).sort_values(["eligible", "n_cells"], ascending=[False, False])


def make_s4(parent_partition: np.ndarray, labels: np.ndarray,
            held_out: str) -> np.ndarray:
    """Derive an S4 partition from a parent split.

    Cells of the held-out type that the parent assigned to TRAIN or CALIBRATION are
    reassigned to a fourth value, "excluded" -- NOT to test. Moving them to test
    would inflate the unseen-type population with cells the parent scheme never
    intended to evaluate, and under S3 would import cells from training datasets into
    a test set defined as one held-out study.
    """
    p = parent_partition.astype(object).copy()
    hit = labels == held_out
    p[hit & (p != "test")] = "excluded"
    return p.astype(str)


def assert_s4_refit_required(representation: str, embedding_path: str | Path) -> None:
    """Fail loudly if a FIT representation is read from A's cache under an S4 split.

    A's classical embeddings are named after the parent split (S1.../S3...), so a
    path lacking an s4 marker is A's artefact and using it here is the leak the brief
    names as invalidating B4 entirely.
    """
    if representation in {"geneformer", "scgpt"}:
        return
    if "__s4-" not in str(embedding_path):
        raise ValueError(
            f"{representation!r} is FIT to a training partition and must be refit "
            f"under each S4 split; got {embedding_path!r}, which is Project A's "
            f"parent-split artefact. Using it would leak the held-out cell type into "
            f"the representation and invalidate B4."
        )


def summarise_s4(part: np.ndarray, labels: np.ndarray, held_out: str) -> dict:
    tr, ca, te, ex = (part == "train"), (part == "calibration"), (part == "test"), (part == "excluded")
    ho = labels == held_out
    return {"held_out_type": held_out,
            "n_train": int(tr.sum()), "n_calibration": int(ca.sum()),
            "n_test": int(te.sum()), "n_excluded": int(ex.sum()),
            "heldout_in_train": int((ho & tr).sum()),        # MUST be 0
            "heldout_in_calibration": int((ho & ca).sum()),  # MUST be 0
            "heldout_in_test": int((ho & te).sum()),
            "n_classes_train": int(len(np.unique(labels[tr]))),
            "unseen_frac_of_test": float((ho & te).sum() / max(te.sum(), 1))}
