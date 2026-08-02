"""Evaluation metrics.

PRIMARY endpoint: macro-F1 over classes present in BOTH the train and test
partition of the contrast being evaluated ("learnable" classes). A class absent
from training cannot be predicted by ANY representation, so including it adds an
identical constant penalty to every arm and dilutes the contrast the study
exists to measure -- it reports corpus composition, not representation quality.

The all-test-classes variant is computed alongside and reported as a SECONDARY
metric, so the deployment-realistic number is never hidden and a reader can see
whether the choice changed a conclusion. Both go into T2/T3, with the class set
and its size recorded per row.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             train_classes: np.ndarray) -> dict:
    test_classes = np.unique(y_true)
    learnable = np.intersect1d(test_classes, np.unique(train_classes))
    out = {
        "n_test": int(len(y_true)),
        "n_classes_test": int(len(test_classes)),
        "n_classes_learnable": int(len(learnable)),
        "n_classes_test_only": int(len(np.setdiff1d(test_classes, train_classes))),
        "classes_learnable": ";".join(map(str, learnable)),
    }
    # PRIMARY: restricted to learnable classes, computed on the cells belonging
    # to them. Cells of test-only classes are excluded from the primary metric
    # (they are unpredictable by construction) but counted above.
    m = np.isin(y_true, learnable)
    if m.sum() > 0 and len(learnable) > 0:
        out["macro_f1"] = float(f1_score(y_true[m], y_pred[m],
                                         labels=learnable, average="macro", zero_division=0))
        out["balanced_acc"] = float(balanced_accuracy_score(y_true[m], y_pred[m]))
        out["per_class_f1"] = {c: float(v) for c, v in zip(
            learnable, f1_score(y_true[m], y_pred[m], labels=learnable,
                                average=None, zero_division=0))}
    else:
        out["macro_f1"] = float("nan"); out["balanced_acc"] = float("nan")
        out["per_class_f1"] = {}
    # SECONDARY: every class in the test partition, absent-from-train scoring 0.
    out["macro_f1_all_test_classes"] = float(f1_score(
        y_true, y_pred, labels=test_classes, average="macro", zero_division=0))
    out["accuracy"] = float((y_true == y_pred).mean())
    return out


def rarity_strata(train_labels: np.ndarray, edges=(0.01, 0.05)) -> dict:
    """Assign each class to a rarity stratum by its TRAIN prevalence (H3).

    Strata are defined on the training partition, not the test partition: rarity
    that matters for annotation is how little evidence the classifier had, not how
    few cells happen to appear at evaluation time.
    """
    cls, cnt = np.unique(train_labels, return_counts=True)
    frac = cnt / cnt.sum()
    out = {}
    for c, f in zip(cls, frac):
        out[c] = "rare" if f < edges[0] else ("uncommon" if f < edges[1] else "common")
    return out

def evaluate_per_dataset(y_true, y_pred, y_train, datasets):
    """Score each dataset separately within one run.

    THIS IS WHAT MAKES THE DATASET THE UNIT OF REPLICATION (guardrail 5).

    Under S1 and S2 a single split spans every dataset, so a run produces one
    pooled macro-F1 and the only thing left to aggregate over is the split seed.
    Testing across split seeds would present 5 resamplings of the same corpus as
    5 replication units -- the same error the guardrail forbids, in a subtler
    form than counting seeds x datasets as one big n. Scoring per dataset gives
    the paired test 13 genuine units under every scheme.

    Learnable classes are resolved PER DATASET: a class present in training but
    absent from dataset X's test cells is not scoreable there, and counting it
    against X would measure that dataset's composition rather than the
    representation. This is the pre-specified primary metric applied at the unit
    of analysis rather than to the pooled pool.

    The classifier fit is untouched -- only the scoring of already-made
    predictions is partitioned -- so this costs essentially nothing.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    datasets = np.asarray(datasets)
    train_classes = set(np.unique(y_train).tolist())
    out = {}
    for ds in np.unique(datasets):
        m = datasets == ds
        yt, yp = y_true[m], y_pred[m]
        test_classes = set(np.unique(yt).tolist())
        learnable = sorted(train_classes & test_classes)
        if not learnable:
            continue
        out[str(ds)] = {
            "n_test": int(m.sum()),
            "n_classes_learnable": len(learnable),
            "n_classes_test_only": len(test_classes - train_classes),
            "macro_f1": float(f1_score(yt, yp, labels=learnable,
                                       average="macro", zero_division=0)),
            "macro_f1_all_test_classes": float(
                f1_score(yt, yp, labels=sorted(test_classes),
                         average="macro", zero_division=0)),
            "accuracy": float((yt == yp).mean()),
        }
    return out
