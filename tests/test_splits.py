"""Leakage-control tests.

These are the tests that matter most in this repository: a violated leakage
control invalidates the study, so both directions are checked -- the legitimate
paths must pass, AND the assertions must actually fire on constructed
violations. A guard that never fires is indistinguishable from no guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scfmbench import splits as S

FRACS = {"train": 0.60, "calibration": 0.20, "test": 0.20}

# Two entries deliberately share GSE143423 -- the real corpus has this hazard
# (the same accession filed under two cancer types) and it must collapse to one group.
DS_GROUPS = {
    "NSCLC_GSE143423": "GSE143423",
    "BRCA_GSE143423": "GSE143423",
    "NSCLC_GSE117570": "GSE117570",
    "BRCA_GSE176078": "GSE176078",
    "LIHC_GSE146115": "GSE146115",
    "LIHC_GSE166635": "GSE166635",
}
LABELS = ["CD4_T", "CD8_T", "Treg", "NK", "B", "Macrophage", "Mast"]


@pytest.fixture(scope="module")
def index() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for ds, grp in DS_GROUPS.items():
        for d in range(4):
            for lab in LABELS:
                for i in range(int(rng.integers(8, 60))):
                    rows.append((f"{ds}_d{d}_{lab}_{i}", ds, grp, f"{ds}_donor{d}", lab))
    return pd.DataFrame(rows, columns=["cell_id", "dataset", "group", "donor", "label"])


@pytest.mark.parametrize("scheme", S.__dict__.get("_SCHEMES", [
    "S1_within_dataset", "S2_leave_donor_out", "S3_leave_dataset_out"]))
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_schemes_are_leakage_free(index, scheme, seed):
    part = S.make_splits(index, scheme, FRACS, seed=seed)
    S.assert_no_leakage(index, part, scheme)
    assert set(part.unique()) == set(S.PARTITIONS)


def test_s1_hits_target_proportions(index):
    part = S.make_splits(index, "S1_within_dataset", FRACS, seed=0)
    frac = part.value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.60, abs=0.02)
    assert frac["calibration"] == pytest.approx(0.20, abs=0.02)


def test_s1_keeps_every_class_in_every_partition(index):
    part = S.make_splits(index, "S1_within_dataset", FRACS, seed=0)
    tab = index.assign(p=part.to_numpy()).pivot_table(
        index="label", columns="p", values="cell_id", aggfunc="count")
    assert tab.notna().all().all(), f"a class is missing from a partition:\n{tab}"


@pytest.mark.parametrize("holdout", sorted(set(DS_GROUPS.values())))
def test_s3_named_holdout_is_entirely_test(index, holdout):
    part = S.make_splits(index, "S3_leave_dataset_out", FRACS, seed=1, holdout_group=holdout)
    S.assert_no_leakage(index, part, "S3_leave_dataset_out")
    df = index.assign(p=part.to_numpy())
    assert set(df.loc[df["group"] == holdout, "p"]) == {"test"}
    train_g = set(df.loc[df["p"] == "train", "group"])
    calib_g = set(df.loc[df["p"] == "calibration", "group"])
    assert not (train_g & calib_g), "a group spans train and calibration"


def test_shared_accession_folds_into_one_test_unit(index):
    """The specific hazard: holding out GSE143423 must hold out BOTH TISCH2
    entries that share it, or cells from the same study sit on both sides."""
    part = S.make_splits(index, "S3_leave_dataset_out", FRACS, seed=1, holdout_group="GSE143423")
    df = index.assign(p=part.to_numpy())
    assert df.loc[df["p"] == "test", "dataset"].nunique() == 2


# ---------------- negative controls: the guards must FIRE ----------------

def test_donor_spanning_partitions_is_caught(index):
    bad = pd.Series(["train"] * len(index), index=index.index)
    bad.iloc[:50] = "test"
    bad.iloc[50:100] = "calibration"
    with pytest.raises(S.LeakageError, match="donor"):
        S.assert_no_leakage(index, bad, "S2_leave_donor_out")


def test_shared_accession_split_across_partitions_is_caught(index):
    """Treating the duplicated accession as two independent datasets is the
    exact silent-inflation failure mode; it must raise."""
    naive = index["dataset"].map(
        {d: ("test" if d == "NSCLC_GSE143423" else "train") for d in DS_GROUPS})
    naive = naive.where(index["dataset"] != "BRCA_GSE143423", "calibration")
    with pytest.raises(S.LeakageError, match="group"):
        S.assert_no_leakage(index, naive, "S3_leave_dataset_out")


def test_calibration_is_reserved_for_project_b():
    S.assert_calibration_untouched({"train", "test"})          # legitimate
    with pytest.raises(S.LeakageError, match="reserved"):
        S.assert_calibration_untouched({"train", "calibration", "test"})


def test_null_label_is_refused_not_coerced(index):
    holed = index.assign(label=index["label"].mask(index.index < 3))
    with pytest.raises(ValueError, match="null harmonised label"):
        S.make_splits(holed, "S1_within_dataset", FRACS, seed=0)


def test_duplicate_cell_id_is_refused(index):
    with pytest.raises(ValueError, match="duplicated cell_id"):
        S.make_splits(pd.concat([index, index.head(2)]), "S1_within_dataset", FRACS, seed=0)


def test_too_few_groups_for_s3_raises_rather_than_splitting_a_group(index):
    one_group = index[index["group"] == "GSE143423"]
    with pytest.raises(S.LeakageError):
        S.make_splits(one_group, "S3_leave_dataset_out", FRACS, seed=0)
