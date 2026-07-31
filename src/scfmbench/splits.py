"""Three-way (train / calibration / test) leakage-controlled split construction.

Three schemes, in increasing order of distribution shift:

  S1_within_dataset   random stratified within each dataset          (no shift)
  S2_leave_donor_out  whole donors held out                          (moderate shift)
  S3_leave_dataset_out whole dataset GROUPS held out                 (high shift)

The `group` column, not `dataset`, is the S3 holdout unit. Two TISCH2 entries
sharing a GSE accession are the same underlying study and must form ONE group:
splitting them across partitions is precisely the leakage S3 exists to prevent
and would silently inflate transfer scores.

The CALIBRATION partition is reserved for downstream Project B (conformal
prediction) and must not be read by this project. `assert_calibration_untouched`
is the guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIN, CALIB, TEST = "train", "calibration", "test"
PARTITIONS = (TRAIN, CALIB, TEST)

REQUIRED_COLUMNS = ("cell_id", "dataset", "group", "donor", "label")


class LeakageError(AssertionError):
    """A split violated a leakage control. Never caught and downgraded to a warning."""


def _check_index(index: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in index.columns]
    if missing:
        raise ValueError(f"cell index missing required columns: {missing}")
    if index["cell_id"].duplicated().any():
        n = int(index["cell_id"].duplicated().sum())
        raise ValueError(f"cell index has {n} duplicated cell_id values")
    if index["label"].isna().any():
        n = int(index["label"].isna().sum())
        raise ValueError(
            f"{n} cells have a null harmonised label. Unmappable labels must be "
            f"DROPPED upstream in stage.harmonise and counted in T4, never carried "
            f"into split construction."
        )


def _assign_units(units: np.ndarray, fracs: dict, rng: np.random.Generator) -> dict:
    """Assign whole units (donors or groups) to partitions by target proportion.

    Units are shuffled then allocated largest-remainder style, so small unit
    counts still populate every requested partition rather than rounding one away.
    Partitions with a zero target fraction are skipped, which is what lets the
    named-holdout S3 path divide the remaining groups two ways.

    A unit is never split. If there are fewer units than requested partitions the
    function raises rather than borrowing cells across a boundary -- the whole
    point of the unit is that it stays intact.
    """
    units = np.asarray(sorted(units))
    wanted = [p for p in PARTITIONS if fracs.get(p, 0.0) > 0.0]
    if len(units) < len(wanted):
        raise LeakageError(
            f"only {len(units)} holdout unit(s) available but {len(wanted)} partitions "
            f"({', '.join(wanted)}) must each be non-empty. Add datasets/donors, or use a "
            f"scheme with a coarser holdout unit, rather than allowing a unit to span "
            f"partitions."
        )
    shuffled = units[rng.permutation(len(units))]
    n = len(shuffled)
    counts = {p: int(np.floor(fracs[p] * n)) for p in wanted}
    for p in wanted:                           # every requested partition gets >=1 unit
        counts[p] = max(counts[p], 1)
    while sum(counts.values()) > n:            # trim the largest first
        counts[max(counts, key=lambda k: counts[k])] -= 1
    while sum(counts.values()) < n:
        counts[wanted[0]] += 1
    out, i = {}, 0
    for p in wanted:
        for u in shuffled[i : i + counts[p]]:
            out[u] = p
        i += counts[p]
    return out


def make_splits(
    index: pd.DataFrame, scheme: str, fracs: dict, seed: int,
    holdout_group: str | None = None,
) -> pd.Series:
    """Return a partition label per row of `index`, aligned to its order."""
    _check_index(index)
    rng = np.random.default_rng(seed)
    part = pd.Series(index=index.index, dtype=object)

    if scheme == "S1_within_dataset":
        # Stratified by (dataset, label) so rare classes appear in all three
        # partitions wherever their count permits.
        for _, rows in index.groupby(["dataset", "label"], sort=True, observed=True):
            idx = rng.permutation(rows.index.to_numpy())
            n = len(idx)
            n_tr = int(round(fracs[TRAIN] * n))
            n_ca = int(round(fracs[CALIB] * n))
            n_tr = min(n_tr, max(n - 2, 0)) if n >= 3 else n
            part.loc[idx[:n_tr]] = TRAIN
            part.loc[idx[n_tr : n_tr + n_ca]] = CALIB
            part.loc[idx[n_tr + n_ca :]] = TEST

    elif scheme == "S2_leave_donor_out":
        # Donors are held out whole, WITHIN each dataset: S2 isolates
        # inter-individual variation, not cross-study batch effects.
        for ds, rows in index.groupby("dataset", sort=True, observed=True):
            mapping = _assign_units(rows["donor"].unique(), fracs, rng)
            part.loc[rows.index] = rows["donor"].map(mapping).to_numpy()

    elif scheme == "S3_leave_dataset_out":
        # Whole GROUPS held out, and groups stay the unit for EVERY partition.
        #
        # When a specific holdout group is named it becomes the test set, and the
        # remaining groups are divided between train and calibration -- again whole,
        # never by donor. Splitting the remainder by donor would place one group's
        # cells in both train and calibration, violating the S3 control ("no dataset
        # may appear in more than one partition") even though the test set itself
        # would still look clean.
        #
        # Consequence worth stating plainly for downstream Project B: under S3 the
        # calibration partition is a DIFFERENT held-out study from test, so
        # calibration and test are not exchangeable and split-conformal coverage is
        # not guaranteed under this scheme. That is a property of genuine
        # distribution shift rather than a defect of the split, and it is precisely
        # the thing Project B exists to measure. The leakage control takes
        # precedence over conformal convenience.
        groups = index["group"].unique()
        if holdout_group is not None:
            if holdout_group not in set(groups):
                raise ValueError(f"holdout_group {holdout_group!r} not in corpus: {sorted(groups)}")
            is_test = index["group"] == holdout_group
            part.loc[is_test] = TEST
            rest = index.loc[~is_test]
            rest_groups = rest["group"].unique()
            if len(rest_groups) < 2:
                raise LeakageError(
                    f"S3 with holdout group {holdout_group!r} leaves only "
                    f"{len(rest_groups)} group(s) for train and calibration; at least 2 are "
                    f"needed to keep whole groups inside a single partition."
                )
            denom = fracs[TRAIN] + fracs[CALIB]
            sub = {TRAIN: fracs[TRAIN] / denom, CALIB: fracs[CALIB] / denom, TEST: 0.0}
            gmap = _assign_units(rest_groups, sub, rng)
            part.loc[rest.index] = rest["group"].map(gmap).to_numpy()
        else:
            mapping = _assign_units(groups, fracs, rng)
            part.loc[index.index] = index["group"].map(mapping).to_numpy()
    else:
        raise ValueError(f"unknown split scheme: {scheme!r}")

    if part.isna().any():
        raise LeakageError(f"{int(part.isna().sum())} cells were left unassigned by {scheme}")
    return part


def assert_no_leakage(index: pd.DataFrame, part: pd.Series, scheme: str) -> None:
    """Hard leakage checks. Raises rather than warns -- a violated control
    invalidates the study, so it must stop the run."""
    df = index.assign(_part=part.to_numpy())

    if scheme in ("S2_leave_donor_out", "S3_leave_dataset_out"):
        spanning = (
            df.groupby("donor", observed=True)["_part"].nunique().loc[lambda s: s > 1].index.tolist()
        )
        if spanning:
            raise LeakageError(
                f"{scheme}: {len(spanning)} donor(s) appear in more than one partition "
                f"(e.g. {spanning[:5]}). No donor may span partitions under S2 or S3."
            )
    if scheme == "S3_leave_dataset_out":
        spanning = (
            df.groupby("group", observed=True)["_part"].nunique().loc[lambda s: s > 1].index.tolist()
        )
        if spanning:
            raise LeakageError(
                f"S3: dataset group(s) {spanning} appear in more than one partition. "
                f"Note that groups, not datasets, are the holdout unit -- entries sharing "
                f"a GSE accession are one group by construction."
            )
    for p in PARTITIONS:
        if not (df["_part"] == p).any():
            raise LeakageError(f"{scheme}: partition {p!r} is empty")


def assert_calibration_untouched(part_used: set[str]) -> None:
    """Guard for the reserved partition. Project A reads TRAIN and TEST only;
    CALIBRATION exists so Project B can compute conformal nonconformity scores
    on these identical cached embeddings without re-embedding or re-splitting."""
    if CALIB in part_used:
        raise LeakageError(
            "the calibration partition is reserved for downstream Project B and must "
            "not be read in Project A. Use train and test only."
        )
