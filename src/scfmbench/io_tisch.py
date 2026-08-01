"""Readers for TISCH2 artefact pairs.

Each dataset is `{DATASET}_expression.h5` (10x-style CSR: matrix/{data,indices,
indptr,shape,barcodes,features/{id,name}}) plus `{DATASET}_CellMetainfo_table.tsv`
carrying the author annotations.

MEMORY CONTRACT: the largest dataset here is 4.33 GiB as CSR and the full corpus
is 10.7 GiB (78 GiB dense) against a 15 GiB host that also runs an OOM killer.
So these readers never densify, never hold two datasets at once, and select
cells BEFORE materialising the matrix -- `read_tisch` takes the barcode subset
it is going to keep, rather than loading everything and subsetting after.
"""
from __future__ import annotations

import pathlib

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

LINEAGE_COL = "Celltype (major-lineage)"
MALIGNANCY_COL = "Celltype (malignancy)"


def read_metainfo(path: str | pathlib.Path) -> pd.DataFrame:
    """Read a metainfo table, normalising the donor column name across datasets.

    TISCH2 tables are inconsistent: some carry Patient, some Sample, some both,
    and the column order differs. Donor identity drives the S2/S3 leakage
    controls, so a missing donor column is an error rather than something to
    fill with a placeholder.
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.rename(columns={df.columns[0]: "cell_id"})
    cols = {c.strip().lower().replace(" ", ""): c for c in df.columns}
    donor_col = cols.get("patient") or cols.get("sample") or cols.get("donor")
    if donor_col is None:
        raise ValueError(
            f"{path}: no Patient/Sample/Donor column found (have {list(df.columns)}). "
            f"Donor identity is required for the S2/S3 leakage controls; it cannot be "
            f"defaulted."
        )
    df["donor_raw"] = df[donor_col].astype(str)
    if LINEAGE_COL not in df.columns:
        raise ValueError(f"{path}: missing {LINEAGE_COL!r}; author labels are required")
    df["label_raw"] = df[LINEAGE_COL].astype(str)
    df["malignancy_raw"] = df[MALIGNANCY_COL].astype(str) if MALIGNANCY_COL in df.columns else "NA"
    return df


def h5_dims(path: str | pathlib.Path) -> tuple[int, int]:
    with h5py.File(path, "r") as h:
        n_gene, n_cell = (int(x) for x in h["matrix/shape"][:])
    return n_cell, n_gene


def read_tisch(
    h5_path: str | pathlib.Path,
    keep_barcodes: np.ndarray | None = None,
) -> ad.AnnData:
    """Read a TISCH2 h5 into AnnData (cells x genes, CSR), optionally restricted
    to `keep_barcodes`.

    The stored matrix is genes x cells CSC-equivalent; slicing columns of the
    stored CSR-by-gene layout is what lets us take a cell subset without ever
    holding the full matrix. Gene symbols come from features/name; duplicates
    are summed rather than silently dropped (see harmonise_genes).
    """
    with h5py.File(h5_path, "r") as h:
        g = h["matrix"]
        n_gene, n_cell = (int(x) for x in g["shape"][:])
        barcodes = np.array([b.decode() if isinstance(b, bytes) else b for b in g["barcodes"][:]])
        names = np.array([b.decode() if isinstance(b, bytes) else b for b in g["features/name"][:]])
        ids = np.array([b.decode() if isinstance(b, bytes) else b for b in g["features/id"][:]])
        # Stored as CSC over cells (indptr length n_cell+1) in 10x convention.
        indptr = g["indptr"][:]
        if len(indptr) == n_cell + 1:
            if keep_barcodes is not None:
                want = pd.Index(barcodes).get_indexer(pd.Index(keep_barcodes))
                want = np.sort(want[want >= 0])
            else:
                want = np.arange(n_cell)
            # Read only the slices belonging to the wanted cells.
            data_parts, idx_parts, counts = [], [], np.empty(len(want), dtype=np.int64)
            for k, ci in enumerate(want):
                lo, hi = int(indptr[ci]), int(indptr[ci + 1])
                counts[k] = hi - lo
                if hi > lo:
                    data_parts.append(g["data"][lo:hi])
                    idx_parts.append(g["indices"][lo:hi])
            data = np.concatenate(data_parts) if data_parts else np.zeros(0, dtype=np.float32)
            indices = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int32)
            new_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            X = sp.csr_matrix((data, indices, new_indptr), shape=(len(want), n_gene))
            bc = barcodes[want]
        else:
            raise ValueError(
                f"{h5_path}: unexpected indptr length {len(indptr)} for shape "
                f"({n_gene} genes, {n_cell} cells); refusing to guess the layout"
            )

    adata = ad.AnnData(
        X=X.astype(np.float32),
        obs=pd.DataFrame(index=pd.Index(bc, name="cell_id")),
        var=pd.DataFrame({"gene_symbol": names, "gene_id": ids},
                         index=pd.Index(names, name="var_names")),
    )
    adata.var_names_make_unique()
    return adata
