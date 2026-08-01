"""QC, gene/label harmonisation, stratified subsampling, gene-overlap matrix.

MEMORY DESIGN (this is the phase that breaks a 15 GiB host, per the brief, and
the host also runs an OOM killer that prefers python processes):
  * datasets are processed STRICTLY ONE AT A TIME and freed before the next;
  * cells to keep are chosen from the metainfo table FIRST, so the expression
    matrix is materialised only for the retained subset -- the full corpus is
    10.7 GiB as CSR and the largest single dataset is 4.33 GiB;
  * matrices stay CSR float32 and are never densified;
  * per-dataset processed h5ad is written and released immediately, so the
    concatenated corpus is assembled once from small pieces.

If a stage dies with no traceback, check `journalctl -u earlyoom` before
suspecting the code: an OOM kill leaves no Python-side evidence.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scfmbench import config, io_tisch, labels, provenance


def qc_dataset(adata: ad.AnnData, qc: dict) -> tuple[ad.AnnData, dict]:
    """Apply documented QC thresholds. No undocumented defaults."""
    n0 = adata.n_obs
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    total = np.asarray(X.sum(axis=1)).ravel()
    mito_mask = np.asarray(adata.var_names.str.upper().str.startswith(qc["mito_prefix"]))
    mito = (np.asarray(X[:, mito_mask].sum(axis=1)).ravel()
            if mito_mask.any() else np.zeros(n0, dtype=np.float32))
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_mito = np.where(total > 0, 100.0 * mito / total, 0.0)

    keep = (n_genes >= qc["min_genes_per_cell"]) & (pct_mito <= qc["max_pct_mito"])
    stats = {
        "cells_before": int(n0),
        "failed_min_genes": int((n_genes < qc["min_genes_per_cell"]).sum()),
        "failed_max_pct_mito": int((pct_mito > qc["max_pct_mito"]).sum()),
        "mito_genes_found": int(mito_mask.sum()),
        "median_genes_per_cell": float(np.median(n_genes)),
        "median_pct_mito": float(np.median(pct_mito)),
    }
    adata = adata[keep].copy()
    # gene filter AFTER cell filter, on the retained cells only
    gene_cells = np.asarray((adata.X > 0).sum(axis=0)).ravel()
    gkeep = gene_cells >= qc["min_cells_per_gene"]
    if not mito_mask.any():
        # Not fatal, but it changes what the QC means: with no detectable
        # mitochondrial genes the max_pct_mito filter is inert for this dataset and
        # every cell trivially passes it. Recorded per dataset so the report can say
        # where the filter actually applied rather than implying it applied uniformly.
        stats["mito_filter_applied"] = False
    else:
        stats["mito_filter_applied"] = True
    stats["genes_before"] = int(adata.n_vars)
    stats["genes_dropped_min_cells"] = int((~gkeep).sum())
    adata = adata[:, gkeep].copy()
    stats["cells_after"] = int(adata.n_obs)
    stats["genes_after"] = int(adata.n_vars)
    return adata, stats


def stratified_cap(obs: pd.DataFrame, target: int, seed: int,
                   class_col: str = "working_class") -> np.ndarray:
    """Subsample toward `target` cells by CAPPING COMMON CLASSES, never by
    uniform downsampling -- uniform sampling would erase the rare immune subsets
    that macro-F1 and hypothesis H3 exist to measure.

    Water-filling: raise a per-class cap until the total reaches the target, so
    classes below the cap are kept whole and only abundant classes are trimmed.
    """
    rng = np.random.default_rng(seed)
    counts = obs[class_col].value_counts()
    if counts.sum() <= target:
        return obs.index.to_numpy()
    lo, hi = 1, int(counts.max())
    while lo < hi:
        mid = (lo + hi) // 2
        if int(np.minimum(counts, mid).sum()) < target:
            lo = mid + 1
        else:
            hi = mid
    cap = lo
    keep = []
    for cls, idx in obs.groupby(class_col, observed=True).groups.items():
        idx = np.asarray(idx)
        keep.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    return np.concatenate(keep)


def gene_overlap_matrix(gene_sets: dict[str, set]) -> pd.DataFrame:
    """Pairwise gene-overlap fraction (required for H4).

    Asymmetric by construction: reported as |A n B| / |A| with A the ROW
    (reference/training) dataset, because H4 asks how much of the reference's
    feature space survives in the query.
    """
    ks = sorted(gene_sets)
    M = pd.DataFrame(index=ks, columns=ks, dtype=float)
    for a in ks:
        for b in ks:
            M.loc[a, b] = len(gene_sets[a] & gene_sets[b]) / max(len(gene_sets[a]), 1)
    return M


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = config.load(args.config)

    raw = pathlib.Path(cfg["data"]["raw"])
    proc = pathlib.Path(cfg["data"]["processed"]); proc.mkdir(parents=True, exist_ok=True)
    res = pathlib.Path(cfg["data"]["results"]); res.mkdir(parents=True, exist_ok=True)
    qc = cfg["qc"]
    lo, hi = cfg["corpus"]["target_cells_per_dataset"]
    seed = cfg["run"]["seeds"][0]

    qc_rows, gene_sets, timings = [], {}, []
    for d in cfg["corpus"]["datasets"]:
        name = d["name"]
        out_h5 = proc / f"{name}.h5ad"
        if out_h5.exists() and not args.force:
            # A cached artefact is trusted only if it OPENS and carries the columns
            # downstream stages need. A crashed or OOM-killed run leaves a partially
            # written h5ad at a plausible size; treating existence as completion
            # silently drops that dataset from qc_summary and T4 while the pipeline
            # reports success. Re-derive rather than trust.
            try:
                a = ad.read_h5ad(out_h5, backed="r")
                ok = a.n_obs > 0 and {"label", "donor", "group"} <= set(a.obs.columns)
                gv = set(a.var_names) if ok else None
                a.file.close()
            except Exception as exc:
                print(f"[requeue] {name}: cached h5ad unreadable "
                      f"({type(exc).__name__}); re-processing", flush=True)
                out_h5.unlink(missing_ok=True)
            else:
                if ok:
                    gene_sets[name] = gv
                    print(f"[skip] {name} already processed", flush=True)
                    continue
                print(f"[requeue] {name}: cached h5ad incomplete; re-processing", flush=True)
                out_h5.unlink(missing_ok=True)

        with provenance.timed(f"harmonise:{name}", timings):
            meta = io_tisch.read_metainfo(raw / f"{name}_CellMetainfo_table.tsv")
            lab = labels.harmonise_labels(meta["label_raw"])
            meta = pd.concat([meta, lab.drop(columns="label_raw")], axis=1)

            # label accounting BEFORE any cell is read
            vc = meta.groupby(["label_raw", "working_class"], dropna=False).size()
            this_lab_rows = []
            for (raw_l, wc), n in vc.items():
                this_lab_rows.append({"dataset": name, "label_raw": raw_l,
                                 "working_class": wc if pd.notna(wc) else "",
                                 "cl_term": labels.MAPPING.get(raw_l, ("", ""))[1],
                                 "n_cells": int(n),
                                 "action": "map" if pd.notna(wc) else "drop",
                                 "drop_reason": labels.DROP_RULES.get(raw_l, "")})

            kept_meta = meta[meta["working_class"].notna()].copy()
            # subsample BEFORE materialising the matrix -- this is the memory contract
            sel_idx = stratified_cap(kept_meta.set_index("cell_id"), hi, seed)
            kept_meta = kept_meta.set_index("cell_id").loc[sel_idx]

            adata = io_tisch.read_tisch(raw / f"{name}_expression.h5",
                                        keep_barcodes=kept_meta.index.to_numpy())
            adata.obs = kept_meta.reindex(adata.obs_names)
            adata, st = qc_dataset(adata, qc)
            st.update({"dataset": name, "cells_selected_pre_qc": int(len(sel_idx))})
            qc_rows.append(st)

            adata.obs["dataset"] = name
            adata.obs["cancer"] = d["cancer"]
            adata.obs["group"] = d["group"]
            adata.obs["donor"] = name + "|" + adata.obs["donor_raw"].astype(str)
            adata.obs["label"] = adata.obs["working_class"]
            gene_sets[name] = set(adata.var_names)

            # Keep only the columns the downstream stages actually consume, and
            # force them to explicit string dtype. `drop_reason` and `cl_term` are
            # all-None for retained cells (by definition -- these cells were not
            # dropped), and h5ad cannot serialise an all-None object column. The
            # label accounting they carried is already persisted in T4, so the
            # right move is to drop them here rather than coerce None to "".
            keep_cols = ["dataset", "cancer", "group", "donor", "label",
                         "label_raw", "donor_raw", "malignancy_raw"]
            adata.obs = adata.obs[[c for c in keep_cols if c in adata.obs.columns]].copy()
            for col in adata.obs.columns:
                adata.obs[col] = adata.obs[col].astype(str)
            if adata.obs["label"].isin(["None", "nan", ""]).any():
                raise ValueError(
                    f"{name}: retained cells carry a null harmonised label after "
                    f"harmonisation; unmappable labels must be dropped upstream, never "
                    f"written into the processed corpus"
                )
            adata.write_h5ad(out_h5, compression="gzip")
            acc_dir = proc / "_accounting"; acc_dir.mkdir(parents=True, exist_ok=True)
            (acc_dir / f"{name}.qc.json").write_text(json.dumps(st))
            (acc_dir / f"{name}.labels.json").write_text(json.dumps(this_lab_rows))
            (acc_dir / f"{name}.genes.json").write_text(json.dumps(sorted(gene_sets[name])))
            print(f"[done] {name}: {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
            del adata, meta, kept_meta
            gc.collect()

    # Accounting tables are assembled from PER-DATASET sidecars, never from only the
    # datasets processed in this invocation. Writing them from `qc_rows` alone was a
    # real bug: on a resumed run every skipped dataset contributes no rows, so the
    # whole-file overwrite reduced a 13-row qc_summary to 1 row while the stage
    # reported success. Resumability must not be able to destroy the accounting.
    acc = proc / "_accounting"
    qc_all, lab_all = [], []
    for d in cfg["corpus"]["datasets"]:
        qf, lf = acc / f"{d['name']}.qc.json", acc / f"{d['name']}.labels.json"
        if qf.exists():
            qc_all.append(json.loads(qf.read_text()))
        if lf.exists():
            lab_all.extend(json.loads(lf.read_text()))
    missing = [d["name"] for d in cfg["corpus"]["datasets"]
               if not (acc / f"{d['name']}.qc.json").exists()]
    if missing:
        raise RuntimeError(
            f"accounting sidecars missing for {missing}; refusing to write a partial "
            f"qc_summary/T4. Re-run those datasets with --force."
        )
    pd.DataFrame(qc_all).to_csv(res / "qc_summary.csv", index=False)
    pd.DataFrame(lab_all).to_csv(res / "T4_label_harmonisation.csv", index=False)
    labels.mapping_table().to_csv(res / "T4_label_mapping_rules.csv", index=False)
    if gene_sets:
        gene_overlap_matrix(gene_sets).to_csv(res / "gene_overlap_matrix.csv")
    json.dump(timings, open(res / "timings_harmonise.json", "w"), indent=2)
    print(json.dumps({"datasets": len(gene_sets)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
