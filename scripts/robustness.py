"""Two post-hoc robustness checks requested after initial reporting.

Neither re-runs embeddings or the grid: both are recomputations of the
already-frozen statistics stage over subsets of the existing T2/T5 rows.
Nothing is re-tuned and no pre-specified analysis is altered -- the same
aggregate_seeds -> paired_contrast -> BH-FDR path is used, so the only thing
that differs from the shipped tables is which rows enter it.

R1  Pretraining-overlap sensitivity (PRE-SPECIFIED in
    results/pretraining_overlap_audit.md, never executed). Every primary
    contrast recomputed with BRCA_GSE176078 excluded (n=12) beside the
    shipped n=13 values. The audit also pre-specified a second check: whether
    the scFM margin is unusually large ON that dataset relative to its margin
    elsewhere, which would itself be evidence of memorisation.

R2  Seed-count sensitivity. Seeds were raised 5 -> 20 post hoc under
    leave-dataset-out. That is where H2 lives, so the affected contrasts are
    recomputed on seeds 0-4 only -- the original pre-boost set -- beside the
    20-seed values.
"""
import sys, pathlib, json
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from scfmbench.stages import s06_stats as S
from statsmodels.stats.multitest import multipletests

RES = pathlib.Path("results")
t2 = pd.read_csv(RES / "T2_results_long.csv")
t2["scheme"] = t2.split_file.str.split("__").str[0]
t2["budget"] = t2.budget.astype(str)
FM = ["geneformer", "scgpt"]
BASE = ["hvg_pca", "scvi", "harmony"]
BUDGETS = ["5", "10", "25", "50", "100", "all"]


def contrasts(df, label):
    """Run the frozen aggregate -> paired-contrast path over one row subset."""
    rows = []
    for sch in sorted(df.scheme.unique()):
        for b in BUDGETS:
            cell = df[(df.scheme == sch) & (df.budget == b)]
            if cell.empty:
                continue
            agg = S.aggregate_seeds(cell, "macro_f1")
            for fm in FM:
                for base in BASE:
                    if not {fm, base} <= set(agg.representation.unique()):
                        continue
                    try:
                        r = S.paired_contrast(agg, fm, base, "macro_f1")
                    except Exception:
                        continue
                    r.update(subset=label, scheme=sch, budget=b,
                             contrast=f"{fm} - {base}")
                    rows.append(r)
    out = pd.DataFrame(rows)
    # paired_contrast returns "p_raw" (Wilcoxon), and on a degenerate cell it
    # returns an ERROR dict with no p at all. Both must be handled explicitly:
    # dropping error rows silently would shrink the correction family and make
    # every surviving p look better than it deserves.
    if "error" in out.columns and out["error"].notna().any():
        bad = out[out["error"].notna()][["scheme", "budget", "contrast", "error"]]
        raise RuntimeError(f"paired_contrast failed on {len(bad)} cells:\n{bad}")
    assert "p_raw" in out.columns, f"expected p_raw, got {list(out.columns)}"
    # FDR over the whole family, matching s06_stats.main() exactly
    out["p_fdr"] = multipletests(out["p_raw"], method="fdr_bh")[1]
    return out


def join(a, b, ka, kb):
    keys = ["scheme", "budget", "contrast"]
    cols = keys + ["delta_mean", "ci95_lo", "ci95_hi", "p_fdr", "cohens_dz",
                   "n_datasets", "n_datasets_favouring_a"]
    m = (a[cols].add_suffix(f"_{ka}").rename(columns={f"{k}_{ka}": k for k in keys})
         .merge(b[cols].add_suffix(f"_{kb}").rename(columns={f"{k}_{kb}": k for k in keys}),
                on=keys, how="inner"))
    m["delta_shift"] = m[f"delta_mean_{kb}"] - m[f"delta_mean_{ka}"]
    m["sign_flip"] = np.sign(m[f"delta_mean_{ka}"]) != np.sign(m[f"delta_mean_{kb}"])
    m["sig_change"] = (m[f"p_fdr_{ka}"] < 0.05) != (m[f"p_fdr_{kb}"] < 0.05)
    return m


# ---------------- R1: pretraining-overlap sensitivity ----------------
FLAGGED = "BRCA_GSE176078"
assert FLAGGED in set(t2.dataset), "flagged dataset absent from T2"
full = contrasts(t2, "n13")

# GATE: the n=13 recomputation must reproduce the SHIPPED T3 on the contrasts
# they share. If it does not, this script is not running the same analysis the
# report describes, and its n=12 column would be measuring something else.
shipped = pd.read_csv(RES / "T3_primary_statistics.csv")
shipped["budget"] = shipped.budget.astype(str)
_key = ["scheme", "budget", "contrast"]
_chk = shipped.merge(full, on=_key, suffixes=("_ship", "_recomp"))
assert len(_chk) > 0, "no overlap between recomputed and shipped contrasts"
_dmax = (_chk.delta_mean_ship - _chk.delta_mean_recomp).abs().max()
assert _dmax < 1e-9, f"recomputation does not reproduce shipped T3 (max diff {_dmax:.2e})"
print(f"[gate] reproduces shipped T3 on {len(_chk)} contrasts (max |diff| {_dmax:.2e})")

excl = contrasts(t2[t2.dataset != FLAGGED], "n12")
r1 = join(full, excl, "n13", "n12")
r1.to_csv(RES / "T10_pretraining_sensitivity.csv", index=False)

# the audit's SECOND pre-specified check: is the scFM margin unusually large
# on the flagged dataset relative to its margin on the other twelve?
mem = []
for sch in sorted(t2.scheme.unique()):
    for b in BUDGETS:
        cell = t2[(t2.scheme == sch) & (t2.budget == b)]
        if cell.empty:
            continue
        agg = S.aggregate_seeds(cell, "macro_f1")
        piv = agg.pivot_table(index="dataset", columns="representation", values="macro_f1")
        for fm in FM:
            if fm not in piv or "hvg_pca" not in piv:
                continue
            d = (piv[fm] - piv["hvg_pca"]).dropna()
            if FLAGGED not in d.index or len(d) < 5:
                continue
            here, other = float(d[FLAGGED]), d.drop(FLAGGED)
            mem.append({"scheme": sch, "budget": b, "contrast": f"{fm} - hvg_pca",
                        "delta_on_flagged": here,
                        "delta_others_mean": float(other.mean()),
                        "delta_others_sd": float(other.std(ddof=1)),
                        "z_vs_others": (here - other.mean()) / other.std(ddof=1),
                        "rank_of_flagged": int((d.rank(ascending=False)[FLAGGED])),
                        "n_datasets": len(d)})
memo = pd.DataFrame(mem)
memo.to_csv(RES / "T11_memorisation_check.csv", index=False)

# ---------------- R2: seed-count sensitivity ----------------
s3 = t2[t2.scheme == "S3_leave_dataset_out"]
boosted = sorted(b for b in s3.budget.unique()
                 if s3[s3.budget == b].seed.nunique() > 5)
boost20 = contrasts(s3[s3.budget.isin(boosted)], "seeds20")
boost5 = contrasts(s3[s3.budget.isin(boosted) & (s3.seed <= 4)], "seeds5")
r2 = join(boost20, boost5, "seeds20", "seeds5")
r2.to_csv(RES / "T12_seed_sensitivity.csv", index=False)

summary = {
    "flagged_dataset": FLAGGED,
    "r1_contrasts": int(len(r1)),
    "r1_sign_flips": int(r1.sign_flip.sum()),
    "r1_significance_changes": int(r1.sig_change.sum()),
    "r1_max_abs_delta_shift": float(r1.delta_shift.abs().max()),
    "r1_median_abs_delta_shift": float(r1.delta_shift.abs().median()),
    "memorisation_max_abs_z": float(memo.z_vs_others.abs().max()),
    "memorisation_n_cells_z_gt_2": int((memo.z_vs_others.abs() > 2).sum()),
    "memorisation_n_cells": int(len(memo)),
    "boosted_budgets": boosted,
    "r2_contrasts": int(len(r2)),
    "r2_sign_flips": int(r2.sign_flip.sum()),
    "r2_significance_changes": int(r2.sig_change.sum()),
    "r2_max_abs_delta_shift": float(r2.delta_shift.abs().max()),
}
(RES / "robustness_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
