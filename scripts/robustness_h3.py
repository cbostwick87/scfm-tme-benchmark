"""H3 (rarity) leg of the pre-specified pretraining-overlap sensitivity check.

The T3 family was covered by robustness.py; H3 lives in the per-class table
and needs the same treatment. Identical pipeline to the shipped H3 analysis
(pre-specified prevalence strata, paired over (dataset, cell type), Wilcoxon
+ bootstrap CI + BH-FDR) with the only difference being which rows enter.
"""
import pathlib, json
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

RES = pathlib.Path("results")
FLAGGED = "BRCA_GSE176078"
BINS = [0, 0.01, 0.05, 0.20, 1.01]
LAB = ["<1% (very rare)", "1-5% (rare)", "5-20% (uncommon)", ">20% (common)"]

t5 = pd.read_csv(RES / "T5_per_class_f1.csv")
t5["budget"] = t5.budget.astype(str)
assert FLAGGED in set(t5.dataset), "flagged dataset absent from T5"


def h3(df, label):
    agg = (df.groupby(["scheme", "budget", "representation", "dataset", "cell_type"],
                      as_index=False)
             .agg(f1=("f1", "mean"), prev=("class_prevalence", "mean")))
    agg["stratum"] = pd.cut(agg.prev, bins=BINS, labels=LAB, include_lowest=True)
    rows = []
    for sch in sorted(agg.scheme.unique()):
        for b in ["10", "all"]:
            for fm in ["geneformer", "scgpt"]:
                for st in LAB:
                    s = agg[(agg.scheme == sch) & (agg.budget == b) & (agg.stratum == st)]
                    piv = s.pivot_table(index=["dataset", "cell_type"],
                                        columns="representation", values="f1")
                    if fm not in piv or "hvg_pca" not in piv:
                        continue
                    pair = piv[[fm, "hvg_pca"]].dropna()
                    if len(pair) < 5:
                        continue
                    d = (pair[fm] - pair["hvg_pca"]).to_numpy()
                    try:
                        _, p = stats.wilcoxon(d, alternative="two-sided",
                                              zero_method="wilcox")
                    except ValueError:
                        p = 1.0
                    rng = np.random.default_rng(0)
                    boot = np.array([rng.choice(d, len(d), replace=True).mean()
                                     for _ in range(5000)])
                    rows.append({"scheme": sch, "budget": b, "stratum": st,
                                 "contrast": f"{fm} - hvg_pca",
                                 "n_class_instances": len(d), "delta_mean": d.mean(),
                                 "ci95_lo": np.percentile(boot, 2.5),
                                 "ci95_hi": np.percentile(boot, 97.5), "p_raw": p})
    out = pd.DataFrame(rows)
    out["p_fdr"] = multipletests(out.p_raw, method="fdr_bh")[1]
    return out.add_suffix(f"_{label}").rename(
        columns={f"{k}_{label}": k for k in ["scheme", "budget", "stratum", "contrast"]})


full = h3(t5, "n13")
excl = h3(t5[t5.dataset != FLAGGED], "n12")

# GATE: the n=13 leg must reproduce the shipped T8 for the rows they share.
shipped = pd.read_csv(RES / "T8_h3_rarity.csv")
shipped["budget"] = shipped.budget.astype(str)
key = ["scheme", "budget", "stratum", "contrast"]
chk = shipped.merge(full, on=key)
dmax = (chk.delta_mean - chk.delta_mean_n13).abs().max()
assert len(chk) > 0 and dmax < 1e-9, f"does not reproduce shipped T8 (max diff {dmax:.2e})"
print(f"[gate] reproduces shipped T8 on {len(chk)} rows (max |diff| {dmax:.2e})")

m = full.merge(excl, on=key, how="inner")
m["delta_shift"] = m.delta_mean_n12 - m.delta_mean_n13
m["sign_flip"] = np.sign(m.delta_mean_n13) != np.sign(m.delta_mean_n12)
m["sig_change"] = (m.p_fdr_n13 < 0.05) != (m.p_fdr_n12 < 0.05)
m.to_csv(RES / "T13_h3_pretraining_sensitivity.csv", index=False)

summary = {"rows": int(len(m)), "sign_flips": int(m.sign_flip.sum()),
           "significance_changes": int(m.sig_change.sum()),
           "max_abs_delta_shift": float(m.delta_shift.abs().max()),
           "median_abs_delta_shift": float(m.delta_shift.abs().median())}
(RES / "robustness_h3_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
cols = key + ["delta_mean_n13", "delta_mean_n12", "p_fdr_n13", "p_fdr_n12",
              "sign_flip", "sig_change"]
sub = m[m.budget == "all"]
print(sub[cols].round(4).to_string(index=False))
