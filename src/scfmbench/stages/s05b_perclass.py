"""Per-class F1 rescoring for H3 (cell-type rarity).

WHY THIS EXISTS AS A SEPARATE STAGE
-----------------------------------
H3 asks whether the scFM-versus-classical difference depends on cell-type
RARITY. Answering it needs per-class F1, and `metrics.evaluate` computes
exactly that -- but the sweep only ever wrote the macro-averaged summary to
T2, so the per-class detail was discarded 29,961 times. That is a design
oversight in the sweep, not a limitation of the data.

Re-running the whole grid to recover it would cost ~53 h. It is not necessary:
T2 records `C_selected` for every run, so the model can be refit at the
ALREADY-SELECTED regularisation strength with NO inner cross-validation. That
matters for more than cost -- because nothing is selected here, this stage
cannot introduce hyperparameter selection on test, which is the guardrail that
would otherwise make a rescoring pass dangerous.

WHAT IS AND IS NOT RECOMPUTED
-----------------------------
Identical to the original run: the split, the label-budget subsample (same
seed, same `budget_indices`), the representation, the standardiser fit on
train only, and C. Therefore the refit model is the same model, and the
macro-F1 this stage derives from the per-class values must reproduce the
macro-F1 already in T2. That equality is ASSERTED per run, not assumed -- a
mismatch means the reconstruction diverged and the run is reported as failed
rather than silently contributing a wrong per-class row.

SCOPE
-----
Rescoring the full 29,961-run grid is unnecessary for H3. Rarity is a property
of a class within a dataset, and the hypothesis is about the SHAPE of the
accuracy-versus-rarity relationship, so this runs on the cells where the
primary contrasts landed: both scFMs plus hvg_pca, at two budgets (10 and
'all', the restricted and unrestricted extremes), across all three schemes.
The subset is stated up front rather than chosen after seeing which subset
gives a cleaner answer.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from scfmbench import config as C
from scfmbench import provenance
from scfmbench.stages.s05_sweep import (budget_indices, load_embedding)
from scfmbench.models.classical import fit_standardiser
from scfmbench.gpu_logreg import GPULogisticRegression


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--representations", nargs="*",
                    default=["geneformer", "scgpt", "hvg_pca"])
    ap.add_argument("--budgets", nargs="*", default=["10", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="cap runs (smoke testing only)")
    args = ap.parse_args(argv)

    cfg = C.load(args.config)
    # Read the SAME config keys the sweep uses -- cfg["data"][...] -- rather than
    # inventing a parallel path scheme that could silently point elsewhere.
    emb_dir = pathlib.Path(cfg["data"]["embeddings"])
    split_dir = pathlib.Path(cfg["data"]["splits"])
    res = pathlib.Path(cfg["data"]["results"])

    idx = pd.read_parquet(split_dir / "cell_index.parquet")
    y_all = idx["label"].to_numpy()
    ds_all = idx["dataset"].to_numpy()
    cid_all = idx["cell_id"].to_numpy()

    t2 = pd.read_csv(res / "T2_results_long.csv")
    # one row per RUN (T2 is per dataset within a run)
    runs = (t2[t2.representation.isin(args.representations)]
              .assign(bstr=lambda d: d.budget.astype(str))
              .query("bstr in @args.budgets")
              .groupby(["representation", "split_file", "budget", "seed"], as_index=False)
              .agg(C_selected=("C_selected", "first"),
                   n_dims_selected=("n_dims_selected", "first")))
    if args.limit:
        runs = runs.head(args.limit)
    print(f"rescoring {len(runs)} runs", flush=True)

    out_rows, failed, timings = [], [], []
    cache: dict[tuple, tuple] = {}
    for i, r in enumerate(runs.itertuples(index=False), 1):
        stem = pathlib.Path(r.split_file).stem
        key = (r.representation, stem)
        if key not in cache:
            cache.clear()          # one representation-split resident at a time
            E, cids = load_embedding(emb_dir, r.representation, stem)
            order = pd.Series(np.arange(len(cids)), index=cids).reindex(cid_all).to_numpy()
            if np.isnan(order).any():
                raise ValueError(f"{key} does not cover the cell index")
            cache[key] = (E[order.astype(int)],)
        (E,) = cache[key]

        part = pd.read_parquet(split_dir / r.split_file)["partition"].to_numpy()
        tr, te = part == "train", part == "test"
        budget = None if str(r.budget) == "all" else int(r.budget)
        sub = budget_indices(y_all[tr], budget, int(r.seed))
        tr_idx = np.flatnonzero(tr)[sub]

        # fit_standardiser takes the TRAINING MATRIX, not indices. Fitting it on
        # the budget-subsampled training rows is what the sweep does, so the
        # refit sees exactly the same scaling as the original run.
        # TRUNCATE TO THE SELECTED DIMENSIONALITY before anything else.
        # hvg_pca's PC count is chosen by inner CV per run (30/50/100 observed),
        # while every other representation is fixed-width (Geneformer 768, scGPT
        # 512, scVI 30, Harmony 100). The first version of this stage used the
        # full stored matrix regardless, so for hvg_pca it refit a
        # HIGHER-DIMENSIONAL model than the run it claimed to reconstruct. That
        # produced 1,966 reconstruction failures -- 100% of them hvg_pca, 99% at
        # budget=10 where the fewest PCs are selected, with a median error of
        # -0.065 macro-F1. The guard caught it. Without the guard those rows
        # would have entered the rarity analysis as per-class scores for a model
        # the study never ran, and they would have been systematically biased
        # against the baseline.
        n_dims = int(r.n_dims_selected)
        if n_dims > E.shape[1]:
            raise ValueError(
                f"run selected {n_dims} dims but the stored matrix has "
                f"{E.shape[1]}; refusing to reconstruct a model that cannot exist")
        Esel = E[:, :n_dims]

        scale = fit_standardiser(Esel[tr_idx])
        Xtr, ytr = scale(Esel[tr_idx]), y_all[tr_idx]
        Xte, yte = scale(Esel[te]), y_all[te]

        with provenance.timed(f"{r.representation}:{stem}:{r.budget}:{r.seed}", timings):
            mdl = GPULogisticRegression(C=float(r.C_selected),
                                        random_state=int(r.seed)).fit(Xtr, ytr)
            yp = mdl.predict(Xte)

        train_classes = set(np.unique(ytr).tolist())
        for ds in np.unique(ds_all[te]):
            m = ds_all[te] == ds
            yt, yh = yte[m], yp[m]
            learnable = sorted(train_classes & set(np.unique(yt).tolist()))
            if not learnable:
                continue
            # RECONSTRUCTION CHECK: this must equal the macro_f1 already in T2.
            macro = float(f1_score(yt, yh, labels=learnable,
                                   average="macro", zero_division=0))
            ref = t2[(t2.representation == r.representation)
                     & (t2.split_file == r.split_file)
                     & (t2.budget.astype(str) == str(r.budget))
                     & (t2.seed == r.seed) & (t2.dataset == ds)]
            if len(ref) and abs(float(ref.macro_f1.iloc[0]) - macro) > 1e-6:
                failed.append({"representation": r.representation, "split_file": r.split_file,
                               "budget": str(r.budget), "seed": int(r.seed), "dataset": str(ds),
                               "t2_macro_f1": float(ref.macro_f1.iloc[0]),
                               "refit_macro_f1": macro})
                continue
            per = f1_score(yt, yh, labels=learnable, average=None, zero_division=0)
            n_by = pd.Series(yt).value_counts()
            for cls, f1v in zip(learnable, per):
                out_rows.append({
                    "representation": r.representation, "split_file": r.split_file,
                    "scheme": stem.split("__")[0], "budget": str(r.budget),
                    "seed": int(r.seed), "dataset": str(ds), "cell_type": str(cls),
                    "f1": float(f1v),
                    "n_test_cells_class": int(n_by.get(cls, 0)),
                    "class_prevalence": float(n_by.get(cls, 0) / len(yt)),
                    "n_test_cells_dataset": int(len(yt)),
                })
        if i % 25 == 0:
            print(f"  {i}/{len(runs)} runs, {len(out_rows)} class-rows, "
                  f"{len(failed)} failed", flush=True)

    df = pd.DataFrame(out_rows)
    df.to_csv(res / "T5_per_class_f1.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(res / "T5_reconstruction_failures.csv", index=False)
    json.dump(timings, open(res / "timings_perclass.json", "w"), indent=2)
    print(json.dumps({"class_rows": len(df), "runs": len(runs),
                      "reconstruction_failures": len(failed)}, indent=2))
    # A reconstruction failure means the refit diverged from the recorded run;
    # fail loudly rather than shipping per-class numbers that do not correspond
    # to the macro numbers already published in T2.
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
