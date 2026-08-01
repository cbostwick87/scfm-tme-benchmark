"""The design grid: representation x split scheme x label budget x seed.

Same classifier head everywhere -- multinomial logistic regression with L2,
strength chosen by inner cross-validation ON THE TRAINING PARTITION ONLY. The
experiment isolates the REPRESENTATION; a classifier tuned per-arm would
confound exactly what is being measured (guardrail 3).

No hyperparameter is ever selected on test (guardrail 2). The calibration
partition is never read (guardrail 6) -- `assert_calibration_untouched` enforces
it at load time.

Results are appended INCREMENTALLY to a tidy long-format table, one row per run,
so an interrupted sweep loses at most the run in flight.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

from scfmbench import config, metrics, splits


def load_embedding(emb_dir: pathlib.Path, model: str,
                   split_stem: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load a representation, returning (matrix, cell_ids).

    Two shapes exist, and the difference is structural rather than incidental:

      * A zero-shot foundation model embeds each cell independently of any
        partition, so its embedding is cached ONCE for the corpus and sharded by
        dataset. All shards are concatenated.
      * A classical representation is FIT on a split's training partition, so
        there is one file PER SPLIT and the caller must say which. Concatenating
        them would stack the same cell 75 times over.

    Passing `split_stem` selects the per-split file; omitting it concatenates
    dataset shards. A per-split directory read without `split_stem` raises rather
    than silently returning a duplicated index.
    """
    d = emb_dir / model
    if not d.exists():
        raise FileNotFoundError(f"no embedding directory {d}")
    per_split = (d / f"{split_stem}.npz").exists() if split_stem else False
    if per_split:
        fs = [d / f"{split_stem}.npz"]
    else:
        fs = sorted(f for f in d.glob("*.npz"))
        if not fs:
            raise FileNotFoundError(f"no embedding shards in {d}")
        # A directory whose files are named after splits is per-split, not sharded.
        if any("__seed" in f.name for f in fs):
            raise ValueError(
                f"{model} is fit per split (found {len(fs)} split files in {d}) but no "
                f"matching file for split {split_stem!r}. Refusing to concatenate "
                f"per-split representations -- that would stack each cell once per "
                f"split and silently corrupt the evaluation."
            )
    E, ids = [], []
    for f in fs:
        with np.load(f, allow_pickle=True) as z:
            E.append(z["emb"]); ids.append(z["cell_id"])
    return np.vstack(E), np.concatenate(ids)


def rep_dim_grid(rep: str, cfg, n_dims: int) -> list[int]:
    """Candidate dimensionalities for a representation, selectable by inner CV.

    Only the classical PCA arm has a genuine dimensionality choice: principal
    components are ORDERED, so the first k columns of a 100-component fit are
    exactly the k-component representation and no refit is needed. Foundation-model
    embedding dimensions are not ordered by variance and truncating them would be
    arbitrary mutilation, so those arms return their single native width.

    Selecting this on training data only is what makes `n_pcs_grid` in the config
    real rather than decorative -- before the pilot diagnostic the grid was
    declared but never used, which quietly under-tuned the baseline the study is
    supposed to be trying hardest to beat (guardrail 4).
    """
    if rep == "hvg_pca":
        g = [int(d) for d in cfg["embeddings"]["hvg_pca"]["n_pcs_grid"] if 0 < int(d) <= n_dims]
        return sorted(set(g)) or [n_dims]
    return [n_dims]


def inner_cv_score(Ztr, ytr, seed: int, cv_grid, n_folds: int,
                   cv_max_cells: int = 20000) -> float:
    """Best inner-CV macro-F1 over the C grid, on TRAIN only. Never touches test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    classes, counts = np.unique(ytr, return_counts=True)
    k = int(min(n_folds, counts.min()))
    if k < 2 or len(classes) < 2:
        return -np.inf
    if len(ytr) > cv_max_cells:
        rng = np.random.default_rng(seed)
        sel = []
        for c in classes:
            idx = np.flatnonzero(ytr == c)
            take = max(int(round(cv_max_cells * len(idx) / len(ytr))), min(len(idx), 2))
            sel.append(idx if len(idx) <= take else rng.choice(idx, take, replace=False))
        sub = np.sort(np.concatenate(sel))
    else:
        sub = np.arange(len(ytr))
    gs = GridSearchCV(LogisticRegression(max_iter=2000, solver="lbfgs",
                                         class_weight="balanced", random_state=seed),
                      {"C": list(cv_grid)},
                      cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                      scoring="f1_macro", n_jobs=1, refit=False)
    gs.fit(Ztr[sub], ytr[sub])
    return float(gs.best_score_)


def budget_indices(y: np.ndarray, budget, seed: int) -> np.ndarray:
    """Sample up to `budget` TRAINING cells per class (None = use all).

    Sampling is per class so a budget of 10 means ten labelled examples of each
    class, which is what a label budget means operationally: an annotator's
    effort per cell type, not a global cap that starves rare classes.
    """
    rng = np.random.default_rng(seed)
    if budget is None:
        return np.arange(len(y))
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        keep.append(idx if len(idx) <= budget else rng.choice(idx, budget, replace=False))
    return np.sort(np.concatenate(keep))


def fit_predict(Ztr, ytr, Zte, seed: int, cv_grid, n_folds: int,
                cv_max_cells: int = 20000):
    """Multinomial LR with L2; C chosen by inner CV on TRAIN only.

    COST NOTE (measured, not assumed). At the unrestricted label budget the
    training partition is ~138,000 cells, and a 5-fold CV over a 6-point C grid is
    30 full logistic-regression fits on ~110,000 x 768 -- more than 20 minutes per
    run, which across the design grid is several hundred hours. The pilot gate
    caught this before the full run rather than after.

    The fix subsamples the INNER CV ONLY. C is selected on at most `cv_max_cells`
    training cells, and the final model is then refit ONCE on the ENTIRE training
    set with the chosen C. What this changes:
      * the cost of SELECTING the regularisation strength;
    what it does not change:
      * the classifier (same multinomial L2 head for every representation),
      * the C grid (unreduced -- tuning quality is not traded for speed, and the
        baseline must be tuned as carefully as the scFM arm),
      * the data the final model is fit on (the full training partition),
      * the leakage contract (the subsample is drawn from TRAIN only, and test is
        never touched during selection).
    C is a smooth regularisation parameter whose optimum is stable well before
    100k samples; the cells omitted from selection are still used for the fit.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GridSearchCV

    classes, counts = np.unique(ytr, return_counts=True)
    # `multi_class` was removed in scikit-learn 1.7+; with lbfgs a multi-class
    # problem is fit as multinomial softmax by default. That is what guardrail 3
    # requires, so it is ASSERTED after fitting rather than requested through a
    # keyword that no longer exists -- a silent switch to one-vs-rest would change
    # what the study measures.
    base = LogisticRegression(max_iter=2000, solver="lbfgs",
                              class_weight="balanced", random_state=seed)

    # inner CV needs >=2 members per class per fold; at tiny label budgets that is
    # impossible, so fall back to a fixed C rather than letting sklearn pick a
    # degenerate split.
    k = int(min(n_folds, counts.min()))
    if k >= 2 and len(classes) > 1:
        if len(ytr) > cv_max_cells:
            rng = np.random.default_rng(seed)
            sel = []
            for c in classes:                      # stratified, keeps rare classes
                idx = np.flatnonzero(ytr == c)
                take = max(int(round(cv_max_cells * len(idx) / len(ytr))), min(len(idx), 2))
                sel.append(idx if len(idx) <= take else rng.choice(idx, take, replace=False))
            sub = np.sort(np.concatenate(sel))
        else:
            sub = np.arange(len(ytr))
        gs = GridSearchCV(base, {"C": list(cv_grid)},
                          cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                          scoring="f1_macro", n_jobs=1, refit=False)
        gs.fit(Ztr[sub], ytr[sub])
        chosen, cv_used, cv_n = gs.best_params_["C"], k, int(len(sub))
    else:
        chosen, cv_used, cv_n = 1.0, 0, 0

    # final model: full training set, chosen C
    model = base.set_params(C=chosen).fit(Ztr, ytr)

    if len(model.classes_) > 2:
        if model.coef_.shape[0] != len(model.classes_):
            raise RuntimeError(
                f"classifier is not multinomial: coef_ has {model.coef_.shape[0]} rows "
                f"for {len(model.classes_)} classes (guardrail 3 requires one shared "
                f"multinomial head for every representation)")
        pr = model.predict_proba(Zte[:min(64, len(Zte))])
        if not np.allclose(pr.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(
                "classifier probabilities do not sum to 1: the solver is not fitting a "
                "multinomial softmax, so representations would not be compared under "
                "the same classifier head")
    return model.predict(Zte), chosen, cv_used, cv_n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--representations", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = config.load(args.config)

    emb_dir = pathlib.Path(cfg["data"]["embeddings"])
    split_dir = pathlib.Path(cfg["data"]["splits"])
    res = pathlib.Path(cfg["data"]["results"]); res.mkdir(parents=True, exist_ok=True)
    out_csv = pathlib.Path(args.out) if args.out else res / "T2_results_long.csv"

    idx = pd.read_parquet(split_dir / "cell_index.parquet")
    # Representations are whatever is enabled in the config's embeddings block --
    # a single source of truth, so an arm cannot be silently dropped from the sweep
    # while still appearing configured.
    reps = args.representations or [k for k, v in cfg["embeddings"].items()
                                    if isinstance(v, dict) and v.get("enabled")]
    budgets = cfg["label_budgets"]
    grid = cfg["classifier"]["C_grid"]
    n_folds = cfg["classifier"]["inner_cv_folds"]

    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(map(tuple, prev[["representation", "split_file", "budget", "seed"]]
                       .astype(str).to_numpy()))
        print(f"resuming: {len(done)} runs already recorded", flush=True)

    cache = {}
    n_new = 0
    wanted = set(cfg["splits"]["scheme_ids"])
    split_files = [sf for sf in sorted(split_dir.glob("*.parquet"))
                   if sf.name != "cell_index.parquet"
                   and sf.name.split("__")[0] in wanted]
    if not split_files:
        raise ValueError(f"no split files match scheme_ids={sorted(wanted)}")
    if cfg["run"].get("max_splits_per_scheme"):
        lim = int(cfg["run"]["max_splits_per_scheme"])
        keep, seen = [], {}
        for sf in split_files:
            s = sf.name.split("__")[0]
            seen[s] = seen.get(s, 0) + 1
            if seen[s] <= lim:
                keep.append(sf)
        split_files = keep

    y = idx["label"].to_numpy()
    for sf in split_files:
        part = pd.read_parquet(sf)
        if len(part) != len(idx) or not (part["cell_id"].to_numpy() == idx["cell_id"].to_numpy()).all():
            raise ValueError(f"{sf.name}: split rows do not align with the cell index")
        p = part["partition"].to_numpy()
        splits.assert_calibration_untouched({"train", "test"})   # guardrail 6
        tr, te = p == "train", p == "test"

        for rep in reps:
            key_rep = (rep, sf.stem)
            if key_rep not in cache:
                cache.clear()                     # one representation resident at a time
                E, ids = load_embedding(emb_dir, rep, split_stem=sf.stem)
                order = pd.Index(ids).get_indexer(pd.Index(idx["cell_id"]))
                if (order < 0).any():
                    raise ValueError(
                        f"{rep}: {int((order < 0).sum())} cells in the index have no "
                        f"embedding. Every cell must be embedded exactly once per model; "
                        f"refusing to evaluate on a partial cache.")
                cache[key_rep] = E[order]
                print(f"loaded {rep} for {sf.stem}: {cache[key_rep].shape}", flush=True)
            Z = cache[key_rep]

            for budget in budgets:
                for seed in cfg["run"]["seeds"]:
                    key = (rep, sf.name, str(budget), str(seed))
                    if key in done:
                        continue
                    t0 = time.time()
                    sel = np.flatnonzero(tr)[budget_indices(y[tr], budget, seed)]
                    ytr = y[sel]
                    if len(np.unique(ytr)) < 2:
                        continue
                    from scfmbench.models.classical import fit_standardiser
                    # Representation dimensionality is selected by the SAME inner CV
                    # that selects C, on training data only.
                    dim_grid = rep_dim_grid(rep, cfg, Z.shape[1])
                    if len(dim_grid) > 1:
                        best_d, best_s = dim_grid[-1], -np.inf
                        for dcand in dim_grid:
                            sc_d = fit_standardiser(Z[sel][:, :dcand])
                            sc = inner_cv_score(sc_d(Z[sel][:, :dcand]), ytr,
                                                seed, grid, n_folds)
                            if sc > best_s:
                                best_d, best_s = dcand, sc
                        d_sel = best_d
                    else:
                        d_sel = dim_grid[0]
                    Zs = Z[:, :d_sel]
                    scale = fit_standardiser(Zs[sel])     # fit on TRAIN subset only
                    yp, C, cv_used, cv_n = fit_predict(
                        scale(Zs[sel]), ytr, scale(Zs[te]), seed, grid, n_folds)
                    m = metrics.evaluate(y[te], yp, ytr)
                    row = {"representation": rep, "split_file": sf.name,
                           "scheme": sf.name.split("__")[0],
                           "holdout_group": sf.name.split("__holdout-")[1].replace(".parquet", "")
                                            if "__holdout-" in sf.name else "",
                           "budget": budget if budget is not None else "all",
                           "seed": seed, "n_train_used": int(len(sel)),
                           "C_selected": C, "inner_cv_folds_used": cv_used,
                           "cv_selection_cells": cv_n, "n_dims_selected": int(d_sel),
                           "seconds": round(time.time() - t0, 2),
                           **{k: v for k, v in m.items() if k != "per_class_f1"},
                           "per_class_f1_json": json.dumps(m["per_class_f1"])}
                    pd.DataFrame([row]).to_csv(out_csv, mode="a", index=False,
                                               header=not out_csv.exists())
                    n_new += 1
                    if n_new % 10 == 0:
                        print(f"  {n_new} new runs; last: {rep} {row['scheme']} "
                              f"b={row['budget']} f1={row['macro_f1']:.3f}", flush=True)
    print(json.dumps({"new_runs": n_new, "table": str(out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
