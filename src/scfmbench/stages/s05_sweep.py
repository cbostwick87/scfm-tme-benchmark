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
from scfmbench.gpu_logreg import GPULogisticRegression, assert_matches_sklearn


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
    from sklearn.metrics import f1_score
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


_GPU_STATE = {"checked": False, "ok": False}


# Canonical T2 column order. EVERY append must use exactly this order.
#
# pandas `to_csv(mode="a")` writes a DataFrame's own column order and does NOT
# reconcile it against the existing header. So if the header was written with one
# ordering and a later append has another -- which happens the moment a column is
# added mid-run -- values land under the WRONG column names, silently. This was
# observed in practice: `budget` came to contain "S1_within_dataset" (the scheme),
# with no error anywhere. A results table that parses cleanly but has scrambled
# columns is worse than one that fails to parse.
T2_COLUMNS = [
    "representation", "split_file", "scheme", "holdout_group", "budget", "seed",
    "dataset", "n_datasets_in_run",
    "n_train_used", "C_selected", "inner_cv_folds_used", "cv_selection_cells",
    "n_dims_selected", "seconds",
    "pooled_macro_f1", "pooled_macro_f1_all_test_classes",
    "n_test", "n_classes_learnable", "n_classes_test_only",
    "macro_f1", "macro_f1_all_test_classes", "accuracy",
]


def _append_rows(rows_out, out_csv) -> None:
    """Append run rows under a fixed column order, or refuse.

    Refusing on an unexpected column set is deliberate: a mismatch means the code
    and the table have diverged, and appending anyway is how column-misaligned
    rows enter a file that still parses.
    """
    df = pd.DataFrame(rows_out)
    missing = set(T2_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(T2_COLUMNS)
    if missing or extra:
        raise RuntimeError(
            f"run row schema does not match T2_COLUMNS (missing={sorted(missing)}, "
            f"unexpected={sorted(extra)}). Refusing to append: pandas would write "
            f"these under the existing header's column names and silently misalign "
            f"the table.")
    df = df[T2_COLUMNS]
    if out_csv.exists():
        hdr = pd.read_csv(out_csv, nrows=0).columns.tolist()
        if hdr != T2_COLUMNS:
            raise RuntimeError(
                f"existing {out_csv.name} has a different column order than the code "
                f"writes. Appending would misalign every subsequent row. Retire or "
                f"migrate the old file rather than appending to it.")
    df.to_csv(out_csv, mode="a", index=False, header=not out_csv.exists())


def seeds_for(cfg, scheme: str, budget) -> list:
    """Seeds to run for a given (scheme, budget).

    The base seed list applies everywhere. An OPTIONAL `run.extra_seeds` block
    adds seeds in specified cells only -- used for the H2 cell (low label budget
    x leave-dataset-out), where seed noise is an order of magnitude larger than
    at the unrestricted budget and the runs are cheap.

    Extra seeds are ADDITIVE and cell-scoped by design: the number of seeds must
    be recorded per row (n_seeds is computed at aggregation time from the rows
    present), and a paired contrast within a cell is unaffected by how many
    seeds a DIFFERENT cell used, because seeds are averaged to a dataset mean
    before any test. Cells with more seeds simply have a less noisy mean.
    """
    base = list(cfg["run"]["seeds"])
    ex = cfg["run"].get("extra_seeds")
    if not ex:
        return base
    b = str(budget if budget is not None else "all")
    ok_scheme = scheme in [str(s) for s in ex.get("schemes", [])]
    ok_budget = b in [str(x if x is not None else "all") for x in ex.get("budgets", [])]
    if ok_scheme and ok_budget:
        return base + [s for s in ex["seeds"] if s not in base]
    return base


def _runkey(rep, split_name, budget, seed) -> tuple:
    """Canonical run key, normalised so a round-trip through CSV cannot change it.

    Seeds and budgets are written as integers but pandas infers a float column the
    moment any row is missing, so "0" on the way out came back as "0.0" and every
    resume key silently failed to match -- 350 completed runs were being recomputed
    while the log reported them complete. Normalising both ends through this one
    function is the fix; comparing ad-hoc str() casts on either side is what broke.
    """
    def norm(v):
        s = str(v).strip()
        if s in ("", "nan", "None"):
            return "all"
        try:
            f = float(s)
            return str(int(f)) if f.is_integer() else s
        except ValueError:
            return s
    return (str(rep), str(split_name), norm(budget), norm(seed))


def _gpu_ok() -> bool:
    """Gate GPU use on a one-time equivalence check against scikit-learn.

    Guardrail 3 requires the same classifier head everywhere, so the GPU solver is
    only trusted after it has been shown to reproduce scikit-learn on real data in
    THIS session. A failure disables the GPU path and falls back rather than
    aborting: the study can afford to be slow, not to be wrong.
    """
    if _GPU_STATE["checked"]:
        return _GPU_STATE["ok"]
    _GPU_STATE["checked"] = True
    try:
        import torch
        if not torch.cuda.is_available():
            print("[gpu] CUDA unavailable; using scikit-learn", flush=True)
            _GPU_STATE["ok"] = False
            return False
        rng = np.random.default_rng(0)
        n, d, k = 4000, 64, 6
        mu = rng.normal(0, 1.2, (k, d))
        yy = rng.integers(0, k, n)
        XX = (mu[yy] + rng.normal(0, 2.0, (n, d))).astype(np.float32)
        assert_matches_sklearn(XX, yy.astype(str), C=1.0, verbose=True)
        _GPU_STATE["ok"] = True
        print("[gpu] equivalence check passed; final refits run on GPU", flush=True)
    except Exception as e:
        print(f"[gpu] equivalence check FAILED ({type(e).__name__}: {str(e)[:160]}); "
              f"falling back to scikit-learn for every fit", flush=True)
        _GPU_STATE["ok"] = False
    return _GPU_STATE["ok"]


def fit_predict(Ztr, ytr, Zte, seed: int, cv_grid, n_folds: int,
                cv_max_cells: int = 20000, use_gpu: bool = True):
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
    from sklearn.metrics import f1_score

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
        # WHERE THE COST ACTUALLY IS (measured, and it corrected an earlier wrong
        # assumption of mine): profiling one unrestricted-budget run gave inner CV
        # 3096 s of a 3119 s total -- 99.3% -- against 13 s for the final GPU refit.
        # The expensive step is selecting C (30 fits), not fitting once.
        #
        # So the inner CV runs on the GPU solver too. Measured on real embeddings,
        # 20,000 x 768 with 14 classes, 6 C x 5 folds: 98.9 s on GPU against 356.3 s
        # on scikit-learn (3.6x), SAME C selected, score-curve correlation 0.9924,
        # and a maximum score difference of 0.00305 across the grid -- far below the
        # 0.02 the study calls negligible. The grid is unreduced and the fold
        # structure is unchanged; only the hardware differs.
        if use_gpu and _gpu_ok():
            skf = StratifiedKFold(k, shuffle=True, random_state=seed)
            scores = {}
            for C in cv_grid:
                fold_scores = []
                for tri, vai in skf.split(Ztr[sub], ytr[sub]):
                    mdl = GPULogisticRegression(C=C, random_state=seed).fit(
                        Ztr[sub][tri], ytr[sub][tri])
                    fold_scores.append(f1_score(ytr[sub][vai], mdl.predict(Ztr[sub][vai]),
                                                average="macro", zero_division=0))
                scores[C] = float(np.mean(fold_scores))
            chosen = max(scores, key=scores.get)
        else:
            gs = GridSearchCV(base, {"C": list(cv_grid)},
                              cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                              scoring="f1_macro", n_jobs=1, refit=False)
            gs.fit(Ztr[sub], ytr[sub])
            chosen = gs.best_params_["C"]
        cv_used, cv_n = k, int(len(sub))
    else:
        chosen, cv_used, cv_n = 1.0, 0, 0

    # Final model: full training set, chosen C.
    #
    # The final refit is the expensive step (~1100 s on CPU at the unrestricted
    # budget, 270 h across the grid) while the T4 sat at 0% utilisation. It is
    # routed to the GPU solver, which is the SAME estimator -- same strictly convex
    # objective, verified to reach the same unique optimum as scikit-learn in fp64
    # (relative objective gap 3e-11 to 4e-09, coefficient correlation 1.000000 at
    # C = 0.01, 1.0, 100.0). The inner CV stays on scikit-learn: it is already cheap
    # because selection is subsampled, and keeping the reference implementation in
    # the loop that CHOOSES C means the hyperparameter is not selected by the new
    # code path.
    #
    # If the GPU is unavailable or the equivalence check fails, this falls back to
    # scikit-learn rather than proceeding: a slow correct answer beats a fast
    # unverified one.
    if use_gpu and _gpu_ok():
        model = GPULogisticRegression(C=chosen, random_state=seed).fit(Ztr, ytr)
    else:
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
    ap.add_argument("--shard", default=None,
                    help="K/N: this worker takes splits K, K+N, K+2N ... of the "
                         "configured split list. Sharding is BY SPLIT so each worker "
                         "holds one representation at a time and keeps the "
                         "single-worker memory profile.")
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

    # Resume reads EVERY shard, so parallel workers see each other's completed
    # runs and a re-run with different sharding never repeats work.
    shard_glob = sorted(out_csv.parent.glob(out_csv.name.replace(".csv", ".shard*.csv")))
    prev_files = ([out_csv] if out_csv.exists() else []) + shard_glob
    done, partial = set(), []
    if prev_files:
        prev = pd.concat([pd.read_csv(f) for f in prev_files], ignore_index=True)
        keycols = ["representation", "split_file", "budget", "seed"]
        counts = prev.groupby(keycols).size()
        if "n_datasets_in_run" in prev.columns:
            expected = prev.groupby(keycols)["n_datasets_in_run"].max()
            ok = counts[counts >= expected.reindex(counts.index)]
            partial = counts[counts < expected.reindex(counts.index)].index.tolist()
        else:
            # rows written before this column existed cannot be verified; treat
            # them as suspect rather than assume they are complete.
            ok = counts.iloc[0:0]
            partial = counts.index.tolist()
        done = {_runkey(*k) for k in ok.index}
        print(f"resuming: {len(done)} complete runs across {len(prev_files)} file(s)",
              flush=True)
        if partial:
            print(f"  {len(partial)} INCOMPLETE runs will be recomputed "
                  f"(partial write or pre-versioning rows); their stale rows are "
                  f"dropped at merge time by keeping the last complete write",
                  flush=True)

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

    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not (0 <= k < n):
            raise ValueError(f"--shard K/N requires 0 <= K < N, got {args.shard}")
        split_files = split_files[k::n]
        out_csv = out_csv.with_name(out_csv.name.replace(".csv", f".shard{k}.csv"))
        print(f"shard {k}/{n}: {len(split_files)} splits -> {out_csv.name}", flush=True)

    y = idx["label"].to_numpy()
    ds_all = idx["dataset"].to_numpy()      # replication unit for guardrail 5
    for sf in split_files:
        part = pd.read_parquet(sf)
        if len(part) != len(idx) or not (part["cell_id"].to_numpy() == idx["cell_id"].to_numpy()).all():
            raise ValueError(f"{sf.name}: split rows do not align with the cell index")
        p = part["partition"].to_numpy()
        splits.assert_calibration_untouched({"train", "test"})   # guardrail 6
        tr, te = p == "train", p == "test"

        for rep in reps:
            # Check whether ANY cell of this (representation, split) still needs
            # computing BEFORE loading the embedding. Loading is ~30-60 s for a
            # 229,801-row matrix, and on a resumed run the loop would otherwise pay
            # that cost for every split whose runs are already complete -- 45 min of
            # pure I/O with nothing recorded, which reads exactly like a stall.
            scheme_name = sf.name.split("__")[0]
            wanted = [(b, s) for b in budgets
                      for s in seeds_for(cfg, scheme_name, b)
                      if _runkey(rep, sf.name, b, s) not in done]
            if not wanted:
                print(f"[skip] {rep} {sf.stem}: all runs already complete", flush=True)
                continue

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
                for seed in seeds_for(cfg, scheme_name, budget):
                    # _runkey, never an ad-hoc str() cast: `budget=None` must
                    # render as "all" and a float-inferred seed as an integer, or
                    # the key silently misses and the run is recomputed.
                    key = _runkey(rep, sf.name, budget, seed)
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
                    # ONE ROW PER DATASET, not per run. Under S1/S2 a split spans
                    # every dataset, so a pooled score would leave the split seed
                    # as the only thing to aggregate over -- and testing across
                    # split seeds presents 5 resamplings of one corpus as 5
                    # replication units, which is exactly what guardrail 5
                    # forbids. Scoring per dataset gives the paired test 13 real
                    # units under every scheme. The classifier fit is unchanged;
                    # only the scoring of existing predictions is partitioned.
                    m_pool = metrics.evaluate(y[te], yp, ytr)
                    per_ds = metrics.evaluate_per_dataset(y[te], yp, ytr, ds_all[te])
                    secs = round(time.time() - t0, 2)
                    base = {"representation": rep, "split_file": sf.name,
                            "scheme": sf.name.split("__")[0],
                            "holdout_group": sf.name.split("__holdout-")[1].replace(".parquet", "")
                                             if "__holdout-" in sf.name else "",
                            "budget": budget if budget is not None else "all",
                            "seed": seed, "n_train_used": int(len(sel)),
                            "C_selected": C, "inner_cv_folds_used": cv_used,
                            "cv_selection_cells": cv_n, "n_dims_selected": int(d_sel),
                            "seconds": secs,
                            "pooled_macro_f1": m_pool["macro_f1"],
                            "pooled_macro_f1_all_test_classes":
                                m_pool["macro_f1_all_test_classes"]}
                    # Each row records how many rows its run OUGHT to have. A run
                    # writes all its dataset rows in one append; if the process dies
                    # mid-write the run key would still be present and resume would
                    # skip it, leaving that run permanently contributing a partial
                    # dataset panel. Recording the expected count makes an incomplete
                    # run detectable instead of silently wrong.
                    n_exp = len(per_ds)
                    rows_out = [{**base, "dataset": dsn, "n_datasets_in_run": n_exp, **vals}
                                for dsn, vals in per_ds.items()]
                    if not rows_out:
                        raise RuntimeError(
                            f"no dataset produced a scoreable result for {rep} "
                            f"{sf.name} budget={budget} seed={seed}; refusing to "
                            f"record a run with no replication unit")
                    _append_rows(rows_out, out_csv)
                    n_new += 1
                    if n_new % 10 == 0:
                        print(f"  {n_new} runs; last: {rep} {base['scheme']} "
                              f"b={base['budget']} pooled_f1={base['pooled_macro_f1']:.3f} "
                              f"({len(rows_out)} datasets)", flush=True)
    print(json.dumps({"new_runs": n_new, "table": str(out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
