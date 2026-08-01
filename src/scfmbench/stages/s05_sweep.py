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


def load_embedding(emb_dir: pathlib.Path, model: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all shards of a model's embedding, returning (matrix, cell_ids)."""
    d = emb_dir / model
    fs = sorted(d.glob("*.npz"))
    if not fs:
        raise FileNotFoundError(f"no embedding shards in {d}")
    E, ids = [], []
    for f in fs:
        with np.load(f, allow_pickle=True) as z:
            E.append(z["emb"]); ids.append(z["cell_id"])
    return np.vstack(E), np.concatenate(ids)


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


def fit_predict(Ztr, ytr, Zte, seed: int, cv_grid, n_folds: int):
    """Multinomial LR with L2; C chosen by inner CV on TRAIN only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GridSearchCV

    classes, counts = np.unique(ytr, return_counts=True)
    # inner CV needs at least 2 members per class per fold; with tiny label
    # budgets that is impossible, so fall back to a fixed C rather than
    # silently letting sklearn pick a degenerate split.
    k = int(min(n_folds, counts.min()))
    base = LogisticRegression(max_iter=2000, multi_class="multinomial",
                              class_weight="balanced", random_state=seed)
    if k >= 2 and len(classes) > 1:
        gs = GridSearchCV(base, {"C": list(cv_grid)},
                          cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                          scoring="f1_macro", n_jobs=1, refit=True)
        gs.fit(Ztr, ytr)
        model, chosen, cv_used = gs.best_estimator_, gs.best_params_["C"], k
    else:
        model = base.set_params(C=1.0).fit(Ztr, ytr)
        chosen, cv_used = 1.0, 0
    return model.predict(Zte), chosen, cv_used


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
    reps = args.representations or list(cfg["representations"])
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
    for rep in reps:
        if rep not in cache:
            E, ids = load_embedding(emb_dir, rep)
            order = pd.Index(ids).get_indexer(pd.Index(idx["cell_id"]))
            if (order < 0).any():
                missing = int((order < 0).sum())
                raise ValueError(
                    f"{rep}: {missing} cells in the index have no embedding. Every cell "
                    f"must be embedded exactly once per model; refusing to evaluate on a "
                    f"partial cache."
                )
            cache[rep] = E[order]
            print(f"loaded {rep}: {cache[rep].shape}", flush=True)
        Z = cache[rep]

        for sf in sorted(split_dir.glob("*.parquet")):
            if sf.name == "cell_index.parquet":
                continue
            part = pd.read_parquet(sf)
            if len(part) != len(idx) or not (part["cell_id"].to_numpy() == idx["cell_id"].to_numpy()).all():
                raise ValueError(f"{sf.name}: split rows do not align with the cell index")
            p = part["partition"].to_numpy()
            splits.assert_calibration_untouched({"train", "test"})   # guardrail 6
            tr = p == "train"; te = p == "test"
            y = idx["label"].to_numpy()

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
                    scale = fit_standardiser(Z[sel])      # fit on TRAIN subset only
                    yp, C, cv_used = fit_predict(scale(Z[sel]), ytr, scale(Z[te]),
                                                 seed, grid, n_folds)
                    m = metrics.evaluate(y[te], yp, ytr)
                    scheme = sf.name.split("__")[0]
                    row = {"representation": rep, "split_file": sf.name, "scheme": scheme,
                           "holdout_group": sf.name.split("__holdout-")[1].replace(".parquet", "")
                                            if "__holdout-" in sf.name else "",
                           "budget": budget if budget is not None else "all",
                           "seed": seed, "n_train_used": int(len(sel)),
                           "C_selected": C, "inner_cv_folds_used": cv_used,
                           "seconds": round(time.time() - t0, 2),
                           **{k: v for k, v in m.items() if k != "per_class_f1"},
                           "per_class_f1_json": json.dumps(m["per_class_f1"])}
                    pd.DataFrame([row]).to_csv(out_csv, mode="a", index=False,
                                               header=not out_csv.exists())
                    n_new += 1
                    if n_new % 10 == 0:
                        print(f"  {n_new} new runs; last: {rep} {scheme} b={row['budget']} "
                              f"f1={row['macro_f1']:.3f}", flush=True)
    print(json.dumps({"new_runs": n_new, "table": str(out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
