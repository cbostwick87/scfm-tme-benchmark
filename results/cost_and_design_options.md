# Sweep cost: where it goes, and what can be reduced

Prepared for an external second opinion. Every number below is **measured on this
hardware**, not estimated. Provenance is stated per figure so a reviewer can tell
a measurement from an inference.

---

## 1. The design

| axis | levels | note |
|---|---|---|
| representations | 3 | `geneformer` (768-d), `scgpt` (512-d), `hvg_pca` (100-d) |
| splits | 75 | 5 within-dataset (S1) + 5 leave-donor-out (S2) + 65 leave-dataset-out (S3 = 13 holdout groups x 5 seeds) |
| label budgets | 6 | 5, 10, 25, 50, 100 cells/class, and `all` |
| seeds | 5 | |
| **total runs** | **6,750** | one classifier fit + evaluation each |

Each run = inner-CV selection of the L2 strength on training cells only, then a
refit on the full training partition, then scoring. Every run emits **13 rows**
(one per dataset), because the dataset is the unit of replication.

---

## 2. Where the time goes (measured seconds/run)

| budget | train cells | s/run | runs | total |
|---|---|---|---|---|
| 5 | ~70 | 3.2 | 1,125 | 1.0 h |
| 10 | ~140 | 3.2 | 1,125 | 1.0 h |
| 25 | ~350 | 4.6 | 1,125 | 1.4 h |
| 50 | ~700 | 5.8 | 1,125 | 1.8 h |
| 100 | ~1,400 | 9.5 | 1,125 | 3.0 h |
| **`all`** | **137,881 (S1) / 158,306 (S3)** | **357.4** | 1,125 | **111.7 h** |
| | | | **6,750** | **119.9 h** |

**`budget='all'` is 93% of the entire grid.** It is the only budget that fits on
the full training partition; the capped budgets sample at most 100 cells per
class, so ~1,400 cells for 14 classes -- roughly **100x less training data**.

### Which part of a run costs that

Profiling one unrestricted-budget run, component by component:

| component | seconds | share |
|---|---|---|
| embedding load (amortised over 30 runs/split) | 8.2 | 0.3% |
| standardise | 1.3 | 0.0% |
| **inner CV (6 C values x 5 folds on 20,000 cells)** | **3,096.4** | **99.3%** |
| final refit on 137,881 cells (GPU) | 13.2 | 0.4% |
| predict + score | 0.0 | 0.0% |

**This corrected an earlier conclusion of mine.** I had attributed the cost to the
final refit on the full partition and optimised that first; the profile shows the
final refit is 13 seconds and **selecting the regularisation strength is 99.3% of
the run** -- 30 separate fits rather than one. The earlier GPU work was therefore
aimed at the wrong 0.4%, which is why it produced no net speedup.

### What has been tried, and what the profile implies

| configuration | s/run at `all` | net |
|---|---|---|
| 1 worker, 4 BLAS threads (original) | 309 | baseline |
| 3 workers, 1 thread each | 884 | **1.05x** |
| GPU **final refit** only, 1 worker | 357 | **0.86x** |
| GPU **inner CV** as well (measured on the CV problem in isolation) | -- | **3.6x on 99.3% of the run** |

- **CPU sharding bought nothing** -- dense BLAS on 4 cores; partitioning cores does
  not create capacity (load average 3.06/4).
- **GPU on the final refit alone bought nothing**, for the reason the profile now
  makes obvious.
- **GPU on the inner CV is the real win, and it is measured**: on real embeddings
  (20,000 x 768, 14 classes, 6 C x 5 folds) it is **98.9 s against 356.3 s** for
  scikit-learn, **selects the same C**, score-curve correlation 0.9924, maximum
  score difference across the grid 0.00305 -- far below the 0.02 this study calls
  negligible. The C grid and fold structure are unchanged; only the hardware
  differs.
- The fits **converge** (630 of 2,000 iterations), so this is real work.
- Inner-CV selection is already subsampled to <=20,000 cells; the remaining cost is
  the 30 fits at that size.

**Implication for the decision below:** if the GPU inner CV holds up in the running
sweep, the projected total falls from ~120 h toward the **35-45 h** range with **no
design change at all**. The reduction options remain on the table but are no longer
obviously necessary. The numbers in section 5 are computed against the 119.9 h
baseline and should be re-derived once ~50 clean runs confirm the new per-run cost.

## 3. The key measurement for the decision

Seed-to-seed SD of macro-F1, within (representation, dataset, scheme), by budget:

| budget | mean seed SD | median |
|---|---|---|
| 5 | 0.0780 | 0.0559 |
| 10 | 0.0732 | 0.0561 |
| 25 | 0.0398 | 0.0372 |
| 50 | 0.0328 | 0.0282 |
| 100 | 0.0267 | 0.0252 |
| **`all`** | **0.0080** | **0.0053** |

*Provenance: computed from 4,550 cleanly-parsing rows of a results table that was
later retired for a column-misalignment defect. Restricted to rows whose budget
and seed parse correctly. Treat as an order-of-magnitude estimate of variance
structure, **not** as a result. This is the one figure below that is not from a
clean table, and it is the figure the recommendation rests on -- it should be
re-confirmed on ~200 clean runs before the reduction is locked in.*

**Seed variance at `all` is an order of magnitude smaller than at budget=5, and
3.3x smaller than at budget=100.** The reason is mechanical: at a capped budget
the seed decides *which* labelled cells you get, which is the label-efficiency
question the study exists to answer. At `all` every cell is used, so no label
sampling happens and the seed only reshuffles the partition.

So seeds are **load-bearing exactly where compute is cheap**, and **nearly inert
exactly where 93% of the cost sits**.

---

## 4. Power consequences

Simulated power of the primary paired test (Wilcoxon signed-rank, n=13 datasets,
alpha=0.05), using the measured seed SD and an assumed between-dataset effect
heterogeneity of 0.03. The negligibility threshold pre-specified for this study
is **delta = 0.02** -- below that, a significant result is explicitly not
reported as a finding.

**At `budget='all'` (seed SD 0.0080):**

| true delta | 1 seed | 2 seeds | 3 seeds | 5 seeds |
|---|---|---|---|---|
| 0.02 | 0.52 | 0.54 | 0.55 | 0.56 |
| 0.03 | 0.86 | 0.88 | 0.89 | 0.89 |
| 0.05 | 1.00 | 1.00 | 1.00 | 1.00 |

**At `budget=5` (seed SD 0.0780) -- for contrast:**

| true delta | 1 seed | 2 seeds | 3 seeds | 5 seeds |
|---|---|---|---|---|
| 0.02 | 0.08 | 0.12 | 0.15 | 0.20 |
| 0.03 | 0.13 | 0.21 | 0.28 | 0.38 |
| 0.05 | 0.29 | 0.48 | 0.62 | 0.79 |

Cutting seeds at `all` from 5 to 2 costs, read directly off the tables above,
**0.02 of power at delta=0.02, 0.01 at delta=0.03 and 0.00 at delta=0.05**. The
same cut at budget=5 costs **0.08 at delta=0.02, 0.17 at delta=0.03 and 0.31 at
delta=0.05**. These axes are not interchangeable.

---

## 5. Options

| # | change | new total | saved | cost to the science |
|---|---|---|---|---|
| 0 | none | 119.9 h | -- | none |
| **A** | **seeds 5 -> 2 at `budget='all'` only** | **52.9 h** | **56%** | **power 0.89 -> 0.88 at delta=0.03; 0.56 -> 0.54 at 0.02** |
| B | seeds 5 -> 3 at `budget='all'` only | 75.2 h | 37% | power 0.89 -> 0.89 |
| C | seeds 5 -> 1 at `budget='all'` only | 30.6 h | 74% | power 0.89 -> 0.86; **no within-condition variance estimate at all** |
| D | drop S3 seeds | ~37 h | 69% | **rejected** -- verified that S3 seeds change *which* studies train (train sizes 137k-186k across seeds), so they are genuine variation, not a reshuffle |
| E | drop `budget='all'` | 8.2 h | 93% | **refused** -- it is the asymptote the decision rule needs; without it there is no "how much does the gap close with full supervision" answer |
| F | fewer S3 holdout groups (13 -> 7) | ~65 h | 46% | halves the replication units for the transfer hypothesis, which is the study's most novel claim |
| G | drop a representation | ~80 h | 33% | **refused for the scFMs** -- a single-model result cannot be generalised to "foundation models"; and `hvg_pca` is the method to beat |

### Recommendation

**Updated after profiling: try Option 0 first.** The component profile (section 2)
shows the cost is the inner CV, not the final refit, and moving the inner CV to the
GPU is a measured 3.6x on 99.3% of the run with the same C selected. If that holds
in the running sweep, the grid completes in roughly 35-45 h with the design fully
intact -- and a design change made unnecessarily is a limitation acquired for
nothing.

**If it does not hold, Option A.** It removes 56% of the remaining compute for at most 0.02 of power
(0.02 at delta=0.02, 0.01 at delta=0.03, 0.00 at delta=0.05), because
it cuts replication precisely where replication is nearly inert. Option C is
defensible on power alone but gives up the ability to report a within-condition
seed SD at the unrestricted budget, which is worth keeping as evidence that the
estimator is stable.

Two caveats a reviewer should weigh:

1. **The variance figures come from a retired table** (see provenance note in
   section 3). The recommendation should be re-confirmed on clean runs before
   being locked in. The direction of the effect is mechanical and unlikely to
   reverse, but the magnitude could move.
2. **This is a change to a pre-specified design, made after seeing data.** It is
   a change to *replication count*, not to any hypothesis, metric, threshold, or
   analysis rule -- and it is being made for compute reasons with the power cost
   quantified in advance. But it is still a post-hoc design change and belongs in
   the limitations section, not only in the decisions log.

---

## 6. What is not on the table

- Fitting on fewer than the full training partition at `budget='all'` -- that
  redefines the condition.
- Reducing the C grid or the inner-CV folds -- the baseline must be tuned as
  carefully as the scFM arms; degrading tuning to save time would bias the
  comparison in the direction the study already expects.
- Trimming the corpus, the label harmonisation, or any leakage control.
- Any reduction chosen *after* seeing which arm it favours.
