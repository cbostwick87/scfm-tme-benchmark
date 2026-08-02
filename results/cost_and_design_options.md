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

**`budget='all'` is 93% of the entire grid.** The reason is structural, not a
bug: it is the only budget that fits on the full training partition. The capped
budgets sample at most 100 cells per class, so ~1,400 cells for 14 classes --
roughly **100x less training data**. The L-BFGS fit dominates everything else.

### What has already been tried

| configuration | s/run at `all` | net throughput |
|---|---|---|
| 1 worker, 4 BLAS threads (original) | 309 | baseline |
| 3 workers, 1 thread each | 884 | **1.05x** |
| GPU refit, 1 worker, 4 threads | 357 | **0.86x** |

- **CPU sharding bought nothing.** The work is dense BLAS on a 4-core box;
  partitioning cores does not create capacity (load average sat at 3.06/4).
- **The GPU refit is 2.47x faster than the sharded configuration but slightly
  slower than the original single-worker CPU setup.** It was verified to solve
  the identical objective (relative gap 3e-11 to 4e-09, coefficient correlation
  1.000000 at C = 0.01/1.0/100.0), so it is a legitimate swap -- it is just not
  a speed win. Retained because it frees CPU for the inner CV.
- The fits **converge** (630 of 2,000 iterations), so the cost is real work, not
  a solver failing to terminate.
- Inner-CV selection is already subsampled to <=20,000 cells; the remaining cost
  is the final refit on the full partition, which cannot be subsampled without
  changing what the study measures.

---

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

Cutting seeds at `all` from 5 to 2 costs **0.01 of power** at every effect size
tested. Cutting them at budget=5 would cost 0.26 at delta=0.05. These axes are
not interchangeable.

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

**Option A.** It removes 56% of the remaining compute for 0.01 of power, because
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
