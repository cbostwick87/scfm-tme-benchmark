# When do zero-shot single-cell foundation model embeddings beat classical representations for immune cell-type annotation in tumour microenvironment data?

All four hypotheses are resolved. Figures F1–F5 accompany this report; every
number below is reproduced from the tables in `results/`.

---

## 1. What this study is for

This is a decision rule, not a demonstration. The question is not whether
foundation models *can* produce useful cell embeddings — they can — but under
which conditions a practitioner annotating immune cells in tumour data should
prefer a zero-shot foundation-model embedding over a well-tuned classical
representation, and under which conditions they should not.

Three axes were varied: **label budget** (5, 10, 25, 50, 100, or all labelled
cells per class), **distribution shift** (within-dataset, held-out donor,
held-out dataset), and **cell-type rarity**. Two foundation models (Geneformer
V2-104M, scGPT-human) were compared against three classical baselines
(HVG+PCA, scVI, Harmony) over **13 datasets spanning 8 cancer types and
229,801 cells**, using an identical classifier head throughout so the
comparison isolates the representation.

**The unit of replication is the dataset (n=13), never the seed.** Seeds are
averaged to a dataset mean before any test.

---

## 2. Limitations — read before the results

These are placed before the findings deliberately. Three of them bias toward
conclusions this study reaches, and a reader who meets them afterwards has
already formed an impression the caveats should have shaped.

### 2.1 Caveats that flatter the classical arms

**scVI receives depth-normalised pseudo-counts, not raw UMI counts.** TISCH2
distributes `log1p(CP10K)`; measured, per-cell `sum(expm1(x))` is exactly
10,000. scVI's negative-binomial likelihood requires integer counts. True
counts are exactly recoverable for 9 of 13 datasets but not for the 4
plate-based ones, and using true counts only where available would inject a
per-dataset artefact into a contrast whose unit of replication *is* the
dataset. `round(expm1(x))` was therefore applied uniformly. Consequence:
library size is ~10⁴ by construction, so scVI's library-size latent is
uninformative.

**scVI's 60-epoch budget was never reached by early stopping.** All 75 fits ran
the full cap, so the budget — not convergence — terminated training.

Together these mean **scVI's numbers are a lower bound for two independent
reasons**, and this study does not establish scVI's ceiling.

**Harmony is fit transductively — a declared deviation from the leakage
control.** harmonypy 2.0 exposes no out-of-sample projection. Harmony therefore
sees test-cell *expression* (never labels) while every other arm does not. This
**advantages a baseline**, biasing toward "classical methods win", which is
this study's own expected direction. It must not be read as evidence for H1.
PCA remains train-only; only the correction step sees test coordinates.

### 2.2 Scope caveats

- **One checkpoint per model family.** Results are about *these two
  checkpoints*, not about "foundation models" as a class.
- **Zero-shot only.** No fine-tuning. Fine-tuned performance is out of scope
  and may differ substantially.
- **n=13 datasets.** Adequate for the large effects reported; underpowered for
  weak associations, which is why H4 is reported as "no evidence for" rather
  than "evidence against".
- **Pretraining-overlap confound, partially bounded.** BRCA_GSE176078 is a
  suspected member of scGPT's CELLxGENE-derived pretraining corpus, retained
  with a pre-specified sensitivity analysis. **That analysis is reported in
  §3.8: excluding the dataset changes no conclusion** — zero sign flips and
  zero significance changes across the 36 primary contrasts, largest shift
  0.0095, and the flagged dataset is not an outlier in the direction
  memorisation would predict (max |z| = 1.29 across 36 cells).
- **Two post-hoc design changes**, both declared: extra seeds under
  leave-dataset-out at budgets 5, 10, 25 and 50 (5→20 — four budgets, not the
  single cell an earlier draft of this bullet implied), and reduced seeds at the
  unrestricted budget (5→2, justified by the measurement that the seed does not
  change the training data there). **The first is audited in §3.9: recomputing
  the affected contrasts on the original 5 seeds changes nothing** — zero sign
  flips, zero significance changes, largest shift 0.0076 — so the boost did not
  produce the H2 conclusion. It also bought almost no precision (CI-width ratio
  0.99), because the replication unit is the dataset, not the seed.
- **Gene-space coverage varies by dataset** (Geneformer vocabulary coverage
  0.560–0.969), a per-dataset caveat rather than a property of the method.
- **The ultra-rare regime is untested.** The <1% prevalence stratum contained
  only 12 (dataset, cell type) instances across the whole corpus — too few to
  test. Conclusions about rarity extend down to ~1% prevalence and no further.
- **The mechanism analysis (§3.7) is descriptive.** It correlates across five
  representations, which are not independent replicates, on a fixed subsample.
  It motivates the next experiment; it does not establish causation.

---

## 3. Results

All contrasts are paired across 13 datasets, with bootstrap 95% CIs, Cohen's
dz, Wilcoxon signed-rank p, and BH-FDR correction across the 108-contrast
family. **49 of 108 contrasts are significant at FDR 0.05; none of them is
flagged negligible** (|Δ| < 0.02).

### 3.1 H1 — in-distribution, at restricted label budgets: no advantage

At every restricted budget from 5 to 100 labelled cells per class, within
dataset, neither foundation model differs detectably from HVG+PCA: all ten
contrasts return "no detectable difference" and **none is significant at
FDR 0.05** (p range 0.150–1.000).

Point estimates run from −0.059 to +0.001 — where they lean at all, they lean
*against* the foundation models. Two details matter for reading this honestly
rather than as a flat null:

- The largest deviation is **scGPT at 5 labels per class, Δ = −0.059
  [−0.115, −0.010]**, winning on only 3 of 13 datasets. Its bootstrap CI does
  *not* cross zero, yet its FDR-corrected p is 0.150. The CI is on the mean
  difference while the pre-specified primary test is Wilcoxon signed-rank on
  ranks, so the two can disagree; the pre-specified test governs. The correct
  reading is a *suggestive but unconfirmed* disadvantage for scGPT at the
  smallest budget — not an established one, and not a flat null either.
- Excluding that cell, the remaining eight contrasts (budgets 10–100) span
  −0.017 to +0.001, every CI crosses zero, and every FDR p > 0.64.

**This confirms the study's stated expectation, and it is the finding with the
most direct practical consequence.**

### 3.2 The exception: unrestricted labels, in distribution

Given *all* available labels within dataset, both models beat HVG+PCA on
**13 of 13 datasets**: Geneformer +0.051 [+0.040, +0.063], scGPT +0.052
[+0.039, +0.063], both FDR p = 0.0018, dz ≈ +2.3. This contradicts the stated
expectation and is reported as such.

### 3.3 The strongest result: the advantage reverses under dataset shift

Under leave-dataset-out, HVG+PCA wins, and **the gap widens as labels
increase**:

| Budget | Geneformer − HVG+PCA | FDR p | Datasets won |
|---|---|---|---|
| 5 | −0.053 | 0.255 | 5/13 |
| 10 | −0.064 | 0.069 | 4/13 |
| 25 | −0.076 | 0.006 | 1/13 |
| 50 | −0.083 | 0.002 | 1/13 |
| 100 | −0.100 | 0.002 | 0/13 |
| all | −0.107 | 0.002 | 0/13 |

scGPT shows the same pattern, more weakly (−0.083 at unrestricted budget,
FDR p = 0.004, 2/13). **This is the opposite of the "foundation models
generalise better" prior.**

### 3.4 H2 — against the other mandatory baselines

Direction depends on the same axis. In distribution at unrestricted labels,
both scFMs beat scVI (+0.092) and Harmony (+0.137). Under dataset shift, both
lose to scVI (−0.086 to −0.111) and to Harmony (−0.071 to −0.096) — despite
scVI operating under two handicaps and Harmony under a declared advantage.

### 3.5 H4 — gene-space overlap: a null

No association between train–test gene-space overlap and the scFM advantage,
across overlap spanning 0.538–0.946: Spearman ρ = −0.34 (p=0.25) and −0.13
(p=0.67) at 10 labels; +0.02 (p=0.96) and −0.09 (p=0.78) at all labels. Every
point estimate is at or below zero — if anything, opposite to the prediction.
Underpowered at n=13 to exclude a weak association.

### 3.6 H3 — rarity: no rescue, and no trend

Per-class F1 was recovered by refitting at the already-selected regularisation
(no inner CV, so no selection on test). Classes were binned by pre-specified
prevalence strata; the paired unit is the (dataset, cell type) pair.

**Rarity does not change the direction of any result.** In distribution at
unrestricted labels, the scFM gain is present in *every* stratum and is largest
in the middle one (Geneformer +0.057 uncommon vs +0.035 rare, +0.037 common).
Under dataset shift the penalty is present in every stratum too, and at
unrestricted labels it is *largest for the rarest testable classes*
(Geneformer −0.168, scGPT −0.175 in the 1–5% stratum, both FDR p < 0.005) —
the opposite of the hypothesis that foundation models earn their keep on rare
populations.

Tested continuously rather than binned, the effect does not trend with
prevalence in 11 of 12 cells (|ρ| ≤ 0.14, p > 0.15). The single exception is
Geneformer under dataset shift at 10 labels (ρ = −0.26, p = 0.010), which is
uncorrected for the 12 tests and should not be read as an established gradient.

The <1% stratum had only 12 class-instances and was not tested. **This study
therefore says nothing about genuinely ultra-rare populations**, which is where
a rarity benefit would most plausibly live.

### 3.7 Mechanism: the embeddings encode dataset identity

The reversal in §3.3 has a measurable candidate explanation. Computing
silhouette width twice on the same cells — once labelled by cell type (the
signal) and once by dataset (the nuisance):

| Representation | Cell type | Dataset |
|---|---|---|
| Geneformer | +0.017 | **+0.047** |
| scGPT | +0.036 | +0.026 |
| HVG+PCA | +0.020 | −0.010 |
| scVI | **+0.059** | −0.035 |
| Harmony | −0.001 | −0.053 |

**Geneformer separates datasets more strongly than it separates cell types.**
Both foundation models have positive dataset silhouette; all three classical
representations have negative. Across the five representations, dataset
silhouette orders perfectly with the macro-F1 lost going from within-dataset to
held-out-dataset (Spearman ρ = +1.00, exact permutation p = 0.017).

This is consistent with the frozen embedding encoding batch structure a linear
head cannot discount, and it is *descriptive*: n = 5 representations are not
independent replicates, the silhouette is computed on a fixed subsample, and
correlation across five methods cannot establish causation. It motivates the
next experiment rather than concluding this one.

---

### 3.8 Robustness check 1 — pretraining-overlap sensitivity

`results/pretraining_overlap_audit.md` pre-specified that every primary contrast
be recomputed with `BRCA_GSE176078` excluded. **That analysis was specified
before any result was known but was never executed**; it is run here and
reported in full. Nothing was re-tuned and no pre-specified analysis was
altered: the same frozen aggregate → paired-contrast → BH-FDR path is used, and
the only difference is which datasets enter it. The n=13 leg reproduces the
shipped T3 exactly on all 108 contrasts (max difference 1e-16), which is
asserted rather than assumed.

**No conclusion changes.** Across the 36 primary contrasts against HVG+PCA
there are **zero sign flips and zero significance changes**; the largest shift
in any point estimate is 0.0095, below the pre-specified negligibility floor of
0.02. The headline contrasts:

| Contrast | n=13 | n=12 (excl. flagged) |
|---|---|---|
| Within dataset, all labels, Geneformer | +0.0513 [+0.0400, +0.0632], p=0.0018 | +0.0533 [+0.0415, +0.0653], p=0.0035 |
| Within dataset, all labels, scGPT | +0.0516 [+0.0392, +0.0631], p=0.0018 | +0.0507 [+0.0377, +0.0635], p=0.0035 |
| Held-out dataset, all labels, Geneformer | −0.1071 [−0.1988, −0.0507], p=0.0018 | −0.1086 [−0.2056, −0.0470], p=0.0035 |
| Held-out dataset, all labels, scGPT | −0.0828 [−0.1354, −0.0411], p=0.0043 | −0.0857 [−0.1417, −0.0408], p=0.0085 |

Across the full 108-contrast family (including scVI and Harmony comparators)
there is **1 sign flip and 3 significance changes**, all reported rather than
suppressed. None is a primary contrast and all four sit on effects at or below
the negligibility floor or just at the FDR boundary: `scgpt − harmony` at
held-out-dataset/10 labels moves +0.0028 → −0.0008 (both p > 0.87, i.e. a sign
flip on a non-effect), and three comparator contrasts cross from p≈0.047 to
p≈0.07 — losing significance purely because dropping a dataset costs one
degree of freedom on n=13.

**H3 (rarity) is unaffected too.** Recomputed over the per-class table, 36 rows:
1 sign flip and 1 significance change, both in the 5–20% stratum at 10 labels on
effects of |Δ| ≤ 0.012, and both far below the negligibility floor. Every
budget-all stratum — where the H3 conclusion lives — retains its sign and its
significance, including the rarest-classes result under dataset shift
(Geneformer −0.168 → −0.177, scGPT −0.175 → −0.187, both still p < 0.003).

**The audit's second pre-specified check also returns nothing.** It asked
whether the scFM margin is unusually large *on the flagged dataset* relative to
its margin elsewhere, which would itself be evidence of memorisation. Across 36
scheme × budget × model cells the largest standardised deviation is |z| = 1.29
and **no cell exceeds |z| = 2**. The flagged dataset is not an outlier in the
direction memorisation would predict.

Tables: `results/T10_pretraining_sensitivity.csv`,
`results/T11_memorisation_check.csv`,
`results/T13_h3_pretraining_sensitivity.csv`.

### 3.9 Robustness check 2 — does the post-hoc seed increase carry H2?

Seeds were raised from 5 to 20 after the design was fixed, under
leave-dataset-out. That is the condition H2 lives in, so a reader may reasonably
ask whether the added replication produced the conclusion rather than merely
measuring it. The affected contrasts are recomputed here on **seeds 0–4 only**
— the original pre-boost set — beside the 20-seed values.

One correction to §2.2 while reporting this: the boost was applied to **four
budgets (5, 10, 25 and 50)**, not to a single low-budget cell. All four are
included below.

**The conclusion is unchanged.** Across all 24 affected contrasts:
**zero sign flips, zero significance changes**, largest shift in any point
estimate 0.0076 — again below the negligibility floor. The H2 comparisons
against scVI and Harmony hold their direction and their significance at 5 seeds.

The more interesting result is *why* it makes so little difference. Median CI
width is **0.0819 at 20 seeds versus 0.0831 at 5** — a ratio of 0.99, i.e. the
extra 15 seeds bought essentially no precision. That is exactly what the
design's own replication rule predicts: the unit of replication is the dataset
(n=13), so the interval is set by between-dataset variance, which added seeds
cannot reduce. The seed boost lowered within-dataset resampling noise, which was
never the binding term.

So the post-hoc change was close to inert. It could not have manufactured the H2
result, and — stated plainly because it cuts against the decision to make it —
it also did not deliver the power increase it was intended to deliver. The
underpowered-cell concern that motivated it is better addressed by more
datasets, not more seeds.

Table: `results/T12_seed_sensitivity.csv`.

## 4. Decision rule

On this evidence, for immune cell-type annotation in TME data:

1. **Annotating within a dataset you already have labels for, on a restricted
   budget** — use HVG+PCA. There is no measurable benefit from a zero-shot
   foundation-model embedding, and the classical pipeline is cheaper, faster,
   and interpretable.
2. **Annotating within a dataset with abundant labels** — a foundation-model
   embedding gives a real but modest gain (~+0.05 macro-F1, consistent across
   all 13 datasets).
3. **Transferring to an unseen dataset** — use HVG+PCA. The foundation-model
   embeddings are measurably worse here, and increasingly so as labels grow.
4. **Do not choose based on gene-space overlap.** It carries no signal.

---

## 5. Downstream implications

The reversal in (3) is the result with consequences beyond this benchmark, and
§3.7 supplies direct evidence for why it happens: **both foundation-model
embeddings carry positive dataset silhouette while all three classical
representations carry negative**, and dataset silhouette orders perfectly with
the transfer penalty across the five methods. The frozen embedding appears to
encode dataset-of-origin structure that a linear head cannot discount, and more
labels in a shifted domain help the classical pipeline more than they help a
representation whose axes partly encode the wrong thing.

That is a candidate mechanism, not a demonstrated one — five representations
are not five independent replicates. The natural next experiments follow
directly: (a) test whether an explicit batch-correction step applied *to the
foundation-model embedding* recovers the transfer performance, and (b) test
whether fine-tuning, which is out of scope here, removes the dataset structure
that zero-shot embedding retains. Either result would sharpen the decision rule
from "do not use these embeddings for transfer" to "use them for transfer only
after correction".

The rarity result adds a second implication: §3.6 rules out the most common
defence of scFM embeddings — that they earn their keep on rare populations.
Under dataset shift the penalty was *largest* for the rarest testable classes.
The genuinely ultra-rare regime (<1% prevalence) remains untested here and is
the one place that defence could still survive.

For practitioners, the operational reading is that **the deployment scenario
determines the answer**, and the scenario where foundation models are most
often advocated — transfer to new data — is the one where they performed worst
here.

---

## 6. Reproducibility

Environment lockfile, split indices, run manifest and DECISIONS log are in the
repository. The calibration partition of the three-way split was written and
never read by this project; it is reserved for downstream conformal-prediction
work. Analysis code was frozen and hash-recorded before the full-grid
statistics were run.

Compute: a single NVIDIA T4 instance with 4 vCPU and 16 GiB RAM.
