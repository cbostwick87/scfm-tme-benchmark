# When do zero-shot single-cell foundation model embeddings beat classical representations for immune cell-type annotation in tumour microenvironment data?

**Status: draft. Figures F2 and F3 pending a per-class rescoring pass now in progress.**

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
  with a pre-specified sensitivity analysis.
- **Two post-hoc design changes**, both declared: extra seeds in the
  low-budget × held-out-dataset cell (5→20), and reduced seeds at the
  unrestricted budget (5→2, justified by the measurement that the seed does not
  change the training data there).
- **Gene-space coverage varies by dataset** (Geneformer vocabulary coverage
  0.560–0.969), a per-dataset caveat rather than a property of the method.

---

## 3. Results

All contrasts are paired across 13 datasets, with bootstrap 95% CIs, Cohen's
dz, Wilcoxon signed-rank p, and BH-FDR correction across the 108-contrast
family. **49 of 108 contrasts are significant at FDR 0.05; none of them is
flagged negligible** (|Δ| < 0.02).

### 3.1 H1 — in-distribution, at restricted label budgets: no advantage

At every restricted budget from 5 to 100 labelled cells per class, within
dataset, neither foundation model differs detectably from HVG+PCA. Deltas span
−0.017 to +0.001, all CIs cross zero, all FDR p > 0.6.

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

### 3.6 H3 — rarity

*Pending the per-class rescoring pass.*

---

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

The reversal in (3) is the result with consequences beyond this benchmark. The
widening gap with label budget suggests the classical pipeline extracts
increasing benefit from labels in a shifted domain while the frozen embedding
does not — consistent with the zero-shot embedding encoding
dataset-of-origin structure that a linear head cannot discount. This is a
hypothesis the present design cannot test, and it is the natural next
experiment.

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
