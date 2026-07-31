# Pretraining-corpus overlap audit

Zero-shot scFM embeddings are not fit to any partition of this benchmark, so they cannot
leak through the split. They can, however, leak through **pretraining**: if a benchmark
dataset was part of a model's pretraining corpus, that model has already seen those cells
and its apparent advantage is partly memorisation. This is the single most common
unaddressed confound in the scFM literature, and this document records what could and
could not be established for each dataset in this corpus.

## What each corpus contains

**Geneformer / Genecorpus-30M.** Approximately 30 million human single-cell transcriptomes
(V1, June 2021), assembled from publicly available scRNA-seq data. The corpus explicitly
**excluded cells with high mutational burden** — malignant cells and immortalised cell
lines — because such cells could confound the model's learned biology. The default
checkpoint used here is **Geneformer-V2-104M** (Dec 2024), trained on a substantially
larger (~104M-cell) corpus whose composition is documented in less detail than V1's.
Neither release publishes a per-accession file list.

**scGPT.** Pretrained on approximately 33 million **normal / non-diseased** human cells
obtained from CZ CELLxGENE Discover. Because the corpus was drawn from CELLxGENE, presence
of a study in CELLxGENE is a checkable proxy for possible membership.

## Method

Two tests were applied per dataset, and the limits of each are stated rather than glossed:

1. **CELLxGENE Census presence (scGPT).** Each dataset's primary publication DOI was
   matched against the Census dataset index at pinned version **2025-11-08** (1,845
   datasets, 313 collections, 217,768,036 cells).

   A first attempt searched Census metadata for GEO accessions and returned zero hits for
   all 13 datasets. **That result was invalid and was discarded**: control probes for
   accessions known to be in Census also returned zero, because Census metadata does not
   carry GEO accession strings. The audit was redone on DOI and citation fields. This is
   recorded because the invalid version would have produced a comfortable, wrong answer —
   "no overlap anywhere" — and it is exactly the failure mode this audit exists to prevent.

2. **Malignant-cell exclusion (Geneformer).** Genecorpus-30M excluded high-mutational-burden
   cells, so the malignant compartment of these tumour datasets is unlikely to appear.
   Non-malignant tumour-infiltrating immune cells — which are what this study classifies —
   cannot be positively excluded without a corpus manifest.

## Result

| dataset | flag | basis |
|---|---|---|
| **BRCA_GSE176078** | **suspected overlap** | Wu et al. 2021 (`10.1038/s41588-021-00911-1`) is present in Census 2025-11-08 as **13 datasets / 225,725 cells**, including T-cell, B-cell, myeloid and plasmablast compartments — the immune cells this study classifies. |
| the other 12 datasets | no evidence of overlap | Primary publication not found in Census 2025-11-08 by DOI. |

## How this is handled in the analysis

- The flag is a column in **T1**, carried through every downstream table.
- `BRCA_GSE176078` is **retained**, not dropped. Dropping the one dataset with identifiable
  overlap would bias the corpus toward datasets whose overlap merely could not be detected,
  which is worse than measuring the effect. Its retention is what makes the sensitivity
  analysis below possible.
- **Pre-specified sensitivity analysis:** every primary contrast (H1, H2) will be recomputed
  with `BRCA_GSE176078` excluded. If the conclusion changes, that is reported as the
  headline caveat rather than a footnote. This is specified now, before any result is
  known, so it cannot become a post-hoc choice.
- If scFM embeddings outperform baselines *specifically* on this dataset relative to their
  margin elsewhere, that pattern is itself evidence of memorisation and will be reported as
  such.

## Honest limits of this audit

- **Absence of evidence is not evidence of absence.** Neither corpus publishes a
  per-accession manifest. A dataset can be absent from Census yet present in a pretraining
  corpus by another route, and TISCH2 re-processes raw GEO submissions independently.
- Census content **changes between versions**; this audit is valid for 2025-11-08 only.
- scGPT's corpus is described as non-diseased, so even a Census-present tumour study may
  have been filtered out of the 33M-cell subset. The flag therefore marks *risk*, not
  confirmed contamination.
- Geneformer-V2's corpus is larger and less precisely documented than V1's, so the
  uncertainty is greater for the checkpoint actually used than the V1 description implies.
- This audit addresses **dataset-level** overlap. It cannot detect a study that contributed
  a *different* set of cells from the same donors to a pretraining corpus.
