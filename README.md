# When do single-cell foundation models beat a well-tuned linear baseline?

**A leakage-controlled audit of zero-shot scFM embeddings for tumour-immune cell-type annotation.**

![headline figure](figures/F1_performance_vs_label_budget.png)

> **Status: IN PROGRESS.** The headline finding will be stated here, in one sentence,
> as soon as the pre-specified analysis is complete - including if it is negative.
> This placeholder is deliberate: the finding is not written before the numbers exist.

## The question

Single-cell foundation models (scFMs) are promoted as general-purpose cell
representations, but the peer-reviewed literature is genuinely split on whether their
embeddings beat simple, well-tuned classical representations. This study asks a
narrower and more useful question than "are scFMs good":

**Under which conditions do zero-shot scGPT and Geneformer embeddings outperform
HVG+PCA, scVI, Harmony and celltypist for immune cell-type annotation in tumour
microenvironment data?**

Three axes are varied systematically: **labelled-example budget** (5 to all cells per
class), **distribution shift** (within-dataset / leave-donor-out / leave-dataset-out),
and **cell-type rarity**. The intended contribution is a **decision rule** - when the
embedding is worth its compute - not a demonstration that one method wins.

## Design commitments

These are the reasons to trust the numbers, and they are enforced in code, not prose:

| Commitment | Where enforced |
|---|---|
| HVG selection, PCA, scVI and Harmony are fit on the **training partition only**, then applied to calibration/test | `src/scfmbench/stages/s04c_classical.py`; every fit function in `src/scfmbench/models/classical.py` takes an explicit `train_mask` and `_check_train_mask` raises `LeakageError` on an all-True mask |
| No donor spans partitions under leave-donor-out; no dataset spans partitions under leave-dataset-out | asserted in `src/scfmbench/splits.py` |
| **Identical classifier head everywhere** - multinomial logistic regression, L2, strength by inner CV on training only | `src/scfmbench/stages/s05_sweep.py` |
| Test partition is never touched for any tuning decision | inner CV only, `s05_sweep.py` |
| **Datasets are the unit of replication** - seeds aggregated first, then tested across datasets | `src/scfmbench/stages/s06_stats.py` |
| Baselines tuned as carefully as the scFM pipeline - HVG+PCA is the method to beat | `configs/default.yaml` baseline grids |
| Pretraining-corpus overlap checked per dataset and reported | table `results/T1_dataset_manifest.csv` |
| Zero-shot only - no fine-tuning of any foundation model | `s04_embed_geneformer.py` (forward passes under `torch.no_grad()`) and `s04b_embed_scgpt.py` (the released `scgpt.tasks.embed_data` inference path); no optimiser is constructed in either stage |
| **Calibration partition never read by this project** - reserved for downstream work | `assert_calibration_untouched` in `src/scfmbench/splits.py` |
| Every deviation and fallback logged with reason and timestamp | [`DECISIONS.md`](DECISIONS.md) |

A negative result - scFMs not beating HVG+PCA in-distribution - is an **expected and
reportable outcome**, not a failure of the study. Positive and negative outcomes are
reported with equal prominence.

## Hypotheses

- **H1** (primary) In the full-label, in-distribution regime, scFM embeddings do **not**
  significantly outperform HVG+PCA on macro-F1.
- **H2** (primary) In the low-label (<=50 cells/class) **and** leave-dataset-out regime,
  scFM embeddings **do** outperform HVG+PCA. This is the most informative cell of the design.
- **H3** (secondary) The scFM advantage is larger for **rare** immune cell types.
- **H4** (exploratory) scFM transfer is less sensitive to low gene overlap than HVG+PCA.

## Reproduction

```bash
# 1. environment
conda env create -f environment.yml && conda activate scfm-bench
#    or, for the exact resolved build:  conda create --name scfm-bench --file conda-lock.txt

# 2. point the pipeline at a data volume (>=60 GiB free)
export SCFM_DATA_ROOT=/path/to/data/volume

# 3. one-command path
make all                      # full study
make pilot                    # 2 datasets, 1 split, 2 budgets, 2 seeds (the mandatory gate)
make figures                  # regenerate every figure from the tidy results table alone
```

Each stage is independently runnable and resumable from cached artefacts:

```
make stage.data        # acquire + hash raw data, build T1 manifest
make stage.harmonise   # QC, gene symbols, label taxonomy -> T4, gene-overlap matrix
make stage.splits      # three-way 60/20/20 splits under S1/S2/S3 + leakage assertions
make stage.embed       # GPU: scGPT + Geneformer; CPU: HVG+PCA, scVI, Harmony, celltypist
make stage.sweep       # design grid -> tidy results table T2
make stage.stats       # H1-H4 -> T3
make stage.figures     # F1-F5
```

### What is not in this repository, and why

Raw data, processed h5ad, cached embeddings and model checkpoints are **not** committed.
They are large, they are inputs and intermediates rather than deliverables, and
reproducibility here is by *documented regeneration* rather than binary storage (no
git-lfs). Every one of them has its retrieval command, source URL and SHA256 recorded in
`results/run_manifest.json`. What **is** committed: all code, configs, the environment
lockfile, tables T1-T4, figures F1-F5, the split manifests, the report, the run manifest
and the decisions log.

## Data

Primary corpus is TISCH2 (Tumour Immune Single-cell Hub 2; Han et al., *Nucleic Acids
Research* 2023, 51(D1):D1425) across NSCLC, BRCA and LIHC, with CELLxGENE Census at a
pinned LTS version as a harmonised-ontology and transfer source. Public data only - no
dataset requiring dbGaP authorisation, a DUA, or institutional approval is used.

## Compute

All results were produced on a single NVIDIA T4 GPU instance with 4 vCPU and 16 GiB RAM.
Host RAM, not VRAM, is the binding constraint at this scale; the pipeline is written for
it (backed h5ad reads, incremental PCA, chunked tokenisation, CSR sparse throughout).
Embedding extraction is a one-time cost - each cell is embedded once per model and cached.

## Report

The full written report, including the mandatory limitations section, is at
[`report/report.md`](report/report.md) (rendered: `report/report.pdf`).

## Licence

MIT - see [LICENSE](LICENSE).
