# QC notes — why the thresholds removed zero cells

The configured QC thresholds (`min_genes_per_cell: 200`, `max_pct_mito: 20`,
`min_cells_per_gene: 3`) removed **0 of 229,801 cells**. That is a real result, not a
silent failure, and it is stated here because "we applied QC" would otherwise imply a
filtering step that did no work.

## Evidence that the filters are inert rather than broken

Measured directly on the **raw** `BRCA_GSE161529` matrix, before any processing in this
pipeline:

| quantity | value |
|---|---|
| minimum genes detected per cell | **exactly 500** |
| 1st percentile genes/cell | 520 |
| median genes/cell | 1,460 |
| cells below 200 genes | 0 |
| cells below 500 genes | 0 |
| MT- genes present in the feature set | 13 |
| pct_mito median / p90 / p99 / max (n=3,000 sample) | 1.65 / 2.69 / 4.27 / **10.78** |
| cells above 20% mito | 0 |

A hard floor at exactly 500 genes per cell is the signature of an upstream filter:
TISCH2 distributes **already quality-controlled** matrices. The mitochondrial
distribution tops out at 10.8%, roughly half the 20% threshold.

So the thresholds are retained deliberately: they are a **guard**, asserting the
assumption that inputs are pre-filtered rather than assuming it. If a future dataset
enters the corpus without upstream QC, they will fire.

## Where the mitochondrial filter cannot apply at all

Three datasets contain **no detectable `MT-` genes**, so `max_pct_mito` is structurally
inapplicable and every cell trivially passes:

- `CRC_GSE108989`
- `HNSC_GSE103322`
- `LIHC_GSE98638`

All three are Smart-seq2-era plate-based datasets whose distributed feature sets exclude
mitochondrial genes. This is recorded per dataset as `mito_filter_applied` in
`qc_summary.csv` so the report can state where the filter applied rather than implying
uniform application across the corpus.

## What actually reduces the corpus

Cell counts are governed by two other steps, both deliberate and both reported:

1. **Label harmonisation** drops 57,926 cells (6.8%) whose author labels are ambiguous
   (e.g. `Immune cells`, `Myeloid`), technical (`Doublet`), proliferating compartments
   that span identities, or non-immune tissue parenchyma. Every drop has a stated reason
   in `T4_label_harmonisation.csv`.
2. **Stratified capping** to 30,000 cells per dataset trims abundant classes only. Rare
   classes are kept whole — see `qc_summary.csv` (`cells_selected_pre_qc`) and the
   water-filling implementation in `s02_harmonise.stratified_cap`.
