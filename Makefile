# ============================================================
# One-command reproduction paths. Every figure and table is regenerable
# from the persisted tidy results table by a named target (P5) -- no manual
# steps, no notebook state.
# ============================================================
CONFIG ?= configs/default.yaml
PY     ?= python -m

.PHONY: all pilot figures tables test lint manifest clean help \
        stage.data stage.harmonise stage.splits stage.embed stage.sweep stage.stats stage.figures

help:
	@grep -E '^[a-zA-Z._-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-18s\033[0m %s\n",$$1,$$2}'

all: stage.data stage.harmonise stage.splits stage.embed stage.sweep stage.stats stage.figures  ## full study

pilot:  ## Phase 0 gate: 2 datasets, 1 split, 2 budgets, 2 seeds
	$(MAKE) all CONFIG=configs/pilot.yaml

stage.data:       ## acquire + hash raw data, resolve duplicate accessions, build T1
	$(PY) scfmbench.stages.s01_data --config $(CONFIG)
stage.harmonise:  ## QC, gene symbols, label taxonomy (T4), gene-overlap matrix
	$(PY) scfmbench.stages.s02_harmonise --config $(CONFIG)
stage.splits:     ## three-way 60/20/20 splits under S1/S2/S3 + leakage assertions
	$(PY) scfmbench.stages.s03_splits --config $(CONFIG)
stage.embed:      ## GPU: scGPT + Geneformer; CPU: HVG+PCA, scVI, Harmony, celltypist
	$(PY) scfmbench.stages.s04_embed --config $(CONFIG)
stage.sweep:      ## design grid -> tidy results table T2
	$(PY) scfmbench.stages.s05_sweep --config $(CONFIG)
stage.stats:      ## H1-H4 -> T3
	$(PY) scfmbench.stages.s06_stats --config $(CONFIG)
stage.figures:    ## F1-F5 from T2 alone
	$(PY) scfmbench.stages.s07_figures --config $(CONFIG)

figures: stage.figures  ## alias
tables: stage.stats     ## alias

manifest:  ## write the run manifest (versions, GPU witness, git commit, seeds)
	$(PY) scfmbench.cli manifest --config $(CONFIG)

test:  ## leakage-control and unit tests
	pytest -q tests/

lint:
	ruff check src/ tests/ || true

clean:  ## remove derived results/figures ONLY -- never raw data or embeddings
	rm -rf results/shards results/_scratch figures/*.png figures/*.pdf figures/*.svg
