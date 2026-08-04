#!/usr/bin/env bash
# Fit scVI ONE SPLIT PER PROCESS.
#
# Why a wrapper instead of a fix inside the loop. Measured on this host:
#   * stage baseline is 8.11 GiB before any split work -- the corpus CSR is
#     2.49 GiB at 334M nonzeros and the stage holds TWO copies (the raw
#     log1p(CP10K) matrix that scVI's pseudo-counts derive from, and the
#     renormalised matrix every other arm uses)
#   * the OOM kill came at 12,849 MiB, so per-split allocations add ~4.7 GiB
#     against roughly 7 GiB of headroom
#   * available memory fell 40% -> 23% -> 8.5% over about two splits, i.e.
#     something accumulates per split rather than the baseline simply being high
#
# An in-loop `del` plus gc.collect() was already tried and did not hold. Rather
# than keep hunting an allocation inside lightning/scvi-tools that does not
# belong to my code, each split runs in its own process: the OS reclaims
# everything at exit, so peak RSS is bounded by one split's requirement and the
# accumulation cannot occur by construction. The cost is re-reading the corpus
# per split (24 s measured) against ~650 s of training, i.e. under 4% overhead
# for a hard memory bound.
#
# This is a documented fallback under the brief's dependency time-box, not a
# silent workaround: the loop is unchanged and still correct, it simply is not
# the entry point for scVI.
set -uo pipefail
cd "$(dirname "$0")/.."
source /mnt/fm-bench/miniforge/etc/profile.d/conda.sh
conda activate /mnt/fm-bench/envs/scfm-bench
export SCFM_DATA_ROOT=/mnt/fm-bench TMPDIR=/mnt/fm-bench/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=src

LOG=/mnt/fm-bench/logs/fit_scvi_isolated.log
OUT=/mnt/fm-bench/embeddings/scvi
SPLITS=/mnt/fm-bench/splits

echo "=== per-split scVI fit $(date -u +%FT%TZ) ===" | tee -a "$LOG"
done_n=0; ran=0; failed=0
for f in "$SPLITS"/*.parquet; do
  b=$(basename "$f" .parquet)
  [ "$b" = "cell_index" ] && continue
  # Resume gate: the shard counts as done only if it OPENS and carries the full
  # cell index. Existence and plausible size are not enough -- that assumption
  # has already cost this project twice.
  if python - "$OUT/$b.npz" <<'PY' 2>/dev/null
import sys, numpy as np
with np.load(sys.argv[1], allow_pickle=True) as z:
    assert len(z["cell_id"]) == 229801 and np.isfinite(z["emb"]).all()
PY
  then done_n=$((done_n+1)); continue; fi

  echo "--- $b ---" | tee -a "$LOG"
  /usr/bin/time -f "  peak RSS %M KiB, wall %e s" \
    python -m scfmbench.stages.s04c_classical \
      --config configs/default.yaml --methods scvi --splits "$b.parquet" \
      >> "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    failed=$((failed+1))
    echo "  FAILED rc=$rc (137/143 = OOM-killed)" | tee -a "$LOG"
  else
    ran=$((ran+1))
  fi
  grep -a "peak RSS" "$LOG" | tail -1
done
echo "=== done: $ran fitted, $done_n already present, $failed failed ===" | tee -a "$LOG"
echo "total shards: $(ls "$OUT"/*.npz 2>/dev/null | wc -l)/75" | tee -a "$LOG"
