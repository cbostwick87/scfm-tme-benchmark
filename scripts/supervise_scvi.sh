#!/usr/bin/env bash
# Supervise the per-split scVI fit: detect an OOM kill, resume, and STOP after a
# bounded number of failures so the operator decides rather than the machine
# grinding for days.
#
# Why this exists. Two OOM kills have already cost roughly six hours each of
# silent stall -- not lost work (every shard is integrity-checked and resumable)
# but lost wall-clock, because nothing noticed the process was gone until a human
# asked. The per-split wrapper bounded peak RSS to 10.4-10.6 GiB against a 12.85
# GiB kill, so a third failure is not expected; if it happens anyway it will be
# visible within one poll interval instead of at the next status request.
#
# The retry budget is deliberately small. The operator's decision rule is that a
# third failure opens the question of dropping scVI from the study and declaring
# the omission (the pre-specified fallback: H2 reported against HVG+PCA alone).
# A supervisor that retried indefinitely would quietly override that decision, so
# it exits non-zero with a written verdict instead.
set -uo pipefail

# PORTABILITY: locations come from the environment, defaulting to the
# machine this was developed on. A fresh clone sets these rather than
# editing the script.
#   SCFM_DATA_ROOT     volume holding raw/ processed/ splits/ embeddings/ logs/
#   SCFM_CONDA_PREFIX  conda install providing etc/profile.d/conda.sh
#   SCFM_ENV           analysis environment to activate
: "${SCFM_DATA_ROOT:=/mnt/fm-bench}"
: "${SCFM_CONDA_PREFIX:=${SCFM_DATA_ROOT}/miniforge}"
: "${SCFM_ENV:=${SCFM_DATA_ROOT}/envs/scfm-bench}"

cd "$(dirname "$0")/.."

MAX_FAILURES=${MAX_FAILURES:-3}
POLL=${POLL:-120}
OUT=${SCFM_DATA_ROOT}/embeddings/scvi
LOG=${SCFM_DATA_ROOT}/logs/supervise_scvi.log
VERDICT=${SCFM_DATA_ROOT}/logs/scvi_verdict.txt

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

count() { ls "$OUT"/*.npz 2>/dev/null | wc -l; }

failures=0
log "supervisor start: $(count)/75 shards present, max_failures=$MAX_FAILURES"

while true; do
  n_before=$(count)
  if [ "$n_before" -ge 75 ]; then
    log "COMPLETE: 75/75 shards"
    echo "COMPLETE at $(date -u +%FT%TZ): 75/75 scVI shards" > "$VERDICT"
    exit 0
  fi

  if ! pgrep -f fit_scvi_isolated >/dev/null; then
    # Not running. Either it finished this pass, or it died.
    if [ "$failures" -ge "$MAX_FAILURES" ]; then
      log "STOPPING after $failures failures at $(count)/75 shards"
      {
        echo "SCVI FIT HALTED at $(date -u +%FT%TZ)"
        echo "shards completed: $(count)/75"
        echo "consecutive supervisor restarts: $failures (budget $MAX_FAILURES)"
        echo
        echo "The operator's decision rule applies: a repeated failure here opens"
        echo "the question of dropping scVI and declaring the omission, rather"
        echo "than continuing to spend days on the arm that is hardest to fit on"
        echo "this host. The pre-specified fallback reports H2 against HVG+PCA"
        echo "alone with the omission stated, not a narrowed hypothesis."
        echo
        echo "Recent kills, if any:"
        journalctl -u earlyoom --no-pager --since "6 hours ago" 2>/dev/null \
          | grep -E "sending SIG" | tail -5
      } > "$VERDICT"
      exit 3
    fi
    failures=$((failures+1))
    log "fit not running at $(count)/75 -- restart $failures/$MAX_FAILURES"
    nohup bash scripts/fit_scvi_isolated.sh \
      >> ${SCFM_DATA_ROOT}/logs/fit_scvi_wrapper.log 2>&1 &
    sleep 30
    continue
  fi

  sleep "$POLL"
  n_after=$(count)
  if [ "$n_after" -gt "$n_before" ]; then
    # Real progress resets the budget: the failures that matter are consecutive
    # ones, not a single transient over the whole multi-hour run.
    if [ "$failures" -gt 0 ]; then
      log "progress $n_before -> $n_after; clearing failure count"
      failures=0
    fi
  fi
done
