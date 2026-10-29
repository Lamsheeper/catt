#!/usr/bin/env bash
# Score a slice of the attribution grid sequentially on one GPU.
#
#   TASK=fb FACTS=50 DOCS="1 5 10 30 50" SEEDS="1001 1002 1003" bash scripts/run_grid.sh
#   JOBS="ekfac layerwise-project" bash scripts/run_grid.sh
#
# Defaults are the paper's slice for one task. The training-dynamics jobs (tracin,
# layerwise-tracin, source) run only on the seeds that have published trajectories
# (1001, 1002). A (cell, job) whose output directory already holds scores is skipped.
# Outputs: $CATT_WORK/attribution/{task}/{run}/{job}/.
#
# The paper's |F| = 50 sweep took ~1,000 GPU-hours over the three tasks; see
# README.md for per-job costs before launching a full grid.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=${TASK:-fb}
FACTS=${FACTS:-50}
case "$TASK" in
  rr) DEFAULT_DOCS="3 5 10 30 50" ;;
  *)  DEFAULT_DOCS="1 5 10 30 50" ;;
esac
DOCS=${DOCS:-$DEFAULT_DOCS}
SEEDS=${SEEDS:-"1001 1002 1003"}
JOBS=${JOBS:-"ekfac layerwise-project trak tracin layerwise-tracin source bm25 repsim"}
WORK=${CATT_WORK:-work}
CATT=${CATT:-"uv run catt"}

$CATT data --task "$TASK" --facts "$FACTS" --docs $DOCS

run_name() {
  if [[ "$TASK" == rr ]]; then echo "route_ab_c${FACTS}_${1}d_sd${2}"; else echo "f${FACTS}_${1}d_sd${2}"; fi
}

for docs in $DOCS; do
  for seed in $SEEDS; do
    run=$(run_name "$docs" "$seed")
    for job in $JOBS; do
      case "$job" in
        tracin|layerwise-tracin|source)
          [[ "$seed" == 1001 || "$seed" == 1002 ]] || continue ;;
      esac
      out="$WORK/attribution/$TASK/$run/$job"
      if compgen -G "$out/scores.*.npz" > /dev/null; then
        echo "skip $run/$job (already scored)"
        continue
      fi
      echo "=== $run/$job"
      $CATT attribute --task "$TASK" --facts "$FACTS" --docs "$docs" --seed "$seed" \
        --job "$job" --out "$out"
    done
  done
done
