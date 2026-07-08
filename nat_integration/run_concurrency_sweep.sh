#!/bin/bash
# NAT concurrency sweep: run the SAME workload at increasing max_concurrency to
# measure how serving latency scales with concurrent agent load. Each level goes
# to its own ARCHIVED dir (no overwrites). Thread-limited so concurrent scagent
# processes don't oversubscribe a small eval node (the serving GPU is what we
# stress; the eval node is just the harness).
#
# Usage: bash run_concurrency_sweep.sh "1 2 4" 4 http://iscn008:8000/v1
#   arg1 = space-separated concurrency levels   (default "1 2 4")
#   arg2 = reps (items per level)               (default 4)
#   arg3 = backend base_url                     (default http://iscn008:8000/v1)
set -u
LEVELS=${1:-"1 2 4"}
REPS=${2:-4}
BASE_URL=${3:-"http://iscn008:8000/v1"}

REPO=/data1/peerd/ibrahih3/cs_agent
NAT=/usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/nat
CFG=$REPO/nat_integration/configs/sweep.yml
STAMP=$(date +%Y%m%d_%H%M%S)
SWEEP=$REPO/nat_sweep_$STAMP
mkdir -p "$SWEEP"

# Keep each scagent process single-threaded so concurrency N ~ N cores (no thrash).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

echo "sweep dir: $SWEEP | levels: [$LEVELS] | reps: $REPS | backend: $BASE_URL"
for K in $LEVELS; do
  OUT="$SWEEP/conc_$K"
  echo "[$(date +%H:%M:%S)] === concurrency=$K -> $OUT ==="
  rm -f "$REPO"/nat_runs/*.steps.jsonl
  "$NAT" eval --config_file "$CFG" \
    --override workflow.base_url "$BASE_URL" \
    --override eval.general.max_concurrency "$K" \
    --override eval.general.output_dir "$OUT" \
    --reps "$REPS" > "$SWEEP/conc_$K.log" 2>&1
  echo "[$(date +%H:%M:%S)]     done (rc=$?)"
done
echo "SWEEP COMPLETE: $SWEEP"
