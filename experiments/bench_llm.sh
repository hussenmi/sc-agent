#!/bin/bash
#SBATCH --job-name=bench_llm
#SBATCH --output=/data1/peerd/ibrahih3/cs_agent/logs/bench_llm_%N_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#
# Wrapper around bench_llm.py for A/B benchmarking LLM endpoints.
# Edit the BENCHMARK SETTINGS block below, then run either of:
#   bash bench_llm.sh        # interactive — writes logs/bench_llm_<host>_<ts>.log
#   sbatch bench_llm.sh      # batch       — writes logs/bench_llm_<node>_<jobid>.log
#
# Only one log file per run either way. (No --error directive: SLURM merges
# stderr into the --output file by default.)
#
# The script must be runnable from a node that can reach all endpoints.
# (Same datacenter is fine — cross-node latency in /data1/peerd is sub-ms.)

# ── BENCHMARK SETTINGS ───────────────────────────────────────────────────────
# Each entry is "URL,MODEL,LABEL". URL is the OpenAI-compatible /v1 endpoint,
# MODEL is the served name (matches SCAGENT_MODEL printed by start_vllm.sh),
# LABEL is whatever you want to see in the report.
# Use absolute hostnames (not "localhost") so this works whether run interactively
# or via sbatch — sbatch lands the job on an arbitrary compute node.
ENDPOINTS=(
  "http://iscb007:8000/v1,Qwen3.6-27B,iscb007-A100-BF16"
  "http://iscg002:8000/v1,Qwen3.6-27B,iscg002-H100-FP8"
)

REPEATS=4          # times to run each prompt (medians/p90 across all)
MAX_TOKENS=300     # generation length cap per request
NO_WARMUP=0        # set to 1 to skip the warmup pass (faster but first request is cold)
# ─────────────────────────────────────────────────────────────────────────────

# Hardcoded project root because $0 doesn't point here under sbatch (SLURM copies
# the submitted script to /var/spool/slurmd/<job>/).
PROJECT_ROOT="/data1/peerd/ibrahih3/cs_agent"
BENCH_PY="$PROJECT_ROOT/experiments/bench_llm.py"
LOG_DIR="$PROJECT_ROOT/logs"
PYTHON=${PYTHON:-python3}

if [[ ! -f "$BENCH_PY" ]]; then
  echo "ERROR: bench_llm.py not found at $BENCH_PY"
  exit 1
fi

mkdir -p "$LOG_DIR"

ARGS=()
for ep in "${ENDPOINTS[@]}"; do
  ARGS+=(--pair "$ep")
done
ARGS+=(--repeats "$REPEATS")
ARGS+=(--max-tokens "$MAX_TOKENS")
[[ "$NO_WARMUP" == "1" ]] && ARGS+=(--no-warmup)

echo "Running: $PYTHON $BENCH_PY ${ARGS[*]}"

# When launched via sbatch, SLURM redirects stdout/stderr to the --output file
# above. When launched interactively, no redirection happens — so tee here for
# interactive runs only, to avoid duplicating the log file under sbatch.
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/bench_llm_$(hostname -s)_${TS}.log"
echo "Logging to: $LOG_FILE"
echo ""
"$PYTHON" "$BENCH_PY" "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
