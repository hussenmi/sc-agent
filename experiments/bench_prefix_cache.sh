#!/bin/bash
#SBATCH --job-name=bench_prefix_cache
#SBATCH --output=/data1/peerd/ibrahih3/cs_agent/logs/bench_prefix_cache_%j.out
#SBATCH --error=/data1/peerd/ibrahih3/cs_agent/logs/bench_prefix_cache_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#
# Prefix caching benchmark — simulates an agent loop to show TTFT improvement
# from prefix caching as session context grows across iterations.
#
# Run interactively:  bash bench_prefix_cache.sh
#   → output goes to terminal AND logs/bench_prefix_cache_<host>_<ts>.log
# Run via sbatch:     sbatch bench_prefix_cache.sh
#   → output goes to logs/bench_prefix_cache_<jobid>.out/err

# ── SETTINGS ─────────────────────────────────────────────────────────────────

ENDPOINTS=(
  "http://iscb007:8000/v1,Qwen3.6-27B,A100"
  "http://iscg002:8000/v1,Qwen3.6-27B,H100"
)

ITERATIONS=20        # agent loop iterations per condition (cached + uncached each)
PREFIX_TOKENS=20000  # target prefix size in tokens (~scagent system prompt + tool schemas)
MAX_TOKENS=30        # keep short — we're measuring TTFT, not decode speed
PAUSE=0.3            # seconds between requests
MODE="both"          # "cached", "uncached", or "both"
OUT="results/prefix_cache.csv"

# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT="/data1/peerd/ibrahih3/cs_agent"
PYTHON=${PYTHON:-python3}

mkdir -p "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs"

_run() {
  echo "Started on $(hostname) at $(date)"
  echo ""
  for ep in "${ENDPOINTS[@]}"; do
    IFS=',' read -r url model hardware <<< "$ep"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Hardware: $hardware  |  URL: $url  |  Model: $model"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "$PYTHON" "$PROJECT_ROOT/experiments/bench_prefix_cache.py" \
      --url           "$url" \
      --model         "$model" \
      --hardware      "$hardware" \
      --out           "$PROJECT_ROOT/$OUT" \
      --mode          "$MODE" \
      --iterations    "$ITERATIONS" \
      --prefix-tokens "$PREFIX_TOKENS" \
      --max-tokens    "$MAX_TOKENS" \
      --pause         "$PAUSE"
    echo ""
  done
  echo "Finished at $(date)"
}

TS=$(date +%Y%m%d_%H%M%S)
LOG="$PROJECT_ROOT/logs/bench_prefix_cache_$(hostname -s)_${TS}.log"
echo "Logging to: $LOG"
_run 2>&1 | tee "$LOG"
