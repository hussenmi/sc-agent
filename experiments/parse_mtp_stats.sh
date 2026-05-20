#!/bin/bash
#SBATCH --job-name=parse_mtp_stats
#SBATCH --output=/data1/peerd/ibrahih3/cs_agent/logs/parse_mtp_stats_%N_%j.log
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#
# Parses SpecDecoding metrics from vLLM server logs and writes a structured CSV.
# vLLM logs acceptance rate, mean acceptance length, and throughput every 10
# seconds during active inference — this extracts all of it.
#
# Run this after any benchmark that generates load against a server with MTP
# enabled (bench_llm.sh, bench_waterfall.sh, bench_prefix_cache.sh all qualify).
#
# Usage:
#   bash parse_mtp_stats.sh              # parse all vLLM logs in logs/
#   bash parse_mtp_stats.sh logs/vllm_iscb007*.log   # specific files
#
# To scope to a specific benchmark window (avoid mixing runs):
#   AFTER="05-07 15:45" BEFORE="05-07 15:55" bash parse_mtp_stats.sh

# ── SETTINGS ─────────────────────────────────────────────────────────────────

# Log files to parse. Glob patterns are expanded by the script.
# Default: all vLLM logs in the logs/ directory.
LOG_PATTERN="${*:-logs/vllm_*.log}"

OUT="results/mtp_stats.csv"

# Optional timestamp filters to scope to a specific benchmark run.
# Format: "MM-DD HH:MM"  e.g. "05-07 15:45"
# Leave empty to parse all entries in the matched logs.
AFTER="${AFTER:-}"
BEFORE="${BEFORE:-}"

# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT="/data1/peerd/ibrahih3/cs_agent"
PARSE_PY="$PROJECT_ROOT/experiments/parse_mtp_stats.py"
PYTHON=${PYTHON:-python3}

if [[ ! -f "$PARSE_PY" ]]; then
  echo "ERROR: parse_mtp_stats.py not found at $PARSE_PY"
  exit 1
fi

mkdir -p "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs"

# Build args
ARGS=()
# Expand the glob pattern — the Python script handles this too, but being
# explicit here ensures we print what's being parsed before calling it.
for f in $LOG_PATTERN; do
  [[ -f "$f" ]] && ARGS+=("$f")
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "No log files matched: $LOG_PATTERN"
  exit 1
fi

echo "Parsing ${#ARGS[@]} log file(s):"
for f in "${ARGS[@]}"; do echo "  $f"; done
echo ""

ARGS+=(--out "$PROJECT_ROOT/$OUT")
[[ -n "$AFTER"  ]] && ARGS+=(--after  "$AFTER")
[[ -n "$BEFORE" ]] && ARGS+=(--before "$BEFORE")

run_parse() {
  cd "$PROJECT_ROOT"
  "$PYTHON" "$PARSE_PY" "${ARGS[@]}"
}

TS=$(date +%Y%m%d_%H%M%S)
LOG="$PROJECT_ROOT/logs/parse_mtp_stats_$(hostname -s)_${TS}.log"
echo "Logging to: $LOG"
echo ""
run_parse 2>&1 | tee "$LOG"
