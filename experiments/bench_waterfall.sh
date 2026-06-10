#!/bin/bash
#SBATCH --job-name=bench_waterfall
#SBATCH --output=/data1/peerd/ibrahih3/cs_agent/logs/bench_waterfall_%N_%j.log
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#
# Run bench_waterfall.py against one server config and append results to the
# shared waterfall CSV. Restart the server between runs with different flags
# (see WATERFALL SEQUENCE below), then re-run with a new LABEL.
#
# Usage:
#   LABEL="A100 + MTP (full)" bash bench_waterfall.sh http://iscb016:8000/v1 Qwen3.6-27B A100
#   LABEL="H100 baseline"     bash bench_waterfall.sh http://iscd003:8000/v1 Qwen3.6-27B H100
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WATERFALL SEQUENCE
#
# For each config, restart the server with the appropriate env vars, wait for
# "Server ready" in the log, then run this script. All 10 runs accumulate into
# one results/waterfall.csv for the plotting script.
#
# start_vllm.sh env var overrides used here:
#   ENFORCE_EAGER=1   — disables CUDA graphs (adds --enforce-eager)
#   KV_DTYPE=bf16     — skips FP8 KV cache (uses BF16 KV)
#   NO_PREFIX_CACHE=1 — disables prefix caching
#   SPEC=0            — disables speculative decoding (MTP)
#
# ── A100 sequence (iscb016, Qwen/Qwen3.6-27B BF16) ──────────────────────
#
#   Config 1 — True baseline (nothing enabled):
#     ssh iscb016 'cd /data1/peerd/ibrahih3/cs_agent && \
#       ENFORCE_EAGER=1 KV_DTYPE=bf16 NO_PREFIX_CACHE=1 SPEC=0 \
#       bash start_vllm.sh Qwen/Qwen3.6-27B 8000 2'
#     LABEL="A100 baseline" bash bench_waterfall.sh http://iscb016:8000/v1 Qwen3.6-27B A100
#
#   Config 2 — + CUDA graphs only:
#     ssh iscb016 'cd /data1/peerd/ibrahih3/cs_agent && \
#       KV_DTYPE=bf16 NO_PREFIX_CACHE=1 SPEC=0 \
#       bash start_vllm.sh Qwen/Qwen3.6-27B 8000 2'
#     LABEL="A100 + CUDA graphs" bash bench_waterfall.sh http://iscb016:8000/v1 Qwen3.6-27B A100
#
#   Config 3 — + CUDA graphs + FP8 KV cache:
#     ssh iscb016 'cd /data1/peerd/ibrahih3/cs_agent && \
#       NO_PREFIX_CACHE=1 SPEC=0 \
#       bash start_vllm.sh Qwen/Qwen3.6-27B 8000 2'
#     LABEL="A100 + FP8 KV" bash bench_waterfall.sh http://iscb016:8000/v1 Qwen3.6-27B A100
#
#   Config 4 — + CUDA graphs + FP8 KV + prefix caching:
#     ssh iscb016 'cd /data1/peerd/ibrahih3/cs_agent && \
#       SPEC=0 bash start_vllm.sh Qwen/Qwen3.6-27B 8000 2'
#     LABEL="A100 + prefix cache" bash bench_waterfall.sh http://iscb016:8000/v1 Qwen3.6-27B A100
#
#   Config 5 — Full stack (+ MTP speculative decoding):
#     ssh iscb016 'cd /data1/peerd/ibrahih3/cs_agent && \
#       bash start_vllm.sh Qwen/Qwen3.6-27B 8000 2'
#     LABEL="A100 + MTP (full)" bash bench_waterfall.sh http://iscb016:8000/v1 Qwen3.6-27B A100
#
# ── H100 sequence (iscd003, Qwen/Qwen3.6-27B-FP8) ───────────────────────
#
#   Config 1 — True baseline:
#     ssh iscd003 'cd /data1/peerd/ibrahih3/cs_agent && \
#       ENFORCE_EAGER=1 KV_DTYPE=bf16 NO_PREFIX_CACHE=1 SPEC=0 \
#       bash start_vllm.sh Qwen/Qwen3.6-27B-FP8 8000 2'
#     LABEL="H100 baseline" bash bench_waterfall.sh http://iscd003:8000/v1 Qwen3.6-27B H100
#
#   Config 2 — + CUDA graphs:
#     ssh iscd003 'cd /data1/peerd/ibrahih3/cs_agent && \
#       KV_DTYPE=bf16 NO_PREFIX_CACHE=1 SPEC=0 \
#       bash start_vllm.sh Qwen/Qwen3.6-27B-FP8 8000 2'
#     LABEL="H100 + CUDA graphs" bash bench_waterfall.sh http://iscd003:8000/v1 Qwen3.6-27B H100
#
#   Config 3 — + CUDA graphs + FP8 KV cache:
#     ssh iscd003 'cd /data1/peerd/ibrahih3/cs_agent && \
#       NO_PREFIX_CACHE=1 SPEC=0 \
#       bash start_vllm.sh Qwen/Qwen3.6-27B-FP8 8000 2'
#     LABEL="H100 + FP8 KV" bash bench_waterfall.sh http://iscd003:8000/v1 Qwen3.6-27B H100
#
#   Config 4 — + CUDA graphs + FP8 KV + prefix caching:
#     ssh iscd003 'cd /data1/peerd/ibrahih3/cs_agent && \
#       SPEC=0 bash start_vllm.sh Qwen/Qwen3.6-27B-FP8 8000 2'
#     LABEL="H100 + prefix cache" bash bench_waterfall.sh http://iscd003:8000/v1 Qwen3.6-27B H100
#
#   Config 5 — Full stack (+ MTP):
#     ssh iscd003 'cd /data1/peerd/ibrahih3/cs_agent && \
#       bash start_vllm.sh Qwen/Qwen3.6-27B-FP8 8000 2'
#     LABEL="H100 + MTP (full)" bash bench_waterfall.sh http://iscd003:8000/v1 Qwen3.6-27B H100
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROJECT_ROOT="/data1/peerd/ibrahih3/cs_agent"
PYTHON=${PYTHON:-python3}

URL="${1:?Usage: LABEL=... bash bench_waterfall.sh <url> <model> <A100|H100>}"
MODEL="${2:?provide model name as second arg}"
HARDWARE="${3:?provide hardware as third arg (A100 or H100)}"
LABEL="${LABEL:?Set LABEL env var, e.g.: LABEL=\"A100 + MTP (full)\" bash bench_waterfall.sh ...}"
REPEATS="${REPEATS:-4}"
MAX_TOKENS="${MAX_TOKENS:-400}"
OUT="${OUT:-$PROJECT_ROOT/results/waterfall.csv}"

mkdir -p "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs"

echo "Config:   $LABEL"
echo "URL:      $URL  |  Model: $MODEL  |  Hardware: $HARDWARE"
echo "Repeats:  $REPEATS  |  Max tokens: $MAX_TOKENS"
echo "Output:   $OUT"
echo ""

cd "$PROJECT_ROOT"

_run() {
  echo "Started on $(hostname) at $(date)"
  echo "Config: $LABEL  |  Hardware: $HARDWARE  |  URL: $URL"
  echo ""
  cd "$PROJECT_ROOT"
  "$PYTHON" "$PROJECT_ROOT/experiments/bench_waterfall.py" \
      --url        "$URL" \
      --model      "$MODEL" \
      --label      "$LABEL" \
      --hardware   "$HARDWARE" \
      --out        "$OUT" \
      --repeats    "$REPEATS" \
      --max-tokens "$MAX_TOKENS"
  echo ""
  echo "Finished at $(date)"
}

TS=$(date +%Y%m%d_%H%M%S)
LOG="$PROJECT_ROOT/logs/bench_waterfall_$(hostname -s)_${TS}.log"
echo "Logging to: $LOG"
_run 2>&1 | tee "$LOG"
