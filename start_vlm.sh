#!/bin/bash
# Serve a vision-language model as a DESCRIBER for scagent's vision sidecar, via
# vLLM in Singularity. OpenAI-compatible /v1, so scagent's SCAGENT_VISION_BASE_URL
# points straight at it.
#
# This is intentionally separate from start_vllm.sh: the sidecar only ever sends
# plain chat-with-image_url requests (no tools, no agentic reasoning, no
# speculative decoding), so none of that machinery applies. Keeping it minimal
# avoids loading tool/reasoning parsers a describer never uses.
#
# Default model: NVIDIA Nemotron Nano 12B v2 VL (BF16, ~24GB) — a chart/document
# specialist, fits one H200. Needs --trust-remote-code (custom NemotronH code) and
# vLLM >= ~0.12 (we use the 0.22 SIF).
#
# Arguments:
#   MODEL  HF repo id   (default: nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)
#   PORT   API port     (default: 8003)
#
# Environment overrides:
#   GPU_IDS=0         GPU index/indices to use (TP size = count). Default "0".
#   CTX=131072        max-model-len.
#   MAX_IMAGES=8      images allowed per prompt (sidecar chunks to 3).
#   SERVED_NAME=Nemotron-Nano-VL   name clients use (set SCAGENT_VISION_MODEL to match).
#   VLLM_SIF=...      container path (default the 0.22 SIF).
#
# Example (run on the node that will host it, e.g. iscn008):
#   GPU_IDS=0 bash start_vlm.sh

MODEL=${1:-"nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"}
PORT=${2:-8003}
GPU_IDS=${GPU_IDS:-"0"}
CTX=${CTX:-131072}
MAX_IMAGES=${MAX_IMAGES:-8}
SERVED_NAME=${SERVED_NAME:-"Nemotron-Nano-VL"}

HF_DIR="/data1/peerd/ibrahih3/hf"
VLLM_SIF=${VLLM_SIF:-"/data1/peerd/ibrahih3/vllm-openai_v0.22.0.sif"}
LOG_DIR="/data1/peerd/ibrahih3/cs_agent/logs"
LOG="$LOG_DIR/vlm_$(hostname -s)_$(echo "$MODEL" | sed 's|/|_|g').log"
mkdir -p "$LOG_DIR"

TP=$(echo "$GPU_IDS" | tr ',' '\n' | grep -c .)

[[ -f "$VLLM_SIF" ]] || { echo "ERROR: vLLM SIF not found: $VLLM_SIF"; exit 1; }
MODEL_CACHE="$HF_DIR/hub/models--$(echo "$MODEL" | sed 's|/|--|g')"
if [[ ! -d "$MODEL_CACHE" ]]; then
  echo "ERROR: weights not found at $MODEL_CACHE"
  echo "       Download: bash download_model.sh $MODEL"
  exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
[[ -z "$GPU_NAME" ]] && { echo "ERROR: no GPU detected"; exit 1; }

if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$PORT "; then
  echo "ERROR: port $PORT already in use (ss -ltnp | grep :$PORT)"; exit 1
fi

echo "GPU:      $GPU_NAME"
echo "Model:    $MODEL"
echo "Served:   $SERVED_NAME  (vision describer)"
echo "GPUs:     $GPU_IDS  (tensor-parallel-size=$TP)"
echo "Context:  $CTX   Images/prompt: $MAX_IMAGES"
echo "Port:     $PORT"
echo "Log:      $LOG"
echo ""

MODEL_TAG="$(echo "$MODEL" | sed 's|/|_|g')_vlm"
mkdir -p "$HF_DIR/vllm_compile_cache/$MODEL_TAG"

SINGULARITYENV_HF_HUB_CACHE=/hf_cache/hub \
SINGULARITYENV_HF_HUB_OFFLINE=1 \
SINGULARITYENV_TRANSFORMERS_OFFLINE=1 \
SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_IDS" \
SINGULARITYENV_PYTHONNOUSERSITE=1 \
SINGULARITYENV_HF_HOME=/hf_cache \
SINGULARITYENV_HF_MODULES_CACHE=/hf_cache/modules \
SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="/hf_cache/vllm_compile_cache/$MODEL_TAG/torchinductor" \
SINGULARITYENV_VLLM_CACHE_ROOT="/hf_cache/vllm_compile_cache/$MODEL_TAG/vllm" \
SINGULARITYENV_VLLM_ENGINE_READY_TIMEOUT_S=2400 \
SINGULARITYENV_TMPDIR=/tmp \
singularity exec --nv \
  --bind "$HF_DIR":/hf_cache \
  "$VLLM_SIF" \
  vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$CTX" \
    --limit-mm-per-prompt "{\"image\": $MAX_IMAGES}" \
    --port "$PORT" \
    --host 0.0.0.0 \
  >> "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"
echo ""

for i in $(seq 1 240); do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "vLLM exited during startup — last lines of log:"; tail -30 "$LOG"
    echo ""; echo "Full log: $LOG"; exit 1
  fi
  if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "Vision describer ready at http://localhost:$PORT/v1"
    echo "  SCAGENT_VISION_MODEL=$SERVED_NAME"
    echo "  SCAGENT_VISION_BASE_URL=http://$(hostname -s):$PORT/v1"
    echo "  SCAGENT_VISION_API_KEY=dummy"
    exit 0
  fi
done
echo "Timed out — check log: $LOG"; exit 1
