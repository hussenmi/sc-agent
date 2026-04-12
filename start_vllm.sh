#!/bin/bash
# Start a local LLM server using vLLM inside a Singularity container.
# The server exposes an OpenAI-compatible API that scagent connects to.
#
# Arguments:
#   MODEL  HuggingFace repo ID of the model to serve (default: Qwen2.5-Coder-32B)
#   PORT   Port the HTTP API listens on — scagent uses SCAGENT_BASE_URL=http://localhost:PORT/v1
#   GPUS   Number of GPUs (auto = minimum needed). More GPUs = larger context window,
#          because extra VRAM goes to the KV cache (conversation memory), not model weights.
#          32B on 1 GPU → 32K ctx.  32B on 2 GPUs → 128K ctx.  70B needs 4 GPUs for 128K ctx.
#
# Examples:
#   bash start_vllm.sh                                             # Qwen2.5-Coder-32B, 1 GPU, 32K ctx
#   bash start_vllm.sh Qwen/Qwen3-32B 8000 2                      # Qwen3-32B, 2 GPUs, 128K ctx
#   bash start_vllm.sh meta-llama/Llama-3.3-70B-Instruct 8000 4   # Llama 70B, 4 GPUs, 128K ctx
#
# Download a model first with:  bash download_model.sh <model_id>

MODEL=${1:-"Qwen/Qwen2.5-Coder-32B-Instruct"}
PORT=${2:-8000}
GPUS=${3:-"auto"}

HF_DIR="/data1/peerd/ibrahih3/hf"
SIF="/data1/peerd/ibrahih3/vllm-openai_gemma4.sif"
LOG="/data1/peerd/ibrahih3/tmp/vllm_$(echo $MODEL | sed 's|/|_|g').log"

mkdir -p /data1/peerd/ibrahih3/tmp

# ── Per-model settings ────────────────────────────────────────────────────────
# Add a new model by appending a line here. Fields:
#   HF repo ID | served name (used in .env) | weight size GB | KV KB/token | parser | max ctx K
# max ctx K: model's native context limit from its config.json (not always 128K)
# The parser must match how the model was trained to emit tool calls.
# Run `vllm --help` inside the container to list valid parser names.
MODEL_TABLE=(
  "Qwen/Qwen2.5-Coder-32B-Instruct          | Qwen2.5-Coder-32B-Instruct  |  64 | 256 | qwen3_xml   | 128"
  "Qwen/Qwen2.5-72B-Instruct               | Qwen2.5-72B-Instruct        | 144 | 640 | qwen3_xml   |  32"
  "Qwen/Qwen3-32B                           | Qwen3-32B                   |  64 | 256 | qwen3_xml   |  40"
  "Qwen/Qwen3-30B-A3B                       | Qwen3-30B-A3B               |  60 | 192 | qwen3_coder | 128"
  "meta-llama/Llama-3.3-70B-Instruct        | Llama-3.3-70B-Instruct      | 140 | 640 | llama3_json | 128"
  "meta-llama/Llama-3.1-70B-Instruct        | Llama-3.1-70B-Instruct      | 140 | 640 | llama3_json | 128"
  "google/gemma-4-31b-it                    | gemma-4-31b-it              |  62 | 1120 | gemma4      | 256"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-32B     |  64 | 256 | hermes      | 128"
  "THUDM/glm-4-9b-chat                      | glm-4-9b-chat               |  18 |  64 | glm45       | 128"
)
# ─────────────────────────────────────────────────────────────────────────────

SERVED_NAME="$MODEL"
MODEL_VRAM=64
KV_KB=256
PARSER="hermes"

for entry in "${MODEL_TABLE[@]}"; do
  IFS='|' read -r repo name mvram kvkb parser ctx_k <<< "$entry"
  repo=$(echo "$repo" | xargs)
  if [[ "$MODEL" == "$repo" ]]; then
    SERVED_NAME=$(echo "$name"  | xargs)
    MODEL_VRAM=$(echo "$mvram"  | xargs)
    KV_KB=$(echo "$kvkb"        | xargs)
    PARSER=$(echo "$parser"     | xargs)
    break
  fi
done

# Resolve GPU count: auto = minimum needed to fit model + reasonable KV cache
GPU_VRAM=80
UTIL=90  # percent

if [[ "$GPUS" == "auto" ]]; then
  for n in 1 2 4 8; do
    USABLE=$(( n * GPU_VRAM * UTIL / 100 ))
    KV_BUDGET=$(( USABLE - MODEL_VRAM ))
    if [[ $KV_BUDGET -ge 8 ]]; then   # need at least 8GB for KV cache
      TP=$n
      break
    fi
  done
else
  TP=$GPUS
fi

# Context window: KV budget → tokens, capped at model native max (128K)
USABLE=$(( TP * GPU_VRAM * UTIL / 100 ))
KV_BUDGET=$(( USABLE - MODEL_VRAM ))
MAX_CTX=$(( KV_BUDGET * 1024 * 1024 / KV_KB ))
NATIVE_CAP=$(( ${ctx_k:-128} * 1024 ))
CTX=$(( MAX_CTX < NATIVE_CAP ? MAX_CTX : NATIVE_CAP ))
CTX_K=$(( CTX / 1024 ))

echo "Model:    $MODEL"
echo "Served:   $SERVED_NAME"
echo "GPUs:     $TP  (tensor-parallel-size)"
echo "Context:  ${CTX_K}K tokens"
echo "Parser:   $PARSER"
echo "Port:     $PORT"
echo "Log:      $LOG"
echo ""

# Sanity check
if [[ $KV_BUDGET -lt 4 ]]; then
  echo "ERROR: Not enough GPUs to run this model ($MODEL needs ~${MODEL_VRAM}GB, have $((TP * GPU_VRAM))GB total)"
  echo "Try:  bash start_vllm.sh $MODEL $PORT $((TP + 1))"
  exit 1
fi

# Check model is downloaded
MODEL_CACHE="$HF_DIR/hub/models--$(echo $MODEL | sed 's|/|--|g')"
if [[ ! -d "$MODEL_CACHE" ]]; then
  echo "ERROR: Model not found at $MODEL_CACHE"
  echo "Download it first with:  bash download_model.sh $MODEL"
  exit 1
fi

SINGULARITYENV_HF_HUB_CACHE=/hf_cache/hub \
SINGULARITYENV_HF_HUB_OFFLINE=1 \
SINGULARITYENV_TRANSFORMERS_OFFLINE=1 \
SINGULARITYENV_PYTHONNOUSERSITE=1 \
SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_vllm \
SINGULARITYENV_VLLM_CACHE_ROOT=/tmp/vllm_cache \
singularity exec --nv \
  --bind "$HF_DIR":/hf_cache \
  "$SIF" \
  vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$CTX" \
    --enable-auto-tool-choice \
    --tool-call-parser "$PARSER" \
    --port "$PORT" \
    --host 0.0.0.0 \
  >> "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"
echo ""

for i in $(seq 1 60); do
  sleep 5
  if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "Server ready at http://localhost:$PORT/v1"
    echo "  SCAGENT_MODEL=$SERVED_NAME"
    echo "  SCAGENT_BASE_URL=http://localhost:$PORT/v1"
    exit 0
  fi
done
echo "Timed out — check log: $LOG"
exit 1
