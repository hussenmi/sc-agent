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
SIF=${VLLM_SIF:-"/data1/peerd/ibrahih3/vllm-openai_gemma4.sif"}
LOG="/data1/peerd/ibrahih3/tmp/vllm_$(echo $MODEL | sed 's|/|_|g').log"

mkdir -p /data1/peerd/ibrahih3/tmp

# ── Per-model settings ────────────────────────────────────────────────────────
# Add a new model by appending a line here. Fields:
#   HF repo ID | served name (used in .env) | weight size GB | KV KB/token | parser | max ctx K | quant | reasoning_parser
# weight size GB: actual loaded size (use FP8 weight size for pre-quantized FP8 checkpoints)
# quant: leave blank for BF16; set to "fp8" for pre-quantized FP8 checkpoints
#   - blank: on H100, online FP8 quantization is applied (--quantization fp8); weight size
#            must reflect BF16 size since BF16 is loaded first before quantizing
#   - fp8:   vLLM auto-detects quantization; weight size reflects FP8 size (~half of BF16);
#            larger context window is safe since weights load directly as FP8
# parser: tool-call parser — must match how the model emits tool calls (vllm --help lists valid names)
# reasoning_parser: set for models with thinking/reasoning mode (e.g. qwen3); leave blank otherwise
MODEL_TABLE=(
  "Qwen/Qwen2.5-Coder-32B-Instruct          | Qwen2.5-Coder-32B-Instruct  |  64 | 256  | qwen3_xml   | 128 |"
  "Qwen/Qwen2.5-72B-Instruct                | Qwen2.5-72B-Instruct        | 144 | 640  | qwen3_xml   |  32 |"
  "Qwen/Qwen3-32B                           | Qwen3-32B                   |  64 | 256  | qwen3_xml   |  40 |"
  "Qwen/Qwen3-30B-A3B                       | Qwen3-30B-A3B               |  60 | 192  | qwen3_coder | 128 |"
  "Qwen/Qwen3.6-27B                         | Qwen3.6-27B                 |  56 | 256  | qwen3_coder | 256 |     | qwen3"
  "Qwen/Qwen3.6-27B-FP8                     | Qwen3.6-27B                 |  31 | 256  | qwen3_coder | 256 | fp8 | qwen3"
  "meta-llama/Llama-3.3-70B-Instruct        | Llama-3.3-70B-Instruct      | 140 | 640  | llama3_json | 128 |"
  "meta-llama/Llama-3.1-70B-Instruct        | Llama-3.1-70B-Instruct      | 140 | 640  | llama3_json | 128 |"
  "google/gemma-4-31b-it                    | gemma-4-31b-it              |  62 | 1120 | gemma4      | 256 |"
  "RedHatAi/gemma-4-31B-it-FP8-Dynamic      | gemma-4-31b-it              |  31 | 1120 | gemma4      | 256 | fp8"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | DeepSeek-R1-Distill-32B     |  64 | 256  | hermes      | 128 |"
  "THUDM/glm-4-9b-chat                      | glm-4-9b-chat               |  18 |  64  | glm45       | 128 |"
)
# ─────────────────────────────────────────────────────────────────────────────

SERVED_NAME="$MODEL"
MODEL_VRAM=64
KV_KB=256
PARSER="hermes"
MODEL_QUANT=""
REASONING_PARSER=""

for entry in "${MODEL_TABLE[@]}"; do
  IFS='|' read -r repo name mvram kvkb parser ctx_k quant reasoning_parser <<< "$entry"
  repo=$(echo "$repo" | xargs)
  if [[ "$MODEL" == "$repo" ]]; then
    SERVED_NAME=$(echo "$name"             | xargs)
    MODEL_VRAM=$(echo "$mvram"             | xargs)
    KV_KB=$(echo "$kvkb"                   | xargs)
    PARSER=$(echo "$parser"                | xargs)
    MODEL_QUANT=$(echo "$quant"            | xargs)
    REASONING_PARSER=$(echo "$reasoning_parser" | xargs)
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

# Detect GPU and apply GPU-specific settings
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
FP8_FLAGS=""
MEM_UTIL="0.90"
EXTRA_FLAGS=""
if [[ "$GPU_NAME" == *"H100"* ]]; then
  # H100: CUDA graph compilation is more aggressive than A100 (batch sizes up to 8192 vs 2048),
  # causing OOM during graph capture. enforce-eager disables CUDA graphs entirely — H100's
  # raw tensor core throughput is fast enough without them for interactive workloads.
  # VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS not needed when CUDA graphs are disabled.
  EXTRA_FLAGS="--enforce-eager"
  echo "GPU:      $GPU_NAME"
else
  echo "GPU:      ${GPU_NAME:-unknown}"
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
    --gpu-memory-utilization "$MEM_UTIL" \
    --max-model-len "$CTX" \
    $FP8_FLAGS \
    $EXTRA_FLAGS \
    --enable-auto-tool-choice \
    --tool-call-parser "$PARSER" \
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    --port "$PORT" \
    --host 0.0.0.0 \
  >> "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"
echo ""

for i in $(seq 1 120); do
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