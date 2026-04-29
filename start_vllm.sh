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
LOG_HOST=$(hostname -s 2>/dev/null || echo unknown)
LOG_TS=$(date +%Y%m%d_%H%M%S)
LOG="/data1/peerd/ibrahih3/tmp/vllm_${LOG_HOST}_${PORT}_$(echo "$MODEL" | sed 's|/|_|g')_${LOG_TS}.log"

mkdir -p /data1/peerd/ibrahih3/tmp

HOST_SHORT=$(hostname -s 2>/dev/null || hostname)
if [[ "$HOST_SHORT" =~ ^isc[bcd] ]]; then
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "WARNING: $HOST_SHORT looks like a compute node, but no Slurm job allocation is active."
    echo "         H100 nodes in particular may kill or reject processes started outside an allocation."
  elif [[ -n "${SLURM_JOB_NODELIST:-}" ]] && [[ "$SLURM_JOB_NODELIST" != *"$HOST_SHORT"* ]]; then
    echo "WARNING: Running on $HOST_SHORT, but current Slurm job $SLURM_JOB_ID is allocated to $SLURM_JOB_NODELIST."
    echo "         Starting vLLM outside the allocated node can fail unpredictably."
  fi
fi

# ── Per-model settings ────────────────────────────────────────────────────────
# Add a new model by appending a line here. Fields:
#   HF repo ID | served name (used in .env) | weight size GB | KV KB/token | parser | max ctx K | quant | reasoning_parser
# weight size GB: actual loaded size (use FP8 weight size for pre-quantized FP8 checkpoints)
# quant: leave blank for BF16; set to "fp8" for pre-quantized FP8 checkpoints
#   - blank: standard BF16 checkpoint sizing
#   - fp8:   pre-quantized FP8 checkpoint sizing (~half of BF16)
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

# Models that resolve to a multimodal ConditionalGeneration class in vLLM
# tend to spend extra memory on encoder setup even for text-only workloads.
IS_HYBRID_MODEL=0
case "$MODEL" in
  Qwen/Qwen3.6-*|google/gemma-4-*|RedHatAi/gemma-4-*)
    IS_HYBRID_MODEL=1
    ;;
esac

# Resolve GPU count: auto = minimum needed to fit model + reasonable KV cache
GPU_VRAM=80
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ "$GPU_MEM_MIB" =~ ^[0-9]+$ ]] && [[ "$GPU_MEM_MIB" -gt 0 ]]; then
  GPU_VRAM=$(( GPU_MEM_MIB / 1024 ))
fi

MEM_UTIL=${VLLM_MEM_UTIL:-"0.90"}
UTIL=$(awk -v u="$MEM_UTIL" 'BEGIN { printf "%d", u * 100 }')

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
EXTRA_FLAGS=""
CACHE_FLAGS=""
MM_FLAGS=""
SCHED_FLAGS=""
GDN_FLAGS=""
if [[ "$GPU_NAME" == *"H100"* ]]; then
  # H100: CUDA graph compilation is more aggressive than A100 (batch sizes up to 8192 vs 2048),
  # causing OOM during graph capture. enforce-eager disables CUDA graphs entirely — H100's
  # raw tensor core throughput is fast enough without them for interactive workloads.
  EXTRA_FLAGS="--enforce-eager"

  # Single-GPU H100 runs with hybrid VLM checkpoints can still die during
  # encoder/KV warmup even after graphs are disabled. Default to a safer
  # profile; all knobs can be overridden via env vars when needed.
  if [[ -z "${VLLM_MEM_UTIL:-}" ]]; then
    MEM_UTIL="0.80"
  fi
  UTIL=$(awk -v u="$MEM_UTIL" 'BEGIN { printf "%d", u * 100 }')

  if [[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]]; then
    KV_CACHE_DTYPE="$VLLM_KV_CACHE_DTYPE"
    CACHE_FLAGS="--kv-cache-dtype $KV_CACHE_DTYPE"
  fi

  MAX_BATCHED_TOKENS=${VLLM_MAX_BATCHED_TOKENS:-4096}
  MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8}
  case "$MODEL" in
    Qwen/Qwen3.6-*)
      # Qwen3.6 GDN layers need a lower batch-token cap during vLLM startup.
      # vLLM issue: https://github.com/vllm-project/vllm/issues/37714
      if [[ -z "${VLLM_MAX_BATCHED_TOKENS:-}" ]]; then
        MAX_BATCHED_TOKENS=2096
      fi
      if [[ -z "${VLLM_GDN_PREFILL_BACKEND:-}" ]]; then
        GDN_FLAGS="--gdn-prefill-backend triton"
      else
        GDN_FLAGS="--gdn-prefill-backend $VLLM_GDN_PREFILL_BACKEND"
      fi
      ;;
  esac
  SCHED_FLAGS="--max-num-batched-tokens $MAX_BATCHED_TOKENS --max-num-seqs $MAX_NUM_SEQS"

  if [[ $TP -eq 1 && $IS_HYBRID_MODEL -eq 1 && "${VLLM_TEXT_ONLY:-0}" == "1" ]]; then
    MM_FLAGS="--language-model-only --mm-processor-cache-gb 0"
    echo "Mode:     text-only (hybrid encoder disabled by VLLM_TEXT_ONLY=1)"
  elif [[ $IS_HYBRID_MODEL -eq 1 ]]; then
    echo "Mode:     multimodal"
  fi

  echo "GPU:      $GPU_NAME → eager mode (safer H100 startup profile)"
else
  echo "GPU:      ${GPU_NAME:-unknown}"
fi

# Context window: KV budget → tokens, capped at model native max (128K)
USABLE=$(( TP * GPU_VRAM * UTIL / 100 ))
KV_BUDGET=$(( USABLE - MODEL_VRAM ))
MAX_CTX=$(( KV_BUDGET * 1024 * 1024 / KV_KB ))
NATIVE_CAP=$(( ${ctx_k:-128} * 1024 ))
CTX=$(( MAX_CTX < NATIVE_CAP ? MAX_CTX : NATIVE_CAP ))

if [[ "$GPU_NAME" == *"H100"* && "$TP" -eq 1 && -z "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  H100_CTX_CAP=$(( ${VLLM_H100_CTX_CAP_K:-32} * 1024 ))
  CTX=$(( CTX < H100_CTX_CAP ? CTX : H100_CTX_CAP ))
fi

CTX_K=$(( CTX / 1024 ))

if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  CTX="$VLLM_MAX_MODEL_LEN"
  CTX_K=$(( CTX / 1024 ))
fi

echo "Model:    $MODEL"
echo "Served:   $SERVED_NAME"
echo "GPUs:     $TP  (tensor-parallel-size)"
echo "Context:  ${CTX_K}K tokens"
echo "Mem util: $MEM_UTIL"
if [[ -n "$CACHE_FLAGS" ]]; then
  echo "KV cache: ${KV_CACHE_DTYPE}"
fi
if [[ -n "$SCHED_FLAGS" ]]; then
  echo "Batching: tokens=$MAX_BATCHED_TOKENS seqs=$MAX_NUM_SEQS"
fi
if [[ -n "$GDN_FLAGS" ]]; then
  echo "GDN:      ${VLLM_GDN_PREFILL_BACKEND:-triton}"
fi
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
    $CACHE_FLAGS \
    $MM_FLAGS \
    $SCHED_FLAGS \
    $GDN_FLAGS \
    $EXTRA_FLAGS \
    ${VLLM_EXTRA_FLAGS:-} \
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
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Server exited before becoming healthy — check log: $LOG"
    tail -n 60 "$LOG"
    exit 1
  fi
  if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "Server ready at http://localhost:$PORT/v1"
    echo "  SCAGENT_MODEL=$SERVED_NAME"
    echo "  SCAGENT_BASE_URL=http://localhost:$PORT/v1"
    exit 0
  fi
done
echo "Timed out — check log: $LOG"
exit 1
