#!/bin/bash
# Start a local LLM server using vLLM inside a Singularity container.
# The server exposes an OpenAI-compatible API that scagent connects to.
#
# Adapts to the GPU it finds — A100, H100 PCIe, H100 NVL, or H100 SXM —
# and picks weights/quant/fusions/speculative method accordingly.
#
# Arguments:
#   MODEL  HuggingFace repo ID (default: Qwen2.5-Coder-32B-Instruct)
#   PORT   API port — scagent uses SCAGENT_BASE_URL=http://localhost:PORT/v1
#   GPUS   Number of GPUs (auto = minimum to fit model + reasonable KV cache)
#
# Environment overrides:
#   THINKING=1   Enable model reasoning/thinking mode (slower TTFT)
#   SPEC=0       Disable speculative decoding
#   SPEC=1       Force-enable for models that have a configured method
#   LONG_CTX=1   Extend context to ~1M tokens via YaRN (Qwen3.6 only).
#                Actual window is capped by KV budget — 2×H100 FP8 ≈ 450K,
#                4×H100 FP8 ≈ 1M. Degrades short-context quality slightly.
#
# Examples:
#   bash start_vllm.sh Qwen/Qwen3.6-27B-FP8 8000 2                          # FP8 — H100 only
#   LONG_CTX=1 bash start_vllm.sh Qwen/Qwen3.6-27B-FP8 8000 4              # FP8 + YaRN 1M
#   bash start_vllm.sh Qwen/Qwen3.6-27B 8000 2                              # BF16 — required on A100
#   bash start_vllm.sh meta-llama/Llama-3.3-70B-Instruct 8000 4
#
# Download a model first with:  bash download_model.sh <model_id>

MODEL=${1:-"Qwen/Qwen2.5-Coder-32B-Instruct"}
PORT=${2:-8000}
GPUS=${3:-"auto"}
THINKING=${THINKING:-0}
SPEC=${SPEC:-"auto"}
LONG_CTX=${LONG_CTX:-1}

HF_DIR="/data1/peerd/ibrahih3/hf"
SIF=${VLLM_SIF:-"/data1/peerd/ibrahih3/vllm-openai_gemma4.sif"}
# Log path includes hostname so launches from different nodes don't clobber each
# other on the shared filesystem.
LOG_DIR="/data1/peerd/ibrahih3/cs_agent/logs"
LOG="$LOG_DIR/vllm_$(hostname -s)_$(echo $MODEL | sed 's|/|_|g').log"

mkdir -p "$LOG_DIR"

# ── Per-model settings ────────────────────────────────────────────────────────
# Add a new model by appending a line. Fields:
#   HF repo ID | served name | weight GB | KV KB/token | parser | max ctx K | quant | reasoning_parser
# weight GB: actual loaded size (FP8 weight size for pre-quantized FP8 checkpoints)
# parser: tool-call parser — must match how the model emits tool calls
# quant: blank for BF16; "fp8" for pre-quantized FP8 checkpoints (rejected on Ampere)
# reasoning_parser: set for models with thinking mode (e.g. qwen3); blank otherwise
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
ctx_k=128

for entry in "${MODEL_TABLE[@]}"; do
  IFS='|' read -r repo name mvram kvkb parser ctxk quant reasoning_parser <<< "$entry"
  repo=$(echo "$repo" | xargs)
  if [[ "$MODEL" == "$repo" ]]; then
    SERVED_NAME=$(echo "$name"             | xargs)
    MODEL_VRAM=$(echo "$mvram"             | xargs)
    KV_KB=$(echo "$kvkb"                   | xargs)
    PARSER=$(echo "$parser"                | xargs)
    ctx_k=$(echo "$ctxk"                   | xargs)
    MODEL_QUANT=$(echo "$quant"            | xargs)
    REASONING_PARSER=$(echo "$reasoning_parser" | xargs)
    break
  fi
done

# ── GPU detection ─────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

if [[ -z "$GPU_NAME" ]]; then
  echo "ERROR: nvidia-smi not available or no GPU detected"
  exit 1
fi

GPU_VRAM=$(( GPU_MEM_MIB / 1024 ))

# Classify hardware. The axis that matters for vLLM kernel selection on Hopper is
# whether NVSwitch is present — SXM boards have it, PCIe and NVL don't. Without
# NVSwitch, FlashInfer's SymmDeviceMemory multicast fails and several fusions
# must be disabled. The SKU name is a proxy: SXM/HBM3 → has NVSwitch.
case "$GPU_CC" in
  8.0)
    HW_CLASS="ampere"                  # A100
    ;;
  9.0)
    if [[ "$GPU_NAME" == *"SXM"* || "$GPU_NAME" == *"HBM3"* ]]; then
      HW_CLASS="hopper_nvswitch"       # H100 SXM
    else
      HW_CLASS="hopper_no_nvswitch"    # H100 PCIe / H100 NVL
    fi
    ;;
  *)
    HW_CLASS="unknown"
    ;;
esac

echo "GPU:      $GPU_NAME (cc=$GPU_CC, ${GPU_VRAM}GB)"
echo "Class:    $HW_CLASS"

# Reject FP8 weights on Ampere — Marlin would silently dequantize to BF16 on every
# forward pass. Slower than just running BF16 directly with no benefit.
if [[ "$HW_CLASS" == "ampere" && "$MODEL_QUANT" == "fp8" ]]; then
  echo ""
  echo "ERROR: FP8 model on A100 has no native FP8 support (compute capability 8.0)."
  echo "       Marlin would dequantize to BF16 on every forward pass — slower than BF16 direct."
  case "$MODEL" in
    Qwen/Qwen3.6-27B-FP8)
      echo "       Use the BF16 sibling instead:"
      echo "         bash start_vllm.sh Qwen/Qwen3.6-27B $PORT $GPUS"
      ;;
    RedHatAi/gemma-4-31B-it-FP8-Dynamic)
      echo "       Use the BF16 sibling instead:"
      echo "         bash start_vllm.sh google/gemma-4-31b-it $PORT $GPUS"
      ;;
  esac
  exit 1
fi

# ── GPU count autoselect ──────────────────────────────────────────────────────
UTIL=90  # percent

if [[ "$GPUS" == "auto" ]]; then
  for n in 1 2 4 8; do
    USABLE=$(( n * GPU_VRAM * UTIL / 100 ))
    KV_BUDGET=$(( USABLE - MODEL_VRAM ))
    if [[ $KV_BUDGET -ge 8 ]]; then
      TP=$n
      break
    fi
  done
else
  TP=$GPUS
fi

# ── Hardware-specific flags ───────────────────────────────────────────────────
MEM_UTIL="0.90"
VLLM_USE_DEEP_GEMM=1
EXTRA_FLAGS=()

case "$HW_CLASS" in
  ampere)
    # BF16 throughput is the best path on Ampere. FP8 KV cache roughly doubles
    # effective KV memory, recovering most of what BF16 weights cost vs FP8.
    # Use E5M2 explicitly: bare "fp8" defaults to E4M3 (fp8e4nv) which Triton
    # cannot compile on cc=8.0. Qwen incidentally avoids the Triton path; Gemma 4
    # fuses FP8 casts into its RMS-norm kernel and crashes there. E5M2 is the
    # format A100 actually supports natively and works for most models.
    #
    # Gemma 4 is the exception: vLLM's attention module hard-asserts kv_cache_dtype
    # in {"fp8","fp8_e4m3"} for the Gemma 4 path, rejecting E5M2 outright. So on
    # A100 + Gemma 4 we have no working FP8 KV option (E4M3 fails Triton, E5M2
    # fails vLLM's assert) and must fall back to BF16 KV cache.
    if [[ "$PARSER" == "gemma4" ]]; then
      VLLM_USE_DEEP_GEMM=0
      echo "Config:   BF16 weights + BF16 KV cache (Gemma 4 on A100 — FP8 KV not supported)"
    else
      EXTRA_FLAGS+=("--kv-cache-dtype" "fp8_e5m2")
      VLLM_USE_DEEP_GEMM=0
      echo "Config:   BF16 weights + FP8 KV cache (E5M2)"
    fi
    ;;
  hopper_no_nvswitch)
    # H100 PCIe and NVL both lack NVSwitch (NVL pairs use an NVLink bridge at best;
    # NVL pairs that aren't bridged talk over PCIe). FlashInfer's SymmDeviceMemory
    # workspace requires NVSwitch for GPU-to-GPU multicasting and
    # fails mid-compile. torch.compile then restarts without the allreduce-rms
    # fusion, running compilation twice. The doubled intermediate tensors stay in
    # PyTorch's CUDA allocator cache (~50 GB) and push the KV-cache profiler to
    # report num_gpu_blocks=0, causing a DeepGEMM crash during graph capture.
    # Pre-disabling the fusion prevents the mid-compile retry and keeps memory stable.
    EXTRA_FLAGS+=("--compilation-config" '{"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true,"fuse_attn_quant":false,"enable_sp":false,"fuse_gemm_comms":false,"fuse_allreduce_rms":false}}')
    EXTRA_FLAGS+=("--disable-custom-all-reduce")
    EXTRA_FLAGS+=("--kv-cache-dtype" "fp8")
    VLLM_USE_DEEP_GEMM=0
    echo "Config:   FP8 weights + FP8 KV cache"
    echo "Config:   SymmMem-dependent fusions disabled (no NVSwitch)"
    echo "Config:   custom all-reduce disabled, DeepGEMM disabled"
    ;;
  hopper_nvswitch)
    EXTRA_FLAGS+=("--kv-cache-dtype" "fp8")
    echo "Config:   FP8 weights + FP8 KV cache + DeepGEMM + SymmMem fusions"
    ;;
  *)
    echo "WARNING:  Unknown GPU class — using vLLM defaults"
    ;;
esac

# Prefix caching: on by default in vLLM V1. Explicit flag for V0 compatibility.
EXTRA_FLAGS+=("--enable-prefix-caching")

# Single-user interactive session: limit concurrent sequences. Default (256)
# pre-allocates KV slots for 256 phantom sessions and blows out memory headroom.
EXTRA_FLAGS+=("--max-num-seqs" "8")

# Thinking mode: off by default for speed (3-10× faster TTFT for routine tool calls).
# Override: THINKING=1 bash start_vllm.sh <model>   or   export THINKING=1
if [[ "$THINKING" == "0" ]]; then
  EXTRA_FLAGS+=("--default-chat-template-kwargs" '{"enable_thinking": false}')
fi

# ── Speculative decoding ──────────────────────────────────────────────────────
# Per the official vLLM recipe page, Qwen3.6-27B uses MTP (Multi-Token Prediction)
# baked into the target model itself — no separate draft model, no extra VRAM,
# no SymmMem fight. Replaces the older DFlash setup which had ~5% acceptance with
# this checkpoint (the Qwen3.6 DFlash drafter is still in training as of 2026-04-27).
#
#   SPEC=auto (default)  — on for models with a configured method
#   SPEC=0               — off
#   SPEC=1               — force-enable (no-op if no method configured)
#
# Speculative is also disabled when THINKING=1 — acceptance collapses on reasoning tokens.
if [[ "$THINKING" == "0" && "$SPEC" != "0" ]]; then
  case "$MODEL" in
    Qwen/Qwen3.6-27B|Qwen/Qwen3.6-27B-FP8)
      EXTRA_FLAGS+=("--speculative-config" '{"method":"mtp","num_speculative_tokens":1}')
      echo "Speculative: MTP (k=1, native to Qwen3.6)"
      ;;
    *)
      if [[ "$SPEC" == "1" ]]; then
        echo "Speculative: requested but no method configured for $MODEL"
      fi
      ;;
  esac
elif [[ "$THINKING" == "1" ]]; then
  echo "Speculative: disabled (THINKING=1)"
elif [[ "$SPEC" == "0" ]]; then
  echo "Speculative: disabled (SPEC=0)"
fi

# ── Long context: YaRN extension to ~1M tokens (Qwen3.6 only) ────────────────
# Native cap is 262K. YaRN factor=4.0 enables up to 1,010,000 tokens but the
# actual window is capped by the KV budget below — you need enough GPUs:
#   2×H100 80GB FP8 model ≈ 452K   |   4×H100 80GB FP8 model ≈ 1M
# Short-context quality degrades slightly at factor 4.0.
LONG_CTX_SUPPORTED=0
if [[ "$LONG_CTX" == "1" ]]; then
  case "$MODEL" in
    Qwen/Qwen3.6-27B|Qwen/Qwen3.6-27B-FP8)
      LONG_CTX_SUPPORTED=1
      ctx_k=1010
      EXTRA_FLAGS+=("--hf-overrides" '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}')
      echo "Long ctx:  YaRN factor=4.0 → up to 1010K (actual limited by KV budget)"
      ;;
    *)
      echo "Long ctx:  LONG_CTX=1 requested but not configured for $MODEL — using default ${ctx_k}K"
      ;;
  esac
fi
if [[ "$LONG_CTX_SUPPORTED" == "1" ]]; then
  export SINGULARITYENV_VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

# ── Context window: KV budget → tokens, capped at model native max ────────────
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

# Persistent compile cache — keyed by model AND hardware class so A100/H100 caches
# don't collide. Without this, a cache hit from a different GPU class can cause a
# crash during CUDA graph capture (compiled kernels reference unavailable instructions).
MODEL_TAG="$(echo "$MODEL" | sed 's|/|_|g')_${HW_CLASS}"
mkdir -p "$HF_DIR/vllm_compile_cache/$MODEL_TAG"

SINGULARITYENV_HF_HUB_CACHE=/hf_cache/hub \
SINGULARITYENV_HF_HUB_OFFLINE=1 \
SINGULARITYENV_TRANSFORMERS_OFFLINE=1 \
SINGULARITYENV_PYTHONNOUSERSITE=1 \
SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="/hf_cache/vllm_compile_cache/$MODEL_TAG/torchinductor" \
SINGULARITYENV_VLLM_CACHE_ROOT="/hf_cache/vllm_compile_cache/$MODEL_TAG/vllm" \
SINGULARITYENV_VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
SINGULARITYENV_VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM" \
SINGULARITYENV_VLLM_ENGINE_READY_TIMEOUT_S=1800 \
SINGULARITYENV_TMPDIR=/tmp \
singularity exec --nv \
  --bind "$HF_DIR":/hf_cache \
  "$SIF" \
  vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --max-model-len "$CTX" \
    "${EXTRA_FLAGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser "$PARSER" \
    ${REASONING_PARSER:+--reasoning-parser "$REASONING_PARSER"} \
    --port "$PORT" \
    --host 0.0.0.0 \
  >> "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"
echo ""

# First startup with CUDA graphs takes longer — torch.compile + DeepGEMM warmup
# can add several minutes. Compiled artifacts are cached under VLLM_CACHE_ROOT
# so subsequent startups are faster. Timeout: 15 min.
# Fails fast if vLLM exits during startup instead of polling until timeout.
for i in $(seq 1 180); do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "vLLM exited during startup — last lines of log:"
    tail -20 "$LOG"
    echo ""
    echo "Full log: $LOG"
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
