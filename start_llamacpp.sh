#!/bin/bash
# Start a local GLM-5.2 (or any GGUF) server using llama.cpp's llama-server
# inside a Singularity container. Exposes an OpenAI-compatible API that scagent
# connects to via SCAGENT_BASE_URL=http://localhost:PORT/v1.
#
# WHY THIS EXISTS (read before relying on it):
#   This is an EVALUATION path, not a serving path. llama.cpp has NO tensor
#   parallelism for MoE models, and GLM-5.2 is a 744B/40B MoE. Across N GPUs it
#   uses LAYER split (-sm layer): the GPUs pool VRAM (so the model fits) but run
#   roughly sequentially per token — you get ~one GPU's worth of compute, not N.
#   Concurrency and prefix caching are also weaker than vLLM. Use this to answer
#   "is GLM-5.2 good for our agent?" today, using the Unsloth GGUF that's already
#   published. For real serving, use AWQ 4-bit + vLLM (true MoE tensor parallel).
#
# Arguments:
#   MODEL  GGUF HuggingFace repo ID   (default: unsloth/GLM-5.2-GGUF)
#   PORT   API port                   (default: 8001 — avoid clashing with vLLM's 8000)
#   GPUS   Number of GPUs             (auto = minimum to fit weights + KV headroom)
#
# Environment overrides:
#   QUANT=UD-IQ4_XS     Which quant subdir/shards to load. Default UD-IQ4_XS (~365GB,
#                       near-lossless, fits 4xH200 with KV room). Use UD-Q4_K_XL (~467GB)
#                       for max quality if you have the VRAM (needs >4 GPUs at full ctx).
#   CTX=32768           Per-session context window (tokens). Total KV = CTX * PARALLEL.
#   PARALLEL=4          Concurrent request slots. KV cache is split CTX-per-slot.
#   KV_TYPE=q8_0        Quantize KV cache (q8_0 / q4_0) to extend context. Requires
#                       flash-attn, which this enables automatically when set. Default
#                       is f16 KV (robust, no FA dependency).
#   THINKING=0          1 = enable GLM reasoning/thinking (slower TTFT). Default off,
#                       matching start_vllm.sh, for fast tool-calling turns.
#   SERVED_NAME=GLM-5.2 Name reported to clients (set SCAGENT_MODEL to match).
#   LLAMACPP_SIF=...    Path to the llama.cpp CUDA SIF (see build command below).
#   LLAMACPP_BIN=...    Path to llama-server inside the SIF (default /app/llama-server).
#
# Examples:
#   bash start_llamacpp.sh                                   # GLM-5.2 UD-IQ4_XS, port 8001, auto GPUs
#   QUANT=UD-Q4_K_XL bash start_llamacpp.sh unsloth/GLM-5.2-GGUF 8001 8
#   KV_TYPE=q8_0 CTX=131072 bash start_llamacpp.sh          # quantized KV for long context

MODEL=${1:-"unsloth/GLM-5.2-GGUF"}
PORT=${2:-8001}
GPUS=${3:-"auto"}

QUANT=${QUANT:-"UD-IQ4_XS"}
CTX=${CTX:-32768}
PARALLEL=${PARALLEL:-4}
KV_TYPE=${KV_TYPE:-""}          # empty = f16 KV (default); set q8_0/q4_0 to quantize
THINKING=${THINKING:-0}
SERVED_NAME=${SERVED_NAME:-"GLM-5.2"}
GPU_IDS=${GPU_IDS:-""}          # explicit GPU index list, e.g. "0,1,5,7" — overrides
                                # autoselect. Use on shared nodes to skip GPUs others use.
# Multi-GPU split mode: layer (default; pools VRAM, ~1 GPU's compute) or tensor
# (real tensor parallelism — splits weights AND KV, parallelizes compute). `tensor`
# needs build 9737+ and fast interconnect (NVLink/NVSwitch). Try tensor for MoE
# speed on a full node; falls back to layer if the model/arch isn't supported.
SPLIT_MODE=${SPLIT_MODE:-layer}

HF_DIR="/data1/peerd/ibrahih3/hf"
LLAMACPP_SIF=${LLAMACPP_SIF:-"/data1/peerd/ibrahih3/llamacpp-server-cuda.sif"}
LLAMACPP_BIN=${LLAMACPP_BIN:-"/app/llama-server"}
LOG_DIR="/data1/peerd/ibrahih3/cs_agent/logs"
LOG="$LOG_DIR/llamacpp_$(hostname -s)_$(echo "${MODEL}_${QUANT}" | sed 's|/|_|g').log"

mkdir -p "$LOG_DIR"

# ── Preconditions: container + weights ────────────────────────────────────────
if [[ ! -f "$LLAMACPP_SIF" ]]; then
  echo "ERROR: llama.cpp SIF not found at $LLAMACPP_SIF"
  echo "       Build it once (needs internet) with:"
  echo ""
  echo "         singularity build $LLAMACPP_SIF docker://ghcr.io/ggml-org/llama.cpp:server-cuda"
  echo ""
  echo "       Or point LLAMACPP_SIF at an existing one."
  exit 1
fi

# The Unsloth GGUF repo contains EVERY quant (multiple TB). Only the chosen quant's
# shards should ever be downloaded — never snapshot the whole repo.
MODEL_CACHE="$HF_DIR/hub/models--$(echo "$MODEL" | sed 's|/|--|g')"
SNAP=$(ls -d "$MODEL_CACHE"/snapshots/*/ 2>/dev/null | head -1)
GGUF=""
if [[ -n "$SNAP" ]]; then
  # Prefer the first shard of a multi-part quant; fall back to a single-file quant.
  GGUF=$(find "$SNAP" -iname "*${QUANT}*00001-of-*.gguf" 2>/dev/null | head -1)
  [[ -z "$GGUF" ]] && GGUF=$(find "$SNAP" -iname "*${QUANT}*.gguf" 2>/dev/null | head -1)
fi

if [[ -z "$GGUF" ]]; then
  echo "ERROR: GGUF for quant '$QUANT' of $MODEL not found under $MODEL_CACHE"
  echo "       Download ONLY this quant (not the whole multi-TB repo) with:"
  echo ""
  echo "         bash download_gguf.sh $MODEL $QUANT"
  echo ""
  exit 1
fi

# Total on-disk size of the selected quant's shards → drives GPU autoselect.
# stat -L follows the HF symlinks into blobs/ (a plain find -printf '%s' would
# report the ~79-byte symlink, not the real shard size).
MODEL_BYTES=$(find "$SNAP" -iname "*${QUANT}*.gguf" -exec stat -L -c '%s' {} \; 2>/dev/null | awk '{s+=$1} END{print s+0}')
MODEL_GB=$(( MODEL_BYTES / 1024 / 1024 / 1024 ))

# ── GPU detection ─────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [[ -z "$GPU_NAME" ]]; then
  echo "ERROR: nvidia-smi not available or no GPU detected"
  exit 1
fi
GPU_VRAM=$(( GPU_MEM_MIB / 1024 ))

echo "GPU:      $GPU_NAME (${GPU_VRAM}GB x ${GPU_COUNT})"
echo "Quant:    $QUANT  (~${MODEL_GB}GB weights)"

# ── GPU count autoselect ──────────────────────────────────────────────────────
# Need enough pooled VRAM for weights + KV cache + compute buffers. Use 88% of
# nominal VRAM and reserve a KV/overhead cushion that scales with total context.
UTIL=88
KV_RESERVE=$(( 16 + (CTX * PARALLEL) / 8192 ))   # rough GB cushion; tune at test time
if [[ -n "$GPU_IDS" ]]; then
  NGPU=$(echo "$GPU_IDS" | tr ',' '\n' | grep -c .)
  echo "GPUs:     explicit GPU_IDS=$GPU_IDS  ($NGPU GPUs — autoselect skipped)"
elif [[ "$GPUS" == "auto" ]]; then
  NGPU=0
  for n in 1 2 4 8; do
    [[ $n -gt $GPU_COUNT ]] && break
    USABLE=$(( n * GPU_VRAM * UTIL / 100 ))
    if [[ $USABLE -ge $(( MODEL_GB + KV_RESERVE )) ]]; then
      NGPU=$n
      break
    fi
  done
  if [[ $NGPU -eq 0 ]]; then
    echo "ERROR: $QUANT (~${MODEL_GB}GB) + ~${KV_RESERVE}GB KV won't fit in $GPU_COUNT x ${GPU_VRAM}GB."
    echo "       Use a smaller quant (e.g. QUANT=UD-IQ3_S) or lower CTX/PARALLEL."
    exit 1
  fi
else
  NGPU=$GPUS
fi

# Container path: HF_DIR is bind-mounted to /hf_cache (see singularity exec below).
GGUF_CONTAINER="/hf_cache${GGUF#"$HF_DIR"}"
TOTAL_CTX=$(( CTX * PARALLEL ))

# ── llama-server flags ────────────────────────────────────────────────────────
# -ngl 999        offload all layers to GPU
# -sm layer       layer split across GPUs (the only mode that works for MoE)
# --jinja         use the GGUF's embedded chat template → enables GLM tool calls
#                 (Unsloth ships fixed templates; no separate tool-call-parser needed)
# -c              TOTAL KV context; each of PARALLEL slots gets CTX = -c / PARALLEL
# --parallel      concurrent request slots (single user + a few sessions)
FLAGS=(
  --model "$GGUF_CONTAINER"
  --alias "$SERVED_NAME"
  -ngl 999
  -sm "$SPLIT_MODE"
  --jinja
  -c "$TOTAL_CTX"
  --parallel "$PARALLEL"
  --host 0.0.0.0
  --port "$PORT"
)

# KV cache: f16 by default (robust). Quantized KV needs flash-attn → enable both.
if [[ -n "$KV_TYPE" ]]; then
  FLAGS+=(--cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE" -fa on)
  echo "Config:   layer-split, KV=$KV_TYPE (flash-attn on)"
else
  echo "Config:   ${SPLIT_MODE}-split, KV=f16"
fi

# Thinking off by default for fast tool-calling turns. If your llama.cpp build
# rejects --chat-template-kwargs, drop this block (older builds lack the flag).
if [[ "$THINKING" == "0" ]]; then
  FLAGS+=(--chat-template-kwargs '{"enable_thinking": false}')
  echo "Config:   thinking disabled"
else
  echo "Config:   thinking ENABLED (slower TTFT)"
fi

echo "Model:    $MODEL ($QUANT)"
echo "Served:   $SERVED_NAME"
echo "GPUs:     $NGPU  (layer split — NOT tensor parallel; MoE has no TP in llama.cpp)"
echo "Context:  ${CTX} tokens/slot x ${PARALLEL} slots = ${TOTAL_CTX} total tokens"
echo "Port:     $PORT"
echo "Log:      $LOG"
echo ""

# ── Port preflight (same rationale as start_vllm.sh) ─────────────────────────
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$PORT "; then
  echo "ERROR: port $PORT is already in use — health checks would hit that server."
  echo "       Free it (ss -ltnp | grep :$PORT) or pick a different PORT."
  exit 1
fi

# CUDA_VISIBLE_DEVICES limits llama.cpp to the chosen devices. GPU_IDS picks
# specific indices (shared nodes); otherwise use the first NGPU devices.
if [[ -n "$GPU_IDS" ]]; then DEVICES="$GPU_IDS"; else DEVICES=$(seq -s, 0 $(( NGPU - 1 ))); fi

# --cleanenv: the host shell's LD_PRELOAD (scagent_rapids libs) leaks into the
#   container and spams errors otherwise.
# LD_LIBRARY_PATH must include /app (llama-server's own .so live there, no rpath)
#   AND the CUDA + --nv driver paths (/.singularity.d/libs) or GPU access breaks.
singularity exec --nv --cleanenv \
  --env "LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64:/.singularity.d/libs" \
  --env "CUDA_VISIBLE_DEVICES=$DEVICES" \
  --bind "$HF_DIR":/hf_cache \
  "$LLAMACPP_SIF" \
  "$LLAMACPP_BIN" "${FLAGS[@]}" \
  >> "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"
echo ""

# Readiness probe. Large GGUFs take minutes to mmap + upload to VRAM; /health
# returns 503 while loading and 200 when ready. Fail fast if the process dies.
for i in $(seq 1 180); do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "llama-server exited during startup — last lines of log:"
    tail -20 "$LOG"
    echo ""
    echo "Full log: $LOG"
    exit 1
  fi
  if curl -s "http://localhost:$PORT/health" 2>/dev/null | grep -q '"ok"'; then
    echo "Server ready at http://localhost:$PORT/v1"
    echo "  SCAGENT_MODEL=$SERVED_NAME"
    echo "  SCAGENT_BASE_URL=http://localhost:$PORT/v1"
    exit 0
  fi
done
echo "Timed out — check log: $LOG"
exit 1
