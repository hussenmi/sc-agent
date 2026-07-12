#!/usr/bin/env bash
# Start a local LLM server on the DGX Spark (GB10 / Blackwell / sm_121, aarch64),
# using NVIDIA's NGC vLLM container. Exposes an OpenAI-compatible API that scagent
# and experiments/bench_*.py connect to — same contract as start_vllm.sh, just a
# different host recipe.
#
# WHY A SEPARATE SCRIPT (not start_vllm.sh):
#   start_vllm.sh targets x86_64 Iris nodes: Singularity .sif images, /data1 shared
#   paths, and GPU-class detection that only knows Ampere (cc 8.0) / Hopper (9.0).
#   The Spark is none of those — it's aarch64, GB10 is sm_121, and stock/PyPI vLLM
#   wheels only compile through sm_120, so vLLM must come from a GB10-aware
#   container. This script encodes that Spark recipe and nothing else.
#
# Arguments:
#   MODEL  HuggingFace repo ID (default: nvidia/Qwen3.6-27B-NVFP4 — native Blackwell FP4)
#   PORT   API port — scagent uses SCAGENT_BASE_URL=http://localhost:PORT/v1
#
# Environment overrides:
#   VLLM_IMAGE      NGC container tag (default nvcr.io/nvidia/vllm:26.05-py3)
#   HF_CACHE        host HF cache mounted into the container (default ~/.cache/huggingface)
#   MAX_MODEL_LEN   context cap (default 262144 = Qwen3.6 native). Lower this first
#                   if the KV cache OOMs — on the 128GB unified pool, FP8 KV for the
#                   full 262K window is ~60GB, which fits alongside ~16GB NVFP4
#                   weights, but leave OS/runtime headroom.
#   MEM_UTIL        --gpu-memory-utilization (default 0.37). On the Spark the LLM
#                   server and scagent's GPU compute (run_scvi, RAPIDS) share the
#                   ONE 121GB unified pool — plus the AnnData object lives in that
#                   same RAM. vLLM grabs this fraction at boot and holds it for life,
#                   so over-reserving steals memory from scvi and risks host OOM on a
#                   big study. 0.37 (~44GB) covers weights + activations + one full
#                   262K-token KV request with prefix headroom, and NOTHING more,
#                   leaving ~77GB for scvi/RAPIDS/AnnData/OS. Raise ONLY if you've
#                   confirmed your largest scvi run fits in what's left (or if the LLM
#                   runs on a different host than the compute chain).
#   MAX_NUM_SEQS    max concurrent sequences (default 4). scagent is single-user;
#                   above ~4 decode streams the GB10 memory-bandwidth tax outweighs
#                   batching gains (per vLLM's Spark writeup).
#   SPEC            1=MTP speculative decoding (default), 0=off. The NVFP4 checkpoint
#                   KEEPS its MTP head (config: mtp_num_hidden_layers=1, and the head
#                   is excluded from quantization), so spec-decode still applies.
#   SPEC_K          num_speculative_tokens for MTP (default 4). Swept on this GB10
#                   (vllm-host/, 2026-07-09): k=4 is best single-stream (~3x baseline;
#                   code content gains most, prose plateaus at k=3). Drop to 2-3 if the
#                   server mostly handles several CONCURRENT agents — deep drafts waste
#                   compute on rejects under load. scagent is single-user, so 4 fits.
#   THINKING        0=off (default, fast tool-calling), 1=on.
#   QUANT           empty=let vLLM auto-detect from the checkpoint (default; the
#                   NVFP4 repo ships hf_quant_config.json, so this normally Just
#                   Works). Set to "modelopt" ONLY if first boot fails to recognize
#                   the NVFP4 format — then it's passed as --quantization modelopt.
#
# Examples:
#   bash start_vllm_spark.sh                                   # Qwen3.6-27B-NVFP4, MTP on
#   MAX_MODEL_LEN=131072 bash start_vllm_spark.sh             # smaller KV footprint
#   SPEC=0 bash start_vllm_spark.sh                            # no speculative decoding
#
# Pre-req: model downloaded into HF_CACHE first, e.g.
#   HF_HUB_ENABLE_HF_TRANSFER=1 uvx --from huggingface_hub hf download nvidia/Qwen3.6-27B-NVFP4
set -uo pipefail

MODEL=${1:-"nvidia/Qwen3.6-27B-NVFP4"}
PORT=${2:-8000}

VLLM_IMAGE=${VLLM_IMAGE:-"nvcr.io/nvidia/vllm:26.05-py3"}
HF_CACHE=${HF_CACHE:-"$HOME/.cache/huggingface"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MEM_UTIL=${MEM_UTIL:-0.37}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
SPEC=${SPEC:-1}
SPEC_K=${SPEC_K:-4}
THINKING=${THINKING:-0}
QUANT=${QUANT:-}   # empty = auto-detect from checkpoint; set "modelopt" only if that fails

# ── Per-model settings ────────────────────────────────────────────────────────
#   HF repo ID | served name | tool_parser | reasoning_parser
# served name MUST match SCAGENT_MODEL in .env or scagent's requests 404.
MODEL_TABLE=(
  "nvidia/Qwen3.6-27B-NVFP4 | Qwen3.6-27B | qwen3_coder | qwen3"
  "Qwen/Qwen3.6-27B-FP8     | Qwen3.6-27B | qwen3_coder | qwen3"
  "Qwen/Qwen3.6-27B         | Qwen3.6-27B | qwen3_coder | qwen3"
)
SERVED_NAME="$MODEL"; TOOL_PARSER="qwen3_coder"; REASONING_PARSER="qwen3"
for entry in "${MODEL_TABLE[@]}"; do
  IFS='|' read -r repo name tparser rparser <<< "$entry"
  if [[ "$MODEL" == "$(echo "$repo" | xargs)" ]]; then
    SERVED_NAME=$(echo "$name"    | xargs)
    TOOL_PARSER=$(echo "$tparser" | xargs)
    REASONING_PARSER=$(echo "$rparser" | xargs)
    break
  fi
done

# ── Pre-flight ────────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot talk to the Docker daemon."
  echo "       You are probably not in the 'docker' group. Fix (one-time, needs sudo):"
  echo "         sudo usermod -aG docker \$USER   # then re-login (or: newgrp docker)"
  exit 1
fi
# Confirm the container runtime can see the GPU before we try to serve.
if ! docker run --rm --gpus all "$VLLM_IMAGE" nvidia-smi -L >/dev/null 2>&1; then
  echo "WARNING: '--gpus all' GPU probe failed against $VLLM_IMAGE."
  echo "         Check the NVIDIA Container Toolkit is configured (nvidia-ctk runtime configure)."
  echo "         Continuing anyway — vLLM will fail loudly if the GPU is truly unavailable."
fi
# Fail fast if the port is taken, or the readiness probe below gets answered by
# whatever already owns it and we falsely report "ready".
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$PORT "; then
  echo "ERROR: port $PORT is already in use (find it: ss -ltnp | grep :$PORT)."
  exit 1
fi
MODEL_CACHE="$HF_CACHE/hub/models--$(echo "$MODEL" | sed 's|/|--|g')"
[[ -d "$MODEL_CACHE" ]] || {
  echo "ERROR: model not cached at $MODEL_CACHE"
  echo "Download first:  HF_HUB_ENABLE_HF_TRANSFER=1 uvx --from huggingface_hub hf download $MODEL"
  exit 1
}

# ── vLLM flags ────────────────────────────────────────────────────────────────
# NOTE: do NOT add --enforce-eager on GB10. CUDA graphs are effectively mandatory
# on sm_121 — disabling them cuts throughput ~55% (per vLLM's DGX Spark writeup).
EXTRA=()

# Speculative decoding: MTP is native to Qwen3.6 and survives NVFP4 quantization
# here (the MTP head is kept out of the FP4 quant set). k=SPEC_K (default 4) is the
# GB10-swept single-stream optimum (vllm-host/ bench, 2026-07-09); k=1 was the older
# un-swept Iris carry-over. Acceptance collapses on reasoning tokens, so spec-decode
# is off when THINKING=1.
if [[ "$SPEC" == "1" && "$THINKING" == "0" ]]; then
  EXTRA+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC_K}")
fi

# Thinking off by default: 3-10x faster TTFT for routine tool calls.
if [[ "$THINKING" == "0" ]]; then
  EXTRA+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi

# Quantization: normally auto-detected from the checkpoint's hf_quant_config.json.
# Only forced if QUANT is set (escape hatch for first-boot auto-detect failure).
if [[ -n "$QUANT" ]]; then
  EXTRA+=(--quantization "$QUANT")
fi

echo "=================================================="
echo "Host:      DGX Spark (GB10 / sm_121 / aarch64)"
echo "Image:     $VLLM_IMAGE"
echo "Model:     $MODEL"
echo "Served:    $SERVED_NAME   (must equal SCAGENT_MODEL in .env)"
echo "Context:   max_model_len=$MAX_MODEL_LEN"
echo "Memory:    gpu_mem_util=$MEM_UTIL  max_num_seqs=$MAX_NUM_SEQS  (128GB unified pool)"
echo "Parsers:   tool=$TOOL_PARSER  reasoning=$REASONING_PARSER"
echo "Quant:     $([[ -n "$QUANT" ]] && echo "$QUANT (forced)" || echo "auto-detect from checkpoint")"
echo "Spec:      $([[ "$SPEC" == "1" && "$THINKING" == "0" ]] && echo "MTP k=$SPEC_K" || echo "off")"
echo "Thinking:  $([[ "$THINKING" == "1" ]] && echo "on" || echo "off")"
echo "Port:      $PORT"
echo "=================================================="

# --ipc=host: vLLM workers use shared memory; the default 64MB /dev/shm is too small.
# HF cache mounted read-write so first-run dynamic-module / compile artifacts persist.
# HF_HUB_OFFLINE=1: weights are already local; don't reach out mid-serve.
exec docker run --rm --gpus all --ipc=host \
  -p "$PORT:$PORT" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_V1=1 \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --reasoning-parser "$REASONING_PARSER" \
    "${EXTRA[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"
