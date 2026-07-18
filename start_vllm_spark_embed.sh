#!/usr/bin/env bash
# Serve an NVFP4 text-EMBEDDING model on the DGX Spark (GB10 / Blackwell / sm_121,
# aarch64) via NVIDIA's NGC vLLM container. Exposes an OpenAI-compatible embeddings
# API (/v1/embeddings) plus NVIDIA's native /v2/embed. Companion to
# start_vllm_spark.sh, which serves generative (chat/tool) models; this one is the
# embedder/retrieval half of a RAG stack on the same box.
#
# WHY THE FLAGS BELOW ARE NON-OBVIOUS (validated on spark-e5d5, 2026-07-16 with
# nvcr.io/nvidia/vllm:26.06-py3 == vLLM 0.22.1, model Nemotron-3-Embed-1B-NVFP4):
#
#   1. NGC container, not PyPI. The model card says `pip install vllm==0.25.0`, but
#      PyPI/stock vLLM wheels only compile through sm_120; GB10 is sm_121, so vLLM
#      must come from a GB10-aware NGC container (same reason as start_vllm_spark.sh).
#
#   2. --hf-overrides architectures. The June NGC image predates the July-15 model,
#      so its registry has no `Ministral3Model` (the arch the checkpoint declares).
#      It DOES have `Ministral3ForCausalLM`; we remap to that and let --runner pooling
#      convert it into an embedder (drops the LM head, adds a pooler). When a newer
#      NGC image registers Ministral3Model natively, this override can be dropped.
#
#   3. --runner pooling (NOT the docs' `--task embed`; that flag was renamed).
#
#   4. --pooler-config pooling_type=MEAN. Nemotron 3 Embed is mean-pooled + L2
#      normalized. The CausalLM->embed conversion would otherwise default to LAST.
#      Normalization is applied automatically by the embed-convert path (verified:
#      output vectors have L2 norm == 1.0).
#
#   NVFP4 works as-is: the checkpoint carries ModelOpt NVFP4 metadata, vLLM auto-
#   detects it (FlashInferCutlassNvFp4 GEMM kernel), and CUDA graphs capture cleanly
#   on sm_121 — do NOT add --enforce-eager. (The Nemotron-3-Nano NVFP4 graph-capture
#   crash, NVIDIA-NeMo/Nemotron#125, does not reproduce for this embedder.)
#
# Prompt format (mean-pooled, prefix-based): queries -> "query: ...",
# passages/documents -> "passage: ..." (aka "document: ...").
#
# Pre-req: model downloaded into HF_CACHE first, e.g.
#   HF_HUB_ENABLE_HF_TRANSFER=1 uvx --from huggingface_hub hf download nvidia/Nemotron-3-Embed-1B-NVFP4
set -uo pipefail

MODEL=${1:-"nvidia/Nemotron-3-Embed-1B-NVFP4"}
PORT=${2:-8100}

VLLM_IMAGE=${VLLM_IMAGE:-"nvcr.io/nvidia/vllm:26.06-py3"}
HF_CACHE=${HF_CACHE:-"$HOME/.cache/huggingface"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}       # model supports 32k; cap KV footprint
MEM_UTIL=${MEM_UTIL:-0.15}                  # ~1GB weights; the embedder is tiny
SERVED_NAME=${SERVED_NAME:-"nemotron-embed"}
POOL_ARCH=${POOL_ARCH:-"Ministral3ForCausalLM"}  # registered arch to remap onto
CONTAINER=${CONTAINER:-"nemembed"}

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found"; exit 1; }
docker info >/dev/null 2>&1 || { echo "ERROR: cannot talk to the Docker daemon (docker group?)"; exit 1; }
MODEL_CACHE="$HF_CACHE/hub/models--$(echo "$MODEL" | sed 's|/|--|g')"
[[ -d "$MODEL_CACHE" ]] || {
  echo "ERROR: model not cached at $MODEL_CACHE"
  echo "Download first: HF_HUB_ENABLE_HF_TRANSFER=1 uvx --from huggingface_hub hf download $MODEL"
  exit 1
}
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$PORT "; then
  echo "ERROR: port $PORT already in use (find it: ss -ltnp | grep :$PORT)."; exit 1
fi

echo "=================================================="
echo "Host:    DGX Spark (GB10 / sm_121 / aarch64)"
echo "Image:   $VLLM_IMAGE"
echo "Model:   $MODEL  (NVFP4 embedder)"
echo "Served:  $SERVED_NAME   -> POST http://localhost:$PORT/v1/embeddings"
echo "Pooling: MEAN + L2 normalize   Arch remap: $POOL_ARCH"
echo "Context: max_model_len=$MAX_MODEL_LEN   gpu_mem_util=$MEM_UTIL"
echo "=================================================="

docker rm -f "$CONTAINER" 2>/dev/null || true
exec docker run --rm --name "$CONTAINER" --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "$PORT:$PORT" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=1 \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL" \
    --runner pooling \
    --hf-overrides "{\"architectures\":[\"$POOL_ARCH\"]}" \
    --pooler-config '{"pooling_type":"MEAN"}' \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --host 0.0.0.0 \
    --port "$PORT"
