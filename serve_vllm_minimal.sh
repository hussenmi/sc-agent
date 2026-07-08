#!/usr/bin/env bash
# vLLM server for an OpenAI-compatible API for one model on a GPU.
# Requires vLLM available on PATH (e.g. `pip install vllm`, or run inside a
# vLLM container). You can adjust the launch wrapper to your environment as needed.
#
# Usage:
#   bash serve_vllm_minimal.sh                                    # default model, 1 GPU
#   bash serve_vllm_minimal.sh Qwen/Qwen2.5-32B-Instruct          # any HF model
#   GPUS=2 bash serve_vllm_minimal.sh <model>                     # multi-GPU (tensor parallel)
#   HF_TOKEN=hf_xxx bash serve_vllm_minimal.sh meta-llama/...     # gated models
#
# Test it once it prints "Application startup complete":
#   curl http://localhost:8000/v1/models
#   curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
#     -d '{"model":"MODEL_ID","messages":[{"role":"user","content":"Say hi"}]}'
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"   # any Hugging Face repo id
PORT="${2:-8000}"                         # OpenAI API at http://localhost:PORT/v1
GPUS="${GPUS:-1}"                         # number of GPUs = tensor-parallel size

ARGS=(
  --tensor-parallel-size "$GPUS"
  --gpu-memory-utilization 0.90
  --port "$PORT"
)
# For tool-calling agents, uncomment (the parser must match the model family):
# ARGS+=(--enable-auto-tool-choice --tool-call-parser hermes)

echo "Serving $MODEL on $GPUS GPU(s) -> http://localhost:$PORT/v1"
vllm serve "$MODEL" "${ARGS[@]}"
