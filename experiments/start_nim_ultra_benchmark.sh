#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data1/peerd/ibrahih3/cs_agent
SIF=/data1/peerd/ibrahih3/nim-nemotron3-ultra_2.0.5-variant.sif
PROFILE=15ce6340a8c59a0a69b1d9a47356d402e0f7ada27eedf21e847090bad0aaa4a7
PORT=${1:-8001}

export NIM_CACHE=/data1/peerd/ibrahih3/nim_cache_nemotron3_ultra
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_CACHE_ROOT=/opt/nim/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/opt/nim/.cache/torchinductor
export TORCH_HOME=/opt/nim/.cache/torch
export TRITON_CACHE_DIR=/opt/nim/.cache/triton
export CUDA_CACHE_PATH=/opt/nim/.cache/cuda
export NIM_PASSTHROUGH_ARGS="--max-model-len 262144 --enable-prefix-caching \
--mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding \
--mamba-cache-philox-rounds 5 --mamba-cache-mode align \
--max-num-batched-tokens 32768 --block-size 64 --max-num-seqs 72 \
--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":1}' \
--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3"

cd "$PROJECT_ROOT"
exec bash start_nim.sh "$SIF" "$PORT" 0,1,2,3,4,5,6,7 "$PROFILE"
