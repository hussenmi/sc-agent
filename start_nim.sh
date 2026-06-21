#!/usr/bin/env bash
# Start an NVIDIA NIM (TensorRT-LLM backend) under Singularity, exposing an
# OpenAI-compatible API. This is the NIM analog of start_vllm.sh — same
# OpenAI /v1 contract, so scagent and experiments/bench_*.py point at it
# unchanged (just a different --pair / SCAGENT_BASE_URL).
#
# Why the env-var dance: NIM ships as a Docker image; we pull it to a .sif via
#   singularity pull docker://nvcr.io/nim/<...>
# Singularity uses HOST networking (no docker-style -p mapping), so the only way
# to avoid colliding with a vLLM server already on :8000 is to move NIM's own
# listen port with NIM_HTTP_API_PORT. The NGC key (model download auth) and GPU
# pinning are passed via SINGULARITYENV_* so they reach the container without
# being echoed on the command line.
#
# Usage:
#   bash start_nim.sh [SIF] [PORT] [GPU]
# Examples:
#   bash start_nim.sh /data1/peerd/ibrahih3/nim-qwen3.6-27b_variant.sif 8002 1
#
# Env overrides:
#   NGC_API_KEY_FILE  file holding the NGC key   (default ~/.ngc_api_key)
#   NIM_CACHE         host dir bound to the NIM model/engine cache
#                     (default /data1/peerd/ibrahih3/nim_cache)
set -euo pipefail

SIF=${1:-/data1/peerd/ibrahih3/nim-qwen3.6-27b_variant.sif}
PORT=${2:-8002}
GPU=${3:-1}
NGC_API_KEY_FILE=${NGC_API_KEY_FILE:-$HOME/.ngc_api_key}
NIM_CACHE=${NIM_CACHE:-/data1/peerd/ibrahih3/nim_cache}

[[ -f "$SIF" ]] || { echo "ERROR: SIF not found: $SIF"; exit 1; }
KEY=$(cat "$NGC_API_KEY_FILE" 2>/dev/null || true)
[[ -n "$KEY" ]] || { echo "ERROR: NGC key not found at $NGC_API_KEY_FILE"; exit 1; }
mkdir -p "$NIM_CACHE" "$NIM_CACHE/tmp"

echo "=================================================="
echo "NIM SIF:   $SIF"
echo "Port:      $PORT      (OpenAI API -> http://localhost:$PORT/v1)"
echo "GPU:       $GPU"
echo "Cache:     $NIM_CACHE  ->  /opt/nim/.cache"
echo "=================================================="
echo "Once ready (model profile selected + engine built/downloaded), point scagent at it:"
echo "  SCAGENT_PROVIDER=openai"
echo "  SCAGENT_BASE_URL=http://localhost:$PORT/v1"
echo "  SCAGENT_MODEL=\$(curl -s http://localhost:$PORT/v1/models | python3 -c 'import sys,json;print(json.load(sys.stdin)[\"data\"][0][\"id\"])')"
echo "=================================================="

# Pass secrets/config into the container without printing them.
export SINGULARITYENV_NGC_API_KEY="$KEY"
export SINGULARITYENV_NIM_HTTP_API_PORT="$PORT"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU"
# VLM NIMs ship a NIM_MEDIA_IO_KWARGS default whose JSON quotes get mangled into
# literal \" when the NIM SDK assembles the vLLM subprocess via shell=True, so the
# backend's json.loads rejects it and the server dies at startup (a NIM-under-
# Singularity quoting bug; works under `docker run`). An empty object has no quotes
# or spaces to mis-escape, parses cleanly, and is fine because we only send images,
# never video. Harmless for text-only NIMs, which ignore it.
export SINGULARITYENV_NIM_MEDIA_IO_KWARGS='{}'
# NIM builds a workspace VFS under TMPDIR. By default the container inherits the
# host TMPDIR (e.g. /data1/.../tmp), which is NOT bound into the container, so it
# can't be created -> "Error constructing Workspace VFS ... No such file or
# directory". Pin it inside the bound, writable cache instead.
export SINGULARITYENV_TMPDIR=/opt/nim/.cache/tmp

# --no-eval: Singularity's default `run` re-evaluates the container's
# ENTRYPOINT/CMD/ARGS through a shell, which mangles quoting — VLM NIMs pass a
# JSON --media-io-kwargs that gets corrupted into literal {\"video\":...} and the
# vLLM backend's json.loads then rejects it, killing startup. --no-eval uses
# OCI-compatible behavior that preserves quoting (per the SIF's own runscript).
# --writable-tmpfs: writable overlay for scratch paths outside the bound cache.
exec singularity run --no-eval --nv \
  --writable-tmpfs \
  --bind "$NIM_CACHE":/opt/nim/.cache \
  "$SIF"
