#!/usr/bin/env bash
# Start a local LLM server using TensorRT-LLM inside a Singularity container.
# The server exposes an OpenAI-compatible API that scagent connects to — the
# TRT-LLM analog of start_vllm.sh, so scagent and experiments/bench_*.py point
# at it unchanged (just a different SCAGENT_BASE_URL).
#
# Backend: PyTorch (NOT the TRT engine-build path). For a brand-new arch like
# Qwen3.6 the engine builder in this RC chokes on mrope; the PyTorch backend
# loads the pre-quantized FP8 checkpoint directly and still uses the optimized
# kernels. This is the path that actually serves a 200 OK today.
#
# Arguments:
#   MODEL  HuggingFace repo ID (default: Qwen/Qwen3.6-27B-FP8)
#   PORT   API port — scagent uses SCAGENT_BASE_URL=http://localhost:PORT/v1
#   GPUS   tensor-parallel size (default: 1 — a 27B FP8 model fits one H200)
#
# Environment overrides:
#   TRTLLM_SIF   path to the TensorRT-LLM .sif
#   MAX_SEQ_LEN  per-request context cap (default 262144 = Qwen3.6 native 256K)
#   MAX_BATCH    max concurrent requests (default 8 — single-user agent)
#   KV_FRACTION  free GPU mem fraction for KV cache (default 0.90)
#   KV_DTYPE     auto|fp8|nvfp4 (default fp8 — matches the vLLM FP8 KV path)
#   THINKING     0=off (default), 1=on. See the note below — TRT-LLM has no
#                server-side enable_thinking flag; 0 only keeps `content` clean
#                via the reasoning parser, it does not stop thinking tokens.
#   SPEC         0=off (default), 1=MTP speculative decoding (k=1, Qwen3.6).
#                Off by default until verified on this RC; it's the biggest
#                perf lever and the most likely thing to misbehave.
#
# Examples:
#   bash start_trtllm.sh Qwen/Qwen3.6-27B-FP8 8001 1
#   SPEC=1 bash start_trtllm.sh Qwen/Qwen3.6-27B-FP8 8001 1
#
# Download a model first with:  bash download_model.sh <model_id>
set -uo pipefail

MODEL=${1:-"Qwen/Qwen3.6-27B-FP8"}
PORT=${2:-8001}
TP=${3:-1}

TRTLLM_SIF=${TRTLLM_SIF:-/tmp/ibrahih3/trtllm/tensorrt-llm-release-1.3.0rc18.sif}
# The container ships WITHOUT ray, but the ray orchestrator needs `import ray`.
# RAY_SITE is a host dir holding a pip-installed ray (2.31 here, matching the
# torch/python in the .sif) that we bind in and prepend to PYTHONPATH.
RAY_SITE=${RAY_SITE:-/tmp/ibrahih3/trtllm/ray_site_231}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-262144}
# Per-iteration prefill budget. The server REJECTS a prompt longer than this
# unless chunked prefill is on (default off). scagent prompts run ~56K (51 tool
# schemas + world state), well over the 8192 default, so we enable chunked
# prefill and use this as the chunk size. Memory-bounded, not a context cap.
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-16384}
MAX_BATCH=${MAX_BATCH:-8}
KV_FRACTION=${KV_FRACTION:-0.90}
KV_DTYPE=${KV_DTYPE:-fp8}
THINKING=${THINKING:-0}
SPEC=${SPEC:-0}

HF_DIR="/data1/peerd/ibrahih3/hf"
# Per-run scratch. The container's MPI singleton init writes a session dir under
# TMPDIR/HOME; the host cwd lives on read-only /data1 inside the container, so we
# MUST point HOME + TMPDIR at a writable bind or import fails before serving:
#   "mkdir ... /data1 ... Read-only file system" -> orte_init / MPI_Init abort.
RUN_ROOT=${TRTLLM_RUN_ROOT:-/tmp/ibrahih3/trtllm}
RUN_HOME="$RUN_ROOT/home"
RUN_TMP="$RUN_ROOT/tmp"
# Logs go to node-local /tmp, not /data1 — the shared weka fs can be full, and a
# log write failing with ENOSPC takes the server down during startup. Weights are
# read-only on /data1, which still works when the fs is full.
LOG_DIR="${TRTLLM_LOG_DIR:-$RUN_ROOT/logs}"
LOG="$LOG_DIR/trtllm_$(hostname -s)_$(echo "$MODEL" | sed 's|/|_|g').log"
mkdir -p "$RUN_HOME" "$RUN_TMP" "$RUN_TMP/hf_home" "$LOG_DIR"
# Wipe any stale Ray session dir before launch. A previous server killed with
# SIGKILL can leave half-torn-down sessions here; the next Ray start then fails
# with "Raylet could not connect to Runtime Env Agent". This is our own scratch,
# so clearing it on each (single-server) launch is safe.
rm -rf "$RUN_TMP/ray" 2>/dev/null || true

# ── Per-model settings ────────────────────────────────────────────────────────
#   HF repo ID | served name | tool_parser | reasoning_parser
# served name MUST match SCAGENT_MODEL in .env (currently "Qwen3.6-27B") or
# scagent's request model field won't resolve and every call 404s.
MODEL_TABLE=(
  "Qwen/Qwen3.6-27B-FP8 | Qwen3.6-27B | qwen3_coder | qwen3"
  "Qwen/Qwen3.6-27B     | Qwen3.6-27B | qwen3_coder | qwen3"
  "Qwen/Qwen3-32B       | Qwen3-32B   | qwen3       | qwen3"
)
SERVED_NAME="$MODEL"; TOOL_PARSER="qwen3"; REASONING_PARSER="qwen3"
for entry in "${MODEL_TABLE[@]}"; do
  IFS='|' read -r repo name tparser rparser <<< "$entry"
  if [[ "$MODEL" == "$(echo "$repo" | xargs)" ]]; then
    SERVED_NAME=$(echo "$name"   | xargs)
    TOOL_PARSER=$(echo "$tparser" | xargs)
    REASONING_PARSER=$(echo "$rparser" | xargs)
    break
  fi
done

# ── GPU check ─────────────────────────────────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
[[ -n "$GPU_NAME" ]] || { echo "ERROR: no GPU / nvidia-smi"; exit 1; }
# On shared nodes, pin to a specific GPU via CUDA_VISIBLE_DEVICES (inherited by
# singularity --nv). If unset, warn and point at the emptiest GPU — landing on a
# contended GPU silently degrades or kills the model worker.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  EMPTIEST=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | xargs)
  echo "WARNING:   CUDA_VISIBLE_DEVICES unset — may grab a busy GPU on a shared node."
  echo "           Emptiest GPU now is $EMPTIEST. Pin with: CUDA_VISIBLE_DEVICES=$EMPTIEST bash $0 $MODEL $PORT $TP"
else
  echo "GPU pin:   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
[[ -f "$TRTLLM_SIF" ]] || { echo "ERROR: SIF not found: $TRTLLM_SIF"; exit 1; }
[[ -d "$RAY_SITE/ray" ]] || { echo "ERROR: ray not found in RAY_SITE=$RAY_SITE (the .sif has no ray; orchestrator_type ray needs it)"; exit 1; }
# Fail fast if the port is taken — otherwise the readiness probe is answered by
# whatever already owns it (a stale TRT-LLM or a vLLM server) and we falsely
# report "ready" while this server never bound the port.
if (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$PORT "; then
  echo "ERROR: port $PORT is already in use (find it: ss -ltnp | grep :$PORT). Free it or pick another PORT."
  exit 1
fi
MODEL_CACHE="$HF_DIR/hub/models--$(echo "$MODEL" | sed 's|/|--|g')"
[[ -d "$MODEL_CACHE" ]] || { echo "ERROR: model not cached: $MODEL_CACHE (run download_model.sh $MODEL)"; exit 1; }

# ── Extra engine options (YAML) ───────────────────────────────────────────────
# CLI flags don't cover KV-block reuse (prefix caching), CUDA graphs, or
# speculative decoding — those go through --extra_llm_api_options. The file must
# live under RUN_TMP (bound to /tmp inside the container) and be referenced by
# its in-container path, or trtllm-serve can't open it.
CONF="$RUN_TMP/trtllm_$(echo "$SERVED_NAME" | sed 's|/|_|g')_serve.yaml"
CONF_INCTR="/tmp/$(basename "$CONF")"
{
  echo "trust_remote_code: true"
  # Ray orchestrator, NOT the default IPC/MPI one. Under Singularity the default
  # executor does MPI_COMM_SELF.Spawn() to launch workers, which fails with
  # MPI_ERR_SPAWN (no orte runtime to spawn into). Ray manages workers itself and
  # is the path that actually serves a 200 OK here.
  echo "orchestrator_type: ray"
  echo "kv_cache_config:"
  echo "  enable_block_reuse: true            # prefix caching"
  echo "  free_gpu_memory_fraction: $KV_FRACTION"
  echo "cuda_graph_config: {}                 # CUDA graphs on (default shapes)"
  if [[ "$SPEC" == "1" ]]; then
    echo "speculative_config:"
    echo "  decoding_type: MTP                 # native Qwen3.6 multi-token prediction"
    echo "  num_nextn_predict_layers: 1"
  fi
} > "$CONF"

echo "=================================================="
echo "SIF:       $TRTLLM_SIF"
echo "Ray:       $RAY_SITE -> /ray_site (PYTHONPATH)"
echo "Model:     $MODEL"
echo "Served:    $SERVED_NAME   (must equal SCAGENT_MODEL)"
echo "GPU:       $GPU_NAME  (TP=$TP)"
echo "Context:   max_seq_len=$MAX_SEQ_LEN  max_num_tokens=$MAX_NUM_TOKENS (chunked prefill)  max_batch=$MAX_BATCH"
echo "KV cache:  dtype=$KV_DTYPE  fraction=$KV_FRACTION  block_reuse=on"
echo "Parsers:   tool=$TOOL_PARSER  reasoning=$REASONING_PARSER"
echo "Spec:      $([[ "$SPEC" == "1" ]] && echo "MTP k=1" || echo "off")"
echo "Thinking:  $([[ "$THINKING" == "1" ]] && echo "on" || echo "off (content kept clean by reasoning parser)")"
echo "Config:    $CONF"
echo "Port:      $PORT"
echo "Log:       $LOG"
echo "=================================================="

# Secrets/config into the container without echoing on the command line.
# Weights resolve from the read-only /data1 hub via HF_HUB_CACHE, but HF_HOME
# must point somewhere WRITABLE — trust_remote_code writes the dynamic module
# cache under $HF_HOME/modules. The host exports HF_HOME=/data1/.../hf (read-only
# inside the container), so override it to a writable path under /tmp.
export SINGULARITYENV_HF_HOME=/tmp/hf_home
export SINGULARITYENV_HF_HUB_CACHE=/hf_cache/hub
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TRANSFORMERS_OFFLINE=1
export SINGULARITYENV_TMPDIR=/tmp
export SINGULARITYENV_OMPI_MCA_orte_tmpdir_base=/tmp
export SINGULARITY_TMPDIR="$RUN_TMP" APPTAINER_TMPDIR="$RUN_TMP"
# Inject the host ray install ahead of the container's site-packages.
export SINGULARITYENV_PYTHONPATH=/ray_site

# Force Ray onto loopback (127.0.0.1) for all daemons. This node is multi-homed
# (two 10.247.x NICs); Ray's default IP probe (socket to 8.8.8.8:53) picks one
# interface, but the runtime_env_agent ends up unreachable on it, so the raylet
# self-terminates ("Raylet could not connect to Runtime Env Agent"). Ray returns
# 127.0.0.1 when ENABLE_RAY_CLUSTER is False; on Linux that defaults True, and
# this internal env var (despite the Windows/OSX name) flips it. We're single
# node TP=1, so loopback is correct and sidesteps the multi-NIC pick entirely.
export SINGULARITYENV_RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0

# Strip SLURM/PMIx env from what the container inherits. If SLURM_* vars are
# present, the .sif's OpenMPI thinks it was srun-launched and tries SLURM PMIx,
# which it wasn't built for -> "Unreachable in pmix3x_client.c" / MPI_Init abort.
# We run single-node via the Ray orchestrator, so none of this is needed.
for _v in $(env | sed -n 's/^\(\(SLURM\|PMI\|PMIX\)_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$_v"; done

singularity exec --nv \
  --home "$RUN_HOME" \
  --bind "$RUN_TMP":/tmp \
  --bind "$RAY_SITE":/ray_site \
  --bind "$HF_DIR":/hf_cache \
  "$TRTLLM_SIF" \
  trtllm-serve serve "$MODEL" \
    --backend pytorch \
    --tp_size "$TP" \
    --served_model_name "$SERVED_NAME" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --max_num_tokens "$MAX_NUM_TOKENS" \
    --enable_chunked_prefill \
    --max_batch_size "$MAX_BATCH" \
    --kv_cache_dtype "$KV_DTYPE" \
    --tool_parser "$TOOL_PARSER" \
    --reasoning_parser "$REASONING_PARSER" \
    --trust_remote_code \
    --extra_llm_api_options "$CONF_INCTR" \
    --host 0.0.0.0 \
    --port "$PORT" \
  > "$LOG" 2>&1 &

PID=$!
echo "PID: $PID"

# Poll for readiness; fail fast if the server exits during startup.
for i in $(seq 1 240); do
  sleep 5
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "trtllm-serve exited during startup — last lines:"; tail -25 "$LOG"
    echo "Full log: $LOG"; exit 1
  fi
  if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1 || \
     curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "Server ready at http://localhost:$PORT/v1"
    echo "  SCAGENT_PROVIDER=openai"
    echo "  SCAGENT_MODEL=$SERVED_NAME"
    echo "  SCAGENT_BASE_URL=http://localhost:$PORT/v1"
    exit 0
  fi
done
echo "Timed out — check log: $LOG"; exit 1
