#!/bin/bash
# Host a vLLM server as a detached SLURM batch job so it survives your laptop
# disconnecting (unlike `salloc`, whose allocation dies with your login session).
#
# Usage:  bash host_model.sh
#
# Run it on the login node. It submits itself with `sbatch`; the batch job then
# runs the server on a compute node and holds the allocation open. Edit the
# CONFIG block below to change model / port / GPUs / walltime, then just re-run.

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3.6-27B-FP8"   # HF repo id — see the MODEL_TABLE in start_vllm.sh
PORT=8000                       # OpenAI-compatible API port
GPUS=1                          # number of GPUs (tensor-parallel size)

RESUBMIT=true                   # true → launch a fresh job before walltime ends,
                                #        so the endpoint stays up. false → stop.
OVERLAP=false                   # false = STABLE ADDRESS: keep the fixed PORT; a
                                #   resubmitted job WAITS for its predecessor to
                                #   release the port before binding (brief gap at
                                #   each handoff while the new server cold-starts).
                                # true  = zero-downtime, but the port may float up
                                #   on a collision — clients must then discover it
                                #   from logs/vllm_current_host.txt.

# SLURM allocation (matches your `salloc` line)
PARTITION="gpu"
GPU_TYPE="nvidia_h200"
NODELIST="iscp001"              # Pin to one node so host:port stays constant and
                                #   .env's SCAGENT_BASE_URL keeps working. MUST match
                                #   the host in .env. Empty = any node in PARTITION
                                #   (then the host moves — only sensible with OVERLAP
                                #   + client-side discovery, not this stable mode).
WALLTIME="12:00:00"
MEM="120G"
GRACE=300                       # seconds before walltime to trigger the resubmit
# ──────────────────────────────────────────────────────────────────────────────

REPO_DIR="/data1/peerd/ibrahih3/cs_agent"
LOG_DIR="$REPO_DIR/logs"
SCRIPT="$REPO_DIR/host_model.sh"   # canonical path — NOT $0, which is the spooled
                                   # copy when running inside a SLURM job.
mkdir -p "$LOG_DIR"

submit() {
  sbatch \
    --job-name=vllm-serve \
    --partition="$PARTITION" \
    --gres=gpu:"$GPU_TYPE":"$GPUS" \
    ${NODELIST:+--nodelist="$NODELIST"} \
    --time="$WALLTIME" \
    --mem="$MEM" \
    --signal=B:USR1@"$GRACE" \
    --output="$LOG_DIR/vllm_slurm_%j.out" \
    --export=ALL,HOST_MODEL_JOB=1 \
    "$SCRIPT"
}

# Return the first free TCP port at or above $1 (scanning up to +20). Two jobs on
# the same node share the node's network namespace, so a collision here is real
# even though their GPUs are separate. Picking a free port lets a handoff (the
# resubmit overlap) or an unrelated server on the node coexist instead of failing.
find_free_port() {
  local p=$1
  for _ in $(seq 0 20); do
    if ! (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$p "; then
      echo "$p"; return 0
    fi
    p=$((p + 1))
  done
  return 1
}

# Wait until TCP port $1 is free on this node, up to $2 seconds (poll every 5s).
# The no-overlap handoff uses this: on a pinned node a resubmitted job may start
# while its predecessor still holds the fixed port, so it waits for that to exit
# and then binds the same port — keeping the endpoint address constant.
wait_for_port_free() {
  local p=$1 max=${2:-420} waited=0
  while (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ":$p "; do
    if [[ $waited -ge $max ]]; then
      return 1
    fi
    echo "[$(date)] Port $p still held on $(hostname); waiting for predecessor to exit (${waited}s)..."
    sleep 5
    waited=$((waited + 5))
  done
  return 0
}

# Submit unless we ARE the batch job this script sbatch'd. We key off our own
# HOST_MODEL_JOB marker, NOT $SLURM_JOB_ID: the latter is also set when you run
# this from inside an unrelated allocation (e.g. an existing salloc on an A100),
# which would make the script serve on that node instead of requesting an H200.
if [[ -z "$HOST_MODEL_JOB" ]]; then
  echo "Submitting vLLM serving job:"
  echo "  model:    $MODEL"
  echo "  port:     $PORT"
  echo "  gpus:     $GPUS x $GPU_TYPE"
  echo "  walltime: $WALLTIME   resubmit=$RESUBMIT"
  submit
  echo ""
  echo "Watch it come up:"
  echo "  squeue -u \$USER"
  echo "  tail -f $LOG_DIR/vllm_slurm_<jobid>.out    # SLURM + which node it landed on"
  echo "Clients connect to:  http://${NODELIST:-<node>}:$PORT/v1   (node is printed in the .out log)"
  exit 0
fi

# ── Inside the SLURM job: run the server and hold the allocation ──────────────
cd "$REPO_DIR" || exit 1

# Decide the serving port.
if [[ "$OVERLAP" == "true" ]]; then
  # Zero-downtime mode: don't wait — take the next free port if the base is busy.
  # The endpoint may move to PORT+1, so clients must discover it from the host file.
  ACTUAL_PORT=$(find_free_port "$PORT")
  if [[ -z "$ACTUAL_PORT" ]]; then
    echo "[$(date)] No free port in $PORT..$((PORT + 20)) on $(hostname) — cannot serve. Exiting."
    exit 1
  fi
  [[ "$ACTUAL_PORT" != "$PORT" ]] && echo "[$(date)] Port $PORT busy on $(hostname); serving on $ACTUAL_PORT instead."
else
  # Stable-address mode: keep the fixed PORT. If a predecessor still holds it (the
  # handoff overlap on the pinned node), wait for it to exit, then bind the same port.
  ACTUAL_PORT="$PORT"
  if ! wait_for_port_free "$PORT" $((GRACE + 120)); then
    echo "[$(date)] Port $PORT still held on $(hostname) after waiting — another server is squatting it. Exiting."
    exit 1
  fi
fi

echo "=================================================================="
echo "vLLM serving job $SLURM_JOB_ID"
echo "  node:  $(hostname)"
echo "  model: $MODEL   port: $ACTUAL_PORT   gpus: $GPUS"
echo "  clients → http://$(hostname):$ACTUAL_PORT/v1"
echo "=================================================================="
# Record host AND the chosen port so collaborators can find the endpoint even
# when it lands on a non-default port after a handoff.
echo "$(hostname)  port=$ACTUAL_PORT  model=$MODEL  job=$SLURM_JOB_ID" > "$LOG_DIR/vllm_current_host.txt"

# Resubmit a fresh job GRACE seconds before walltime, so the endpoint survives
# the handoff. SLURM sends USR1 to this batch shell (the B: prefix) at that point.
resubmit_done=false
on_timeout() {
  if [[ "$RESUBMIT" == "true" && "$resubmit_done" == "false" ]]; then
    resubmit_done=true
    echo "[$(date)] Walltime approaching — submitting a fresh serving job..."
    submit
  else
    echo "[$(date)] Walltime approaching — resubmit disabled, job will end."
  fi
}
trap on_timeout USR1

# Launch the server. start_vllm.sh backgrounds vLLM and returns 0 once /health
# responds (or non-zero if startup failed).
bash start_vllm.sh "$MODEL" "$ACTUAL_PORT" "$GPUS"
if [[ $? -ne 0 ]]; then
  echo "[$(date)] start_vllm.sh failed to bring the server up — see the vLLM log above. Exiting."
  exit 1
fi

# Hold the allocation open while the server stays healthy. If vLLM dies, the
# health check fails and we let the job end (a resubmit, if enabled, will already
# have been queued near walltime). The trap interrupts sleep to resubmit on time.
echo "[$(date)] Server up — holding allocation. Health-polling every 30s."
while curl -s "http://localhost:$ACTUAL_PORT/health" >/dev/null 2>&1; do
  sleep 30
done

echo "[$(date)] Health check failed — vLLM has exited. Ending job $SLURM_JOB_ID."
