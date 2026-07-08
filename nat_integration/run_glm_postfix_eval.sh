#!/usr/bin/env bash
# Post-fix 2a validation: GLM-5.2 LuCA eval, reps=4, on the fixed harness.
# Compare completion rate vs pre-fix (1/4). Driver GPU for scimilarity via
# CUDA_VISIBLE_DEVICES; LLM = GLM @ iscp001; vision = NIM @ iscn011.
set -euo pipefail
cd /data1/peerd/ibrahih3/cs_agent
export WANDB_API_KEY="$(grep ^WANDB_API_KEY= .env | cut -d= -f2- | tr -d '"')"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "host=$(hostname) cwd=$(pwd) gpu=$CUDA_VISIBLE_DEVICES wandb_key_len=${#WANDB_API_KEY}"
exec /usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/nat eval \
  --config_file nat_integration/configs/luca_eval_glm_postfix.yml --reps "${REPS:-4}"
