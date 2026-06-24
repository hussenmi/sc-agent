#!/usr/bin/env bash
# Post-fix Nemotron LuCA eval, reps=4 (completes the 3-model post-fix matrix).
set -euo pipefail
cd /data1/peerd/ibrahih3/cs_agent
export WANDB_API_KEY="$(grep ^WANDB_API_KEY= .env | cut -d= -f2- | tr -d '"')"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES wandb_key_len=${#WANDB_API_KEY}"
exec /usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/nat eval \
  --config_file nat_integration/configs/luca_eval_nemotron_postfix.yml --reps "${REPS:-4}"
