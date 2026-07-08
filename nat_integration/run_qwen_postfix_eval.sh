#!/usr/bin/env bash
# Post-fix agency study: Qwen3.6 LuCA eval, reps=4, fixed harness (agency contract).
# Compare agency proxies + accuracy/completion vs pre-fix (nat_luca_out_qwen).
set -euo pipefail
cd /data1/peerd/ibrahih3/cs_agent
export WANDB_API_KEY="$(grep ^WANDB_API_KEY= .env | cut -d= -f2- | tr -d '"')"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
echo "host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES wandb_key_len=${#WANDB_API_KEY}"
exec /usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/nat eval \
  --config_file nat_integration/configs/luca_eval_qwen_postfix.yml --reps "${REPS:-4}"
