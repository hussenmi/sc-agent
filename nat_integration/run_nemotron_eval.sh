#!/usr/bin/env bash
# Launch the standardized n=4 Nemotron-3-Ultra NAT eval (pre-fix baseline).
# Run on the driver node (iscn011): GPU1 for scagent/scimilarity, GPU0 = VLM NIM.
set -euo pipefail
cd /data1/peerd/ibrahih3/cs_agent
export WANDB_API_KEY="$(grep ^WANDB_API_KEY= .env | cut -d= -f2- | tr -d '"')"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
echo "cwd=$(pwd)  wandb_key_len=${#WANDB_API_KEY}  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exec /usersoftware/peerd/ibrahih3/envs/nvidia-nat/bin/nat eval \
  --config_file nat_integration/configs/luca_eval_nemotron.yml --reps "${REPS:-4}"
