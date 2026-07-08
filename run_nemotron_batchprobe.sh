#!/usr/bin/env bash
# Probe: does Nemotron respect the batch-effect floor when NOT told to?
# Generic "analyze" instruction (no batch hint) on the multi-donor Reyfman dataset.
set -euo pipefail
cd /data1/peerd/ibrahih3/cs_agent
export CUDA_VISIBLE_DEVICES=1            # iscn011 GPU1 (GPU0 = VLM NIM)
export SCAGENT_MODEL=Nemotron-3-Ultra
export SCAGENT_BASE_URL=http://iscp001:8000/v1
export SCAGENT_PROVIDER=openai
echo "host=$(hostname) cwd=$(pwd) model=$SCAGENT_MODEL base=$SCAGENT_BASE_URL gpu=$CUDA_VISIBLE_DEVICES"
exec /usersoftware/peerd/ibrahih3/envs/scagent/bin/scagent analyze \
  "analyze this dataset thoroughly and report" \
  --data /data1/peerd/ibrahih3/cs_agent/test_data/salcher/Reyfman_all_raw.h5ad \
  --output /data1/peerd/ibrahih3/cs_agent/nat_runs \
  --name nemotron_batchprobe \
  --single-run --quiet --smart \
  --provider openai --model Nemotron-3-Ultra \
  --max-iterations 75
