#!/usr/bin/env bash
# Regression: confirm the spine fix does NOT introduce a false batch floor on a
# single-sample dataset (LUNG_T06, the LuCA eval case). Expect: no batch_decision
# obligation, normal completion, annotation finalized.
set -euo pipefail
cd /data1/peerd/ibrahih3/cs_agent
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export SCAGENT_MODEL=Nemotron-3-Ultra
export SCAGENT_BASE_URL=http://iscp001:8000/v1
export SCAGENT_PROVIDER=openai
echo "host=$(hostname) cwd=$(pwd) model=$SCAGENT_MODEL gpu=$CUDA_VISIBLE_DEVICES"
exec /usersoftware/peerd/ibrahih3/envs/scagent/bin/scagent analyze \
  "analyze this dataset thoroughly and report" \
  --data /data1/peerd/ibrahih3/cs_agent/test_data/salcher/LUNG_T06_raw.h5ad \
  --output /data1/peerd/ibrahih3/cs_agent/nat_runs \
  --name nemotron_singlesample_regression \
  --single-run --quiet --smart \
  --provider openai --model Nemotron-3-Ultra \
  --max-iterations 75
