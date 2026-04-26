#!/bin/bash
# Download any HuggingFace model into the shared HF cache
#
# Usage:
#   bash download_model.sh Qwen/Qwen3-32B
#   bash download_model.sh meta-llama/Llama-3.3-70B-Instruct
#   bash download_model.sh google/gemma-4-27b-it

MODEL=${1:?"Usage: bash download_model.sh <hf_repo_id>  e.g. Qwen/Qwen3-32B"}

HF_DIR="/data1/peerd/ibrahih3/hf"
PYTHON="/usersoftware/peerd/ibrahih3/envs/scagent/bin/python3"

if ! "$PYTHON" -c "import huggingface_hub" 2>/dev/null; then
  echo "ERROR: huggingface_hub not found in scagent env."
  echo "Fix with:  uv pip install huggingface_hub hf_xet --python $PYTHON"
  exit 1
fi

echo "Downloading: $MODEL"
echo "Destination: $HF_DIR/hub/"
echo ""

HF_HOME="$HF_DIR" "$PYTHON" - <<PYEOF
import sys
from huggingface_hub import snapshot_download

repo = "$MODEL"
print(f"Starting download of {repo}...")
path = snapshot_download(
    repo_id=repo,
    cache_dir="$HF_DIR/hub",
    ignore_patterns=["*.pt", "original/*"],   # skip old-format weights
)
print(f"\nDone! Saved to: {path}")
PYEOF
