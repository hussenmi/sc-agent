#!/bin/bash
# Download ONLY one quant from a GGUF HuggingFace repo into the shared HF cache.
#
# Unsloth-style GGUF repos hold every quant (IQ1 ... BF16 = multiple TB). Plain
# download_model.sh would snapshot the whole thing. This grabs just the shards
# matching the chosen quant, so start_llamacpp.sh can find them.
#
# Usage:
#   bash download_gguf.sh <repo_id> <quant>
#   bash download_gguf.sh unsloth/GLM-5.2-GGUF UD-IQ4_XS
#   bash download_gguf.sh unsloth/GLM-5.2-GGUF UD-Q4_K_XL

MODEL=${1:?"Usage: bash download_gguf.sh <hf_repo_id> <quant>   e.g. unsloth/GLM-5.2-GGUF UD-IQ4_XS"}
QUANT=${2:?"Usage: bash download_gguf.sh <hf_repo_id> <quant>   e.g. unsloth/GLM-5.2-GGUF UD-IQ4_XS"}

HF_DIR="/data1/peerd/ibrahih3/hf"
PYTHON="/usersoftware/peerd/ibrahih3/envs/scagent/bin/python3"

if ! "$PYTHON" -c "import huggingface_hub" 2>/dev/null; then
  echo "ERROR: huggingface_hub not found in scagent env."
  echo "Fix with:  uv pip install huggingface_hub hf_xet --python $PYTHON"
  exit 1
fi

echo "Downloading: $MODEL  (quant: $QUANT only)"
echo "Destination: $HF_DIR/hub/"
echo ""

HF_HOME="$HF_DIR" "$PYTHON" - "$MODEL" "$QUANT" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

repo, quant = sys.argv[1], sys.argv[2]
# Match the quant whether it lives in a subfolder or as flat shard filenames.
patterns = [f"*{quant}*"]
print(f"Starting download of {repo} matching {patterns}...")
path = snapshot_download(
    repo_id=repo,
    cache_dir=f"{__import__('os').environ['HF_HOME']}/hub",
    allow_patterns=patterns,
)
print(f"\nDone! Snapshot at: {path}")

import glob, os
shards = sorted(glob.glob(os.path.join(path, "**", f"*{quant}*.gguf"), recursive=True))
if not shards:
    print(f"\nWARNING: no *.gguf files matched '{quant}'. Check the quant name on the repo page.")
    sys.exit(1)
total = sum(os.path.getsize(s) for s in shards)
print(f"Got {len(shards)} shard(s), {total / 1024**3:.1f} GB total.")
print(f"First shard: {shards[0]}")
PYEOF
