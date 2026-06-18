#!/bin/bash
# =============================================================================
# scagent GPU Setup Script for IRIS HPC
# =============================================================================
#
# Usage: source setup_gpu.sh
#
# Activates the GPU (rapids_singlecell) conda environment and enables GPU
# acceleration. This is the counterpart to setup.sh:
#   - setup.sh      -> CPU uv venv (Python 3.10) for dev/tests
#   - setup_gpu.sh  -> GPU conda env "scagent_rapids" (Python 3.14, RAPIDS 26.04)
#
# scagent is installed editable into the conda env, so code changes and branch
# switches take effect with no reinstall. This script only activates + exports;
# it never installs.
# =============================================================================

# Don't use 'set -e' in sourced scripts - it can kill the parent shell!

SCAGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="/data1/peerd/ibrahih3/tools/miniconda3"
GPU_ENV="/usersoftware/peerd/ibrahih3/envs/scagent_rapids"

echo "============================================"
echo "scagent GPU setup for IRIS HPC"
echo "============================================"

# Drop any active uv/venv first — otherwise its bin stays ahead of the conda
# env on PATH and shadows its python. `deactivate` is unreliable in
# non-interactive shells (the function may be undefined), so strip the venv's
# bin from PATH explicitly and clear VIRTUAL_ENV.
if [ -n "${VIRTUAL_ENV}" ]; then
    echo "Removing active venv from PATH: ${VIRTUAL_ENV}"
    deactivate 2>/dev/null || true
    PATH="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -vxF "${VIRTUAL_ENV}/bin" | paste -sd ':' -)"
    export PATH
    unset VIRTUAL_ENV
fi

# Activate the conda GPU env (by absolute path).
if [ ! -d "${GPU_ENV}" ]; then
    echo "ERROR: GPU env not found at ${GPU_ENV}"
    return 1 2>/dev/null || exit 1
fi
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${GPU_ENV}" || {
    echo "ERROR: failed to activate ${GPU_ENV}"
    return 1 2>/dev/null || exit 1
}
# Force the conda env's bin to the front of PATH so its python always wins,
# regardless of any leftover entries from the shell snapshot.
export PATH="${GPU_ENV}/bin:${PATH}"

# Two conda libs must load before the older pip-wheel CUDA/runtime libs that
# torch pulls into the process, or imports fail:
#   - libstdc++.so.6  : numpy needs GLIBCXX_3.4.29; the system /lib64 one is older.
#   - libnvJitLink.so.13 : cuml's libcuvs needs __nvJitLinkComplete_13_2, which the
#     pip nvidia-nvjitlink-cu13 that torch bundles lacks (torch loads it first, so
#     cuml binds to the wrong one -> "undefined symbol").
# Preload the conda copies so they win regardless of import order.
export LD_PRELOAD="${GPU_ENV}/lib/libstdc++.so.6:${GPU_ENV}/lib/libnvJitLink.so.13${LD_PRELOAD:+:${LD_PRELOAD}}"

# Same environment variables as setup.sh.
export SCAGENT_HOME="${SCAGENT_DIR}"
export SCIMILARITY_MODEL_PATH="/data1/peerd/ibrahih3/scimilarity/docs/notebooks/models/model_v1.1"
if [ -x "/usersoftware/peerd/ibrahih3/envs/cellbender/bin/cellbender" ]; then
    export SCAGENT_CELLBENDER="/usersoftware/peerd/ibrahih3/envs/cellbender/bin/cellbender"
fi
export PYTHONPATH="${SCAGENT_DIR}:${PYTHONPATH}"

# Enable GPU acceleration. Override with `export SCAGENT_GPU=0` before sourcing
# to force the CPU code path while still using this env.
export SCAGENT_GPU="${SCAGENT_GPU:-1}"

# =============================================================================
# Verify
# =============================================================================
echo ""
echo "Python: $(which python)"
echo "Version: $(python --version 2>&1)"
python - <<'PY' 2>/dev/null || echo "Note: GPU stack import failed — check the env."
import os, scagent, rapids_singlecell as rsc, cupy
print(f"scagent v{scagent.__version__} | rapids_singlecell {rsc.__version__} | cupy {cupy.__version__}")
print(f"CUDA devices visible: {cupy.cuda.runtime.getDeviceCount()}")
print(f"SCAGENT_GPU={os.environ.get('SCAGENT_GPU')}")
PY

echo ""
echo "Environment variables set:"
echo "  SCAGENT_HOME=${SCAGENT_HOME}"
echo "  SCAGENT_GPU=${SCAGENT_GPU}"
echo "  SCIMILARITY_MODEL_PATH=${SCIMILARITY_MODEL_PATH}"
echo ""
echo "GPU env ready. (Use 'source setup.sh' for the CPU venv / tests.)"
echo ""
