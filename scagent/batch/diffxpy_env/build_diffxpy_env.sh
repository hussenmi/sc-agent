#!/usr/bin/env bash
# Build the isolated diffxpy environment for scagent.
#
# diffxpy (batchglm + TensorFlow) is a frozen 2020-era stack that conflicts with
# scagent's modern scanpy / anndata / RAPIDS deps, so it lives in its OWN conda
# env and is driven across a process boundary (see scagent/batch/diffxpy.py) —
# the same isolation model as CellBender. This script creates that env with the
# exact, verified pins in requirements.txt.
#
# Usage:
#   bash scagent/batch/diffxpy_env/build_diffxpy_env.sh [ENV_PREFIX]
#
# ENV_PREFIX defaults to /usersoftware/peerd/$USER/envs/scagent_diffxpy, which is
# where setup.sh / setup_gpu.sh look. After building, those scripts export
# SCAGENT_DIFFXPY automatically; or point it there by hand:
#   export SCAGENT_DIFFXPY="$ENV_PREFIX/bin/python"
#
# NOTE: the pip install MUST run in a clean environment. If the calling shell has
# another conda env active (e.g. RAPIDS), its PYTHONHOME/PYTHONPATH leak into the
# py3.9 interpreter and corrupt the install — so we invoke pip via `env -i`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${HERE}/requirements.txt"
ENV_PREFIX="${1:-/usersoftware/peerd/${USER}/envs/scagent_diffxpy}"

# Locate a conda executable.
CONDA_BIN="${CONDA_EXE:-}"
if [ -z "${CONDA_BIN}" ] || [ ! -x "${CONDA_BIN}" ]; then
    CONDA_BIN="$(command -v conda || true)"
fi
if [ -z "${CONDA_BIN}" ]; then
    echo "ERROR: could not find a conda executable (set CONDA_EXE)." >&2
    exit 1
fi

echo "Building diffxpy env at: ${ENV_PREFIX}"
"${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.9 pip

PY="${ENV_PREFIX}/bin/python"
if [ ! -x "${PY}" ]; then
    echo "ERROR: ${PY} was not created." >&2
    exit 1
fi

echo "Installing the verified diffxpy stack (clean env -i to avoid leakage)..."
env -i HOME="${HOME}" PATH="${ENV_PREFIX}/bin:/usr/bin:/bin" \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_INPUT=1 \
    "${PY}" -m pip install -r "${REQ}"

echo "Verifying imports..."
env -i HOME="${HOME}" PATH="${ENV_PREFIX}/bin:/usr/bin:/bin" \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES="" \
    "${PY}" -c "import numpy, scipy, pandas, anndata, diffxpy.api, tensorflow; print('diffxpy env OK')"

echo ""
echo "Done. Point scagent at it with:"
echo "  export SCAGENT_DIFFXPY=${PY}"
