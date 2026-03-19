#!/bin/bash
# =============================================================================
# scagent Setup Script for IRIS HPC
# =============================================================================
#
# Usage: source setup.sh [--local]
#
# Options:
#   --local    Store venv in project directory instead of /usersoftware
#
# This script follows IRIS HPC conventions:
#   - Environments stored in /usersoftware/peerd/$USER/
#   - Uses uv for fast, reliable package management
#   - Configures paths for scagent and dependencies
# =============================================================================

# Don't use 'set -e' in sourced scripts - it can kill the parent shell!

SCAGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SW_DIR="/usersoftware/peerd/${USER}"
VENV_NAME="scagent"
UV_PATH="/data1/peerd/ibrahih3/tools/miniconda3/bin/uv"

# Parse arguments
USE_LOCAL=false
for arg in "$@"; do
    case $arg in
        --local)
            USE_LOCAL=true
            ;;
    esac
done

# Determine venv location
if [ "$USE_LOCAL" = true ]; then
    VENV_DIR="${SCAGENT_DIR}/.venv"
    echo "Using local venv: ${VENV_DIR}"
else
    VENV_DIR="${USER_SW_DIR}/envs/${VENV_NAME}"
    echo "Using usersoftware venv: ${VENV_DIR}"
fi

echo "============================================"
echo "scagent setup for IRIS HPC"
echo "============================================"
echo "Project: ${SCAGENT_DIR}"
echo "Venv: ${VENV_DIR}"
echo ""

# =============================================================================
# Create /usersoftware directory structure if needed
# =============================================================================
if [ "$USE_LOCAL" = false ]; then
    if [ ! -d "${USER_SW_DIR}" ]; then
        echo "Creating user software directory: ${USER_SW_DIR}"
        mkdir -p "${USER_SW_DIR}/envs"
    elif [ ! -d "${USER_SW_DIR}/envs" ]; then
        mkdir -p "${USER_SW_DIR}/envs"
    fi
fi

# =============================================================================
# Ensure uv is available
# =============================================================================
if [ -x "${UV_PATH}" ]; then
    UV_CMD="${UV_PATH}"
elif command -v uv &> /dev/null; then
    UV_CMD="uv"
else
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_CMD="${HOME}/.local/bin/uv"
fi
echo "Using uv: ${UV_CMD}"

# =============================================================================
# Create virtual environment
# =============================================================================
if [ ! -d "${VENV_DIR}" ]; then
    echo ""
    echo "Creating virtual environment..."
    ${UV_CMD} venv "${VENV_DIR}" --python 3.10
fi

# =============================================================================
# Activate and install
# =============================================================================
echo ""
echo "Activating environment..."
source "${VENV_DIR}/bin/activate"

echo "Installing scagent and dependencies..."
cd "${SCAGENT_DIR}"
# Install with agent extra for full functionality
${UV_CMD} pip install -e ".[agent]" --quiet

# =============================================================================
# Set environment variables
# =============================================================================
export SCAGENT_HOME="${SCAGENT_DIR}"
export SCIMILARITY_MODEL_PATH="/data1/peerd/ibrahih3/scimilarity/docs/notebooks/models/model_v1.1"
export PYTHONPATH="${SCAGENT_DIR}:${PYTHONPATH}"

# =============================================================================
# Load SAIL modules if available
# =============================================================================
if command -v module &> /dev/null; then
    # Add SAIL modulefiles if available
    if [ -d "/usersoftware/collab002/sail/tools/Modules/modulefiles" ]; then
        module use --append /usersoftware/collab002/sail/tools/Modules/modulefiles 2>/dev/null || true
    fi
fi

# =============================================================================
# Verify installation
# =============================================================================
echo ""
echo "============================================"
echo "scagent environment ready!"
echo "============================================"
echo ""
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""

# Test import
python -c "import scagent; print(f'scagent v{scagent.__version__} loaded successfully')" 2>/dev/null || {
    echo "Note: If import fails, try: source setup.sh --local"
}

echo ""
echo "Quick start:"
echo "  from scagent.core import load_data, inspect_data"
echo "  from scagent.agent import SCAgent"
echo ""
echo "Environment variables set:"
echo "  SCAGENT_HOME=${SCAGENT_HOME}"
echo "  SCIMILARITY_MODEL_PATH=${SCIMILARITY_MODEL_PATH}"
echo ""
