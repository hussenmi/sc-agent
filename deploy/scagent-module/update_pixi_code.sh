#!/bin/bash
# =============================================================================
# Push working-dir code changes into the deployed pixi scagent module (in place)
# =============================================================================
# The fast path for the testing phase: rebuild a wheel from the repo and force-
# reinstall scagent into the live pixi env. No env re-pack, no version bump.
# Testers get the new code on their next `scagent` launch.
#
#   * Updates ONLY the scagent package. Does NOT add deps or change the env.
#     For a new dependency, edit pixi/pixi.toml and re-run install_pixi_module.sh.
#   * Leaves the shared .env untouched.
#   * Needs PyPI access (build isolation fetches the hatchling backend).
#
# Run as the owner of the install (ibrahih3).
# =============================================================================
set -euo pipefail

REPO="/data1/peerd/ibrahih3/cs_agent"
BASE="/usersoftware/collab002/sail/tools/Modules"
BUILD_DIR="${BASE}/lib/scagent-0.1.0-pixi"
ENV_PREFIX="${BUILD_DIR}/.pixi/envs/default"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_freeze_scagent.sh"

[ -d "$ENV_PREFIX" ] || { echo "ERROR: pixi env not found at $ENV_PREFIX — run install_pixi_module.sh first"; exit 1; }

echo ">> Unlocking env for reinstall"
chmod -R u+w "$BUILD_DIR"

echo ">> Rebuilding + freezing scagent from $REPO"
freeze_scagent "$ENV_PREFIX"

echo ">> Re-locking permissions"
chmod -R 555 "$BUILD_DIR"
chmod u+w "$BUILD_DIR/.env" 2>/dev/null && chmod 640 "$BUILD_DIR/.env" || true

echo ">> Verify (from a clean env, like a real \`module load\`)"
( cd /tmp && env -u PYTHONPATH -u VIRTUAL_ENV "$ENV_PREFIX/bin/python" -c \
    "import scagent; print('scagent', scagent.__version__, 'from', scagent.__file__)" )
echo "Done. Testers get the new code on their next \`scagent\` launch (restart open sessions)."
