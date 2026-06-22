#!/bin/bash
# =============================================================================
# Deploy scagent as an Lmod module on Iris  (pixi / RAPIDS variant)
# =============================================================================
# Replaces the old conda-pack build (install_scagent_module.sh) with a
# lockfile-reproducible pixi env. Builds the env directly in the shared Modules
# tree from the committed pixi.toml + pixi.lock, freezes scagent into it
# (non-editable, so the module does NOT track your working tree), places the
# shared .env, locks permissions, and installs the modulefile.
#
# Why pixi: a real conda solve, so NO LD_PRELOAD hack is needed (cupy/cuml/
# rapids_singlecell + torch + scvi-tools coexist), and the build is one
# `pixi install` from a lockfile instead of an opaque conda-pack blob.
#
# Run ON A GPU NODE, as a member of grp_hpc_collab002. Needs network (conda +
# PyPI) unless the pixi cache is already warm.
#
#   bash install_pixi_module.sh --dry-run   # validate + print; write nothing
#   bash install_pixi_module.sh             # build + deploy
# =============================================================================
set -euo pipefail

VERSION="0.1.0"
BASE="/usersoftware/collab002/sail/tools/Modules"
BUILD_DIR="${BASE}/lib/scagent-${VERSION}-pixi"
ENV_PREFIX="${BUILD_DIR}/.pixi/envs/default"
REPO="/data1/peerd/ibrahih3/cs_agent"
GROUP="grp_hpc_collab002"
PIXI="/usersoftware/peerd/ibrahih3/.pixi/bin/pixi"
export PIXI_CACHE_DIR="/usersoftware/peerd/ibrahih3/.pixi/cache"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_freeze_scagent.sh"
SRC_MANIFEST="${HERE}/pixi/pixi.toml"
SRC_LOCK="${HERE}/pixi/pixi.lock"
OLD_CONDA_PREFIX="${BASE}/lib/scagent-${VERSION}"   # the conda-pack env (kept until verified)

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1
run() { if [ "$DRY_RUN" = 1 ]; then printf '   [dry-run] %s\n' "$*"; else "$@"; fi; }

[ "$DRY_RUN" = 1 ] && echo "*** DRY RUN — read-only checks run for real; nothing is written ***"

echo ">> Preflight (read-only)"
fail=0
[ -x "$PIXI" ] && echo "   ok   pixi present ($("$PIXI" --version))" || { echo "   FAIL pixi not at $PIXI"; fail=1; }
smi_out="$(nvidia-smi 2>&1 || true)"
if [[ "$smi_out" =~ CUDA\ Version:[[:space:]]+(1[23]\.[0-9]+) ]]; then
    echo "   ok   GPU + CUDA present (CUDA Version: ${BASH_REMATCH[1]})"
else
    echo "   FAIL no GPU / CUDA 12.x|13.x — run on a GPU node"; fail=1
fi
id -nG | tr ' ' '\n' | grep -qx "$GROUP" && echo "   ok   you are in $GROUP" || { echo "   FAIL not in $GROUP"; fail=1; }
[ -f "$SRC_MANIFEST" ] && echo "   ok   manifest present: $SRC_MANIFEST" || { echo "   FAIL missing $SRC_MANIFEST"; fail=1; }
[ -f "$SRC_LOCK" ] && echo "   ok   lock present: $SRC_LOCK" || { echo "   FAIL missing $SRC_LOCK"; fail=1; }
[ -d "$REPO" ] && echo "   ok   repo present: $REPO" || { echo "   FAIL missing repo $REPO"; fail=1; }
[ -f "$HERE/modulefile/${VERSION}-pixi" ] && echo "   ok   modulefile draft present" || { echo "   FAIL missing modulefile/${VERSION}-pixi"; fail=1; }
if [ -e "$BUILD_DIR" ] && [ "$DRY_RUN" != 1 ]; then echo "   WARN $BUILD_DIR exists — will be reused/overwritten by pixi"; fi
avail=$(df -BG --output=avail "$BASE" 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${avail:-0}" -ge 20 ] && echo "   ok   ${avail}G free on Modules FS (need ~17G)" || echo "   WARN only ${avail:-?}G free (~17G needed)"
[ "$fail" != 0 ] && { echo ">> Preflight FAILED — aborting."; exit 1; }
echo ">> Preflight passed."

echo ">> 1/5  Staging build dir + frozen deploy manifest"
run mkdir -p "$BUILD_DIR"
# Derive the deploy manifest from the committed dev manifest: rewrite the scagent
# dep from an editable relative path to a NON-editable absolute path, so the env
# is a frozen snapshot of the repo (does not track the working tree afterwards).
if [ "$DRY_RUN" = 1 ]; then
    echo "   [dry-run] rewrite scagent dep -> { path = \"$REPO\", editable = false } into $BUILD_DIR/pixi.toml"
    echo "   [dry-run] cp $SRC_LOCK $BUILD_DIR/pixi.lock"
else
    chmod -R u+w "$BUILD_DIR" 2>/dev/null || true
    REPO_ESC=$(printf '%s' "$REPO" | sed 's/[&|\\]/\\&/g')
    sed -E "s|^scagent = \{ path = \"\.\./\.\./\.\.\".*|scagent = { path = \"$REPO_ESC\", editable = false, extras = [\"all\"] }|" \
        "$SRC_MANIFEST" > "$BUILD_DIR/pixi.toml"
    grep -q "editable = false" "$BUILD_DIR/pixi.toml" || { echo "   ERROR: manifest rewrite failed (scagent line not transformed)"; exit 1; }
    cp "$SRC_LOCK" "$BUILD_DIR/pixi.lock"
    echo "   wrote frozen $BUILD_DIR/pixi.toml"
fi

echo ">> 2/5  Building the env (pixi install; ~17G, minutes) + freezing scagent"
# scagent's editable->non-editable change invalidates only its own lock entry, so
# pixi re-locks the pypi side (same versions, different install mode) and reuses
# the conda solve. Lock is updated in-place in the build dir.
run "$PIXI" install --manifest-path "$BUILD_DIR/pixi.toml"
# pixi resolves scagent's DEPENDENCIES from the path dep, but it caches the built
# scagent wheel by version and will serve a STALE copy if the code changed without
# a version bump (verified). So we always overwrite scagent itself with a fresh
# wheel built from the repo -> the deployed code is guaranteed current.
if [ "$DRY_RUN" != 1 ]; then
    freeze_scagent "$ENV_PREFIX"
fi

echo ">> 3/5  Verify the env is frozen (scagent imports from the env, not the repo)"
if [ "$DRY_RUN" != 1 ]; then
    "$ENV_PREFIX/bin/scagent" --version
    # Run from /tmp with PYTHONPATH/VIRTUAL_ENV stripped, mirroring what a clean
    # `module load` gives — otherwise a dev shell that sourced setup*.sh (which
    # exports PYTHONPATH=<repo>) would shadow site-packages and falsely fail this.
    ( cd /tmp && env -u PYTHONPATH -u VIRTUAL_ENV "$ENV_PREFIX/bin/python" - <<PY
import os, scagent
p = os.path.dirname(scagent.__file__)
assert "$ENV_PREFIX" in p and "site-packages" in p, f"NOT frozen: scagent imports from {p}"
print("   frozen OK:", p)
PY
    )
fi

echo ">> 4/5  Shared .env + permissions"
if [ "$DRY_RUN" != 1 ]; then
    if [ -f "$BUILD_DIR/.env" ]; then
        echo "   .env already present — leaving it"
    elif [ -f "$OLD_CONDA_PREFIX/.env" ]; then
        cp "$OLD_CONDA_PREFIX/.env" "$BUILD_DIR/.env"
        echo "   copied shared .env from the conda module"
    else
        cp "$HERE/.env.template" "$BUILD_DIR/.env"
        echo "   !! seeded .env from template — EDIT $BUILD_DIR/.env and set the lab key"
    fi
    chgrp -R "$GROUP" "$BUILD_DIR"
    chmod -R 555 "$BUILD_DIR"
    chmod u+w "$BUILD_DIR/.env"; chmod 640 "$BUILD_DIR/.env"   # group-read only (secret)
else
    echo "   [dry-run] place shared .env, chgrp $GROUP, chmod 555 (env) + 640 (.env)"
fi

echo ">> 5/5  Modulefile + .version (flips the live default to pixi)"
run install -d "${BASE}/modulefiles/scagent"
run cp "$HERE/modulefile/${VERSION}-pixi" "${BASE}/modulefiles/scagent/${VERSION}"
if [ "$DRY_RUN" = 1 ]; then
    echo "   [dry-run] echo $VERSION > ${BASE}/modulefiles/scagent/.version"
else
    echo "$VERSION" > "${BASE}/modulefiles/scagent/.version"
    chgrp "$GROUP" "${BASE}/modulefiles/scagent/${VERSION}" "${BASE}/modulefiles/scagent/.version"
    chmod 644 "${BASE}/modulefiles/scagent/${VERSION}" "${BASE}/modulefiles/scagent/.version"
fi

[ "$DRY_RUN" = 1 ] && { echo; echo "*** DRY RUN complete — nothing written. ***"; exit 0; }
echo
echo "Done. The live 'scagent' module now uses the pixi env."
echo "Verify on a GPU node:  module load scagent && scagent --version"
echo "Once happy, remove the old conda env to reclaim ~15G:"
echo "    chmod -R u+w '$OLD_CONDA_PREFIX' && rm -rf '$OLD_CONDA_PREFIX'"
