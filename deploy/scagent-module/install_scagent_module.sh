#!/bin/bash
# =============================================================================
# Deploy scagent as an Lmod module on Iris  (GPU / RAPIDS variant)
# =============================================================================
# DRAFT — review before running. Run ON A GPU NODE, as a member of
# grp_hpc_collab002, with PyPI access (to build the wheel + fetch conda-pack).
#
# What it does:
#   1. Relocates the scagent_rapids env into the shared Modules lib/ tree with
#      conda-pack (the env is pip-heavy, so `conda --clone` can't reproduce it).
#      Result: a frozen ~15G copy, independent of your personal env.
#   2. Replaces the EDITABLE scagent install with a frozen, non-editable one
#      (the prebuilt wheel), so lab results don't track your working tree.
#   3. Installs the activation shim + shared .env + modulefile + .version.
#   4. Locks down permissions (group-readable install; .env not world-readable).
#
# It does NOT relocate the SCimilarity model or cellbender — those still point
# at Hussen's personal paths (see "Follow-ups" in README.md).
#
# Usage:
#   bash install_scagent_module.sh --dry-run   # validate + print steps, write nothing
#   bash install_scagent_module.sh             # actually deploy
# =============================================================================
set -euo pipefail

VERSION="0.1.0"
BASE="/usersoftware/collab002/sail/tools/Modules"
PREFIX="${BASE}/lib/scagent-${VERSION}"
SRC_ENV="/usersoftware/peerd/ibrahih3/envs/scagent_rapids"
REPO="/data1/peerd/ibrahih3/cs_agent"
GROUP="grp_hpc_collab002"
# Staging dir for the conda-pack tarball + throwaway venv. On /usersoftware
# (59G free); NOT /data1 scratch, which is full. Cleaned up via the EXIT trap.
STAGE="${BASE}/.scagent-stage-$$"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# run: execute a mutating command, or just print it in dry-run mode.
run() {
    if [ "$DRY_RUN" = 1 ]; then
        printf '   [dry-run] %s\n' "$*"
    else
        "$@"
    fi
}

[ "$DRY_RUN" = 1 ] && echo "*** DRY RUN — read-only checks run for real; nothing is written ***"

echo ">> Preflight (read-only; runs for real even in dry-run)"
fail=0
# Capture first, then match with bash regex — avoids the pipefail+grep -q SIGPIPE
# trap (nvidia-smi keeps writing after grep -q closes the pipe -> 141 -> false fail).
smi_out="$(nvidia-smi 2>&1 || true)"
if [[ "$smi_out" =~ CUDA\ Version:[[:space:]]+(1[23]\.[0-9]+) ]]; then
    echo "   ok   GPU + CUDA present (CUDA Version: ${BASH_REMATCH[1]})"
else
    echo "   FAIL no GPU / CUDA 12.x|13.x — guard would block; run on a GPU node"; fail=1
fi
if command -v python >/dev/null && python -c "import venv" 2>/dev/null; then echo "   ok   python + venv present (for the conda-pack venv)"; else echo "   FAIL python venv module missing"; fail=1; fi
if [ -d "$SRC_ENV" ]; then echo "   ok   source env exists ($(du -sh "$SRC_ENV" 2>/dev/null | cut -f1))"; else echo "   FAIL source env missing: $SRC_ENV"; fail=1; fi
if [ -d "$PREFIX" ]; then echo "   FAIL $PREFIX already exists — bump VERSION or remove"; fail=1; else echo "   ok   target prefix is free"; fi
if id -nG | tr ' ' '\n' | grep -qx "$GROUP"; then echo "   ok   you are in $GROUP"; else echo "   FAIL not in $GROUP — can't write the shared tree"; fail=1; fi
avail=$(df -BG --output=avail "$BASE" 2>/dev/null | tail -1 | tr -dc '0-9')
# Peak need ~27G: the ~15G unpacked env + the conda-pack tarball staged alongside.
if [ "${avail:-0}" -ge 30 ]; then echo "   ok   ${avail}G free on Modules FS (need ~27G peak: env + tarball)"; else echo "   WARN only ${avail:-?}G free on Modules FS (~27G peak needed)"; fi
for f in module-activate.sh .env.template scagent/0.1.0 scagent/.version; do
    if [ -f "$HERE/$f" ]; then echo "   ok   draft file present: $f"; else echo "   FAIL missing draft file: $f"; fail=1; fi
done
# Build the wheel NOW (before the expensive clone) so a build/network failure
# aborts early. scagent uses the hatchling backend, which is NOT in the env, so
# the build needs internet (build isolation fetches it) — but installing the
# resulting pure-python wheel later needs no backend and no network. Fail-fast.
WHEEL_DIR="$(mktemp -d)"
trap 'rm -rf "$WHEEL_DIR" "${STAGE:-}"' EXIT
WHEEL=""
if python -m pip wheel --no-deps -w "$WHEEL_DIR" "$REPO" >"$WHEEL_DIR/build.log" 2>&1; then
    WHEEL="$(ls "$WHEEL_DIR"/scagent-*.whl 2>/dev/null | head -1)"
fi
if [ -n "$WHEEL" ]; then echo "   ok   wheel built: $(basename "$WHEEL")"; else echo "   FAIL could not build scagent wheel (see $WHEEL_DIR/build.log; needs PyPI access for hatchling)"; fail=1; fi
if [ "$fail" != 0 ]; then echo ">> Preflight FAILED — aborting."; rm -rf "$WHEEL_DIR"; exit 1; fi
echo ">> Preflight passed."

echo ">> 1/6  Relocating env into the shared tree via conda-pack (~15G; takes a while)"
# Uncompressed tar: this is a local relocate, so skip gzip — much faster pack +
# extract, and we have the disk headroom. conda-pack infers format from extension.
TARBALL="$STAGE/scagent-env.tar"
if [ "$DRY_RUN" = 1 ]; then
    echo "   [dry-run] mkdir -p $STAGE"
    echo "   [dry-run] python -m venv $STAGE/cpack && $STAGE/cpack/bin/pip install conda-pack"
    echo "   [dry-run] $STAGE/cpack/bin/conda-pack -p $SRC_ENV --ignore-editable-packages -o $TARBALL"
    echo "   [dry-run] mkdir -p $PREFIX && tar -xf $TARBALL -C $PREFIX"
    echo "   [dry-run] $PREFIX/bin/conda-unpack   # rewrites baked-in absolute paths"
else
    mkdir -p "$STAGE"
    # Run conda-pack from a throwaway venv so the SOURCE env is never modified.
    python -m venv "$STAGE/cpack"
    "$STAGE/cpack/bin/pip" install -q conda-pack
    # --ignore-editable-packages: don't choke on the editable scagent (we install
    # the frozen wheel in step 2 instead).
    "$STAGE/cpack/bin/conda-pack" -p "$SRC_ENV" --ignore-editable-packages -o "$TARBALL"
    mkdir -p "$PREFIX"
    tar -xf "$TARBALL" -C "$PREFIX"
    "$PREFIX/bin/conda-unpack"
    rm -rf "$STAGE"          # free the tarball + venv now (trap also covers failures)
fi

echo ">> 2/6  Freezing scagent: install the prebuilt wheel non-editable into the clone"
if [ "$DRY_RUN" = 1 ]; then
    echo "   [dry-run] $PREFIX/bin/python -m pip uninstall -y scagent"
    echo "   [dry-run] $PREFIX/bin/python -m pip install --no-deps $(basename "$WHEEL")  (prebuilt wheel: no backend/network)"
    echo "   [dry-run] $PREFIX/bin/scagent --help  (verify CLI)"
else
    # Install the wheel built in preflight. --no-deps: the packed env already has
    # every dependency. No build backend / network needed.
    "$PREFIX/bin/python" -m pip uninstall -y scagent || true
    # Belt-and-suspenders: drop any editable hook conda-pack may have copied, or
    # scagent would import from the repo working tree instead of the frozen copy.
    rm -f "$PREFIX"/lib/python*/site-packages/__editable__*scagent*.pth \
          "$PREFIX"/lib/python*/site-packages/_editable_impl_scagent.pth 2>/dev/null || true
    "$PREFIX/bin/python" -m pip install --no-deps "$WHEEL"
    "$PREFIX/bin/scagent" --help >/dev/null && echo "   scagent CLI OK"
fi

echo ">> 3/6  Installing activation shim"
run cp "$HERE/module-activate.sh" "$PREFIX/module-activate.sh"

echo ">> 4/6  Placing shared .env"
if [ "$DRY_RUN" != 1 ] && [ -f "$PREFIX/.env" ]; then
    echo "   .env already present — leaving it"
else
    run cp "$HERE/.env.template" "$PREFIX/.env"
    echo "   !! EDIT $PREFIX/.env and set the lab API key before users load the module"
fi

echo ">> 5/6  Permissions"
run chgrp -R "$GROUP" "$PREFIX"
run chmod -R 555 "$PREFIX"            # read+execute for user/group/other (per Modules AGENTS.md)
run chmod u+w "$PREFIX/.env"
run chmod 640 "$PREFIX/.env"          # group-read only — holds the secret key

echo ">> 6/6  Modulefile + .version"
run install -d "${BASE}/modulefiles/scagent"
run cp "$HERE/scagent/0.1.0" "${BASE}/modulefiles/scagent/${VERSION}"
if [ "$DRY_RUN" = 1 ]; then
    echo "   [dry-run] echo $VERSION > ${BASE}/modulefiles/scagent/.version"
else
    echo "$VERSION" > "${BASE}/modulefiles/scagent/.version"
fi
run chgrp "$GROUP" "${BASE}/modulefiles/scagent/${VERSION}" "${BASE}/modulefiles/scagent/.version"
run chmod 644 "${BASE}/modulefiles/scagent/${VERSION}" "${BASE}/modulefiles/scagent/.version"

[ "$DRY_RUN" = 1 ] && { echo; echo "*** DRY RUN complete — no changes made. Re-run without --dry-run to deploy. ***"; exit 0; }

echo
echo "Done. Verify on a GPU node:"
echo "    module load scagent && scagent --help"
