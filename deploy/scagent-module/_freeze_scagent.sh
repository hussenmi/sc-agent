# Shared helper: build a fresh scagent wheel from $REPO and force-reinstall it into
# a target env prefix. Sourced by install_pixi_module.sh and update_pixi_code.sh.
#
# Why a wheel instead of letting pixi rebuild the path dep: pixi/uv caches the built
# scagent wheel by version, so a code change without a version bump is NOT picked up
# by `pixi install` (it sees the lock satisfied and no-ops). Building + force-
# reinstalling guarantees the deployed code matches the repo. Same approach (and the
# same .gitignore completeness guard) as the old conda update_scagent_code.sh.
#
# Requires: $REPO set; network/PyPI access (build isolation fetches the hatchling
# backend); git available (for the completeness check).

freeze_scagent() {
    local env_prefix="$1"
    local py="$env_prefix/bin/python"
    local wd; wd="$(mktemp -d)"

    echo "   building scagent wheel from $REPO"
    "$py" -m pip wheel --no-deps --no-cache-dir -w "$wd" "$REPO" >"$wd/build.log" 2>&1 || {
        echo "   ERROR: wheel build failed (see $wd/build.log; needs PyPI access for hatchling)"; cat "$wd/build.log" | tail -20; rm -rf "$wd"; return 1; }
    local whl; whl="$(ls "$wd"/scagent-*.whl 2>/dev/null | head -1)"
    [ -n "$whl" ] || { echo "   ERROR: no wheel produced"; rm -rf "$wd"; return 1; }

    # Guard the .gitignore packaging trap: every tracked scagent/ file must be in
    # the wheel (a broad ignore pattern once silently dropped run_manager.py).
    local miss
    miss="$(cd "$REPO" && "$py" - "$whl" <<'PY'
import sys, zipfile, subprocess
inw = set(zipfile.ZipFile(sys.argv[1]).namelist())
tracked = subprocess.check_output(["git", "ls-files", "scagent/"]).decode().split()
print("\n".join(f for f in tracked if f not in inw))
PY
)"
    if [ -n "$miss" ]; then
        echo "   ERROR: these tracked files are missing from the wheel (gitignore pattern?):"
        echo "$miss" | sed 's/^/        /'; rm -rf "$wd"; return 1
    fi

    echo "   force-reinstalling $(basename "$whl") into the env"
    "$py" -m pip install --no-deps --no-index --force-reinstall "$whl" >/dev/null
    rm -rf "$wd"
    echo "   scagent frozen: $("$env_prefix/bin/scagent" --version)"
}
