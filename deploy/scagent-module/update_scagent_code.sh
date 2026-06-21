#!/bin/bash
# =============================================================================
# Push working-dir code changes into the deployed scagent module (in place)
# =============================================================================
# Rebuilds a wheel from the repo and reinstalls it into the frozen module prefix.
# Use this when you've edited scagent PYTHON CODE and want testers to get it on
# their next `scagent` launch — no version bump, no env re-pack.
#
#   * Only updates the scagent package. Does NOT add new dependencies or change
#     the conda env. If your change needs a NEW dep, pip-install it into
#     $PREFIX/bin separately, or re-pack the env (see install_scagent_module.sh).
#   * Leaves the shared .env untouched.
#   * Needs PyPI access (the hatchling build backend isn't in the env).
#
# Run on a GPU node, as the owner of the install (ibrahih3).
# =============================================================================
set -euo pipefail

REPO="/data1/peerd/ibrahih3/cs_agent"
PREFIX="/usersoftware/collab002/sail/tools/Modules/lib/scagent-0.1.0"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

echo ">> Building wheel from $REPO"
python -m pip wheel --no-deps --no-cache-dir -w "$W" "$REPO" >/dev/null
WHL="$(ls "$W"/scagent-*.whl | head -1)"
echo "   built $(basename "$WHL")"

# Guard against the .gitignore packaging trap: confirm every tracked source file
# made it into the wheel before we ship it.
miss="$(cd "$REPO" && python - "$WHL" <<'PY'
import sys, zipfile, subprocess
inw = set(zipfile.ZipFile(sys.argv[1]).namelist())
tracked = subprocess.check_output(["git","ls-files","scagent/"]).decode().split()
print("\n".join(f for f in tracked if f not in inw))
PY
)"
if [ -n "$miss" ]; then
    echo ">> ABORT: these tracked files are missing from the wheel (gitignore pattern?):"
    echo "$miss" | sed 's/^/     /'
    exit 1
fi
echo "   wheel completeness OK"

echo ">> Reinstalling into $PREFIX"
chmod -R u+w "$PREFIX"
"$PREFIX/bin/python" -m pip install --no-deps --no-index --force-reinstall "$WHL" >/dev/null
chmod -R 555 "$PREFIX"
chmod 640 "$PREFIX/.env"

echo ">> Verify"
"$PREFIX/bin/scagent" --version
echo "Done. Testers get the new code on their next \`scagent\` launch (restart any open sessions)."
