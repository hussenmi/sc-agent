# scagent code-execution sandbox image for OpenShell (CPU-only).
#
# Build (on the OpenShell host's Docker daemon):
#     docker build -t scagent-sbx:cpu docker/scagent-sandbox
#
# scagent points at this via SCAGENT_SANDBOX_IMAGE (default: scagent-sbx:cpu)
# and creates a fresh per-session sandbox from it. Build FROM the OpenShell
# community base so the sandbox supervisor / unprivileged `sandbox` user /
# uv-managed venv wiring stays intact; we only add the scientific stack that
# run_code needs.
#
# This CPU subset covers the vast majority of run_code (custom scanpy / pandas
# ops). Torch / RAPIDS / scimilarity live in scagent's specialized tools, not in
# run_code, so they are deliberately omitted here. A "fat" image (and whether the
# sandbox network policy needs a PyPI carve-out for install_package) is a
# documented follow-up in docs/openshell_integration.md.
FROM ghcr.io/nvidia/openshell-community/sandboxes/base:latest

# Install into the base's existing uv venv (VIRTUAL_ENV=/sandbox/.venv). Wheels
# resolve cleanly for py3.14/aarch64 — no compilation (validated in Spike 2).
RUN uv pip install --python /sandbox/.venv/bin/python \
        scanpy \
        anndata \
        leidenalg \
        igraph \
    && /sandbox/.venv/bin/python -c "import scanpy, anndata, leidenalg, igraph; import importlib.metadata as m; print('scanpy', m.version('scanpy'), 'anndata', m.version('anndata'))"
