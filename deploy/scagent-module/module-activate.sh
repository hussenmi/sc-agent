#!/bin/bash
# =============================================================================
# scagent module activation shim
# =============================================================================
# Sourced (via `source-sh`) by the scagent modulefile to reproduce
# `conda activate <prefix>` WITHOUT requiring conda on the user's PATH.
#
# It exports CONDA_PREFIX and runs the env's etc/conda/activate.d/*.sh hooks
# (libarrow, ucx, gdal, proj4, libglib, libxml2) that set the runtime library
# and data paths RAPIDS / pyarrow / gdal need. The modulefile captures the
# resulting environment delta and reverses it on `module unload`.
#
# Lives at the root of the install prefix; derives the prefix from its own
# location so it has no hard-coded paths.
# =============================================================================

PREFIX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONDA_PREFIX="$PREFIX"

if [ -d "$PREFIX/etc/conda/activate.d" ]; then
    for f in "$PREFIX"/etc/conda/activate.d/*.sh; do
        [ -r "$f" ] && source "$f"
    done
fi
