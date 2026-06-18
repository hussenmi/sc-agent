"""
Optional GPU acceleration for scagent's compute steps via rapids_singlecell.

CPU (scanpy) is the default. Set ``SCAGENT_GPU=1`` to route the heavy steps
(PCA, neighbors, UMAP, Leiden, Louvain) through rapids_singlecell on a CUDA
device. If the env var is unset, or rapids_singlecell / a GPU is unavailable,
everything transparently falls back to the existing scanpy code path.

Usage in a compute function::

    from .gpu import on_gpu

    with on_gpu(adata) as gpu:
        if gpu:
            import rapids_singlecell as rsc
            rsc.pp.pca(adata, n_comps=n_comps)
        else:
            sc.tl.pca(adata, n_comps=n_comps)

``on_gpu`` moves the AnnData to the GPU on entry (``convert_all=True`` so
``obsm``/``obsp``/layers travel too, keeping the neighbors→UMAP→Leiden chain
valid) and back to host on exit. It is re-entrant: if the object is already on
the GPU (an outer ``on_gpu`` moved it), the inner context is a no-op transfer,
so a whole pipeline pays a single round-trip.
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from anndata import AnnData

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def _import_rapids_singlecell_quietly():
    """Import rapids_singlecell without letting it hijack the root logger.

    rapids_singlecell (``decoupler_gpu/_helper/_log.py``) runs
    ``logging.basicConfig(level=INFO, format="%(asctime)s | [%(levelname)s] ...")``
    on first import, which attaches an INFO console handler to the *root* logger.
    That makes every library's — and scagent's own — INFO records print to the
    terminal in that timestamped format, cluttering the clean spinner UI. We
    snapshot the root logger and undo any handler/level it adds, so scagent keeps
    its quiet console policy (step detail still flows to the run manifest /
    agent.log). Only handlers rapids_singlecell newly added are removed; any
    pre-existing root handlers are preserved.
    """
    root = logging.getLogger()
    handlers_before = root.handlers[:]
    level_before = root.level
    import rapids_singlecell  # noqa: F401  (triggers the basicConfig side effect)

    for handler in root.handlers[:]:
        if handler not in handlers_before:
            root.removeHandler(handler)
    root.setLevel(level_before)
    return rapids_singlecell


@functools.lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Whether GPU acceleration is enabled and usable.

    True only when ``SCAGENT_GPU`` is truthy *and* rapids_singlecell + cupy
    import *and* at least one CUDA device is present. Result is cached; call
    ``gpu_available.cache_clear()`` after changing the env var (tests do this).
    """
    flag = os.environ.get("SCAGENT_GPU", "").strip().lower()
    if flag not in _TRUTHY:
        return False
    try:
        import cupy  # noqa: F401

        _import_rapids_singlecell_quietly()
        n_devices = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - depends on GPU stack
        logger.warning(
            "SCAGENT_GPU is set but the GPU stack is unavailable (%s); using CPU.",
            exc,
        )
        return False
    if n_devices < 1:  # pragma: no cover - depends on hardware
        logger.warning("SCAGENT_GPU is set but no CUDA device was found; using CPU.")
        return False
    logger.info("GPU acceleration enabled via rapids_singlecell (%d device(s)).", n_devices)
    return True


def is_on_gpu(adata: AnnData) -> bool:
    """True if ``adata.X`` currently lives on the GPU (a cupy array/matrix)."""
    top = type(adata.X).__module__.split(".", 1)[0]
    return top in ("cupy", "cupyx")


@contextmanager
def on_gpu(adata: AnnData) -> Iterator[bool]:
    """Context manager that places ``adata`` on the GPU when enabled.

    Yields True inside a GPU context (rapids_singlecell ops valid), False when
    GPU is disabled/unavailable (caller should use the scanpy path). Re-entrant:
    only the outermost context that performs the transfer moves data back to the
    host on exit, so nested calls share one round-trip.
    """
    if not gpu_available():
        yield False
        return

    from rapids_singlecell.get import anndata_to_CPU, anndata_to_GPU

    moved = not is_on_gpu(adata)
    if moved:
        anndata_to_GPU(adata, convert_all=True)
    try:
        yield True
    finally:
        if moved:
            anndata_to_CPU(adata, convert_all=True)
