"""Annotation modules for scagent."""

from .celltypist import run_celltypist, prepare_for_celltypist
from .scimilarity import run_scimilarity, prepare_for_scimilarity

__all__ = [
    "run_celltypist",
    "prepare_for_celltypist",
    "run_scimilarity",
    "prepare_for_scimilarity",
]
