"""Batch correction modules for scagent."""

from .bbknn import run_bbknn
from .diffxpy import diffxpy_available, run_two_group_de
from .entropy import compute_batch_entropy
from .harmony import run_harmony
from .scanorama import run_scanorama
from .scib import run_scib_benchmark
from .scvi import run_scvi

__all__ = [
    "run_scanorama",
    "run_harmony",
    "run_scvi",
    "run_bbknn",
    "compute_batch_entropy",
    "run_scib_benchmark",
    "diffxpy_available",
    "run_two_group_de",
]
