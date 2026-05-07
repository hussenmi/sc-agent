"""Annotation modules for scagent."""

from .celltypist import (
    available_celltypist_models_for_organism,
    celltypist_models_description,
    infer_celltypist_model_organism,
    prepare_for_celltypist,
    run_celltypist,
)
from .scimilarity import run_scimilarity, prepare_for_scimilarity

__all__ = [
    "run_celltypist",
    "prepare_for_celltypist",
    "celltypist_models_description",
    "infer_celltypist_model_organism",
    "available_celltypist_models_for_organism",
    "run_scimilarity",
    "prepare_for_scimilarity",
]
