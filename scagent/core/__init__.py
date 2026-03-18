"""Core analysis modules for scagent."""

from .inspector import DataState, inspect_data, recommend_next_steps
from .io import load_data, load_10x_h5, load_h5ad
from .qc import run_qc_pipeline, calculate_qc_metrics, filter_cells_by_mt, filter_genes, detect_doublets
from .normalization import normalize_data, preserve_raw_counts
from .dimred import run_pca, compute_neighbors, compute_umap
from .clustering import run_leiden, run_phenograph

__all__ = [
    # Inspector
    "DataState",
    "inspect_data",
    "recommend_next_steps",
    # IO
    "load_data",
    "load_10x_h5",
    "load_h5ad",
    # QC
    "run_qc_pipeline",
    "calculate_qc_metrics",
    "filter_cells_by_mt",
    "filter_genes",
    "detect_doublets",
    # Normalization
    "normalize_data",
    "preserve_raw_counts",
    # Dimensionality reduction
    "run_pca",
    "compute_neighbors",
    "compute_umap",
    # Clustering
    "run_leiden",
    "run_phenograph",
]
