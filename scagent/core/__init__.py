"""Core analysis modules for scagent."""

from .inspector import (
    ClusteringRecord,
    DataState,
    MetadataCandidate,
    MetadataResolution,
    clustering_record_to_dict,
    default_cluster_key_for_method,
    format_resolution_token,
    get_clustering_registry,
    infer_cluster_key,
    inspect_data,
    metadata_candidate_to_dict,
    metadata_resolution_to_dict,
    promote_clustering_to_primary,
    rank_obs_metadata_candidates,
    recommend_next_steps,
    register_clustering,
    resolve_batch_metadata,
)
from .io import load_data, load_10x_h5, load_h5ad
from .qc import run_qc_pipeline, calculate_qc_metrics, filter_cells_by_mt, filter_genes, detect_doublets, run_decontx
from .normalization import normalize_data, preserve_raw_counts
from .dimred import run_pca, compute_neighbors, compute_umap
from .clustering import run_leiden, run_phenograph

__all__ = [
    # Inspector
    "DataState",
    "MetadataCandidate",
    "MetadataResolution",
    "ClusteringRecord",
    "inspect_data",
    "recommend_next_steps",
    "rank_obs_metadata_candidates",
    "resolve_batch_metadata",
    "metadata_candidate_to_dict",
    "metadata_resolution_to_dict",
    "get_clustering_registry",
    "clustering_record_to_dict",
    "default_cluster_key_for_method",
    "format_resolution_token",
    "infer_cluster_key",
    "register_clustering",
    "promote_clustering_to_primary",
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
    "run_decontx",
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
