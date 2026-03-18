"""
Lab's standard parameters for single-cell analysis.

These defaults are derived from the SCALE workshop notebooks (sessions 1-6)
and represent best practices for single-cell RNA-seq analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass(frozen=True)
class QCDefaults:
    """Quality control default parameters."""

    # Mitochondrial content thresholds
    mt_threshold_cells: float = 25.0  # For dissociated cells
    mt_threshold_nuclei: float = 5.0  # For single nuclei

    # Gene filtering
    min_cells_per_gene: float = np.exp(4)  # ~55 cells

    # Scrublet doublet detection
    scrublet_expected_doublet_rate: float = 0.06
    scrublet_sim_ratio: float = 2.0
    scrublet_n_prin_comps: int = 30
    scrublet_random_state: int = 0

    # Ribosomal gene prefixes (human)
    ribo_prefixes: tuple = ('RPS', 'RPL')

    # Mitochondrial gene prefix (human)
    mt_prefix: str = 'MT-'


@dataclass(frozen=True)
class HVGDefaults:
    """Highly variable gene selection defaults."""

    n_top_genes: int = 4000
    flavor: str = 'seurat_v3'  # Requires raw counts in layer


@dataclass(frozen=True)
class DimRedDefaults:
    """Dimensionality reduction defaults."""

    # PCA
    n_pcs: int = 30

    # Neighbors
    n_neighbors: int = 30
    metric: str = 'euclidean'

    # UMAP
    umap_min_dist: float = 0.1


@dataclass(frozen=True)
class ClusteringDefaults:
    """Clustering defaults."""

    # Leiden
    leiden_resolution: float = 1.0
    leiden_random_state: int = 0

    # PhenoGraph
    phenograph_k: int = 30
    phenograph_clustering_algo: str = 'leiden'
    phenograph_jaccard: bool = True
    phenograph_metric: str = 'euclidean'


@dataclass(frozen=True)
class CellTypistDefaults:
    """CellTypist annotation defaults."""

    # CRITICAL: CellTypist requires specific normalization
    target_sum: int = 10000

    # Default model
    model: str = 'Immune_All_Low.pkl'

    # Majority voting
    majority_voting: bool = True


@dataclass(frozen=True)
class ScimilarityDefaults:
    """Scimilarity annotation defaults."""

    # Normalization (same as CellTypist)
    target_sum: int = 10000

    # Model path placeholder
    model_path: str = ''


@dataclass(frozen=True)
class BatchDefaults:
    """Batch correction defaults."""

    # Scanorama
    scanorama_dimred: int = 30
    scanorama_knn: int = 30
    scanorama_return_dimred: bool = True

    # Harmony
    harmony_basis: str = 'X_pca'
    harmony_adjusted_basis: str = 'X_pca_harmony'


# Create singleton instances
QC_DEFAULTS = QCDefaults()
HVG_DEFAULTS = HVGDefaults()
DIMRED_DEFAULTS = DimRedDefaults()
CLUSTERING_DEFAULTS = ClusteringDefaults()
CELLTYPIST_DEFAULTS = CellTypistDefaults()
SCIMILARITY_DEFAULTS = ScimilarityDefaults()
BATCH_DEFAULTS = BatchDefaults()


# Utility function to get all defaults as a dictionary
def get_all_defaults() -> dict:
    """Return all default parameters as a nested dictionary."""
    return {
        'qc': {
            'mt_threshold_cells': QC_DEFAULTS.mt_threshold_cells,
            'mt_threshold_nuclei': QC_DEFAULTS.mt_threshold_nuclei,
            'min_cells_per_gene': QC_DEFAULTS.min_cells_per_gene,
            'scrublet_expected_doublet_rate': QC_DEFAULTS.scrublet_expected_doublet_rate,
            'scrublet_sim_ratio': QC_DEFAULTS.scrublet_sim_ratio,
            'scrublet_n_prin_comps': QC_DEFAULTS.scrublet_n_prin_comps,
        },
        'hvg': {
            'n_top_genes': HVG_DEFAULTS.n_top_genes,
            'flavor': HVG_DEFAULTS.flavor,
        },
        'dimred': {
            'n_pcs': DIMRED_DEFAULTS.n_pcs,
            'n_neighbors': DIMRED_DEFAULTS.n_neighbors,
            'metric': DIMRED_DEFAULTS.metric,
            'umap_min_dist': DIMRED_DEFAULTS.umap_min_dist,
        },
        'clustering': {
            'leiden_resolution': CLUSTERING_DEFAULTS.leiden_resolution,
            'phenograph_k': CLUSTERING_DEFAULTS.phenograph_k,
        },
        'celltypist': {
            'target_sum': CELLTYPIST_DEFAULTS.target_sum,
            'model': CELLTYPIST_DEFAULTS.model,
        },
        'batch': {
            'scanorama_dimred': BATCH_DEFAULTS.scanorama_dimred,
            'scanorama_knn': BATCH_DEFAULTS.scanorama_knn,
        },
    }
