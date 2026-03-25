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


@dataclass(frozen=True)
class DEGDefaults:
    """Differential expression analysis defaults."""

    # Method
    method: str = 'wilcoxon'  # Recommended for single-cell
    n_genes: int = 100        # Top genes to store per cluster

    # Validation thresholds
    min_cluster_size: int = 20       # ERROR if smaller
    warn_cluster_size: int = 50      # WARNING if smaller
    max_logfc_sanity: float = 10.0   # Flag genes with |logFC| > this

    # Group imbalance
    max_imbalance_ratio: float = 20.0  # WARNING if largest/smallest > this

    # Batch confounding
    batch_confound_threshold: float = 0.7  # Cramer's V threshold for WARNING

    # Preferred layers (in order of preference)
    preferred_raw_layers: tuple = ('raw_counts', 'counts', 'raw_data')

    # Gene set compatibility
    default_geneset: str = 'MSigDB_Hallmark_2020'


@dataclass(frozen=True)
class GSEADefaults:
    """Gene Set Enrichment Analysis defaults."""

    # GSEA parameters
    min_size: int = 5         # Min genes in pathway
    max_size: int = 500       # Max genes in pathway
    permutation_num: int = 1000
    seed: int = 42

    # Default gene sets
    default_geneset: str = 'MSigDB_Hallmark_2020'

    # Available gene sets with descriptions
    available_genesets: tuple = (
        'MSigDB_Hallmark_2020',
        'KEGG_2021_Human',
        'GO_Biological_Process_2021',
        'Reactome_2022',
        'WikiPathways_2021_Human',
    )

    # Significance thresholds
    fdr_threshold: float = 0.25  # Standard GSEA threshold
    nes_min: float = 1.0         # Minimum |NES| to report

    # Literature search
    max_pathways_per_cluster: int = 3
    max_papers_per_pathway: int = 5


# Create singleton instances
QC_DEFAULTS = QCDefaults()
HVG_DEFAULTS = HVGDefaults()
DIMRED_DEFAULTS = DimRedDefaults()
CLUSTERING_DEFAULTS = ClusteringDefaults()
CELLTYPIST_DEFAULTS = CellTypistDefaults()
SCIMILARITY_DEFAULTS = ScimilarityDefaults()
BATCH_DEFAULTS = BatchDefaults()
DEG_DEFAULTS = DEGDefaults()
GSEA_DEFAULTS = GSEADefaults()


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
        'deg': {
            'method': DEG_DEFAULTS.method,
            'n_genes': DEG_DEFAULTS.n_genes,
            'min_cluster_size': DEG_DEFAULTS.min_cluster_size,
            'warn_cluster_size': DEG_DEFAULTS.warn_cluster_size,
            'max_logfc_sanity': DEG_DEFAULTS.max_logfc_sanity,
            'max_imbalance_ratio': DEG_DEFAULTS.max_imbalance_ratio,
            'batch_confound_threshold': DEG_DEFAULTS.batch_confound_threshold,
        },
        'gsea': {
            'min_size': GSEA_DEFAULTS.min_size,
            'max_size': GSEA_DEFAULTS.max_size,
            'permutation_num': GSEA_DEFAULTS.permutation_num,
            'fdr_threshold': GSEA_DEFAULTS.fdr_threshold,
            'default_geneset': GSEA_DEFAULTS.default_geneset,
        },
    }
