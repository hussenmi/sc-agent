"""
scagent: Single-cell RNA-seq Analysis Agent

A standardized single-cell RNA-seq analysis package that:
- Encapsulates lab best practices from workshop notebooks
- Works with Claude Code as an importable module
- Can run as a standalone autonomous agent via Claude API
- Includes data state inspection for autonomous decision making

Quick Start
-----------
>>> import scagent
>>> from scagent.core import load_data, inspect_data, run_qc_pipeline
>>> adata = load_data("my_data.h5")
>>> state = inspect_data(adata)
>>> print(state)

For autonomous analysis:
>>> from scagent.agent import SCAgent
>>> agent = SCAgent()
>>> result = agent.analyze("QC and cluster this PBMC data", data_path="pbmc.h5")
"""

__version__ = "0.1.0"

# Core functionality
from .core import (
    # Inspector
    DataState,
    inspect_data,
    recommend_next_steps,
    # IO
    load_data,
    load_10x_h5,
    load_h5ad,
    # QC
    run_qc_pipeline,
    calculate_qc_metrics,
    filter_cells_by_mt,
    filter_genes,
    detect_doublets,
    # Normalization
    normalize_data,
    preserve_raw_counts,
    # Dimensionality reduction
    run_pca,
    compute_neighbors,
    compute_umap,
    # Clustering
    run_leiden,
    run_phenograph,
)

# Configuration
from .config import (
    QC_DEFAULTS,
    HVG_DEFAULTS,
    DIMRED_DEFAULTS,
    CLUSTERING_DEFAULTS,
    CELLTYPIST_DEFAULTS,
    SCIMILARITY_DEFAULTS,
    BATCH_DEFAULTS,
)

# Annotation
from .annotation import (
    run_celltypist,
    prepare_for_celltypist,
    run_scimilarity,
    prepare_for_scimilarity,
)

# Batch correction
from .batch import (
    run_scanorama,
    run_harmony,
)

# Agent
from .agent import (
    SCAgent,
    AgentWorldState,
    ArtifactRecord,
    DecisionRecord,
    StateDelta,
    VerificationResult,
)

__all__ = [
    # Version
    "__version__",
    # Core - Inspector
    "DataState",
    "inspect_data",
    "recommend_next_steps",
    # Core - IO
    "load_data",
    "load_10x_h5",
    "load_h5ad",
    # Core - QC
    "run_qc_pipeline",
    "calculate_qc_metrics",
    "filter_cells_by_mt",
    "filter_genes",
    "detect_doublets",
    # Core - Normalization
    "normalize_data",
    "preserve_raw_counts",
    # Core - Dimensionality reduction
    "run_pca",
    "compute_neighbors",
    "compute_umap",
    # Core - Clustering
    "run_leiden",
    "run_phenograph",
    # Config
    "QC_DEFAULTS",
    "HVG_DEFAULTS",
    "DIMRED_DEFAULTS",
    "CLUSTERING_DEFAULTS",
    "CELLTYPIST_DEFAULTS",
    "SCIMILARITY_DEFAULTS",
    "BATCH_DEFAULTS",
    # Annotation
    "run_celltypist",
    "prepare_for_celltypist",
    "run_scimilarity",
    "prepare_for_scimilarity",
    # Batch
    "run_scanorama",
    "run_harmony",
    # Agent
    "SCAgent",
    "AgentWorldState",
    "ArtifactRecord",
    "DecisionRecord",
    "StateDelta",
    "VerificationResult",
]
