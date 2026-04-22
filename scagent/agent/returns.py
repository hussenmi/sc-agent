"""
Structured return objects for agent tools.

Following the design principle: return compact, structured summaries
that give the LLM exactly what it needs to continue intelligently.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json
from anndata import AnnData

from ..core.inspector import (
    DataState,
    clustering_record_to_dict,
    inspect_data,
    metadata_candidate_to_dict,
    semantic_roles_to_dict,
)
from ..analysis import infer_biological_context


@dataclass
class ToolReturn:
    """Base structured return for all tools."""

    status: str  # "ok", "error", "warning"
    tool: str
    message: str = ""
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class QCReturn(ToolReturn):
    """Return from QC pipeline."""

    tool: str = "run_qc"
    input_path: str = ""
    output_path: str = ""
    before: Dict[str, int] = field(default_factory=dict)
    after: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    state: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ClusteringReturn(ToolReturn):
    """Return from clustering."""

    tool: str = "run_clustering"
    method: str = ""
    n_clusters: int = 0
    cluster_sizes: Dict[str, int] = field(default_factory=dict)
    resolution: float = 1.0
    state: Dict[str, bool] = field(default_factory=dict)


@dataclass
class InspectReturn(ToolReturn):
    """Return from data inspection."""

    tool: str = "inspect_data"
    shape: Dict[str, int] = field(default_factory=dict)
    data_type: str = "unknown"
    processing: Dict[str, bool] = field(default_factory=dict)
    embeddings: List[str] = field(default_factory=list)
    clustering: Dict[str, Any] = field(default_factory=dict)
    batch_info: Dict[str, Any] = field(default_factory=dict)
    biological_context: Dict[str, Any] = field(default_factory=dict)
    recommended_next_steps: List[str] = field(default_factory=list)


def make_state_dict(state: DataState) -> Dict[str, bool]:
    """Convert DataState to compact dict for LLM."""
    return {
        "has_raw_counts": state.has_raw_layer,
        "has_qc_metrics": state.has_qc_metrics,
        "has_doublets": state.has_doublet_scores,
        "is_normalized": state.is_normalized,
        "has_hvg": state.has_hvg,
        "has_pca": state.has_pca,
        "has_neighbors": state.has_neighbors,
        "has_umap": state.has_umap,
        "has_clusters": state.has_clusters,
        "has_celltypes": state.has_celltype_annotations,
    }


def build_qc_return(
    adata: AnnData,
    input_path: str,
    output_path: str,
    n_before: int,
    n_genes_before: int,
    warnings: List[str] = None,
) -> QCReturn:
    """Build structured QC return."""

    state = inspect_data(adata)

    return QCReturn(
        status="ok",
        input_path=input_path,
        output_path=output_path,
        before={"n_obs": n_before, "n_vars": n_genes_before},
        after={"n_obs": adata.n_obs, "n_vars": adata.n_vars},
        metrics={
            "cells_removed": n_before - adata.n_obs,
            "genes_removed": n_genes_before - adata.n_vars,
            "doublet_rate": float(adata.obs['predicted_doublet'].mean()) if 'predicted_doublet' in adata.obs else 0,
            "median_pct_mt": float(adata.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in adata.obs else 0,
        },
        warnings=warnings or [],
        state=make_state_dict(state),
    )


def build_inspect_return(adata: AnnData, goal: str = None) -> InspectReturn:
    """Build structured inspect return."""

    from ..core import recommend_next_steps

    state = inspect_data(adata)
    biological_context = infer_biological_context(adata)

    return InspectReturn(
        status="ok",
        shape={"n_obs": state.n_cells, "n_vars": state.n_genes},
        data_type=state.data_type,
        processing=make_state_dict(state),
        embeddings=[k for k in ['X_pca', 'X_umap', 'X_tsne'] if k in adata.obsm],
        clustering={
            "has_clusters": state.has_clusters,
            "cluster_key": state.cluster_key,
            "n_clusters": state.n_clusters,
            "available_clusterings": [
                clustering_record_to_dict(record)
                for record in state.clusterings
            ],
        },
        biological_context={
            **biological_context.to_dict(),
            "semantic_obs_roles": semantic_roles_to_dict(state.semantic_obs_roles),
            "cell_type_key": state.cell_type_key,
        },
        batch_info={
            "batch_key": state.batch_key,
            "n_batches": state.n_batches,
            "corrected": state.batch_correction_applied,
            "metadata_candidates": [
                metadata_candidate_to_dict(candidate)
                for candidate in state.metadata_candidates
            ],
        },
        recommended_next_steps=recommend_next_steps(state, goal) if goal else [],
    )
