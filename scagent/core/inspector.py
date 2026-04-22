"""
Data state inspector for scagent.

This is the MOST CRITICAL module - it detects the current state of the data
and recommends what analysis steps are needed to reach a user's goal.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

SCAGENT_UNS_KEY = "scagent"
SCAGENT_CLUSTERING_REGISTRY_KEY = "clusterings"
SCAGENT_PRIMARY_CLUSTER_SOURCE_KEY = "primary_cluster_sources"

DEFAULT_CLUSTER_KEYS = {
    "leiden": "leiden",
    "phenograph": "pheno_leiden",
    "louvain": "louvain",
}

METADATA_ROLE_ALIASES = {
    "batch": {
        "batch",
        "batch_id",
        "batchid",
        "library",
        "library_id",
        "libraryid",
        "lane",
        "run",
        "run_id",
        "channel",
    },
    "sample": {
        "sample",
        "sample_id",
        "sampleid",
        "orig_ident",
        "orig.ident",
        "origident",
        "pool",
        "pool_id",
        "dataset",
        "library",
        "library_id",
        "libraryid",
    },
    "donor": {
        "donor",
        "donor_id",
        "donorid",
        "patient",
        "patient_id",
        "patientid",
        "subject",
        "subject_id",
        "subjectid",
        "individual",
        "individual_id",
    },
    "condition": {
        "condition",
        "group",
        "status",
        "state",
        "disease",
        "diagnosis",
        "treatment",
        "treated",
        "stim",
        "stimulation",
        "timepoint",
        "time_point",
    },
}

PARTITION_ROLES = {"batch", "sample", "donor"}

SEMANTIC_OBS_ROLE_ALIASES = {
    **METADATA_ROLE_ALIASES,
    "cell_type": {
        "cell_type",
        "celltype",
        "celltypes",
        "cell_type_label",
        "cell_type_labels",
        "cell_label",
        "cell_labels",
        "cell_identity",
        "identity",
        "annotation",
        "annotations",
        "manual_annotation",
        "manual_annotations",
        "manual_labels",
        "author_cell_type",
        "author_celltype",
        "broad_cell_type",
        "fine_cell_type",
        "predicted_labels",
        "majority_voting",
        "celltypist_predicted_labels",
        "celltypist_majority_voting",
        "predictions_unconstrained",
        "representative_prediction",
        "scimilarity_predictions_unconstrained",
        "scimilarity_representative_prediction",
        "cell_ontology_class",
        "cell_ontology_term",
    },
    "cluster": {
        "cluster",
        "clusters",
        "cluster_id",
        "clusterid",
        "clustering",
        "leiden",
        "louvain",
        "phenograph",
        "pheno_leiden",
        "seurat_clusters",
        "seurat_cluster",
        "snn_res",
        "rna_snn_res",
        "integrated_snn_res",
        "res",
        "resolution",
    },
    "disease": {
        "disease",
        "disease_status",
        "diagnosis",
        "pathology",
        "phenotype",
    },
    "tissue": {
        "tissue",
        "tissue_type",
        "organ",
        "site",
        "anatomical_site",
        "compartment",
    },
    "qc_total_counts": {
        "total_counts",
        "n_counts",
        "ncounts",
        "n_count",
        "ncount_rna",
        "ncount",
        "umi_counts",
        "n_umi",
    },
    "qc_n_genes": {
        "n_genes_by_counts",
        "n_genes",
        "num_genes",
        "n_feature",
        "n_features",
        "nfeature_rna",
        "nfeature",
        "genes_detected",
    },
    "qc_pct_mt": {
        "pct_counts_mt",
        "percent_mt",
        "percent.mt",
        "pct_mt",
        "mt_pct",
        "mito_pct",
        "percent_mito",
        "percent_mitochondrial",
    },
    "qc_pct_ribo": {
        "pct_counts_ribo",
        "percent_ribo",
        "percent.ribo",
        "pct_ribo",
        "ribo_pct",
        "percent_ribosomal",
    },
    "doublet_score": {
        "doublet_score",
        "doubletscore",
        "scrublet_score",
        "scrublet_doublet_score",
    },
    "doublet_label": {
        "predicted_doublet",
        "doublet",
        "is_doublet",
        "doublet_call",
        "doublet_label",
        "doublet_class",
    },
}

NUMERIC_SEMANTIC_ROLES = {
    "qc_total_counts",
    "qc_n_genes",
    "qc_pct_mt",
    "qc_pct_ribo",
    "doublet_score",
}

CELL_TYPE_VALUE_PATTERNS = [
    r"\bt\s*cell\b",
    r"\bb\s*cell\b",
    r"\bnk\b",
    r"\bcd4\b",
    r"\bcd8\b",
    r"\bmonocyte\b",
    r"\bmacrophage\b",
    r"\bdendritic\b",
    r"\bplasma\b",
    r"\bmast\b",
    r"\bneutrophil\b",
    r"\bgranulocyte\b",
    r"\bepithelial\b",
    r"\bendothelial\b",
    r"\bfibroblast\b",
    r"\bastrocyte\b",
    r"\bmicroglia\b",
    r"\bneuron\b",
    r"\bhepatocyte\b",
    r"\bkeratinocyte\b",
    r"\bmyeloid\b",
    r"\blymphocyte\b",
]


@dataclass
class MetadataCandidate:
    """Ranked candidate metadata column for collaborative decisions."""

    column: str
    role: str
    n_unique: int
    unique_fraction: float
    confidence: float
    rationale: str
    dtype: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class MetadataResolution:
    """Outcome of resolving a metadata column for a downstream task."""

    status: str = "no_candidate"
    requested_column: Optional[str] = None
    applied_column: Optional[str] = None
    recommended_column: Optional[str] = None
    recommended_role: Optional[str] = None
    needs_user_confirmation: bool = False
    reason: str = ""
    candidates: List[MetadataCandidate] = field(default_factory=list)


@dataclass
class ClusteringRecord:
    """Tracked clustering result stored on an AnnData object."""

    key: str
    method: str
    n_clusters: int
    resolution: Optional[float] = None
    is_primary: bool = False
    source_key: Optional[str] = None
    created_by: str = "inferred"


@dataclass
class DataState:
    """Comprehensive representation of an AnnData object's processing state."""

    # Basic info
    shape: Tuple[int, int] = (0, 0)
    n_cells: int = 0
    n_genes: int = 0

    # Data type
    data_type: str = "unknown"  # "cells", "nuclei", or "unknown"

    # Raw data
    has_raw_layer: bool = False
    raw_layer_name: str = ""
    has_raw: bool = False          # True if adata.raw is set
    raw_n_vars: int = 0            # Number of genes in adata.raw (often > n_genes after HVG)
    is_counts: bool = False        # True if X contains integer counts

    # QC state
    has_qc_metrics: bool = False
    has_mt_metrics: bool = False
    has_ribo_metrics: bool = False
    has_doublet_scores: bool = False
    doublet_detection_method: str = ""

    # Normalization state
    is_normalized: bool = False
    is_log_transformed: bool = False
    normalization_method: str = ""

    # HVG state
    has_hvg: bool = False
    n_hvg: int = 0
    hvg_flavor: str = ""

    # Dimensionality reduction
    has_pca: bool = False
    n_pcs: int = 0
    has_neighbors: bool = False
    n_neighbors: int = 0
    has_umap: bool = False
    has_tsne: bool = False

    # Clustering
    has_clusters: bool = False
    cluster_key: str = ""
    n_clusters: int = 0
    clustering_method: str = ""
    clusterings: List[ClusteringRecord] = field(default_factory=list)

    # Annotations
    has_celltypist: bool = False
    celltypist_model: str = ""
    has_scimilarity: bool = False
    has_celltype_annotations: bool = False
    cell_type_key: str = ""
    cell_type_candidates: List[MetadataCandidate] = field(default_factory=list)
    semantic_obs_roles: Dict[str, List[MetadataCandidate]] = field(default_factory=dict)

    # Batch info
    batch_key: Optional[str] = None
    n_batches: int = 0
    metadata_candidates: List[MetadataCandidate] = field(default_factory=list)
    batch_correction_applied: bool = False
    batch_correction_method: str = ""

    # Gene ID format
    gene_id_format: str = "unknown"  # "symbol", "ensembl", "entrez", "mixed", "unknown"
    has_gene_symbols: bool = False
    has_ensembl_ids: bool = False
    sample_gene_names: List[str] = field(default_factory=list)

    # Additional observations
    obs_columns: List[str] = field(default_factory=list)
    var_columns: List[str] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    obsm_keys: List[str] = field(default_factory=list)
    obsp_keys: List[str] = field(default_factory=list)


def _ensure_scagent_uns(adata: AnnData) -> Dict[str, Any]:
    """Ensure the scagent namespace exists in adata.uns."""
    namespace = adata.uns.get(SCAGENT_UNS_KEY)
    if not isinstance(namespace, dict):
        namespace = {}
        adata.uns[SCAGENT_UNS_KEY] = namespace
    return namespace


def clustering_record_to_dict(record: ClusteringRecord) -> Dict[str, Any]:
    """Serialize a clustering record for tool responses."""
    payload = {
        "key": record.key,
        "method": record.method,
        "n_clusters": int(record.n_clusters),
        "is_primary": bool(record.is_primary),
        "created_by": record.created_by,
    }
    if record.resolution is not None:
        payload["resolution"] = float(record.resolution)
    if record.source_key:
        payload["source_key"] = record.source_key
    return payload


def metadata_candidate_to_dict(candidate: MetadataCandidate) -> Dict[str, Any]:
    """Serialize a metadata candidate for tool responses."""
    return {
        "column": candidate.column,
        "role": candidate.role,
        "n_unique": int(candidate.n_unique),
        "unique_fraction": float(candidate.unique_fraction),
        "confidence": float(candidate.confidence),
        "rationale": candidate.rationale,
        "dtype": candidate.dtype,
        "examples": list(candidate.examples),
    }


def semantic_roles_to_dict(
    roles: Dict[str, List[MetadataCandidate]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Serialize semantic obs-role candidates for LLM-facing responses."""
    return {
        role: [metadata_candidate_to_dict(candidate) for candidate in candidates]
        for role, candidates in roles.items()
        if candidates
    }


def metadata_resolution_to_dict(resolution: MetadataResolution) -> Dict[str, Any]:
    """Serialize a metadata resolution decision for tool responses."""
    return {
        "status": resolution.status,
        "requested_column": resolution.requested_column,
        "applied_column": resolution.applied_column,
        "recommended_column": resolution.recommended_column,
        "recommended_role": resolution.recommended_role,
        "needs_user_confirmation": resolution.needs_user_confirmation,
        "reason": resolution.reason,
        "candidates": [
            metadata_candidate_to_dict(candidate)
            for candidate in resolution.candidates
        ],
    }


def _is_integer_matrix(X) -> bool:
    """Check if matrix contains integer values (counts).

    Uses the dtype as a fast shortcut — integer dtypes are definitely counts,
    float dtypes with a log1p-range max are definitely not.  Only falls back
    to sampling when the dtype is ambiguous (float that could still be counts).
    Sampling uses a cheap head-slice instead of np.random.choice to avoid
    allocating an index array the size of the full NNZ count.
    """
    # Fast dtype shortcut: integer dtypes are always counts
    dtype = getattr(X, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.integer):
        return True

    if sp.issparse(X):
        data = X.data
    else:
        data = X.ravel()

    if len(data) == 0:
        return True

    # Cheap head-slice — avoids O(nnz) np.random.choice for large matrices
    sample = data[:min(10000, len(data))]
    return bool(np.allclose(sample, np.round(sample)))


def _is_subdtype(dtype, kind) -> bool:
    try:
        return bool(np.issubdtype(dtype, kind))
    except TypeError:
        return False


def _detect_data_type(adata: AnnData) -> str:
    """
    Detect if data is from cells or nuclei based on MT content distribution.

    Nuclei typically have very low MT content (<5%) because mitochondria
    are in the cytoplasm, while cells can have higher MT content.
    """
    if "pct_counts_mt" not in adata.obs.columns:
        return "unknown"

    mt_pct = adata.obs["pct_counts_mt"].values
    median_mt = np.median(mt_pct)
    max_mt = np.max(mt_pct)

    if median_mt < 2.0 and max_mt < 10.0:
        return "nuclei"
    return "cells"


def _normalize_column_name(column: str) -> str:
    """Normalize column names for heuristic matching."""
    text = str(column).strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _column_name_role_scores(
    column: str,
    aliases_by_role: Optional[Dict[str, Iterable[str]]] = None,
    *,
    fuzzy: bool = False,
) -> Dict[str, float]:
    """Score how strongly a column name suggests each metadata role."""
    aliases_by_role = aliases_by_role or METADATA_ROLE_ALIASES
    normalized = _normalize_column_name(column)
    compact = normalized.replace("_", "")
    tokens = set(filter(None, normalized.split("_")))
    scores: Dict[str, float] = {}

    for role, aliases in aliases_by_role.items():
        normalized_aliases = {_normalize_column_name(alias) for alias in aliases}
        compact_aliases = {alias.replace("_", "") for alias in normalized_aliases}
        if normalized in normalized_aliases:
            scores[role] = 1.0
            continue
        if compact in compact_aliases:
            scores[role] = max(scores.get(role, 0.0), 0.96)
            continue

        alias_tokens = {
            token
            for alias in normalized_aliases
            for token in alias.split("_")
            if token
        }
        overlap = len(tokens & alias_tokens)
        if overlap:
            scores[role] = min(0.85, 0.45 + 0.18 * overlap)

        if compact and fuzzy:
            best_ratio = max(
                (SequenceMatcher(None, compact, alias).ratio() for alias in compact_aliases),
                default=0.0,
            )
            if best_ratio >= 0.88:
                scores[role] = max(scores.get(role, 0.0), min(0.92, best_ratio))
            elif best_ratio >= 0.80 and len(compact) >= 5:
                scores[role] = max(scores.get(role, 0.0), 0.72)

        if len(compact) >= 5:
            for alias in compact_aliases:
                if len(alias) >= 5 and (compact in alias or alias in compact):
                    scores[role] = max(scores.get(role, 0.0), 0.78)

    if normalized.endswith("_id"):
        if "sample" in normalized or "orig" in normalized:
            scores["sample"] = max(scores.get("sample", 0.0), 0.85)
        elif any(token in normalized for token in ("donor", "patient", "subject", "individual")):
            scores["donor"] = max(scores.get("donor", 0.0), 0.85)
        elif any(token in normalized for token in ("batch", "library", "lane", "run", "channel")):
            scores["batch"] = max(scores.get("batch", 0.0), 0.85)

    return scores


def _series_examples(values: Iterable[Any], limit: int = 3) -> List[str]:
    """Create compact human-readable examples from a sequence of values."""
    examples: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        text = str(value)
        if text not in examples:
            examples.append(text)
        if len(examples) >= limit:
            break
    return examples


def _is_discrete_obs_column(adata: AnnData, column: str) -> bool:
    """Check whether a column is suitable as a low-cardinality partition key."""
    series = adata.obs[column]
    n_obs = max(1, adata.n_obs)
    n_unique = int(series.nunique(dropna=True))
    if n_unique <= 1:
        return False
    if n_unique >= n_obs:
        return False
    if (n_unique / n_obs) >= 0.9:
        return False

    dtype_name = str(series.dtype)
    if dtype_name == "category" or dtype_name == "bool" or "string" in dtype_name or dtype_name == "object":
        return True

    if np.issubdtype(series.dtype, np.integer):
        return True

    if np.issubdtype(series.dtype, np.floating):
        non_na = series.dropna()
        if non_na.empty:
            return False
        return np.allclose(non_na.to_numpy(), np.round(non_na.to_numpy()))

    return False


def _categorical_structure_score(series, n_obs: int, role: str) -> float:
    n_unique = int(series.nunique(dropna=True))
    if n_unique <= 1 or n_unique >= n_obs:
        return 0.0

    unique_fraction = n_unique / max(1, n_obs)
    dtype_name = str(series.dtype)
    score = 0.0
    if dtype_name == "category" or dtype_name == "bool" or "string" in dtype_name or dtype_name == "object":
        score += 0.3
    elif _is_subdtype(series.dtype, np.integer):
        score += 0.22
    elif _is_subdtype(series.dtype, np.floating):
        non_na = series.dropna()
        if non_na.empty or not np.allclose(non_na.to_numpy(), np.round(non_na.to_numpy())):
            return 0.0
        score += 0.12

    if role == "cell_type":
        if 2 <= n_unique <= min(200, max(8, int(n_obs * 0.35))):
            score += 0.32
        elif unique_fraction < 0.65:
            score += 0.16
    elif role == "cluster":
        if 2 <= n_unique <= min(80, max(3, int(n_obs * 0.25))):
            score += 0.34
        elif unique_fraction < 0.35:
            score += 0.12
    else:
        if n_unique <= min(24, max(3, int(n_obs * 0.2))):
            score += 0.3
        elif unique_fraction <= 0.3:
            score += 0.14

    if unique_fraction <= 0.05:
        score += 0.18
    elif unique_fraction <= 0.2:
        score += 0.1

    return min(1.0, score)


def _numeric_structure_score(series, role: str) -> float:
    if not _is_subdtype(series.dtype, np.number):
        return 0.0
    non_na = series.dropna()
    if non_na.empty:
        return 0.0

    score = 0.45
    values = non_na.to_numpy()
    if role in {"qc_pct_mt", "qc_pct_ribo", "doublet_score"}:
        finite = values[np.isfinite(values)]
        if len(finite) and float(np.nanmin(finite)) >= 0:
            score += 0.15
        if len(finite) and float(np.nanmax(finite)) <= 100:
            score += 0.12
    elif role in {"qc_total_counts", "qc_n_genes"}:
        if float(np.nanmax(values)) > 10:
            score += 0.12
        if np.allclose(values[: min(1000, len(values))], np.round(values[: min(1000, len(values))])):
            score += 0.08
    return min(1.0, score)


def _semantic_value_score(series, role: str) -> float:
    values = [str(value).strip().lower() for value in series.dropna().unique().tolist()[:50]]
    if not values:
        return 0.0

    if role == "cell_type":
        text = " | ".join(values)
        matches = sum(1 for pattern in CELL_TYPE_VALUE_PATTERNS if re.search(pattern, text))
        if matches >= 3:
            return 0.45
        if matches == 2:
            return 0.34
        if matches == 1:
            return 0.22
        if any("cell" in value for value in values):
            return 0.14

    if role == "cluster":
        numeric_like = 0
        for value in values:
            try:
                float(value)
                numeric_like += 1
            except ValueError:
                if value.startswith(("cluster", "clust", "c")):
                    numeric_like += 1
        if numeric_like and numeric_like / len(values) >= 0.75:
            return 0.32

    if role == "doublet_label":
        normalized_values = {value.replace(" ", "_") for value in values}
        known = {"true", "false", "0", "1", "doublet", "singlet", "multiplet", "negative", "positive"}
        if normalized_values and normalized_values <= known:
            return 0.35

    return 0.0


def _semantic_min_confidence(role: str) -> float:
    if role in {"cell_type", "cluster"}:
        return 0.38
    if role in NUMERIC_SEMANTIC_ROLES:
        return 0.48
    return 0.34


def _score_obs_semantic_candidate(
    adata: AnnData,
    column: str,
    role: str,
) -> Optional[MetadataCandidate]:
    series = adata.obs[column]
    n_obs = max(1, adata.n_obs)
    n_unique = int(series.nunique(dropna=True))
    if n_unique <= 1:
        return None

    name_score = _column_name_role_scores(
        column,
        SEMANTIC_OBS_ROLE_ALIASES,
        fuzzy=True,
    ).get(role, 0.0)
    if role in NUMERIC_SEMANTIC_ROLES and name_score < 0.7:
        return None

    if role in NUMERIC_SEMANTIC_ROLES:
        structure_score = _numeric_structure_score(series, role)
        value_score = 0.0
    else:
        structure_score = _categorical_structure_score(series, n_obs, role)
        value_score = _semantic_value_score(series, role)

    if structure_score == 0.0 and name_score < 0.9:
        return None

    confidence = min(0.99, 0.56 * name_score + 0.27 * structure_score + 0.17 * value_score)
    if name_score == 0.0:
        confidence *= 0.72
    if confidence < _semantic_min_confidence(role):
        return None

    examples = _series_examples(series.dropna().unique().tolist(), limit=4)
    rationale_parts = [f"{n_unique} unique values"]
    if name_score >= 0.9:
        rationale_parts.append(f"name strongly suggests {role}")
    elif name_score >= 0.65:
        rationale_parts.append(f"name resembles {role}")
    elif name_score > 0:
        rationale_parts.append(f"name weakly suggests {role}")
    if structure_score >= 0.55:
        rationale_parts.append("values have the expected structure")
    elif structure_score > 0:
        rationale_parts.append("values are structurally plausible")
    if value_score > 0:
        rationale_parts.append("example values support the role")
    if examples:
        rationale_parts.append("examples: " + ", ".join(examples))

    return MetadataCandidate(
        column=str(column),
        role=role,
        n_unique=n_unique,
        unique_fraction=n_unique / n_obs,
        confidence=round(confidence, 3),
        rationale="; ".join(rationale_parts),
        dtype=str(series.dtype),
        examples=examples,
    )


def rank_obs_semantic_candidates(
    adata: AnnData,
    *,
    roles: Optional[Iterable[str]] = None,
    limit_per_role: int = 6,
) -> Dict[str, List[MetadataCandidate]]:
    """Rank obs columns by semantic role using names, structure, and examples."""
    roles_set = set(roles) if roles is not None else set(SEMANTIC_OBS_ROLE_ALIASES)
    ranked: Dict[str, List[MetadataCandidate]] = {}

    for column in adata.obs.columns:
        for role in roles_set:
            candidate = _score_obs_semantic_candidate(adata, column, role)
            if candidate is not None:
                ranked.setdefault(role, []).append(candidate)

    for role, candidates in list(ranked.items()):
        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        ranked[role] = candidates[:limit_per_role]

    return ranked


def rank_obs_metadata_candidates(
    adata: AnnData,
    *,
    roles: Optional[Iterable[str]] = None,
    limit: int = 6,
) -> List[MetadataCandidate]:
    """Rank likely metadata columns for collaborative decisions."""
    roles_set = set(roles) if roles is not None else set(METADATA_ROLE_ALIASES)
    candidates: List[MetadataCandidate] = []
    n_obs = max(1, adata.n_obs)

    for column in adata.obs.columns:
        if not _is_discrete_obs_column(adata, column):
            continue

        series = adata.obs[column]
        n_unique = int(series.nunique(dropna=True))
        unique_fraction = n_unique / n_obs
        dtype_name = str(series.dtype)
        role_scores = {
            role: score
            for role, score in _column_name_role_scores(column).items()
            if role in roles_set
        }

        if role_scores:
            role = max(role_scores, key=role_scores.get)
            name_score = role_scores[role]
        elif len(roles_set) == 1:
            role = next(iter(roles_set))
            name_score = 0.0
        else:
            continue

        structure_score = 0.0
        if dtype_name == "category" or dtype_name == "bool" or "string" in dtype_name or dtype_name == "object":
            structure_score += 0.25
        elif np.issubdtype(series.dtype, np.integer):
            structure_score += 0.15
        elif np.issubdtype(series.dtype, np.floating):
            structure_score += 0.1

        if n_unique <= min(12, max(3, int(n_obs * 0.1))):
            structure_score += 0.3
        elif n_unique <= max(24, int(n_obs * 0.2)):
            structure_score += 0.2
        else:
            structure_score += 0.05

        if unique_fraction <= 0.05:
            structure_score += 0.2
        elif unique_fraction <= 0.2:
            structure_score += 0.12
        else:
            structure_score += 0.03

        confidence = min(0.99, 0.58 * name_score + 0.42 * structure_score)
        if name_score == 0.0:
            confidence *= 0.65

        if confidence < 0.3:
            continue

        examples = _series_examples(series.dropna().unique().tolist())
        rationale_parts = [f"{n_unique} unique values"]
        if name_score >= 0.85:
            rationale_parts.append(f"name strongly suggests {role}")
        elif name_score >= 0.45:
            rationale_parts.append(f"name suggests {role}")
        else:
            rationale_parts.append("discrete metadata-like values")
        if examples:
            rationale_parts.append("examples: " + ", ".join(examples))

        candidates.append(
            MetadataCandidate(
                column=str(column),
                role=role,
                n_unique=n_unique,
                unique_fraction=unique_fraction,
                confidence=round(confidence, 3),
                rationale="; ".join(rationale_parts),
                dtype=dtype_name,
                examples=examples,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.confidence,
            candidate.role in PARTITION_ROLES,
            candidate.unique_fraction,
        ),
        reverse=True,
    )
    return candidates[:limit]


def resolve_batch_metadata(
    adata: AnnData,
    requested_column: Optional[str] = None,
) -> MetadataResolution:
    """Resolve the most appropriate column for per-batch operations like Scrublet."""
    candidates = rank_obs_metadata_candidates(adata, roles=PARTITION_ROLES)
    result = MetadataResolution(
        requested_column=requested_column,
        candidates=candidates,
    )

    if requested_column:
        if requested_column not in adata.obs.columns:
            result.status = "invalid_requested"
            result.reason = (
                f"'{requested_column}' is not present in adata.obs. "
                "Use one of the recommended candidates or proceed without batch stratification."
            )
            if candidates:
                result.recommended_column = candidates[0].column
                result.recommended_role = candidates[0].role
            return result

        if not _is_discrete_obs_column(adata, requested_column):
            result.status = "invalid_requested"
            result.reason = (
                f"'{requested_column}' exists but is not a suitable partition column "
                "because it is constant or nearly unique per cell."
            )
            if candidates:
                result.recommended_column = candidates[0].column
                result.recommended_role = candidates[0].role
            return result

        match = next((candidate for candidate in candidates if candidate.column == requested_column), None)
        result.status = "user_selected"
        result.applied_column = requested_column
        result.recommended_column = requested_column
        result.recommended_role = match.role if match else None
        n_unique = int(adata.obs[requested_column].nunique(dropna=True))
        result.reason = f"Using the user-specified column '{requested_column}' ({n_unique} groups)."
        return result

    if not candidates:
        result.status = "no_candidate"
        result.reason = (
            "I did not find an obvious low-cardinality batch/sample/donor column. "
            "Per-batch methods can run on the full dataset unless the user later provides one."
        )
        return result

    top = candidates[0]
    gap = top.confidence - candidates[1].confidence if len(candidates) > 1 else top.confidence
    result.recommended_column = top.column
    result.recommended_role = top.role

    obvious = (
        top.confidence >= 0.78
        and gap >= 0.16
        and top.n_unique <= max(2, int(max(2, adata.n_obs) * 0.3))
    )

    if obvious:
        result.status = "auto_selected"
        result.applied_column = top.column
        result.reason = (
            f"Auto-selected '{top.column}' for per-batch processing because it is a strong "
            f"{top.role}-like column with {top.n_unique} groups."
        )
    else:
        result.status = "needs_confirmation"
        result.needs_user_confirmation = True
        result.reason = (
            f"'{top.column}' looks like the best {top.role}-like candidate, but the metadata is "
            "not clear enough to rely on automatically."
        )

    return result


def _detect_raw_layer(adata: AnnData) -> Tuple[bool, str]:
    """Detect if a raw counts layer exists in adata.layers.

    Note: adata.raw is checked separately in inspect_data and reported via
    has_raw / raw_n_vars fields.
    """
    common_raw_names = ["raw_counts", "raw_data", "counts", "raw"]

    for name in common_raw_names:
        if name in adata.layers and _is_integer_matrix(adata.layers[name]):
            return True, name

    # Check adata.raw — but only if it actually contains integer counts.
    # adata.raw is often log-normalized data stored before HVG selection,
    # not true raw counts. Verify before reporting it as a raw counts source.
    if adata.raw is not None and _is_integer_matrix(adata.raw.X):
        return True, "__raw__"

    return False, ""


def _detect_gene_id_format(adata: AnnData) -> Tuple[str, bool, bool, List[str]]:
    """
    Detect the format of gene identifiers.

    Returns: (format, has_symbols, has_ensembl, sample_names)
    """
    info = _characterize_features(adata)
    return (
        info["gene_id_format"],
        info["has_gene_symbols"],
        info["has_ensembl_ids"],
        info["sample_gene_names"],
    )


def _characterize_features(adata: AnnData) -> dict:
    """
    Comprehensive characterization of var_names and obs_names for the LLM.

    Returns a dict that is embedded directly into the inspect_data result so
    the agent has everything it needs to decide whether gene names need
    transformation before QC or annotation.
    """
    gene_names = adata.var_names.tolist()
    sample_size = min(200, len(gene_names))
    sample = gene_names[:sample_size]

    ensembl_re = re.compile(r"^ENS[A-Z]{0,3}G\d{11}")
    entrez_re = re.compile(r"^\d{1,8}$")
    symbol_re = re.compile(r"^[A-Z][A-Z0-9\-\.]{1,20}$", re.IGNORECASE)
    genome_prefix_re = re.compile(r"^([A-Za-z0-9]+_{2,})")

    # --- genome prefix detection ---
    prefix_counts: dict = {}
    for g in sample:
        m = genome_prefix_re.match(str(g))
        if m:
            prefix_counts[m.group(1)] = prefix_counts.get(m.group(1), 0) + 1
    genome_prefix = None
    if prefix_counts:
        top_prefix, top_count = max(prefix_counts.items(), key=lambda x: x[1])
        if top_count > sample_size * 0.3:
            genome_prefix = top_prefix

    # strip prefix for downstream pattern matching
    def _strip(name: str) -> str:
        return genome_prefix_re.sub("", name) if genome_prefix else name

    stripped = [_strip(g) for g in sample]

    ensembl_count = sum(1 for g in stripped if ensembl_re.match(g))
    entrez_count = sum(1 for g in stripped if entrez_re.match(g))
    symbol_count = sum(1 for g in stripped if symbol_re.match(g) and not ensembl_re.match(g))

    has_symbols_col = "gene_symbols" in adata.var.columns or "gene_name" in adata.var.columns
    has_ensembl_col = "gene_ids" in adata.var.columns or "ensembl_id" in adata.var.columns

    if ensembl_count > sample_size * 0.5:
        fmt = "ensembl"
    elif entrez_count > sample_size * 0.5:
        fmt = "entrez"
    elif symbol_count > sample_size * 0.3:
        fmt = "symbol"
    else:
        fmt = "mixed" if (ensembl_count > 0 and symbol_count > 0) else "unknown"

    # --- special / non-human gene populations ---
    special_populations: List[dict] = []
    all_genes_set = set(gene_names)

    # viral / custom genomes — anything with a prefix different from the main one
    alt_prefixes: dict = {}
    for g in gene_names:
        m = genome_prefix_re.match(str(g))
        if m:
            p = m.group(1)
            if p != genome_prefix:
                alt_prefixes[p] = alt_prefixes.get(p, 0) + 1
    for p, cnt in alt_prefixes.items():
        special_populations.append({"prefix": p, "n_genes": cnt,
                                     "example": next(g for g in gene_names if g.startswith(p))})

    # --- MT / ribo detectability ---
    mt_genes_found = [g for g in gene_names if re.search(r'(?:^|_)MT-', g)]
    ribo_genes_found = [g for g in gene_names if re.search(r'(?:^|_)(?:RPS|RPL)', g)]

    mt_warning = None
    ribo_warning = None
    if len(mt_genes_found) == 0:
        mt_warning = (
            f"No MT- genes found with standard prefix search. "
            f"{'Genome prefix detected: ' + repr(genome_prefix) + ' — strip it before QC.' if genome_prefix else 'Inspect var_names format before running QC.'}"
        )
    if len(ribo_genes_found) == 0:
        ribo_warning = (
            f"No RPS/RPL genes found with standard prefix search. "
            f"{'Genome prefix detected: ' + repr(genome_prefix) + ' — strip it before QC.' if genome_prefix else 'Inspect var_names format before running QC.'}"
        )

    # --- obs_names (barcode) characterization ---
    obs_sample = adata.obs_names.tolist()[:10]
    tenx_re = re.compile(r'^[ACGT]{16}(-\d+)?$')
    n_tenx = sum(1 for b in obs_sample if tenx_re.match(b))
    obs_format = "10x_barcode" if n_tenx > 7 else "custom"
    suffixes = set()
    for b in obs_sample:
        m = re.search(r'-(\d+)$', b)
        if m:
            suffixes.add(m.group(1))

    return {
        # gene id format (backwards-compat with existing callers)
        "gene_id_format": fmt,
        "has_gene_symbols": has_symbols_col or fmt == "symbol",
        "has_ensembl_ids": has_ensembl_col or fmt == "ensembl",
        "sample_gene_names": gene_names[:10],
        # extended info for LLM — raw facts, no pre-interpreted flags
        "genome_prefix": genome_prefix,
        "special_gene_populations": special_populations,
        "mt_genes_detected": len(mt_genes_found),
        "mt_gene_examples": mt_genes_found[:3],
        "ribo_genes_detected": len(ribo_genes_found),
        "ribo_gene_examples": ribo_genes_found[:3],
        "obs_names_format": obs_format,
        "obs_names_sample": obs_sample,
        "obs_names_suffixes_detected": sorted(suffixes),
    }


def _detect_normalization(adata: AnnData) -> Tuple[bool, bool, str]:
    """
    Detect if data is normalized and log-transformed.

    Uses metadata shortcuts to avoid touching adata.X when possible — X.max()
    on a large sparse matrix can scan hundreds of millions of values.

    Returns: (is_normalized, is_log_transformed, method)
    """
    # --- Metadata shortcuts (no X access) ---
    # log1p in uns is written by sc.pp.log1p and is definitive.
    if "log1p" in adata.uns:
        return True, True, "log1p"

    # HVG + PCA presence strongly implies prior normalization even without log1p.
    if "highly_variable" in adata.var.columns and "X_pca" in adata.obsm:
        method = "log1p" if "log1p" in adata.uns else "unknown"
        return True, False, method

    # Integer dtype means raw counts — no normalization has occurred.
    dtype = getattr(adata.X, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.integer):
        return False, False, ""

    # --- Fallback: sample a small prefix of X.data (cheap head-slice) ---
    if sp.issparse(adata.X):
        data = adata.X.data
        if len(data) == 0:
            return False, False, ""
        sample_data = data[:min(10000, len(data))]
        max_val = float(sample_data.max())
    else:
        flat = adata.X.ravel()
        if len(flat) == 0:
            return False, False, ""
        sample_data = flat[:10000]
        max_val = float(sample_data.max())

    has_floats = not np.allclose(sample_data, np.round(sample_data))
    is_log = max_val < 15 and has_floats
    is_normalized = has_floats

    return is_normalized, is_log, ""


def normalize_clustering_method(method: str) -> str:
    """Normalize clustering method names used across tools and registries."""
    text = str(method).strip().lower()
    if text in {"pheno", "pheno_leiden", "phenograph"}:
        return "phenograph"
    if text == "louvain":
        return "louvain"
    return "leiden"


def default_cluster_key_for_method(method: str) -> str:
    """Return the compatibility alias used for a clustering method."""
    return DEFAULT_CLUSTER_KEYS.get(normalize_clustering_method(method), str(method))


def format_resolution_token(resolution: float) -> str:
    """Create a deterministic, filesystem-safe resolution token."""
    numeric = float(resolution)
    if numeric.is_integer():
        return str(int(numeric))
    text = f"{numeric:.6g}"
    return text.replace(".", "_").replace("-", "neg_")


def infer_cluster_key(method: str, resolution: float) -> str:
    """Infer a deterministic cluster key for a non-primary clustering result."""
    base = default_cluster_key_for_method(method)
    return f"{base}_res_{format_resolution_token(resolution)}"


def _infer_obs_clustering_entries(adata: AnnData) -> List[ClusteringRecord]:
    """Infer clustering-like columns directly from obs for backwards compatibility."""
    entries: List[ClusteringRecord] = []
    seen = set()

    for key in adata.obs.columns:
        text = str(key)
        if text in seen:
            continue
        if text not in {"leiden", "louvain", "pheno_leiden", "clusters", "cluster"} and not (
            text.startswith(("leiden", "louvain", "pheno_leiden"))
        ):
            continue

        if not _is_discrete_obs_column(adata, key):
            continue

        n_clusters = int(adata.obs[key].nunique(dropna=True))
        if n_clusters <= 1:
            continue

        if text.startswith("pheno_leiden"):
            method = "phenograph"
        elif text.startswith("louvain"):
            method = "louvain"
        else:
            method = "leiden"

        entries.append(
            ClusteringRecord(
                key=text,
                method=method,
                n_clusters=n_clusters,
                resolution=None,
                is_primary=(text == default_cluster_key_for_method(method)),
                created_by="inferred",
            )
        )
        seen.add(text)

    semantic_clusters = rank_obs_semantic_candidates(adata, roles={"cluster"}).get("cluster", [])
    for candidate in semantic_clusters:
        if candidate.column in seen:
            continue
        n_clusters = int(adata.obs[candidate.column].nunique(dropna=True))
        if n_clusters <= 1:
            continue
        entries.append(
            ClusteringRecord(
                key=candidate.column,
                method="inferred",
                n_clusters=n_clusters,
                resolution=None,
                is_primary=False,
                created_by="inferred",
            )
        )
        seen.add(candidate.column)

    return entries


def get_clustering_registry(adata: AnnData) -> List[ClusteringRecord]:
    """Return tracked clustering results, including inferred legacy keys."""
    namespace = _ensure_scagent_uns(adata)
    registry = namespace.get(SCAGENT_CLUSTERING_REGISTRY_KEY, {})
    primary_sources = namespace.get(SCAGENT_PRIMARY_CLUSTER_SOURCE_KEY, {})
    records: Dict[str, ClusteringRecord] = {}

    if isinstance(registry, dict):
        for key, payload in registry.items():
            if key not in adata.obs.columns or not isinstance(payload, dict):
                continue
            method = normalize_clustering_method(payload.get("method", key))
            records[key] = ClusteringRecord(
                key=key,
                method=method,
                n_clusters=int(adata.obs[key].nunique(dropna=True)),
                resolution=payload.get("resolution"),
                is_primary=(key == default_cluster_key_for_method(method)),
                source_key=payload.get("source_key"),
                created_by=str(payload.get("created_by", "tool")),
            )

    for inferred in _infer_obs_clustering_entries(adata):
        records.setdefault(inferred.key, inferred)

    for method in {"leiden", "phenograph", "louvain"}:
        alias = default_cluster_key_for_method(method)
        if alias in records:
            records[alias].is_primary = True
            source_key = primary_sources.get(method)
            if isinstance(source_key, str) and source_key and source_key in adata.obs.columns:
                records[alias].source_key = source_key

    return sorted(
        records.values(),
        key=lambda record: (
            not record.is_primary,
            record.method,
            record.key,
        ),
    )


def register_clustering(
    adata: AnnData,
    *,
    cluster_key: str,
    method: str,
    resolution: Optional[float],
    created_by: str = "tool",
    source_key: Optional[str] = None,
) -> None:
    """Register a clustering result in adata.uns for later inspection."""
    namespace = _ensure_scagent_uns(adata)
    registry = namespace.setdefault(SCAGENT_CLUSTERING_REGISTRY_KEY, {})
    registry[cluster_key] = {
        "method": normalize_clustering_method(method),
        "resolution": float(resolution) if resolution is not None else None,
        "source_key": source_key,
        "created_by": created_by,
    }


def promote_clustering_to_primary(
    adata: AnnData,
    *,
    cluster_key: str,
    method: str,
    resolution: Optional[float],
    created_by: str = "tool",
) -> str:
    """Promote a clustering result to the compatibility alias for its method."""
    normalized_method = normalize_clustering_method(method)
    alias = default_cluster_key_for_method(normalized_method)
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs")

    if alias != cluster_key:
        adata.obs[alias] = adata.obs[cluster_key].astype("category")

    namespace = _ensure_scagent_uns(adata)
    primary_sources = namespace.setdefault(SCAGENT_PRIMARY_CLUSTER_SOURCE_KEY, {})
    primary_sources[normalized_method] = cluster_key

    register_clustering(
        adata,
        cluster_key=cluster_key,
        method=normalized_method,
        resolution=resolution,
        created_by=created_by,
    )
    register_clustering(
        adata,
        cluster_key=alias,
        method=normalized_method,
        resolution=resolution,
        created_by=created_by,
        source_key=cluster_key,
    )
    return alias


def _detect_clustering(
    adata: AnnData,
) -> Tuple[bool, str, int, str, List[ClusteringRecord]]:
    """Detect clustering state and available tracked clustering results."""
    registry = get_clustering_registry(adata)
    if registry:
        primary = next((record for record in registry if record.is_primary), registry[0])
        return True, primary.key, primary.n_clusters, primary.method, registry
    return False, "", 0, "", []


def inspect_data(adata: AnnData) -> DataState:
    """
    Comprehensive data inspection that returns a DataState object.

    This function analyzes an AnnData object to determine its current
    processing state, which is essential for autonomous analysis.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to inspect.

    Returns
    -------
    DataState
        A dataclass containing all detected states.
    """
    state = DataState()

    state.shape = adata.shape
    state.n_cells = adata.n_obs
    state.n_genes = adata.n_vars

    state.obs_columns = list(adata.obs.columns)
    state.var_columns = list(adata.var.columns)
    state.layers = list(adata.layers.keys())
    state.obsm_keys = list(adata.obsm.keys())
    state.obsp_keys = list(adata.obsp.keys())
    state.semantic_obs_roles = rank_obs_semantic_candidates(adata)

    state.has_raw_layer, state.raw_layer_name = _detect_raw_layer(adata)
    if adata.raw is not None:
        state.has_raw = True
        state.raw_n_vars = adata.raw.n_vars
    state.is_counts = _is_integer_matrix(adata.X)

    gene_fmt, has_sym, has_ens, sample_genes = _detect_gene_id_format(adata)
    state.gene_id_format = gene_fmt
    state.has_gene_symbols = has_sym
    state.has_ensembl_ids = has_ens
    state.sample_gene_names = sample_genes

    qc_roles = state.semantic_obs_roles
    qc_obs_cols = ["n_genes_by_counts", "total_counts", "n_genes"]
    state.has_qc_metrics = any(column in adata.obs.columns for column in qc_obs_cols) or bool(
        qc_roles.get("qc_total_counts") or qc_roles.get("qc_n_genes") or qc_roles.get("qc_pct_mt")
    )
    state.has_mt_metrics = "pct_counts_mt" in adata.obs.columns or bool(qc_roles.get("qc_pct_mt"))
    state.has_ribo_metrics = "pct_counts_ribo" in adata.obs.columns or bool(qc_roles.get("qc_pct_ribo"))

    if (
        "doublet_score" in adata.obs.columns
        or "predicted_doublet" in adata.obs.columns
        or qc_roles.get("doublet_score")
        or qc_roles.get("doublet_label")
    ):
        state.has_doublet_scores = True
        state.doublet_detection_method = "scrublet" if "scrublet" in adata.uns else "unknown"

    if state.has_mt_metrics:
        state.data_type = _detect_data_type(adata)

    is_norm, is_log, norm_method = _detect_normalization(adata)
    state.is_normalized = is_norm
    state.is_log_transformed = is_log
    state.normalization_method = norm_method

    if "highly_variable" in adata.var.columns:
        state.has_hvg = True
        state.n_hvg = int(adata.var["highly_variable"].sum())
        if "hvg" in adata.uns:
            state.hvg_flavor = adata.uns["hvg"].get("flavor", "")

    if "X_pca" in adata.obsm:
        state.has_pca = True
        state.n_pcs = adata.obsm["X_pca"].shape[1]

    if "neighbors" in adata.uns or "connectivities" in adata.obsp:
        state.has_neighbors = True
        if "neighbors" in adata.uns and "params" in adata.uns["neighbors"]:
            state.n_neighbors = adata.uns["neighbors"]["params"].get("n_neighbors", 0)

    state.has_umap = "X_umap" in adata.obsm
    state.has_tsne = "X_tsne" in adata.obsm

    has_clusters, cluster_key, n_clusters, cluster_method, clusterings = _detect_clustering(adata)
    state.has_clusters = has_clusters
    state.cluster_key = cluster_key
    state.n_clusters = n_clusters
    state.clustering_method = cluster_method
    state.clusterings = clusterings

    celltypist_cols = [
        "predicted_labels",
        "majority_voting",
        "celltype_majority_voting",
        "celltypist_predicted_labels",
        "celltypist_majority_voting",
    ]
    if any(column in adata.obs.columns for column in celltypist_cols):
        state.has_celltypist = True

    scimilarity_cols = [
        "predictions_unconstrained",
        "representative_prediction",
        "scimilarity_predictions_unconstrained",
        "scimilarity_representative_prediction",
    ]
    if any(column in adata.obs.columns for column in scimilarity_cols):
        state.has_scimilarity = True
    state.cell_type_candidates = state.semantic_obs_roles.get("cell_type", [])
    if state.cell_type_candidates:
        state.cell_type_key = state.cell_type_candidates[0].column
    state.has_celltype_annotations = bool(
        state.cell_type_candidates or state.has_celltypist or state.has_scimilarity
    )

    batch_resolution = resolve_batch_metadata(adata)
    state.metadata_candidates = batch_resolution.candidates
    state.batch_key = batch_resolution.applied_column
    if state.batch_key:
        state.n_batches = int(adata.obs[state.batch_key].nunique(dropna=True))

    if "X_scanorama" in adata.obsm:
        state.batch_correction_applied = True
        state.batch_correction_method = "scanorama"
    elif "X_pca_harmony" in adata.obsm:
        state.batch_correction_applied = True
        state.batch_correction_method = "harmony"

    return state


def recommend_next_steps(state: DataState, goal: str) -> List[str]:
    """
    Recommend analysis steps to reach a user's goal.

    Parameters
    ----------
    state : DataState
        Current data state from inspect_data().
    goal : str
        User's analysis goal. Common goals:
        - 'qc': Perform quality control
        - 'cluster': Get clusters
        - 'annotate': Get cell type annotations
        - 'umap': Generate UMAP visualization
        - 'deg': Differential expression analysis
        - 'batch_correct': Batch correction

    Returns
    -------
    List[str]
        Ordered list of recommended analysis steps.
    """
    steps = []
    goal = goal.lower()

    if goal == "qc":
        if not state.has_qc_metrics:
            steps.append("calculate_qc_metrics")
        if not state.has_mt_metrics:
            steps.append("calculate_mt_metrics")
        if not state.has_doublet_scores:
            steps.append("detect_doublets")
        steps.append("filter_cells_by_qc")
        steps.append("filter_genes")
        return steps

    if not state.has_raw_layer and not state.has_raw and state.is_counts:
        steps.append("preserve_raw_counts")

    if not state.has_qc_metrics:
        steps.append("calculate_qc_metrics")
    if not state.has_doublet_scores:
        steps.append("detect_doublets")

    if not state.is_normalized:
        steps.append("normalize_data")

    if goal in ["cluster", "umap", "annotate", "deg", "batch_correct"]:
        if not state.has_hvg:
            steps.append("select_hvg")

        if goal == "batch_correct" or (state.n_batches > 1 and not state.batch_correction_applied):
            if goal == "batch_correct":
                steps.append("run_batch_correction")

        if not state.has_pca:
            steps.append("run_pca")

        if not state.has_neighbors:
            steps.append("compute_neighbors")

        if goal in ["umap", "annotate"] and not state.has_umap:
            steps.append("compute_umap")

        if goal in ["cluster", "annotate", "deg"] and not state.has_clusters:
            steps.append("run_clustering")

    if goal == "annotate" and not state.has_celltypist:
        steps.append("run_celltypist")

    if goal == "deg":
        if state.has_clusters:
            steps.append("run_deg_analysis")
        else:
            steps.append("run_clustering")
            steps.append("run_deg_analysis")

    return steps


def summarize_state(state: DataState) -> str:
    """
    Generate a human-readable summary of the data state.

    Parameters
    ----------
    state : DataState
        Data state from inspect_data().

    Returns
    -------
    str
        Human-readable summary.
    """
    lines = []
    lines.append(f"Data shape: {state.n_cells:,} cells x {state.n_genes:,} genes")
    lines.append(f"Data type: {state.data_type}")

    processing = []
    if state.has_raw:
        extra = f", {state.raw_n_vars:,} genes" if state.raw_n_vars != state.n_genes else ""
        processing.append(f"raw counts in adata.raw{extra}")
    if state.has_raw_layer:
        processing.append(f"raw counts in layer '{state.raw_layer_name}'")
    if state.has_qc_metrics:
        processing.append("QC metrics computed")
    if state.has_doublet_scores:
        processing.append(f"doublets detected ({state.doublet_detection_method})")
    if state.is_normalized:
        processing.append("normalized")
    if state.is_log_transformed:
        processing.append("log-transformed")
    if state.has_hvg:
        processing.append(f"{state.n_hvg} HVGs selected")

    if processing:
        lines.append("Processing: " + ", ".join(processing))

    embeddings = []
    if state.has_pca:
        embeddings.append(f"PCA ({state.n_pcs} PCs)")
    if state.has_neighbors:
        embeddings.append(f"neighbors (k={state.n_neighbors})")
    if state.has_umap:
        embeddings.append("UMAP")
    if state.has_tsne:
        embeddings.append("tSNE")

    if embeddings:
        lines.append("Embeddings: " + ", ".join(embeddings))

    if state.has_clusters:
        lines.append(f"Clustering: {state.n_clusters} clusters ({state.clustering_method})")
        if state.clusterings:
            available = ", ".join(
                f"{record.key} ({record.n_clusters})"
                for record in state.clusterings[:4]
            )
            lines.append("Available clusterings: " + available)

    annotations = []
    if state.has_celltypist:
        annotations.append("CellTypist")
    if state.has_scimilarity:
        annotations.append("Scimilarity")
    if annotations:
        lines.append("Annotations: " + ", ".join(annotations))

    if state.batch_key:
        batch_info = f"Batch: {state.n_batches} batches (key='{state.batch_key}')"
        if state.batch_correction_applied:
            batch_info += f", corrected with {state.batch_correction_method}"
        lines.append(batch_info)
    elif state.metadata_candidates:
        top = state.metadata_candidates[0]
        lines.append(
            f"Metadata candidate: {top.column} looks {top.role}-like "
            f"({top.n_unique} groups, confidence {top.confidence:.2f})"
        )

    return "\n".join(lines)
