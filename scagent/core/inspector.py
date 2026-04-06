"""
Data state inspector for scagent.

This is the MOST CRITICAL module - it detects the current state of the data
and recommends what analysis steps are needed to reach a user's goal.
"""

from dataclasses import dataclass, field
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
    is_counts: bool = False  # True if X contains integer counts

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
    """Check if matrix contains integer values (counts)."""
    if sp.issparse(X):
        data = X.data
    else:
        data = X.flatten()

    sample_size = min(10000, len(data))
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        data = data[indices]

    return np.allclose(data, np.round(data))


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


def _column_name_role_scores(column: str) -> Dict[str, float]:
    """Score how strongly a column name suggests each metadata role."""
    normalized = _normalize_column_name(column)
    tokens = set(filter(None, normalized.split("_")))
    scores: Dict[str, float] = {}

    for role, aliases in METADATA_ROLE_ALIASES.items():
        normalized_aliases = {_normalize_column_name(alias) for alias in aliases}
        if normalized in normalized_aliases:
            scores[role] = 1.0
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
    """Detect if a raw counts layer exists."""
    common_raw_names = ["raw_counts", "raw_data", "counts", "raw"]

    for name in common_raw_names:
        if name in adata.layers and _is_integer_matrix(adata.layers[name]):
            return True, name

    return False, ""


def _detect_gene_id_format(adata: AnnData) -> Tuple[str, bool, bool, List[str]]:
    """
    Detect the format of gene identifiers.

    Returns: (format, has_symbols, has_ensembl, sample_names)
    """
    gene_names = adata.var_names.tolist()
    sample = gene_names[:20]

    ensembl_pattern = re.compile(r"^ENS[A-Z]{0,3}G\d{11}")
    entrez_pattern = re.compile(r"^\d{1,8}$")
    symbol_pattern = re.compile(r"^[A-Z][A-Z0-9-]{1,15}$", re.IGNORECASE)

    ensembl_count = sum(1 for gene in sample if ensembl_pattern.match(str(gene)))
    entrez_count = sum(1 for gene in sample if entrez_pattern.match(str(gene)))
    symbol_count = sum(
        1
        for gene in sample
        if symbol_pattern.match(str(gene)) and not ensembl_pattern.match(str(gene))
    )

    has_symbols = "gene_symbols" in adata.var.columns or "gene_name" in adata.var.columns
    has_ensembl = "gene_ids" in adata.var.columns or "ensembl_id" in adata.var.columns

    if ensembl_count > len(sample) * 0.5:
        fmt = "ensembl"
    elif entrez_count > len(sample) * 0.5:
        fmt = "entrez"
    elif symbol_count > len(sample) * 0.3:
        fmt = "symbol"
    else:
        fmt = "mixed" if (ensembl_count > 0 and symbol_count > 0) else "unknown"

    return fmt, has_symbols or fmt == "symbol", has_ensembl or fmt == "ensembl", sample[:5]


def _detect_normalization(adata: AnnData) -> Tuple[bool, bool, str]:
    """
    Detect if data is normalized and log-transformed.

    Returns: (is_normalized, is_log_transformed, method)
    """
    method = ""
    if "log1p" in adata.uns:
        method = "log1p"

    if sp.issparse(adata.X):
        max_val = adata.X.max()
        sample_data = adata.X.data[:min(10000, len(adata.X.data))]
    else:
        max_val = np.max(adata.X)
        sample_data = adata.X.flatten()[:10000]

    has_floats = not np.allclose(sample_data, np.round(sample_data))
    is_log = max_val < 15 and has_floats
    is_normalized = has_floats

    return is_normalized, is_log, method


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

    state.has_raw_layer, state.raw_layer_name = _detect_raw_layer(adata)
    state.is_counts = _is_integer_matrix(adata.X)

    gene_fmt, has_sym, has_ens, sample_genes = _detect_gene_id_format(adata)
    state.gene_id_format = gene_fmt
    state.has_gene_symbols = has_sym
    state.has_ensembl_ids = has_ens
    state.sample_gene_names = sample_genes

    qc_obs_cols = ["n_genes_by_counts", "total_counts", "n_genes"]
    state.has_qc_metrics = any(column in adata.obs.columns for column in qc_obs_cols)
    state.has_mt_metrics = "pct_counts_mt" in adata.obs.columns
    state.has_ribo_metrics = "pct_counts_ribo" in adata.obs.columns

    if "doublet_score" in adata.obs.columns or "predicted_doublet" in adata.obs.columns:
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

    if not state.has_raw_layer and state.is_counts:
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
    if state.has_raw_layer:
        processing.append(f"raw counts in '{state.raw_layer_name}'")
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
