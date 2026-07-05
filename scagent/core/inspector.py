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
    # Representation the clustering was computed on (e.g. "X_pca", "X_scVI").
    # Used to enforce that annotation binds to a post-integration clustering.
    use_rep: Optional[str] = None


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
    raw_is_counts: bool = False    # True if adata.raw.X holds integer-valued counts
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
    if record.use_rep:
        payload["use_rep"] = record.use_rep
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


def _sample_matrix_values(X, n: int = 20000, seed: int = 0) -> np.ndarray:
    """Return a representative sample of a matrix's values as a host numpy array.

    Samples RANDOMLY across the full data rather than taking a head slice: the
    first stored values of a CSR matrix are just the first few cells, so a head
    slice can misjudge integer-ness / range for the matrix as a whole. For sparse
    matrices only the stored (nonzero) values are sampled — zeros are trivially
    integer and irrelevant to "are these counts / are there decimals". Returns an
    empty array when there is nothing to sample.
    """
    # cupy/cupyx arrays (e.g. left in a layer by an interrupted GPU step) are not
    # recognised by scipy.issparse and have no .ravel(); pull to host first.
    if type(X).__module__.split(".", 1)[0] in ("cupy", "cupyx"):
        try:
            gpu = X.data if hasattr(X, "data") else X.reshape(-1)
            data = np.asarray(gpu.get())
        except Exception:
            return np.array([])
    elif sp.issparse(X):
        data = X.data
    else:
        # Dense array, or an exotic/backed matrix type. Guard the conversion so an
        # object we can't materialise (e.g. a backed _CSRDataset) yields an empty
        # sample rather than crashing inspection.
        try:
            data = np.asarray(X).ravel()
        except Exception:
            return np.array([])

    m = len(data)
    if m == 0:
        return np.array([])
    if m <= n:
        return np.asarray(data)
    idx = np.sort(np.random.default_rng(seed).choice(m, size=n, replace=False))
    return np.asarray(data[idx])


def _is_integer_matrix(X) -> bool:
    """Check whether a matrix holds integer-valued counts.

    The decision is made from VALUES, not dtype: float32 that is entirely
    integer-valued (e.g. 1.0, 20.0, 5643.0 — common for CELLxGENE raw counts) is
    treated as counts. Only an integer dtype is a fast shortcut; float dtypes are
    always sampled (representatively) and tested for integer-ness. Empty matrices
    are treated as counts (conservative, non-fatal).
    """
    dtype = getattr(X, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.integer):
        return True
    sample = _sample_matrix_values(X)
    if len(sample) == 0:
        return True
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
    # Identifier-like columns (cell barcodes, per-cell IDs) are never label
    # columns, no matter how their name scores. A cell_type/cluster column is a
    # categorical label with bounded cardinality; a near-unique column is an
    # identifier. Without this, e.g. `cell_barcode` (≈unique per cell) gets a
    # cell_type role solely from sharing the token "cell" with the role aliases.
    if role in {"cell_type", "cluster"} and unique_fraction >= 0.65:
        return 0.0
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

    # doublet_label is a small, closed vocabulary (singlet/doublet/true/false/0/1).
    # Require the VALUES to actually look like doublet calls — not just a fuzzy
    # name match. Otherwise a cell-type column like 'scanvi_label' matches on the
    # "label" substring (name_score 0.63) and falsely flips has_doublets=True with
    # no real doublet call (run_2026_07_02_150701 screenshot). The canonical named
    # columns ('predicted_doublet', 'doublet_score') are matched exactly elsewhere.
    if role == "doublet_label" and value_score == 0.0:
        return None

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
    """Detect a genuine raw-counts LAYER in adata.layers.

    Only reports real named layers. adata.raw is deliberately NOT reported here:
    it is tracked separately via has_raw / raw_is_counts / raw_n_vars. Folding
    adata.raw in as a fake "__raw__" layer used to make inspection claim "raw
    counts in layer '__raw__'", a layer that does not exist — which sent the model
    chasing a non-existent layer (run_2026_07_02_150701 burned ~5 iterations).
    """
    common_raw_names = ["raw_counts", "raw_data", "counts", "raw"]

    for name in common_raw_names:
        if name in adata.layers and _is_integer_matrix(adata.layers[name]):
            return True, name

    return False, ""


# Preferred layer names to search for a raw-counts matrix, most-specific first.
_COUNTS_LAYER_NAMES = ["counts", "raw_counts", "raw_data", "soupx_counts", "spliced"]


def find_counts_matrix(adata: AnnData, prefer_layer: Optional[str] = None):
    """Locate a raw-counts matrix and the ``var`` frame that matches it.

    Counts can live in a named layer, in ``adata.raw`` (which carries its *own*
    ``var`` — often more genes than ``adata.var`` after HVG subsetting), or in
    ``adata.X`` itself. This resolver returns whichever is integer-VALUED — the
    check is on the values, not the dtype, because counts are frequently stored
    as ``float32`` (a naive "X is float ⇒ no counts" test misses ``adata.raw``
    entirely, which is exactly how SCimilarity was fed log-normalized data).

    Parameters
    ----------
    adata : AnnData
    prefer_layer : str, optional
        A specific layer to use if it exists and is integer-valued.

    Returns
    -------
    dict or None
        ``{"X", "var", "source", "n_vars"}`` where ``source`` is
        ``"layer:<name>"``, ``"raw"``, or ``"X"``; or None if no integer-valued
        counts matrix is found anywhere.
    """
    candidates: List[str] = []
    if prefer_layer:
        candidates.append(prefer_layer)
    candidates += [n for n in _COUNTS_LAYER_NAMES if n != prefer_layer]

    for name in candidates:
        if name in adata.layers and _is_integer_matrix(adata.layers[name]):
            return {
                "X": adata.layers[name],
                "var": adata.var,
                "source": f"layer:{name}",
                "n_vars": adata.n_vars,
            }

    if adata.raw is not None and _is_integer_matrix(adata.raw.X):
        return {
            "X": adata.raw.X,
            "var": adata.raw.var,
            "source": "raw",
            "n_vars": adata.raw.n_vars,
        }

    if _is_integer_matrix(adata.X):
        return {
            "X": adata.X,
            "var": adata.var,
            "source": "X",
            "n_vars": adata.n_vars,
        }

    return None


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

    # Content-validated column detection (recognizes feature_name/gene_symbols/…
    # and rejects mislabelled columns). See core.genes.
    from . import genes as _genes
    symbol_col = _genes.find_symbol_column(adata)
    ensembl_col = _genes.find_ensembl_column(adata)
    has_symbols_col = symbol_col is not None
    has_ensembl_col = ensembl_col is not None

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
            f"{'Genome prefix detected: ' + repr(genome_prefix) + ' — standard detection patterns may not match.' if genome_prefix else 'Inspect var_names format before running QC.'}"
        )
    if len(ribo_genes_found) == 0:
        ribo_warning = (
            f"No RPS/RPL genes found with standard prefix search. "
            f"{'Genome prefix detected: ' + repr(genome_prefix) + ' — standard detection patterns may not match.' if genome_prefix else 'Inspect var_names format before running QC.'}"
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
        # Which var column carries symbols/IDs, and whether var_names can be
        # converted to gene symbols offline (needed by SCimilarity/CellTypist).
        "symbol_column": symbol_col,
        "ensembl_column": ensembl_col,
        "convertible_to_symbols": fmt == "symbol" or symbol_col is not None,
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

    # --- Fallback: representative value sample (not a head slice) ---
    # Decide from values, not dtype: float X that is entirely integer-valued is
    # raw counts, not normalized data.
    sample_data = _sample_matrix_values(adata.X)
    if len(sample_data) == 0:
        return False, False, ""
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
                use_rep=payload.get("use_rep"),
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
    use_rep: Optional[str] = None,
) -> None:
    """Register a clustering result in adata.uns for later inspection."""
    namespace = _ensure_scagent_uns(adata)
    registry = namespace.setdefault(SCAGENT_CLUSTERING_REGISTRY_KEY, {})
    registry[cluster_key] = {
        "method": normalize_clustering_method(method),
        "resolution": float(resolution) if resolution is not None else None,
        "source_key": source_key,
        "created_by": created_by,
        "use_rep": use_rep,
    }


def promote_clustering_to_primary(
    adata: AnnData,
    *,
    cluster_key: str,
    method: str,
    resolution: Optional[float],
    created_by: str = "tool",
    use_rep: Optional[str] = None,
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
        use_rep=use_rep,
    )
    register_clustering(
        adata,
        cluster_key=alias,
        method=normalized_method,
        resolution=resolution,
        created_by=created_by,
        source_key=cluster_key,
        use_rep=use_rep,
    )
    return alias


# Obsm keys produced by batch-integration methods, in the same precedence
# inspect_data uses. Single source of truth for "an integrated embedding exists"
# (method convention, not biology). bbknn corrects the neighbor graph, not an
# embedding, so it is not listed here.
_INTEGRATION_EMBEDDING_KEYS = ("X_scanorama", "X_pca_harmony", "X_scVI")


def integrated_embedding_keys(adata: AnnData) -> List[str]:
    """Return the obsm keys of any batch-corrected embeddings present.

    Used to enforce that, once integration has produced a corrected embedding,
    downstream clustering/annotation binds to it rather than to a stale
    pre-integration representation.
    """
    return [key for key in _INTEGRATION_EMBEDDING_KEYS if key in adata.obsm]


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
        state.raw_is_counts = _is_integer_matrix(adata.raw.X)
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

    if adata.uns.get("bbknn_batch_key") is not None:
        state.batch_correction_applied = True
        state.batch_correction_method = "bbknn"
    elif "X_scanorama" in adata.obsm:
        state.batch_correction_applied = True
        state.batch_correction_method = "scanorama"
    elif "X_pca_harmony" in adata.obsm:
        state.batch_correction_applied = True
        state.batch_correction_method = "harmony"
    elif "X_scVI" in adata.obsm:
        state.batch_correction_applied = True
        state.batch_correction_method = "scvi"

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

        if not state.has_pca:
            steps.append("run_pca")

        if state.n_batches > 1 and not state.batch_correction_applied:
            if goal == "batch_correct":
                steps.append("run_batch_correction")
            else:
                steps.append("assess_batch_strategy")
                if state.has_pca:
                    steps.append("run_batch_correction_if_needed")

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
        if state.raw_is_counts:
            processing.append(f"raw counts in adata.raw{extra}")
        else:
            processing.append(f"adata.raw present{extra} (non-integer values — not raw counts)")
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


def obs_columns_detail(obs_df, n_obs: int, max_values: int = 8) -> dict:
    """
    Compact per-column summary of adata.obs for LLM role inference.

    For each column returns dtype, n_unique, and either:
    - representative unique values (categorical / low-cardinality, n_unique <= 15)
    - min/max/mean stats (continuous numeric, n_unique > 15)

    Columns that are essentially unique per cell (barcodes, index-like) are
    flagged as high_cardinality. All columns are included — no truncation.

    Returns {"columns": {...}, "total_obs_cols": N}
    """
    import math
    import pandas as pd

    cols = list(obs_df.columns)
    total = len(cols)
    detail: dict = {}

    for col in cols:
        series = obs_df[col]
        dtype_str = str(series.dtype)
        non_null = series.dropna()
        n_unique = int(series.nunique(dropna=True))

        if n_unique == 0:
            detail[col] = {"dtype": dtype_str, "n_unique": 0}
            continue

        if n_obs > 0 and n_unique >= n_obs * 0.9 and n_unique > 50:
            detail[col] = {"dtype": dtype_str, "n_unique": n_unique, "note": "high_cardinality"}
            continue

        is_numeric = pd.api.types.is_numeric_dtype(series)
        if is_numeric and n_unique > 15:
            try:
                vals = non_null.to_numpy(dtype=float)
                detail[col] = {
                    "dtype": dtype_str,
                    "n_unique": n_unique,
                    "min": round(float(vals.min()), 4),
                    "max": round(float(vals.max()), 4),
                    "mean": round(float(vals.mean()), 4),
                }
            except Exception:
                detail[col] = {"dtype": dtype_str, "n_unique": n_unique}
        else:
            try:
                unique_vals = non_null.unique().tolist()
                samples = []
                for v in unique_vals:
                    if isinstance(v, float) and math.isnan(v):
                        continue
                    sv = str(v)
                    if len(sv) > 50:
                        sv = sv[:47] + "..."
                    samples.append(sv)
                    if len(samples) >= max_values:
                        break
                entry: dict = {"dtype": dtype_str, "n_unique": n_unique, "values": samples}
                if n_unique > max_values:
                    entry["values_truncated"] = True
                detail[col] = entry
            except Exception:
                detail[col] = {"dtype": dtype_str, "n_unique": n_unique}

    return {"columns": detail, "total_obs_cols": total}


def _truncate_value(value: Any, limit: int = 60) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _column_facts(series, n_ref: int, max_values: int = 10) -> dict:
    """Judgment-free factual summary of one obs/var column.

    Reports dtype, cardinality, missingness, and either a value-count
    distribution (categorical / low-cardinality) or numeric stats. Deliberately
    makes NO role decision — e.g. a near-unique column reports
    ``unique_fraction`` close to 1.0 and lets the consumer conclude it is an
    identifier rather than baking that judgment in here.
    """
    import pandas as pd

    dtype_str = str(series.dtype)
    n_missing = int(series.isna().sum())
    non_null = series.dropna()
    n_unique = int(non_null.nunique())
    facts: dict = {
        "dtype": dtype_str,
        "n_unique": n_unique,
        "n_missing": n_missing,
        "unique_fraction": round(n_unique / n_ref, 4) if n_ref else 0.0,
    }
    if n_unique == 0:
        return facts

    is_numeric = pd.api.types.is_numeric_dtype(series)
    if is_numeric and n_unique > max_values:
        try:
            vals = non_null.to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite):
                facts.update(
                    {
                        "min": round(float(finite.min()), 4),
                        "max": round(float(finite.max()), 4),
                        "mean": round(float(finite.mean()), 4),
                        "all_integer": bool(np.allclose(finite, np.round(finite))),
                    }
                )
        except Exception:
            pass
        return facts

    # Categorical / low-cardinality: a value-count distribution. For an
    # identifier column the top values each have count 1, which together with
    # unique_fraction makes "this is a barcode, not a label" self-evident.
    try:
        value_counts = non_null.value_counts()
        facts["top_values"] = [
            {"value": _truncate_value(idx), "count": int(count)}
            for idx, count in value_counts.head(max_values).items()
        ]
        if n_unique > max_values:
            facts["values_truncated"] = True
    except Exception:
        pass
    return facts


def _x_facts(X, sample_n: int = 20000) -> dict:
    """Factual characterization of a matrix from a representative value sample.

    Reports the VALUE evidence needed to decide "are these raw counts?" without a
    verdict: the fraction of sampled values that are integer-valued, min/max, and
    whether any are negative. dtype is reported too, but the fraction is what
    matters — float32 that is 100% integer-valued (fraction_integer_valued ≈ 1.0,
    min ≥ 0) is raw counts despite the float dtype. For sparse matrices the sample
    is over stored (nonzero) values.
    """
    facts: dict = {"dtype": str(getattr(X, "dtype", "unknown")), "is_sparse": bool(sp.issparse(X))}
    try:
        sample = _sample_matrix_values(X, n=sample_n)
        if len(sample):
            integer_valued = np.isclose(sample, np.round(sample))
            facts["n_sampled"] = int(len(sample))
            facts["sample_min"] = round(float(sample.min()), 4)
            facts["sample_max"] = round(float(sample.max()), 4)
            facts["fraction_integer_valued"] = round(float(np.mean(integer_valued)), 6)
            facts["all_integer_sample"] = bool(integer_valued.all())
            facts["has_negative_sample"] = bool(float(sample.min()) < 0)
    except Exception:
        pass
    return facts


def _gene_namespace_facts(adata: AnnData, sample_n: int = 5000) -> dict:
    """Raw gene-identifier signals (counts only, no species conclusion)."""
    names = [str(name) for name in adata.var_names[:sample_n]]
    if not names:
        return {}
    return {
        "n_checked": len(names),
        "ensembl_human_ensg": sum(1 for n in names if n.startswith("ENSG")),
        "ensembl_mouse_ensmusg": sum(1 for n in names if n.startswith("ENSMUSG")),
        "uppercase_symbol_like": sum(1 for n in names if re.match(r"^[A-Z0-9-]{2,}$", n)),
        "title_symbol_like": sum(1 for n in names if re.match(r"^[A-Z][a-z0-9-]{1,}$", n)),
        "mt_prefixed": sum(1 for n in names if n.upper().startswith("MT-")),
    }


def dataset_facts(adata: AnnData, max_values: int = 10) -> dict:
    """Comprehensive, judgment-free fact sheet for an AnnData object.

    Everything observable without interpretation: shape, X characteristics,
    layers / embeddings / uns keys, raw availability, per-column obs & var facts,
    gene-namespace signals, and example var names. Contains NO role / species /
    "is this cell types" decisions — those belong to the judgment layer, which
    consumes this sheet. Keeping facts and judgments separate is the point:
    facts are cheap, deterministic, and testable; judgments are not.
    """
    n_obs, n_vars = int(adata.n_obs), int(adata.n_vars)
    return {
        "shape": {"n_obs": n_obs, "n_vars": n_vars},
        "X": _x_facts(adata.X),
        "layers": list(adata.layers.keys()),
        # Per-layer value facts so the model can see which matrix (if any) holds
        # integer counts, rather than trusting a pre-computed verdict.
        "layer_facts": {name: _x_facts(adata.layers[name]) for name in adata.layers.keys()},
        "obsm_keys": list(adata.obsm.keys()),
        "varm_keys": list(adata.varm.keys()),
        "uns_keys": list(adata.uns.keys()),
        "raw": {
            "present": adata.raw is not None,
            "n_vars": int(adata.raw.n_vars) if adata.raw is not None else 0,
            # Value facts for adata.raw.X (min/max/all_integer_sample) — lets the
            # model tell "float32 but integer-valued counts" from log-normalized
            # data via the actual values (decimal points), not the dtype alone.
            "X": _x_facts(adata.raw.X) if adata.raw is not None else {},
        },
        "obs_columns": {col: _column_facts(adata.obs[col], n_obs, max_values) for col in adata.obs.columns},
        "var_columns": {col: _column_facts(adata.var[col], n_vars, max_values) for col in adata.var.columns},
        "var_names_examples": [str(name) for name in adata.var_names[:8]],
        "gene_namespace": _gene_namespace_facts(adata),
    }
