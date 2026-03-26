"""
Biological context capture for scagent.

Phase 2 adds a lightweight, provenance-aware biological context layer so
interpretation can depend on more than technical state alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import re

from anndata import AnnData

from ..core.inspector import inspect_data


@dataclass
class BiologicalContext:
    """Biological context for interpretation and literature search."""

    tissue: str = "unknown"
    species: str = "unknown"
    condition: str = "unknown"
    sample_type: str = "unknown"
    expected_celltypes: Optional[List[str]] = None

    inferred_tissue: Optional[str] = None
    confidence: float = 0.0

    provenance: Dict[str, str] = field(default_factory=dict)
    user_provided: Dict[str, Any] = field(default_factory=dict)
    metadata_derived: Dict[str, Any] = field(default_factory=dict)
    marker_inferred: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v not in (None, [], {}, "")}


def _normalize_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _infer_species(adata: AnnData) -> tuple[str, str]:
    """Infer species from gene identifiers with provenance label."""
    sample_names = [str(name) for name in adata.var_names[:100]]
    sample_var_values: List[str] = []
    for key in ("gene_ids", "ensembl_id", "gene_symbols", "gene_name"):
        if key in adata.var.columns:
            sample_var_values.extend([str(v) for v in adata.var[key].astype(str).head(100).tolist()])

    combined = sample_names + sample_var_values
    if not combined:
        return "unknown", "unknown"

    upper_symbol_like = sum(1 for name in sample_names if re.match(r"^[A-Z0-9-]{2,}$", name))
    title_symbol_like = sum(1 for name in sample_names if re.match(r"^[A-Z][a-z0-9-]{1,}$", name))
    ensg = sum(1 for value in combined if value.startswith("ENSG"))
    ensmusg = sum(1 for value in combined if value.startswith("ENSMUSG"))

    if ensmusg > 0 or title_symbol_like > upper_symbol_like * 1.3:
        return "mouse", "metadata_derived"
    if ensg > 0 or upper_symbol_like >= max(10, title_symbol_like):
        return "human", "metadata_derived"
    return "unknown", "unknown"


def _infer_sample_type(text_context: str, detected_type: str) -> tuple[str, str]:
    if any(token in text_context for token in ["nuclei", "nucleus", "snrna", "single nucleus"]):
        return "nuclei", "user_provided"
    if detected_type in {"cells", "nuclei"}:
        return detected_type, "metadata_derived"
    return "unknown", "unknown"


def _infer_tissue_from_text(text_context: str) -> tuple[Optional[str], Optional[str]]:
    mapping = [
        ("pbmc", "PBMC"),
        ("peripheral blood", "PBMC"),
        ("whole blood", "blood"),
        ("bone marrow", "bone marrow"),
        ("spleen", "spleen"),
        ("lymph node", "lymph node"),
        ("thymus", "thymus"),
        ("tumor microenvironment", "tumor"),
        ("tumor", "tumor"),
        ("melanoma", "tumor"),
        ("lung", "lung"),
        ("colon", "colon"),
        ("brain", "brain"),
        ("skin", "skin"),
        ("liver", "liver"),
    ]
    for token, value in mapping:
        if token in text_context:
            return value, "user_provided"
    return None, None


def _infer_condition_from_text(text_context: str) -> tuple[str, str]:
    mapping = [
        (["healthy", "control", "unstimulated"], "healthy"),
        (["stimulated", "activation", "activated"], "stimulated"),
        (["infection", "infected", "viral", "bacterial"], "infection"),
        (["tumor", "cancer", "malignant"], "tumor"),
        (["inflamed", "inflammation", "inflammatory"], "inflammation"),
        (["disease", "patient"], "disease"),
    ]
    for tokens, value in mapping:
        if any(token in text_context for token in tokens):
            return value, "user_provided"
    return "unknown", "unknown"


def _infer_tissue_from_annotations(adata: AnnData) -> tuple[Optional[str], Dict[str, Any]]:
    """
    Infer broad tissue context from annotation composition.

    This stays intentionally conservative. The goal is to identify obvious
    PBMC-like mixtures, not to guess specific tissue identity from weak clues.
    """
    annotation_key = None
    for key in (
        "celltypist_majority_voting",
        "celltypist_predicted_labels",
        "scimilarity_representative_prediction",
        "scimilarity_cluster_majority",
    ):
        if key in adata.obs.columns:
            annotation_key = key
            break

    if annotation_key is None:
        return None, {}

    labels = adata.obs[annotation_key].astype(str).str.lower()
    immune_hits = {
        "t_cell": labels.str.contains("t cell|alpha-beta|gamma-delta|mait|regulatory t").any(),
        "nk": labels.str.contains("nk|natural killer").any(),
        "b_cell": labels.str.contains("b cell|plasma").any(),
        "myeloid": labels.str.contains("monocyte|macrophage|dendritic|dc").any(),
        "platelet": labels.str.contains("platelet|megakary").any(),
    }
    nonimmune_hits = {
        "epithelial": labels.str.contains("epithelial").any(),
        "fibroblast": labels.str.contains("fibroblast").any(),
        "endothelial": labels.str.contains("endothelial").any(),
        "hepatocyte": labels.str.contains("hepatocyte").any(),
        "neuron": labels.str.contains("neuron|glia|astrocyte|oligodendro").any(),
    }

    broad_immune_lineages = sum(bool(v) for v in immune_hits.values())
    broad_nonimmune_lineages = sum(bool(v) for v in nonimmune_hits.values())

    evidence = {
        "annotation_key": annotation_key,
        "immune_lineages": [k for k, v in immune_hits.items() if v],
        "nonimmune_lineages": [k for k, v in nonimmune_hits.items() if v],
    }

    if broad_immune_lineages >= 3 and broad_nonimmune_lineages == 0:
        return "PBMC", evidence
    return None, evidence


def _expected_celltypes_for_tissue(tissue: str) -> Optional[List[str]]:
    normalized = (tissue or "").lower()
    if normalized == "pbmc":
        return [
            "T cells",
            "NK cells",
            "B cells",
            "monocytes",
            "dendritic cells",
            "platelets",
        ]
    if normalized == "tumor":
        return [
            "T cells",
            "NK cells",
            "myeloid cells",
            "tumor cells",
            "stromal cells",
        ]
    return None


def infer_biological_context(
    adata: AnnData,
    *,
    text_context: Optional[str] = None,
) -> BiologicalContext:
    """
    Infer biological context from user text, metadata, and coarse annotations.

    Provenance stays explicit so downstream interpretation can tell what came
    from user hints versus metadata versus marker/annotation heuristics.
    """
    state = inspect_data(adata)
    context = BiologicalContext()

    normalized_text = _normalize_text(text_context)

    species, species_source = _infer_species(adata)
    context.species = species
    if species_source != "unknown":
        context.provenance["species"] = species_source
        if species_source == "metadata_derived":
            context.metadata_derived["species"] = species

    sample_type, sample_source = _infer_sample_type(normalized_text, state.data_type)
    context.sample_type = sample_type
    if sample_source != "unknown":
        context.provenance["sample_type"] = sample_source
        if sample_source == "user_provided":
            context.user_provided["sample_type"] = sample_type
        else:
            context.metadata_derived["sample_type"] = sample_type

    tissue, tissue_source = _infer_tissue_from_text(normalized_text)
    if tissue:
        context.tissue = tissue
        context.provenance["tissue"] = tissue_source or "user_provided"
        context.user_provided["tissue"] = tissue
    else:
        inferred_tissue, evidence = _infer_tissue_from_annotations(adata)
        if inferred_tissue:
            context.tissue = inferred_tissue
            context.inferred_tissue = inferred_tissue
            context.provenance["tissue"] = "marker_inferred"
            context.marker_inferred["tissue"] = inferred_tissue
            if evidence:
                context.marker_inferred["tissue_evidence"] = evidence

    condition, condition_source = _infer_condition_from_text(normalized_text)
    context.condition = condition
    if condition_source != "unknown":
        context.provenance["condition"] = condition_source
        context.user_provided["condition"] = condition

    context.expected_celltypes = _expected_celltypes_for_tissue(context.tissue)

    confidence = 0.0
    if context.provenance.get("tissue") == "user_provided":
        confidence += 0.35
    elif context.provenance.get("tissue") == "marker_inferred":
        confidence += 0.2
        context.notes.append("Tissue context is inferred from broad annotation composition and should be treated as provisional.")
    if context.provenance.get("species") == "metadata_derived":
        confidence += 0.25
    if context.provenance.get("sample_type") in {"user_provided", "metadata_derived"}:
        confidence += 0.15
    if context.provenance.get("condition") == "user_provided":
        confidence += 0.15
    if context.tissue == "unknown":
        context.notes.append("Tissue context was not explicit; literature search may be broader than ideal.")
    if context.condition == "unknown":
        context.notes.append("Condition/perturbation context was not provided.")
    context.confidence = round(min(confidence, 1.0), 2)

    return context


def context_query_hint(context: BiologicalContext | Dict[str, Any]) -> str:
    """Render a compact context string for literature search or reports."""
    if isinstance(context, dict):
        tissue = context.get("tissue", "unknown")
        sample_type = context.get("sample_type", "unknown")
        species = context.get("species", "unknown")
        condition = context.get("condition", "unknown")
    else:
        tissue = context.tissue
        sample_type = context.sample_type
        species = context.species
        condition = context.condition

    parts = []
    if tissue != "unknown":
        parts.append(tissue)
    if sample_type != "unknown":
        parts.append(sample_type)
    if species != "unknown":
        parts.append(species)
    if condition != "unknown":
        parts.append(condition)
    return ", ".join(parts)
