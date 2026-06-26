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

from ..core.inspector import inspect_data, rank_obs_semantic_candidates


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
    # Derived from a context string the *model/tool* supplied (e.g. the `context`
    # arg of load_data/inspect_data), NOT from the user. Kept separate so the
    # model can't mistake its own guessed context for user ground truth.
    context_supplied: Dict[str, Any] = field(default_factory=dict)
    metadata_derived: Dict[str, Any] = field(default_factory=dict)
    marker_inferred: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {k: v for k, v in payload.items() if v not in (None, [], {}, "")}


def _normalize_text(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _species_from_text(text_context: str) -> tuple[str, str]:
    """Infer species from explicit user/request text."""
    if not text_context:
        return "unknown", "unknown"
    mouse_hit = bool(re.search(r"\b(mouse|murine|mus musculus|mm10|mm39)\b", text_context))
    human_hit = bool(re.search(r"\b(human|homo sapiens|hg19|hg38|grch37|grch38)\b", text_context))
    if mouse_hit and not human_hit:
        return "mouse", "user_provided"
    if human_hit and not mouse_hit:
        return "human", "user_provided"
    if mouse_hit and human_hit:
        return "unknown", "ambiguous_user_text"
    return "unknown", "unknown"


def _metadata_species(adata: AnnData) -> tuple[str, str]:
    """Infer species from explicit AnnData metadata."""
    organism = str(adata.uns.get("organism", adata.uns.get("species", ""))).lower()
    if any(token in organism for token in ("mouse", "murine", "mus musculus")):
        return "mouse", "metadata_derived"
    if any(token in organism for token in ("human", "homo sapiens")):
        return "human", "metadata_derived"
    return "unknown", "unknown"


def _genome_column_species(adata: AnnData) -> tuple[str, str, List[str]]:
    """Infer species from the ``genome`` column 10x CellRanger writes into ``var``.

    CellRanger ``.h5`` outputs tag each feature with the reference assembly
    (e.g. ``GRCh38`` for human, ``mm10``/``GRCm39`` for mouse). This is the most
    authoritative species signal when present. A barnyard / multi-reference
    dataset carries more than one assembly and is reported as ambiguous.

    Returns ``(species, source, observed_values)``.
    """
    col = next((k for k in ("genome", "Genome") if k in adata.var.columns), None)
    if col is None:
        return "unknown", "unknown", []
    try:
        values = [str(v).strip() for v in adata.var[col].astype(str).unique() if str(v).strip()]
    except Exception:
        return "unknown", "unknown", []
    if not values:
        return "unknown", "unknown", []

    human_tokens = ("grch38", "grch37", "hg19", "hg38", "homo", "human")
    mouse_tokens = ("grcm39", "grcm38", "mm39", "mm10", "mm9", "mus", "mouse")
    species_found = set()
    for v in values:
        lv = v.lower()
        if any(t in lv for t in human_tokens):
            species_found.add("human")
        elif any(t in lv for t in mouse_tokens):
            species_found.add("mouse")
    if len(species_found) == 1:
        return species_found.pop(), "var_genome_column", values
    if len(species_found) > 1:
        return "unknown", "conflicting_genome_column", values
    # Genome column present but the assembly string is unrecognized — let the
    # downstream gene-identifier / marker checks decide rather than blocking.
    return "unknown", "unknown", values


def _infer_species(adata: AnnData, text_context: str = "") -> tuple[str, str, Dict[str, Any]]:
    """Infer species from user text, metadata, the genome column, and gene identifiers."""
    evidence: Dict[str, Any] = {}

    text_species, text_source = _species_from_text(text_context)
    if text_source == "user_provided":
        evidence["user_text_species"] = text_species
        return text_species, text_source, evidence
    if text_source == "ambiguous_user_text":
        evidence["ambiguous_user_text"] = True
        return "unknown", text_source, evidence

    metadata_species, metadata_source = _metadata_species(adata)
    if metadata_source != "unknown":
        evidence["metadata_species"] = metadata_species
        return metadata_species, metadata_source, evidence

    # 10x CellRanger genome assembly column — authoritative when present.
    genome_species, genome_source, genome_values = _genome_column_species(adata)
    if genome_values:
        evidence["genome_column_values"] = genome_values
    if genome_source == "var_genome_column":
        evidence["genome_column_species"] = genome_species
        return genome_species, genome_source, evidence
    if genome_source == "conflicting_genome_column":
        evidence["conflicting_genome_column"] = True
        return "unknown", genome_source, evidence

    sample_names = [str(name) for name in adata.var_names[:50000]]
    sample_var_values: List[str] = []
    for key in ("gene_ids", "ensembl_id", "gene_symbols", "gene_name"):
        if key in adata.var.columns:
            sample_var_values.extend([str(v) for v in adata.var[key].astype(str).head(50000).tolist()])

    combined = sample_names + sample_var_values
    if not combined:
        return "unknown", "unknown", evidence

    upper_symbol_like = sum(1 for name in sample_names if re.match(r"^[A-Z0-9-]{2,}$", name))
    title_symbol_like = sum(1 for name in sample_names if re.match(r"^[A-Z][a-z0-9-]{1,}$", name))
    ensg = sum(1 for value in combined if value.startswith("ENSG"))
    ensmusg = sum(1 for value in combined if value.startswith("ENSMUSG"))
    # Mouse MHC genes are hyphenated (H2-K1, H2-D1, H2-Aa). Require the hyphen so
    # this does NOT match human histone genes (H2AFZ, H2AC6, H2BC12), which are
    # abundant and would otherwise collide with HLA to read as "conflicting".
    h2_genes = sum(1 for name in sample_names if re.match(r"^H2-[A-Za-z0-9]+", name))
    hla_genes = sum(1 for name in sample_names if re.match(r"^HLA[-A-Za-z0-9]*", name))

    evidence.update({
        "n_genes_checked": len(sample_names),
        "upper_symbol_like": upper_symbol_like,
        "title_symbol_like": title_symbol_like,
        "ensg": ensg,
        "ensmusg": ensmusg,
        "h2_genes": h2_genes,
        "hla_genes": hla_genes,
    })

    if ensmusg > 0 and ensg == 0:
        return "mouse", "gene_identifier", evidence
    if ensg > 0 and ensmusg == 0:
        return "human", "gene_identifier", evidence
    if h2_genes > 0 and hla_genes == 0:
        return "mouse", "marker_gene_evidence", evidence
    if hla_genes > 0 and h2_genes == 0:
        return "human", "marker_gene_evidence", evidence
    if ensmusg > 0 and ensg > 0:
        return "unknown", "conflicting_gene_identifiers", evidence
    if h2_genes > 0 and hla_genes > 0:
        return "unknown", "conflicting_marker_gene_evidence", evidence
    # Gene-symbol casing is not reliable enough to choose species. Some mouse
    # pipelines uppercase symbols, and some mixed references have inconsistent
    # casing. Record the evidence, but require user text, metadata, gene IDs, or
    # species-specific marker families before selecting human or mouse.
    if title_symbol_like > upper_symbol_like * 1.3 or upper_symbol_like >= max(50, title_symbol_like * 5):
        return "unknown", "gene_name_case_heuristic_ambiguous", evidence
    return "unknown", "unknown", evidence


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
    ranked = rank_obs_semantic_candidates(adata, roles={"cell_type"}).get("cell_type", [])
    annotation_key = ranked[0].column if ranked else None

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
    hint_context: Optional[str] = None,
    _precomputed_state=None,
) -> BiologicalContext:
    """
    Infer biological context from user text, metadata, and coarse annotations.

    Provenance stays explicit so downstream interpretation can tell what came
    from user hints versus metadata versus marker/annotation heuristics.

    Parameters
    ----------
    _precomputed_state : DataState, optional
        Pre-computed inspect_data result. Pass this when the caller already ran
        inspect_data to avoid a redundant (and expensive) second matrix scan.
    """
    state = _precomputed_state if _precomputed_state is not None else inspect_data(adata)
    context = BiologicalContext()

    # `text_context` is genuine user text (the request); `hint_context` is text a
    # tool/model supplied (e.g. load_data's `context` arg). Species/sample_type
    # use only the user text, so a model-guessed hint can't masquerade as a
    # user_provided species. Tissue/condition fall back to the hint text but are
    # then labeled `context_supplied`, never `user_provided`.
    normalized_text = _normalize_text(text_context)
    normalized_hint = _normalize_text(hint_context)

    species, species_source, species_evidence = _infer_species(adata, normalized_text)
    context.species = species
    if species_source != "unknown":
        context.provenance["species"] = species_source
        if species_source == "user_provided":
            context.user_provided["species"] = species
        elif species_source in {"metadata_derived", "gene_identifier", "marker_gene_evidence"}:
            context.metadata_derived["species"] = species
            if species_evidence:
                context.metadata_derived["species_evidence"] = species_evidence
        else:
            context.notes.append(f"Species could not be resolved automatically: {species_source}.")
    elif species_evidence:
        context.metadata_derived["species_evidence"] = species_evidence

    sample_type, sample_source = _infer_sample_type(normalized_text, state.data_type)
    context.sample_type = sample_type
    if sample_source != "unknown":
        context.provenance["sample_type"] = sample_source
        if sample_source == "user_provided":
            context.user_provided["sample_type"] = sample_type
        else:
            context.metadata_derived["sample_type"] = sample_type

    tissue, tissue_source = _infer_tissue_from_text(normalized_text)
    if not tissue and normalized_hint:
        hint_tissue, _ = _infer_tissue_from_text(normalized_hint)
        if hint_tissue:
            tissue, tissue_source = hint_tissue, "context_supplied"
    if tissue:
        context.tissue = tissue
        context.provenance["tissue"] = tissue_source or "user_provided"
        if tissue_source == "context_supplied":
            context.context_supplied["tissue"] = tissue
        else:
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
    if condition_source == "unknown" and normalized_hint:
        hint_condition, hint_condition_source = _infer_condition_from_text(normalized_hint)
        if hint_condition_source != "unknown":
            condition, condition_source = hint_condition, "context_supplied"
    context.condition = condition
    if condition_source != "unknown":
        context.provenance["condition"] = condition_source
        if condition_source == "context_supplied":
            context.context_supplied["condition"] = condition
        else:
            context.user_provided["condition"] = condition

    context.expected_celltypes = _expected_celltypes_for_tissue(context.tissue)

    confidence = 0.0
    if context.provenance.get("tissue") == "user_provided":
        confidence += 0.35
    elif context.provenance.get("tissue") == "context_supplied":
        confidence += 0.15
        context.notes.append("Tissue came from a supplied context string, not the user — treat as provisional and verify against gene/annotation evidence.")
    elif context.provenance.get("tissue") == "marker_inferred":
        confidence += 0.2
        context.notes.append("Tissue context is inferred from broad annotation composition and should be treated as provisional.")
    if context.provenance.get("species") in {"user_provided", "metadata_derived", "gene_identifier", "marker_gene_evidence"}:
        confidence += 0.25
    if context.provenance.get("sample_type") in {"user_provided", "metadata_derived"}:
        confidence += 0.15
    if context.provenance.get("condition") == "user_provided":
        confidence += 0.15
    elif context.provenance.get("condition") == "context_supplied":
        confidence += 0.07
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
