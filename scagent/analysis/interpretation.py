"""
Structured pathway interpretation for scagent.

Phase 5 turns pathway reporting into a reusable, evidence-aware analysis layer
so markdown and JSON outputs can share the same constrained interpretation
object instead of rebuilding the reasoning inline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class PathwayInterpretation:
    pathway: str
    direction: str

    nes: Optional[float] = None
    fdr: Optional[float] = None
    statistical_confidence: str = "weak"

    cell_type_consistent: bool = False
    tissue_consistent: bool = False
    plausibility: str = "uncertain"

    leading_genes: List[str] = field(default_factory=list)
    literature_support: int = 0
    review_support: int = 0

    biological_meaning: str = ""
    caveats: List[str] = field(default_factory=list)
    suggested_validation: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {}, "")}


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def pathway_function_hint(pathway_term: str) -> Optional[str]:
    """Return a lightweight biological interpretation hint for common pathway families."""
    term = (pathway_term or "").lower()
    hints = [
        (["allograft", "rejection"], "broad immune activation, antigen presentation, and cytotoxic lymphocyte programs rather than literal transplant biology"),
        (["p53"], "cell stress, DNA damage response, and apoptosis-related programs"),
        (["il2", "stat5"], "cytokine signaling and immune activation programs"),
        (["tnf", "nf-kb"], "inflammatory signaling and innate immune activation"),
        (["tnf", "nfkb"], "inflammatory signaling and innate immune activation"),
        (["interferon", "gamma"], "interferon-driven inflammatory and antigen-presentation responses"),
        (["interferon"], "interferon-driven antiviral and inflammatory responses"),
        (["oxidative", "phosphorylation"], "mitochondrial respiration and energy metabolism"),
        (["glycolysis"], "glycolytic metabolism and rapid energy-demand programs"),
        (["hypoxia"], "cellular hypoxia and stress adaptation"),
        (["mtor"], "growth, nutrient sensing, and anabolic signaling"),
        (["e2f"], "cell-cycle progression and proliferation"),
        (["apoptosis"], "programmed cell death and survival control"),
        (["apical", "junction"], "cell adhesion, cytoskeletal remodeling, and tissue-interaction programs"),
        (["kras"], "RAS/MAPK-linked activation and signaling programs"),
        (["coagulation"], "coagulation-linked inflammatory and immunothrombotic programs"),
        (["complement"], "innate immune complement activation and inflammatory effector programs"),
        (["myc"], "growth, biosynthetic activity, and proliferative drive"),
        (["epithelial", "mesenchymal"], "motility, adhesion remodeling, and mesenchymal-like transition programs"),
    ]
    for keywords, hint in hints:
        if all(keyword in term for keyword in keywords):
            return hint
    return None


def _statistical_confidence(fdr: Optional[float]) -> str:
    if fdr is None:
        return "weak"
    if fdr < 0.05:
        return "strong"
    if fdr < 0.25:
        return "moderate"
    return "weak"


def _cluster_consistency(cluster_context: Dict[str, Any]) -> bool:
    agreement = cluster_context.get("annotation_agreement")
    marker_support = cluster_context.get("marker_support")
    if agreement in {"aligned", "broadly_aligned"} and marker_support in {"strong", "moderate"}:
        return True
    return False


def _tissue_consistency(pathway_term: str, biological_context: Optional[Dict[str, Any]]) -> bool:
    if not biological_context:
        return False
    tissue = (biological_context.get("tissue") or "").lower()
    term = (pathway_term or "").lower()
    if tissue == "pbmc":
        mismatched = ["epithelial", "neuron", "hepat", "cardiac"]
        return not any(token in term for token in mismatched)
    return True


def _plausibility(
    *,
    statistical_confidence: str,
    cell_type_consistent: bool,
    tissue_consistent: bool,
    literature_support: int,
    confidence_level: str,
) -> str:
    if statistical_confidence == "strong" and cell_type_consistent and tissue_consistent and literature_support > 0:
        return "expected"
    if statistical_confidence in {"strong", "moderate"} and (cell_type_consistent or tissue_consistent):
        if confidence_level == "low":
            return "provisional"
        return "plausible"
    if literature_support > 0 and (cell_type_consistent or tissue_consistent):
        return "plausible"
    return "uncertain"


def _meaning_sentence(
    pathway_term: str,
    direction: str,
    cell_type: str,
    hint: Optional[str],
) -> str:
    direction_phrase = {
        "upregulated": "relative enrichment",
        "downregulated": "relative depletion",
    }.get(direction, "altered activity")
    if hint:
        return f"{direction_phrase.capitalize()} of {pathway_term} in {cell_type} is most consistent with {hint}."
    return f"{direction_phrase.capitalize()} of {pathway_term} in {cell_type} should be interpreted cautiously in the current biological context."


def _validation_suggestions(
    pathway_term: str,
    leading_genes: List[str],
    cluster_context: Dict[str, Any],
    interpretation_cautions: List[str],
) -> List[str]:
    term = (pathway_term or "").lower()
    suggestions: List[str] = []
    if leading_genes:
        suggestions.append(f"Inspect expression of leading-edge genes: {', '.join(leading_genes[:3])}.")
    if "interferon" in term:
        suggestions.append("Review interferon-response markers on UMAP or dotplot for this cluster.")
    elif "coagulation" in term:
        suggestions.append("Check platelet or immunothrombotic markers to verify whether the coagulation signal matches the inferred lineage.")
    elif "allograft" in term:
        suggestions.append("Treat this as a broad immune-activation program and compare with cytotoxic and antigen-presentation markers.")
    elif "apical" in term or "junction" in term:
        suggestions.append("Inspect adhesion and cytoskeletal genes to verify whether the signal reflects migration or tissue interaction.")

    if cluster_context.get("marker_support") in {"weak", "absent"}:
        suggestions.append("Recheck canonical markers before making a fine-grained biological claim.")
    if cluster_context.get("annotation_agreement") in {"myeloid_disagreement", "lymphoid_disagreement", "conflicting"}:
        suggestions.append("Compare annotation methods and marker panels before treating the cluster identity as settled.")
    if interpretation_cautions:
        suggestions.append("Review the listed cluster caveats before using this pathway as a strong biological conclusion.")
    return suggestions[:4]


def infer_pathway_interpretation(
    pathway: Dict[str, Any],
    cluster_context: Dict[str, Any],
    research_data: Dict[str, Any],
    *,
    biological_context: Optional[Dict[str, Any]] = None,
    interpretation_cautions: Optional[List[str]] = None,
) -> PathwayInterpretation:
    """Build a constrained, structured interpretation for one pathway call."""
    term = pathway.get("term", "pathway")
    direction = pathway.get("direction", "unknown")
    leading_genes = list(pathway.get("genes", [])[:5])
    fdr = pathway.get("fdr")
    confidence_level = cluster_context.get("confidence_level", "low")
    selected_papers = research_data.get("findings", {}).get("selected_papers", [])
    reviews = research_data.get("findings", {}).get("review_articles", [])
    hint = pathway_function_hint(term)
    caveats = list(interpretation_cautions or [])

    if confidence_level == "low":
        caveats.append("Cluster-confidence checks were low.")
    elif confidence_level == "moderate":
        caveats.append("Cluster-confidence checks were moderate.")

    if cluster_context.get("annotation_agreement") in {"myeloid_disagreement", "lymphoid_disagreement", "conflicting"}:
        caveats.append("Annotation sources do not fully agree on cluster identity.")
    if cluster_context.get("marker_support") in {"weak", "absent"}:
        caveats.append("Marker support for the inferred lineage is limited.")
    if fdr is not None and fdr >= 0.25:
        caveats.append("This pathway did not pass the default significance threshold.")
    if not selected_papers:
        caveats.append("Recent primary literature support was limited.")
    caveats = _dedupe_keep_order(caveats)

    statistical_confidence = _statistical_confidence(fdr)
    cell_type_consistent = _cluster_consistency(cluster_context)
    tissue_consistent = _tissue_consistency(term, biological_context)
    plausibility = _plausibility(
        statistical_confidence=statistical_confidence,
        cell_type_consistent=cell_type_consistent,
        tissue_consistent=tissue_consistent,
        literature_support=len(selected_papers),
        confidence_level=confidence_level,
    )

    meaning = _meaning_sentence(
        term,
        direction,
        cluster_context.get("cell_type", "this cluster"),
        hint,
    )
    suggestions = _dedupe_keep_order(
        _validation_suggestions(term, leading_genes, cluster_context, caveats)
    )

    return PathwayInterpretation(
        pathway=term,
        direction=direction,
        nes=pathway.get("nes"),
        fdr=fdr,
        statistical_confidence=statistical_confidence,
        cell_type_consistent=cell_type_consistent,
        tissue_consistent=tissue_consistent,
        plausibility=plausibility,
        leading_genes=leading_genes,
        literature_support=len(selected_papers),
        review_support=len(reviews),
        biological_meaning=meaning,
        caveats=caveats,
        suggested_validation=suggestions,
    )
