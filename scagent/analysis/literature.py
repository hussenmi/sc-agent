"""
Literature relevance scoring for pathway interpretation.

Phase 4 formalizes how PubMed hits are ranked so evidence selection can use
biological context consistently rather than relying only on free-text heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import re

from .cluster_confidence import normalize_annotation_lineage


@dataclass
class LiteratureContextProfile:
    tissue: str = "unknown"
    condition: str = "unknown"
    species: str = "unknown"
    sample_type: str = "unknown"
    lineage: str = "unknown"
    cluster_confidence: Optional[float] = None
    cluster_confidence_level: str = "unknown"
    tissue_terms: List[str] = field(default_factory=list)
    condition_terms: List[str] = field(default_factory=list)
    species_terms: List[str] = field(default_factory=list)
    broad_cell_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {}, "")}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def build_literature_context(
    context_text: str,
    *,
    cell_type: str = "",
    cluster_confidence: Optional[float] = None,
) -> LiteratureContextProfile:
    """Parse lightweight biological context from the request/report context."""
    text = _normalize_text(context_text)
    lineage = normalize_annotation_lineage(cell_type)

    profile = LiteratureContextProfile(
        lineage=lineage,
        cluster_confidence=cluster_confidence,
    )

    if cluster_confidence is not None:
        if cluster_confidence >= 0.8:
            profile.cluster_confidence_level = "high"
        elif cluster_confidence >= 0.55:
            profile.cluster_confidence_level = "moderate"
        else:
            profile.cluster_confidence_level = "low"

    tissue_map = [
        (["pbmc", "peripheral blood"], ("PBMC", ["pbmc", "peripheral blood mononuclear", "blood"])),
        (["tumor", "melanoma", "carcinoma", "adenocarcinoma", "microenvironment"], ("tumor", ["tumor", "tumour", "microenvironment", "cancer"])),
        (["bone marrow"], ("bone marrow", ["bone marrow"])),
        (["spleen"], ("spleen", ["spleen", "splenic"])),
        (["lymph node"], ("lymph node", ["lymph node", "nodal"])),
        (["lung"], ("lung", ["lung", "pulmonary"])),
        (["brain"], ("brain", ["brain", "cns", "central nervous system"])),
        (["liver"], ("liver", ["liver", "hepatic"])),
    ]
    for tokens, (label, terms) in tissue_map:
        if any(token in text for token in tokens):
            profile.tissue = label
            profile.tissue_terms = terms
            break

    condition_map = [
        (["healthy", "control", "unstimulated"], ("healthy", ["healthy", "control", "baseline", "unstimulated"])),
        (["stimulated", "activated", "activation"], ("stimulated", ["stimulated", "activated", "activation"])),
        (["inflammation", "inflamed", "inflammatory"], ("inflammation", ["inflammation", "inflammatory"])),
        (["infection", "infected", "viral", "bacterial"], ("infection", ["infection", "infected", "viral", "bacterial"])),
        (["disease", "patient"], ("disease", ["disease", "patient"])),
    ]
    for tokens, (label, terms) in condition_map:
        if any(token in text for token in tokens):
            profile.condition = label
            profile.condition_terms = terms
            break

    if "human" in text:
        profile.species = "human"
        profile.species_terms = ["human"]
    elif "mouse" in text or "murine" in text or "mus musculus" in text:
        profile.species = "mouse"
        profile.species_terms = ["mouse", "murine", "mus musculus"]

    if "nuclei" in text or "single nucleus" in text or "snrna" in text:
        profile.sample_type = "nuclei"
    elif "cells" in text or "single cell" in text or "scrna" in text:
        profile.sample_type = "cells"

    broad_terms = {
        "t_cell": ["t cell", "lymphocyte"],
        "nk": ["natural killer", "nk cell", "lymphocyte"],
        "b_cell": ["b cell", "lymphocyte"],
        "plasma": ["plasma cell", "b cell"],
        "monocyte": ["monocyte", "myeloid"],
        "dendritic": ["dendritic cell", "myeloid"],
        "pdc": ["plasmacytoid dendritic", "dendritic cell"],
    }
    profile.broad_cell_terms = broad_terms.get(lineage, [])

    return profile


def score_paper_relevance(
    paper: Dict[str, Any],
    *,
    pathway_profile: Dict[str, Any],
    cell_type_terms: List[str],
    genes_list: List[str],
    context_profile: LiteratureContextProfile,
    pathway_tokens_fn,
    prefer_reviews: bool = False,
) -> Dict[str, Any]:
    """Score one paper for biological fit to the pathway/cell-type question."""
    haystack = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    reasons: List[str] = []
    score = 0.0

    phrase_hits = []
    for phrase in pathway_profile.get("scoring_terms", []):
        lowered_phrase = phrase.lower()
        if lowered_phrase and lowered_phrase in haystack:
            phrase_hits.append(phrase)
    if phrase_hits:
        score += 3.5
        reasons.append(f"pathway phrase '{phrase_hits[0]}'")

    matched_tokens = []
    for phrase in pathway_profile.get("scoring_terms", []):
        for tok in pathway_tokens_fn(phrase):
            if tok in haystack and tok not in matched_tokens:
                matched_tokens.append(tok)
    if matched_tokens:
        score += min(3.0, 0.75 * len(matched_tokens))
        reasons.append(f"pathway tokens {', '.join(matched_tokens[:3])}")

    cell_match = False
    matched_cell_term = None
    for idx, term in enumerate(cell_type_terms):
        if term and term in haystack:
            cell_match = True
            matched_cell_term = term
            score += max(2.0, 4.0 - idx)
            reasons.append(f"cell-type term '{term}'")
            break
    if not cell_match:
        cell_tokens = [tok for term in cell_type_terms for tok in pathway_tokens_fn(term) if tok not in {"cell"}]
        token_hits = [tok for tok in cell_tokens if tok in haystack]
        if token_hits:
            score += 1.5
            reasons.append("cell-type context")
            cell_match = True

    broad_lineage_hits = [term for term in context_profile.broad_cell_terms if term in haystack]
    if broad_lineage_hits and not matched_cell_term:
        boost = 1.25 if context_profile.cluster_confidence_level == "low" else 0.75
        score += boost
        reasons.append(f"broad lineage '{broad_lineage_hits[0]}'")

    matched_genes = []
    for gene in genes_list[:5]:
        if gene.lower() in haystack:
            matched_genes.append(gene)
    if matched_genes:
        score += 1.5 * len(matched_genes)
        reasons.append(f"leading-edge genes {', '.join(matched_genes[:3])}")

    tissue_hits = [term for term in context_profile.tissue_terms if term in haystack]
    if tissue_hits:
        score += 1.5
        reasons.append(f"tissue context '{tissue_hits[0]}'")

    condition_hits = [term for term in context_profile.condition_terms if term in haystack]
    if condition_hits:
        score += 1.0
        reasons.append(f"condition context '{condition_hits[0]}'")

    if context_profile.species == "human":
        if "human" in haystack:
            score += 0.75
            reasons.append("human context")
        elif any(term in haystack for term in ["mouse", "murine", "mus musculus"]):
            score -= 0.75
            reasons.append("species mismatch")
    elif context_profile.species == "mouse":
        if any(term in haystack for term in ["mouse", "murine", "mus musculus"]):
            score += 0.75
            reasons.append("mouse context")
        elif "human" in haystack:
            score -= 0.75
            reasons.append("species mismatch")

    year = str(paper.get("year") or "")
    if year.isdigit() and int(year) >= 2024:
        score += 0.5

    journal = (paper.get("journal") or "").lower()
    title = (paper.get("title") or "").lower()
    is_review = "review" in journal or "review" in title
    if prefer_reviews and is_review:
        score += 0.75
    elif is_review:
        score += 0.25

    penalty_hits = [term for term in pathway_profile.get("penalty_terms", []) if term in haystack]
    if penalty_hits and not matched_genes and matched_cell_term not in set(cell_type_terms[:2]):
        score -= 2.0
        reasons.append("generic transplant context")

    if context_profile.tissue == "PBMC" and any(term in haystack for term in ["tumor microenvironment", "solid tumor", "carcinoma"]):
        score -= 1.0
        reasons.append("off-context tumor setting")

    if context_profile.cluster_confidence_level == "low" and not broad_lineage_hits and not matched_genes:
        score -= 0.5
        reasons.append("weak support for low-confidence cluster")

    if not cell_match and not matched_genes:
        score -= 1.0

    scored = dict(paper)
    scored["relevance_score"] = round(score, 2)
    scored["match_reasons"] = _dedupe_keep_order(reasons)[:5]
    scored["context_profile"] = context_profile.to_dict()
    return scored
