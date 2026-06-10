"""Local Cytopus KnowledgeBase wrapper for deterministic marker adjudication.

Cytopus (https://cytopus.readthedocs.io) is a curated, hierarchical, immune/
tumor-focused cell-type gene-set knowledge base shipped as a local Python
package. This module wraps it for the annotation validator so cluster labels can
be adjudicated against curated markers **deterministically and locally** — no
remote MCP call, no agent self-report, no "did you actually query" gate.

Design notes
------------
- Cytopus gene sets are receptor/TF-curated, so big empirical DEG markers like
  LYZ / NKG7 / GNLY are absent and canonical lineage markers are split across the
  hierarchy (e.g. CD3D lives in the ``abT`` parent). We therefore (a) UNION a
  label's gene set with its lineage ancestors, and (b) adjudicate by RELATIVE
  overlap (which candidate's markers best match the DEGs) rather than an absolute
  threshold.
- Coverage is deep for immune/tumor lineages but absent for platelet/erythroid/
  MAIT-identity/most non-immune types. ``covered()`` reports this; the caller
  falls back to PanglaoDB when a label is uncovered or the winning margin is thin.

The package is optional: if cytopus is not importable, ``available()`` is False
and all queries return empty/"fall back" results so callers degrade to PanglaoDB.
"""

from __future__ import annotations

import functools
import re
from typing import Any, Dict, List, Optional, Set


# --- Free-text label -> Cytopus key -------------------------------------------
# Maps the labels CellTypist / Scimilarity / users produce onto Cytopus keys
# (which use a controlled, abbreviated vocabulary). Keys here resolve either to a
# gene-set-bearing identity or to a broad node expanded via lineage union below.
_SYNONYMS: Dict[str, str] = {
    # monocyte / macrophage
    "monocyte": "mono", "monocytes": "mono",
    "classical monocyte": "mono", "classical monocytes": "mono",
    "cd14 monocyte": "mono", "cd14+ monocyte": "mono",
    "non classical monocyte": "mono", "nonclassical monocyte": "mono",
    "non-classical monocyte": "mono", "non-classical monocytes": "mono",
    "cd16 monocyte": "mono", "cd16+ monocyte": "mono", "fcgr3a monocyte": "mono",
    "intermediate monocyte": "mono",
    "macrophage": "Mac", "macrophages": "Mac",
    # dendritic cells
    "dendritic cell": "cDC2", "dendritic cells": "cDC2", "dc": "cDC2",
    "myeloid dendritic cell": "cDC2", "conventional dendritic cell": "cDC2",
    "dc1": "cDC1", "cdc1": "cDC1", "dc2": "cDC2", "cdc2": "cDC2",
    "dc3": "cDC3", "cdc3": "cDC3",
    "plasmacytoid dendritic cell": "p-DC", "plasmacytoid dendritic cells": "p-DC",
    "pdc": "p-DC", "p-dc": "p-DC", "p dc": "p-DC",
    "langerhans cell": "Langerhans", "langerhans": "Langerhans",
    "mo-dc": "mo-DC", "monocyte-derived dc": "mo-DC", "monocyte derived dc": "mo-DC",
    # B cells
    "b cell": "B", "b cells": "B", "b-cell": "B", "b lymphocyte": "B",
    "b-cell lineage": "B", "b cell lineage": "B",
    "naive b cell": "B-naive", "naive b cells": "B-naive", "b naive": "B-naive",
    "memory b cell": "B-memory", "memory b cells": "B-memory", "b memory": "B-memory",
    "switched memory b cell": "B-memory-switched",
    "class-switched memory b cell": "B-memory-switched",
    "unswitched memory b cell": "B-memory-non-switched",
    "germinal center b cell": "GC-B", "gc b cell": "GC-B",
    "plasma cell": "plasma", "plasma cells": "plasma",
    "plasmablast": "plasma-blast", "plasma blast": "plasma-blast",
    "follicular dendritic cell": "FDC", "fdc": "FDC",
    # T cells
    "t cell": "T", "t cells": "T", "t lymphocyte": "T", "cd3 t cell": "T",
    "cd4 t cell": "CD4-T", "cd4+ t cell": "CD4-T", "helper t cell": "CD4-T",
    "cd4 t": "CD4-T", "cd4-positive t cell": "CD4-T",
    "cd8 t cell": "CD8-T", "cd8+ t cell": "CD8-T", "cytotoxic t cell": "CD8-T",
    "cytotoxic t": "CD8-T", "cd8 t": "CD8-T", "cd8-positive t cell": "CD8-T",
    "regulatory t cell": "Treg", "treg": "Treg", "regulatory t cells": "Treg",
    "naive t cell": "T-naive", "t naive": "T-naive",
    "central memory t cell": "TCM", "tcm": "TCM",
    "effector memory t cell": "TEM", "tem": "TEM",
    "tissue resident memory t cell": "TRM", "trm": "TRM",
    "tfh": "TFH", "follicular helper t cell": "TFH",
    "gamma delta t cell": "gdT", "gamma-delta t cell": "gdT", "gd t cell": "gdT",
    "gdt": "gdT",
    "stem cell memory t cell": "TSCM", "tscm": "TSCM",
    # NK / ILC
    "nk cell": "NK", "nk cells": "NK", "natural killer cell": "NK",
    "natural killer": "NK", "natural killer cells": "NK",
    "cd16 nk cell": "CD56dim-NK", "cd16+ nk cell": "CD56dim-NK",
    "cd56dim nk cell": "CD56dim-NK", "cytotoxic nk cell": "CD56dim-NK",
    "cd56bright nk cell": "CD56bright-NK", "cd56 bright nk cell": "CD56bright-NK",
    "innate lymphoid cell": "ILC1", "ilc": "ILC1", "ilc1": "ILC1",
    "ilc2": "ILC2", "ilc3": "ILC3-NCRpos",
    # granulocytes / mast
    "granulocyte": "gran", "granulocytes": "gran",
    "neutrophil": "gran", "neutrophils": "gran",
    "mast cell": "mast", "mast cells": "mast", "mast": "mast",
    # stromal / epithelial / endothelial (limited Cytopus coverage)
    "endothelial cell": "endo-systemic-venous", "endothelial": "endo-systemic-venous",
    "fibroblast": "fibro", "fibroblasts": "fibro", "stromal cell": "fibro",
    "epithelial cell": "epi", "epithelial": "epi",
}

# When scoring a subtype, also include markers from these lineage ancestors so
# canonical lineage genes (e.g. CD3 from abT) are part of the comparison.
_LINEAGE_UNION: Dict[str, List[str]] = {
    "CD8-T": ["abT", "CD8-T"], "CD4-T": ["abT", "CD4-T"],
    "Treg": ["abT", "CD4-T", "Treg"], "TFH": ["abT", "CD4-T", "TFH"],
    "TCM": ["abT", "TCM"], "TEM": ["abT", "TEM"], "TRM": ["abT", "TRM"],
    "TSCM": ["abT", "TSCM"], "T-naive": ["abT", "T-naive"], "gdT": ["abT", "gdT"],
    "T": ["abT", "CD4-T", "CD8-T"],
    "CD56dim-NK": ["NK", "CD56dim-NK"], "CD56bright-NK": ["NK", "CD56bright-NK"],
    "NK-adaptive": ["NK", "NK-adaptive"],
    "B-naive": ["B", "B-naive"], "B-memory": ["B", "B-memory"],
    "B-memory-switched": ["B", "B-memory", "B-memory-switched"],
    "B-memory-non-switched": ["B", "B-memory", "B-memory-non-switched"],
    "GC-B": ["B", "GC-B"],
    "plasma": ["plasma"], "plasma-blast": ["plasma", "plasma-blast"],
}

# Default competing labels considered during adjudication so the relative score
# always has a discriminating field even if the caller passes few competitors.
_DEFAULT_PANEL = [
    "mono", "Mac", "cDC1", "cDC2", "p-DC", "CD4-T", "CD8-T", "Treg",
    "B-naive", "B-memory", "plasma", "NK", "CD56dim-NK", "gran", "mast",
]


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


@functools.lru_cache(maxsize=1)
def _load_kb():
    import contextlib
    import io
    import cytopus as cp  # noqa: import guarded by caller via available()
    # cp.KnowledgeBase() prints a banner ("KnowledgeBase object containing ...")
    # to stdout on init; silence it so it does not leak into the agent terminal.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return cp.KnowledgeBase()


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """True iff the cytopus package is importable and the KB loads."""
    try:
        _load_kb()
        return True
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _identities() -> Dict[str, List[str]]:
    return dict(_load_kb().identities)


@functools.lru_cache(maxsize=1)
def _celltypes() -> Set[str]:
    return set(_load_kb().celltypes)


def _keyword_resolve(text: str) -> Optional[str]:
    """Token/keyword fuzzy resolver for compound/styled labels that exact-match
    misses (e.g. 'Tem/Trm cytotoxic T cells', 'CD16+ NK cells', 'Tcm/Naive
    helper T cells'). Ordered: most specific rule first. Returns a Cytopus key
    or None (None -> caller falls back to PanglaoDB).
    """
    t = f" {text} "
    has = lambda *ws: all(f" {w}" in t or t.strip().find(w) >= 0 for w in ws)

    def contains(*subs: str) -> bool:
        return any(s in text for s in subs)

    # MAIT / iNKT have no Cytopus identity gene set -> leave to PanglaoDB.
    if contains("mait", "inkt"):
        return None
    # T-cell lineage (check subtype keywords before generic "t")
    if contains("regulatory t", "treg"):
        return "Treg"
    if contains("gamma delta", "gamma-delta", "gd t", "γδ"):
        return "gdT"
    if "t" in text and contains("cytotoxic", "cd8"):
        return "CD8-T"
    if "t" in text and contains("helper", "cd4"):
        return "CD4-T"
    if contains("t cell", "t lymphocyte", "t-cell") or (
        "t" in text and contains("naive", "memory", "effector", "tcm", "tem", "trm")
    ):
        return "T"
    # NK
    if contains("nk", "natural killer"):
        if contains("bright", "cd56bright", "regulatory"):
            return "CD56bright-NK"
        if contains("cd16", "dim", "cytotoxic"):
            return "CD56dim-NK"
        return "NK"
    # B-cell lineage
    if contains("plasmablast", "plasma blast"):
        return "plasma-blast"
    if contains("plasma"):
        return "plasma"
    if "b" in text and contains("naive"):
        return "B-naive"
    if "b" in text and contains("memory"):
        return "B-memory"
    if contains("germinal center"):
        return "GC-B"
    if contains("b cell", "b-cell", "b lymphocyte"):
        return "B"
    # DC
    if contains("plasmacytoid"):
        return "p-DC"
    if contains("dendritic", "dc"):
        if contains("cdc1", "dc1"):
            return "cDC1"
        if contains("cdc3", "dc3"):
            return "cDC3"
        return "cDC2"
    # Myeloid / granulocyte / mast
    if contains("macrophage"):
        return "Mac"
    if contains("monocyte", "mono"):
        return "mono"
    if contains("neutrophil", "granulocyte"):
        return "gran"
    if contains("mast"):
        return "mast"
    if contains("fibroblast", "stromal"):
        return "fibro"
    if contains("endothel"):
        return "endo-systemic-venous"
    if contains("epitheli"):
        return "epi"
    return None


def resolve_label(label: Any) -> Optional[str]:
    """Map a free-text label to a Cytopus key, or None if unmapped/unavailable."""
    if not available():
        return None
    text = normalize_label(label)
    if not text:
        return None
    if text in _SYNONYMS:
        return _SYNONYMS[text]
    # strip a trailing "cell(s)" and retry
    stripped = re.sub(r"\b cells?\b", "", text).strip()
    if stripped and stripped in _SYNONYMS:
        return _SYNONYMS[stripped]
    # direct key match (case-insensitive) against the controlled vocabulary
    ids = _identities()
    for key in ids:
        if key.lower() == text or key.lower() == stripped:
            return key
    # token/keyword fuzzy fallback for compound/styled labels
    return _keyword_resolve(text)


def markers_for(label: Any) -> Set[str]:
    """Return the curated marker set for a label, unioned with lineage ancestors.

    Empty set if the label cannot be resolved or cytopus is unavailable.
    """
    key = resolve_label(label)
    if key is None:
        return set()
    ids = _identities()
    genes: Set[str] = set(ids.get(key, []))
    for anc in _LINEAGE_UNION.get(key, [key]):
        genes |= set(ids.get(anc, []))
    # if the key was a broad celltype with no own identity set, expand subsets
    if not genes and key in _celltypes():
        try:
            expanded = _load_kb().get_identities([key], include_subsets=True)
            for v in expanded.values():
                genes |= set(v)
        except Exception:
            pass
    return {g.upper() for g in genes}


def covered(label: Any) -> bool:
    """True iff this label resolves to a non-empty Cytopus marker set."""
    return bool(markers_for(label))


def adjudicate(
    candidate_label: Any,
    competing_labels: Optional[List[Any]] = None,
    deg_genes: Optional[List[Any]] = None,
    *,
    min_margin: int = 1,
) -> Dict[str, Any]:
    """Adjudicate a candidate label against competitors using Cytopus marker
    overlap with the cluster's DEGs (relative scoring, hierarchy-aware).

    Returns a dict:
      - ``available`` (bool): cytopus usable at all
      - ``candidate_covered`` (bool): candidate maps to a Cytopus marker set
      - ``deg_overlap`` (Dict[label -> {overlap, markers_matched, n_markers}])
      - ``best_label`` / ``best_overlap`` / ``margin`` (winner vs runner-up)
      - ``candidate_overlap`` (int) for the candidate specifically
      - ``candidate_is_best`` (bool)
      - ``needs_external_fallback`` (bool): True when the candidate is uncovered,
        the winning margin is below ``min_margin``, or the candidate is not best —
        i.e. defer to PanglaoDB.
    """
    if not available():
        return {"available": False, "needs_external_fallback": True,
                "candidate_covered": False, "deg_overlap": {}}

    deg_set = {str(g).strip().upper() for g in (deg_genes or []) if str(g).strip()}
    cand_norm = normalize_label(candidate_label)

    # Build the field of labels to compare: candidate + competitors + a default
    # discriminating panel (deduplicated by resolved Cytopus key).
    raw_labels: List[Any] = [candidate_label] + list(competing_labels or []) + _DEFAULT_PANEL
    by_key: Dict[str, Any] = {}
    for lbl in raw_labels:
        key = resolve_label(lbl)
        if key is not None:
            by_key.setdefault(key, lbl)

    deg_overlap: Dict[str, Dict[str, Any]] = {}
    for key, lbl in by_key.items():
        m = markers_for(lbl)
        matched = sorted(m & deg_set)
        deg_overlap[normalize_label(lbl) or key] = {
            "cytopus_key": key,
            "overlap": len(matched),
            "markers_matched": matched,
            "n_cytopus_markers": len(m),
        }

    candidate_covered = covered(candidate_label)
    candidate_overlap = next(
        (v["overlap"] for k, v in deg_overlap.items() if k == cand_norm), 0
    )

    ranked = sorted(deg_overlap.items(), key=lambda kv: kv[1]["overlap"], reverse=True)
    best_label = ranked[0][0] if ranked else None
    best_overlap = ranked[0][1]["overlap"] if ranked else 0
    runner_overlap = ranked[1][1]["overlap"] if len(ranked) > 1 else 0
    margin = best_overlap - runner_overlap
    candidate_is_best = bool(
        candidate_covered and best_label == cand_norm and best_overlap > 0
    )

    needs_external_fallback = bool(
        (not candidate_covered)
        or (not candidate_is_best)
        or (margin < min_margin)
        or (candidate_overlap == 0)
    )
    return {
        "available": True,
        "candidate_covered": candidate_covered,
        "candidate_overlap": candidate_overlap,
        "candidate_is_best": candidate_is_best,
        "best_label": best_label,
        "best_overlap": best_overlap,
        "margin": margin,
        "deg_overlap": deg_overlap,
        "needs_external_fallback": needs_external_fallback,
    }
