"""
Claude API tool definitions for scagent.

Tools are organized into two layers:
1. Action tools - mutate/generate analysis artifacts
2. Inspection tools - read-only queries for more detail

All tools return structured JSON for LLM reasoning.
"""

# Configure tqdm for cleaner progress bars (must be before any imports that use tqdm)
import os
os.environ.setdefault('TQDM_NCOLS', '60')
os.environ.setdefault('TQDM_MININTERVAL', '0.5')  # Update less frequently

from typing import List, Dict, Any, Optional
import hashlib
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def _make_annotation_proposal_fingerprint(
    cluster_key: str,
    cluster_ids,
    deg_key: str,
    annotation_key: str,
    n_obs: int,
    adata=None,
) -> str:
    """Stable fingerprint of the inputs that define an annotation proposal.

    When ``adata`` is provided, the fingerprint also includes a hash of the
    live per-cell ``(obs_name, cluster_label)`` pairs so it changes when
    cluster membership shifts — even if cluster ids, cluster count, and
    ``n_obs`` stay the same. This is what makes "same labels, swapped
    members" detectable downstream in ``finalize_annotation``.

    When ``adata`` is omitted, the fingerprint covers only the proposal's
    static descriptors; that variant is intended for parity checks of the
    stored proposal payload, not as a live-state guarantee.
    """
    try:
        ids_norm = sorted(str(c) for c in (cluster_ids or []))
    except Exception:
        ids_norm = []
    digest = hashlib.sha1()
    digest.update(
        json.dumps(
            {
                "cluster_key": str(cluster_key or ""),
                "annotation_key": str(annotation_key or ""),
                "deg_key": str(deg_key or ""),
                "n_obs": int(n_obs or 0),
                "cluster_ids": ids_norm,
            },
            sort_keys=True,
        ).encode("utf-8")
    )
    if adata is not None and cluster_key and cluster_key in adata.obs.columns:
        try:
            cluster_series = adata.obs[cluster_key].astype(str)
            obs_names_iter = adata.obs_names.astype(str)
            membership = hashlib.sha1()
            for name, label in zip(obs_names_iter, cluster_series.values):
                membership.update(str(name).encode("utf-8", errors="replace"))
                membership.update(b"\t")
                membership.update(str(label).encode("utf-8", errors="replace"))
                membership.update(b"\n")
            digest.update(b"|membership=")
            digest.update(membership.hexdigest().encode("utf-8"))
        except Exception:
            digest.update(b"|membership=unavailable")
    return digest.hexdigest()[:16]

DEFAULT_STRUCTURE_EXCLUDE_PATTERNS = [
    r"^MT-",
    r"^mt-",
    r"^RPL",
    r"^RPS",
    r"^MRPL",
    r"^MRPS",
    r"^Rpl",
    r"^Rps",
    r"^Mrpl",
    r"^Mrps",
    r"^MALAT1$",
    r"^Malat1$",
    r"^HB[ABDEGMQZ]",
    r"^Hb[ab]",
    r"^RP\d",
    r"^AC\d",
    r"^AL\d",
    r"^AP\d",
    r"^LINC\d",
    r"\.\d+$",
]

ANNOTATION_NUISANCE_GENE_PATTERNS = [
    r"^MT-",
    r"^mt-",
    r"^RPL",
    r"^RPS",
    r"^MRPL",
    r"^MRPS",
    r"^Rpl",
    r"^Rps",
    r"^Mrpl",
    r"^Mrps",
    r"^MALAT1$",
    r"^Malat1$",
    r"^HB[ABDEGMQZ]",
    r"^Hb[ab]",
    r"^RP\d",
    r"^AC\d",
    r"^AL\d",
    r"^AP\d",
    r"^LINC\d",
    r"\.\d+$",
]

ANNOTATION_BROAD_SUPPORT_GENE_PATTERNS = [
    # Broad immune / antigen-presentation genes. Useful context, but not
    # enough by themselves to overturn a lineage call.
    r"^PTPRC$",
    r"^CD52$",
    r"^CORO1A$",
    r"^B2M$",
    r"^CD74$",
    r"^HLA-",
    r"^H2-",
    # Housekeeping, cytoskeleton, and high-abundance structural genes.
    r"^ACT[ABG]",
    r"^GAPDH$",
    r"^TUB[AB]",
    r"^UBB$",
    r"^UBC$",
    r"^EEF",
    r"^RAN$",
    # Stress / heat-shock / immediate-early programs.
    r"^HSP",
    r"^HSPA",
    r"^HSPB",
    r"^HSPD",
    r"^HSPH",
    r"^DNAJ",
    r"^FOS",
    r"^JUN",
    r"^EGR",
    r"^DUSP",
    r"^IER",
    # Interferon and generic inflammatory response genes.
    r"^IFIT",
    r"^IFITM",
    r"^ISG",
    r"^IFI",
    r"^MX[12]$",
    r"^OAS",
    r"^RSAD2$",
    r"^S100A[89]$",
    # Broad myeloid/innate context. These support a family but are not
    # discriminating enough for cross-family overrides alone.
    r"^LYZ$",
    r"^LST1$",
    r"^TYROBP$",
    r"^FCER1G$",
    r"^AIF1$",
    r"^LGALS3$",
    r"^CST3$",
    r"^CTSB$",
    r"^CTSL$",
    r"^FTL$",
    r"^FTH1$",
]


def _annotation_nuisance_reason(gene: Any) -> Optional[str]:
    """Return the nuisance-pattern reason for a gene, if it is non-specific."""
    text = str(gene or "").strip()
    if not text:
        return "empty_gene"
    for pattern in ANNOTATION_NUISANCE_GENE_PATTERNS:
        try:
            if re.search(pattern, text):
                return pattern
        except Exception:
            continue
    return None


def _annotation_broad_support_reason(gene: Any) -> Optional[str]:
    """Return why a gene is broad/non-discriminating annotation support.

    These are not treated as garbage: many are biologically meaningful. The
    point is narrower — they should not be the only evidence for fine labels
    or for overriding a CellTypist/Scimilarity reference consensus.
    """
    text = str(gene or "").strip()
    if not text:
        return "empty_gene"
    for pattern in ANNOTATION_BROAD_SUPPORT_GENE_PATTERNS:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return pattern
        except Exception:
            continue
    return None


def _resolve_run_path(raw: Any, run_manager: Any = None, must_exist: bool = True) -> Optional[Path]:
    """Resolve a path argument forgivingly across tools.

    - Absolute paths are returned as-is.
    - Relative paths and bare basenames are tried against (in order):
      ``run_manager.run_dir / raw``, ``Path.cwd() / raw``, ``raw``.
    - With ``must_exist=True`` (default, used for reads): returns the first
      existing candidate, or ``None`` if no candidate exists.
    - With ``must_exist=False`` (used for writes): returns the run_dir-anchored
      path (or cwd-anchored if run_manager is None), creating no files.
    """
    if raw is None:
        return None
    try:
        raw_path = Path(str(raw)).expanduser()
    except Exception:
        return None
    if raw_path.is_absolute():
        if must_exist and not raw_path.exists():
            return None
        return raw_path
    candidates: List[Path] = []
    run_dir: Optional[Path] = None
    if run_manager is not None:
        try:
            run_dir = Path(run_manager.run_dir)
            candidates.append(run_dir / raw_path)
        except Exception:
            run_dir = None
    candidates.append(Path.cwd() / raw_path)
    candidates.append(raw_path)
    if must_exist:
        for c in candidates:
            try:
                if c.exists():
                    return c
            except Exception:
                continue
        return None
    # write path: prefer run_dir anchor when available
    if run_dir is not None:
        return run_dir / raw_path
    return Path.cwd() / raw_path


def _format_validation_failures_per_cluster(failures: List[str]) -> str:
    """Group cluster-prefixed failures by cluster id so the model sees all
    issues at once instead of a 6-item truncation. ``failures`` items that
    start with ``"Cluster <id>:"`` are grouped; others go to a ``(global)``
    bucket. The output keeps the original message bodies (post-prefix).
    """
    if not failures:
        return ""
    per_cluster: Dict[str, List[str]] = {}
    globals_: List[str] = []
    cluster_prefix = re.compile(r"^Cluster\s+([^:]+):\s*(.*)$", re.DOTALL)
    for f in failures:
        if not isinstance(f, str):
            continue
        m = cluster_prefix.match(f)
        if m:
            cid = m.group(1).strip()
            body = m.group(2).strip()
            per_cluster.setdefault(cid, []).append(body)
        else:
            globals_.append(f.strip())
    n_clusters = len(per_cluster)
    n_issues = sum(len(v) for v in per_cluster.values()) + len(globals_)
    header = f"{n_issues} issues across {n_clusters} cluster(s)"
    lines: List[str] = [header]
    def _sort_key(cid: str):
        try:
            return (0, int(cid))
        except Exception:
            return (1, cid)
    for cid in sorted(per_cluster.keys(), key=_sort_key):
        bodies = per_cluster[cid]
        lines.append(f"Cluster {cid}: " + "; ".join(bodies))
    for g in globals_:
        lines.append(f"(global): {g}")
    return " | ".join(lines)


def _validate_annotation_evidence(
    *,
    adata: Any,
    proposal: Dict[str, Any],
    evidence: Dict[str, Any],
    world_state: Any,
    tool_input_unavailable_sources: Any = None,
    allow_partial: bool = False,
    apply_auto_fixes: bool = True,
) -> Dict[str, Any]:
    """Validate annotation evidence against a prepare_annotation proposal.

    Centralised so both ``stage_annotation_evidence`` (preview) and
    ``finalize_annotation`` (commit) run identical checks. Returns a report
    dict; never raises. When ``apply_auto_fixes`` is true, deterministic rule
    violations (e.g., ``confidence=high`` paired with broad-lineage-only
    PanglaoDB support) are silently corrected on the returned ``evidence_str``
    and an entry is appended to ``auto_fixes``.

    Returns
    -------
    dict with keys:
      - ``validation_failures`` (List[str])
      - ``per_cluster_validation`` (Dict[str, Dict[str, Any]])
      - ``auto_fixes`` (List[str])
      - ``evidence_str`` (Dict[str, Any]) — possibly mutated
      - ``missing_clusters``, ``unknown_clusters`` (List[str])
      - ``missing_reference_sources``, ``unexplained_missing_sources`` (List[str])
      - ``tool_recorded_unavailable_sources``, ``manual_unavailable_sources``,
        ``unavailable_reference_sources`` (Dict[str, Any])
      - ``scimilarity_availability`` (Optional[Dict[str, Any]])
      - ``any_panglaodb`` (bool)
      - ``ambiguous_set`` (set), ``reference_keys`` (List[str])
      - ``proposal_clusters``, ``proposal_cluster_entries``
    """
    proposal = proposal or {}
    proposal_clusters = [str(c) for c in proposal.get("cluster_ids", [])]
    evidence_str = {str(k): v for k, v in (evidence or {}).items()}
    missing_clusters = [c for c in proposal_clusters if c not in evidence_str]
    unknown_clusters = [c for c in evidence_str.keys() if c not in proposal_clusters]

    ambiguous_set = set(str(c) for c in proposal.get("ambiguous_clusters", []))
    reference_keys = [
        str(k) for k in (proposal.get("reference_annotation_keys") or [])
        if str(k).strip()
    ]
    proposal_cluster_entries = {
        str(entry.get("cluster_id")): entry
        for entry in proposal.get("clusters", [])
        if isinstance(entry, dict) and entry.get("cluster_id") is not None
    }

    def _normalize_label(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _singular_label(value: Any) -> str:
        tokens = re.findall(r"[a-z0-9]+", _normalize_label(value))

        def _singular_token(token: str) -> str:
            if token == "cells":
                return "cell"
            if len(token) > 4 and token.endswith("ies"):
                return token[:-3] + "y"
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                return token[:-1]
            return token

        return " ".join(_singular_token(token) for token in tokens)

    def _label_tokens(value: Any) -> List[str]:
        return re.findall(r"[a-z0-9]+", _singular_label(value))

    def _substantive_label_tokens(value: Any) -> List[str]:
        generic = {
            "cell", "cells", "positive", "negative", "pos", "neg",
            "human", "mouse", "derived", "like", "and", "or", "of", "the",
        }
        return [token for token in _label_tokens(value) if token not in generic]

    def _label_acronyms(value: Any) -> set:
        tokens = _label_tokens(value)
        substantive = _substantive_label_tokens(value)
        acronyms = set()
        if len(tokens) >= 2:
            acronyms.add("".join(token[0] for token in tokens if token))
        if len(substantive) >= 2:
            acronyms.add("".join(token[0] for token in substantive if token))
        return {a for a in acronyms if len(a) >= 2}

    def _label_aliases(value: Any) -> set:
        raw_text = str(value or "").strip()
        text = _normalize_label(value)
        singular = _singular_label(text)
        aliases = {text, singular} if text else set()
        if not text:
            return aliases
        compact = re.sub(r"[^a-z0-9]+", "", singular)
        if compact:
            aliases.add(compact)
        aliases.update(_label_acronyms(text))
        substantive = _substantive_label_tokens(text)
        if len(substantive) == 1 and len(substantive[0]) <= 4:
            aliases.add(substantive[0])
        raw_compact = re.sub(r"[^A-Za-z0-9]+", "", raw_text)
        if raw_compact and len(raw_compact) <= 5 and raw_compact.lower() == compact:
            aliases.add(raw_compact.lower())
        return {a for a in aliases if a}

    def _split_label_candidates(value: Any) -> List[str]:
        if isinstance(value, list):
            values = value
        elif isinstance(value, str):
            values = re.split(r"\s*(?:/|\||;|\bor\b)\s*", value)
        else:
            values = []
        labels: List[str] = []
        for raw in values:
            label = str(raw or "").strip()
            if label and label not in labels:
                labels.append(label)
        return labels

    def _labels_exact_or_alias(value_a: Any, value_b: Any) -> bool:
        candidates_a = _split_label_candidates(value_a) or [value_a]
        candidates_b = _split_label_candidates(value_b) or [value_b]
        for candidate_a in candidates_a:
            aliases_a = _label_aliases(candidate_a)
            if not aliases_a:
                continue
            for candidate_b in candidates_b:
                aliases_b = _label_aliases(candidate_b)
                if aliases_b and aliases_a.intersection(aliases_b):
                    return True
        return False

    def _label_family_for_annotation(value: Any) -> Optional[str]:
        text = _normalize_label(value)
        if not text:
            return None
        if "platelet" in text or "megakary" in text:
            return "platelet"
        if "plasma" in text:
            return "plasma"
        if "monocyte" in text or "macrophage" in text:
            return "monocyte"
        if "plasmacytoid dendritic" in text or text == "pdc" or " pdc" in f" {text}":
            return "pdc"
        if "dendritic" in text or text in {"dc", "cdc", "cdc1", "cdc2"} or " cdc" in f" {text}":
            return "dendritic"
        if "natural killer" in text or " nk" in f" {text}" or text.startswith("nk"):
            return "nk"
        if "b cell" in text or text.startswith("b ") or " b " in f" {text} ":
            return "b"
        if "t cell" in text or text.startswith("t ") or " t " in f" {text} " or "mait" in text or "treg" in text:
            return "t"
        if "neutrophil" in text:
            return "neutrophil"
        if "mast" in text or "basophil" in text:
            return "mast_basophil"
        if "eryth" in text or "red blood" in text:
            return "erythroid"
        if "epithelial" in text or "ciliated" in text or "club cell" in text:
            return "epithelial"
        if "endothelial" in text:
            return "endothelial"
        if "fibroblast" in text or "stromal" in text or "smooth muscle" in text:
            return "stromal"
        if "hsc" in text or "mpp" in text or "progenitor" in text or "stem" in text:
            return "progenitor"
        return None

    def _labels_biologically_compatible(a: Any, b: Any) -> bool:
        a_norm = _normalize_label(a)
        b_norm = _normalize_label(b)
        if not a_norm or not b_norm:
            return False
        if _labels_exact_or_alias(a_norm, b_norm):
            return True
        a_singular = re.sub(r"\bcells\b", "cell", a_norm)
        b_singular = re.sub(r"\bcells\b", "cell", b_norm)
        if a_singular in b_singular or b_singular in a_singular:
            return True
        a_family = _label_family_for_annotation(a_norm)
        b_family = _label_family_for_annotation(b_norm)
        return bool(a_family and a_family == b_family)

    def _reference_source_group(annotation_key: Any) -> str:
        key = str(annotation_key or "").lower()
        if "celltypist" in key:
            return "celltypist"
        if "scimilarity" in key:
            return "scimilarity"
        return str(annotation_key or "other")

    def _reference_consensus_for_cluster(
        proposal_entry: Dict[str, Any],
        min_fraction: float = 0.35,
    ) -> Dict[str, Any]:
        entries = proposal_entry.get("reference_annotations") or []
        usable: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = entry.get("top_label")
            if not isinstance(label, str) or not label.strip():
                continue
            try:
                frac = float(entry.get("top_fraction") or 0.0)
            except Exception:
                frac = 0.0
            if frac < min_fraction:
                continue
            source = _reference_source_group(entry.get("annotation_key"))
            usable.append({
                "source": source,
                "annotation_key": entry.get("annotation_key"),
                "label": label.strip(),
                "fraction": round(frac, 4),
                "family": _label_family_for_annotation(label),
            })

        groups: List[Dict[str, Any]] = []
        for item in usable:
            placed = False
            for group in groups:
                if _labels_biologically_compatible(item["label"], group["label"]):
                    group["members"].append(item)
                    if item["fraction"] > group.get("score", 0.0):
                        group["label"] = item["label"]
                        group["score"] = item["fraction"]
                        group["family"] = item.get("family")
                    placed = True
                    break
            if not placed:
                groups.append({
                    "label": item["label"],
                    "family": item.get("family"),
                    "score": item["fraction"],
                    "members": [item],
                })

        for group in groups:
            sources = sorted({m["source"] for m in group["members"] if m.get("source")})
            group["sources"] = sources
            group["n_sources"] = len(sources)
            group["labels"] = sorted({m["label"] for m in group["members"] if m.get("label")})

        consensus = sorted(
            [g for g in groups if g.get("n_sources", 0) >= 2],
            key=lambda g: (g.get("n_sources", 0), g.get("score", 0.0)),
            reverse=True,
        )
        if not consensus:
            return {
                "has_consensus": False,
                "reference_groups": groups,
                "usable_references": usable,
            }

        best = consensus[0]
        return {
            "has_consensus": True,
            "label": best.get("label"),
            "family": best.get("family"),
            "sources": best.get("sources", []),
            "labels": best.get("labels", []),
            "score": best.get("score"),
            "reference_groups": groups,
            "usable_references": usable,
        }

    def _list_labels_from_evidence(value: Any) -> List[str]:
        labels: List[str] = []
        if isinstance(value, list):
            iterable = value
        elif isinstance(value, dict):
            iterable = value.values()
        elif isinstance(value, str):
            iterable = [value]
        else:
            iterable = []
        for item in iterable:
            if isinstance(item, dict):
                raw = item.get("label") or item.get("cell_type") or item.get("candidate") or item.get("top_label")
            else:
                raw = item
            if isinstance(raw, str) and raw.strip():
                labels.append(raw.strip())
        return labels

    def _confidence_rank(value: Any) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(str(value or ""), -1)

    def _confidence_exceeds(value: Any, cap: str) -> bool:
        return _confidence_rank(value) > _confidence_rank(cap)

    def _reason_code(reason: Any) -> str:
        if isinstance(reason, dict):
            value = reason.get("reason") or reason.get("unavailable_reason") or reason.get("status")
        else:
            value = reason
        return str(value or "").strip().lower()

    def _scimilarity_availability() -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "source": "scimilarity",
            "available": False,
            "package_available": False,
            "model_available": False,
            "model_paths_checked": [],
            "available_model_paths": [],
        }
        try:
            import importlib.util
            if importlib.util.find_spec("scimilarity") is None:
                info["reason"] = "package_missing"
                return info
            info["package_available"] = True
        except Exception as e:
            info["reason"] = "package_check_failed"
            info["message"] = str(e)
            return info
        try:
            from ..annotation.scimilarity import (
                DEFAULT_MODEL_PATH as _SCI_DEFAULT_MODEL_PATH,
                _MODEL_PATH_HUMAN as _SCI_MODEL_PATH_HUMAN,
                _MODEL_PATH_MOUSE as _SCI_MODEL_PATH_MOUSE,
            )
            candidate_paths = [
                os.environ.get("SCIMILARITY_MODEL_PATH"),
                os.environ.get("SCIMILARITY_MODEL_PATH_MOUSE"),
                _SCI_MODEL_PATH_HUMAN,
                _SCI_MODEL_PATH_MOUSE,
                _SCI_DEFAULT_MODEL_PATH,
            ]
        except Exception as e:
            info["reason"] = "model_path_lookup_failed"
            info["message"] = str(e)
            candidate_paths = [
                os.environ.get("SCIMILARITY_MODEL_PATH"),
                os.environ.get("SCIMILARITY_MODEL_PATH_MOUSE"),
            ]
        seen_paths = set()
        for candidate in candidate_paths:
            if not candidate:
                continue
            path_text = str(candidate)
            if path_text in seen_paths:
                continue
            seen_paths.add(path_text)
            exists = Path(path_text).exists()
            info["model_paths_checked"].append({"path": path_text, "exists": bool(exists)})
            if exists:
                info["available_model_paths"].append(path_text)
        info["model_available"] = bool(info["available_model_paths"])
        info["available"] = bool(info["package_available"] and info["model_available"])
        if not info["available"] and "reason" not in info:
            info["reason"] = "model_path_missing" if info["package_available"] else "package_missing"
        return info

    # Resolve unavailable-reference-source reasons from world state + caller.
    tool_recorded_unavailable_sources: Dict[str, Any] = {}
    if world_state is not None:
        annotation_val = getattr(world_state, "annotation_validation", None)
        if isinstance(annotation_val, dict):
            for source, reason in (annotation_val.get("reference_source_unavailable") or {}).items():
                if source:
                    tool_recorded_unavailable_sources[str(source).lower()] = reason

    manual_unavailable_sources: Dict[str, Any] = {}
    if isinstance(tool_input_unavailable_sources, dict):
        for source, reason in tool_input_unavailable_sources.items():
            if source:
                manual_unavailable_sources[str(source).lower()] = reason

    unavailable_reference_sources = dict(tool_recorded_unavailable_sources)
    for source, reason in manual_unavailable_sources.items():
        if source != "scimilarity":
            unavailable_reference_sources[source] = reason

    missing_reference_sources = [
        str(source).lower()
        for source in (proposal.get("missing_reference_sources") or [])
        if str(source).strip()
    ]
    unexplained_missing_sources = [
        source for source in missing_reference_sources
        if source != "scimilarity" and source not in unavailable_reference_sources
    ]

    scimilarity_availability: Optional[Dict[str, Any]] = None
    scimilarity_tool_unavailable = tool_recorded_unavailable_sources.get("scimilarity")
    scimilarity_manual_unavailable = manual_unavailable_sources.get("scimilarity")
    if "scimilarity" in missing_reference_sources:
        scimilarity_availability = _scimilarity_availability()

    # Pull PanglaoDB query log from world_state.
    queried_celltypes_normalized: set = set()
    queried_celltypes_raw: List[str] = []
    queried_gene_symbols_normalized: set = set()
    if world_state is not None:
        annotation_val = getattr(world_state, "annotation_validation", None)
        if isinstance(annotation_val, dict):
            for entry in annotation_val.get("reference_marker_queries", []) or []:
                if not isinstance(entry, dict):
                    continue
                ct = entry.get("cell_type")
                if isinstance(ct, str) and ct.strip():
                    queried_celltypes_raw.append(ct.strip())
                    queried_celltypes_normalized.add(_normalize_label(ct))
                gene = entry.get("gene_symbol") or entry.get("gene")
                if isinstance(gene, str) and gene.strip():
                    queried_gene_symbols_normalized.add(gene.strip().upper())

    def _queried_label_candidates(labels: List[str]) -> List[str]:
        candidates: List[str] = []
        queried_aliases = {
            alias
            for queried in queried_celltypes_normalized
            for alias in _label_aliases(queried)
        }
        for label_value in labels:
            aliases = _label_aliases(label_value)
            if aliases.intersection(queried_aliases):
                candidates.append(label_value)
        return candidates

    # Per-cluster DEG gene sets from the proposal.
    cluster_deg_genes: Dict[str, set] = {}
    for cid, entry in proposal_cluster_entries.items():
        genes = set()
        for deg in (entry.get("top_degs") or []):
            if isinstance(deg, dict):
                gene = deg.get("gene")
            else:
                gene = deg
            if isinstance(gene, str) and gene.strip():
                genes.add(gene.strip().upper())
        cluster_deg_genes[str(cid)] = genes

    validation_failures: List[str] = []
    per_cluster_validation: Dict[str, Dict[str, Any]] = {}
    auto_fixes: List[str] = []

    if missing_clusters and not allow_partial:
        validation_failures.append(
            f"Missing evidence for {len(missing_clusters)} clusters: {missing_clusters[:10]}"
        )
    if unknown_clusters:
        validation_failures.append(
            f"evidence_summary references {len(unknown_clusters)} clusters not in the proposal: {unknown_clusters[:10]}"
        )
    if unexplained_missing_sources:
        validation_failures.append(
            "Missing reference source(s) lack concrete unavailable reasons: "
            f"{unexplained_missing_sources}. Run the compatible reference tool(s), or pass "
            "reference_source_unavailable with specific non-Scimilarity reasons such as package_missing, "
            "model_download_failed, model_path_missing, no_species_compatible_model, or user_opted_out."
        )
    if "scimilarity" in missing_reference_sources:
        if scimilarity_availability and scimilarity_availability.get("available"):
            validation_failures.append(
                "Scimilarity is available but missing from the annotation proposal. "
                "Run run_scimilarity with the dataset organism before finalize_annotation; "
                "manual reference_source_unavailable cannot excuse Scimilarity when the package "
                f"and model path exist ({scimilarity_availability.get('available_model_paths')})."
            )
        elif not scimilarity_tool_unavailable:
            if scimilarity_manual_unavailable:
                validation_failures.append(
                    "Scimilarity was manually marked unavailable in finalize_annotation, but no prior "
                    "run_scimilarity tool failure recorded that blocker. Run run_scimilarity first, or "
                    "let the tool record a real package/model/runtime failure."
                )
            else:
                validation_failures.append(
                    "Missing Scimilarity source lacks a tool-recorded unavailable reason. "
                    "Run run_scimilarity before finalizing; if it truly cannot run, the failed tool "
                    "result will record the concrete blocker."
                )
        else:
            sci_reason = _reason_code(scimilarity_tool_unavailable)
            sci_status = (
                str(scimilarity_tool_unavailable.get("status") or "").lower()
                if isinstance(scimilarity_tool_unavailable, dict)
                else ""
            )
            if sci_reason in {"organism_ambiguous", "needs_input"} or sci_status == "needs_input":
                validation_failures.append(
                    "Scimilarity is not resolved yet: the prior run_scimilarity call needed an explicit "
                    "organism. Resolve the organism and rerun Scimilarity before finalizing annotation."
                )

    any_panglaodb = False
    for cid, ev in evidence_str.items():
        if not isinstance(ev, dict):
            validation_failures.append(f"Cluster {cid}: evidence is not an object.")
            continue
        checks: Dict[str, Any] = {"cluster_id": cid}
        label = ev.get("label")
        if not isinstance(label, str) or not label.strip():
            validation_failures.append(f"Cluster {cid}: missing or empty 'label'.")
            checks["label_ok"] = False
        else:
            checks["label"] = label
            checks["label_ok"] = True

        pq = bool(ev.get("panglaodb_queried", False))
        checks["panglaodb_queried"] = pq
        if pq:
            any_panglaodb = True
        else:
            validation_failures.append(
                f"Cluster {cid}: panglaodb_queried must be true after external marker validation."
            )

        panglaodb_label_used = ev.get("panglaodb_label_used")
        panglaodb_label_candidates = _split_label_candidates(panglaodb_label_used)
        effective_panglaodb_label_used = (
            panglaodb_label_candidates[0] if panglaodb_label_candidates else None
        )
        if isinstance(panglaodb_label_used, str) and panglaodb_label_used.strip():
            checks["panglaodb_label_used"] = panglaodb_label_used.strip()
        label_in_history = False
        reverse_hit = False
        if queried_celltypes_normalized or queried_gene_symbols_normalized:
            candidate_labels: List[str] = []
            candidate_labels.extend(panglaodb_label_candidates)
            if isinstance(label, str) and label.strip():
                candidate_labels.append(label.strip())
            queried_candidates = _queried_label_candidates(candidate_labels)
            if queried_candidates:
                label_in_history = True
                if (
                    panglaodb_label_candidates
                    and effective_panglaodb_label_used != queried_candidates[0]
                ):
                    auto_fixes.append(
                        f"Cluster {cid}: interpreted panglaodb_label_used={panglaodb_label_used!r} "
                        f"as queried label {queried_candidates[0]!r}."
                    )
                effective_panglaodb_label_used = queried_candidates[0]
            supporting_for_reverse = ev.get("supporting_genes") or []
            if isinstance(supporting_for_reverse, list):
                for g in supporting_for_reverse:
                    if isinstance(g, str) and g.strip().upper() in queried_gene_symbols_normalized:
                        reverse_hit = True
                        break
            if not label_in_history and not reverse_hit:
                validation_failures.append(
                    f"Cluster {cid}: label {label!r} (panglaodb_label_used={panglaodb_label_used!r}) "
                    "was not found in the PanglaoDB call history recorded in world state. "
                    "Either query PanglaoDB for this label or record the gene_symbol reverse query that supports it."
                )
            checks["panglaodb_label_in_call_history"] = label_in_history
            checks["panglaodb_reverse_gene_hit"] = reverse_hit
        else:
            checks["panglaodb_call_history_available"] = False

        supporting = ev.get("supporting_genes") or []
        checks["n_supporting_genes"] = len(supporting) if isinstance(supporting, list) else 0
        checks["supporting_genes"] = supporting if isinstance(supporting, list) else []
        nuisance_supporting: List[Dict[str, str]] = []
        broad_supporting: List[Dict[str, str]] = []
        non_nuisance_supporting: List[str] = []
        matched_degs: List[str] = []
        non_nuisance_matched_degs: List[str] = []
        discriminating_matched_degs: List[str] = []
        if not isinstance(supporting, list) or len(supporting) == 0:
            validation_failures.append(
                f"Cluster {cid}: supporting_genes is empty — every label must cite at least one cluster DEG that matched a PanglaoDB marker."
            )
        else:
            for gene in supporting:
                if not isinstance(gene, str) or not gene.strip():
                    continue
                nuisance_reason = _annotation_nuisance_reason(gene)
                if nuisance_reason:
                    nuisance_supporting.append({"gene": gene.strip(), "reason": nuisance_reason})
                else:
                    non_nuisance_supporting.append(gene.strip())
                    broad_reason = _annotation_broad_support_reason(gene)
                    if broad_reason:
                        broad_supporting.append({"gene": gene.strip(), "reason": broad_reason})
            checks["nuisance_supporting_genes"] = nuisance_supporting
            checks["broad_supporting_genes"] = broad_supporting
            checks["non_nuisance_supporting_genes"] = non_nuisance_supporting
            if not non_nuisance_supporting:
                validation_failures.append(
                    f"Cluster {cid}: supporting_genes are all nuisance/non-specific genes "
                    f"({[g.get('gene') for g in nuisance_supporting[:10]]}). "
                    "MT, ribosomal, hemoglobin, MALAT1, and generic locus genes cannot be the sole support for a cell-type label."
                )
            cluster_genes = cluster_deg_genes.get(cid, set())
            if cluster_genes:
                supporting_upper = {
                    str(g).strip().upper() for g in supporting if isinstance(g, str) and g.strip()
                }
                matched_degs = sorted(supporting_upper & cluster_genes)
                nuisance_upper = {
                    str(g.get("gene")).strip().upper()
                    for g in nuisance_supporting
                    if isinstance(g, dict) and g.get("gene")
                }
                non_nuisance_matched_degs = [
                    g for g in matched_degs if g not in nuisance_upper
                ]
                broad_upper = {
                    str(g.get("gene")).strip().upper()
                    for g in broad_supporting
                    if isinstance(g, dict) and g.get("gene")
                }
                discriminating_matched_degs = [
                    g for g in non_nuisance_matched_degs if g not in broad_upper
                ]
                checks["supporting_genes_matched_cluster_degs"] = matched_degs
                checks["n_supporting_genes_matched_cluster_degs"] = len(matched_degs)
                checks["non_nuisance_supporting_genes_matched_cluster_degs"] = non_nuisance_matched_degs
                checks["n_non_nuisance_supporting_genes_matched_cluster_degs"] = len(non_nuisance_matched_degs)
                checks["discriminating_supporting_genes_matched_cluster_degs"] = discriminating_matched_degs
                checks["n_discriminating_supporting_genes_matched_cluster_degs"] = len(discriminating_matched_degs)
                if not matched_degs:
                    validation_failures.append(
                        f"Cluster {cid}: none of supporting_genes={supporting[:10]} appear in this "
                        f"cluster's top DEGs. Cite genes that are actually differentially expressed in cluster {cid}."
                    )
                elif not non_nuisance_matched_degs:
                    validation_failures.append(
                        f"Cluster {cid}: supporting genes matched cluster DEGs, but all matched genes are nuisance/non-specific "
                        f"({matched_degs[:10]}). Provide at least one non-nuisance lineage marker among this cluster's DEGs."
                    )
                elif not discriminating_matched_degs:
                    validation_failures.append(
                        f"Cluster {cid}: supporting genes matched cluster DEGs, but all matched non-nuisance genes are broad/context markers "
                        f"({non_nuisance_matched_degs[:10]}). Add at least one discriminating marker DEG for the final label; "
                        "broad immune, stress, interferon, MHC, housekeeping, and generic myeloid markers cannot decide a label alone."
                    )

        final_label_text = label if isinstance(label, str) else ""
        panglao_label_text = (
            effective_panglaodb_label_used.strip()
            if isinstance(effective_panglaodb_label_used, str) and effective_panglaodb_label_used.strip()
            else final_label_text
        )
        panglao_exact_or_fine = _labels_exact_or_alias(final_label_text, panglao_label_text)
        panglao_compatible = _labels_biologically_compatible(final_label_text, panglao_label_text)
        broad_lineage_only = bool(
            panglao_label_text
            and final_label_text
            and not panglao_exact_or_fine
            and panglao_compatible
            and not reverse_hit
        )
        if panglao_label_text and final_label_text and not panglao_compatible and not reverse_hit:
            validation_failures.append(
                f"Cluster {cid}: panglaodb_label_used={panglao_label_text!r} is not biologically compatible "
                f"with final label {final_label_text!r}. Use a compatible PanglaoDB label, exact label query, "
                "or reverse marker support."
            )
        if reverse_hit and non_nuisance_matched_degs:
            panglaodb_support_level = "reverse_marker_plus_deg"
        elif label_in_history and panglao_exact_or_fine and non_nuisance_matched_degs:
            panglaodb_support_level = "fine_label_plus_deg"
        elif label_in_history and broad_lineage_only and non_nuisance_matched_degs:
            panglaodb_support_level = "broad_lineage_only"
        elif label_in_history:
            panglaodb_support_level = "label_queried_but_weak_deg_support"
        else:
            panglaodb_support_level = "self_attested_or_unavailable_history"
        checks["panglaodb_support_level"] = panglaodb_support_level

        proposal_entry = proposal_cluster_entries.get(cid, {})
        reference_consensus = _reference_consensus_for_cluster(proposal_entry)
        if reference_consensus.get("has_consensus"):
            checks["reference_consensus"] = {
                "label": reference_consensus.get("label"),
                "family": reference_consensus.get("family"),
                "sources": reference_consensus.get("sources", []),
                "labels": reference_consensus.get("labels", []),
                "score": reference_consensus.get("score"),
            }
            checks["final_label_matches_reference_consensus"] = _labels_biologically_compatible(
                final_label_text,
                reference_consensus.get("label"),
            )
        elif reference_consensus.get("reference_groups"):
            checks["reference_groups"] = reference_consensus.get("reference_groups")

        competing = ev.get("competing_labels_considered")
        if cid in ambiguous_set:
            if not isinstance(competing, list) or len(competing) == 0:
                inferred_competing: List[str] = []
                for comp in proposal_entry.get("competing_labels", []) or []:
                    if isinstance(comp, dict):
                        comp_label = comp.get("label") or comp.get("cell_type") or comp.get("candidate")
                    else:
                        comp_label = comp
                    if isinstance(comp_label, str) and comp_label.strip():
                        inferred_competing.append(comp_label.strip())
                label_for_filter = label.strip() if isinstance(label, str) else ""
                inferred_competing = [
                    x for i, x in enumerate(inferred_competing)
                    if x and x != label_for_filter and x not in inferred_competing[:i]
                ]
                if inferred_competing:
                    ev = dict(ev)
                    ev["competing_labels_considered"] = inferred_competing
                    evidence_str[cid] = ev
                    competing = inferred_competing
                    auto_fixes.append(
                        f"Cluster {cid}: filled competing_labels_considered from prepare_annotation proposal."
                    )
                else:
                    validation_failures.append(
                        f"Cluster {cid} was flagged ambiguous by prepare_annotation but evidence provides no competing_labels_considered."
                    )
        checks["competing_labels_considered"] = competing or []
        ref_support = ev.get("reference_annotation_support")
        if reference_keys:
            ref_support_ok = False
            if isinstance(ref_support, dict):
                ref_support_ok = any(
                    value not in (None, "", [], {})
                    for value in ref_support.values()
                )
            elif isinstance(ref_support, list):
                ref_support_ok = len(ref_support) > 0
            elif isinstance(ref_support, str):
                ref_support_ok = bool(ref_support.strip())
            if not ref_support_ok:
                validation_failures.append(
                    f"Cluster {cid}: missing reference_annotation_support for reference columns {reference_keys}."
                )
        if "reference_annotation_support" in ev:
            checks["reference_annotation_support"] = ev.get("reference_annotation_support")
        if "reference_annotation_conflicts" in ev:
            conflicts = ev.get("reference_annotation_conflicts")
            checks["reference_annotation_conflicts"] = conflicts if isinstance(conflicts, list) else [str(conflicts)]
        if "reverse_marker_support" in ev:
            checks["reverse_marker_support"] = ev.get("reverse_marker_support")
        if "panglaodb_label_used" in ev:
            checks["panglaodb_label_used"] = (
                str(effective_panglaodb_label_used)
                if effective_panglaodb_label_used
                else str(ev.get("panglaodb_label_used"))
            )
        if "external_sources" in ev:
            external_sources = ev.get("external_sources")
            checks["external_sources"] = external_sources if isinstance(external_sources, list) else [str(external_sources)]
        reasoning = ev.get("reasoning")
        if not isinstance(reasoning, str) or len(reasoning.strip()) < 20:
            validation_failures.append(
                f"Cluster {cid}: reasoning must explain the final label using reference labels, DEGs, PanglaoDB evidence, and alternatives considered."
            )
        else:
            checks["reasoning"] = reasoning.strip()

        conf = ev.get("confidence")
        if conf not in {"high", "medium", "low"}:
            validation_failures.append(f"Cluster {cid}: confidence must be 'high', 'medium', or 'low'.")
        checks["confidence_original"] = conf

        # Deterministic confidence-cap fixes — auto-downgrade rather than reject
        # when the support level already tells us what the right ceiling is.
        if apply_auto_fixes and conf == "high":
            if panglaodb_support_level == "broad_lineage_only":
                ev = dict(ev)
                ev["confidence"] = "medium"
                evidence_str[cid] = ev
                conf = "medium"
                auto_fixes.append(
                    f"Cluster {cid}: auto-lowered confidence high → medium because PanglaoDB only validated "
                    f"a broader parent lineage ({panglao_label_text!r}) for {final_label_text!r}."
                )
            elif panglaodb_support_level == "label_queried_but_weak_deg_support":
                ev = dict(ev)
                ev["confidence"] = "medium"
                evidence_str[cid] = ev
                conf = "medium"
                auto_fixes.append(
                    f"Cluster {cid}: auto-lowered confidence high → medium because PanglaoDB label was queried "
                    "but cluster DEGs only weakly match its markers."
                )
            elif panglaodb_support_level == "self_attested_or_unavailable_history":
                ev = dict(ev)
                ev["confidence"] = "low"
                evidence_str[cid] = ev
                conf = "low"
                auto_fixes.append(
                    f"Cluster {cid}: auto-lowered confidence high → low because no PanglaoDB call history "
                    "backs this label (self-attested)."
                )
        checks["confidence"] = conf
        # If auto-fixes are disabled we still emit the prior strict messages
        # so callers that want the raw rejection report can see them.
        if not apply_auto_fixes:
            if conf == "high" and panglaodb_support_level == "broad_lineage_only":
                validation_failures.append(
                    f"Cluster {cid}: confidence cannot be high when PanglaoDB only validates a broader parent lineage "
                    f"({panglao_label_text!r}) for final label {final_label_text!r}. Add exact/fine-label or reverse-marker support, "
                    "or lower confidence."
                )
            if conf == "high" and panglaodb_support_level in {
                "label_queried_but_weak_deg_support",
                "self_attested_or_unavailable_history",
            }:
                validation_failures.append(
                    f"Cluster {cid}: confidence cannot be high with weak PanglaoDB/DEG support "
                    f"({panglaodb_support_level})."
                )

        qc_caveats = proposal_entry.get("qc_annotation_caveats") or []
        if isinstance(qc_caveats, list) and qc_caveats:
            checks["qc_annotation_caveats"] = qc_caveats
            strictest_cap = "high"
            for caveat in qc_caveats:
                if not isinstance(caveat, dict):
                    continue
                cap = caveat.get("confidence_cap")
                if cap in {"low", "medium", "high"} and _confidence_rank(cap) < _confidence_rank(strictest_cap):
                    strictest_cap = cap
            checks["qc_confidence_cap"] = strictest_cap
            if conf in {"high", "medium", "low"} and _confidence_exceeds(conf, strictest_cap):
                if apply_auto_fixes:
                    ev = dict(ev)
                    ev["confidence"] = strictest_cap
                    evidence_str[cid] = ev
                    auto_fixes.append(
                        f"Cluster {cid}: auto-lowered confidence {conf!r} → {strictest_cap!r} to honour "
                        "the QC-derived cap (high MT, doublet enrichment, low complexity, or structure-QC review)."
                    )
                    conf = strictest_cap
                    checks["confidence"] = conf
                else:
                    validation_failures.append(
                        f"Cluster {cid}: confidence={conf!r} exceeds QC-derived cap {strictest_cap!r}. "
                        "Clusters with high MT, doublet enrichment, low complexity, or structure-QC review/conflict "
                        "must be labeled lower-confidence unless the evidence explicitly resolves the caveat."
                    )

        source_synthesis = ev.get("source_synthesis")
        if isinstance(source_synthesis, str) and source_synthesis.strip():
            text = source_synthesis.strip()
            source_synthesis = {
                "agreement": text,
                "final_decision_basis": (
                    reasoning.strip()
                    if isinstance(reasoning, str) and len(reasoning.strip()) >= 20
                    else text
                ),
            }
            ev = dict(ev)
            ev["source_synthesis"] = source_synthesis
            evidence_str[cid] = ev
            auto_fixes.append(
                f"Cluster {cid}: converted string source_synthesis into structured agreement/final_decision_basis."
            )
        elif isinstance(source_synthesis, dict):
            source_synthesis = dict(source_synthesis)
            if (
                not source_synthesis.get("final_decision_basis")
                and isinstance(source_synthesis.get("decision_basis"), str)
                and source_synthesis.get("decision_basis", "").strip()
            ):
                source_synthesis["final_decision_basis"] = source_synthesis["decision_basis"].strip()
                ev = dict(ev)
                ev["source_synthesis"] = source_synthesis
                evidence_str[cid] = ev
                auto_fixes.append(
                    f"Cluster {cid}: used source_synthesis.decision_basis as final_decision_basis."
                )
            if (
                not source_synthesis.get("agreement")
                and any(source_synthesis.get(k) for k in ("celltypist", "scimilarity", "panglaodb_evidence"))
            ):
                parts = [
                    str(source_synthesis.get(k))
                    for k in ("celltypist", "scimilarity", "panglaodb_evidence")
                    if source_synthesis.get(k)
                ]
                source_synthesis["agreement"] = "; ".join(parts)
                ev = dict(ev)
                ev["source_synthesis"] = source_synthesis
                evidence_str[cid] = ev
                auto_fixes.append(
                    f"Cluster {cid}: derived source_synthesis.agreement from source fields."
                )
        if reference_keys or missing_reference_sources or qc_caveats:
            if not isinstance(source_synthesis, dict) or not source_synthesis:
                validation_failures.append(
                    f"Cluster {cid}: source_synthesis is required. Summarize CellTypist/Scimilarity candidates, "
                    "DEG/PanglaoDB support, QC caveats, source agreement/discordance, and the final decision basis."
                )
            else:
                checks["source_synthesis"] = source_synthesis
                agreement = source_synthesis.get("agreement")
                basis = source_synthesis.get("final_decision_basis")
                if not isinstance(agreement, str) or not agreement.strip():
                    validation_failures.append(
                        f"Cluster {cid}: source_synthesis.agreement is required."
                    )
                if not isinstance(basis, str) or len(basis.strip()) < 20:
                    validation_failures.append(
                        f"Cluster {cid}: source_synthesis.final_decision_basis must explain the synthesized decision."
                    )

        if reference_consensus.get("has_consensus") and not checks.get("final_label_matches_reference_consensus"):
            consensus_label = reference_consensus.get("label")
            consensus_sources = reference_consensus.get("sources", [])
            candidate_competing_labels = []
            candidate_competing_labels.extend(_list_labels_from_evidence(competing))
            candidate_competing_labels.extend(_list_labels_from_evidence(ev.get("reference_annotation_conflicts")))
            consensus_considered = any(
                _labels_biologically_compatible(item, consensus_label)
                for item in candidate_competing_labels
            )
            basis_text = ""
            agreement_text = ""
            if isinstance(source_synthesis, dict):
                basis_text = str(source_synthesis.get("final_decision_basis") or "")
                agreement_text = str(source_synthesis.get("agreement") or "")
            combined_source_text = f"{agreement_text} {basis_text}".lower()
            sources_named = [
                source for source in consensus_sources
                if str(source).lower() in combined_source_text
            ]
            high_bar_checks = {
                "fine_or_reverse_panglaodb_support": panglaodb_support_level in {
                    "fine_label_plus_deg",
                    "reverse_marker_plus_deg",
                },
                "n_discriminating_deg_markers": len(discriminating_matched_degs),
                "has_three_discriminating_deg_markers": len(discriminating_matched_degs) >= 3,
                "reference_consensus_considered_as_competitor": consensus_considered,
                "source_synthesis_names_consensus_sources": len(sources_named) == len(consensus_sources),
                "source_synthesis_basis_is_detailed": len(basis_text.strip()) >= 60,
            }
            high_bar_passed = all(high_bar_checks.values())
            checks["reference_consensus_override"] = {
                "consensus_label": consensus_label,
                "consensus_sources": consensus_sources,
                "consensus_labels": reference_consensus.get("labels", []),
                "final_label": final_label_text,
                "high_bar_checks": high_bar_checks,
                "passed": high_bar_passed,
            }
            if not high_bar_passed:
                validation_failures.append(
                    f"Cluster {cid}: final label {final_label_text!r} crosses lineages away from "
                    f"CellTypist/Scimilarity reference consensus {consensus_label!r} "
                    f"({consensus_sources}). Cross-lineage overrides require exact/reverse PanglaoDB "
                    "support, at least three discriminating non-broad supporting DEGs, the consensus label "
                    "listed as a competing label/conflict, and a detailed source_synthesis explaining why "
                    "the reference consensus lost."
                )

        per_cluster_validation[cid] = checks

    if not any_panglaodb:
        validation_failures.append(
            "No cluster has panglaodb_queried=true. At least one PanglaoDB query must back the labels — "
            "this tool refuses to write annotations without external marker validation."
        )

    return {
        "validation_failures": validation_failures,
        "per_cluster_validation": per_cluster_validation,
        "auto_fixes": auto_fixes,
        "evidence_str": evidence_str,
        "missing_clusters": missing_clusters,
        "unknown_clusters": unknown_clusters,
        "missing_reference_sources": missing_reference_sources,
        "unexplained_missing_sources": unexplained_missing_sources,
        "tool_recorded_unavailable_sources": tool_recorded_unavailable_sources,
        "manual_unavailable_sources": manual_unavailable_sources,
        "unavailable_reference_sources": unavailable_reference_sources,
        "scimilarity_availability": scimilarity_availability,
        "any_panglaodb": any_panglaodb,
        "ambiguous_set": ambiguous_set,
        "reference_keys": reference_keys,
        "proposal_clusters": proposal_clusters,
        "proposal_cluster_entries": proposal_cluster_entries,
    }


def get_tools(include_describe_image: bool = False) -> List[Dict[str, Any]]:
    """
    Get Claude API tool definitions for single-cell analysis.

    Parameters
    ----------
    include_describe_image : bool
        If True, also register the ``describe_image`` tool that routes a saved
        figure through the vision sidecar. The agent should only enable this
        when ``_use_sidecar_for_images()`` is True so the tool stays invisible
        to multimodal-main runs.

    Returns
    -------
    List[Dict]
        List of tool definitions in Claude API format.
    """
    # Action tools (mutate state)
    action_tools = [
        {
            "name": "load_data",
            "description": "Replace the primary in-memory dataset with a new file. Use this when the user explicitly wants to switch focus to a different dataset. Always save the current primary with save_data first if it has been processed. Returns full inspection info (shape, state, obs columns, batch metadata) — do NOT call inspect_data afterwards.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to the h5ad or 10X h5 file to load as the new primary dataset."},
                    "goal": {"type": "string", "description": "Analysis goal hint (e.g., 'qc', 'cluster', 'annotate')"},
                    "context": {"type": "string", "description": "Optional biological context hint (e.g., 'PBMC healthy human')"}
                },
                "required": ["data_path"]
            }
        },
        {
            "name": "run_cellbender",
            "description": (
                "Run CellBender remove-background on a raw/unfiltered droplet matrix before standard scagent analysis. "
                "Use this for ambient RNA/background removal when the user provides raw droplet data. "
                "Do not run this on already filtered, normalized, or post-CellBender data. "
                "The tool validates inputs, captures stdout/stderr logs, verifies the output h5, and only loads "
                "the cleaned output as the primary dataset when load_output=true."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_path": {"type": "string", "description": "Path to the raw/unfiltered input for CellBender. Prefer raw_feature_bc_matrix.h5; CellBender also supports some raw matrix directories and other unfiltered formats."},
                    "output_path": {"type": "string", "description": "Path for CellBender's cleaned output h5. Defaults to <run_dir>/cellbender/<input_stem>_cellbender.h5."},
                    "expected_cells": {"type": "integer", "description": "Optional CellBender --expected-cells value. For CellBender v0.3+, omit this initially unless defaults fail or the user/source provides a reason."},
                    "total_droplets_included": {"type": "integer", "description": "Optional CellBender --total-droplets-included value. For CellBender v0.3+, omit this initially unless defaults fail or UMI-curve review supports a manual value."},
                    "fpr": {"type": "number", "description": "Optional CellBender --fpr value. Default is CellBender's own conservative setting; larger values remove more background but risk removing signal."},
                    "epochs": {"type": "integer", "description": "Optional CellBender --epochs value."},
                    "use_cuda": {"type": "boolean", "description": "If true, pass --cuda to CellBender. Only use when GPU availability has been checked."},
                    "cellbender_executable": {"type": "string", "description": "Executable or absolute path. Defaults to SCAGENT_CELLBENDER, then 'cellbender' on PATH."},
                    "extra_args": {"type": "array", "items": {"type": "string"}, "description": "Advanced extra command-line args for cellbender remove-background. Do not include --input or --output."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds. Default is 86400 (24 hours)."},
                    "workdir": {"type": "string", "description": "Working directory for the CellBender subprocess. Defaults to the output directory."},
                    "load_output": {"type": "boolean", "description": "If true, load the CellBender output h5 as the primary in-memory dataset after success. Default false."},
                    "force_replace_primary": {"type": "boolean", "description": "Required with load_output=true when a primary dataset is already loaded."}
                },
                "required": ["input_path"]
            }
        },
        {
            "name": "run_qc",
            "description": "Compute QC metrics and run doublet detection. Default behavior (flag_only=true) computes metrics, stores QC flags as obs columns, generates violin plots with log1p-transformed counts for readable axes, and does NOT remove any cells — removal happens later at cluster level via run_cluster_qc. Use confirm_filtering=true only when the user explicitly requests global threshold-based filtering as a fallback.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad or 10X h5 file (required for initial load, optional if data already in memory)"},
                    "output_path": {"type": "string", "description": "Optional path to save a processed h5ad. Prefer saving only final outputs unless the user explicitly asks for checkpoints."},
                    "flag_only": {"type": "boolean", "description": "If true (default), compute metrics and store QC flags (qc_flag_high_mt, qc_flag_low_lib, qc_flag_low_genes) but do not remove any cells. This is the standard first pass — removal decisions are made after clustering via run_cluster_qc."},
                    "mt_flag_threshold": {"type": "number", "description": "MT% threshold for qc_flag_high_mt (default: 25 for cells, 5 for nuclei). Used for flagging only when flag_only=true."},
                    "lib_flag_threshold": {"type": "number", "description": "Library size threshold for qc_flag_low_lib (default: 500 counts). Used for flagging only when flag_only=true."},
                    "genes_flag_threshold": {"type": "integer", "description": "n_genes threshold for qc_flag_low_genes (default: 200). Used for flagging only when flag_only=true."},
                    "preview_only": {"type": "boolean", "description": "Legacy alias for flag_only. If true, do not filter. Compute metrics, estimate removals, and generate QC figures."},
                    "confirm_filtering": {"type": "boolean", "description": "Required to apply global threshold-based cell/gene filtering. Use only when the user explicitly requests this approach instead of cluster-level QC. Set true only after previewing thresholds and removal counts."},
                    "data_type": {"type": "string", "enum": ["single_cell", "single_nucleus"], "description": "Hint for MT threshold direction: 'single_nucleus' starts at 5%, 'single_cell' at 20%. The actual threshold must be chosen from the QC figure — always inspect the distribution before filtering."},
                    "mt_threshold": {"type": "number", "description": "Max MT% threshold. Overrides data_type if provided."},
                    "filter_mt": {"type": "boolean", "description": "If false, compute and report MT metrics but do not apply a hard MT% cell filter. Use this for source pipelines that inspect MT but do not remove cells by MT%."},
                    "min_genes": {"type": "integer", "description": "Minimum detected genes per cell before cell removal. This is cell-level filtering, distinct from min_cells per gene."},
                    "min_cells": {"type": "integer", "description": "Minimum cells a gene must be expressed in to be kept (default: 3). In preview, shows how many genes would be removed. Present this to the user alongside the projected removal count and confirm before applying."},
                    "remove_ribo": {"type": "boolean", "description": "Advanced QC gene-filter option. Leave false in the standard workflow; normalize_and_hvg performs the project-default ribosomal gene removal before normalization/HVG."},
                    "remove_mt": {"type": "boolean", "description": "Remove mitochondrial genes from the feature set (default: false)"},
                    "detect_doublets_flag": {"type": "boolean", "description": "Run Scrublet doublet detection (default: true)"},
                    "remove_doublets": {"type": "boolean", "description": "If true, remove cells flagged as predicted doublets in apply mode. Preview mode reports the count only."},
                    "scrublet_expected_doublet_rate": {"type": "number", "description": "Scrublet expected_doublet_rate (default: 0.06)."},
                    "scrublet_sim_doublet_ratio": {"type": "number", "description": "Scrublet sim_doublet_ratio (default: 2.0)."},
                    "scrublet_n_prin_comps": {"type": "integer", "description": "Scrublet n_prin_comps / PCA components (default: 30). Set to 40 to match some published pipelines."},
                    "scrublet_min_counts": {"type": "integer", "description": "Scrublet scrub_doublets min_counts preprocessing parameter (default: 2)."},
                    "scrublet_min_cells": {"type": "integer", "description": "Scrublet scrub_doublets min_cells preprocessing parameter (default: 3)."},
                    "scrublet_min_gene_variability_pctl": {"type": "number", "description": "Scrublet scrub_doublets min_gene_variability_pctl preprocessing parameter (default: 85)."},
                    "scrublet_random_state": {"type": "integer", "description": "Random seed for Scrublet (default: 0)."},
                    "force_doublet_recompute": {"type": "boolean", "description": "If true, recompute Scrublet scores even if doublet columns already exist."},
                    "figure_dir": {"type": "string", "description": "Directory for QC figures. Plots are generated from the full pre-filter data."},
                    "batch_key": {"type": "string", "description": "Batch column for per-batch doublet detection"}
                },
                "required": []
            }
        },
        {
            "name": "normalize_and_hvg",
            "description": "Normalize, log-transform, and select highly variable genes. Preserves raw counts in a layer. By default, ribosomal genes are removed from the analysis object before normalization/HVG so they cannot drive embedding or marker interpretation; set remove_ribosomal_genes=false when the user/source explicitly wants to keep them.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "n_hvg": {"type": "integer", "description": "Number of HVGs (default: 4000)"},
                    "target_sum": {"type": "number", "description": "Target counts per cell for normalize_total (default: 10000). Use source/paper value when reproducing a workflow."},
                    "log_transform": {"type": "boolean", "description": "Apply log1p after normalize_total (default: true)."},
                    "raw_layer_name": {"type": "string", "description": "Layer used to preserve/reset raw integer counts (default: raw_counts)."},
                    "normalization_source": {"type": "string", "enum": ["auto", "raw_counts", "current_X"], "description": "Source matrix for normalization (default: auto). 'auto' resets from raw_layer_name when X already looks processed; 'raw_counts' forces a raw-count rebuild; 'current_X' is an expert override."},
                    "preserve_input_x_layer": {"type": "string", "description": "When resetting from raw counts, preserve the pre-reset X in this layer if absent (default: pre_scagent_X). Set to empty string to disable."},
                    "set_raw_after_normalization": {"type": "boolean", "description": "If true, set adata.raw = adata.copy() after normalization/log1p and before later scaling/PCA (default: true)."},
                    "hvg_flavor": {"type": "string", "enum": ["seurat", "seurat_v3", "cell_ranger"], "description": "Scanpy HVG flavor (default: seurat_v3). seurat_v3 uses VST on raw counts and supports batch_key (ranks by median rank across batches). seurat works on log-normalized data."},
                    "hvg_layer": {"type": "string", "description": "Layer for HVG calculation. seurat_v3 requires raw integer counts; if omitted, auto-detects from 'raw_counts', 'raw_data', 'counts' in that order. Only set explicitly if your raw counts are in a non-standard layer."},
                    "batch_key": {"type": "string", "description": "obs column for batch-stratified HVG selection. Recommended for multi-sample data. Supported by all flavors including seurat_v3."},
                    "remove_ribosomal_genes": {"type": "boolean", "description": "If true (default), physically remove ribosomal genes from this analysis object before normalization/HVG. Set false only when the user/source explicitly says to keep ribosomal genes."},
                    "ribosomal_remove_patterns": {"type": "array", "items": {"type": "string"}, "description": "Regex pattern(s) for ribosomal genes to remove when remove_ribosomal_genes=true. Defaults to human/mouse cytosolic and mitochondrial ribosomal prefixes: RPL/RPS/MRPL/MRPS and Rpl/Rps/Mrpl/Mrps."},
                    "exclude_ribosomal_from_hvg": {"type": "boolean", "description": "Compatibility/advanced option used only when remove_ribosomal_genes=false. If true, keep ribosomal genes in adata but exclude them before HVG selection. Set false only if the user/source explicitly wants ribosomal genes included in HVG/PCA."},
                    "hvg_exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Additional regex pattern(s) for features that must not be marked highly_variable. Use for evidence-backed source/workflow exclusions."},
                    "hvg_exclusion_mode": {"type": "string", "enum": ["post", "pre"], "description": "How to apply HVG exclusions. 'pre' computes HVGs only on allowed features; 'post' runs HVG then forces excluded features to false (default: pre)."},
                    "hvg_exclude_match_mode": {"type": "string", "enum": ["match", "contains", "fullmatch"], "description": "Regex matching mode for hvg_exclude_patterns against var_names (default: match)."},
                    "hvg_exclusion_source": {"type": "string", "description": "Short provenance for any additional feature-exclusion rule, e.g. source repo file/function/line or paper method."}
                },
                "required": []
            }
        },
        {
            "name": "run_pca",
            "description": "Run PCA only. Does not compute neighbors, UMAP, clustering, or batch correction.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "n_comps": {"type": "integer", "description": "Number of PCA components (default: 30)"},
                    "svd_solver": {"type": "string", "description": "SVD solver passed to scanpy.tl.pca (default: arpack)"},
                    "mask_var": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Boolean var column for PCA feature mask, or null for all genes (default: highly_variable)"}
                },
                "required": []
            }
        },
        {
            "name": "run_neighbors",
            "description": "Compute a neighbor graph only. Does not run PCA, UMAP, clustering, or batch correction.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "n_neighbors": {"type": "integer", "description": "Number of neighbors (default: 30)"},
                    "n_pcs": {"type": "integer", "description": "Number of PCs to use from the representation (optional)"},
                    "use_rep": {"type": "string", "description": "Representation in adata.obsm to use (default: X_pca)"},
                    "metric": {"type": "string", "description": "Distance metric (default: euclidean)"},
                    "key_added": {"type": "string", "description": "Optional alternate neighbors key. Omit to write the default graph."}
                },
                "required": []
            }
        },
        {
            "name": "run_umap",
            "description": "Compute UMAP only from an existing neighbor graph. Does not recompute PCA, neighbors, batch correction, or clustering.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "min_dist": {"type": "number", "description": "UMAP min_dist (default: 0.5, Scanpy's default)"},
                    "spread": {"type": "number", "description": "UMAP spread (default: 1.0)"},
                    "n_components": {"type": "integer", "description": "Number of UMAP dimensions (default: 2)"},
                    "neighbors_key": {"type": "string", "description": "Optional neighbors key to use. Omit to use adata.uns['neighbors']."},
                    "random_state": {"type": "integer", "description": "Random seed (default: 0)"}
                },
                "required": []
            }
        },
        {
            "name": "run_clustering",
            "description": "Run Leiden or PhenoGraph clustering. Preserves alternative clustering results under explicit keys so comparisons do not overwrite the primary clustering by accident.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "method": {"type": "string", "enum": ["leiden", "louvain", "phenograph"], "description": "Method (default: leiden)"},
                    "resolution": {"type": "number", "description": "Resolution (default: 1.0)"},
                    "k": {"type": "integer", "description": "PhenoGraph nearest-neighbor k (default: 30; ignored for Leiden)"},
                    "use_rep": {"type": "string", "description": "Representation for PhenoGraph clustering (default: X_pca; ignored for Leiden)"},
                    "random_state": {"type": "integer", "description": "Random seed for clustering when supported (default: 0)"},
                    "cluster_key": {"type": "string", "description": "Optional explicit obs column to store this clustering result. If omitted, scagent will keep primary aliases like 'leiden' stable and store comparisons under deterministic keys like 'leiden_res_0_5'."},
                    "make_primary": {"type": "boolean", "description": "If true, promote this clustering to the default alias for the method (for example 'leiden') while preserving the explicit result key."}
                },
                "required": []
            }
        },
        {
            "name": "compare_clusterings",
            "description": "Run a safe clustering comparison across multiple resolutions without overwriting earlier results. Use this instead of chaining several run_clustering calls when the goal is to compare resolutions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "method": {"type": "string", "enum": ["leiden", "louvain", "phenograph"], "description": "Method (default: leiden)"},
                    "resolutions": {"type": "array", "items": {"type": "number"}, "description": "List of resolutions to compare"},
                    "k": {"type": "integer", "description": "PhenoGraph nearest-neighbor k (default: 30; ignored for Leiden)"},
                    "use_rep": {"type": "string", "description": "Representation for PhenoGraph clustering (default: X_pca; ignored for Leiden)"},
                    "random_state": {"type": "integer", "description": "Random seed for clustering when supported (default: 0)"},
                    "generate_figures": {"type": "boolean", "description": "If true and UMAP is present, save one figure per clustering"},
                    "figure_dir": {"type": "string", "description": "Optional directory for generated comparison figures"},
                    "include_images": {"type": "boolean", "description": "If true, include image data for generated figures"},
                    "promote_resolution": {"type": "number", "description": "Optional resolution to promote to the primary alias after comparison"}
                },
                "required": ["resolutions"]
            }
        },
        {
            "name": "run_celltypist",
            "description": "Annotate cell types with CellTypist. Handles target_sum=10000 normalization automatically and checks dataset organism against model metadata before running.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "model": {"type": "string", "description": "Model name (default: Immune_All_Low.pkl)"},
                    "organism": {"type": "string", "enum": ["human", "mouse"], "description": "Dataset organism. Use explicit user-provided species when available; if ambiguous, ask before annotation."},
                    "allow_cross_species": {"type": "boolean", "description": "Expert override to run a species-mismatched CellTypist model as non-definitive exploratory output (default: false)."},
                    "majority_voting": {"type": "boolean", "description": "Use majority voting (default: true)"},
                    "cluster_key": {"type": "string", "description": "Cluster column to use for CellTypist majority voting (default: leiden)"}
                },
                "required": []
            }
        },
        {
            "name": "run_scimilarity",
            "description": "Annotate cell types with Scimilarity (embedding-based). Uses pretrained embeddings and kNN to annotate cells. Requires a known organism ('human' or 'mouse') or an explicit model_path; if species is ambiguous, ask before running. In the Iris scagent environment, Scimilarity model files are expected to be available, so run this whenever the organism is known instead of treating DEG-only annotation as sufficient.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data if already loaded)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "cluster_key": {"type": "string", "description": "Cluster column to use for cluster-level representative predictions (default: leiden)"},
                    "organism": {"type": "string", "description": "Dataset organism: 'human' or 'mouse'. Use explicit user-provided species when available."},
                    "model_path": {"type": "string", "description": "Optional explicit Scimilarity model directory. Overrides organism-based default paths."}
                },
                "required": []
            }
        },
        {
            "name": "prepare_annotation",
            "description": (
                "Prepare a structured annotation proposal for all clusters: compute DEGs, score marker genes "
                "against clusters using normalized expression fractions (not raw means), flag ambiguous clusters "
                "where top-2 candidate labels are close, and identify shared markers that don't discriminate. "
                "If CellTypist or Scimilarity columns are present, summarize their dominant cluster-level labels "
                "as reference-derived candidate labels so DEG/PanglaoDB validation can adjudicate them. "
                "Also stage reverse PanglaoDB marker lookups for a panel of top non-nuisance DEGs per cluster, "
                "so plausible alternative labels can be discovered from observed markers instead of relying on "
                "a single gene or hard-coded ambiguity lists. "
                "Stores the proposal in adata.uns['annotation_proposal'] and returns per-cluster candidates "
                "with competing labels and the specific PanglaoDB label and reverse-marker queries you must run next. "
                "This is the validation/adjudication stage, not a replacement for reference-based annotation "
                "when a compatible CellTypist or Scimilarity model is available. After this, query PanglaoDB "
                "for each proposed label, reference-derived candidate label, competing label, and staged DEG "
                "gene-symbol reverse lookup, then call finalize_annotation with the evidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster_key": {"type": "string", "description": "obs column with cluster labels (default: leiden)"},
                    "marker_dict": {
                        "type": "object",
                        "description": (
                            "Optional dict mapping cell-type label to list of marker gene names. "
                            "If provided, scoring uses normalized expression fraction (fraction of cells "
                            "expressing each marker > 0), averaged across all markers in the list. "
                            "This is less biased than raw mean expression and length-normalized. "
                            "Example: {\"T cell\": [\"CD3D\", \"CD3E\"], \"B cell\": [\"CD19\", \"MS4A1\"]}"
                        ),
                        "additionalProperties": {"type": "array", "items": {"type": "string"}}
                    },
                    "n_deg_genes": {"type": "integer", "description": "Number of top DEGs to extract per cluster for PanglaoDB comparison (default: 20)"},
                    "deg_key": {"type": "string", "description": "adata.uns key for existing DEG results (default: rank_genes_groups). If the key exists, DEGs are read from it; otherwise rank_genes_groups is run automatically."},
                    "annotation_key": {"type": "string", "description": "Name of the obs column that finalize_annotation will write (default: cell_type). Stored in the proposal so finalize_annotation knows where to write."},
                    "reference_annotation_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional obs columns containing reference-based candidate labels, such as "
                            "celltypist_majority_voting, celltypist_predicted_labels, "
                            "scimilarity_predictions_unconstrained, or scimilarity_representative_prediction. "
                            "If omitted, prepare_annotation auto-detects those standard columns when present."
                        )
                    },
                    "panglaodb_species": {"type": "string", "enum": ["Hs", "Mm"], "description": "Optional PanglaoDB species code to include in staged queries: Hs for human, Mm for mouse."},
                    "reverse_lookup_n_genes_per_cluster": {"type": "integer", "description": "Number of top non-nuisance DEGs per cluster to stage for PanglaoDB gene_symbol reverse lookup (default: 10; use 0 to disable)."},
                    "reverse_lookup_max_unique_genes": {"type": "integer", "description": "Maximum unique DEG gene_symbol reverse lookup queries to stage across all clusters (default: 60). Genes are selected round-robin across clusters so small/high-numbered clusters are represented without overwhelming context."},
                    "reverse_lookup_exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Regex patterns for genes to exclude from reverse lookup as nuisance markers. Defaults exclude MT, ribosomal, MALAT1, and common hemoglobin genes; cell-cycle genes are not excluded by default."
                    },
                    "ambiguity_threshold": {"type": "number", "description": "Max allowed difference between top-2 normalized scores for a cluster to be flagged as ambiguous (default: 0.10). Clusters with top-2 delta below this are flagged."},
                    "shared_marker_threshold": {"type": "number", "description": "Fraction of cell-type lists a gene must appear in to be considered a shared/non-discriminating marker (default: 0.5)."}
                },
                "required": []
            }
        },
        {
            "name": "stage_annotation_evidence",
            "description": (
                "Incrementally stage final annotation evidence after prepare_annotation and PanglaoDB queries. "
                "Use this when there are many clusters so evidence can be submitted in batches instead of one "
                "large finalize_annotation call. Merges entries into adata.uns['annotation_evidence_summary']; "
                "does not write labels. "
                "STAGING ALSO VALIDATES: this tool runs the full finalize-time validator on the merged evidence "
                "and surfaces every issue per cluster, so problems are visible at staging time rather than only "
                "when finalize is called. Deterministic fixes (e.g., lowering confidence when only a broader "
                "lineage was validated) are applied automatically and reported in `validation.auto_fixes`. "
                "Required per-cluster fields and rules: "
                "(1) `label`, `confidence` ∈ {high, medium, low}, `panglaodb_queried` (true), `supporting_genes` "
                "(non-empty, must overlap this cluster's top DEGs, must include at least one non-nuisance "
                "marker — MT/ribosomal/hemoglobin/MALAT1 genes alone do not count). "
                "(2) `panglaodb_label_used`: the PanglaoDB cell_type you queried that backs this label; must be "
                "biologically compatible with `label` (same lineage family) OR provide `reverse_marker_support` "
                "naming the gene_symbol query that returned this label. "
                "(3) Confidence is auto-capped from PanglaoDB support level: `medium` max when only a broad parent "
                "lineage was queried; `low` max when no PanglaoDB call history backs the label. QC-derived caps "
                "(high MT, doublet enrichment, low complexity, structure-QC review) auto-lower confidence too. "
                "(4) `reasoning`: ≥20 chars explaining the chosen label. "
                "(5) `source_synthesis`: `{agreement, final_decision_basis}` required when CellTypist/Scimilarity "
                "reference columns were used or QC caveats apply. "
                "(6) `competing_labels_considered`: required for clusters that prepare_annotation flagged ambiguous. "
                "(7) If CellTypist and Scimilarity agree on a lineage, the final label must stay compatible "
                "with that consensus unless the evidence clears the cross-lineage override gate: exact/reverse "
                "PanglaoDB support, at least three discriminating non-broad DEG markers, the consensus label "
                "listed as a competitor/conflict, and source_synthesis explicitly explaining why the consensus lost."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "evidence_summary": {
                        "type": ["object", "string"],
                        "description": (
                            "Dict mapping cluster_id to annotation evidence. Each entry should include label, "
                            "panglaodb_queried, supporting_genes, confidence, reasoning, and where relevant "
                            "competing_labels_considered. If reference annotation columns were used by "
                            "prepare_annotation, include reference_annotation_support for each cluster; add "
                            "reference_annotation_conflicts when CellTypist/Scimilarity disagree. May also be "
                            "a JSON string encoding the same dict. Use external_sources for literature/web "
                            "evidence used to resolve ambiguous cases."
                        ),
                        "additionalProperties": {"type": "object"},
                    },
                    "evidence_path": {
                        "type": "string",
                        "description": (
                            "Optional path to a JSON file containing the evidence_summary dict. Relative paths "
                            "are resolved from the run directory when available, then the current working directory. "
                            "Use this for large evidence payloads instead of passing huge JSON through the tool call."
                        ),
                    },
                    "replace": {
                        "type": "boolean",
                        "description": "If true, replace any previously staged annotation evidence. Default false merges entries.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "finalize_annotation",
            "description": (
                "Write final cell-type annotation labels to adata.obs after PanglaoDB evidence has been "
                "collected. Requires: (1) prepare_annotation was called first (proposal in adata.uns), "
                "(2) evidence_summary maps every cluster to a label with PanglaoDB evidence, or evidence "
                "has already been staged with stage_annotation_evidence. "
                "Writes adata.obs[annotation_key] and records the full evidence in adata.uns['annotation_validation']. "
                "This is step 2 of 2 — never call this before querying PanglaoDB for each proposed label "
                "and each competing label from the prepare_annotation output. "
                "Recommended flow: use stage_annotation_evidence (which runs the same validator and applies "
                "auto-fixes) to surface and resolve all issues, then call finalize_annotation. If you pass "
                "evidence directly, the same per-cluster rules apply: "
                "(1) `label`, `confidence` ∈ {high, medium, low}, `panglaodb_queried`=true, `supporting_genes` "
                "(non-empty, overlapping the cluster's top DEGs, at least one non-nuisance lineage marker). "
                "(2) `panglaodb_label_used` biologically compatible with `label`, or `reverse_marker_support` "
                "naming the gene_symbol query that returned this label. "
                "(3) Confidence is auto-capped to `medium` when PanglaoDB only validated a broader parent "
                "lineage, and to `low` when no PanglaoDB call history backs the label; QC-derived caps "
                "(high MT, doublets, low complexity, structure-QC review) also auto-lower confidence. "
                "(4) `reasoning` ≥20 chars. (5) `source_synthesis={agreement, final_decision_basis}` required "
                "when reference columns or QC caveats apply. (6) `competing_labels_considered` required for "
                "clusters that prepare_annotation flagged ambiguous. (7) CellTypist+Scimilarity consensus is "
                "trusted with DEGs as the biological anchor; cross-lineage overrides must pass the high-bar "
                "consensus override checks described in stage_annotation_evidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "evidence_summary": {
                        "type": "object",
                        "description": (
                            "Optional if stage_annotation_evidence has already staged all clusters. "
                            "Dict mapping cluster_id (as string) to annotation evidence. "
                            "Each entry must include: 'label' (final cell-type string), "
                            "'panglaodb_queried' (true/false), 'supporting_genes' (list of marker "
                            "genes that matched PanglaoDB), 'confidence' ('high'/'medium'/'low'). "
                            "When reference labels were used, include 'reference_annotation_support' "
                            "and 'reference_annotation_conflicts' so the final record preserves whether "
                            "CellTypist/Scimilarity agreed with the DEG/PanglaoDB evidence. "
                            "Include 'source_synthesis' with the CellTypist/Scimilarity/DEG/PanglaoDB/QC "
                            "agreement summary and final decision basis. "
                            "For more than a few clusters, stage evidence with evidence_path rather than "
                            "passing a large inline JSON string."
                            "When reverse marker lookup was used, include 'reverse_marker_support' "
                            "(candidate PanglaoDB labels and the DEG genes supporting each) and "
                            "'panglaodb_label_used' if the final biological label had to be validated "
                            "through a broader PanglaoDB vocabulary label. "
                            "Example: {\"0\": {\"label\": \"T cell\", \"panglaodb_queried\": true, "
                            "\"supporting_genes\": [\"CD3D\", \"CD3E\"], \"confidence\": \"high\"}}"
                        ),
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "panglaodb_queried": {"type": "boolean"},
                                "supporting_genes": {"type": "array", "items": {"type": "string"}},
                                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                "reference_annotation_support": {"type": "object"},
                                "reference_annotation_conflicts": {"type": "array", "items": {"type": "string"}},
                                "source_synthesis": {"type": "object"},
                                "reverse_marker_support": {"type": "object"},
                                "panglaodb_label_used": {"type": "string"},
                                "external_sources": {"type": "array", "items": {"type": "string"}},
                                "reasoning": {"type": "string"}
                            },
                            "required": ["label", "panglaodb_queried"]
                        }
                    },
                    "annotation_key": {"type": "string", "description": "obs column to write labels into (default: reads from adata.uns['annotation_proposal']['annotation_key'] or 'cell_type')"},
                    "cluster_key": {"type": "string", "description": "obs column with cluster ids (default: reads from adata.uns['annotation_proposal']['cluster_key'] or 'leiden')"},
                    "validate_only": {"type": "boolean", "description": "If true, run all evidence checks and return the validation report without writing adata.obs[annotation_key]. Use this before finalizing large staged evidence."},
                    "reference_source_unavailable": {
                        "type": "object",
                        "description": (
                            "Optional mapping of missing non-Scimilarity reference sources to concrete "
                            "unavailable reasons, e.g. {'celltypist': {'reason': 'no_species_compatible_model'}}. "
                            "Scimilarity unavailability is accepted only when a prior run_scimilarity tool call "
                            "recorded the blocker; manual finalize_annotation input cannot excuse an available "
                            "Scimilarity model."
                        ),
                    },
                    "overwrite": {"type": "boolean", "description": "If true, overwrite an existing annotation column (default: false — raises an error if the column already exists)"}
                },
                "required": []
            }
        },
        {
            "name": "run_batch_correction",
            "description": (
                "Correct batch effects using Harmony, BBKNN, Scanorama, or scVI. "
                "Harmony: fast, corrects PCA embeddings, good for mild-to-moderate batch effects. "
                "BBKNN: fast, builds a batch-balanced k-NN graph in PCA space; correction lives in "
                "the neighbor graph (not a separate embedding). Run UMAP/clustering separately unless explicitly requested. "
                "Good default when you have many samples (e.g. >10 batches). "
                "Scanorama: MNN-based, also corrects gene expression, good for partially overlapping datasets. "
                "scVI: deep generative model, models raw counts directly, best for complex/strong batch effects "
                "but requires raw_counts layer and takes longer to train (recommended max_epochs=200). "
                "This tool only performs batch correction. Run run_neighbors and run_umap as separate steps afterwards."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional)"},
                    "batch_key": {"type": "string", "description": "Column in adata.obs containing batch labels"},
                    "method": {
                        "type": "string",
                        "enum": ["harmony", "bbknn", "scanorama", "scvi"],
                        "description": (
                            "Correction method (default: harmony). "
                            "bbknn: batch-balanced graph, good for many batches (>10 samples). "
                            "scvi: best for complex effects but needs raw_counts layer."
                        )
                    },
                    "n_pcs": {"type": "integer", "description": "BBKNN only: number of PCA components to use (default: 30)"},
                    "neighbors_within_batch": {"type": "integer", "description": "BBKNN only: neighbors contributed per batch per cell (default: 3; total = n_batches × this value)"},
                    "n_latent": {"type": "integer", "description": "scVI only: latent space dimensions (default: 30)"},
                    "max_epochs": {"type": "integer", "description": "scVI only: training epochs (default: 200; use fewer only for quick tests)"},
                    "store_normalized": {"type": "boolean", "description": "scVI only: store scVI-normalized expression in layers['scvi_normalized'] (default: false)"}
                },
                "required": []
            }
        },
        {
            "name": "score_integration",
            "description": (
                "Score batch integration quality using neighborhood batch mixing entropy. "
                "For each cell, examines its k nearest neighbors in the chosen embedding and "
                "computes the Shannon entropy of batch labels — high entropy means batches are "
                "well-mixed. The score is normalized to [0, 1] where 1 = perfect mixing. "
                "Call this after run_batch_correction to quantify whether integration worked. "
                "Can also be called on uncorrected embeddings (use_rep='X_pca') as a baseline "
                "to compare before/after. Stores per-cell scores in obs['integration_entropy']."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "batch_key": {
                        "type": "string",
                        "description": "obs column with batch labels (same key used for batch correction)."
                    },
                    "use_rep": {
                        "type": "string",
                        "description": "Embedding to evaluate (default: 'X_umap'). Use the corrected embedding for post-integration score, or 'X_pca' for pre-integration baseline."
                    },
                    "n_neighbors": {
                        "type": "integer",
                        "description": "Neighborhood size for entropy calculation (default: 50). Larger = stabler but slower."
                    }
                },
                "required": ["batch_key"]
            }
        },
        {
            "name": "benchmark_integration",
            "description": (
                "Benchmark batch integration quality using scib-metrics — the same evaluation "
                "used in workshop session 5 to decide which correction method to keep. "
                "Computes bio-conservation metrics (NMI, ARI, silhouette label, cLISI) and "
                "batch-correction metrics (silhouette batch, iLISI, kBET, graph connectivity, PCR) "
                "across all corrected embeddings present in adata.obsm, always including X_pca as "
                "the uncorrected baseline. Returns a ranked table and the best-performing method. "
                "Requires: scib-metrics (pip install scib-metrics), a label_key with cell type or "
                "cluster annotations, and at least one corrected embedding from run_batch_correction."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "batch_key": {
                        "type": "string",
                        "description": "obs column with batch labels (same as used for batch correction)."
                    },
                    "label_key": {
                        "type": "string",
                        "description": "obs column with cell type or cluster labels for bio-conservation metrics (e.g. 'leiden', 'cell_type', 'celltypist_cell_type')."
                    },
                    "embedding_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "obsm keys to benchmark. Auto-detected if omitted (includes X_pca baseline + any corrected embeddings present)."
                    },
                    "fast": {
                        "type": "boolean",
                        "description": "Skip slow metrics (kBET) for a quicker result (default: false)."
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save results CSV and results-table figure (optional but recommended)."
                    }
                },
                "required": ["batch_key", "label_key"]
            }
        },
        {
            "name": "run_deg",
            "description": "Run validated differential expression analysis between groups. Validates input data (matrix type, cluster sizes, batch confounding) and attaches validity metadata for downstream GSEA interpretation. Returns validation warnings alongside DEG results.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional)"},
                    "groupby": {"type": "string", "description": "Group column (default: leiden)"},
                    "method": {"type": "string", "enum": ["wilcoxon", "t-test", "logreg"], "description": "Method (default: wilcoxon)"},
                    "layer": {"type": "string", "description": "Optional expression layer to use for DEG (for example scran_norm)"},
                    "use_raw": {"type": "boolean", "description": "Whether to use adata.raw for DEG when layer is not set. If omitted, follows Scanpy default (uses adata.raw when present)."},
                    "key_added": {"type": "string", "description": "Key in adata.uns for DEG results (default: rank_genes_groups)"},
                    "n_genes": {"type": "integer", "description": "Number of ranked genes to store per group (default: 100)"},
                    "target_geneset": {"type": "string", "description": "Target gene set database for compatibility check (default: MSigDB_Hallmark_2020)"}
                },
                "required": []
            }
        },
        {
            "name": "run_pseudobulk_deg",
            "description": (
                "Run pseudobulk differential expression analysis using DESeq2. "
                "Aggregates raw counts to the sample level (one observation per biological replicate "
                "per cell type) before running statistics — this respects replicate independence and "
                "is strongly preferred over single-cell Wilcoxon when biological replicates are available. "
                "Requires: raw integer counts in a layer (default 'raw_counts'), a sample column with "
                "≥ 2 replicates per condition, and a condition column. "
                "Use run_deg (Wilcoxon) when no replicates are available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sample_col": {
                        "type": "string",
                        "description": "Column in adata.obs identifying biological replicates (e.g. 'sample_id', 'donor'). Each unique value must be an independent sample."
                    },
                    "condition_col": {
                        "type": "string",
                        "description": "Column in adata.obs defining the condition to compare (e.g. 'disease', 'treatment', 'timepoint')."
                    },
                    "condition_a": {
                        "type": "string",
                        "description": "Reference condition (denominator in log fold change, e.g. 'healthy', 'control')."
                    },
                    "condition_b": {
                        "type": "string",
                        "description": "Test condition (numerator in log fold change, e.g. 'disease', 'treated'). Positive LFC means upregulated here."
                    },
                    "groups_col": {
                        "type": "string",
                        "description": "Column in adata.obs containing cell type or cluster labels (e.g. 'leiden', 'cell_type'). Used to subset to a specific cell type."
                    },
                    "cell_type": {
                        "type": "string",
                        "description": "Specific cell type label from groups_col to analyze. If omitted, runs on all cells together (use when adata is already subset)."
                    },
                    "layer": {
                        "type": "string",
                        "description": "Layer containing raw integer counts (default: 'raw_counts'). DESeq2 requires non-normalized counts."
                    },
                    "min_cells": {
                        "type": "integer",
                        "description": "Minimum cells a sample must contribute to pseudobulk to be retained (default: 10). Samples below this threshold are dropped."
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Adjusted p-value threshold for significance reporting (default: 0.05)."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save full results as CSV (optional)."
                    }
                },
                "required": ["sample_col", "condition_col", "condition_a", "condition_b", "groups_col"]
            }
        },
        {
            "name": "generate_figure",
            "description": "Generate and save a visualization (UMAP, violin, dotplot, etc.). For clustering comparisons, always use an explicit cluster key returned by run_clustering or compare_clusterings rather than a bare primary alias unless you intentionally want the promoted default.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save PNG figure"},
                    "plot_type": {"type": "string", "enum": ["umap", "tsne", "violin", "dotplot", "heatmap"], "description": "Plot type. Use 'tsne' when the dataset has obsm['X_tsne'] but no UMAP (e.g. when reproducing a paper that uses t-SNE)."},
                    "color_by": {"type": "string", "description": "Column or gene to color by"},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Genes for dotplot/heatmap"},
                    "include_image": {"type": "boolean", "description": "If true, include image data for model review (default: true)"}
                },
                "required": ["output_path", "plot_type"]
            }
        },
        {
            "name": "run_gsea",
            "description": "Run Gene Set Enrichment Analysis on DEG results. Identifies enriched biological pathways/processes. Requires DEG to be run first. Returns top enriched pathways with NES scores and FDR values.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad with DEG results (optional - uses in-memory data)"},
                    "output_dir": {"type": "string", "description": "Directory to save GSEA results"},
                    "cluster": {"type": "string", "description": "Cluster to analyze (or 'all' for all clusters)"},
                    "gene_sets": {"type": "string", "description": "Gene set database: KEGG_2021_Human, GO_Biological_Process_2021, Reactome_2022, MSigDB_Hallmark_2020 (default: KEGG_2021_Human)"},
                    "min_size": {"type": "integer", "description": "Min genes in pathway (default: 5)"},
                    "max_size": {"type": "integer", "description": "Max genes in pathway (default: 500)"},
                    "permutation_num": {"type": "integer", "description": "Permutations for p-value (default: 1000)"}
                },
                "required": ["output_dir", "cluster"]
            }
        },
        {
            "name": "run_spectra",
            "description": (
                "Run Spectra semi-supervised factor analysis to discover gene programs. "
                "Spectra fits a factor model guided by cell-type-specific gene set priors — "
                "it produces both gene-set-guided factors (e.g. a T cell exhaustion program) "
                "and de novo factors that explain residual variation not covered by the priors. "
                "Outputs per-cell factor scores in obsm['SPECTRA_cell_scores'] (visualizable on UMAP), "
                "top marker genes per factor in uns['SPECTRA_markers'], and gene loadings in uns['SPECTRA_factors']. "
                "Requires: log-normalized counts in adata.X, a cell_type_key, and the Spectra-sc package. "
                "Gene set dictionary format: JSON with cell type keys (one entry per cell type, even if empty {}) "
                "plus a 'global' key. If none provided, runs in de novo mode (unsupervised). "
                "Workshop note: num_epochs=100 for demos, 10000 for serious analysis."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "cell_type_key": {
                        "type": "string",
                        "description": "obs column with cell type labels (e.g. 'celltypist_cell_type', 'leiden'). Every unique value must have an entry in the gene set dictionary."
                    },
                    "gene_set_dict_path": {
                        "type": "string",
                        "description": "Path to a JSON file containing the gene set dictionary. Format: {cell_type: {gene_set_name: [gene, ...]}, 'global': {gene_set_name: [gene, ...]}}. Missing cell types are auto-filled with empty entries."
                    },
                    "use_default_gene_sets": {
                        "type": "boolean",
                        "description": "Use Spectra's built-in default gene sets instead of a custom dictionary (default: false)."
                    },
                    "lam": {
                        "type": "number",
                        "description": "Regularization toward input gene sets (default: 0.1). Lower = stronger adherence to provided gene sets. Range: 0.001–0.5."
                    },
                    "num_epochs": {
                        "type": "integer",
                        "description": "Training iterations (default: 1000). Use 100 for a quick test, 10000 for publication-quality results."
                    },
                    "n_top_vals": {
                        "type": "integer",
                        "description": "Top genes per factor stored in SPECTRA_markers (default: 50)."
                    },
                    "use_highly_variable": {
                        "type": "boolean",
                        "description": "Restrict to highly variable genes plus gene set genes (default: true)."
                    },
                    "use_cell_types": {
                        "type": "boolean",
                        "description": "Fit cell-type-specific factors in addition to global factors (default: true)."
                    },
                    "overlap_threshold": {
                        "type": "number",
                        "description": "Minimum overlap coefficient to label a factor with a gene set name (default: 0.2)."
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save the Spectra model and UMAP factor score figures (recommended)."
                    }
                },
                "required": ["cell_type_key"]
            }
        },
        {
            "name": "query_cells",
            "description": (
                "Search the Scimilarity reference database (~24M cells) for cells most similar to a query. "
                "Two modes:\n"
                "- 'cells': query using specific cells (by obs_names list or a boolean obs column). "
                "Uses the mean Scimilarity embedding of the selected cells.\n"
                "- 'centroid': query using the centroid of a cluster or cell type group "
                "(provide group_key + group_value). More robust for heterogeneous populations.\n"
                "Returns the top-k matching reference cells with their metadata: cell type, tissue, disease, study, distance. "
                "Useful for: validating ambiguous annotations, finding analogous cell states in other datasets, "
                "characterising novel populations. "
                "Requires Scimilarity to be installed and run_scimilarity to have been run (or X_scimilarity in obsm). "
                "For centroid mode, raw counts must be available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["cells", "centroid"],
                        "description": "'cells' (default): query by specific cells. 'centroid': query by cluster/celltype centroid."
                    },
                    "cell_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of obs_names to use as query (cells mode). Use this for querying a specific set of cells."
                    },
                    "obs_column": {
                        "type": "string",
                        "description": "obs column where True/1 marks query cells (cells mode). Alternative to cell_ids."
                    },
                    "group_key": {
                        "type": "string",
                        "description": "obs column containing group labels (centroid mode), e.g. 'leiden' or 'celltypist_majority_voting'."
                    },
                    "group_value": {
                        "type": "string",
                        "description": "Which group to use as the centroid query (centroid mode), e.g. '3' or 'Macrophage'."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of nearest reference cells to retrieve (default: 50). Increase to 500+ for broader characterisation."
                    },
                    "raw_layer": {
                        "type": "string",
                        "description": "Layer with raw integer counts (used in centroid mode). Leave unset to auto-detect."
                    },
                    "organism": {
                        "type": "string",
                        "description": "Reference organism for Scimilarity query: 'human' or 'mouse'. Usually inherited from prior run_scimilarity."
                    },
                    "model_path": {
                        "type": "string",
                        "description": "Optional explicit Scimilarity model directory."
                    }
                },
                "required": []
            }
        },
        {
            "name": "score_gene_signature",
            "description": (
                "Score each cell for a gene signature using Scanpy's implementation of the Seurat method "
                "(average expression of the gene list minus the average of a random control set of similar "
                "expression level). Scores are stored in adata.obs under 'score_name'. "
                "Optionally run cell cycle scoring (S/G2M/G1 phases) as a special case. "
                "Works on normalized data; no raw counts required. "
                "Use for: cell cycle regression, pathway activity, viral signature, stress response, etc."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "gene_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene names to score. Genes not found in the dataset are silently dropped; the tool reports how many were matched."
                    },
                    "score_name": {
                        "type": "string",
                        "description": "Column name to store the score in adata.obs (default: 'gene_signature_score'). Use a descriptive name, e.g. 'IFN_response_score'."
                    },
                    "cell_cycle": {
                        "type": "boolean",
                        "description": "If true, run cell cycle scoring instead. Requires s_genes and g2m_genes. Adds 'S_score', 'G2M_score', and 'phase' to adata.obs."
                    },
                    "s_genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "S-phase gene list for cell cycle scoring (only used when cell_cycle=true)."
                    },
                    "g2m_genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "G2M-phase gene list for cell cycle scoring (only used when cell_cycle=true)."
                    },
                    "layer": {
                        "type": "string",
                        "description": "Layer to use for scoring. Defaults to adata.X (normalized counts). Do not use raw counts — the score uses expression levels, not counts."
                    },
                    "n_bins": {
                        "type": "integer",
                        "description": "Number of expression bins for control gene sampling (default: 25). Increase if you have very few genes."
                    },
                    "ctrl_size": {
                        "type": "integer",
                        "description": "Number of control genes sampled per bin (default: 50). Set to len(gene_list) for a tighter control."
                    }
                },
                "required": []
            }
        },
        {
            "name": "run_cluster_qc",
            "description": "Compute a per-cluster QC summary table and classify each cluster by quality using multi-metric assessment (MT%, library size, n_genes, doublet score). Does NOT remove any cells; it nominates proposed-removal and ambiguous clusters for structure QC adjudication. Call this after first clustering to identify low-quality, low-complexity, doublet-enriched, or ambiguous clusters before annotation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster_key": {"type": "string", "description": "obs column to group by (default: leiden)"},
                    "doublet_threshold": {"type": "number", "description": "Mean doublet score above which to flag as doublet-enriched (default: 0.3)"},
                    "mt_threshold": {"type": "number", "description": "Mean MT% above which a cluster is flagged as elevated (default: 25)"},
                    "low_lib_fraction": {"type": "number", "description": "Fraction of global median library size below which lib size is considered low (default: 0.5)"},
                    "low_genes_fraction": {"type": "number", "description": "Fraction of global median n_genes below which gene count is considered low (default: 0.5)"},
                    "save_checkpoint": {"type": "boolean", "description": "Save an h5ad checkpoint before any removal (default: true)"},
                    "checkpoint_path": {"type": "string", "description": "Path for checkpoint file (default: <output_dir>/checkpoint_pre_cleanup.h5ad)"}
                },
                "required": []
            }
        },
        {
            "name": "run_cluster_structure_qc",
            "description": (
                "Adjudicate proposed/ambiguous cluster-level QC calls with covariance-structure evidence. "
                "For flagged clusters, selects top informative HVGs, computes gene-gene Pearson correlation "
                "module metrics, saves clustered correlation heatmaps, and computes technical Moran's I for "
                "MT% and library size on the existing KNN graph. Does NOT remove cells; returns synthesized "
                "cleanup recommendations for evidence-based reporting and cleanup decisions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster_key": {"type": "string", "description": "obs column to group by (default: leiden)"},
                    "clusters_to_analyze": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Cluster IDs to analyze. Defaults to latest run_cluster_qc metric_flagged_clusters + ambiguous clusters.",
                    },
                    "n_genes": {"type": "integer", "description": "Maximum genes for structure analysis and heatmap (default: 150)."},
                    "min_cells": {"type": "integer", "description": "Minimum cells required for correlation structure analysis (default: 15)."},
                    "moran_min_cells": {"type": "integer", "description": "Minimum cells required for technical Moran's I summaries (default: 40)."},
                    "corr_threshold": {"type": "number", "description": "Absolute correlation threshold for high-correlation pair fraction (default: 0.3)."},
                    "figure_dir": {
                        "type": "string",
                        "description": (
                            "Base directory for clustered correlation heatmaps "
                            "(default: <run_dir>/figures/cluster_qc). The tool creates provenance-safe "
                            "cluster_key/pass_NNN subdirectories automatically."
                        ),
                    },
                },
                "required": [],
            },
        },
        {
            "name": "save_data",
            "description": "Save the current in-memory AnnData object without modifying it. Use this as the final save step after analysis and annotation are complete.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Path to save the current in-memory h5ad"}
                },
                "required": ["output_path"]
            }
        },
    ]

    # Meta tools (agent control)
    # Note: ask_user is intentionally absent. The agent follows a turn-based
    # model (like Claude Code / Codex): run all tools to completion, produce a
    # final response with numbered options, then wait for the user's next message.
    # The user's reply comes back through the CLI loop as a normal analyze() call
    # so state, data, and conversation history are always fully maintained.
    meta_tools = [
        {
            "name": "run_code",
            "description": "FLEXIBLE FALLBACK: Execute custom Python code on the AnnData object. This is your most versatile tool - use it for ANY valid request not covered by specialized tools. DO NOT `import os`, `import sys`, `subprocess`, or `shutil` — they are hard-blocked. Use the namespace helpers `ensure_dir(path)` (mkdir + return Path), `Path(output_dir) / 'sub'` (path joins), `write_report(name, content)` (save markdown), and `register_artifact(path, role=..., metadata=...)` (record a file you wrote so its absolute path comes back in `result.artifacts_created` for the next tool to use). Access: adata, sc (scanpy), plt (matplotlib), np, pd, output_dir, Path, ensure_dir(path), write_report(name, content), register_artifact(path). Examples: custom plots (variance explained, gene correlations, histograms), data filtering (remove clusters, subset cells), calculations (cluster sizes, gene stats), or any scanpy/pandas operation. For destructive edits, build and validate a candidate first, then assign adata = candidate only as the final step. ALWAYS prefer this over saying 'I can't do that'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. DO NOT `import os`, `import sys`, `subprocess`, or `shutil` — these are hard-blocked and will fail the call. For every filesystem op use the namespace helpers instead: `ensure_dir(path)` to make a directory and return it as a Path; `Path(output_dir) / 'subdir' / 'file.json'` to join paths; `write_report(name, content)` to save markdown to reports/name.md (auto-registers as an artifact). When you write a file with `Path(...).write_text(...)`, `json.dump(...)`, or `fig.savefig(...)` that the *next* tool call will need to reference, call `register_artifact(path)` so its absolute path comes back in `result.artifacts_created` — then paste that path verbatim into the next tool, no guessing. Namespace also provides: adata, sc, plt, np, pd, output_dir, Path. The working directory is set to output_dir for the duration of this call, so bare relative paths like 'evidence.json' land inside the run folder; absolute paths still work normally for reads elsewhere. Example figure save: fig_dir = ensure_dir(Path(output_dir) / 'figures'); out = fig_dir / 'plot.png'; fig.savefig(out); register_artifact(out, role='figure'). Always use write_report() instead of open() when saving text results — never write .txt files. For destructive edits, do not mutate adata in-place; create candidate = adata[keep_mask].copy(), validate candidate, then assign adata = candidate as the final step. When loading 10x h5 files with sc.read_10x_h5(), always call .var_names_make_unique() on each AnnData before concatenating. Use series.iloc[pos] not series[pos] for positional pandas access."},
                    "description": {"type": "string", "description": "Brief description of what the code does"},
                    "save_to": {"type": "string", "description": "Optional path to save adata after execution"}
                },
                "required": ["code", "description"]
            }
        },
        {
            "name": "run_shell",
            "description": (
                "Run a shell command and return stdout/stderr. Use for system checks, "
                "CLI tools, and anything that isn't Python. "
                "Examples: 'nvidia-smi' (GPU availability and memory), 'free -h' (RAM), "
                "'df -h .' (disk space), 'which cellbender' (tool installed?), "
                "'cellbender remove-background --input raw.h5 --output clean.h5' (run CellBender), "
                "'pip show scib-metrics' (package version), 'ls -lh /path/to/data'. "
                "stdout and stderr are both captured and returned. "
                "Commands that modify or delete files outside the output directory, "
                "write to device files, or escalate privileges are blocked."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run. Executed via bash -c. Use absolute paths for reliability."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60). Use a longer value for slow CLI tools like CellBender."
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory for the command (default: current session output directory)."
                    }
                },
                "required": ["command"]
            }
        },
        {
            "name": "web_search",
            "description": (
                "Search the web for documentation, API references, package guides, troubleshooting, and tutorials. "
                "Use the `site` parameter to target specific documentation sources. "
                "Common bioinformatics doc sites: scanpy.readthedocs.io, anndata.readthedocs.io, "
                "celltypist.readthedocs.io, gseapy.readthedocs.io, harmonypy.readthedocs.io, "
                "scvi-tools.readthedocs.io, muon.readthedocs.io, squidpy.readthedocs.io. "
                "For troubleshooting use scverse.discourse.org or github.com. "
                "Use search_papers instead for peer-reviewed scientific evidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — be specific (e.g., 'scanpy normalize_total target_sum parameter' rather than 'scanpy normalize')"},
                    "site": {"type": "string", "description": "Optional domain to restrict results (e.g., 'scanpy.readthedocs.io')"},
                    "max_results": {"type": "integer", "description": "Maximum results (default: 5)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_papers",
            "description": (
                "Search PubMed for peer-reviewed scientific literature. Use for: cell type markers, "
                "pathway biology, disease mechanisms, method papers, and any claim that needs a citation. "
                "Automatically normalises GSEA/gene set names (strips HALLMARK_, REACTOME_, GO_ prefixes). "
                "Returns PMID, first author, year, journal, abstract, and PubMed URL. "
                "Use web_search for package documentation instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text query or PubMed search string. GSEA set names are normalised automatically (e.g. 'HALLMARK_TNFA_SIGNALING_VIA_NFKB' → 'TNF alpha signaling NF-kB'). Be specific: include cell type, disease, or gene names for better results."},
                    "max_results": {"type": "integer", "description": "Maximum papers (default: 5)"},
                    "recent_years": {"type": "integer", "description": "Restrict to last N years (default: 5). Use 10-15 for foundational method papers."},
                    "reviews_only": {"type": "boolean", "description": "Return review articles only — good for overviews of a topic"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "fetch_url",
            "description": (
                "Fetch the full text of a web page. Use after web_search when snippets are not enough — "
                "e.g. to read a function's full parameter list, a method's README, or a paper abstract. "
                "Works well for readthedocs, GitHub READMEs, and static HTML pages. "
                "JavaScript-heavy sites (Notion, some dashboards) may return little content."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default: 4000; increase to 8000 for long API pages)"}
                },
                "required": ["url"]
            }
        },
        {
            "name": "install_package",
            "description": "Request installation of a Python package. Requires user approval. Use when you need a package that isn't installed (e.g., gseapy, mygene, biomart).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "Package name (pip format)"},
                    "reason": {"type": "string", "description": "Why this package is needed"}
                },
                "required": ["package", "reason"]
            }
        },
        {
            "name": "pause_and_ask",
            "description": (
                "Pause the analysis and ask the user for guidance. "
                "Use ONLY when you genuinely cannot proceed without information only the user can provide — "
                "e.g. ambiguous batch key, surprising results that change the analysis direction, "
                "or a fork where both paths have large and different downstream consequences. "
                "Do NOT use for routine preprocessing steps, algorithm defaults, or reversible choices. "
                "After calling this tool, present the question in your response and end your turn."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question to ask the user. Be concrete — state what you found and what you need to know."
                    },
                    "context": {
                        "type": "string",
                        "description": "Why you cannot infer the answer yourself. Reference the actual data (e.g. 'I see 3 columns that could be the batch key: sample_id, batch, donor_id')."
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of discrete choices if applicable. Omit for open-ended questions."
                    }
                },
                "required": ["question", "context"]
            }
        },
    ]

    # Inspection tools (read-only)
    inspection_tools = [
        {
            "name": "inspect_data",
            "description": "Inspect data state: shape, processing status, available embeddings, what steps are done, likely metadata columns, and tracked clustering results. Use this first to understand the data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to a single h5ad or 10x h5 file (optional - uses in-memory data). Do NOT pass a directory; for multi-sample loading use run_code."},
                    "goal": {"type": "string", "description": "Analysis goal to get recommendations (e.g., 'cluster', 'annotate')"},
                    "context": {"type": "string", "description": "Optional biological context hint from the user or file path (e.g., 'PBMC healthy human cells')"}
                },
                "required": []
            }
        },
        {
            "name": "inspect_session",
            "description": "Inspect the unified agent session state: active dataset summary, artifacts, recent actions, unresolved decisions, and latest verification.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "include_history": {"type": "boolean", "description": "Include recent events and resolved decisions (default: true)"}
                },
                "required": []
            }
        },
        {
            "name": "list_celltypist_models",
            "description": (
                "List CellTypist models from the installed CellTypist catalog with inferred organism, "
                "description, and local cache state. Use this before CellTypist annotation when the "
                "default immune model may not match the dataset organism/tissue."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "organism": {"type": "string", "enum": ["human", "mouse"], "description": "Optional organism filter."},
                    "query": {"type": "string", "description": "Optional case-insensitive text filter over model name and description, e.g. PBMC, skin, gut, brain."},
                    "force_update": {"type": "boolean", "description": "Ask CellTypist to refresh its model catalog before listing (default: false)."},
                    "limit": {"type": "integer", "description": "Maximum number of model records to return (default: 50)."}
                },
                "required": []
            }
        },
        {
            "name": "check_celltypist_model",
            "description": (
                "Check whether a specific CellTypist model is compatible with the dataset organism, "
                "whether it is locally cached, whether download is required, and which alternative "
                "models may fit the organism/query."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "CellTypist model name or explicit path (default: Immune_All_Low.pkl)."},
                    "organism": {"type": "string", "enum": ["human", "mouse"], "description": "Dataset organism for compatibility checking."},
                    "query": {"type": "string", "description": "Optional tissue/context query used to filter recommendations."},
                    "force_update": {"type": "boolean", "description": "Ask CellTypist to refresh its model catalog before checking (default: false)."}
                },
                "required": []
            }
        },
        {
            "name": "list_artifacts",
            "description": "List known artifacts from the current session ledger or a saved run manifest.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "run_path": {"type": "string", "description": "Optional run directory or manifest.json path when inspecting a saved run."},
                    "artifact_kind": {"type": "string", "description": "Optional artifact kind filter (for example figure, report, data, log)."},
                    "limit": {"type": "integer", "description": "Maximum artifacts to return (default: 20)"}
                },
                "required": []
            }
        },
        {
            "name": "get_cluster_sizes",
            "description": "Get cell counts per cluster.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"},
                    "cluster_key": {"type": "string", "description": "Cluster column (default: leiden)"}
                },
                "required": []
            }
        },
        {
            "name": "get_top_markers",
            "description": "Get top marker genes for a cluster (requires DEG analysis first).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad with DEG results (optional - uses in-memory data)"},
                    "cluster": {"type": "string", "description": "Cluster ID"},
                    "n_genes": {"type": "integer", "description": "Number of genes (default: 10)"},
                    "key": {"type": "string", "description": "DEG result key in adata.uns (default: rank_genes_groups)"}
                },
                "required": ["cluster"]
            }
        },
        {
            "name": "summarize_qc_metrics",
            "description": "Get summary statistics of QC metrics (library size, genes, MT%, doublet scores).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"}
                },
                "required": []
            }
        },
        {
            "name": "get_celltypes",
            "description": "Get cell type annotation summary (counts per type).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"},
                    "annotation_key": {"type": "string", "description": "Annotation column (default: auto-detect)"}
                },
                "required": []
            }
        },
        {
            "name": "list_obs_columns",
            "description": "List available columns in obs (cell metadata).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"}
                },
                "required": []
            }
        },
        {
            "name": "review_figure",
            "description": "Attach and review an existing saved figure with the LLM. Use this when the user wants the agent to interpret QC plots, UMAPs, or other already-generated artifacts.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "figure_path": {"type": "string", "description": "Path to an existing figure image file"},
                    "question": {"type": "string", "description": "Optional prompt to guide the review of the figure"},
                    "include_image": {"type": "boolean", "description": "If true, include the image data for model review (default: true)"}
                },
                "required": ["figure_path"]
            }
        },
        {
            "name": "review_artifact",
            "description": "Review an existing artifact from the session or workspace. Supports figures, text reports, JSON outputs, logs, and AnnData files in read-only mode.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "artifact_path": {"type": "string", "description": "Absolute or run-relative path to the artifact."},
                    "artifact_id": {"type": "string", "description": "Artifact id from list_artifacts or inspect_session."},
                    "question": {"type": "string", "description": "Optional prompt to guide the review."},
                    "include_image": {"type": "boolean", "description": "If true, include image data when reviewing image artifacts (default: true)."},
                    "max_chars": {"type": "integer", "description": "Maximum text characters to return for text-like artifacts (default: 4000)."}
                },
                "required": []
            }
        },
        {
            "name": "inspect_run_state",
            "description": "Inspect a saved run manifest or the active run ledger: status, steps, artifacts, decisions, and recent world-state snapshots.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "run_path": {"type": "string", "description": "Run directory or manifest.json path. Optional when an active run exists."},
                    "include_history": {"type": "boolean", "description": "Include recent events and snapshots (default: true)."}
                },
                "required": []
            }
        },
        {
            "name": "inspect_workspace",
            "description": "Read-only workspace inspection for the current project or run directory. Use this sparingly for awareness and recovery.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path to inspect (default: current working directory)."},
                    "max_depth": {"type": "integer", "description": "Maximum directory depth to traverse (default: 2)."},
                    "limit": {"type": "integer", "description": "Maximum entries to return (default: 50)."}
                },
                "required": []
            }
        },
        {
            "name": "read_file",
            "description": (
                "Read and return the content of a file. Supports PDF (text extraction and optional page rendering), "
                "plain text, Markdown, CSV, TSV, and JSON. Use this to read a paper, protocol, metadata table, "
                "marker gene list, or any other reference document the user provides. "
                "For large PDFs, use pages to read specific sections (e.g. methods). "
                "Set render_pages=true to render PDF pages as images — the first page is sent directly to the vision "
                "model so figures and plots embedded in the document are visible. Additional rendered pages are saved "
                "to figures/pdf_pages/ and can be reviewed with review_figure. "
                "Rendered pages are the right approach for scanned PDFs or pages where the content is primarily visual."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file."},
                    "pages": {"type": "string", "description": "For PDFs: page range to extract, e.g. '1-5' or '3' or '2,4,6'. Default: all pages."},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default: 20000). Increase for longer documents."},
                    "render_pages": {"type": "boolean", "description": "PDF only. Render pages as images (108 DPI PNG) in addition to text extraction. The first rendered page is sent to the vision model inline; others are saved to figures/pdf_pages/ for review_figure. Use when the document has figures, plots, or is scanned. Default: false."},
                },
                "required": ["path"]
            }
        },
        {
            "name": "research_findings",
            "description": (
                "Search PubMed for literature about a specific pathway or gene set in the context of a cell type. "
                "Returns recent papers and review articles. Used internally after GSEA to ground pathway "
                "interpretations in published evidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pathway": {"type": "string", "description": "Pathway or gene set name (e.g. 'HALLMARK_TNFA_SIGNALING_VIA_NFKB')."},
                    "cell_type": {"type": "string", "description": "Cell type context for the search (e.g. 'CD8 T cells')."},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Top leading-edge genes to include in the query."},
                    "context": {"type": "string", "description": "Additional biological context (tissue, disease, species)."},
                    "recent_years": {"type": "integer", "description": "Limit search to this many recent years (default: 3)."},
                },
                "required": ["pathway"]
            }
        },
    ]

    tools = action_tools + meta_tools + inspection_tools

    if include_describe_image:
        tools.append({
            "name": "describe_image",
            "description": (
                "Get a structured textual description of a saved figure from the vision "
                "sidecar model. Use this to (re-)inspect a figure or ask a specific "
                "follow-up question about it. This is the only way you can examine a "
                "figure — the main model you are is text-only. The sidecar returns "
                "sections: WHAT_THIS_IS / KEY_OBSERVATIONS / NUMBERS_VISIBLE / "
                "ANOMALIES / ACTIONABLE_FLAGS / OPEN_QUESTIONS."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "figure_path": {
                        "type": "string",
                        "description": "Absolute or run-relative path to the figure file (PNG/JPG).",
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "Optional focused question for the sidecar, e.g. "
                            "'do clusters 4 and 7 separate by batch?' or "
                            "'is there a small island top-right of the UMAP?'"
                        ),
                    },
                },
                "required": ["figure_path"],
            },
        })

    return tools


def get_openai_tools(include_describe_image: bool = False) -> List[Dict[str, Any]]:
    """
    Get OpenAI-format tool definitions.

    OpenAI uses a different schema format than Anthropic.
    """
    anthropic_tools = get_tools(include_describe_image=include_describe_image)
    openai_tools = []

    for tool in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
        })

    return openai_tools


def encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64 for vision API."""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type for image."""
    ext = image_path.lower().split(".")[-1]
    mime_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return mime_types.get(ext, "image/png")


def _dataframe_preview(df, n: int = 5, max_cols: int = 20) -> dict:
    """Return a JSON-serialisable head() preview of a DataFrame for LLM display."""
    import math
    subset = df.iloc[:n, :max_cols]
    truncated_cols = df.shape[1] > max_cols

    def _clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        try:
            # Categorical → string so json.dumps doesn't choke
            return v.item() if hasattr(v, "item") else str(v) if not isinstance(v, (int, float, bool, type(None))) else v
        except Exception:
            return str(v)

    rows = []
    for idx, row in subset.iterrows():
        entry = {"_index": str(idx)}
        entry.update({col: _clean(val) for col, val in row.items()})
        rows.append(entry)

    return {
        "columns": ["_index"] + list(subset.columns),
        "rows": rows,
        "total_rows": df.shape[0],
        "total_cols": df.shape[1],
        "truncated_cols": truncated_cols,
    }


def process_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any],
    adata=None,
    world_state=None,
    run_manager=None,
) -> tuple:
    """
    Process a tool call and return structured JSON result.

    Returns
    -------
    tuple
        (json_result_string, updated_adata)
    """
    import numpy as np

    from ..core import (
        clustering_record_to_dict,
        default_cluster_key_for_method,
        get_clustering_registry,
        infer_cluster_key,
        inspect_data,
        load_data,
        metadata_candidate_to_dict,
        metadata_resolution_to_dict,
        obs_columns_detail as _obs_columns_detail,
        promote_clustering_to_primary,
        recommend_next_steps,
        register_clustering,
        resolve_batch_metadata,
        run_qc_pipeline,
        normalize_data,
        run_pca,
        compute_neighbors,
        compute_umap,
        run_leiden,
        run_phenograph,
        calculate_qc_metrics,
        detect_doublets,
    )
    from ..core.normalization import select_hvg
    from ..core.clustering import run_differential_expression, get_top_markers
    from ..annotation import (
        available_celltypist_models_for_organism,
        celltypist_model_records,
        check_celltypist_model,
        infer_celltypist_model_organism,
        run_celltypist,
        run_scimilarity,
    )
    from ..batch import run_scanorama, run_harmony, run_scvi, run_bbknn

    from .decision_policy import (
        decision_for_batch_strategy,
        decision_for_clustering_selection,
    )
    from .world_state import ArtifactRecord, StateDelta, VerificationResult

    def make_state(adata):
        """Create compact state dict."""
        state = inspect_data(adata)
        return {
            "has_raw_counts": state.has_raw_layer or state.has_raw,
            "raw_in_adata_raw": state.has_raw,
            "raw_in_layer": state.has_raw_layer,
            "raw_layer_name": state.raw_layer_name if state.has_raw_layer else None,
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

    starting_state = make_state(adata) if adata is not None else {}

    def _stage_from_state(state_dict: Dict[str, Any]) -> str:
        if not state_dict:
            return "uninitialized"
        if state_dict.get("has_celltypes"):
            return "annotated"
        if state_dict.get("has_clusters"):
            return "clustered"
        if state_dict.get("has_umap") or state_dict.get("has_neighbors"):
            return "embedded"
        if state_dict.get("is_normalized") or state_dict.get("has_hvg"):
            return "normalized"
        if state_dict.get("has_qc_metrics") or state_dict.get("has_doublets"):
            return "qc"
        if state_dict.get("has_raw_counts"):
            return "loaded"
        return "unknown"

    def _artifact_kind_from_path(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return "figure"
        if suffix in {".h5ad", ".h5", ".loom"}:
            return "data"
        if suffix in {".json"}:
            return "json"
        if suffix in {".md", ".txt", ".csv", ".tsv"}:
            return "report"
        if suffix in {".log"}:
            return "log"
        return "artifact"

    def _artifact_payload(path: str, *, role: str = "artifact", metadata: Dict[str, Any] | None = None):
        if not path:
            return None
        artifact = ArtifactRecord.from_path(
            path,
            kind=_artifact_kind_from_path(path),
            role=role,
            source_tool=tool_name,
            metadata=metadata or {},
        )
        return artifact.to_dict()

    def _build_state_delta(
        current_adata,
        *,
        summary: str,
        dataset_changed: bool,
        notes: List[str] | None = None,
    ) -> Dict[str, Any]:
        after_state = make_state(current_adata) if current_adata is not None else {}
        changed_flags = {}
        all_keys = set(starting_state.keys()) | set(after_state.keys())
        for key in sorted(all_keys):
            before = starting_state.get(key)
            after = after_state.get(key)
            if before != after:
                changed_flags[key] = {"before": before, "after": after}
        return StateDelta(
            tool=tool_name,
            summary=summary,
            dataset_changed=dataset_changed,
            stage_before=_stage_from_state(starting_state),
            stage_after=_stage_from_state(after_state),
            changed_flags=changed_flags,
            notes=notes or [],
        ).to_dict()

    def _build_verification(
        status: str,
        summary: str,
        checks: List[Dict[str, Any]],
        recovery_options: List[str] | None = None,
    ) -> Dict[str, Any]:
        if status == "passed" and any(check.get("status") == "failed" for check in checks):
            status = "warning"
        return VerificationResult(
            status=status,
            summary=summary,
            checks=checks,
            recovery_options=recovery_options or [],
        ).to_dict()

    def _check(name: str, passed: bool, details: str) -> Dict[str, Any]:
        return {"name": name, "status": "passed" if passed else "failed", "details": details}

    def _neighbors_provenance(adata_obj) -> Dict[str, Any]:
        neighbors = adata_obj.uns.get("neighbors", {}) if adata_obj is not None else {}
        params = neighbors.get("params", {}) if isinstance(neighbors, dict) else {}
        connectivities = adata_obj.obsp.get("connectivities") if adata_obj is not None and "connectivities" in adata_obj.obsp else None
        distances = adata_obj.obsp.get("distances") if adata_obj is not None and "distances" in adata_obj.obsp else None
        return {
            "has_neighbors": bool(adata_obj is not None and "neighbors" in adata_obj.uns),
            "params": _sanitize_uns_value(dict(params)) if isinstance(params, dict) else _sanitize_uns_value(params),
            "connectivities_nnz": int(connectivities.nnz) if connectivities is not None and hasattr(connectivities, "nnz") else None,
            "distances_nnz": int(distances.nnz) if distances is not None and hasattr(distances, "nnz") else None,
            "connectivities_key": neighbors.get("connectivities_key") if isinstance(neighbors, dict) else None,
            "distances_key": neighbors.get("distances_key") if isinstance(neighbors, dict) else None,
        }

    def _provenance_same(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
        return before == after

    def _finalize_result(
        result: Dict[str, Any],
        updated_adata,
        *,
        dataset_changed: bool,
        summary: str,
        artifacts_created: List[Dict[str, Any]] | None = None,
        decisions_raised: List[Dict[str, Any]] | None = None,
        verification: Dict[str, Any] | None = None,
        notes: List[str] | None = None,
    ):
        result.setdefault(
            "state_delta",
            _build_state_delta(
                updated_adata,
                summary=summary,
                dataset_changed=dataset_changed,
                notes=notes,
            ),
        )
        result.setdefault("artifacts_created", artifacts_created or [])
        result.setdefault("decisions_raised", decisions_raised or [])
        result.setdefault(
            "verification",
            verification
            or _build_verification(
                "passed",
                f"{tool_name} completed without verification issues.",
                [],
            ),
        )
        return json.dumps(result, indent=2), updated_adata

    def _confirmed_decision_value(key: str):
        if world_state is None:
            return None
        getter = getattr(world_state, "get_confirmed_value", None)
        if getter is None:
            return None
        return getter(key)

    def _clusterings_payload(adata_obj):
        return [
            clustering_record_to_dict(record)
            for record in get_clustering_registry(adata_obj)
        ]

    def _batch_relevance(state=None, *, goal: Any = None, context: str = "") -> bool:
        goal_text = str(goal or "").strip().lower()
        if goal_text in {"batch_correct", "cluster", "annotate", "umap", "deg"}:
            if state is not None and getattr(state, "n_batches", 0) and getattr(state, "n_batches", 0) > 1:
                return True
        if goal_text == "batch_correct":
            return True
        context_text = str(context or "").lower()
        if state is not None and getattr(state, "n_batches", 0) and getattr(state, "n_batches", 0) > 1:
            if re.search(r"\b(analy[sz]e|cluster|annotat|umap|integrat|multi[- ]sample|samples?|donors?|patients?|libraries|batches)\b", context_text):
                return True
        return bool(
            re.search(
                r"\b(batch|integration|integrate|harmony|scanorama|correct(?:ion)?|multi[- ]sample|samples?|donors?|patients?|libraries)\b",
                context_text,
            )
        )

    def _analysis_guidance(state, *, goal: Any = None, context: str = "") -> Dict[str, Any]:
        batch_relevant_now = _batch_relevance(state, goal=goal, context=context)
        batch_strategy = {
            "status": "not_applicable",
            "batch_key": state.batch_key,
            "n_batches": state.n_batches,
        }
        if state.batch_key and state.n_batches > 1:
            if state.batch_correction_applied:
                batch_strategy = {
                    "status": "corrected",
                    "batch_key": state.batch_key,
                    "n_batches": state.n_batches,
                    "method": state.batch_correction_method or "unknown",
                    "next_action": "Use the corrected graph/embedding for UMAP, clustering, and annotation.",
                }
            elif state.has_neighbors or state.has_umap or state.has_clusters:
                batch_strategy = {
                    "status": "needs_review",
                    "batch_key": state.batch_key,
                    "n_batches": state.n_batches,
                    "next_action": "Score or inspect batch mixing; rerun batch correction if sample structure remains.",
                }
            elif state.has_pca:
                batch_strategy = {
                    "status": "needs_decision",
                    "batch_key": state.batch_key,
                    "n_batches": state.n_batches,
                    "next_action": "Run score_integration on X_pca and/or run_batch_correction before neighbors/UMAP/clustering.",
                }
            else:
                batch_strategy = {
                    "status": "pending_pca",
                    "batch_key": state.batch_key,
                    "n_batches": state.n_batches,
                    "next_action": "Carry batch metadata through preprocessing, then decide after PCA.",
                }

        if not state.has_qc_metrics:
            next_priority = "qc_preview"
        elif not state.is_normalized:
            next_priority = "normalize_and_hvg"
        elif batch_relevant_now and state.has_pca and not state.batch_correction_applied and not state.has_neighbors:
            next_priority = "batch_strategy"
        elif batch_relevant_now and state.has_neighbors and not state.batch_correction_applied:
            next_priority = "review_batch_strategy"
        elif not (state.has_pca and state.has_neighbors and state.has_umap):
            next_priority = "run_pca"
        elif not state.has_clusters:
            next_priority = "run_clustering"
        elif (
            world_state is not None
            and isinstance(getattr(world_state, "data_summary", None), dict)
            and (world_state.data_summary.get("cluster_qc", {}) or {}).get("status") == "needed"
        ):
            next_priority = "run_cluster_qc"
        else:
            next_priority = "annotation_or_deg"

        notes = [
            "For a routine first pass, keep the workflow QC-first before moving into normalization, embedding, and clustering.",
        ]
        if batch_relevant_now:
            notes.append(
                "Batch handling is relevant for this request. For multi-sample data, record a batch strategy before neighbors/UMAP/clustering: run correction, or score/inspect mixing and record why correction is unnecessary."
            )
        else:
            notes.append(
                "Do not make batch correction a front-and-center decision right now; batch/sample metadata only matters later for explicit integration workflows or per-batch operations."
            )

        return {
            "next_priority": next_priority,
            "batch_relevant_now": batch_relevant_now,
            "batch_strategy": batch_strategy,
            "notes": notes,
        }

    def _batch_correction_present(adata_obj) -> bool:
        if adata_obj is None:
            return False
        return bool(
            adata_obj.uns.get("bbknn_batch_key") is not None
            or any(key in adata_obj.obsm for key in ("X_pca_harmony", "X_scVI", "X_scanorama"))
        )

    def _available_annotation_keys(adata_obj) -> List[str]:
        ocd = _obs_columns_detail(adata_obj.obs, adata_obj.n_obs).get("columns", {})
        return [
            col for col, info in ocd.items()
            if info.get("note") != "high_cardinality"
            and 2 <= info.get("n_unique", 0) <= 300
            and info.get("dtype") in ("object", "category")
        ]

    def _available_plot_colors(adata_obj) -> List[str]:
        preferred = [
            "leiden",
            "sample_id",
            "batch",
            "sample",
            "pct_counts_mt",
            "total_counts",
            "n_genes_by_counts",
        ]
        available: List[str] = []
        for candidate in preferred + list(adata_obj.obs.columns):
            if candidate in adata_obj.obs.columns and candidate not in available:
                available.append(candidate)
        return available[:20]

    def _smart_unavailable_result(
        *,
        tool: str,
        message: str,
        adata_obj,
        recovery_options: List[str],
        missing_prerequisites: List[str] | None = None,
        extra: Dict[str, Any] | None = None,
    ):
        payload = {
            "status": "warning",
            "tool": tool,
            "message": message,
            "missing_prerequisites": missing_prerequisites or [],
            "recovery_options": recovery_options,
            "available_clusterings": _clusterings_payload(adata_obj) if adata_obj is not None else [],
            "available_annotation_keys": _available_annotation_keys(adata_obj) if adata_obj is not None else [],
            "available_plot_colors": _available_plot_colors(adata_obj) if adata_obj is not None else [],
            "state": make_state(adata_obj) if adata_obj is not None else {},
        }
        if extra:
            payload.update(extra)
        return _finalize_result(
            payload,
            adata_obj,
            dataset_changed=False,
            summary=message,
            verification=_build_verification(
                "warning",
                message,
                [],
                recovery_options=recovery_options,
            ),
        )

    def _error_result(
        *,
        tool: str,
        message: str,
        adata_obj=None,
        recovery_options: List[str] | None = None,
        install_hint: str | None = None,
        extra: Dict[str, Any] | None = None,
    ):
        """Return a standardized error tuple ``(json_str, adata)``.

        Every tool error should go through this helper so the LLM always
        sees the same shape: ``{status, tool, message, recovery_options,
        available_columns}``.  ``install_hint`` is a shortcut that appends
        a "pip install …" option automatically.
        """
        opts = list(recovery_options or [])
        if install_hint:
            opts.append(f"Install the missing package: {install_hint}")
        payload: Dict[str, Any] = {
            "status": "error",
            "tool": tool,
            "message": message,
            "recovery_options": opts,
        }
        if adata_obj is not None:
            payload["available_columns"] = list(adata_obj.obs.columns[:30])
        if extra:
            payload.update(extra)
        return json.dumps(payload, indent=2), adata_obj

    def get_adata(tool_input, existing_adata, update_memory: bool = True, prefer_memory: bool = False):
        """Get adata from memory or load from disk.

        If ``update_memory`` is False, loading from disk is treated as read-only
        and does not replace the active in-memory AnnData tracked by the agent.
        """
        data_path = tool_input.get("data_path")
        if prefer_memory and existing_adata is not None:
            return existing_adata, existing_adata
        # If adata is already in memory and no specific path given, use it
        if existing_adata is not None and (data_path is None or data_path == "memory"):
            return existing_adata, existing_adata
        # Otherwise load from disk
        if data_path and data_path != "memory":
            loaded = load_data(data_path)
            if update_memory:
                return loaded, loaded
            return loaded, existing_adata
        raise ValueError("No data available. Provide data_path or load data first.")

    def fix_output_path(output_path: str, tool_name: str) -> str:
        """Normalize output_path values for h5ad-producing tools.

        Relative paths with no directory component (e.g. 'result.h5ad') are
        resolved inside the run directory when run_manager is available, so
        saved files land alongside figures and reports rather than in cwd.
        """
        import os as os_module
        if output_path is None:
            return None
        if os_module.path.isdir(output_path):
            if tool_name == "save_data":
                return os_module.path.join(output_path, "final_result.h5ad")
            return None
        # Bare filename with no directory component → put it in the run dir
        if (run_manager is not None
                and not os_module.path.isabs(output_path)
                and os_module.path.dirname(output_path) == ""):
            return os_module.path.join(run_manager.run_dir, output_path)
        return output_path

    def _resolve_integer_counts_layer(adata, requested_layer: str = "raw_counts"):
        """
        Find a layer with true integer counts, or raise a clear ValueError.

        Checks (in order):
        1. The explicitly requested layer name
        2. Common raw layer names (raw_counts, raw_data, counts)
        3. adata.raw — but ONLY if it contains integer values

        Returns (layer_name_or_sentinel, X_matrix) where layer_name_or_sentinel
        is either a key in adata.layers or '__raw__' if adata.raw is the source.
        Raises ValueError with a user-facing message if no integer counts found.
        """
        from ..core.inspector import _is_integer_matrix

        # 1. Explicitly requested layer
        if requested_layer and requested_layer in adata.layers:
            if _is_integer_matrix(adata.layers[requested_layer]):
                return requested_layer, adata.layers[requested_layer]
            else:
                raise ValueError(
                    f"Layer '{requested_layer}' exists but does not contain integer counts "
                    f"(found float values — likely already normalized). "
                    f"This tool requires raw UMI/read counts. "
                    f"Available layers: {list(adata.layers.keys())}"
                )

        # 2. Common raw layer names
        for name in ["raw_counts", "raw_data", "counts"]:
            if name in adata.layers and _is_integer_matrix(adata.layers[name]):
                return name, adata.layers[name]

        # 3. adata.raw — only if truly integer
        if adata.raw is not None:
            if _is_integer_matrix(adata.raw.X):
                return "__raw__", adata.raw.X
            else:
                raise ValueError(
                    "adata.raw exists but contains non-integer values (likely log-normalized). "
                    "This tool requires raw UMI/read counts. "
                    "The original integer counts are not present in this object — "
                    "reload from the source file or use a checkpoint saved before normalization."
                )

        # 4. Nothing found
        raise ValueError(
            "No integer count layer found. "
            f"Checked: layer '{requested_layer}', 'raw_counts', 'raw_data', 'counts', and adata.raw. "
            f"Available layers: {list(adata.layers.keys())}. "
            "This tool requires raw UMI/read counts. Save them with normalize_and_hvg "
            "which preserves raw counts in layers['raw_counts'] before normalizing."
        )

    def _state_preservation_warning(tool_input, existing_adata):
        if existing_adata is not None and tool_input.get("data_path") not in (None, "memory"):
            return ["Ignored data_path and continued with the in-memory dataset to preserve prior analysis state."]
        return []

    def _validate_obs_column(adata_obj, column_name: str, warnings: List[str], *, required: bool = False, context: str = "parameter"):
        """Validate an obs column reference and either warn or raise a clean error."""
        if not column_name:
            return None
        if column_name not in adata_obj.obs.columns:
            available = list(adata_obj.obs.columns)
            if required:
                raise ValueError(
                    f"{context} '{column_name}' is not present in adata.obs. "
                    f"Available columns: {available}"
                )
            warnings.append(f"Ignored invalid {context} '{column_name}' because it is not present in adata.obs.")
            return None
        return column_name

    def _same_resolution(left, right) -> bool:
        if left is None or right is None:
            return False
        return abs(float(left) - float(right)) < 1e-9

    def _resolve_clustering_output_key(adata_obj, method: str, resolution: float, requested_key: str | None):
        normalized_method = "phenograph" if str(method).lower() == "phenograph" else "leiden"
        alias = default_cluster_key_for_method(normalized_method)
        if requested_key:
            return requested_key, requested_key == alias

        registry = {record["key"]: record for record in _clusterings_payload(adata_obj)}
        if alias not in adata_obj.obs:
            return alias, True

        alias_record = registry.get(alias)
        if alias_record and _same_resolution(alias_record.get("resolution"), resolution):
            return alias, True
        if alias_record is None and _same_resolution(resolution, 1.0):
            return alias, True
        return infer_cluster_key(normalized_method, resolution), False

    def _apply_clustering(
        adata_obj,
        *,
        method: str,
        resolution: float,
        cluster_key: str,
        make_primary: bool,
        k: int | None = None,
        use_rep: str | None = None,
        random_state: int = 0,
    ):
        normalized_method = "phenograph" if str(method).lower() == "phenograph" else "leiden"
        if normalized_method == "leiden":
            run_leiden(adata_obj, resolution=resolution, random_state=random_state, key_added=cluster_key)
        else:
            phenograph_kwargs = {
                "resolution": resolution,
                "key_added": cluster_key,
                "random_state": random_state,
            }
            if k is not None:
                phenograph_kwargs["k"] = int(k)
            if use_rep is not None:
                phenograph_kwargs["use_rep"] = use_rep
            run_phenograph(adata_obj, **phenograph_kwargs)

        register_clustering(
            adata_obj,
            cluster_key=cluster_key,
            method=normalized_method,
            resolution=resolution,
            created_by="tool",
        )
        primary_alias = default_cluster_key_for_method(normalized_method)
        primary_cluster_key = primary_alias if primary_alias in adata_obj.obs.columns else ""
        primary_alias_created = False
        if make_primary:
            primary_cluster_key = promote_clustering_to_primary(
                adata_obj,
                cluster_key=cluster_key,
                method=normalized_method,
                resolution=resolution,
                created_by="tool",
            )
            primary_alias_created = primary_cluster_key in adata_obj.obs.columns
        primary_alias_available = bool(primary_cluster_key and primary_cluster_key in adata_obj.obs.columns)
        created_obs_columns = [cluster_key]
        if primary_alias_created and primary_cluster_key != cluster_key:
            created_obs_columns.append(primary_cluster_key)

        sizes = adata_obj.obs[cluster_key].value_counts().to_dict()
        return {
            "cluster_key": cluster_key,
            "primary_alias": primary_alias,
            "primary_cluster_key": primary_cluster_key,
            "primary_alias_available": primary_alias_available,
            "primary_alias_created": primary_alias_created,
            "created_obs_columns": created_obs_columns,
            "method": normalized_method,
            "resolution": float(resolution),
            "n_clusters": len(sizes),
            "cluster_sizes": {str(key): int(value) for key, value in sizes.items()},
            "clusterings": _clusterings_payload(adata_obj),
        }

    def _render_figure(
        adata_obj,
        *,
        plot_type: str,
        output_path: str,
        color_by: str | None = None,
        genes=None,
        include_image: bool = True,
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc

        genes = genes or []
        if plot_type == "umap":
            if "X_umap" not in adata_obj.obsm:
                raise ValueError("UMAP embedding not found. Run run_pca, run_neighbors, and run_umap first.")
            if color_by in ("", None):
                color_by = None
            elif color_by not in adata_obj.obs.columns and color_by not in adata_obj.var_names:
                raise ValueError(f"'{color_by}' is not available for {plot_type.upper()} coloring.")
        elif plot_type == "tsne":
            if "X_tsne" not in adata_obj.obsm:
                raise ValueError("t-SNE embedding not found in obsm['X_tsne']. Compute it via sc.tl.tsne or run_code first.")
            if color_by in ("", None):
                color_by = None
            elif color_by not in adata_obj.obs.columns and color_by not in adata_obj.var_names:
                raise ValueError(f"'{color_by}' is not available for {plot_type.upper()} coloring.")

        # For large datasets, rasterized scatter is orders of magnitude faster than
        # vector rendering (the matplotlib default).  vector_friendly=False tells
        # scanpy to rasterize scatter points — identical PNG output, seconds not minutes.
        n_cells = adata_obj.n_obs
        large_dataset = n_cells > 50_000
        if large_dataset:
            sc.settings.set_figure_params(vector_friendly=False)
            dot_size = max(1, min(5, 120_000 // n_cells))
        else:
            dot_size = None

        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_type == "umap":
            kwargs = dict(ax=ax, show=False)
            if dot_size is not None:
                kwargs["size"] = dot_size
            if color_by is None:
                sc.pl.umap(adata_obj, **kwargs)
            else:
                sc.pl.umap(adata_obj, color=color_by, **kwargs)
        elif plot_type == "tsne":
            kwargs = dict(ax=ax, show=False)
            if dot_size is not None:
                kwargs["size"] = dot_size
            if color_by is None:
                sc.pl.tsne(adata_obj, **kwargs)
            else:
                sc.pl.tsne(adata_obj, color=color_by, **kwargs)
        elif plot_type == "violin":
            sc.pl.violin(adata_obj, keys=genes or [color_by], groupby=color_by, ax=ax, show=False)
        elif plot_type == "dotplot" and genes:
            sc.pl.dotplot(adata_obj, var_names=genes, groupby=color_by, show=False)
        elif plot_type == "heatmap" and genes:
            sc.pl.heatmap(adata_obj, var_names=genes, groupby=color_by, show=False)
        else:
            raise ValueError(f"Unsupported plot configuration: plot_type={plot_type}")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Restore default figure params so subsequent plots in the same session
        # are not affected.
        if large_dataset:
            sc.settings.set_figure_params(vector_friendly=True)

        result = {
            "status": "ok",
            "tool": "generate_figure",
            "output_path": output_path,
            "plot_type": plot_type,
            "color_by": color_by,
        }
        if include_image:
            try:
                result["image_base64"] = encode_image_base64(output_path)
                result["image_mime"] = get_image_mime_type(output_path)
            except Exception as enc_err:
                # Don't let an encoding failure blow up the whole tool call —
                # the figure is already written to disk and the path is in
                # the result. Log and continue without the inline blob.
                logger.warning(
                    "Failed to encode figure %s as base64 (%s); returning path only.",
                    output_path, enc_err,
                )
                result["image_encode_error"] = str(enc_err)
        return result

    def _generate_cluster_highlight_grid(adata_obj, color_by, output_path):
        """
        Grid of UMAP panels — one per cluster — each cluster highlighted in colour,
        all other cells shown in light gray.  Returns the saved file path or None.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _mplt
        import numpy as _np

        if "X_umap" not in adata_obj.obsm:
            return None
        if color_by not in adata_obj.obs.columns:
            return None

        coords = adata_obj.obsm["X_umap"]
        labels = adata_obj.obs[color_by].astype(str)

        try:
            unique_clusters = sorted(labels.unique(), key=lambda x: int(x))
        except (ValueError, TypeError):
            unique_clusters = sorted(labels.unique())

        n_clusters = len(unique_clusters)
        if n_clusters < 2 or n_clusters > 50:
            return None

        n_cols = min(5, n_clusters)
        n_rows = int(_np.ceil(n_clusters / n_cols))

        # Palette: tab20 → tab20b → tab20c, cycling every 20
        _palettes = [_mplt.cm.tab20, _mplt.cm.tab20b, _mplt.cm.tab20c]
        def _cluster_color(i):
            return _palettes[(i // 20) % 3]((i % 20) / 20)

        n_cells = adata_obj.n_obs
        s_bg = max(0.4, min(4.0, 80_000 / n_cells))
        s_fg = max(0.8, min(7.0, 120_000 / n_cells))

        panel_w, panel_h = 2.6, 2.6
        fig, axes = _mplt.subplots(
            n_rows, n_cols,
            figsize=(panel_w * n_cols, panel_h * n_rows),
            squeeze=False,
        )

        for idx, cluster in enumerate(unique_clusters):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]

            fg = (labels == cluster).values
            bg = ~fg

            if bg.any():
                ax.scatter(
                    coords[bg, 0], coords[bg, 1],
                    c="#CCCCCC", s=s_bg, alpha=0.25,
                    linewidths=0, rasterized=True,
                )
            ax.scatter(
                coords[fg, 0], coords[fg, 1],
                c=[_cluster_color(idx)], s=s_fg, alpha=0.9,
                linewidths=0, rasterized=True,
            )
            ax.set_title(
                f"Cluster {cluster}  ({int(fg.sum()):,})",
                fontsize=8, pad=3,
            )
            ax.set_axis_off()

        for idx in range(n_clusters, n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        fig.subplots_adjust(hspace=0.25, wspace=0.04)

        base, ext = os.path.splitext(output_path)
        grid_path = f"{base}_grid{ext}"
        fig.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor="white")
        _mplt.close(fig)
        return grid_path

    def _stringify_dataframe_columns(df):
        if df is None:
            return df
        for col in df.columns:
            try:
                dtype_str = str(df[col].dtype)
            except Exception:
                dtype_str = ""
            if dtype_str in {"object", "category"}:
                df[col] = df[col].astype(str)
        return df

    def _sanitize_uns_value(value):
        import numpy as _np
        import pandas as _pd
        try:
            import scipy.sparse as _sp
        except Exception:
            _sp = None

        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (_np.integer, _np.floating, _np.bool_)):
            return value.item()
        if _sp is not None and _sp.issparse(value):
            return value.toarray().tolist()
        if isinstance(value, _np.ndarray):
            if value.dtype.names is not None:
                return {str(name): _sanitize_uns_value(value[name]) for name in value.dtype.names}
            if value.dtype.kind in "biufc":
                return value.tolist()
            if value.dtype.kind in "SU":
                return value.astype(str).tolist()
            return [_sanitize_uns_value(v) for v in value.tolist()]
        if isinstance(value, (_pd.Series, _pd.Index)):
            return [_sanitize_uns_value(v) for v in value.tolist()]
        if isinstance(value, _pd.DataFrame):
            safe_df = value.copy()
            _stringify_dataframe_columns(safe_df)
            return {str(col): [_sanitize_uns_value(v) for v in safe_df[col].tolist()] for col in safe_df.columns}
        if isinstance(value, dict):
            return {str(k): _sanitize_uns_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            sanitized_items = [_sanitize_uns_value(v) for v in value]
            normalized_items = []
            for item in sanitized_items:
                if isinstance(item, (dict, list, tuple, set)):
                    try:
                        normalized_items.append(json.dumps(item, default=str, sort_keys=True))
                    except Exception:
                        normalized_items.append(str(item))
                else:
                    normalized_items.append(item)
            return normalized_items
        return str(value)

    def _make_serializable_copy(current_adata, aggressive_uns: bool = False):
        sanitized = current_adata.copy()
        _stringify_dataframe_columns(sanitized.obs)
        _stringify_dataframe_columns(sanitized.var)
        if sanitized.raw is not None:
            _stringify_dataframe_columns(sanitized.raw.var)
        if aggressive_uns:
            sanitized.uns = {str(k): _sanitize_uns_value(v) for k, v in sanitized.uns.items()}
        return sanitized

    def write_h5ad_safe(current_adata, output_path: str) -> Dict[str, Any]:
        details = {"save_mode": "direct", "warnings": []}
        first_error_msg = None
        second_error_msg = None
        try:
            current_adata.write_h5ad(output_path)
            return details
        except Exception as first_error:
            first_error_msg = str(first_error)
            details["warnings"].append(f"Direct save failed; retrying with obs/var cleanup: {first_error_msg}")

        try:
            sanitized = _make_serializable_copy(current_adata, aggressive_uns=False)
            sanitized.write_h5ad(output_path)
            details["save_mode"] = "clean_obs_var"
            return details
        except Exception as second_error:
            second_error_msg = str(second_error)
            details["warnings"].append(f"Obs/var cleanup save failed; retrying with uns cleanup: {second_error_msg}")

        try:
            sanitized = _make_serializable_copy(current_adata, aggressive_uns=True)
            sanitized.write_h5ad(output_path)
            details["save_mode"] = "clean_obs_var_uns"
            return details
        except Exception as third_error:
            raise RuntimeError(
                "Unable to save AnnData after serialization cleanup. "
                f"Direct error: {first_error_msg}; obs/var cleanup error: {second_error_msg}; uns cleanup error: {third_error}"
            )

    def search_web(query: str, site: str = "", max_results: int = 5) -> Dict[str, Any]:
        """Search web/docs using Tavily (primary), DuckDuckGo (secondary), or Google CSE (last fallback)."""
        import os as os_module
        from urllib.parse import urlparse
        import requests

        scoped_query = f"site:{site} {query}" if site else query
        search_errors = []

        def normalize_domain(value: str) -> str:
            value = (value or "").strip().lower()
            if not value:
                return ""
            if "://" not in value:
                value = f"https://{value}"
            parsed = urlparse(value)
            return parsed.netloc.replace("www.", "")

        def extract_query_tokens(value: str) -> List[str]:
            raw_tokens = [tok.strip(" ,:;()[]{}").lower() for tok in value.split()]
            tokens = [tok for tok in raw_tokens if tok and tok not in {
                "documentation", "docs", "api", "function", "method", "tutorial", "guide",
                "site"
            }]
            return tokens

        query_tokens = extract_query_tokens(query)
        technical_tokens = [
            tok for tok in query_tokens
            if "_" in tok or "." in tok or any(ch.isdigit() for ch in tok) or len(tok) >= 8
        ]
        priority_tokens = technical_tokens or query_tokens[-2:]

        def score_result(item: Dict[str, Any], preferred_domain: str = "") -> tuple:
            domain = normalize_domain(item.get("url", ""))
            title = (item.get("title") or "").lower()
            snippet = (item.get("snippet") or "").lower()
            url = (item.get("url") or "").lower()
            domain_match = 1 if preferred_domain and preferred_domain in domain else 0
            exact_token_match = 0
            partial_token_match = 0
            api_page_bonus = 0

            for token in priority_tokens:
                if token and token in url:
                    exact_token_match += 1
                if token and token in title:
                    partial_token_match += 1

            api_indicators = [
                "/generated/",
                "/api/",
                "api.",
                "reference",
                "class",
                "function",
            ]
            if any(ind in url for ind in api_indicators):
                api_page_bonus = 1

            query_bonus = 0
            for token in query_tokens:
                if token in title:
                    query_bonus += 2
                elif token in snippet:
                    query_bonus += 1
            return (
                exact_token_match,
                domain_match,
                api_page_bonus,
                partial_token_match,
                query_bonus,
                len(snippet),
            )

        def run_ddg(search_query: str) -> List[Dict[str, Any]]:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=max_results))

            snippets = []
            for item in results:
                snippets.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", "") or item.get("url", ""),
                    "snippet": item.get("body", "")[:300],
                })
            return snippets

        def dedupe_and_rank(snippets: List[Dict[str, Any]], preferred_domain: str = "") -> List[Dict[str, Any]]:
            deduped: Dict[str, Dict[str, Any]] = {}
            for item in snippets:
                url = item.get("url", "")
                key = url or f"{item.get('title', '')}|{item.get('snippet', '')}"
                if key not in deduped:
                    deduped[key] = item
            ranked = sorted(
                deduped.values(),
                key=lambda item: score_result(item, preferred_domain=preferred_domain),
                reverse=True,
            )
            return ranked[:max_results]

        preferred_domain = normalize_domain(site)

        def run_tavily(search_query: str, include_domains: List[str]) -> List[Dict[str, Any]]:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": search_query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_domains": include_domains,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            snippets = []
            for item in data.get("results", []):
                snippets.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "")[:300],
                })
            return snippets

        # === TAVILY (Primary - best for AI agents) ===
        tavily_api_key = os_module.environ.get("TAVILY_API_KEY")
        if tavily_api_key:
            try:
                snippets = []
                used_dual_query = False
                if site:
                    snippets.extend(run_tavily(query, [site]))
                    snippets.extend(run_tavily(query, []))
                    used_dual_query = True
                else:
                    snippets.extend(run_tavily(query, []))

                if snippets:
                    snippets = dedupe_and_rank(snippets, preferred_domain=preferred_domain)
                    return {
                        "status": "ok",
                        "backend": "tavily",
                        "query": query,
                        "results": snippets,
                        "used_dual_query": used_dual_query,
                        "site_filter_requested": bool(site),
                    }
                search_errors.append({"backend": "tavily", "type": "no_results"})
            except requests.HTTPError as e:
                response = getattr(e, "response", None)
                search_errors.append({
                    "backend": "tavily",
                    "type": "http_error",
                    "status_code": response.status_code if response is not None else None,
                    "message": response.text[:200] if response is not None and response.text else str(e),
                })
            except Exception as e:
                search_errors.append({"backend": "tavily", "type": type(e).__name__, "message": str(e)})

        # === DUCKDUCKGO (Secondary) ===
        try:
            ddg_queries = [scoped_query]
            if site:
                ddg_queries.append(query)
                ddg_queries.append(f"{query} {site}")

            snippets = []
            retried_without_site = False
            for idx, ddg_query in enumerate(ddg_queries):
                current = run_ddg(ddg_query)
                if idx > 0 and current:
                    retried_without_site = True
                snippets.extend(current)
                if len(snippets) >= max_results * 2:
                    break

            snippets = dedupe_and_rank(snippets, preferred_domain=preferred_domain)

            if snippets:
                return {
                    "status": "ok",
                    "backend": "duckduckgo",
                    "query": scoped_query,
                    "results": snippets,
                    "retried_without_site": retried_without_site,
                    "fallback_used": True,
                    "backends_tried": [e["backend"] for e in search_errors],
                    "errors": search_errors if search_errors else None,
                }
        except ImportError:
            search_errors.append({"backend": "duckduckgo", "type": "not_installed", "message": "duckduckgo-search not installed"})

        # === GOOGLE (Last fallback) ===
        google_api_key = os_module.environ.get("GOOGLE_API_KEY")
        google_cx = os_module.environ.get("GOOGLE_CX")

        if google_api_key and google_cx:
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": google_api_key,
                    "cx": google_cx,
                    "q": scoped_query,
                    "num": max(1, min(max_results, 10)),
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                snippets = []
                for item in data.get("items", []):
                    snippets.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")[:300],
                    })

                if snippets:
                    snippets = dedupe_and_rank(snippets, preferred_domain=preferred_domain)
                    return {
                        "status": "ok",
                        "backend": "google",
                        "query": scoped_query,
                        "results": snippets,
                        "fallback_used": True,
                        "backends_tried": [e["backend"] for e in search_errors],
                        "errors": search_errors if search_errors else None,
                    }
                search_errors.append({"backend": "google", "type": "no_results"})
            except requests.HTTPError as e:
                response = getattr(e, "response", None)
                search_errors.append({
                    "backend": "google",
                    "type": "http_error",
                    "status_code": response.status_code if response is not None else None,
                    "message": response.text[:200] if response is not None and response.text else str(e),
                })
            except Exception as e:
                search_errors.append({"backend": "google", "type": type(e).__name__, "message": str(e)})

        if search_errors:
            return {
                "status": "warning",
                "query": scoped_query,
                "backend": "none",
                "results": [],
                "message": "No search backend returned results for this query.",
                "errors": search_errors,
            }

    def search_pubmed(query: str, max_results: int = 5, recent_years: int = 5, reviews_only: bool = False) -> List[Dict[str, Any]]:
        """Search PubMed and return structured article metadata."""
        import requests
        from datetime import datetime

        current_year = datetime.now().year
        min_date = f"{current_year - recent_years}/01/01"
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        search_term = query
        if reviews_only:
            search_term += " AND review[pt]"

        search_params = {
            "db": "pubmed",
            "term": search_term,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "mindate": min_date,
            "maxdate": f"{current_year}/12/31",
            "datetype": "pdat",
        }

        try:
            resp = requests.get(esearch_url, params=search_params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "rettype": "abstract",
            }
            resp = requests.get(efetch_url, params=fetch_params, timeout=10)
            resp.raise_for_status()
            xml = resp.text

            def strip_tags(s: str) -> str:
                return re.sub(r"<[^>]+>", "", s).strip()

            results = []
            articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL)
            for article in articles:
                title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.DOTALL)
                pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
                if not (title_match and pmid_match):
                    continue

                # Abstract: join all AbstractText sections (handles structured abstracts)
                abstract_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
                abstract = " ".join(strip_tags(p) for p in abstract_parts)[:1000]

                pmid = pmid_match.group(1)
                title = strip_tags(title_match.group(1))

                year_match = re.search(r"<PubDate>.*?<Year>(\d+)</Year>", article, re.DOTALL)
                year = year_match.group(1) if year_match else "N/A"

                journal_match = re.search(r"<ISOAbbreviation>(.*?)</ISOAbbreviation>", article)
                if not journal_match:
                    journal_match = re.search(r"<Title>(.*?)</Title>", article)
                journal = strip_tags(journal_match.group(1)) if journal_match else "N/A"

                # First author
                last_name = re.search(r"<LastName>(.*?)</LastName>", article)
                first_name = re.search(r"<Initials>(.*?)</Initials>", article)
                first_author = ""
                if last_name:
                    first_author = strip_tags(last_name.group(1))
                    if first_name:
                        first_author += f" {strip_tags(first_name.group(1))}"

                # DOI
                doi_match = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', article)
                doi = strip_tags(doi_match.group(1)) if doi_match else None

                results.append({
                    "pmid": pmid,
                    "title": title,
                    "first_author": first_author,
                    "year": year,
                    "journal": journal,
                    "abstract": abstract,
                    "doi": doi,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })

            return results
        except Exception:
            return []

    def fetch_url_text(url: str, max_chars: int = 4000) -> Dict[str, Any]:
        """Fetch a URL and return a compact, structured text summary."""
        import html
        import io
        from urllib.parse import urlparse
        import requests

        headers = {
            "User-Agent": "scagent/0.1 (+single-cell analysis agent)"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        raw_bytes = resp.content
        text = resp.text
        title = ""
        meta_description = ""
        extracted_with = "plain_text"
        warning = None

        def clean_whitespace(value: str) -> str:
            return re.sub(r"\s+", " ", value).strip()

        def extract_html_text(html_text: str) -> Dict[str, Any]:
            nonlocal warning
            try:
                from bs4 import BeautifulSoup  # type: ignore
            except ImportError:
                BeautifulSoup = None

            if BeautifulSoup is not None:
                soup = BeautifulSoup(html_text, "html.parser")

                # Strip boilerplate before extraction
                for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                for tag in soup.find_all(attrs={"role": ["navigation", "banner", "complementary"]}):
                    tag.decompose()
                for tag in soup.find_all(class_=lambda c: c and any(
                    kw in (" ".join(c) if isinstance(c, list) else c)
                    for kw in ("sidebar", "toctree", "nav", "menu", "breadcrumb", "footer", "header")
                )):
                    tag.decompose()

                page_title = clean_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""
                meta = soup.find("meta", attrs={"name": "description"})
                meta_desc = clean_whitespace(meta.get("content", "")) if meta else ""

                # Prefer semantic containers (readthedocs uses .rst-content, sphinx uses .document)
                main = (
                    soup.find("main") or
                    soup.find("article") or
                    soup.find(class_=lambda c: c and any(
                        kw in (" ".join(c) if isinstance(c, list) else c)
                        for kw in ("rst-content", "document", "content", "body-content")
                    )) or
                    soup.body or soup
                )
                parts = []
                for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "pre", "code"]):
                    text_part = clean_whitespace(tag.get_text(" ", strip=True))
                    if text_part and len(text_part) > 3:
                        parts.append(text_part)

                if not parts:
                    parts = [clean_whitespace(main.get_text(" ", strip=True))]

                return {
                    "title": page_title,
                    "meta_description": meta_desc,
                    "text": clean_whitespace(" ".join(parts)),
                    "extracted_with": "beautifulsoup4",
                }

            warning = "beautifulsoup4 not installed; used regex-based HTML extraction."
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
            meta_match = re.search(
                r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
                html_text,
                re.IGNORECASE | re.DOTALL,
            )
            page_title = html.unescape(clean_whitespace(title_match.group(1))) if title_match else ""
            meta_desc = html.unescape(clean_whitespace(meta_match.group(1))) if meta_match else ""
            cleaned = re.sub(r"<script.*?</script>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"<style.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"<noscript.*?</noscript>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
            cleaned = html.unescape(cleaned)
            cleaned = clean_whitespace(cleaned)
            return {
                "title": page_title,
                "meta_description": meta_desc,
                "text": cleaned,
                "extracted_with": "regex_html",
            }

        def extract_pdf_text(data: bytes) -> Dict[str, Any]:
            try:
                from pypdf import PdfReader  # type: ignore
            except ImportError:
                try:
                    from PyPDF2 import PdfReader  # type: ignore
                except ImportError as e:
                    raise ImportError("PDF reader dependency not installed") from e

            reader = PdfReader(io.BytesIO(data))
            pages = []
            for page in reader.pages[:5]:
                page_text = page.extract_text() or ""
                if page_text:
                    pages.append(clean_whitespace(page_text))

            return {
                "title": "",
                "meta_description": "",
                "text": clean_whitespace(" ".join(pages)),
                "extracted_with": "pdf_reader",
            }

        lower_content_type = content_type.lower()
        if "html" in lower_content_type:
            extracted = extract_html_text(text)
            title = extracted["title"]
            meta_description = extracted["meta_description"]
            cleaned = extracted["text"]
            extracted_with = extracted["extracted_with"]
        elif "pdf" in lower_content_type or urlparse(str(resp.url)).path.lower().endswith(".pdf"):
            try:
                extracted = extract_pdf_text(raw_bytes)
                title = extracted["title"]
                meta_description = extracted["meta_description"]
                cleaned = extracted["text"]
                extracted_with = extracted["extracted_with"]
            except ImportError:
                cleaned = ""
                extracted_with = "pdf_unsupported"
                warning = "PDF content fetched but no PDF text extraction library is installed."
        else:
            cleaned = clean_whitespace(text)

        result = {
            "status": "ok",
            "url": url,
            "final_url": str(resp.url),
            "content_type": content_type,
            "domain": urlparse(str(resp.url)).netloc,
            "title": title,
            "meta_description": meta_description,
            "extracted_with": extracted_with,
            "text": cleaned[:max_chars],
            "text_length": len(cleaned),
            "truncated": len(cleaned) > max_chars,
        }
        if warning:
            result["warning"] = warning
        if not cleaned:
            result["status"] = "warning"
            result["message"] = "Fetched the URL, but extracted little or no readable text."
        return result

    try:
        # ===== META TOOLS =====
        if tool_name == "ask_user":
            # This is handled specially by the agent loop - just return the question
            return json.dumps({
                "status": "needs_input",
                "tool": "ask_user",
                "question": tool_input["question"],
                "options": tool_input.get("options", []),
                "option_actions": tool_input.get("option_actions", []),
                "default": tool_input.get("default", ""),
                "decision_key": tool_input.get("decision_key", ""),
            }, indent=2), adata

        elif tool_name == "run_code":
            # Execute custom Python code on adata
            import scanpy as sc
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            code = tool_input["code"]
            description = tool_input.get("description") or "Custom Python code"
            save_to = tool_input.get("save_to")
            save_warning = None

            # Security: basic checks (not foolproof, but helps)
            # Note: "import os" is blocked but pathlib.Path is allowed in namespace
            forbidden = ["import os", "import sys", "subprocess", "eval(",
                        "__import__", "rm -rf", "shutil.rmtree", "requests.",
                        "os.system", "os.popen", "os.exec"]
            # Map each forbidden token to a concrete in-namespace replacement so
            # the agent does not retry with the same banned idiom.
            _filesystem_alternative = (
                "Use the run_code namespace helpers instead — they cover every legitimate filesystem op:\n"
                "  • Make a directory: ensure_dir(Path(output_dir) / 'subdir')\n"
                "  • Join paths:       Path(output_dir) / 'subdir' / 'file.json'\n"
                "  • Save text/report: write_report('name', content)  → reports/name.md\n"
                "  • Save figure:      fig_dir = ensure_dir(Path(output_dir) / 'figures'); fig.savefig(fig_dir / 'plot.png')\n"
                "  • Save JSON:        (Path(output_dir) / 'evidence.json').write_text(json.dumps(...))"
            )
            _forbidden_guidance = {
                "import os": _filesystem_alternative,
                "os.system": _filesystem_alternative,
                "os.popen": _filesystem_alternative,
                "os.exec": _filesystem_alternative,
                "shutil.rmtree": "Removing files/directories from run_code is not allowed. If you need to drop cells, build a keep-mask candidate (candidate = adata[mask].copy()) and assign adata = candidate instead.",
                "import sys": "sys is not needed inside run_code — the agent already manages the Python process. If you need a path constant, use output_dir or Path(__file__).",
                "subprocess": "Shell-outs are not allowed from run_code. Use the run_shell tool for legitimate shell commands.",
                "__import__": "Dynamic imports are blocked. Import normally at the top of your snippet or use what is already in the namespace (sc, np, pd, plt, Path).",
                "eval(": "eval() is blocked. Build the value directly in code rather than evaluating a string.",
                "requests.": "Network calls from run_code are not allowed. Use the fetch_url or search_papers tool for HTTP.",
                "rm -rf": "Shell deletion is blocked. Build candidate AnnData via masking instead of deleting files.",
            }
            for f in forbidden:
                if f in code:
                    guidance = _forbidden_guidance.get(f, _filesystem_alternative)
                    return _error_result(
                        tool="run_code",
                        message=(
                            f"Forbidden operation: {f!r} is blocked inside run_code.\n\n"
                            f"{guidance}"
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "Rewrite the snippet using ensure_dir / Path / write_report from the run_code namespace — do NOT retry with the same import or call.",
                            "For shell commands, use the run_shell tool instead.",
                            "For HTTP, use fetch_url or search_papers.",
                        ],
                    )

            provenance_bypass_patterns = [
                (
                    r"\badata\s*\[[^\]]+\]\s*\.\s*write(?:_h5ad)?\s*\(",
                    "Do not write a primary AnnData subset to disk from run_code. "
                    "This can bypass cleanup provenance and destructive-removal preflight.",
                ),
                (
                    r"\badata\s*=\s*(?:sc|ad|anndata)\.read_h5ad\s*\(",
                    "Do not replace the primary AnnData by reading an h5ad inside run_code. "
                    "Use load_data for intentional dataset switches; use native cleanup tools or "
                    "authorized validated subsetting for cell removal.",
                ),
            ]
            for pattern, message in provenance_bypass_patterns:
                if re.search(pattern, code, flags=re.DOTALL):
                    return _error_result(
                        tool="run_code",
                        message=message,
                        adata_obj=adata,
                        recovery_options=[
                            "If the goal is cleanup, run run_cluster_structure_qc and remove only the synthesized clusters.",
                            "If structure QC synthesized no removal set, keep the reviewed clusters and continue with analysis.",
                            "If the user explicitly requests a different dataset, use load_data rather than sc.read_h5ad in run_code.",
                        ],
                    )

            in_place_destructive_patterns = [
                (r"\badata\._inplace_subset_(obs|var)\s*\(", "Do not use AnnData in-place subsetting in run_code."),
                (r"\badata\.(obs|var)\.drop\s*\([^)]*inplace\s*=\s*True", "Do not use inplace=True drops on adata.obs/adata.var in run_code."),
                (r"\bdel\s+adata\.(obs|var)\s*\[", "Do not delete adata.obs/adata.var columns in run_code."),
                (r"\badata\.(obs|var)\.pop\s*\(", "Do not pop columns from adata.obs/adata.var in run_code."),
                (r"\badata\.(obs|var)\s*=\s*adata\.(obs|var)\.drop\s*\(", "Do not replace adata.obs/adata.var with a dropped-column copy in run_code."),
                (r"\badata\.(obs|var)\s*=\s*adata\.(obs|var)\.loc\s*\[\s*:\s*,", "Do not replace adata.obs/adata.var with a column subset in run_code."),
                (r"\badata\.(obs|var)\s*=\s*adata\.(obs|var)\s*\[\s*\[", "Do not replace adata.obs/adata.var with a column subset in run_code."),
            ]
            for pattern, message in in_place_destructive_patterns:
                if re.search(pattern, code, flags=re.DOTALL):
                    return _error_result(
                        tool="run_code",
                        message=(
                            f"{message} Build a candidate object or dataframe, validate it, "
                            "then assign it back only after all checks pass."
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "For cell filtering, use: candidate = adata[~mask].copy(); validate candidate; adata = candidate.",
                            "For new annotations or metrics, add new obs/var columns instead of deleting existing reference columns.",
                            "If the user explicitly wants columns deleted, ask them to confirm the exact columns first.",
                        ],
                    )

            # Load data if needed
            if adata is None and "data_path" in tool_input:
                adata = get_adata(tool_input, adata)

            preflight_checks = []

            def _expected_cell_count(text: str) -> int | None:
                """Extract an expected removal count from phrases like '(61 cells)'."""
                for match in re.finditer(r"\(([\d,]+)\s+cells?\)", text or "", flags=re.IGNORECASE):
                    try:
                        return int(match.group(1).replace(",", ""))
                    except ValueError:
                        continue
                return None

            def _literal_string_list(expr: str) -> list[str]:
                import ast

                try:
                    value = ast.literal_eval(expr.strip())
                except Exception:
                    return []
                if isinstance(value, (list, tuple, set)):
                    return [str(item) for item in value]
                if isinstance(value, str):
                    return [value]
                return []

            def _cluster_removal_plan() -> Dict[str, Any] | None:
                """Extract a cluster-removal plan from generated run_code."""
                if adata is None or not hasattr(adata, "obs"):
                    return None
                combined_text = f"{description}\n{code}"
                if not re.search(r"\b(remove|drop|filter|exclude|subset)\b", combined_text, flags=re.IGNORECASE):
                    return None
                if "[~" not in code and ".copy()" not in code:
                    return None
                expected = _expected_cell_count(description) or _expected_cell_count(code)

                obs_ref = r"adata\.obs\[['\"]([^'\"]+)['\"]\](?:\.astype\(['\"]str['\"]\))?"

                isin_match = re.search(
                    rf"{obs_ref}\.isin\(([^)]*)\)",
                    code,
                    flags=re.DOTALL,
                )
                if isin_match:
                    cluster_key = isin_match.group(1)
                    isin_arg = isin_match.group(2).strip()
                    labels = _literal_string_list(isin_arg)
                    if not labels and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", isin_arg):
                        assign_match = re.search(
                            rf"{re.escape(isin_arg)}\s*=\s*(\[[^\]]*\])",
                            code,
                            flags=re.DOTALL,
                        )
                        if assign_match:
                            labels = _literal_string_list(assign_match.group(1))
                        if not labels:
                            # Detect the common bypass pattern:
                            #   clusters_to_keep = [...]
                            #   clusters_to_keep.remove('15')
                            #   keep_mask = adata.obs[key].isin(clusters_to_keep)
                            #   adata = adata[keep_mask].copy()
                            # In this case the labels removed from the keep-list are
                            # exactly the clusters being filtered out.
                            removed_from_keep = re.findall(
                                rf"{re.escape(isin_arg)}\.remove\(\s*['\"]([^'\"]+)['\"]\s*\)",
                                code,
                            )
                            if removed_from_keep:
                                labels = [str(label) for label in removed_from_keep]
                else:
                    comparison_match = re.search(
                        rf"{obs_ref}\s*(==|!=)\s*['\"]([^'\"]+)['\"]",
                        code,
                    )
                    if not comparison_match:
                        return None
                    cluster_key = comparison_match.group(1)
                    labels = [str(comparison_match.group(3))]
                if not labels or cluster_key not in adata.obs.columns:
                    return None

                observed = int(adata.obs[cluster_key].astype(str).isin(labels).sum())
                return {
                    "name": "destructive_cluster_removal",
                    "cluster_key": cluster_key,
                    "labels": labels,
                    "expected_cells": expected,
                    "observed_cells": observed,
                }

            def _primary_adata_subset_reassignment_detected() -> bool:
                """Detect primary AnnData replacement via row-subset copy.

                This catches non-literal keep-mask variants that intentionally
                avoid the exact cluster-removal regex. If the code reassigns the
                live `adata` to a subset, but we cannot extract a verifiable
                cluster-removal plan, the safe behavior is to block before
                execution.
                """
                subset_vars = set(
                    re.findall(
                        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*adata\s*\[[^\]]+\]\s*\.copy\s*\(",
                        code,
                        flags=re.DOTALL,
                    )
                )
                if re.search(r"\badata\s*=\s*adata\s*\[[^\]]+\]\s*\.copy\s*\(", code, flags=re.DOTALL):
                    return True
                for var_name in subset_vars:
                    if re.search(rf"\badata\s*=\s*{re.escape(var_name)}\b", code):
                        return True
                return False

            def _validate_cleanup_authorization(plan: Dict[str, Any]) -> Dict[str, Any]:
                authorization = tool_input.get("cleanup_authorization") or {}
                proposal = authorization.get("proposal") or {}
                source = authorization.get("source", "")
                expected = plan.get("expected_cells")
                if expected is None and proposal:
                    expected = proposal.get("cells_in_proposed_removal")
                    plan["expected_cells"] = expected

                check = {
                    **plan,
                    "authorization_source": source or "none",
                    "status": "passed",
                    "failures": [],
                }
                if not authorization:
                    check["status"] = "failed"
                    check["failures"].append("No cleanup authorization was provided.")
                    return check

                if source in {"auto_policy", "auto_structure_qc", "user_confirmation"}:
                    proposal_key = proposal.get("cluster_key")
                    proposal_labels = {str(label) for label in proposal.get("proposed_removal", [])}
                    plan_labels = {str(label) for label in plan.get("labels", [])}
                    if proposal_key and plan.get("cluster_key") != proposal_key:
                        check["failures"].append(
                            f"Code targets cluster key {plan.get('cluster_key')!r}, but authorization is for {proposal_key!r}."
                        )
                    if proposal_labels and plan_labels != proposal_labels:
                        check["failures"].append(
                            "Code targets labels "
                            f"{sorted(plan_labels)}, but authorization is for {sorted(proposal_labels)}."
                        )

                if expected is not None and int(plan.get("observed_cells", -1)) != int(expected):
                    check["failures"].append(
                        f"Expected {expected} cells, observed {plan.get('observed_cells')} in the current AnnData."
                    )
                if check["failures"]:
                    check["status"] = "failed"
                return check

            removal_plan = _cluster_removal_plan()
            if removal_plan is not None:
                removal_check = _validate_cleanup_authorization(removal_plan)
                preflight_checks.append(removal_check)
                if removal_check["status"] != "passed":
                    labels = ", ".join(removal_check.get("labels", []))
                    return _error_result(
                        tool="run_code",
                        message=(
                            "Destructive cluster-removal preflight failed for "
                            f"{removal_check.get('cluster_key')} in [{labels}]: "
                            + "; ".join(removal_check.get("failures", []))
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "Ask the user for confirmation, or use an explicit user-granted auto-cleanup policy.",
                            "Inspect current cluster sizes before retrying.",
                            "Only remove clusters that match the current cluster QC proposal and cell counts.",
                        ],
                    )
            elif _primary_adata_subset_reassignment_detected():
                return _error_result(
                    tool="run_code",
                    message=(
                        "Primary AnnData row-subsetting was detected, but the code does not expose a "
                        "verifiable authorized cluster-removal plan. This can bypass cleanup provenance "
                        "and destructive-removal preflight."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        "If structure QC synthesized a cleanup set, remove exactly those clusters using a literal clusters_to_remove list.",
                        "If structure QC synthesized no removal set, keep the reviewed clusters and continue with analysis.",
                        "If the user wants an explicit override, create a proper cleanup checkpoint/proposal first; do not bypass via keep-mask subsetting.",
                    ],
                )

            # Hard block: direct cluster-to-label annotation bypass.
            # Catches the canonical anti-pattern
            #   adata.obs['cell_type'] = adata.obs['leiden'].map({0: 'T cell', ...})
            # which sidesteps the prepare_annotation → PanglaoDB → finalize_annotation
            # validation chain.
            _cluster_keys_known: set = set()
            if world_state is not None:
                try:
                    for _rec in (getattr(world_state, "clustering_registry", []) or []):
                        if isinstance(_rec, dict) and _rec.get("key"):
                            _cluster_keys_known.add(str(_rec["key"]).lower())
                except Exception:
                    pass
            _cluster_name_tokens = ("leiden", "louvain", "phenograph", "cluster")

            for _match in re.finditer(
                r'\badata\.obs\s*\[\s*[\'"]([^\'"]+)[\'"]\s*\]\s*=\s*'
                r'adata\.obs\s*\[\s*[\'"]([^\'"]+)[\'"]\s*\]\s*\.\s*map\s*\(\s*(\{[^}]{0,4000}\})',
                code,
                flags=re.DOTALL,
            ):
                lhs_col = _match.group(1)
                rhs_col = _match.group(2)
                map_literal = _match.group(3) or ""
                rhs_lower = rhs_col.lower()
                lhs_lower = lhs_col.lower()
                is_cluster_rhs = (
                    rhs_lower in _cluster_keys_known
                    or any(tok in rhs_lower for tok in _cluster_name_tokens)
                )
                # Quoted string values on the RHS dict are a strong tell that
                # this is a label dict rather than a numeric remap.
                has_string_values = bool(
                    re.search(r":\s*[\'\"][A-Za-z][^\'\"]{0,200}[\'\"]", map_literal)
                )
                # An LHS that looks like an annotation column reinforces the signal.
                looks_like_annotation_lhs = any(
                    tok in lhs_lower
                    for tok in ("cell_type", "celltype", "annotation", "label", "ident")
                )
                if not (is_cluster_rhs and (has_string_values or looks_like_annotation_lhs)):
                    continue

                _validated = False
                if adata is not None and hasattr(adata, "uns"):
                    _val = adata.uns.get("annotation_validation") or {}
                    _validated = bool(
                        isinstance(_val, dict)
                        and _val.get("panglaodb_validated")
                        and _val.get("finalized")
                    )
                if _validated:
                    # Already validated this proposal — allow the assignment.
                    continue

                _proposal_exists = bool(
                    adata is not None
                    and hasattr(adata, "uns")
                    and (adata.uns.get("annotation_proposal") or {}).get("cluster_ids")
                )
                return _error_result(
                    tool="run_code",
                    message=(
                        f"Direct cluster-to-label annotation bypass detected: "
                        f"adata.obs[{lhs_col!r}] = adata.obs[{rhs_col!r}].map({{...}}). "
                        "Cell-type labels must go through prepare_annotation → PanglaoDB queries → "
                        "finalize_annotation so they are validated against external markers and cluster DEGs. "
                        "Refusing to write annotation column without validation."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        (
                            "Call prepare_annotation to extract DEGs and the list of PanglaoDB queries needed."
                            if not _proposal_exists else
                            "A proposal already exists; proceed to PanglaoDB queries then finalize_annotation."
                        ),
                        "Query PanglaoDB for each proposed and competing label.",
                        "Stage evidence with stage_annotation_evidence, then call finalize_annotation to write labels.",
                    ],
                )

            # Soft warning: direct obs annotation assignment without the prepare/finalize workflow.
            # Intentionally excludes type-conversion and boolean idioms so we do not
            # warn for benign computations. The targeted hard block above catches
            # the cluster→label .map() bypass even when .map( is excluded here.
            _anno_assign = re.search(
                r'\badata\.obs\s*\[\s*[\'"][^\'"]+[\'"]\s*\]\s*='
                r'(?!.*\.astype|.*\.map\(|.*int|.*float|.*bool|.*isin)',
                code,
                flags=re.DOTALL,
            )
            if _anno_assign and adata is not None:
                _proposal_exists = bool((adata.uns.get('annotation_proposal') or {}) if hasattr(adata, 'uns') else False)
                _validated = bool((adata.uns.get('annotation_validation') or {}).get('panglaodb_validated') if hasattr(adata, 'uns') else False)
                if not _validated:
                    preflight_checks.append({
                        "name": "direct_annotation_assignment",
                        "status": "warning",
                        "details": (
                            "Direct obs column assignment detected. If this is a cell-type annotation, "
                            "use prepare_annotation → PanglaoDB queries → finalize_annotation instead "
                            "of assigning labels directly in run_code. "
                            + ("prepare_annotation has been called; proceed to PanglaoDB queries then finalize_annotation."
                               if _proposal_exists else
                               "prepare_annotation has NOT been called yet — call it first to get DEGs, "
                               "scoring, and the list of PanglaoDB queries needed before finalizing labels.")
                        ),
                    })

            # Helper function for safe directory creation
            from pathlib import Path as _Path
            def ensure_dir(path):
                """Create directory if it doesn't exist and return it as a Path."""
                p = _Path(path)
                p.mkdir(parents=True, exist_ok=True)
                return p

            _run_dir = _Path(tool_input.get("output_dir", ".")).resolve()
            _run_dir.mkdir(parents=True, exist_ok=True)

            # Track files produced by user code so they can be surfaced in the
            # tool result. Anything register_artifact()'d (or written via
            # write_report) appears in result.artifacts_created with an
            # absolute path the next tool call can paste verbatim.
            _written_artifacts: List[Dict[str, Any]] = []

            def _register_artifact_record(path, role=None, metadata=None):
                # Resolve to an absolute path, anchoring relative inputs in the
                # run directory (run_code chdirs into _run_dir, so this matches
                # what the user code actually wrote).
                try:
                    p = _Path(path)
                except Exception:
                    return None
                try:
                    if p.is_absolute():
                        abs_p = p.resolve()
                    elif p.exists():
                        abs_p = p.resolve()
                    else:
                        abs_p = (_run_dir / p).resolve()
                except Exception:
                    abs_p = p
                # Build an ArtifactRecord-compliant dict via the shared helper
                # so world_state.register_artifact accepts it. Falls back to a
                # minimal dict if the closure isn't available for some reason.
                try:
                    rec = _artifact_payload(
                        str(abs_p),
                        role=str(role) if role else "artifact",
                        metadata=dict(metadata) if isinstance(metadata, dict) else {},
                    )
                except Exception:
                    rec = None
                if not isinstance(rec, dict):
                    rec = {
                        "path": str(abs_p),
                        "role": str(role) if role else "artifact",
                        "metadata": dict(metadata) if isinstance(metadata, dict) else {},
                    }
                # Deduplicate by absolute path.
                for existing in _written_artifacts:
                    if existing.get("path") == rec.get("path"):
                        if role and not existing.get("role"):
                            existing["role"] = rec.get("role", "artifact")
                        new_meta = rec.get("metadata") or {}
                        if new_meta:
                            existing_meta = existing.get("metadata") or {}
                            existing_meta.update(new_meta)
                            existing["metadata"] = existing_meta
                        return existing
                _written_artifacts.append(rec)
                return rec

            def write_report(name: str, content: str) -> str:
                """Write a markdown report to reports/name.md and return the path.

                Always use this instead of open() when saving analysis results —
                it ensures reports land in the right directory as readable .md files.
                The returned path is also auto-registered as an artifact so it
                appears in the tool result's artifacts_created list.

                Example:
                    write_report('cluster_summary', '## Cluster Summary\\n\\n...')
                """
                reports_dir = ensure_dir(_run_dir / "reports")
                safe_name = name.replace(" ", "_").rstrip(".md")
                path = reports_dir / f"{safe_name}.md"
                path.write_text(content)
                _register_artifact_record(path, role="report", metadata={"name": safe_name})
                return str(path)

            def register_artifact(path, role: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
                """Record a file written by this run_code call so its absolute
                path comes back in result.artifacts_created.

                Call this after fig.savefig(), json.dump(), or any direct file
                write you want the next tool call to be able to reference by
                path. The returned dict has an absolute ``path`` field plus
                optional ``role`` and ``metadata``.

                Example:
                    p = Path(output_dir) / "evidence.json"
                    p.write_text(json.dumps(evidence))
                    register_artifact(p, role="annotation_evidence_json")
                """
                return _register_artifact_record(path, role=role, metadata=metadata)

            # Execute in controlled namespace
            # Note: Path and ensure_dir are provided - no need to import os
            namespace = {
                "adata": adata,
                "sc": sc,
                "np": np,
                "pd": pd,
                "plt": plt,
                "scanpy": sc,
                "matplotlib": matplotlib,
                "output_dir": str(_run_dir),
                "Path": _Path,
                "ensure_dir": ensure_dir,
                "write_report": write_report,
                "register_artifact": register_artifact,
            }

            # Capture stdout so LLM can see print outputs
            import io
            import sys
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout

            # Capture any figures created
            plt.close('all')

            import warnings as _warnings
            exec_error = None
            _caught = []
            _orig_cwd = os.getcwd()
            try:
                sys.stdout = stdout_capture
                os.chdir(_run_dir)
                with _warnings.catch_warnings(record=True) as _caught:
                    _warnings.simplefilter("always")
                    exec(code, namespace)
            except Exception as _exec_err:
                exec_error = _exec_err
            finally:
                sys.stdout = old_stdout
                try:
                    os.chdir(_orig_cwd)
                except Exception:
                    pass

            captured_output = stdout_capture.getvalue()

            # Append actionable warnings to captured output so the agent sees and acts on them.
            # Suppress purely cosmetic pandas FutureWarnings that require no action.
            _cosmetic = {
                "Series.__getitem__ treating keys as positions",
                "The default of observed=False",
            }
            for w in _caught:
                msg = str(w.message)
                if not any(c in msg for c in _cosmetic):
                    captured_output += f"\nWarning ({w.category.__name__}): {msg}"

            if exec_error is not None:
                err_type = type(exec_error).__name__
                err_msg = str(exec_error)
                reassigned_adata_discarded = namespace.get("adata", adata) is not adata

                # Give the LLM targeted guidance based on the error type
                if err_type in ("TypeError", "AttributeError") or "unexpected keyword" in err_msg or "got an unexpected" in err_msg:
                    hint = "This looks like an API mismatch. Look up the function's documentation with web_search + fetch_url before retrying — do not guess."
                elif err_type in ("KeyError", "IndexError") or "not in" in err_msg:
                    hint = "This looks like a missing column or key. Check adata.obs.columns, adata.var.columns, or adata.obsm with inspect_data or a quick run_code before retrying."
                elif err_type in ("ModuleNotFoundError", "ImportError"):
                    hint = f"Package not installed. Use the install_package tool to request installation of the missing package."
                elif err_type == "SyntaxError":
                    hint = "Syntax error in the generated code — fix it directly."
                elif err_type == "NameError":
                    hint = "NameError — check that all variables used are defined in the namespace (adata, sc, np, pd, plt, Path, ensure_dir, write_report, output_dir)."
                else:
                    hint = "Diagnose the error before retrying: if it's an API issue look up docs; if it's a data issue inspect adata state."

                return json.dumps({
                    "status": "error",
                    "tool": "run_code",
                    "description": description,
                    "error_type": err_type,
                    "message": f"{err_type}: {err_msg}",
                    "output": captured_output[:500] if captured_output else None,
                    "adata_committed": False,
                    "reassigned_adata_discarded": reassigned_adata_discarded,
                    "recovery_options": [hint],
                }, indent=2), adata

            adata = namespace.get("adata", adata)
            custom_output_path = namespace.get("output_path")

            # After a clean execution, ensure var_names and obs_names are unique on
            # the live adata. Delaying this until after the error check prevents a
            # partially reassigned AnnData from becoming the session state when a
            # later print/plot/save line fails.
            var_names_fixed = False
            obs_names_fixed = False
            if adata is not None and not adata.var_names.is_unique:
                adata.var_names_make_unique()
                var_names_fixed = True
            if adata is not None and adata.obs_names.duplicated().any():
                adata.obs_names_make_unique()
                obs_names_fixed = True

            # Check if any figures were created
            figures_saved = []
            if plt.get_fignums():
                # There are open figures - check if code saved them
                pass

            if save_to:
                save_warning = "run_code ignored save_to; use save_data to save AnnData after custom code modifications"

            # Save code to file if output directory exists
            code_file = None
            if "output_dir" in tool_input:
                code_dir = os.path.join(tool_input["output_dir"], "code")
                os.makedirs(code_dir, exist_ok=True)

                # Create filename from description
                safe_desc = "".join(c if c.isalnum() or c in "_ " else "_" for c in description)
                safe_desc = safe_desc.replace(" ", "_")[:50]
                code_file = os.path.join(code_dir, f"{safe_desc}.py")

                with open(code_file, "w") as f:
                    f.write(f'"""\n{description}\n\nAuto-generated by scagent\n"""\n\n')
                    f.write("import scanpy as sc\n")
                    f.write("import numpy as np\n")
                    f.write("import pandas as pd\n")
                    f.write("import matplotlib.pyplot as plt\n\n")
                    f.write("# Load data (adjust path as needed)\n")
                    f.write("# adata = sc.read_h5ad('path/to/data.h5ad')\n\n")
                    f.write("# Generated code:\n")
                    f.write(code)

            result = {
                "status": "ok",
                "tool": "run_code",
                "description": description,
                "adata_committed": True,
            }
            if adata is not None:
                result["shape"] = {"n_cells": adata.n_obs, "n_genes": adata.n_vars}
            if var_names_fixed:
                result.setdefault("auto_fixes", []).append("var_names had duplicates — called .var_names_make_unique()")
            if obs_names_fixed:
                result.setdefault("auto_fixes", []).append("obs_names had duplicates — called .obs_names_make_unique()")
            if save_to:
                result["ignored_save_to"] = save_to
            if save_warning:
                result.setdefault("warnings", []).append(save_warning)
            if preflight_checks:
                result["preflight_checks"] = preflight_checks
            if code_file:
                result["code_file"] = code_file
            if custom_output_path:
                result["output_path"] = str(custom_output_path)
                # Auto-register so it also appears in artifacts_created.
                _register_artifact_record(custom_output_path, role="output_path")
            if _written_artifacts:
                result["artifacts_created"] = list(_written_artifacts)
            if captured_output:
                # Truncate if too long
                result["output"] = captured_output[:2000]
                if len(captured_output) > 2000:
                    result["output_truncated"] = True

            return json.dumps(result, indent=2), adata

        elif tool_name in {"web_search", "web_search_docs"}:
            query = tool_input["query"]
            site = tool_input.get("site", "")
            max_results = tool_input.get("max_results", 5)
            search_result = search_web(query, site=site, max_results=max_results)
            search_result["tool"] = "web_search"
            return json.dumps(search_result, indent=2), adata

        elif tool_name == "search_papers":
            raw_query = tool_input["query"]
            max_results = tool_input.get("max_results", 5)
            recent_years = tool_input.get("recent_years", 5)
            reviews_only = tool_input.get("reviews_only", False)

            # Normalise GSEA gene set names: HALLMARK_TNFA_SIGNALING_VIA_NFKB → TNF alpha signaling NF-kB
            import re as _re
            query = _re.sub(r"^(HALLMARK|REACTOME|KEGG|GO|WP|BIOCARTA|PID|NABA)_", "", raw_query, flags=_re.IGNORECASE)
            query = query.replace("_", " ").strip()

            results = search_pubmed(
                query=query,
                max_results=max_results,
                recent_years=recent_years,
                reviews_only=reviews_only,
            )

            return json.dumps({
                "status": "ok",
                "tool": "search_papers",
                "query": query,
                "original_query": raw_query if raw_query != query else None,
                "reviews_only": reviews_only,
                "years_searched": f"last {recent_years} years",
                "count": len(results),
                "results": results,
            }, indent=2), adata

        elif tool_name == "research_findings":
            pathway = tool_input["pathway"]
            cell_type = tool_input.get("cell_type", "")
            genes = tool_input.get("genes", [])
            context = tool_input.get("context", "")
            recent_years = tool_input.get("recent_years", 3)

            # Build a focused query: pathway + cell type + top genes
            gene_str = " ".join(genes[:5]) if genes else ""
            query_parts = [p for p in [pathway, cell_type, gene_str, context] if p]
            query = " ".join(query_parts)

            # Normalise GSEA-style pathway names before searching
            import re as _re
            query = _re.sub(r"^(HALLMARK|REACTOME|KEGG|GO|WP|BIOCARTA|PID|NABA)_", "", query, flags=_re.IGNORECASE)
            query = query.replace("_", " ").strip()

            recent_papers = search_pubmed(query=query, max_results=5, recent_years=recent_years, reviews_only=False)
            reviews = search_pubmed(query=query, max_results=3, recent_years=recent_years, reviews_only=True)

            return json.dumps({
                "status": "ok",
                "tool": "research_findings",
                "pathway": pathway,
                "cell_type": cell_type,
                "query": query,
                "findings": {
                    "selected_papers": recent_papers,
                    "review_articles": reviews,
                },
            }, indent=2), adata

        elif tool_name == "fetch_url":
            url = tool_input["url"]
            max_chars = tool_input.get("max_chars", 4000)
            fetched = fetch_url_text(url, max_chars=max_chars)
            fetched["tool"] = "fetch_url"
            return json.dumps(fetched, indent=2), adata

        elif tool_name == "run_cellbender":
            import shutil
            import subprocess

            input_raw = str(tool_input.get("input_path", "")).strip()
            if not input_raw:
                return _error_result(
                    tool="run_cellbender",
                    message="input_path is required.",
                    adata_obj=adata,
                    recovery_options=["Provide the raw/unfiltered 10x h5 file path as input_path."],
                )

            input_path = Path(input_raw).expanduser()
            if not input_path.exists():
                return _error_result(
                    tool="run_cellbender",
                    message=f"Input file does not exist: {input_path}",
                    adata_obj=adata,
                    recovery_options=[
                        "Check the path to the raw/unfiltered 10x h5 file.",
                        "Use inspect_workspace or run_shell with ls -lh to locate the file.",
                    ],
                )
            if not (input_path.is_file() or input_path.is_dir()):
                return _error_result(
                    tool="run_cellbender",
                    message=f"Input path is neither a file nor directory: {input_path}",
                    adata_obj=adata,
                    recovery_options=["Pass a raw/unfiltered input supported by CellBender, usually raw_feature_bc_matrix.h5."],
                )

            load_output = bool(tool_input.get("load_output", False))
            force_replace_primary = bool(tool_input.get("force_replace_primary", False))
            if load_output and adata is not None and not force_replace_primary:
                return _error_result(
                    tool="run_cellbender",
                    message=(
                        "A primary dataset is already loaded. Refusing to replace it with CellBender output "
                        "unless force_replace_primary=true."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        "Set load_output=false to run CellBender as preprocessing only.",
                        "Save the current dataset first, then retry with force_replace_primary=true if you really want to switch primary data.",
                    ],
                )

            executable = (
                str(tool_input.get("cellbender_executable") or "").strip()
                or os.environ.get("SCAGENT_CELLBENDER", "").strip()
                or "cellbender"
            )
            resolved_executable = shutil.which(executable)
            if not resolved_executable:
                return _error_result(
                    tool="run_cellbender",
                    message=f"CellBender executable not found: {executable}",
                    adata_obj=adata,
                    recovery_options=[
                        "Activate an environment where cellbender is installed before starting scagent.",
                        "Set SCAGENT_CELLBENDER to the absolute CellBender executable path.",
                        "Pass cellbender_executable with an absolute executable path.",
                        "Install CellBender in the configured scagent environment if it is not installed.",
                    ],
                )

            default_base = Path(run_manager.run_dir) if run_manager is not None else Path(".")
            output_raw = str(tool_input.get("output_path") or "").strip()
            if output_raw:
                output_path = Path(output_raw).expanduser()
                if not output_path.is_absolute() and run_manager is not None:
                    output_path = default_base / output_path
            else:
                output_path = default_base / "cellbender" / f"{input_path.stem}_cellbender.h5"
            if output_path.suffix.lower() != ".h5":
                return _error_result(
                    tool="run_cellbender",
                    message=f"CellBender output_path must end in .h5: {output_path}",
                    adata_obj=adata,
                    recovery_options=["Choose an output_path ending in .h5."],
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                return _error_result(
                    tool="run_cellbender",
                    message=f"Output path already exists: {output_path}",
                    adata_obj=adata,
                    recovery_options=[
                        "Choose a new output_path.",
                        "Move or archive the existing output before rerunning CellBender.",
                    ],
                    extra={"output_path": str(output_path)},
                )

            extra_args = tool_input.get("extra_args") or []
            if isinstance(extra_args, str):
                return _error_result(
                    tool="run_cellbender",
                    message="extra_args must be an array of strings, not a single string.",
                    adata_obj=adata,
                    recovery_options=["Pass extra_args like [\"--low-count-threshold\", \"5\"]."],
                )
            if not isinstance(extra_args, list) or not all(isinstance(arg, str) for arg in extra_args):
                return _error_result(
                    tool="run_cellbender",
                    message="extra_args must be an array of strings.",
                    adata_obj=adata,
                    recovery_options=["Remove non-string values from extra_args."],
                )
            reserved_extra = {"--input", "--output"}
            if any(arg in reserved_extra for arg in extra_args):
                return _error_result(
                    tool="run_cellbender",
                    message="extra_args must not include --input or --output; use input_path and output_path instead.",
                    adata_obj=adata,
                    recovery_options=["Remove --input/--output from extra_args."],
                )

            def _positive_int_arg(name: str) -> int | None:
                value = tool_input.get(name)
                if value is None:
                    return None
                ivalue = int(value)
                if ivalue <= 0:
                    raise ValueError(f"{name} must be positive")
                return ivalue

            try:
                expected_cells = _positive_int_arg("expected_cells")
                total_droplets = _positive_int_arg("total_droplets_included")
                epochs = _positive_int_arg("epochs")
                timeout = int(tool_input.get("timeout", 86400))
                if timeout <= 0:
                    raise ValueError("timeout must be positive")
                fpr = tool_input.get("fpr")
                if fpr is not None:
                    fpr = float(fpr)
                    if fpr < 0 or fpr > 1:
                        raise ValueError("fpr must be between 0 and 1")
            except Exception as e:
                return _error_result(
                    tool="run_cellbender",
                    message=f"Invalid CellBender parameter: {e}",
                    adata_obj=adata,
                    recovery_options=["Use positive integers for count/epoch parameters and 0 <= fpr <= 1."],
                )

            argv = [
                resolved_executable,
                "remove-background",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            if expected_cells is not None:
                argv.extend(["--expected-cells", str(expected_cells)])
            if total_droplets is not None:
                argv.extend(["--total-droplets-included", str(total_droplets)])
            if fpr is not None:
                argv.extend(["--fpr", str(fpr)])
            if epochs is not None:
                argv.extend(["--epochs", str(epochs)])
            if bool(tool_input.get("use_cuda", False)):
                argv.append("--cuda")
            argv.extend(extra_args)

            workdir_raw = str(tool_input.get("workdir") or "").strip()
            workdir = Path(workdir_raw).expanduser() if workdir_raw else output_path.parent
            workdir.mkdir(parents=True, exist_ok=True)

            stdout_log = output_path.with_suffix(output_path.suffix + ".stdout.log")
            stderr_log = output_path.with_suffix(output_path.suffix + ".stderr.log")

            def _tail(path: Path, max_chars: int = 4000) -> str:
                if not path.exists():
                    return ""
                text = path.read_text(errors="replace")
                return text[-max_chars:] if len(text) > max_chars else text

            timed_out = False
            try:
                with stdout_log.open("w") as stdout_fh, stderr_log.open("w") as stderr_fh:
                    proc = subprocess.run(
                        argv,
                        stdout=stdout_fh,
                        stderr=stderr_fh,
                        text=True,
                        timeout=timeout,
                        cwd=str(workdir),
                    )
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = -1
            except Exception as e:
                return _error_result(
                    tool="run_cellbender",
                    message=f"CellBender failed to start: {e}",
                    adata_obj=adata,
                    recovery_options=[
                        "Check that CellBender is executable in this environment.",
                        "Check that workdir is writable.",
                    ],
                    extra={
                        "command": " ".join(argv),
                        "stdout_log": str(stdout_log),
                        "stderr_log": str(stderr_log),
                    },
                )

            output_exists = output_path.exists() and output_path.stat().st_size > 0
            stdout_tail = _tail(stdout_log)
            stderr_tail = _tail(stderr_log)
            artifacts_created = []
            output_artifact = _artifact_payload(
                str(output_path),
                role="cellbender_output",
                metadata={"format": "10x_h5", "loaded_as_primary": False},
            ) if output_exists else None
            if output_artifact:
                artifacts_created.append(output_artifact)
            companion_candidates = [
                (output_path.with_name(f"{output_path.stem}_filtered.h5"), "cellbender_filtered_output"),
                (output_path.with_name(f"{output_path.stem}_report.html"), "cellbender_report"),
                (output_path.with_suffix(".pdf"), "cellbender_report"),
                (output_path.with_suffix(".log"), "cellbender_log"),
                (output_path.with_name(f"{output_path.stem}_metrics.csv"), "cellbender_metrics"),
                (output_path.with_name(f"{output_path.stem}_cell_barcodes.csv"), "cellbender_cell_barcodes"),
                (output_path.with_name(f"{output_path.stem}_posterior.h5"), "cellbender_posterior"),
                (workdir / "ckpt.tar.gz", "cellbender_checkpoint"),
            ]
            seen_artifact_paths = {str(output_path.resolve())} if output_exists else set()
            for path, role in companion_candidates:
                if path.exists() and str(path.resolve()) not in seen_artifact_paths:
                    artifact = _artifact_payload(str(path), role=role)
                    if artifact:
                        artifacts_created.append(artifact)
                        seen_artifact_paths.add(str(path.resolve()))
            for log_path, stream_name in ((stdout_log, "stdout"), (stderr_log, "stderr")):
                if log_path.exists():
                    artifact = _artifact_payload(
                        str(log_path),
                        role="cellbender_log",
                        metadata={"stream": stream_name},
                    )
                    if artifact:
                        artifacts_created.append(artifact)

            updated_adata = adata
            loaded_as_primary = False
            load_error = None
            if returncode == 0 and output_exists and load_output:
                try:
                    updated_adata = load_data(str(output_path))
                    loaded_as_primary = True
                    if output_artifact:
                        output_artifact["metadata"]["loaded_as_primary"] = True
                except Exception as e:
                    load_error = str(e)

            ok = returncode == 0 and output_exists and load_error is None
            status = "ok" if ok else "error"
            if timed_out:
                message = f"CellBender timed out after {timeout}s."
            elif returncode != 0:
                message = f"CellBender exited with return code {returncode}."
            elif not output_exists:
                message = "CellBender exited successfully but the expected output h5 was not created."
            elif load_error:
                message = f"CellBender output was created, but loading it failed: {load_error}"
            else:
                message = f"CellBender completed and wrote {output_path}."

            result = {
                "status": status,
                "tool": "run_cellbender",
                "message": message,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "stdout_log": str(stdout_log),
                "stderr_log": str(stderr_log),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "returncode": returncode,
                "timed_out": timed_out,
                "command": " ".join(argv),
                "command_argv": argv,
                "workdir": str(workdir),
                "expected_cells": expected_cells,
                "total_droplets_included": total_droplets,
                "fpr": fpr,
                "epochs": epochs,
                "use_cuda": bool(tool_input.get("use_cuda", False)),
                "load_output": load_output,
                "loaded_as_primary": loaded_as_primary,
                "state": make_state(updated_adata) if updated_adata is not None else {},
            }

            return _finalize_result(
                result,
                updated_adata,
                dataset_changed=loaded_as_primary,
                summary=message,
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed" if ok else "failed",
                    message,
                    [
                        _check("cellbender_returncode_zero", returncode == 0, f"Return code: {returncode}"),
                        _check("cellbender_output_exists", output_exists, f"Output path: {output_path}"),
                        _check(
                            "cellbender_output_loaded",
                            (not load_output) or loaded_as_primary,
                            "Output loaded as primary dataset." if loaded_as_primary else "Output was not loaded as primary dataset.",
                        ),
                    ],
                    recovery_options=[
                        "Inspect stderr_log for CellBender errors.",
                        "Check GPU availability and retry with use_cuda=false if CUDA failed.",
                        "Adjust expected_cells, total_droplets_included, fpr, or epochs based on the dataset.",
                    ] if not ok else [],
                ),
            )

        elif tool_name == "run_shell":
            import subprocess
            import shlex

            command = tool_input.get("command", "").strip()
            timeout = int(tool_input.get("timeout", 60))
            workdir = tool_input.get("workdir") or tool_input.get("output_dir") or "."

            # Block destructive patterns — focused on things that can't be undone
            _blocked = [
                ("rm -rf /", "recursive deletion of root"),
                ("rm -rf ~", "recursive deletion of home directory"),
                ("rm -rf $HOME", "recursive deletion of home directory"),
                ("> /dev/", "writing to device file"),
                ("dd if=/dev/zero of=/dev/", "disk overwrite"),
                ("mkfs", "filesystem formatting"),
                (":(){ :|:& };:", "fork bomb"),
                ("sudo rm", "privileged deletion"),
                ("chmod -R 777 /", "global permission change"),
            ]
            for pattern, reason in _blocked:
                if pattern in command:
                    return _error_result(
                        tool="run_shell",
                        message=f"Blocked: {reason}.",
                        adata_obj=adata,
                        recovery_options=["Rewrite the command to avoid destructive operations."],
                        extra={"command": command},
                    )

            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=workdir,
                )
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                return _error_result(
                    tool="run_shell",
                    message=f"Command timed out after {timeout}s.",
                    adata_obj=adata,
                    recovery_options=[
                        f"Increase the timeout (currently {timeout}s).",
                        "Break the command into smaller steps.",
                    ],
                    extra={"command": command},
                )
            except Exception as e:
                return _error_result(
                    tool="run_shell",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=["Check command syntax and that required tools are installed."],
                    extra={"command": command},
                )

            return json.dumps({
                "status": "ok" if returncode == 0 else "error",
                "tool": "run_shell",
                "command": command,
                "returncode": returncode,
                "stdout": stdout[:8000] if stdout else "",
                "stderr": stderr[:2000] if stderr else "",
                "truncated": len(stdout) > 8000,
            }, indent=2), adata

        elif tool_name == "install_package":
            # Request package installation - requires user approval
            package = tool_input["package"]
            reason = tool_input["reason"]

            return json.dumps({
                "status": "needs_approval",
                "tool": "install_package",
                "package": package,
                "reason": reason,
                "message": f"Agent wants to install '{package}': {reason}"
            }, indent=2), adata

        # ===== INSPECTION TOOLS =====
        elif tool_name == "inspect_data":
            # update_memory=True only when there is no primary loaded yet.
            # When primary is already in memory, data_path loads a copy for
            # inspection only — the primary is NOT replaced.
            should_update = tool_input.get("data_path") is not None and adata is None
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=should_update)
            state = inspect_data(working_adata)
            batch_resolution = resolve_batch_metadata(working_adata)
            goal = tool_input.get("goal")
            context_hint = tool_input.get("context", "")
            guidance_context = context_hint
            if tool_input.get("data_path"):
                context_hint = " ".join(part for part in [context_hint, str(tool_input.get("data_path"))] if part)
            confirmed_batch_key = _confirmed_decision_value("batch_key")
            guidance = _analysis_guidance(state, goal=goal, context=guidance_context)

            raw_info = {"adata_raw": None, "layers": []}
            if state.has_raw:
                raw_info["adata_raw"] = {
                    "n_vars": state.raw_n_vars,
                    "note": "full gene set before HVG subsetting" if state.raw_n_vars > state.n_genes else "same gene set as X",
                }
            if state.has_raw_layer:
                raw_info["layers"].append(state.raw_layer_name)

            from ..core.inspector import _characterize_features
            feature_info = _characterize_features(working_adata)
            result = {
                "status": "ok",
                "tool": "inspect_data",
                "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
                "data_type": state.data_type,
                "raw": raw_info,
                "state": make_state(working_adata),
                "embeddings": [k for k in working_adata.obsm.keys()],
                "layers": list(working_adata.layers.keys()),
                "genes": {
                    "format": state.gene_id_format,
                    "has_symbols": state.has_gene_symbols,
                    "has_ensembl": state.has_ensembl_ids,
                    "sample": feature_info["sample_gene_names"],
                    "var_columns": list(working_adata.var.columns)[:10],
                    "genome_prefix": feature_info["genome_prefix"],
                    "special_gene_populations": feature_info["special_gene_populations"],
                    "mt_genes_detected": feature_info["mt_genes_detected"],
                    "mt_gene_examples": feature_info["mt_gene_examples"],
                    "ribo_genes_detected": feature_info["ribo_genes_detected"],
                    "ribo_gene_examples": feature_info["ribo_gene_examples"],
                },
                "obs_names": {
                    "format": feature_info["obs_names_format"],
                    "sample": feature_info["obs_names_sample"],
                    "suffixes_detected": feature_info["obs_names_suffixes_detected"],
                },
                "clustering": {
                    "has_clusters": state.has_clusters,
                    "cluster_key": state.cluster_key,
                    "n_clusters": state.n_clusters,
                    "available_clusterings": _clusterings_payload(working_adata),
                },
                "annotations": {
                    "has_celltypes": state.has_celltype_annotations,
                    "cell_type_key": state.cell_type_key,
                    "cell_type_candidates": [
                        metadata_candidate_to_dict(candidate)
                        for candidate in state.cell_type_candidates
                    ],
                    "sources": [
                        source
                        for source, present in (
                            ("celltypist", state.has_celltypist),
                            ("scimilarity", state.has_scimilarity),
                            (
                                "external_or_manual",
                                bool(state.cell_type_candidates)
                                and not (state.has_celltypist or state.has_scimilarity),
                            ),
                        )
                        if present
                    ],
                },
                "batch": {
                    "confirmed_batch_key": confirmed_batch_key,
                    "inferred_batch_key": state.batch_key,
                    "n_batches": state.n_batches,
                    "batch_correction_applied": state.batch_correction_applied,
                    "batch_correction_method": state.batch_correction_method,
                    "status": batch_resolution.status,
                    "recommended_batch_key": batch_resolution.recommended_column,
                    "recommended_role": batch_resolution.recommended_role,
                    "needs_confirmation": batch_resolution.needs_user_confirmation,
                    "reason": batch_resolution.reason,
                    "relevance": "current" if guidance["batch_relevant_now"] else "later_optional",
                    "strategy": guidance["batch_strategy"],
                    "candidates": [
                        metadata_candidate_to_dict(candidate)
                        for candidate in state.metadata_candidates
                    ],
                },
                "metadata_candidates": [
                    metadata_candidate_to_dict(candidate)
                    for candidate in state.metadata_candidates
                ],
                "batch_key": confirmed_batch_key,
                "recommended_batch_key": batch_resolution.recommended_column,
                "available_clusterings": _clusterings_payload(working_adata),
                "analysis_guidance": guidance,
                "obs_names_sample": working_adata.obs_names[:10].tolist(),
                "var_names_sample": working_adata.var_names[:10].tolist(),
                "obs_preview": _dataframe_preview(working_adata.obs),
                "var_preview": _dataframe_preview(working_adata.var),
                "obs_columns_detail": _obs_columns_detail(working_adata.obs, working_adata.n_obs),
            }
            if goal:
                result["recommended_steps"] = recommend_next_steps(state, goal)
            decisions = []
            batch_decision = decision_for_batch_strategy(
                metadata_resolution_to_dict(batch_resolution),
                context="inspect_data",
                source_tool="inspect_data",
                batch_relevant=guidance["batch_relevant_now"],
            )
            if batch_decision is not None:
                decisions.append(batch_decision)

            return _finalize_result(
                result,
                updated_adata,
                dataset_changed=False,
                summary="Inspected the active AnnData state and collaborative metadata candidates.",
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "inspect_data returned a coherent dataset summary.",
                    [
                        _check("shape_available", "shape" in result, "Dataset shape is present."),
                        _check(
                            "batch_section_present",
                            "batch" in result,
                            "Batch metadata summary is present.",
                        ),
                    ],
                ),
            )

        elif tool_name == "inspect_session":
            include_history = bool(tool_input.get("include_history", True))
            if world_state is None:
                session_payload = {
                    "status": "ok",
                    "tool": "inspect_session",
                    "message": "No active AgentWorldState was provided. Falling back to the current AnnData state only.",
                    "world_state": {
                        "analysis_stage": _stage_from_state(starting_state),
                        "data_summary": {"state": starting_state},
                    },
                }
                return _finalize_result(
                    session_payload,
                    adata,
                    dataset_changed=False,
                    summary="Inspected a minimal session fallback because no AgentWorldState was available.",
                )

            snapshot = world_state.snapshot()
            if not include_history:
                snapshot.pop("resolved_decisions", None)
                snapshot.pop("artifacts", None)

            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "inspect_session",
                    "world_state": snapshot,
                },
                adata,
                dataset_changed=False,
                summary="Inspected the unified agent session state.",
                verification=_build_verification(
                    "passed",
                    "Session state was available for inspection.",
                    [
                        _check(
                            "analysis_stage_present",
                            bool(snapshot.get("analysis_stage")),
                            "Session snapshot includes an analysis stage.",
                        ),
                    ],
                ),
            )

        elif tool_name == "list_celltypist_models":
            organism = (tool_input.get("organism") or "").strip().lower()
            if organism not in {"human", "mouse"}:
                organism = None
            query = tool_input.get("query")
            force_update = bool(tool_input.get("force_update", False))
            limit = int(tool_input.get("limit", 50))
            try:
                records = celltypist_model_records(
                    organism=organism,
                    query=query,
                    force_update=force_update,
                )
            except Exception as e:
                return _error_result(
                    tool="list_celltypist_models",
                    message=f"Could not read CellTypist model catalog: {e}",
                    adata_obj=adata,
                    recovery_options=[
                        "Check that CellTypist is installed in the active environment.",
                        "Retry with force_update=false if catalog refresh failed due network access.",
                    ],
                    install_hint="pip install celltypist" if "celltypist" in str(e).lower() else None,
                )
            shown = records[:limit]
            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "list_celltypist_models",
                    "organism": organism,
                    "query": query,
                    "force_update": force_update,
                    "n_models": len(records),
                    "n_returned": len(shown),
                    "models": shown,
                    "model_discovery": {
                        "list_models_code": "celltypist.models.models_description()",
                        "download_model_code": "celltypist.models.download_models(model='<model>.pkl')",
                        "refresh_catalog_code": "celltypist.models.download_models(force_update=True)",
                        "official_models_url": "https://www.celltypist.org/models",
                    },
                },
                adata,
                dataset_changed=False,
                summary=f"Listed {len(shown)} CellTypist model(s).",
            )

        elif tool_name == "check_celltypist_model":
            model = tool_input.get("model") or "Immune_All_Low.pkl"
            organism = (tool_input.get("organism") or "").strip().lower()
            if organism not in {"human", "mouse"}:
                organism = None
            query = tool_input.get("query")
            force_update = bool(tool_input.get("force_update", False))
            try:
                check = check_celltypist_model(
                    model,
                    organism=organism,
                    query=query,
                    force_update=force_update,
                )
            except Exception as e:
                return _error_result(
                    tool="check_celltypist_model",
                    message=f"Could not check CellTypist model '{model}': {e}",
                    adata_obj=adata,
                    recovery_options=[
                        "Use list_celltypist_models to inspect available model names.",
                        "Retry with force_update=false if catalog refresh failed due network access.",
                    ],
                    install_hint="pip install celltypist" if "celltypist" in str(e).lower() else None,
                    extra={"model": model, "requested_organism": organism},
                )
            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "check_celltypist_model",
                    **check,
                    "model_discovery": {
                        "list_models_code": "celltypist.models.models_description()",
                        "download_model_code": "celltypist.models.download_models(model='<model>.pkl')",
                        "refresh_catalog_code": "celltypist.models.download_models(force_update=True)",
                        "official_models_url": "https://www.celltypist.org/models",
                    },
                },
                adata,
                dataset_changed=False,
                summary=f"Checked CellTypist model '{model}'.",
            )

        elif tool_name == "list_artifacts":
            limit = int(tool_input.get("limit", 20))
            artifact_kind = tool_input.get("artifact_kind")
            artifacts: List[Dict[str, Any]] = []

            if world_state is not None:
                artifacts = [artifact.to_dict() for artifact in world_state.artifacts]
            elif run_manager is not None:
                artifacts = list(run_manager.manifest.artifact_registry)
            else:
                run_path = tool_input.get("run_path")
                if run_path:
                    manifest_path = Path(run_path)
                    if manifest_path.is_dir():
                        manifest_path = manifest_path / "manifest.json"
                    if not manifest_path.exists():
                        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
                    with open(manifest_path) as handle:
                        manifest_payload = json.load(handle)
                    artifacts = manifest_payload.get("artifact_registry", [])

            if artifact_kind:
                artifacts = [artifact for artifact in artifacts if artifact.get("kind") == artifact_kind]

            artifacts = artifacts[-limit:]
            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "list_artifacts",
                    "artifacts": artifacts,
                    "n_artifacts": len(artifacts),
                },
                adata,
                dataset_changed=False,
                summary="Listed known session or run artifacts.",
            )

        elif tool_name == "get_cluster_sizes":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            key = tool_input.get("cluster_key", "leiden")
            if key not in working_adata.obs:
                return _error_result(
                    tool="get_cluster_sizes",
                    message=f"No cluster column '{key}'.",
                    adata_obj=working_adata,
                    recovery_options=[
                        "Run clustering first, then request cluster sizes.",
                        "Use list_obs_columns to find the correct cluster key.",
                    ],
                )

            sizes = working_adata.obs[key].value_counts().to_dict()
            return json.dumps({
                "status": "ok",
                "tool": "get_cluster_sizes",
                "cluster_key": key,
                "n_clusters": len(sizes),
                "sizes": {str(k): int(v) for k, v in sizes.items()}
            }, indent=2), updated_adata

        elif tool_name == "get_top_markers":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            cluster = tool_input["cluster"]
            n_genes = tool_input.get("n_genes", 10)
            key = tool_input.get("key", "rank_genes_groups")

            if key not in working_adata.uns:
                return _smart_unavailable_result(
                    tool="get_top_markers",
                    message="Top markers are not available because differential expression has not been run yet.",
                    adata_obj=updated_adata,
                    missing_prerequisites=["deg"],
                    recovery_options=[
                        "Run differential expression on the current clustering first.",
                        "Inspect available clusterings before choosing a DEG grouping.",
                    ],
                    extra={"cluster": cluster, "key": key},
                )

            markers_df = get_top_markers(working_adata, group=cluster, n_genes=n_genes, key=key)
            markers = markers_df[['names', 'scores', 'logfoldchanges', 'pvals_adj']].to_dict('records')

            return json.dumps({
                "status": "ok",
                "tool": "get_top_markers",
                "cluster": cluster,
                "key": key,
                "markers": markers
            }, indent=2), updated_adata

        elif tool_name == "summarize_qc_metrics":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)

            import pandas as _pd

            metrics = {}
            obs = working_adata.obs

            # Map canonical role names → columns. Check exact names first, then
            # fall back to pattern matching on all numeric obs columns.
            _role_patterns = {
                "qc_total_counts":  ("total_counts", ["total_count", "n_counts", "sum_counts"]),
                "qc_n_genes":       ("n_genes_by_counts", ["n_genes", "ngenes", "num_genes", "n_features"]),
                "qc_pct_mt":        ("pct_counts_mt", ["pct_mt", "percent_mt", "mito_pct", "pct_mito", "mt_pct"]),
                "qc_pct_ribo":      ("pct_counts_ribo", ["pct_ribo", "percent_ribo", "ribo_pct"]),
                "doublet_score":    ("doublet_score", ["scrublet_score", "doublet_prob", "dbl_score"]),
            }
            seen = set()
            for role, (canonical, aliases) in _role_patterns.items():
                col = None
                if canonical in obs:
                    col = canonical
                else:
                    for alias in aliases:
                        if alias in obs:
                            col = alias
                            break
                    if col is None:
                        lower_cols = {c.lower(): c for c in obs.columns}
                        for alias in [canonical] + aliases:
                            if alias.lower() in lower_cols:
                                col = lower_cols[alias.lower()]
                                break
                if col and col not in seen:
                    seen.add(col)
                    values = obs[col]
                    if _pd.api.types.is_numeric_dtype(values):
                        metrics[role] = {
                            "column": col,
                            "median": float(values.median()),
                            "mean": float(values.mean()),
                            "min": float(values.min()),
                            "max": float(values.max()),
                        }

            # Doublet label column
            _doublet_label_names = [
                "predicted_doublet", "is_doublet", "doublet_label",
                "doublet", "scrublet_doublet", "dbl_label",
            ]
            doublet_label = next(
                (c for c in _doublet_label_names if c in obs), None
            )
            if doublet_label is None:
                lower_cols = {c.lower(): c for c in obs.columns}
                for name in _doublet_label_names:
                    if name.lower() in lower_cols:
                        doublet_label = lower_cols[name.lower()]
                        break

            doublet_info = {}
            if doublet_label and doublet_label in obs:
                labels = working_adata.obs[doublet_label]
                if labels.dtype == bool:
                    positive = labels
                else:
                    positive = labels.astype(str).str.lower().isin(
                        {"true", "1", "doublet", "multiplet", "positive"}
                    )
                doublet_info = {
                    "column": doublet_label,
                    "n_doublets": int(positive.sum()),
                    "doublet_rate": float(positive.mean())
                }

            return json.dumps({
                "status": "ok",
                "tool": "summarize_qc_metrics",
                "n_cells": working_adata.n_obs,
                "metrics": metrics,
                "doublets": doublet_info
            }, indent=2), updated_adata

        elif tool_name == "get_celltypes":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)

            # Find annotation column
            key = tool_input.get("annotation_key")
            if not key:
                # Use obs_columns_detail from world state if available, otherwise compute
                ocd = (
                    (world_state.data_summary.get("obs_columns_detail") or {})
                    if world_state is not None
                    else {}
                )
                if not ocd:
                    ocd = _obs_columns_detail(working_adata.obs, working_adata.n_obs).get("columns", {})
                # Candidate columns: categorical, n_unique 2-300, not flagged high_cardinality
                candidates = [
                    col for col, info in ocd.items()
                    if info.get("note") != "high_cardinality"
                    and 2 <= info.get("n_unique", 0) <= 300
                    and info.get("dtype") in ("object", "category")
                ]
                if len(candidates) == 1:
                    key = candidates[0]
                elif candidates:
                    return _finalize_result(
                        {
                            "status": "needs_choice",
                            "tool": "get_celltypes",
                            "message": (
                                "Multiple obs columns could be cell type annotations. "
                                "Identify the right one from obs_columns_detail and call "
                                "get_celltypes with annotation_key."
                            ),
                            "obs_columns_detail": {c: ocd[c] for c in candidates},
                            "recovery_options": [
                                "Call get_celltypes again with annotation_key set to the intended column.",
                                "Ask the user which column contains the cell type labels.",
                            ],
                        },
                        updated_adata,
                        dataset_changed=False,
                        summary="Cell type annotation column needs disambiguation.",
                    )

            if not key or key not in working_adata.obs:
                ocd = _obs_columns_detail(working_adata.obs, working_adata.n_obs).get("columns", {})
                return _smart_unavailable_result(
                    tool="get_celltypes",
                    message="No cell type annotations found. Check obs_columns_detail for available columns.",
                    adata_obj=updated_adata,
                    missing_prerequisites=["annotation"],
                    recovery_options=[
                        "Run cell type annotation on the current clustering.",
                        "Call get_celltypes with annotation_key set to the correct obs column.",
                    ],
                    extra={"obs_columns_detail": ocd},
                )

            counts = working_adata.obs[key].value_counts()
            total_cells = working_adata.n_obs

            # Build detailed breakdown with percentages
            breakdown = {}
            for ct, count in counts.items():
                breakdown[str(ct)] = {
                    "count": int(count),
                    "percent": round(100.0 * count / total_cells, 1)
                }

            # Group by major categories if there are many types
            major_types = {}
            if len(counts) > 5:
                for ct in counts.head(10).index:
                    major_types[str(ct)] = {
                        "count": int(counts[ct]),
                        "percent": round(100.0 * counts[ct] / total_cells, 1)
                    }

            return json.dumps({
                "status": "ok",
                "tool": "get_celltypes",
                "annotation_key": key,
                "total_cells": total_cells,
                "n_types": len(counts),
                "top_10_types": major_types if major_types else breakdown,
                "all_types": breakdown
            }, indent=2), updated_adata

        elif tool_name == "list_obs_columns":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            return json.dumps({
                "status": "ok",
                "tool": "list_obs_columns",
                "n_columns": len(working_adata.obs.columns),
                "obs_columns_detail": _obs_columns_detail(working_adata.obs, working_adata.n_obs),
            }, indent=2), updated_adata

        elif tool_name in {"review_figure", "review_artifact"}:
            artifact_path = tool_input.get("artifact_path") or tool_input.get("figure_path")
            artifact_id = tool_input.get("artifact_id")
            include_image = bool(tool_input.get("include_image", True))
            question = tool_input.get("question", "")
            max_chars = int(tool_input.get("max_chars", 4000))

            if not artifact_path and artifact_id and world_state is not None:
                matched = next(
                    (artifact for artifact in world_state.artifacts if artifact.artifact_id == artifact_id),
                    None,
                )
                if matched is not None:
                    artifact_path = matched.path

            if not artifact_path:
                raise ValueError("Provide artifact_path/figure_path or artifact_id to review an artifact.")

            _resolved_artifact = _resolve_run_path(artifact_path, run_manager=run_manager, must_exist=True)
            if _resolved_artifact is None:
                raise FileNotFoundError(f"Artifact not found: {artifact_path}")
            artifact_path = str(_resolved_artifact)

            artifact_kind = _artifact_kind_from_path(artifact_path)
            result = {
                "status": "ok",
                "tool": tool_name,
                "artifact_path": artifact_path,
                "artifact_kind": artifact_kind,
                "question": question,
                "size_bytes": os.path.getsize(artifact_path),
            }
            if tool_name == "review_figure":
                result["figure_path"] = artifact_path

            if artifact_kind == "figure":
                if include_image:
                    try:
                        result["image_base64"] = encode_image_base64(artifact_path)
                        result["image_mime"] = get_image_mime_type(artifact_path)
                    except Exception as enc_err:
                        logger.warning(
                            "Failed to encode artifact %s as base64 (%s); returning path only.",
                            artifact_path, enc_err,
                        )
                        result["image_encode_error"] = str(enc_err)
            elif artifact_kind == "json":
                with open(artifact_path) as handle:
                    payload = json.load(handle)
                excerpt = json.dumps(payload, indent=2)[:max_chars]
                result["content_excerpt"] = excerpt
                result["json_keys"] = list(payload.keys())[:25] if isinstance(payload, dict) else []
            elif artifact_kind in {"report", "log"}:
                with open(artifact_path) as handle:
                    content = handle.read()
                result["content_excerpt"] = content[:max_chars]
                result["truncated"] = len(content) > max_chars
            elif artifact_kind == "data":
                reviewed_adata = load_data(artifact_path)
                reviewed_state = inspect_data(reviewed_adata)
                result["data_summary"] = {
                    "shape": {"n_cells": reviewed_state.n_cells, "n_genes": reviewed_state.n_genes},
                    "data_type": reviewed_state.data_type,
                    "batch_key": reviewed_state.batch_key,
                    "cluster_key": reviewed_state.cluster_key,
                    "n_clusters": reviewed_state.n_clusters,
                }

            verification_checks = [
                _check("artifact_exists", os.path.exists(artifact_path), f"Artifact exists at {artifact_path}."),
            ]
            if artifact_kind == "figure" and include_image:
                verification_checks.append(
                    _check(
                        "image_payload_attached",
                        "image_base64" in result,
                        "Image payload attached for model review.",
                    )
                )
            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary=f"Reviewed existing {artifact_kind} artifact.",
                verification=_build_verification(
                    "passed",
                    f"{artifact_kind.title()} artifact was available for review.",
                    verification_checks,
                ),
            )

        elif tool_name == "inspect_run_state":
            include_history = bool(tool_input.get("include_history", True))
            manifest_payload = None
            if run_manager is not None:
                manifest_payload = run_manager.manifest.to_dict()
            else:
                run_path = tool_input.get("run_path")
                if not run_path:
                    raise ValueError("Provide run_path when no active run manager is available.")
                manifest_path = Path(run_path)
                if manifest_path.is_dir():
                    manifest_path = manifest_path / "manifest.json"
                if not manifest_path.exists():
                    raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
                with open(manifest_path) as handle:
                    manifest_payload = json.load(handle)

            result = {
                "status": "ok",
                "tool": "inspect_run_state",
                "run_id": manifest_payload.get("run_id"),
                "run_status": manifest_payload.get("status"),
                "request": manifest_payload.get("request"),
                "n_steps": len(manifest_payload.get("steps_completed", [])),
                "n_artifacts": len(manifest_payload.get("artifact_registry", [])),
                "n_decisions": len(manifest_payload.get("user_decisions", [])),
                "n_verifications": len(manifest_payload.get("verification_history", [])),
                "latest_world_state": (manifest_payload.get("world_state_snapshots", []) or [{}])[-1],
            }
            if include_history:
                result["recent_events"] = manifest_payload.get("session_events", [])[-10:]
                result["recent_steps"] = manifest_payload.get("steps_completed", [])[-10:]

            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary="Inspected the active or persisted run ledger.",
            )

        elif tool_name == "inspect_workspace":
            workspace_root = Path.cwd().resolve()
            allowed_roots = [workspace_root]
            if run_manager is not None:
                allowed_roots.append(Path(run_manager.run_dir).resolve())

            requested = tool_input.get("path", ".")
            requested_path = Path(requested)
            if not requested_path.is_absolute():
                requested_path = (workspace_root / requested_path).resolve()
            else:
                requested_path = requested_path.resolve()

            if not any(
                requested_path == root or root in requested_path.parents
                for root in allowed_roots
            ):
                raise ValueError(
                    f"inspect_workspace is limited to the project workspace and active run directory. "
                    f"Requested path: {requested_path}"
                )

            max_depth = int(tool_input.get("max_depth", 2))
            limit = int(tool_input.get("limit", 50))
            entries: List[Dict[str, Any]] = []

            def _walk(path_obj: Path, depth: int) -> None:
                if len(entries) >= limit:
                    return
                if path_obj.is_file():
                    entries.append(
                        {
                            "path": str(path_obj),
                            "kind": "file",
                            "size_bytes": path_obj.stat().st_size,
                        }
                    )
                    return
                if not path_obj.is_dir() or depth > max_depth:
                    return

                for child in sorted(path_obj.iterdir(), key=lambda candidate: candidate.name):
                    if len(entries) >= limit:
                        break
                    if child.name == "__pycache__":
                        continue
                    entries.append(
                        {
                            "path": str(child),
                            "kind": "directory" if child.is_dir() else "file",
                            "size_bytes": child.stat().st_size if child.is_file() else None,
                            "depth": depth,
                        }
                    )
                    if child.is_dir():
                        _walk(child, depth + 1)

            _walk(requested_path, 0)
            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "inspect_workspace",
                    "path": str(requested_path),
                    "entries": entries,
                    "n_entries": len(entries),
                },
                adata,
                dataset_changed=False,
                summary="Inspected the workspace in read-only mode.",
            )

        # ===== ACTION TOOLS =====
        elif tool_name == "load_data":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=True)
            if working_adata is None:
                return _error_result(tool="load_data", message="data_path is required", suggestions=["Provide a valid path to an h5ad or 10X h5 file."])
            state = inspect_data(working_adata)
            batch_resolution = resolve_batch_metadata(working_adata)
            from ..core.inspector import _characterize_features
            feature_info = _characterize_features(working_adata)
            ocd = _obs_columns_detail(working_adata.obs, working_adata.n_obs)
            result = {
                "status": "ok",
                "tool": "load_data",
                "loaded": tool_input.get("data_path"),
                "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
                "data_type": state.data_type,
                "state": make_state(working_adata),
                "embeddings": [k for k in working_adata.obsm.keys()],
                "layers": list(working_adata.layers.keys()),
                "genes": {
                    "format": state.gene_id_format,
                    "has_symbols": state.has_gene_symbols,
                    "sample": feature_info["sample_gene_names"],
                    "mt_genes_detected": feature_info["mt_genes_detected"],
                },
                "obs_columns_detail": ocd,
                "batch_metadata": {
                    "status": batch_resolution.status,
                    "recommended_batch_key": batch_resolution.recommended_column,
                    "recommended_role": batch_resolution.recommended_role,
                    "needs_confirmation": batch_resolution.needs_user_confirmation,
                    "reason": batch_resolution.reason,
                    "candidates": [metadata_candidate_to_dict(c) for c in (batch_resolution.candidates or [])],
                },
            }
            return json.dumps(result, indent=2), updated_adata

        elif tool_name == "run_qc":
            import pandas as pd

            warnings = []
            if adata is not None and tool_input.get("data_path") not in (None, "memory"):
                warnings.append(
                    "Ignored data_path and continued with the in-memory dataset to preserve prior analysis state."
                )

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_before, g_before = adata.n_obs, adata.n_vars

            detect_doublets_flag = tool_input.get("detect_doublets_flag", True)
            remove_ribo = tool_input.get("remove_ribo", False)
            remove_mt = tool_input.get("remove_mt", False)
            remove_doublets = bool(tool_input.get("remove_doublets", False))
            filter_mt = bool(tool_input.get("filter_mt", True))
            min_genes = tool_input.get("min_genes")
            min_cells = tool_input.get("min_cells", 3)
            requested_mt_threshold = tool_input.get("mt_threshold")
            # flag_only is the new primary mode; preview_only is a legacy alias
            flag_only = bool(tool_input.get("flag_only", True))
            preview_only = bool(tool_input.get("preview_only", False)) or flag_only
            confirm_filtering = bool(tool_input.get("confirm_filtering", False))
            scrublet_expected_doublet_rate = float(tool_input.get("scrublet_expected_doublet_rate") or 0.06)
            scrublet_sim_doublet_ratio = float(tool_input.get("scrublet_sim_doublet_ratio") or 2.0)
            scrublet_n_prin_comps = int(tool_input.get("scrublet_n_prin_comps") or 30)
            scrublet_min_counts = int(tool_input.get("scrublet_min_counts") or 2)
            scrublet_min_cells = int(tool_input.get("scrublet_min_cells") or 3)
            scrublet_min_gene_variability_pctl = float(tool_input.get("scrublet_min_gene_variability_pctl") or 85.0)
            scrublet_random_state = int(tool_input.get("scrublet_random_state") or 0)
            force_doublet_recompute = bool(tool_input.get("force_doublet_recompute", False))
            requested_scrublet_params = {
                "expected_doublet_rate": scrublet_expected_doublet_rate,
                "sim_doublet_ratio": scrublet_sim_doublet_ratio,
                "n_prin_comps": scrublet_n_prin_comps,
                "min_counts": scrublet_min_counts,
                "min_cells": scrublet_min_cells,
                "min_gene_variability_pctl": scrublet_min_gene_variability_pctl,
                "random_state": scrublet_random_state,
                "force_recompute": force_doublet_recompute,
            }
            requested_batch_key = tool_input.get("batch_key") or _confirmed_decision_value("batch_key")
            if requested_batch_key and "batch_key" not in tool_input and _confirmed_decision_value("batch_key"):
                warnings.append(
                    f"Using previously confirmed batch_key '{requested_batch_key}' from session state."
                )

            if detect_doublets_flag:
                batch_resolution = resolve_batch_metadata(
                    adata,
                    requested_column=requested_batch_key,
                )
            else:
                batch_resolution = None

            batch_key = batch_resolution.applied_column if batch_resolution else None
            if batch_resolution:
                if batch_resolution.status == "auto_selected" and batch_key:
                    warnings.append(batch_resolution.reason)
                elif batch_resolution.status == "invalid_requested":
                    warnings.append(batch_resolution.reason)
                elif batch_resolution.status == "needs_confirmation":
                    warnings.append(
                        f"{batch_resolution.reason} "
                        f"{'Previewing' if preview_only else 'Running'} doublets without per-batch stratification for now."
                    )
                elif batch_resolution.status == "no_candidate":
                    warnings.append(batch_resolution.reason)

            # Compute QC metrics directly on adata — these are non-destructive obs/var
            # annotations (pct_counts_mt, doublet_score, etc.). Writing to adata here
            # (not a throwaway copy) means apply mode can reuse them without recomputing.
            # Actual cell/gene filtering only happens later if not preview_only.
            _metrics_precomputed = 'pct_counts_mt' in adata.obs.columns
            # Never recompute doublets during the filtering step — scores from preview are authoritative.
            _effective_force_recompute = force_doublet_recompute and not confirm_filtering
            _doublets_precomputed = (
                detect_doublets_flag
                and 'predicted_doublet' in adata.obs.columns
                and not _effective_force_recompute
            )
            doublet_predictions_source = (
                "precomputed"
                if _doublets_precomputed
                else ("computed" if detect_doublets_flag else "not_run")
            )
            stored_scrublet_params = (
                adata.uns.get("scrublet_params", {})
                if _doublets_precomputed and isinstance(adata.uns.get("scrublet_params", {}), dict)
                else {}
            )
            scrublet_params_for_report = requested_scrublet_params.copy()
            if stored_scrublet_params:
                scrublet_params_for_report = {
                    "expected_doublet_rate": float(stored_scrublet_params.get("expected_doublet_rate", scrublet_expected_doublet_rate)),
                    "sim_doublet_ratio": float(stored_scrublet_params.get("sim_doublet_ratio", scrublet_sim_doublet_ratio)),
                    "n_prin_comps": int(stored_scrublet_params.get("n_prin_comps", scrublet_n_prin_comps)),
                    "n_prin_comps_used": int(stored_scrublet_params.get("n_prin_comps_used", stored_scrublet_params.get("n_prin_comps", scrublet_n_prin_comps))),
                    "min_counts": int(stored_scrublet_params.get("min_counts", scrublet_min_counts)),
                    "min_cells": int(stored_scrublet_params.get("min_cells", scrublet_min_cells)),
                    "min_gene_variability_pctl": float(stored_scrublet_params.get("min_gene_variability_pctl", scrublet_min_gene_variability_pctl)),
                    "random_state": int(stored_scrublet_params.get("random_state", scrublet_random_state)),
                    "force_recompute": force_doublet_recompute,
                    "batch_key": stored_scrublet_params.get("batch_key", batch_key),
                }
                changed_requested_params = [
                    key for key in (
                        "scrublet_expected_doublet_rate",
                        "scrublet_sim_doublet_ratio",
                        "scrublet_n_prin_comps",
                        "scrublet_min_counts",
                        "scrublet_min_cells",
                        "scrublet_min_gene_variability_pctl",
                        "scrublet_random_state",
                    )
                    if key in tool_input
                ]
                if changed_requested_params:
                    warnings.append(
                        "Existing Scrublet predictions were reused. Requested Scrublet parameters "
                        "were not recomputed; pass force_doublet_recompute=true to regenerate doublet calls."
                    )

            try:
                if not _metrics_precomputed:
                    calculate_qc_metrics(adata, inplace=True)
                if detect_doublets_flag and not _doublets_precomputed:
                    if force_doublet_recompute:
                        for col in ("doublet_score", "predicted_doublet"):
                            if col in adata.obs.columns:
                                del adata.obs[col]
                    detect_doublets(
                        adata,
                        batch_key=batch_key,
                        expected_doublet_rate=scrublet_expected_doublet_rate,
                        sim_doublet_ratio=scrublet_sim_doublet_ratio,
                        n_prin_comps=scrublet_n_prin_comps,
                        scrublet_min_counts=scrublet_min_counts,
                        scrublet_min_cells=scrublet_min_cells,
                        scrublet_min_gene_variability_pctl=scrublet_min_gene_variability_pctl,
                        random_state=scrublet_random_state,
                        inplace=True,
                    )
                    doublet_predictions_source = "computed"
                    stored_scrublet_params = adata.uns.get("scrublet_params", {}) if isinstance(adata.uns.get("scrublet_params", {}), dict) else {}
                    if stored_scrublet_params:
                        scrublet_params_for_report = {
                            "expected_doublet_rate": float(stored_scrublet_params.get("expected_doublet_rate", scrublet_expected_doublet_rate)),
                            "sim_doublet_ratio": float(stored_scrublet_params.get("sim_doublet_ratio", scrublet_sim_doublet_ratio)),
                            "n_prin_comps": int(stored_scrublet_params.get("n_prin_comps", scrublet_n_prin_comps)),
                            "n_prin_comps_used": int(stored_scrublet_params.get("n_prin_comps_used", stored_scrublet_params.get("n_prin_comps", scrublet_n_prin_comps))),
                            "min_counts": int(stored_scrublet_params.get("min_counts", scrublet_min_counts)),
                            "min_cells": int(stored_scrublet_params.get("min_cells", scrublet_min_cells)),
                            "min_gene_variability_pctl": float(stored_scrublet_params.get("min_gene_variability_pctl", scrublet_min_gene_variability_pctl)),
                            "random_state": int(stored_scrublet_params.get("random_state", scrublet_random_state)),
                            "force_recompute": force_doublet_recompute,
                            "batch_key": stored_scrublet_params.get("batch_key", batch_key),
                        }
            except ValueError as e:
                if detect_doublets_flag and "skimage is not installed" in str(e):
                    warnings.append("Scrublet auto-threshold requires skimage; reran without doublet detection.")
                    detect_doublets_flag = False
                    doublet_predictions_source = "not_run"
                    scrublet_params_for_report = requested_scrublet_params.copy()
                    if not _metrics_precomputed:
                        calculate_qc_metrics(adata, inplace=True)
                else:
                    raise

            # Compute and store QC flag columns for cluster-level cleanup later.
            # Flags are observations, not filters — no cells are removed here.
            if flag_only and 'pct_counts_mt' in adata.obs.columns:
                median_mt_for_flags = float(adata.obs['pct_counts_mt'].median())
                mt_flag_threshold = float(tool_input.get("mt_flag_threshold") or (5.0 if median_mt_for_flags < 2.0 else 25.0))
                lib_flag_threshold = float(tool_input.get("lib_flag_threshold") or 500.0)
                genes_flag_threshold = int(tool_input.get("genes_flag_threshold") or 200)
                adata.obs['qc_flag_high_mt'] = adata.obs['pct_counts_mt'] > mt_flag_threshold
                adata.obs['qc_flag_low_lib'] = adata.obs['total_counts'] < lib_flag_threshold
                adata.obs['qc_flag_low_genes'] = adata.obs['n_genes_by_counts'] < genes_flag_threshold
                n_flag_mt = int(adata.obs['qc_flag_high_mt'].sum())
                n_flag_lib = int(adata.obs['qc_flag_low_lib'].sum())
                n_flag_genes = int(adata.obs['qc_flag_low_genes'].sum())

            # Use adata directly — no copy needed, metrics are already there
            qc_preview = adata

            requested_data_type = tool_input.get("data_type")  # "single_cell" | "single_nucleus" | None
            if requested_mt_threshold is not None:
                mt_threshold = float(requested_mt_threshold)
                data_type_confirmed = True
            else:
                data_type_confirmed = False
                # No explicit threshold — pick a sentinel based on data type hint only.
                # The real threshold must come from inspecting the QC figures.
                median_mt_preview = float(qc_preview.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in qc_preview.obs else 0.0
                mt_threshold = 5.0 if (requested_data_type == "single_nucleus" or median_mt_preview < 2.0) else 20.0
                warnings.append(
                    f"mt_threshold not set explicitly (median MT={median_mt_preview:.2f}%). "
                    "Review the MT% distribution in the QC figure and choose a data-driven threshold "
                    "based on where the high-MT tail separates from the main population. "
                    "Then re-run with mt_threshold=<value> to get exact removal counts."
                )
            if not filter_mt:
                warnings.append(
                    "Hard MT% cell filtering is disabled for this run; MT metrics are "
                    "reported for QC review only."
                )

            _mt_col = qc_preview.obs['pct_counts_mt'] if 'pct_counts_mt' in qc_preview.obs else None
            cells_over_mt = int((_mt_col >= mt_threshold).sum()) if _mt_col is not None else 0
            # Report exact removal counts at a range of thresholds so the model can present
            # data-driven options without anchoring to any single "standard" value.
            mt_threshold_options = {
                str(t): {
                    "threshold": t,
                    "cells_flagged": int((_mt_col >= t).sum()) if _mt_col is not None else 0,
                    "pct_flagged": round(float((_mt_col >= t).mean()) * 100, 1) if _mt_col is not None else 0,
                }
                for t in [5, 10, 15, 20, 25, 30]
            } if not data_type_confirmed else None
            predicted_doublets = int(qc_preview.obs['predicted_doublet'].sum()) if 'predicted_doublet' in qc_preview.obs else 0
            genes_low_cells = (
                int((qc_preview.var['n_cells_by_counts'] < min_cells).sum())
                if min_cells is not None and 'n_cells_by_counts' in qc_preview.var.columns
                else 0
            )
            cells_low_genes = (
                int((qc_preview.obs['n_genes_by_counts'] < int(min_genes)).sum())
                if min_genes is not None and 'n_genes_by_counts' in qc_preview.obs.columns
                else 0
            )
            ribo_genes = int(qc_preview.var['ribo'].sum()) if 'ribo' in qc_preview.var.columns and remove_ribo else 0
            mt_genes = int(qc_preview.var['mt'].sum()) if 'mt' in qc_preview.var.columns and remove_mt else 0
            n_mt_genes_detected = int(qc_preview.var['mt'].sum()) if 'mt' in qc_preview.var.columns else 0
            n_ribo_genes_detected = int(qc_preview.var['ribo'].sum()) if 'ribo' in qc_preview.var.columns else 0

            cell_removal_mask = pd.Series(False, index=qc_preview.obs_names)
            if min_genes is not None and 'n_genes_by_counts' in qc_preview.obs.columns:
                cell_removal_mask |= qc_preview.obs['n_genes_by_counts'] < int(min_genes)
            if filter_mt and 'pct_counts_mt' in qc_preview.obs.columns:
                cell_removal_mask |= qc_preview.obs['pct_counts_mt'] >= mt_threshold
            if remove_doublets and 'predicted_doublet' in qc_preview.obs.columns:
                cell_removal_mask |= qc_preview.obs['predicted_doublet'].astype(bool)
            projected_cells_removed = int(cell_removal_mask.sum())
            projected_cells_retained = int(n_before - projected_cells_removed)

            gene_removal_mask = pd.Series(False, index=qc_preview.var_names)
            if min_cells is not None and 'n_cells_by_counts' in qc_preview.var.columns:
                gene_removal_mask |= qc_preview.var['n_cells_by_counts'] < int(min_cells)
            if remove_ribo and 'ribo' in qc_preview.var.columns:
                gene_removal_mask |= qc_preview.var['ribo'].astype(bool)
            if remove_mt and 'mt' in qc_preview.var.columns:
                gene_removal_mask |= qc_preview.var['mt'].astype(bool)
            projected_genes_removed = int(gene_removal_mask.sum())
            projected_genes_retained = int(g_before - projected_genes_removed)

            figure_outputs = []
            figure_dir = tool_input.get("figure_dir")
            if figure_dir:
                import scanpy as sc
                import matplotlib.pyplot as plt
                os.makedirs(figure_dir, exist_ok=True)
                try:
                    qc_plot_adata = adata.copy()

                    # Use log1p columns from calculate_qc_metrics (log1p=True adds these automatically).
                    # Fall back to computing log10 only if log1p columns are absent.
                    if "log1p_total_counts" not in qc_plot_adata.obs.columns and "total_counts" in qc_plot_adata.obs.columns:
                        qc_plot_adata.obs["log1p_total_counts"] = np.log1p(qc_plot_adata.obs["total_counts"])
                    if "log1p_n_genes_by_counts" not in qc_plot_adata.obs.columns and "n_genes_by_counts" in qc_plot_adata.obs.columns:
                        qc_plot_adata.obs["log1p_n_genes_by_counts"] = np.log1p(qc_plot_adata.obs["n_genes_by_counts"])

                    # --- Figure 1: Main QC violin plots (log1p counts, log1p genes, MT%, ribo%) ---
                    main_violin_keys = []
                    if "log1p_total_counts" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("log1p_total_counts")
                    if "log1p_n_genes_by_counts" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("log1p_n_genes_by_counts")
                    if "pct_counts_mt" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("pct_counts_mt")
                    if "pct_counts_ribo" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("pct_counts_ribo")

                    if main_violin_keys:
                        sc.pl.violin(qc_plot_adata, main_violin_keys, jitter=0.2, multi_panel=True, show=False)
                        plt.suptitle("QC Metrics Distribution", y=1.02)
                        violin_path = os.path.join(figure_dir, "qc_violin_metrics.png")
                        plt.savefig(violin_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        figure_outputs.append(violin_path)

                    # --- Figure 2: Doublet scores (if available) ---
                    if "doublet_score" in qc_plot_adata.obs.columns:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        # Violin of doublet scores
                        sc.pl.violin(qc_plot_adata, "doublet_score", jitter=0.2, ax=axes[0], show=False)
                        axes[0].set_title("Doublet Score Distribution")
                        # Histogram with threshold
                        axes[1].hist(qc_plot_adata.obs["doublet_score"], bins=50, edgecolor='black', alpha=0.7)
                        axes[1].axvline(0.25, color='red', linestyle='--', label='Typical threshold (0.25)')
                        axes[1].set_xlabel("Doublet Score")
                        axes[1].set_ylabel("Cell Count")
                        axes[1].set_title("Doublet Score Histogram")
                        axes[1].legend()
                        fig.tight_layout()
                        doublet_path = os.path.join(figure_dir, "qc_doublet_scores.png")
                        fig.savefig(doublet_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        figure_outputs.append(doublet_path)

                    # --- Figure 3: Histograms for counts and genes (log scale) ---
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    if "total_counts" in qc_plot_adata.obs.columns:
                        axes[0].hist(qc_plot_adata.obs["total_counts"], bins=100, edgecolor='black', alpha=0.7)
                        axes[0].set_xscale('log')
                        axes[0].set_xlabel("Total Counts (log scale)")
                        axes[0].set_ylabel("Cell Count")
                        axes[0].set_title("Library Size Distribution")
                    if "n_genes_by_counts" in qc_plot_adata.obs.columns:
                        axes[1].hist(qc_plot_adata.obs["n_genes_by_counts"], bins=100, edgecolor='black', alpha=0.7)
                        axes[1].set_xscale('log')
                        axes[1].set_xlabel("Genes Detected (log scale)")
                        axes[1].set_ylabel("Cell Count")
                        axes[1].set_title("Genes per Cell Distribution")
                    fig.tight_layout()
                    hist_path = os.path.join(figure_dir, "qc_histograms.png")
                    fig.savefig(hist_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    figure_outputs.append(hist_path)

                    # --- Figure 4: MT% histogram (threshold line only when user confirmed a value) ---
                    if "pct_counts_mt" in qc_plot_adata.obs.columns:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(qc_plot_adata.obs["pct_counts_mt"], bins=100, edgecolor='black', alpha=0.7)
                        if data_type_confirmed:
                            ax.axvline(mt_threshold, color='red', linestyle='--', linewidth=2,
                                       label=f'Threshold: {mt_threshold:.1f}%')
                            ax.legend()
                        ax.set_xlabel("Mitochondrial %")
                        ax.set_ylabel("Cell Count")
                        ax.set_title("Mitochondrial Content Distribution")
                        fig.tight_layout()
                        mt_hist_path = os.path.join(figure_dir, "qc_mt_histogram.png")
                        fig.savefig(mt_hist_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        figure_outputs.append(mt_hist_path)

                    # --- Figure 5: Scatter plots (genes vs counts, MT vs counts) ---
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    if 'n_genes_by_counts' in qc_plot_adata.obs.columns and 'total_counts' in qc_plot_adata.obs.columns:
                        sc.pl.scatter(qc_plot_adata, x='total_counts', y='n_genes_by_counts', ax=axes[0], show=False)
                        axes[0].set_xscale('log')
                        axes[0].set_yscale('log')
                        axes[0].set_title("Genes vs Counts (log-log)")
                    if 'pct_counts_mt' in qc_plot_adata.obs.columns and 'total_counts' in qc_plot_adata.obs.columns:
                        sc.pl.scatter(qc_plot_adata, x='total_counts', y='pct_counts_mt', ax=axes[1], show=False)
                        axes[1].set_xscale('log')
                        if data_type_confirmed:
                            axes[1].axhline(mt_threshold, color='red', linestyle='--', linewidth=2,
                                            label=f'MT threshold: {mt_threshold:.1f}%')
                            axes[1].legend()
                        axes[1].set_title("MT% vs Counts")
                    fig.tight_layout()
                    scatter_path = os.path.join(figure_dir, "qc_scatter.png")
                    fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    figure_outputs.append(scatter_path)

                    # --- Figure 6: Ribo vs MT scatter (if both available) ---
                    if 'pct_counts_mt' in qc_plot_adata.obs.columns and 'pct_counts_ribo' in qc_plot_adata.obs.columns:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sc.pl.scatter(qc_plot_adata, x='pct_counts_mt', y='pct_counts_ribo', ax=ax, show=False)
                        if data_type_confirmed:
                            ax.axvline(mt_threshold, color='red', linestyle='--', linewidth=1, label=f'MT threshold')
                            ax.legend()
                        ax.set_title("Ribosomal vs Mitochondrial Content")
                        fig.tight_layout()
                        ribo_mt_path = os.path.join(figure_dir, "qc_ribo_vs_mt.png")
                        fig.savefig(ribo_mt_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        figure_outputs.append(ribo_mt_path)

                except Exception as e:
                    warnings.append(f"QC figure generation failed: {e}")

            qc_decisions = {
                "mt_threshold": {
                    "value": mt_threshold,
                    "filter_enabled": filter_mt,
                    "data_type_confirmed": data_type_confirmed,
                    **({"threshold_options": mt_threshold_options} if mt_threshold_options else {}),
                    "reason": (
                        f"Cells with pct_counts_mt >= {mt_threshold:.1f}% are flagged for review; "
                        + (
                            "this threshold will be applied in filtering."
                            if filter_mt
                            else "hard MT% filtering is disabled for this run."
                        )
                    ),
                    "cells_flagged": cells_over_mt,
                },
                "min_genes": {
                    "value": int(min_genes) if min_genes is not None else None,
                    "enabled": min_genes is not None,
                    "reason": (
                        f"Cells with fewer than {int(min_genes)} detected genes are usually low-quality."
                        if min_genes is not None
                        else "No cell-level gene-count filter was requested."
                    ),
                    "cells_flagged": cells_low_genes,
                },
                "min_cells_per_gene": {
                    "value": int(min_cells) if min_cells is not None else None,
                    "reason": (
                        f"Genes detected in fewer than {int(min_cells)} cells add noise and little clustering signal."
                        if min_cells is not None
                        else "No min-cells-per-gene filter requested."
                    ),
                    "genes_flagged": genes_low_cells,
                },
                "doublet_detection": {
                    "enabled": bool(detect_doublets_flag),
                    "reason": (
                        "Scrublet flags likely multiplets; these are reported so the user can decide whether to exclude them."
                    ),
                    "cells_flagged": predicted_doublets,
                    "remove_on_apply": remove_doublets,
                    "predictions_source": doublet_predictions_source,
                    "parameters": scrublet_params_for_report,
                },
                "remove_ribo": {
                    "enabled": bool(remove_ribo),
                    "reason": "Ribosomal genes can dominate variance and dilute biologically informative structure.",
                    "genes_flagged": ribo_genes,
                },
                "remove_mt_genes": {
                    "enabled": bool(remove_mt),
                    "reason": "Mitochondrial genes are often excluded from downstream feature selection to reduce QC-driven signal.",
                    "genes_flagged": mt_genes,
                },
                "gene_detection_counts": {
                    "n_mt_genes_detected": n_mt_genes_detected,
                    "n_ribo_genes_detected": n_ribo_genes_detected,
                },
            }
            filtering_plan = {
                "confirmation_required": not preview_only and not confirm_filtering,
                "confirmed": bool(confirm_filtering and not preview_only),
                "parameters": {
                    "mt_threshold": mt_threshold,
                    "filter_mt": filter_mt,
                    "data_type": requested_data_type,
                    "data_type_confirmed": data_type_confirmed,
                    "min_genes": int(min_genes) if min_genes is not None else None,
                    "min_cells_per_gene": int(min_cells) if min_cells is not None else None,
                    "remove_ribo": bool(remove_ribo),
                    "remove_mt_genes": bool(remove_mt),
                    "detect_doublets": bool(detect_doublets_flag),
                    "remove_doublets": bool(remove_doublets),
                    "batch_key_for_doublets": batch_key,
                    "scrublet": scrublet_params_for_report,
                },
                "cell_filters": {
                    "low_genes": {
                        "enabled": min_genes is not None,
                        "threshold": int(min_genes) if min_genes is not None else None,
                        "cells_flagged": cells_low_genes,
                        "will_remove": min_genes is not None,
                    },
                    "high_mt": {
                        "enabled": bool(filter_mt),
                        "threshold_pct": mt_threshold,
                        "cells_flagged": cells_over_mt,
                        "will_remove": bool(filter_mt),
                    },
                    "doublets": {
                        "enabled": bool(detect_doublets_flag),
                        "cells_flagged": predicted_doublets,
                        "will_remove": bool(remove_doublets),
                    },
                },
                "gene_filters": {
                    "low_cells": {
                        "enabled": min_cells is not None,
                        "threshold": int(min_cells) if min_cells is not None else None,
                        "genes_flagged": genes_low_cells,
                        "will_remove": min_cells is not None,
                    },
                    "ribosomal": {
                        "enabled": bool(remove_ribo),
                        "genes_flagged": ribo_genes,
                        "will_remove": bool(remove_ribo),
                    },
                    "mitochondrial": {
                        "enabled": bool(remove_mt),
                        "genes_flagged": mt_genes,
                        "will_remove": bool(remove_mt),
                    },
                },
                "projected_after_filtering": {
                    "cells_removed": projected_cells_removed,
                    "cells_retained": projected_cells_retained,
                    "genes_removed_before_cell_filtering": projected_genes_removed,
                    "genes_retained_before_cell_filtering": projected_genes_retained,
                    "note": (
                        "Gene removals are estimated before cell filtering; final gene removals can change "
                        "after cells are removed."
                    ),
                },
            }

            cell_filter_bits = []
            if min_genes is not None:
                cell_filter_bits.append(
                    f"removing {cells_low_genes} cells with fewer than {int(min_genes)} genes"
                )
            if filter_mt:
                cell_filter_bits.append(
                    f"filtering {cells_over_mt} cells with pct_counts_mt >= {mt_threshold:.1f}%"
                )
            else:
                cell_filter_bits.append(
                    f"reporting {cells_over_mt} cells with pct_counts_mt >= {mt_threshold:.1f}% without applying an MT filter"
                )
            doublet_action = "removing" if remove_doublets else "flagging"
            recommendation = (
                "I recommend "
                + ", ".join(cell_filter_bits)
                + f", removing {genes_low_cells} low-detection genes, and "
                + (
                    f"{doublet_action} doublets ({predicted_doublets} cells)."
                    if detect_doublets_flag
                    else "skipping doublet detection."
                )
            )
            if batch_resolution and batch_resolution.needs_user_confirmation and tool_input.get("batch_key"):
                recommendation += (
                    f" I could not confirm the requested per-batch column automatically; "
                    f"'{batch_resolution.recommended_column}' looks closest."
                )

            batch_strategy = (
                metadata_resolution_to_dict(batch_resolution)
                if batch_resolution
                else {
                    "status": "not_applicable",
                    "requested_column": requested_batch_key,
                    "applied_column": None,
                    "recommended_column": None,
                    "recommended_role": None,
                    "needs_user_confirmation": False,
                    "reason": "Doublet detection is disabled for this QC run.",
                    "candidates": [],
                }
            )
            batch_strategy["used_for_doublets"] = batch_key
            decisions = []
            batch_decision = decision_for_batch_strategy(
                batch_strategy,
                context="doublet_detection",
                source_tool="run_qc",
                batch_relevant=bool(tool_input.get("batch_key")),
            )
            if batch_decision is not None:
                decisions.append(batch_decision)
            artifact_payloads = [
                artifact
                for artifact in (
                    _artifact_payload(path, role="qc_figure", metadata={"mode": "preview" if preview_only else "applied"})
                    for path in figure_outputs
                )
                if artifact is not None
            ]

            if preview_only:
                # In flag_only mode: strip all filter-suggestive content from the result.
                # If we return mt_threshold_options, recommendation, filtering_plan, etc.,
                # the model will respond to that data and propose global filtering — even if
                # the system prompt says not to. The result must only contain observational
                # metrics and an unambiguous next-step instruction.
                _obs = qc_preview.obs
                flag_summary = {}
                if flag_only:
                    flag_summary = {
                        "qc_flag_high_mt": int(_obs['qc_flag_high_mt'].sum()) if 'qc_flag_high_mt' in _obs else 0,
                        "qc_flag_low_lib": int(_obs['qc_flag_low_lib'].sum()) if 'qc_flag_low_lib' in _obs else 0,
                        "qc_flag_low_genes": int(_obs['qc_flag_low_genes'].sum()) if 'qc_flag_low_genes' in _obs else 0,
                        "predicted_doublets": predicted_doublets,
                    }
                preview_result = {
                    "status": "ok",
                    "tool": "run_qc",
                    "mode": "flag_only" if flag_only else "preview",
                    "before": {"n_cells": n_before, "n_genes": g_before},
                    "metrics": {
                        "median_pct_mt": round(float(_obs['pct_counts_mt'].median()), 2) if 'pct_counts_mt' in _obs else None,
                        "median_total_counts": round(float(_obs['total_counts'].median()), 0) if 'total_counts' in _obs else None,
                        "median_n_genes": round(float(_obs['n_genes_by_counts'].median()), 0) if 'n_genes_by_counts' in _obs else None,
                        "doublet_rate_pct": round(float(_obs['predicted_doublet'].mean()) * 100, 1) if 'predicted_doublet' in _obs else None,
                        "n_mt_genes_detected": n_mt_genes_detected,
                        "n_ribo_genes_detected": n_ribo_genes_detected,
                    },
                    **({"flags_stored_in_obs": flag_summary} if flag_only else {"qc_decisions": qc_decisions}),
                    "next_step": (
                        "QC metrics and flags are stored in adata.obs. No cells were removed. "
                        "Proceed immediately to normalize_and_hvg — filtering decisions are deferred to cluster-level QC after embedding."
                        if flag_only else
                        "Review the figures and proposed thresholds, then confirm to apply filtering."
                    ),
                    "warnings": warnings,
                    "figures": figure_outputs,
                    "batch_strategy": batch_strategy,
                    "state": make_state(adata)
                }
                verification_checks = [
                    _check("qc_metrics_computed", "pct_counts_mt" in qc_preview.obs.columns, "QC metrics were computed on the preview copy."),
                    _check(
                        "batch_strategy_reported",
                        bool(batch_strategy.get("status")),
                        "Batch strategy is attached to the QC preview.",
                    ),
                ]
                if figure_outputs:
                    verification_checks.append(
                        _check(
                            "preview_figures_exist",
                            all(os.path.exists(path) for path in figure_outputs),
                            "QC preview figures were written to disk.",
                        )
                    )
                return _finalize_result(
                    preview_result,
                    adata,
                    dataset_changed=False,
                    summary="QC metrics computed and flags stored. No cells removed — proceeding to normalization.",
                    artifacts_created=artifact_payloads,
                    decisions_raised=decisions,
                    verification=_build_verification(
                        "passed",
                        "QC flag-only pass completed.",
                        verification_checks,
                    ),
                )

            if not confirm_filtering:
                confirmation_result = {
                    "status": "needs_confirmation",
                    "tool": "run_qc",
                    "mode": "confirmation_required",
                    "message": (
                        "QC filtering was not applied. Review the thresholds, parameters, and removal "
                        "counts, then confirm before I remove cells or genes."
                    ),
                    "required_next_action": "ask_user",
                    "before": {"n_cells": n_before, "n_genes": g_before},
                    "after": {"n_cells": n_before, "n_genes": g_before},
                    "recommendation": recommendation,
                    "filtering_plan": filtering_plan,
                    "qc_decisions": qc_decisions,
                    "metrics": {
                        "median_pct_mt": float(qc_preview.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in qc_preview.obs else None,
                        "doublet_rate": float(qc_preview.obs['predicted_doublet'].mean()) if 'predicted_doublet' in qc_preview.obs else None,
                        "cells_below_min_genes": cells_low_genes,
                        "genes_below_min_cells": genes_low_cells,
                        "predicted_doublets": predicted_doublets,
                        "cells_over_mt_threshold": cells_over_mt,
                    },
                    "warnings": warnings,
                    "figures": figure_outputs,
                    "batch_strategy": batch_strategy,
                    "state": make_state(adata),
                }
                return _finalize_result(
                    confirmation_result,
                    adata,
                    dataset_changed=False,
                    summary="Prepared a QC filtering plan and paused for explicit confirmation.",
                    artifacts_created=artifact_payloads,
                    decisions_raised=decisions,
                    verification=_build_verification(
                        "passed",
                        "QC filtering did not modify the dataset because confirmation is required.",
                        [
                            _check(
                                "dataset_unchanged",
                                adata.n_obs == n_before and adata.n_vars == g_before,
                                "No cells or genes were removed before confirmation.",
                            )
                        ],
                    ),
                )

            try:
                run_qc_pipeline(
                    adata,
                    mt_threshold=mt_threshold,
                    filter_mt=filter_mt,
                    min_genes=int(min_genes) if min_genes is not None else None,
                    min_cells=min_cells,
                    remove_ribo=remove_ribo,
                    detect_doublets_flag=detect_doublets_flag,
                    remove_doublets=remove_doublets,
                    batch_key=batch_key,
                    scrublet_expected_doublet_rate=scrublet_expected_doublet_rate,
                    scrublet_sim_doublet_ratio=scrublet_sim_doublet_ratio,
                    scrublet_n_prin_comps=scrublet_n_prin_comps,
                    scrublet_min_counts=scrublet_min_counts,
                    scrublet_min_cells=scrublet_min_cells,
                    scrublet_min_gene_variability_pctl=scrublet_min_gene_variability_pctl,
                    scrublet_random_state=scrublet_random_state,
                    force_doublet_recompute=force_doublet_recompute,
                )
            except ValueError as e:
                if detect_doublets_flag and "skimage is not installed" in str(e):
                    warnings.append("Scrublet auto-threshold requires skimage; reran QC without doublet detection.")
                    detect_doublets_flag = False
                    run_qc_pipeline(
                        adata,
                        mt_threshold=mt_threshold,
                        filter_mt=filter_mt,
                        min_genes=int(min_genes) if min_genes is not None else None,
                        min_cells=min_cells,
                        remove_ribo=remove_ribo,
                        detect_doublets_flag=False,
                        remove_doublets=False,
                        batch_key=batch_key,
                    )
                else:
                    raise

            actual_cells_removed = n_before - adata.n_obs
            actual_genes_removed = g_before - adata.n_vars
            post_filter_doublets = int(adata.obs['predicted_doublet'].sum()) if 'predicted_doublet' in adata.obs else None
            post_filter_doublet_rate = (
                float(adata.obs['predicted_doublet'].mean())
                if 'predicted_doublet' in adata.obs
                else None
            )
            pre_filter_doublet_rate = (
                float(predicted_doublets / n_before)
                if detect_doublets_flag and n_before
                else None
            )
            qc_decisions["min_cells_per_gene"]["genes_flagged_before_cell_filtering"] = genes_low_cells
            qc_decisions["min_cells_per_gene"]["genes_removed_after_cell_filtering"] = actual_genes_removed
            qc_decisions["doublet_detection"]["cells_flagged_before_filtering"] = predicted_doublets
            qc_decisions["doublet_detection"]["cells_remaining_after_filtering"] = post_filter_doublets

            output_path = fix_output_path(tool_input.get("output_path"), "run_qc")
            if output_path:
                write_h5ad_safe(adata, output_path)
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifact_payloads.append(artifact)

            qc_result = {
                "status": "ok",
                "tool": "run_qc",
                "input_path": tool_input.get("data_path", "memory"),
                "output_path": output_path,
                "saved": output_path is not None,
                "before": {"n_cells": n_before, "n_genes": g_before},
                "after": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "recommendation": recommendation,
                "filtering_plan": filtering_plan,
                "qc_decisions": qc_decisions,
                "metrics": {
                    "cells_removed": actual_cells_removed,
                    "genes_removed": actual_genes_removed,
                    "doublet_rate": pre_filter_doublet_rate,
                    "post_filter_doublet_rate": post_filter_doublet_rate,
                    "median_pct_mt": float(adata.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in adata.obs else None,
                    "cells_below_min_genes": cells_low_genes,
                    "genes_below_min_cells": genes_low_cells,
                    "genes_below_min_cells_before_cell_filtering": genes_low_cells,
                    "genes_removed_after_cell_filtering": actual_genes_removed,
                    "predicted_doublets": predicted_doublets,
                    "predicted_doublets_before_filtering": predicted_doublets,
                    "predicted_doublets_remaining": post_filter_doublets,
                    "cells_over_mt_threshold": cells_over_mt,
                },
                "warnings": warnings,
                "figures": figure_outputs,
                "batch_strategy": batch_strategy,
                "state": make_state(adata)
            }
            verification_checks = [
                _check("qc_metrics_present", "pct_counts_mt" in adata.obs.columns, "MT QC metric is present after QC."),
                _check(
                    "batch_key_valid",
                    batch_key is None or batch_key in adata.obs.columns,
                    f"Batch key '{batch_key}' is available on the filtered AnnData." if batch_key else "QC ran without per-batch stratification.",
                ),
            ]
            if detect_doublets_flag:
                verification_checks.append(
                    _check(
                        "doublet_scores_present",
                        "predicted_doublet" in adata.obs.columns,
                        "Doublet scores/predictions are available after QC.",
                    )
                )
            return _finalize_result(
                qc_result,
                adata,
                dataset_changed=True,
                summary="Applied QC filtering and recorded the batch-aware doublet strategy.",
                artifacts_created=artifact_payloads,
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "QC completed and the resulting AnnData passed post-action checks.",
                    verification_checks,
                ),
            )

        elif tool_name == "normalize_and_hvg":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_hvg = int(tool_input.get("n_hvg", 4000))
            target_sum = tool_input.get("target_sum", 10000)
            target_sum = float(target_sum) if target_sum is not None else None
            log_transform = bool(tool_input.get("log_transform", True))
            raw_layer_name = tool_input.get("raw_layer_name", "raw_counts")
            normalization_source = tool_input.get("normalization_source") or "auto"
            legacy_force_reset = tool_input.get("force_reset_from_raw")
            if legacy_force_reset is not None and "normalization_source" not in tool_input:
                normalization_source = "raw_counts" if bool(legacy_force_reset) else "current_X"
            preserve_input_x_layer = tool_input.get("preserve_input_x_layer", "pre_scagent_X")
            set_raw_after_normalization = bool(tool_input.get("set_raw_after_normalization", True))
            hvg_flavor = tool_input.get("hvg_flavor", "seurat_v3")
            hvg_layer = tool_input.get("hvg_layer") or None
            batch_key = tool_input.get("batch_key")
            default_ribo_patterns = [
                r"^(RPL|RPS|MRPL|MRPS)",
                r"^(Rpl|Rps|Mrpl|Mrps)",
            ]
            remove_ribo_param_present = "remove_ribosomal_genes" in tool_input
            if remove_ribo_param_present:
                remove_ribosomal_genes = bool(tool_input.get("remove_ribosomal_genes"))
            elif tool_input.get("exclude_ribosomal_from_hvg") is False:
                # Backward compatibility: older prompts used this as the
                # only way to say ribosomal genes should remain usable.
                remove_ribosomal_genes = False
            else:
                remove_ribosomal_genes = True

            if "exclude_ribosomal_from_hvg" in tool_input:
                exclude_ribosomal_from_hvg = bool(tool_input.get("exclude_ribosomal_from_hvg"))
            else:
                exclude_ribosomal_from_hvg = not remove_ribosomal_genes

            ribosomal_remove_patterns = tool_input.get("ribosomal_remove_patterns") or default_ribo_patterns
            if isinstance(ribosomal_remove_patterns, str):
                ribosomal_remove_patterns = [ribosomal_remove_patterns]
            ribosomal_remove_patterns = [
                str(pattern) for pattern in ribosomal_remove_patterns if str(pattern)
            ]
            hvg_exclude_patterns = tool_input.get("hvg_exclude_patterns") or []
            if isinstance(hvg_exclude_patterns, str):
                hvg_exclude_patterns = [hvg_exclude_patterns]
            hvg_exclude_patterns = [str(pattern) for pattern in hvg_exclude_patterns if str(pattern)]
            if exclude_ribosomal_from_hvg:
                hvg_exclude_patterns = default_ribo_patterns + hvg_exclude_patterns
            hvg_exclusion_mode = tool_input.get("hvg_exclusion_mode", "pre")
            hvg_exclude_match_mode = tool_input.get("hvg_exclude_match_mode", "match")
            hvg_exclusion_source = tool_input.get("hvg_exclusion_source")
            if exclude_ribosomal_from_hvg:
                if hvg_exclusion_source:
                    hvg_exclusion_source = (
                        "scagent retained ribosomal genes by explicit request "
                        "and excluded them before HVG; "
                        f"additional source: {hvg_exclusion_source}"
                    )
                else:
                    hvg_exclusion_source = (
                        "scagent retained ribosomal genes by explicit request "
                        "and excluded them before HVG"
                    )

            before_shape = (adata.n_obs, adata.n_vars)

            def _feature_mask_from_patterns(var_names, patterns, match_mode="match"):
                names = [str(name) for name in var_names]
                mask = np.zeros(len(names), dtype=bool)
                for pattern in patterns:
                    regex = re.compile(pattern)
                    if match_mode == "contains":
                        mask |= np.asarray([bool(regex.search(name)) for name in names])
                    elif match_mode == "fullmatch":
                        mask |= np.asarray([bool(regex.fullmatch(name)) for name in names])
                    else:
                        mask |= np.asarray([bool(regex.match(name)) for name in names])
                return mask

            ribosomal_removal_meta = {
                "enabled": bool(remove_ribosomal_genes),
                "patterns": ribosomal_remove_patterns,
                "match_mode": "match",
                "source": (
                    "scagent project default ribosomal gene removal before "
                    "normalization/HVG"
                    if remove_ribosomal_genes
                    else "ribosomal genes retained by explicit request/source setting"
                ),
                "n_removed": 0,
            }
            if remove_ribosomal_genes:
                ribo_mask = _feature_mask_from_patterns(
                    adata.var_names,
                    ribosomal_remove_patterns,
                    match_mode="match",
                )
                n_ribo_removed = int(ribo_mask.sum())
                ribosomal_removal_meta["n_removed"] = n_ribo_removed
                if n_ribo_removed:
                    removed_names = [str(name) for name in adata.var_names[ribo_mask][:25]]
                    ribosomal_removal_meta["example_removed_genes"] = removed_names
                    adata = adata[:, ~ribo_mask].copy()
                feature_removals = dict(adata.uns.get("feature_removals", {}))
                feature_removals["ribosomal_genes"] = ribosomal_removal_meta
                adata.uns["feature_removals"] = feature_removals

            def _integer_like_matrix(matrix, n_rows: int = 100, n_cols: int = 100) -> bool:
                if matrix is None:
                    return False
                if hasattr(matrix, "data") and hasattr(matrix, "nnz"):
                    values = np.asarray(matrix.data[:max(n_rows * n_cols, 1)])
                    if values.size == 0:
                        return True
                    return bool(np.allclose(values, np.round(values)))
                sample = matrix[:min(n_rows, matrix.shape[0]), :min(n_cols, matrix.shape[1])]
                if hasattr(sample, "toarray"):
                    sample = sample.toarray()
                sample = np.asarray(sample)
                return bool(np.allclose(sample, np.round(sample)))

            try:
                normalize_data(
                    adata,
                    target_sum=target_sum,
                    log_transform=log_transform,
                    preserve_raw=True,
                    raw_layer_name=raw_layer_name,
                    normalization_source=normalization_source,
                    preserve_input_x_layer=preserve_input_x_layer,
                )
                if set_raw_after_normalization:
                    adata.raw = adata.copy()
                select_hvg(
                    adata,
                    n_top_genes=n_hvg,
                    flavor=hvg_flavor,
                    layer=hvg_layer,
                    batch_key=batch_key,
                    exclude_patterns=hvg_exclude_patterns,
                    exclusion_mode=hvg_exclusion_mode,
                    exclude_match_mode=hvg_exclude_match_mode,
                    exclusion_source=hvg_exclusion_source,
                )
            except ValueError as e:
                return _error_result(
                    tool="normalize_and_hvg",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Retry normalize_and_hvg with normalization_source='raw_counts' "
                        f"so adata.X is restored from layers['{raw_layer_name}'] before normalization.",
                        "If the dataset was already normalized in this session, "
                        "do not call scanpy normalize/log1p manually on the current X; "
                        "reset from raw counts first.",
                        "If the dataset was loaded already-normalized from disk, "
                        "load the raw-counts version instead, or place raw "
                        f"counts into adata.layers['{raw_layer_name}'] before retrying.",
                    ],
                )

            output_path = fix_output_path(tool_input.get("output_path"), "normalize_and_hvg")
            if output_path:
                write_h5ad_safe(adata, output_path)

            artifact_payloads = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifact_payloads.append(artifact)

            hvg_meta = adata.uns.get("hvg", {})
            exclusion_meta = hvg_meta.get("feature_exclusions", {})
            raw_counts_present = raw_layer_name in adata.layers
            raw_counts_integer_like = (
                _integer_like_matrix(adata.layers[raw_layer_name])
                if raw_counts_present
                else False
            )
            raw_shape = list(adata.raw.shape) if adata.raw is not None else None
            normalized_counts_target = None
            if log_transform:
                import scipy.sparse as sp
                if sp.issparse(adata.X):
                    X_counts = adata.X.copy()
                    X_counts.data = np.expm1(X_counts.data)
                    normalized_counts_target = float(np.median(np.asarray(X_counts.sum(axis=1)).ravel()))
                else:
                    X_counts = np.expm1(np.asarray(adata.X))
                    normalized_counts_target = float(np.median(X_counts.sum(axis=1)))

            result_payload = {
                "status": "ok",
                "tool": "normalize_and_hvg",
                "output_path": output_path,
                "saved": output_path is not None,
                "before": {"n_cells": before_shape[0], "n_genes": before_shape[1]},
                "after": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "target_sum": target_sum,
                "log_transform": log_transform,
                "normalization": adata.uns.get("normalization", {}),
                "normalization_source": adata.uns.get("normalization", {}).get("normalization_source", normalization_source),
                "resolved_source": adata.uns.get("normalization", {}).get("resolved_source"),
                "reset_from_raw_counts": adata.uns.get("normalization", {}).get("reset_from_raw_counts"),
                "reset_reason": adata.uns.get("normalization", {}).get("reset_reason"),
                "input_x_preserved_layer": adata.uns.get("normalization", {}).get("input_x_preserved_layer"),
                "raw_layer_name": raw_layer_name,
                "raw_counts_present": raw_counts_present,
                "raw_counts_integer_like": raw_counts_integer_like,
                "adata_raw_set": adata.raw is not None,
                "adata_raw_shape": raw_shape,
                "set_raw_after_normalization": set_raw_after_normalization,
                "n_hvg": int(adata.var['highly_variable'].sum()),
                "feature_removals": {
                    "ribosomal_genes": ribosomal_removal_meta,
                },
                "hvg": {
                    "requested_flavor": hvg_meta.get("requested_flavor", hvg_flavor),
                    "flavor": hvg_meta.get("flavor", hvg_flavor),
                    "method": "scanpy.pp.highly_variable_genes",
                    "n_top_genes": int(hvg_meta.get("n_top_genes", n_hvg)),
                    "batch_key": hvg_meta.get("batch_key", batch_key),
                    "layer": hvg_meta.get("layer", hvg_layer),
                    "n_hvg_selected": int(adata.var['highly_variable'].sum()),
                },
                "remove_ribosomal_genes": remove_ribosomal_genes,
                "exclude_ribosomal_from_hvg": exclude_ribosomal_from_hvg,
                "feature_exclusions": exclusion_meta,
                "metrics": {
                    "n_hvg_selected": int(adata.var['highly_variable'].sum()),
                    "normalized_counts_target_median": normalized_counts_target,
                    "n_removed_ribosomal_genes": int(ribosomal_removal_meta.get("n_removed", 0) or 0),
                    "n_excluded_features": int(exclusion_meta.get("n_excluded", 0) or 0),
                    "excluded_features_marked_hvg": int(exclusion_meta.get("excluded_hvg_after_forcing", 0) or 0),
                },
                "warnings": warnings,
                "state": make_state(adata)
            }
            verification_checks = [
                _check(
                    "raw_counts_present",
                    raw_counts_present,
                    f"Raw counts layer '{raw_layer_name}' is present.",
                ),
                _check(
                    "raw_counts_integer_like",
                    raw_counts_integer_like,
                    f"Raw counts layer '{raw_layer_name}' appears integer-like in a matrix sample.",
                ),
                _check(
                    "hvg_count",
                    int(adata.var['highly_variable'].sum()) > 0,
                    f"{int(adata.var['highly_variable'].sum())} HVGs are marked.",
                ),
            ]
            if remove_ribosomal_genes:
                verification_checks.append(
                    _check(
                        "ribosomal_gene_removal_recorded",
                        "feature_removals" in adata.uns
                        and "ribosomal_genes" in adata.uns["feature_removals"],
                        (
                            f"Removed {int(ribosomal_removal_meta.get('n_removed', 0) or 0)} "
                            "ribosomal genes before normalization/HVG."
                        ),
                    )
                )
            if int(exclusion_meta.get("n_excluded", 0) or 0):
                verification_checks.append(
                    _check(
                        "excluded_features_not_hvg",
                        int(exclusion_meta.get("excluded_hvg_after_forcing", 0) or 0) == 0,
                        "No excluded features remain marked highly_variable.",
                    )
                )
            if set_raw_after_normalization:
                verification_checks.append(
                    _check(
                        "adata_raw_set",
                        adata.raw is not None,
                        "adata.raw was set after normalization/log1p.",
                    )
                )
            return _finalize_result(
                result_payload,
                adata,
                dataset_changed=True,
                summary=(
                    f"Normalized data to target_sum={target_sum}, "
                    f"selected {int(adata.var['highly_variable'].sum())} HVGs, "
                    f"removed {int(ribosomal_removal_meta.get('n_removed', 0) or 0)} ribosomal genes, "
                    f"and applied {int(exclusion_meta.get('n_excluded', 0) or 0)} feature exclusions before HVG selection."
                ),
                artifacts_created=artifact_payloads,
                verification=_build_verification(
                    "passed",
                    "Normalization and HVG selection completed with provenance checks.",
                    verification_checks,
                ),
            )

        elif tool_name == "run_pca":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_comps = int(tool_input.get("n_comps") or tool_input.get("n_pcs") or 50)
            svd_solver = tool_input.get("svd_solver", "arpack")
            mask_var = tool_input.get("mask_var", "highly_variable")

            run_pca(adata, n_comps=n_comps, mask_var=mask_var, svd_solver=svd_solver)

            output_path = fix_output_path(tool_input.get("output_path"), "run_pca")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Elbow detection: kneedle algorithm (max perpendicular distance from diagonal)
            variance_ratios = adata.uns["pca"]["variance_ratio"]
            n_shown = len(variance_ratios)

            def _find_pca_elbow(ratios):
                n = len(ratios)
                if n < 3:
                    return n
                x = np.arange(n, dtype=float)
                y = np.array(ratios, dtype=float)
                x_n = x / (n - 1)
                y_range = float(y.max() - y.min())
                y_n = (y - y.min()) / (y_range if y_range > 0 else 1.0)
                dx = float(x_n[-1] - x_n[0])
                dy = float(y_n[-1] - y_n[0])
                denom = float(np.sqrt(dx**2 + dy**2))
                if denom == 0:
                    return n // 2
                dist = np.abs(dy * x_n - dx * y_n + x_n[-1] * y_n[0] - y_n[-1] * x_n[0]) / denom
                return int(np.argmax(dist)) + 1  # 1-indexed

            elbow_pc = _find_pca_elbow(variance_ratios)
            elbow_buffer_n_pcs = 10
            elbow_buffered_n_pcs = min(elbow_pc + elbow_buffer_n_pcs, n_shown)
            conservative_floor_n_pcs = min(30, n_shown)
            suggested_n_pcs = max(elbow_buffered_n_pcs, conservative_floor_n_pcs)
            pca_selection_rationale = (
                f"Elbow detection placed the knee at PC{elbow_pc}. The buffered knee keeps "
                f"{elbow_buffered_n_pcs} PCs by adding {elbow_buffer_n_pcs} PCs beyond the knee, "
                "which is a conservative margin meant to retain biological signal that may sit "
                "just past the sharpest variance drop, not treating the knee as a hard cutoff. "
                f"The tool recommends {suggested_n_pcs} PCs because single-cell analyses usually "
                f"benefit from retaining at least {conservative_floor_n_pcs} PCs when available, "
                "so subtler immune states, rare populations, and technical structure are not "
                "discarded too early. "
                "The agent should still override this with explicit reasoning if the dataset "
                "is very small, clearly over-noisy, or the user/source specifies a different value."
            )

            # Scree plot
            scree_path = None
            scree_b64 = None
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as _plt

                _base = Path(run_manager.run_dir) if run_manager else Path(".")
                figures_dir = _base / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                scree_path = str(figures_dir / "pca_variance_explained.png")

                pcs = np.arange(1, n_shown + 1)
                cumvar = np.cumsum(variance_ratios) * 100

                fig, (ax1, ax2) = _plt.subplots(1, 2, figsize=(12, 4))

                ax1.bar(pcs, variance_ratios * 100, color="steelblue", alpha=0.7, width=0.8)
                ax1.axvline(elbow_pc, color="darkorange", linestyle="--", linewidth=1.5,
                            label=f"Elbow PC{elbow_pc}")
                ax1.axvline(suggested_n_pcs, color="firebrick", linestyle="--", linewidth=1.5,
                            label=f"Suggested n_pcs={suggested_n_pcs}")
                ax1.set_xlabel("Principal Component")
                ax1.set_ylabel("Variance Explained (%)")
                ax1.set_title("Variance per PC")
                ax1.legend(fontsize=9)

                ax2.plot(pcs, cumvar, "o-", markersize=3, color="steelblue")
                ax2.axvline(elbow_pc, color="darkorange", linestyle="--", linewidth=1.5,
                            label=f"Elbow PC{elbow_pc}")
                ax2.axvline(suggested_n_pcs, color="firebrick", linestyle="--", linewidth=1.5,
                            label=f"Suggested n_pcs={suggested_n_pcs}")
                ax2.axhline(80, color="gray", linestyle=":", linewidth=1, label="80% threshold")
                ax2.set_xlabel("Principal Component")
                ax2.set_ylabel("Cumulative Variance (%)")
                ax2.set_title("Cumulative Variance Explained")
                ax2.legend(fontsize=9)

                _plt.tight_layout()
                _plt.savefig(scree_path, dpi=150, bbox_inches="tight")
                _plt.close(fig)
                scree_b64 = encode_image_base64(scree_path)
            except Exception as _scree_err:
                logger.warning("Scree plot generation failed: %s", _scree_err)

            result = {
                "status": "ok",
                "tool": "run_pca",
                "output_path": output_path,
                "saved": output_path is not None,
                "n_comps": n_comps,
                "svd_solver": svd_solver,
                "mask_var": mask_var,
                "variance_explained_total": float(variance_ratios.sum()),
                "variance_ratio_per_pc": [round(float(v), 5) for v in variance_ratios],
                "elbow_pc": elbow_pc,
                "elbow_buffer_n_pcs": elbow_buffer_n_pcs,
                "elbow_buffered_n_pcs": elbow_buffered_n_pcs,
                "conservative_floor_n_pcs": conservative_floor_n_pcs,
                "suggested_n_pcs": suggested_n_pcs,
                "pca_selection_rationale": pca_selection_rationale,
                "scree_plot": scree_path,
                "side_effects": {
                    "pca_computed": True,
                    "neighbors_recomputed": False,
                    "umap_recomputed": False,
                    "clustering_recomputed": False,
                },
                "warnings": warnings,
                "state": make_state(adata),
            }
            if scree_b64:
                result["image_base64"] = scree_b64
                result["image_mime"] = "image/png"

            return _finalize_result(
                result,
                adata,
                dataset_changed=True,
                summary=(
                    f"Ran PCA with n_comps={n_comps}; elbow at PC{elbow_pc}, "
                    f"elbow+{elbow_buffer_n_pcs}={elbow_buffered_n_pcs}, suggested n_pcs={suggested_n_pcs} "
                    "for run_neighbors."
                ),
                verification=_build_verification(
                    "passed",
                    "PCA was computed without downstream graph or embedding side effects.",
                    [
                        _check("pca_present", "X_pca" in adata.obsm, "PCA embedding exists in adata.obsm['X_pca']."),
                        _check("no_umap_side_effect", "X_umap" not in adata.obsm or starting_state.get("has_umap"), "run_pca did not create a new UMAP embedding."),
                    ],
                ),
            )

        elif tool_name == "run_neighbors":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_neighbors = int(tool_input.get("n_neighbors") or 30)
            n_pcs = tool_input.get("n_pcs")
            n_pcs = int(n_pcs) if n_pcs is not None else None
            use_rep = tool_input.get("use_rep", "X_pca")
            metric = tool_input.get("metric", "euclidean")
            key_added = tool_input.get("key_added")

            if use_rep not in adata.obsm:
                available_reps = [k for k in adata.obsm.keys()]
                return _smart_unavailable_result(
                    tool="run_neighbors",
                    message=f"Representation '{use_rep}' not found in adata.obsm.",
                    adata_obj=adata,
                    missing_prerequisites=["pca"] if use_rep == "X_pca" else [use_rep],
                    recovery_options=[
                        f"Run run_pca first to compute '{use_rep}'." if use_rep == "X_pca"
                        else f"Compute '{use_rep}' before calling run_neighbors.",
                        f"Set use_rep to one of the available representations: {available_reps}",
                    ],
                    extra={"available_representations": available_reps},
                )

            if use_rep == "X_pca" and not _batch_correction_present(adata):
                batch_resolution = resolve_batch_metadata(adata)
                batch_key = batch_resolution.applied_column
                n_batches = int(adata.obs[batch_key].nunique(dropna=True)) if batch_key else 0
                if n_batches > 1:
                    warnings.append(
                        f"Detected {n_batches} groups in batch key '{batch_key}' and no recorded batch correction. "
                        "For open-ended multi-sample analyses, run score_integration and/or run_batch_correction "
                        "before building an uncorrected PCA neighbor graph unless this is intentional."
                    )

            compute_neighbors(
                adata,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                use_rep=use_rep,
                metric=metric,
                key_added=key_added,
            )

            output_path = fix_output_path(tool_input.get("output_path"), "run_neighbors")
            if output_path:
                write_h5ad_safe(adata, output_path)

            graph_key = key_added or "neighbors"
            result = {
                "status": "ok",
                "tool": "run_neighbors",
                "output_path": output_path,
                "saved": output_path is not None,
                "n_neighbors": n_neighbors,
                "n_pcs": n_pcs,
                "use_rep": use_rep,
                "metric": metric,
                "neighbors_key": graph_key,
                "neighbors_provenance": _neighbors_provenance(adata) if key_added is None else _sanitize_uns_value(adata.uns.get(key_added, {})),
                "side_effects": {
                    "pca_recomputed": False,
                    "neighbors_recomputed": True,
                    "umap_recomputed": False,
                    "clustering_recomputed": False,
                },
                "warnings": warnings,
                "state": make_state(adata),
            }
            return _finalize_result(
                result,
                adata,
                dataset_changed=True,
                summary=f"Computed neighbors only using {use_rep} with n_neighbors={n_neighbors}.",
                verification=_build_verification(
                    "passed",
                    "Neighbor graph was computed without UMAP or clustering side effects.",
                    [
                        _check("neighbors_present", graph_key in adata.uns, f"Neighbors key '{graph_key}' exists in adata.uns."),
                        _check("no_umap_side_effect", "X_umap" not in adata.obsm or starting_state.get("has_umap"), "run_neighbors did not create a new UMAP embedding."),
                    ],
                ),
            )

        elif tool_name == "run_umap":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            neighbors_key = tool_input.get("neighbors_key")
            _neighbors_lookup = neighbors_key or "neighbors"
            if _neighbors_lookup not in adata.uns:
                available_graphs = [k for k in adata.uns if "neighbor" in k.lower() or k == "neighbors"]
                return _smart_unavailable_result(
                    tool="run_umap",
                    message=f"Neighbor graph '{_neighbors_lookup}' not found in adata.uns.",
                    adata_obj=adata,
                    missing_prerequisites=["neighbors"],
                    recovery_options=[
                        "Run run_neighbors first to compute the neighbor graph.",
                        *(
                            [f"Or set neighbors_key to one of the existing graphs: {available_graphs}"]
                            if available_graphs else []
                        ),
                    ],
                    extra={"available_neighbor_graphs": available_graphs},
                )

            neighbors_before = _neighbors_provenance(adata)
            min_dist = float(tool_input.get("min_dist", 0.5))
            spread = float(tool_input.get("spread", 1.0))
            n_components = int(tool_input.get("n_components", 2))
            random_state = int(tool_input.get("random_state", 0))

            if not _batch_correction_present(adata):
                batch_resolution = resolve_batch_metadata(adata)
                batch_key = batch_resolution.applied_column
                n_batches = int(adata.obs[batch_key].nunique(dropna=True)) if batch_key else 0
                if n_batches > 1:
                    warnings.append(
                        f"Detected {n_batches} groups in batch key '{batch_key}' and no recorded batch correction. "
                        "UMAP will reflect the existing uncorrected neighbor graph unless batch correction is run first."
                    )

            compute_umap(
                adata,
                min_dist=min_dist,
                spread=spread,
                n_components=n_components,
                neighbors_key=neighbors_key,
                random_state=random_state,
            )
            neighbors_after = _neighbors_provenance(adata)
            graph_preserved = _provenance_same(neighbors_before, neighbors_after)

            output_path = fix_output_path(tool_input.get("output_path"), "run_umap")
            if output_path:
                write_h5ad_safe(adata, output_path)

            result = {
                "status": "ok",
                "tool": "run_umap",
                "output_path": output_path,
                "saved": output_path is not None,
                "min_dist": min_dist,
                "spread": spread,
                "n_components": n_components,
                "neighbors_key": neighbors_key or "neighbors",
                "random_state": random_state,
                "neighbors_before": neighbors_before,
                "neighbors_after": neighbors_after,
                "neighbor_graph_preserved": graph_preserved,
                "side_effects": {
                    "pca_recomputed": False,
                    "neighbors_recomputed": False,
                    "umap_recomputed": True,
                    "clustering_recomputed": False,
                },
                "warnings": warnings,
                "state": make_state(adata),
            }
            return _finalize_result(
                result,
                adata,
                dataset_changed=True,
                summary=f"Computed UMAP only from the existing neighbor graph with min_dist={min_dist}.",
                verification=_build_verification(
                    "passed" if graph_preserved else "warning",
                    "UMAP was computed from the existing neighbor graph.",
                    [
                        _check("umap_present", "X_umap" in adata.obsm, "UMAP embedding exists in adata.obsm['X_umap']."),
                        _check("neighbor_graph_preserved", graph_preserved, "Neighbor graph provenance and sparsity were unchanged by run_umap."),
                    ],
                ),
            )

        elif tool_name == "run_clustering":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "leiden")
            resolution = float(tool_input.get("resolution", 1.0))
            k = tool_input.get("k")
            use_rep = tool_input.get("use_rep")
            random_state = int(tool_input.get("random_state", 0))
            requested_cluster_key = tool_input.get("cluster_key")
            cluster_key, default_make_primary = _resolve_clustering_output_key(
                adata,
                method,
                resolution,
                requested_cluster_key,
            )
            make_primary = tool_input.get("make_primary")
            if make_primary is None:
                make_primary = default_make_primary
            if requested_cluster_key and requested_cluster_key == default_cluster_key_for_method(method) and not make_primary:
                warnings.append(
                    f"cluster_key '{requested_cluster_key}' is the primary alias for {method}; forcing make_primary=true."
                )
                make_primary = True

            result_payload = _apply_clustering(
                adata,
                method=method,
                resolution=resolution,
                cluster_key=cluster_key,
                make_primary=bool(make_primary),
                k=k,
                use_rep=use_rep,
                random_state=random_state,
            )

            output_path = fix_output_path(tool_input.get("output_path"), "run_clustering")
            if output_path:
                write_h5ad_safe(adata, output_path)
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)

            clustering_result = {
                "status": "ok",
                "tool": "run_clustering",
                "output_path": output_path,
                "saved": output_path is not None,
                "method": result_payload["method"],
                "resolution": result_payload["resolution"],
                "cluster_key": result_payload["cluster_key"],
                "created_obs_columns": result_payload["created_obs_columns"],
                "primary_alias": result_payload["primary_alias"],
                "primary_cluster_key": result_payload["primary_cluster_key"],
                "primary_alias_available": result_payload["primary_alias_available"],
                "primary_alias_created": result_payload["primary_alias_created"],
                "make_primary": bool(make_primary),
                "parameters": {
                    "k": int(k) if k is not None else None,
                    "use_rep": use_rep,
                    "random_state": random_state,
                },
                "n_clusters": result_payload["n_clusters"],
                "cluster_sizes": result_payload["cluster_sizes"],
                "available_clusterings": result_payload["clusterings"],
                "warnings": warnings,
                "state": make_state(adata)
            }
            primary_key = result_payload["primary_cluster_key"]
            primary_check = (
                _check(
                    "primary_alias_available",
                    result_payload["primary_alias_available"],
                    f"Primary clustering alias '{primary_key}' is available.",
                )
                if bool(make_primary)
                else _check(
                    "primary_alias_not_requested",
                    True,
                    "Primary alias was not requested; no alias availability is implied.",
                )
            )
            verification_checks = [
                _check(
                    "cluster_key_created",
                    result_payload["cluster_key"] in adata.obs.columns,
                    f"Clustering column '{result_payload['cluster_key']}' exists in adata.obs.",
                ),
                _check(
                    "cluster_count_matches",
                    adata.obs[result_payload["cluster_key"]].nunique() == result_payload["n_clusters"],
                    "Reported cluster count matches the stored clustering column.",
                ),
                primary_check,
            ]
            return _finalize_result(
                clustering_result,
                adata,
                dataset_changed=True,
                summary=(
                    f"Ran {result_payload['method']} clustering at resolution {result_payload['resolution']} "
                    f"and stored results in '{result_payload['cluster_key']}'."
                ),
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "Clustering results were created and verified against the AnnData state.",
                    verification_checks,
                ),
            )

        elif tool_name == "compare_clusterings":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "leiden")
            resolutions = [float(value) for value in tool_input.get("resolutions", [])]
            if not resolutions:
                raise ValueError("compare_clusterings requires at least one resolution.")
            k = tool_input.get("k")
            use_rep = tool_input.get("use_rep")
            random_state = int(tool_input.get("random_state", 0))

            compare_results = []
            figure_dir = tool_input.get("figure_dir")
            generate_figures = bool(tool_input.get("generate_figures", False))
            include_images = bool(tool_input.get("include_images", False))
            promote_resolution = tool_input.get("promote_resolution")
            image_payloads = []

            if figure_dir:
                os.makedirs(figure_dir, exist_ok=True)

            for resolution in resolutions:
                cluster_key = infer_cluster_key(method, resolution)
                result_payload = _apply_clustering(
                    adata,
                    method=method,
                    resolution=resolution,
                    cluster_key=cluster_key,
                    make_primary=False,
                    k=k,
                    use_rep=use_rep,
                    random_state=random_state,
                )
                compare_entry = {
                    "resolution": result_payload["resolution"],
                    "cluster_key": result_payload["cluster_key"],
                    "n_clusters": result_payload["n_clusters"],
                    "cluster_sizes": result_payload["cluster_sizes"],
                    "parameters": {
                        "k": int(k) if k is not None else None,
                        "use_rep": use_rep,
                        "random_state": random_state,
                    },
                }

                if generate_figures and "X_umap" in adata.obsm:
                    path_root = figure_dir or "."
                    figure_name = f"umap_{cluster_key}.png"
                    figure_path = os.path.join(path_root, figure_name)
                    figure_result = _render_figure(
                        adata,
                        plot_type="umap",
                        output_path=figure_path,
                        color_by=cluster_key,
                        include_image=include_images,
                    )
                    compare_entry["figure_path"] = figure_path
                    if include_images and "image_base64" in figure_result:
                        image_payloads.append({
                            "cluster_key": cluster_key,
                            "output_path": figure_path,
                            "image_base64": figure_result["image_base64"],
                            "image_mime": figure_result["image_mime"],
                        })

                compare_results.append(compare_entry)

            if promote_resolution is not None:
                promote_key = infer_cluster_key(method, float(promote_resolution))
                if promote_key not in adata.obs.columns:
                    raise ValueError(
                        f"Cannot promote resolution {promote_resolution}; "
                        f"expected clustering key '{promote_key}' was not generated."
                    )
                promote_clustering_to_primary(
                    adata,
                    cluster_key=promote_key,
                    method=method,
                    resolution=float(promote_resolution),
                    created_by="tool",
                )

            result = {
                "status": "ok",
                "tool": "compare_clusterings",
                "method": "phenograph" if str(method).lower() == "phenograph" else "leiden",
                "comparisons": compare_results,
                "available_clusterings": _clusterings_payload(adata),
                "warnings": warnings,
                "state": make_state(adata),
            }
            if image_payloads:
                first = image_payloads[0]
                result["image_base64"] = first["image_base64"]
                result["image_mime"] = first["image_mime"]
                result["image_context"] = {
                    "cluster_key": first["cluster_key"],
                    "output_path": first["output_path"],
                }
            artifacts_created = [
                artifact
                for artifact in (
                    _artifact_payload(
                        comparison.get("figure_path"),
                        role="comparison_figure",
                        metadata={"cluster_key": comparison.get("cluster_key")},
                    )
                    for comparison in compare_results
                    if comparison.get("figure_path")
                )
                if artifact is not None
            ]
            decisions = []
            if promote_resolution is None:
                clustering_decision = decision_for_clustering_selection(
                    compare_results,
                    source_tool="compare_clusterings",
                )
                if clustering_decision is not None:
                    decisions.append(clustering_decision)
            verification_checks = [
                _check(
                    "comparison_keys_created",
                    all(comparison["cluster_key"] in adata.obs.columns for comparison in compare_results),
                    "Every compared clustering key exists in adata.obs.",
                ),
            ]
            if artifacts_created:
                verification_checks.append(
                    _check(
                        "comparison_figures_exist",
                        all(os.path.exists(artifact["path"]) for artifact in artifacts_created),
                        "Generated comparison figures exist on disk.",
                    )
                )
            return _finalize_result(
                result,
                adata,
                dataset_changed=True,
                summary="Generated a safe multi-resolution clustering comparison without overwriting prior results.",
                artifacts_created=artifacts_created,
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "All requested clustering comparisons were preserved and verified.",
                    verification_checks,
                ),
            )

        elif tool_name == "run_celltypist":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            model = tool_input.get("model", "Immune_All_Low.pkl")
            majority = tool_input.get("majority_voting", True)
            cluster_key = tool_input.get("cluster_key", "leiden")
            organism = (tool_input.get("organism") or "").strip().lower()
            biological_context = tool_input.get("biological_context") or {}
            if not biological_context and world_state is not None:
                biological_context = (
                    (getattr(world_state, "data_summary", {}) or {}).get("biological_context", {})
                )
            if not organism and isinstance(biological_context, dict):
                organism = str(biological_context.get("species") or "").strip().lower()
            if organism not in {"human", "mouse"}:
                organism = ""
            allow_cross_species = bool(tool_input.get("allow_cross_species", False))

            model_info = infer_celltypist_model_organism(model)
            model_organism = model_info.get("organism")
            if not organism and model_organism in {"human", "mouse"}:
                return json.dumps({
                    "status": "needs_input",
                    "tool": "run_celltypist",
                    "reference_source": "celltypist",
                    "unavailable_reference_source": "celltypist",
                    "unavailable_reason": "organism_ambiguous",
                    "message": (
                        "CellTypist needs dataset organism before using "
                        f"model '{model}'. The model appears to be {model_organism}, "
                        "but the current species context is ambiguous."
                    ),
                    "required_input": "organism",
                    "options": ["human", "mouse"],
                    "model": model,
                    "model_info": model_info,
                    "biological_context": biological_context,
                    "model_discovery": {
                        "list_models_code": "celltypist.models.models_description()",
                        "download_model_code": "celltypist.models.download_models(model='<model>.pkl')",
                        "refresh_catalog_code": "celltypist.models.download_models(force_update=True)",
                    },
                    "recovery_options": [
                        "If the prompt or metadata says human, rerun with organism='human'.",
                        "If the prompt or metadata says mouse, rerun with organism='mouse'.",
                        "If species is genuinely unclear, inspect HLA vs H2 markers and ENSG vs ENSMUSG IDs first.",
                    ],
                }, indent=2), adata
            if (
                organism
                and model_organism in {"human", "mouse"}
                and organism != model_organism
                and not allow_cross_species
            ):
                compatible_models = available_celltypist_models_for_organism(organism)
                return json.dumps({
                    "status": "needs_input",
                    "tool": "run_celltypist",
                    "reference_source": "celltypist",
                    "unavailable_reference_source": "celltypist",
                    "unavailable_reason": "species_model_mismatch",
                    "message": (
                        f"Refusing to run CellTypist model '{model}' because the "
                        f"dataset organism is '{organism}', but the model appears "
                        f"to be '{model_organism}'."
                    ),
                    "required_input": "species_compatible_annotation_strategy",
                    "model": model,
                    "model_info": model_info,
                    "requested_organism": organism,
                    "model_organism": model_organism,
                    "compatible_celltypist_models": compatible_models[:20],
                    "biological_context": biological_context,
                    "model_discovery": {
                        "list_models_code": "celltypist.models.models_description()",
                        "download_model_code": "celltypist.models.download_models(model='<model>.pkl')",
                        "refresh_catalog_code": "celltypist.models.download_models(force_update=True)",
                        "official_models_url": "https://www.celltypist.org/models",
                    },
                    "recovery_options": [
                        f"Choose a {organism}-compatible CellTypist model if one matches the tissue.",
                        f"Run Scimilarity with organism='{organism}' and validate markers.",
                        "Use marker-based/manual annotation and report that CellTypist had no appropriate model.",
                        "Set allow_cross_species=true only for an explicitly caveated exploratory run.",
                    ],
                }, indent=2), adata
            if majority and cluster_key not in adata.obs.columns:
                return _smart_unavailable_result(
                    tool="run_celltypist",
                    message=(
                        f"CellTypist majority voting needs a valid clustering column, but '{cluster_key}' "
                        "is not available on the current in-memory dataset."
                    ),
                    adata_obj=adata,
                    missing_prerequisites=["clustering"],
                    recovery_options=[
                        "Run clustering first, then rerun CellTypist.",
                        "Choose one of the available clustering keys for annotation.",
                    ],
                    extra={
                        "reference_source": "celltypist",
                        "unavailable_reference_source": "celltypist",
                        "unavailable_reason": "missing_clustering",
                        "requested_cluster_key": cluster_key,
                    },
                )
            if majority:
                cluster_key = _validate_obs_column(
                    adata,
                    cluster_key,
                    warnings,
                    required=True,
                    context="cluster_key",
                )

            try:
                run_celltypist(
                    adata,
                    model=model,
                    organism=organism or None,
                    allow_cross_species=allow_cross_species,
                    majority_voting=majority,
                    over_clustering=cluster_key if majority else None,
                )
            except (ValueError, RuntimeError, ImportError) as e:
                # Most often: missing raw-counts layer when adata.X is already
                # log-normalized. Surface a recoverable error rather than
                # crashing the tool loop.
                message = str(e)
                lower_message = message.lower()
                if "raw integer counts" in lower_message or "raw-counts layer" in lower_message:
                    unavailable_reason = "raw_counts_missing"
                    recovery_options = [
                        "Ensure raw integer counts are in adata.layers['raw_counts'] "
                        "before running CellTypist (normalize_and_hvg preserves them "
                        "automatically; data loaded externally may not).",
                        "If you have raw counts under a different layer name, "
                        "pass it as raw_layer when invoking via run_code.",
                    ]
                elif "download failed" in lower_message:
                    unavailable_reason = "model_download_failed"
                    recovery_options = [
                        "Use list_celltypist_models to choose a model already cached locally.",
                        "Retry later if CellTypist model hosting or network access was unavailable.",
                        "Use Scimilarity and marker validation while recording CellTypist as unavailable.",
                    ]
                elif "model path does not exist" in lower_message:
                    unavailable_reason = "model_path_missing"
                    recovery_options = [
                        "Choose an existing explicit model path.",
                        "Use a named CellTypist catalog model instead of a path.",
                        "Use Scimilarity and marker validation while recording CellTypist as unavailable.",
                    ]
                elif isinstance(e, ImportError):
                    unavailable_reason = "package_missing"
                    recovery_options = [
                        "Install CellTypist in the active environment.",
                        "Use Scimilarity and marker validation while recording CellTypist as unavailable.",
                    ]
                else:
                    unavailable_reason = "model_runtime_error"
                    recovery_options = [
                        "Use check_celltypist_model to verify species compatibility and cache/download state.",
                        "Use list_celltypist_models to choose another compatible model.",
                        "Use Scimilarity and marker validation while recording CellTypist as unavailable.",
                    ]
                return _error_result(
                    tool="run_celltypist",
                    message=message,
                    adata_obj=adata,
                    recovery_options=recovery_options,
                    install_hint="pip install celltypist" if isinstance(e, ImportError) else None,
                    extra={
                        "reference_source": "celltypist",
                        "unavailable_reference_source": "celltypist",
                        "unavailable_reason": unavailable_reason,
                        "model": model,
                        "requested_organism": organism or None,
                        "model_organism": model_organism,
                        "model_info": model_info,
                    },
                )

            output_path = fix_output_path(tool_input.get("output_path"), "run_celltypist")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Get detailed type breakdown
            key = 'celltypist_majority_voting' if majority and 'celltypist_majority_voting' in adata.obs else 'celltypist_predicted_labels'
            all_counts = adata.obs[key].value_counts() if key in adata.obs else {}
            total_cells = adata.n_obs
            celltypist_meta = adata.uns.get("celltypist", {}) if hasattr(adata, "uns") else {}

            # Build detailed breakdown with counts and percentages
            type_breakdown = {}
            for ct, count in all_counts.items():
                type_breakdown[str(ct)] = {
                    "count": int(count),
                    "percent": round(100.0 * count / total_cells, 1)
                }
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)
            celltypist_result = {
                "status": "ok",
                "tool": "run_celltypist",
                "output_path": output_path,
                "saved": output_path is not None,
                "model": model,
                "requested_organism": celltypist_meta.get("requested_organism") or organism or None,
                "model_organism": celltypist_meta.get("model_organism") or model_organism,
                "model_organism_source": celltypist_meta.get("model_organism_source") or model_info.get("source"),
                "model_description": celltypist_meta.get("model_description") or model_info.get("description"),
                "model_cached": celltypist_meta.get("model_cached"),
                "model_cache_path": celltypist_meta.get("model_cache_path"),
                "allow_cross_species": allow_cross_species,
                "majority_voting": majority,
                "cluster_key_used": cluster_key if majority else None,
                "biological_context": biological_context,
                "model_discovery": {
                    "list_models_code": "celltypist.models.models_description()",
                    "download_model_code": "celltypist.models.download_models(model='<model>.pkl')",
                    "refresh_catalog_code": "celltypist.models.download_models(force_update=True)",
                    "official_models_url": "https://www.celltypist.org/models",
                },
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "cell_type_breakdown": type_breakdown,
                "warnings": warnings,
                "state": make_state(adata)
            }
            return _finalize_result(
                celltypist_result,
                adata,
                dataset_changed=True,
                summary="Ran CellTypist annotation and recorded the annotation source in session state.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "CellTypist annotations were written to AnnData.",
                    [
                        _check("annotation_key_present", key in adata.obs.columns, f"Annotation column '{key}' exists."),
                        _check(
                            "cluster_key_valid",
                            not majority or cluster_key in adata.obs.columns,
                            f"Cluster key '{cluster_key}' is valid for majority voting." if majority else "Majority voting was disabled.",
                        ),
                    ],
                ),
            )

        elif tool_name == "run_scimilarity":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            model_path = tool_input.get("model_path")
            organism = (tool_input.get("organism") or "").strip().lower()
            biological_context = tool_input.get("biological_context") or {}
            if not biological_context and world_state is not None:
                biological_context = (
                    (getattr(world_state, "data_summary", {}) or {}).get("biological_context", {})
                )
            if not organism and isinstance(biological_context, dict):
                organism = str(biological_context.get("species") or "").strip().lower()
            if organism not in {"human", "mouse"}:
                organism = ""
            cluster_key = tool_input.get("cluster_key", "leiden")
            cluster_key = _validate_obs_column(
                adata,
                cluster_key,
                warnings,
                required=False,
                context="cluster_key",
            ) or "leiden"

            if not model_path and not organism:
                return json.dumps({
                    "status": "needs_input",
                    "tool": "run_scimilarity",
                    "reference_source": "scimilarity",
                    "unavailable_reference_source": "scimilarity",
                    "unavailable_reason": "organism_ambiguous",
                    "message": (
                        "Scimilarity needs an explicit organism before annotation. "
                        "Human and mouse models are separate, and the current species context is ambiguous."
                    ),
                    "required_input": "organism",
                    "options": ["human", "mouse"],
                    "biological_context": biological_context,
                    "recovery_options": [
                        "If the prompt or metadata says the dataset is human, rerun with organism='human'.",
                        "If the prompt or metadata says the dataset is mouse, rerun with organism='mouse'.",
                        "If species is genuinely unclear, inspect gene IDs/markers first (HLA vs H2, ENSG vs ENSMUSG).",
                    ],
                }, indent=2), adata

            try:
                run_scimilarity(
                    adata,
                    model_path=model_path or None,
                    organism=organism or None,
                    cluster_key=cluster_key,
                )
            except ImportError as e:
                return _error_result(
                    tool="run_scimilarity",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Activate the scagent environment that includes scimilarity, then rerun run_scimilarity.",
                        "If this environment intentionally lacks scimilarity, continue with CellTypist plus DEG/PanglaoDB and report package_missing.",
                    ],
                    extra={
                        "reference_source": "scimilarity",
                        "unavailable_reference_source": "scimilarity",
                        "unavailable_reason": "package_missing",
                        "requested_organism": organism or None,
                        "model_path": model_path,
                    },
                )
            except FileNotFoundError as e:
                return _error_result(
                    tool="run_scimilarity",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Check SCIMILARITY_MODEL_PATH or SCIMILARITY_MODEL_PATH_MOUSE.",
                        "Rerun with an explicit model_path that points to the Scimilarity model directory.",
                    ],
                    extra={
                        "reference_source": "scimilarity",
                        "unavailable_reference_source": "scimilarity",
                        "unavailable_reason": "model_path_missing",
                        "requested_organism": organism or None,
                        "model_path": model_path,
                    },
                )
            except ValueError as e:
                message = str(e)
                reason = "organism_ambiguous" if "organism" in message.lower() or "species" in message.lower() else "model_runtime_error"
                return _error_result(
                    tool="run_scimilarity",
                    message=message,
                    adata_obj=adata,
                    recovery_options=[
                        "If the dataset species is known, rerun with organism='human' or organism='mouse'.",
                        "If the model path was explicit, confirm it matches the dataset organism and gene symbols.",
                    ],
                    extra={
                        "reference_source": "scimilarity",
                        "unavailable_reference_source": "scimilarity",
                        "unavailable_reason": reason,
                        "requested_organism": organism or None,
                        "model_path": model_path,
                    },
                )
            except Exception as e:
                message = str(e)
                lowered = message.lower()
                if any(token in lowered for token in ("model path", "not found", "no such file", "does not exist")):
                    reason = "model_path_missing"
                else:
                    reason = "model_runtime_error"
                return _error_result(
                    tool="run_scimilarity",
                    message=message,
                    adata_obj=adata,
                    recovery_options=[
                        "Review the Scimilarity traceback/message, organism, model path, and raw-count availability before deciding it is unavailable.",
                        "If the model path exists and the package imports, fix the runtime issue and rerun rather than finalizing without Scimilarity.",
                    ],
                    extra={
                        "reference_source": "scimilarity",
                        "unavailable_reference_source": "scimilarity",
                        "unavailable_reason": reason,
                        "requested_organism": organism or None,
                        "model_path": model_path,
                        "error_type": type(e).__name__,
                    },
                )
            scimilarity_meta = adata.uns.get("scimilarity", {}) if hasattr(adata, "uns") else {}

            output_path = fix_output_path(tool_input.get("output_path"), "run_scimilarity")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Get detailed type breakdown
            key = 'scimilarity_predictions_unconstrained'
            if key not in adata.obs:
                key = 'scimilarity_representative_prediction'

            all_counts = adata.obs[key].value_counts() if key in adata.obs else {}
            total_cells = adata.n_obs

            # Build detailed breakdown with counts and percentages
            type_breakdown = {}
            for ct, count in all_counts.items():
                type_breakdown[str(ct)] = {
                    "count": int(count),
                    "percent": round(100.0 * count / total_cells, 1)
                }
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)
            scimilarity_result = {
                "status": "ok",
                "tool": "run_scimilarity",
                "output_path": output_path,
                "saved": output_path is not None,
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "cluster_key_used": cluster_key,
                "has_embeddings": "X_scimilarity" in adata.obsm,
                "requested_organism": scimilarity_meta.get("requested_organism") or organism or None,
                "selected_organism": scimilarity_meta.get("selected_organism"),
                "model_path": scimilarity_meta.get("model_path") or model_path,
                "model_source": "explicit_model_path" if model_path else "organism_default",
                "biological_context": biological_context,
                "cell_type_breakdown": type_breakdown,
                "warnings": warnings,
                "state": make_state(adata)
            }
            return _finalize_result(
                scimilarity_result,
                adata,
                dataset_changed=True,
                summary="Ran Scimilarity annotation and recorded the representative predictions.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "Scimilarity outputs were written to AnnData.",
                    [
                        _check("annotation_key_present", key in adata.obs.columns, f"Annotation column '{key}' exists."),
                        _check(
                            "organism_selected",
                            scimilarity_meta.get("selected_organism") in {"human", "mouse"} or bool(model_path),
                            f"Scimilarity model organism: {scimilarity_meta.get('selected_organism') or 'explicit model path'}.",
                        ),
                        _check(
                            "cluster_key_valid",
                            cluster_key in adata.obs.columns,
                            f"Cluster key '{cluster_key}' is available for cluster-level summaries.",
                        ),
                    ],
                ),
            )

        elif tool_name == "query_cells":
            from ..annotation.scimilarity import query_cells as _query_cells

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            if adata is None:
                return _error_result(
                    tool="query_cells",
                    message="No data loaded. Load a dataset first.",
                    recovery_options=["Load data with inspect_data or provide a data_path."],
                )

            query_type = tool_input.get("query_type", "cells")
            k = tool_input.get("k", 50)
            raw_layer = tool_input.get("raw_layer")

            # Validate mode-specific inputs early for a clear error
            if query_type == "centroid":
                if not tool_input.get("group_key") or not tool_input.get("group_value"):
                    return _error_result(
                        tool="query_cells",
                        message="centroid mode requires group_key and group_value.",
                        adata_obj=adata,
                        recovery_options=[
                            "Provide both group_key (obs column) and group_value (category within it).",
                            "Use list_obs_columns to find available grouping columns.",
                        ],
                    )
            else:
                if not tool_input.get("cell_ids") and not tool_input.get("obs_column"):
                    return _error_result(
                        tool="query_cells",
                        message="cells mode requires either cell_ids (list of obs_names) or obs_column.",
                        adata_obj=adata,
                        recovery_options=[
                            "Provide a list of cell_ids (obs_names) or an obs_column to select cells from.",
                        ],
                    )

            try:
                result = _query_cells(
                    adata,
                    query_type=query_type,
                    cell_ids=tool_input.get("cell_ids"),
                    obs_column=tool_input.get("obs_column"),
                    group_key=tool_input.get("group_key"),
                    group_value=tool_input.get("group_value"),
                    k=k,
                    model_path=tool_input.get("model_path") or None,
                    organism=tool_input.get("organism") or None,
                    raw_layer=raw_layer,
                )
            except (FileNotFoundError, ImportError) as e:
                return _error_result(
                    tool="query_cells",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify the SCimilarity model path exists (see download_model.sh).",
                    ],
                    install_hint="pip install scimilarity" if "Import" in type(e).__name__ else None,
                )
            except Exception as e:
                return _error_result(
                    tool="query_cells",
                    message=f"Cell query failed: {e}",
                    adata_obj=adata,
                    recovery_options=[
                        "Verify adata has raw counts (raw_layer) and correct obs structure.",
                    ],
                )

            # Build a human-readable summary
            top_ct = result.get("top_celltypes", {})
            top_tissue = result.get("top_tissues", {})
            top_disease = result.get("top_diseases", {})
            coherence = result.get("coherence")
            mean_dist = result.get("mean_dist")

            summary_parts = [
                f"Retrieved {result['n_results']} reference cells (k={k}).",
                f"Mean distance: {mean_dist:.4f}." if mean_dist is not None else "",
                f"Query coherence: {coherence}%." if coherence is not None else "",
                "Top cell types: " + ", ".join(f"{ct} ({n})" for ct, n in list(top_ct.items())[:5]) + "." if top_ct else "",
                "Top tissues: " + ", ".join(f"{t} ({n})" for t, n in list(top_tissue.items())[:5]) + "." if top_tissue else "",
            ]
            summary = " ".join(p for p in summary_parts if p)

            return json.dumps({
                "status": "ok",
                "tool": "query_cells",
                **result,
                "message": summary,
            }, indent=2), adata

        elif tool_name == "score_gene_signature":
            import numpy as np
            import scanpy as sc
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            if adata is None:
                return _error_result(
                    tool="score_gene_signature",
                    message="No data loaded. Load a dataset first.",
                    recovery_options=["Load data with inspect_data or provide a data_path."],
                )

            from ..core.inspector import inspect_data as _inspect_for_norm
            _norm_state = _inspect_for_norm(adata)
            if not _norm_state.is_normalized:
                return _error_result(
                    tool="score_gene_signature",
                    message=(
                        "Data does not appear to be normalized. score_gene_signature works on "
                        "log-normalized expression values (adata.X), not raw counts. "
                        "Run normalize_and_hvg first."
                    ),
                    adata_obj=adata,
                    recovery_options=["Run normalize_and_hvg before scoring gene signatures."],
                )

            cell_cycle = tool_input.get("cell_cycle", False)

            if cell_cycle:
                # ---- Cell cycle scoring ----
                s_genes = tool_input.get("s_genes") or []
                g2m_genes = tool_input.get("g2m_genes") or []
                if not s_genes or not g2m_genes:
                    return _error_result(
                        tool="score_gene_signature",
                        message=(
                            "cell_cycle=true requires both s_genes and g2m_genes. "
                            "Provide lists of S-phase and G2M-phase marker genes."
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "Provide both s_genes and g2m_genes lists.",
                            "Use standard Tirosh et al. 2016 cell cycle gene sets (search_papers can find them).",
                        ],
                    )

                # Filter to genes present in the dataset
                var_names = set(adata.var_names)
                s_found = [g for g in s_genes if g in var_names]
                g2m_found = [g for g in g2m_genes if g in var_names]

                if not s_found or not g2m_found:
                    return _error_result(
                        tool="score_gene_signature",
                        message=(
                            f"Cell cycle scoring failed: found {len(s_found)}/{len(s_genes)} S-phase genes "
                            f"and {len(g2m_found)}/{len(g2m_genes)} G2M-phase genes in the dataset. "
                            "Need at least one gene per phase. Check gene name format (human HGNC symbols)."
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "Verify gene names match the dataset format (inspect sample var_names with inspect_data).",
                            "Genes may be in mouse format (e.g., Ccnb1) vs. human (CCNB1) — convert if needed.",
                        ],
                    )

                try:
                    sc.tl.score_genes_cell_cycle(
                        adata,
                        s_genes=s_found,
                        g2m_genes=g2m_found,
                    )
                except Exception as e:
                    return _error_result(
                        tool="score_gene_signature",
                        message=f"Cell cycle scoring failed: {e}",
                        adata_obj=adata,
                        recovery_options=["Check gene name formats and ensure data is log-normalized."],
                    )

                phase_counts = adata.obs["phase"].value_counts().to_dict()
                return json.dumps({
                    "status": "ok",
                    "tool": "score_gene_signature",
                    "mode": "cell_cycle",
                    "s_genes_matched": len(s_found),
                    "s_genes_total": len(s_genes),
                    "g2m_genes_matched": len(g2m_found),
                    "g2m_genes_total": len(g2m_genes),
                    "scores_added": ["S_score", "G2M_score", "phase"],
                    "phase_distribution": phase_counts,
                    "message": (
                        f"Cell cycle scoring complete. Phase distribution: "
                        + ", ".join(f"{k}: {v}" for k, v in sorted(phase_counts.items()))
                        + ". Scores stored in adata.obs['S_score'], ['G2M_score'], ['phase']."
                    ),
                }, indent=2), adata

            else:
                # ---- Generic gene signature scoring ----
                gene_list = tool_input.get("gene_list") or []
                if not gene_list:
                    return _error_result(
                        tool="score_gene_signature",
                        message="Provide gene_list (list of gene names) or set cell_cycle=true.",
                        adata_obj=adata,
                        recovery_options=[
                            "Provide a gene_list of marker genes to score.",
                            "Set cell_cycle=true with s_genes and g2m_genes for cell cycle scoring.",
                        ],
                    )

                score_name = tool_input.get("score_name", "gene_signature_score")
                layer = tool_input.get("layer")
                n_bins = tool_input.get("n_bins", 25)
                ctrl_size = tool_input.get("ctrl_size", 50)

                # Validate layer if provided
                if layer and layer not in adata.layers:
                    return _error_result(
                        tool="score_gene_signature",
                        message=(
                            f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}. "
                            "Leave layer unset to use adata.X (recommended)."
                        ),
                        adata_obj=adata,
                        recovery_options=["Use one of the available layers or omit layer to use adata.X."],
                    )

                # Filter gene_list to genes present in the dataset and report coverage
                var_names = set(adata.var_names)
                matched = [g for g in gene_list if g in var_names]
                missing = [g for g in gene_list if g not in var_names]

                if not matched:
                    return _error_result(
                        tool="score_gene_signature",
                        message=(
                            f"None of the {len(gene_list)} provided genes were found in the dataset. "
                            f"Check gene name format — dataset uses: {list(adata.var_names[:5])}..."
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "Verify gene name format matches the dataset (human HGNC vs. mouse, Ensembl IDs vs. symbols).",
                            "Use inspect_data to see example var_names.",
                        ],
                        extra={"genes_not_found": gene_list[:20]},
                    )

                # Warn if coverage is low but still proceed
                coverage_pct = len(matched) / len(gene_list) * 100

                try:
                    sc.tl.score_genes(
                        adata,
                        gene_list=matched,
                        score_name=score_name,
                        n_bins=n_bins,
                        ctrl_size=ctrl_size,
                        layer=layer,
                    )
                except Exception as e:
                    return _error_result(
                        tool="score_gene_signature",
                        message=f"Gene scoring failed: {e}",
                        adata_obj=adata,
                        recovery_options=["Verify gene list and layer contain valid expression data."],
                    )

                scores = adata.obs[score_name]
                return json.dumps({
                    "status": "ok",
                    "tool": "score_gene_signature",
                    "mode": "signature",
                    "score_name": score_name,
                    "genes_requested": len(gene_list),
                    "genes_matched": len(matched),
                    "genes_missing": len(missing),
                    "coverage_pct": round(coverage_pct, 1),
                    "missing_genes": missing[:20] if missing else [],
                    "score_stats": {
                        "mean": round(float(scores.mean()), 4),
                        "std": round(float(scores.std()), 4),
                        "min": round(float(scores.min()), 4),
                        "max": round(float(scores.max()), 4),
                        "pct_positive": round(float((scores > 0).mean() * 100), 1),
                    },
                    "message": (
                        f"Scored {len(matched)}/{len(gene_list)} genes ({coverage_pct:.0f}% coverage). "
                        f"Score stored in adata.obs['{score_name}']. "
                        f"Mean score: {scores.mean():.4f}, "
                        f"{(scores > 0).mean() * 100:.1f}% of cells are positive."
                        + (f" Warning: {len(missing)} genes not found in dataset." if missing else "")
                    ),
                }, indent=2), adata

        elif tool_name == "run_spectra":
            from ..analysis.spectra import run_spectra

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)

            cell_type_key = tool_input.get("cell_type_key")
            if not cell_type_key or cell_type_key not in adata.obs.columns:
                return _error_result(
                    tool="run_spectra",
                    message=f"cell_type_key '{cell_type_key}' not found in adata.obs.",
                    adata_obj=adata,
                    recovery_options=[
                        "Run cell type annotation (run_celltypist or run_scimilarity) first.",
                        "Use list_obs_columns to find the correct annotation column.",
                    ],
                )

            output_dir = fix_output_path(tool_input.get("output_dir"), "run_spectra")

            try:
                result = run_spectra(
                    adata,
                    cell_type_key=cell_type_key,
                    gene_set_dict_path=tool_input.get("gene_set_dict_path"),
                    use_default_gene_sets=bool(tool_input.get("use_default_gene_sets", False)),
                    lam=float(tool_input.get("lam", 0.1)),
                    rho=0.001,
                    num_epochs=int(tool_input.get("num_epochs", 1000)),
                    n_top_vals=int(tool_input.get("n_top_vals", 50)),
                    use_highly_variable=bool(tool_input.get("use_highly_variable", True)),
                    use_weights=True,
                    use_cell_types=bool(tool_input.get("use_cell_types", True)),
                    label_factors=True,
                    overlap_threshold=float(tool_input.get("overlap_threshold", 0.2)),
                    output_dir=output_dir,
                )
            except ImportError as e:
                return _error_result(
                    tool="run_spectra",
                    message=str(e),
                    adata_obj=adata,
                    install_hint="pip install Spectra-sc",
                )
            except Exception as e:
                return _error_result(
                    tool="run_spectra",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify cell_type_key is valid and gene set paths are correct.",
                    ],
                )

            artifacts_created = []
            if result.get("figure_path"):
                artifact = _artifact_payload(
                    result["figure_path"],
                    role="figure",
                    metadata={"kind": "spectra_factor_umaps"},
                )
                if artifact:
                    artifacts_created.append(artifact)

            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "run_spectra",
                    "cell_type_key": cell_type_key,
                    "n_factors": result["n_factors"],
                    "factor_labels": result["factor_labels"],
                    "top_markers_per_factor": result["top_markers_per_factor"],
                    "model_path": result.get("model_path"),
                    "figure_path": result.get("figure_path"),
                    "obsm_key": "SPECTRA_cell_scores",
                    "note": (
                        "Factor scores in adata.obsm['SPECTRA_cell_scores'] — "
                        "color UMAP by individual columns to visualize each gene program. "
                        "Top marker genes per factor are in adata.uns['SPECTRA_markers']."
                    ),
                    "warnings": warnings,
                    "state": make_state(adata),
                },
                adata,
                dataset_changed=True,
                summary=f"Spectra discovered {result['n_factors']} gene program factors using '{cell_type_key}' as cell type key.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "Spectra completed successfully.",
                    [
                        _check(
                            "cell_scores_present",
                            "SPECTRA_cell_scores" in adata.obsm,
                            f"Factor scores written to adata.obsm['SPECTRA_cell_scores'] ({result['n_factors']} factors).",
                        ),
                        _check(
                            "markers_present",
                            "SPECTRA_markers" in adata.uns,
                            "Top marker genes stored in adata.uns['SPECTRA_markers'].",
                        ),
                    ],
                ),
            )

        elif tool_name == "save_data":
            if adata is None:
                return _error_result(
                    tool="save_data",
                    message="No in-memory data available to save. Run an analysis tool first.",
                    recovery_options=["Load and process data before saving."],
                )

            output_path = fix_output_path(tool_input.get("output_path"), "save_data")
            if not output_path:
                return _error_result(
                    tool="save_data",
                    message="Provide an .h5ad output_path or a directory where the final_result.h5ad can be written.",
                    adata_obj=adata,
                    recovery_options=["Provide output_path as a .h5ad file path or directory."],
                )

            save_details = write_h5ad_safe(adata, output_path)
            artifact = _artifact_payload(output_path, role="saved_dataset", metadata={"save_mode": save_details.get("save_mode", "direct")})
            save_result = {
                "status": "ok",
                "tool": "save_data",
                "output_path": output_path,
                "saved": True,
                "save_mode": save_details.get("save_mode", "direct"),
                "warnings": save_details.get("warnings", []),
                "shape": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "state": make_state(adata)
            }
            return _finalize_result(
                save_result,
                adata,
                dataset_changed=False,
                summary="Saved the current in-memory AnnData as a final dataset artifact.",
                artifacts_created=[artifact] if artifact is not None else [],
                verification=_build_verification(
                    "passed",
                    "The AnnData output file exists and was registered as an artifact.",
                    [
                        _check("output_exists", os.path.exists(output_path), f"Saved output exists at {output_path}."),
                    ],
                ),
            )

        elif tool_name == "run_batch_correction":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "harmony")
            requested_batch_key = tool_input.get("batch_key") or _confirmed_decision_value("batch_key")
            if not requested_batch_key:
                raise ValueError(
                    "No batch_key was provided and no confirmed batch_key exists in session state. "
                    "Confirm the correct batch column first."
                )
            batch_key = _validate_obs_column(adata, requested_batch_key, warnings, required=True, context="batch_key")

            # Get batch sizes for output
            batch_sizes = adata.obs[batch_key].value_counts().to_dict()
            if method == "harmony":
                run_harmony(adata, batch_key=batch_key)
                corrected_rep = "X_pca_harmony"
            elif method == "bbknn":
                n_pcs = int(tool_input.get("n_pcs") or 30)
                neighbors_within_batch = int(tool_input.get("neighbors_within_batch") or 3)
                # BBKNN requires PCA — validate before running
                if "X_pca" not in adata.obsm:
                    return _error_result(
                        tool="run_batch_correction",
                        message="BBKNN requires PCA in adata.obsm['X_pca']. Run run_pca first.",
                        adata_obj=adata,
                        recovery_options=["Run run_pca to compute PCA, then retry BBKNN."],
                        extra={"method": "bbknn"},
                    )
                run_bbknn(
                    adata,
                    batch_key=batch_key,
                    n_pcs=n_pcs,
                    neighbors_within_batch=neighbors_within_batch,
                )
                # BBKNN correction lives in the neighbor graph, not an obsm key.
                # corrected_rep=None signals downstream logic to use the BBKNN graph as-is.
                corrected_rep = None
            elif method == "scvi":
                n_latent = int(tool_input.get("n_latent") or 30)
                max_epochs = int(tool_input.get("max_epochs") or 200)
                store_normalized = bool(tool_input.get("store_normalized", False))
                # scVI requires raw integer counts — validate before training
                try:
                    _resolve_integer_counts_layer(adata, "raw_counts")
                except ValueError as e:
                    return _error_result(
                        tool="run_batch_correction",
                        message=str(e),
                        adata_obj=adata,
                        recovery_options=[
                            "Ensure raw integer counts are in adata.layers['raw_counts'] "
                            "(normalize_and_hvg preserves them automatically).",
                            "Try method='harmony' or 'bbknn' instead — they work on PCA embeddings and do not need raw counts.",
                        ],
                        extra={"method": "scvi"},
                    )
                run_scvi(
                    adata,
                    batch_key=batch_key,
                    n_latent=n_latent,
                    max_epochs=max_epochs,
                    store_normalized=store_normalized,
                )
                corrected_rep = "X_scVI"
            else:
                run_scanorama(adata, batch_key=batch_key)
                corrected_rep = "X_scanorama"

            neighbors_recomputed = method == "bbknn"
            umap_recomputed = False

            output_path = fix_output_path(tool_input.get("output_path"), "run_batch_correction")
            if output_path:
                write_h5ad_safe(adata, output_path)
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)
            batch_strategy = {
                "status": "user_selected" if tool_input.get("batch_key") else "auto_selected",
                "requested_column": tool_input.get("batch_key"),
                "applied_column": batch_key,
                "recommended_column": batch_key,
                "recommended_role": "batch",
                "needs_user_confirmation": False,
                "reason": f"Using '{batch_key}' for batch correction.",
                "candidates": [{"column": batch_key}],
            }
            decisions = []
            batch_decision = decision_for_batch_strategy(
                batch_strategy,
                context="batch_correction",
                source_tool="run_batch_correction",
            )
            if batch_decision is not None:
                decisions.append(batch_decision)
            extra = {}
            if method == "scvi":
                extra["n_latent"] = int(tool_input.get("n_latent") or 30)
                extra["max_epochs"] = int(tool_input.get("max_epochs") or 200)
                extra["scvi_normalized_stored"] = bool(tool_input.get("store_normalized", False))
            elif method == "bbknn":
                extra["n_pcs"] = int(tool_input.get("n_pcs") or 30)
                extra["neighbors_within_batch"] = int(tool_input.get("neighbors_within_batch") or 3)
                extra["total_neighbors_per_cell"] = len(batch_sizes) * extra["neighbors_within_batch"]

            # BBKNN correction lives in the neighbor graph (adata.obsp), not an obsm key
            corrected_embedding_label = (
                "BBKNN graph (adata.obsp['connectivities'])"
                if corrected_rep is None
                else corrected_rep
            )
            batch_result = {
                "status": "ok",
                "tool": "run_batch_correction",
                "output_path": output_path,
                "saved": output_path is not None,
                "method": method,
                "batch_key": batch_key,
                "n_batches": len(batch_sizes),
                "batch_sizes": {str(k): int(v) for k, v in batch_sizes.items()},
                "corrected_embedding": corrected_embedding_label,
                "neighbors_recomputed": neighbors_recomputed,
                "umap_recomputed": False,
                "note": "Batch correction complete. Run run_umap next (and run_neighbors first for Harmony/Scanorama/scVI).",
                "warnings": warnings,
                "state": make_state(adata),
                **extra,
            }

            # For BBKNN the correction is in the neighbor graph, not an obsm key
            if corrected_rep is None:
                corrected_present = adata.uns.get("bbknn_batch_key") == batch_key
                corrected_check_label = "BBKNN neighbor graph stored in adata.obsp['connectivities']."
            else:
                corrected_present = corrected_rep in adata.obsm
                corrected_check_label = f"Corrected embedding '{corrected_rep}' exists in adata.obsm."

            return _finalize_result(
                batch_result,
                adata,
                dataset_changed=True,
                summary=(
                    f"Applied {method} batch correction using '{batch_key}'. "
                    f"Downstream UMAP recomputed: {umap_recomputed}."
                ),
                artifacts_created=artifacts_created,
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "Batch correction completed and the corrected representation is available.",
                    [
                        _check("batch_key_present", batch_key in adata.obs.columns, f"Batch key '{batch_key}' exists in adata.obs."),
                        _check("corrected_embedding_present", corrected_present, corrected_check_label),
                        _check("umap_not_recomputed", not umap_recomputed, "UMAP was not recomputed — run run_umap to refresh the layout."),
                    ],
                ),
            )

        elif tool_name == "score_integration":
            from ..batch.entropy import compute_batch_entropy

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            batch_key = tool_input.get("batch_key")
            use_rep = tool_input.get("use_rep", "X_umap")
            n_neighbors = int(tool_input.get("n_neighbors", 50))

            if not batch_key or batch_key not in adata.obs.columns:
                return _error_result(
                    tool="score_integration",
                    message=f"batch_key '{batch_key}' not found in adata.obs.",
                    adata_obj=adata,
                    recovery_options=[
                        "Use list_obs_columns to find the correct batch column name.",
                        "Confirm the batch column with the user before scoring.",
                    ],
                )

            try:
                result = compute_batch_entropy(
                    adata,
                    batch_key=batch_key,
                    use_rep=use_rep,
                    n_neighbors=n_neighbors,
                )
            except (ValueError, ImportError) as e:
                return _error_result(
                    tool="score_integration",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify batch_key has ≥2 unique values and use_rep embedding exists in adata.obsm.",
                    ],
                )

            # Store per-cell entropy in obs for downstream visualization
            adata.obs["integration_entropy"] = result["per_cell_entropy"]

            entropy_mean = result["entropy_mean"]
            if entropy_mean >= 0.8:
                interpretation = "Excellent mixing — batches are well-integrated."
            elif entropy_mean >= 0.6:
                interpretation = "Good mixing — minor batch structure may remain."
            elif entropy_mean >= 0.4:
                interpretation = "Moderate mixing — consider a stronger correction method (e.g. scVI)."
            else:
                interpretation = "Poor mixing — strong batch structure persists. Try scVI or check batch_key."

            return json.dumps({
                "status": "ok",
                "tool": "score_integration",
                "use_rep": use_rep,
                "batch_key": batch_key,
                "n_neighbors": result["n_neighbors"],
                "n_batches": result["n_batches"],
                "entropy_mean": round(entropy_mean, 4),
                "entropy_median": round(result["entropy_median"], 4),
                "entropy_per_batch": {k: round(v, 4) for k, v in result["entropy_per_batch"].items()},
                "interpretation": interpretation,
                "note": (
                    "Per-cell scores stored in adata.obs['integration_entropy']. "
                    "Color UMAP by 'integration_entropy' to see spatial mixing patterns."
                ),
                "state": make_state(adata),
            }, indent=2), adata

        elif tool_name == "benchmark_integration":
            from ..batch.scib import run_scib_benchmark

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            batch_key = tool_input.get("batch_key")
            label_key = tool_input.get("label_key")
            embedding_keys = tool_input.get("embedding_keys") or None
            fast = bool(tool_input.get("fast", False))
            output_dir = fix_output_path(tool_input.get("output_dir"), "benchmark_integration")

            for col, name in [(batch_key, "batch_key"), (label_key, "label_key")]:
                if not col or col not in adata.obs.columns:
                    return _error_result(
                        tool="benchmark_integration",
                        message=f"{name} '{col}' not found in adata.obs.",
                        adata_obj=adata,
                        recovery_options=[
                            "Use list_obs_columns to find valid batch and label columns.",
                            "Run cell type annotation first if label_key is missing.",
                        ],
                    )

            try:
                bench = run_scib_benchmark(
                    adata,
                    batch_key=batch_key,
                    label_key=label_key,
                    embedding_keys=embedding_keys,
                    fast=fast,
                    output_dir=output_dir,
                )
            except ImportError as e:
                return _error_result(
                    tool="benchmark_integration",
                    message=str(e),
                    adata_obj=adata,
                    install_hint="pip install scib-metrics",
                )
            except Exception as e:
                return _error_result(
                    tool="benchmark_integration",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify batch/label keys and that embeddings are properly formatted.",
                    ],
                )

            artifacts_created = []
            if bench.get("output_figure"):
                artifact = _artifact_payload(
                    bench["output_figure"],
                    role="figure",
                    metadata={"kind": "scib_benchmark_table"},
                )
                if artifact:
                    artifacts_created.append(artifact)
            if bench.get("output_csv"):
                artifact = _artifact_payload(
                    bench["output_csv"],
                    role="artifact",
                    metadata={"kind": "scib_results_csv"},
                )
                if artifact:
                    artifacts_created.append(artifact)

            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "benchmark_integration",
                    "batch_key": batch_key,
                    "label_key": label_key,
                    "embeddings_benchmarked": bench["embeddings_benchmarked"],
                    "scores_by_embedding": bench["scores_by_embedding"],
                    "best_method": bench["best_method"],
                    "results_table": bench["results_table"],
                    "output_csv": bench.get("output_csv"),
                    "output_figure": bench.get("output_figure"),
                    "note": (
                        f"Best embedding by total scib score: {bench['best_method']}. "
                        "Use run_batch_correction with the corresponding method if not already applied."
                    ) if bench["best_method"] else "",
                    "state": make_state(adata),
                },
                adata,
                dataset_changed=False,
                summary=f"scib-metrics benchmark complete across {len(bench['embeddings_benchmarked'])} embeddings. Best: {bench['best_method']}.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "scib-metrics benchmark completed successfully.",
                    [
                        _check(
                            "embeddings_benchmarked",
                            len(bench["embeddings_benchmarked"]) > 0,
                            f"Benchmarked {len(bench['embeddings_benchmarked'])} embeddings.",
                        ),
                        _check(
                            "best_method_identified",
                            bench["best_method"] is not None,
                            f"Best method: {bench['best_method']}.",
                        ),
                    ],
                ),
            )

        elif tool_name == "run_deg":
            from ..analysis.deg import run_validated_deg, get_deg_caveats
            from ..config.defaults import DEG_DEFAULTS

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            requested_groupby = tool_input.get("groupby", "leiden")
            if requested_groupby not in adata.obs.columns:
                return _smart_unavailable_result(
                    tool="run_deg",
                    message=(
                        f"Differential expression needs a valid grouping column, but '{requested_groupby}' "
                        "is not available on the current in-memory dataset."
                    ),
                    adata_obj=adata,
                    missing_prerequisites=["grouping"],
                    recovery_options=[
                        "Run clustering first, then run DEG on the cluster key.",
                        "Use one of the available annotation or metadata columns as the DEG grouping.",
                    ],
                    extra={"requested_groupby": requested_groupby},
                )
            groupby = _validate_obs_column(
                adata,
                requested_groupby,
                warnings,
                required=True,
                context="groupby"
            )
            method = tool_input.get("method", "wilcoxon")
            layer = tool_input.get("layer")
            use_raw = tool_input.get("use_raw")
            key_added = tool_input.get("key_added", "rank_genes_groups")
            n_genes = int(tool_input.get("n_genes", 100))
            target_geneset = tool_input.get("target_geneset", DEG_DEFAULTS.default_geneset)

            # Run validated DEG - this validates inputs, runs rank_genes_groups,
            # validates outputs, and attaches validity metadata to adata.uns
            try:
                _, validity_report = run_validated_deg(
                    adata,
                    groupby=groupby,
                    method=method,
                    n_genes=n_genes,
                    layer=layer,
                    use_raw=use_raw,
                    key_added=key_added,
                    target_geneset=target_geneset,
                    min_cluster_size=DEG_DEFAULTS.min_cluster_size,
                    warn_cluster_size=DEG_DEFAULTS.warn_cluster_size,
                    imbalance_ratio=DEG_DEFAULTS.max_imbalance_ratio,
                    block_on_errors=True,
                    batch_confound_threshold=DEG_DEFAULTS.batch_confound_threshold,
                    max_logfc=DEG_DEFAULTS.max_logfc_sanity,
                    inplace=True,
                )
            except Exception as e:
                return _error_result(
                    tool="run_deg",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify groupby column is valid and has ≥2 groups.",
                        "If a layer was specified, ensure it exists and contains valid expression data.",
                    ],
                )

            output_path = fix_output_path(tool_input.get("output_path"), "run_deg")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Get top 5 markers per cluster for immediate insight
            groups = list(adata.obs[groupby].unique())
            top_markers_summary = {}
            for group in groups[:15]:  # Limit to first 15 clusters for response size
                try:
                    markers_df = get_top_markers(adata, group=str(group), n_genes=5, key=key_added)
                    top_markers_summary[str(group)] = [
                        {
                            "gene": row['names'],
                            "logfc": round(row['logfoldchanges'], 2),
                            "pval_adj": float(f"{row['pvals_adj']:.2e}")
                        }
                        for _, row in markers_df.iterrows()
                    ]
                except Exception:
                    pass

            # Build validation summary for response
            validity_summary = {
                "is_valid": validity_report.is_valid,
                "has_warnings": validity_report.has_warnings,
                "n_errors": len(validity_report.errors),
                "n_warnings": len(validity_report.warnings),
                "matrix_type": validity_report.matrix_type,
                "matrix_source": validity_report.matrix_source,
                "use_raw": validity_report.use_raw,
                "layer_used": validity_report.layer_used,
                "data_species": validity_report.data_species,
                "gene_id_format": validity_report.gene_id_format,
            }

            # Include specific issues for agent awareness
            if validity_report.errors:
                validity_summary["errors"] = [e.message for e in validity_report.errors]
            if validity_report.warnings:
                validity_summary["warnings"] = [w.message for w in validity_report.warnings]

            # Get caveats that should propagate to GSEA
            deg_caveats = get_deg_caveats(adata)

            return json.dumps({
                "status": "ok",
                "tool": "run_deg",
                "output_path": output_path,
                "saved": output_path is not None,
                "groupby": groupby,
                "method": method,
                "key_added": key_added,
                "n_genes": n_genes,
                "n_groups": len(groups),
                "requested_layer": layer,
                "requested_use_raw": use_raw,
                "layer_used": validity_report.layer_used,
                "use_raw": validity_report.use_raw,
                "matrix_source": validity_report.matrix_source,
                "matrix_type": validity_report.matrix_type,
                "cluster_sizes": validity_report.cluster_sizes,
                "validity": validity_summary,
                "caveats_for_gsea": deg_caveats,
                "top_markers_per_cluster": top_markers_summary,
                "note": "Validity metadata stored in adata.uns['deg_validity'] - will propagate to GSEA",
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_pseudobulk_deg":
            from ..analysis.pseudobulk import run_pseudobulk_deg

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)

            sample_col = tool_input["sample_col"]
            condition_col = tool_input["condition_col"]
            condition_a = tool_input["condition_a"]
            condition_b = tool_input["condition_b"]
            groups_col = tool_input["groups_col"]
            cell_type = tool_input.get("cell_type")
            layer = tool_input.get("layer", "raw_counts")
            min_cells = int(tool_input.get("min_cells") or 10)
            alpha = float(tool_input.get("alpha") or 0.05)
            output_path = tool_input.get("output_path")

            # Validate integer counts before aggregating
            try:
                _resolve_integer_counts_layer(adata, layer)
            except ValueError as e:
                return _error_result(
                    tool="run_pseudobulk_deg",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Ensure raw integer counts are in the specified layer (normalize_and_hvg preserves them).",
                    ],
                )

            try:
                result = run_pseudobulk_deg(
                    adata,
                    sample_col=sample_col,
                    condition_col=condition_col,
                    condition_a=condition_a,
                    condition_b=condition_b,
                    groups_col=groups_col,
                    cell_type=cell_type,
                    layer=layer,
                    min_cells=min_cells,
                    alpha=alpha,
                    output_path=output_path,
                )
            except (ImportError, ValueError) as e:
                return _error_result(
                    tool="run_pseudobulk_deg",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify sample_col, condition_col, and groups_col exist and are valid.",
                        "Ensure ≥2 samples per condition for DESeq2 replication.",
                    ],
                    install_hint="pip install decoupler pydeseq2" if "Import" in type(e).__name__ else None,
                )

            result["warnings"] = warnings
            result["state"] = make_state(adata)
            return json.dumps(result, indent=2), adata

        elif tool_name == "run_gsea":
            import scanpy as sc
            from ..analysis.deg import get_deg_validity, get_cluster_caveats

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            output_dir = tool_input["output_dir"]
            cluster = tool_input["cluster"]
            gene_sets = tool_input.get("gene_sets", "KEGG_2021_Human")
            min_size = tool_input.get("min_size", 5)
            max_size = tool_input.get("max_size", 500)
            permutation_num = tool_input.get("permutation_num", 1000)

            # Check DEG results exist
            if 'rank_genes_groups' not in adata.uns:
                return _error_result(
                    tool="run_gsea",
                    message="No DEG results found. Run run_deg first.",
                    adata_obj=adata,
                    recovery_options=["Run run_deg to generate rank_genes_groups before running GSEA."],
                )

            # Get DEG validity info for caveats
            deg_validity = get_deg_validity(adata)
            deg_caveats = adata.uns.get("deg_caveats", [])

            try:
                import gseapy
            except ImportError:
                return _error_result(
                    tool="run_gsea",
                    message="gseapy not installed.",
                    adata_obj=adata,
                    install_hint="pip install gseapy",
                )

            os.makedirs(output_dir, exist_ok=True)

            # Get clusters to analyze
            groupby = adata.uns['rank_genes_groups']['params']['groupby']
            max_clusters = tool_input.get("max_clusters")
            if cluster == 'all':
                clusters_to_analyze = list(adata.obs[groupby].unique())
                if max_clusters is not None:
                    clusters_to_analyze = clusters_to_analyze[: int(max_clusters)]
            else:
                clusters_to_analyze = [cluster]
            all_results = {}

            for clust in clusters_to_analyze:
                try:
                    # Get DEG results for this cluster
                    deg_df = sc.get.rank_genes_groups_df(adata, group=str(clust))

                    # Create ranked gene list (gene -> score)
                    # Use scores from DEG (stat values work well for GSEA)
                    df_rank = deg_df[['names', 'scores']].dropna()
                    df_rank = df_rank.set_index('names')['scores']

                    # Run GSEA prerank
                    gsea_outdir = os.path.join(output_dir, f"cluster_{clust}")
                    pre_res = gseapy.prerank(
                        rnk=df_rank,
                        gene_sets=gene_sets,
                        threads=1,
                        min_size=min_size,
                        max_size=max_size,
                        permutation_num=permutation_num,
                        outdir=gsea_outdir,
                        seed=42,
                        verbose=False,
                    )

                    # Get top results
                    res_df = pre_res.res2d
                    res_df = res_df.sort_values('NES', ascending=False)

                    # Top 5 upregulated and top 5 downregulated
                    top_up = res_df[res_df['NES'] > 0].head(5)
                    top_down = res_df[res_df['NES'] < 0].tail(5)

                    cluster_results = {
                        "upregulated_pathways": [
                            {
                                "term": row['Term'],
                                "nes": round(row['NES'], 2),
                                "fdr": float(f"{row['FDR q-val']:.2e}"),
                                "genes": row['Lead_genes'].split(';')[:5] if row['Lead_genes'] else []
                            }
                            for _, row in top_up.iterrows()
                        ],
                        "downregulated_pathways": [
                            {
                                "term": row['Term'],
                                "nes": round(row['NES'], 2),
                                "fdr": float(f"{row['FDR q-val']:.2e}"),
                                "genes": row['Lead_genes'].split(';')[:5] if row['Lead_genes'] else []
                            }
                            for _, row in top_down.iterrows()
                        ],
                        "total_significant": int((res_df['FDR q-val'] < 0.25).sum()),
                    }
                    all_results[str(clust)] = cluster_results

                except Exception as e:
                    all_results[str(clust)] = {"error": str(e)}

            # Add per-cluster caveats based on DEG validity
            cluster_caveats = {}
            for clust in clusters_to_analyze:
                caveats = get_cluster_caveats(adata, str(clust))
                if caveats:
                    cluster_caveats[str(clust)] = caveats

            # Build response with validity metadata
            response = {
                "status": "ok",
                "tool": "run_gsea",
                "output_dir": output_dir,
                "gene_sets": gene_sets,
                "clusters_analyzed": clusters_to_analyze,
                "results": all_results,
                "recommended_next_steps": [
                    "Use search_papers on the most significant pathway terms for biological interpretation and recent reviews.",
                    "Use web_search for pathway database or software documentation questions."
                ],
                "note": "NES > 0 means pathway upregulated in this cluster. FDR < 0.25 is typically significant.",
            }

            # Add DEG validity info if present
            if deg_validity:
                response["deg_validity"] = {
                    "is_valid": deg_validity.get("is_valid", True),
                    "has_warnings": deg_validity.get("has_warnings", False),
                    "matrix_type": deg_validity.get("matrix_type"),
                    "data_species": deg_validity.get("data_species"),
                    "gene_id_format": deg_validity.get("gene_id_format"),
                }
            if deg_caveats:
                response["deg_caveats"] = deg_caveats
            if cluster_caveats:
                response["cluster_caveats"] = cluster_caveats

            return json.dumps(response, indent=2), adata

        elif tool_name == "generate_figure":
            adata, _ = get_adata(tool_input, adata)
            plot_type = tool_input["plot_type"]
            output_path = tool_input.get("output_path")
            if not output_path:
                safe_color = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(tool_input.get("color_by", "plot")))
                output_path = f"{plot_type}_{safe_color or 'plot'}.png"
            color_by = tool_input.get("color_by")
            genes = tool_input.get("genes", [])
            include_image = tool_input.get("include_image", True)
            _embedding_key = {"umap": "X_umap", "tsne": "X_tsne"}.get(plot_type)
            if _embedding_key is not None and _embedding_key not in adata.obsm:
                return _smart_unavailable_result(
                    tool="generate_figure",
                    message=(
                        f"{plot_type.upper()} cannot be rendered because the "
                        f"embedding (obsm['{_embedding_key}']) is not available "
                        "on the current in-memory dataset."
                    ),
                    adata_obj=adata,
                    missing_prerequisites=["embedding"],
                    recovery_options=(
                        ["Run PCA, neighbors, and UMAP first.",
                         "If you only need a summary of current state, inspect the session instead of plotting."]
                        if plot_type == "umap" else
                        [f"Compute t-SNE first (e.g. sc.tl.tsne(adata) via run_code) so obsm['{_embedding_key}'] is populated.",
                         "If a UMAP exists instead, switch plot_type to 'umap'."]
                    ),
                    extra={"plot_type": plot_type, "color_by": color_by},
                )
            if plot_type in ("umap", "tsne") and color_by not in (None, "") and color_by not in adata.obs.columns and color_by not in adata.var_names:
                return _smart_unavailable_result(
                    tool="generate_figure",
                    message=f"{plot_type.upper()} coloring key '{color_by}' is not available on the current in-memory dataset.",
                    adata_obj=adata,
                    missing_prerequisites=["valid_color_key"],
                    recovery_options=[
                        "Use one of the available obs columns or genes for coloring.",
                        f"Render a plain {plot_type.upper()} without coloring.",
                    ],
                    extra={"plot_type": plot_type, "requested_color_by": color_by},
                )
            result = _render_figure(
                adata,
                plot_type=plot_type,
                output_path=output_path,
                color_by=color_by,
                genes=genes,
                include_image=include_image,
            )
            result["available_clusterings"] = _clusterings_payload(adata)
            artifact = _artifact_payload(
                output_path,
                role="figure",
                metadata={"plot_type": plot_type, "color_by": color_by},
            )
            artifacts = [artifact] if artifact is not None else []

            # Auto-generate cluster highlight grid when coloring by a categorical column
            # (covers leiden/louvain/pheno results; skips continuous gene expression colorings)
            if (
                plot_type == "umap"
                and color_by is not None
                and color_by in adata.obs.columns
                and hasattr(adata.obs[color_by], "cat")
            ):
                try:
                    grid_path = _generate_cluster_highlight_grid(adata, color_by, output_path)
                    if grid_path and os.path.exists(grid_path):
                        grid_artifact = _artifact_payload(
                            grid_path,
                            role="figure",
                            metadata={"plot_type": "umap_cluster_grid", "color_by": color_by},
                        )
                        if grid_artifact:
                            artifacts.append(grid_artifact)
                        result["cluster_grid_path"] = grid_path
                except Exception as _grid_err:
                    logger.warning("Cluster highlight grid generation failed: %s", _grid_err)

            verification_checks = [
                _check("figure_exists", os.path.exists(output_path), f"Figure exists at {output_path}."),
            ]
            if plot_type in ("umap", "tsne") and color_by not in (None, ""):
                verification_checks.append(
                    _check(
                        "color_key_valid",
                        color_by in adata.obs.columns or color_by in adata.var_names,
                        f"Color key '{color_by}' exists in AnnData.",
                    )
                )
            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary=f"Generated a {plot_type} figure colored by '{color_by}'.",
                artifacts_created=artifacts,
                verification=_build_verification(
                    "passed",
                    "Figure output was created and verified.",
                    verification_checks,
                ),
            )

        elif tool_name == "read_file":
            import re as _re
            raw_arg = tool_input["path"]
            resolved = _resolve_run_path(raw_arg, run_manager=run_manager, must_exist=True)
            if resolved is None:
                return _error_result(
                    tool="read_file",
                    message=f"File not found: {raw_arg}",
                    adata_obj=adata,
                    recovery_options=[
                        "Verify the file path exists and is accessible.",
                        "Bare filenames are resolved against the run directory first; pass an absolute path to read files outside it.",
                    ],
                )
            file_path = resolved

            max_chars = int(tool_input.get("max_chars") or 20000)
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                try:
                    import fitz  # pymupdf
                except ImportError:
                    return _error_result(
                        tool="read_file",
                        message="pymupdf not installed.",
                        adata_obj=adata,
                        install_hint="pip install pymupdf",
                    )

                doc = fitz.open(str(file_path))
                n_pages = len(doc)

                # Parse page selection
                pages_param = tool_input.get("pages", "").strip()
                if pages_param:
                    selected = set()
                    for part in _re.split(r"[,\s]+", pages_param):
                        if "-" in part:
                            a, b = part.split("-", 1)
                            selected.update(range(int(a) - 1, int(b)))
                        elif part.isdigit():
                            selected.add(int(part) - 1)
                    page_indices = sorted(p for p in selected if 0 <= p < n_pages)
                else:
                    page_indices = list(range(n_pages))

                # Text extraction
                parts = []
                for i in page_indices:
                    text = doc[i].get_text().strip()
                    if text:
                        parts.append(f"[Page {i+1}]\n{text}")

                full_text = "\n\n".join(parts)
                truncated = len(full_text) > max_chars
                content = full_text[:max_chars]

                result = {
                    "status": "ok",
                    "tool": "read_file",
                    "path": str(file_path),
                    "type": "pdf",
                    "total_pages": n_pages,
                    "pages_read": [i + 1 for i in page_indices],
                    "truncated": truncated,
                    "chars_returned": len(content),
                    "content": content,
                }

                # Optional page rendering — lets vision model see figures and plots
                render_pages = bool(tool_input.get("render_pages", False))
                if render_pages:
                    import base64 as _b64
                    if run_manager is not None:
                        pdf_pages_dir = run_manager.dirs["figures"] / "pdf_pages"
                    else:
                        import tempfile
                        pdf_pages_dir = Path(tempfile.mkdtemp())
                    pdf_pages_dir.mkdir(parents=True, exist_ok=True)

                    rendered_paths = []
                    for i in page_indices:
                        page = doc[i]
                        mat = fitz.Matrix(1.5, 1.5)  # 108 DPI — good quality, reasonable size
                        pix = page.get_pixmap(matrix=mat)
                        out_path = pdf_pages_dir / f"page_{i + 1}.png"
                        pix.save(str(out_path))
                        rendered_paths.append(str(out_path))

                    if rendered_paths:
                        # First page goes to the vision pipeline via _pending_image
                        result["image_base64"] = _b64.b64encode(
                            open(rendered_paths[0], "rb").read()
                        ).decode()
                        result["image_mime"] = "image/png"
                        result["image_context"] = {"output_path": rendered_paths[0]}
                        result["rendered_page_paths"] = rendered_paths
                        if len(rendered_paths) > 1:
                            result["note"] = (
                                f"Page 1 sent to vision model inline. "
                                f"Remaining {len(rendered_paths) - 1} page(s) saved to "
                                f"figures/pdf_pages/ — use review_figure to inspect them."
                            )

                doc.close()
                return json.dumps(result, indent=2), adata

            else:
                # Plain text, markdown, CSV, TSV, JSON, etc.
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    return _error_result(
                        tool="read_file",
                        message=str(e),
                        adata_obj=adata,
                        recovery_options=["Verify the file is readable and in a supported text format."],
                    )

                truncated = len(raw) > max_chars
                content = raw[:max_chars]
                return json.dumps({
                    "status": "ok",
                    "tool": "read_file",
                    "path": str(file_path),
                    "type": suffix.lstrip(".") or "text",
                    "truncated": truncated,
                    "chars_returned": len(content),
                    "content": content,
                }, indent=2), adata

        elif tool_name == "run_cluster_qc":
            import pandas as pd

            if adata is None:
                return _error_result(
                    tool="run_cluster_qc",
                    message="No in-memory data available. Load and cluster data first.",
                    recovery_options=["Run load_data, then normalize/cluster before running cluster QC."],
                )

            cluster_key = tool_input.get("cluster_key", "leiden")
            if cluster_key not in adata.obs.columns:
                available = [c for c in adata.obs.columns if adata.obs[c].dtype.name == "category"]
                return _error_result(
                    tool="run_cluster_qc",
                    message=f"Cluster key '{cluster_key}' not found in adata.obs. Available categorical columns: {available}",
                    adata_obj=adata,
                    recovery_options=["Run run_clustering first, or specify the correct cluster_key."],
                )

            required_cols = ["total_counts", "n_genes_by_counts", "pct_counts_mt"]
            missing = [c for c in required_cols if c not in adata.obs.columns]
            if missing:
                return _error_result(
                    tool="run_cluster_qc",
                    message=f"QC metrics not computed. Missing obs columns: {missing}. Run run_qc first.",
                    adata_obj=adata,
                    recovery_options=["Run run_qc (flag_only=true) to compute QC metrics before cluster QC."],
                )

            doublet_threshold = float(tool_input.get("doublet_threshold", 0.3))
            mt_threshold = float(tool_input.get("mt_threshold", 25.0))
            low_lib_frac = float(tool_input.get("low_lib_fraction", 0.5))
            low_genes_frac = float(tool_input.get("low_genes_fraction", 0.5))
            save_checkpoint = bool(tool_input.get("save_checkpoint", True))

            has_doublet = "doublet_score" in adata.obs.columns
            has_flag_mt = "qc_flag_high_mt" in adata.obs.columns
            has_flag_lib = "qc_flag_low_lib" in adata.obs.columns

            agg_dict = {
                "n_cells": ("total_counts", "count"),
                "mean_lib_size": ("total_counts", "mean"),
                "mean_n_genes": ("n_genes_by_counts", "mean"),
                "mean_mt": ("pct_counts_mt", "mean"),
            }
            if has_doublet:
                agg_dict["mean_doublet"] = ("doublet_score", "mean")
            if has_flag_mt:
                agg_dict["frac_flagged_mt"] = ("qc_flag_high_mt", "mean")
            if has_flag_lib:
                agg_dict["frac_flagged_lib"] = ("qc_flag_low_lib", "mean")
            if "pct_counts_ribo" in adata.obs.columns:
                agg_dict["mean_ribo"] = ("pct_counts_ribo", "mean")

            cluster_qc = adata.obs.groupby(cluster_key).agg(**agg_dict).round(2)

            global_lib = float(adata.obs["total_counts"].median())
            global_genes = float(adata.obs["n_genes_by_counts"].median())

            cluster_decisions = {}
            proposed_removal = []
            ambiguous = []
            clean = []

            for cluster in cluster_qc.index:
                row = cluster_qc.loc[cluster]
                mean_mt = float(row["mean_mt"])
                mean_lib = float(row["mean_lib_size"])
                mean_genes = float(row["mean_n_genes"])
                mean_doublet = float(row.get("mean_doublet", 0.0))

                lib_low = mean_lib < low_lib_frac * global_lib
                genes_low = mean_genes < low_genes_frac * global_genes
                mt_high = mean_mt > mt_threshold
                doublet_high = mean_doublet > doublet_threshold and "mean_doublet" in row.index
                high_library = mean_lib > 1.5 * global_lib

                evidence = {
                    "low_library": bool(lib_low),
                    "low_genes": bool(genes_low),
                    "high_mt": bool(mt_high),
                    "high_doublet_score": bool(doublet_high),
                    "high_library": bool(high_library),
                    "mean_library_fraction_of_global_median": round(mean_lib / global_lib, 2) if global_lib else None,
                    "mean_genes_fraction_of_global_median": round(mean_genes / global_genes, 2) if global_genes else None,
                }
                reasons = []
                if lib_low:
                    reasons.append(
                        f"mean library size is below {low_lib_frac:.2g}x the global median"
                    )
                if genes_low:
                    reasons.append(
                        f"mean detected genes is below {low_genes_frac:.2g}x the global median"
                    )
                if mt_high:
                    reasons.append(f"mean MT% is above {mt_threshold:g}%")
                if doublet_high:
                    reasons.append(f"mean doublet score is above {doublet_threshold:g}")
                if high_library:
                    reasons.append("mean library size is substantially above the global median")

                if mt_high and lib_low and genes_low:
                    recommended_action = "propose_removal"
                    severity = "obvious"
                    proposed_removal.append(str(cluster))
                elif mt_high and not lib_low:
                    recommended_action = "review"
                    severity = "ambiguous"
                    ambiguous.append(str(cluster))
                elif doublet_high and high_library:
                    recommended_action = "propose_removal"
                    severity = "obvious"
                    proposed_removal.append(str(cluster))
                elif lib_low and genes_low and not mt_high:
                    recommended_action = "propose_removal"
                    severity = "obvious"
                    proposed_removal.append(str(cluster))
                else:
                    recommended_action = "keep"
                    severity = "clean"
                    if not reasons:
                        reasons.append("cluster-level QC metrics are within expected ranges")
                    clean.append(str(cluster))

                cluster_decisions[str(cluster)] = {
                    "recommended_action": recommended_action,
                    "severity": severity,
                    "evidence": evidence,
                    "reasons": reasons,
                }

            cluster_labels = adata.obs[cluster_key].astype(str)
            cells_proposed = int(cluster_labels.isin(proposed_removal).sum())
            cells_ambiguous = int(cluster_labels.isin(ambiguous).sum())
            cells_total = adata.n_obs
            metric_flagged_clusters = list(proposed_removal)
            cells_metric_flagged = cells_proposed

            checkpoint_path = None
            if save_checkpoint:
                import os as _os
                _base = str(run_manager.run_dir) if run_manager else "."
                cp_default = _os.path.join(_base, "checkpoint_pre_cleanup.h5ad")
                checkpoint_path = tool_input.get("checkpoint_path") or cp_default
                write_h5ad_safe(adata, checkpoint_path)

            cluster_table = cluster_qc.reset_index().rename(columns={cluster_key: "cluster"})
            cluster_table["cluster"] = cluster_table["cluster"].astype(str)
            cluster_table["recommended_action"] = cluster_table["cluster"].map(
                lambda c: cluster_decisions.get(str(c), {}).get("recommended_action", "keep")
            )
            cluster_table["severity"] = cluster_table["cluster"].map(
                lambda c: cluster_decisions.get(str(c), {}).get("severity", "clean")
            )
            cluster_table["evidence"] = cluster_table["cluster"].map(
                lambda c: cluster_decisions.get(str(c), {}).get("evidence", {})
            )
            cluster_table["reasons"] = cluster_table["cluster"].map(
                lambda c: cluster_decisions.get(str(c), {}).get("reasons", [])
            )
            cluster_table = cluster_table.to_dict("records")

            result = {
                "status": "ok",
                "tool": "run_cluster_qc",
                "cluster_key": cluster_key,
                "n_clusters": len(cluster_qc),
                "global_lib_median": round(global_lib, 1),
                "global_genes_median": round(global_genes, 1),
                "cluster_table": cluster_table,
                "cluster_decisions": cluster_decisions,
                "metric_flagged_clusters": metric_flagged_clusters,
                "cells_in_metric_flagged_clusters": cells_metric_flagged,
                "pct_metric_flagged": round(cells_metric_flagged / cells_total * 100, 1),
                "metric_qc_interpretation": (
                    "Metric QC flagged these clusters as problematic/suspicious and in need of "
                    "structure QC adjudication; this is not a removal decision."
                ),
                "proposed_removal": proposed_removal,
                "ambiguous": ambiguous,
                "clean": clean,
                "cells_in_proposed_removal": cells_proposed,
                "cells_ambiguous": cells_ambiguous,
                "pct_proposed": round(cells_proposed / cells_total * 100, 1),
                "cells_remaining_if_removed": cells_total - cells_proposed,
                "checkpoint_path": checkpoint_path,
                "thresholds_used": {
                    "mt_threshold": mt_threshold,
                    "doublet_threshold": doublet_threshold,
                    "low_lib_fraction": low_lib_frac,
                    "low_genes_fraction": low_genes_frac,
                },
                "state": make_state(adata),
            }
            artifacts = []
            if checkpoint_path:
                artifacts.append(_artifact_payload(checkpoint_path, role="checkpoint", metadata={"stage": "pre_cluster_qc_cleanup"}))
            return _finalize_result(
                result, adata,
                dataset_changed=False,
                summary=(
                    f"Cluster QC: {len(metric_flagged_clusters)} metric-flagged cluster(s) "
                    f"({cells_metric_flagged} cells, {result['pct_metric_flagged']}%) and "
                    f"{len(ambiguous)} ambiguous cluster(s) require structure QC adjudication."
                ),
                artifacts_created=artifacts,
            )

        elif tool_name == "run_cluster_structure_qc":
            import math
            import os as _os

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            import numpy as _np
            import scipy.sparse as _sp
            from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
            from scipy.spatial.distance import pdist

            if adata is None:
                return _error_result(
                    tool="run_cluster_structure_qc",
                    message="No in-memory data available. Load, QC, embed, and cluster data first.",
                    recovery_options=["Run load_data, run_qc, embedding, clustering, then run_cluster_qc."],
                )

            cluster_key = tool_input.get("cluster_key", "leiden")
            if cluster_key not in adata.obs.columns:
                return _error_result(
                    tool="run_cluster_structure_qc",
                    message=f"Cluster key '{cluster_key}' not found in adata.obs.",
                    adata_obj=adata,
                    recovery_options=["Run run_clustering first, or pass the correct cluster_key."],
                )

            n_genes = max(2, int(tool_input.get("n_genes", 150)))
            min_cells = max(2, int(tool_input.get("min_cells", 15)))
            moran_min_cells = max(2, int(tool_input.get("moran_min_cells", 40)))
            corr_threshold = float(tool_input.get("corr_threshold", 0.3))

            latest_cluster_qc = {}
            if world_state is not None:
                latest_cluster_qc = (
                    getattr(world_state, "cluster_qc_registry", {}) or {}
                ).get(str(cluster_key), {}) or {}

            requested_clusters = tool_input.get("clusters_to_analyze")
            if requested_clusters:
                clusters_to_analyze = [str(c) for c in requested_clusters]
            else:
                clusters_to_analyze = [
                    str(c)
                    for c in (
                        latest_cluster_qc.get("proposed_removal", [])
                        + latest_cluster_qc.get("ambiguous", [])
                    )
                ]
            clusters_to_analyze = list(dict.fromkeys(clusters_to_analyze))
            if not clusters_to_analyze:
                return _error_result(
                    tool="run_cluster_structure_qc",
                    message=(
                        "No clusters were provided and no latest run_cluster_qc proposed/ambiguous "
                        "clusters were available in world state."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        "Run run_cluster_qc first.",
                        "Or pass clusters_to_analyze explicitly.",
                    ],
                )

            cluster_labels = adata.obs[cluster_key].astype(str)
            present_clusters = set(cluster_labels.unique())
            missing_clusters = [c for c in clusters_to_analyze if c not in present_clusters]
            clusters_to_analyze = [c for c in clusters_to_analyze if c in present_clusters]
            if not clusters_to_analyze:
                return _error_result(
                    tool="run_cluster_structure_qc",
                    message=f"None of the requested clusters are present in '{cluster_key}': {missing_clusters}",
                    adata_obj=adata,
                    recovery_options=["Inspect cluster sizes and retry with current cluster IDs."],
                )

            if tool_input.get("figure_dir"):
                base_figure_dir = Path(str(tool_input["figure_dir"]))
            elif run_manager:
                base_figure_dir = Path(run_manager.run_dir) / "figures" / "cluster_qc"
            else:
                base_figure_dir = Path("figures") / "cluster_qc"

            def _safe_path_component(value):
                text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("._")
                return text or "unknown"

            safe_cluster_key = _safe_path_component(cluster_key)
            structure_history = []
            try:
                history_root = adata.uns.get("cluster_structure_qc_history", {})
                if isinstance(history_root, dict):
                    existing = history_root.get(str(cluster_key), [])
                    if isinstance(existing, list):
                        structure_history = existing
            except Exception:
                structure_history = []

            existing_latest = None
            try:
                latest_root = adata.uns.get("cluster_structure_qc", {})
                if isinstance(latest_root, dict):
                    existing_latest = latest_root.get(str(cluster_key))
            except Exception:
                existing_latest = None

            if structure_history:
                structure_qc_pass = len(structure_history) + 1
            elif existing_latest:
                # Data saved by older scagent versions may have a latest structure-QC
                # record but no history list. Start at pass 002 to avoid reusing paths.
                structure_qc_pass = 2
            else:
                structure_qc_pass = 1

            structure_qc_run_id = f"{safe_cluster_key}__pass_{structure_qc_pass:03d}"
            figure_dir = base_figure_dir / safe_cluster_key / f"pass_{structure_qc_pass:03d}"
            figure_dir.mkdir(parents=True, exist_ok=True)

            exclude_patterns = tool_input.get("exclude_patterns") or DEFAULT_STRUCTURE_EXCLUDE_PATTERNS
            try:
                exclude_regexes = [re.compile(str(p)) for p in exclude_patterns if str(p).strip()]
            except Exception:
                exclude_regexes = [re.compile(p) for p in DEFAULT_STRUCTURE_EXCLUDE_PATTERNS]
            mt_regexes = [re.compile(r"^MT-"), re.compile(r"^mt-")]

            var_names = _np.asarray(adata.var_names.astype(str))
            hvg_mask = None
            if "highly_variable" in adata.var.columns:
                try:
                    hvg_mask = _np.asarray(adata.var["highly_variable"].fillna(False).astype(bool))
                except Exception:
                    hvg_mask = None

            def _matches_any(gene: str, regexes) -> bool:
                for rx in regexes:
                    try:
                        if rx.search(gene):
                            return True
                    except Exception:
                        continue
                return False

            def _dense_matrix(matrix):
                return matrix.toarray() if _sp.issparse(matrix) else _np.asarray(matrix)

            def _cluster_z(values, mask):
                arr = _np.asarray(values, dtype=float)
                finite = _np.isfinite(arr)
                if not finite.any():
                    return None, None
                global_mean = float(_np.nanmean(arr))
                global_std = float(_np.nanstd(arr))
                cluster_mean = float(_np.nanmean(arr[mask])) if mask.any() else None
                if cluster_mean is None or global_std <= 0 or not _np.isfinite(global_std):
                    return cluster_mean, None
                return cluster_mean, float((cluster_mean - global_mean) / global_std)

            def _local_moran(values, graph):
                arr = _np.asarray(values, dtype=float)
                finite = _np.isfinite(arr)
                if not finite.all():
                    arr = arr.copy()
                    arr[~finite] = _np.nanmean(arr[finite]) if finite.any() else 0.0
                x_centered = arr - float(arr.mean())
                w_sum = float(graph.sum())
                denom = float(x_centered @ x_centered)
                n_obs = len(arr)
                if denom <= 0 or w_sum <= 0:
                    return _np.zeros(n_obs), 0.0
                lag = _np.asarray(graph @ x_centered).ravel()
                local_i = (n_obs / w_sum) * x_centered * lag / (denom / n_obs)
                global_i = float((n_obs / w_sum) * float(x_centered @ lag) / denom)
                return local_i, global_i

            graph = adata.obsp.get("connectivities") if hasattr(adata, "obsp") else None
            if graph is not None and not _sp.issparse(graph):
                graph = _sp.csr_matrix(graph)
            moran_available = graph is not None and graph.shape == (adata.n_obs, adata.n_obs)
            local_mt = local_lib = None
            global_moran_mt = global_moran_lib = None
            if moran_available:
                if "pct_counts_mt" in adata.obs.columns:
                    local_mt, global_moran_mt = _local_moran(adata.obs["pct_counts_mt"].values, graph)
                if "total_counts" in adata.obs.columns:
                    local_lib, global_moran_lib = _local_moran(adata.obs["total_counts"].values, graph)

            metric_decisions = latest_cluster_qc.get("cluster_decisions") or {}
            metric_table = {
                str(row.get("cluster")): row
                for row in (latest_cluster_qc.get("cluster_table") or [])
                if isinstance(row, dict) and row.get("cluster") is not None
            }

            def _select_genes(mask):
                x_cluster = adata.X[mask, :]
                means = _np.asarray(x_cluster.mean(axis=0)).ravel()
                means = _np.nan_to_num(means, nan=-_np.inf, posinf=-_np.inf, neginf=-_np.inf)
                order = _np.argsort(means)[::-1]
                top_order = order[: max(n_genes * 4, n_genes)]
                non_nuisance = [
                    int(idx)
                    for idx in top_order
                    if not _matches_any(str(var_names[idx]), exclude_regexes)
                ]
                hvg_filtered = [
                    idx
                    for idx in non_nuisance
                    if hvg_mask is not None and bool(hvg_mask[idx])
                ]
                if len(hvg_filtered) >= 20:
                    selected = hvg_filtered[:n_genes]
                    strategy = "top_expressed_hvg_non_nuisance"
                else:
                    selected = [
                        int(idx)
                        for idx in top_order
                        if not _matches_any(str(var_names[idx]), mt_regexes)
                    ][:n_genes]
                    strategy = "fallback_top_expressed_mt_excluded"
                excluded_nuisance = [
                    str(var_names[idx])
                    for idx in top_order[:n_genes]
                    if _matches_any(str(var_names[idx]), exclude_regexes)
                ][:25]
                return selected, {
                    "gene_selection_strategy": strategy,
                    "genes_after_hvg_filter": len(hvg_filtered),
                    "excluded_nuisance_genes_preview": excluded_nuisance,
                }

            def _pca_support(x):
                x = _np.asarray(x, dtype=float)
                x = x - x.mean(axis=0, keepdims=True)
                total_var = float(_np.sum(_np.var(x, axis=0)))
                if total_var <= 0:
                    return None, []
                try:
                    singular = _np.linalg.svd(x, compute_uv=False)
                except Exception:
                    return None, []
                denom = max(x.shape[0] - 1, 1)
                explained = (singular ** 2) / denom
                ratios = (explained / explained.sum()).tolist() if explained.sum() > 0 else []
                return (float(ratios[0]) if ratios else None), [float(v) for v in ratios[:5]]

            def _structure_interpretation(mean_abs_corr, frac_pairs):
                if mean_abs_corr is None:
                    return "inconclusive"
                if mean_abs_corr < 0.08 and frac_pairs < 0.05:
                    return "unstructured"
                if mean_abs_corr < 0.12:
                    return "weak"
                if mean_abs_corr < 0.18:
                    return "moderate"
                return "strong"

            def _synthesize(metric_severity, structure_interp, moran_i_mt, mt_z, moran_i_lib, lib_z):
                strong_structure = structure_interp in {"moderate", "strong"}
                weak_structure = structure_interp in {"unstructured", "weak"}
                mt_pocket = (
                    moran_i_mt is not None
                    and mt_z is not None
                    and moran_i_mt > 0.3
                    and mt_z > 0.5
                )
                low_lib_pocket = (
                    moran_i_lib is not None
                    and lib_z is not None
                    and moran_i_lib > 0.3
                    and lib_z < -0.5
                )
                bad_quality_pocket = bool(mt_pocket or low_lib_pocket)
                metric = str(metric_severity or "unknown")

                if structure_interp in {"inconclusive", "skipped_small_cluster", "skipped_low_gene_count"}:
                    return "inconclusive", "review"
                if metric == "obvious" and weak_structure:
                    return "confirmed_junk", "remove"
                if metric == "obvious" and strong_structure:
                    return ("conflicting" if bad_quality_pocket else "obvious_but_structured"), "review"
                if metric == "ambiguous" and weak_structure:
                    return "unstructured_ambiguous", "remove" if bad_quality_pocket else "review"
                if metric == "ambiguous" and strong_structure:
                    return ("conflicting", "review") if bad_quality_pocket else ("structured_ambiguous", "keep")
                if weak_structure and bad_quality_pocket:
                    return "unstructured_ambiguous", "review"
                if strong_structure:
                    return "structured_ambiguous", "keep"
                return "inconclusive", "review"

            cluster_results = []
            structure_evidence = {}
            artifacts = []
            heatmap_paths = []
            heatmap_artifacts = []
            synthesized_removal = []
            rescued_clusters = []
            conflicting_clusters = []
            confirmed_junk = []
            unstructured_ambiguous = []
            structured_ambiguous = []

            for cluster_id in clusters_to_analyze:
                mask = (cluster_labels == str(cluster_id)).values
                n_cells = int(mask.sum())
                metric_decision = metric_decisions.get(str(cluster_id), {}) if isinstance(metric_decisions, dict) else {}
                metric_severity = metric_decision.get("severity") or metric_table.get(str(cluster_id), {}).get("severity")
                metric_action = metric_decision.get("recommended_action") or metric_table.get(str(cluster_id), {}).get("recommended_action")
                reasons_added = []
                heatmap_path = None
                record = {
                    "cluster_id": str(cluster_id),
                    "n_cells": n_cells,
                    "metric_severity_original": metric_severity,
                    "metric_recommended_action": metric_action,
                    "moran_computed": False,
                    "moran_skip_reason": None,
                    "heatmap_path": None,
                    "structure_analysis_skipped": False,
                    "structure_skip_reason": None,
                }

                if n_cells < min_cells:
                    record.update({
                        "structure_analysis_skipped": True,
                        "structure_skip_reason": f"cluster has {n_cells} cells (< {min_cells} minimum)",
                        "structure_interpretation": "skipped_small_cluster",
                        "synthesis": "inconclusive",
                        "synthesis_lean": "review",
                        "reasons_added": [f"Structure analysis skipped: cluster has {n_cells} cells (< {min_cells} minimum)."],
                    })
                    cluster_results.append(record)
                    structure_evidence[str(cluster_id)] = record
                    continue

                selected_genes, selection_info = _select_genes(mask)
                x_sub = _dense_matrix(adata.X[mask, :][:, selected_genes]).astype(float)
                variances = _np.nanvar(x_sub, axis=0)
                keep_var = _np.isfinite(variances) & (variances > 1e-12)
                selected_genes = [idx for idx, keep in zip(selected_genes, keep_var) if bool(keep)]
                x_sub = x_sub[:, keep_var]
                selected_gene_names = [str(var_names[idx]) for idx in selected_genes]

                record.update(selection_info)
                record["n_genes_selected"] = len(selected_genes)
                record["selected_genes_preview"] = selected_gene_names[:20]

                if x_sub.shape[1] < 2:
                    record.update({
                        "structure_analysis_skipped": True,
                        "structure_skip_reason": "fewer than 2 nonzero-variance selected genes",
                        "structure_interpretation": "skipped_low_gene_count",
                        "synthesis": "inconclusive",
                        "synthesis_lean": "review",
                        "reasons_added": ["Structure analysis skipped: fewer than 2 nonzero-variance selected genes."],
                    })
                    cluster_results.append(record)
                    structure_evidence[str(cluster_id)] = record
                    continue

                corr_matrix = _np.corrcoef(x_sub.T)
                corr_matrix = _np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
                off_diag = ~_np.eye(corr_matrix.shape[0], dtype=bool)
                abs_off = _np.abs(corr_matrix[off_diag])
                mean_abs_corr = float(abs_off.mean()) if abs_off.size else None
                frac_pairs = float((abs_off >= corr_threshold).sum() / abs_off.size) if abs_off.size else None
                pc1, eigenspectrum_top5 = _pca_support(x_sub)

                order = _np.arange(corr_matrix.shape[0])
                n_modules = 1
                module_size_summary = [int(corr_matrix.shape[0])]
                linkage_status = "not_run"
                if corr_matrix.shape[0] >= 3:
                    try:
                        distances = pdist(x_sub.T, metric="correlation")
                        distances = _np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=1.0)
                        z = linkage(distances, method="average")
                        order = leaves_list(z)
                        module_labels = fcluster(z, t=0.7, criterion="distance")
                        _, counts = _np.unique(module_labels, return_counts=True)
                        n_modules = int(len(counts))
                        module_size_summary = [int(v) for v in sorted(counts, reverse=True)[:10]]
                        linkage_status = "ok"
                    except Exception as e:
                        linkage_status = f"failed: {e}"

                reordered = corr_matrix[_np.ix_(order, order)]
                safe_cluster_id = _safe_path_component(cluster_id)
                heatmap_path = figure_dir / f"cluster_{safe_cluster_id}_correlation.png"
                fig_width = max(5.0, min(9.0, x_sub.shape[1] / 20))
                fig, ax = _plt.subplots(figsize=(fig_width, fig_width))
                image = ax.imshow(reordered, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                ax.set_title(
                    f"{cluster_key} / pass {structure_qc_pass:03d} / cluster {cluster_id}\n"
                    f"{x_sub.shape[1]} genes x {n_cells} cells | mean_abs_corr={mean_abs_corr:.3f}"
                )
                if x_sub.shape[1] <= 60:
                    ordered_names = [selected_gene_names[int(i)] for i in order]
                    ax.set_xticks(range(len(ordered_names)))
                    ax.set_yticks(range(len(ordered_names)))
                    ax.set_xticklabels(ordered_names, rotation=90, fontsize=5)
                    ax.set_yticklabels(ordered_names, fontsize=5)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(heatmap_path, dpi=180)
                _plt.close(fig)
                heatmap_path_str = str(heatmap_path)
                heatmap_artifact = _artifact_payload(
                    heatmap_path_str,
                    role="cluster_structure_heatmap",
                    metadata={
                        "cluster": str(cluster_id),
                        "cluster_key": cluster_key,
                        "structure_qc_run_id": structure_qc_run_id,
                        "structure_qc_pass": structure_qc_pass,
                    },
                )
                if heatmap_artifact:
                    artifacts.append(heatmap_artifact)
                    heatmap_artifacts.append(heatmap_artifact)
                    heatmap_paths.append(heatmap_artifact.get("path", heatmap_path_str))

                structure_interp = _structure_interpretation(mean_abs_corr, frac_pairs)
                mt_mean = mt_z = lib_mean = lib_z = None
                moran_i_mt = moran_i_lib = None
                if n_cells < moran_min_cells:
                    moran_skip_reason = f"cluster has {n_cells} cells (< {moran_min_cells} minimum)"
                elif not moran_available:
                    moran_skip_reason = "neighbor connectivities graph is not available"
                else:
                    moran_skip_reason = None
                    if local_mt is not None:
                        moran_i_mt = float(_np.asarray(local_mt)[mask].mean())
                        mt_mean, mt_z = _cluster_z(adata.obs["pct_counts_mt"].values, mask)
                    if local_lib is not None:
                        moran_i_lib = float(_np.asarray(local_lib)[mask].mean())
                        lib_mean, lib_z = _cluster_z(adata.obs["total_counts"].values, mask)
                    record["moran_computed"] = True

                synthesis, synthesis_lean = _synthesize(
                    metric_severity,
                    structure_interp,
                    moran_i_mt,
                    mt_z,
                    moran_i_lib,
                    lib_z,
                )

                if structure_interp in {"unstructured", "weak"}:
                    reasons_added.append(
                        f"Correlation structure is {structure_interp} "
                        f"(mean_abs_corr={mean_abs_corr:.3f}, frac_abs_corr>={corr_threshold:g}={frac_pairs:.3f})."
                    )
                elif structure_interp in {"moderate", "strong"}:
                    reasons_added.append(
                        f"Well-structured transcriptional program detected "
                        f"(mean_abs_corr={mean_abs_corr:.3f}, modules={n_modules})."
                    )
                if moran_i_mt is not None:
                    direction = "elevated" if (mt_z is not None and mt_z > 0.5) else "not elevated"
                    reasons_added.append(
                        f"Local MT% Moran's I={moran_i_mt:.3f} with cluster MT z={mt_z:.2f} ({direction})."
                        if mt_z is not None
                        else f"Local MT% Moran's I={moran_i_mt:.3f}."
                    )
                if moran_i_lib is not None:
                    if lib_z is not None and lib_z < -0.5:
                        lib_direction = "low-library pocket"
                    elif lib_z is not None and lib_z > 0.5:
                        lib_direction = "high-library pocket"
                    else:
                        lib_direction = "near global library size"
                    reasons_added.append(
                        f"Local library-size Moran's I={moran_i_lib:.3f} with cluster library z={lib_z:.2f} ({lib_direction})."
                        if lib_z is not None
                        else f"Local library-size Moran's I={moran_i_lib:.3f}."
                    )
                if moran_skip_reason:
                    reasons_added.append(f"Technical Moran's I skipped: {moran_skip_reason}.")

                record.update({
                    "mean_abs_corr": mean_abs_corr,
                    "frac_pairs_above_threshold": frac_pairs,
                    "corr_threshold": corr_threshold,
                    "n_modules": n_modules,
                    "module_size_summary": module_size_summary,
                    "linkage_status": linkage_status,
                    "variance_explained_pc1": pc1,
                    "eigenspectrum_top5": eigenspectrum_top5,
                    "structure_interpretation": structure_interp,
                    "moran_i_mt": moran_i_mt,
                    "moran_i_lib": moran_i_lib,
                    "global_moran_i_mt": global_moran_mt,
                    "global_moran_i_lib": global_moran_lib,
                    "cluster_mean_mt": mt_mean,
                    "cluster_mean_total_counts": lib_mean,
                    "cluster_mt_z": mt_z,
                    "cluster_lib_z": lib_z,
                    "moran_skip_reason": moran_skip_reason,
                    "heatmap_path": heatmap_path_str,
                    "synthesis": synthesis,
                    "synthesis_lean": synthesis_lean,
                    "reasons_added": reasons_added,
                })

                if synthesis_lean == "remove":
                    synthesized_removal.append(str(cluster_id))
                if synthesis == "confirmed_junk":
                    confirmed_junk.append(str(cluster_id))
                if synthesis == "structured_ambiguous":
                    structured_ambiguous.append(str(cluster_id))
                    rescued_clusters.append(str(cluster_id))
                if synthesis == "unstructured_ambiguous":
                    unstructured_ambiguous.append(str(cluster_id))
                if synthesis == "conflicting":
                    conflicting_clusters.append(str(cluster_id))

                cluster_results.append(record)
                structure_evidence[str(cluster_id)] = record

            cells_in_synthesized_removal = int(cluster_labels.isin(synthesized_removal).sum())
            result = {
                "status": "ok",
                "tool": "run_cluster_structure_qc",
                "cluster_key": cluster_key,
                "clusters_analyzed": clusters_to_analyze,
                "missing_clusters": missing_clusters,
                "n_clusters_analyzed": len(cluster_results),
                "cluster_structure_evidence": cluster_results,
                "structure_evidence_by_cluster": structure_evidence,
                "synthesized_removal": synthesized_removal,
                "cells_in_synthesized_removal": cells_in_synthesized_removal,
                "pct_synthesized_removal": round(cells_in_synthesized_removal / adata.n_obs * 100, 1),
                "rescued_clusters": rescued_clusters,
                "confirmed_junk": confirmed_junk,
                "structured_ambiguous": structured_ambiguous,
                "unstructured_ambiguous": unstructured_ambiguous,
                "conflicting": conflicting_clusters,
                "structure_qc_run_id": structure_qc_run_id,
                "structure_qc_pass": structure_qc_pass,
                "base_figure_dir": str(base_figure_dir),
                "figure_dir": str(figure_dir),
                "heatmap_paths": heatmap_paths,
                "heatmap_artifacts": heatmap_artifacts,
                "visual_evidence_guidance": (
                    "Correlation heatmaps are saved as figure artifacts. When using visual heatmap "
                    "evidence in reasoning, cite the cluster's heatmap_path or artifact path. If a "
                    "cluster has no heatmap_path because structure analysis was skipped, do not claim "
                    "visual heatmap evidence for that cluster."
                ),
                "thresholds_used": {
                    "n_genes": n_genes,
                    "min_cells": min_cells,
                    "moran_min_cells": moran_min_cells,
                    "corr_threshold": corr_threshold,
                    "structure_thresholds": {
                        "unstructured_mean_abs_corr": 0.08,
                        "weak_mean_abs_corr": 0.12,
                        "moderate_mean_abs_corr": 0.18,
                        "technical_moran_high": 0.3,
                        "technical_z_direction": 0.5,
                    },
                },
                "state": make_state(adata),
            }

            if run_manager:
                report_path = run_manager.write_json_report(
                    f"cluster_structure_qc_{structure_qc_run_id}",
                    {
                        k: v
                        for k, v in result.items()
                        if k not in {"state"}
                    },
                )
                result["structure_qc_json"] = report_path
                artifacts.append(
                    _artifact_payload(
                        report_path,
                        role="cluster_structure_qc_report",
                        metadata={
                            "cluster_key": cluster_key,
                            "structure_qc_run_id": structure_qc_run_id,
                            "structure_qc_pass": structure_qc_pass,
                        },
                    )
                )
                def _fmt_num(value, digits=3):
                    try:
                        if value is None or (isinstance(value, float) and not math.isfinite(value)):
                            return "NA"
                        return f"{float(value):.{digits}f}"
                    except Exception:
                        return "NA"

                def _fmt_list(values, limit=4):
                    if not isinstance(values, list) or not values:
                        return "none"
                    shown = [str(v) for v in values[:limit]]
                    return ", ".join(shown) + (f", +{len(values) - limit} more" if len(values) > limit else "")

                def _safe_cell(value):
                    text = str(value if value is not None else "NA")
                    return text.replace("|", "\\|").replace("\n", " ")

                md_lines = [
                    f"# Cluster Structure QC - `{cluster_key}`",
                    "",
                    "This report is the human-readable companion to the machine-readable structure QC JSON. "
                    "Metric QC nominates suspicious clusters; this pass adjudicates them with gene-gene "
                    "correlation structure, saved heatmaps, and technical Moran's I context.",
                    "",
                    "## Summary",
                    "",
                    f"- Clusters analyzed: **{len(cluster_results)}**",
                    f"- Synthesized removal set: **{_fmt_list(synthesized_removal, limit=12)}** "
                    f"({cells_in_synthesized_removal} cells; {result['pct_synthesized_removal']}%)",
                    f"- Rescued/structured clusters: **{_fmt_list(rescued_clusters, limit=12)}**",
                    f"- Conflicting clusters for review: **{_fmt_list(conflicting_clusters, limit=12)}**",
                    f"- Heatmaps saved under: `{figure_dir}`",
                    "",
                    "## Per-Cluster Evidence",
                    "",
                    "| Cluster | Cells | Metric severity | Structure | Mean abs corr | Modules | MT Moran / z | Library Moran / z | Synthesis | Heatmap |",
                    "|---|---:|---|---|---:|---:|---|---|---|---|",
                ]
                for rec in cluster_results:
                    heatmap = rec.get("heatmap_path")
                    heatmap_cell = f"`{heatmap}`" if heatmap else "not generated"
                    md_lines.append(
                        "| "
                        + " | ".join([
                            _safe_cell(rec.get("cluster_id")),
                            _safe_cell(rec.get("n_cells")),
                            _safe_cell(rec.get("metric_severity_original")),
                            _safe_cell(rec.get("structure_interpretation")),
                            _fmt_num(rec.get("mean_abs_corr")),
                            _safe_cell(rec.get("n_modules")),
                            f"{_fmt_num(rec.get('moran_i_mt'))} / {_fmt_num(rec.get('cluster_mt_z'), 2)}",
                            f"{_fmt_num(rec.get('moran_i_lib'))} / {_fmt_num(rec.get('cluster_lib_z'), 2)}",
                            _safe_cell(f"{rec.get('synthesis')} ({rec.get('synthesis_lean')})"),
                            _safe_cell(heatmap_cell),
                        ])
                        + " |"
                    )
                md_lines.extend(["", "## Reasoning Notes", ""])
                for rec in cluster_results:
                    md_lines.append(f"### Cluster {rec.get('cluster_id')}")
                    md_lines.append("")
                    md_lines.append(f"- Synthesis: **{rec.get('synthesis')}**; lean: **{rec.get('synthesis_lean')}**.")
                    md_lines.append(f"- Gene selection: `{rec.get('gene_selection_strategy', 'NA')}`; selected genes: **{rec.get('n_genes_selected', 'NA')}**.")
                    if rec.get("selected_genes_preview"):
                        md_lines.append(f"- Selected gene preview: {_fmt_list(rec.get('selected_genes_preview'), limit=12)}.")
                    for reason in rec.get("reasons_added", []) or []:
                        md_lines.append(f"- {reason}")
                    if rec.get("heatmap_path"):
                        md_lines.append(f"- Heatmap artifact: `{rec.get('heatmap_path')}`.")
                    if rec.get("structure_analysis_skipped"):
                        md_lines.append(f"- Structure analysis skipped: {rec.get('structure_skip_reason')}.")
                    md_lines.append("")
                md_lines.extend([
                    "## Interpretation Guide",
                    "",
                    "- Low/flat correlation structure supports apoptotic, ambient, or otherwise unstructured droplets when it agrees with poor metric QC.",
                    "- Moderate or strong gene-gene structure is evidence that a cluster contains a coherent transcriptional program; if metric QC is poor, treat this as a rescue or conflict signal rather than automatic removal.",
                    "- Moran's I is interpreted with direction: high MT Moran matters most when the cluster also has elevated MT z-score; library Moran is interpreted as low- or high-library depending on the cluster z-score.",
                    "- Heatmap statements in the final analysis should cite the heatmap path above.",
                    "",
                ])
                markdown_path = run_manager.write_text_report(
                    f"cluster_structure_qc_{structure_qc_run_id}_summary",
                    "\n".join(md_lines),
                    ext="md",
                )
                result["structure_qc_markdown"] = markdown_path
                artifacts.append(
                    _artifact_payload(
                        markdown_path,
                        role="cluster_structure_qc_summary",
                        metadata={
                            "cluster_key": cluster_key,
                            "structure_qc_run_id": structure_qc_run_id,
                            "structure_qc_pass": structure_qc_pass,
                        },
                    )
                )

            adata.uns.setdefault("cluster_structure_qc", {})
            adata.uns["cluster_structure_qc"][str(cluster_key)] = {
                "structure_qc_run_id": structure_qc_run_id,
                "structure_qc_pass": structure_qc_pass,
                "base_figure_dir": str(base_figure_dir),
                "figure_dir": str(figure_dir),
                "clusters_analyzed": clusters_to_analyze,
                "structure_evidence_by_cluster": structure_evidence,
                "synthesized_removal": synthesized_removal,
                "rescued_clusters": rescued_clusters,
                "confirmed_junk": confirmed_junk,
                "conflicting": conflicting_clusters,
                "thresholds_used": result["thresholds_used"],
                "heatmap_paths": heatmap_paths,
                "structure_qc_json": result.get("structure_qc_json"),
                "structure_qc_markdown": result.get("structure_qc_markdown"),
            }
            if not isinstance(adata.uns.get("cluster_structure_qc_history"), dict):
                adata.uns["cluster_structure_qc_history"] = {}
            if not isinstance(adata.uns["cluster_structure_qc_history"].get(str(cluster_key)), list):
                adata.uns["cluster_structure_qc_history"][str(cluster_key)] = []
            adata.uns["cluster_structure_qc_history"][str(cluster_key)].append(
                {
                    "structure_qc_run_id": structure_qc_run_id,
                    "structure_qc_pass": structure_qc_pass,
                    "cluster_key": str(cluster_key),
                    "shape": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                    "clusters_analyzed": clusters_to_analyze,
                    "synthesized_removal": synthesized_removal,
                    "cells_in_synthesized_removal": cells_in_synthesized_removal,
                    "pct_synthesized_removal": result["pct_synthesized_removal"],
                    "figure_dir": str(figure_dir),
                    "heatmap_paths": heatmap_paths,
                    "structure_qc_json": result.get("structure_qc_json"),
                    "structure_qc_markdown": result.get("structure_qc_markdown"),
                }
            )

            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary=(
                    f"Cluster structure QC analyzed {len(cluster_results)} cluster(s); "
                    f"{len(synthesized_removal)} synthesized removal candidate(s), "
                    f"{len(rescued_clusters)} rescued structured ambiguous cluster(s)."
                ),
                artifacts_created=[artifact for artifact in artifacts if artifact],
                verification=_build_verification(
                    "passed",
                    "Cluster structure QC completed without mutating AnnData cells or genes.",
                    [
                        _check("dataset_shape_unchanged", True, "No cells or genes were removed."),
                        _check(
                            "heatmap_artifacts_created",
                            len(heatmap_paths) == len([r for r in cluster_results if r.get("heatmap_path")])
                            and all(_os.path.exists(path) for path in heatmap_paths),
                            f"Saved {len(heatmap_paths)} cluster-structure heatmap figure artifact(s).",
                        ),
                    ],
                ),
            )

        elif tool_name == "prepare_annotation":
            import re as _re
            import scanpy as sc
            import scipy.sparse as sp
            import numpy as _np

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            if adata is None:
                return _error_result(
                    tool="prepare_annotation",
                    message="No data in memory. Load a dataset first.",
                    adata_obj=adata,
                    recovery_options=["Run load_data with the path to your h5ad."],
                )

            cluster_key = tool_input.get("cluster_key", "leiden")
            if cluster_key not in adata.obs.columns:
                return _smart_unavailable_result(
                    tool="prepare_annotation",
                    message=f"Cluster column '{cluster_key}' not found in adata.obs.",
                    adata_obj=adata,
                    missing_prerequisites=["clustering"],
                    recovery_options=[
                        "Run run_clustering first to produce a cluster column.",
                        f"Pass an existing cluster column via cluster_key. Available columns: {list(adata.obs.columns)[:30]}",
                    ],
                )

            annotation_key = tool_input.get("annotation_key", "cell_type")
            marker_dict = tool_input.get("marker_dict") or {}
            n_deg_genes = int(tool_input.get("n_deg_genes", 20))
            deg_key = tool_input.get("deg_key", "rank_genes_groups")
            ambiguity_threshold = float(tool_input.get("ambiguity_threshold", 0.10))
            shared_marker_threshold = float(tool_input.get("shared_marker_threshold", 0.5))
            expression_threshold = float(tool_input.get("expression_threshold", 0.0))
            force_recompute = bool(tool_input.get("force_recompute_deg", False))
            deg_method = tool_input.get("deg_method", "wilcoxon")
            panglaodb_species = tool_input.get("panglaodb_species")
            reverse_lookup_n_genes = max(
                0, int(tool_input.get("reverse_lookup_n_genes_per_cluster", 10))
            )
            reverse_lookup_max_unique = max(
                0, int(tool_input.get("reverse_lookup_max_unique_genes", 60))
            )
            default_reverse_exclude_patterns = [
                r"^MT-",
                r"^mt-",
                r"^RPL",
                r"^RPS",
                r"^MRPL",
                r"^MRPS",
                r"^Rpl",
                r"^Rps",
                r"^Mrpl",
                r"^Mrps",
                r"^MALAT1$",
                r"^Malat1$",
                r"^HB[ABDEGMQZ]",
                r"^Hb[ab]",
                r"^RP\d",
                r"^AC\d",
                r"^AL\d",
                r"^AP\d",
                r"^LINC\d",
                r"\.\d+$",
            ]
            supplied_reverse_exclude = tool_input.get("reverse_lookup_exclude_patterns")
            reverse_exclude_patterns = (
                [str(p) for p in supplied_reverse_exclude]
                if isinstance(supplied_reverse_exclude, list)
                else default_reverse_exclude_patterns
            )
            try:
                reverse_exclude_regexes = [
                    _re.compile(p) for p in reverse_exclude_patterns if str(p).strip()
                ]
            except Exception:
                reverse_exclude_regexes = [
                    _re.compile(p) for p in default_reverse_exclude_patterns
                ]

            def _reverse_exclude_reason(gene: str) -> Optional[str]:
                for rx in reverse_exclude_regexes:
                    try:
                        if rx.search(gene):
                            return rx.pattern
                    except Exception:
                        continue
                return None

            supplied_reference_keys = tool_input.get("reference_annotation_keys")
            standard_reference_keys = [
                "celltypist_majority_voting",
                "celltypist_predicted_labels",
                "scimilarity_predictions_unconstrained",
                "scimilarity_representative_prediction",
            ]
            if supplied_reference_keys is None:
                reference_annotation_keys = [
                    k for k in standard_reference_keys if k in adata.obs.columns
                ]
            else:
                reference_annotation_keys = [
                    str(k) for k in supplied_reference_keys
                    if isinstance(k, str) and str(k).strip()
                ]
            valid_reference_keys = [
                k for k in reference_annotation_keys if k in adata.obs.columns
            ]
            missing_reference_keys = [
                k for k in reference_annotation_keys if k not in adata.obs.columns
            ]
            reference_source_coverage: Dict[str, List[str]] = {
                "celltypist": [],
                "scimilarity": [],
                "other": [],
            }
            for key in valid_reference_keys:
                lower_key = key.lower()
                if "celltypist" in lower_key:
                    reference_source_coverage["celltypist"].append(key)
                elif "scimilarity" in lower_key:
                    reference_source_coverage["scimilarity"].append(key)
                else:
                    reference_source_coverage["other"].append(key)
            missing_reference_sources = [
                source
                for source in ("celltypist", "scimilarity")
                if not reference_source_coverage.get(source)
            ]
            if not valid_reference_keys:
                reference_annotation_notice = (
                    "No CellTypist/Scimilarity annotation columns were supplied or auto-detected. "
                    "If a compatible reference model is available, run it before treating this proposal "
                    "as the main source of candidate labels."
                )
            elif missing_reference_sources:
                present_sources = [
                    source for source in ("celltypist", "scimilarity")
                    if reference_source_coverage.get(source)
                ]
                reference_annotation_notice = (
                    "Reference annotation is partial: present sources="
                    f"{present_sources or ['other']}; missing sources={missing_reference_sources}. "
                    "Run the missing compatible reference tool before finalizing, or record the concrete "
                    "reason it was unavailable in the staged evidence."
                )
            else:
                reference_annotation_notice = None

            cluster_series = adata.obs[cluster_key].astype(str)
            cluster_ids = sorted(cluster_series.unique(), key=lambda s: (len(s), s))
            cluster_sizes = {c: int((cluster_series == c).sum()) for c in cluster_ids}

            need_recompute = force_recompute or (deg_key not in adata.uns)
            if not need_recompute:
                cached = adata.uns.get(deg_key, {})
                cached_groupby = None
                params = cached.get("params") if isinstance(cached, dict) else None
                if isinstance(params, dict):
                    cached_groupby = params.get("groupby")
                if cached_groupby != cluster_key:
                    need_recompute = True

            if need_recompute:
                sc.tl.rank_genes_groups(
                    adata,
                    groupby=cluster_key,
                    method=deg_method,
                    key_added=deg_key,
                    use_raw=False,
                    n_genes=max(n_deg_genes, 50),
                )

            rgg = adata.uns.get(deg_key, {})
            names = rgg.get("names") if isinstance(rgg, dict) else None
            scores = rgg.get("scores") if isinstance(rgg, dict) else None
            pvals_adj = rgg.get("pvals_adj") if isinstance(rgg, dict) else None
            logfcs = rgg.get("logfoldchanges") if isinstance(rgg, dict) else None

            top_degs_per_cluster: Dict[str, List[Dict[str, Any]]] = {}
            if names is not None and hasattr(names, "dtype") and names.dtype.names:
                rec_groups = list(names.dtype.names)
                for g in rec_groups:
                    entries = []
                    n_take = min(n_deg_genes, len(names[g]))
                    for i in range(n_take):
                        try:
                            gene = str(names[g][i])
                        except Exception:
                            continue
                        entry = {"gene": gene}
                        if scores is not None:
                            try:
                                entry["score"] = float(scores[g][i])
                            except Exception:
                                pass
                        if logfcs is not None:
                            try:
                                entry["logfc"] = float(logfcs[g][i])
                            except Exception:
                                pass
                        if pvals_adj is not None:
                            try:
                                entry["pval_adj"] = float(pvals_adj[g][i])
                            except Exception:
                                pass
                        entries.append(entry)
                    top_degs_per_cluster[str(g)] = entries
            elif isinstance(names, dict):
                for g, gene_values in names.items():
                    entries = []
                    try:
                        genes_iter = list(gene_values)
                    except Exception:
                        genes_iter = []
                    n_take = min(n_deg_genes, len(genes_iter))
                    for i in range(n_take):
                        try:
                            gene = str(genes_iter[i])
                        except Exception:
                            continue
                        entry = {"gene": gene}
                        if isinstance(scores, dict) and g in scores:
                            try:
                                entry["score"] = float(list(scores[g])[i])
                            except Exception:
                                pass
                        if isinstance(logfcs, dict) and g in logfcs:
                            try:
                                entry["logfc"] = float(list(logfcs[g])[i])
                            except Exception:
                                pass
                        if isinstance(pvals_adj, dict) and g in pvals_adj:
                            try:
                                entry["pval_adj"] = float(list(pvals_adj[g])[i])
                            except Exception:
                                pass
                        entries.append(entry)
                    top_degs_per_cluster[str(g)] = entries

            shared_markers: List[str] = []
            label_marker_lists: Dict[str, List[str]] = {}
            if marker_dict:
                var_set = set(adata.var_names.astype(str))
                for label, genes in marker_dict.items():
                    if not isinstance(genes, list):
                        continue
                    present = [str(g) for g in genes if str(g) in var_set]
                    label_marker_lists[str(label)] = present

                if shared_marker_threshold > 0 and label_marker_lists:
                    gene_label_count: Dict[str, int] = {}
                    for genes in label_marker_lists.values():
                        for g in set(genes):
                            gene_label_count[g] = gene_label_count.get(g, 0) + 1
                    n_labels = len(label_marker_lists)
                    threshold_count = max(2, int(round(shared_marker_threshold * n_labels)))
                    shared_markers = sorted(
                        g for g, c in gene_label_count.items() if c >= threshold_count
                    )

            score_matrix: Dict[str, Dict[str, float]] = {}
            ambiguous_clusters: List[str] = []
            cluster_summaries: List[Dict[str, Any]] = []
            reference_annotation_summary: Dict[str, List[Dict[str, Any]]] = {}
            reverse_lookup_by_cluster: Dict[str, List[str]] = {}
            reverse_lookup_excluded_by_cluster: Dict[str, List[Dict[str, str]]] = {}

            X_layer = tool_input.get("deg_layer")
            X_source = adata.layers[X_layer] if X_layer and X_layer in adata.layers else adata.X

            if label_marker_lists:
                gene_to_idx = {str(g): i for i, g in enumerate(adata.var_names)}
                cluster_masks = {c: (cluster_series == c).values for c in cluster_ids}
                for label, genes in label_marker_lists.items():
                    discriminating = [g for g in genes if g not in shared_markers] or genes
                    idxs = [gene_to_idx[g] for g in discriminating if g in gene_to_idx]
                    if not idxs:
                        for c in cluster_ids:
                            score_matrix.setdefault(c, {})[label] = 0.0
                        continue
                    sub = X_source[:, idxs]
                    if sp.issparse(sub):
                        expressed = (sub > expression_threshold).astype("float32")
                    else:
                        expressed = (_np.asarray(sub) > expression_threshold).astype("float32")
                    for c in cluster_ids:
                        mask = cluster_masks[c]
                        n_in = int(mask.sum())
                        if n_in == 0:
                            score_matrix.setdefault(c, {})[label] = 0.0
                            continue
                        if sp.issparse(expressed):
                            sub_expr = expressed[mask, :]
                            frac = float(sub_expr.sum() / (n_in * len(idxs)))
                        else:
                            frac = float(expressed[mask, :].mean())
                        score_matrix.setdefault(c, {})[label] = frac

            if reverse_lookup_n_genes > 0:
                for c in cluster_ids:
                    selected: List[str] = []
                    excluded: List[Dict[str, str]] = []
                    seen_cluster_genes: set = set()
                    for entry in top_degs_per_cluster.get(c, []):
                        gene = str(entry.get("gene") or "").strip()
                        if not gene or gene in seen_cluster_genes:
                            continue
                        seen_cluster_genes.add(gene)
                        reason = _reverse_exclude_reason(gene)
                        if reason:
                            excluded.append({"gene": gene, "reason": f"matched exclude pattern {reason}"})
                            continue
                        selected.append(gene)
                        if len(selected) >= reverse_lookup_n_genes:
                            break
                    reverse_lookup_by_cluster[c] = selected
                    reverse_lookup_excluded_by_cluster[c] = excluded[:20]

            if valid_reference_keys:
                for c in cluster_ids:
                    mask = (cluster_series == c).values
                    per_cluster_reference: List[Dict[str, Any]] = []
                    n_in = int(mask.sum())
                    for key in valid_reference_keys:
                        values = adata.obs.loc[mask, key].astype(str)
                        values = values[
                            ~values.str.lower().isin({"", "nan", "none", "unknown", "unassigned"})
                        ]
                        if values.empty or n_in == 0:
                            per_cluster_reference.append({
                                "annotation_key": key,
                                "top_label": None,
                                "top_fraction": 0.0,
                                "top_count": 0,
                                "top_labels": [],
                            })
                            continue
                        counts = values.value_counts(dropna=True).head(5)
                        top_label = str(counts.index[0])
                        top_count = int(counts.iloc[0])
                        per_cluster_reference.append({
                            "annotation_key": key,
                            "top_label": top_label,
                            "top_fraction": round(float(top_count / n_in), 4),
                            "top_count": top_count,
                            "top_labels": [
                                {
                                    "label": str(label),
                                    "count": int(count),
                                    "fraction": round(float(count / n_in), 4),
                                }
                                for label, count in counts.items()
                            ],
                        })
                    reference_annotation_summary[c] = per_cluster_reference

            def _norm_label(label: Any) -> str:
                text = str(label or "").lower()
                text = _re.sub(r"[^a-z0-9]+", " ", text)
                text = _re.sub(r"\s+", " ", text).strip()
                return text

            def _label_family(label: Any) -> Optional[str]:
                text = _norm_label(label)
                if not text:
                    return None
                if "platelet" in text or "megakary" in text:
                    return "platelet"
                if "plasma" in text:
                    return "plasma"
                if "monocyte" in text or "macrophage" in text:
                    return "monocyte"
                if "plasmacytoid dendritic" in text or text == "pdc" or " pdc" in f" {text}":
                    return "pdc"
                if "dendritic" in text or text in {"dc", "cdc", "cdc1", "cdc2"} or " cdc" in f" {text}":
                    return "dendritic"
                if "natural killer" in text or " nk" in f" {text}" or text.startswith("nk"):
                    return "nk"
                if "b cell" in text or text.startswith("b ") or " b " in f" {text} ":
                    return "b"
                if "t cell" in text or text.startswith("t ") or " t " in f" {text} " or "mait" in text or "treg" in text:
                    return "t"
                if "neutrophil" in text:
                    return "neutrophil"
                if "mast" in text or "basophil" in text:
                    return "mast_basophil"
                if "eryth" in text or "red blood" in text:
                    return "erythroid"
                if "epithelial" in text:
                    return "epithelial"
                if "endothelial" in text:
                    return "endothelial"
                if "fibroblast" in text or "stromal" in text:
                    return "stromal"
                return None

            def _labels_compatible(a: Any, b: Any) -> bool:
                a_norm = _norm_label(a)
                b_norm = _norm_label(b)
                if not a_norm or not b_norm:
                    return False
                if a_norm == b_norm:
                    return True
                a_singular = _re.sub(r"\bcells\b", "cell", a_norm)
                b_singular = _re.sub(r"\bcells\b", "cell", b_norm)
                if a_singular in b_singular or b_singular in a_singular:
                    return True
                a_family = _label_family(a_norm)
                b_family = _label_family(b_norm)
                return bool(a_family and a_family == b_family)

            def _cluster_annotation_qc_caveats(cid: str) -> List[Dict[str, Any]]:
                caveats: List[Dict[str, Any]] = []
                mask = (cluster_series == str(cid)).values
                n_in = int(mask.sum())
                if n_in <= 0:
                    return caveats

                def _obs_mean(col: str):
                    if col not in adata.obs.columns:
                        return None
                    try:
                        return float(_np.asarray(adata.obs.loc[mask, col], dtype=float).mean())
                    except Exception:
                        return None

                mt_mean = _obs_mean("pct_counts_mt")
                if mt_mean is not None and mt_mean >= 25.0:
                    caveats.append({
                        "type": "high_mt_cluster",
                        "severity": "strong",
                        "value": round(mt_mean, 3),
                        "confidence_cap": "low" if mt_mean >= 40.0 else "medium",
                        "message": f"Cluster mean mitochondrial percentage is high ({mt_mean:.2f}%).",
                    })
                if "qc_flag_high_mt" in adata.obs.columns:
                    try:
                        frac = float(adata.obs.loc[mask, "qc_flag_high_mt"].astype(bool).mean())
                        if frac >= 0.5:
                            caveats.append({
                                "type": "high_mt_flag_fraction",
                                "severity": "strong",
                                "value": round(frac, 4),
                                "confidence_cap": "medium",
                                "message": f"{frac:.1%} of cells are high-MT flagged.",
                            })
                    except Exception:
                        pass

                doublet_mean = _obs_mean("doublet_score")
                if doublet_mean is not None and doublet_mean >= 0.30:
                    caveats.append({
                        "type": "doublet_enriched_cluster",
                        "severity": "strong",
                        "value": round(doublet_mean, 4),
                        "confidence_cap": "low" if doublet_mean >= 0.50 else "medium",
                        "message": f"Cluster mean doublet score is high ({doublet_mean:.3f}).",
                    })
                if "predicted_doublet" in adata.obs.columns:
                    try:
                        frac = float(adata.obs.loc[mask, "predicted_doublet"].astype(bool).mean())
                        if frac >= 0.30:
                            caveats.append({
                                "type": "predicted_doublet_enriched_cluster",
                                "severity": "strong",
                                "value": round(frac, 4),
                                "confidence_cap": "low" if frac >= 0.80 else "medium",
                                "message": f"{frac:.1%} of cells are predicted doublets.",
                            })
                    except Exception:
                        pass

                lib_mean = _obs_mean("total_counts")
                gene_mean = _obs_mean("n_genes_by_counts")
                try:
                    global_lib = float(_np.nanmedian(_np.asarray(adata.obs["total_counts"], dtype=float)))
                    if lib_mean is not None and global_lib > 0 and lib_mean < 0.5 * global_lib:
                        caveats.append({
                            "type": "low_library_cluster",
                            "severity": "moderate",
                            "value": round(lib_mean / global_lib, 4),
                            "confidence_cap": "medium",
                            "message": "Cluster mean library size is <0.5x global median.",
                        })
                except Exception:
                    pass
                try:
                    global_genes = float(_np.nanmedian(_np.asarray(adata.obs["n_genes_by_counts"], dtype=float)))
                    if gene_mean is not None and global_genes > 0 and gene_mean < 0.5 * global_genes:
                        caveats.append({
                            "type": "low_gene_complexity_cluster",
                            "severity": "moderate",
                            "value": round(gene_mean / global_genes, 4),
                            "confidence_cap": "medium",
                            "message": "Cluster mean detected genes is <0.5x global median.",
                        })
                except Exception:
                    pass

                structure_qc = {}
                try:
                    structure_qc = (
                        adata.uns.get("cluster_structure_qc", {})
                        .get(str(cluster_key), {})
                    )
                except Exception:
                    structure_qc = {}
                if isinstance(structure_qc, dict):
                    evidence_by_cluster = structure_qc.get("structure_evidence_by_cluster") or {}
                    structure_record = evidence_by_cluster.get(str(cid)) or {}
                    synthesis = str(structure_record.get("synthesis") or "")
                    lean = str(structure_record.get("synthesis_lean") or "")
                    if str(cid) in [str(c) for c in structure_qc.get("synthesized_removal", []) or []] or lean == "remove":
                        caveats.append({
                            "type": "structure_qc_synthesized_removal",
                            "severity": "strong",
                            "confidence_cap": "low",
                            "message": "Structure QC synthesized this cluster for removal; do not assign a high-confidence biological label.",
                        })
                    elif str(cid) in [str(c) for c in structure_qc.get("conflicting", []) or []] or synthesis in {"conflicting", "obvious_but_structured", "inconclusive"} or lean == "review":
                        caveats.append({
                            "type": "structure_qc_review",
                            "severity": "moderate",
                            "confidence_cap": "medium",
                            "synthesis": synthesis or None,
                            "message": "Structure QC marked this cluster for review/conflict; confidence is capped unless resolved explicitly.",
                        })
                return caveats

            def _reference_proposal(ref_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
                real_entries = [
                    e for e in ref_entries
                    if isinstance(e, dict) and e.get("top_label")
                ]
                if not real_entries:
                    return {
                        "label": None,
                        "score": None,
                        "competing_labels": [],
                        "is_ambiguous": True,
                        "ambiguity_delta": None,
                        "warnings": ["No usable reference annotation label for this cluster."],
                    }

                def _key_priority(entry: Dict[str, Any]) -> int:
                    key = str(entry.get("annotation_key", "")).lower()
                    if "celltypist_majority" in key:
                        return 0
                    if "celltypist" in key:
                        return 1
                    if "scimilarity_representative" in key:
                        return 2
                    if "scimilarity" in key:
                        return 3
                    return 4

                preferred = sorted(real_entries, key=_key_priority)[0]
                proposed = str(preferred.get("top_label"))
                proposed_fraction = float(preferred.get("top_fraction") or 0.0)
                candidates: Dict[str, Dict[str, Any]] = {}
                warnings: List[str] = []

                for entry in real_entries:
                    key = str(entry.get("annotation_key") or "")
                    labels = entry.get("top_labels")
                    if not isinstance(labels, list) or not labels:
                        labels = [{
                            "label": entry.get("top_label"),
                            "fraction": entry.get("top_fraction", 0.0),
                            "count": entry.get("top_count", 0),
                        }]
                    for rank, label_entry in enumerate(labels[:5]):
                        label = str(label_entry.get("label") or "").strip()
                        if not label:
                            continue
                        try:
                            frac = float(label_entry.get("fraction") or 0.0)
                        except Exception:
                            frac = 0.0
                        existing = candidates.get(label)
                        if existing is None or frac > existing.get("score", 0.0):
                            candidates[label] = {
                                "label": label,
                                "score": round(frac, 4),
                                "source": key,
                                "rank": rank + 1,
                            }

                competing = []
                incompatible_top = False
                for label, candidate in candidates.items():
                    if label == proposed:
                        continue
                    compatible = _labels_compatible(proposed, label)
                    item = dict(candidate)
                    item["compatible_with_proposed"] = compatible
                    if not compatible:
                        competing.append(item)
                        if item.get("rank") == 1 and float(item.get("score") or 0.0) >= 0.35:
                            incompatible_top = True
                    elif float(item.get("score") or 0.0) >= 0.5:
                        # Keep high-support fine-grained variants visible for reporting without
                        # turning broad-lineage agreement into an avoidable ambiguity failure.
                        item["same_lineage_variant"] = True
                        competing.append(item)

                non_majority_entries = [
                    e for e in real_entries
                    if "majority" not in str(e.get("annotation_key", "")).lower()
                ]
                low_raw_support = False
                if non_majority_entries:
                    best_non_majority_support = max(
                        float(e.get("top_fraction") or 0.0)
                        for e in non_majority_entries
                        if _labels_compatible(proposed, e.get("top_label"))
                    ) if any(_labels_compatible(proposed, e.get("top_label")) for e in non_majority_entries) else 0.0
                    low_raw_support = best_non_majority_support > 0 and best_non_majority_support < 0.35
                    if low_raw_support:
                        warnings.append(
                            "Reference majority label has low raw per-cell support; use DEGs and reverse markers carefully."
                        )

                is_ambiguous = bool(incompatible_top or low_raw_support)
                return {
                    "label": proposed,
                    "score": round(proposed_fraction, 4),
                    "competing_labels": competing[:6],
                    "is_ambiguous": is_ambiguous,
                    "ambiguity_delta": None,
                    "warnings": warnings,
                    "source": preferred.get("annotation_key"),
                }

            for c in cluster_ids:
                summary: Dict[str, Any] = {
                    "cluster_id": c,
                    "n_cells": cluster_sizes[c],
                    "top_degs": [d["gene"] for d in top_degs_per_cluster.get(c, [])][:n_deg_genes],
                    "top_degs_detail": top_degs_per_cluster.get(c, []),
                    "reverse_lookup_genes": reverse_lookup_by_cluster.get(c, []),
                    "reverse_lookup_excluded_genes": reverse_lookup_excluded_by_cluster.get(c, []),
                    "qc_annotation_caveats": _cluster_annotation_qc_caveats(c),
                }
                if reference_annotation_summary.get(c):
                    summary["reference_annotations"] = reference_annotation_summary[c]
                if score_matrix.get(c):
                    ranked = sorted(score_matrix[c].items(), key=lambda kv: kv[1], reverse=True)
                    top_label, top_score = ranked[0]
                    runner_up = ranked[1] if len(ranked) > 1 else (None, 0.0)
                    delta = float(top_score - runner_up[1])
                    is_ambiguous = delta < ambiguity_threshold and top_score > 0
                    summary.update({
                        "proposed_label": top_label,
                        "proposed_score": round(top_score, 4),
                        "competing_labels": [
                            {"label": lbl, "score": round(s, 4), "delta_from_top": round(top_score - s, 4)}
                            for lbl, s in ranked[1:4] if s > 0
                        ],
                        "is_ambiguous": is_ambiguous,
                        "ambiguity_delta": round(delta, 4),
                    })
                    if is_ambiguous:
                        ambiguous_clusters.append(c)
                else:
                    ref_proposal = _reference_proposal(reference_annotation_summary.get(c, []))
                    if ref_proposal.get("label"):
                        is_ambiguous = bool(ref_proposal.get("is_ambiguous"))
                        summary.update({
                            "proposed_label": ref_proposal.get("label"),
                            "proposed_score": ref_proposal.get("score"),
                            "proposed_label_source": "reference_annotation",
                            "proposed_label_reference_key": ref_proposal.get("source"),
                            "competing_labels": ref_proposal.get("competing_labels", []),
                            "is_ambiguous": is_ambiguous,
                            "ambiguity_delta": ref_proposal.get("ambiguity_delta"),
                            "reference_confidence_warnings": ref_proposal.get("warnings", []),
                        })
                        if is_ambiguous:
                            ambiguous_clusters.append(c)
                    else:
                        summary.update({
                            "proposed_label": None,
                            "proposed_score": None,
                            "competing_labels": [],
                            "is_ambiguous": True,
                            "ambiguity_delta": None,
                            "reference_confidence_warnings": ref_proposal.get("warnings", []),
                        })
                        ambiguous_clusters.append(c)
                cluster_summaries.append(summary)

            panglaodb_queries: List[Dict[str, Any]] = []
            seen_queries: set = set()
            for entry in cluster_summaries:
                proposed = entry.get("proposed_label")
                if proposed and proposed not in seen_queries:
                    panglaodb_queries.append({
                        "cell_type": proposed,
                        "reason": f"proposed label for cluster {entry['cluster_id']}",
                    })
                    seen_queries.add(proposed)
                if entry.get("is_ambiguous"):
                    for comp in entry.get("competing_labels", []):
                        label = comp.get("label")
                        if label and label not in seen_queries:
                            panglaodb_queries.append({
                                "cell_type": label,
                                "reason": f"competing label for ambiguous cluster {entry['cluster_id']}",
                            })
                            seen_queries.add(label)
                for ref_entry in entry.get("reference_annotations", []):
                    label = ref_entry.get("top_label")
                    if label and label not in seen_queries:
                        panglaodb_queries.append({
                            "cell_type": label,
                            "reason": (
                                f"reference-derived candidate from {ref_entry.get('annotation_key')} "
                                f"for cluster {entry['cluster_id']}"
                            ),
                        })
                        seen_queries.add(label)

            panglaodb_reverse_queries: List[Dict[str, Any]] = []
            reverse_gene_to_clusters: Dict[str, List[str]] = {}
            for c, genes in reverse_lookup_by_cluster.items():
                for g in genes:
                    reverse_gene_to_clusters.setdefault(g, []).append(c)
            selected_reverse_genes: List[str] = []
            seen_reverse_genes: set = set()
            max_depth = max((len(v) for v in reverse_lookup_by_cluster.values()), default=0)
            for rank in range(max_depth):
                for c in cluster_ids:
                    genes = reverse_lookup_by_cluster.get(c, [])
                    if rank >= len(genes):
                        continue
                    gene = genes[rank]
                    if gene in seen_reverse_genes:
                        continue
                    selected_reverse_genes.append(gene)
                    seen_reverse_genes.add(gene)
                    if len(selected_reverse_genes) >= reverse_lookup_max_unique:
                        break
                if len(selected_reverse_genes) >= reverse_lookup_max_unique:
                    break
            for gene in selected_reverse_genes:
                query: Dict[str, Any] = {
                    "gene_symbol": gene,
                    "reason": (
                        "reverse marker lookup from top cluster DEGs; aggregate returned "
                        "cell types across multiple genes before choosing candidate labels"
                    ),
                    "clusters": reverse_gene_to_clusters.get(gene, []),
                }
                if panglaodb_species in {"Hs", "Mm"}:
                    query["species"] = panglaodb_species
                panglaodb_reverse_queries.append(query)

            proposal_fingerprint = _make_annotation_proposal_fingerprint(
                cluster_key=cluster_key,
                cluster_ids=cluster_ids,
                deg_key=deg_key,
                annotation_key=annotation_key,
                n_obs=int(adata.n_obs),
                adata=adata,
            )

            proposal = {
                "cluster_key": cluster_key,
                "annotation_key": annotation_key,
                "n_clusters": len(cluster_ids),
                "cluster_ids": cluster_ids,
                "clusters": cluster_summaries,
                "shared_markers": shared_markers,
                "ambiguous_clusters": ambiguous_clusters,
                "label_marker_lists_used": {k: list(v) for k, v in label_marker_lists.items()},
                "reference_annotation_keys": valid_reference_keys,
                "missing_reference_annotation_keys": missing_reference_keys,
                "reference_source_coverage": reference_source_coverage,
                "missing_reference_sources": missing_reference_sources,
                "reference_annotation_summary": reference_annotation_summary,
                "reverse_lookup_n_genes_per_cluster": reverse_lookup_n_genes,
                "reverse_lookup_max_unique_genes": reverse_lookup_max_unique,
                "reverse_lookup_exclude_patterns": reverse_exclude_patterns,
                "reverse_lookup_by_cluster": reverse_lookup_by_cluster,
                "reverse_lookup_excluded_by_cluster": reverse_lookup_excluded_by_cluster,
                "scoring_method": (
                    "normalized_expression_fraction"
                    if label_marker_lists
                    else ("reference_annotation_plus_deg" if valid_reference_keys else "deg_only")
                ),
                "ambiguity_threshold": ambiguity_threshold,
                "shared_marker_threshold": shared_marker_threshold,
                "panglaodb_queries_required": panglaodb_queries,
                "panglaodb_reverse_marker_queries_required": panglaodb_reverse_queries,
                "panglaodb_species": panglaodb_species,
                "deg_key": deg_key,
                "deg_method": deg_method,
                "fingerprint": proposal_fingerprint,
                "n_obs_at_propose": int(adata.n_obs),
            }
            try:
                adata.uns["annotation_proposal"] = proposal
            except Exception:
                pass

            # Clear staged evidence when the new proposal does not match what was
            # last staged. Without this, finalize_annotation will silently reuse
            # stale labels from a prior clustering whose cluster ids happen to
            # overlap. Carry forward only when the fingerprint matches exactly.
            try:
                prior_fp = adata.uns.get("annotation_evidence_fingerprint")
            except Exception:
                prior_fp = None
            evidence_cleared = False
            n_evidence_cleared = 0
            if prior_fp != proposal_fingerprint:
                try:
                    prior_evidence = adata.uns.get("annotation_evidence_summary")
                    if isinstance(prior_evidence, dict):
                        n_evidence_cleared = len(prior_evidence)
                except Exception:
                    n_evidence_cleared = 0
                try:
                    if "annotation_evidence_summary" in adata.uns:
                        del adata.uns["annotation_evidence_summary"]
                    if "annotation_evidence_fingerprint" in adata.uns:
                        del adata.uns["annotation_evidence_fingerprint"]
                except Exception:
                    try:
                        adata.uns["annotation_evidence_summary"] = {}
                    except Exception:
                        pass
                evidence_cleared = True

            result = {
                "status": "ok",
                "tool": "prepare_annotation",
                "cluster_key": cluster_key,
                "annotation_key": annotation_key,
                "n_clusters": len(cluster_ids),
                "n_ambiguous": len(ambiguous_clusters),
                "ambiguous_clusters": ambiguous_clusters,
                "shared_markers_flagged": shared_markers,
                "scoring_method": proposal["scoring_method"],
                "reference_annotation_keys": valid_reference_keys,
                "missing_reference_annotation_keys": missing_reference_keys,
                "reference_source_coverage": reference_source_coverage,
                "missing_reference_sources": missing_reference_sources,
                "reference_annotation_summary": reference_annotation_summary,
                "reference_annotation_notice": reference_annotation_notice,
                "clusters": cluster_summaries,
                "panglaodb_queries_required": panglaodb_queries,
                "panglaodb_reverse_marker_queries_required": panglaodb_reverse_queries,
                "reverse_lookup_n_genes_per_cluster": reverse_lookup_n_genes,
                "reverse_lookup_max_unique_genes": reverse_lookup_max_unique,
                "reverse_lookup_exclude_patterns": reverse_exclude_patterns,
                "panglaodb_species": panglaodb_species,
                "proposal_fingerprint": proposal_fingerprint,
                "stale_evidence_cleared": evidence_cleared,
                "n_stale_evidence_entries_cleared": n_evidence_cleared,
                "prior_evidence_fingerprint": prior_fp,
                "next_steps": [
                    "If CellTypist or Scimilarity is compatible but absent from reference_annotation_keys, run the missing reference annotation before finalizing, or record the concrete unavailability reason in staged evidence.",
                    "For each entry in panglaodb_queries_required, call bc_get_panglaodb_marker_genes (mouse or human as appropriate).",
                    "For each entry in panglaodb_reverse_marker_queries_required, call bc_get_panglaodb_marker_genes with gene_symbol and species; aggregate returned cell_type values per cluster across multiple genes.",
                    "Do not infer alternatives from a single top gene. Treat reverse-lookup labels as candidates only when supported by multiple DEG genes, then query those cell_type labels directly.",
                    "Compare PanglaoDB markers against each cluster's top_degs and any reference_annotations to confirm, revise, broaden, or reject each candidate label.",
                    "For ambiguous clusters, query competing labels too — the goal is adjudication, not confirmation.",
                    "Stage each cluster with label, supporting_genes, PanglaoDB evidence, CellTypist/Scimilarity agreement or conflict, competing labels considered, and explicit reasoning. For more than five clusters, use evidence_path instead of large inline JSON.",
                    "Before writing labels, call finalize_annotation with validate_only=true; only call it without validate_only after evidence validation passes.",
                ],
                "state": make_state(adata),
            }
            return _finalize_result(
                result, adata,
                dataset_changed=False,
                summary=(
                    f"Annotation proposal staged for {len(cluster_ids)} clusters "
                    f"({len(ambiguous_clusters)} ambiguous, {len(shared_markers)} shared markers flagged). "
                    f"Now query PanglaoDB for {len(panglaodb_queries)} candidate labels "
                    f"and {len(panglaodb_reverse_queries)} reverse marker genes."
                ),
                verification=_build_verification(
                    "passed",
                    "Annotation proposal written to adata.uns['annotation_proposal'].",
                    [
                        _check(
                            "proposal_stored",
                            "annotation_proposal" in adata.uns,
                            "annotation_proposal present on adata.uns.",
                        ),
                        _check(
                            "degs_available",
                            bool(top_degs_per_cluster),
                            f"Top DEGs extracted for {len(top_degs_per_cluster)} clusters.",
                        ),
                    ],
                ),
            )

        elif tool_name == "stage_annotation_evidence":
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            if adata is None:
                return _error_result(
                    tool="stage_annotation_evidence",
                    message="No data in memory.",
                    adata_obj=adata,
                    recovery_options=["Load data and run prepare_annotation first."],
                )

            proposal = adata.uns.get("annotation_proposal")
            if not isinstance(proposal, dict) or not proposal.get("cluster_ids"):
                return _error_result(
                    tool="stage_annotation_evidence",
                    message=(
                        "No annotation_proposal found on adata.uns. Stage evidence only after "
                        "prepare_annotation has created the cluster proposal."
                    ),
                    adata_obj=adata,
                    recovery_options=["Call prepare_annotation first, then stage evidence in batches."],
                )

            proposal_fp = proposal.get("fingerprint")
            staged_fp = adata.uns.get("annotation_evidence_fingerprint")
            if (
                isinstance(proposal_fp, str)
                and isinstance(staged_fp, str)
                and staged_fp != proposal_fp
            ):
                # Previously staged evidence belongs to a different proposal —
                # do not silently merge labels from a stale clustering. Wipe and
                # require the agent to re-stage against the current proposal.
                try:
                    if "annotation_evidence_summary" in adata.uns:
                        del adata.uns["annotation_evidence_summary"]
                    if "annotation_evidence_fingerprint" in adata.uns:
                        del adata.uns["annotation_evidence_fingerprint"]
                except Exception:
                    pass
                return _error_result(
                    tool="stage_annotation_evidence",
                    message=(
                        "Staged evidence fingerprint does not match the current proposal "
                        f"(staged={staged_fp}, proposal={proposal_fp}). Stale evidence was discarded; "
                        "re-stage against the current proposal."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        "Call stage_annotation_evidence again with evidence keyed by the current cluster ids.",
                        "If you intended to keep old labels, re-run prepare_annotation with the same clustering and re-stage explicitly.",
                    ],
                )

            incoming = tool_input.get("evidence_summary")
            evidence_source = "direct"
            evidence_path = tool_input.get("evidence_path")
            if incoming is None and evidence_path:
                resolved_path = _resolve_run_path(evidence_path, run_manager=run_manager, must_exist=True)
                if resolved_path is None:
                    return _error_result(
                        tool="stage_annotation_evidence",
                        message=f"evidence_path was provided but no file was found: {evidence_path}",
                        adata_obj=adata,
                        recovery_options=[
                            "Write the evidence JSON file inside the run directory (Path(output_dir) / 'evidence.json').write_text(json.dumps(...))",
                            "Then pass evidence_path with the bare filename — it is resolved against the run directory automatically.",
                            "Use register_artifact(path) inside run_code to surface the absolute path in the previous tool result.",
                        ],
                    )
                try:
                    incoming = json.loads(resolved_path.read_text())
                    evidence_source = f"file:{resolved_path}"
                except Exception as exc:
                    return _error_result(
                        tool="stage_annotation_evidence",
                        message=f"Could not parse evidence_path as JSON: {type(exc).__name__}: {exc}",
                        adata_obj=adata,
                        recovery_options=["Ensure the file contains a JSON object keyed by cluster id."],
                    )
            elif isinstance(incoming, str):
                try:
                    incoming = json.loads(incoming)
                    evidence_source = "json_string"
                except Exception as exc:
                    incoming_len = len(incoming)
                    likely_truncated = incoming_len > 6000 and type(exc).__name__ == "JSONDecodeError"
                    return _error_result(
                        tool="stage_annotation_evidence",
                        message=(
                            f"Could not parse evidence_summary JSON string: {type(exc).__name__}: {exc}. "
                            + (
                                "The inline evidence payload is large and likely truncated; write evidence to a JSON file and pass evidence_path instead."
                                if likely_truncated
                                else "Pass evidence_summary as a JSON object rather than a string when possible."
                            )
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "For more than five clusters, write evidence to a JSON file in the run directory and pass evidence_path.",
                            "For small batches, pass evidence_summary as an object, not a JSON-encoded string.",
                        ],
                        extra={
                            "inline_evidence_length": incoming_len,
                            "likely_truncated_inline_evidence": likely_truncated,
                            "recommended_next_call": {
                                "tool": "stage_annotation_evidence",
                                "arguments": {"evidence_path": "annotation_evidence.json", "replace": False},
                            },
                        },
                    )
            if not isinstance(incoming, dict) or not incoming:
                return _error_result(
                    tool="stage_annotation_evidence",
                    message="evidence_summary or evidence_path is required and must contain at least one cluster entry.",
                    adata_obj=adata,
                    recovery_options=[
                        "Pass evidence_summary={cluster_id: {label, panglaodb_queried, supporting_genes, confidence}} for one or more clusters.",
                        "For large payloads, write a JSON object to disk and pass evidence_path.",
                    ],
                )

            invalid_entries = [str(k) for k, v in incoming.items() if not isinstance(v, dict)]
            if invalid_entries:
                return _error_result(
                    tool="stage_annotation_evidence",
                    message=f"Evidence entries must be objects. Invalid cluster ids: {invalid_entries[:10]}",
                    adata_obj=adata,
                    recovery_options=["Wrap each cluster's label, genes, confidence, and reasoning in a dict."],
                )

            replace = bool(tool_input.get("replace", False))
            staged_existing = adata.uns.get("annotation_evidence_summary")
            if replace or not isinstance(staged_existing, dict):
                staged: Dict[str, Any] = {}
            else:
                staged = {str(k): v for k, v in staged_existing.items() if isinstance(v, dict)}

            normalized_incoming = {str(k): v for k, v in incoming.items()}
            staged.update(normalized_incoming)
            try:
                adata.uns["annotation_evidence_summary"] = staged
            except Exception:
                pass
            # Stamp the staged-evidence fingerprint so finalize_annotation and
            # subsequent stage calls can detect proposal/evidence drift.
            try:
                if isinstance(proposal_fp, str) and proposal_fp:
                    adata.uns["annotation_evidence_fingerprint"] = proposal_fp
            except Exception:
                pass

            proposal_clusters = [str(c) for c in proposal.get("cluster_ids", [])]
            covered = [c for c in proposal_clusters if c in staged]
            missing = [c for c in proposal_clusters if c not in staged]
            unknown = [c for c in staged.keys() if c not in proposal_clusters]

            # Validate the merged evidence so the model sees every issue at
            # staging time, not only when finalize_annotation is called. Run
            # auto-fixes (e.g., confidence cap from PanglaoDB support level)
            # and persist the corrected evidence back to adata.uns so the
            # corrections survive into finalize. We treat ``allow_partial`` as
            # true here regardless of caller intent — incomplete coverage is
            # the normal state during multi-batch staging and shouldn't be a
            # validation failure on its own (it's surfaced in ``coverage``).
            _stage_validation = _validate_annotation_evidence(
                adata=adata,
                proposal=proposal,
                evidence=staged,
                world_state=world_state,
                tool_input_unavailable_sources=(
                    tool_input.get("reference_source_unavailable")
                    or tool_input.get("reference_sources_unavailable")
                    or {}
                ),
                allow_partial=True,
                apply_auto_fixes=True,
            )
            stage_evidence_str = _stage_validation["evidence_str"]
            stage_auto_fixes = _stage_validation["auto_fixes"]
            stage_failures = _stage_validation["validation_failures"]
            stage_per_cluster = _stage_validation["per_cluster_validation"]
            # Persist any auto-fixed evidence so finalize sees the corrections.
            try:
                adata.uns["annotation_evidence_summary"] = stage_evidence_str
                staged = stage_evidence_str
            except Exception:
                pass

            n_covered_with_evidence = len(covered)
            n_proposal = len(proposal_clusters)
            full_coverage = n_covered_with_evidence == n_proposal and n_proposal > 0
            has_blocking_issues = bool(stage_failures)
            validation_status = "ok" if not has_blocking_issues else "issues"

            result = {
                "status": "ok",
                "tool": "stage_annotation_evidence",
                "n_entries_received": len(normalized_incoming),
                "n_entries_staged_total": len(staged),
                "evidence_source": evidence_source,
                "replace": replace,
                "coverage": {
                    "n_proposal_clusters": n_proposal,
                    "n_covered": n_covered_with_evidence,
                    "n_missing": len(missing),
                    "missing_clusters": missing[:50],
                    "unknown_clusters": unknown[:50],
                },
                "validation": {
                    "status": validation_status,
                    "n_issues": len(stage_failures),
                    "auto_fixes": stage_auto_fixes,
                    "validation_failures": stage_failures,
                    "validation_failures_grouped": (
                        _format_validation_failures_per_cluster(stage_failures)
                        if stage_failures else ""
                    ),
                    "per_cluster_validation": stage_per_cluster,
                },
                "next_steps": (
                    [
                        f"Fix the {len(stage_failures)} validation issue(s) above before calling finalize_annotation.",
                        "Resubmit stage_annotation_evidence with the corrected entries (only the failing clusters need to be re-sent).",
                    ]
                    if has_blocking_issues
                    else (
                        [
                            "Call finalize_annotation to write the labels; evidence_summary can be omitted if the staged evidence is complete.",
                        ]
                        if full_coverage
                        else [
                            "Continue staging evidence until all proposal clusters are covered.",
                            "Then call finalize_annotation; evidence_summary can be omitted if staged evidence is complete.",
                        ]
                    )
                ),
                "state": make_state(adata),
            }
            summary = (
                f"Staged annotation evidence for {len(normalized_incoming)} clusters "
                f"({n_covered_with_evidence}/{n_proposal} proposal clusters covered)"
            )
            if stage_auto_fixes:
                summary += f"; {len(stage_auto_fixes)} auto-fix(es) applied"
            if has_blocking_issues:
                summary += f"; {len(stage_failures)} validation issue(s) — see validation report"
            return _finalize_result(
                result, adata,
                dataset_changed=False,
                summary=summary + ".",
                verification=_build_verification(
                    "passed" if not has_blocking_issues else "warn",
                    (
                        "Annotation evidence staged and validated without writing cell-type labels."
                        if not has_blocking_issues
                        else "Annotation evidence staged but failed validation — labels not written."
                    ),
                    [
                        _check(
                            "evidence_staged",
                            isinstance(adata.uns.get("annotation_evidence_summary"), dict),
                            "adata.uns['annotation_evidence_summary'] present.",
                        ),
                        _check(
                            "labels_not_written",
                            True,
                            "stage_annotation_evidence did not write or overwrite annotation labels.",
                        ),
                        _check(
                            "validation_ok",
                            not has_blocking_issues,
                            (
                                "All staged evidence passes validation."
                                if not has_blocking_issues
                                else f"{len(stage_failures)} validation issue(s) need to be addressed before finalize."
                            ),
                        ),
                    ],
                ),
            )

        elif tool_name == "finalize_annotation":
            import pandas as _pd

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            if adata is None:
                return _error_result(
                    tool="finalize_annotation",
                    message="No data in memory.",
                    adata_obj=adata,
                    recovery_options=["Load data first."],
                )

            proposal = adata.uns.get("annotation_proposal")
            if not isinstance(proposal, dict) or not proposal.get("cluster_ids"):
                return _error_result(
                    tool="finalize_annotation",
                    message=(
                        "No annotation_proposal found on adata.uns. finalize_annotation requires "
                        "prepare_annotation to be run first."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        "Call prepare_annotation first to produce DEGs and a proposal.",
                        "Then query PanglaoDB for each candidate label and call finalize_annotation with evidence.",
                    ],
                )

            # Cross-check that the proposal still matches the live clustering.
            # We recompute the fingerprint from the live ``adata`` (cluster
            # membership included), not from the stored proposal fields, so
            # that same-label/same-size membership swaps after prepare are
            # detected. Using stored fields would be circular.
            proposal_fp = proposal.get("fingerprint")
            live_cluster_key = proposal.get("cluster_key", tool_input.get("cluster_key", "leiden"))
            current_fp = _make_annotation_proposal_fingerprint(
                cluster_key=live_cluster_key,
                cluster_ids=(
                    sorted(adata.obs[live_cluster_key].astype(str).unique().tolist())
                    if live_cluster_key in adata.obs.columns
                    else proposal.get("cluster_ids", [])
                ),
                deg_key=proposal.get("deg_key", "rank_genes_groups"),
                annotation_key=proposal.get("annotation_key", "cell_type"),
                n_obs=int(adata.n_obs),
                adata=adata,
            )
            if isinstance(proposal_fp, str) and proposal_fp and proposal_fp != current_fp:
                return _error_result(
                    tool="finalize_annotation",
                    message=(
                        "Annotation proposal fingerprint does not match the live AnnData "
                        f"(stored={proposal_fp}, live={current_fp}). Cluster membership or "
                        "cell set drifted since prepare_annotation. Re-run prepare_annotation "
                        "to regenerate the proposal before finalizing."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        "Call prepare_annotation again with the active cluster_key.",
                        "Re-stage evidence against the new proposal, then call finalize_annotation.",
                    ],
                )

            incoming_evidence = tool_input.get("evidence_summary")
            staged_evidence = adata.uns.get("annotation_evidence_summary")
            staged_fp = adata.uns.get("annotation_evidence_fingerprint")
            if (
                isinstance(staged_evidence, dict) and staged_evidence
                and isinstance(proposal_fp, str) and proposal_fp
                and isinstance(staged_fp, str) and staged_fp != proposal_fp
            ):
                # Refuse to finalize using evidence that was staged against a
                # different proposal. This is the central anti-stale-evidence
                # guard — without it, a re-clustering with overlapping ids
                # would silently inherit old labels.
                return _error_result(
                    tool="finalize_annotation",
                    message=(
                        "Staged annotation evidence fingerprint does not match the current proposal "
                        f"(staged={staged_fp}, proposal={proposal_fp}). Refusing to finalize on stale evidence."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        "Call stage_annotation_evidence to re-stage evidence against the current proposal.",
                        "Or pass evidence_summary directly in this finalize_annotation call.",
                    ],
                )
            evidence: Dict[str, Any] = {}
            used_staged_evidence = False
            if isinstance(staged_evidence, dict) and staged_evidence:
                evidence.update({str(k): v for k, v in staged_evidence.items() if isinstance(v, dict)})
                used_staged_evidence = True
            if isinstance(incoming_evidence, dict) and incoming_evidence:
                evidence.update({str(k): v for k, v in incoming_evidence.items()})
            if not evidence:
                return _error_result(
                    tool="finalize_annotation",
                    message=(
                        "Annotation evidence is required and must map every cluster to a label with evidence. "
                        "Pass evidence_summary directly or stage it first with stage_annotation_evidence."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        "Use stage_annotation_evidence in batches, then call finalize_annotation after all clusters are covered.",
                        "Or pass evidence_summary={cluster_id: {label, panglaodb_queried, supporting_genes, confidence}} for every cluster.",
                    ],
                )

            cluster_key = tool_input.get("cluster_key") or proposal.get("cluster_key", "leiden")
            annotation_key = tool_input.get("annotation_key") or proposal.get("annotation_key", "cell_type")
            overwrite = bool(tool_input.get("overwrite", False))
            allow_partial = bool(tool_input.get("allow_partial", False))
            validate_only = bool(tool_input.get("validate_only", False) or tool_input.get("dry_run", False))

            if cluster_key not in adata.obs.columns:
                return _error_result(
                    tool="finalize_annotation",
                    message=f"Cluster column '{cluster_key}' missing from adata.obs.",
                    adata_obj=adata,
                    recovery_options=["Re-run prepare_annotation with the correct cluster_key."],
                )
            if annotation_key in adata.obs.columns and not overwrite and not validate_only:
                return _error_result(
                    tool="finalize_annotation",
                    message=(
                        f"Column '{annotation_key}' already exists on adata.obs. "
                        "Pass overwrite=true to replace it, or use a different annotation_key."
                    ),
                    adata_obj=adata,
                    recovery_options=[
                        f"Retry with overwrite=true if you intend to replace '{annotation_key}'.",
                        "Pick a new annotation_key that does not collide with an existing column.",
                    ],
                )

            _validation_report = _validate_annotation_evidence(
                adata=adata,
                proposal=proposal,
                evidence=evidence,
                world_state=world_state,
                tool_input_unavailable_sources=(
                    tool_input.get("reference_source_unavailable")
                    or tool_input.get("reference_sources_unavailable")
                    or {}
                ),
                allow_partial=allow_partial,
                apply_auto_fixes=True,
            )
            validation_failures = _validation_report["validation_failures"]
            per_cluster_validation = _validation_report["per_cluster_validation"]
            auto_fixes = _validation_report["auto_fixes"]
            evidence_str = _validation_report["evidence_str"]
            proposal_clusters = _validation_report["proposal_clusters"]
            proposal_cluster_entries = _validation_report["proposal_cluster_entries"]
            ambiguous_set = _validation_report["ambiguous_set"]
            reference_keys = _validation_report["reference_keys"]
            missing_reference_sources = _validation_report["missing_reference_sources"]
            tool_recorded_unavailable_sources = _validation_report["tool_recorded_unavailable_sources"]
            manual_unavailable_sources = _validation_report["manual_unavailable_sources"]
            unavailable_reference_sources = _validation_report["unavailable_reference_sources"]
            unexplained_missing_sources = _validation_report["unexplained_missing_sources"]
            scimilarity_availability = _validation_report["scimilarity_availability"]
            any_panglaodb = _validation_report["any_panglaodb"]


            if validation_failures:
                return _error_result(
                    tool="finalize_annotation",
                    message="Evidence validation failed: " + _format_validation_failures_per_cluster(validation_failures),
                    adata_obj=adata,
                    recovery_options=[
                        "Query PanglaoDB for each proposed label and competing label.",
                        "Fill supporting_genes with the cluster DEGs that match PanglaoDB markers.",
                        "Use non-nuisance marker genes; MT/ribosomal/hemoglobin/MALAT1 genes cannot be sole label support.",
                        "Include source_synthesis and lower confidence for QC/structure-review clusters.",
                        "For ambiguous clusters, list the alternative labels you considered in competing_labels_considered.",
                        "Run missing CellTypist/Scimilarity sources; Scimilarity needs a prior tool-recorded blocker if it truly cannot run.",
                        "Resubmit finalize_annotation with the corrected evidence_summary.",
                    ],
                    extra={
                        "validation_failures": validation_failures,
                        "per_cluster_validation": per_cluster_validation,
                        "auto_fixes": auto_fixes,
                        "missing_reference_sources": missing_reference_sources,
                        "reference_source_unavailable_tool_recorded": tool_recorded_unavailable_sources,
                        "reference_source_unavailable_manual": manual_unavailable_sources,
                        "accepted_reference_source_unavailable": unavailable_reference_sources,
                        "scimilarity_availability": scimilarity_availability,
                    },
                )

            if validate_only:
                cluster_to_label_preview: Dict[str, str] = {}
                for cid in proposal_clusters:
                    ev = evidence_str.get(cid)
                    if isinstance(ev, dict) and isinstance(ev.get("label"), str):
                        cluster_to_label_preview[cid] = ev["label"]
                    elif allow_partial:
                        cluster_to_label_preview[cid] = "Unassigned"
                preview_series = adata.obs[cluster_key].astype(str).map(cluster_to_label_preview)
                mapping_has_missing = bool((not allow_partial) and preview_series.isna().any())
                if mapping_has_missing:
                    return _error_result(
                        tool="finalize_annotation",
                        message="Validation passed, but mapping would produce NaNs because some obs cluster ids are absent from the evidence.",
                        adata_obj=adata,
                        recovery_options=["Re-run prepare_annotation; verify the cluster_key matches the staged evidence."],
                        extra={
                            "validation_failures": [
                                "Mapping would produce NaNs — some cluster ids in obs were not in the evidence."
                            ],
                            "per_cluster_validation": per_cluster_validation,
                            "auto_fixes": auto_fixes,
                        },
                    )
                preview_series = preview_series.fillna("Unassigned")
                label_counts_preview: Dict[str, int] = {}
                for value in preview_series.astype(str).values:
                    label_counts_preview[value] = label_counts_preview.get(value, 0) + 1
                validation_payload_preview = {
                    "annotation_key": annotation_key,
                    "cluster_key": cluster_key,
                    "panglaodb_validated": True,
                    "validate_only": True,
                    "used_staged_evidence": used_staged_evidence,
                    "auto_fixes": auto_fixes,
                    "reference_annotation_keys": reference_keys,
                    "reference_source_coverage": proposal.get("reference_source_coverage") or {},
                    "missing_reference_sources": proposal.get("missing_reference_sources") or [],
                    "reference_source_unavailable": unavailable_reference_sources,
                    "reference_source_unavailable_tool_recorded": tool_recorded_unavailable_sources,
                    "reference_source_unavailable_manual": manual_unavailable_sources,
                    "unexplained_missing_reference_sources": unexplained_missing_sources,
                    "scimilarity_availability": scimilarity_availability,
                    "n_clusters_validated": len([c for c in per_cluster_validation.values() if c.get("panglaodb_queried")]),
                    "per_cluster_evidence": per_cluster_validation,
                    "label_counts": label_counts_preview,
                    "finalized": False,
                }
                return _finalize_result(
                    {
                        "status": "ok",
                        "tool": "finalize_annotation",
                        "validate_only": True,
                        "annotation_key": annotation_key,
                        "cluster_key": cluster_key,
                        "n_clusters_validated": len(cluster_to_label_preview),
                        "label_counts": label_counts_preview,
                        "annotation_validation": validation_payload_preview,
                        "used_staged_evidence": used_staged_evidence,
                        "auto_fixes": auto_fixes,
                        "state": make_state(adata),
                    },
                    adata,
                    dataset_changed=False,
                    summary=(
                        f"Annotation evidence validation passed for {len(cluster_to_label_preview)} clusters. "
                        "No labels were written because validate_only=true."
                    ),
                    verification=_build_verification(
                        "passed",
                        "Annotation evidence passed validation without writing labels.",
                        [
                            _check("validation_only", True, "validate_only=true; annotation column was not written."),
                            _check("panglaodb_evidence_present", any_panglaodb, "At least one cluster has PanglaoDB-backed evidence."),
                        ],
                    ),
                )

            cluster_to_label: Dict[str, str] = {}
            for cid in proposal_clusters:
                ev = evidence_str.get(cid)
                if isinstance(ev, dict) and isinstance(ev.get("label"), str):
                    cluster_to_label[cid] = ev["label"]
                elif allow_partial:
                    cluster_to_label[cid] = "Unassigned"

            try:
                series = adata.obs[cluster_key].astype(str).map(cluster_to_label)
                if not allow_partial and series.isna().any():
                    return _error_result(
                        tool="finalize_annotation",
                        message="Mapping produced NaNs — some cluster ids in obs were not in the evidence.",
                        adata_obj=adata,
                        recovery_options=["Re-run prepare_annotation; verify the cluster_key matches."],
                    )
                series = series.fillna("Unassigned")
                adata.obs[annotation_key] = _pd.Categorical(series.values)
            except Exception as e:
                return _error_result(
                    tool="finalize_annotation",
                    message=f"Failed to write annotation column: {e}",
                    adata_obj=adata,
                    recovery_options=["Inspect cluster_key dtype and evidence_summary keys; ensure they're strings."],
                )

            label_counts: Dict[str, int] = {}
            for v in adata.obs[annotation_key].astype(str).values:
                label_counts[v] = label_counts.get(v, 0) + 1

            validation_payload = {
                "annotation_key": annotation_key,
                "cluster_key": cluster_key,
                "panglaodb_validated": True,
                "used_staged_evidence": used_staged_evidence,
                "auto_fixes": auto_fixes,
                "reference_annotation_keys": reference_keys,
                "reference_source_coverage": proposal.get("reference_source_coverage") or {},
                "missing_reference_sources": proposal.get("missing_reference_sources") or [],
                "reference_source_unavailable": unavailable_reference_sources,
                "reference_source_unavailable_tool_recorded": tool_recorded_unavailable_sources,
                "reference_source_unavailable_manual": manual_unavailable_sources,
                "unexplained_missing_reference_sources": unexplained_missing_sources,
                "scimilarity_availability": scimilarity_availability,
                "n_clusters_validated": len([c for c in per_cluster_validation.values() if c.get("panglaodb_queried")]),
                "per_cluster_evidence": per_cluster_validation,
                "label_counts": label_counts,
                "finalized": True,
            }
            try:
                adata.uns["annotation_validation"] = validation_payload
            except Exception:
                pass

            artifacts: List[Dict[str, Any]] = []
            if run_manager:
                def _fmt_report_value(value, limit=120):
                    if value is None:
                        return "NA"
                    if isinstance(value, dict):
                        parts = [f"{k}: {v}" for k, v in value.items() if v not in (None, "", [], {})]
                        text = "; ".join(parts) if parts else "none"
                    elif isinstance(value, list):
                        text = ", ".join(str(v) for v in value) if value else "none"
                    else:
                        text = str(value)
                    text = text.replace("|", "\\|").replace("\n", " ")
                    return text[:limit] + ("..." if len(text) > limit else "")

                annotation_json_path = run_manager.write_json_report(
                    f"annotation_validation_{annotation_key}",
                    validation_payload,
                )
                artifacts.append(
                    _artifact_payload(
                        annotation_json_path,
                        role="annotation_validation_json",
                        metadata={"annotation_key": annotation_key, "cluster_key": cluster_key},
                    )
                )

                md_lines = [
                    f"# Annotation Validation - `{annotation_key}`",
                    "",
                    "This report summarizes the finalized consensus annotation evidence. Automated reference "
                    "labels are treated as candidates; final labels are written only after DEG and external "
                    "marker validation.",
                    "",
                    "## Summary",
                    "",
                    f"- Cluster key: `{cluster_key}`",
                    f"- Clusters labeled: **{len(cluster_to_label)}**",
                    f"- Unique labels: **{len(label_counts)}**",
                    f"- Used staged evidence: **{used_staged_evidence}**",
                    f"- Reference annotation columns: **{_fmt_report_value(reference_keys)}**",
                    f"- Missing reference sources: **{_fmt_report_value(proposal.get('missing_reference_sources') or [])}**",
                    "",
                    "## Cell-Type Counts",
                    "",
                    "| Label | Cells |",
                    "|---|---:|",
                ]
                for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                    md_lines.append(f"| {_fmt_report_value(label)} | {int(count)} |")

                md_lines.extend([
                    "",
                    "## Per-Cluster Annotation Evidence",
                    "",
                    "| Cluster | Final label | Confidence | Support level | QC cap | Supporting genes | PanglaoDB label used | Competing labels considered | Reference support/conflict |",
                    "|---|---|---|---|---|---|---|---|---|",
                ])
                for cid in proposal_clusters:
                    ev = per_cluster_validation.get(cid, {})
                    ref_bits = []
                    if ev.get("reference_annotation_support"):
                        ref_bits.append("support: " + _fmt_report_value(ev.get("reference_annotation_support"), limit=90))
                    if ev.get("reference_annotation_conflicts"):
                        ref_bits.append("conflict: " + _fmt_report_value(ev.get("reference_annotation_conflicts"), limit=90))
                    md_lines.append(
                        "| "
                        + " | ".join([
                            _fmt_report_value(cid),
                            _fmt_report_value(ev.get("label")),
                            _fmt_report_value(ev.get("confidence")),
                            _fmt_report_value(ev.get("panglaodb_support_level")),
                            _fmt_report_value(ev.get("qc_confidence_cap", "none")),
                            _fmt_report_value(ev.get("supporting_genes"), limit=80),
                            _fmt_report_value(ev.get("panglaodb_label_used", "same/as stated")),
                            _fmt_report_value(ev.get("competing_labels_considered"), limit=90),
                            _fmt_report_value("; ".join(ref_bits) if ref_bits else "not recorded", limit=140),
                        ])
                        + " |"
                    )

                md_lines.extend(["", "## Reasoning By Cluster", ""])
                for cid in proposal_clusters:
                    ev = per_cluster_validation.get(cid, {})
                    proposal_entry = proposal_cluster_entries.get(cid, {})
                    top_degs = proposal_entry.get("top_degs") or []
                    md_lines.append(f"### Cluster {cid}: {ev.get('label', 'Unassigned')}")
                    md_lines.append("")
                    md_lines.append(f"- Confidence: **{ev.get('confidence', 'NA')}**.")
                    if ev.get("panglaodb_support_level"):
                        md_lines.append(f"- PanglaoDB support level: **{ev.get('panglaodb_support_level')}**.")
                    if ev.get("qc_annotation_caveats"):
                        md_lines.append(f"- QC/structure caveats: {_fmt_report_value(ev.get('qc_annotation_caveats'), limit=300)}.")
                    md_lines.append(f"- Supporting genes: {_fmt_report_value(ev.get('supporting_genes'))}.")
                    if ev.get("non_nuisance_supporting_genes"):
                        md_lines.append(f"- Non-nuisance support: {_fmt_report_value(ev.get('non_nuisance_supporting_genes'))}.")
                    if top_degs:
                        md_lines.append(f"- Top DEGs reviewed: {_fmt_report_value(top_degs[:12])}.")
                    if ev.get("source_synthesis"):
                        md_lines.append(f"- Source synthesis: {_fmt_report_value(ev.get('source_synthesis'), limit=400)}.")
                    if ev.get("reference_annotation_support"):
                        md_lines.append(f"- Reference support: {_fmt_report_value(ev.get('reference_annotation_support'), limit=300)}.")
                    if ev.get("reference_annotation_conflicts"):
                        md_lines.append(f"- Reference conflicts: {_fmt_report_value(ev.get('reference_annotation_conflicts'), limit=300)}.")
                    if ev.get("competing_labels_considered"):
                        md_lines.append(f"- Competing labels considered: {_fmt_report_value(ev.get('competing_labels_considered'))}.")
                    if ev.get("reverse_marker_support"):
                        md_lines.append(f"- Reverse marker support: {_fmt_report_value(ev.get('reverse_marker_support'), limit=300)}.")
                    md_lines.append(f"- Reasoning: {ev.get('reasoning', 'No reasoning recorded')}")
                    md_lines.append("")

                md_lines.extend([
                    "## Reporting Notes",
                    "",
                    "- CellTypist majority-voted cluster labels are not the same as raw per-cell unanimity; use raw prediction fractions when claiming agreement strength.",
                    "- If `panglaodb_label_used` is broader than the final label, the report should say the broad lineage was externally validated and the fine subtype was resolved from DEGs/reference labels.",
                    "- This Markdown report is derived from `adata.uns['annotation_validation']`; the companion JSON preserves the full machine-readable evidence.",
                    "",
                ])
                annotation_md_path = run_manager.write_text_report(
                    f"annotation_validation_{annotation_key}_summary",
                    "\n".join(md_lines),
                    ext="md",
                )
                artifacts.append(
                    _artifact_payload(
                        annotation_md_path,
                        role="annotation_validation_summary",
                        metadata={"annotation_key": annotation_key, "cluster_key": cluster_key},
                    )
                )

            result = {
                "status": "ok",
                "tool": "finalize_annotation",
                "annotation_key": annotation_key,
                "cluster_key": cluster_key,
                "n_clusters_labeled": len(cluster_to_label),
                "label_counts": label_counts,
                "cell_type_breakdown": label_counts,
                "annotation_validation": validation_payload,
                "used_staged_evidence": used_staged_evidence,
                "auto_fixes": auto_fixes,
                "annotation_validation_json": (
                    annotation_json_path if run_manager else None
                ),
                "annotation_validation_markdown": (
                    annotation_md_path if run_manager else None
                ),
                "state": make_state(adata),
            }
            return _finalize_result(
                result, adata,
                dataset_changed=True,
                summary=(
                    f"Wrote final annotation '{annotation_key}' for {len(cluster_to_label)} clusters "
                    f"({len(label_counts)} unique labels) with external marker evidence."
                ),
                artifacts_created=[artifact for artifact in artifacts if artifact],
                verification=_build_verification(
                    "passed",
                    "Annotation finalized with external marker evidence.",
                    [
                        _check(
                            "annotation_column_written",
                            annotation_key in adata.obs.columns,
                            f"adata.obs['{annotation_key}'] present.",
                        ),
                        _check(
                            "validation_recorded",
                            "annotation_validation" in adata.uns,
                            "adata.uns['annotation_validation'] recorded.",
                        ),
                        _check(
                            "panglaodb_evidence_present",
                            any_panglaodb,
                            "At least one cluster has PanglaoDB-backed evidence.",
                        ),
                    ],
                ),
            )

        else:
            return _error_result(
                tool=tool_name,
                message=f"Unknown tool: {tool_name}",
                adata_obj=adata,
                recovery_options=["Check tool name spelling or use inspect_session to list available tools."],
            )

    except Exception as e:
        return _error_result(
            tool=tool_name,
            message=str(e),
            adata_obj=adata,
            recovery_options=["Review the error and tool input parameters before retrying."],
            extra={"error_type": type(e).__name__},
        )
