"""
Unified agent world state for scagent.

This module keeps the agent's operational picture in one place so decisions,
artifacts, and verification results do not depend on prompt memory alone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import importlib.util
import json
import os


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifact_id_from_path(path: str) -> str:
    normalized = os.path.abspath(path)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    stem = Path(normalized).stem or "artifact"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in stem)
    return f"{safe_stem}_{digest}"


def _stage_from_processing(processing: Dict[str, Any]) -> str:
    if not processing:
        return "uninitialized"
    if processing.get("has_celltypes"):
        return "annotated"
    if processing.get("has_clusters"):
        return "clustered"
    if processing.get("has_umap") or processing.get("has_neighbors"):
        return "embedded"
    if processing.get("is_normalized") or processing.get("has_hvg"):
        return "normalized"
    if processing.get("has_qc_metrics") or processing.get("has_doublets"):
        return "qc"
    if processing.get("has_raw_counts"):
        return "loaded"
    return "unknown"


@dataclass
class ArtifactRecord:
    artifact_id: str
    path: str
    kind: str
    role: str = "artifact"
    source_tool: str = ""
    created_at: str = field(default_factory=_utc_now_iso)
    exists: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    review_count: int = 0
    last_reviewed_at: Optional[str] = None
    last_review_question: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArtifactRecord":
        return cls(**payload)

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        kind: str,
        role: str = "artifact",
        source_tool: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ArtifactRecord":
        normalized = os.path.abspath(path)
        return cls(
            artifact_id=artifact_id_from_path(normalized),
            path=normalized,
            kind=kind,
            role=role,
            source_tool=source_tool,
            exists=os.path.exists(normalized),
            metadata=metadata or {},
        )


@dataclass
class DecisionRecord:
    decision_id: str
    key: str
    policy_action: str
    status: str
    rationale: str
    recommended_value: Any = None
    applied_value: Any = None
    impact: str = "medium"
    candidates: List[Any] = field(default_factory=list)
    created_by_tool: str = ""
    created_at: str = field(default_factory=_utc_now_iso)
    resolved_at: Optional[str] = None
    user_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DecisionRecord":
        return cls(**payload)


@dataclass
class StateDelta:
    tool: str
    summary: str
    dataset_changed: bool
    stage_before: str
    stage_after: str
    changed_flags: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    status: str
    summary: str
    checks: List[Dict[str, Any]] = field(default_factory=list)
    recovery_options: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VerificationResult":
        return cls(**payload)


@dataclass
class AgentWorldState:
    created_at: str = field(default_factory=_utc_now_iso)
    active_request: str = ""
    analysis_stage: str = "uninitialized"
    data_summary: Dict[str, Any] = field(default_factory=dict)
    metadata_candidates: List[Dict[str, Any]] = field(default_factory=list)
    clustering_registry: List[Dict[str, Any]] = field(default_factory=list)
    annotation_sources: List[str] = field(default_factory=list)
    cluster_qc_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    outstanding_decisions: List[DecisionRecord] = field(default_factory=list)
    resolved_decisions: List[DecisionRecord] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    context_hints: List[str] = field(default_factory=list)
    annotation_validation: Dict[str, Any] = field(default_factory=dict)
    last_action: Dict[str, Any] = field(default_factory=dict)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    latest_verification: Dict[str, Any] = field(default_factory=dict)
    # Permanent record of key parameters and results from each major analysis step.
    # Never trimmed from the system prompt — used for notebook generation, method
    # sections, and follow-up questions about what was done.
    step_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Inspect-data cache: store a structural fingerprint of the last adata
        # we synced from so we can skip re-running inspect_data when nothing has
        # changed.  These are not dataclass fields — they stay out of snapshots.
        self._inspect_cache_key: Optional[tuple] = None
        self._inspect_cache_state = None  # cached DataState object

    def _derive_capabilities(self, adata) -> Dict[str, Any]:
        processing = self.data_summary.get("processing", {})
        cluster_qc = self.data_summary.get("cluster_qc", {})
        cluster_keys = [
            record.get("key")
            for record in self.clustering_registry
            if record.get("key")
        ]
        annotation_keys = []
        if adata is not None:
            semantic_roles = self.data_summary.get("semantic_obs_roles", {})
            annotation_keys = [
                candidate.get("column")
                for candidate in semantic_roles.get("cell_type", [])
                if candidate.get("column")
            ]
            for column in adata.obs.columns:
                if any(token in column.lower() for token in ("celltyp", "scimilar", "annotation", "label")):
                    if column not in annotation_keys:
                        annotation_keys.append(column)
        deg_available = bool(adata is not None and "rank_genes_groups" in adata.uns)
        primary_cluster_key = self.data_summary.get("cluster_key")
        obs_columns = list(adata.obs.columns) if adata is not None else []
        plot_color_candidates: List[str] = []
        preferred = [
            primary_cluster_key,
            "sample_id",
            "batch",
            "sample",
            "pct_counts_mt",
            "total_counts",
            "n_genes_by_counts",
        ]
        for candidate in preferred + obs_columns:
            if candidate and candidate in obs_columns and candidate not in plot_color_candidates:
                plot_color_candidates.append(candidate)
        # Build available_actions list - what the agent CAN do right now
        available_actions: List[str] = []
        blocked_actions: List[Dict[str, str]] = []

        if adata is not None:
            # Always available
            available_actions.extend(["run_code", "inspect_data", "save_data", "pause_and_ask"])

            # QC
            if not processing.get("has_qc_metrics"):
                available_actions.append("run_qc")

            # Normalization - available if we have raw counts and not yet normalized
            if processing.get("has_raw_counts") and not processing.get("is_normalized"):
                available_actions.append("normalize_and_hvg")
            elif not processing.get("has_raw_counts"):
                blocked_actions.append({"action": "normalize_and_hvg", "needs": "raw counts"})

            # Dimensionality reduction
            if processing.get("is_normalized") or processing.get("has_hvg"):
                available_actions.extend(["run_pca"])
                if processing.get("has_pca"):
                    available_actions.append("run_neighbors")
                if processing.get("has_neighbors"):
                    available_actions.append("run_umap")
            else:
                blocked_actions.append({"action": "run_pca", "needs": "normalized data with HVGs"})

            # Clustering
            if processing.get("has_neighbors"):
                available_actions.extend(["run_clustering", "compare_clusterings"])
            elif processing.get("has_pca"):
                available_actions.extend(["run_clustering", "compare_clusterings"])  # PhenoGraph works on PCA
            else:
                blocked_actions.append({"action": "run_clustering", "needs": "neighbors graph or PCA"})

            # Annotation
            if processing.get("has_clusters"):
                available_actions.extend(["run_celltypist", "run_scimilarity"])
                if processing.get("has_qc_metrics"):
                    available_actions.append("run_cluster_qc")
            else:
                blocked_actions.append({"action": "run_celltypist", "needs": "clustering"})
                blocked_actions.append({"action": "run_scimilarity", "needs": "clustering"})
                blocked_actions.append({"action": "run_cluster_qc", "needs": "QC metrics and clustering"})

            # DEG
            if processing.get("has_clusters") or self.annotation_sources:
                available_actions.append("run_deg")
            else:
                blocked_actions.append({"action": "run_deg", "needs": "clusters or annotations"})

            # Pseudobulk DEG — needs groups to aggregate AND raw counts for DESeq2
            pseudobulk_available = (
                importlib.util.find_spec("scagent.analysis.pseudobulk") is not None
            )
            if (
                pseudobulk_available
                and (processing.get("has_clusters") or self.annotation_sources)
                and processing.get("has_raw_counts")
            ):
                available_actions.append("run_pseudobulk_deg")
            elif not pseudobulk_available:
                blocked_actions.append({
                    "action": "run_pseudobulk_deg",
                    "needs": "pseudobulk implementation is not installed in this checkout",
                })
            elif not (processing.get("has_clusters") or self.annotation_sources):
                blocked_actions.append({"action": "run_pseudobulk_deg", "needs": "clusters or annotations"})
            else:
                blocked_actions.append({"action": "run_pseudobulk_deg", "needs": "raw counts layer (required for DESeq2 aggregation)"})

            # GSEA
            if deg_available:
                available_actions.append("run_gsea")
            else:
                blocked_actions.append({"action": "run_gsea", "needs": "DEG results"})

            # Plotting
            if processing.get("has_umap"):
                available_actions.append("generate_figure")

            # Batch correction
            if processing.get("is_normalized"):
                available_actions.append("run_batch_correction")

            # Cell query — available when Scimilarity embedding is present
            _has_scimilarity_emb = adata is not None and "X_scimilarity" in adata.obsm
            if _has_scimilarity_emb:
                available_actions.append("query_cells")
            else:
                blocked_actions.append({"action": "query_cells", "needs": "X_scimilarity embedding (run run_scimilarity first)"})

            # Gene signature scoring — needs normalized data
            if processing.get("is_normalized"):
                available_actions.append("score_gene_signature")
            else:
                blocked_actions.append({"action": "score_gene_signature", "needs": "normalized data"})

            # Spectra — needs annotations/clusters for cell_type_key, normalized data
            spectra_available = importlib.util.find_spec("scagent.analysis.spectra") is not None
            if (
                spectra_available
                and (processing.get("has_clusters") or self.annotation_sources)
                and processing.get("is_normalized")
            ):
                available_actions.append("run_spectra")
            elif not spectra_available:
                blocked_actions.append({
                    "action": "run_spectra",
                    "needs": "Spectra implementation is not installed in this checkout",
                })
            else:
                blocked_actions.append({"action": "run_spectra", "needs": "normalized data and cell type labels or clusters"})

            # Integration scoring and benchmarking
            _has_corrected_rep = adata is not None and (
                any(k in adata.obsm for k in ("X_pca_harmony", "X_scVI", "X_scanorama"))
                or adata.uns.get("bbknn_batch_key") is not None
            )
            _has_batch_key = bool(self.data_summary.get("batch_key"))
            if _has_batch_key and (processing.get("has_pca") or _has_corrected_rep):
                available_actions.append("score_integration")
            # scib benchmark needs a label_key (clusters or annotations) + corrected embedding
            if _has_corrected_rep and _has_batch_key and (
                processing.get("has_clusters") or self.annotation_sources
            ):
                available_actions.append("benchmark_integration")

        return {
            "has_raw_counts": bool(processing.get("has_raw_counts")),
            "has_normalized_matrix": bool(processing.get("is_normalized")),
            "has_hvg": bool(processing.get("has_hvg")),
            "has_pca": bool(processing.get("has_pca")),
            "has_neighbors": bool(processing.get("has_neighbors")),
            "has_umap": bool(processing.get("has_umap")),
            "has_clusters": bool(processing.get("has_clusters")),
            "has_annotations": bool(self.annotation_sources),
            "deg_available": deg_available,
            "cluster_keys": cluster_keys,
            "primary_cluster_key": primary_cluster_key,
            "annotation_keys": annotation_keys,
            "obs_columns": obs_columns[:200],
            "plot_color_candidates": plot_color_candidates[:10],
            "can_plot_umap": bool(processing.get("has_umap")),
            "can_plot_cluster_umap": bool(processing.get("has_umap") and processing.get("has_clusters")),
            "can_run_clustering": bool(processing.get("has_neighbors") or processing.get("has_pca")),
            "can_run_cluster_qc": bool(processing.get("has_qc_metrics") and processing.get("has_clusters")),
            "cluster_qc_fresh": bool(cluster_qc.get("fresh")),
            "cluster_qc_status": cluster_qc.get("status"),
            "can_run_annotation": bool(processing.get("has_clusters")),
            "can_run_deg": bool(processing.get("has_clusters") or self.annotation_sources),
            "can_review_markers": deg_available,
            # NEW: Explicit action availability for LLM reasoning
            "available_actions": available_actions,
            "blocked_actions": blocked_actions,
            "run_code_note": "run_code is ALWAYS available for custom plots, filtering, or any valid analysis",
        }

    def _batch_strategy_summary(self, state, processing: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize whether a multi-partition dataset has an explicit batch plan."""
        batch_key = self.get_confirmed_value("batch_key") or state.batch_key
        n_batches = int(state.n_batches or 0)
        if (not batch_key or n_batches < 2) and state.metadata_candidates:
            top_candidate = state.metadata_candidates[0]
            batch_key = batch_key or top_candidate.column
            n_batches = max(n_batches, int(top_candidate.n_unique or 0))

        if not batch_key or n_batches < 2:
            return {
                "status": "not_applicable",
                "batch_key": batch_key,
                "n_batches": n_batches,
                "reason": "No multi-group sample/batch/donor partition was detected.",
            }

        if state.batch_correction_applied:
            return {
                "status": "corrected",
                "batch_key": batch_key,
                "n_batches": n_batches,
                "method": state.batch_correction_method or "unknown",
                "reason": "A batch-corrected representation or graph is present.",
            }

        selected_strategy = self.get_confirmed_value("multi_sample_strategy")
        if selected_strategy:
            strategy_action = (
                selected_strategy.get("action")
                if isinstance(selected_strategy, dict)
                else selected_strategy
            )
            strategy_details = (
                selected_strategy.get("details")
                if isinstance(selected_strategy, dict)
                else None
            )
            strategy_summaries = {
                "investigate_integration": (
                    "investigate_requested",
                    "Run an uncorrected first pass (PCA, neighbors, UMAP, clustering) and assess "
                    "batch mixing. Do not integrate yet: once the first pass produces a clustering, "
                    "the runtime re-opens the multi_sample_strategy decision so the user picks "
                    "integrate/keep/separate based on the diagnostic.",
                ),
                "integrate_scvi": (
                    "scvi_requested",
                    "Confirm the sample key if needed, then integrate with scVI before the final graph and clustering.",
                ),
                "keep_unintegrated": (
                    "uncorrected_requested",
                    "Keep samples combined in one analysis without batch correction.",
                ),
                "analyze_separately": (
                    "separate_analysis_requested",
                    "Run sample-specific analyses rather than constructing a shared integrated representation.",
                ),
                "custom": (
                    "custom_strategy",
                    strategy_details or "Follow the user's custom multi-sample strategy.",
                ),
            }
            status, next_action = strategy_summaries.get(
                strategy_action,
                ("strategy_selected", f"Follow the selected strategy: {strategy_action}."),
            )
            return {
                "status": status,
                "batch_key": batch_key,
                "n_batches": n_batches,
                "method": "scvi" if strategy_action == "integrate_scvi" else None,
                "selected_strategy": selected_strategy,
                "reason": "The user explicitly selected how the samples should be handled.",
                "next_action": next_action,
            }

        if processing.get("has_neighbors") or processing.get("has_umap") or processing.get("has_clusters"):
            status = "needs_review"
            next_action = "Present the sample-handling choice with pause_and_ask (investigate / integrate with scVI / keep uncorrected / analyze separately), then end the turn."
            reason = (
                "A multi-group batch key is present, but neighbors/UMAP/clustering already exist "
                "without a user-selected sample-handling strategy."
            )
        elif processing.get("has_pca"):
            status = "needs_decision"
            next_action = "Present the sample-handling strategy choice with pause_and_ask; do not correct automatically."
            reason = "A multi-group sample-like key is present after PCA, but no strategy has been selected."
        elif processing.get("is_normalized") or processing.get("has_hvg"):
            status = "pending_pca"
            next_action = "Use the user's selected strategy before constructing the final graph."
            reason = "A multi-group sample-like key is present, but its presence alone does not justify correction."
        else:
            status = "pending_preprocessing"
            next_action = "Present the strategy choice with pause_and_ask before any preprocessing; preprocessing is blocked until a strategy is selected."
            reason = "A multi-group sample-like key is present early in the workflow; correction is opt-in."

        return {
            "status": status,
            "batch_key": batch_key,
            "n_batches": n_batches,
            "method": None,
            "reason": reason,
            "next_action": next_action,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "active_request": self.active_request,
            "analysis_stage": self.analysis_stage,
            "data_summary": self.data_summary,
            "metadata_candidates": self.metadata_candidates,
            "clustering_registry": self.clustering_registry,
            "annotation_sources": self.annotation_sources,
            "cluster_qc_registry": self.cluster_qc_registry,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "outstanding_decisions": [decision.to_dict() for decision in self.outstanding_decisions],
            "resolved_decisions": [decision.to_dict() for decision in self.resolved_decisions],
            "user_preferences": self.user_preferences,
            "context_hints": self.context_hints,
            "annotation_validation": self.annotation_validation,
            "last_action": self.last_action,
            "recent_events": self.recent_events,
            "latest_verification": self.latest_verification,
            "step_log": self.step_log,
        }

    def snapshot(self) -> Dict[str, Any]:
        # Strip semantic_obs_roles from the LLM snapshot — the LLM uses
        # obs_columns_detail (also in data_summary) for role inference instead.
        # semantic_obs_roles stays in data_summary for _derive_capabilities().
        data_summary_for_llm = {
            k: v for k, v in self.data_summary.items() if k != "semantic_obs_roles"
        }
        return {
            "analysis_stage": self.analysis_stage,
            "active_request": self.active_request,
            "data_summary": data_summary_for_llm,
            "metadata_candidates": self.metadata_candidates[:3],
            "clustering_registry": self.clustering_registry[:6],
            "annotation_sources": self.annotation_sources,
            "cluster_qc": self.data_summary.get("cluster_qc", {}),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts[-8:]],
            "outstanding_decisions": [decision.to_dict() for decision in self.outstanding_decisions[-5:]],
            "resolved_decisions": [decision.to_dict() for decision in self.resolved_decisions[-5:]],
            "user_preferences": self.user_preferences,
            "context_hints": self.context_hints[-8:],
            "annotation_validation": self.annotation_validation,
            "latest_verification": self.latest_verification,
            "last_action": self.last_action,
            # Scientific-spine obligations that are triggered-but-unmet. Surfaced
            # every turn (not just in the inspect_data tool result) so a required
            # decision/step is in front of the model from the first turn, regardless
            # of which tools it has called. Empty list = nothing pending.
            "unmet_obligations": self.unmet_obligations(),
            # Cap in the system-prompt snapshot to keep context small on long
            # sessions. The full step_log is still available via to_dict() for
            # notebook generation and reporting.
            "step_log": self.step_log[-25:],
        }

    def render_runtime_context(self) -> str:
        return json.dumps(self.snapshot(), indent=2)

    def set_active_request(self, request: str) -> None:
        self.active_request = request

    def add_context_hint(self, hint: str) -> None:
        """Persist a user/tool-provided biological or workflow hint."""
        normalized = " ".join(str(hint or "").split())
        if not normalized:
            return
        if normalized not in self.context_hints:
            self.context_hints.append(normalized)
            self.context_hints = self.context_hints[-20:]

    @staticmethod
    def _adata_fingerprint(adata) -> tuple:
        """Cheap structural key — changes whenever adata is meaningfully modified."""
        return (
            adata.n_obs,
            adata.n_vars,
            tuple(sorted(adata.obs.columns)),
            tuple(sorted(adata.var.columns)),
            tuple(sorted(adata.uns.keys())),
            tuple(sorted(adata.obsm.keys())),
            tuple(sorted(adata.obsp.keys())),
            tuple(sorted(adata.layers.keys())),
        )

    @staticmethod
    def _cell_set_fingerprint(adata) -> str:
        """Fingerprint the current cell set so cluster-QC freshness survives column changes."""
        digest = hashlib.sha1()
        digest.update(str(adata.n_obs).encode("utf-8"))
        digest.update(b"|")
        digest.update(str(adata.n_vars).encode("utf-8"))
        for name in adata.obs_names:
            digest.update(b"|")
            digest.update(str(name).encode("utf-8", errors="replace"))
        return digest.hexdigest()[:16]

    def _cluster_qc_summary(self, adata, cluster_key: Optional[str], processing: Dict[str, Any]) -> Dict[str, Any]:
        if adata is None:
            return {"status": "not_applicable", "reason": "no data loaded"}
        if not processing.get("has_clusters") or not cluster_key:
            return {"status": "not_applicable", "reason": "no clustering available"}
        if not processing.get("has_qc_metrics"):
            return {
                "status": "not_ready",
                "cluster_key": cluster_key,
                "fresh": False,
                "reason": "QC metrics are not present, so cluster-level QC cannot run yet",
            }

        cell_set = self._cell_set_fingerprint(adata)
        n_clusters = int(adata.obs[cluster_key].nunique()) if cluster_key in adata.obs.columns else None
        record = self.cluster_qc_registry.get(str(cluster_key))
        if record and record.get("cell_set") == cell_set and record.get("n_clusters") == n_clusters:
            status = "fresh_clean"
            if record.get("proposed_removal") or record.get("ambiguous"):
                status = "fresh_review_required"
            return {
                "status": status,
                "cluster_key": cluster_key,
                "fresh": True,
                "checked_at": record.get("checked_at"),
                "metric_flagged_clusters": record.get("metric_flagged_clusters", record.get("proposed_removal", [])),
                "cells_in_metric_flagged_clusters": record.get("cells_in_metric_flagged_clusters", record.get("cells_in_proposed_removal")),
                "pct_metric_flagged": record.get("pct_metric_flagged", record.get("pct_proposed")),
                "proposed_removal": record.get("proposed_removal", []),
                "ambiguous": record.get("ambiguous", []),
                "cells_in_proposed_removal": record.get("cells_in_proposed_removal"),
                "pct_proposed": record.get("pct_proposed"),
                "structure_clusters_analyzed": record.get("structure_clusters_analyzed", []),
                "synthesized_removal": record.get("synthesized_removal", []),
                "cells_in_synthesized_removal": record.get("cells_in_synthesized_removal"),
                "rescued_clusters": record.get("rescued_clusters", []),
                "conflicting": record.get("conflicting", []),
                "reason": "cluster QC is fresh for the active clustering and current cell set",
            }

        stale_reason = "no cluster-level QC has been run for the active clustering"
        if record:
            stale_reason = "cluster-level QC is stale because the cell set or cluster count changed"
        return {
            "status": "needed",
            "cluster_key": cluster_key,
            "fresh": False,
            "reason": stale_reason,
            "recommended_next_action": "run_cluster_qc",
        }

    def invalidate_inspect_cache(self) -> None:
        """Force the next sync_from_adata to re-run inspect_data."""
        self._inspect_cache_key = None
        self._inspect_cache_state = None

    def sync_from_adata(self, adata, request_text: Optional[str] = None) -> None:
        if adata is None:
            self.data_summary = {}
            self.metadata_candidates = []
            self.clustering_registry = []
            self.annotation_sources = []
            self.analysis_stage = "uninitialized"
            self.invalidate_inspect_cache()
            return

        from ..core import inspect_data
        from ..core.inspector import (
            clustering_record_to_dict,
            metadata_candidate_to_dict,
            obs_columns_detail,
            semantic_roles_to_dict,
        )
        from ..analysis.context import infer_biological_context


        # Re-use the cached DataState if adata's structure hasn't changed.
        # inspect_data touches adata.X (expensive on large datasets); caching it
        # means simple follow-up questions cost nothing here.
        fingerprint = self._adata_fingerprint(adata)
        if fingerprint == self._inspect_cache_key and self._inspect_cache_state is not None:
            state = self._inspect_cache_state
        else:
            state = inspect_data(adata)
            self._inspect_cache_key = fingerprint
            self._inspect_cache_state = state
        processing = {
            # "raw counts are available" — in a layer, adata.raw, OR the X matrix
            # itself (X is integer counts, even when stored as float32). Previously
            # this was has_raw_layer only, so a raw-count X with no separate layer
            # (e.g. *_raw.h5ad files) misreported as has_raw_counts=false and
            # confused the model into thinking X wasn't raw.
            "has_raw_counts": bool(state.has_raw_layer or state.has_raw or state.is_counts),
            # Explicit: does the live X matrix contain raw integer counts right now.
            "x_is_raw_counts": bool(state.is_counts),
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
        context_text = " ".join(
            part for part in [self.active_request, request_text, *self.context_hints] if part
        )
        biological_context = infer_biological_context(
            adata,
            text_context=context_text,
            _precomputed_state=state,
        ).to_dict()

        # A model-recorded inspection decision (see record_inspection) overrides
        # the heuristic *judgments* — which column is the cell type, the species.
        # Facts (obs_columns_detail, shape, is_counts, …) stay heuristic and are
        # recomputed every sync; only the judgment layer defers to the model.
        # Absent a recorded decision, behavior is exactly as before.
        inspection = self.get_confirmed_value("inspection")
        cell_type_key = state.cell_type_key
        cluster_key = state.cluster_key
        if inspection:
            cell_type_key = inspection.get("cell_type_col")
            processing["has_celltypes"] = cell_type_key is not None
            if inspection.get("cluster_col"):
                cluster_key = inspection["cluster_col"]
            if inspection.get("species"):
                biological_context["species"] = inspection["species"]
                biological_context["species_source"] = "model_inspection"
            if inspection.get("tissue"):
                from ..analysis.context import _expected_celltypes_for_tissue
                biological_context["tissue"] = inspection["tissue"]
                biological_context["tissue_source"] = "model_inspection"
                biological_context["expected_celltypes"] = _expected_celltypes_for_tissue(
                    inspection["tissue"]
                )
            if inspection.get("condition"):
                biological_context["condition"] = inspection["condition"]
                biological_context["condition_source"] = "model_inspection"

        self.analysis_stage = _stage_from_processing(processing)
        self.data_summary = {
            "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
            "data_type": state.data_type,
            "processing": processing,
            "batch_key": self.get_confirmed_value("batch_key"),
            "recommended_batch_key": state.batch_key,
            "n_batches": state.n_batches,
            "batch_correction_applied": state.batch_correction_applied,
            "batch_correction_method": state.batch_correction_method,
            "cluster_key": cluster_key,
            "n_clusters": state.n_clusters,
            "cell_type_key": cell_type_key,
            "semantic_obs_roles": semantic_roles_to_dict(state.semantic_obs_roles),
            "obs_columns_detail": obs_columns_detail(adata.obs, adata.n_obs),
            "biological_context": biological_context,
        }
        self.data_summary["batch_strategy"] = self._batch_strategy_summary(state, processing)
        self.data_summary["cluster_qc"] = self._cluster_qc_summary(
            adata,
            state.cluster_key,
            processing,
        )
        self.metadata_candidates = [
            metadata_candidate_to_dict(candidate)
            for candidate in state.metadata_candidates
        ]
        self.clustering_registry = [
            clustering_record_to_dict(record)
            for record in state.clusterings
        ]
        annotation_sources: List[str] = []
        if state.has_celltypist:
            annotation_sources.append("celltypist")
        if state.has_scimilarity:
            annotation_sources.append("scimilarity")
        # "external_or_manual" is a judgment (is some obs column a real label?).
        # When the model has recorded an inspection, trust its cell_type_col call
        # over the heuristic candidate — so a barcode column ruled out as labels
        # does not keep surfacing as an annotation source.
        if inspection:
            has_manual_labels = inspection.get("cell_type_col") is not None
        else:
            has_manual_labels = bool(state.cell_type_candidates)
        if has_manual_labels and not (state.has_celltypist or state.has_scimilarity):
            annotation_sources.append("external_or_manual")
        self.annotation_sources = annotation_sources

        # Surface the recorded inspection so the model sees its own settled
        # decision in the snapshot instead of re-deriving roles every turn.
        if inspection:
            self.data_summary["inspection"] = inspection

        self.data_summary["capabilities"] = self._derive_capabilities(adata)

    def register_artifact(self, artifact_payload: Dict[str, Any]) -> None:
        artifact = (
            artifact_payload
            if isinstance(artifact_payload, ArtifactRecord)
            else ArtifactRecord.from_dict(artifact_payload)
        )
        for index, existing in enumerate(self.artifacts):
            if existing.path == artifact.path:
                self.artifacts[index] = artifact
                return
        self.artifacts.append(artifact)

    def mark_artifact_reviewed(self, path: str, question: str = "") -> None:
        normalized = os.path.abspath(path)
        for artifact in self.artifacts:
            if artifact.path == normalized:
                artifact.review_count += 1
                artifact.last_reviewed_at = _utc_now_iso()
                artifact.last_review_question = question
                return

    def record_decision(self, decision_payload: Dict[str, Any]) -> None:
        decision = (
            decision_payload
            if isinstance(decision_payload, DecisionRecord)
            else DecisionRecord.from_dict(decision_payload)
        )
        target_list = self.resolved_decisions if decision.status != "open" else self.outstanding_decisions
        other_list = self.outstanding_decisions if target_list is self.resolved_decisions else self.resolved_decisions
        other_list[:] = [existing for existing in other_list if existing.key != decision.key]
        target_list[:] = [existing for existing in target_list if existing.key != decision.key]
        target_list.append(decision)
        if decision.status in {"resolved", "user_corrected"}:
            self.user_preferences[decision.key] = decision.applied_value

    def resolve_decision(self, key: str, value: Any, *, source: str = "user", message: str = "") -> None:
        rationale = f"User confirmed {key}={value!r}."
        matching = next((decision for decision in reversed(self.outstanding_decisions) if decision.key == key), None)
        if matching is not None:
            rationale = matching.rationale
            candidates = matching.candidates
            impact = matching.impact
            created_by_tool = matching.created_by_tool
        else:
            candidates = [value]
            impact = "high"
            created_by_tool = source

        resolved = DecisionRecord(
            decision_id=f"{key}_{artifact_id_from_path(str(value))}",
            key=key,
            policy_action="auto_execute" if source != "user" else "recommend_and_confirm",
            status="user_corrected" if source == "user" else "resolved",
            rationale=rationale,
            recommended_value=value,
            applied_value=value,
            impact=impact,
            candidates=candidates,
            created_by_tool=created_by_tool,
            resolved_at=_utc_now_iso(),
            user_message=message,
            metadata={"source": source},
        )
        self.record_decision(resolved.to_dict())

    def get_confirmed_value(self, key: str) -> Any:
        return self.user_preferences.get(key)

    def record_inspection(self, payload: Dict[str, Any], adata=None) -> Dict[str, Any]:
        """Record the model's inspection judgment (column roles + species).

        This is the judgment layer of the facts/judgment split: the model reads
        the deterministic fact sheet and reports which obs column is the cell
        type / batch / donor / sample and the species. We validate the claim
        (named columns must exist; species constrained), then store it as a
        resolved decision so it (a) overrides the heuristic in sync_from_adata,
        (b) surfaces in the snapshot, and (c) flows to the manifest. batch_col is
        routed through the existing ``batch_key`` slot the pipeline already reads.

        Returns ``{"status": "ok", "inspection": {...}}`` or
        ``{"status": "error", "errors": [...]}``; on error nothing is stored.
        """
        obs_cols = set(adata.obs.columns) if adata is not None else set()
        errors: List[str] = []
        for field_name in ("cell_type_col", "batch_col", "donor_col", "sample_col", "cluster_col"):
            value = payload.get(field_name)
            if value is not None and adata is not None and value not in obs_cols:
                errors.append(
                    f"{field_name}={value!r} is not an obs column. "
                    f"Available: {sorted(obs_cols)}"
                )
        species = payload.get("species")
        species_norm = str(species).lower() if species is not None else None
        if species_norm is not None and species_norm not in {"human", "mouse", "unknown"}:
            errors.append(f"species={species!r} must be one of human, mouse, unknown.")
        if errors:
            return {"status": "error", "errors": errors}

        def _clean_text(value):
            text = str(value).strip() if value is not None else ""
            return text or None

        inspection = {
            "cell_type_col": payload.get("cell_type_col"),
            "batch_col": payload.get("batch_col"),
            "donor_col": payload.get("donor_col"),
            "sample_col": payload.get("sample_col"),
            "cluster_col": payload.get("cluster_col"),
            "species": species_norm,
            "tissue": _clean_text(payload.get("tissue")),
            "condition": _clean_text(payload.get("condition")),
            "rationale": str(payload.get("rationale", "")),
            "recorded_at": _utc_now_iso(),
        }
        self.resolve_decision("inspection", inspection, source="model_inspection")
        if species_norm:
            self.resolve_decision("species", species_norm, source="model_inspection")
        if inspection["batch_col"]:
            self.resolve_decision("batch_key", inspection["batch_col"], source="model_inspection")
        # Re-sync so data_summary reflects the new decision immediately.
        if adata is not None:
            self.sync_from_adata(adata, request_text=self.active_request)
        return {"status": "ok", "inspection": inspection}

    def apply_tool_result(self, tool_name: str, result: Dict[str, Any], adata=None) -> None:
        if adata is not None:
            self.sync_from_adata(adata, request_text=self.active_request)

        self._update_annotation_validation(tool_name, result)

        for artifact_payload in result.get("artifacts_created", []):
            self.register_artifact(artifact_payload)
        for decision_payload in result.get("decisions_raised", []):
            self.record_decision(decision_payload)

        if tool_name in {"review_artifact", "review_figure"}:
            reviewed_path = result.get("artifact_path") or result.get("figure_path")
            if reviewed_path:
                self.mark_artifact_reviewed(reviewed_path, question=result.get("question", ""))

        verification = result.get("verification") or {}
        self.latest_verification = verification
        self.last_action = {
            "tool": tool_name,
            "status": result.get("status"),
            "summary": (result.get("state_delta") or {}).get("summary", ""),
            "verification": verification,
            "timestamp": _utc_now_iso(),
        }
        self.recent_events.append(
            {
                "tool": tool_name,
                "status": result.get("status"),
                "timestamp": _utc_now_iso(),
                "summary": (result.get("state_delta") or {}).get("summary", ""),
            }
        )
        self.recent_events = self.recent_events[-25:]

        # Extract and permanently log key parameters/results for each major step.
        # This survives context trimming and is the source of truth for notebook
        # generation and retrospective questions about what was done.
        entry = self._extract_step_entry(tool_name, result)
        if entry:
            # Append every successful run so re-running a step with different
            # parameters preserves the full history (important for notebook
            # generation and "what thresholds did we try?" follow-ups).
            # The snapshot() view is capped separately to keep the system
            # prompt compact; to_dict() keeps the full log for reporting.
            self.step_log.append(entry)
            # Guard against unbounded growth on very long sessions.
            if len(self.step_log) > 200:
                self.step_log = self.step_log[-200:]

        if tool_name in {"run_clustering", "compare_clusterings"} and result.get("status") in {"ok", "success"}:
            keys: List[str] = []
            if result.get("cluster_key"):
                keys.append(str(result.get("cluster_key")))
            if result.get("primary_cluster_key"):
                keys.append(str(result.get("primary_cluster_key")))
            for comparison in result.get("comparisons", []) or []:
                if comparison.get("cluster_key"):
                    keys.append(str(comparison.get("cluster_key")))
            for key in keys:
                self.cluster_qc_registry.pop(key, None)
            if adata is not None:
                self.data_summary["cluster_qc"] = self._cluster_qc_summary(
                    adata,
                    self.data_summary.get("cluster_key"),
                    self.data_summary.get("processing", {}),
                )

        if tool_name == "run_cluster_qc" and result.get("status") in {"ok", "success"} and adata is not None:
            cluster_key = result.get("cluster_key")
            if cluster_key:
                n_clusters = None
                if cluster_key in adata.obs.columns:
                    n_clusters = int(adata.obs[cluster_key].nunique())
                self.cluster_qc_registry[str(cluster_key)] = {
                    "cluster_key": str(cluster_key),
                    "cell_set": self._cell_set_fingerprint(adata),
                    "shape": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                    "n_clusters": n_clusters,
                    "checked_at": _utc_now_iso(),
                    "metric_flagged_clusters": [
                        str(c) for c in result.get("metric_flagged_clusters", result.get("proposed_removal", [])) or []
                    ],
                    "cells_in_metric_flagged_clusters": result.get(
                        "cells_in_metric_flagged_clusters",
                        result.get("cells_in_proposed_removal"),
                    ),
                    "pct_metric_flagged": result.get("pct_metric_flagged", result.get("pct_proposed")),
                    "proposed_removal": [str(c) for c in result.get("proposed_removal", []) or []],
                    "ambiguous": [str(c) for c in result.get("ambiguous", []) or []],
                    "cells_in_proposed_removal": result.get("cells_in_proposed_removal"),
                    "pct_proposed": result.get("pct_proposed"),
                    "thresholds_used": result.get("thresholds_used", {}),
                    "cluster_decisions": result.get("cluster_decisions", {}),
                    "cluster_table": result.get("cluster_table", []),
                }
                self.data_summary["cluster_qc"] = self._cluster_qc_summary(
                    adata,
                    self.data_summary.get("cluster_key"),
                    self.data_summary.get("processing", {}),
                )

        if tool_name == "run_cluster_structure_qc" and result.get("status") in {"ok", "success"} and adata is not None:
            cluster_key = result.get("cluster_key")
            if cluster_key:
                key = str(cluster_key)
                record = self.cluster_qc_registry.setdefault(
                    key,
                    {
                        "cluster_key": key,
                        "cell_set": self._cell_set_fingerprint(adata),
                        "shape": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                    },
                )
                record["structure_checked_at"] = _utc_now_iso()
                record["structure_qc_run_id"] = result.get("structure_qc_run_id")
                record["structure_qc_pass"] = result.get("structure_qc_pass")
                record["structure_figure_dir"] = result.get("figure_dir")
                record["structure_heatmap_paths"] = result.get("heatmap_paths", [])
                record["structure_qc_json"] = result.get("structure_qc_json")
                record["structure_qc_markdown"] = result.get("structure_qc_markdown")
                record["structure_evidence"] = result.get("structure_evidence_by_cluster", {})
                record["structure_clusters_analyzed"] = [
                    str(c) for c in result.get("clusters_analyzed", []) or []
                ]
                record["synthesized_removal"] = [
                    str(c) for c in result.get("synthesized_removal", []) or []
                ]
                record["cells_in_synthesized_removal"] = result.get("cells_in_synthesized_removal")
                record["pct_synthesized_removal"] = result.get("pct_synthesized_removal")
                record["rescued_clusters"] = [
                    str(c) for c in result.get("rescued_clusters", []) or []
                ]
                record["confirmed_junk"] = [
                    str(c) for c in result.get("confirmed_junk", []) or []
                ]
                record["conflicting"] = [
                    str(c) for c in result.get("conflicting", []) or []
                ]
                record["structure_thresholds_used"] = result.get("thresholds_used", {})
                record.setdefault("structure_history", []).append(
                    {
                        "structure_qc_run_id": result.get("structure_qc_run_id"),
                        "structure_qc_pass": result.get("structure_qc_pass"),
                        "checked_at": record["structure_checked_at"],
                        "clusters_analyzed": record["structure_clusters_analyzed"],
                        "synthesized_removal": record["synthesized_removal"],
                        "cells_in_synthesized_removal": record["cells_in_synthesized_removal"],
                        "pct_synthesized_removal": record["pct_synthesized_removal"],
                        "figure_dir": record["structure_figure_dir"],
                        "heatmap_paths": record["structure_heatmap_paths"],
                        "structure_qc_json": record["structure_qc_json"],
                        "structure_qc_markdown": record["structure_qc_markdown"],
                    }
                )
                self.data_summary["cluster_qc"] = self._cluster_qc_summary(
                    adata,
                    self.data_summary.get("cluster_key"),
                    self.data_summary.get("processing", {}),
                )

    def unmet_obligations(self) -> List[Dict[str, Any]]:
        """Scientific-spine obligations that are *triggered but not satisfied*.

        A read-only **view** over state this object already computes — it adds no
        new tracking. Each entry is a plain dict describing a floor the harness
        should bind the model to (vs. the advisory `blocked_actions`/decisions it
        only serializes). The enforcement (re-prompt → block + fallback, or entry
        gating) lives in the agent loop; this method only *reports* what is unmet.

        Obligations are deliberately limited to load-bearing **scientific-validity**
        checkpoints, not tool-ordering prerequisites:

        - ``annotation_finalize`` (completion): annotation was entered
          (``prepare_annotation`` set ``required``) but never finalized. This is the
          same predicate the save/report guard uses; surfacing it here lets the
          *terminal* exit (a no-tool-call stop) be guarded too, not just save/report.
        - ``batch_decision`` (entry): a multi-sample dataset needs the
          ``multi_sample_strategy`` decision resolved before clustering/annotation.

        Both are **floors, not ceilings** — `satisfied` means *a decision was made /
        annotation was finalized*, never a particular outcome. A model that already
        does the right thing is never bound.
        """
        out: List[Dict[str, Any]] = []

        av = self.annotation_validation if isinstance(self.annotation_validation, dict) else {}
        if (
            av.get("required")
            and not av.get("finalized")
            and av.get("status") != "validated_and_finalized"
        ):
            out.append({
                "key": "annotation_finalize",
                "kind": "completion",
                "blocks_terminal": True,
                "finalize_attempts": int(av.get("finalize_attempts", 0) or 0),
                "status": av.get("status"),
                "guidance": (
                    "Annotation was started (prepare_annotation) but not finalized "
                    f"(annotation_validation.status={av.get('status')!r}). Do not end the "
                    "run with prose. Either: (1) call stage_annotation_evidence for any "
                    "uncovered clusters, then finalize_annotation; (2) if a cluster cannot "
                    "be resolved, stage it with confidence=low and finalize anyway "
                    "(PanglaoDB is optional, not a gate); or (3) as a last resort, call "
                    "save_data(allow_unvalidated=true) to persist a clearly-marked "
                    "incomplete result. Emit the tool call now — do not run more queries."
                ),
            })

        if self.multi_sample_decision_unresolved():
            n = self._multi_sample_group_count()
            out.append({
                "key": "batch_decision",
                "kind": "entry",
                "blocks_terminal": True,
                "guidance": (
                    f"This dataset has {n} sample-like groups but the multi_sample_strategy "
                    "decision is unresolved. Present the choice to the user now with "
                    "pause_and_ask — investigate (uncorrected first pass → diagnose_batch_effect), "
                    "integrate, keep one combined uncorrected analysis, or analyze separately — "
                    "then end your turn. Preprocessing is blocked until a strategy is selected, so "
                    "do not run QC/normalization first and do not deliberate about proceeding: "
                    "'investigate' is an option you offer here, not a step you take before asking."
                ),
            })

        return out

    def note_spine_intervention(self, keys: List[str], action: str) -> None:
        """Record a coordination-harness intervention for telemetry/audit.

        Lands in `recent_events` (→ snapshot → manifest), so spine adherence —
        how often the floor had to nudge or force a fallback, and on which
        obligations — is measurable post-hoc (e.g. by the NAT eval) rather than
        only in agent.log.
        """
        self.recent_events.append({
            "tool": "spine_obligation_gate",
            "status": action,  # "nudge" | "forced_fallback"
            "timestamp": _utc_now_iso(),
            "summary": f"unmet obligations: {', '.join(keys)}",
        })
        self.recent_events = self.recent_events[-25:]

    def _multi_sample_group_count(self) -> int:
        """Largest detected sample-like group count (batch_key or top candidate)."""
        ds = self.data_summary or {}
        n = int(ds.get("n_batches") or 0)
        for cand in (self.metadata_candidates or []):
            try:
                n = max(n, int(getattr(cand, "n_unique", None) or (cand.get("n_unique") if isinstance(cand, dict) else 0) or 0))
            except (TypeError, ValueError):
                continue
        return n

    def multi_sample_decision_unresolved(self) -> bool:
        """True iff the data is multi-sample and no multi_sample_strategy is set.

        Floor predicate for the batch entry obligation. ``satisfied`` = a strategy
        was chosen (any option); ``moot`` (returns False) when the data is
        single-sample, so it can never fire on a single-sample dataset.
        """
        if self.get_confirmed_value("multi_sample_strategy"):
            return False
        return self._multi_sample_group_count() >= 2

    def _update_annotation_validation(self, tool_name: str, result: Dict[str, Any]) -> None:
        """Track whether automated annotation has external marker validation."""
        status = result.get("status")
        if tool_name in {"run_celltypist", "run_scimilarity"} and status not in {"ok", "success"}:
            source_name = tool_name.removeprefix("run_")
            existing = self.annotation_validation if isinstance(self.annotation_validation, dict) else {}
            unavailable = dict(existing.get("reference_source_unavailable") or {})
            reason = (
                result.get("unavailable_reason")
                or result.get("missing_reason")
                or result.get("message")
                or str(status or "unknown")
            )
            unavailable[source_name] = {
                "tool": tool_name,
                "status": status,
                "reason": reason,
                "message": result.get("message"),
                "model": result.get("model") or result.get("celltypist_model"),
                "model_path": result.get("model_path"),
                "requested_organism": result.get("requested_organism"),
                "model_organism": result.get("model_organism"),
                "selected_organism": result.get("selected_organism"),
                "timestamp": _utc_now_iso(),
            }
            self.annotation_validation = {
                **existing,
                "required": True,
                "status": existing.get("status") or "reference_source_unavailable",
                "reference_source_unavailable": unavailable,
                "instruction": existing.get("instruction") or (
                    "Run compatible CellTypist and Scimilarity sources when possible. "
                    "If a source cannot run, keep its concrete unavailable reason and "
                    "continue with remaining sources plus DEG/external marker adjudication."
                ),
            }
            return
        if tool_name == "finalize_annotation" and status not in {"ok", "success"}:
            # Count only *genuine* finalize attempts — ones where evidence was
            # actually evaluated and failed validation, not the trivial
            # "stage evidence first" rejection. The save guard uses this so it
            # can degrade to an honestly-labeled unvalidated save after the
            # agent has truly tried, instead of blocking forever and ending the
            # run with no final dataset on disk.
            is_genuine_attempt = bool(result.get("validation_failures")) or (
                "validation failed" in str(result.get("message", "")).lower()
            )
            if is_genuine_attempt and isinstance(self.annotation_validation, dict):
                self.annotation_validation["finalize_attempts"] = (
                    int(self.annotation_validation.get("finalize_attempts", 0)) + 1
                )
                self.annotation_validation["last_finalize_error"] = result.get("message")
            return
        if status not in {"ok", "success"}:
            return

        if tool_name in {"run_celltypist", "run_scimilarity"}:
            breakdown = result.get("cell_type_breakdown") or {}
            expected_labels = sorted(str(label) for label in breakdown.keys())
            existing = self.annotation_validation if isinstance(self.annotation_validation, dict) else {}
            candidate_sources = dict(existing.get("candidate_sources") or {})
            unavailable = dict(existing.get("reference_source_unavailable") or {})
            unavailable.pop(tool_name.removeprefix("run_"), None)
            candidate_sources[tool_name] = {
                "annotation_key": result.get("annotation_key"),
                "organism": (
                    result.get("requested_organism")
                    or result.get("selected_organism")
                    or result.get("model_organism")
                ),
                "n_labels": len(expected_labels),
                "expected_labels": expected_labels,
            }
            reference_keys = [
                str(source.get("annotation_key"))
                for source in candidate_sources.values()
                if source.get("annotation_key")
            ]
            union_labels = sorted({
                str(label)
                for source in candidate_sources.values()
                for label in source.get("expected_labels", [])
            })
            existing_queries = (
                list(existing.get("reference_marker_queries", []))
                if isinstance(existing.get("reference_marker_queries"), list) else []
            )
            self.annotation_validation = {
                "required": True,
                "status": "pending_annotation_consensus",
                "annotation_tool": "multi_source_consensus",
                "annotation_key": result.get("annotation_key"),
                "organism": (
                    result.get("requested_organism")
                    or result.get("selected_organism")
                    or result.get("model_organism")
                    or existing.get("organism")
                ),
                "candidate_sources": candidate_sources,
                "reference_source_unavailable": unavailable,
                "reference_annotation_keys": reference_keys,
                "expected_annotation_labels": union_labels,
                "reference_marker_source": "local Cytopus markers (primary) + reference consensus; PanglaoDB fallback",
                "reference_marker_queries": existing_queries,
                "validation_mode": "adjudicate_best_supported_label_not_confirmation",
                "competing_label_policy": (
                    "For ambiguous clusters, query and compare plausible alternative "
                    "labels instead of only searching for support for the first "
                    "automated label."
                ),
                "deg_required": True,
                "deg_completed": bool(existing.get("deg_completed", False)),
                "finalized": False,
                "instruction": (
                    "Run both CellTypist and Scimilarity when compatible, run DEG by cluster, "
                    "call prepare_annotation with all reference annotation keys, query PanglaoDB only "
                    "for clusters requiring external adjudication, use literature/web sources for "
                    "unresolved ambiguous labels, stage evidence, then finalize_annotation."
                ),
            }
            return

        if tool_name == "run_deg" and self.annotation_validation.get("required"):
            self.annotation_validation["deg_completed"] = True
            if not self.annotation_validation.get("reference_marker_queries"):
                self.annotation_validation["status"] = "deg_ready_pending_reference_marker_validation"
            return

        if tool_name == "prepare_annotation":
            existing_validation = self.annotation_validation if isinstance(self.annotation_validation, dict) else {}
            cluster_summaries = result.get("clusters") or []
            ambiguous = result.get("ambiguous_clusters") or []
            queries_required = result.get("panglaodb_queries_required") or []
            reverse_queries_required = result.get("panglaodb_reverse_marker_queries_required") or []
            required_clusters = [str(c) for c in (result.get("panglaodb_required_clusters") or [])]
            optional_clusters = [str(c) for c in (result.get("panglaodb_optional_clusters") or [])]
            existing_queries = (
                list(existing_validation.get("reference_marker_queries", []))
                if isinstance(existing_validation, dict) else []
            )
            self.annotation_validation = {
                "required": True,
                "status": (
                    "proposal_staged_pending_external_adjudication"
                    if required_clusters
                    else "proposal_staged_reference_deg_sufficient"
                ),
                "annotation_tool": "manual_marker_workflow",
                "annotation_key": result.get("annotation_key"),
                "cluster_key": result.get("cluster_key"),
                "n_clusters": result.get("n_clusters"),
                "n_ambiguous": len(ambiguous),
                "ambiguous_clusters": ambiguous,
                "shared_markers_flagged": result.get("shared_markers_flagged") or [],
                "scoring_method": result.get("scoring_method"),
                "reference_annotation_keys": result.get("reference_annotation_keys") or [],
                "reference_source_coverage": result.get("reference_source_coverage") or {},
                "missing_reference_sources": result.get("missing_reference_sources") or [],
                "reference_source_unavailable": existing_validation.get("reference_source_unavailable") or {},
                "reference_annotation_notice": result.get("reference_annotation_notice"),
                "candidate_sources": existing_validation.get("candidate_sources") or {},
                "panglaodb_queries_required": queries_required,
                "panglaodb_reverse_marker_queries_required": reverse_queries_required,
                "panglaodb_required_clusters": required_clusters,
                "panglaodb_optional_clusters": optional_clusters,
                "reference_marker_queries": existing_queries,
                "reference_marker_source": "local Cytopus markers (primary) + reference consensus + DEGs; PanglaoDB fallback",
                "deg_required": True,
                "deg_completed": True,
                "finalized": False,
                "instruction": (
                    (
                        "Query PanglaoDB only for clusters in panglaodb_required_clusters using "
                        "panglaodb_queries_required and panglaodb_reverse_marker_queries_required, "
                        "aggregate reverse gene-symbol hits across multiple DEGs, compare markers "
                        "against each required cluster's top_degs, then stage/finalize annotation. "
                        "PanglaoDB is optional, not a gate: if a flagged cluster cannot be resolved "
                        "(label not covered, e.g. CMP/MEP/early-erythroid, or query inconclusive), "
                        "stage it with panglaodb_queried=false and confidence=low — the validator "
                        "accepts reference+DEG evidence and will not block finalize. Do not loop."
                    )
                    if required_clusters else
                    "No cluster was flagged for upfront PanglaoDB adjudication. Stage evidence from "
                    "reference labels plus submitted DEG support; query PanglaoDB reactively only if "
                    "stage_annotation_evidence/finalize_annotation reports a cluster still requires it."
                ),
            }
            return

        if tool_name == "stage_annotation_evidence":
            coverage = result.get("coverage") or {}
            if not isinstance(self.annotation_validation, dict):
                self.annotation_validation = {}
            self.annotation_validation.update({
                "required": True,
                "status": (
                    "evidence_staged_ready_to_finalize"
                    if coverage.get("n_missing") == 0
                    else "evidence_partially_staged_pending_more_clusters"
                ),
                "annotation_tool": "manual_marker_workflow",
                "n_evidence_staged": result.get("n_entries_staged_total"),
                "n_evidence_covered": coverage.get("n_covered"),
                "n_evidence_missing": coverage.get("n_missing"),
                "missing_evidence_clusters": coverage.get("missing_clusters") or [],
                "instruction": (
                    "Continue staging cluster evidence until coverage is complete, then call "
                    "finalize_annotation; evidence_summary can be omitted when staged evidence covers all clusters."
                ),
            })
            return

        if tool_name == "finalize_annotation":
            payload = result.get("annotation_validation") or {}
            existing_validation = self.annotation_validation if isinstance(self.annotation_validation, dict) else {}
            existing_queries = (
                list(existing_validation.get("reference_marker_queries", []))
                if isinstance(existing_validation, dict) else []
            )
            self.annotation_validation = {
                "required": True,
                "status": "validated_and_finalized",
                "annotation_tool": "manual_marker_workflow",
                "annotation_key": result.get("annotation_key"),
                "cluster_key": result.get("cluster_key"),
                "n_clusters_validated": payload.get("n_clusters_validated"),
                "label_counts": result.get("label_counts") or {},
                "panglaodb_validated": True,
                "external_validation_policy": payload.get("external_validation_policy"),
                "validation_strategy": payload.get("validation_strategy"),
                "validation_tier_breakdown": payload.get("validation_tier_breakdown") or {},
                "n_cytopus_adjudicated": payload.get("n_cytopus_adjudicated"),
                "n_panglaodb_adjudicated": payload.get("n_panglaodb_adjudicated"),
                "marker_adjudication_sources": payload.get("marker_adjudication_sources")
                    or ["cytopus_local", "reference_consensus", "submitted_deg", "panglaodb_fallback"],
                "panglaodb_required_clusters": payload.get("panglaodb_required_clusters") or [],
                "finalized": True,
                "reference_marker_queries": existing_queries,
                "reference_marker_source": "Cytopus (local) primary; reference consensus + DEGs; PanglaoDB fallback",
                "reference_annotation_keys": payload.get("reference_annotation_keys") or [],
                "reference_source_coverage": payload.get("reference_source_coverage") or {},
                "missing_reference_sources": payload.get("missing_reference_sources") or [],
                "reference_source_unavailable": payload.get("reference_source_unavailable") or existing_validation.get("reference_source_unavailable") or {},
                "candidate_sources": existing_validation.get("candidate_sources") or {},
                "per_cluster_evidence": payload.get("per_cluster_evidence", {}),
            }
            return

        if tool_name == "bc_get_panglaodb_marker_genes":
            query = result.get("marker_query") or result.get("query") or {}
            entry = {
                "source": "PanglaoDB",
                "species": query.get("species") or result.get("species"),
                "cell_type": query.get("cell_type") or result.get("cell_type"),
                # Persist gene_symbol so reverse-marker queries are auditable.
                # Without this, finalize_annotation cannot verify "reverse_marker_support"
                # evidence against the actual PanglaoDB call history.
                "gene_symbol": (
                    query.get("gene_symbol")
                    or query.get("gene")
                    or result.get("gene_symbol")
                    or result.get("gene")
                ),
                "min_sensitivity": query.get("min_sensitivity") or result.get("min_sensitivity"),
                "queried_at": _utc_now_iso(),
            }
            existing = self.annotation_validation.get("reference_marker_queries") or []
            existing.append(entry)
            if not self.annotation_validation:
                self.annotation_validation = {
                    "required": False,
                    "status": "reference_markers_queried",
                    "reference_marker_source": "PanglaoDB",
                }
            self.annotation_validation["reference_marker_queries"] = existing[-50:]
            if not self.annotation_validation.get("finalized"):
                self.annotation_validation["status"] = "reference_markers_queried_pending_synthesis"
            self.annotation_validation["reference_marker_source"] = "PanglaoDB"

    @staticmethod
    def _extract_step_entry(tool_name: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a compact, permanent log entry from a tool result."""
        if result.get("status") != "ok":
            return None

        ts = _utc_now_iso()



        if tool_name == "run_cellbender":
            return {
                "tool": "run_cellbender",
                "timestamp": ts,
                "input_path": result.get("input_path"),
                "output_path": result.get("output_path"),
                "returncode": result.get("returncode"),
                "use_cuda": result.get("use_cuda"),
                "expected_cells": result.get("expected_cells"),
                "total_droplets_included": result.get("total_droplets_included"),
                "fpr": result.get("fpr"),
                "epochs": result.get("epochs"),
                "loaded_as_primary": result.get("loaded_as_primary"),
            }

        if tool_name == "score_integration":
            return {
                "tool": "score_integration",
                "timestamp": ts,
                "use_rep": result.get("use_rep"),
                "batch_key": result.get("batch_key"),
                "n_neighbors": result.get("n_neighbors"),
                "entropy_mean": result.get("entropy_mean"),
                "entropy_median": result.get("entropy_median"),
                "interpretation": result.get("interpretation"),
            }

        if tool_name == "benchmark_integration":
            return {
                "tool": "benchmark_integration",
                "timestamp": ts,
                "batch_key": result.get("batch_key"),
                "label_key": result.get("label_key"),
                "embeddings_benchmarked": result.get("embeddings_benchmarked"),
                "scores_by_embedding": result.get("scores_by_embedding"),
                "best_method": result.get("best_method"),
            }

        if tool_name == "run_qc":
            before = result.get("before", {})
            after = result.get("after", {})
            metrics = result.get("metrics", {})
            # Pull applied thresholds from qc_decisions if present
            decisions = result.get("qc_decisions") or {}
            return {
                "tool": "run_qc",
                "timestamp": ts,
                "cells_before": before.get("n_cells"),
                "cells_after": after.get("n_cells"),
                "genes_before": before.get("n_genes"),
                "genes_after": after.get("n_genes"),
                "cells_removed": metrics.get("cells_removed"),
                "genes_removed": metrics.get("genes_removed"),
                "mt_threshold": decisions.get("mt_threshold") or decisions.get("pct_counts_mt"),
                "min_genes": decisions.get("min_genes"),
                "min_cells_per_gene": decisions.get("min_cells_per_gene"),
                "max_genes": decisions.get("max_genes"),
                "min_counts": decisions.get("min_counts"),
                "doublet_detection": decisions.get("doublet_detection"),
                "doublet_rate": metrics.get("doublet_rate"),
                "median_pct_mt": metrics.get("median_pct_mt"),
            }

        if tool_name == "normalize_and_hvg":
            hvg = result.get("hvg") or {}
            exclusions = result.get("feature_exclusions") or {}
            removals = result.get("feature_removals") or {}
            ribosomal_removal = removals.get("ribosomal_genes") or {}
            return {
                "tool": "normalize_and_hvg",
                "timestamp": ts,
                "target_sum": result.get("target_sum"),
                "log_transform": result.get("log_transform"),
                "normalization_source": result.get("normalization_source"),
                "resolved_source": result.get("resolved_source"),
                "reset_from_raw_counts": result.get("reset_from_raw_counts"),
                "reset_reason": result.get("reset_reason"),
                "input_x_preserved_layer": result.get("input_x_preserved_layer"),
                "raw_layer_name": result.get("raw_layer_name"),
                "raw_counts_present": result.get("raw_counts_present"),
                "raw_counts_integer_like": result.get("raw_counts_integer_like"),
                "adata_raw_set": result.get("adata_raw_set"),
                "adata_raw_shape": result.get("adata_raw_shape"),
                "n_hvg_selected": result.get("n_hvg"),
                "remove_ribosomal_genes": result.get("remove_ribosomal_genes"),
                "feature_removals": {
                    "ribosomal_genes": {
                        "enabled": ribosomal_removal.get("enabled"),
                        "patterns": ribosomal_removal.get("patterns"),
                        "match_mode": ribosomal_removal.get("match_mode"),
                        "source": ribosomal_removal.get("source"),
                        "n_removed": ribosomal_removal.get("n_removed"),
                    }
                },
                "hvg_method": hvg.get("method"),
                "hvg_flavor": hvg.get("flavor"),
                "hvg_requested_flavor": hvg.get("requested_flavor"),
                "batch_key": hvg.get("batch_key"),
                "hvg_layer": hvg.get("layer"),
                "feature_exclusions": {
                    "applied": exclusions.get("applied"),
                    "patterns": exclusions.get("patterns"),
                    "match_mode": exclusions.get("match_mode"),
                    "mode": exclusions.get("mode"),
                    "source": exclusions.get("source"),
                    "n_excluded": exclusions.get("n_excluded"),
                    "excluded_hvg_before_forcing": exclusions.get("excluded_hvg_before_forcing"),
                    "excluded_hvg_after_forcing": exclusions.get("excluded_hvg_after_forcing"),
                },
            }


        if tool_name == "run_pca":
            return {
                "tool": "run_pca",
                "timestamp": ts,
                "n_comps": result.get("n_comps"),
                "svd_solver": result.get("svd_solver"),
                "mask_var": result.get("mask_var"),
                "variance_explained": result.get("variance_explained"),
                "side_effects": result.get("side_effects"),
            }

        if tool_name == "run_neighbors":
            return {
                "tool": "run_neighbors",
                "timestamp": ts,
                "n_neighbors": result.get("n_neighbors"),
                "n_pcs": result.get("n_pcs"),
                "use_rep": result.get("use_rep"),
                "neighbors_key": result.get("neighbors_key"),
                "side_effects": result.get("side_effects"),
            }

        if tool_name == "run_umap":
            return {
                "tool": "run_umap",
                "timestamp": ts,
                "neighbors_key": result.get("neighbors_key"),
                "min_dist": result.get("min_dist"),
                "spread": result.get("spread"),
                "n_components": result.get("n_components"),
                "neighbor_graph_preserved": result.get("neighbor_graph_preserved"),
                "side_effects": result.get("side_effects"),
            }

        if tool_name == "run_clustering":
            return {
                "tool": "run_clustering",
                "timestamp": ts,
                "method": result.get("method"),
                "resolution": result.get("resolution"),
                "n_clusters": result.get("n_clusters"),
                "cluster_key": result.get("cluster_key"),
                "created_obs_columns": result.get("created_obs_columns"),
                "primary_alias": result.get("primary_alias"),
                "primary_cluster_key": result.get("primary_cluster_key"),
                "primary_alias_available": result.get("primary_alias_available"),
            }

        if tool_name == "compare_clusterings":
            return {
                "tool": "compare_clusterings",
                "timestamp": ts,
                "method": result.get("method"),
                "resolutions_tested": result.get("resolutions_tested"),
                "selected_resolution": result.get("selected_resolution"),
                "n_clusters": result.get("n_clusters"),
                "cluster_key": result.get("cluster_key"),
            }

        if tool_name == "run_batch_correction":
            entry = {
                "tool": "run_batch_correction",
                "timestamp": ts,
                "method": result.get("method"),
                "batch_key": result.get("batch_key"),
                "n_batches": result.get("n_batches"),
                "corrected_embedding": result.get("corrected_embedding"),
            }
            if result.get("n_neighbors") is not None:
                entry["n_neighbors"] = result.get("n_neighbors")
            entry["neighbors_recomputed"] = result.get("neighbors_recomputed")
            if result.get("method") == "scvi":
                entry["n_latent"] = result.get("n_latent")
                entry["max_epochs"] = result.get("max_epochs")
            if result.get("method") == "bbknn":
                entry["n_pcs"] = result.get("n_pcs")
                entry["neighbors_within_batch"] = result.get("neighbors_within_batch")
                entry["total_neighbors_per_cell"] = result.get("total_neighbors_per_cell")
            return entry

        if tool_name in {"run_celltypist", "run_scimilarity"}:
            return {
                "tool": tool_name,
                "timestamp": ts,
                "status": result.get("status"),
                "model": result.get("model") or result.get("celltypist_model"),
                "model_path": result.get("model_path"),
                "model_cached": result.get("model_cached"),
                "model_cache_path": result.get("model_cache_path"),
                "requested_organism": result.get("requested_organism"),
                "selected_organism": result.get("selected_organism"),
                "model_organism": result.get("model_organism"),
                "model_organism_source": result.get("model_organism_source"),
                "unavailable_reason": result.get("unavailable_reason"),
                "message": result.get("message"),
                "allow_cross_species": result.get("allow_cross_species"),
                "majority_voting": result.get("majority_voting"),
                "n_cell_types": result.get("n_cell_types") or result.get("n_types"),
                "label_key": result.get("label_key") or result.get("annotation_key"),
            }

        if tool_name == "prepare_annotation":
            return {
                "tool": "prepare_annotation",
                "timestamp": ts,
                "cluster_key": result.get("cluster_key"),
                "annotation_key": result.get("annotation_key"),
                "n_clusters": result.get("n_clusters"),
                "n_ambiguous": result.get("n_ambiguous"),
                "ambiguous_clusters": result.get("ambiguous_clusters"),
                "shared_markers_flagged": result.get("shared_markers_flagged"),
                "scoring_method": result.get("scoring_method"),
                "reference_annotation_keys": result.get("reference_annotation_keys"),
                "reference_source_coverage": result.get("reference_source_coverage"),
                "missing_reference_sources": result.get("missing_reference_sources"),
                "reference_annotation_notice": result.get("reference_annotation_notice"),
                "panglaodb_queries_required": result.get("panglaodb_queries_required"),
                "panglaodb_reverse_marker_queries_required": result.get("panglaodb_reverse_marker_queries_required"),
                "reverse_lookup_n_genes_per_cluster": result.get("reverse_lookup_n_genes_per_cluster"),
                "reverse_lookup_max_unique_genes": result.get("reverse_lookup_max_unique_genes"),
            }

        if tool_name == "stage_annotation_evidence":
            coverage = result.get("coverage") or {}
            return {
                "tool": "stage_annotation_evidence",
                "timestamp": ts,
                "n_entries_received": result.get("n_entries_received"),
                "n_entries_staged_total": result.get("n_entries_staged_total"),
                "n_covered": coverage.get("n_covered"),
                "n_missing": coverage.get("n_missing"),
                "missing_clusters": coverage.get("missing_clusters"),
                "role": "manual_annotation_evidence_staging",
            }

        if tool_name == "finalize_annotation":
            return {
                "tool": "finalize_annotation",
                "timestamp": ts,
                "annotation_key": result.get("annotation_key"),
                "cluster_key": result.get("cluster_key"),
                "n_clusters_labeled": result.get("n_clusters_labeled"),
                "label_counts": result.get("label_counts"),
                "panglaodb_validated": True,
                "role": "manual_annotation_finalized",
            }

        if tool_name == "bc_get_panglaodb_marker_genes":
            query = result.get("marker_query") or result.get("query") or {}
            return {
                "tool": "bc_get_panglaodb_marker_genes",
                "timestamp": ts,
                "species": query.get("species") or result.get("species"),
                "cell_type": query.get("cell_type") or result.get("cell_type"),
                "min_sensitivity": query.get("min_sensitivity") or result.get("min_sensitivity"),
                "status": result.get("status"),
                "role": "annotation_reference_marker_validation",
            }

        if tool_name == "run_deg":
            return {
                "tool": "run_deg",
                "timestamp": ts,
                "groupby": result.get("groupby"),
                "method": result.get("method"),
                "n_genes": result.get("n_genes"),
                "key_added": result.get("key_added"),
                "use_raw": result.get("use_raw"),
                "layer_used": result.get("layer_used"),
                "matrix_source": result.get("matrix_source"),
                "matrix_type": result.get("matrix_type"),
            }

        if tool_name == "run_pseudobulk_deg":
            return {
                "tool": "run_pseudobulk_deg",
                "timestamp": ts,
                "cell_type": result.get("cell_type"),
                "sample_col": result.get("sample_col"),
                "condition_col": result.get("condition_col"),
                "condition_a": result.get("condition_a"),
                "condition_b": result.get("condition_b"),
                "n_samples": result.get("n_samples"),
                "n_genes_tested": result.get("n_genes_tested"),
                "n_significant": result.get("n_significant"),
                "alpha": result.get("alpha"),
            }

        if tool_name == "run_gsea":
            return {
                "tool": "run_gsea",
                "timestamp": ts,
                "gene_sets": result.get("gene_sets"),
                "groupby": result.get("groupby"),
            }

        if tool_name == "run_spectra":
            return {
                "tool": "run_spectra",
                "timestamp": ts,
                "cell_type_key": result.get("cell_type_key"),
                "n_factors": result.get("n_factors"),
                "factor_labels": result.get("factor_labels"),
                "model_path": result.get("model_path"),
            }

        if tool_name == "query_cells":
            return {
                "tool": "query_cells",
                "timestamp": ts,
                "query_type": result.get("query_type"),
                "n_query_cells": result.get("n_query_cells"),
                "k": result.get("k"),
                "n_results": result.get("n_results"),
                "mean_dist": result.get("mean_dist"),
                "coherence": result.get("coherence"),
                "top_celltypes": result.get("top_celltypes"),
                "top_tissues": result.get("top_tissues"),
            }

        if tool_name == "score_gene_signature":
            entry: Dict[str, Any] = {
                "tool": "score_gene_signature",
                "timestamp": ts,
                "mode": result.get("mode"),
            }
            if result.get("mode") == "cell_cycle":
                entry["phase_distribution"] = result.get("phase_distribution")
                entry["s_genes_matched"] = result.get("s_genes_matched")
                entry["g2m_genes_matched"] = result.get("g2m_genes_matched")
            else:
                entry["score_name"] = result.get("score_name")
                entry["genes_matched"] = result.get("genes_matched")
                entry["genes_requested"] = result.get("genes_requested")
                entry["coverage_pct"] = result.get("coverage_pct")
                entry["score_stats"] = result.get("score_stats")
            return entry

        if tool_name == "run_cluster_qc":
            return {
                "tool": "run_cluster_qc",
                "timestamp": ts,
                "cluster_key": result.get("cluster_key"),
                "n_clusters": result.get("n_clusters"),
                "metric_flagged_clusters": result.get("metric_flagged_clusters", result.get("proposed_removal", [])),
                "cells_in_metric_flagged_clusters": result.get(
                    "cells_in_metric_flagged_clusters",
                    result.get("cells_in_proposed_removal"),
                ),
                "pct_metric_flagged": result.get("pct_metric_flagged", result.get("pct_proposed")),
                "proposed_removal": result.get("proposed_removal", []),
                "ambiguous": result.get("ambiguous", []),
                "cells_proposed": result.get("cells_in_proposed_removal"),
                "pct_proposed": result.get("pct_proposed"),
                "checkpoint_path": result.get("checkpoint_path"),
            }

        if tool_name == "run_cluster_structure_qc":
            return {
                "tool": "run_cluster_structure_qc",
                "timestamp": ts,
                "cluster_key": result.get("cluster_key"),
                "clusters_analyzed": result.get("clusters_analyzed", []),
                "synthesized_removal": result.get("synthesized_removal", []),
                "cells_in_synthesized_removal": result.get("cells_in_synthesized_removal"),
                "confirmed_junk": result.get("confirmed_junk", []),
                "structured_ambiguous": result.get("structured_ambiguous", []),
                "unstructured_ambiguous": result.get("unstructured_ambiguous", []),
                "rescued_clusters": result.get("rescued_clusters", []),
                "conflicting": result.get("conflicting", []),
                "structure_qc_run_id": result.get("structure_qc_run_id"),
                "structure_qc_pass": result.get("structure_qc_pass"),
                "figure_dir": result.get("figure_dir"),
                "heatmap_paths": result.get("heatmap_paths", []),
                "structure_qc_json": result.get("structure_qc_json"),
                "structure_qc_markdown": result.get("structure_qc_markdown"),
            }

        return None
