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
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    outstanding_decisions: List[DecisionRecord] = field(default_factory=list)
    resolved_decisions: List[DecisionRecord] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
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
            available_actions.extend(["run_code", "inspect_data", "save_data", "ask_user"])

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
            else:
                blocked_actions.append({"action": "run_celltypist", "needs": "clustering"})
                blocked_actions.append({"action": "run_scimilarity", "needs": "clustering"})

            # DEG
            if processing.get("has_clusters") or self.annotation_sources:
                available_actions.append("run_deg")
            else:
                blocked_actions.append({"action": "run_deg", "needs": "clusters or annotations"})

            # Pseudobulk DEG — needs groups to aggregate AND raw counts for DESeq2
            if (processing.get("has_clusters") or self.annotation_sources) and processing.get("has_raw_counts"):
                available_actions.append("run_pseudobulk_deg")
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
            if (processing.get("has_clusters") or self.annotation_sources) and processing.get("is_normalized"):
                available_actions.append("run_spectra")
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
            "can_run_annotation": bool(processing.get("has_clusters")),
            "can_run_deg": bool(processing.get("has_clusters") or self.annotation_sources),
            "can_review_markers": deg_available,
            # NEW: Explicit action availability for LLM reasoning
            "available_actions": available_actions,
            "blocked_actions": blocked_actions,
            "run_code_note": "run_code is ALWAYS available for custom plots, filtering, or any valid analysis",
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
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "outstanding_decisions": [decision.to_dict() for decision in self.outstanding_decisions],
            "resolved_decisions": [decision.to_dict() for decision in self.resolved_decisions],
            "user_preferences": self.user_preferences,
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
            "artifacts": [artifact.to_dict() for artifact in self.artifacts[-8:]],
            "outstanding_decisions": [decision.to_dict() for decision in self.outstanding_decisions[-5:]],
            "resolved_decisions": [decision.to_dict() for decision in self.resolved_decisions[-5:]],
            "latest_verification": self.latest_verification,
            "last_action": self.last_action,
            # Cap in the system-prompt snapshot to keep context small on long
            # sessions. The full step_log is still available via to_dict() for
            # notebook generation and reporting.
            "step_log": self.step_log[-25:],
        }

    def render_runtime_context(self) -> str:
        return json.dumps(self.snapshot(), indent=2)

    def set_active_request(self, request: str) -> None:
        self.active_request = request

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
            "has_raw_counts": state.has_raw_layer,
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
        self.analysis_stage = _stage_from_processing(processing)
        self.data_summary = {
            "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
            "data_type": state.data_type,
            "processing": processing,
            "batch_key": self.get_confirmed_value("batch_key"),
            "recommended_batch_key": state.batch_key,
            "n_batches": state.n_batches,
            "batch_correction_applied": state.batch_correction_applied,
            "cluster_key": state.cluster_key,
            "n_clusters": state.n_clusters,
            "cell_type_key": state.cell_type_key,
            "semantic_obs_roles": semantic_roles_to_dict(state.semantic_obs_roles),
            "obs_columns_detail": obs_columns_detail(adata.obs, adata.n_obs),
        }
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
        if state.cell_type_candidates and not (state.has_celltypist or state.has_scimilarity):
            annotation_sources.append("external_or_manual")
        self.annotation_sources = annotation_sources

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

    def apply_tool_result(self, tool_name: str, result: Dict[str, Any], adata=None) -> None:
        if adata is not None:
            self.sync_from_adata(adata, request_text=self.active_request)

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

    @staticmethod
    def _extract_step_entry(tool_name: str, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a compact, permanent log entry from a tool result."""
        if result.get("status") != "ok":
            return None

        ts = _utc_now_iso()



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
            return {
                "tool": "normalize_and_hvg",
                "timestamp": ts,
                "target_sum": result.get("target_sum"),
                "log_transform": result.get("log_transform"),
                "raw_layer_name": result.get("raw_layer_name"),
                "raw_counts_present": result.get("raw_counts_present"),
                "raw_counts_integer_like": result.get("raw_counts_integer_like"),
                "adata_raw_set": result.get("adata_raw_set"),
                "adata_raw_shape": result.get("adata_raw_shape"),
                "n_hvg_selected": result.get("n_hvg"),
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
                "model": result.get("model") or result.get("celltypist_model"),
                "majority_voting": result.get("majority_voting"),
                "n_cell_types": result.get("n_cell_types"),
                "label_key": result.get("label_key") or result.get("annotation_key"),
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
                "proposed_removal": result.get("proposed_removal", []),
                "ambiguous": result.get("ambiguous", []),
                "cells_proposed": result.get("cells_in_proposed_removal"),
                "pct_proposed": result.get("pct_proposed"),
                "checkpoint_path": result.get("checkpoint_path"),
            }

        return None
