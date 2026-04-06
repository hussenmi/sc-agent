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
    biological_context: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    outstanding_decisions: List[DecisionRecord] = field(default_factory=list)
    resolved_decisions: List[DecisionRecord] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    last_action: Dict[str, Any] = field(default_factory=dict)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)
    latest_verification: Dict[str, Any] = field(default_factory=dict)

    def _derive_capabilities(self, adata) -> Dict[str, Any]:
        processing = self.data_summary.get("processing", {})
        cluster_keys = [
            record.get("key")
            for record in self.clustering_registry
            if record.get("key")
        ]
        annotation_keys = []
        if adata is not None:
            annotation_keys = [
                column
                for column in adata.obs.columns
                if any(token in column.lower() for token in ("celltyp", "scimilar", "annotation", "label"))
            ]
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
                available_actions.append("run_dimred")
            else:
                blocked_actions.append({"action": "run_dimred", "needs": "normalized data with HVGs"})

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
            "biological_context": self.biological_context,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "outstanding_decisions": [decision.to_dict() for decision in self.outstanding_decisions],
            "resolved_decisions": [decision.to_dict() for decision in self.resolved_decisions],
            "user_preferences": self.user_preferences,
            "last_action": self.last_action,
            "recent_events": self.recent_events,
            "latest_verification": self.latest_verification,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "analysis_stage": self.analysis_stage,
            "active_request": self.active_request,
            "data_summary": self.data_summary,
            "metadata_candidates": self.metadata_candidates[:3],
            "clustering_registry": self.clustering_registry[:6],
            "annotation_sources": self.annotation_sources,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts[-8:]],
            "outstanding_decisions": [decision.to_dict() for decision in self.outstanding_decisions[-5:]],
            "resolved_decisions": [decision.to_dict() for decision in self.resolved_decisions[-5:]],
            "latest_verification": self.latest_verification,
            "last_action": self.last_action,
        }

    def render_runtime_context(self) -> str:
        return json.dumps(self.snapshot(), indent=2)

    def set_active_request(self, request: str) -> None:
        self.active_request = request

    def sync_from_adata(self, adata, request_text: Optional[str] = None) -> None:
        if adata is None:
            self.data_summary = {}
            self.metadata_candidates = []
            self.clustering_registry = []
            self.annotation_sources = []
            self.analysis_stage = "uninitialized"
            return

        from ..analysis import infer_biological_context
        from ..core import inspect_data
        from ..core.inspector import (
            clustering_record_to_dict,
            metadata_candidate_to_dict,
        )

        state = inspect_data(adata)
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
            "has_celltypes": state.has_celltypist or state.has_scimilarity,
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
        self.annotation_sources = annotation_sources

        context_text = request_text or self.active_request
        self.biological_context = infer_biological_context(
            adata,
            text_context=context_text or "",
        ).to_dict()
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
