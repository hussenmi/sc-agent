"""
Code-level decision policy helpers for collaborative agent behavior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .world_state import DecisionRecord, artifact_id_from_path


def _candidate_values(payload: Dict[str, Any]) -> List[Any]:
    values: List[Any] = []
    for candidate in payload.get("candidates", []):
        if isinstance(candidate, dict) and "column" in candidate:
            values.append(candidate["column"])
        else:
            values.append(candidate)
    return values


def decision_for_batch_strategy(
    strategy: Dict[str, Any],
    *,
    context: str,
    source_tool: str,
    batch_relevant: bool = False,
) -> Optional[Dict[str, Any]]:
    if not strategy:
        return None

    status = strategy.get("status", "")
    recommended = strategy.get("recommended_column")
    applied = strategy.get("applied_column")
    requested = strategy.get("requested_column")
    rationale = strategy.get("reason", "")
    candidates = _candidate_values(strategy)

    if status == "not_applicable":
        return None

    # Batch/sample metadata should not become a front-and-center decision unless
    # the current task actually depends on it or the user explicitly raised it.
    if not batch_relevant and status not in {"invalid_requested"} and not requested:
        return None

    policy_action = "recommend_and_confirm"
    decision_status = "open"
    if status in {"auto_selected", "user_selected"}:
        policy_action = "auto_execute"
        decision_status = "auto_applied"
    elif status == "no_candidate":
        policy_action = "must_ask"
    elif status == "invalid_requested":
        policy_action = "recommend_and_confirm"
    elif status == "needs_confirmation":
        policy_action = "recommend_and_confirm"

    decision = DecisionRecord(
        decision_id=f"batch_key_{artifact_id_from_path(str(recommended or applied or context))}",
        key="batch_key",
        policy_action=policy_action,
        status=decision_status,
        rationale=rationale,
        recommended_value=recommended,
        applied_value=applied,
        impact="high",
        candidates=candidates,
        created_by_tool=source_tool,
        metadata={
            "context": context,
            "recommended_role": strategy.get("recommended_role"),
            "needs_user_confirmation": strategy.get("needs_user_confirmation", False),
            "batch_relevant": batch_relevant,
        },
    )
    return decision.to_dict()


def decision_for_clustering_selection(
    comparisons: List[Dict[str, Any]],
    *,
    source_tool: str,
) -> Optional[Dict[str, Any]]:
    if len(comparisons) < 2:
        return None

    recommended = None
    for comparison in comparisons:
        if abs(float(comparison.get("resolution", 0.0)) - 1.0) < 1e-9:
            recommended = comparison.get("cluster_key")
            break
    if recommended is None:
        recommended = comparisons[0].get("cluster_key")

    candidates = [
        {
            "cluster_key": comparison.get("cluster_key"),
            "resolution": comparison.get("resolution"),
            "n_clusters": comparison.get("n_clusters"),
        }
        for comparison in comparisons
    ]
    rationale = (
        "Multiple clustering resolutions were generated safely. "
        "Choose one to promote as the primary clustering if you want downstream "
        "annotation and DEG defaults to follow a specific resolution."
    )
    decision = DecisionRecord(
        decision_id=f"primary_clustering_{artifact_id_from_path(str(recommended))}",
        key="primary_clustering",
        policy_action="recommend_and_confirm",
        status="open",
        rationale=rationale,
        recommended_value=recommended,
        applied_value=None,
        impact="high",
        candidates=candidates,
        created_by_tool=source_tool,
        metadata={"comparison_count": len(comparisons)},
    )
    return decision.to_dict()
