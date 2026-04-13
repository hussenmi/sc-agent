"""
Analysis module for scagent.

This module contains tools for validated analysis including:
- DEG validation and execution
- Cluster confidence assessment (Phase 3)
- Biological context capture (Phase 2)
- Literature relevance scoring (Phase 4)
- Structured pathway interpretation (Phase 5)
"""

from .deg import (
    # Enums
    IssueSeverity,
    IssueCategory,

    # Dataclasses
    DEGValidityIssue,
    DEGValidityReport,

    # Main functions
    validate_deg_input,
    validate_deg_output,
    run_validated_deg,

    # Utility functions
    get_deg_validity,
    get_deg_caveats,
    get_cluster_caveats,
)
from .context import (
    BiologicalContext,
    infer_biological_context,
    context_query_hint,
)
from .cluster_confidence import (
    ClusterConfidence,
    get_best_annotation_key,
    cluster_annotation_summary,
    normalize_annotation_lineage,
    annotation_agreement_summary,
    expected_marker_panel,
    marker_support_summary,
    get_cluster_top_markers,
    infer_cluster_confidence,
)
from .literature import (
    LiteratureContextProfile,
    build_literature_context,
    score_paper_relevance,
)
from .interpretation import (
    PathwayInterpretation,
    pathway_function_hint,
    infer_pathway_interpretation,
)
from .pseudobulk import run_pseudobulk_deg
from .spectra import run_spectra

__all__ = [
    # Enums
    "IssueSeverity",
    "IssueCategory",

    # Dataclasses
    "DEGValidityIssue",
    "DEGValidityReport",

    # Main functions
    "validate_deg_input",
    "validate_deg_output",
    "run_validated_deg",

    # Utility functions
    "get_deg_validity",
    "get_deg_caveats",
    "get_cluster_caveats",
    # Biological context
    "BiologicalContext",
    "infer_biological_context",
    "context_query_hint",
    # Cluster confidence
    "ClusterConfidence",
    "get_best_annotation_key",
    "cluster_annotation_summary",
    "normalize_annotation_lineage",
    "annotation_agreement_summary",
    "expected_marker_panel",
    "marker_support_summary",
    "get_cluster_top_markers",
    "infer_cluster_confidence",
    # Literature relevance
    "LiteratureContextProfile",
    "build_literature_context",
    "score_paper_relevance",
    # Structured interpretation
    "PathwayInterpretation",
    "pathway_function_hint",
    "infer_pathway_interpretation",
    # Pseudobulk DEG
    "run_pseudobulk_deg",
    # Spectra factor analysis
    "run_spectra",
]
