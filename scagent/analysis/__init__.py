"""
Analysis module for scagent.

This module contains tools for validated analysis including:
- DEG validation and execution
- Cluster confidence assessment (Phase 3)
- Biological context capture (Phase 2)
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
]
