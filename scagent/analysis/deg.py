"""
Differential Expression Gene (DEG) validation and analysis for scagent.

This module ensures DEG analysis is run on biologically appropriate data
and propagates validity metadata through downstream analyses (GSEA, interpretation).

Key principles:
- Validation happens BEFORE running rank_genes_groups
- Issues are categorized by severity (error, warning, info)
- Validity metadata is attached to adata.uns for downstream use
- Caveats propagate to GSEA and pathway interpretation

Phase 1 of the Biological Relevance Roadmap.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Set
from enum import Enum
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
import scanpy as sc
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class IssueSeverity(Enum):
    """Severity levels for DEG validity issues."""
    ERROR = "error"      # Blocks analysis - results would be invalid
    WARNING = "warning"  # Analysis can proceed but results need caveats
    INFO = "info"        # Informational - good to know but not problematic


class IssueCategory(Enum):
    """Categories of DEG validity issues."""
    MATRIX_TYPE = "matrix_type"           # Raw vs normalized vs scaled
    LAYER_COMPATIBILITY = "layer"         # Layer/method compatibility
    CLUSTER_SIZE = "cluster_size"         # Too few cells in clusters
    GROUP_IMBALANCE = "group_imbalance"   # Large size differences between groups
    BATCH_CONFOUNDING = "batch_confound"  # Batch correlates with clusters
    GENE_ID_FORMAT = "gene_id_format"     # Gene IDs incompatible with pathway DBs
    SPECIES_MISMATCH = "species"          # Data species vs pathway DB species
    EFFECT_SIZE = "effect_size"           # Suspiciously large logFC values
    STATISTICAL = "statistical"           # Statistical test issues
    DATA_QUALITY = "data_quality"         # General data quality issues


# Gene set databases and their expected gene ID formats
GENESET_SPECIES = {
    "KEGG_2021_Human": "human",
    "GO_Biological_Process_2021": "human",
    "Reactome_2022": "human",
    "MSigDB_Hallmark_2020": "human",
    "WikiPathways_2021_Human": "human",
    "KEGG_2021_Mouse": "mouse",
    "GO_Biological_Process_Mouse": "mouse",
}

# Common human gene symbol patterns
HUMAN_GENE_PATTERNS = [
    r'^[A-Z][A-Z0-9]+$',           # Standard symbols: TP53, CD4, HLA-A
    r'^[A-Z]+\d+[A-Z]*$',          # With numbers: IL2, CCL5
    r'^[A-Z]+-[A-Z0-9]+$',         # With hyphens: HLA-DRA
    r'^MT-[A-Z0-9]+$',             # Mitochondrial: MT-CO1
]

# Common mouse gene patterns (typically Title case or lowercase)
MOUSE_GENE_PATTERNS = [
    r'^[A-Z][a-z0-9]+$',           # Title case: Trp53, Cd4
    r'^[a-z][a-z0-9]+$',           # Lowercase: gapdh
]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DEGValidityIssue:
    """A single validity issue detected during DEG validation."""

    severity: IssueSeverity
    category: IssueCategory
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    # Suggested action to resolve or mitigate
    suggestion: str = ""

    # Affected clusters (if applicable)
    affected_clusters: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
            "affected_clusters": self.affected_clusters,
        }

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        return f"{prefix} {self.message}"


@dataclass
class DEGValidityReport:
    """
    Comprehensive report of DEG input validation.

    This report is attached to adata.uns['deg_validity'] and propagates
    to downstream GSEA and interpretation steps.
    """

    # Validation metadata
    validated_at: str = ""
    groupby: str = ""
    method: str = ""
    layer_used: Optional[str] = None

    # Issues found
    issues: List[DEGValidityIssue] = field(default_factory=list)

    # Summary statistics
    n_groups: int = 0
    n_cells_total: int = 0
    cluster_sizes: Dict[str, int] = field(default_factory=dict)

    # Detected data characteristics
    data_species: str = "unknown"  # human, mouse, unknown
    gene_id_format: str = "unknown"  # symbol, ensembl, entrez, mixed
    matrix_type: str = "unknown"  # counts, log_normalized, scaled

    # Batch information
    batch_key: Optional[str] = None
    batch_cluster_correlation: float = 0.0

    # Validation outcome
    is_valid: bool = True  # False if any ERROR severity issues
    has_warnings: bool = False

    @property
    def errors(self) -> List[DEGValidityIssue]:
        """Get all ERROR severity issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> List[DEGValidityIssue]:
        """Get all WARNING severity issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def infos(self) -> List[DEGValidityIssue]:
        """Get all INFO severity issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.INFO]

    def add_issue(self, issue: DEGValidityIssue):
        """Add an issue and update validity flags."""
        self.issues.append(issue)
        if issue.severity == IssueSeverity.ERROR:
            self.is_valid = False
        elif issue.severity == IssueSeverity.WARNING:
            self.has_warnings = True

    def get_caveats_for_cluster(self, cluster: str) -> List[str]:
        """Get caveat messages relevant to a specific cluster."""
        caveats = []
        for issue in self.issues:
            if issue.severity in [IssueSeverity.ERROR, IssueSeverity.WARNING]:
                if not issue.affected_clusters or cluster in issue.affected_clusters:
                    caveats.append(issue.message)
        return caveats

    def get_interpretation_modifiers(self) -> Dict[str, Any]:
        """
        Get modifiers that should affect downstream interpretation.

        Returns a dict that can be used by GSEA/interpretation to:
        - Add caveats to reports
        - Adjust confidence levels
        - Flag clusters needing extra scrutiny
        """
        modifiers = {
            "global_caveats": [],
            "cluster_caveats": {},
            "confidence_penalty": 0.0,
            "flagged_clusters": set(),
        }

        for issue in self.issues:
            if issue.severity == IssueSeverity.ERROR:
                modifiers["global_caveats"].append(f"[CRITICAL] {issue.message}")
                modifiers["confidence_penalty"] += 0.5
            elif issue.severity == IssueSeverity.WARNING:
                modifiers["global_caveats"].append(issue.message)
                modifiers["confidence_penalty"] += 0.2
                for cluster in issue.affected_clusters:
                    modifiers["flagged_clusters"].add(cluster)
                    if cluster not in modifiers["cluster_caveats"]:
                        modifiers["cluster_caveats"][cluster] = []
                    modifiers["cluster_caveats"][cluster].append(issue.message)

        modifiers["flagged_clusters"] = list(modifiers["flagged_clusters"])
        modifiers["confidence_penalty"] = min(1.0, modifiers["confidence_penalty"])

        return modifiers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage in adata.uns."""
        return {
            "validated_at": self.validated_at,
            "groupby": self.groupby,
            "method": self.method,
            "layer_used": self.layer_used,
            "issues": [i.to_dict() for i in self.issues],
            "n_groups": self.n_groups,
            "n_cells_total": self.n_cells_total,
            "cluster_sizes": self.cluster_sizes,
            "data_species": self.data_species,
            "gene_id_format": self.gene_id_format,
            "matrix_type": self.matrix_type,
            "batch_key": self.batch_key,
            "batch_cluster_correlation": self.batch_cluster_correlation,
            "is_valid": self.is_valid,
            "has_warnings": self.has_warnings,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "n_infos": len(self.infos),
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = []
        lines.append(f"DEG Validity Report ({self.validated_at})")
        lines.append(f"  Groups: {self.n_groups} ({self.n_cells_total:,} cells)")
        lines.append(f"  Method: {self.method}, Layer: {self.layer_used or 'X'}")
        lines.append(f"  Species: {self.data_species}, Gene format: {self.gene_id_format}")

        if self.is_valid and not self.has_warnings:
            lines.append("  Status: VALID (no issues)")
        elif self.is_valid:
            lines.append(f"  Status: VALID with {len(self.warnings)} warning(s)")
        else:
            lines.append(f"  Status: INVALID ({len(self.errors)} error(s))")

        if self.errors:
            lines.append("\n  ERRORS:")
            for e in self.errors:
                lines.append(f"    - {e.message}")

        if self.warnings:
            lines.append("\n  WARNINGS:")
            for w in self.warnings:
                lines.append(f"    - {w.message}")

        return "\n".join(lines)


# =============================================================================
# Validation Helper Functions
# =============================================================================

def _detect_matrix_type(X) -> str:
    """
    Detect whether matrix contains counts, log-normalized, or scaled data.

    Returns: 'counts', 'log_normalized', 'scaled', or 'unknown'
    """
    if sp.issparse(X):
        data = X.data
        max_val = X.max()
        min_val = X.min()
    else:
        data = X.flatten()
        max_val = np.max(X)
        min_val = np.min(X)

    # Sample for efficiency
    sample_size = min(100000, len(data))
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        sample_data = data[indices]
    else:
        sample_data = data

    # Check for integer values (counts)
    is_integer = np.allclose(sample_data, np.round(sample_data))

    # Check value ranges
    has_negative = min_val < -0.01

    if is_integer and not has_negative and max_val > 50:
        return "counts"
    elif has_negative:
        # Scaled data typically has negative values (mean-centered)
        return "scaled"
    elif max_val < 20 and not is_integer:
        # Log-normalized typically has max < ~15-20
        return "log_normalized"
    elif not is_integer and max_val > 20:
        # Normalized but not log-transformed
        return "normalized"
    else:
        return "unknown"


def _detect_species_from_genes(gene_names: List[str]) -> str:
    """
    Detect species based on gene naming conventions.

    Human genes: ALL CAPS (TP53, CD4, HLA-DRA)
    Mouse genes: Title case or lowercase (Trp53, Cd4, Gapdh)
    """
    sample = gene_names[:100]

    human_count = 0
    mouse_count = 0

    for gene in sample:
        gene_str = str(gene)
        # Skip mitochondrial and ribosomal
        if gene_str.startswith(('MT-', 'mt-', 'RPS', 'RPL', 'Rps', 'Rpl')):
            continue

        # Human pattern: all uppercase
        if re.match(r'^[A-Z][A-Z0-9]+(-[A-Z0-9]+)?$', gene_str):
            human_count += 1
        # Mouse pattern: title case or lowercase
        elif re.match(r'^[A-Z][a-z0-9]+$', gene_str) or re.match(r'^[a-z][a-z0-9]+$', gene_str):
            mouse_count += 1

    if human_count > mouse_count * 2:
        return "human"
    elif mouse_count > human_count * 2:
        return "mouse"
    else:
        return "unknown"


def _check_cluster_sizes(
    adata: AnnData,
    groupby: str,
    min_cells: int = 20,
    warn_cells: int = 50,
) -> Tuple[Dict[str, int], List[DEGValidityIssue]]:
    """
    Check cluster sizes and flag small clusters.

    Parameters
    ----------
    min_cells : int
        Clusters smaller than this trigger an ERROR (DEG unreliable)
    warn_cells : int
        Clusters smaller than this trigger a WARNING
    """
    issues = []
    sizes = adata.obs[groupby].value_counts().to_dict()
    sizes = {str(k): int(v) for k, v in sizes.items()}

    tiny_clusters = [k for k, v in sizes.items() if v < min_cells]
    small_clusters = [k for k, v in sizes.items() if min_cells <= v < warn_cells]

    if tiny_clusters:
        issues.append(DEGValidityIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.CLUSTER_SIZE,
            message=f"Clusters {tiny_clusters} have <{min_cells} cells - DEG results unreliable",
            details={"cluster_sizes": {k: sizes[k] for k in tiny_clusters}},
            suggestion=f"Consider excluding these clusters or merging with similar clusters",
            affected_clusters=tiny_clusters,
        ))

    if small_clusters:
        issues.append(DEGValidityIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.CLUSTER_SIZE,
            message=f"Clusters {small_clusters} have <{warn_cells} cells - interpret with caution",
            details={"cluster_sizes": {k: sizes[k] for k in small_clusters}},
            suggestion="DEG power is reduced for small clusters",
            affected_clusters=small_clusters,
        ))

    return sizes, issues


def _check_group_imbalance(
    cluster_sizes: Dict[str, int],
    imbalance_ratio: float = 20.0,
) -> List[DEGValidityIssue]:
    """
    Check for severe group size imbalances.

    Large imbalances can bias DEG results (larger groups dominate).
    """
    issues = []

    if len(cluster_sizes) < 2:
        return issues

    sizes = list(cluster_sizes.values())
    max_size = max(sizes)
    min_size = min(sizes)

    if min_size > 0 and max_size / min_size > imbalance_ratio:
        largest = [k for k, v in cluster_sizes.items() if v == max_size][0]
        smallest = [k for k, v in cluster_sizes.items() if v == min_size][0]

        issues.append(DEGValidityIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.GROUP_IMBALANCE,
            message=f"Large cluster size imbalance ({max_size:,} vs {min_size} cells, ratio {max_size/min_size:.1f}x)",
            details={
                "largest_cluster": largest,
                "largest_size": max_size,
                "smallest_cluster": smallest,
                "smallest_size": min_size,
                "ratio": max_size / min_size,
            },
            suggestion="Consider downsampling large clusters for DEG or using methods robust to imbalance",
            affected_clusters=[smallest],
        ))

    return issues


def _check_batch_confounding(
    adata: AnnData,
    groupby: str,
    batch_key: Optional[str] = None,
    correlation_threshold: float = 0.7,
) -> Tuple[float, List[DEGValidityIssue]]:
    """
    Check if batch and cluster assignments are confounded.

    Uses Cramer's V to measure association between categorical variables.
    """
    issues = []

    # Try to find batch key
    if batch_key is None:
        for key in ['batch', 'sample', 'sample_id', 'donor', 'library']:
            if key in adata.obs.columns:
                batch_key = key
                break

    if batch_key is None or batch_key not in adata.obs.columns:
        return 0.0, issues

    if adata.obs[batch_key].nunique() < 2:
        return 0.0, issues

    # Compute Cramer's V
    try:
        from scipy.stats import chi2_contingency

        contingency = pd.crosstab(adata.obs[groupby], adata.obs[batch_key])
        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1

        if min_dim > 0 and n > 0:
            cramers_v = np.sqrt(chi2 / (n * min_dim))
        else:
            cramers_v = 0.0

        if cramers_v > correlation_threshold:
            issues.append(DEGValidityIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.BATCH_CONFOUNDING,
                message=f"Cluster-batch confounding detected (Cramer's V = {cramers_v:.2f})",
                details={
                    "batch_key": batch_key,
                    "cramers_v": cramers_v,
                    "n_batches": int(adata.obs[batch_key].nunique()),
                },
                suggestion="Consider batch correction before DEG or account for batch in model",
            ))
        elif cramers_v > 0.5:
            issues.append(DEGValidityIssue(
                severity=IssueSeverity.INFO,
                category=IssueCategory.BATCH_CONFOUNDING,
                message=f"Moderate cluster-batch association (Cramer's V = {cramers_v:.2f})",
                details={"batch_key": batch_key, "cramers_v": cramers_v},
                suggestion="Monitor for batch-driven DEG results",
            ))

        return cramers_v, issues

    except Exception as e:
        logger.warning(f"Could not compute batch confounding: {e}")
        return 0.0, issues


def _check_gene_pathway_compatibility(
    gene_names: List[str],
    gene_format: str,
    target_geneset: str = "KEGG_2021_Human",
) -> List[DEGValidityIssue]:
    """
    Check if gene IDs are compatible with pathway databases.

    Most pathway databases use gene symbols. Ensembl IDs need conversion.
    """
    issues = []

    # Check format
    if gene_format == "ensembl":
        issues.append(DEGValidityIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.GENE_ID_FORMAT,
            message="Gene IDs are Ensembl format - pathway databases expect gene symbols",
            details={"detected_format": gene_format, "sample": gene_names[:5]},
            suggestion="Convert to gene symbols or ensure var contains 'gene_symbols' column",
        ))
    elif gene_format == "entrez":
        issues.append(DEGValidityIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.GENE_ID_FORMAT,
            message="Gene IDs are Entrez format - may need conversion for some databases",
            details={"detected_format": gene_format},
            suggestion="Consider converting to gene symbols",
        ))

    return issues


def _check_species_geneset_match(
    data_species: str,
    target_geneset: str,
) -> List[DEGValidityIssue]:
    """Check if data species matches gene set database species."""
    issues = []

    geneset_species = GENESET_SPECIES.get(target_geneset, "human")

    if data_species != "unknown" and geneset_species != data_species:
        issues.append(DEGValidityIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.SPECIES_MISMATCH,
            message=f"Data appears to be {data_species} but gene set '{target_geneset}' is for {geneset_species}",
            details={
                "data_species": data_species,
                "geneset": target_geneset,
                "geneset_species": geneset_species,
            },
            suggestion=f"Use {data_species}-specific gene sets or convert gene symbols",
        ))

    return issues


def _check_layer_method_compatibility(
    matrix_type: str,
    method: str,
    layer: Optional[str],
) -> List[DEGValidityIssue]:
    """
    Check if the matrix type is appropriate for the DEG method.

    Best practices:
    - Wilcoxon: log-normalized or counts (NOT scaled)
    - t-test: log-normalized (NOT scaled, NOT counts)
    - logreg: log-normalized or scaled
    """
    issues = []

    if method == "wilcoxon":
        if matrix_type == "scaled":
            issues.append(DEGValidityIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.MATRIX_TYPE,
                message="Wilcoxon test should not be run on scaled (z-scored) data",
                details={"matrix_type": matrix_type, "method": method, "layer": layer},
                suggestion="Use log-normalized data or raw counts layer",
            ))
        elif matrix_type == "counts":
            # Counts are OK for Wilcoxon but log-normalized is preferred
            issues.append(DEGValidityIssue(
                severity=IssueSeverity.INFO,
                category=IssueCategory.MATRIX_TYPE,
                message="Running Wilcoxon on raw counts - log-normalized is often preferred",
                details={"matrix_type": matrix_type},
                suggestion="Consider using log-normalized data for better effect size estimates",
            ))

    elif method == "t-test":
        if matrix_type == "scaled":
            issues.append(DEGValidityIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.MATRIX_TYPE,
                message="t-test should not be run on scaled data - log fold changes will be meaningless",
                details={"matrix_type": matrix_type, "method": method},
                suggestion="Use log-normalized data",
            ))
        elif matrix_type == "counts":
            issues.append(DEGValidityIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.MATRIX_TYPE,
                message="t-test on raw counts may not be appropriate - assumes normal distribution",
                details={"matrix_type": matrix_type, "method": method},
                suggestion="Use log-normalized data for t-test",
            ))

    return issues


def _check_effect_size_sanity(
    deg_results: Optional[pd.DataFrame] = None,
    max_logfc: float = 10.0,
) -> List[DEGValidityIssue]:
    """
    Post-DEG check: flag suspiciously large effect sizes.

    LogFC > 10 often indicates technical issues (wrong layer, batch effects).
    """
    issues = []

    if deg_results is None:
        return issues

    if 'logfoldchanges' not in deg_results.columns:
        return issues

    extreme_genes = deg_results[deg_results['logfoldchanges'].abs() > max_logfc]

    if len(extreme_genes) > 0:
        n_extreme = len(extreme_genes)
        top_extreme = extreme_genes.nlargest(3, 'logfoldchanges')['names'].tolist()

        issues.append(DEGValidityIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.EFFECT_SIZE,
            message=f"{n_extreme} genes have |logFC| > {max_logfc} - may indicate technical issues",
            details={
                "n_extreme": n_extreme,
                "max_logfc_threshold": max_logfc,
                "top_extreme_genes": top_extreme,
            },
            suggestion="Check if correct layer was used or if batch effects are present",
        ))

    return issues


# =============================================================================
# Main Validation Function
# =============================================================================

def validate_deg_input(
    adata: AnnData,
    groupby: str = "leiden",
    method: str = "wilcoxon",
    layer: Optional[str] = None,
    target_geneset: str = "MSigDB_Hallmark_2020",
    min_cluster_size: int = 20,
    warn_cluster_size: int = 50,
    imbalance_ratio: float = 20.0,
    check_batch: bool = True,
    batch_key: Optional[str] = None,
    batch_confound_threshold: float = 0.7,
) -> DEGValidityReport:
    """
    Comprehensive validation of DEG input data.

    This function checks all aspects of the data before running
    rank_genes_groups to ensure biologically meaningful results.

    Parameters
    ----------
    adata : AnnData
        AnnData object with clusters.
    groupby : str
        Column in adata.obs containing group assignments.
    method : str
        DEG method: 'wilcoxon', 't-test', 'logreg'.
    layer : str, optional
        Layer to use for DEG. If None, uses adata.X.
    target_geneset : str
        Target gene set database for compatibility check.
    min_cluster_size : int
        Minimum cells per cluster (below this = ERROR).
    warn_cluster_size : int
        Warning threshold for cluster size.
    check_batch : bool
        Whether to check for batch confounding.
    batch_key : str, optional
        Explicit batch key to use.

    Returns
    -------
    DEGValidityReport
        Comprehensive validation report.
    """
    report = DEGValidityReport(
        validated_at=datetime.now().isoformat(),
        groupby=groupby,
        method=method,
        layer_used=layer,
        n_cells_total=adata.n_obs,
    )

    # Check groupby exists
    if groupby not in adata.obs.columns:
        report.add_issue(DEGValidityIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.DATA_QUALITY,
            message=f"Group column '{groupby}' not found in adata.obs",
            suggestion="Run clustering first or specify correct groupby column",
        ))
        return report

    report.n_groups = adata.obs[groupby].nunique()

    # Detect matrix type
    if layer is not None and layer in adata.layers:
        X = adata.layers[layer]
    else:
        X = adata.X

    report.matrix_type = _detect_matrix_type(X)

    # Check layer/method compatibility
    report.issues.extend(_check_layer_method_compatibility(
        report.matrix_type, method, layer
    ))

    # Detect species and gene format
    gene_names = adata.var_names.tolist()
    report.data_species = _detect_species_from_genes(gene_names)

    # Detect gene ID format
    sample_genes = gene_names[:50]
    ensembl_count = sum(1 for g in sample_genes if str(g).startswith('ENS'))
    if ensembl_count > len(sample_genes) * 0.5:
        report.gene_id_format = "ensembl"
    elif all(str(g).isdigit() for g in sample_genes[:10] if g):
        report.gene_id_format = "entrez"
    else:
        report.gene_id_format = "symbol"

    # Check gene-pathway compatibility
    report.issues.extend(_check_gene_pathway_compatibility(
        gene_names[:10], report.gene_id_format, target_geneset
    ))

    # Check species-geneset match
    report.issues.extend(_check_species_geneset_match(
        report.data_species, target_geneset
    ))

    # Check cluster sizes
    report.cluster_sizes, size_issues = _check_cluster_sizes(
        adata, groupby, min_cluster_size, warn_cluster_size
    )
    report.issues.extend(size_issues)

    # Check group imbalance
    report.issues.extend(_check_group_imbalance(
        report.cluster_sizes,
        imbalance_ratio=imbalance_ratio,
    ))

    # Check batch confounding
    if check_batch:
        report.batch_key = batch_key
        correlation, batch_issues = _check_batch_confounding(
            adata,
            groupby,
            batch_key,
            correlation_threshold=batch_confound_threshold,
        )
        report.batch_cluster_correlation = correlation
        if batch_issues:
            report.batch_key = batch_issues[0].details.get("batch_key")
        report.issues.extend(batch_issues)

    # Update validity flags
    for issue in report.issues:
        if issue.severity == IssueSeverity.ERROR:
            report.is_valid = False
        elif issue.severity == IssueSeverity.WARNING:
            report.has_warnings = True

    return report


def validate_deg_output(
    adata: AnnData,
    groupby: str = "leiden",
    key: str = "rank_genes_groups",
    max_logfc: float = 10.0,
) -> List[DEGValidityIssue]:
    """
    Post-DEG validation to check for suspicious results.

    Run this after rank_genes_groups to flag potential issues.
    """
    issues = []

    if key not in adata.uns:
        return issues

    # Check each group for extreme effect sizes
    groups = list(adata.obs[groupby].unique())

    for group in groups[:20]:  # Keep bounded for very large cluster counts
        group_str = str(group)
        try:
            deg_df = sc.get.rank_genes_groups_df(adata, group=group_str)
            group_issues = _check_effect_size_sanity(deg_df, max_logfc)
            for issue in group_issues:
                issue.affected_clusters = [group_str]
                issue.details["cluster"] = group_str
                issue.message = (
                    f"Cluster {group_str}: {issue.message}"
                )
            issues.extend(group_issues)
        except Exception:
            pass

    return issues


# =============================================================================
# Main DEG Runner with Validation
# =============================================================================

def run_validated_deg(
    adata: AnnData,
    groupby: str = "leiden",
    method: str = "wilcoxon",
    n_genes: int = 100,
    layer: Optional[str] = None,
    use_raw: bool = True,
    key_added: str = "rank_genes_groups",
    target_geneset: str = "MSigDB_Hallmark_2020",
    min_cluster_size: int = 20,
    warn_cluster_size: int = 50,
    imbalance_ratio: float = 20.0,
    check_batch: bool = True,
    batch_key: Optional[str] = None,
    batch_confound_threshold: float = 0.7,
    max_logfc: float = 10.0,
    block_on_errors: bool = True,
    inplace: bool = True,
) -> Tuple[Optional[AnnData], DEGValidityReport]:
    """
    Run differential expression with comprehensive validation.

    This is the recommended way to run DEG - it validates inputs,
    runs rank_genes_groups, validates outputs, and attaches
    validity metadata for downstream use.

    Parameters
    ----------
    adata : AnnData
        AnnData object with clusters.
    groupby : str
        Column in adata.obs containing group assignments.
    method : str
        DEG method: 'wilcoxon', 't-test', 'logreg'.
    n_genes : int
        Number of top genes to store per group.
    layer : str, optional
        Layer to use. If None, Wilcoxon uses the active normalized/log1p
        analysis matrix; other methods may opt into a preserved raw-count layer.
    use_raw : bool
        Whether non-Wilcoxon methods may prefer a preserved raw-count layer.
    key_added : str
        Key in adata.uns for results.
    target_geneset : str
        Gene set database for compatibility checks.
    min_cluster_size : int
        Minimum cells per cluster.
    warn_cluster_size : int
        Warning threshold for small clusters.
    check_batch : bool
        Check for batch confounding.
    batch_key : str, optional
        Batch column to use.
    block_on_errors : bool
        If True, raise exception on ERROR severity issues.
    inplace : bool
        Modify adata in place.

    Returns
    -------
    Tuple[AnnData or None, DEGValidityReport]
        Modified AnnData (or None if inplace) and validation report.

    Raises
    ------
    ValueError
        If block_on_errors=True and validation finds ERROR severity issues.
    """
    if not inplace:
        adata = adata.copy()

    # Preserve workshop best practice: keep raw counts in a layer, but run
    # Wilcoxon DEG on the active normalized/log1p analysis matrix unless the
    # caller explicitly requests a different layer.
    actual_layer = layer
    if actual_layer is None and use_raw and method != "wilcoxon":
        for raw_layer in ["raw_counts", "counts", "raw_data"]:
            if raw_layer in adata.layers:
                actual_layer = raw_layer
                logger.info(f"Using '{actual_layer}' layer for DEG")
                break

    # Run pre-validation
    report = validate_deg_input(
        adata,
        groupby=groupby,
        method=method,
        layer=actual_layer,
        target_geneset=target_geneset,
        min_cluster_size=min_cluster_size,
        warn_cluster_size=warn_cluster_size,
        imbalance_ratio=imbalance_ratio,
        check_batch=check_batch,
        batch_key=batch_key,
        batch_confound_threshold=batch_confound_threshold,
    )

    # Log validation results
    logger.info(report.summary())

    # Block on errors if requested
    if block_on_errors and not report.is_valid:
        error_messages = [e.message for e in report.errors]
        raise ValueError(
            f"DEG validation failed with {len(report.errors)} error(s):\n"
            + "\n".join(f"  - {m}" for m in error_messages)
        )

    # Run DEG
    logger.info(f"Running {method} DEG by '{groupby}' ({report.n_groups} groups)")

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        key_added=key_added,
        layer=actual_layer,
    )

    # Run post-validation
    output_issues = validate_deg_output(
        adata,
        groupby=groupby,
        key=key_added,
        max_logfc=max_logfc,
    )
    for issue in output_issues:
        report.add_issue(issue)

    # Store validity report in adata.uns
    adata.uns["deg_validity"] = report.to_dict()

    # Also store a quick-access caveats list
    adata.uns["deg_caveats"] = [
        issue.message for issue in report.issues
        if issue.severity in [IssueSeverity.ERROR, IssueSeverity.WARNING]
    ]

    logger.info(
        f"DEG complete: {report.n_groups} groups, "
        f"{len(report.errors)} errors, {len(report.warnings)} warnings"
    )

    if not inplace:
        return adata, report

    return None, report


def _normalize_validity_issues(issues: Any) -> List[Dict[str, Any]]:
    """Normalize saved issue payloads into a list of dict records."""
    import json
    import numpy as np

    if issues is None:
        return []
    if isinstance(issues, np.ndarray):
        issues = issues.tolist()
    if not isinstance(issues, list):
        return []

    normalized = []
    for issue in issues:
        if isinstance(issue, dict):
            normalized.append(issue)
            continue
        if isinstance(issue, str):
            stripped = issue.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except Exception:
                continue
            if isinstance(parsed, dict):
                normalized.append(parsed)
    return normalized


def get_deg_validity(adata: AnnData) -> Optional[Dict[str, Any]]:
    """
    Retrieve DEG validity metadata from adata.

    Returns None if DEG hasn't been run with validation.
    """
    validity = adata.uns.get("deg_validity")
    if not isinstance(validity, dict):
        return validity
    normalized = dict(validity)
    normalized["issues"] = _normalize_validity_issues(validity.get("issues"))
    return normalized


def get_deg_caveats(adata: AnnData) -> List[str]:
    """
    Get list of DEG caveats for interpretation.

    Returns empty list if no caveats or DEG not run.
    """
    caveats = adata.uns.get("deg_caveats", [])
    if caveats is None:
        return []
    if isinstance(caveats, list):
        return [str(c) for c in caveats if str(c).strip()]
    return [str(caveats)] if str(caveats).strip() else []


def get_cluster_caveats(adata: AnnData, cluster: str) -> List[str]:
    """
    Get caveats specific to a cluster.

    Useful for per-cluster interpretation in GSEA reports.
    """
    validity = get_deg_validity(adata)
    if validity is None or not isinstance(validity, dict):
        return []

    caveats = []
    for issue in validity.get("issues", []):
        if not isinstance(issue, dict):
            continue
        if issue.get("severity") in ["error", "warning"]:
            affected = issue.get("affected_clusters", []) or []
            if not affected or cluster in affected:
                message = issue.get("message")
                if message:
                    caveats.append(message)

    return caveats
