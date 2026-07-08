"""
Robust gene-identifier handling for scagent.

Single source of truth for:
  * classifying var_names as Ensembl IDs / Entrez IDs / gene symbols,
  * finding the column in ``var`` that carries gene symbols (or Ensembl IDs),
  * converting ``var_names`` to gene symbols so downstream tools that expect
    HGNC/MGI symbols (SCimilarity, CellTypist, marker overlap, plotting) work.

Design constraints (see AGENTS.md):
  * **Offline-first.** The authoritative, network-free path is the dataset's own
    symbol column in ``var`` — ``feature_name`` (CELLxGENE), ``gene_symbols``
    (10x/CellRanger), etc. ``mygene`` is only an optional last-resort fallback
    when no such column exists, and it fails soft when there is no network.
  * **Content-validated, not name-matched.** A column is only trusted as a symbol
    column if its *contents* actually look like symbols, so an oddly-named column
    still gets picked up and a mislabelled one is rejected.
  * **Non-destructive.** Original identifiers are preserved into ``var`` before
    ``var_names`` is overwritten, and genes are never dropped — duplicate symbols
    are made unique rather than deleted.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import pandas as pd
from anndata import AnnData

# --- identifier patterns -------------------------------------------------------
# Ensembl gene IDs across species: ENSG (human), ENSMUSG (mouse), ENSRNOG, ...
# Optional trailing version suffix (".12") is tolerated by the classifier.
_ENSEMBL_RE = re.compile(r"^ENS[A-Z]{0,4}G\d{6,}(?:\.\d+)?$")
_ENTREZ_RE = re.compile(r"^\d{1,9}$")
# A gene symbol: starts with a letter, then letters/digits/-/./_ (e.g. CD3D,
# HLA-DRB1, A1BG-AS1, MT-CO1, RP11-1.2). Deliberately permissive.
_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\.\-_]{0,30}$")
# Multi-genome CellRanger prefixes: "GRCh38_CD3D", "GRCh38___CD3D", "hg19_ACTB".
_GENOME_PREFIX_RE = re.compile(r"^([A-Za-z0-9]+_{1,10})(?=[A-Za-z])")
# Suffix added by AnnData.var_names_make_unique(): "CD3D-1" -> "CD3D".
_MAKE_UNIQUE_SUFFIX_RE = re.compile(r"^(.+)-\d+$")

# Ordered candidate column names, most-specific/most-common first. Presence is a
# hint only; contents are always validated before a column is trusted.
_SYMBOL_COL_CANDIDATES = [
    "feature_name",   # CELLxGENE
    "gene_symbols",   # 10x / CellRanger features.tsv
    "gene_symbol",
    "gene_name",
    "gene_names",
    "hgnc_symbol",
    "mgi_symbol",
    "symbol",
    "Symbol",
    "SYMBOL",
    "GeneSymbol",
    "Gene",
    "gene",
]
_ENSEMBL_COL_CANDIDATES = [
    "gene_ids",       # 10x / scanpy sc.read_10x_mtx default
    "gene_id",
    "ensembl_id",
    "ensembl_ids",
    "ensembl",
    "ENSEMBL",
    "feature_id",
]

_SAMPLE_SIZE = 500


# --- classification ------------------------------------------------------------
def classify_gene_id(name: str) -> str:
    """Classify a single identifier as 'ensembl' | 'entrez' | 'symbol' | 'other'."""
    s = str(name)
    if _ENSEMBL_RE.match(s):
        return "ensembl"
    if _ENTREZ_RE.match(s):
        return "entrez"
    if _SYMBOL_RE.match(s):
        return "symbol"
    return "other"


def infer_id_format(names) -> str:
    """Infer the dominant identifier format of a collection of names.

    Returns 'ensembl' | 'entrez' | 'symbol' | 'mixed' | 'unknown'. Genome
    prefixes are stripped before classification so 'GRCh38_CD3D' reads as a
    symbol.
    """
    names = [str(n) for n in names]
    if not names:
        return "unknown"
    sample, _ = strip_genome_prefix(names[:_SAMPLE_SIZE])
    n = len(sample)
    counts = {"ensembl": 0, "entrez": 0, "symbol": 0, "other": 0}
    for g in sample:
        counts[classify_gene_id(g)] += 1
    if counts["ensembl"] > n * 0.5:
        return "ensembl"
    if counts["entrez"] > n * 0.5:
        return "entrez"
    if counts["symbol"] > n * 0.5:
        return "symbol"
    if counts["ensembl"] > 0 and counts["symbol"] > 0:
        return "mixed"
    return "unknown"


def strip_ensembl_version(name: str) -> str:
    """'ENSG00000000003.14' -> 'ENSG00000000003'; other names unchanged."""
    s = str(name)
    if _ENSEMBL_RE.match(s) and "." in s:
        return s.split(".", 1)[0]
    return s


def strip_genome_prefix(names) -> Tuple[List[str], Optional[str]]:
    """Strip a dominant multi-genome CellRanger prefix from names.

    Returns (new_names, prefix). If fewer than 30% of names share a single
    prefix, nothing is stripped and prefix is None.
    """
    names = [str(n) for n in names]
    prefix_counts: dict = {}
    for g in names[:_SAMPLE_SIZE]:
        m = _GENOME_PREFIX_RE.match(g)
        if m:
            prefix_counts[m.group(1)] = prefix_counts.get(m.group(1), 0) + 1
    if not prefix_counts:
        return names, None
    top_prefix, top_count = max(prefix_counts.items(), key=lambda x: x[1])
    if top_count <= min(len(names), _SAMPLE_SIZE) * 0.3:
        return names, None
    stripped = [
        g[len(top_prefix):] if g.startswith(top_prefix) else g for g in names
    ]
    return stripped, top_prefix


def _column_values(adata: AnnData, col: str) -> List[str]:
    ser = adata.var[col]
    if isinstance(ser.dtype, pd.CategoricalDtype):
        ser = ser.astype(str)
    return [str(v) for v in ser.tolist()]


def _looks_like_symbols(values) -> bool:
    """True if the majority of values look like gene symbols (and not Ensembl)."""
    vals = [str(v) for v in values[:_SAMPLE_SIZE] if v not in (None, "", "nan", "NaN")]
    if not vals:
        return False
    good = sum(
        1
        for v in vals
        if classify_gene_id(v) == "symbol" and not _ENSEMBL_RE.match(v)
    )
    return good > len(vals) * 0.5


def _looks_like_ensembl(values) -> bool:
    vals = [str(v) for v in values[:_SAMPLE_SIZE] if v not in (None, "", "nan", "NaN")]
    if not vals:
        return False
    return sum(1 for v in vals if _ENSEMBL_RE.match(v)) > len(vals) * 0.5


def find_symbol_column(adata: AnnData) -> Optional[str]:
    """Return the name of the ``var`` column holding gene symbols, or None.

    Tries the well-known candidate names first (validating contents), then falls
    back to scanning every column for one whose contents look like symbols.
    """
    cols = list(adata.var.columns)
    for cand in _SYMBOL_COL_CANDIDATES:
        if cand in cols and _looks_like_symbols(_column_values(adata, cand)):
            return cand
    # Fallback: any column whose contents look like symbols.
    for col in cols:
        try:
            if _looks_like_symbols(_column_values(adata, col)):
                return col
        except Exception:
            continue
    return None


def find_ensembl_column(adata: AnnData) -> Optional[str]:
    """Return the name of the ``var`` column holding Ensembl IDs, or None."""
    cols = list(adata.var.columns)
    for cand in _ENSEMBL_COL_CANDIDATES:
        if cand in cols and _looks_like_ensembl(_column_values(adata, cand)):
            return cand
    for col in cols:
        try:
            if _looks_like_ensembl(_column_values(adata, col)):
                return col
        except Exception:
            continue
    return None


def needs_symbol_conversion(adata: AnnData) -> bool:
    """True if var_names are not already gene symbols (conversion would help)."""
    return infer_id_format(adata.var_names) != "symbol"


# --- report --------------------------------------------------------------------
@dataclass
class GeneConversionReport:
    changed: bool = False
    from_format: str = "unknown"
    to_format: str = "unknown"
    source: str = "none"          # 'column:<name>' | 'mygene' | 'already_symbols'
    symbol_column: Optional[str] = None
    genome_prefix_stripped: Optional[str] = None
    n_genes: int = 0
    n_mapped: int = 0             # genes whose name became a real symbol
    n_unmapped: int = 0          # genes left as their original id (no symbol found)
    n_duplicates_made_unique: int = 0
    original_ids_saved_to: Optional[str] = None
    message: str = ""
    sample_before: List[str] = field(default_factory=list)
    sample_after: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- conversion ----------------------------------------------------------------
def convert_var_to_symbols(
    adata: AnnData,
    *,
    inplace: bool = False,
    use_mygene: bool = False,
    organism: Optional[str] = None,
) -> Tuple[AnnData, GeneConversionReport]:
    """Convert ``adata.var_names`` to gene symbols.

    Steps, in order (each skipped when not applicable):
      1. Strip a dominant multi-genome prefix (``GRCh38_CD3D`` -> ``CD3D``).
      2. If already symbols, stop (only prefix-stripping counts as a change).
      3. Preserve the original identifiers into a ``var`` column.
      4. Map to symbols using the dataset's own symbol column (offline,
         authoritative); genes with no symbol keep their original id.
      5. Optionally, if no symbol column exists and ``use_mygene`` is set, query
         mygene.info (best-effort, fails soft when offline).
      6. Make duplicate symbols unique (no genes dropped).

    Returns (adata, report). With ``inplace=False`` a copy is returned and the
    input is untouched.
    """
    if not inplace:
        adata = adata.copy()

    report = GeneConversionReport(n_genes=adata.n_vars)
    original = [str(n) for n in adata.var_names.tolist()]
    report.sample_before = original[:10]
    report.from_format = infer_id_format(original)

    # 1. genome prefix
    names, prefix = strip_genome_prefix(original)
    if prefix:
        report.genome_prefix_stripped = prefix
        report.changed = True

    fmt_after_prefix = infer_id_format(names)

    # 2. already symbols -> done (prefix strip may still have changed things)
    if fmt_after_prefix == "symbol":
        report.to_format = "symbol"
        report.source = "already_symbols" if not prefix else f"genome_prefix:{prefix}"
        report.n_mapped = report.n_genes
        _assign_unique(adata, names, report)
        report.message = (
            "var_names already gene symbols" if not prefix
            else f"stripped genome prefix '{prefix}'"
        )
        report.sample_after = [str(n) for n in adata.var_names[:10]]
        return adata, report

    # 3. preserve original ids
    if report.from_format == "ensembl":
        save_col = "ensembl_id" if "ensembl_id" not in adata.var.columns else "ensembl_id_original"
        if save_col not in adata.var.columns:
            adata.var[save_col] = original
            report.original_ids_saved_to = save_col

    # 4. map via in-file symbol column
    symbol_col = find_symbol_column(adata)
    mapped_names: Optional[List[str]] = None
    if symbol_col is not None:
        col_vals = _column_values(adata, symbol_col)
        mapped_names = [
            col_vals[i]
            if col_vals[i] not in (None, "", "nan", "NaN") and col_vals[i].lower() != "nan"
            else names[i]
            for i in range(len(names))
        ]
        report.source = f"column:{symbol_col}"
        report.symbol_column = symbol_col

    # 5. optional mygene fallback
    elif use_mygene and fmt_after_prefix in ("ensembl", "entrez"):
        mapped_names = _mygene_map(names, fmt_after_prefix, organism)
        if mapped_names is not None:
            report.source = "mygene"

    if mapped_names is None:
        # No symbol source available — leave names as-is (minus any prefix).
        report.to_format = fmt_after_prefix
        report.n_unmapped = report.n_genes
        _assign_unique(adata, names, report)
        report.message = (
            "No gene-symbol column found in var and no mygene mapping available; "
            "var_names left unchanged. Downstream tools expecting symbols may fail."
        )
        report.sample_after = [str(n) for n in adata.var_names[:10]]
        return adata, report

    # count how many actually became symbols (differ from their original id)
    n_mapped = sum(
        1
        for orig, new in zip(names, mapped_names, strict=False)
        if str(new) != str(orig) and classify_gene_id(str(new)) == "symbol"
    )
    report.n_mapped = n_mapped
    report.n_unmapped = report.n_genes - n_mapped
    report.to_format = "symbol"
    report.changed = True
    _assign_unique(adata, [str(n) for n in mapped_names], report)
    report.message = (
        f"Mapped {n_mapped}/{report.n_genes} var_names to gene symbols via "
        f"{report.source}."
    )
    report.sample_after = [str(n) for n in adata.var_names[:10]]
    return adata, report


def _assign_unique(adata: AnnData, names: List[str], report: GeneConversionReport) -> None:
    """Assign names to var_names and make duplicates unique without dropping genes."""
    adata.var_names = pd.Index([str(n) for n in names])
    if not adata.var_names.is_unique:
        n_before_dupe = int(adata.var_names.duplicated().sum())
        adata.var_names_make_unique()
        report.n_duplicates_made_unique = n_before_dupe


def _mygene_map(
    names: List[str], fmt: str, organism: Optional[str]
) -> Optional[List[str]]:
    """Best-effort Ensembl/Entrez -> symbol mapping via mygene. Fails soft.

    Returns a list aligned to ``names`` (unmapped entries keep their original id),
    or None if mygene is unavailable or the query fails (e.g. no network).
    """
    try:
        import mygene  # type: ignore
    except Exception:
        return None
    scopes = "ensembl.gene" if fmt == "ensembl" else "entrezgene"
    species = {"human": "human", "mouse": "mouse"}.get(
        str(organism).lower() if organism else "", "all"
    )
    query = [strip_ensembl_version(n) for n in names]
    try:
        mg = mygene.MyGeneInfo()
        res = mg.querymany(
            query,
            scopes=scopes,
            fields="symbol",
            species=species,
            as_dataframe=False,
            verbose=False,
        )
    except Exception:
        return None
    best: dict = {}
    for hit in res:
        q = hit.get("query")
        sym = hit.get("symbol")
        if q and sym and q not in best:
            best[q] = sym
    return [best.get(q, orig) for q, orig in zip(query, names, strict=False)]
