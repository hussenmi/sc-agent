"""
Data loading utilities for scagent.

Supports multiple input formats:
- 10X Genomics h5 files
- AnnData h5ad files
- Matrix Market (mtx) format
"""

import os
from pathlib import Path
from typing import Any, Union, Optional, List
import scanpy as sc
from anndata import AnnData
import logging

logger = logging.getLogger(__name__)


def load_data(
    path: Union[str, Path],
    format: Optional[str] = None,
    make_var_unique: bool = True,
) -> AnnData:
    """
    Load single-cell data from various formats.

    Automatically detects format based on file extension if not specified.

    Parameters
    ----------
    path : str or Path
        Path to the data file or directory.
    format : str, optional
        File format: '10x_h5', 'h5ad', 'mtx'. Auto-detected if None.
    make_var_unique : bool, default True
        Make variable names unique (recommended for 10X data).

    Returns
    -------
    AnnData
        Loaded AnnData object.

    Examples
    --------
    >>> adata = load_data('filtered_feature_bc_matrix.h5')
    >>> adata = load_data('processed.h5ad')
    >>> adata = load_data('matrix.mtx.gz', format='mtx')
    """
    path = Path(path)

    # Auto-detect format
    if format is None:
        suffixes = [s.lower() for s in path.suffixes]
        if path.suffix.lower() == '.h5ad':
            format = 'h5ad'
        elif len(suffixes) >= 2 and suffixes[-2] == '.h5ad' and suffixes[-1] == '.gz':
            format = 'h5ad_gz'
        elif path.suffix.lower() == '.h5':
            format = '10x_h5'
        elif path.suffix.lower() in ['.mtx']:
            format = 'mtx'
        elif path.is_dir():
            # Validate that this looks like a 10x MTX directory before assuming MTX format.
            # Directories containing h5 files (e.g. per-sample download folders) are not MTX.
            mtx_candidates = list(path.glob("matrix.mtx*"))
            h5_candidates = list(path.glob("*.h5"))
            if mtx_candidates:
                format = 'mtx'
            elif h5_candidates:
                raise ValueError(
                    f"'{path}' is a directory containing {len(h5_candidates)} .h5 file(s), "
                    "not a 10x MTX directory. To load multiple samples, use run_code to load "
                    "each file individually with sc.read_10x_h5() and concatenate with "
                    "anndata.concat(). Call .var_names_make_unique() on each before concatenating."
                )
            else:
                raise ValueError(
                    f"'{path}' is a directory but does not contain matrix.mtx or matrix.mtx.gz. "
                    "Expected a 10x MTX directory with matrix.mtx.gz, barcodes.tsv.gz, and features.tsv.gz."
                )
        elif path.suffix.lower() == '.gz':
            # .gz but not .h5ad.gz — assume MTX directory (e.g. matrix.mtx.gz passed directly)
            format = 'mtx'
        else:
            raise ValueError(
                f"Cannot auto-detect format for {path}. "
                "Please specify format='10x_h5', 'h5ad', 'h5ad_gz', or 'mtx'"
            )

    # Load data
    if format == '10x_h5':
        adata = load_10x_h5(path, make_var_unique=make_var_unique)
    elif format == 'h5ad':
        adata = load_h5ad(path)
    elif format == 'h5ad_gz':
        adata = load_h5ad_gz(path)
    elif format == 'mtx':
        adata = load_mtx(path, make_var_unique=make_var_unique)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Loaded data: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


def load_10x_h5(
    path: Union[str, Path],
    make_var_unique: bool = True,
) -> AnnData:
    """
    Load 10X Genomics h5 file.

    Parameters
    ----------
    path : str or Path
        Path to the h5 file.
    make_var_unique : bool, default True
        Make variable names unique.

    Returns
    -------
    AnnData
        Loaded AnnData object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading 10X h5 file: {path}")
    adata = sc.read_10x_h5(str(path))

    if make_var_unique:
        adata.var_names_make_unique()

    return adata


def load_h5ad(path: Union[str, Path]) -> AnnData:
    """
    Load AnnData h5ad file.

    Parameters
    ----------
    path : str or Path
        Path to the h5ad file.

    Returns
    -------
    AnnData
        Loaded AnnData object.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading h5ad file: {path}")
    return sc.read_h5ad(str(path))


def load_h5ad_gz(path: Union[str, Path]) -> AnnData:
    """
    Load a gzip-compressed h5ad file (e.g. dataset.h5ad.gz from GEO).

    Decompresses to a temporary file, reads it, then cleans up.
    Shows a progress indicator because decompression of large files can take time.

    Parameters
    ----------
    path : str or Path
        Path to the .h5ad.gz file.

    Returns
    -------
    AnnData
        Loaded AnnData object.
    """
    import gzip
    import shutil
    import tempfile

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    file_size_mb = path.stat().st_size / (1024 ** 2)
    logger.info(f"Decompressing {path.name} ({file_size_mb:.0f} MB compressed) ...")
    print(f"  Decompressing {path.name} ({file_size_mb:.0f} MB compressed) — this may take a minute...")

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with gzip.open(str(path), "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=64 * 1024 * 1024)  # 64 MB chunks
        decompressed_mb = Path(tmp_path).stat().st_size / (1024 ** 2)
        logger.info(f"Decompressed to {decompressed_mb:.0f} MB, loading h5ad ...")
        print(f"  Decompressed ({decompressed_mb:.0f} MB), loading ...")
        return sc.read_h5ad(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def load_mtx(
    path: Union[str, Path],
    make_var_unique: bool = True,
) -> AnnData:
    """
    Load Matrix Market format (10X-style directory).

    Expected directory structure:
    - matrix.mtx.gz
    - barcodes.tsv.gz
    - features.tsv.gz (or genes.tsv.gz)

    Parameters
    ----------
    path : str or Path
        Path to the mtx file or directory containing mtx files.
    make_var_unique : bool, default True
        Make variable names unique.

    Returns
    -------
    AnnData
        Loaded AnnData object.
    """
    path = Path(path)

    # If path is a file, get its parent directory
    if path.is_file():
        path = path.parent

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    logger.info(f"Loading MTX from directory: {path}")
    adata = sc.read_10x_mtx(str(path))

    if make_var_unique:
        adata.var_names_make_unique()

    return adata


def concat_datasets(
    datasets: List[AnnData],
    batch_key: str = 'batch_id',
    batch_names: Optional[List[str]] = None,
    join: str = 'outer',
    **kwargs: Any,
) -> AnnData:
    """
    Concatenate multiple AnnData objects.

    Parameters
    ----------
    datasets : List[AnnData]
        List of AnnData objects to concatenate.
    batch_key : str, default 'batch_id'
        Key to store batch information in obs.
    batch_names : List[str], optional
        Names for each batch. If None, uses integer indices.
    join : {'outer', 'inner'}, default 'outer'
        Outer keeps every gene observed in any dataset. Inner keeps only genes
        shared by every dataset.
    **kwargs
        Tolerated ``anndata.concat``-style aliases so a near-miss call form does
        not dead-end the load step (callers frequently conflate the two APIs):
        ``label=`` (alias for ``batch_key``), ``keys=`` (alias for
        ``batch_names``), and ``fill_value=`` (honored; otherwise 0 for outer
        joins, None for inner). Structural kwargs this helper manages itself
        (``axis``, ``index_unique``, ``merge``) are accepted and ignored. Any
        other unexpected kwarg raises a TypeError naming the accepted parameters.

    Returns
    -------
    AnnData
        Concatenated AnnData object.
    """
    import anndata

    # Tolerate the anndata.concat-style kwargs models commonly reach for, so a
    # single near-miss does not cascade into hand-rolled concat fallbacks.
    if batch_key == 'batch_id' and kwargs.get('label'):
        batch_key = kwargs.pop('label')
    else:
        kwargs.pop('label', None)
    if batch_names is None and kwargs.get('keys') is not None:
        batch_names = kwargs.pop('keys')
    else:
        kwargs.pop('keys', None)
    has_fill_override = 'fill_value' in kwargs
    fill_value_override = kwargs.pop('fill_value', None)
    for _structural in ('axis', 'index_unique', 'merge'):
        kwargs.pop(_structural, None)
    if kwargs:
        raise TypeError(
            f"concat_datasets() got unexpected keyword argument(s) {sorted(kwargs)}. "
            "Accepted: datasets, batch_key, batch_names, join ('outer'/'inner'). "
            "anndata.concat aliases label=, keys=, fill_value= are tolerated; "
            "axis/index_unique/merge are managed internally."
        )

    if not datasets:
        raise ValueError("datasets must contain at least one AnnData object")

    if batch_names is None:
        batch_names = [str(i) for i in range(len(datasets))]
    else:
        batch_names = [str(name).strip() for name in batch_names]

    if len(batch_names) != len(datasets):
        raise ValueError(
            f"batch_names has {len(batch_names)} entries for {len(datasets)} datasets"
        )
    if any(not name for name in batch_names):
        raise ValueError("batch_names cannot contain empty values")
    if len(set(batch_names)) != len(batch_names):
        raise ValueError("batch_names must be unique")
    if join not in {"outer", "inner"}:
        raise ValueError("join must be either 'outer' or 'inner'")

    expected_counts = {}
    for dataset, name in zip(datasets, batch_names, strict=True):
        dataset.var_names_make_unique()
        expected_counts[name] = int(dataset.n_obs)

    logger.info(f"Concatenating {len(datasets)} datasets")

    fill_value = fill_value_override if has_fill_override else (0 if join == "outer" else None)
    adata = anndata.concat(
        datasets,
        axis=0,
        join=join,
        label=batch_key,
        keys=batch_names,
        index_unique='-',
        fill_value=fill_value,
        merge='same',
    )

    observed_counts = {
        str(name): int(count)
        for name, count in adata.obs[batch_key].value_counts().to_dict().items()
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"Concatenation produced unexpected '{batch_key}' labels: "
            f"expected {expected_counts}, observed {observed_counts}"
        )
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    if adata.obs_names.duplicated().any():
        raise RuntimeError("Concatenation produced duplicate cell barcodes")

    logger.info(f"Concatenated shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata


def _peek_tabular(candidate: Path) -> Optional[dict]:
    """Cheaply classify a text file as a likely count matrix without loading it.

    Reads only the header and the first data row (gzip-aware) and infers the
    delimiter, column count, and whether the first column looks like a row label
    (gene id / barcode). Returns None for files that do not look tabular (e.g.
    binary, or a single-column list). This is what lets discovery recognize the
    common "one CSV/TSV count matrix per sample" layout without us maintaining a
    hard list of every extension a matrix might use.
    """
    import csv
    import gzip
    import io as _io

    name = candidate.name.lower()
    opener = gzip.open if name.endswith(".gz") else open
    try:
        with opener(candidate, "rt", errors="replace") as handle:  # type: ignore[operator]
            header = handle.readline()
            first_row = handle.readline()
    except (OSError, UnicodeError):
        return None
    if not header:
        return None
    # Sniff the delimiter from the header; fall back to the delimiter that yields
    # the most fields. A real matrix has many columns (cells), so a lone-column
    # "matrix" is almost certainly a list/metadata file, not a count matrix.
    sample = header + (first_row or "")
    delimiter = None
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;| ").delimiter
    except csv.Error:
        counts = {d: header.count(d) for d in (",", "\t", ";", "|")}
        delimiter = max(counts, key=counts.get) if any(counts.values()) else None
    if not delimiter:
        return None
    n_fields = len(next(csv.reader(_io.StringIO(header), delimiter=delimiter)))
    if n_fields < 2:
        return None
    first_cell = ""
    if first_row:
        row_fields = next(csv.reader(_io.StringIO(first_row), delimiter=delimiter), [])
        first_cell = row_fields[0] if row_fields else ""
    return {
        "delimiter": delimiter,
        "n_header_fields": n_fields,
        "first_cell": first_cell,
    }


def discover_data_inputs(path: Union[str, Path]) -> dict:
    """Describe supported single-cell inputs without loading them.

    Recognizes the binary/standard formats (h5ad, 10x h5, loom, mtx, 10x mtx
    directories) and, via a cheap header peek, plain-text count matrices
    (csv/tsv/txt, optionally gzipped). ``source_datasets`` — the set that drives
    the ``multi_dataset_loading`` decision — is the largest group of files that
    share a structural signature (same format, and for text the same delimiter
    and comparable column count), i.e. a replicate set that plausibly combines
    into one object. When no such group of >=2 exists, every recognized dataset
    is returned so a single explicit input still loads. This is deliberately
    structural rather than an extension whitelist: a folder of 40 per-sample CSVs
    registers as 40 source datasets (so the combine decision fires) while a stray
    ``metadata.csv`` sitting beside real matrices does not hijack the group.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    def _binary_format_for(candidate: Path) -> Optional[str]:
        lower = candidate.name.lower()
        if lower.endswith(".h5ad.gz"):
            return "h5ad_gz"
        if lower.endswith(".h5ad"):
            return "h5ad"
        if lower.endswith(".h5"):
            return "10x_h5"
        if lower.endswith(".loom"):
            return "loom"
        if lower.endswith(".mtx") or lower.endswith(".mtx.gz"):
            return "mtx"
        return None

    def _tabular_format_for(candidate: Path) -> Optional[str]:
        lower = candidate.name.lower()
        for stem, fmt in ((".csv", "csv"), (".tsv", "tsv"), (".txt", "txt")):
            if lower.endswith(stem) or lower.endswith(stem + ".gz"):
                return fmt
        return None

    def _likely_combined(candidate: Path) -> bool:
        name = candidate.name.lower()
        return any(token in name for token in ("combined", "concatenated", "merged"))

    # A directory that IS a 10x mtx bundle is ONE dataset — its loose
    # barcodes/features/matrix component files must not be mistaken for separate
    # per-sample tables (the tsv/csv peek below would otherwise pick up
    # features.tsv). Short-circuit to a single-dataset result.
    if root.is_dir() and any(root.glob("matrix.mtx*")):
        entry = {
            "path": str(root),
            "name": root.name,
            "format": "10x_mtx_directory",
            "size_bytes": None,
            "likely_combined_output": _likely_combined(root),
        }
        return {
            "path": str(root),
            "datasets": [entry],
            "source_datasets": [entry],
            "likely_combined_outputs": [],
            "excluded_datasets": [],
            "n_datasets": 1,
            "n_source_datasets": 1,
        }

    candidates = [root] if root.is_file() else sorted(root.iterdir())
    datasets = []
    for candidate in candidates:
        data_format: Optional[str] = None
        tabular: Optional[dict] = None
        if candidate.is_dir():
            if any(candidate.glob("matrix.mtx*")):
                data_format = "10x_mtx_directory"
        elif candidate.is_file():
            data_format = _binary_format_for(candidate)
            if data_format is None and _tabular_format_for(candidate) is not None:
                tabular = _peek_tabular(candidate)
                if tabular is not None:
                    data_format = _tabular_format_for(candidate)
        if data_format is None:
            continue
        entry = {
            "path": str(candidate),
            "name": candidate.name,
            "format": data_format,
            "size_bytes": candidate.stat().st_size if candidate.is_file() else None,
            "likely_combined_output": _likely_combined(candidate),
        }
        if tabular is not None:
            # Load hints so a consumer knows how to read the matrix without
            # re-sniffing (genes-as-rows is the common orientation for these).
            entry["delimiter"] = tabular["delimiter"]
            entry["n_header_fields"] = tabular["n_header_fields"]
        datasets.append(entry)

    def _signature(dataset: dict) -> tuple:
        # Files combine into one object only if they are structurally alike. For
        # text matrices, bucket by column count (log2) so per-sample matrices with
        # slightly different cell counts still group, while a metadata table with a
        # handful of columns does not join a group of wide count matrices.
        fmt = dataset["format"]
        if "n_header_fields" in dataset:
            import math
            bucket = int(math.log2(max(dataset["n_header_fields"], 1)))
            return (fmt, dataset.get("delimiter"), bucket)
        return (fmt,)

    non_combined = [d for d in datasets if not d["likely_combined_output"]]
    groups: dict = {}
    for dataset in non_combined:
        groups.setdefault(_signature(dataset), []).append(dataset)
    replicate_groups = [g for g in groups.values() if len(g) >= 2]
    if replicate_groups:
        # The largest structurally-consistent replicate set is the source group.
        source_datasets = max(replicate_groups, key=len)
    else:
        source_datasets = non_combined if non_combined else list(datasets)

    source_paths = {d["path"] for d in source_datasets}
    return {
        "path": str(root),
        "datasets": datasets,
        "source_datasets": source_datasets,
        "likely_combined_outputs": [
            dataset for dataset in datasets if dataset["likely_combined_output"]
        ],
        # Recognized datasets excluded from the source group (name-flagged combined
        # outputs, or structural odd-ones-out like a stray metadata table beside a
        # replicate set). Surfaced so a consumer can see what was set aside and why.
        "excluded_datasets": [
            dataset for dataset in datasets if dataset["path"] not in source_paths
        ],
        "n_datasets": len(datasets),
        "n_source_datasets": len(source_datasets),
    }
