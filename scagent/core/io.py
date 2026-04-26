"""
Data loading utilities for scagent.

Supports multiple input formats:
- 10X Genomics h5 files
- AnnData h5ad files
- Matrix Market (mtx) format
"""

import os
from pathlib import Path
from typing import Union, Optional, List
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

    Returns
    -------
    AnnData
        Concatenated AnnData object.
    """
    import anndata

    if batch_names is None:
        batch_names = [str(i) for i in range(len(datasets))]

    logger.info(f"Concatenating {len(datasets)} datasets")

    adata = anndata.concat(
        datasets,
        axis=0,
        join='outer',
        label=batch_key,
        keys=batch_names,
        index_unique='-',
        fill_value=0,
    )

    logger.info(f"Concatenated shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return adata
