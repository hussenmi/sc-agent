"""
Normalization methods for scagent.

Implements:
- Raw counts preservation
- Library size normalization
- Log transformation
"""

from typing import Optional, Sequence
import re
import numpy as np
import scanpy as sc
from anndata import AnnData
import logging

logger = logging.getLogger(__name__)


def _matrix_integer_like(matrix, n_rows: int = 100, n_cols: int = 100) -> bool:
    """Return True when a small matrix sample looks like raw integer counts."""
    if matrix is None:
        return False
    sample = matrix[:min(n_rows, matrix.shape[0]), :min(n_cols, matrix.shape[1])]
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    sample = np.asarray(sample)
    return bool(np.allclose(sample, np.round(sample)))


def preserve_raw_counts(
    adata: AnnData,
    layer_name: str = 'raw_counts',
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Preserve raw counts in a layer before normalization.

    CRITICAL: This should be called BEFORE any normalization.

    Parameters
    ----------
    adata : AnnData
        AnnData object with raw counts in X.
    layer_name : str, default 'raw_counts'
        Name for the layer to store raw counts.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    if layer_name in adata.layers:
        logger.warning(f"Layer '{layer_name}' already exists, overwriting")

    adata.layers[layer_name] = adata.X.copy()
    logger.info(f"Raw counts preserved in layer '{layer_name}'")

    if not inplace:
        return adata


def normalize_data(
    adata: AnnData,
    target_sum: Optional[float] = None,
    log_transform: bool = True,
    preserve_raw: bool = True,
    raw_layer_name: str = 'raw_counts',
    force_reset_from_raw: bool = False,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Normalize and optionally log-transform data.

    If ``adata`` has already been normalized (scanpy sets ``adata.uns['log1p']``
    after ``sc.pp.log1p``), this function resets ``adata.X`` from the raw-counts
    layer before normalizing again. This prevents silent double-normalization
    when the tool is re-run with different parameters. If no raw-counts layer
    is available in that situation, ``ValueError`` is raised rather than
    producing a corrupt matrix.

    Parameters
    ----------
    adata : AnnData
        AnnData object with count data.
    target_sum : float, optional
        Target sum for normalization. If None, uses median library size.
    log_transform : bool, default True
        Apply log1p transformation after normalization.
    preserve_raw : bool, default True
        Preserve raw counts in a layer before normalization.
    raw_layer_name : str, default 'raw_counts'
        Name for the raw counts layer.
    force_reset_from_raw : bool, default False
        If true and ``raw_layer_name`` exists, reset ``adata.X`` from that
        layer before normalizing even when ``adata.uns['log1p']`` is absent.
        This prevents retry paths from accidentally normalizing an already
        normalized matrix.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.

    Raises
    ------
    ValueError
        If ``adata.X`` appears already-normalized and no raw-counts layer
        is available to reset from.
    """
    if not inplace:
        adata = adata.copy()

    reset_from_raw = False
    if force_reset_from_raw:
        if raw_layer_name in adata.layers:
            if not _matrix_integer_like(adata.layers[raw_layer_name]):
                raise ValueError(
                    f"Layer '{raw_layer_name}' exists but does not look like "
                    "raw integer counts. Refusing to use it for normalization reset."
                )
            logger.info(
                "Resetting adata.X from layers['%s'] before normalization.",
                raw_layer_name,
            )
            adata.X = adata.layers[raw_layer_name].copy()
            adata.uns.pop('log1p', None)
            reset_from_raw = True
        elif 'log1p' in adata.uns:
            raise ValueError(
                "force_reset_from_raw=True was requested, but no raw-counts "
                f"layer named '{raw_layer_name}' is available. Re-normalizing "
                "would risk double-normalization."
            )

    already_normalized = 'log1p' in adata.uns

    if already_normalized:
        if raw_layer_name in adata.layers:
            logger.warning(
                "adata.uns['log1p'] is set — resetting adata.X from "
                "layers['%s'] before re-normalizing to avoid double "
                "normalization.",
                raw_layer_name,
            )
            adata.X = adata.layers[raw_layer_name].copy()
            reset_from_raw = True
            # Clear the log1p marker so scanpy does not refuse log1p again
            # and so downstream tools do not think X is still log-scaled.
            adata.uns.pop('log1p', None)
        else:
            raise ValueError(
                "adata appears already normalized (adata.uns['log1p'] is set) "
                f"and no raw-counts layer named '{raw_layer_name}' is available "
                "to reset from. Re-normalizing would corrupt adata.X. "
                "Load the dataset fresh from raw counts, or copy raw counts "
                f"into adata.layers['{raw_layer_name}'] before retrying."
            )

    # Preserve raw counts (after any reset, the current X is the raw matrix)
    if preserve_raw and raw_layer_name not in adata.layers:
        if not _matrix_integer_like(adata.X):
            raise ValueError(
                "No raw-counts layer is available, and adata.X does not look "
                "like raw integer counts. Refusing to preserve the current X as "
                f"layers['{raw_layer_name}'] because it may already be normalized."
            )
        preserve_raw_counts(adata, layer_name=raw_layer_name, inplace=True)

    # Normalize
    if target_sum is not None:
        logger.info(f"Normalizing to target_sum={target_sum}")
    else:
        logger.info("Normalizing to median library size")

    sc.pp.normalize_total(adata, target_sum=target_sum, inplace=True)

    # Log transform
    if log_transform:
        sc.pp.log1p(adata)
        logger.info("Applied log1p transformation")

    adata.uns["normalization"] = {
        "target_sum": target_sum,
        "log_transform": bool(log_transform),
        "raw_layer_name": raw_layer_name,
        "preserve_raw": bool(preserve_raw),
        "reset_from_raw_counts": bool(reset_from_raw),
    }

    if not inplace:
        return adata


def _normalize_patterns(patterns: Optional[str | Sequence[str]]) -> list[str]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        return [patterns]
    return [str(pattern) for pattern in patterns if str(pattern)]


def _feature_exclusion_mask(
    var_names,
    patterns: Optional[str | Sequence[str]],
    *,
    match_mode: str = "match",
) -> np.ndarray:
    """Return a boolean mask for source-defined HVG/PCA feature exclusions."""
    names = np.asarray([str(name) for name in var_names])
    normalized = _normalize_patterns(patterns)
    if not normalized:
        return np.zeros(len(names), dtype=bool)

    mask = np.zeros(len(names), dtype=bool)
    for pattern in normalized:
        regex = re.compile(pattern)
        if match_mode == "contains":
            mask |= np.asarray([bool(regex.search(name)) for name in names])
        elif match_mode == "fullmatch":
            mask |= np.asarray([bool(regex.fullmatch(name)) for name in names])
        else:
            mask |= np.asarray([bool(regex.match(name)) for name in names])
    return mask


def select_hvg(
    adata: AnnData,
    n_top_genes: int = 4000,
    flavor: str = 'seurat_v3',
    layer: Optional[str] = None,
    batch_key: Optional[str] = None,
    exclude_patterns: Optional[str | Sequence[str]] = None,
    exclusion_mode: str = "post",
    exclude_match_mode: str = "match",
    exclusion_source: Optional[str] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Select highly variable genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    n_top_genes : int, default 4000
        Number of top highly variable genes to select.
    flavor : str, default 'seurat_v3'
        Flavor for HVG selection. 'seurat_v3' requires raw counts.
    layer : str, optional
        Layer to use for HVG calculation. If None and flavor='seurat_v3',
        tries to use 'raw_counts' or 'raw_data' layer.
    batch_key : str, optional
        obs column to stratify HVG selection by batch. Recommended for
        multi-sample data to avoid selecting batch-specific genes.
        Supported by all flavors including 'seurat_v3' (ranks genes by
        median rank across batches).
    exclude_patterns : str or sequence of str, optional
        Regex pattern(s) for source-defined features that must not drive HVG,
        PCA, or clustering. This is generic provenance-driven behavior; do
        not hard-code dataset-specific patterns here.
    exclusion_mode : {'post', 'pre'}, default 'post'
        ``post`` matches workflows that run HVG selection then force excluded
        features to ``highly_variable=False``. ``pre`` computes HVGs only on
        allowed features and copies flags back to the full AnnData.
    exclude_match_mode : {'match', 'contains', 'fullmatch'}, default 'match'
        Regex matching mode applied to var_names.
    exclusion_source : str, optional
        Human-readable provenance for the feature-exclusion rule.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    # For seurat_v3, we need raw integer counts. Falling back to a
    # log-normalized adata.X silently produces wrong HVGs, so refuse
    # rather than warn-and-continue.
    if flavor == 'seurat_v3' and layer is None:
        for name in ['raw_counts', 'raw_data', 'counts']:
            if name in adata.layers:
                layer = name
                break
        if layer is None:
            raise ValueError(
                "select_hvg(flavor='seurat_v3') requires raw integer counts "
                "in a layer (looked for 'raw_counts', 'raw_data', 'counts'). "
                "adata.X is likely log-normalized, which would silently "
                "produce wrong HVGs. Either pass an explicit `layer=`, copy "
                "raw counts into adata.layers['raw_counts'], or use a "
                "log-friendly flavor like 'seurat' / 'cell_ranger'."
            )

    patterns = _normalize_patterns(exclude_patterns)
    exclude_mask = _feature_exclusion_mask(
        adata.var_names,
        patterns,
        match_mode=exclude_match_mode,
    )
    n_excluded = int(exclude_mask.sum())
    if exclusion_mode not in {"post", "pre"}:
        raise ValueError("exclusion_mode must be either 'post' or 'pre'.")
    if exclude_match_mode not in {"match", "contains", "fullmatch"}:
        raise ValueError("exclude_match_mode must be 'match', 'contains', or 'fullmatch'.")

    batch_msg = f" (batch-stratified by '{batch_key}')" if batch_key else ""
    exclusion_msg = f"; excluding {n_excluded} source-defined features ({exclusion_mode})" if n_excluded else ""
    logger.info(f"Selecting {n_top_genes} HVGs with {flavor} flavor{batch_msg}{exclusion_msg}")

    hvg_kwargs = dict(n_top_genes=n_top_genes, flavor=flavor, layer=layer)
    if batch_key:
        hvg_kwargs["batch_key"] = batch_key

    excluded_hvg_before = 0
    if n_excluded and exclusion_mode == "pre":
        allowed_mask = ~exclude_mask
        work = adata[:, allowed_mask].copy()
        sc.pp.highly_variable_genes(work, **hvg_kwargs)

        hvg_columns = [
            "highly_variable",
            "means",
            "dispersions",
            "dispersions_norm",
            "variances",
            "variances_norm",
            "highly_variable_rank",
            "highly_variable_nbatches",
            "highly_variable_intersection",
        ]
        for column in hvg_columns:
            if column not in work.var.columns:
                continue
            if work.var[column].dtype == bool:
                adata.var[column] = False
            else:
                adata.var[column] = np.nan
            adata.var.loc[work.var_names, column] = work.var[column].values
        if "highly_variable" not in adata.var.columns:
            adata.var["highly_variable"] = False
        adata.var.loc[exclude_mask, "highly_variable"] = False
        adata.uns["hvg"] = dict(work.uns.get("hvg", {}))
    else:
        sc.pp.highly_variable_genes(adata, **hvg_kwargs)
        if n_excluded and "highly_variable" in adata.var.columns:
            excluded_hvg_before = int(adata.var.loc[exclude_mask, "highly_variable"].sum())
            adata.var.loc[exclude_mask, "highly_variable"] = False

    adata.var["hvg_excluded"] = exclude_mask
    excluded_hvg_after = (
        int(adata.var.loc[exclude_mask, "highly_variable"].sum())
        if n_excluded and "highly_variable" in adata.var.columns
        else 0
    )
    n_hvg = adata.var['highly_variable'].sum()
    logger.info(f"Selected {n_hvg} highly variable genes")

    hvg_meta = dict(adata.uns.get("hvg", {}))
    hvg_meta.update({
        "flavor": flavor,
        "n_top_genes": int(n_top_genes),
        "batch_key": batch_key,
        "layer": layer,
        "n_hvg_selected": int(n_hvg),
        "feature_exclusions": {
            "applied": bool(n_excluded),
            "patterns": patterns,
            "match_mode": exclude_match_mode,
            "mode": exclusion_mode,
            "source": exclusion_source,
            "n_excluded": n_excluded,
            "excluded_hvg_before_forcing": excluded_hvg_before,
            "excluded_hvg_after_forcing": excluded_hvg_after,
        },
    })
    adata.uns["hvg"] = hvg_meta

    if not inplace:
        return adata
