"""
Normalization methods for scagent.

Implements:
- Raw counts preservation
- Library size normalization
- Log transformation
"""

from typing import Optional, Union
import numpy as np
import scanpy as sc
from anndata import AnnData
import logging

logger = logging.getLogger(__name__)


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
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Normalize and optionally log-transform data.

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
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    # Preserve raw counts
    if preserve_raw and raw_layer_name not in adata.layers:
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

    if not inplace:
        return adata


def select_hvg(
    adata: AnnData,
    n_top_genes: int = 4000,
    flavor: str = 'seurat_v3',
    layer: Optional[str] = None,
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
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    # For seurat_v3, we need raw counts
    if flavor == 'seurat_v3' and layer is None:
        # Try to find raw counts layer
        for name in ['raw_counts', 'raw_data', 'counts']:
            if name in adata.layers:
                layer = name
                break
        if layer is None:
            logger.warning(
                "No raw counts layer found for seurat_v3 flavor. "
                "Results may be suboptimal."
            )

    logger.info(f"Selecting {n_top_genes} HVGs with {flavor} flavor")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=flavor,
        layer=layer,
    )

    n_hvg = adata.var['highly_variable'].sum()
    logger.info(f"Selected {n_hvg} highly variable genes")

    if not inplace:
        return adata
