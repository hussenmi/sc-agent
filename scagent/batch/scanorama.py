"""
Scanorama batch correction for scagent.

Scanorama uses MNN (mutual nearest neighbors) to align batches.
It outputs both corrected gene expression and low-dimensional embeddings.
"""

from typing import Optional, List
import numpy as np
import scanpy as sc
from anndata import AnnData
import logging

from ..config.defaults import BATCH_DEFAULTS

logger = logging.getLogger(__name__)


def run_scanorama(
    adata: AnnData,
    batch_key: str,
    dimred: int = BATCH_DEFAULTS.scanorama_dimred,
    knn: int = BATCH_DEFAULTS.scanorama_knn,
    return_dimred: bool = BATCH_DEFAULTS.scanorama_return_dimred,
    key_added: str = 'X_scanorama',
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run Scanorama batch correction.

    Scanorama corrects batch effects by:
    1. Finding MNN pairs between batches
    2. Using these pairs to learn batch correction vectors
    3. Applying corrections in both gene expression and embedding space

    Parameters
    ----------
    adata : AnnData
        AnnData object with batch information.
    batch_key : str
        Column in adata.obs containing batch information.
    dimred : int, default 30
        Number of dimensions for SVD/embedding.
    knn : int, default 30
        Number of nearest neighbors for MNN.
    return_dimred : bool, default True
        Return the corrected low-dimensional embedding.
    key_added : str, default 'X_scanorama'
        Key to store corrected embedding in adata.obsm.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    try:
        import scanorama
    except ImportError:
        raise ImportError("scanorama not installed. Install with: pip install scanorama")

    if not inplace:
        adata = adata.copy()

    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

    logger.info(f"Running Scanorama batch correction with dimred={dimred}, knn={knn}")

    # Split by batch
    batches = adata.obs[batch_key].unique()
    adatas = [adata[adata.obs[batch_key] == batch].copy() for batch in batches]

    logger.info(f"Processing {len(batches)} batches: {list(batches)}")

    # Run Scanorama
    corrected = scanorama.correct_scanpy(
        adatas,
        return_dimred=return_dimred,
        verbose=True,
        dimred=dimred,
        knn=knn,
    )

    # Concatenate corrected data
    import anndata
    adata_corrected = anndata.concat(corrected, axis=0)

    # Reorder to match original
    adata_corrected = adata_corrected[adata.obs_names]

    # Transfer results back to original adata
    if return_dimred and key_added in adata_corrected.obsm:
        adata.obsm[key_added] = adata_corrected.obsm[key_added]
        logger.info(f"Scanorama embedding stored in adata.obsm['{key_added}']")

    # Store corrected expression
    adata.layers['X_scanorama_corr'] = adata_corrected.X.copy()
    logger.info("Scanorama-corrected expression stored in adata.layers['X_scanorama_corr']")

    logger.info("Scanorama batch correction complete")

    if not inplace:
        return adata


def compute_umap_from_scanorama(
    adata: AnnData,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    scanorama_key: str = 'X_scanorama',
    neighbors_key: str = 'neighbors_scanorama',
    umap_key: str = 'X_umap_scanorama',
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute UMAP from Scanorama-corrected embedding.

    Parameters
    ----------
    adata : AnnData
        AnnData object with Scanorama embedding.
    n_neighbors : int, default 30
        Number of neighbors for the graph.
    min_dist : float, default 0.1
        Minimum distance for UMAP.
    scanorama_key : str, default 'X_scanorama'
        Key for Scanorama embedding in adata.obsm.
    neighbors_key : str, default 'neighbors_scanorama'
        Key to store neighbor graph.
    umap_key : str, default 'X_umap_scanorama'
        Key to store UMAP embedding.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    if scanorama_key not in adata.obsm:
        raise ValueError(f"Scanorama embedding '{scanorama_key}' not found. Run run_scanorama first.")

    logger.info(f"Computing UMAP from Scanorama embedding")

    # Compute neighbors on Scanorama embedding
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=scanorama_key,
        metric='euclidean',
        key_added=neighbors_key,
    )

    # Compute UMAP
    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        min_dist=min_dist,
    )

    # Rename UMAP to avoid overwriting
    if 'X_umap' in adata.obsm:
        adata.obsm[umap_key] = adata.obsm['X_umap'].copy()
        logger.info(f"UMAP stored in adata.obsm['{umap_key}']")

    if not inplace:
        return adata
