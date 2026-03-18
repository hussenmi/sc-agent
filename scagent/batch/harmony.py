"""
Harmony batch correction for scagent.

Harmony iteratively adjusts PCA embeddings to remove batch effects
while preserving biological variation.
"""

from typing import Optional
import scanpy as sc
from anndata import AnnData
import logging

from ..config.defaults import BATCH_DEFAULTS

logger = logging.getLogger(__name__)


def run_harmony(
    adata: AnnData,
    batch_key: str,
    basis: str = BATCH_DEFAULTS.harmony_basis,
    adjusted_basis: str = BATCH_DEFAULTS.harmony_adjusted_basis,
    max_iter: int = 10,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run Harmony batch correction.

    Harmony corrects batch effects in PCA space by iteratively:
    1. Clustering cells
    2. Maximizing diversity of batches within clusters
    3. Computing correction factors

    Note: Harmony does NOT correct gene expression, only embeddings.

    Parameters
    ----------
    adata : AnnData
        AnnData object with PCA computed.
    batch_key : str
        Column in adata.obs containing batch information.
    basis : str, default 'X_pca'
        Key for PCA embedding to correct.
    adjusted_basis : str, default 'X_pca_harmony'
        Key for corrected embedding.
    max_iter : int, default 10
        Maximum number of Harmony iterations.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

    if basis not in adata.obsm:
        raise ValueError(f"Basis '{basis}' not found. Run PCA first.")

    logger.info(f"Running Harmony batch correction on {basis}")

    sc.external.pp.harmony_integrate(
        adata,
        key=batch_key,
        basis=basis,
        adjusted_basis=adjusted_basis,
        max_iter_harmony=max_iter,
    )

    logger.info(f"Harmony-corrected embedding stored in adata.obsm['{adjusted_basis}']")

    if not inplace:
        return adata


def compute_umap_from_harmony(
    adata: AnnData,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    harmony_key: str = 'X_pca_harmony',
    neighbors_key: str = 'neighbors_harmony',
    umap_key: str = 'X_umap_harmony',
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute UMAP from Harmony-corrected embedding.

    Parameters
    ----------
    adata : AnnData
        AnnData object with Harmony-corrected PCA.
    n_neighbors : int, default 30
        Number of neighbors for the graph.
    min_dist : float, default 0.1
        Minimum distance for UMAP.
    harmony_key : str, default 'X_pca_harmony'
        Key for Harmony-corrected PCA in adata.obsm.
    neighbors_key : str, default 'neighbors_harmony'
        Key to store neighbor graph.
    umap_key : str, default 'X_umap_harmony'
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

    if harmony_key not in adata.obsm:
        raise ValueError(f"Harmony embedding '{harmony_key}' not found. Run run_harmony first.")

    logger.info(f"Computing UMAP from Harmony embedding")

    # Compute neighbors on Harmony embedding
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=harmony_key,
        metric='euclidean',
        key_added=neighbors_key,
    )

    # Compute UMAP
    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        min_dist=min_dist,
        key_added=umap_key,
    )

    logger.info(f"UMAP stored in adata.obsm['{umap_key}']")

    if not inplace:
        return adata
