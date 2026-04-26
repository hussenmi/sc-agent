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

    # Split by batch — filter to HVGs if available (matches workshop approach;
    # running on all genes is slow and reduces alignment quality)
    batches = adata.obs[batch_key].unique()
    if "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"]
        adatas = [adata[adata.obs[batch_key] == batch, hvg_mask].copy() for batch in batches]
        logger.info(f"Running Scanorama on {hvg_mask.sum()} HVGs")
    else:
        adatas = [adata[adata.obs[batch_key] == batch].copy() for batch in batches]
        logger.info(f"No HVGs found — running Scanorama on all {adata.n_vars} genes")

    logger.info(f"Processing {len(batches)} batches: {list(batches)}")

    # Run Scanorama — correct_scanpy returns a list of adatas (one per batch),
    # each with obsm["X_scanorama"] when return_dimred=True
    corrected_adatas = scanorama.correct_scanpy(
        adatas,
        return_dimred=return_dimred,
        verbose=True,
        dimred=dimred,
        knn=knn,
    )

    # Reassemble embedding in original cell order
    if return_dimred:
        scanorama_emb = np.zeros((adata.n_obs, dimred))
        for i, batch in enumerate(batches):
            mask = adata.obs[batch_key] == batch
            scanorama_emb[mask] = corrected_adatas[i].obsm[key_added]
        adata.obsm[key_added] = scanorama_emb
        logger.info(f"Scanorama embedding stored in adata.obsm['{key_added}']")

    # Reassemble corrected expression in original cell order.
    # Scanorama only corrects the genes it was run on (HVGs when available).
    # The corrected matrix must be padded to adata.n_vars for layers storage;
    # non-HVG positions stay at zero.
    import scipy.sparse as _sp
    n_hvg = corrected_adatas[0].n_vars
    # Build a full-width matrix (n_cells × all_genes) padded with zeros
    corr_expr = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)
    # Find which gene columns to fill
    if "highly_variable" in adata.var.columns:
        hvg_col_indices = np.where(adata.var["highly_variable"].values)[0]
    else:
        # Ran on all genes — no padding needed, columns match directly
        hvg_col_indices = np.arange(adata.n_vars)
    if n_hvg == len(hvg_col_indices):
        for i, batch in enumerate(batches):
            mask = adata.obs[batch_key] == batch
            X_batch = corrected_adatas[i].X
            X_batch_dense = X_batch.toarray() if _sp.issparse(X_batch) else X_batch
            corr_expr[np.ix_(mask, hvg_col_indices)] = X_batch_dense
        adata.layers['X_scanorama_corr'] = corr_expr
        logger.info(
            "Scanorama-corrected expression stored in adata.layers['X_scanorama_corr'] "
            f"({n_hvg} HVG columns non-zero, {adata.n_vars - n_hvg} zero-padded)"
        )
    else:
        logger.warning(
            "Skipping X_scanorama_corr: corrected gene count (%d) doesn't match "
            "HVG count (%d) — embedding only.", n_hvg, len(hvg_col_indices)
        )

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

    # Compute UMAP directly into umap_key — sc.tl.umap without key_added always
    # writes to 'X_umap', so we must pass key_added to avoid overwriting any
    # existing uncorrected UMAP before we can copy it.
    sc.tl.umap(
        adata,
        neighbors_key=neighbors_key,
        min_dist=min_dist,
        key_added=umap_key,
    )
    logger.info(f"UMAP stored in adata.obsm['{umap_key}']")

    if not inplace:
        return adata
