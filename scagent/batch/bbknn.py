"""
BBKNN (Batch-Balanced K-Nearest Neighbors) batch correction for scagent.

BBKNN constructs a batch-balanced k-nearest-neighbor graph in PCA space.
Unlike Harmony, Scanorama, or scVI, there is no corrected embedding key —
the batch correction is embedded in the neighbor graph itself, which is used
directly for UMAP and Leiden clustering.

Advantages:
- Fast (comparable to Harmony), no iterative training required
- Produces batch-balanced neighborhoods for both UMAP and clustering
- Works on PCA; does not require raw counts

Limitation:
- Does not produce a corrected gene expression matrix or a named obsm embedding;
  the correction lives in adata.obsp['connectivities'] / adata.uns['neighbors']
- Cannot be evaluated with scib embedding-based metrics the same way as Harmony/scVI
"""

from typing import Optional
import logging

from anndata import AnnData

from ..config.defaults import BATCH_DEFAULTS

logger = logging.getLogger(__name__)


def run_bbknn(
    adata: AnnData,
    batch_key: str,
    n_pcs: int = BATCH_DEFAULTS.bbknn_n_pcs,
    neighbors_within_batch: int = BATCH_DEFAULTS.bbknn_neighbors_within_batch,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run BBKNN batch correction.

    BBKNN builds a batch-balanced k-nearest-neighbor graph in PCA space,
    replacing adata.uns['neighbors'] and adata.obsp['connectivities'] with a
    batch-corrected graph. UMAP and Leiden clustering should be run directly on
    this graph — do NOT recompute sc.pp.neighbors() afterward, as that would
    overwrite the BBKNN graph.

    Total neighbors per cell = n_batches × neighbors_within_batch.
    With many batches (e.g. 19 patients), neighbors_within_batch=3 gives
    19 × 3 = 57 neighbors, which is a well-balanced graph.

    Parameters
    ----------
    adata : AnnData
        AnnData object with PCA computed in adata.obsm['X_pca'].
    batch_key : str
        Column in adata.obs containing batch labels.
    n_pcs : int, default 30
        Number of PCA components to use.
    neighbors_within_batch : int, default 3
        Neighbors contributed by each batch per cell. Total neighbors =
        n_batches × neighbors_within_batch.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.

    Raises
    ------
    ImportError
        If bbknn is not installed.
    ValueError
        If batch_key or PCA is missing.
    """
    try:
        import bbknn as bbknn_lib
    except ImportError:
        raise ImportError("bbknn is not installed. Install with: pip install bbknn")

    if not inplace:
        adata = adata.copy()

    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

    if "X_pca" not in adata.obsm:
        raise ValueError("PCA not found in adata.obsm['X_pca']. Run PCA first.")

    n_batches = adata.obs[batch_key].nunique()
    total_neighbors = n_batches * neighbors_within_batch
    logger.info(
        f"Running BBKNN: {n_batches} batches, n_pcs={n_pcs}, "
        f"neighbors_within_batch={neighbors_within_batch} "
        f"(total neighbors per cell: {total_neighbors})"
    )

    bbknn_lib.bbknn(
        adata,
        batch_key=batch_key,
        n_pcs=n_pcs,
        neighbors_within_batch=neighbors_within_batch,
    )

    # Tag so world_state and downstream tools can detect BBKNN was applied.
    # BBKNN does not produce an obsm key — the correction lives in the neighbor graph.
    adata.uns["bbknn_batch_key"] = batch_key

    logger.info(
        f"BBKNN complete. Batch-corrected graph stored in adata.obsp['connectivities']. "
        f"Run sc.tl.umap(adata) next — do NOT recompute sc.pp.neighbors() first."
    )

    if not inplace:
        return adata
