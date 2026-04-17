"""
Dimensionality reduction methods for scagent.

Implements:
- PCA
- Neighbor graph computation
- UMAP
- Force-directed layout
"""

from typing import Optional
import scanpy as sc
from anndata import AnnData
import logging

from ..config.defaults import DIMRED_DEFAULTS

logger = logging.getLogger(__name__)


def run_pca(
    adata: AnnData,
    n_comps: int = DIMRED_DEFAULTS.n_pcs,
    mask_var: Optional[str] = "highly_variable",
    random_state: int = 0,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run PCA on the data.

    Parameters
    ----------
    adata : AnnData
        AnnData object (should be normalized and log-transformed).
    n_comps : int, default 30
        Number of principal components to compute.
    mask_var : str or None, default 'highly_variable'
        Boolean column in adata.var selecting genes for PCA. The default
        restricts to HVGs; pass None to use all genes. (Replaces the old
        deprecated `use_highly_variable` flag.)
    random_state : int, default 0
        Random seed for reproducibility.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    # Resolve mask: if the caller requested HVGs but the column is missing,
    # fall back to all genes with a warning rather than failing.
    if mask_var is not None and mask_var not in adata.var.columns:
        logger.warning(
            "mask_var=%r not found in adata.var; running PCA on all genes",
            mask_var,
        )
        mask_var = None

    logger.info(f"Running PCA with {n_comps} components")

    sc.tl.pca(
        adata,
        n_comps=n_comps,
        mask_var=mask_var,
        random_state=random_state,
    )

    logger.info(f"PCA complete. Variance explained: {adata.uns['pca']['variance_ratio'].sum():.2%}")

    if not inplace:
        return adata


def compute_neighbors(
    adata: AnnData,
    n_neighbors: int = DIMRED_DEFAULTS.n_neighbors,
    n_pcs: Optional[int] = None,
    use_rep: str = 'X_pca',
    metric: str = DIMRED_DEFAULTS.metric,
    key_added: Optional[str] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute k-nearest neighbor graph.

    Parameters
    ----------
    adata : AnnData
        AnnData object with PCA or other representation.
    n_neighbors : int, default 30
        Number of neighbors for the graph.
    n_pcs : int, optional
        Number of PCs to use. If None, uses all available.
    use_rep : str, default 'X_pca'
        Representation to use for computing neighbors.
    metric : str, default 'euclidean'
        Distance metric.
    key_added : str, optional
        Key to add to adata.uns for multiple neighbor graphs.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    if use_rep not in adata.obsm:
        raise ValueError(f"Representation '{use_rep}' not found. Run PCA first.")

    logger.info(f"Computing neighbors with k={n_neighbors} using {use_rep}")

    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        metric=metric,
        key_added=key_added,
    )

    logger.info("Neighbor graph computed")

    if not inplace:
        return adata


def compute_umap(
    adata: AnnData,
    min_dist: float = DIMRED_DEFAULTS.umap_min_dist,
    spread: float = 1.0,
    n_components: int = 2,
    neighbors_key: Optional[str] = None,
    random_state: int = 0,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute UMAP embedding.

    Parameters
    ----------
    adata : AnnData
        AnnData object with neighbor graph computed.
    min_dist : float, default 0.1
        Minimum distance between points in UMAP.
    spread : float, default 1.0
        Spread of UMAP embedding.
    n_components : int, default 2
        Number of UMAP dimensions.
    neighbors_key : str, optional
        Key in adata.uns for neighbor graph (for multiple neighbor graphs).
    random_state : int, default 0
        Random seed for reproducibility.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    # Check for neighbor graph
    if 'neighbors' not in adata.uns and neighbors_key is None:
        raise ValueError("Neighbor graph not found. Run compute_neighbors first.")

    logger.info(f"Computing UMAP with min_dist={min_dist}")

    sc.tl.umap(
        adata,
        min_dist=min_dist,
        spread=spread,
        n_components=n_components,
        neighbors_key=neighbors_key,
        random_state=random_state,
    )

    logger.info("UMAP computed")

    if not inplace:
        return adata


def compute_force_directed_layout(
    adata: AnnData,
    layout: str = 'fa',
    random_state: int = 0,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Compute force-directed graph layout.

    Parameters
    ----------
    adata : AnnData
        AnnData object with neighbor graph computed.
    layout : str, default 'fa'
        Layout algorithm: 'fa' (ForceAtlas2) or 'fr' (Fruchterman-Reingold).
    random_state : int, default 0
        Random seed for reproducibility.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    logger.info(f"Computing force-directed layout ({layout})")

    sc.tl.draw_graph(adata, layout=layout, random_state=random_state)

    logger.info("Force-directed layout computed")

    if not inplace:
        return adata


def run_dimensionality_reduction(
    adata: AnnData,
    n_pcs: int = DIMRED_DEFAULTS.n_pcs,
    n_neighbors: int = DIMRED_DEFAULTS.n_neighbors,
    umap_min_dist: float = DIMRED_DEFAULTS.umap_min_dist,
    compute_fdl: bool = False,
    random_state: int = 0,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run complete dimensionality reduction pipeline: PCA -> neighbors -> UMAP.

    Parameters
    ----------
    adata : AnnData
        AnnData object (normalized, log-transformed, with HVGs).
    n_pcs : int, default 30
        Number of principal components.
    n_neighbors : int, default 30
        Number of neighbors for the graph.
    umap_min_dist : float, default 0.1
        Minimum distance for UMAP.
    compute_fdl : bool, default False
        Also compute force-directed layout.
    random_state : int, default 0
        Random seed for reproducibility.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    logger.info("Running dimensionality reduction pipeline...")

    # PCA
    run_pca(adata, n_comps=n_pcs, random_state=random_state, inplace=True)

    # Neighbors
    compute_neighbors(adata, n_neighbors=n_neighbors, inplace=True)

    # UMAP
    compute_umap(adata, min_dist=umap_min_dist, random_state=random_state, inplace=True)

    # Optional FDL
    if compute_fdl:
        compute_force_directed_layout(adata, random_state=random_state, inplace=True)

    logger.info("Dimensionality reduction pipeline complete")

    if not inplace:
        return adata
