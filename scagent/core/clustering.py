"""
Clustering methods for scagent.

Implements:
- Leiden clustering
- PhenoGraph clustering (Leiden on Jaccard graph)
"""

from typing import Optional
import numpy as np
import scipy.sparse as sp
import scanpy as sc
from anndata import AnnData
import logging

from ..config.defaults import CLUSTERING_DEFAULTS

logger = logging.getLogger(__name__)


def run_leiden(
    adata: AnnData,
    resolution: float = CLUSTERING_DEFAULTS.leiden_resolution,
    random_state: int = CLUSTERING_DEFAULTS.leiden_random_state,
    key_added: str = 'leiden',
    neighbors_key: Optional[str] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run Leiden clustering.

    Parameters
    ----------
    adata : AnnData
        AnnData object with neighbor graph computed.
    resolution : float, default 1.0
        Resolution parameter for Leiden clustering.
    random_state : int, default 0
        Random seed for reproducibility.
    key_added : str, default 'leiden'
        Key to add to adata.obs for cluster assignments.
    neighbors_key : str, optional
        Key in adata.uns for neighbor graph.
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

    logger.info(f"Running Leiden clustering with resolution={resolution}")

    sc.tl.leiden(
        adata,
        resolution=resolution,
        random_state=random_state,
        key_added=key_added,
        neighbors_key=neighbors_key,
    )

    n_clusters = adata.obs[key_added].nunique()
    logger.info(f"Leiden clustering complete: {n_clusters} clusters")

    if not inplace:
        return adata


def run_phenograph(
    adata: AnnData,
    k: int = CLUSTERING_DEFAULTS.phenograph_k,
    clustering_algo: str = CLUSTERING_DEFAULTS.phenograph_clustering_algo,
    resolution: float = CLUSTERING_DEFAULTS.leiden_resolution,
    primary_metric: str = CLUSTERING_DEFAULTS.phenograph_metric,
    use_rep: str = 'X_pca',
    key_added: str = 'pheno_leiden',
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run PhenoGraph clustering (Leiden on Jaccard-weighted KNN graph).

    PhenoGraph builds a Jaccard-weighted graph from KNN distances,
    which is more robust to noise than standard KNN graphs.

    IMPORTANT: After PhenoGraph, the Jaccard graph is stored in COO format.
    This function converts it to CSR format for Scanpy compatibility.

    Parameters
    ----------
    adata : AnnData
        AnnData object with PCA computed.
    k : int, default 30
        Number of nearest neighbors.
    clustering_algo : str, default 'leiden'
        Clustering algorithm: 'leiden' or 'louvain'.
    resolution : float, default 1.0
        Resolution parameter for clustering.
    primary_metric : str, default 'euclidean'
        Distance metric for KNN.
    use_rep : str, default 'X_pca'
        Representation to use for KNN.
    key_added : str, default 'pheno_leiden'
        Key to add to adata.obs for cluster assignments.
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

    logger.info(f"Running PhenoGraph clustering with k={k}, resolution={resolution}")

    sc.external.tl.phenograph(
        adata,
        clustering_algo=clustering_algo,
        k=k,
        jaccard=True,
        primary_metric=primary_metric,
        resolution_parameter=resolution,
    )

    # CRITICAL: Convert Jaccard graph from COO to CSR format
    # PhenoGraph stores the graph in COO format, but Scanpy expects CSR
    if 'pheno_jaccard_ig' in adata.obsp:
        if not sp.isspmatrix_csr(adata.obsp['pheno_jaccard_ig']):
            adata.obsp['pheno_jaccard_ig'] = sp.csr_matrix(adata.obsp['pheno_jaccard_ig'])
            logger.info("Converted pheno_jaccard_ig to CSR format")

    n_clusters = adata.obs[key_added].nunique()
    logger.info(f"PhenoGraph clustering complete: {n_clusters} clusters")

    if not inplace:
        return adata


def run_louvain(
    adata: AnnData,
    resolution: float = CLUSTERING_DEFAULTS.leiden_resolution,
    random_state: int = CLUSTERING_DEFAULTS.leiden_random_state,
    key_added: str = 'louvain',
    neighbors_key: Optional[str] = None,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run Louvain clustering.

    Note: Leiden is generally preferred over Louvain for better performance.

    Parameters
    ----------
    adata : AnnData
        AnnData object with neighbor graph computed.
    resolution : float, default 1.0
        Resolution parameter for Louvain clustering.
    random_state : int, default 0
        Random seed for reproducibility.
    key_added : str, default 'louvain'
        Key to add to adata.obs for cluster assignments.
    neighbors_key : str, optional
        Key in adata.uns for neighbor graph.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    logger.info(f"Running Louvain clustering with resolution={resolution}")

    sc.tl.louvain(
        adata,
        resolution=resolution,
        random_state=random_state,
        key_added=key_added,
        neighbors_key=neighbors_key,
    )

    n_clusters = adata.obs[key_added].nunique()
    logger.info(f"Louvain clustering complete: {n_clusters} clusters")

    if not inplace:
        return adata


def run_differential_expression(
    adata: AnnData,
    groupby: str = 'leiden',
    method: str = 'wilcoxon',
    n_genes: int = 100,
    key_added: str = 'rank_genes_groups',
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run differential expression analysis between clusters.

    Parameters
    ----------
    adata : AnnData
        AnnData object with clusters.
    groupby : str, default 'leiden'
        Column in adata.obs containing cluster assignments.
    method : str, default 'wilcoxon'
        Statistical method: 'wilcoxon', 't-test', 'logreg'.
    n_genes : int, default 100
        Number of top genes to store.
    key_added : str, default 'rank_genes_groups'
        Key to add to adata.uns.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    if not inplace:
        adata = adata.copy()

    if groupby not in adata.obs.columns:
        raise ValueError(f"Groupby column '{groupby}' not found. Run clustering first.")

    logger.info(f"Running differential expression analysis by {groupby}")

    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        n_genes=n_genes,
        key_added=key_added,
    )

    logger.info("Differential expression analysis complete")

    if not inplace:
        return adata


def get_top_markers(
    adata: AnnData,
    group: str,
    n_genes: int = 20,
    key: str = 'rank_genes_groups',
) -> 'pd.DataFrame':
    """
    Get top marker genes for a specific cluster.

    Parameters
    ----------
    adata : AnnData
        AnnData object with DE results.
    group : str
        Cluster to get markers for.
    n_genes : int, default 20
        Number of top genes to return.
    key : str, default 'rank_genes_groups'
        Key in adata.uns containing DE results.

    Returns
    -------
    pd.DataFrame
        DataFrame with marker genes and statistics.
    """
    if key not in adata.uns:
        raise ValueError(f"DE results not found at key '{key}'. Run differential expression first.")

    return sc.get.rank_genes_groups_df(adata, group=group).head(n_genes)
