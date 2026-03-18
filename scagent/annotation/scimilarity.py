"""
Scimilarity cell type annotation for scagent.

Scimilarity uses pretrained embeddings to annotate cell types.
Requires gene alignment to the model's gene space.
"""

from typing import Optional, List
import scanpy as sc
from anndata import AnnData
import numpy as np
import logging

from ..config.defaults import SCIMILARITY_DEFAULTS

logger = logging.getLogger(__name__)

# Default model path on the HPC system (IRIS)
DEFAULT_MODEL_PATH = "/data1/peerd/ibrahih3/scimilarity/docs/notebooks/models/model_v1.1"

import os
# Allow override via environment variable
DEFAULT_MODEL_PATH = os.environ.get("SCIMILARITY_MODEL_PATH", DEFAULT_MODEL_PATH)


def prepare_for_scimilarity(
    adata: AnnData,
    model_path: str = DEFAULT_MODEL_PATH,
    raw_layer: Optional[str] = None,
) -> AnnData:
    """
    Prepare data for Scimilarity annotation.

    This function:
    1. Creates a new AnnData with raw counts
    2. Aligns genes to the Scimilarity gene space
    3. Normalizes to target_sum=10000 and log-transforms

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    model_path : str
        Path to Scimilarity model directory.
    raw_layer : str, optional
        Layer containing raw counts.

    Returns
    -------
    AnnData
        New AnnData object prepared for Scimilarity.
    """
    try:
        from scimilarity import CellAnnotation
        from scimilarity.utils import lognorm_counts, align_dataset
    except ImportError:
        raise ImportError("scimilarity not installed. Install with: pip install scimilarity")

    logger.info("Preparing data for Scimilarity annotation...")

    # Load model to get gene order
    ca = CellAnnotation(model_path=model_path)

    # Find raw counts
    if raw_layer is None:
        for layer in ['raw_counts', 'raw_data', 'counts']:
            if layer in adata.layers:
                raw_layer = layer
                break

    if raw_layer is not None:
        logger.info(f"Using raw counts from layer '{raw_layer}'")
        X = adata.layers[raw_layer]
    else:
        logger.warning("No raw counts layer found, using adata.X")
        X = adata.X

    # Create minimal AnnData for Scimilarity
    adata_sci = AnnData(
        X,
        obs=adata.obs[['leiden']] if 'leiden' in adata.obs.columns else adata.obs[[]],
        var=adata.var[['gene_ids']] if 'gene_ids' in adata.var.columns else adata.var[[]],
    )

    # Copy UMAP if available
    if 'X_umap' in adata.obsm:
        adata_sci.obsm['X_umap'] = adata.obsm['X_umap'].copy()

    # Align to Scimilarity gene space
    adata_sci = align_dataset(adata_sci, ca.gene_order)
    logger.info(f"Aligned to Scimilarity gene space: {adata_sci.n_vars} genes")

    # Normalize (Scimilarity expects counts in layers['counts'])
    adata_sci.layers['counts'] = adata_sci.X.copy()
    adata_sci = lognorm_counts(adata_sci)  # Normalizes to 10k target sum

    return adata_sci, ca


def run_scimilarity(
    adata: AnnData,
    model_path: str = DEFAULT_MODEL_PATH,
    target_celltypes: Optional[List[str]] = None,
    raw_layer: Optional[str] = None,
    cluster_key: str = 'leiden',
    transfer_results: bool = True,
    inplace: bool = True,
) -> Optional[AnnData]:
    """
    Run Scimilarity cell type annotation.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    model_path : str
        Path to Scimilarity model directory.
    target_celltypes : List[str], optional
        List of expected cell types to safelist.
    raw_layer : str, optional
        Layer containing raw counts.
    cluster_key : str, default 'leiden'
        Column in adata.obs containing cluster assignments.
    transfer_results : bool, default True
        Transfer results back to original adata.
    inplace : bool, default True
        Modify adata in place.

    Returns
    -------
    AnnData or None
        Returns AnnData if inplace=False, None otherwise.
    """
    try:
        from scimilarity import CellAnnotation
    except ImportError:
        raise ImportError("scimilarity not installed. Install with: pip install scimilarity")

    if not inplace:
        adata = adata.copy()

    logger.info(f"Running Scimilarity annotation with model at {model_path}")

    # Prepare data
    adata_sci, ca = prepare_for_scimilarity(adata, model_path=model_path, raw_layer=raw_layer)

    # Get embeddings
    adata_sci.obsm["X_scimilarity"] = ca.get_embeddings(adata_sci.X)
    logger.info(f"Computed Scimilarity embeddings: {adata_sci.obsm['X_scimilarity'].shape}")

    # Get unconstrained predictions
    predictions, nn_idxs, nn_dists, nn_stats = ca.get_predictions_knn(
        adata_sci.obsm["X_scimilarity"]
    )
    adata_sci.obs["predictions_unconstrained"] = predictions.values

    # Get cluster-level representative predictions
    if cluster_key in adata.obs.columns:
        adata_sci.obs[cluster_key] = adata.obs[cluster_key].values

        # Count celltypes per cluster
        df_cluster = adata_sci.obs[[cluster_key, 'predictions_unconstrained']]
        celltype_counts = df_cluster.groupby(cluster_key)["predictions_unconstrained"].value_counts().unstack(fill_value=0)
        representatives = celltype_counts.idxmax(axis=1)
        adata_sci.obs['representative_prediction'] = adata_sci.obs[cluster_key].map(representatives.to_dict())

    # Optionally use safelisted cell types
    if target_celltypes is not None:
        logger.info(f"Applying safelist with {len(target_celltypes)} cell types")
        ca.safelist_celltypes(target_celltypes)
        adata_sci = ca.annotate_dataset(adata_sci)

    # Transfer results back to original adata
    if transfer_results:
        cols_to_transfer = ['predictions_unconstrained', 'representative_prediction']

        for col in cols_to_transfer:
            if col in adata_sci.obs.columns:
                adata.obs[f'scimilarity_{col}'] = adata_sci.obs[col].loc[adata.obs_names].values

        # Also transfer embeddings
        adata.obsm['X_scimilarity'] = adata_sci.obsm['X_scimilarity']

        logger.info("Scimilarity results transferred to adata")

    # Log summary
    n_celltypes = adata_sci.obs['predictions_unconstrained'].nunique()
    logger.info(f"Scimilarity complete: {n_celltypes} cell types predicted")

    if not inplace:
        return adata


# Common cell types for PBMC data
PBMC_CELLTYPES = [
    "CD4-positive, alpha-beta T cell",
    "CD8-positive, alpha-beta T cell",
    "regulatory T cell",
    "B cell",
    "plasma cell",
    "natural killer cell",
    "classical monocyte",
    "non-classical monocyte",
    "conventional dendritic cell",
    "plasmacytoid dendritic cell",
]
