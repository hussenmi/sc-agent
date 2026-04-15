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


def query_cells(
    adata: AnnData,
    query_type: str = "cells",
    cell_ids: Optional[List[str]] = None,
    obs_column: Optional[str] = None,
    group_key: Optional[str] = None,
    group_value: Optional[str] = None,
    k: int = 50,
    model_path: str = DEFAULT_MODEL_PATH,
    raw_layer: Optional[str] = None,
) -> dict:
    """
    Query the Scimilarity reference database for cells similar to a query.

    Two modes:
    - "cells": query using specific cells identified by obs_names or a boolean obs column.
      Uses the mean embedding of the selected cells as the query vector.
    - "centroid": query using the centroid of a cluster or cell type group.
      Internally calls search_centroid_nearest on the aligned+prepared adata.

    Prerequisites: Scimilarity must be installed and the model must include the
    cellsearch kNN index (cellsearch/ subdirectory with full_kNN.bin).

    Parameters
    ----------
    adata : AnnData
        AnnData object. If X_scimilarity is already in obsm, embeddings are reused.
    query_type : str
        "cells" (default) or "centroid".
    cell_ids : list of str, optional
        obs_names of cells to use as query (mode "cells").
    obs_column : str, optional
        obs column where True/1 marks query cells (mode "cells").
        Used when cell_ids is not provided.
    group_key : str, optional
        obs column containing group labels (mode "centroid").
    group_value : str, optional
        Value in group_key to use as the centroid (mode "centroid").
    k : int, default 50
        Number of nearest neighbours to retrieve.
    model_path : str
        Path to Scimilarity model directory.
    raw_layer : str, optional
        Layer with raw counts (used only for centroid mode, which requires counts).

    Returns
    -------
    dict with keys:
        query_type, n_query_cells, k, results_metadata (DataFrame as dict),
        top_celltypes (value_counts), top_tissues, top_diseases,
        coherence (centroid mode only), mean_dist.
    """
    try:
        from scimilarity import CellQuery
    except ImportError:
        raise ImportError("scimilarity not installed. Install with: pip install scimilarity")

    import os
    knn_path = os.path.join(model_path, "cellsearch", "full_kNN.bin")
    if not os.path.exists(knn_path):
        raise FileNotFoundError(
            f"CellQuery kNN index not found at {knn_path}. "
            "The model directory needs a cellsearch/ subdirectory with full_kNN.bin. "
            "Contact your system admin or set SCIMILARITY_MODEL_PATH."
        )

    cq = CellQuery(model_path=model_path)

    if query_type == "centroid":
        if not group_key or not group_value:
            raise ValueError("centroid mode requires group_key and group_value.")
        if group_key not in adata.obs.columns:
            raise ValueError(f"group_key '{group_key}' not found in adata.obs.")
        if group_value not in adata.obs[group_key].values:
            raise ValueError(
                f"group_value '{group_value}' not found in adata.obs['{group_key}']. "
                f"Available: {sorted(adata.obs[group_key].unique().tolist())}"
            )

        # Prepare the aligned adata — centroid search needs layers["counts"]
        adata_sci, _ = prepare_for_scimilarity(adata, model_path=model_path, raw_layer=raw_layer)

        # Mark query cells in adata_sci (1 = in group, 0 = not)
        centroid_col = "__query_group__"
        adata_sci.obs[centroid_col] = (
            adata.obs[group_key].values == group_value
        ).astype(int)

        n_query_cells = int(adata_sci.obs[centroid_col].sum())
        if n_query_cells == 0:
            raise ValueError(f"No cells matched group_key='{group_key}', group_value='{group_value}'.")

        centroid_emb, nn_idxs, nn_dists, metadata, qc_stats = cq.search_centroid_nearest(
            adata_sci,
            centroid_key=centroid_col,
            k=k,
            ef=max(k, 100),
        )
        coherence = qc_stats.get("query_coherence")
        # metadata is a single DataFrame (one query → one result set)
        results_df = metadata

    else:  # "cells" mode
        # Resolve which cells to query
        if cell_ids is not None:
            missing = [c for c in cell_ids if c not in adata.obs_names]
            if missing:
                raise ValueError(
                    f"{len(missing)} cell IDs not found in adata.obs_names: {missing[:5]}..."
                )
            mask = adata.obs_names.isin(cell_ids)
        elif obs_column is not None:
            if obs_column not in adata.obs.columns:
                raise ValueError(f"obs_column '{obs_column}' not found in adata.obs.")
            mask = adata.obs[obs_column].astype(bool)
        else:
            raise ValueError("cells mode requires cell_ids or obs_column.")

        n_query_cells = int(mask.sum())
        if n_query_cells == 0:
            raise ValueError("No cells selected. Check cell_ids or obs_column values.")

        # Get or compute embeddings for selected cells
        if "X_scimilarity" in adata.obsm:
            query_embedding = np.array(adata.obsm["X_scimilarity"])[mask]
        else:
            logger.info("X_scimilarity not found — computing embeddings for query cells")
            adata_sci, ca = prepare_for_scimilarity(adata, model_path=model_path, raw_layer=raw_layer)
            all_embeddings = ca.get_embeddings(adata_sci.X)
            query_embedding = all_embeddings[mask]

        # Use mean embedding across selected cells as the query vector
        mean_embedding = query_embedding.mean(axis=0, keepdims=True)

        nn_idxs, nn_dists, metadata = cq.search_nearest(
            mean_embedding,
            k=k,
            ef=max(k, 100),
        )
        results_df = metadata
        coherence = None

    # Summarise results
    mean_dist = float(results_df["query_nn_dist"].mean()) if "query_nn_dist" in results_df.columns else None

    def top_counts(col, n=10):
        if col in results_df.columns:
            return results_df[col].value_counts().head(n).to_dict()
        return {}

    return {
        "query_type": query_type,
        "n_query_cells": n_query_cells,
        "k": k,
        "n_results": len(results_df),
        "mean_dist": round(mean_dist, 4) if mean_dist is not None else None,
        "coherence": round(float(coherence), 2) if coherence is not None else None,
        "top_celltypes": top_counts("cell_type", 10),
        "top_tissues": top_counts("tissue", 10),
        "top_diseases": top_counts("disease", 10),
        "results_preview": results_df.head(10).to_dict(orient="records"),
    }


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
