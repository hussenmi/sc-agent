"""
Batch mixing entropy for integration quality scoring.

Implements neighborhood-based Shannon entropy to quantify how well batch
labels are mixed in a low-dimensional embedding. High entropy = well-mixed
batches = effective integration.

Workshop reference: Session 5 (batch correction methods and evaluation).
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_batch_entropy(
    adata,
    batch_key: str,
    use_rep: str = "X_umap",
    n_neighbors: int = 50,
) -> Dict[str, Any]:
    """
    Compute per-cell batch mixing entropy and return a summary.

    For every cell, find its k nearest neighbors in ``use_rep`` and compute
    the Shannon entropy of the batch label distribution among those neighbors.
    Entropy is normalized by log2(n_batches) so the score lies in [0, 1]:

    - 1.0  → perfectly mixed (ideal integration)
    - 0.0  → all neighbors from the same batch (no mixing)

    Parameters
    ----------
    adata : AnnData
    batch_key : str
        obs column with batch labels.
    use_rep : str, default 'X_umap'
        Embedding to use for neighbor search. Use the *corrected* embedding
        (e.g. 'X_pca_harmony', 'X_scVI') to score post-integration quality,
        or 'X_pca' to get a pre-integration baseline.
    n_neighbors : int, default 50
        Neighborhood size. Larger values give stabler estimates but are slower.

    Returns
    -------
    dict
        entropy_mean, entropy_median, entropy_per_batch, per_cell_entropy,
        n_batches, use_rep, n_neighbors.
    """
    from sklearn.neighbors import NearestNeighbors

    if use_rep not in adata.obsm:
        raise ValueError(
            f"Representation '{use_rep}' not found in adata.obsm. "
            f"Available: {list(adata.obsm.keys())}"
        )
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs.")

    X = np.array(adata.obsm[use_rep])
    batch_labels = adata.obs[batch_key].astype(str).values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    if n_batches < 2:
        raise ValueError(f"Need at least 2 batches to compute entropy; found {n_batches}.")

    # Map batch labels to integers
    batch_map = {b: i for i, b in enumerate(unique_batches)}
    batch_int = np.array([batch_map[b] for b in batch_labels])

    logger.info(
        "Computing batch mixing entropy in '%s' (%d cells, %d batches, k=%d)...",
        use_rep, adata.n_obs, n_batches, n_neighbors,
    )

    # kNN search
    k = min(n_neighbors, adata.n_obs - 1)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(X)
    _, indices = nbrs.kneighbors(X)

    # Per-cell Shannon entropy
    max_entropy = np.log2(n_batches)
    per_cell_entropy = np.empty(adata.n_obs, dtype=float)
    for i in range(adata.n_obs):
        neighbor_batches = batch_int[indices[i]]
        counts = np.bincount(neighbor_batches, minlength=n_batches).astype(float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        raw_entropy = -np.sum(probs * np.log2(probs))
        per_cell_entropy[i] = raw_entropy / max_entropy  # normalize to [0, 1]

    # Per-batch breakdown
    entropy_per_batch = {}
    for batch_name in unique_batches:
        mask = batch_labels == batch_name
        entropy_per_batch[batch_name] = float(np.mean(per_cell_entropy[mask]))

    return {
        "use_rep": use_rep,
        "n_neighbors": k,
        "n_batches": n_batches,
        "entropy_mean": float(np.mean(per_cell_entropy)),
        "entropy_median": float(np.median(per_cell_entropy)),
        "entropy_per_batch": entropy_per_batch,
        "per_cell_entropy": per_cell_entropy,  # ndarray — caller decides what to store
    }
