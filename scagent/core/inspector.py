"""
Data state inspector for scagent.

This is the MOST CRITICAL module - it detects the current state of the data
and recommends what analysis steps are needed to reach a user's goal.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from anndata import AnnData
import scipy.sparse as sp


@dataclass
class DataState:
    """Comprehensive representation of an AnnData object's processing state."""

    # Basic info
    shape: Tuple[int, int] = (0, 0)
    n_cells: int = 0
    n_genes: int = 0

    # Data type
    data_type: str = 'unknown'  # 'cells', 'nuclei', or 'unknown'

    # Raw data
    has_raw_layer: bool = False
    raw_layer_name: str = ''
    is_counts: bool = False  # True if X contains integer counts

    # QC state
    has_qc_metrics: bool = False
    has_mt_metrics: bool = False
    has_ribo_metrics: bool = False
    has_doublet_scores: bool = False
    doublet_detection_method: str = ''

    # Normalization state
    is_normalized: bool = False
    is_log_transformed: bool = False
    normalization_method: str = ''

    # HVG state
    has_hvg: bool = False
    n_hvg: int = 0
    hvg_flavor: str = ''

    # Dimensionality reduction
    has_pca: bool = False
    n_pcs: int = 0
    has_neighbors: bool = False
    n_neighbors: int = 0
    has_umap: bool = False
    has_tsne: bool = False

    # Clustering
    has_clusters: bool = False
    cluster_key: str = ''
    n_clusters: int = 0
    clustering_method: str = ''

    # Annotations
    has_celltypist: bool = False
    celltypist_model: str = ''
    has_scimilarity: bool = False

    # Batch info
    batch_key: Optional[str] = None
    n_batches: int = 0
    batch_correction_applied: bool = False
    batch_correction_method: str = ''

    # Additional observations
    obs_columns: List[str] = field(default_factory=list)
    var_columns: List[str] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    obsm_keys: List[str] = field(default_factory=list)
    obsp_keys: List[str] = field(default_factory=list)


def _is_integer_matrix(X) -> bool:
    """Check if matrix contains integer values (counts)."""
    if sp.issparse(X):
        data = X.data
    else:
        data = X.flatten()

    # Sample for efficiency
    sample_size = min(10000, len(data))
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        data = data[indices]

    return np.allclose(data, np.round(data))


def _detect_data_type(adata: AnnData) -> str:
    """
    Detect if data is from cells or nuclei based on MT content distribution.

    Nuclei typically have very low MT content (<5%) because mitochondria
    are in the cytoplasm, while cells can have higher MT content.
    """
    if 'pct_counts_mt' not in adata.obs.columns:
        return 'unknown'

    mt_pct = adata.obs['pct_counts_mt'].values
    median_mt = np.median(mt_pct)
    max_mt = np.max(mt_pct)

    # Nuclei typically have MT < 5% for most cells
    if median_mt < 2.0 and max_mt < 10.0:
        return 'nuclei'
    else:
        return 'cells'


def _detect_batch_key(adata: AnnData) -> Optional[str]:
    """Detect the batch key from common column names."""
    common_batch_keys = [
        'batch', 'batch_id', 'sample', 'sample_id', 'donor', 'donor_id',
        'library', 'library_id', 'patient', 'condition', 'experiment'
    ]

    for key in common_batch_keys:
        if key in adata.obs.columns:
            # Check if it has multiple values
            if adata.obs[key].nunique() > 1:
                return key

    return None


def _detect_raw_layer(adata: AnnData) -> Tuple[bool, str]:
    """Detect if a raw counts layer exists."""
    common_raw_names = ['raw_counts', 'raw_data', 'counts', 'raw']

    for name in common_raw_names:
        if name in adata.layers:
            # Verify it contains integers
            if _is_integer_matrix(adata.layers[name]):
                return True, name

    return False, ''


def _detect_normalization(adata: AnnData) -> Tuple[bool, bool, str]:
    """
    Detect if data is normalized and log-transformed.

    Returns: (is_normalized, is_log_transformed, method)
    """
    # Check uns for normalization info
    method = ''
    if 'log1p' in adata.uns:
        method = 'log1p'

    # Heuristic: normalized data typically has values < 20 after log
    if sp.issparse(adata.X):
        max_val = adata.X.max()
        # Check a sample for non-integer values
        sample_data = adata.X.data[:min(10000, len(adata.X.data))]
    else:
        max_val = np.max(adata.X)
        sample_data = adata.X.flatten()[:10000]

    has_floats = not np.allclose(sample_data, np.round(sample_data))

    # Log-transformed data typically has max values < 15
    is_log = max_val < 15 and has_floats

    # Normalized but not log-transformed would have larger values
    is_normalized = has_floats

    return is_normalized, is_log, method


def _detect_clustering(adata: AnnData) -> Tuple[bool, str, int, str]:
    """Detect clustering state."""
    cluster_keys = ['leiden', 'louvain', 'pheno_leiden', 'clusters', 'cluster']

    for key in cluster_keys:
        if key in adata.obs.columns:
            n_clusters = adata.obs[key].nunique()
            method = 'phenograph' if 'pheno' in key else key
            return True, key, n_clusters, method

    return False, '', 0, ''


def inspect_data(adata: AnnData) -> DataState:
    """
    Comprehensive data inspection that returns a DataState object.

    This function analyzes an AnnData object to determine its current
    processing state, which is essential for autonomous analysis.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to inspect.

    Returns
    -------
    DataState
        A dataclass containing all detected states.
    """
    state = DataState()

    # Basic info
    state.shape = adata.shape
    state.n_cells = adata.n_obs
    state.n_genes = adata.n_vars

    # Store metadata keys
    state.obs_columns = list(adata.obs.columns)
    state.var_columns = list(adata.var.columns)
    state.layers = list(adata.layers.keys())
    state.obsm_keys = list(adata.obsm.keys())
    state.obsp_keys = list(adata.obsp.keys())

    # Raw data detection
    state.has_raw_layer, state.raw_layer_name = _detect_raw_layer(adata)
    state.is_counts = _is_integer_matrix(adata.X)

    # QC metrics
    qc_obs_cols = ['n_genes_by_counts', 'total_counts', 'n_genes']
    state.has_qc_metrics = any(col in adata.obs.columns for col in qc_obs_cols)
    state.has_mt_metrics = 'pct_counts_mt' in adata.obs.columns
    state.has_ribo_metrics = 'pct_counts_ribo' in adata.obs.columns

    # Doublet detection
    if 'doublet_score' in adata.obs.columns or 'predicted_doublet' in adata.obs.columns:
        state.has_doublet_scores = True
        state.doublet_detection_method = 'scrublet' if 'scrublet' in adata.uns else 'unknown'

    # Data type (cells vs nuclei)
    if state.has_mt_metrics:
        state.data_type = _detect_data_type(adata)

    # Normalization
    is_norm, is_log, norm_method = _detect_normalization(adata)
    state.is_normalized = is_norm
    state.is_log_transformed = is_log
    state.normalization_method = norm_method

    # HVG
    if 'highly_variable' in adata.var.columns:
        state.has_hvg = True
        state.n_hvg = int(adata.var['highly_variable'].sum())
        if 'hvg' in adata.uns:
            state.hvg_flavor = adata.uns['hvg'].get('flavor', '')

    # PCA
    if 'X_pca' in adata.obsm:
        state.has_pca = True
        state.n_pcs = adata.obsm['X_pca'].shape[1]

    # Neighbors
    if 'neighbors' in adata.uns or 'connectivities' in adata.obsp:
        state.has_neighbors = True
        if 'neighbors' in adata.uns and 'params' in adata.uns['neighbors']:
            state.n_neighbors = adata.uns['neighbors']['params'].get('n_neighbors', 0)

    # UMAP/tSNE
    state.has_umap = 'X_umap' in adata.obsm
    state.has_tsne = 'X_tsne' in adata.obsm

    # Clustering
    has_clusters, cluster_key, n_clusters, cluster_method = _detect_clustering(adata)
    state.has_clusters = has_clusters
    state.cluster_key = cluster_key
    state.n_clusters = n_clusters
    state.clustering_method = cluster_method

    # Cell type annotations
    celltypist_cols = ['predicted_labels', 'majority_voting', 'celltype_majority_voting']
    if any(col in adata.obs.columns for col in celltypist_cols):
        state.has_celltypist = True

    scimilarity_cols = ['predictions_unconstrained', 'representative_prediction']
    if any(col in adata.obs.columns for col in scimilarity_cols):
        state.has_scimilarity = True

    # Batch info
    state.batch_key = _detect_batch_key(adata)
    if state.batch_key:
        state.n_batches = adata.obs[state.batch_key].nunique()

    # Batch correction
    if 'X_scanorama' in adata.obsm:
        state.batch_correction_applied = True
        state.batch_correction_method = 'scanorama'
    elif 'X_pca_harmony' in adata.obsm:
        state.batch_correction_applied = True
        state.batch_correction_method = 'harmony'

    return state


def recommend_next_steps(state: DataState, goal: str) -> List[str]:
    """
    Recommend analysis steps to reach a user's goal.

    Parameters
    ----------
    state : DataState
        Current data state from inspect_data().
    goal : str
        User's analysis goal. Common goals:
        - 'qc': Perform quality control
        - 'cluster': Get clusters
        - 'annotate': Get cell type annotations
        - 'umap': Generate UMAP visualization
        - 'deg': Differential expression analysis
        - 'batch_correct': Batch correction

    Returns
    -------
    List[str]
        Ordered list of recommended analysis steps.
    """
    steps = []
    goal = goal.lower()

    # QC goal
    if goal == 'qc':
        if not state.has_qc_metrics:
            steps.append('calculate_qc_metrics')
        if not state.has_mt_metrics:
            steps.append('calculate_mt_metrics')
        if not state.has_doublet_scores:
            steps.append('detect_doublets')
        steps.append('filter_cells_by_qc')
        steps.append('filter_genes')
        return steps

    # All other goals require at least basic preprocessing

    # Step 1: Raw counts preservation
    if not state.has_raw_layer and state.is_counts:
        steps.append('preserve_raw_counts')

    # Step 2: QC if not done
    if not state.has_qc_metrics:
        steps.append('calculate_qc_metrics')
    if not state.has_doublet_scores:
        steps.append('detect_doublets')

    # Step 3: Normalization
    if not state.is_normalized:
        steps.append('normalize_data')

    # For visualization and clustering goals
    if goal in ['cluster', 'umap', 'annotate', 'deg', 'batch_correct']:
        # HVG selection
        if not state.has_hvg:
            steps.append('select_hvg')

        # Batch correction (if applicable and requested)
        if goal == 'batch_correct' or (state.n_batches > 1 and not state.batch_correction_applied):
            if goal == 'batch_correct':
                steps.append('run_batch_correction')

        # PCA
        if not state.has_pca:
            steps.append('run_pca')

        # Neighbors
        if not state.has_neighbors:
            steps.append('compute_neighbors')

        # UMAP
        if goal in ['umap', 'annotate'] and not state.has_umap:
            steps.append('compute_umap')

        # Clustering
        if goal in ['cluster', 'annotate', 'deg'] and not state.has_clusters:
            steps.append('run_clustering')

    # Annotation-specific
    if goal == 'annotate':
        if not state.has_celltypist:
            steps.append('run_celltypist')

    # DEG requires clusters
    if goal == 'deg':
        if state.has_clusters:
            steps.append('run_deg_analysis')
        else:
            steps.append('run_clustering')
            steps.append('run_deg_analysis')

    return steps


def summarize_state(state: DataState) -> str:
    """
    Generate a human-readable summary of the data state.

    Parameters
    ----------
    state : DataState
        Data state from inspect_data().

    Returns
    -------
    str
        Human-readable summary.
    """
    lines = []
    lines.append(f"Data shape: {state.n_cells:,} cells x {state.n_genes:,} genes")
    lines.append(f"Data type: {state.data_type}")

    # Processing state
    processing = []
    if state.has_raw_layer:
        processing.append(f"raw counts in '{state.raw_layer_name}'")
    if state.has_qc_metrics:
        processing.append("QC metrics computed")
    if state.has_doublet_scores:
        processing.append(f"doublets detected ({state.doublet_detection_method})")
    if state.is_normalized:
        processing.append("normalized")
    if state.is_log_transformed:
        processing.append("log-transformed")
    if state.has_hvg:
        processing.append(f"{state.n_hvg} HVGs selected")

    if processing:
        lines.append("Processing: " + ", ".join(processing))

    # Embeddings
    embeddings = []
    if state.has_pca:
        embeddings.append(f"PCA ({state.n_pcs} PCs)")
    if state.has_neighbors:
        embeddings.append(f"neighbors (k={state.n_neighbors})")
    if state.has_umap:
        embeddings.append("UMAP")
    if state.has_tsne:
        embeddings.append("tSNE")

    if embeddings:
        lines.append("Embeddings: " + ", ".join(embeddings))

    # Clustering
    if state.has_clusters:
        lines.append(f"Clustering: {state.n_clusters} clusters ({state.clustering_method})")

    # Annotations
    annotations = []
    if state.has_celltypist:
        annotations.append("CellTypist")
    if state.has_scimilarity:
        annotations.append("Scimilarity")
    if annotations:
        lines.append("Annotations: " + ", ".join(annotations))

    # Batch info
    if state.batch_key:
        batch_info = f"Batch: {state.n_batches} batches (key='{state.batch_key}')"
        if state.batch_correction_applied:
            batch_info += f", corrected with {state.batch_correction_method}"
        lines.append(batch_info)

    return "\n".join(lines)
