"""
Claude API tool definitions for scagent.

Tools are organized into two layers:
1. Action tools - mutate/generate analysis artifacts
2. Inspection tools - read-only queries for more detail

All tools return structured JSON for LLM reasoning.
"""

# Configure tqdm for cleaner progress bars (must be before any imports that use tqdm)
import os
os.environ.setdefault('TQDM_NCOLS', '60')
os.environ.setdefault('TQDM_MININTERVAL', '0.5')  # Update less frequently

from typing import List, Dict, Any
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


def get_tools() -> List[Dict[str, Any]]:
    """
    Get Claude API tool definitions for single-cell analysis.

    Returns
    -------
    List[Dict]
        List of tool definitions in Claude API format.
    """
    # Action tools (mutate state)
    action_tools = [
        {
            "name": "load_data",
            "description": "Replace the primary in-memory dataset with a new file. Use this when the user explicitly wants to switch focus to a different dataset. Always save the current primary with save_data first if it has been processed. Returns full inspection info (shape, state, obs columns, batch metadata) — do NOT call inspect_data afterwards.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to the h5ad or 10X h5 file to load as the new primary dataset."},
                    "goal": {"type": "string", "description": "Analysis goal hint (e.g., 'qc', 'cluster', 'annotate')"},
                    "context": {"type": "string", "description": "Optional biological context hint (e.g., 'PBMC healthy human')"}
                },
                "required": ["data_path"]
            }
        },
        {
            "name": "run_qc",
            "description": "Run or preview the quality control pipeline: QC metrics, doublet detection, cell/gene filtering. Use preview_only=true first in collaborative workflows so the user can review proposed removals before filters are applied. Do not assume intermediate h5ad saving is desired.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad or 10X h5 file (required for initial load, optional if data already in memory)"},
                    "output_path": {"type": "string", "description": "Optional path to save a processed h5ad. Prefer saving only final outputs unless the user explicitly asks for checkpoints."},
                    "preview_only": {"type": "boolean", "description": "If true, do not filter. Instead compute full QC metrics, estimate removals, and generate pre-filter QC figures."},
                    "confirm_filtering": {"type": "boolean", "description": "Required to apply cell/gene filtering. Set true only after the user has explicitly confirmed the previewed thresholds, parameters, and removal counts."},
                    "data_type": {"type": "string", "enum": ["single_cell", "single_nucleus"], "description": "Hint for MT threshold direction: 'single_nucleus' starts at 5%, 'single_cell' at 20%. The actual threshold must be chosen from the QC figure — always inspect the distribution before filtering."},
                    "mt_threshold": {"type": "number", "description": "Max MT% threshold. Overrides data_type if provided."},
                    "filter_mt": {"type": "boolean", "description": "If false, compute and report MT metrics but do not apply a hard MT% cell filter. Use this for source pipelines that inspect MT but do not remove cells by MT%."},
                    "min_genes": {"type": "integer", "description": "Minimum detected genes per cell before cell removal. This is cell-level filtering, distinct from min_cells per gene."},
                    "min_cells": {"type": "integer", "description": "Minimum cells a gene must be expressed in to be kept (default: 3). In preview, shows how many genes would be removed. Present this to the user alongside the projected removal count and confirm before applying."},
                    "remove_ribo": {"type": "boolean", "description": "Remove ribosomal genes (default: false — only set true if user explicitly requests it)"},
                    "remove_mt": {"type": "boolean", "description": "Remove mitochondrial genes from the feature set (default: false)"},
                    "detect_doublets_flag": {"type": "boolean", "description": "Run Scrublet doublet detection (default: true)"},
                    "remove_doublets": {"type": "boolean", "description": "If true, remove cells flagged as predicted doublets in apply mode. Preview mode reports the count only."},
                    "scrublet_expected_doublet_rate": {"type": "number", "description": "Scrublet expected_doublet_rate (default: 0.06)."},
                    "scrublet_sim_doublet_ratio": {"type": "number", "description": "Scrublet sim_doublet_ratio (default: 2.0)."},
                    "scrublet_n_prin_comps": {"type": "integer", "description": "Scrublet n_prin_comps / PCA components (default: 30). Set to 40 to match some published pipelines."},
                    "scrublet_min_counts": {"type": "integer", "description": "Scrublet scrub_doublets min_counts preprocessing parameter (default: 2)."},
                    "scrublet_min_cells": {"type": "integer", "description": "Scrublet scrub_doublets min_cells preprocessing parameter (default: 3)."},
                    "scrublet_min_gene_variability_pctl": {"type": "number", "description": "Scrublet scrub_doublets min_gene_variability_pctl preprocessing parameter (default: 85)."},
                    "scrublet_random_state": {"type": "integer", "description": "Random seed for Scrublet (default: 0)."},
                    "force_doublet_recompute": {"type": "boolean", "description": "If true, recompute Scrublet scores even if doublet columns already exist."},
                    "figure_dir": {"type": "string", "description": "Directory for QC figures. Plots are generated from the full pre-filter data."},
                    "batch_key": {"type": "string", "description": "Batch column for per-batch doublet detection"}
                },
                "required": []
            }
        },
        {
            "name": "normalize_and_hvg",
            "description": "Normalize, log-transform, and select highly variable genes. Preserves raw counts in layer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "n_hvg": {"type": "integer", "description": "Number of HVGs (default: 4000)"},
                    "target_sum": {"type": "number", "description": "Target counts per cell for normalize_total (default: 10000). Use source/paper value when reproducing a workflow."},
                    "log_transform": {"type": "boolean", "description": "Apply log1p after normalize_total (default: true)."},
                    "raw_layer_name": {"type": "string", "description": "Layer used to preserve/reset raw integer counts (default: raw_counts)."},
                    "force_reset_from_raw": {"type": "boolean", "description": "If true, reset adata.X from raw_layer_name before normalization when available. Use for retries to avoid double-normalization (default: true)."},
                    "set_raw_after_normalization": {"type": "boolean", "description": "If true, set adata.raw = adata.copy() after normalization/log1p and before later scaling/PCA (default: true)."},
                    "hvg_flavor": {"type": "string", "enum": ["seurat", "seurat_v3", "cell_ranger"], "description": "Scanpy HVG flavor (default: seurat_v3). seurat_v3 uses VST on raw counts and supports batch_key (ranks by median rank across batches). seurat works on log-normalized data."},
                    "hvg_layer": {"type": "string", "description": "Layer for HVG calculation. seurat_v3 requires raw integer counts; if omitted, auto-detects from 'raw_counts', 'raw_data', 'counts' in that order. Only set explicitly if your raw counts are in a non-standard layer."},
                    "batch_key": {"type": "string", "description": "obs column for batch-stratified HVG selection. Recommended for multi-sample data. Supported by all flavors including seurat_v3."},
                    "hvg_exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Regex pattern(s) for source-defined features that must not be marked highly_variable. Only pass evidence-backed source/workflow exclusions; do not invent dataset-specific patterns."},
                    "hvg_exclusion_mode": {"type": "string", "enum": ["post", "pre"], "description": "How to apply source-defined HVG exclusions. 'post' runs HVG then forces excluded features to false; 'pre' computes HVGs only on allowed features (default: post)."},
                    "hvg_exclude_match_mode": {"type": "string", "enum": ["match", "contains", "fullmatch"], "description": "Regex matching mode for hvg_exclude_patterns against var_names (default: match)."},
                    "hvg_exclusion_source": {"type": "string", "description": "Short provenance for the feature-exclusion rule, e.g. source repo file/function/line or paper method."}
                },
                "required": []
            }
        },
        {
            "name": "run_pca",
            "description": "Run PCA only. Does not compute neighbors, UMAP, clustering, or batch correction.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "n_comps": {"type": "integer", "description": "Number of PCA components (default: 30)"},
                    "svd_solver": {"type": "string", "description": "SVD solver passed to scanpy.tl.pca (default: arpack)"},
                    "mask_var": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "Boolean var column for PCA feature mask, or null for all genes (default: highly_variable)"}
                },
                "required": []
            }
        },
        {
            "name": "run_neighbors",
            "description": "Compute a neighbor graph only. Does not run PCA, UMAP, clustering, or batch correction.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "n_neighbors": {"type": "integer", "description": "Number of neighbors (default: 30)"},
                    "n_pcs": {"type": "integer", "description": "Number of PCs to use from the representation (optional)"},
                    "use_rep": {"type": "string", "description": "Representation in adata.obsm to use (default: X_pca)"},
                    "metric": {"type": "string", "description": "Distance metric (default: euclidean)"},
                    "key_added": {"type": "string", "description": "Optional alternate neighbors key. Omit to write the default graph."}
                },
                "required": []
            }
        },
        {
            "name": "run_umap",
            "description": "Compute UMAP only from an existing neighbor graph. Does not recompute PCA, neighbors, batch correction, or clustering.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "min_dist": {"type": "number", "description": "UMAP min_dist (default: 0.5, Scanpy's default)"},
                    "spread": {"type": "number", "description": "UMAP spread (default: 1.0)"},
                    "n_components": {"type": "integer", "description": "Number of UMAP dimensions (default: 2)"},
                    "neighbors_key": {"type": "string", "description": "Optional neighbors key to use. Omit to use adata.uns['neighbors']."},
                    "random_state": {"type": "integer", "description": "Random seed (default: 0)"}
                },
                "required": []
            }
        },
        {
            "name": "run_clustering",
            "description": "Run Leiden or PhenoGraph clustering. Preserves alternative clustering results under explicit keys so comparisons do not overwrite the primary clustering by accident.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "method": {"type": "string", "enum": ["leiden", "louvain", "phenograph"], "description": "Method (default: leiden)"},
                    "resolution": {"type": "number", "description": "Resolution (default: 1.0)"},
                    "cluster_key": {"type": "string", "description": "Optional explicit obs column to store this clustering result. If omitted, scagent will keep primary aliases like 'leiden' stable and store comparisons under deterministic keys like 'leiden_res_0_5'."},
                    "make_primary": {"type": "boolean", "description": "If true, promote this clustering to the default alias for the method (for example 'leiden') while preserving the explicit result key."}
                },
                "required": []
            }
        },
        {
            "name": "compare_clusterings",
            "description": "Run a safe clustering comparison across multiple resolutions without overwriting earlier results. Use this instead of chaining several run_clustering calls when the goal is to compare resolutions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "method": {"type": "string", "enum": ["leiden", "louvain", "phenograph"], "description": "Method (default: leiden)"},
                    "resolutions": {"type": "array", "items": {"type": "number"}, "description": "List of resolutions to compare"},
                    "generate_figures": {"type": "boolean", "description": "If true and UMAP is present, save one figure per clustering"},
                    "figure_dir": {"type": "string", "description": "Optional directory for generated comparison figures"},
                    "include_images": {"type": "boolean", "description": "If true, include image data for generated figures"},
                    "promote_resolution": {"type": "number", "description": "Optional resolution to promote to the primary alias after comparison"}
                },
                "required": ["resolutions"]
            }
        },
        {
            "name": "run_celltypist",
            "description": "Annotate cell types with CellTypist. Handles target_sum=10000 normalization automatically.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "model": {"type": "string", "description": "Model name (default: Immune_All_Low.pkl)"},
                    "majority_voting": {"type": "boolean", "description": "Use majority voting (default: true)"},
                    "cluster_key": {"type": "string", "description": "Cluster column to use for CellTypist majority voting (default: leiden)"}
                },
                "required": []
            }
        },
        {
            "name": "run_scimilarity",
            "description": "Annotate cell types with Scimilarity (embedding-based). Uses pretrained embeddings and kNN to annotate cells. Model path is pre-configured via SCIMILARITY_MODEL_PATH env var - do NOT ask the user for a model path. Different from CellTypist - use when you want embedding-based annotation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data if already loaded)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "cluster_key": {"type": "string", "description": "Cluster column to use for cluster-level representative predictions (default: leiden)"}
                },
                "required": []
            }
        },
        {
            "name": "run_batch_correction",
            "description": (
                "Correct batch effects using Harmony, BBKNN, Scanorama, or scVI. "
                "Harmony: fast, corrects PCA embeddings, good for mild-to-moderate batch effects. "
                "BBKNN: fast, builds a batch-balanced k-NN graph in PCA space; correction lives in "
                "the neighbor graph (not a separate embedding). Run UMAP/clustering separately unless explicitly requested. "
                "Good default when you have many samples (e.g. >10 batches). "
                "Scanorama: MNN-based, also corrects gene expression, good for partially overlapping datasets. "
                "scVI: deep generative model, models raw counts directly, best for complex/strong batch effects "
                "but requires raw_counts layer and takes longer to train (recommended max_epochs=200). "
                "This tool only performs batch correction. Run run_neighbors and run_umap as separate steps afterwards."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional)"},
                    "batch_key": {"type": "string", "description": "Column in adata.obs containing batch labels"},
                    "method": {
                        "type": "string",
                        "enum": ["harmony", "bbknn", "scanorama", "scvi"],
                        "description": (
                            "Correction method (default: harmony). "
                            "bbknn: batch-balanced graph, good for many batches (>10 samples). "
                            "scvi: best for complex effects but needs raw_counts layer."
                        )
                    },
                    "n_pcs": {"type": "integer", "description": "BBKNN only: number of PCA components to use (default: 30)"},
                    "neighbors_within_batch": {"type": "integer", "description": "BBKNN only: neighbors contributed per batch per cell (default: 3; total = n_batches × this value)"},
                    "n_latent": {"type": "integer", "description": "scVI only: latent space dimensions (default: 30)"},
                    "max_epochs": {"type": "integer", "description": "scVI only: training epochs (default: 200; use fewer only for quick tests)"},
                    "store_normalized": {"type": "boolean", "description": "scVI only: store scVI-normalized expression in layers['scvi_normalized'] (default: false)"}
                },
                "required": []
            }
        },
        {
            "name": "score_integration",
            "description": (
                "Score batch integration quality using neighborhood batch mixing entropy. "
                "For each cell, examines its k nearest neighbors in the chosen embedding and "
                "computes the Shannon entropy of batch labels — high entropy means batches are "
                "well-mixed. The score is normalized to [0, 1] where 1 = perfect mixing. "
                "Call this after run_batch_correction to quantify whether integration worked. "
                "Can also be called on uncorrected embeddings (use_rep='X_pca') as a baseline "
                "to compare before/after. Stores per-cell scores in obs['integration_entropy']."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "batch_key": {
                        "type": "string",
                        "description": "obs column with batch labels (same key used for batch correction)."
                    },
                    "use_rep": {
                        "type": "string",
                        "description": "Embedding to evaluate (default: 'X_umap'). Use the corrected embedding for post-integration score, or 'X_pca' for pre-integration baseline."
                    },
                    "n_neighbors": {
                        "type": "integer",
                        "description": "Neighborhood size for entropy calculation (default: 50). Larger = stabler but slower."
                    }
                },
                "required": ["batch_key"]
            }
        },
        {
            "name": "benchmark_integration",
            "description": (
                "Benchmark batch integration quality using scib-metrics — the same evaluation "
                "used in workshop session 5 to decide which correction method to keep. "
                "Computes bio-conservation metrics (NMI, ARI, silhouette label, cLISI) and "
                "batch-correction metrics (silhouette batch, iLISI, kBET, graph connectivity, PCR) "
                "across all corrected embeddings present in adata.obsm, always including X_pca as "
                "the uncorrected baseline. Returns a ranked table and the best-performing method. "
                "Requires: scib-metrics (pip install scib-metrics), a label_key with cell type or "
                "cluster annotations, and at least one corrected embedding from run_batch_correction."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "batch_key": {
                        "type": "string",
                        "description": "obs column with batch labels (same as used for batch correction)."
                    },
                    "label_key": {
                        "type": "string",
                        "description": "obs column with cell type or cluster labels for bio-conservation metrics (e.g. 'leiden', 'cell_type', 'celltypist_cell_type')."
                    },
                    "embedding_keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "obsm keys to benchmark. Auto-detected if omitted (includes X_pca baseline + any corrected embeddings present)."
                    },
                    "fast": {
                        "type": "boolean",
                        "description": "Skip slow metrics (kBET) for a quicker result (default: false)."
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save results CSV and results-table figure (optional but recommended)."
                    }
                },
                "required": ["batch_key", "label_key"]
            }
        },
        {
            "name": "run_deg",
            "description": "Run validated differential expression analysis between groups. Validates input data (matrix type, cluster sizes, batch confounding) and attaches validity metadata for downstream GSEA interpretation. Returns validation warnings alongside DEG results.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional)"},
                    "groupby": {"type": "string", "description": "Group column (default: leiden)"},
                    "method": {"type": "string", "enum": ["wilcoxon", "t-test", "logreg"], "description": "Method (default: wilcoxon)"},
                    "layer": {"type": "string", "description": "Optional expression layer to use for DEG (for example scran_norm)"},
                    "use_raw": {"type": "boolean", "description": "Whether to use adata.raw for DEG when layer is not set. If omitted, follows Scanpy default (uses adata.raw when present)."},
                    "key_added": {"type": "string", "description": "Key in adata.uns for DEG results (default: rank_genes_groups)"},
                    "n_genes": {"type": "integer", "description": "Number of ranked genes to store per group (default: 100)"},
                    "target_geneset": {"type": "string", "description": "Target gene set database for compatibility check (default: MSigDB_Hallmark_2020)"}
                },
                "required": []
            }
        },
        {
            "name": "run_pseudobulk_deg",
            "description": (
                "Run pseudobulk differential expression analysis using DESeq2. "
                "Aggregates raw counts to the sample level (one observation per biological replicate "
                "per cell type) before running statistics — this respects replicate independence and "
                "is strongly preferred over single-cell Wilcoxon when biological replicates are available. "
                "Requires: raw integer counts in a layer (default 'raw_counts'), a sample column with "
                "≥ 2 replicates per condition, and a condition column. "
                "Use run_deg (Wilcoxon) when no replicates are available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sample_col": {
                        "type": "string",
                        "description": "Column in adata.obs identifying biological replicates (e.g. 'sample_id', 'donor'). Each unique value must be an independent sample."
                    },
                    "condition_col": {
                        "type": "string",
                        "description": "Column in adata.obs defining the condition to compare (e.g. 'disease', 'treatment', 'timepoint')."
                    },
                    "condition_a": {
                        "type": "string",
                        "description": "Reference condition (denominator in log fold change, e.g. 'healthy', 'control')."
                    },
                    "condition_b": {
                        "type": "string",
                        "description": "Test condition (numerator in log fold change, e.g. 'disease', 'treated'). Positive LFC means upregulated here."
                    },
                    "groups_col": {
                        "type": "string",
                        "description": "Column in adata.obs containing cell type or cluster labels (e.g. 'leiden', 'cell_type'). Used to subset to a specific cell type."
                    },
                    "cell_type": {
                        "type": "string",
                        "description": "Specific cell type label from groups_col to analyze. If omitted, runs on all cells together (use when adata is already subset)."
                    },
                    "layer": {
                        "type": "string",
                        "description": "Layer containing raw integer counts (default: 'raw_counts'). DESeq2 requires non-normalized counts."
                    },
                    "min_cells": {
                        "type": "integer",
                        "description": "Minimum cells a sample must contribute to pseudobulk to be retained (default: 10). Samples below this threshold are dropped."
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Adjusted p-value threshold for significance reporting (default: 0.05)."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save full results as CSV (optional)."
                    }
                },
                "required": ["sample_col", "condition_col", "condition_a", "condition_b", "groups_col"]
            }
        },
        {
            "name": "generate_figure",
            "description": "Generate and save a visualization (UMAP, violin, dotplot, etc.). For clustering comparisons, always use an explicit cluster key returned by run_clustering or compare_clusterings rather than a bare primary alias unless you intentionally want the promoted default.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save PNG figure"},
                    "plot_type": {"type": "string", "enum": ["umap", "violin", "dotplot", "heatmap"], "description": "Plot type"},
                    "color_by": {"type": "string", "description": "Column or gene to color by"},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Genes for dotplot/heatmap"},
                    "include_image": {"type": "boolean", "description": "If true, include image data for model review (default: true)"}
                },
                "required": ["output_path", "plot_type"]
            }
        },
        {
            "name": "run_gsea",
            "description": "Run Gene Set Enrichment Analysis on DEG results. Identifies enriched biological pathways/processes. Requires DEG to be run first. Returns top enriched pathways with NES scores and FDR values.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad with DEG results (optional - uses in-memory data)"},
                    "output_dir": {"type": "string", "description": "Directory to save GSEA results"},
                    "cluster": {"type": "string", "description": "Cluster to analyze (or 'all' for all clusters)"},
                    "gene_sets": {"type": "string", "description": "Gene set database: KEGG_2021_Human, GO_Biological_Process_2021, Reactome_2022, MSigDB_Hallmark_2020 (default: KEGG_2021_Human)"},
                    "min_size": {"type": "integer", "description": "Min genes in pathway (default: 5)"},
                    "max_size": {"type": "integer", "description": "Max genes in pathway (default: 500)"},
                    "permutation_num": {"type": "integer", "description": "Permutations for p-value (default: 1000)"}
                },
                "required": ["output_dir", "cluster"]
            }
        },
        {
            "name": "run_spectra",
            "description": (
                "Run Spectra semi-supervised factor analysis to discover gene programs. "
                "Spectra fits a factor model guided by cell-type-specific gene set priors — "
                "it produces both gene-set-guided factors (e.g. a T cell exhaustion program) "
                "and de novo factors that explain residual variation not covered by the priors. "
                "Outputs per-cell factor scores in obsm['SPECTRA_cell_scores'] (visualizable on UMAP), "
                "top marker genes per factor in uns['SPECTRA_markers'], and gene loadings in uns['SPECTRA_factors']. "
                "Requires: log-normalized counts in adata.X, a cell_type_key, and the Spectra-sc package. "
                "Gene set dictionary format: JSON with cell type keys (one entry per cell type, even if empty {}) "
                "plus a 'global' key. If none provided, runs in de novo mode (unsupervised). "
                "Workshop note: num_epochs=100 for demos, 10000 for serious analysis."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "cell_type_key": {
                        "type": "string",
                        "description": "obs column with cell type labels (e.g. 'celltypist_cell_type', 'leiden'). Every unique value must have an entry in the gene set dictionary."
                    },
                    "gene_set_dict_path": {
                        "type": "string",
                        "description": "Path to a JSON file containing the gene set dictionary. Format: {cell_type: {gene_set_name: [gene, ...]}, 'global': {gene_set_name: [gene, ...]}}. Missing cell types are auto-filled with empty entries."
                    },
                    "use_default_gene_sets": {
                        "type": "boolean",
                        "description": "Use Spectra's built-in default gene sets instead of a custom dictionary (default: false)."
                    },
                    "lam": {
                        "type": "number",
                        "description": "Regularization toward input gene sets (default: 0.1). Lower = stronger adherence to provided gene sets. Range: 0.001–0.5."
                    },
                    "num_epochs": {
                        "type": "integer",
                        "description": "Training iterations (default: 1000). Use 100 for a quick test, 10000 for publication-quality results."
                    },
                    "n_top_vals": {
                        "type": "integer",
                        "description": "Top genes per factor stored in SPECTRA_markers (default: 50)."
                    },
                    "use_highly_variable": {
                        "type": "boolean",
                        "description": "Restrict to highly variable genes plus gene set genes (default: true)."
                    },
                    "use_cell_types": {
                        "type": "boolean",
                        "description": "Fit cell-type-specific factors in addition to global factors (default: true)."
                    },
                    "overlap_threshold": {
                        "type": "number",
                        "description": "Minimum overlap coefficient to label a factor with a gene set name (default: 0.2)."
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save the Spectra model and UMAP factor score figures (recommended)."
                    }
                },
                "required": ["cell_type_key"]
            }
        },
        {
            "name": "query_cells",
            "description": (
                "Search the Scimilarity reference database (~24M cells) for cells most similar to a query. "
                "Two modes:\n"
                "- 'cells': query using specific cells (by obs_names list or a boolean obs column). "
                "Uses the mean Scimilarity embedding of the selected cells.\n"
                "- 'centroid': query using the centroid of a cluster or cell type group "
                "(provide group_key + group_value). More robust for heterogeneous populations.\n"
                "Returns the top-k matching reference cells with their metadata: cell type, tissue, disease, study, distance. "
                "Useful for: validating ambiguous annotations, finding analogous cell states in other datasets, "
                "characterising novel populations. "
                "Requires Scimilarity to be installed and run_scimilarity to have been run (or X_scimilarity in obsm). "
                "For centroid mode, raw counts must be available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["cells", "centroid"],
                        "description": "'cells' (default): query by specific cells. 'centroid': query by cluster/celltype centroid."
                    },
                    "cell_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of obs_names to use as query (cells mode). Use this for querying a specific set of cells."
                    },
                    "obs_column": {
                        "type": "string",
                        "description": "obs column where True/1 marks query cells (cells mode). Alternative to cell_ids."
                    },
                    "group_key": {
                        "type": "string",
                        "description": "obs column containing group labels (centroid mode), e.g. 'leiden' or 'celltypist_majority_voting'."
                    },
                    "group_value": {
                        "type": "string",
                        "description": "Which group to use as the centroid query (centroid mode), e.g. '3' or 'Macrophage'."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of nearest reference cells to retrieve (default: 50). Increase to 500+ for broader characterisation."
                    },
                    "raw_layer": {
                        "type": "string",
                        "description": "Layer with raw integer counts (used in centroid mode). Leave unset to auto-detect."
                    }
                },
                "required": []
            }
        },
        {
            "name": "score_gene_signature",
            "description": (
                "Score each cell for a gene signature using Scanpy's implementation of the Seurat method "
                "(average expression of the gene list minus the average of a random control set of similar "
                "expression level). Scores are stored in adata.obs under 'score_name'. "
                "Optionally run cell cycle scoring (S/G2M/G1 phases) as a special case. "
                "Works on normalized data; no raw counts required. "
                "Use for: cell cycle regression, pathway activity, viral signature, stress response, etc."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "gene_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of gene names to score. Genes not found in the dataset are silently dropped; the tool reports how many were matched."
                    },
                    "score_name": {
                        "type": "string",
                        "description": "Column name to store the score in adata.obs (default: 'gene_signature_score'). Use a descriptive name, e.g. 'IFN_response_score'."
                    },
                    "cell_cycle": {
                        "type": "boolean",
                        "description": "If true, run cell cycle scoring instead. Requires s_genes and g2m_genes. Adds 'S_score', 'G2M_score', and 'phase' to adata.obs."
                    },
                    "s_genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "S-phase gene list for cell cycle scoring (only used when cell_cycle=true)."
                    },
                    "g2m_genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "G2M-phase gene list for cell cycle scoring (only used when cell_cycle=true)."
                    },
                    "layer": {
                        "type": "string",
                        "description": "Layer to use for scoring. Defaults to adata.X (normalized counts). Do not use raw counts — the score uses expression levels, not counts."
                    },
                    "n_bins": {
                        "type": "integer",
                        "description": "Number of expression bins for control gene sampling (default: 25). Increase if you have very few genes."
                    },
                    "ctrl_size": {
                        "type": "integer",
                        "description": "Number of control genes sampled per bin (default: 50). Set to len(gene_list) for a tighter control."
                    }
                },
                "required": []
            }
        },
        {
            "name": "save_data",
            "description": "Save the current in-memory AnnData object without modifying it. Use this as the final save step after analysis and annotation are complete.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Path to save the current in-memory h5ad"}
                },
                "required": ["output_path"]
            }
        },
    ]

    # Meta tools (agent control)
    # Note: ask_user is intentionally absent. The agent follows a turn-based
    # model (like Claude Code / Codex): run all tools to completion, produce a
    # final response with numbered options, then wait for the user's next message.
    # The user's reply comes back through the CLI loop as a normal analyze() call
    # so state, data, and conversation history are always fully maintained.
    meta_tools = [
        {
            "name": "run_code",
            "description": "FLEXIBLE FALLBACK: Execute custom Python code on the AnnData object. This is your most versatile tool - use it for ANY valid request not covered by specialized tools. Examples: custom plots (variance explained, gene correlations, histograms), data filtering (remove clusters, subset cells), calculations (cluster sizes, gene stats), or any scanpy/pandas operation. Access: adata, sc (scanpy), plt (matplotlib), np, pd, output_dir, Path, ensure_dir(path). Use ensure_dir() to create directories before saving. ALWAYS prefer this over saying 'I can't do that'.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Has access to: adata, sc, plt, np, pd, output_dir, Path, ensure_dir(), write_report(). Key helpers: ensure_dir(path) creates the dir and returns a Path — use it for figures: fig_dir = ensure_dir(Path(output_dir) / 'figures'); out = fig_dir / 'plot.png'. write_report(name, content) saves a markdown report to reports/name.md and returns the path — always use this instead of open() when saving text results, never write .txt files. Do NOT import os. When loading 10x h5 files with sc.read_10x_h5(), always call .var_names_make_unique() on each AnnData before concatenating. Use series.iloc[pos] not series[pos] for positional pandas access."},
                    "description": {"type": "string", "description": "Brief description of what the code does"},
                    "save_to": {"type": "string", "description": "Optional path to save adata after execution"}
                },
                "required": ["code", "description"]
            }
        },
        {
            "name": "run_shell",
            "description": (
                "Run a shell command and return stdout/stderr. Use for system checks, "
                "CLI tools, and anything that isn't Python. "
                "Examples: 'nvidia-smi' (GPU availability and memory), 'free -h' (RAM), "
                "'df -h .' (disk space), 'which cellbender' (tool installed?), "
                "'cellbender remove-background --input raw.h5 --output clean.h5' (run CellBender), "
                "'pip show scib-metrics' (package version), 'ls -lh /path/to/data'. "
                "stdout and stderr are both captured and returned. "
                "Commands that modify or delete files outside the output directory, "
                "write to device files, or escalate privileges are blocked."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run. Executed via bash -c. Use absolute paths for reliability."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 60). Use a longer value for slow CLI tools like CellBender."
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory for the command (default: current session output directory)."
                    }
                },
                "required": ["command"]
            }
        },
        {
            "name": "web_search",
            "description": (
                "Search the web for documentation, API references, package guides, troubleshooting, and tutorials. "
                "Use the `site` parameter to target specific documentation sources. "
                "Common bioinformatics doc sites: scanpy.readthedocs.io, anndata.readthedocs.io, "
                "celltypist.readthedocs.io, gseapy.readthedocs.io, harmonypy.readthedocs.io, "
                "scvi-tools.readthedocs.io, muon.readthedocs.io, squidpy.readthedocs.io. "
                "For troubleshooting use scverse.discourse.org or github.com. "
                "Use search_papers instead for peer-reviewed scientific evidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query — be specific (e.g., 'scanpy normalize_total target_sum parameter' rather than 'scanpy normalize')"},
                    "site": {"type": "string", "description": "Optional domain to restrict results (e.g., 'scanpy.readthedocs.io')"},
                    "max_results": {"type": "integer", "description": "Maximum results (default: 5)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_papers",
            "description": (
                "Search PubMed for peer-reviewed scientific literature. Use for: cell type markers, "
                "pathway biology, disease mechanisms, method papers, and any claim that needs a citation. "
                "Automatically normalises GSEA/gene set names (strips HALLMARK_, REACTOME_, GO_ prefixes). "
                "Returns PMID, first author, year, journal, abstract, and PubMed URL. "
                "Use web_search for package documentation instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text query or PubMed search string. GSEA set names are normalised automatically (e.g. 'HALLMARK_TNFA_SIGNALING_VIA_NFKB' → 'TNF alpha signaling NF-kB'). Be specific: include cell type, disease, or gene names for better results."},
                    "max_results": {"type": "integer", "description": "Maximum papers (default: 5)"},
                    "recent_years": {"type": "integer", "description": "Restrict to last N years (default: 5). Use 10-15 for foundational method papers."},
                    "reviews_only": {"type": "boolean", "description": "Return review articles only — good for overviews of a topic"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "fetch_url",
            "description": (
                "Fetch the full text of a web page. Use after web_search when snippets are not enough — "
                "e.g. to read a function's full parameter list, a method's README, or a paper abstract. "
                "Works well for readthedocs, GitHub READMEs, and static HTML pages. "
                "JavaScript-heavy sites (Notion, some dashboards) may return little content."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default: 4000; increase to 8000 for long API pages)"}
                },
                "required": ["url"]
            }
        },
        {
            "name": "install_package",
            "description": "Request installation of a Python package. Requires user approval. Use when you need a package that isn't installed (e.g., gseapy, mygene, biomart).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "Package name (pip format)"},
                    "reason": {"type": "string", "description": "Why this package is needed"}
                },
                "required": ["package", "reason"]
            }
        },
        {
            "name": "pause_and_ask",
            "description": (
                "Pause the analysis and ask the user for guidance. "
                "Use ONLY when you genuinely cannot proceed without information only the user can provide — "
                "e.g. ambiguous batch key, surprising results that change the analysis direction, "
                "or a fork where both paths have large and different downstream consequences. "
                "Do NOT use for routine preprocessing steps, algorithm defaults, or reversible choices. "
                "After calling this tool, present the question in your response and end your turn."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question to ask the user. Be concrete — state what you found and what you need to know."
                    },
                    "context": {
                        "type": "string",
                        "description": "Why you cannot infer the answer yourself. Reference the actual data (e.g. 'I see 3 columns that could be the batch key: sample_id, batch, donor_id')."
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of discrete choices if applicable. Omit for open-ended questions."
                    }
                },
                "required": ["question", "context"]
            }
        },
    ]

    # Inspection tools (read-only)
    inspection_tools = [
        {
            "name": "inspect_data",
            "description": "Inspect data state: shape, processing status, available embeddings, what steps are done, likely metadata columns, and tracked clustering results. Use this first to understand the data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to a single h5ad or 10x h5 file (optional - uses in-memory data). Do NOT pass a directory; for multi-sample loading use run_code."},
                    "goal": {"type": "string", "description": "Analysis goal to get recommendations (e.g., 'cluster', 'annotate')"},
                    "context": {"type": "string", "description": "Optional biological context hint from the user or file path (e.g., 'PBMC healthy human cells')"}
                },
                "required": []
            }
        },
        {
            "name": "inspect_session",
            "description": "Inspect the unified agent session state: active dataset summary, artifacts, recent actions, unresolved decisions, and latest verification.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "include_history": {"type": "boolean", "description": "Include recent events and resolved decisions (default: true)"}
                },
                "required": []
            }
        },
        {
            "name": "list_artifacts",
            "description": "List known artifacts from the current session ledger or a saved run manifest.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "run_path": {"type": "string", "description": "Optional run directory or manifest.json path when inspecting a saved run."},
                    "artifact_kind": {"type": "string", "description": "Optional artifact kind filter (for example figure, report, data, log)."},
                    "limit": {"type": "integer", "description": "Maximum artifacts to return (default: 20)"}
                },
                "required": []
            }
        },
        {
            "name": "get_cluster_sizes",
            "description": "Get cell counts per cluster.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"},
                    "cluster_key": {"type": "string", "description": "Cluster column (default: leiden)"}
                },
                "required": []
            }
        },
        {
            "name": "get_top_markers",
            "description": "Get top marker genes for a cluster (requires DEG analysis first).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad with DEG results (optional - uses in-memory data)"},
                    "cluster": {"type": "string", "description": "Cluster ID"},
                    "n_genes": {"type": "integer", "description": "Number of genes (default: 10)"},
                    "key": {"type": "string", "description": "DEG result key in adata.uns (default: rank_genes_groups)"}
                },
                "required": ["cluster"]
            }
        },
        {
            "name": "summarize_qc_metrics",
            "description": "Get summary statistics of QC metrics (library size, genes, MT%, doublet scores).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"}
                },
                "required": []
            }
        },
        {
            "name": "get_celltypes",
            "description": "Get cell type annotation summary (counts per type).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"},
                    "annotation_key": {"type": "string", "description": "Annotation column (default: auto-detect)"}
                },
                "required": []
            }
        },
        {
            "name": "list_obs_columns",
            "description": "List available columns in obs (cell metadata).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"}
                },
                "required": []
            }
        },
        {
            "name": "review_figure",
            "description": "Attach and review an existing saved figure with the LLM. Use this when the user wants the agent to interpret QC plots, UMAPs, or other already-generated artifacts.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "figure_path": {"type": "string", "description": "Path to an existing figure image file"},
                    "question": {"type": "string", "description": "Optional prompt to guide the review of the figure"},
                    "include_image": {"type": "boolean", "description": "If true, include the image data for model review (default: true)"}
                },
                "required": ["figure_path"]
            }
        },
        {
            "name": "review_artifact",
            "description": "Review an existing artifact from the session or workspace. Supports figures, text reports, JSON outputs, logs, and AnnData files in read-only mode.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "artifact_path": {"type": "string", "description": "Absolute or run-relative path to the artifact."},
                    "artifact_id": {"type": "string", "description": "Artifact id from list_artifacts or inspect_session."},
                    "question": {"type": "string", "description": "Optional prompt to guide the review."},
                    "include_image": {"type": "boolean", "description": "If true, include image data when reviewing image artifacts (default: true)."},
                    "max_chars": {"type": "integer", "description": "Maximum text characters to return for text-like artifacts (default: 4000)."}
                },
                "required": []
            }
        },
        {
            "name": "inspect_run_state",
            "description": "Inspect a saved run manifest or the active run ledger: status, steps, artifacts, decisions, and recent world-state snapshots.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "run_path": {"type": "string", "description": "Run directory or manifest.json path. Optional when an active run exists."},
                    "include_history": {"type": "boolean", "description": "Include recent events and snapshots (default: true)."}
                },
                "required": []
            }
        },
        {
            "name": "inspect_workspace",
            "description": "Read-only workspace inspection for the current project or run directory. Use this sparingly for awareness and recovery.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path to inspect (default: current working directory)."},
                    "max_depth": {"type": "integer", "description": "Maximum directory depth to traverse (default: 2)."},
                    "limit": {"type": "integer", "description": "Maximum entries to return (default: 50)."}
                },
                "required": []
            }
        },
        {
            "name": "read_file",
            "description": (
                "Read and return the content of a file. Supports PDF (text extraction and optional page rendering), "
                "plain text, Markdown, CSV, TSV, and JSON. Use this to read a paper, protocol, metadata table, "
                "marker gene list, or any other reference document the user provides. "
                "For large PDFs, use pages to read specific sections (e.g. methods). "
                "Set render_pages=true to render PDF pages as images — the first page is sent directly to the vision "
                "model so figures and plots embedded in the document are visible. Additional rendered pages are saved "
                "to figures/pdf_pages/ and can be reviewed with review_figure. "
                "Rendered pages are the right approach for scanned PDFs or pages where the content is primarily visual."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file."},
                    "pages": {"type": "string", "description": "For PDFs: page range to extract, e.g. '1-5' or '3' or '2,4,6'. Default: all pages."},
                    "max_chars": {"type": "integer", "description": "Maximum characters to return (default: 20000). Increase for longer documents."},
                    "render_pages": {"type": "boolean", "description": "PDF only. Render pages as images (108 DPI PNG) in addition to text extraction. The first rendered page is sent to the vision model inline; others are saved to figures/pdf_pages/ for review_figure. Use when the document has figures, plots, or is scanned. Default: false."},
                },
                "required": ["path"]
            }
        },
        {
            "name": "research_findings",
            "description": (
                "Search PubMed for literature about a specific pathway or gene set in the context of a cell type. "
                "Returns recent papers and review articles. Used internally after GSEA to ground pathway "
                "interpretations in published evidence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pathway": {"type": "string", "description": "Pathway or gene set name (e.g. 'HALLMARK_TNFA_SIGNALING_VIA_NFKB')."},
                    "cell_type": {"type": "string", "description": "Cell type context for the search (e.g. 'CD8 T cells')."},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Top leading-edge genes to include in the query."},
                    "context": {"type": "string", "description": "Additional biological context (tissue, disease, species)."},
                    "recent_years": {"type": "integer", "description": "Limit search to this many recent years (default: 3)."},
                },
                "required": ["pathway"]
            }
        },
    ]

    return action_tools + meta_tools + inspection_tools


def get_openai_tools() -> List[Dict[str, Any]]:
    """
    Get OpenAI-format tool definitions.

    OpenAI uses a different schema format than Anthropic.
    """
    anthropic_tools = get_tools()
    openai_tools = []

    for tool in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            }
        })

    return openai_tools


def encode_image_base64(image_path: str) -> str:
    """Encode an image file to base64 for vision API."""
    import base64
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type for image."""
    ext = image_path.lower().split(".")[-1]
    mime_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return mime_types.get(ext, "image/png")


def _dataframe_preview(df, n: int = 5, max_cols: int = 20) -> dict:
    """Return a JSON-serialisable head() preview of a DataFrame for LLM display."""
    import math
    subset = df.iloc[:n, :max_cols]
    truncated_cols = df.shape[1] > max_cols

    def _clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        try:
            # Categorical → string so json.dumps doesn't choke
            return v.item() if hasattr(v, "item") else str(v) if not isinstance(v, (int, float, bool, type(None))) else v
        except Exception:
            return str(v)

    rows = []
    for idx, row in subset.iterrows():
        entry = {"_index": str(idx)}
        entry.update({col: _clean(val) for col, val in row.items()})
        rows.append(entry)

    return {
        "columns": ["_index"] + list(subset.columns),
        "rows": rows,
        "total_rows": df.shape[0],
        "total_cols": df.shape[1],
        "truncated_cols": truncated_cols,
    }


def process_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any],
    adata=None,
    world_state=None,
    run_manager=None,
) -> tuple:
    """
    Process a tool call and return structured JSON result.

    Returns
    -------
    tuple
        (json_result_string, updated_adata)
    """
    import numpy as np

    from ..core import (
        clustering_record_to_dict,
        default_cluster_key_for_method,
        get_clustering_registry,
        infer_cluster_key,
        inspect_data,
        load_data,
        metadata_candidate_to_dict,
        metadata_resolution_to_dict,
        obs_columns_detail as _obs_columns_detail,
        promote_clustering_to_primary,
        recommend_next_steps,
        register_clustering,
        resolve_batch_metadata,
        run_qc_pipeline,
        normalize_data,
        run_pca,
        compute_neighbors,
        compute_umap,
        run_leiden,
        run_phenograph,
        calculate_qc_metrics,
        detect_doublets,
    )
    from ..core.normalization import select_hvg
    from ..core.clustering import run_differential_expression, get_top_markers
    from ..annotation import run_celltypist, run_scimilarity
    from ..batch import run_scanorama, run_harmony, run_scvi, run_bbknn

    from .decision_policy import (
        decision_for_batch_strategy,
        decision_for_clustering_selection,
    )
    from .world_state import ArtifactRecord, StateDelta, VerificationResult

    def make_state(adata):
        """Create compact state dict."""
        state = inspect_data(adata)
        return {
            "has_raw_counts": state.has_raw_layer or state.has_raw,
            "raw_in_adata_raw": state.has_raw,
            "raw_in_layer": state.has_raw_layer,
            "raw_layer_name": state.raw_layer_name if state.has_raw_layer else None,
            "has_qc_metrics": state.has_qc_metrics,
            "has_doublets": state.has_doublet_scores,
            "is_normalized": state.is_normalized,
            "has_hvg": state.has_hvg,
            "has_pca": state.has_pca,
            "has_neighbors": state.has_neighbors,
            "has_umap": state.has_umap,
            "has_clusters": state.has_clusters,
            "has_celltypes": state.has_celltype_annotations,
        }

    starting_state = make_state(adata) if adata is not None else {}

    def _stage_from_state(state_dict: Dict[str, Any]) -> str:
        if not state_dict:
            return "uninitialized"
        if state_dict.get("has_celltypes"):
            return "annotated"
        if state_dict.get("has_clusters"):
            return "clustered"
        if state_dict.get("has_umap") or state_dict.get("has_neighbors"):
            return "embedded"
        if state_dict.get("is_normalized") or state_dict.get("has_hvg"):
            return "normalized"
        if state_dict.get("has_qc_metrics") or state_dict.get("has_doublets"):
            return "qc"
        if state_dict.get("has_raw_counts"):
            return "loaded"
        return "unknown"

    def _artifact_kind_from_path(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            return "figure"
        if suffix in {".h5ad", ".h5", ".loom"}:
            return "data"
        if suffix in {".json"}:
            return "json"
        if suffix in {".md", ".txt", ".csv", ".tsv"}:
            return "report"
        if suffix in {".log"}:
            return "log"
        return "artifact"

    def _artifact_payload(path: str, *, role: str = "artifact", metadata: Dict[str, Any] | None = None):
        if not path:
            return None
        artifact = ArtifactRecord.from_path(
            path,
            kind=_artifact_kind_from_path(path),
            role=role,
            source_tool=tool_name,
            metadata=metadata or {},
        )
        return artifact.to_dict()

    def _build_state_delta(
        current_adata,
        *,
        summary: str,
        dataset_changed: bool,
        notes: List[str] | None = None,
    ) -> Dict[str, Any]:
        after_state = make_state(current_adata) if current_adata is not None else {}
        changed_flags = {}
        all_keys = set(starting_state.keys()) | set(after_state.keys())
        for key in sorted(all_keys):
            before = starting_state.get(key)
            after = after_state.get(key)
            if before != after:
                changed_flags[key] = {"before": before, "after": after}
        return StateDelta(
            tool=tool_name,
            summary=summary,
            dataset_changed=dataset_changed,
            stage_before=_stage_from_state(starting_state),
            stage_after=_stage_from_state(after_state),
            changed_flags=changed_flags,
            notes=notes or [],
        ).to_dict()

    def _build_verification(
        status: str,
        summary: str,
        checks: List[Dict[str, Any]],
        recovery_options: List[str] | None = None,
    ) -> Dict[str, Any]:
        if status == "passed" and any(check.get("status") == "failed" for check in checks):
            status = "warning"
        return VerificationResult(
            status=status,
            summary=summary,
            checks=checks,
            recovery_options=recovery_options or [],
        ).to_dict()

    def _check(name: str, passed: bool, details: str) -> Dict[str, Any]:
        return {"name": name, "status": "passed" if passed else "failed", "details": details}

    def _neighbors_provenance(adata_obj) -> Dict[str, Any]:
        neighbors = adata_obj.uns.get("neighbors", {}) if adata_obj is not None else {}
        params = neighbors.get("params", {}) if isinstance(neighbors, dict) else {}
        connectivities = adata_obj.obsp.get("connectivities") if adata_obj is not None and "connectivities" in adata_obj.obsp else None
        distances = adata_obj.obsp.get("distances") if adata_obj is not None and "distances" in adata_obj.obsp else None
        return {
            "has_neighbors": bool(adata_obj is not None and "neighbors" in adata_obj.uns),
            "params": _sanitize_uns_value(dict(params)) if isinstance(params, dict) else _sanitize_uns_value(params),
            "connectivities_nnz": int(connectivities.nnz) if connectivities is not None and hasattr(connectivities, "nnz") else None,
            "distances_nnz": int(distances.nnz) if distances is not None and hasattr(distances, "nnz") else None,
            "connectivities_key": neighbors.get("connectivities_key") if isinstance(neighbors, dict) else None,
            "distances_key": neighbors.get("distances_key") if isinstance(neighbors, dict) else None,
        }

    def _provenance_same(before: Dict[str, Any], after: Dict[str, Any]) -> bool:
        return before == after

    def _finalize_result(
        result: Dict[str, Any],
        updated_adata,
        *,
        dataset_changed: bool,
        summary: str,
        artifacts_created: List[Dict[str, Any]] | None = None,
        decisions_raised: List[Dict[str, Any]] | None = None,
        verification: Dict[str, Any] | None = None,
        notes: List[str] | None = None,
    ):
        result.setdefault(
            "state_delta",
            _build_state_delta(
                updated_adata,
                summary=summary,
                dataset_changed=dataset_changed,
                notes=notes,
            ),
        )
        result.setdefault("artifacts_created", artifacts_created or [])
        result.setdefault("decisions_raised", decisions_raised or [])
        result.setdefault(
            "verification",
            verification
            or _build_verification(
                "passed",
                f"{tool_name} completed without verification issues.",
                [],
            ),
        )
        return json.dumps(result, indent=2), updated_adata

    def _confirmed_decision_value(key: str):
        if world_state is None:
            return None
        getter = getattr(world_state, "get_confirmed_value", None)
        if getter is None:
            return None
        return getter(key)

    def _clusterings_payload(adata_obj):
        return [
            clustering_record_to_dict(record)
            for record in get_clustering_registry(adata_obj)
        ]

    def _batch_relevance(goal: Any = None, context: str = "") -> bool:
        goal_text = str(goal or "").strip().lower()
        if goal_text == "batch_correct":
            return True
        context_text = str(context or "").lower()
        return bool(
            re.search(
                r"\b(batch|integration|integrate|harmony|scanorama|correct(?:ion)?|multi[- ]sample)\b",
                context_text,
            )
        )

    def _analysis_guidance(state, *, goal: Any = None, context: str = "") -> Dict[str, Any]:
        if not state.has_qc_metrics:
            next_priority = "qc_preview"
        elif not state.is_normalized:
            next_priority = "normalize_and_hvg"
        elif not (state.has_pca and state.has_neighbors and state.has_umap):
            next_priority = "run_pca"
        elif not state.has_clusters:
            next_priority = "run_clustering"
        else:
            next_priority = "annotation_or_deg"

        batch_relevant_now = _batch_relevance(goal, context)
        notes = [
            "For a routine first pass, keep the workflow QC-first before moving into normalization, embedding, and clustering.",
        ]
        if batch_relevant_now:
            notes.append(
                "Batch handling is relevant for this request, so confirm the partition column before correction or other batch-sensitive steps."
            )
        else:
            notes.append(
                "Do not make batch correction a front-and-center decision right now; batch/sample metadata only matters later for explicit integration workflows or per-batch operations."
            )

        return {
            "next_priority": next_priority,
            "batch_relevant_now": batch_relevant_now,
            "notes": notes,
        }

    def _available_annotation_keys(adata_obj) -> List[str]:
        ocd = _obs_columns_detail(adata_obj.obs, adata_obj.n_obs).get("columns", {})
        return [
            col for col, info in ocd.items()
            if info.get("note") != "high_cardinality"
            and 2 <= info.get("n_unique", 0) <= 300
            and info.get("dtype") in ("object", "category")
        ]

    def _available_plot_colors(adata_obj) -> List[str]:
        preferred = [
            "leiden",
            "sample_id",
            "batch",
            "sample",
            "pct_counts_mt",
            "total_counts",
            "n_genes_by_counts",
        ]
        available: List[str] = []
        for candidate in preferred + list(adata_obj.obs.columns):
            if candidate in adata_obj.obs.columns and candidate not in available:
                available.append(candidate)
        return available[:20]

    def _smart_unavailable_result(
        *,
        tool: str,
        message: str,
        adata_obj,
        recovery_options: List[str],
        missing_prerequisites: List[str] | None = None,
        extra: Dict[str, Any] | None = None,
    ):
        payload = {
            "status": "warning",
            "tool": tool,
            "message": message,
            "missing_prerequisites": missing_prerequisites or [],
            "recovery_options": recovery_options,
            "available_clusterings": _clusterings_payload(adata_obj) if adata_obj is not None else [],
            "available_annotation_keys": _available_annotation_keys(adata_obj) if adata_obj is not None else [],
            "available_plot_colors": _available_plot_colors(adata_obj) if adata_obj is not None else [],
            "state": make_state(adata_obj) if adata_obj is not None else {},
        }
        if extra:
            payload.update(extra)
        return _finalize_result(
            payload,
            adata_obj,
            dataset_changed=False,
            summary=message,
            verification=_build_verification(
                "warning",
                message,
                [],
                recovery_options=recovery_options,
            ),
        )

    def _error_result(
        *,
        tool: str,
        message: str,
        adata_obj=None,
        recovery_options: List[str] | None = None,
        install_hint: str | None = None,
        extra: Dict[str, Any] | None = None,
    ):
        """Return a standardized error tuple ``(json_str, adata)``.

        Every tool error should go through this helper so the LLM always
        sees the same shape: ``{status, tool, message, recovery_options,
        available_columns}``.  ``install_hint`` is a shortcut that appends
        a "pip install …" option automatically.
        """
        opts = list(recovery_options or [])
        if install_hint:
            opts.append(f"Install the missing package: {install_hint}")
        payload: Dict[str, Any] = {
            "status": "error",
            "tool": tool,
            "message": message,
            "recovery_options": opts,
        }
        if adata_obj is not None:
            payload["available_columns"] = list(adata_obj.obs.columns[:30])
        if extra:
            payload.update(extra)
        return json.dumps(payload, indent=2), adata_obj

    def get_adata(tool_input, existing_adata, update_memory: bool = True, prefer_memory: bool = False):
        """Get adata from memory or load from disk.

        If ``update_memory`` is False, loading from disk is treated as read-only
        and does not replace the active in-memory AnnData tracked by the agent.
        """
        data_path = tool_input.get("data_path")
        if prefer_memory and existing_adata is not None:
            return existing_adata, existing_adata
        # If adata is already in memory and no specific path given, use it
        if existing_adata is not None and (data_path is None or data_path == "memory"):
            return existing_adata, existing_adata
        # Otherwise load from disk
        if data_path and data_path != "memory":
            loaded = load_data(data_path)
            if update_memory:
                return loaded, loaded
            return loaded, existing_adata
        raise ValueError("No data available. Provide data_path or load data first.")

    def fix_output_path(output_path: str, tool_name: str) -> str:
        """Normalize output_path values for h5ad-producing tools.

        Relative paths with no directory component (e.g. 'result.h5ad') are
        resolved inside the run directory when run_manager is available, so
        saved files land alongside figures and reports rather than in cwd.
        """
        import os as os_module
        if output_path is None:
            return None
        if os_module.path.isdir(output_path):
            if tool_name == "save_data":
                return os_module.path.join(output_path, "final_result.h5ad")
            return None
        # Bare filename with no directory component → put it in the run dir
        if (run_manager is not None
                and not os_module.path.isabs(output_path)
                and os_module.path.dirname(output_path) == ""):
            return os_module.path.join(run_manager.run_dir, output_path)
        return output_path

    def _resolve_integer_counts_layer(adata, requested_layer: str = "raw_counts"):
        """
        Find a layer with true integer counts, or raise a clear ValueError.

        Checks (in order):
        1. The explicitly requested layer name
        2. Common raw layer names (raw_counts, raw_data, counts)
        3. adata.raw — but ONLY if it contains integer values

        Returns (layer_name_or_sentinel, X_matrix) where layer_name_or_sentinel
        is either a key in adata.layers or '__raw__' if adata.raw is the source.
        Raises ValueError with a user-facing message if no integer counts found.
        """
        from ..core.inspector import _is_integer_matrix

        # 1. Explicitly requested layer
        if requested_layer and requested_layer in adata.layers:
            if _is_integer_matrix(adata.layers[requested_layer]):
                return requested_layer, adata.layers[requested_layer]
            else:
                raise ValueError(
                    f"Layer '{requested_layer}' exists but does not contain integer counts "
                    f"(found float values — likely already normalized). "
                    f"This tool requires raw UMI/read counts. "
                    f"Available layers: {list(adata.layers.keys())}"
                )

        # 2. Common raw layer names
        for name in ["raw_counts", "raw_data", "counts"]:
            if name in adata.layers and _is_integer_matrix(adata.layers[name]):
                return name, adata.layers[name]

        # 3. adata.raw — only if truly integer
        if adata.raw is not None:
            if _is_integer_matrix(adata.raw.X):
                return "__raw__", adata.raw.X
            else:
                raise ValueError(
                    "adata.raw exists but contains non-integer values (likely log-normalized). "
                    "This tool requires raw UMI/read counts. "
                    "The original integer counts are not present in this object — "
                    "reload from the source file or use a checkpoint saved before normalization."
                )

        # 4. Nothing found
        raise ValueError(
            "No integer count layer found. "
            f"Checked: layer '{requested_layer}', 'raw_counts', 'raw_data', 'counts', and adata.raw. "
            f"Available layers: {list(adata.layers.keys())}. "
            "This tool requires raw UMI/read counts. Save them with normalize_and_hvg "
            "which preserves raw counts in layers['raw_counts'] before normalizing."
        )

    def _state_preservation_warning(tool_input, existing_adata):
        if existing_adata is not None and tool_input.get("data_path") not in (None, "memory"):
            return ["Ignored data_path and continued with the in-memory dataset to preserve prior analysis state."]
        return []

    def _validate_obs_column(adata_obj, column_name: str, warnings: List[str], *, required: bool = False, context: str = "parameter"):
        """Validate an obs column reference and either warn or raise a clean error."""
        if not column_name:
            return None
        if column_name not in adata_obj.obs.columns:
            available = list(adata_obj.obs.columns)
            if required:
                raise ValueError(
                    f"{context} '{column_name}' is not present in adata.obs. "
                    f"Available columns: {available}"
                )
            warnings.append(f"Ignored invalid {context} '{column_name}' because it is not present in adata.obs.")
            return None
        return column_name

    def _same_resolution(left, right) -> bool:
        if left is None or right is None:
            return False
        return abs(float(left) - float(right)) < 1e-9

    def _resolve_clustering_output_key(adata_obj, method: str, resolution: float, requested_key: str | None):
        normalized_method = "phenograph" if str(method).lower() == "phenograph" else "leiden"
        alias = default_cluster_key_for_method(normalized_method)
        if requested_key:
            return requested_key, requested_key == alias

        registry = {record["key"]: record for record in _clusterings_payload(adata_obj)}
        if alias not in adata_obj.obs:
            return alias, True

        alias_record = registry.get(alias)
        if alias_record and _same_resolution(alias_record.get("resolution"), resolution):
            return alias, True
        if alias_record is None and _same_resolution(resolution, 1.0):
            return alias, True
        return infer_cluster_key(normalized_method, resolution), False

    def _apply_clustering(
        adata_obj,
        *,
        method: str,
        resolution: float,
        cluster_key: str,
        make_primary: bool,
    ):
        normalized_method = "phenograph" if str(method).lower() == "phenograph" else "leiden"
        if normalized_method == "leiden":
            run_leiden(adata_obj, resolution=resolution, key_added=cluster_key)
        else:
            run_phenograph(adata_obj, resolution=resolution, key_added=cluster_key)

        register_clustering(
            adata_obj,
            cluster_key=cluster_key,
            method=normalized_method,
            resolution=resolution,
            created_by="tool",
        )
        primary_alias = default_cluster_key_for_method(normalized_method)
        primary_cluster_key = primary_alias if primary_alias in adata_obj.obs.columns else ""
        primary_alias_created = False
        if make_primary:
            primary_cluster_key = promote_clustering_to_primary(
                adata_obj,
                cluster_key=cluster_key,
                method=normalized_method,
                resolution=resolution,
                created_by="tool",
            )
            primary_alias_created = primary_cluster_key in adata_obj.obs.columns
        primary_alias_available = bool(primary_cluster_key and primary_cluster_key in adata_obj.obs.columns)
        created_obs_columns = [cluster_key]
        if primary_alias_created and primary_cluster_key != cluster_key:
            created_obs_columns.append(primary_cluster_key)

        sizes = adata_obj.obs[cluster_key].value_counts().to_dict()
        return {
            "cluster_key": cluster_key,
            "primary_alias": primary_alias,
            "primary_cluster_key": primary_cluster_key,
            "primary_alias_available": primary_alias_available,
            "primary_alias_created": primary_alias_created,
            "created_obs_columns": created_obs_columns,
            "method": normalized_method,
            "resolution": float(resolution),
            "n_clusters": len(sizes),
            "cluster_sizes": {str(key): int(value) for key, value in sizes.items()},
            "clusterings": _clusterings_payload(adata_obj),
        }

    def _render_figure(
        adata_obj,
        *,
        plot_type: str,
        output_path: str,
        color_by: str | None = None,
        genes=None,
        include_image: bool = True,
    ):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import scanpy as sc

        genes = genes or []
        if plot_type == "umap":
            if "X_umap" not in adata_obj.obsm:
                raise ValueError("UMAP embedding not found. Run run_pca, run_neighbors, and run_umap first.")
            if color_by in ("", None):
                color_by = None
            elif color_by not in adata_obj.obs.columns and color_by not in adata_obj.var_names:
                raise ValueError(f"'{color_by}' is not available for UMAP coloring.")

        # For large datasets, rasterized scatter is orders of magnitude faster than
        # vector rendering (the matplotlib default).  vector_friendly=False tells
        # scanpy to rasterize scatter points — identical PNG output, seconds not minutes.
        n_cells = adata_obj.n_obs
        large_dataset = n_cells > 50_000
        if large_dataset:
            sc.settings.set_figure_params(vector_friendly=False)
            dot_size = max(1, min(5, 120_000 // n_cells))
        else:
            dot_size = None

        fig, ax = plt.subplots(figsize=(10, 8))

        if plot_type == "umap":
            kwargs = dict(ax=ax, show=False)
            if dot_size is not None:
                kwargs["size"] = dot_size
            if color_by is None:
                sc.pl.umap(adata_obj, **kwargs)
            else:
                sc.pl.umap(adata_obj, color=color_by, **kwargs)
        elif plot_type == "violin":
            sc.pl.violin(adata_obj, keys=genes or [color_by], groupby=color_by, ax=ax, show=False)
        elif plot_type == "dotplot" and genes:
            sc.pl.dotplot(adata_obj, var_names=genes, groupby=color_by, show=False)
        elif plot_type == "heatmap" and genes:
            sc.pl.heatmap(adata_obj, var_names=genes, groupby=color_by, show=False)
        else:
            raise ValueError(f"Unsupported plot configuration: plot_type={plot_type}")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Restore default figure params so subsequent plots in the same session
        # are not affected.
        if large_dataset:
            sc.settings.set_figure_params(vector_friendly=True)

        result = {
            "status": "ok",
            "tool": "generate_figure",
            "output_path": output_path,
            "plot_type": plot_type,
            "color_by": color_by,
        }
        if include_image:
            try:
                result["image_base64"] = encode_image_base64(output_path)
                result["image_mime"] = get_image_mime_type(output_path)
            except Exception as enc_err:
                # Don't let an encoding failure blow up the whole tool call —
                # the figure is already written to disk and the path is in
                # the result. Log and continue without the inline blob.
                logger.warning(
                    "Failed to encode figure %s as base64 (%s); returning path only.",
                    output_path, enc_err,
                )
                result["image_encode_error"] = str(enc_err)
        return result

    def _stringify_dataframe_columns(df):
        if df is None:
            return df
        for col in df.columns:
            try:
                dtype_str = str(df[col].dtype)
            except Exception:
                dtype_str = ""
            if dtype_str in {"object", "category"}:
                df[col] = df[col].astype(str)
        return df

    def _sanitize_uns_value(value):
        import numpy as _np
        import pandas as _pd
        try:
            import scipy.sparse as _sp
        except Exception:
            _sp = None

        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (_np.integer, _np.floating, _np.bool_)):
            return value.item()
        if _sp is not None and _sp.issparse(value):
            return value.toarray().tolist()
        if isinstance(value, _np.ndarray):
            if value.dtype.names is not None:
                return {str(name): _sanitize_uns_value(value[name]) for name in value.dtype.names}
            if value.dtype.kind in "biufc":
                return value.tolist()
            if value.dtype.kind in "SU":
                return value.astype(str).tolist()
            return [_sanitize_uns_value(v) for v in value.tolist()]
        if isinstance(value, (_pd.Series, _pd.Index)):
            return [_sanitize_uns_value(v) for v in value.tolist()]
        if isinstance(value, _pd.DataFrame):
            safe_df = value.copy()
            _stringify_dataframe_columns(safe_df)
            return {str(col): [_sanitize_uns_value(v) for v in safe_df[col].tolist()] for col in safe_df.columns}
        if isinstance(value, dict):
            return {str(k): _sanitize_uns_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            sanitized_items = [_sanitize_uns_value(v) for v in value]
            normalized_items = []
            for item in sanitized_items:
                if isinstance(item, (dict, list, tuple, set)):
                    try:
                        normalized_items.append(json.dumps(item, default=str, sort_keys=True))
                    except Exception:
                        normalized_items.append(str(item))
                else:
                    normalized_items.append(item)
            return normalized_items
        return str(value)

    def _make_serializable_copy(current_adata, aggressive_uns: bool = False):
        sanitized = current_adata.copy()
        _stringify_dataframe_columns(sanitized.obs)
        _stringify_dataframe_columns(sanitized.var)
        if sanitized.raw is not None:
            _stringify_dataframe_columns(sanitized.raw.var)
        if aggressive_uns:
            sanitized.uns = {str(k): _sanitize_uns_value(v) for k, v in sanitized.uns.items()}
        return sanitized

    def write_h5ad_safe(current_adata, output_path: str) -> Dict[str, Any]:
        details = {"save_mode": "direct", "warnings": []}
        first_error_msg = None
        second_error_msg = None
        try:
            current_adata.write_h5ad(output_path)
            return details
        except Exception as first_error:
            first_error_msg = str(first_error)
            details["warnings"].append(f"Direct save failed; retrying with obs/var cleanup: {first_error_msg}")

        try:
            sanitized = _make_serializable_copy(current_adata, aggressive_uns=False)
            sanitized.write_h5ad(output_path)
            details["save_mode"] = "clean_obs_var"
            return details
        except Exception as second_error:
            second_error_msg = str(second_error)
            details["warnings"].append(f"Obs/var cleanup save failed; retrying with uns cleanup: {second_error_msg}")

        try:
            sanitized = _make_serializable_copy(current_adata, aggressive_uns=True)
            sanitized.write_h5ad(output_path)
            details["save_mode"] = "clean_obs_var_uns"
            return details
        except Exception as third_error:
            raise RuntimeError(
                "Unable to save AnnData after serialization cleanup. "
                f"Direct error: {first_error_msg}; obs/var cleanup error: {second_error_msg}; uns cleanup error: {third_error}"
            )

    def search_web(query: str, site: str = "", max_results: int = 5) -> Dict[str, Any]:
        """Search web/docs using Tavily (primary), DuckDuckGo (secondary), or Google CSE (last fallback)."""
        import os as os_module
        from urllib.parse import urlparse
        import requests

        scoped_query = f"site:{site} {query}" if site else query
        search_errors = []

        def normalize_domain(value: str) -> str:
            value = (value or "").strip().lower()
            if not value:
                return ""
            if "://" not in value:
                value = f"https://{value}"
            parsed = urlparse(value)
            return parsed.netloc.replace("www.", "")

        def extract_query_tokens(value: str) -> List[str]:
            raw_tokens = [tok.strip(" ,:;()[]{}").lower() for tok in value.split()]
            tokens = [tok for tok in raw_tokens if tok and tok not in {
                "documentation", "docs", "api", "function", "method", "tutorial", "guide",
                "site"
            }]
            return tokens

        query_tokens = extract_query_tokens(query)
        technical_tokens = [
            tok for tok in query_tokens
            if "_" in tok or "." in tok or any(ch.isdigit() for ch in tok) or len(tok) >= 8
        ]
        priority_tokens = technical_tokens or query_tokens[-2:]

        def score_result(item: Dict[str, Any], preferred_domain: str = "") -> tuple:
            domain = normalize_domain(item.get("url", ""))
            title = (item.get("title") or "").lower()
            snippet = (item.get("snippet") or "").lower()
            url = (item.get("url") or "").lower()
            domain_match = 1 if preferred_domain and preferred_domain in domain else 0
            exact_token_match = 0
            partial_token_match = 0
            api_page_bonus = 0

            for token in priority_tokens:
                if token and token in url:
                    exact_token_match += 1
                if token and token in title:
                    partial_token_match += 1

            api_indicators = [
                "/generated/",
                "/api/",
                "api.",
                "reference",
                "class",
                "function",
            ]
            if any(ind in url for ind in api_indicators):
                api_page_bonus = 1

            query_bonus = 0
            for token in query_tokens:
                if token in title:
                    query_bonus += 2
                elif token in snippet:
                    query_bonus += 1
            return (
                exact_token_match,
                domain_match,
                api_page_bonus,
                partial_token_match,
                query_bonus,
                len(snippet),
            )

        def run_ddg(search_query: str) -> List[Dict[str, Any]]:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(search_query, max_results=max_results))

            snippets = []
            for item in results:
                snippets.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", "") or item.get("url", ""),
                    "snippet": item.get("body", "")[:300],
                })
            return snippets

        def dedupe_and_rank(snippets: List[Dict[str, Any]], preferred_domain: str = "") -> List[Dict[str, Any]]:
            deduped: Dict[str, Dict[str, Any]] = {}
            for item in snippets:
                url = item.get("url", "")
                key = url or f"{item.get('title', '')}|{item.get('snippet', '')}"
                if key not in deduped:
                    deduped[key] = item
            ranked = sorted(
                deduped.values(),
                key=lambda item: score_result(item, preferred_domain=preferred_domain),
                reverse=True,
            )
            return ranked[:max_results]

        preferred_domain = normalize_domain(site)

        def run_tavily(search_query: str, include_domains: List[str]) -> List[Dict[str, Any]]:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_api_key,
                    "query": search_query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_domains": include_domains,
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            snippets = []
            for item in data.get("results", []):
                snippets.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "")[:300],
                })
            return snippets

        # === TAVILY (Primary - best for AI agents) ===
        tavily_api_key = os_module.environ.get("TAVILY_API_KEY")
        if tavily_api_key:
            try:
                snippets = []
                used_dual_query = False
                if site:
                    snippets.extend(run_tavily(query, [site]))
                    snippets.extend(run_tavily(query, []))
                    used_dual_query = True
                else:
                    snippets.extend(run_tavily(query, []))

                if snippets:
                    snippets = dedupe_and_rank(snippets, preferred_domain=preferred_domain)
                    return {
                        "status": "ok",
                        "backend": "tavily",
                        "query": query,
                        "results": snippets,
                        "used_dual_query": used_dual_query,
                        "site_filter_requested": bool(site),
                    }
                search_errors.append({"backend": "tavily", "type": "no_results"})
            except requests.HTTPError as e:
                response = getattr(e, "response", None)
                search_errors.append({
                    "backend": "tavily",
                    "type": "http_error",
                    "status_code": response.status_code if response is not None else None,
                    "message": response.text[:200] if response is not None and response.text else str(e),
                })
            except Exception as e:
                search_errors.append({"backend": "tavily", "type": type(e).__name__, "message": str(e)})

        # === DUCKDUCKGO (Secondary) ===
        try:
            ddg_queries = [scoped_query]
            if site:
                ddg_queries.append(query)
                ddg_queries.append(f"{query} {site}")

            snippets = []
            retried_without_site = False
            for idx, ddg_query in enumerate(ddg_queries):
                current = run_ddg(ddg_query)
                if idx > 0 and current:
                    retried_without_site = True
                snippets.extend(current)
                if len(snippets) >= max_results * 2:
                    break

            snippets = dedupe_and_rank(snippets, preferred_domain=preferred_domain)

            if snippets:
                return {
                    "status": "ok",
                    "backend": "duckduckgo",
                    "query": scoped_query,
                    "results": snippets,
                    "retried_without_site": retried_without_site,
                    "fallback_used": True,
                    "backends_tried": [e["backend"] for e in search_errors],
                    "errors": search_errors if search_errors else None,
                }
        except ImportError:
            search_errors.append({"backend": "duckduckgo", "type": "not_installed", "message": "duckduckgo-search not installed"})

        # === GOOGLE (Last fallback) ===
        google_api_key = os_module.environ.get("GOOGLE_API_KEY")
        google_cx = os_module.environ.get("GOOGLE_CX")

        if google_api_key and google_cx:
            try:
                url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    "key": google_api_key,
                    "cx": google_cx,
                    "q": scoped_query,
                    "num": max(1, min(max_results, 10)),
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                snippets = []
                for item in data.get("items", []):
                    snippets.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")[:300],
                    })

                if snippets:
                    snippets = dedupe_and_rank(snippets, preferred_domain=preferred_domain)
                    return {
                        "status": "ok",
                        "backend": "google",
                        "query": scoped_query,
                        "results": snippets,
                        "fallback_used": True,
                        "backends_tried": [e["backend"] for e in search_errors],
                        "errors": search_errors if search_errors else None,
                    }
                search_errors.append({"backend": "google", "type": "no_results"})
            except requests.HTTPError as e:
                response = getattr(e, "response", None)
                search_errors.append({
                    "backend": "google",
                    "type": "http_error",
                    "status_code": response.status_code if response is not None else None,
                    "message": response.text[:200] if response is not None and response.text else str(e),
                })
            except Exception as e:
                search_errors.append({"backend": "google", "type": type(e).__name__, "message": str(e)})

        if search_errors:
            return {
                "status": "warning",
                "query": scoped_query,
                "backend": "none",
                "results": [],
                "message": "No search backend returned results for this query.",
                "errors": search_errors,
            }

    def search_pubmed(query: str, max_results: int = 5, recent_years: int = 5, reviews_only: bool = False) -> List[Dict[str, Any]]:
        """Search PubMed and return structured article metadata."""
        import requests
        from datetime import datetime

        current_year = datetime.now().year
        min_date = f"{current_year - recent_years}/01/01"
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

        search_term = query
        if reviews_only:
            search_term += " AND review[pt]"

        search_params = {
            "db": "pubmed",
            "term": search_term,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "mindate": min_date,
            "maxdate": f"{current_year}/12/31",
            "datetype": "pdat",
        }

        try:
            resp = requests.get(esearch_url, params=search_params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            ids = data.get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            fetch_params = {
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "rettype": "abstract",
            }
            resp = requests.get(efetch_url, params=fetch_params, timeout=10)
            resp.raise_for_status()
            xml = resp.text

            def strip_tags(s: str) -> str:
                return re.sub(r"<[^>]+>", "", s).strip()

            results = []
            articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL)
            for article in articles:
                title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.DOTALL)
                pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
                if not (title_match and pmid_match):
                    continue

                # Abstract: join all AbstractText sections (handles structured abstracts)
                abstract_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
                abstract = " ".join(strip_tags(p) for p in abstract_parts)[:1000]

                pmid = pmid_match.group(1)
                title = strip_tags(title_match.group(1))

                year_match = re.search(r"<PubDate>.*?<Year>(\d+)</Year>", article, re.DOTALL)
                year = year_match.group(1) if year_match else "N/A"

                journal_match = re.search(r"<ISOAbbreviation>(.*?)</ISOAbbreviation>", article)
                if not journal_match:
                    journal_match = re.search(r"<Title>(.*?)</Title>", article)
                journal = strip_tags(journal_match.group(1)) if journal_match else "N/A"

                # First author
                last_name = re.search(r"<LastName>(.*?)</LastName>", article)
                first_name = re.search(r"<Initials>(.*?)</Initials>", article)
                first_author = ""
                if last_name:
                    first_author = strip_tags(last_name.group(1))
                    if first_name:
                        first_author += f" {strip_tags(first_name.group(1))}"

                # DOI
                doi_match = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', article)
                doi = strip_tags(doi_match.group(1)) if doi_match else None

                results.append({
                    "pmid": pmid,
                    "title": title,
                    "first_author": first_author,
                    "year": year,
                    "journal": journal,
                    "abstract": abstract,
                    "doi": doi,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })

            return results
        except Exception:
            return []

    def fetch_url_text(url: str, max_chars: int = 4000) -> Dict[str, Any]:
        """Fetch a URL and return a compact, structured text summary."""
        import html
        import io
        from urllib.parse import urlparse
        import requests

        headers = {
            "User-Agent": "scagent/0.1 (+single-cell analysis agent)"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        raw_bytes = resp.content
        text = resp.text
        title = ""
        meta_description = ""
        extracted_with = "plain_text"
        warning = None

        def clean_whitespace(value: str) -> str:
            return re.sub(r"\s+", " ", value).strip()

        def extract_html_text(html_text: str) -> Dict[str, Any]:
            nonlocal warning
            try:
                from bs4 import BeautifulSoup  # type: ignore
            except ImportError:
                BeautifulSoup = None

            if BeautifulSoup is not None:
                soup = BeautifulSoup(html_text, "html.parser")

                # Strip boilerplate before extraction
                for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                for tag in soup.find_all(attrs={"role": ["navigation", "banner", "complementary"]}):
                    tag.decompose()
                for tag in soup.find_all(class_=lambda c: c and any(
                    kw in (" ".join(c) if isinstance(c, list) else c)
                    for kw in ("sidebar", "toctree", "nav", "menu", "breadcrumb", "footer", "header")
                )):
                    tag.decompose()

                page_title = clean_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""
                meta = soup.find("meta", attrs={"name": "description"})
                meta_desc = clean_whitespace(meta.get("content", "")) if meta else ""

                # Prefer semantic containers (readthedocs uses .rst-content, sphinx uses .document)
                main = (
                    soup.find("main") or
                    soup.find("article") or
                    soup.find(class_=lambda c: c and any(
                        kw in (" ".join(c) if isinstance(c, list) else c)
                        for kw in ("rst-content", "document", "content", "body-content")
                    )) or
                    soup.body or soup
                )
                parts = []
                for tag in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "dt", "dd", "pre", "code"]):
                    text_part = clean_whitespace(tag.get_text(" ", strip=True))
                    if text_part and len(text_part) > 3:
                        parts.append(text_part)

                if not parts:
                    parts = [clean_whitespace(main.get_text(" ", strip=True))]

                return {
                    "title": page_title,
                    "meta_description": meta_desc,
                    "text": clean_whitespace(" ".join(parts)),
                    "extracted_with": "beautifulsoup4",
                }

            warning = "beautifulsoup4 not installed; used regex-based HTML extraction."
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
            meta_match = re.search(
                r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
                html_text,
                re.IGNORECASE | re.DOTALL,
            )
            page_title = html.unescape(clean_whitespace(title_match.group(1))) if title_match else ""
            meta_desc = html.unescape(clean_whitespace(meta_match.group(1))) if meta_match else ""
            cleaned = re.sub(r"<script.*?</script>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"<style.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"<noscript.*?</noscript>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r"<[^>]+>", " ", cleaned)
            cleaned = html.unescape(cleaned)
            cleaned = clean_whitespace(cleaned)
            return {
                "title": page_title,
                "meta_description": meta_desc,
                "text": cleaned,
                "extracted_with": "regex_html",
            }

        def extract_pdf_text(data: bytes) -> Dict[str, Any]:
            try:
                from pypdf import PdfReader  # type: ignore
            except ImportError:
                try:
                    from PyPDF2 import PdfReader  # type: ignore
                except ImportError as e:
                    raise ImportError("PDF reader dependency not installed") from e

            reader = PdfReader(io.BytesIO(data))
            pages = []
            for page in reader.pages[:5]:
                page_text = page.extract_text() or ""
                if page_text:
                    pages.append(clean_whitespace(page_text))

            return {
                "title": "",
                "meta_description": "",
                "text": clean_whitespace(" ".join(pages)),
                "extracted_with": "pdf_reader",
            }

        lower_content_type = content_type.lower()
        if "html" in lower_content_type:
            extracted = extract_html_text(text)
            title = extracted["title"]
            meta_description = extracted["meta_description"]
            cleaned = extracted["text"]
            extracted_with = extracted["extracted_with"]
        elif "pdf" in lower_content_type or urlparse(str(resp.url)).path.lower().endswith(".pdf"):
            try:
                extracted = extract_pdf_text(raw_bytes)
                title = extracted["title"]
                meta_description = extracted["meta_description"]
                cleaned = extracted["text"]
                extracted_with = extracted["extracted_with"]
            except ImportError:
                cleaned = ""
                extracted_with = "pdf_unsupported"
                warning = "PDF content fetched but no PDF text extraction library is installed."
        else:
            cleaned = clean_whitespace(text)

        result = {
            "status": "ok",
            "url": url,
            "final_url": str(resp.url),
            "content_type": content_type,
            "domain": urlparse(str(resp.url)).netloc,
            "title": title,
            "meta_description": meta_description,
            "extracted_with": extracted_with,
            "text": cleaned[:max_chars],
            "text_length": len(cleaned),
            "truncated": len(cleaned) > max_chars,
        }
        if warning:
            result["warning"] = warning
        if not cleaned:
            result["status"] = "warning"
            result["message"] = "Fetched the URL, but extracted little or no readable text."
        return result

    try:
        # ===== META TOOLS =====
        if tool_name == "ask_user":
            # This is handled specially by the agent loop - just return the question
            return json.dumps({
                "status": "needs_input",
                "tool": "ask_user",
                "question": tool_input["question"],
                "options": tool_input.get("options", []),
                "option_actions": tool_input.get("option_actions", []),
                "default": tool_input.get("default", ""),
                "decision_key": tool_input.get("decision_key", ""),
            }, indent=2), adata

        elif tool_name == "run_code":
            # Execute custom Python code on adata
            import scanpy as sc
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            code = tool_input["code"]
            description = tool_input["description"]
            save_to = tool_input.get("save_to")
            save_warning = None

            # Security: basic checks (not foolproof, but helps)
            # Note: "import os" is blocked but pathlib.Path is allowed in namespace
            forbidden = ["import os", "import sys", "subprocess", "eval(",
                        "__import__", "rm -rf", "shutil.rmtree", "requests.",
                        "os.system", "os.popen", "os.exec"]
            for f in forbidden:
                if f in code:
                    return _error_result(
                        tool="run_code",
                        message=f"Forbidden operation: {f}. Use Path from namespace for file operations.",
                        adata_obj=adata,
                        recovery_options=[
                            "Rewrite the code using the provided namespace (Path, ensure_dir, write_report).",
                            "For shell commands, use the run_shell tool instead.",
                        ],
                    )

            # Load data if needed
            if adata is None and "data_path" in tool_input:
                adata = get_adata(tool_input, adata)

            # Helper function for safe directory creation
            from pathlib import Path as _Path
            def ensure_dir(path):
                """Create directory if it doesn't exist and return it as a Path."""
                p = _Path(path)
                p.mkdir(parents=True, exist_ok=True)
                return p

            _run_dir = _Path(tool_input.get("output_dir", "."))

            def write_report(name: str, content: str) -> str:
                """Write a markdown report to reports/name.md and return the path.

                Always use this instead of open() when saving analysis results —
                it ensures reports land in the right directory as readable .md files.

                Example:
                    write_report('cluster_summary', '## Cluster Summary\\n\\n...')
                """
                reports_dir = ensure_dir(_run_dir / "reports")
                safe_name = name.replace(" ", "_").rstrip(".md")
                path = reports_dir / f"{safe_name}.md"
                path.write_text(content)
                return str(path)

            # Execute in controlled namespace
            # Note: Path and ensure_dir are provided - no need to import os
            namespace = {
                "adata": adata,
                "sc": sc,
                "np": np,
                "pd": pd,
                "plt": plt,
                "scanpy": sc,
                "matplotlib": matplotlib,
                "output_dir": tool_input.get("output_dir", "."),
                "Path": _Path,
                "ensure_dir": ensure_dir,
                "write_report": write_report,
            }

            # Capture stdout so LLM can see print outputs
            import io
            import sys
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout

            # Capture any figures created
            plt.close('all')

            import warnings as _warnings
            exec_error = None
            _caught = []
            try:
                sys.stdout = stdout_capture
                with _warnings.catch_warnings(record=True) as _caught:
                    _warnings.simplefilter("always")
                    exec(code, namespace)
            except Exception as _exec_err:
                exec_error = _exec_err
            finally:
                sys.stdout = old_stdout

            captured_output = stdout_capture.getvalue()

            # Append actionable warnings to captured output so the agent sees and acts on them.
            # Suppress purely cosmetic pandas FutureWarnings that require no action.
            _cosmetic = {
                "Series.__getitem__ treating keys as positions",
                "The default of observed=False",
            }
            for w in _caught:
                msg = str(w.message)
                if not any(c in msg for c in _cosmetic):
                    captured_output += f"\nWarning ({w.category.__name__}): {msg}"

            # After execution, ensure var_names and obs_names are unique on the live adata.
            # 10x h5 files can contain duplicate gene symbols; anndata.concat() propagates
            # them. Leaving duplicates causes silent wrong-gene indexing downstream.
            # obs_names duplicates arise when concat is called without keys/suffixes.
            var_names_fixed = False
            obs_names_fixed = False
            adata = namespace.get("adata", adata)
            if adata is not None and not adata.var_names.is_unique:
                adata.var_names_make_unique()
                var_names_fixed = True
            if adata is not None and adata.obs_names.duplicated().any():
                adata.obs_names_make_unique()
                obs_names_fixed = True

            if exec_error is not None:
                err_type = type(exec_error).__name__
                err_msg = str(exec_error)

                # Give the LLM targeted guidance based on the error type
                if err_type in ("TypeError", "AttributeError") or "unexpected keyword" in err_msg or "got an unexpected" in err_msg:
                    hint = "This looks like an API mismatch. Look up the function's documentation with web_search + fetch_url before retrying — do not guess."
                elif err_type in ("KeyError", "IndexError") or "not in" in err_msg:
                    hint = "This looks like a missing column or key. Check adata.obs.columns, adata.var.columns, or adata.obsm with inspect_data or a quick run_code before retrying."
                elif err_type in ("ModuleNotFoundError", "ImportError"):
                    hint = f"Package not installed. Use the install_package tool to request installation of the missing package."
                elif err_type == "SyntaxError":
                    hint = "Syntax error in the generated code — fix it directly."
                elif err_type == "NameError":
                    hint = "NameError — check that all variables used are defined in the namespace (adata, sc, np, pd, plt, Path, ensure_dir, write_report, output_dir)."
                else:
                    hint = "Diagnose the error before retrying: if it's an API issue look up docs; if it's a data issue inspect adata state."

                return json.dumps({
                    "status": "error",
                    "tool": "run_code",
                    "description": description,
                    "error_type": err_type,
                    "message": f"{err_type}: {err_msg}",
                    "output": captured_output[:500] if captured_output else None,
                    "recovery_options": [hint],
                }, indent=2), adata

            adata = namespace.get("adata", adata)
            custom_output_path = namespace.get("output_path")

            # Check if any figures were created
            figures_saved = []
            if plt.get_fignums():
                # There are open figures - check if code saved them
                pass

            if save_to:
                save_warning = "run_code ignored save_to; use save_data to save AnnData after custom code modifications"

            # Save code to file if output directory exists
            code_file = None
            if "output_dir" in tool_input:
                code_dir = os.path.join(tool_input["output_dir"], "code")
                os.makedirs(code_dir, exist_ok=True)

                # Create filename from description
                safe_desc = "".join(c if c.isalnum() or c in "_ " else "_" for c in description)
                safe_desc = safe_desc.replace(" ", "_")[:50]
                code_file = os.path.join(code_dir, f"{safe_desc}.py")

                with open(code_file, "w") as f:
                    f.write(f'"""\n{description}\n\nAuto-generated by scagent\n"""\n\n')
                    f.write("import scanpy as sc\n")
                    f.write("import numpy as np\n")
                    f.write("import pandas as pd\n")
                    f.write("import matplotlib.pyplot as plt\n\n")
                    f.write("# Load data (adjust path as needed)\n")
                    f.write("# adata = sc.read_h5ad('path/to/data.h5ad')\n\n")
                    f.write("# Generated code:\n")
                    f.write(code)

            result = {
                "status": "ok",
                "tool": "run_code",
                "description": description,
            }
            if adata is not None:
                result["shape"] = {"n_cells": adata.n_obs, "n_genes": adata.n_vars}
            if var_names_fixed:
                result.setdefault("auto_fixes", []).append("var_names had duplicates — called .var_names_make_unique()")
            if obs_names_fixed:
                result.setdefault("auto_fixes", []).append("obs_names had duplicates — called .obs_names_make_unique()")
            if save_to:
                result["ignored_save_to"] = save_to
            if save_warning:
                result.setdefault("warnings", []).append(save_warning)
            if code_file:
                result["code_file"] = code_file
            if custom_output_path:
                result["output_path"] = str(custom_output_path)
            if captured_output:
                # Truncate if too long
                result["output"] = captured_output[:2000]
                if len(captured_output) > 2000:
                    result["output_truncated"] = True

            return json.dumps(result, indent=2), adata

        elif tool_name in {"web_search", "web_search_docs"}:
            query = tool_input["query"]
            site = tool_input.get("site", "")
            max_results = tool_input.get("max_results", 5)
            search_result = search_web(query, site=site, max_results=max_results)
            search_result["tool"] = "web_search"
            return json.dumps(search_result, indent=2), adata

        elif tool_name == "search_papers":
            raw_query = tool_input["query"]
            max_results = tool_input.get("max_results", 5)
            recent_years = tool_input.get("recent_years", 5)
            reviews_only = tool_input.get("reviews_only", False)

            # Normalise GSEA gene set names: HALLMARK_TNFA_SIGNALING_VIA_NFKB → TNF alpha signaling NF-kB
            import re as _re
            query = _re.sub(r"^(HALLMARK|REACTOME|KEGG|GO|WP|BIOCARTA|PID|NABA)_", "", raw_query, flags=_re.IGNORECASE)
            query = query.replace("_", " ").strip()

            results = search_pubmed(
                query=query,
                max_results=max_results,
                recent_years=recent_years,
                reviews_only=reviews_only,
            )

            return json.dumps({
                "status": "ok",
                "tool": "search_papers",
                "query": query,
                "original_query": raw_query if raw_query != query else None,
                "reviews_only": reviews_only,
                "years_searched": f"last {recent_years} years",
                "count": len(results),
                "results": results,
            }, indent=2), adata

        elif tool_name == "research_findings":
            pathway = tool_input["pathway"]
            cell_type = tool_input.get("cell_type", "")
            genes = tool_input.get("genes", [])
            context = tool_input.get("context", "")
            recent_years = tool_input.get("recent_years", 3)

            # Build a focused query: pathway + cell type + top genes
            gene_str = " ".join(genes[:5]) if genes else ""
            query_parts = [p for p in [pathway, cell_type, gene_str, context] if p]
            query = " ".join(query_parts)

            # Normalise GSEA-style pathway names before searching
            import re as _re
            query = _re.sub(r"^(HALLMARK|REACTOME|KEGG|GO|WP|BIOCARTA|PID|NABA)_", "", query, flags=_re.IGNORECASE)
            query = query.replace("_", " ").strip()

            recent_papers = search_pubmed(query=query, max_results=5, recent_years=recent_years, reviews_only=False)
            reviews = search_pubmed(query=query, max_results=3, recent_years=recent_years, reviews_only=True)

            return json.dumps({
                "status": "ok",
                "tool": "research_findings",
                "pathway": pathway,
                "cell_type": cell_type,
                "query": query,
                "findings": {
                    "selected_papers": recent_papers,
                    "review_articles": reviews,
                },
            }, indent=2), adata

        elif tool_name == "fetch_url":
            url = tool_input["url"]
            max_chars = tool_input.get("max_chars", 4000)
            fetched = fetch_url_text(url, max_chars=max_chars)
            fetched["tool"] = "fetch_url"
            return json.dumps(fetched, indent=2), adata

        elif tool_name == "run_shell":
            import subprocess
            import shlex

            command = tool_input.get("command", "").strip()
            timeout = int(tool_input.get("timeout", 60))
            workdir = tool_input.get("workdir") or tool_input.get("output_dir") or "."

            # Block destructive patterns — focused on things that can't be undone
            _blocked = [
                ("rm -rf /", "recursive deletion of root"),
                ("rm -rf ~", "recursive deletion of home directory"),
                ("rm -rf $HOME", "recursive deletion of home directory"),
                ("> /dev/", "writing to device file"),
                ("dd if=/dev/zero of=/dev/", "disk overwrite"),
                ("mkfs", "filesystem formatting"),
                (":(){ :|:& };:", "fork bomb"),
                ("sudo rm", "privileged deletion"),
                ("chmod -R 777 /", "global permission change"),
            ]
            for pattern, reason in _blocked:
                if pattern in command:
                    return _error_result(
                        tool="run_shell",
                        message=f"Blocked: {reason}.",
                        adata_obj=adata,
                        recovery_options=["Rewrite the command to avoid destructive operations."],
                        extra={"command": command},
                    )

            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=workdir,
                )
                stdout = proc.stdout.strip()
                stderr = proc.stderr.strip()
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                return _error_result(
                    tool="run_shell",
                    message=f"Command timed out after {timeout}s.",
                    adata_obj=adata,
                    recovery_options=[
                        f"Increase the timeout (currently {timeout}s).",
                        "Break the command into smaller steps.",
                    ],
                    extra={"command": command},
                )
            except Exception as e:
                return _error_result(
                    tool="run_shell",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=["Check command syntax and that required tools are installed."],
                    extra={"command": command},
                )

            return json.dumps({
                "status": "ok" if returncode == 0 else "error",
                "tool": "run_shell",
                "command": command,
                "returncode": returncode,
                "stdout": stdout[:8000] if stdout else "",
                "stderr": stderr[:2000] if stderr else "",
                "truncated": len(stdout) > 8000,
            }, indent=2), adata

        elif tool_name == "install_package":
            # Request package installation - requires user approval
            package = tool_input["package"]
            reason = tool_input["reason"]

            return json.dumps({
                "status": "needs_approval",
                "tool": "install_package",
                "package": package,
                "reason": reason,
                "message": f"Agent wants to install '{package}': {reason}"
            }, indent=2), adata

        # ===== INSPECTION TOOLS =====
        elif tool_name == "inspect_data":
            # update_memory=True only when there is no primary loaded yet.
            # When primary is already in memory, data_path loads a copy for
            # inspection only — the primary is NOT replaced.
            should_update = tool_input.get("data_path") is not None and adata is None
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=should_update)
            state = inspect_data(working_adata)
            batch_resolution = resolve_batch_metadata(working_adata)
            goal = tool_input.get("goal")
            context_hint = tool_input.get("context", "")
            guidance_context = context_hint
            if tool_input.get("data_path"):
                context_hint = " ".join(part for part in [context_hint, str(tool_input.get("data_path"))] if part)
            confirmed_batch_key = _confirmed_decision_value("batch_key")
            guidance = _analysis_guidance(state, goal=goal, context=guidance_context)

            raw_info = {"adata_raw": None, "layers": []}
            if state.has_raw:
                raw_info["adata_raw"] = {
                    "n_vars": state.raw_n_vars,
                    "note": "full gene set before HVG subsetting" if state.raw_n_vars > state.n_genes else "same gene set as X",
                }
            if state.has_raw_layer:
                raw_info["layers"].append(state.raw_layer_name)

            from ..core.inspector import _characterize_features
            feature_info = _characterize_features(working_adata)
            result = {
                "status": "ok",
                "tool": "inspect_data",
                "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
                "data_type": state.data_type,
                "raw": raw_info,
                "state": make_state(working_adata),
                "embeddings": [k for k in working_adata.obsm.keys()],
                "layers": list(working_adata.layers.keys()),
                "genes": {
                    "format": state.gene_id_format,
                    "has_symbols": state.has_gene_symbols,
                    "has_ensembl": state.has_ensembl_ids,
                    "sample": feature_info["sample_gene_names"],
                    "var_columns": list(working_adata.var.columns)[:10],
                    "genome_prefix": feature_info["genome_prefix"],
                    "special_gene_populations": feature_info["special_gene_populations"],
                    "mt_genes_detected": feature_info["mt_genes_detected"],
                    "mt_gene_examples": feature_info["mt_gene_examples"],
                    "ribo_genes_detected": feature_info["ribo_genes_detected"],
                    "ribo_gene_examples": feature_info["ribo_gene_examples"],
                },
                "obs_names": {
                    "format": feature_info["obs_names_format"],
                    "sample": feature_info["obs_names_sample"],
                    "suffixes_detected": feature_info["obs_names_suffixes_detected"],
                },
                "clustering": {
                    "has_clusters": state.has_clusters,
                    "cluster_key": state.cluster_key,
                    "n_clusters": state.n_clusters,
                    "available_clusterings": _clusterings_payload(working_adata),
                },
                "annotations": {
                    "has_celltypes": state.has_celltype_annotations,
                    "cell_type_key": state.cell_type_key,
                    "cell_type_candidates": [
                        metadata_candidate_to_dict(candidate)
                        for candidate in state.cell_type_candidates
                    ],
                    "sources": [
                        source
                        for source, present in (
                            ("celltypist", state.has_celltypist),
                            ("scimilarity", state.has_scimilarity),
                            (
                                "external_or_manual",
                                bool(state.cell_type_candidates)
                                and not (state.has_celltypist or state.has_scimilarity),
                            ),
                        )
                        if present
                    ],
                },
                "batch": {
                    "confirmed_batch_key": confirmed_batch_key,
                    "inferred_batch_key": state.batch_key,
                    "n_batches": state.n_batches,
                    "status": batch_resolution.status,
                    "recommended_batch_key": batch_resolution.recommended_column,
                    "recommended_role": batch_resolution.recommended_role,
                    "needs_confirmation": batch_resolution.needs_user_confirmation,
                    "reason": batch_resolution.reason,
                    "relevance": "current" if guidance["batch_relevant_now"] else "later_optional",
                    "candidates": [
                        metadata_candidate_to_dict(candidate)
                        for candidate in state.metadata_candidates
                    ],
                },
                "metadata_candidates": [
                    metadata_candidate_to_dict(candidate)
                    for candidate in state.metadata_candidates
                ],
                "batch_key": confirmed_batch_key,
                "recommended_batch_key": batch_resolution.recommended_column,
                "available_clusterings": _clusterings_payload(working_adata),
                "analysis_guidance": guidance,
                "obs_names_sample": working_adata.obs_names[:10].tolist(),
                "var_names_sample": working_adata.var_names[:10].tolist(),
                "obs_preview": _dataframe_preview(working_adata.obs),
                "var_preview": _dataframe_preview(working_adata.var),
                "obs_columns_detail": _obs_columns_detail(working_adata.obs, working_adata.n_obs),
            }
            if goal:
                result["recommended_steps"] = recommend_next_steps(state, goal)
            decisions = []
            batch_decision = decision_for_batch_strategy(
                metadata_resolution_to_dict(batch_resolution),
                context="inspect_data",
                source_tool="inspect_data",
                batch_relevant=guidance["batch_relevant_now"],
            )
            if batch_decision is not None:
                decisions.append(batch_decision)

            return _finalize_result(
                result,
                updated_adata,
                dataset_changed=False,
                summary="Inspected the active AnnData state and collaborative metadata candidates.",
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "inspect_data returned a coherent dataset summary.",
                    [
                        _check("shape_available", "shape" in result, "Dataset shape is present."),
                        _check(
                            "batch_section_present",
                            "batch" in result,
                            "Batch metadata summary is present.",
                        ),
                    ],
                ),
            )

        elif tool_name == "inspect_session":
            include_history = bool(tool_input.get("include_history", True))
            if world_state is None:
                session_payload = {
                    "status": "ok",
                    "tool": "inspect_session",
                    "message": "No active AgentWorldState was provided. Falling back to the current AnnData state only.",
                    "world_state": {
                        "analysis_stage": _stage_from_state(starting_state),
                        "data_summary": {"state": starting_state},
                    },
                }
                return _finalize_result(
                    session_payload,
                    adata,
                    dataset_changed=False,
                    summary="Inspected a minimal session fallback because no AgentWorldState was available.",
                )

            snapshot = world_state.snapshot()
            if not include_history:
                snapshot.pop("resolved_decisions", None)
                snapshot.pop("artifacts", None)

            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "inspect_session",
                    "world_state": snapshot,
                },
                adata,
                dataset_changed=False,
                summary="Inspected the unified agent session state.",
                verification=_build_verification(
                    "passed",
                    "Session state was available for inspection.",
                    [
                        _check(
                            "analysis_stage_present",
                            bool(snapshot.get("analysis_stage")),
                            "Session snapshot includes an analysis stage.",
                        ),
                    ],
                ),
            )

        elif tool_name == "list_artifacts":
            limit = int(tool_input.get("limit", 20))
            artifact_kind = tool_input.get("artifact_kind")
            artifacts: List[Dict[str, Any]] = []

            if world_state is not None:
                artifacts = [artifact.to_dict() for artifact in world_state.artifacts]
            elif run_manager is not None:
                artifacts = list(run_manager.manifest.artifact_registry)
            else:
                run_path = tool_input.get("run_path")
                if run_path:
                    manifest_path = Path(run_path)
                    if manifest_path.is_dir():
                        manifest_path = manifest_path / "manifest.json"
                    if not manifest_path.exists():
                        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
                    with open(manifest_path) as handle:
                        manifest_payload = json.load(handle)
                    artifacts = manifest_payload.get("artifact_registry", [])

            if artifact_kind:
                artifacts = [artifact for artifact in artifacts if artifact.get("kind") == artifact_kind]

            artifacts = artifacts[-limit:]
            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "list_artifacts",
                    "artifacts": artifacts,
                    "n_artifacts": len(artifacts),
                },
                adata,
                dataset_changed=False,
                summary="Listed known session or run artifacts.",
            )

        elif tool_name == "get_cluster_sizes":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            key = tool_input.get("cluster_key", "leiden")
            if key not in working_adata.obs:
                return _error_result(
                    tool="get_cluster_sizes",
                    message=f"No cluster column '{key}'.",
                    adata_obj=working_adata,
                    recovery_options=[
                        "Run clustering first, then request cluster sizes.",
                        "Use list_obs_columns to find the correct cluster key.",
                    ],
                )

            sizes = working_adata.obs[key].value_counts().to_dict()
            return json.dumps({
                "status": "ok",
                "tool": "get_cluster_sizes",
                "cluster_key": key,
                "n_clusters": len(sizes),
                "sizes": {str(k): int(v) for k, v in sizes.items()}
            }, indent=2), updated_adata

        elif tool_name == "get_top_markers":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            cluster = tool_input["cluster"]
            n_genes = tool_input.get("n_genes", 10)
            key = tool_input.get("key", "rank_genes_groups")

            if key not in working_adata.uns:
                return _smart_unavailable_result(
                    tool="get_top_markers",
                    message="Top markers are not available because differential expression has not been run yet.",
                    adata_obj=updated_adata,
                    missing_prerequisites=["deg"],
                    recovery_options=[
                        "Run differential expression on the current clustering first.",
                        "Inspect available clusterings before choosing a DEG grouping.",
                    ],
                    extra={"cluster": cluster, "key": key},
                )

            markers_df = get_top_markers(working_adata, group=cluster, n_genes=n_genes, key=key)
            markers = markers_df[['names', 'scores', 'logfoldchanges', 'pvals_adj']].to_dict('records')

            return json.dumps({
                "status": "ok",
                "tool": "get_top_markers",
                "cluster": cluster,
                "key": key,
                "markers": markers
            }, indent=2), updated_adata

        elif tool_name == "summarize_qc_metrics":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)

            import pandas as _pd

            metrics = {}
            obs = working_adata.obs

            # Map canonical role names → columns. Check exact names first, then
            # fall back to pattern matching on all numeric obs columns.
            _role_patterns = {
                "qc_total_counts":  ("total_counts", ["total_count", "n_counts", "sum_counts"]),
                "qc_n_genes":       ("n_genes_by_counts", ["n_genes", "ngenes", "num_genes", "n_features"]),
                "qc_pct_mt":        ("pct_counts_mt", ["pct_mt", "percent_mt", "mito_pct", "pct_mito", "mt_pct"]),
                "qc_pct_ribo":      ("pct_counts_ribo", ["pct_ribo", "percent_ribo", "ribo_pct"]),
                "doublet_score":    ("doublet_score", ["scrublet_score", "doublet_prob", "dbl_score"]),
            }
            seen = set()
            for role, (canonical, aliases) in _role_patterns.items():
                col = None
                if canonical in obs:
                    col = canonical
                else:
                    for alias in aliases:
                        if alias in obs:
                            col = alias
                            break
                    if col is None:
                        lower_cols = {c.lower(): c for c in obs.columns}
                        for alias in [canonical] + aliases:
                            if alias.lower() in lower_cols:
                                col = lower_cols[alias.lower()]
                                break
                if col and col not in seen:
                    seen.add(col)
                    values = obs[col]
                    if _pd.api.types.is_numeric_dtype(values):
                        metrics[role] = {
                            "column": col,
                            "median": float(values.median()),
                            "mean": float(values.mean()),
                            "min": float(values.min()),
                            "max": float(values.max()),
                        }

            # Doublet label column
            _doublet_label_names = [
                "predicted_doublet", "is_doublet", "doublet_label",
                "doublet", "scrublet_doublet", "dbl_label",
            ]
            doublet_label = next(
                (c for c in _doublet_label_names if c in obs), None
            )
            if doublet_label is None:
                lower_cols = {c.lower(): c for c in obs.columns}
                for name in _doublet_label_names:
                    if name.lower() in lower_cols:
                        doublet_label = lower_cols[name.lower()]
                        break

            doublet_info = {}
            if doublet_label and doublet_label in obs:
                labels = working_adata.obs[doublet_label]
                if labels.dtype == bool:
                    positive = labels
                else:
                    positive = labels.astype(str).str.lower().isin(
                        {"true", "1", "doublet", "multiplet", "positive"}
                    )
                doublet_info = {
                    "column": doublet_label,
                    "n_doublets": int(positive.sum()),
                    "doublet_rate": float(positive.mean())
                }

            return json.dumps({
                "status": "ok",
                "tool": "summarize_qc_metrics",
                "n_cells": working_adata.n_obs,
                "metrics": metrics,
                "doublets": doublet_info
            }, indent=2), updated_adata

        elif tool_name == "get_celltypes":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)

            # Find annotation column
            key = tool_input.get("annotation_key")
            if not key:
                # Use obs_columns_detail from world state if available, otherwise compute
                ocd = (
                    (world_state.data_summary.get("obs_columns_detail") or {})
                    if world_state is not None
                    else {}
                )
                if not ocd:
                    ocd = _obs_columns_detail(working_adata.obs, working_adata.n_obs).get("columns", {})
                # Candidate columns: categorical, n_unique 2-300, not flagged high_cardinality
                candidates = [
                    col for col, info in ocd.items()
                    if info.get("note") != "high_cardinality"
                    and 2 <= info.get("n_unique", 0) <= 300
                    and info.get("dtype") in ("object", "category")
                ]
                if len(candidates) == 1:
                    key = candidates[0]
                elif candidates:
                    return _finalize_result(
                        {
                            "status": "needs_choice",
                            "tool": "get_celltypes",
                            "message": (
                                "Multiple obs columns could be cell type annotations. "
                                "Identify the right one from obs_columns_detail and call "
                                "get_celltypes with annotation_key."
                            ),
                            "obs_columns_detail": {c: ocd[c] for c in candidates},
                            "recovery_options": [
                                "Call get_celltypes again with annotation_key set to the intended column.",
                                "Ask the user which column contains the cell type labels.",
                            ],
                        },
                        updated_adata,
                        dataset_changed=False,
                        summary="Cell type annotation column needs disambiguation.",
                    )

            if not key or key not in working_adata.obs:
                ocd = _obs_columns_detail(working_adata.obs, working_adata.n_obs).get("columns", {})
                return _smart_unavailable_result(
                    tool="get_celltypes",
                    message="No cell type annotations found. Check obs_columns_detail for available columns.",
                    adata_obj=updated_adata,
                    missing_prerequisites=["annotation"],
                    recovery_options=[
                        "Run cell type annotation on the current clustering.",
                        "Call get_celltypes with annotation_key set to the correct obs column.",
                    ],
                    extra={"obs_columns_detail": ocd},
                )

            counts = working_adata.obs[key].value_counts()
            total_cells = working_adata.n_obs

            # Build detailed breakdown with percentages
            breakdown = {}
            for ct, count in counts.items():
                breakdown[str(ct)] = {
                    "count": int(count),
                    "percent": round(100.0 * count / total_cells, 1)
                }

            # Group by major categories if there are many types
            major_types = {}
            if len(counts) > 5:
                for ct in counts.head(10).index:
                    major_types[str(ct)] = {
                        "count": int(counts[ct]),
                        "percent": round(100.0 * counts[ct] / total_cells, 1)
                    }

            return json.dumps({
                "status": "ok",
                "tool": "get_celltypes",
                "annotation_key": key,
                "total_cells": total_cells,
                "n_types": len(counts),
                "top_10_types": major_types if major_types else breakdown,
                "all_types": breakdown
            }, indent=2), updated_adata

        elif tool_name == "list_obs_columns":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            return json.dumps({
                "status": "ok",
                "tool": "list_obs_columns",
                "n_columns": len(working_adata.obs.columns),
                "obs_columns_detail": _obs_columns_detail(working_adata.obs, working_adata.n_obs),
            }, indent=2), updated_adata

        elif tool_name in {"review_figure", "review_artifact"}:
            artifact_path = tool_input.get("artifact_path") or tool_input.get("figure_path")
            artifact_id = tool_input.get("artifact_id")
            include_image = bool(tool_input.get("include_image", True))
            question = tool_input.get("question", "")
            max_chars = int(tool_input.get("max_chars", 4000))

            if not artifact_path and artifact_id and world_state is not None:
                matched = next(
                    (artifact for artifact in world_state.artifacts if artifact.artifact_id == artifact_id),
                    None,
                )
                if matched is not None:
                    artifact_path = matched.path

            if not artifact_path:
                raise ValueError("Provide artifact_path/figure_path or artifact_id to review an artifact.")

            artifact_path = os.path.abspath(artifact_path)
            if not os.path.exists(artifact_path):
                raise FileNotFoundError(f"Artifact not found: {artifact_path}")

            artifact_kind = _artifact_kind_from_path(artifact_path)
            result = {
                "status": "ok",
                "tool": tool_name,
                "artifact_path": artifact_path,
                "artifact_kind": artifact_kind,
                "question": question,
                "size_bytes": os.path.getsize(artifact_path),
            }
            if tool_name == "review_figure":
                result["figure_path"] = artifact_path

            if artifact_kind == "figure":
                if include_image:
                    try:
                        result["image_base64"] = encode_image_base64(artifact_path)
                        result["image_mime"] = get_image_mime_type(artifact_path)
                    except Exception as enc_err:
                        logger.warning(
                            "Failed to encode artifact %s as base64 (%s); returning path only.",
                            artifact_path, enc_err,
                        )
                        result["image_encode_error"] = str(enc_err)
            elif artifact_kind == "json":
                with open(artifact_path) as handle:
                    payload = json.load(handle)
                excerpt = json.dumps(payload, indent=2)[:max_chars]
                result["content_excerpt"] = excerpt
                result["json_keys"] = list(payload.keys())[:25] if isinstance(payload, dict) else []
            elif artifact_kind in {"report", "log"}:
                with open(artifact_path) as handle:
                    content = handle.read()
                result["content_excerpt"] = content[:max_chars]
                result["truncated"] = len(content) > max_chars
            elif artifact_kind == "data":
                reviewed_adata = load_data(artifact_path)
                reviewed_state = inspect_data(reviewed_adata)
                result["data_summary"] = {
                    "shape": {"n_cells": reviewed_state.n_cells, "n_genes": reviewed_state.n_genes},
                    "data_type": reviewed_state.data_type,
                    "batch_key": reviewed_state.batch_key,
                    "cluster_key": reviewed_state.cluster_key,
                    "n_clusters": reviewed_state.n_clusters,
                }

            verification_checks = [
                _check("artifact_exists", os.path.exists(artifact_path), f"Artifact exists at {artifact_path}."),
            ]
            if artifact_kind == "figure" and include_image:
                verification_checks.append(
                    _check(
                        "image_payload_attached",
                        "image_base64" in result,
                        "Image payload attached for model review.",
                    )
                )
            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary=f"Reviewed existing {artifact_kind} artifact.",
                verification=_build_verification(
                    "passed",
                    f"{artifact_kind.title()} artifact was available for review.",
                    verification_checks,
                ),
            )

        elif tool_name == "inspect_run_state":
            include_history = bool(tool_input.get("include_history", True))
            manifest_payload = None
            if run_manager is not None:
                manifest_payload = run_manager.manifest.to_dict()
            else:
                run_path = tool_input.get("run_path")
                if not run_path:
                    raise ValueError("Provide run_path when no active run manager is available.")
                manifest_path = Path(run_path)
                if manifest_path.is_dir():
                    manifest_path = manifest_path / "manifest.json"
                if not manifest_path.exists():
                    raise FileNotFoundError(f"Run manifest not found: {manifest_path}")
                with open(manifest_path) as handle:
                    manifest_payload = json.load(handle)

            result = {
                "status": "ok",
                "tool": "inspect_run_state",
                "run_id": manifest_payload.get("run_id"),
                "run_status": manifest_payload.get("status"),
                "request": manifest_payload.get("request"),
                "n_steps": len(manifest_payload.get("steps_completed", [])),
                "n_artifacts": len(manifest_payload.get("artifact_registry", [])),
                "n_decisions": len(manifest_payload.get("user_decisions", [])),
                "n_verifications": len(manifest_payload.get("verification_history", [])),
                "latest_world_state": (manifest_payload.get("world_state_snapshots", []) or [{}])[-1],
            }
            if include_history:
                result["recent_events"] = manifest_payload.get("session_events", [])[-10:]
                result["recent_steps"] = manifest_payload.get("steps_completed", [])[-10:]

            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary="Inspected the active or persisted run ledger.",
            )

        elif tool_name == "inspect_workspace":
            workspace_root = Path.cwd().resolve()
            allowed_roots = [workspace_root]
            if run_manager is not None:
                allowed_roots.append(Path(run_manager.run_dir).resolve())

            requested = tool_input.get("path", ".")
            requested_path = Path(requested)
            if not requested_path.is_absolute():
                requested_path = (workspace_root / requested_path).resolve()
            else:
                requested_path = requested_path.resolve()

            if not any(
                requested_path == root or root in requested_path.parents
                for root in allowed_roots
            ):
                raise ValueError(
                    f"inspect_workspace is limited to the project workspace and active run directory. "
                    f"Requested path: {requested_path}"
                )

            max_depth = int(tool_input.get("max_depth", 2))
            limit = int(tool_input.get("limit", 50))
            entries: List[Dict[str, Any]] = []

            def _walk(path_obj: Path, depth: int) -> None:
                if len(entries) >= limit:
                    return
                if path_obj.is_file():
                    entries.append(
                        {
                            "path": str(path_obj),
                            "kind": "file",
                            "size_bytes": path_obj.stat().st_size,
                        }
                    )
                    return
                if not path_obj.is_dir() or depth > max_depth:
                    return

                for child in sorted(path_obj.iterdir(), key=lambda candidate: candidate.name):
                    if len(entries) >= limit:
                        break
                    if child.name == "__pycache__":
                        continue
                    entries.append(
                        {
                            "path": str(child),
                            "kind": "directory" if child.is_dir() else "file",
                            "size_bytes": child.stat().st_size if child.is_file() else None,
                            "depth": depth,
                        }
                    )
                    if child.is_dir():
                        _walk(child, depth + 1)

            _walk(requested_path, 0)
            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "inspect_workspace",
                    "path": str(requested_path),
                    "entries": entries,
                    "n_entries": len(entries),
                },
                adata,
                dataset_changed=False,
                summary="Inspected the workspace in read-only mode.",
            )

        # ===== ACTION TOOLS =====
        elif tool_name == "load_data":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=True)
            if working_adata is None:
                return _error_result(tool="load_data", message="data_path is required", suggestions=["Provide a valid path to an h5ad or 10X h5 file."])
            state = inspect_data(working_adata)
            batch_resolution = resolve_batch_metadata(working_adata)
            from ..core.inspector import _characterize_features
            feature_info = _characterize_features(working_adata)
            ocd = _obs_columns_detail(working_adata.obs, working_adata.n_obs)
            result = {
                "status": "ok",
                "tool": "load_data",
                "loaded": tool_input.get("data_path"),
                "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
                "data_type": state.data_type,
                "state": make_state(working_adata),
                "embeddings": [k for k in working_adata.obsm.keys()],
                "layers": list(working_adata.layers.keys()),
                "genes": {
                    "format": state.gene_id_format,
                    "has_symbols": state.has_gene_symbols,
                    "sample": feature_info["sample_gene_names"],
                    "mt_genes_detected": feature_info["mt_genes_detected"],
                },
                "obs_columns_detail": ocd,
                "batch_metadata": {
                    "status": batch_resolution.status,
                    "recommended_batch_key": batch_resolution.recommended_column,
                    "recommended_role": batch_resolution.recommended_role,
                    "needs_confirmation": batch_resolution.needs_user_confirmation,
                    "reason": batch_resolution.reason,
                    "candidates": [metadata_candidate_to_dict(c) for c in (batch_resolution.candidates or [])],
                },
            }
            return json.dumps(result, indent=2), updated_adata

        elif tool_name == "run_qc":
            import pandas as pd

            warnings = []
            if adata is not None and tool_input.get("data_path") not in (None, "memory"):
                warnings.append(
                    "Ignored data_path and continued with the in-memory dataset to preserve prior analysis state."
                )

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_before, g_before = adata.n_obs, adata.n_vars

            detect_doublets_flag = tool_input.get("detect_doublets_flag", True)
            remove_ribo = tool_input.get("remove_ribo", False)
            remove_mt = tool_input.get("remove_mt", False)
            remove_doublets = bool(tool_input.get("remove_doublets", False))
            filter_mt = bool(tool_input.get("filter_mt", True))
            min_genes = tool_input.get("min_genes")
            min_cells = tool_input.get("min_cells", 3)
            requested_mt_threshold = tool_input.get("mt_threshold")
            preview_only = bool(tool_input.get("preview_only", False))
            confirm_filtering = bool(tool_input.get("confirm_filtering", False))
            scrublet_expected_doublet_rate = float(tool_input.get("scrublet_expected_doublet_rate") or 0.06)
            scrublet_sim_doublet_ratio = float(tool_input.get("scrublet_sim_doublet_ratio") or 2.0)
            scrublet_n_prin_comps = int(tool_input.get("scrublet_n_prin_comps") or 30)
            scrublet_min_counts = int(tool_input.get("scrublet_min_counts") or 2)
            scrublet_min_cells = int(tool_input.get("scrublet_min_cells") or 3)
            scrublet_min_gene_variability_pctl = float(tool_input.get("scrublet_min_gene_variability_pctl") or 85.0)
            scrublet_random_state = int(tool_input.get("scrublet_random_state") or 0)
            force_doublet_recompute = bool(tool_input.get("force_doublet_recompute", False))
            requested_scrublet_params = {
                "expected_doublet_rate": scrublet_expected_doublet_rate,
                "sim_doublet_ratio": scrublet_sim_doublet_ratio,
                "n_prin_comps": scrublet_n_prin_comps,
                "min_counts": scrublet_min_counts,
                "min_cells": scrublet_min_cells,
                "min_gene_variability_pctl": scrublet_min_gene_variability_pctl,
                "random_state": scrublet_random_state,
                "force_recompute": force_doublet_recompute,
            }
            requested_batch_key = tool_input.get("batch_key") or _confirmed_decision_value("batch_key")
            if requested_batch_key and "batch_key" not in tool_input and _confirmed_decision_value("batch_key"):
                warnings.append(
                    f"Using previously confirmed batch_key '{requested_batch_key}' from session state."
                )

            if detect_doublets_flag:
                batch_resolution = resolve_batch_metadata(
                    adata,
                    requested_column=requested_batch_key,
                )
            else:
                batch_resolution = None

            batch_key = batch_resolution.applied_column if batch_resolution else None
            if batch_resolution:
                if batch_resolution.status == "auto_selected" and batch_key:
                    warnings.append(batch_resolution.reason)
                elif batch_resolution.status == "invalid_requested":
                    warnings.append(batch_resolution.reason)
                elif batch_resolution.status == "needs_confirmation":
                    warnings.append(
                        f"{batch_resolution.reason} "
                        f"{'Previewing' if preview_only else 'Running'} doublets without per-batch stratification for now."
                    )
                elif batch_resolution.status == "no_candidate":
                    warnings.append(batch_resolution.reason)

            # Compute QC metrics directly on adata — these are non-destructive obs/var
            # annotations (pct_counts_mt, doublet_score, etc.). Writing to adata here
            # (not a throwaway copy) means apply mode can reuse them without recomputing.
            # Actual cell/gene filtering only happens later if not preview_only.
            _metrics_precomputed = 'pct_counts_mt' in adata.obs.columns
            # Never recompute doublets during the filtering step — scores from preview are authoritative.
            _effective_force_recompute = force_doublet_recompute and not confirm_filtering
            _doublets_precomputed = (
                detect_doublets_flag
                and 'predicted_doublet' in adata.obs.columns
                and not _effective_force_recompute
            )
            doublet_predictions_source = (
                "precomputed"
                if _doublets_precomputed
                else ("computed" if detect_doublets_flag else "not_run")
            )
            stored_scrublet_params = (
                adata.uns.get("scrublet_params", {})
                if _doublets_precomputed and isinstance(adata.uns.get("scrublet_params", {}), dict)
                else {}
            )
            scrublet_params_for_report = requested_scrublet_params.copy()
            if stored_scrublet_params:
                scrublet_params_for_report = {
                    "expected_doublet_rate": float(stored_scrublet_params.get("expected_doublet_rate", scrublet_expected_doublet_rate)),
                    "sim_doublet_ratio": float(stored_scrublet_params.get("sim_doublet_ratio", scrublet_sim_doublet_ratio)),
                    "n_prin_comps": int(stored_scrublet_params.get("n_prin_comps", scrublet_n_prin_comps)),
                    "n_prin_comps_used": int(stored_scrublet_params.get("n_prin_comps_used", stored_scrublet_params.get("n_prin_comps", scrublet_n_prin_comps))),
                    "min_counts": int(stored_scrublet_params.get("min_counts", scrublet_min_counts)),
                    "min_cells": int(stored_scrublet_params.get("min_cells", scrublet_min_cells)),
                    "min_gene_variability_pctl": float(stored_scrublet_params.get("min_gene_variability_pctl", scrublet_min_gene_variability_pctl)),
                    "random_state": int(stored_scrublet_params.get("random_state", scrublet_random_state)),
                    "force_recompute": force_doublet_recompute,
                    "batch_key": stored_scrublet_params.get("batch_key", batch_key),
                }
                changed_requested_params = [
                    key for key in (
                        "scrublet_expected_doublet_rate",
                        "scrublet_sim_doublet_ratio",
                        "scrublet_n_prin_comps",
                        "scrublet_min_counts",
                        "scrublet_min_cells",
                        "scrublet_min_gene_variability_pctl",
                        "scrublet_random_state",
                    )
                    if key in tool_input
                ]
                if changed_requested_params:
                    warnings.append(
                        "Existing Scrublet predictions were reused. Requested Scrublet parameters "
                        "were not recomputed; pass force_doublet_recompute=true to regenerate doublet calls."
                    )

            try:
                if not _metrics_precomputed:
                    calculate_qc_metrics(adata, inplace=True)
                if detect_doublets_flag and not _doublets_precomputed:
                    if force_doublet_recompute:
                        for col in ("doublet_score", "predicted_doublet"):
                            if col in adata.obs.columns:
                                del adata.obs[col]
                    detect_doublets(
                        adata,
                        batch_key=batch_key,
                        expected_doublet_rate=scrublet_expected_doublet_rate,
                        sim_doublet_ratio=scrublet_sim_doublet_ratio,
                        n_prin_comps=scrublet_n_prin_comps,
                        scrublet_min_counts=scrublet_min_counts,
                        scrublet_min_cells=scrublet_min_cells,
                        scrublet_min_gene_variability_pctl=scrublet_min_gene_variability_pctl,
                        random_state=scrublet_random_state,
                        inplace=True,
                    )
                    doublet_predictions_source = "computed"
                    stored_scrublet_params = adata.uns.get("scrublet_params", {}) if isinstance(adata.uns.get("scrublet_params", {}), dict) else {}
                    if stored_scrublet_params:
                        scrublet_params_for_report = {
                            "expected_doublet_rate": float(stored_scrublet_params.get("expected_doublet_rate", scrublet_expected_doublet_rate)),
                            "sim_doublet_ratio": float(stored_scrublet_params.get("sim_doublet_ratio", scrublet_sim_doublet_ratio)),
                            "n_prin_comps": int(stored_scrublet_params.get("n_prin_comps", scrublet_n_prin_comps)),
                            "n_prin_comps_used": int(stored_scrublet_params.get("n_prin_comps_used", stored_scrublet_params.get("n_prin_comps", scrublet_n_prin_comps))),
                            "min_counts": int(stored_scrublet_params.get("min_counts", scrublet_min_counts)),
                            "min_cells": int(stored_scrublet_params.get("min_cells", scrublet_min_cells)),
                            "min_gene_variability_pctl": float(stored_scrublet_params.get("min_gene_variability_pctl", scrublet_min_gene_variability_pctl)),
                            "random_state": int(stored_scrublet_params.get("random_state", scrublet_random_state)),
                            "force_recompute": force_doublet_recompute,
                            "batch_key": stored_scrublet_params.get("batch_key", batch_key),
                        }
            except ValueError as e:
                if detect_doublets_flag and "skimage is not installed" in str(e):
                    warnings.append("Scrublet auto-threshold requires skimage; reran without doublet detection.")
                    detect_doublets_flag = False
                    doublet_predictions_source = "not_run"
                    scrublet_params_for_report = requested_scrublet_params.copy()
                    if not _metrics_precomputed:
                        calculate_qc_metrics(adata, inplace=True)
                else:
                    raise

            # Use adata directly — no copy needed, metrics are already there
            qc_preview = adata

            requested_data_type = tool_input.get("data_type")  # "single_cell" | "single_nucleus" | None
            if requested_mt_threshold is not None:
                mt_threshold = float(requested_mt_threshold)
                data_type_confirmed = True
            else:
                data_type_confirmed = False
                # No explicit threshold — pick a sentinel based on data type hint only.
                # The real threshold must come from inspecting the QC figures.
                median_mt_preview = float(qc_preview.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in qc_preview.obs else 0.0
                mt_threshold = 5.0 if (requested_data_type == "single_nucleus" or median_mt_preview < 2.0) else 20.0
                warnings.append(
                    f"mt_threshold not set explicitly (median MT={median_mt_preview:.2f}%). "
                    "Review the MT% distribution in the QC figure and choose a data-driven threshold "
                    "based on where the high-MT tail separates from the main population. "
                    "Then re-run with mt_threshold=<value> to get exact removal counts."
                )
            if not filter_mt:
                warnings.append(
                    "Hard MT% cell filtering is disabled for this run; MT metrics are "
                    "reported for QC review only."
                )

            _mt_col = qc_preview.obs['pct_counts_mt'] if 'pct_counts_mt' in qc_preview.obs else None
            cells_over_mt = int((_mt_col >= mt_threshold).sum()) if _mt_col is not None else 0
            # Report exact removal counts at a range of thresholds so the model can present
            # data-driven options without anchoring to any single "standard" value.
            mt_threshold_options = {
                str(t): {
                    "threshold": t,
                    "cells_flagged": int((_mt_col >= t).sum()) if _mt_col is not None else 0,
                    "pct_flagged": round(float((_mt_col >= t).mean()) * 100, 1) if _mt_col is not None else 0,
                }
                for t in [5, 10, 15, 20, 25, 30]
            } if not data_type_confirmed else None
            predicted_doublets = int(qc_preview.obs['predicted_doublet'].sum()) if 'predicted_doublet' in qc_preview.obs else 0
            genes_low_cells = (
                int((qc_preview.var['n_cells_by_counts'] < min_cells).sum())
                if min_cells is not None and 'n_cells_by_counts' in qc_preview.var.columns
                else 0
            )
            cells_low_genes = (
                int((qc_preview.obs['n_genes_by_counts'] < int(min_genes)).sum())
                if min_genes is not None and 'n_genes_by_counts' in qc_preview.obs.columns
                else 0
            )
            ribo_genes = int(qc_preview.var['ribo'].sum()) if 'ribo' in qc_preview.var.columns and remove_ribo else 0
            mt_genes = int(qc_preview.var['mt'].sum()) if 'mt' in qc_preview.var.columns and remove_mt else 0
            n_mt_genes_detected = int(qc_preview.var['mt'].sum()) if 'mt' in qc_preview.var.columns else 0
            n_ribo_genes_detected = int(qc_preview.var['ribo'].sum()) if 'ribo' in qc_preview.var.columns else 0

            cell_removal_mask = pd.Series(False, index=qc_preview.obs_names)
            if min_genes is not None and 'n_genes_by_counts' in qc_preview.obs.columns:
                cell_removal_mask |= qc_preview.obs['n_genes_by_counts'] < int(min_genes)
            if filter_mt and 'pct_counts_mt' in qc_preview.obs.columns:
                cell_removal_mask |= qc_preview.obs['pct_counts_mt'] >= mt_threshold
            if remove_doublets and 'predicted_doublet' in qc_preview.obs.columns:
                cell_removal_mask |= qc_preview.obs['predicted_doublet'].astype(bool)
            projected_cells_removed = int(cell_removal_mask.sum())
            projected_cells_retained = int(n_before - projected_cells_removed)

            gene_removal_mask = pd.Series(False, index=qc_preview.var_names)
            if min_cells is not None and 'n_cells_by_counts' in qc_preview.var.columns:
                gene_removal_mask |= qc_preview.var['n_cells_by_counts'] < int(min_cells)
            if remove_ribo and 'ribo' in qc_preview.var.columns:
                gene_removal_mask |= qc_preview.var['ribo'].astype(bool)
            if remove_mt and 'mt' in qc_preview.var.columns:
                gene_removal_mask |= qc_preview.var['mt'].astype(bool)
            projected_genes_removed = int(gene_removal_mask.sum())
            projected_genes_retained = int(g_before - projected_genes_removed)

            figure_outputs = []
            figure_dir = tool_input.get("figure_dir")
            if figure_dir:
                import scanpy as sc
                import matplotlib.pyplot as plt
                os.makedirs(figure_dir, exist_ok=True)
                try:
                    qc_plot_adata = adata.copy()

                    # Add log-transformed columns for visualization
                    if "total_counts" in qc_plot_adata.obs.columns:
                        qc_plot_adata.obs["log10_total_counts"] = np.log10(qc_plot_adata.obs["total_counts"] + 1)
                    if "n_genes_by_counts" in qc_plot_adata.obs.columns:
                        qc_plot_adata.obs["log10_n_genes"] = np.log10(qc_plot_adata.obs["n_genes_by_counts"] + 1)

                    # --- Figure 1: Main QC violin plots (log counts, log genes, MT%, ribo%) ---
                    main_violin_keys = []
                    if "log10_total_counts" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("log10_total_counts")
                    if "log10_n_genes" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("log10_n_genes")
                    if "pct_counts_mt" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("pct_counts_mt")
                    if "pct_counts_ribo" in qc_plot_adata.obs.columns:
                        main_violin_keys.append("pct_counts_ribo")

                    if main_violin_keys:
                        sc.pl.violin(qc_plot_adata, main_violin_keys, jitter=0.2, multi_panel=True, show=False)
                        plt.suptitle("QC Metrics Distribution", y=1.02)
                        violin_path = os.path.join(figure_dir, "qc_violin_metrics.png")
                        plt.savefig(violin_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        figure_outputs.append(violin_path)

                    # --- Figure 2: Doublet scores (if available) ---
                    if "doublet_score" in qc_plot_adata.obs.columns:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        # Violin of doublet scores
                        sc.pl.violin(qc_plot_adata, "doublet_score", jitter=0.2, ax=axes[0], show=False)
                        axes[0].set_title("Doublet Score Distribution")
                        # Histogram with threshold
                        axes[1].hist(qc_plot_adata.obs["doublet_score"], bins=50, edgecolor='black', alpha=0.7)
                        axes[1].axvline(0.25, color='red', linestyle='--', label='Typical threshold (0.25)')
                        axes[1].set_xlabel("Doublet Score")
                        axes[1].set_ylabel("Cell Count")
                        axes[1].set_title("Doublet Score Histogram")
                        axes[1].legend()
                        fig.tight_layout()
                        doublet_path = os.path.join(figure_dir, "qc_doublet_scores.png")
                        fig.savefig(doublet_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        figure_outputs.append(doublet_path)

                    # --- Figure 3: Histograms for counts and genes (log scale) ---
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    if "total_counts" in qc_plot_adata.obs.columns:
                        axes[0].hist(qc_plot_adata.obs["total_counts"], bins=100, edgecolor='black', alpha=0.7)
                        axes[0].set_xscale('log')
                        axes[0].set_xlabel("Total Counts (log scale)")
                        axes[0].set_ylabel("Cell Count")
                        axes[0].set_title("Library Size Distribution")
                    if "n_genes_by_counts" in qc_plot_adata.obs.columns:
                        axes[1].hist(qc_plot_adata.obs["n_genes_by_counts"], bins=100, edgecolor='black', alpha=0.7)
                        axes[1].set_xscale('log')
                        axes[1].set_xlabel("Genes Detected (log scale)")
                        axes[1].set_ylabel("Cell Count")
                        axes[1].set_title("Genes per Cell Distribution")
                    fig.tight_layout()
                    hist_path = os.path.join(figure_dir, "qc_histograms.png")
                    fig.savefig(hist_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    figure_outputs.append(hist_path)

                    # --- Figure 4: MT% histogram (threshold line only when user confirmed a value) ---
                    if "pct_counts_mt" in qc_plot_adata.obs.columns:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.hist(qc_plot_adata.obs["pct_counts_mt"], bins=100, edgecolor='black', alpha=0.7)
                        if data_type_confirmed:
                            ax.axvline(mt_threshold, color='red', linestyle='--', linewidth=2,
                                       label=f'Threshold: {mt_threshold:.1f}%')
                            ax.legend()
                        ax.set_xlabel("Mitochondrial %")
                        ax.set_ylabel("Cell Count")
                        ax.set_title("Mitochondrial Content Distribution")
                        fig.tight_layout()
                        mt_hist_path = os.path.join(figure_dir, "qc_mt_histogram.png")
                        fig.savefig(mt_hist_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        figure_outputs.append(mt_hist_path)

                    # --- Figure 5: Scatter plots (genes vs counts, MT vs counts) ---
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    if 'n_genes_by_counts' in qc_plot_adata.obs.columns and 'total_counts' in qc_plot_adata.obs.columns:
                        sc.pl.scatter(qc_plot_adata, x='total_counts', y='n_genes_by_counts', ax=axes[0], show=False)
                        axes[0].set_xscale('log')
                        axes[0].set_yscale('log')
                        axes[0].set_title("Genes vs Counts (log-log)")
                    if 'pct_counts_mt' in qc_plot_adata.obs.columns and 'total_counts' in qc_plot_adata.obs.columns:
                        sc.pl.scatter(qc_plot_adata, x='total_counts', y='pct_counts_mt', ax=axes[1], show=False)
                        axes[1].set_xscale('log')
                        if data_type_confirmed:
                            axes[1].axhline(mt_threshold, color='red', linestyle='--', linewidth=2,
                                            label=f'MT threshold: {mt_threshold:.1f}%')
                            axes[1].legend()
                        axes[1].set_title("MT% vs Counts")
                    fig.tight_layout()
                    scatter_path = os.path.join(figure_dir, "qc_scatter.png")
                    fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    figure_outputs.append(scatter_path)

                    # --- Figure 6: Ribo vs MT scatter (if both available) ---
                    if 'pct_counts_mt' in qc_plot_adata.obs.columns and 'pct_counts_ribo' in qc_plot_adata.obs.columns:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sc.pl.scatter(qc_plot_adata, x='pct_counts_mt', y='pct_counts_ribo', ax=ax, show=False)
                        if data_type_confirmed:
                            ax.axvline(mt_threshold, color='red', linestyle='--', linewidth=1, label=f'MT threshold')
                            ax.legend()
                        ax.set_title("Ribosomal vs Mitochondrial Content")
                        fig.tight_layout()
                        ribo_mt_path = os.path.join(figure_dir, "qc_ribo_vs_mt.png")
                        fig.savefig(ribo_mt_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        figure_outputs.append(ribo_mt_path)

                except Exception as e:
                    warnings.append(f"QC figure generation failed: {e}")

            qc_decisions = {
                "mt_threshold": {
                    "value": mt_threshold,
                    "filter_enabled": filter_mt,
                    "data_type_confirmed": data_type_confirmed,
                    **({"threshold_options": mt_threshold_options} if mt_threshold_options else {}),
                    "reason": (
                        f"Cells with pct_counts_mt >= {mt_threshold:.1f}% are flagged for review; "
                        + (
                            "this threshold will be applied in filtering."
                            if filter_mt
                            else "hard MT% filtering is disabled for this run."
                        )
                    ),
                    "cells_flagged": cells_over_mt,
                },
                "min_genes": {
                    "value": int(min_genes) if min_genes is not None else None,
                    "enabled": min_genes is not None,
                    "reason": (
                        f"Cells with fewer than {int(min_genes)} detected genes are usually low-quality."
                        if min_genes is not None
                        else "No cell-level gene-count filter was requested."
                    ),
                    "cells_flagged": cells_low_genes,
                },
                "min_cells_per_gene": {
                    "value": int(min_cells) if min_cells is not None else None,
                    "reason": (
                        f"Genes detected in fewer than {int(min_cells)} cells add noise and little clustering signal."
                        if min_cells is not None
                        else "No min-cells-per-gene filter requested."
                    ),
                    "genes_flagged": genes_low_cells,
                },
                "doublet_detection": {
                    "enabled": bool(detect_doublets_flag),
                    "reason": (
                        "Scrublet flags likely multiplets; these are reported so the user can decide whether to exclude them."
                    ),
                    "cells_flagged": predicted_doublets,
                    "remove_on_apply": remove_doublets,
                    "predictions_source": doublet_predictions_source,
                    "parameters": scrublet_params_for_report,
                },
                "remove_ribo": {
                    "enabled": bool(remove_ribo),
                    "reason": "Ribosomal genes can dominate variance and dilute biologically informative structure.",
                    "genes_flagged": ribo_genes,
                },
                "remove_mt_genes": {
                    "enabled": bool(remove_mt),
                    "reason": "Mitochondrial genes are often excluded from downstream feature selection to reduce QC-driven signal.",
                    "genes_flagged": mt_genes,
                },
                "gene_detection_counts": {
                    "n_mt_genes_detected": n_mt_genes_detected,
                    "n_ribo_genes_detected": n_ribo_genes_detected,
                },
            }
            filtering_plan = {
                "confirmation_required": not preview_only and not confirm_filtering,
                "confirmed": bool(confirm_filtering and not preview_only),
                "parameters": {
                    "mt_threshold": mt_threshold,
                    "filter_mt": filter_mt,
                    "data_type": requested_data_type,
                    "data_type_confirmed": data_type_confirmed,
                    "min_genes": int(min_genes) if min_genes is not None else None,
                    "min_cells_per_gene": int(min_cells) if min_cells is not None else None,
                    "remove_ribo": bool(remove_ribo),
                    "remove_mt_genes": bool(remove_mt),
                    "detect_doublets": bool(detect_doublets_flag),
                    "remove_doublets": bool(remove_doublets),
                    "batch_key_for_doublets": batch_key,
                    "scrublet": scrublet_params_for_report,
                },
                "cell_filters": {
                    "low_genes": {
                        "enabled": min_genes is not None,
                        "threshold": int(min_genes) if min_genes is not None else None,
                        "cells_flagged": cells_low_genes,
                        "will_remove": min_genes is not None,
                    },
                    "high_mt": {
                        "enabled": bool(filter_mt),
                        "threshold_pct": mt_threshold,
                        "cells_flagged": cells_over_mt,
                        "will_remove": bool(filter_mt),
                    },
                    "doublets": {
                        "enabled": bool(detect_doublets_flag),
                        "cells_flagged": predicted_doublets,
                        "will_remove": bool(remove_doublets),
                    },
                },
                "gene_filters": {
                    "low_cells": {
                        "enabled": min_cells is not None,
                        "threshold": int(min_cells) if min_cells is not None else None,
                        "genes_flagged": genes_low_cells,
                        "will_remove": min_cells is not None,
                    },
                    "ribosomal": {
                        "enabled": bool(remove_ribo),
                        "genes_flagged": ribo_genes,
                        "will_remove": bool(remove_ribo),
                    },
                    "mitochondrial": {
                        "enabled": bool(remove_mt),
                        "genes_flagged": mt_genes,
                        "will_remove": bool(remove_mt),
                    },
                },
                "projected_after_filtering": {
                    "cells_removed": projected_cells_removed,
                    "cells_retained": projected_cells_retained,
                    "genes_removed_before_cell_filtering": projected_genes_removed,
                    "genes_retained_before_cell_filtering": projected_genes_retained,
                    "note": (
                        "Gene removals are estimated before cell filtering; final gene removals can change "
                        "after cells are removed."
                    ),
                },
            }

            cell_filter_bits = []
            if min_genes is not None:
                cell_filter_bits.append(
                    f"removing {cells_low_genes} cells with fewer than {int(min_genes)} genes"
                )
            if filter_mt:
                cell_filter_bits.append(
                    f"filtering {cells_over_mt} cells with pct_counts_mt >= {mt_threshold:.1f}%"
                )
            else:
                cell_filter_bits.append(
                    f"reporting {cells_over_mt} cells with pct_counts_mt >= {mt_threshold:.1f}% without applying an MT filter"
                )
            doublet_action = "removing" if remove_doublets else "flagging"
            recommendation = (
                "I recommend "
                + ", ".join(cell_filter_bits)
                + f", removing {genes_low_cells} low-detection genes, and "
                + (
                    f"{doublet_action} doublets ({predicted_doublets} cells)."
                    if detect_doublets_flag
                    else "skipping doublet detection."
                )
            )
            if batch_resolution and batch_resolution.needs_user_confirmation and tool_input.get("batch_key"):
                recommendation += (
                    f" I could not confirm the requested per-batch column automatically; "
                    f"'{batch_resolution.recommended_column}' looks closest."
                )

            batch_strategy = (
                metadata_resolution_to_dict(batch_resolution)
                if batch_resolution
                else {
                    "status": "not_applicable",
                    "requested_column": requested_batch_key,
                    "applied_column": None,
                    "recommended_column": None,
                    "recommended_role": None,
                    "needs_user_confirmation": False,
                    "reason": "Doublet detection is disabled for this QC run.",
                    "candidates": [],
                }
            )
            batch_strategy["used_for_doublets"] = batch_key
            decisions = []
            batch_decision = decision_for_batch_strategy(
                batch_strategy,
                context="doublet_detection",
                source_tool="run_qc",
                batch_relevant=bool(tool_input.get("batch_key")),
            )
            if batch_decision is not None:
                decisions.append(batch_decision)
            artifact_payloads = [
                artifact
                for artifact in (
                    _artifact_payload(path, role="qc_figure", metadata={"mode": "preview" if preview_only else "applied"})
                    for path in figure_outputs
                )
                if artifact is not None
            ]

            if preview_only:
                preview_result = {
                    "status": "ok",
                    "tool": "run_qc",
                    "mode": "preview",
                    "before": {"n_cells": n_before, "n_genes": g_before},
                    "after": {"n_cells": n_before, "n_genes": g_before},
                    "data_type_confirmed": data_type_confirmed,
                    "recommendation": recommendation,
                    "filtering_plan": filtering_plan,
                    "qc_decisions": qc_decisions,
                    "metrics": {
                        "median_pct_mt": float(qc_preview.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in qc_preview.obs else None,
                        "doublet_rate": float(qc_preview.obs['predicted_doublet'].mean()) if 'predicted_doublet' in qc_preview.obs else None,
                        "cells_below_min_genes": cells_low_genes,
                        "genes_below_min_cells": genes_low_cells,
                        "predicted_doublets": predicted_doublets,
                        "cells_over_mt_threshold": cells_over_mt,
                    },
                    "warnings": warnings,
                    "figures": figure_outputs,
                    "batch_strategy": batch_strategy,
                    "state": make_state(adata)
                }
                verification_checks = [
                    _check("qc_metrics_computed", "pct_counts_mt" in qc_preview.obs.columns, "QC metrics were computed on the preview copy."),
                    _check(
                        "batch_strategy_reported",
                        bool(batch_strategy.get("status")),
                        "Batch strategy is attached to the QC preview.",
                    ),
                ]
                if figure_outputs:
                    verification_checks.append(
                        _check(
                            "preview_figures_exist",
                            all(os.path.exists(path) for path in figure_outputs),
                            "QC preview figures were written to disk.",
                        )
                    )
                return _finalize_result(
                    preview_result,
                    adata,
                    dataset_changed=False,
                    summary="Previewed QC metrics, thresholds, and doublet strategy without filtering cells.",
                    artifacts_created=artifact_payloads,
                    decisions_raised=decisions,
                    verification=_build_verification(
                        "passed",
                        "QC preview completed and reported a concrete batch strategy.",
                        verification_checks,
                    ),
                )

            if not confirm_filtering:
                confirmation_result = {
                    "status": "needs_confirmation",
                    "tool": "run_qc",
                    "mode": "confirmation_required",
                    "message": (
                        "QC filtering was not applied. Review the thresholds, parameters, and removal "
                        "counts, then confirm before I remove cells or genes."
                    ),
                    "required_next_action": "ask_user",
                    "before": {"n_cells": n_before, "n_genes": g_before},
                    "after": {"n_cells": n_before, "n_genes": g_before},
                    "recommendation": recommendation,
                    "filtering_plan": filtering_plan,
                    "qc_decisions": qc_decisions,
                    "metrics": {
                        "median_pct_mt": float(qc_preview.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in qc_preview.obs else None,
                        "doublet_rate": float(qc_preview.obs['predicted_doublet'].mean()) if 'predicted_doublet' in qc_preview.obs else None,
                        "cells_below_min_genes": cells_low_genes,
                        "genes_below_min_cells": genes_low_cells,
                        "predicted_doublets": predicted_doublets,
                        "cells_over_mt_threshold": cells_over_mt,
                    },
                    "warnings": warnings,
                    "figures": figure_outputs,
                    "batch_strategy": batch_strategy,
                    "state": make_state(adata),
                }
                return _finalize_result(
                    confirmation_result,
                    adata,
                    dataset_changed=False,
                    summary="Prepared a QC filtering plan and paused for explicit confirmation.",
                    artifacts_created=artifact_payloads,
                    decisions_raised=decisions,
                    verification=_build_verification(
                        "passed",
                        "QC filtering did not modify the dataset because confirmation is required.",
                        [
                            _check(
                                "dataset_unchanged",
                                adata.n_obs == n_before and adata.n_vars == g_before,
                                "No cells or genes were removed before confirmation.",
                            )
                        ],
                    ),
                )

            try:
                run_qc_pipeline(
                    adata,
                    mt_threshold=mt_threshold,
                    filter_mt=filter_mt,
                    min_genes=int(min_genes) if min_genes is not None else None,
                    min_cells=min_cells,
                    remove_ribo=remove_ribo,
                    detect_doublets_flag=detect_doublets_flag,
                    remove_doublets=remove_doublets,
                    batch_key=batch_key,
                    scrublet_expected_doublet_rate=scrublet_expected_doublet_rate,
                    scrublet_sim_doublet_ratio=scrublet_sim_doublet_ratio,
                    scrublet_n_prin_comps=scrublet_n_prin_comps,
                    scrublet_min_counts=scrublet_min_counts,
                    scrublet_min_cells=scrublet_min_cells,
                    scrublet_min_gene_variability_pctl=scrublet_min_gene_variability_pctl,
                    scrublet_random_state=scrublet_random_state,
                    force_doublet_recompute=force_doublet_recompute,
                )
            except ValueError as e:
                if detect_doublets_flag and "skimage is not installed" in str(e):
                    warnings.append("Scrublet auto-threshold requires skimage; reran QC without doublet detection.")
                    detect_doublets_flag = False
                    run_qc_pipeline(
                        adata,
                        mt_threshold=mt_threshold,
                        filter_mt=filter_mt,
                        min_genes=int(min_genes) if min_genes is not None else None,
                        min_cells=min_cells,
                        remove_ribo=remove_ribo,
                        detect_doublets_flag=False,
                        remove_doublets=False,
                        batch_key=batch_key,
                    )
                else:
                    raise

            actual_cells_removed = n_before - adata.n_obs
            actual_genes_removed = g_before - adata.n_vars
            post_filter_doublets = int(adata.obs['predicted_doublet'].sum()) if 'predicted_doublet' in adata.obs else None
            post_filter_doublet_rate = (
                float(adata.obs['predicted_doublet'].mean())
                if 'predicted_doublet' in adata.obs
                else None
            )
            pre_filter_doublet_rate = (
                float(predicted_doublets / n_before)
                if detect_doublets_flag and n_before
                else None
            )
            qc_decisions["min_cells_per_gene"]["genes_flagged_before_cell_filtering"] = genes_low_cells
            qc_decisions["min_cells_per_gene"]["genes_removed_after_cell_filtering"] = actual_genes_removed
            qc_decisions["doublet_detection"]["cells_flagged_before_filtering"] = predicted_doublets
            qc_decisions["doublet_detection"]["cells_remaining_after_filtering"] = post_filter_doublets

            output_path = fix_output_path(tool_input.get("output_path"), "run_qc")
            if output_path:
                write_h5ad_safe(adata, output_path)
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifact_payloads.append(artifact)

            qc_result = {
                "status": "ok",
                "tool": "run_qc",
                "input_path": tool_input.get("data_path", "memory"),
                "output_path": output_path,
                "saved": output_path is not None,
                "before": {"n_cells": n_before, "n_genes": g_before},
                "after": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "recommendation": recommendation,
                "filtering_plan": filtering_plan,
                "qc_decisions": qc_decisions,
                "metrics": {
                    "cells_removed": actual_cells_removed,
                    "genes_removed": actual_genes_removed,
                    "doublet_rate": pre_filter_doublet_rate,
                    "post_filter_doublet_rate": post_filter_doublet_rate,
                    "median_pct_mt": float(adata.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in adata.obs else None,
                    "cells_below_min_genes": cells_low_genes,
                    "genes_below_min_cells": genes_low_cells,
                    "genes_below_min_cells_before_cell_filtering": genes_low_cells,
                    "genes_removed_after_cell_filtering": actual_genes_removed,
                    "predicted_doublets": predicted_doublets,
                    "predicted_doublets_before_filtering": predicted_doublets,
                    "predicted_doublets_remaining": post_filter_doublets,
                    "cells_over_mt_threshold": cells_over_mt,
                },
                "warnings": warnings,
                "figures": figure_outputs,
                "batch_strategy": batch_strategy,
                "state": make_state(adata)
            }
            verification_checks = [
                _check("qc_metrics_present", "pct_counts_mt" in adata.obs.columns, "MT QC metric is present after QC."),
                _check(
                    "batch_key_valid",
                    batch_key is None or batch_key in adata.obs.columns,
                    f"Batch key '{batch_key}' is available on the filtered AnnData." if batch_key else "QC ran without per-batch stratification.",
                ),
            ]
            if detect_doublets_flag:
                verification_checks.append(
                    _check(
                        "doublet_scores_present",
                        "predicted_doublet" in adata.obs.columns,
                        "Doublet scores/predictions are available after QC.",
                    )
                )
            return _finalize_result(
                qc_result,
                adata,
                dataset_changed=True,
                summary="Applied QC filtering and recorded the batch-aware doublet strategy.",
                artifacts_created=artifact_payloads,
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "QC completed and the resulting AnnData passed post-action checks.",
                    verification_checks,
                ),
            )

        elif tool_name == "normalize_and_hvg":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_hvg = int(tool_input.get("n_hvg", 4000))
            target_sum = tool_input.get("target_sum", 10000)
            target_sum = float(target_sum) if target_sum is not None else None
            log_transform = bool(tool_input.get("log_transform", True))
            raw_layer_name = tool_input.get("raw_layer_name", "raw_counts")
            force_reset_from_raw = bool(tool_input.get("force_reset_from_raw", True))
            set_raw_after_normalization = bool(tool_input.get("set_raw_after_normalization", True))
            hvg_flavor = tool_input.get("hvg_flavor", "seurat_v3")
            hvg_layer = tool_input.get("hvg_layer") or None
            batch_key = tool_input.get("batch_key")
            hvg_exclude_patterns = tool_input.get("hvg_exclude_patterns") or []
            if isinstance(hvg_exclude_patterns, str):
                hvg_exclude_patterns = [hvg_exclude_patterns]
            hvg_exclusion_mode = tool_input.get("hvg_exclusion_mode", "post")
            hvg_exclude_match_mode = tool_input.get("hvg_exclude_match_mode", "match")
            hvg_exclusion_source = tool_input.get("hvg_exclusion_source")

            before_shape = (adata.n_obs, adata.n_vars)

            def _integer_like_matrix(matrix, n_rows: int = 100, n_cols: int = 100) -> bool:
                if matrix is None:
                    return False
                sample = matrix[:min(n_rows, matrix.shape[0]), :min(n_cols, matrix.shape[1])]
                if hasattr(sample, "toarray"):
                    sample = sample.toarray()
                sample = np.asarray(sample)
                return bool(np.allclose(sample, np.round(sample)))

            try:
                normalize_data(
                    adata,
                    target_sum=target_sum,
                    log_transform=log_transform,
                    preserve_raw=True,
                    raw_layer_name=raw_layer_name,
                    force_reset_from_raw=force_reset_from_raw,
                )
                if set_raw_after_normalization:
                    adata.raw = adata.copy()
                select_hvg(
                    adata,
                    n_top_genes=n_hvg,
                    flavor=hvg_flavor,
                    layer=hvg_layer,
                    batch_key=batch_key,
                    exclude_patterns=hvg_exclude_patterns,
                    exclusion_mode=hvg_exclusion_mode,
                    exclude_match_mode=hvg_exclude_match_mode,
                    exclusion_source=hvg_exclusion_source,
                )
            except ValueError as e:
                return _error_result(
                    tool="normalize_and_hvg",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Retry normalize_and_hvg with force_reset_from_raw=true "
                        f"so adata.X is restored from layers['{raw_layer_name}'] before normalization.",
                        "If the dataset was already normalized in this session, "
                        "do not call scanpy normalize/log1p manually on the current X; "
                        "reset from raw counts first.",
                        "If the dataset was loaded already-normalized from disk, "
                        "load the raw-counts version instead, or place raw "
                        f"counts into adata.layers['{raw_layer_name}'] before retrying.",
                    ],
                )

            output_path = fix_output_path(tool_input.get("output_path"), "normalize_and_hvg")
            if output_path:
                write_h5ad_safe(adata, output_path)

            artifact_payloads = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifact_payloads.append(artifact)

            hvg_meta = adata.uns.get("hvg", {})
            exclusion_meta = hvg_meta.get("feature_exclusions", {})
            raw_counts_present = raw_layer_name in adata.layers
            raw_counts_integer_like = (
                _integer_like_matrix(adata.layers[raw_layer_name])
                if raw_counts_present
                else False
            )
            raw_shape = list(adata.raw.shape) if adata.raw is not None else None
            normalized_counts_target = None
            if log_transform:
                import scipy.sparse as sp
                if sp.issparse(adata.X):
                    X_counts = adata.X.copy()
                    X_counts.data = np.expm1(X_counts.data)
                    normalized_counts_target = float(np.median(np.asarray(X_counts.sum(axis=1)).ravel()))
                else:
                    X_counts = np.expm1(np.asarray(adata.X))
                    normalized_counts_target = float(np.median(X_counts.sum(axis=1)))

            result_payload = {
                "status": "ok",
                "tool": "normalize_and_hvg",
                "output_path": output_path,
                "saved": output_path is not None,
                "before": {"n_cells": before_shape[0], "n_genes": before_shape[1]},
                "after": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "target_sum": target_sum,
                "log_transform": log_transform,
                "normalization": adata.uns.get("normalization", {}),
                "raw_layer_name": raw_layer_name,
                "raw_counts_present": raw_counts_present,
                "raw_counts_integer_like": raw_counts_integer_like,
                "adata_raw_set": adata.raw is not None,
                "adata_raw_shape": raw_shape,
                "set_raw_after_normalization": set_raw_after_normalization,
                "n_hvg": int(adata.var['highly_variable'].sum()),
                "hvg": {
                    "requested_flavor": hvg_meta.get("requested_flavor", hvg_flavor),
                    "flavor": hvg_meta.get("flavor", hvg_flavor),
                    "method": "scanpy.pp.highly_variable_genes",
                    "n_top_genes": int(hvg_meta.get("n_top_genes", n_hvg)),
                    "batch_key": hvg_meta.get("batch_key", batch_key),
                    "layer": hvg_meta.get("layer", hvg_layer),
                    "n_hvg_selected": int(adata.var['highly_variable'].sum()),
                },
                "feature_exclusions": exclusion_meta,
                "metrics": {
                    "n_hvg_selected": int(adata.var['highly_variable'].sum()),
                    "normalized_counts_target_median": normalized_counts_target,
                    "n_excluded_features": int(exclusion_meta.get("n_excluded", 0) or 0),
                    "excluded_features_marked_hvg": int(exclusion_meta.get("excluded_hvg_after_forcing", 0) or 0),
                },
                "warnings": warnings,
                "state": make_state(adata)
            }
            verification_checks = [
                _check(
                    "raw_counts_present",
                    raw_counts_present,
                    f"Raw counts layer '{raw_layer_name}' is present.",
                ),
                _check(
                    "raw_counts_integer_like",
                    raw_counts_integer_like,
                    f"Raw counts layer '{raw_layer_name}' appears integer-like in a matrix sample.",
                ),
                _check(
                    "hvg_count",
                    int(adata.var['highly_variable'].sum()) > 0,
                    f"{int(adata.var['highly_variable'].sum())} HVGs are marked.",
                ),
                _check(
                    "excluded_features_not_hvg",
                    int(exclusion_meta.get("excluded_hvg_after_forcing", 0) or 0) == 0,
                    "No excluded features remain marked highly_variable.",
                ),
            ]
            if set_raw_after_normalization:
                verification_checks.append(
                    _check(
                        "adata_raw_set",
                        adata.raw is not None,
                        "adata.raw was set after normalization/log1p.",
                    )
                )
            return _finalize_result(
                result_payload,
                adata,
                dataset_changed=True,
                summary=(
                    f"Normalized data to target_sum={target_sum}, "
                    f"selected {int(adata.var['highly_variable'].sum())} HVGs, "
                    f"and applied {int(exclusion_meta.get('n_excluded', 0) or 0)} source-defined feature exclusions."
                ),
                artifacts_created=artifact_payloads,
                verification=_build_verification(
                    "passed",
                    "Normalization and HVG selection completed with provenance checks.",
                    verification_checks,
                ),
            )

        elif tool_name == "run_pca":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_comps = int(tool_input.get("n_comps") or tool_input.get("n_pcs") or 30)
            svd_solver = tool_input.get("svd_solver", "arpack")
            mask_var = tool_input.get("mask_var", "highly_variable")

            run_pca(adata, n_comps=n_comps, mask_var=mask_var, svd_solver=svd_solver)

            output_path = fix_output_path(tool_input.get("output_path"), "run_pca")
            if output_path:
                write_h5ad_safe(adata, output_path)

            result = {
                "status": "ok",
                "tool": "run_pca",
                "output_path": output_path,
                "saved": output_path is not None,
                "n_comps": n_comps,
                "svd_solver": svd_solver,
                "mask_var": mask_var,
                "variance_explained": float(adata.uns["pca"]["variance_ratio"].sum()),
                "side_effects": {
                    "pca_computed": True,
                    "neighbors_recomputed": False,
                    "umap_recomputed": False,
                    "clustering_recomputed": False,
                },
                "warnings": warnings,
                "state": make_state(adata),
            }
            return _finalize_result(
                result,
                adata,
                dataset_changed=True,
                summary=f"Ran PCA only with n_comps={n_comps} and svd_solver={svd_solver}.",
                verification=_build_verification(
                    "passed",
                    "PCA was computed without downstream graph or embedding side effects.",
                    [
                        _check("pca_present", "X_pca" in adata.obsm, "PCA embedding exists in adata.obsm['X_pca']."),
                        _check("no_umap_side_effect", "X_umap" not in adata.obsm or starting_state.get("has_umap"), "run_pca did not create a new UMAP embedding."),
                    ],
                ),
            )

        elif tool_name == "run_neighbors":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            n_neighbors = int(tool_input.get("n_neighbors") or 30)
            n_pcs = tool_input.get("n_pcs")
            n_pcs = int(n_pcs) if n_pcs is not None else None
            use_rep = tool_input.get("use_rep", "X_pca")
            metric = tool_input.get("metric", "euclidean")
            key_added = tool_input.get("key_added")

            if use_rep not in adata.obsm:
                available_reps = [k for k in adata.obsm.keys()]
                return _smart_unavailable_result(
                    tool="run_neighbors",
                    message=f"Representation '{use_rep}' not found in adata.obsm.",
                    adata_obj=adata,
                    missing_prerequisites=["pca"] if use_rep == "X_pca" else [use_rep],
                    recovery_options=[
                        f"Run run_pca first to compute '{use_rep}'." if use_rep == "X_pca"
                        else f"Compute '{use_rep}' before calling run_neighbors.",
                        f"Set use_rep to one of the available representations: {available_reps}",
                    ],
                    extra={"available_representations": available_reps},
                )

            compute_neighbors(
                adata,
                n_neighbors=n_neighbors,
                n_pcs=n_pcs,
                use_rep=use_rep,
                metric=metric,
                key_added=key_added,
            )

            output_path = fix_output_path(tool_input.get("output_path"), "run_neighbors")
            if output_path:
                write_h5ad_safe(adata, output_path)

            graph_key = key_added or "neighbors"
            result = {
                "status": "ok",
                "tool": "run_neighbors",
                "output_path": output_path,
                "saved": output_path is not None,
                "n_neighbors": n_neighbors,
                "n_pcs": n_pcs,
                "use_rep": use_rep,
                "metric": metric,
                "neighbors_key": graph_key,
                "neighbors_provenance": _neighbors_provenance(adata) if key_added is None else _sanitize_uns_value(adata.uns.get(key_added, {})),
                "side_effects": {
                    "pca_recomputed": False,
                    "neighbors_recomputed": True,
                    "umap_recomputed": False,
                    "clustering_recomputed": False,
                },
                "warnings": warnings,
                "state": make_state(adata),
            }
            return _finalize_result(
                result,
                adata,
                dataset_changed=True,
                summary=f"Computed neighbors only using {use_rep} with n_neighbors={n_neighbors}.",
                verification=_build_verification(
                    "passed",
                    "Neighbor graph was computed without UMAP or clustering side effects.",
                    [
                        _check("neighbors_present", graph_key in adata.uns, f"Neighbors key '{graph_key}' exists in adata.uns."),
                        _check("no_umap_side_effect", "X_umap" not in adata.obsm or starting_state.get("has_umap"), "run_neighbors did not create a new UMAP embedding."),
                    ],
                ),
            )

        elif tool_name == "run_umap":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            neighbors_key = tool_input.get("neighbors_key")
            _neighbors_lookup = neighbors_key or "neighbors"
            if _neighbors_lookup not in adata.uns:
                available_graphs = [k for k in adata.uns if "neighbor" in k.lower() or k == "neighbors"]
                return _smart_unavailable_result(
                    tool="run_umap",
                    message=f"Neighbor graph '{_neighbors_lookup}' not found in adata.uns.",
                    adata_obj=adata,
                    missing_prerequisites=["neighbors"],
                    recovery_options=[
                        "Run run_neighbors first to compute the neighbor graph.",
                        *(
                            [f"Or set neighbors_key to one of the existing graphs: {available_graphs}"]
                            if available_graphs else []
                        ),
                    ],
                    extra={"available_neighbor_graphs": available_graphs},
                )

            neighbors_before = _neighbors_provenance(adata)
            min_dist = float(tool_input.get("min_dist", 0.5))
            spread = float(tool_input.get("spread", 1.0))
            n_components = int(tool_input.get("n_components", 2))
            random_state = int(tool_input.get("random_state", 0))

            compute_umap(
                adata,
                min_dist=min_dist,
                spread=spread,
                n_components=n_components,
                neighbors_key=neighbors_key,
                random_state=random_state,
            )
            neighbors_after = _neighbors_provenance(adata)
            graph_preserved = _provenance_same(neighbors_before, neighbors_after)

            output_path = fix_output_path(tool_input.get("output_path"), "run_umap")
            if output_path:
                write_h5ad_safe(adata, output_path)

            result = {
                "status": "ok",
                "tool": "run_umap",
                "output_path": output_path,
                "saved": output_path is not None,
                "min_dist": min_dist,
                "spread": spread,
                "n_components": n_components,
                "neighbors_key": neighbors_key or "neighbors",
                "random_state": random_state,
                "neighbors_before": neighbors_before,
                "neighbors_after": neighbors_after,
                "neighbor_graph_preserved": graph_preserved,
                "side_effects": {
                    "pca_recomputed": False,
                    "neighbors_recomputed": False,
                    "umap_recomputed": True,
                    "clustering_recomputed": False,
                },
                "warnings": warnings,
                "state": make_state(adata),
            }
            return _finalize_result(
                result,
                adata,
                dataset_changed=True,
                summary=f"Computed UMAP only from the existing neighbor graph with min_dist={min_dist}.",
                verification=_build_verification(
                    "passed" if graph_preserved else "warning",
                    "UMAP was computed from the existing neighbor graph.",
                    [
                        _check("umap_present", "X_umap" in adata.obsm, "UMAP embedding exists in adata.obsm['X_umap']."),
                        _check("neighbor_graph_preserved", graph_preserved, "Neighbor graph provenance and sparsity were unchanged by run_umap."),
                    ],
                ),
            )

        elif tool_name == "run_clustering":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "leiden")
            resolution = float(tool_input.get("resolution", 1.0))
            requested_cluster_key = tool_input.get("cluster_key")
            cluster_key, default_make_primary = _resolve_clustering_output_key(
                adata,
                method,
                resolution,
                requested_cluster_key,
            )
            make_primary = tool_input.get("make_primary")
            if make_primary is None:
                make_primary = default_make_primary
            if requested_cluster_key and requested_cluster_key == default_cluster_key_for_method(method) and not make_primary:
                warnings.append(
                    f"cluster_key '{requested_cluster_key}' is the primary alias for {method}; forcing make_primary=true."
                )
                make_primary = True

            result_payload = _apply_clustering(
                adata,
                method=method,
                resolution=resolution,
                cluster_key=cluster_key,
                make_primary=bool(make_primary),
            )

            output_path = fix_output_path(tool_input.get("output_path"), "run_clustering")
            if output_path:
                write_h5ad_safe(adata, output_path)
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)

            clustering_result = {
                "status": "ok",
                "tool": "run_clustering",
                "output_path": output_path,
                "saved": output_path is not None,
                "method": result_payload["method"],
                "resolution": result_payload["resolution"],
                "cluster_key": result_payload["cluster_key"],
                "created_obs_columns": result_payload["created_obs_columns"],
                "primary_alias": result_payload["primary_alias"],
                "primary_cluster_key": result_payload["primary_cluster_key"],
                "primary_alias_available": result_payload["primary_alias_available"],
                "primary_alias_created": result_payload["primary_alias_created"],
                "make_primary": bool(make_primary),
                "n_clusters": result_payload["n_clusters"],
                "cluster_sizes": result_payload["cluster_sizes"],
                "available_clusterings": result_payload["clusterings"],
                "warnings": warnings,
                "state": make_state(adata)
            }
            primary_key = result_payload["primary_cluster_key"]
            primary_check = (
                _check(
                    "primary_alias_available",
                    result_payload["primary_alias_available"],
                    f"Primary clustering alias '{primary_key}' is available.",
                )
                if bool(make_primary)
                else _check(
                    "primary_alias_not_requested",
                    True,
                    "Primary alias was not requested; no alias availability is implied.",
                )
            )
            verification_checks = [
                _check(
                    "cluster_key_created",
                    result_payload["cluster_key"] in adata.obs.columns,
                    f"Clustering column '{result_payload['cluster_key']}' exists in adata.obs.",
                ),
                _check(
                    "cluster_count_matches",
                    adata.obs[result_payload["cluster_key"]].nunique() == result_payload["n_clusters"],
                    "Reported cluster count matches the stored clustering column.",
                ),
                primary_check,
            ]
            return _finalize_result(
                clustering_result,
                adata,
                dataset_changed=True,
                summary=(
                    f"Ran {result_payload['method']} clustering at resolution {result_payload['resolution']} "
                    f"and stored results in '{result_payload['cluster_key']}'."
                ),
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "Clustering results were created and verified against the AnnData state.",
                    verification_checks,
                ),
            )

        elif tool_name == "compare_clusterings":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "leiden")
            resolutions = [float(value) for value in tool_input.get("resolutions", [])]
            if not resolutions:
                raise ValueError("compare_clusterings requires at least one resolution.")

            compare_results = []
            figure_dir = tool_input.get("figure_dir")
            generate_figures = bool(tool_input.get("generate_figures", False))
            include_images = bool(tool_input.get("include_images", False))
            promote_resolution = tool_input.get("promote_resolution")
            image_payloads = []

            if figure_dir:
                os.makedirs(figure_dir, exist_ok=True)

            for resolution in resolutions:
                cluster_key = infer_cluster_key(method, resolution)
                result_payload = _apply_clustering(
                    adata,
                    method=method,
                    resolution=resolution,
                    cluster_key=cluster_key,
                    make_primary=False,
                )
                compare_entry = {
                    "resolution": result_payload["resolution"],
                    "cluster_key": result_payload["cluster_key"],
                    "n_clusters": result_payload["n_clusters"],
                    "cluster_sizes": result_payload["cluster_sizes"],
                }

                if generate_figures and "X_umap" in adata.obsm:
                    path_root = figure_dir or "."
                    figure_name = f"umap_{cluster_key}.png"
                    figure_path = os.path.join(path_root, figure_name)
                    figure_result = _render_figure(
                        adata,
                        plot_type="umap",
                        output_path=figure_path,
                        color_by=cluster_key,
                        include_image=include_images,
                    )
                    compare_entry["figure_path"] = figure_path
                    if include_images and "image_base64" in figure_result:
                        image_payloads.append({
                            "cluster_key": cluster_key,
                            "output_path": figure_path,
                            "image_base64": figure_result["image_base64"],
                            "image_mime": figure_result["image_mime"],
                        })

                compare_results.append(compare_entry)

            if promote_resolution is not None:
                promote_key = infer_cluster_key(method, float(promote_resolution))
                if promote_key not in adata.obs.columns:
                    raise ValueError(
                        f"Cannot promote resolution {promote_resolution}; "
                        f"expected clustering key '{promote_key}' was not generated."
                    )
                promote_clustering_to_primary(
                    adata,
                    cluster_key=promote_key,
                    method=method,
                    resolution=float(promote_resolution),
                    created_by="tool",
                )

            result = {
                "status": "ok",
                "tool": "compare_clusterings",
                "method": "phenograph" if str(method).lower() == "phenograph" else "leiden",
                "comparisons": compare_results,
                "available_clusterings": _clusterings_payload(adata),
                "warnings": warnings,
                "state": make_state(adata),
            }
            if image_payloads:
                first = image_payloads[0]
                result["image_base64"] = first["image_base64"]
                result["image_mime"] = first["image_mime"]
                result["image_context"] = {
                    "cluster_key": first["cluster_key"],
                    "output_path": first["output_path"],
                }
            artifacts_created = [
                artifact
                for artifact in (
                    _artifact_payload(
                        comparison.get("figure_path"),
                        role="comparison_figure",
                        metadata={"cluster_key": comparison.get("cluster_key")},
                    )
                    for comparison in compare_results
                    if comparison.get("figure_path")
                )
                if artifact is not None
            ]
            decisions = []
            if promote_resolution is None:
                clustering_decision = decision_for_clustering_selection(
                    compare_results,
                    source_tool="compare_clusterings",
                )
                if clustering_decision is not None:
                    decisions.append(clustering_decision)
            verification_checks = [
                _check(
                    "comparison_keys_created",
                    all(comparison["cluster_key"] in adata.obs.columns for comparison in compare_results),
                    "Every compared clustering key exists in adata.obs.",
                ),
            ]
            if artifacts_created:
                verification_checks.append(
                    _check(
                        "comparison_figures_exist",
                        all(os.path.exists(artifact["path"]) for artifact in artifacts_created),
                        "Generated comparison figures exist on disk.",
                    )
                )
            return _finalize_result(
                result,
                adata,
                dataset_changed=True,
                summary="Generated a safe multi-resolution clustering comparison without overwriting prior results.",
                artifacts_created=artifacts_created,
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "All requested clustering comparisons were preserved and verified.",
                    verification_checks,
                ),
            )

        elif tool_name == "run_celltypist":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            model = tool_input.get("model", "Immune_All_Low.pkl")
            majority = tool_input.get("majority_voting", True)
            cluster_key = tool_input.get("cluster_key", "leiden")
            if majority and cluster_key not in adata.obs.columns:
                return _smart_unavailable_result(
                    tool="run_celltypist",
                    message=(
                        f"CellTypist majority voting needs a valid clustering column, but '{cluster_key}' "
                        "is not available on the current in-memory dataset."
                    ),
                    adata_obj=adata,
                    missing_prerequisites=["clustering"],
                    recovery_options=[
                        "Run clustering first, then rerun CellTypist.",
                        "Choose one of the available clustering keys for annotation.",
                    ],
                    extra={"requested_cluster_key": cluster_key},
                )
            if majority:
                cluster_key = _validate_obs_column(
                    adata,
                    cluster_key,
                    warnings,
                    required=True,
                    context="cluster_key",
                )

            try:
                run_celltypist(
                    adata,
                    model=model,
                    majority_voting=majority,
                    over_clustering=cluster_key if majority else None,
                )
            except ValueError as e:
                # Most often: missing raw-counts layer when adata.X is already
                # log-normalized. Surface a recoverable error rather than
                # crashing the tool loop.
                return _error_result(
                    tool="run_celltypist",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Ensure raw integer counts are in adata.layers['raw_counts'] "
                        "before running CellTypist (normalize_and_hvg preserves them "
                        "automatically; data loaded externally may not).",
                        "If you have raw counts under a different layer name, "
                        "pass it as raw_layer when invoking via run_code.",
                    ],
                )

            output_path = fix_output_path(tool_input.get("output_path"), "run_celltypist")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Get detailed type breakdown
            key = 'celltypist_majority_voting' if majority and 'celltypist_majority_voting' in adata.obs else 'celltypist_predicted_labels'
            all_counts = adata.obs[key].value_counts() if key in adata.obs else {}
            total_cells = adata.n_obs

            # Build detailed breakdown with counts and percentages
            type_breakdown = {}
            for ct, count in all_counts.items():
                type_breakdown[str(ct)] = {
                    "count": int(count),
                    "percent": round(100.0 * count / total_cells, 1)
                }
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)
            celltypist_result = {
                "status": "ok",
                "tool": "run_celltypist",
                "output_path": output_path,
                "saved": output_path is not None,
                "model": model,
                "majority_voting": majority,
                "cluster_key_used": cluster_key if majority else None,
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "cell_type_breakdown": type_breakdown,
                "warnings": warnings,
                "state": make_state(adata)
            }
            return _finalize_result(
                celltypist_result,
                adata,
                dataset_changed=True,
                summary="Ran CellTypist annotation and recorded the annotation source in session state.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "CellTypist annotations were written to AnnData.",
                    [
                        _check("annotation_key_present", key in adata.obs.columns, f"Annotation column '{key}' exists."),
                        _check(
                            "cluster_key_valid",
                            not majority or cluster_key in adata.obs.columns,
                            f"Cluster key '{cluster_key}' is valid for majority voting." if majority else "Majority voting was disabled.",
                        ),
                    ],
                ),
            )

        elif tool_name == "run_scimilarity":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            model_path = tool_input.get("model_path")
            cluster_key = tool_input.get("cluster_key", "leiden")
            cluster_key = _validate_obs_column(
                adata,
                cluster_key,
                warnings,
                required=False,
                context="cluster_key",
            ) or "leiden"

            # Only pass model_path if specified, otherwise use default
            if model_path:
                run_scimilarity(adata, model_path=model_path, cluster_key=cluster_key)
            else:
                run_scimilarity(adata, cluster_key=cluster_key)

            output_path = fix_output_path(tool_input.get("output_path"), "run_scimilarity")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Get detailed type breakdown
            key = 'scimilarity_predictions_unconstrained'
            if key not in adata.obs:
                key = 'scimilarity_representative_prediction'

            all_counts = adata.obs[key].value_counts() if key in adata.obs else {}
            total_cells = adata.n_obs

            # Build detailed breakdown with counts and percentages
            type_breakdown = {}
            for ct, count in all_counts.items():
                type_breakdown[str(ct)] = {
                    "count": int(count),
                    "percent": round(100.0 * count / total_cells, 1)
                }
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)
            scimilarity_result = {
                "status": "ok",
                "tool": "run_scimilarity",
                "output_path": output_path,
                "saved": output_path is not None,
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "cluster_key_used": cluster_key,
                "has_embeddings": "X_scimilarity" in adata.obsm,
                "cell_type_breakdown": type_breakdown,
                "warnings": warnings,
                "state": make_state(adata)
            }
            return _finalize_result(
                scimilarity_result,
                adata,
                dataset_changed=True,
                summary="Ran Scimilarity annotation and recorded the representative predictions.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "Scimilarity outputs were written to AnnData.",
                    [
                        _check("annotation_key_present", key in adata.obs.columns, f"Annotation column '{key}' exists."),
                        _check(
                            "cluster_key_valid",
                            cluster_key in adata.obs.columns,
                            f"Cluster key '{cluster_key}' is available for cluster-level summaries.",
                        ),
                    ],
                ),
            )

        elif tool_name == "query_cells":
            from ..annotation.scimilarity import query_cells as _query_cells, DEFAULT_MODEL_PATH

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            if adata is None:
                return _error_result(
                    tool="query_cells",
                    message="No data loaded. Load a dataset first.",
                    recovery_options=["Load data with inspect_data or provide a data_path."],
                )

            query_type = tool_input.get("query_type", "cells")
            k = tool_input.get("k", 50)
            raw_layer = tool_input.get("raw_layer")

            # Validate mode-specific inputs early for a clear error
            if query_type == "centroid":
                if not tool_input.get("group_key") or not tool_input.get("group_value"):
                    return _error_result(
                        tool="query_cells",
                        message="centroid mode requires group_key and group_value.",
                        adata_obj=adata,
                        recovery_options=[
                            "Provide both group_key (obs column) and group_value (category within it).",
                            "Use list_obs_columns to find available grouping columns.",
                        ],
                    )
            else:
                if not tool_input.get("cell_ids") and not tool_input.get("obs_column"):
                    return _error_result(
                        tool="query_cells",
                        message="cells mode requires either cell_ids (list of obs_names) or obs_column.",
                        adata_obj=adata,
                        recovery_options=[
                            "Provide a list of cell_ids (obs_names) or an obs_column to select cells from.",
                        ],
                    )

            try:
                result = _query_cells(
                    adata,
                    query_type=query_type,
                    cell_ids=tool_input.get("cell_ids"),
                    obs_column=tool_input.get("obs_column"),
                    group_key=tool_input.get("group_key"),
                    group_value=tool_input.get("group_value"),
                    k=k,
                    model_path=DEFAULT_MODEL_PATH,
                    raw_layer=raw_layer,
                )
            except (FileNotFoundError, ImportError) as e:
                return _error_result(
                    tool="query_cells",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify the SCimilarity model path exists (see download_model.sh).",
                    ],
                    install_hint="pip install scimilarity" if "Import" in type(e).__name__ else None,
                )
            except Exception as e:
                return _error_result(
                    tool="query_cells",
                    message=f"Cell query failed: {e}",
                    adata_obj=adata,
                    recovery_options=[
                        "Verify adata has raw counts (raw_layer) and correct obs structure.",
                    ],
                )

            # Build a human-readable summary
            top_ct = result.get("top_celltypes", {})
            top_tissue = result.get("top_tissues", {})
            top_disease = result.get("top_diseases", {})
            coherence = result.get("coherence")
            mean_dist = result.get("mean_dist")

            summary_parts = [
                f"Retrieved {result['n_results']} reference cells (k={k}).",
                f"Mean distance: {mean_dist:.4f}." if mean_dist is not None else "",
                f"Query coherence: {coherence}%." if coherence is not None else "",
                "Top cell types: " + ", ".join(f"{ct} ({n})" for ct, n in list(top_ct.items())[:5]) + "." if top_ct else "",
                "Top tissues: " + ", ".join(f"{t} ({n})" for t, n in list(top_tissue.items())[:5]) + "." if top_tissue else "",
            ]
            summary = " ".join(p for p in summary_parts if p)

            return json.dumps({
                "status": "ok",
                "tool": "query_cells",
                **result,
                "message": summary,
            }, indent=2), adata

        elif tool_name == "score_gene_signature":
            import numpy as np
            import scanpy as sc
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            if adata is None:
                return _error_result(
                    tool="score_gene_signature",
                    message="No data loaded. Load a dataset first.",
                    recovery_options=["Load data with inspect_data or provide a data_path."],
                )

            from ..core.inspector import inspect_data as _inspect_for_norm
            _norm_state = _inspect_for_norm(adata)
            if not _norm_state.is_normalized:
                return _error_result(
                    tool="score_gene_signature",
                    message=(
                        "Data does not appear to be normalized. score_gene_signature works on "
                        "log-normalized expression values (adata.X), not raw counts. "
                        "Run normalize_and_hvg first."
                    ),
                    adata_obj=adata,
                    recovery_options=["Run normalize_and_hvg before scoring gene signatures."],
                )

            cell_cycle = tool_input.get("cell_cycle", False)

            if cell_cycle:
                # ---- Cell cycle scoring ----
                s_genes = tool_input.get("s_genes") or []
                g2m_genes = tool_input.get("g2m_genes") or []
                if not s_genes or not g2m_genes:
                    return _error_result(
                        tool="score_gene_signature",
                        message=(
                            "cell_cycle=true requires both s_genes and g2m_genes. "
                            "Provide lists of S-phase and G2M-phase marker genes."
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "Provide both s_genes and g2m_genes lists.",
                            "Use standard Tirosh et al. 2016 cell cycle gene sets (search_papers can find them).",
                        ],
                    )

                # Filter to genes present in the dataset
                var_names = set(adata.var_names)
                s_found = [g for g in s_genes if g in var_names]
                g2m_found = [g for g in g2m_genes if g in var_names]

                if not s_found or not g2m_found:
                    return _error_result(
                        tool="score_gene_signature",
                        message=(
                            f"Cell cycle scoring failed: found {len(s_found)}/{len(s_genes)} S-phase genes "
                            f"and {len(g2m_found)}/{len(g2m_genes)} G2M-phase genes in the dataset. "
                            "Need at least one gene per phase. Check gene name format (human HGNC symbols)."
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "Verify gene names match the dataset format (inspect sample var_names with inspect_data).",
                            "Genes may be in mouse format (e.g., Ccnb1) vs. human (CCNB1) — convert if needed.",
                        ],
                    )

                try:
                    sc.tl.score_genes_cell_cycle(
                        adata,
                        s_genes=s_found,
                        g2m_genes=g2m_found,
                    )
                except Exception as e:
                    return _error_result(
                        tool="score_gene_signature",
                        message=f"Cell cycle scoring failed: {e}",
                        adata_obj=adata,
                        recovery_options=["Check gene name formats and ensure data is log-normalized."],
                    )

                phase_counts = adata.obs["phase"].value_counts().to_dict()
                return json.dumps({
                    "status": "ok",
                    "tool": "score_gene_signature",
                    "mode": "cell_cycle",
                    "s_genes_matched": len(s_found),
                    "s_genes_total": len(s_genes),
                    "g2m_genes_matched": len(g2m_found),
                    "g2m_genes_total": len(g2m_genes),
                    "scores_added": ["S_score", "G2M_score", "phase"],
                    "phase_distribution": phase_counts,
                    "message": (
                        f"Cell cycle scoring complete. Phase distribution: "
                        + ", ".join(f"{k}: {v}" for k, v in sorted(phase_counts.items()))
                        + ". Scores stored in adata.obs['S_score'], ['G2M_score'], ['phase']."
                    ),
                }, indent=2), adata

            else:
                # ---- Generic gene signature scoring ----
                gene_list = tool_input.get("gene_list") or []
                if not gene_list:
                    return _error_result(
                        tool="score_gene_signature",
                        message="Provide gene_list (list of gene names) or set cell_cycle=true.",
                        adata_obj=adata,
                        recovery_options=[
                            "Provide a gene_list of marker genes to score.",
                            "Set cell_cycle=true with s_genes and g2m_genes for cell cycle scoring.",
                        ],
                    )

                score_name = tool_input.get("score_name", "gene_signature_score")
                layer = tool_input.get("layer")
                n_bins = tool_input.get("n_bins", 25)
                ctrl_size = tool_input.get("ctrl_size", 50)

                # Validate layer if provided
                if layer and layer not in adata.layers:
                    return _error_result(
                        tool="score_gene_signature",
                        message=(
                            f"Layer '{layer}' not found. Available layers: {list(adata.layers.keys())}. "
                            "Leave layer unset to use adata.X (recommended)."
                        ),
                        adata_obj=adata,
                        recovery_options=["Use one of the available layers or omit layer to use adata.X."],
                    )

                # Filter gene_list to genes present in the dataset and report coverage
                var_names = set(adata.var_names)
                matched = [g for g in gene_list if g in var_names]
                missing = [g for g in gene_list if g not in var_names]

                if not matched:
                    return _error_result(
                        tool="score_gene_signature",
                        message=(
                            f"None of the {len(gene_list)} provided genes were found in the dataset. "
                            f"Check gene name format — dataset uses: {list(adata.var_names[:5])}..."
                        ),
                        adata_obj=adata,
                        recovery_options=[
                            "Verify gene name format matches the dataset (human HGNC vs. mouse, Ensembl IDs vs. symbols).",
                            "Use inspect_data to see example var_names.",
                        ],
                        extra={"genes_not_found": gene_list[:20]},
                    )

                # Warn if coverage is low but still proceed
                coverage_pct = len(matched) / len(gene_list) * 100

                try:
                    sc.tl.score_genes(
                        adata,
                        gene_list=matched,
                        score_name=score_name,
                        n_bins=n_bins,
                        ctrl_size=ctrl_size,
                        layer=layer,
                    )
                except Exception as e:
                    return _error_result(
                        tool="score_gene_signature",
                        message=f"Gene scoring failed: {e}",
                        adata_obj=adata,
                        recovery_options=["Verify gene list and layer contain valid expression data."],
                    )

                scores = adata.obs[score_name]
                return json.dumps({
                    "status": "ok",
                    "tool": "score_gene_signature",
                    "mode": "signature",
                    "score_name": score_name,
                    "genes_requested": len(gene_list),
                    "genes_matched": len(matched),
                    "genes_missing": len(missing),
                    "coverage_pct": round(coverage_pct, 1),
                    "missing_genes": missing[:20] if missing else [],
                    "score_stats": {
                        "mean": round(float(scores.mean()), 4),
                        "std": round(float(scores.std()), 4),
                        "min": round(float(scores.min()), 4),
                        "max": round(float(scores.max()), 4),
                        "pct_positive": round(float((scores > 0).mean() * 100), 1),
                    },
                    "message": (
                        f"Scored {len(matched)}/{len(gene_list)} genes ({coverage_pct:.0f}% coverage). "
                        f"Score stored in adata.obs['{score_name}']. "
                        f"Mean score: {scores.mean():.4f}, "
                        f"{(scores > 0).mean() * 100:.1f}% of cells are positive."
                        + (f" Warning: {len(missing)} genes not found in dataset." if missing else "")
                    ),
                }, indent=2), adata

        elif tool_name == "run_spectra":
            from ..analysis.spectra import run_spectra

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)

            cell_type_key = tool_input.get("cell_type_key")
            if not cell_type_key or cell_type_key not in adata.obs.columns:
                return _error_result(
                    tool="run_spectra",
                    message=f"cell_type_key '{cell_type_key}' not found in adata.obs.",
                    adata_obj=adata,
                    recovery_options=[
                        "Run cell type annotation (run_celltypist or run_scimilarity) first.",
                        "Use list_obs_columns to find the correct annotation column.",
                    ],
                )

            output_dir = fix_output_path(tool_input.get("output_dir"), "run_spectra")

            try:
                result = run_spectra(
                    adata,
                    cell_type_key=cell_type_key,
                    gene_set_dict_path=tool_input.get("gene_set_dict_path"),
                    use_default_gene_sets=bool(tool_input.get("use_default_gene_sets", False)),
                    lam=float(tool_input.get("lam", 0.1)),
                    rho=0.001,
                    num_epochs=int(tool_input.get("num_epochs", 1000)),
                    n_top_vals=int(tool_input.get("n_top_vals", 50)),
                    use_highly_variable=bool(tool_input.get("use_highly_variable", True)),
                    use_weights=True,
                    use_cell_types=bool(tool_input.get("use_cell_types", True)),
                    label_factors=True,
                    overlap_threshold=float(tool_input.get("overlap_threshold", 0.2)),
                    output_dir=output_dir,
                )
            except ImportError as e:
                return _error_result(
                    tool="run_spectra",
                    message=str(e),
                    adata_obj=adata,
                    install_hint="pip install Spectra-sc",
                )
            except Exception as e:
                return _error_result(
                    tool="run_spectra",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify cell_type_key is valid and gene set paths are correct.",
                    ],
                )

            artifacts_created = []
            if result.get("figure_path"):
                artifact = _artifact_payload(
                    result["figure_path"],
                    role="figure",
                    metadata={"kind": "spectra_factor_umaps"},
                )
                if artifact:
                    artifacts_created.append(artifact)

            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "run_spectra",
                    "cell_type_key": cell_type_key,
                    "n_factors": result["n_factors"],
                    "factor_labels": result["factor_labels"],
                    "top_markers_per_factor": result["top_markers_per_factor"],
                    "model_path": result.get("model_path"),
                    "figure_path": result.get("figure_path"),
                    "obsm_key": "SPECTRA_cell_scores",
                    "note": (
                        "Factor scores in adata.obsm['SPECTRA_cell_scores'] — "
                        "color UMAP by individual columns to visualize each gene program. "
                        "Top marker genes per factor are in adata.uns['SPECTRA_markers']."
                    ),
                    "warnings": warnings,
                    "state": make_state(adata),
                },
                adata,
                dataset_changed=True,
                summary=f"Spectra discovered {result['n_factors']} gene program factors using '{cell_type_key}' as cell type key.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "Spectra completed successfully.",
                    [
                        _check(
                            "cell_scores_present",
                            "SPECTRA_cell_scores" in adata.obsm,
                            f"Factor scores written to adata.obsm['SPECTRA_cell_scores'] ({result['n_factors']} factors).",
                        ),
                        _check(
                            "markers_present",
                            "SPECTRA_markers" in adata.uns,
                            "Top marker genes stored in adata.uns['SPECTRA_markers'].",
                        ),
                    ],
                ),
            )

        elif tool_name == "save_data":
            if adata is None:
                return _error_result(
                    tool="save_data",
                    message="No in-memory data available to save. Run an analysis tool first.",
                    recovery_options=["Load and process data before saving."],
                )

            output_path = fix_output_path(tool_input.get("output_path"), "save_data")
            if not output_path:
                return _error_result(
                    tool="save_data",
                    message="Provide an .h5ad output_path or a directory where the final_result.h5ad can be written.",
                    adata_obj=adata,
                    recovery_options=["Provide output_path as a .h5ad file path or directory."],
                )

            save_details = write_h5ad_safe(adata, output_path)
            artifact = _artifact_payload(output_path, role="saved_dataset", metadata={"save_mode": save_details.get("save_mode", "direct")})
            save_result = {
                "status": "ok",
                "tool": "save_data",
                "output_path": output_path,
                "saved": True,
                "save_mode": save_details.get("save_mode", "direct"),
                "warnings": save_details.get("warnings", []),
                "shape": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "state": make_state(adata)
            }
            return _finalize_result(
                save_result,
                adata,
                dataset_changed=False,
                summary="Saved the current in-memory AnnData as a final dataset artifact.",
                artifacts_created=[artifact] if artifact is not None else [],
                verification=_build_verification(
                    "passed",
                    "The AnnData output file exists and was registered as an artifact.",
                    [
                        _check("output_exists", os.path.exists(output_path), f"Saved output exists at {output_path}."),
                    ],
                ),
            )

        elif tool_name == "run_batch_correction":
            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            method = tool_input.get("method", "harmony")
            requested_batch_key = tool_input.get("batch_key") or _confirmed_decision_value("batch_key")
            if not requested_batch_key:
                raise ValueError(
                    "No batch_key was provided and no confirmed batch_key exists in session state. "
                    "Confirm the correct batch column first."
                )
            batch_key = _validate_obs_column(adata, requested_batch_key, warnings, required=True, context="batch_key")

            # Get batch sizes for output
            batch_sizes = adata.obs[batch_key].value_counts().to_dict()
            if method == "harmony":
                run_harmony(adata, batch_key=batch_key)
                corrected_rep = "X_pca_harmony"
            elif method == "bbknn":
                n_pcs = int(tool_input.get("n_pcs") or 30)
                neighbors_within_batch = int(tool_input.get("neighbors_within_batch") or 3)
                # BBKNN requires PCA — validate before running
                if "X_pca" not in adata.obsm:
                    return _error_result(
                        tool="run_batch_correction",
                        message="BBKNN requires PCA in adata.obsm['X_pca']. Run run_pca first.",
                        adata_obj=adata,
                        recovery_options=["Run run_pca to compute PCA, then retry BBKNN."],
                        extra={"method": "bbknn"},
                    )
                run_bbknn(
                    adata,
                    batch_key=batch_key,
                    n_pcs=n_pcs,
                    neighbors_within_batch=neighbors_within_batch,
                )
                # BBKNN correction lives in the neighbor graph, not an obsm key.
                # corrected_rep=None signals downstream logic to use the BBKNN graph as-is.
                corrected_rep = None
            elif method == "scvi":
                n_latent = int(tool_input.get("n_latent") or 30)
                max_epochs = int(tool_input.get("max_epochs") or 200)
                store_normalized = bool(tool_input.get("store_normalized", False))
                # scVI requires raw integer counts — validate before training
                try:
                    _resolve_integer_counts_layer(adata, "raw_counts")
                except ValueError as e:
                    return _error_result(
                        tool="run_batch_correction",
                        message=str(e),
                        adata_obj=adata,
                        recovery_options=[
                            "Ensure raw integer counts are in adata.layers['raw_counts'] "
                            "(normalize_and_hvg preserves them automatically).",
                            "Try method='harmony' or 'bbknn' instead — they work on PCA embeddings and do not need raw counts.",
                        ],
                        extra={"method": "scvi"},
                    )
                run_scvi(
                    adata,
                    batch_key=batch_key,
                    n_latent=n_latent,
                    max_epochs=max_epochs,
                    store_normalized=store_normalized,
                )
                corrected_rep = "X_scVI"
            else:
                run_scanorama(adata, batch_key=batch_key)
                corrected_rep = "X_scanorama"

            neighbors_recomputed = method == "bbknn"
            umap_recomputed = False

            output_path = fix_output_path(tool_input.get("output_path"), "run_batch_correction")
            if output_path:
                write_h5ad_safe(adata, output_path)
            artifacts_created = []
            if output_path:
                artifact = _artifact_payload(output_path, role="checkpoint", metadata={"format": "h5ad"})
                if artifact is not None:
                    artifacts_created.append(artifact)
            batch_strategy = {
                "status": "user_selected" if tool_input.get("batch_key") else "auto_selected",
                "requested_column": tool_input.get("batch_key"),
                "applied_column": batch_key,
                "recommended_column": batch_key,
                "recommended_role": "batch",
                "needs_user_confirmation": False,
                "reason": f"Using '{batch_key}' for batch correction.",
                "candidates": [{"column": batch_key}],
            }
            decisions = []
            batch_decision = decision_for_batch_strategy(
                batch_strategy,
                context="batch_correction",
                source_tool="run_batch_correction",
            )
            if batch_decision is not None:
                decisions.append(batch_decision)
            extra = {}
            if method == "scvi":
                extra["n_latent"] = int(tool_input.get("n_latent") or 30)
                extra["max_epochs"] = int(tool_input.get("max_epochs") or 200)
                extra["scvi_normalized_stored"] = bool(tool_input.get("store_normalized", False))
            elif method == "bbknn":
                extra["n_pcs"] = int(tool_input.get("n_pcs") or 30)
                extra["neighbors_within_batch"] = int(tool_input.get("neighbors_within_batch") or 3)
                extra["total_neighbors_per_cell"] = len(batch_sizes) * extra["neighbors_within_batch"]

            # BBKNN correction lives in the neighbor graph (adata.obsp), not an obsm key
            corrected_embedding_label = (
                "BBKNN graph (adata.obsp['connectivities'])"
                if corrected_rep is None
                else corrected_rep
            )
            batch_result = {
                "status": "ok",
                "tool": "run_batch_correction",
                "output_path": output_path,
                "saved": output_path is not None,
                "method": method,
                "batch_key": batch_key,
                "n_batches": len(batch_sizes),
                "batch_sizes": {str(k): int(v) for k, v in batch_sizes.items()},
                "corrected_embedding": corrected_embedding_label,
                "neighbors_recomputed": neighbors_recomputed,
                "umap_recomputed": False,
                "note": "Batch correction complete. Run run_umap next (and run_neighbors first for Harmony/Scanorama/scVI).",
                "warnings": warnings,
                "state": make_state(adata),
                **extra,
            }

            # For BBKNN the correction is in the neighbor graph, not an obsm key
            if corrected_rep is None:
                corrected_present = adata.uns.get("bbknn_batch_key") == batch_key
                corrected_check_label = "BBKNN neighbor graph stored in adata.obsp['connectivities']."
            else:
                corrected_present = corrected_rep in adata.obsm
                corrected_check_label = f"Corrected embedding '{corrected_rep}' exists in adata.obsm."

            return _finalize_result(
                batch_result,
                adata,
                dataset_changed=True,
                summary=(
                    f"Applied {method} batch correction using '{batch_key}'. "
                    f"Downstream UMAP recomputed: {umap_recomputed}."
                ),
                artifacts_created=artifacts_created,
                decisions_raised=decisions,
                verification=_build_verification(
                    "passed",
                    "Batch correction completed and the corrected representation is available.",
                    [
                        _check("batch_key_present", batch_key in adata.obs.columns, f"Batch key '{batch_key}' exists in adata.obs."),
                        _check("corrected_embedding_present", corrected_present, corrected_check_label),
                        _check("umap_not_recomputed", not umap_recomputed, "UMAP was not recomputed — run run_umap to refresh the layout."),
                    ],
                ),
            )

        elif tool_name == "score_integration":
            from ..batch.entropy import compute_batch_entropy

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            batch_key = tool_input.get("batch_key")
            use_rep = tool_input.get("use_rep", "X_umap")
            n_neighbors = int(tool_input.get("n_neighbors", 50))

            if not batch_key or batch_key not in adata.obs.columns:
                return _error_result(
                    tool="score_integration",
                    message=f"batch_key '{batch_key}' not found in adata.obs.",
                    adata_obj=adata,
                    recovery_options=[
                        "Use list_obs_columns to find the correct batch column name.",
                        "Confirm the batch column with the user before scoring.",
                    ],
                )

            try:
                result = compute_batch_entropy(
                    adata,
                    batch_key=batch_key,
                    use_rep=use_rep,
                    n_neighbors=n_neighbors,
                )
            except (ValueError, ImportError) as e:
                return _error_result(
                    tool="score_integration",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify batch_key has ≥2 unique values and use_rep embedding exists in adata.obsm.",
                    ],
                )

            # Store per-cell entropy in obs for downstream visualization
            adata.obs["integration_entropy"] = result["per_cell_entropy"]

            entropy_mean = result["entropy_mean"]
            if entropy_mean >= 0.8:
                interpretation = "Excellent mixing — batches are well-integrated."
            elif entropy_mean >= 0.6:
                interpretation = "Good mixing — minor batch structure may remain."
            elif entropy_mean >= 0.4:
                interpretation = "Moderate mixing — consider a stronger correction method (e.g. scVI)."
            else:
                interpretation = "Poor mixing — strong batch structure persists. Try scVI or check batch_key."

            return json.dumps({
                "status": "ok",
                "tool": "score_integration",
                "use_rep": use_rep,
                "batch_key": batch_key,
                "n_neighbors": result["n_neighbors"],
                "n_batches": result["n_batches"],
                "entropy_mean": round(entropy_mean, 4),
                "entropy_median": round(result["entropy_median"], 4),
                "entropy_per_batch": {k: round(v, 4) for k, v in result["entropy_per_batch"].items()},
                "interpretation": interpretation,
                "note": (
                    "Per-cell scores stored in adata.obs['integration_entropy']. "
                    "Color UMAP by 'integration_entropy' to see spatial mixing patterns."
                ),
                "state": make_state(adata),
            }, indent=2), adata

        elif tool_name == "benchmark_integration":
            from ..batch.scib import run_scib_benchmark

            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            batch_key = tool_input.get("batch_key")
            label_key = tool_input.get("label_key")
            embedding_keys = tool_input.get("embedding_keys") or None
            fast = bool(tool_input.get("fast", False))
            output_dir = fix_output_path(tool_input.get("output_dir"), "benchmark_integration")

            for col, name in [(batch_key, "batch_key"), (label_key, "label_key")]:
                if not col or col not in adata.obs.columns:
                    return _error_result(
                        tool="benchmark_integration",
                        message=f"{name} '{col}' not found in adata.obs.",
                        adata_obj=adata,
                        recovery_options=[
                            "Use list_obs_columns to find valid batch and label columns.",
                            "Run cell type annotation first if label_key is missing.",
                        ],
                    )

            try:
                bench = run_scib_benchmark(
                    adata,
                    batch_key=batch_key,
                    label_key=label_key,
                    embedding_keys=embedding_keys,
                    fast=fast,
                    output_dir=output_dir,
                )
            except ImportError as e:
                return _error_result(
                    tool="benchmark_integration",
                    message=str(e),
                    adata_obj=adata,
                    install_hint="pip install scib-metrics",
                )
            except Exception as e:
                return _error_result(
                    tool="benchmark_integration",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify batch/label keys and that embeddings are properly formatted.",
                    ],
                )

            artifacts_created = []
            if bench.get("output_figure"):
                artifact = _artifact_payload(
                    bench["output_figure"],
                    role="figure",
                    metadata={"kind": "scib_benchmark_table"},
                )
                if artifact:
                    artifacts_created.append(artifact)
            if bench.get("output_csv"):
                artifact = _artifact_payload(
                    bench["output_csv"],
                    role="artifact",
                    metadata={"kind": "scib_results_csv"},
                )
                if artifact:
                    artifacts_created.append(artifact)

            return _finalize_result(
                {
                    "status": "ok",
                    "tool": "benchmark_integration",
                    "batch_key": batch_key,
                    "label_key": label_key,
                    "embeddings_benchmarked": bench["embeddings_benchmarked"],
                    "scores_by_embedding": bench["scores_by_embedding"],
                    "best_method": bench["best_method"],
                    "results_table": bench["results_table"],
                    "output_csv": bench.get("output_csv"),
                    "output_figure": bench.get("output_figure"),
                    "note": (
                        f"Best embedding by total scib score: {bench['best_method']}. "
                        "Use run_batch_correction with the corresponding method if not already applied."
                    ) if bench["best_method"] else "",
                    "state": make_state(adata),
                },
                adata,
                dataset_changed=False,
                summary=f"scib-metrics benchmark complete across {len(bench['embeddings_benchmarked'])} embeddings. Best: {bench['best_method']}.",
                artifacts_created=artifacts_created,
                verification=_build_verification(
                    "passed",
                    "scib-metrics benchmark completed successfully.",
                    [
                        _check(
                            "embeddings_benchmarked",
                            len(bench["embeddings_benchmarked"]) > 0,
                            f"Benchmarked {len(bench['embeddings_benchmarked'])} embeddings.",
                        ),
                        _check(
                            "best_method_identified",
                            bench["best_method"] is not None,
                            f"Best method: {bench['best_method']}.",
                        ),
                    ],
                ),
            )

        elif tool_name == "run_deg":
            from ..analysis.deg import run_validated_deg, get_deg_caveats
            from ..config.defaults import DEG_DEFAULTS

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            requested_groupby = tool_input.get("groupby", "leiden")
            if requested_groupby not in adata.obs.columns:
                return _smart_unavailable_result(
                    tool="run_deg",
                    message=(
                        f"Differential expression needs a valid grouping column, but '{requested_groupby}' "
                        "is not available on the current in-memory dataset."
                    ),
                    adata_obj=adata,
                    missing_prerequisites=["grouping"],
                    recovery_options=[
                        "Run clustering first, then run DEG on the cluster key.",
                        "Use one of the available annotation or metadata columns as the DEG grouping.",
                    ],
                    extra={"requested_groupby": requested_groupby},
                )
            groupby = _validate_obs_column(
                adata,
                requested_groupby,
                warnings,
                required=True,
                context="groupby"
            )
            method = tool_input.get("method", "wilcoxon")
            layer = tool_input.get("layer")
            use_raw = tool_input.get("use_raw")
            key_added = tool_input.get("key_added", "rank_genes_groups")
            n_genes = int(tool_input.get("n_genes", 100))
            target_geneset = tool_input.get("target_geneset", DEG_DEFAULTS.default_geneset)

            # Run validated DEG - this validates inputs, runs rank_genes_groups,
            # validates outputs, and attaches validity metadata to adata.uns
            try:
                _, validity_report = run_validated_deg(
                    adata,
                    groupby=groupby,
                    method=method,
                    n_genes=n_genes,
                    layer=layer,
                    use_raw=use_raw,
                    key_added=key_added,
                    target_geneset=target_geneset,
                    min_cluster_size=DEG_DEFAULTS.min_cluster_size,
                    warn_cluster_size=DEG_DEFAULTS.warn_cluster_size,
                    imbalance_ratio=DEG_DEFAULTS.max_imbalance_ratio,
                    block_on_errors=True,
                    batch_confound_threshold=DEG_DEFAULTS.batch_confound_threshold,
                    max_logfc=DEG_DEFAULTS.max_logfc_sanity,
                    inplace=True,
                )
            except Exception as e:
                return _error_result(
                    tool="run_deg",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify groupby column is valid and has ≥2 groups.",
                        "If a layer was specified, ensure it exists and contains valid expression data.",
                    ],
                )

            output_path = fix_output_path(tool_input.get("output_path"), "run_deg")
            if output_path:
                write_h5ad_safe(adata, output_path)

            # Get top 5 markers per cluster for immediate insight
            groups = list(adata.obs[groupby].unique())
            top_markers_summary = {}
            for group in groups[:15]:  # Limit to first 15 clusters for response size
                try:
                    markers_df = get_top_markers(adata, group=str(group), n_genes=5, key=key_added)
                    top_markers_summary[str(group)] = [
                        {
                            "gene": row['names'],
                            "logfc": round(row['logfoldchanges'], 2),
                            "pval_adj": float(f"{row['pvals_adj']:.2e}")
                        }
                        for _, row in markers_df.iterrows()
                    ]
                except Exception:
                    pass

            # Build validation summary for response
            validity_summary = {
                "is_valid": validity_report.is_valid,
                "has_warnings": validity_report.has_warnings,
                "n_errors": len(validity_report.errors),
                "n_warnings": len(validity_report.warnings),
                "matrix_type": validity_report.matrix_type,
                "matrix_source": validity_report.matrix_source,
                "use_raw": validity_report.use_raw,
                "layer_used": validity_report.layer_used,
                "data_species": validity_report.data_species,
                "gene_id_format": validity_report.gene_id_format,
            }

            # Include specific issues for agent awareness
            if validity_report.errors:
                validity_summary["errors"] = [e.message for e in validity_report.errors]
            if validity_report.warnings:
                validity_summary["warnings"] = [w.message for w in validity_report.warnings]

            # Get caveats that should propagate to GSEA
            deg_caveats = get_deg_caveats(adata)

            return json.dumps({
                "status": "ok",
                "tool": "run_deg",
                "output_path": output_path,
                "saved": output_path is not None,
                "groupby": groupby,
                "method": method,
                "key_added": key_added,
                "n_genes": n_genes,
                "n_groups": len(groups),
                "requested_layer": layer,
                "requested_use_raw": use_raw,
                "layer_used": validity_report.layer_used,
                "use_raw": validity_report.use_raw,
                "matrix_source": validity_report.matrix_source,
                "matrix_type": validity_report.matrix_type,
                "cluster_sizes": validity_report.cluster_sizes,
                "validity": validity_summary,
                "caveats_for_gsea": deg_caveats,
                "top_markers_per_cluster": top_markers_summary,
                "note": "Validity metadata stored in adata.uns['deg_validity'] - will propagate to GSEA",
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_pseudobulk_deg":
            from ..analysis.pseudobulk import run_pseudobulk_deg

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)

            sample_col = tool_input["sample_col"]
            condition_col = tool_input["condition_col"]
            condition_a = tool_input["condition_a"]
            condition_b = tool_input["condition_b"]
            groups_col = tool_input["groups_col"]
            cell_type = tool_input.get("cell_type")
            layer = tool_input.get("layer", "raw_counts")
            min_cells = int(tool_input.get("min_cells") or 10)
            alpha = float(tool_input.get("alpha") or 0.05)
            output_path = tool_input.get("output_path")

            # Validate integer counts before aggregating
            try:
                _resolve_integer_counts_layer(adata, layer)
            except ValueError as e:
                return _error_result(
                    tool="run_pseudobulk_deg",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Ensure raw integer counts are in the specified layer (normalize_and_hvg preserves them).",
                    ],
                )

            try:
                result = run_pseudobulk_deg(
                    adata,
                    sample_col=sample_col,
                    condition_col=condition_col,
                    condition_a=condition_a,
                    condition_b=condition_b,
                    groups_col=groups_col,
                    cell_type=cell_type,
                    layer=layer,
                    min_cells=min_cells,
                    alpha=alpha,
                    output_path=output_path,
                )
            except (ImportError, ValueError) as e:
                return _error_result(
                    tool="run_pseudobulk_deg",
                    message=str(e),
                    adata_obj=adata,
                    recovery_options=[
                        "Verify sample_col, condition_col, and groups_col exist and are valid.",
                        "Ensure ≥2 samples per condition for DESeq2 replication.",
                    ],
                    install_hint="pip install decoupler pydeseq2" if "Import" in type(e).__name__ else None,
                )

            result["warnings"] = warnings
            result["state"] = make_state(adata)
            return json.dumps(result, indent=2), adata

        elif tool_name == "run_gsea":
            import scanpy as sc
            from ..analysis.deg import get_deg_validity, get_cluster_caveats

            warnings = _state_preservation_warning(tool_input, adata)
            adata, _ = get_adata(tool_input, adata, prefer_memory=True)
            output_dir = tool_input["output_dir"]
            cluster = tool_input["cluster"]
            gene_sets = tool_input.get("gene_sets", "KEGG_2021_Human")
            min_size = tool_input.get("min_size", 5)
            max_size = tool_input.get("max_size", 500)
            permutation_num = tool_input.get("permutation_num", 1000)

            # Check DEG results exist
            if 'rank_genes_groups' not in adata.uns:
                return _error_result(
                    tool="run_gsea",
                    message="No DEG results found. Run run_deg first.",
                    adata_obj=adata,
                    recovery_options=["Run run_deg to generate rank_genes_groups before running GSEA."],
                )

            # Get DEG validity info for caveats
            deg_validity = get_deg_validity(adata)
            deg_caveats = adata.uns.get("deg_caveats", [])

            try:
                import gseapy
            except ImportError:
                return _error_result(
                    tool="run_gsea",
                    message="gseapy not installed.",
                    adata_obj=adata,
                    install_hint="pip install gseapy",
                )

            os.makedirs(output_dir, exist_ok=True)

            # Get clusters to analyze
            groupby = adata.uns['rank_genes_groups']['params']['groupby']
            max_clusters = tool_input.get("max_clusters")
            if cluster == 'all':
                clusters_to_analyze = list(adata.obs[groupby].unique())
                if max_clusters is not None:
                    clusters_to_analyze = clusters_to_analyze[: int(max_clusters)]
            else:
                clusters_to_analyze = [cluster]
            all_results = {}

            for clust in clusters_to_analyze:
                try:
                    # Get DEG results for this cluster
                    deg_df = sc.get.rank_genes_groups_df(adata, group=str(clust))

                    # Create ranked gene list (gene -> score)
                    # Use scores from DEG (stat values work well for GSEA)
                    df_rank = deg_df[['names', 'scores']].dropna()
                    df_rank = df_rank.set_index('names')['scores']

                    # Run GSEA prerank
                    gsea_outdir = os.path.join(output_dir, f"cluster_{clust}")
                    pre_res = gseapy.prerank(
                        rnk=df_rank,
                        gene_sets=gene_sets,
                        threads=1,
                        min_size=min_size,
                        max_size=max_size,
                        permutation_num=permutation_num,
                        outdir=gsea_outdir,
                        seed=42,
                        verbose=False,
                    )

                    # Get top results
                    res_df = pre_res.res2d
                    res_df = res_df.sort_values('NES', ascending=False)

                    # Top 5 upregulated and top 5 downregulated
                    top_up = res_df[res_df['NES'] > 0].head(5)
                    top_down = res_df[res_df['NES'] < 0].tail(5)

                    cluster_results = {
                        "upregulated_pathways": [
                            {
                                "term": row['Term'],
                                "nes": round(row['NES'], 2),
                                "fdr": float(f"{row['FDR q-val']:.2e}"),
                                "genes": row['Lead_genes'].split(';')[:5] if row['Lead_genes'] else []
                            }
                            for _, row in top_up.iterrows()
                        ],
                        "downregulated_pathways": [
                            {
                                "term": row['Term'],
                                "nes": round(row['NES'], 2),
                                "fdr": float(f"{row['FDR q-val']:.2e}"),
                                "genes": row['Lead_genes'].split(';')[:5] if row['Lead_genes'] else []
                            }
                            for _, row in top_down.iterrows()
                        ],
                        "total_significant": int((res_df['FDR q-val'] < 0.25).sum()),
                    }
                    all_results[str(clust)] = cluster_results

                except Exception as e:
                    all_results[str(clust)] = {"error": str(e)}

            # Add per-cluster caveats based on DEG validity
            cluster_caveats = {}
            for clust in clusters_to_analyze:
                caveats = get_cluster_caveats(adata, str(clust))
                if caveats:
                    cluster_caveats[str(clust)] = caveats

            # Build response with validity metadata
            response = {
                "status": "ok",
                "tool": "run_gsea",
                "output_dir": output_dir,
                "gene_sets": gene_sets,
                "clusters_analyzed": clusters_to_analyze,
                "results": all_results,
                "recommended_next_steps": [
                    "Use search_papers on the most significant pathway terms for biological interpretation and recent reviews.",
                    "Use web_search for pathway database or software documentation questions."
                ],
                "note": "NES > 0 means pathway upregulated in this cluster. FDR < 0.25 is typically significant.",
            }

            # Add DEG validity info if present
            if deg_validity:
                response["deg_validity"] = {
                    "is_valid": deg_validity.get("is_valid", True),
                    "has_warnings": deg_validity.get("has_warnings", False),
                    "matrix_type": deg_validity.get("matrix_type"),
                    "data_species": deg_validity.get("data_species"),
                    "gene_id_format": deg_validity.get("gene_id_format"),
                }
            if deg_caveats:
                response["deg_caveats"] = deg_caveats
            if cluster_caveats:
                response["cluster_caveats"] = cluster_caveats

            return json.dumps(response, indent=2), adata

        elif tool_name == "generate_figure":
            adata, _ = get_adata(tool_input, adata)
            plot_type = tool_input["plot_type"]
            output_path = tool_input.get("output_path")
            if not output_path:
                safe_color = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(tool_input.get("color_by", "plot")))
                output_path = f"{plot_type}_{safe_color or 'plot'}.png"
            color_by = tool_input.get("color_by")
            genes = tool_input.get("genes", [])
            include_image = tool_input.get("include_image", True)
            if plot_type == "umap" and "X_umap" not in adata.obsm:
                return _smart_unavailable_result(
                    tool="generate_figure",
                    message="UMAP cannot be rendered because the embedding is not available on the current in-memory dataset.",
                    adata_obj=adata,
                    missing_prerequisites=["embedding"],
                    recovery_options=[
                        "Run PCA, neighbors, and UMAP first.",
                        "If you only need a summary of current state, inspect the session instead of plotting.",
                    ],
                    extra={"plot_type": plot_type, "color_by": color_by},
                )
            if plot_type == "umap" and color_by not in (None, "") and color_by not in adata.obs.columns and color_by not in adata.var_names:
                return _smart_unavailable_result(
                    tool="generate_figure",
                    message=f"UMAP coloring key '{color_by}' is not available on the current in-memory dataset.",
                    adata_obj=adata,
                    missing_prerequisites=["valid_color_key"],
                    recovery_options=[
                        "Use one of the available obs columns or genes for coloring.",
                        "Render a plain UMAP without coloring.",
                    ],
                    extra={"plot_type": plot_type, "requested_color_by": color_by},
                )
            result = _render_figure(
                adata,
                plot_type=plot_type,
                output_path=output_path,
                color_by=color_by,
                genes=genes,
                include_image=include_image,
            )
            result["available_clusterings"] = _clusterings_payload(adata)
            artifact = _artifact_payload(
                output_path,
                role="figure",
                metadata={"plot_type": plot_type, "color_by": color_by},
            )
            verification_checks = [
                _check("figure_exists", os.path.exists(output_path), f"Figure exists at {output_path}."),
            ]
            if plot_type == "umap":
                verification_checks.append(
                    _check(
                        "color_key_valid",
                        color_by in adata.obs.columns or color_by in adata.var_names,
                        f"Color key '{color_by}' exists in AnnData.",
                    )
                )
            return _finalize_result(
                result,
                adata,
                dataset_changed=False,
                summary=f"Generated a {plot_type} figure colored by '{color_by}'.",
                artifacts_created=[artifact] if artifact is not None else [],
                verification=_build_verification(
                    "passed",
                    "Figure output was created and verified.",
                    verification_checks,
                ),
            )

        elif tool_name == "read_file":
            import re as _re
            file_path = Path(tool_input["path"]).expanduser()
            if not file_path.exists():
                return _error_result(
                    tool="read_file",
                    message=f"File not found: {file_path}",
                    adata_obj=adata,
                    recovery_options=["Verify the file path exists and is accessible."],
                )

            max_chars = int(tool_input.get("max_chars") or 20000)
            suffix = file_path.suffix.lower()

            if suffix == ".pdf":
                try:
                    import fitz  # pymupdf
                except ImportError:
                    return _error_result(
                        tool="read_file",
                        message="pymupdf not installed.",
                        adata_obj=adata,
                        install_hint="pip install pymupdf",
                    )

                doc = fitz.open(str(file_path))
                n_pages = len(doc)

                # Parse page selection
                pages_param = tool_input.get("pages", "").strip()
                if pages_param:
                    selected = set()
                    for part in _re.split(r"[,\s]+", pages_param):
                        if "-" in part:
                            a, b = part.split("-", 1)
                            selected.update(range(int(a) - 1, int(b)))
                        elif part.isdigit():
                            selected.add(int(part) - 1)
                    page_indices = sorted(p for p in selected if 0 <= p < n_pages)
                else:
                    page_indices = list(range(n_pages))

                # Text extraction
                parts = []
                for i in page_indices:
                    text = doc[i].get_text().strip()
                    if text:
                        parts.append(f"[Page {i+1}]\n{text}")

                full_text = "\n\n".join(parts)
                truncated = len(full_text) > max_chars
                content = full_text[:max_chars]

                result = {
                    "status": "ok",
                    "tool": "read_file",
                    "path": str(file_path),
                    "type": "pdf",
                    "total_pages": n_pages,
                    "pages_read": [i + 1 for i in page_indices],
                    "truncated": truncated,
                    "chars_returned": len(content),
                    "content": content,
                }

                # Optional page rendering — lets vision model see figures and plots
                render_pages = bool(tool_input.get("render_pages", False))
                if render_pages:
                    import base64 as _b64
                    if run_manager is not None:
                        pdf_pages_dir = run_manager.dirs["figures"] / "pdf_pages"
                    else:
                        import tempfile
                        pdf_pages_dir = Path(tempfile.mkdtemp())
                    pdf_pages_dir.mkdir(parents=True, exist_ok=True)

                    rendered_paths = []
                    for i in page_indices:
                        page = doc[i]
                        mat = fitz.Matrix(1.5, 1.5)  # 108 DPI — good quality, reasonable size
                        pix = page.get_pixmap(matrix=mat)
                        out_path = pdf_pages_dir / f"page_{i + 1}.png"
                        pix.save(str(out_path))
                        rendered_paths.append(str(out_path))

                    if rendered_paths:
                        # First page goes to the vision pipeline via _pending_image
                        result["image_base64"] = _b64.b64encode(
                            open(rendered_paths[0], "rb").read()
                        ).decode()
                        result["image_mime"] = "image/png"
                        result["image_context"] = {"output_path": rendered_paths[0]}
                        result["rendered_page_paths"] = rendered_paths
                        if len(rendered_paths) > 1:
                            result["note"] = (
                                f"Page 1 sent to vision model inline. "
                                f"Remaining {len(rendered_paths) - 1} page(s) saved to "
                                f"figures/pdf_pages/ — use review_figure to inspect them."
                            )

                doc.close()
                return json.dumps(result, indent=2), adata

            else:
                # Plain text, markdown, CSV, TSV, JSON, etc.
                try:
                    raw = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as e:
                    return _error_result(
                        tool="read_file",
                        message=str(e),
                        adata_obj=adata,
                        recovery_options=["Verify the file is readable and in a supported text format."],
                    )

                truncated = len(raw) > max_chars
                content = raw[:max_chars]
                return json.dumps({
                    "status": "ok",
                    "tool": "read_file",
                    "path": str(file_path),
                    "type": suffix.lstrip(".") or "text",
                    "truncated": truncated,
                    "chars_returned": len(content),
                    "content": content,
                }, indent=2), adata

        else:
            return _error_result(
                tool=tool_name,
                message=f"Unknown tool: {tool_name}",
                adata_obj=adata,
                recovery_options=["Check tool name spelling or use inspect_session to list available tools."],
            )

    except Exception as e:
        return _error_result(
            tool=tool_name,
            message=str(e),
            adata_obj=adata,
            recovery_options=["Review the error and tool input parameters before retrying."],
            extra={"error_type": type(e).__name__},
        )
