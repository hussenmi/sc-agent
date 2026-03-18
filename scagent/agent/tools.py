"""
Claude API tool definitions for scagent.

Tools are organized into two layers:
1. Action tools - mutate/generate analysis artifacts
2. Inspection tools - read-only queries for more detail

All tools return structured JSON for LLM reasoning.
"""

from typing import List, Dict, Any
import json


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
            "name": "run_qc",
            "description": "Run quality control pipeline: QC metrics, doublet detection, cell/gene filtering. Returns structured summary with before/after counts and metrics.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad or 10X h5 file"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad"},
                    "mt_threshold": {"type": "number", "description": "Max MT% (default: auto-detect)"},
                    "remove_ribo": {"type": "boolean", "description": "Remove ribosomal genes (default: true)"},
                    "batch_key": {"type": "string", "description": "Batch column for per-batch doublet detection"}
                },
                "required": ["data_path", "output_path"]
            }
        },
        {
            "name": "normalize_and_hvg",
            "description": "Normalize, log-transform, and select highly variable genes. Preserves raw counts in layer.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad"},
                    "n_hvg": {"type": "integer", "description": "Number of HVGs (default: 4000)"}
                },
                "required": ["data_path", "output_path"]
            }
        },
        {
            "name": "run_dimred",
            "description": "Run PCA, compute neighbor graph, and UMAP embedding.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad"},
                    "n_pcs": {"type": "integer", "description": "Number of PCs (default: 30)"},
                    "n_neighbors": {"type": "integer", "description": "Number of neighbors (default: 30)"}
                },
                "required": ["data_path", "output_path"]
            }
        },
        {
            "name": "run_clustering",
            "description": "Run Leiden or PhenoGraph clustering.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad"},
                    "method": {"type": "string", "enum": ["leiden", "phenograph"], "description": "Method (default: leiden)"},
                    "resolution": {"type": "number", "description": "Resolution (default: 1.0)"}
                },
                "required": ["data_path", "output_path"]
            }
        },
        {
            "name": "run_celltypist",
            "description": "Annotate cell types with CellTypist. Handles target_sum=10000 normalization automatically.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad"},
                    "model": {"type": "string", "description": "Model name (default: Immune_All_Low.pkl)"},
                    "majority_voting": {"type": "boolean", "description": "Use majority voting (default: true)"}
                },
                "required": ["data_path", "output_path"]
            }
        },
        {
            "name": "run_batch_correction",
            "description": "Correct batch effects using Harmony or Scanorama.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad"},
                    "batch_key": {"type": "string", "description": "Batch column name"},
                    "method": {"type": "string", "enum": ["harmony", "scanorama"], "description": "Method (default: harmony)"}
                },
                "required": ["data_path", "output_path", "batch_key"]
            }
        },
        {
            "name": "run_deg",
            "description": "Run differential expression analysis between groups.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad"},
                    "groupby": {"type": "string", "description": "Group column (default: leiden)"},
                    "method": {"type": "string", "enum": ["wilcoxon", "t-test", "logreg"], "description": "Method (default: wilcoxon)"}
                },
                "required": ["data_path", "output_path"]
            }
        },
        {
            "name": "generate_figure",
            "description": "Generate and save a visualization (UMAP, violin, dotplot, etc.).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad"},
                    "output_path": {"type": "string", "description": "Path to save PNG figure"},
                    "plot_type": {"type": "string", "enum": ["umap", "violin", "dotplot", "heatmap"], "description": "Plot type"},
                    "color_by": {"type": "string", "description": "Column or gene to color by"},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Genes for dotplot/heatmap"}
                },
                "required": ["data_path", "output_path", "plot_type"]
            }
        },
    ]

    # Inspection tools (read-only)
    inspection_tools = [
        {
            "name": "inspect_data",
            "description": "Inspect data state: shape, processing status, available embeddings, what steps are done. Use this first to understand the data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file"},
                    "goal": {"type": "string", "description": "Analysis goal to get recommendations (e.g., 'cluster', 'annotate')"}
                },
                "required": ["data_path"]
            }
        },
        {
            "name": "get_cluster_sizes",
            "description": "Get cell counts per cluster.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file"},
                    "cluster_key": {"type": "string", "description": "Cluster column (default: leiden)"}
                },
                "required": ["data_path"]
            }
        },
        {
            "name": "get_top_markers",
            "description": "Get top marker genes for a cluster (requires DEG analysis first).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad with DEG results"},
                    "cluster": {"type": "string", "description": "Cluster ID"},
                    "n_genes": {"type": "integer", "description": "Number of genes (default: 10)"}
                },
                "required": ["data_path", "cluster"]
            }
        },
        {
            "name": "summarize_qc_metrics",
            "description": "Get summary statistics of QC metrics (library size, genes, MT%, doublet scores).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file"}
                },
                "required": ["data_path"]
            }
        },
        {
            "name": "get_celltypes",
            "description": "Get cell type annotation summary (counts per type).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file"},
                    "annotation_key": {"type": "string", "description": "Annotation column (default: auto-detect)"}
                },
                "required": ["data_path"]
            }
        },
        {
            "name": "list_obs_columns",
            "description": "List available columns in obs (cell metadata).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file"}
                },
                "required": ["data_path"]
            }
        },
    ]

    return action_tools + inspection_tools


def process_tool_call(tool_name: str, tool_input: Dict[str, Any], adata=None) -> tuple:
    """
    Process a tool call and return structured JSON result.

    Returns
    -------
    tuple
        (json_result_string, updated_adata)
    """
    import numpy as np
    from ..core import (
        inspect_data, load_data, run_qc_pipeline, normalize_data,
        run_pca, compute_neighbors, compute_umap,
        run_leiden, run_phenograph, recommend_next_steps
    )
    from ..core.normalization import select_hvg
    from ..core.clustering import run_differential_expression, get_top_markers
    from ..annotation import run_celltypist, run_scimilarity
    from ..batch import run_scanorama, run_harmony

    def make_state(adata):
        """Create compact state dict."""
        state = inspect_data(adata)
        return {
            "has_raw_counts": state.has_raw_layer,
            "has_qc_metrics": state.has_qc_metrics,
            "has_doublets": state.has_doublet_scores,
            "is_normalized": state.is_normalized,
            "has_hvg": state.has_hvg,
            "has_pca": state.has_pca,
            "has_neighbors": state.has_neighbors,
            "has_umap": state.has_umap,
            "has_clusters": state.has_clusters,
            "has_celltypes": state.has_celltypist or state.has_scimilarity,
        }

    try:
        # ===== INSPECTION TOOLS =====
        if tool_name == "inspect_data":
            adata = load_data(tool_input["data_path"])
            state = inspect_data(adata)
            goal = tool_input.get("goal")

            result = {
                "status": "ok",
                "tool": "inspect_data",
                "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
                "data_type": state.data_type,
                "state": make_state(adata),
                "embeddings": [k for k in adata.obsm.keys()],
                "layers": list(adata.layers.keys()),
                "clustering": {
                    "has_clusters": state.has_clusters,
                    "cluster_key": state.cluster_key,
                    "n_clusters": state.n_clusters
                },
                "batch": {
                    "batch_key": state.batch_key,
                    "n_batches": state.n_batches
                },
            }
            if goal:
                result["recommended_steps"] = recommend_next_steps(state, goal)

            return json.dumps(result, indent=2), adata

        elif tool_name == "get_cluster_sizes":
            adata = load_data(tool_input["data_path"])
            key = tool_input.get("cluster_key", "leiden")
            if key not in adata.obs:
                return json.dumps({"status": "error", "message": f"No cluster column '{key}'"}), adata

            sizes = adata.obs[key].value_counts().to_dict()
            return json.dumps({
                "status": "ok",
                "tool": "get_cluster_sizes",
                "cluster_key": key,
                "n_clusters": len(sizes),
                "sizes": {str(k): int(v) for k, v in sizes.items()}
            }, indent=2), adata

        elif tool_name == "get_top_markers":
            adata = load_data(tool_input["data_path"])
            cluster = tool_input["cluster"]
            n_genes = tool_input.get("n_genes", 10)

            markers_df = get_top_markers(adata, group=cluster, n_genes=n_genes)
            markers = markers_df[['names', 'scores', 'logfoldchanges', 'pvals_adj']].to_dict('records')

            return json.dumps({
                "status": "ok",
                "tool": "get_top_markers",
                "cluster": cluster,
                "markers": markers
            }, indent=2), adata

        elif tool_name == "summarize_qc_metrics":
            adata = load_data(tool_input["data_path"])

            metrics = {}
            for col in ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'doublet_score']:
                if col in adata.obs:
                    metrics[col] = {
                        "median": float(adata.obs[col].median()),
                        "mean": float(adata.obs[col].mean()),
                        "min": float(adata.obs[col].min()),
                        "max": float(adata.obs[col].max())
                    }

            doublet_info = {}
            if 'predicted_doublet' in adata.obs:
                doublet_info = {
                    "n_doublets": int(adata.obs['predicted_doublet'].sum()),
                    "doublet_rate": float(adata.obs['predicted_doublet'].mean())
                }

            return json.dumps({
                "status": "ok",
                "tool": "summarize_qc_metrics",
                "n_cells": adata.n_obs,
                "metrics": metrics,
                "doublets": doublet_info
            }, indent=2), adata

        elif tool_name == "get_celltypes":
            adata = load_data(tool_input["data_path"])

            # Find annotation column
            key = tool_input.get("annotation_key")
            if not key:
                for candidate in ['celltypist_majority_voting', 'celltypist_predicted_labels',
                                  'scimilarity_representative_prediction', 'cell_type', 'celltype']:
                    if candidate in adata.obs:
                        key = candidate
                        break

            if not key or key not in adata.obs:
                return json.dumps({"status": "error", "message": "No cell type annotations found"}), adata

            counts = adata.obs[key].value_counts().to_dict()
            return json.dumps({
                "status": "ok",
                "tool": "get_celltypes",
                "annotation_key": key,
                "n_types": len(counts),
                "counts": {str(k): int(v) for k, v in counts.items()}
            }, indent=2), adata

        elif tool_name == "list_obs_columns":
            adata = load_data(tool_input["data_path"])
            return json.dumps({
                "status": "ok",
                "tool": "list_obs_columns",
                "columns": list(adata.obs.columns),
                "n_columns": len(adata.obs.columns)
            }, indent=2), adata

        # ===== ACTION TOOLS =====
        elif tool_name == "run_qc":
            adata = load_data(tool_input["data_path"])
            n_before, g_before = adata.n_obs, adata.n_vars

            warnings = []
            batch_key = tool_input.get("batch_key")
            if not batch_key:
                # Auto-detect
                for k in ['batch', 'batch_id', 'sample', 'sample_id']:
                    if k in adata.obs and adata.obs[k].nunique() > 1:
                        batch_key = k
                        warnings.append(f"batch_key auto-detected as '{k}'")
                        break

            run_qc_pipeline(
                adata,
                mt_threshold=tool_input.get("mt_threshold"),
                remove_ribo=tool_input.get("remove_ribo", True),
                batch_key=batch_key,
            )
            adata.write_h5ad(tool_input["output_path"])

            return json.dumps({
                "status": "ok",
                "tool": "run_qc",
                "input_path": tool_input["data_path"],
                "output_path": tool_input["output_path"],
                "before": {"n_cells": n_before, "n_genes": g_before},
                "after": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "metrics": {
                    "cells_removed": n_before - adata.n_obs,
                    "genes_removed": g_before - adata.n_vars,
                    "doublet_rate": float(adata.obs['predicted_doublet'].mean()) if 'predicted_doublet' in adata.obs else None,
                    "median_pct_mt": float(adata.obs['pct_counts_mt'].median()) if 'pct_counts_mt' in adata.obs else None,
                },
                "warnings": warnings,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "normalize_and_hvg":
            adata = load_data(tool_input["data_path"])
            normalize_data(adata)
            n_hvg = tool_input.get("n_hvg", 4000)
            select_hvg(adata, n_top_genes=n_hvg)
            adata.write_h5ad(tool_input["output_path"])

            return json.dumps({
                "status": "ok",
                "tool": "normalize_and_hvg",
                "output_path": tool_input["output_path"],
                "n_hvg": int(adata.var['highly_variable'].sum()),
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_dimred":
            adata = load_data(tool_input["data_path"])
            n_pcs = tool_input.get("n_pcs", 30)
            n_neighbors = tool_input.get("n_neighbors", 30)

            run_pca(adata, n_comps=n_pcs)
            compute_neighbors(adata, n_neighbors=n_neighbors)
            compute_umap(adata)
            adata.write_h5ad(tool_input["output_path"])

            return json.dumps({
                "status": "ok",
                "tool": "run_dimred",
                "output_path": tool_input["output_path"],
                "n_pcs": n_pcs,
                "n_neighbors": n_neighbors,
                "variance_explained": float(adata.uns['pca']['variance_ratio'].sum()),
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_clustering":
            adata = load_data(tool_input["data_path"])
            method = tool_input.get("method", "leiden")
            resolution = tool_input.get("resolution", 1.0)

            if method == "leiden":
                run_leiden(adata, resolution=resolution)
                cluster_key = "leiden"
            else:
                run_phenograph(adata, resolution=resolution)
                cluster_key = "pheno_leiden"

            adata.write_h5ad(tool_input["output_path"])
            sizes = adata.obs[cluster_key].value_counts().to_dict()

            return json.dumps({
                "status": "ok",
                "tool": "run_clustering",
                "output_path": tool_input["output_path"],
                "method": method,
                "resolution": resolution,
                "n_clusters": len(sizes),
                "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_celltypist":
            adata = load_data(tool_input["data_path"])
            model = tool_input.get("model", "Immune_All_Low.pkl")
            majority = tool_input.get("majority_voting", True)

            run_celltypist(adata, model=model, majority_voting=majority)
            adata.write_h5ad(tool_input["output_path"])

            # Get type counts
            key = 'celltypist_majority_voting' if majority and 'celltypist_majority_voting' in adata.obs else 'celltypist_predicted_labels'
            counts = adata.obs[key].value_counts().head(10).to_dict() if key in adata.obs else {}

            return json.dumps({
                "status": "ok",
                "tool": "run_celltypist",
                "output_path": tool_input["output_path"],
                "model": model,
                "majority_voting": majority,
                "n_types": len(adata.obs[key].unique()) if key in adata.obs else 0,
                "top_types": {str(k): int(v) for k, v in counts.items()},
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_batch_correction":
            adata = load_data(tool_input["data_path"])
            method = tool_input.get("method", "harmony")
            batch_key = tool_input["batch_key"]

            if method == "harmony":
                run_harmony(adata, batch_key=batch_key)
            else:
                run_scanorama(adata, batch_key=batch_key)

            adata.write_h5ad(tool_input["output_path"])

            return json.dumps({
                "status": "ok",
                "tool": "run_batch_correction",
                "output_path": tool_input["output_path"],
                "method": method,
                "batch_key": batch_key,
                "n_batches": adata.obs[batch_key].nunique(),
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_deg":
            adata = load_data(tool_input["data_path"])
            groupby = tool_input.get("groupby", "leiden")
            method = tool_input.get("method", "wilcoxon")

            run_differential_expression(adata, groupby=groupby, method=method)
            adata.write_h5ad(tool_input["output_path"])

            return json.dumps({
                "status": "ok",
                "tool": "run_deg",
                "output_path": tool_input["output_path"],
                "groupby": groupby,
                "method": method,
                "n_groups": adata.obs[groupby].nunique(),
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "generate_figure":
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import scanpy as sc

            adata = load_data(tool_input["data_path"])
            plot_type = tool_input["plot_type"]
            output_path = tool_input["output_path"]
            color_by = tool_input.get("color_by", "leiden")
            genes = tool_input.get("genes", [])

            fig, ax = plt.subplots(figsize=(10, 8))

            if plot_type == "umap":
                sc.pl.umap(adata, color=color_by, ax=ax, show=False)
            elif plot_type == "violin":
                sc.pl.violin(adata, keys=genes or [color_by], groupby=color_by, ax=ax, show=False)
            elif plot_type == "dotplot" and genes:
                sc.pl.dotplot(adata, var_names=genes, groupby=color_by, show=False)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            return json.dumps({
                "status": "ok",
                "tool": "generate_figure",
                "output_path": output_path,
                "plot_type": plot_type,
                "color_by": color_by
            }, indent=2), adata

        else:
            return json.dumps({
                "status": "error",
                "message": f"Unknown tool: {tool_name}"
            }), adata

    except Exception as e:
        return json.dumps({
            "status": "error",
            "tool": tool_name,
            "message": str(e),
            "error_type": type(e).__name__
        }, indent=2), adata
