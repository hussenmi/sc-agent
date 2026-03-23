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
                    "data_path": {"type": "string", "description": "Path to input h5ad or 10X h5 file (required for initial load, optional if data already in memory)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - only save at key checkpoints)"},
                    "mt_threshold": {"type": "number", "description": "Max MT% (default: auto-detect)"},
                    "remove_ribo": {"type": "boolean", "description": "Remove ribosomal genes (default: true)"},
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
                    "n_hvg": {"type": "integer", "description": "Number of HVGs (default: 4000)"}
                },
                "required": []
            }
        },
        {
            "name": "run_dimred",
            "description": "Run PCA, compute neighbor graph, and UMAP embedding.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "n_pcs": {"type": "integer", "description": "Number of PCs (default: 30)"},
                    "n_neighbors": {"type": "integer", "description": "Number of neighbors (default: 30)"}
                },
                "required": []
            }
        },
        {
            "name": "run_clustering",
            "description": "Run Leiden or PhenoGraph clustering.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"},
                    "method": {"type": "string", "enum": ["leiden", "phenograph"], "description": "Method (default: leiden)"},
                    "resolution": {"type": "number", "description": "Resolution (default: 1.0)"}
                },
                "required": []
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
                    "majority_voting": {"type": "boolean", "description": "Use majority voting (default: true)"}
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
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional - data persists in memory)"}
                },
                "required": []
            }
        },
        {
            "name": "run_batch_correction",
            "description": "Correct batch effects using Harmony or Scanorama.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional)"},
                    "batch_key": {"type": "string", "description": "Batch column name"},
                    "method": {"type": "string", "enum": ["harmony", "scanorama"], "description": "Method (default: harmony)"}
                },
                "required": ["batch_key"]
            }
        },
        {
            "name": "run_deg",
            "description": "Run differential expression analysis between groups.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save processed h5ad (optional)"},
                    "groupby": {"type": "string", "description": "Group column (default: leiden)"},
                    "method": {"type": "string", "enum": ["wilcoxon", "t-test", "logreg"], "description": "Method (default: wilcoxon)"}
                },
                "required": []
            }
        },
        {
            "name": "generate_figure",
            "description": "Generate and save a visualization (UMAP, violin, dotplot, etc.).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to input h5ad (optional - uses in-memory data)"},
                    "output_path": {"type": "string", "description": "Path to save PNG figure"},
                    "plot_type": {"type": "string", "enum": ["umap", "violin", "dotplot", "heatmap"], "description": "Plot type"},
                    "color_by": {"type": "string", "description": "Column or gene to color by"},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Genes for dotplot/heatmap"}
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
    meta_tools = [
        {
            "name": "ask_user",
            "description": "Ask the user a question and wait for their response. Use this when you need clarification about: data type (cells vs nuclei), batch structure, which annotation model to use, gene ID format, or any ambiguous situation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the user"},
                    "options": {"type": "array", "items": {"type": "string"}, "description": "Optional list of choices"},
                    "default": {"type": "string", "description": "Default answer if user just presses enter"}
                },
                "required": ["question"]
            }
        },
        {
            "name": "run_code",
            "description": "Execute custom Python code on the AnnData object. Use this for operations not covered by other tools, like: gene ID conversion, custom filtering, subsetting, merging datasets, plotting, or any data manipulation. The code has access to 'adata', 'sc' (scanpy), 'plt' (matplotlib), 'np', 'pd'. For plots, save to a file path.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Has access to adata, sc, plt, np, pd."},
                    "description": {"type": "string", "description": "Brief description of what the code does"},
                    "save_to": {"type": "string", "description": "Optional path to save adata after execution"}
                },
                "required": ["code", "description"]
            }
        },
        {
            "name": "web_search_docs",
            "description": "Search general web and documentation sources. Use for package docs, API references, troubleshooting, method pages, and implementation details. Prefer this for software/documentation questions, not for scientific evidence claims.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "site": {"type": "string", "description": "Optional site filter (e.g., 'scanpy.readthedocs.io', 'gseapy.readthedocs.io')"},
                    "max_results": {"type": "integer", "description": "Maximum results to return (default: 5)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "search_papers",
            "description": "Search scientific literature using PubMed. Use for papers, reviews, recent pathway/cell-type findings, and scientific evidence. Returns PMID, title, year, journal, abstract excerpt, and PubMed URL.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PubMed-style search query or plain text topic"},
                    "max_results": {"type": "integer", "description": "Maximum papers to return (default: 5)"},
                    "recent_years": {"type": "integer", "description": "Limit to the last N publication years (default: 5)"},
                    "reviews_only": {"type": "boolean", "description": "Restrict to review articles"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "fetch_url",
            "description": "Fetch and summarize a web page or article landing page. Use after a search step when you need the page contents rather than just search snippets.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Maximum text characters to return (default: 4000)"}
                },
                "required": ["url"]
            }
        },
        {
            "name": "web_search",
            "description": "Backward-compatible alias for documentation/web search. Prefer web_search_docs for docs and search_papers for literature.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "site": {"type": "string", "description": "Optional site filter"},
                    "max_results": {"type": "integer", "description": "Maximum results to return (default: 5)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "research_findings",
            "description": "Conduct focused literature research on GSEA/pathway findings. Uses PubMed queries tailored to pathway, cell type, and leading-edge genes, and returns structured citations plus interpretation guidance. Use AFTER run_gsea to understand what enriched pathways mean biologically.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pathway": {"type": "string", "description": "Pathway name to research (e.g., 'Oxidative phosphorylation', 'TNF signaling')"},
                    "cell_type": {"type": "string", "description": "Cell type context (e.g., 'classical monocytes', 'CD8 T cells', 'B cells')"},
                    "genes": {"type": "array", "items": {"type": "string"}, "description": "Leading edge genes from GSEA to include in search"},
                    "context": {"type": "string", "description": "Additional context (e.g., 'PBMC', 'tumor microenvironment', 'inflammation')"},
                    "recent_years": {"type": "integer", "description": "Limit to papers from last N years (default: 3)"}
                },
                "required": ["pathway", "cell_type"]
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
    ]

    # Inspection tools (read-only)
    inspection_tools = [
        {
            "name": "inspect_data",
            "description": "Inspect data state: shape, processing status, available embeddings, what steps are done. Use this first to understand the data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to h5ad file (optional - uses in-memory data)"},
                    "goal": {"type": "string", "description": "Analysis goal to get recommendations (e.g., 'cluster', 'annotate')"}
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
                    "n_genes": {"type": "integer", "description": "Number of genes (default: 10)"}
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

    def get_adata(tool_input, existing_adata, update_memory: bool = True):
        """Get adata from memory or load from disk.

        If ``update_memory`` is False, loading from disk is treated as read-only
        and does not replace the active in-memory AnnData tracked by the agent.
        """
        data_path = tool_input.get("data_path")
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
        """Fix output_path if it's a directory by adding a filename."""
        import os as os_module
        if output_path is None:
            return None
        if os_module.path.isdir(output_path):
            # It's a directory, construct a proper filename
            filename = f"{tool_name}_result.h5ad"
            return os_module.path.join(output_path, filename)
        return output_path

    def search_web(query: str, site: str = "", max_results: int = 5) -> Dict[str, Any]:
        """Search web/docs using Tavily (primary), Google CSE (backup), or DuckDuckGo (fallback)."""
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

        # === GOOGLE (Backup) ===
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

        # === DUCKDUCKGO (Final fallback) ===
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

            return {
                "status": "ok" if snippets else "warning",
                "backend": "duckduckgo",
                "query": scoped_query,
                "results": snippets,
                "retried_without_site": retried_without_site,
                "fallback_used": True,
                "backends_tried": [e["backend"] for e in search_errors],
                "errors": search_errors if search_errors else None,
                "message": "" if snippets else "No results found for this query.",
            }
        except ImportError:
            search_errors.append({"backend": "duckduckgo", "type": "not_installed", "message": "duckduckgo-search not installed"})
            return {
                "status": "error",
                "query": scoped_query,
                "message": "No search backend available. Set TAVILY_API_KEY, or configure Google, or install duckduckgo-search.",
                "errors": search_errors,
            }

    def search_pubmed(query: str, max_results: int = 5, recent_years: int = 5, reviews_only: bool = False) -> List[Dict[str, Any]]:
        """Search PubMed and return structured article metadata."""
        import re
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

            results = []
            articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL)
            for article in articles:
                title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.DOTALL)
                abstract_match = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
                pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
                year_match = re.search(r"<PubDate>.*?<Year>(\d+)</Year>", article, re.DOTALL)
                journal_match = re.search(r"<Title>(.*?)</Title>", article)

                if title_match and pmid_match:
                    title = re.sub(r"<[^>]+>", "", title_match.group(1))
                    abstract = re.sub(r"<[^>]+>", "", abstract_match.group(1))[:700] if abstract_match else ""
                    pmid = pmid_match.group(1)
                    year = year_match.group(1) if year_match else "N/A"
                    journal = journal_match.group(1) if journal_match else "N/A"
                    results.append({
                        "pmid": pmid,
                        "title": title,
                        "year": year,
                        "journal": journal,
                        "abstract": abstract,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    })

            return results
        except Exception:
            return []

    def fetch_url_text(url: str, max_chars: int = 4000) -> Dict[str, Any]:
        """Fetch a URL and return a compact, structured text summary."""
        import html
        import io
        import re
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
                for tag in soup(["script", "style", "noscript", "svg"]):
                    tag.decompose()

                page_title = clean_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else ""
                meta = soup.find("meta", attrs={"name": "description"})
                meta_desc = clean_whitespace(meta.get("content", "")) if meta else ""

                main = soup.find("main") or soup.find("article") or soup.body or soup
                parts = []
                for tag in main.find_all(["h1", "h2", "h3", "p", "li"]):
                    text_part = clean_whitespace(tag.get_text(" ", strip=True))
                    if text_part:
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
            "text_excerpt": cleaned[:max_chars],
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
                "default": tool_input.get("default", ""),
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

            # Security: basic checks (not foolproof, but helps)
            forbidden = ["import os", "import sys", "subprocess", "eval(",
                        "__import__", "rm -rf", "shutil.rmtree", "requests."]
            for f in forbidden:
                if f in code:
                    return json.dumps({
                        "status": "error",
                        "tool": "run_code",
                        "message": f"Forbidden operation: {f}"
                    }, indent=2), adata

            # Load data if needed
            if adata is None and "data_path" in tool_input:
                adata = get_adata(tool_input, adata)

            # Execute in controlled namespace
            namespace = {
                "adata": adata,
                "sc": sc,
                "np": np,
                "pd": pd,
                "plt": plt,
                "scanpy": sc,
                "matplotlib": matplotlib,
            }

            # Capture stdout so LLM can see print outputs
            import io
            import sys
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout

            # Capture any figures created
            plt.close('all')

            try:
                sys.stdout = stdout_capture
                exec(code, namespace)
            finally:
                sys.stdout = old_stdout

            captured_output = stdout_capture.getvalue()
            adata = namespace.get("adata", adata)

            # Check if any figures were created
            figures_saved = []
            if plt.get_fignums():
                # There are open figures - check if code saved them
                pass

            if save_to:
                adata.write_h5ad(save_to)

            # Save code to file if output directory exists
            code_file = None
            if "output_dir" in tool_input:
                import os
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
            if save_to:
                result["saved_to"] = save_to
            if code_file:
                result["code_file"] = code_file
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
            search_result["tool"] = tool_name
            search_result["search_kind"] = "docs"
            if tool_name == "web_search":
                search_result["deprecated_alias"] = True
                search_result["warning"] = (
                    "web_search is retained for backward compatibility. Prefer web_search_docs for documentation lookup."
                )
            search_result["note"] = (
                "Use this for documentation, troubleshooting, and implementation details. "
                "For scientific literature, prefer search_papers or research_findings."
            )
            return json.dumps(search_result, indent=2), adata

        elif tool_name == "search_papers":
            query = tool_input["query"]
            max_results = tool_input.get("max_results", 5)
            recent_years = tool_input.get("recent_years", 5)
            reviews_only = tool_input.get("reviews_only", False)

            results = search_pubmed(
                query=query,
                max_results=max_results,
                recent_years=recent_years,
                reviews_only=reviews_only,
            )

            return json.dumps({
                "status": "ok",
                "tool": "search_papers",
                "backend": "pubmed",
                "query": query,
                "reviews_only": reviews_only,
                "years_searched": f"last {recent_years} years",
                "results": results,
                "note": "Use fetch_url on a selected PubMed URL if you want the landing page text. Prefer PMID-backed evidence for scientific claims.",
            }, indent=2), adata

        elif tool_name == "fetch_url":
            url = tool_input["url"]
            max_chars = tool_input.get("max_chars", 4000)
            fetched = fetch_url_text(url, max_chars=max_chars)
            fetched["tool"] = "fetch_url"
            return json.dumps(fetched, indent=2), adata

        elif tool_name == "research_findings":
            pathway = tool_input["pathway"]
            cell_type = tool_input["cell_type"]
            genes = tool_input.get("genes", [])
            context = tool_input.get("context", "")
            recent_years = tool_input.get("recent_years", 3)

            all_findings = {
                "pathway": pathway,
                "cell_type": cell_type,
                "pubmed_results": [],
                "review_articles": [],
                "gene_specific": [],
            }

            # Simplify cell type for better PubMed matches
            # "classical monocytes" -> "monocyte", "CD8+ T cells" -> "T cell"
            cell_type_simple = cell_type.lower()
            for prefix in ["classical ", "non-classical ", "cd4+ ", "cd8+ ", "naive ", "memory ", "regulatory "]:
                cell_type_simple = cell_type_simple.replace(prefix, "")
            cell_type_simple = cell_type_simple.rstrip("s")  # monocytes -> monocyte

            # Search 1: Pathway + cell type (ignore context for main query - too restrictive)
            query1 = f'("{pathway}"[Title/Abstract]) AND ("{cell_type_simple}"[Title/Abstract])'
            all_findings["pubmed_results"] = search_pubmed(query1, max_results=5, recent_years=recent_years)

            # If no results, try broader search
            if not all_findings["pubmed_results"]:
                query1_broad = f'("{pathway}") AND ("{cell_type_simple}")'
                all_findings["pubmed_results"] = search_pubmed(query1_broad, max_results=5, recent_years=recent_years)

            # Search 2: Review articles on this topic
            query2 = f'("{pathway}") AND ("{cell_type_simple}")'
            all_findings["review_articles"] = search_pubmed(
                query2,
                max_results=3,
                recent_years=recent_years,
                reviews_only=True,
            )

            # Search 3: Key genes in cell type context
            if genes and len(genes) >= 2:
                gene_str = " OR ".join([f'"{g}"[Title/Abstract]' for g in genes[:3]])
                query3 = f'({gene_str}) AND ("{cell_type_simple}"[Title/Abstract])'
                all_findings["gene_specific"] = search_pubmed(query3, max_results=4, recent_years=recent_years)

            # Count findings
            total_results = (
                len(all_findings["pubmed_results"]) +
                len(all_findings["review_articles"]) +
                len(all_findings["gene_specific"])
            )

            return json.dumps({
                "status": "ok",
                "tool": "research_findings",
                "pathway": pathway,
                "cell_type": cell_type,
                "genes_researched": genes[:5] if genes else [],
                "context": context,
                "years_searched": f"last {recent_years} years",
                "total_papers_found": total_results,
                "findings": all_findings,
                "recommended_next_steps": [
                    "Prioritize review articles to understand the pathway in this cell context.",
                    "Use primary studies to support concrete biological claims.",
                    "Cite PMIDs when summarizing pathway relevance."
                ],
                "note": "Papers are from PubMed. Review abstracts for biological function, disease context, and therapeutic implications."
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
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            state = inspect_data(working_adata)
            goal = tool_input.get("goal")

            result = {
                "status": "ok",
                "tool": "inspect_data",
                "shape": {"n_cells": state.n_cells, "n_genes": state.n_genes},
                "data_type": state.data_type,
                "state": make_state(working_adata),
                "embeddings": [k for k in working_adata.obsm.keys()],
                "layers": list(working_adata.layers.keys()),
                "genes": {
                    "format": state.gene_id_format,
                    "has_symbols": state.has_gene_symbols,
                    "has_ensembl": state.has_ensembl_ids,
                    "sample": state.sample_gene_names,
                    "var_columns": list(working_adata.var.columns)[:10],
                },
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

            return json.dumps(result, indent=2), updated_adata

        elif tool_name == "get_cluster_sizes":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)
            key = tool_input.get("cluster_key", "leiden")
            if key not in working_adata.obs:
                return json.dumps({"status": "error", "message": f"No cluster column '{key}'"}), updated_adata

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

            markers_df = get_top_markers(working_adata, group=cluster, n_genes=n_genes)
            markers = markers_df[['names', 'scores', 'logfoldchanges', 'pvals_adj']].to_dict('records')

            return json.dumps({
                "status": "ok",
                "tool": "get_top_markers",
                "cluster": cluster,
                "markers": markers
            }, indent=2), updated_adata

        elif tool_name == "summarize_qc_metrics":
            working_adata, updated_adata = get_adata(tool_input, adata, update_memory=False)

            metrics = {}
            for col in ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'doublet_score']:
                if col in working_adata.obs:
                    metrics[col] = {
                        "median": float(working_adata.obs[col].median()),
                        "mean": float(working_adata.obs[col].mean()),
                        "min": float(working_adata.obs[col].min()),
                        "max": float(working_adata.obs[col].max())
                    }

            doublet_info = {}
            if 'predicted_doublet' in working_adata.obs:
                doublet_info = {
                    "n_doublets": int(working_adata.obs['predicted_doublet'].sum()),
                    "doublet_rate": float(working_adata.obs['predicted_doublet'].mean())
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
                for candidate in ['celltypist_majority_voting', 'celltypist_predicted_labels',
                                  'scimilarity_representative_prediction', 'cell_type', 'celltype']:
                    if candidate in working_adata.obs:
                        key = candidate
                        break

            if not key or key not in working_adata.obs:
                return json.dumps({"status": "error", "message": "No cell type annotations found"}), updated_adata

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
                "columns": list(working_adata.obs.columns),
                "n_columns": len(working_adata.obs.columns)
            }, indent=2), updated_adata

        # ===== ACTION TOOLS =====
        elif tool_name == "run_qc":
            adata, _ = get_adata(tool_input, adata)
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
            output_path = fix_output_path(tool_input.get("output_path"), "run_qc")
            if output_path:
                adata.write_h5ad(output_path)

            return json.dumps({
                "status": "ok",
                "tool": "run_qc",
                "input_path": tool_input["data_path"],
                "output_path": output_path,
                "saved": output_path is not None,
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
            adata, _ = get_adata(tool_input, adata)
            normalize_data(adata)
            n_hvg = tool_input.get("n_hvg", 4000)
            select_hvg(adata, n_top_genes=n_hvg)

            output_path = fix_output_path(tool_input.get("output_path"), "normalize_and_hvg")
            if output_path:
                adata.write_h5ad(output_path)

            return json.dumps({
                "status": "ok",
                "tool": "normalize_and_hvg",
                "output_path": output_path,
                "saved": output_path is not None,
                "n_hvg": int(adata.var['highly_variable'].sum()),
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_dimred":
            adata, _ = get_adata(tool_input, adata)
            n_pcs = tool_input.get("n_pcs", 30)
            n_neighbors = tool_input.get("n_neighbors", 30)

            run_pca(adata, n_comps=n_pcs)
            compute_neighbors(adata, n_neighbors=n_neighbors)
            compute_umap(adata)

            output_path = fix_output_path(tool_input.get("output_path"), "run_dimred")
            if output_path:
                adata.write_h5ad(output_path)

            return json.dumps({
                "status": "ok",
                "tool": "run_dimred",
                "output_path": output_path,
                "saved": output_path is not None,
                "n_pcs": n_pcs,
                "n_neighbors": n_neighbors,
                "variance_explained": float(adata.uns['pca']['variance_ratio'].sum()),
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_clustering":
            adata, _ = get_adata(tool_input, adata)
            method = tool_input.get("method", "leiden")
            resolution = tool_input.get("resolution", 1.0)

            if method == "leiden":
                run_leiden(adata, resolution=resolution)
                cluster_key = "leiden"
            else:
                run_phenograph(adata, resolution=resolution)
                cluster_key = "pheno_leiden"

            output_path = fix_output_path(tool_input.get("output_path"), "run_clustering")
            if output_path:
                adata.write_h5ad(output_path)
            sizes = adata.obs[cluster_key].value_counts().to_dict()

            return json.dumps({
                "status": "ok",
                "tool": "run_clustering",
                "output_path": output_path,
                "saved": output_path is not None,
                "method": method,
                "resolution": resolution,
                "n_clusters": len(sizes),
                "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_celltypist":
            adata, _ = get_adata(tool_input, adata)
            model = tool_input.get("model", "Immune_All_Low.pkl")
            majority = tool_input.get("majority_voting", True)

            run_celltypist(adata, model=model, majority_voting=majority)

            output_path = fix_output_path(tool_input.get("output_path"), "run_celltypist")
            if output_path:
                adata.write_h5ad(output_path)

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

            return json.dumps({
                "status": "ok",
                "tool": "run_celltypist",
                "output_path": output_path,
                "saved": output_path is not None,
                "model": model,
                "majority_voting": majority,
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "cell_type_breakdown": type_breakdown,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_scimilarity":
            adata, _ = get_adata(tool_input, adata)
            model_path = tool_input.get("model_path")

            # Only pass model_path if specified, otherwise use default
            if model_path:
                run_scimilarity(adata, model_path=model_path)
            else:
                run_scimilarity(adata)

            output_path = fix_output_path(tool_input.get("output_path"), "run_scimilarity")
            if output_path:
                adata.write_h5ad(output_path)

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

            return json.dumps({
                "status": "ok",
                "tool": "run_scimilarity",
                "output_path": output_path,
                "saved": output_path is not None,
                "total_cells": total_cells,
                "n_types": len(all_counts),
                "annotation_key": key,
                "has_embeddings": "X_scimilarity" in adata.obsm,
                "cell_type_breakdown": type_breakdown,
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "save_data":
            if adata is None:
                return json.dumps({
                    "status": "error",
                    "tool": "save_data",
                    "message": "No in-memory data available to save. Run an analysis tool first."
                }, indent=2), adata

            output_path = fix_output_path(tool_input.get("output_path"), "save_data")
            adata.write_h5ad(output_path)

            return json.dumps({
                "status": "ok",
                "tool": "save_data",
                "output_path": output_path,
                "saved": True,
                "shape": {"n_cells": adata.n_obs, "n_genes": adata.n_vars},
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_batch_correction":
            adata, _ = get_adata(tool_input, adata)
            method = tool_input.get("method", "harmony")
            batch_key = tool_input["batch_key"]

            # Get batch sizes for output
            batch_sizes = adata.obs[batch_key].value_counts().to_dict()

            if method == "harmony":
                run_harmony(adata, batch_key=batch_key)
                corrected_rep = 'X_pca_harmony'
            else:
                run_scanorama(adata, batch_key=batch_key)
                corrected_rep = 'X_scanorama'

            # Recompute neighbors and UMAP on corrected embedding
            compute_neighbors(adata, n_neighbors=30, use_rep=corrected_rep)
            compute_umap(adata)

            output_path = fix_output_path(tool_input.get("output_path"), "run_batch_correction")
            if output_path:
                adata.write_h5ad(output_path)

            return json.dumps({
                "status": "ok",
                "tool": "run_batch_correction",
                "output_path": output_path,
                "saved": output_path is not None,
                "method": method,
                "batch_key": batch_key,
                "n_batches": len(batch_sizes),
                "batch_sizes": {str(k): int(v) for k, v in batch_sizes.items()},
                "corrected_embedding": corrected_rep,
                "umap_recomputed": True,
                "note": f"UMAP recomputed using corrected {corrected_rep} embedding",
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_deg":
            adata, _ = get_adata(tool_input, adata)
            groupby = tool_input.get("groupby", "leiden")
            method = tool_input.get("method", "wilcoxon")

            # Best practice: use raw counts for DEG
            run_differential_expression(adata, groupby=groupby, method=method, use_raw=True)

            output_path = fix_output_path(tool_input.get("output_path"), "run_deg")
            if output_path:
                adata.write_h5ad(output_path)

            # Get top 5 markers per cluster for immediate insight
            groups = list(adata.obs[groupby].unique())
            top_markers_summary = {}
            for group in groups[:15]:  # Limit to first 15 clusters for response size
                try:
                    markers_df = get_top_markers(adata, group=str(group), n_genes=5)
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

            return json.dumps({
                "status": "ok",
                "tool": "run_deg",
                "output_path": output_path,
                "saved": output_path is not None,
                "groupby": groupby,
                "method": method,
                "n_groups": len(groups),
                "used_raw_counts": True,
                "top_markers_per_cluster": top_markers_summary,
                "note": "Use get_top_markers tool for more detailed analysis of specific clusters",
                "state": make_state(adata)
            }, indent=2), adata

        elif tool_name == "run_gsea":
            import os
            import scanpy as sc

            adata, _ = get_adata(tool_input, adata)
            output_dir = tool_input["output_dir"]
            cluster = tool_input["cluster"]
            gene_sets = tool_input.get("gene_sets", "KEGG_2021_Human")
            min_size = tool_input.get("min_size", 5)
            max_size = tool_input.get("max_size", 500)
            permutation_num = tool_input.get("permutation_num", 1000)

            # Check DEG results exist
            if 'rank_genes_groups' not in adata.uns:
                return json.dumps({
                    "status": "error",
                    "tool": "run_gsea",
                    "message": "No DEG results found. Run run_deg first."
                }, indent=2), adata

            try:
                import gseapy
            except ImportError:
                return json.dumps({
                    "status": "error",
                    "tool": "run_gsea",
                    "message": "gseapy not installed. Use install_package tool first.",
                    "install_command": "pip install gseapy"
                }, indent=2), adata

            os.makedirs(output_dir, exist_ok=True)

            # Get clusters to analyze
            groupby = adata.uns['rank_genes_groups']['params']['groupby']
            if cluster == 'all':
                clusters_to_analyze = list(adata.obs[groupby].unique())[:10]  # Limit to 10
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
                    gsea_outdir = os.path.join(output_dir, f"gsea_cluster_{clust}")
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

            return json.dumps({
                "status": "ok",
                "tool": "run_gsea",
                "output_dir": output_dir,
                "gene_sets": gene_sets,
                "clusters_analyzed": clusters_to_analyze,
                "results": all_results,
                "recommended_next_steps": [
                    "Use research_findings on the most significant pathway terms for biological interpretation.",
                    "Use search_papers for broader literature questions or recent reviews.",
                    "Use web_search_docs for pathway database or software documentation questions."
                ],
                "note": "NES > 0 means pathway upregulated in this cluster. FDR < 0.25 is typically significant."
            }, indent=2), adata

        elif tool_name == "generate_figure":
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import scanpy as sc

            adata, _ = get_adata(tool_input, adata)
            plot_type = tool_input["plot_type"]
            output_path = tool_input["output_path"]
            color_by = tool_input.get("color_by", "leiden")
            genes = tool_input.get("genes", [])
            include_image = tool_input.get("include_image", True)

            fig, ax = plt.subplots(figsize=(10, 8))

            if plot_type == "umap":
                sc.pl.umap(adata, color=color_by, ax=ax, show=False)
            elif plot_type == "violin":
                sc.pl.violin(adata, keys=genes or [color_by], groupby=color_by, ax=ax, show=False)
            elif plot_type == "dotplot" and genes:
                sc.pl.dotplot(adata, var_names=genes, groupby=color_by, show=False)
            elif plot_type == "heatmap" and genes:
                sc.pl.heatmap(adata, var_names=genes, groupby=color_by, show=False)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            result = {
                "status": "ok",
                "tool": "generate_figure",
                "output_path": output_path,
                "plot_type": plot_type,
                "color_by": color_by,
            }

            # Include base64 image for vision models
            if include_image:
                result["image_base64"] = encode_image_base64(output_path)
                result["image_mime"] = get_image_mime_type(output_path)

            return json.dumps(result, indent=2), adata

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
