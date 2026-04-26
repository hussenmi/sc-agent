# scagent

**Single-cell RNA-seq Analysis Agent** - A collaborative single-cell analysis toolkit that encapsulates lab best practices and can reason about your data.

## Features

### Core Analysis
- **Data Inspection**: Automatically detect data state, gene ID format, and recommend next steps
- **QC Pipeline**: Batch-aware doublet detection (Scrublet), MT/ribo filtering
- **Normalization**: Library size normalization with raw counts preservation
- **Dimensionality Reduction**: PCA, neighbors, UMAP with lab-validated parameters
- **Clustering**: Leiden and PhenoGraph (Jaccard-weighted)
- **Cell Type Annotation**: CellTypist and Scimilarity with proper target_sum=10000
- **Batch Correction**: Scanorama and Harmony
- **Differential Expression**: Wilcoxon, t-test, logistic regression

### Autonomous Agent
- **Multi-provider**: Works with OpenAI, Anthropic, Groq, or experimental ChatGPT/Codex login
- **32 Built-in Tools**: QC, clustering, annotation, visualization, research, and more
- **Code Generation**: Dynamically writes and executes Python for custom analyses
- **Vision Support**: Agent can see and analyze generated plots
- **Web Search**: Look up gene functions, pathways, best practices
- **Interactive**: Can ask user questions when clarification needed
- **Package Installation**: Request new packages with user approval
- **Reproducibility**: All runs create manifests with full provenance

## Installation

```bash
cd /data1/peerd/ibrahih3/cs_agent
source setup.sh
```

Or with uv directly:
```bash
uv sync
uv pip install -e ".[agent]"
```

## Configuration

Create a `.env` file (copy from `.env.example`):

```bash
# API Keys (only one API provider required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Defaults
SCAGENT_PROVIDER=openai
SCAGENT_MODEL=gpt-5.4-mini
```

For experimental ChatGPT/Codex subscription-backed runs, log in first and then select
the Codex provider:

```bash
scagent login chatgpt
scagent analyze --data pbmc.h5 --provider codex
```

This uses Codex/ChatGPT login rather than OpenAI Platform API billing. It is not
identical to API function-calling mode yet, so keep `openai`, `anthropic`, or `groq`
as the production path for now.

## Quick Start

### Library Usage

```python
import scagent
from scagent.core import load_data, inspect_data, run_qc_pipeline

# Load and inspect
adata = load_data("your_data.h5")
state = inspect_data(adata)
print(f"Cells: {state.n_cells}, Genes: {state.n_genes}")
print(f"Gene format: {state.gene_id_format}")

# Run QC
run_qc_pipeline(adata)
```

### Autonomous Agent

```python
from scagent.agent import SCAgent

# Initialize (reads from .env)
agent = SCAgent()  # collaborative checkpoints on by default

# Simple analysis: inspect, recommend, ask, then proceed
result = agent.analyze(
    "QC and cluster this PBMC data, then identify cell types",
    data_path="pbmc.h5"
)

# Complex multi-step analysis
result = agent.analyze("""
    1. Run QC and clustering
    2. Find clusters with highest MT% - could indicate stressed cells
    3. Investigate marker genes for those clusters
    4. Generate UMAP colored by MT%
    5. Provide synthesis with quality concerns
""", data_path="pbmc.h5")

# Interactive follow-up (continues conversation with loaded data)
result = agent.analyze(
    "Now show me DEG for cluster 5 vs all others",
    continue_conversation=True  # Keeps conversation history
)
```

### CLI Interactive Mode

```bash
# Start a collaborative session (default)
scagent analyze --data pbmc.h5

# After initial analysis completes, continue with follow-ups:
# > What are the top markers for cluster 3?
# > Generate a heatmap of these genes
# > done  (or exit / quit / q)
```

## Two Modes of Operation

scagent offers two ways to run analyses:

### 1. Agent Mode (LLM-Guided)

The agent inspects your data, reasons about what to do, adapts to problems, and provides interpretation. It works collaboratively: it summarizes findings at major checkpoints, recommends a next step, and asks before applying consequential analysis decisions.

```bash
# CLI
scagent analyze "QC and cluster this data" --data pbmc.h5
scagent analyze "QC and cluster this data" --data pbmc.h5 --single-run

# Python
agent = SCAgent()
agent.analyze("QC and cluster", data_path="pbmc.h5")
```

**Pros:** Adapts to unexpected data, handles errors gracefully, supports follow-up questions, provides insights
**Cons:** Requires API key, costs money, slower (LLM round-trips)

**Best for:** Exploratory analysis, unfamiliar data, when you want interpretation and checkpointed collaboration

### 2. Direct Mode (No LLM)

Call core functions directly - fast, free, deterministic, but no adaptation.

```bash
# CLI (limited commands)
scagent qc data.h5 output.h5ad --mt-threshold 25
scagent inspect data.h5ad

# Python (full control)
from scagent.core import load_data, run_qc_pipeline, run_clustering_pipeline

adata = load_data("data.h5")
run_qc_pipeline(adata, mt_threshold=25)
run_clustering_pipeline(adata)
adata.write_h5ad("result.h5ad")
```

**Pros:** Free, fast, reproducible, scriptable
**Cons:** Fails on unexpected data, no reasoning, no interpretation

**Best for:** Batch processing, known-good data, scripting, CI/CD pipelines

### Comparison

| | Agent Mode | Direct Mode |
|---|---|---|
| LLM involved | Yes | No |
| Cost | API calls | Free |
| Speed | Slower | Fast |
| Error handling | Reasons and adapts | Crashes |
| Interpretation | Yes ("cluster 5 looks like T cells") | No |
| Scripting | Adaptable but grounded | Deterministic |

### Agent Capabilities

The agent can:
- **Use built-in tools**: 18 pre-defined analysis functions
- **Generate custom code**: For anything not covered by tools
- **See outputs**: Reads stdout from code, views generated plots
- **Reason and adapt**: Adjusts approach based on what it finds
- **Ask questions**: When clarification is needed
- **Install packages**: With user approval

Example of agent reasoning:
```
Agent: "Cluster 13 has highest MT% (mean 15.14%) and its markers
        are mitochondrial genes - this is likely a stressed/dying
        cell population, not a real immune cell type."
```

## Available Tools (24 total)

| Category | Tools |
|----------|-------|
| **Analysis** | `run_qc`, `normalize_and_hvg`, `run_dimred`, `run_clustering`, `run_celltypist`, `run_scimilarity`, `run_batch_correction`, `run_deg`, `run_gsea` |
| **Visualization** | `generate_figure` (UMAP, violin, dotplot, heatmap) |
| **Inspection** | `inspect_data`, `get_cluster_sizes`, `get_top_markers`, `summarize_qc_metrics`, `get_celltypes`, `list_obs_columns` |
| **Research** | `web_search_docs`, `search_papers`, `fetch_url`, `research_findings` |
| **Meta** | `ask_user`, `run_code`, `install_package` |

## Standard Analysis Order

The agent follows this pipeline sequence:

```
1. QC (metrics, doublets, filtering)
2. Normalize + log transform (preserve raw_counts first!)
3. Select HVGs (4000 genes)
4. PCA (30 components)
5. Compute neighbors (k=30)
6. Compute UMAP
7. Clustering (Leiden) ← MUST come before CellTypist!
8. Cell type annotation (CellTypist)
9. DEG analysis (run Wilcoxon on the normalized/log1p analysis matrix; keep raw counts preserved in a layer)
10. GSEA / pathway analysis
```

## Lab Parameters (Best Practices)

| Parameter | Value | Notes |
|-----------|-------|-------|
| MT threshold (cells) | 25% | |
| MT threshold (nuclei) | 5% | |
| Min cells per gene | ~55 (np.exp(4)) | |
| Scrublet doublet rate | 0.06 | |
| HVG count | 4000 | seurat_v3 flavor |
| N PCs | 30 | |
| N neighbors | 30 | |
| Leiden resolution | 1.0 | |
| UMAP min_dist | 0.1 | |
| **CellTypist target_sum** | **10000** | CRITICAL |
| Scanorama dimred/knn | 30 | |

### Parameter Handling

The agent follows a **"lab defaults + ask before changing"** approach:

1. **Uses lab-validated defaults** for all parameters (MT=25%, k=30, etc.)
2. **Detects when defaults don't fit** (e.g., very low MT% suggests nuclei data)
3. **Asks user before deviating** ("MT% is very low, use 5% threshold instead?")

This ensures the pipeline is **predictable** (always starts with validated parameters) and **transparent** (user approves any changes). The agent will never silently change parameters.

### Automatic Best Practices

The agent automatically follows these best practices:

- **CellTypist**: Normalizes to `target_sum=10000` from raw counts layer (not from scaled data). Returns complete cell type breakdown with counts and percentages.
- **DEG Analysis**: Preserves raw counts in a layer, but runs Wilcoxon on the active normalized/log1p analysis matrix to match the workshop notebooks. Returns validation warnings plus top 5 markers per cluster immediately for quick insight.
- **Batch Correction**: After Harmony/Scanorama, automatically recomputes neighbors and UMAP on the corrected embedding.
- **GSEA**: Uses DEG scores for prerank GSEA with GSEApy. Returns top up/downregulated pathways with NES scores, FDR values, and leading edge genes. Supports KEGG, GO, Reactome, MSigDB Hallmark databases.
- **Documentation Search**: `web_search_docs` uses Tavily when configured (`TAVILY_API_KEY`), falls back to DuckDuckGo, and only tries Google Programmable Search as a last fallback. Best for software docs, APIs, troubleshooting, and method pages.
- **Paper Search**: `search_papers` uses PubMed E-utilities to return recent papers with PMID, title, abstract excerpt, journal, year, and PubMed URL.
- **Literature Research**: After GSEA, `research_findings` performs focused PubMed searches around enriched pathways, cell types, and leading-edge genes, and returns structured citations for interpretation.
- **Source Reading**: `fetch_url` reads selected web pages and, when dependencies are available, extracts cleaner HTML text and basic PDF text for downstream reasoning.
- **GSEA Evidence Reports**: Successful `run_gsea` calls automatically write `reports/gsea_evidence.md` and `reports/gsea_evidence.json`, combining pathway output with targeted PubMed evidence for the most relevant pathways.
- **File Management**: Keeps data in memory between tool calls and does not write intermediate `.h5ad` files unless checkpoint saving is explicitly enabled.

Search/research design note:
- [`SEARCH_RESEARCH_ARCHITECTURE.md`](/Users/hibrahim/Desktop/iris_peerd/cs_agent/SEARCH_RESEARCH_ARCHITECTURE.md)

### Where Results Are Stored

All results accumulate in a single AnnData object:

| Analysis | Columns/Keys | Location |
|----------|--------------|----------|
| QC | `total_counts`, `n_genes_by_counts`, `pct_counts_mt`, `pct_counts_ribo` | `adata.obs` |
| Doublets | `predicted_doublet`, `doublet_score` | `adata.obs` |
| Clustering | `leiden` or `pheno_leiden` | `adata.obs` |
| CellTypist | `celltypist_predicted_labels`, `celltypist_conf_score`, `celltypist_majority_voting` | `adata.obs` |
| Scimilarity | `scimilarity_predictions_unconstrained`, `scimilarity_representative_prediction` | `adata.obs` |
| PCA | `X_pca` | `adata.obsm` |
| UMAP | `X_umap` | `adata.obsm` |
| Scimilarity embeddings | `X_scimilarity` | `adata.obsm` |
| HVG | `highly_variable`, `means`, `dispersions` | `adata.var` |
| Raw counts | `raw_counts` | `adata.layers` |
| DEG | `rank_genes_groups` | `adata.uns` |

## Run Output Structure

Each agent run creates a structured directory:

```
run_2026_03_18_230927_full_analysis/
├── manifest.json           # Full reproducibility log
├── reports/
│   └── summary.md          # Analysis summary
│   └── gsea_evidence.md    # Pathway evidence summary (after GSEA)
│   └── gsea_evidence.json  # Machine-readable pathway evidence
├── logs/
│   └── agent.log           # Tool-level execution log
├── figures/
│   └── *.png               # Visualizations
└── result.h5ad             # Final saved AnnData (if requested)
```

If checkpoint saving is enabled, an additional `intermediate/` folder is created for checkpoint `.h5ad` files.

## CLI Reference

```bash
# Agent mode - LLM-guided analysis
scagent analyze "your request" --data file.h5      # Collaborative agent run
scagent analyze --data file.h5                     # Let the agent choose a first pass
scagent analyze --data file.h5 --single-run        # Exit after the initial collaborative turn
scagent analyze --data file.h5 --provider openai   # Use OpenAI instead of Anthropic
scagent login chatgpt                              # Log in for experimental Codex provider
scagent analyze --data file.h5 --provider codex    # Use ChatGPT/Codex login instead of API key

# Direct mode - no LLM
scagent qc input.h5 output.h5ad                   # Run QC pipeline directly
scagent qc input.h5 output.h5ad --mt-threshold 20 # Custom MT threshold
scagent inspect data.h5ad                         # Show data state
scagent inspect data.h5ad --goal cluster          # Get recommendations for goal

# Chat - quick questions (no data)
scagent chat "What MT threshold should I use for nuclei?"
```

## Testing

```bash
# Test tools only (no API call)
python test_agent.py --tools-only

# Test with OpenAI
python test_agent.py --provider openai

# Full analysis test
python test_agent.py --provider openai --full
```

## License

MIT
