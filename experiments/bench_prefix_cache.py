#!/usr/bin/env python3
"""
Prefix cache benchmark — simulates an agent loop to show how prefix caching
affects TTFT as context grows across iterations.

What this measures:
  A real scagent session sends ~20K tokens of fixed overhead (system prompt +
  tool schemas) on EVERY iteration. With prefix caching, the server reuses
  cached KV values for that prefix — only the new tokens at the tail need
  to be processed. The longer the session runs, the better the cache hit rate
  because the cached portion grows relative to the new tokens.

  This script simulates that by:
    - Building a large fixed prefix (~PREFIX_TOKENS tokens) representing the
      system prompt + tool schemas
    - Sending N iterations where each adds a new "turn" to the conversation
    - Measuring TTFT per iteration in two conditions:
        cached:   same prefix every time → cache hits after iteration 1
        uncached: unique nonce in prefix each time → always cold prefill
    - Both conditions have the same total token count per iteration (fair comparison)

Usage:
  # Run both conditions against the current server (prefix caching must be ON):
  python bench_prefix_cache.py \\
    --url http://iscb007:8000/v1 \\
    --model Qwen3.6-27B \\
    --hardware A100 \\
    --out results/prefix_cache.csv

  # Only one condition:
  python bench_prefix_cache.py --url ... --mode cached
  python bench_prefix_cache.py --url ... --mode uncached

  # Use a different prefix size (default 20000 tokens):
  python bench_prefix_cache.py --url ... --prefix-tokens 15000

Notes:
  - Server MUST have --enable-prefix-caching active (default in start_vllm.sh).
  - The "uncached" condition works by inserting a unique 8-char hex nonce at the
    start of the system prompt, making each prefix a cache miss while keeping
    total token count nearly identical to the cached condition.
  - Short completions (--max-tokens 30) keep the benchmark fast and focus
    measurement on TTFT, not decode speed.
  - A small inter-request pause (--pause 0.3s) lets the KV cache settle.
"""

import argparse
import csv
import json
import os
import random
import string
import sys
import time
from datetime import datetime

import requests

# ── Realistic prefix components ───────────────────────────────────────────────
# These represent the structure of scagent's system prompt + tool schemas.
# The actual content is less important than the token count — what matters is
# that the prefix is identical across iterations (cached) or unique (uncached).

_SYSTEM_PROMPT_BASE = """\
You are scagent, an expert single-cell RNA-sequencing analysis assistant \
integrated with a Python execution environment. You help bioinformaticians \
perform end-to-end scRNA-seq analysis workflows including quality control, \
normalization, dimensionality reduction, clustering, cell type annotation, \
differential expression analysis, and visualization.

You have access to a set of specialized tools that directly manipulate an \
in-memory AnnData object. Each tool call is executed in a sandboxed Python \
environment. You receive structured JSON results from each tool call and use \
them to decide the next analysis step.

Core principles:
- Execute one analysis step at a time and verify results before proceeding.
- Preserve existing metadata (obs columns, var columns, embeddings) unless \
  explicitly instructed to remove them.
- QC is flag-first: compute metrics and flag cells, but require explicit \
  confirmation before removing flagged cells.
- Annotation is not final until validated against external marker databases \
  (PanglaoDB or equivalent).
- When a tool result contains unexpected numbers (e.g. very few clusters, \
  unusually high mitochondrial fraction), pause and report before continuing.
- run_code is the escape hatch for analysis not covered by specialized tools.

The current analysis state is tracked in world_state and injected into this \
prompt at each iteration. Always consult world_state before choosing the next \
analysis step to avoid repeating completed work.

Technical context:
- Primary dataset: one AnnData object held in memory as `adata`
- Available: scanpy, numpy, pandas, matplotlib, scipy, sklearn
- Outputs are written to the current run directory
- Figures are saved as PNG and returned as image context
"""

_TOOL_TEMPLATE = """\

─── Tool: {name} ───────────────────────────────────────────
Description: {description}
Parameters:
  {params}
Returns: JSON object with keys: {returns}
Example call:
  {{"tool": "{name}", "parameters": {{{example}}}}}
"""

_TOOLS = [
    ("run_qc", "Compute QC metrics and flag low-quality cells.",
     "min_genes (int), max_genes (int), max_pct_mito (float), min_counts (int)",
     "cells_before, cells_after, flagged_count, mito_median, genes_median",
     '"min_genes": 200, "max_genes": 6000, "max_pct_mito": 20'),
    ("normalize_and_hvg", "Log-normalize counts and select highly variable genes.",
     "target_sum (float), n_top_genes (int), normalization_source (str), flavor (str)",
     "n_hvg, normalization_key, hvg_key",
     '"target_sum": 10000, "n_top_genes": 3000'),
    ("run_pca", "Run PCA on HVG-normalized data.",
     "n_comps (int), use_highly_variable (bool), random_state (int)",
     "n_components_used, variance_explained, key_added",
     '"n_comps": 50'),
    ("run_neighbors", "Compute k-nearest-neighbor graph for clustering and UMAP.",
     "n_neighbors (int), n_pcs (int), use_rep (str), metric (str)",
     "n_neighbors_used, use_rep, key_added",
     '"n_neighbors": 15, "n_pcs": 30'),
    ("run_umap", "Compute 2D UMAP embedding.",
     "min_dist (float), spread (float), random_state (int), color_by (str)",
     "figure_path, key_added",
     '"min_dist": 0.3, "color_by": "leiden_0.5"'),
    ("leiden_clustering", "Run Leiden community detection.",
     "resolution (float), key (str), n_iterations (int), random_state (int)",
     "n_clusters, key_added, cluster_sizes",
     '"resolution": 0.5, "key": "leiden_0.5"'),
    ("run_deg", "Differential expression analysis between clusters.",
     "groupby (str), method (str), n_genes (int), key_added (str)",
     "n_groups, top_markers_per_group, key_added",
     '"groupby": "leiden_0.5", "method": "wilcoxon"'),
    ("annotate_celltypist", "Annotate clusters using CellTypist trained models.",
     "model (str), majority_voting (bool), min_prop (float)",
     "labels_added, model_used, n_cell_types",
     '"model": "Immune_All_High"'),
    ("annotate_scimilarity", "Annotate clusters using scimilarity cell atlas.",
     "query_key (str), threshold (float)",
     "labels_added, n_matched, n_unmatched",
     '"threshold": 0.7'),
    ("validate_markers", "Validate cluster annotations against PanglaoDB marker genes.",
     "annotation_key (str), top_n (int), species (str)",
     "validation_report, confidence_scores, conflicting_labels",
     '"annotation_key": "celltypist_labels"'),
    ("run_harmony", "Batch correction using Harmony.",
     "batch_key (str), theta (float), max_iter (int)",
     "corrected_rep_key, n_batches_corrected",
     '"batch_key": "sample"'),
    ("run_scvi", "Batch correction and latent space using scVI.",
     "batch_key (str), n_latent (int), max_epochs (int), use_gpu (bool)",
     "latent_key, model_path, training_history",
     '"batch_key": "sample", "n_latent": 30'),
    ("plot_qc_metrics", "Plot QC metric distributions.",
     "metrics (list), groupby (str), show_thresholds (bool)",
     "figure_path",
     '"metrics": ["n_genes_by_counts", "pct_counts_mt"]'),
    ("plot_embedding", "Plot UMAP or PCA colored by a metadata column.",
     "basis (str), color (str), size (float), alpha (float)",
     "figure_path",
     '"basis": "X_umap", "color": "leiden_0.5"'),
    ("plot_dotplot", "Dot plot of marker gene expression per cluster.",
     "var_names (list), groupby (str), standard_scale (str)",
     "figure_path",
     '"groupby": "leiden_0.5"'),
    ("run_gsea", "Gene set enrichment analysis on DEG results.",
     "deg_key (str), organism (str), gene_sets (list)",
     "enriched_pathways, figure_path",
     '"deg_key": "rank_genes_leiden_0.5", "organism": "human"'),
    ("save_adata", "Save the current AnnData object to disk.",
     "filename (str), compression (str)",
     "file_path, file_size_mb",
     '"filename": "analysis_clustered.h5ad"'),
    ("load_adata", "Load an AnnData object from disk.",
     "path (str), backed (bool)",
     "n_obs, n_vars, obs_columns, var_columns",
     '"path": "/data/pbmc.h5ad"'),
    ("inspect_adata", "Report the current state of the AnnData object.",
     "show_obs (bool), show_var (bool), show_uns (bool)",
     "shape, obs_columns, var_columns, embeddings, uns_keys",
     '"show_obs": true'),
    ("run_code", "Execute arbitrary Python code in the analysis namespace.",
     "code (str), description (str)",
     "stdout, stderr, figures_saved, adata_modified",
     '"code": "print(adata.obs.columns.tolist())", "description": "inspect obs columns"'),
]


def _build_prefix(target_tokens: int, nonce: str = "") -> str:
    """Build a realistic system-prompt-sized prefix of approximately target_tokens tokens.

    Uses the nonce parameter to make the prefix unique per request when
    cache-busting is needed. The nonce is prepended as a comment-like marker
    that doesn't affect model behavior but ensures the cache key differs.
    """
    nonce_line = f"[session-id: {nonce}]\n" if nonce else ""
    base = nonce_line + _SYSTEM_PROMPT_BASE

    # Append tool schemas until we approach the target size.
    # Rough estimate: Qwen tokenizer ≈ 3.8 chars/token for English technical text.
    chars_per_token = 3.8
    tool_text = ""
    for name, desc, params, returns, example in _TOOLS:
        tool_text += _TOOL_TEMPLATE.format(
            name=name, description=desc, params=params,
            returns=returns, example=example,
        )

    # Repeat the tool block until we reach the target
    current_tokens = len(base) / chars_per_token
    while current_tokens < target_tokens - 500:
        base += tool_text
        current_tokens = len(base) / chars_per_token

    return base


# Pre-compute the fixed prefix once (expensive for large targets)
_PREFIX_CACHE: dict = {}


def get_prefix(target_tokens: int, nonce: str = "") -> str:
    key = (target_tokens, nonce)
    if key not in _PREFIX_CACHE:
        _PREFIX_CACHE[key] = _build_prefix(target_tokens, nonce)
    return _PREFIX_CACHE[key]


# Realistic agent conversation turns to simulate growing history
_TURNS = [
    ("Run quality control on the loaded data.", "Running QC... n_genes threshold 200-6000, max_mito 20%. Flagged 847 low-quality cells."),
    ("Filter out the flagged low-quality cells.", "Filtered: 12000 → 11153 cells. Removing 847 flagged cells."),
    ("Normalize and select highly variable genes.", "Log-normalized to 10K. Selected 3000 HVGs using seurat_v3 flavor."),
    ("Run PCA with 50 components.", "PCA complete. Top 3 PCs explain 15.2%, 8.7%, 6.3% variance respectively."),
    ("Compute neighbors graph with 15 neighbors.", "KNN graph built with 15 neighbors using 30 PCs. Added to obsp['connectivities']."),
    ("Generate a UMAP embedding.", "UMAP complete. Saved to X_umap. Figure saved."),
    ("Run Leiden clustering at resolution 0.5.", "Leiden clustering: 9 clusters at resolution 0.5. Sizes range from 312 to 2847."),
    ("Run differential expression for all clusters.", "DEG complete. Wilcoxon rank-sum. Results in rank_genes_leiden_0.5."),
    ("Annotate with CellTypist using Immune_All_High model.", "CellTypist annotation: T cells (38%), B cells (22%), Monocytes (18%), NK cells (12%), other (10%)."),
    ("Validate annotations against PanglaoDB markers.", "Validation: 7/9 clusters have high-confidence marker overlap. Clusters 3 and 7 are ambiguous."),
    ("Plot a UMAP colored by cell type annotation.", "UMAP by annotation saved. Clear separation of T/B/Monocyte populations visible."),
    ("Run GSEA on the top DEGs for monocytes.", "GSEA: top pathways — inflammatory response, cytokine signaling, innate immune response."),
    ("Save the annotated AnnData to disk.", "Saved to analysis_annotated.h5ad (847 MB)."),
    ("Show a summary of the current analysis state.", "Analysis state: QC ✓, Normalize ✓, PCA ✓, UMAP ✓, Cluster ✓, DEG ✓, Annotate ✓, Validate ✓"),
    ("Plot a dotplot of top markers for each cluster.", "Dotplot saved. Clear marker specificity for T cell subtypes in clusters 0, 2, and 5."),
]


def bench_request(
    base_url: str,
    model: str,
    system_content: str,
    history: list,
    current_user_msg: str,
    max_tokens: int,
) -> dict:
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": current_user_msg})

    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    ttft = None
    completion_tokens = None
    prompt_tokens = None

    with requests.post(
        f"{base_url}/chat/completions",
        json=body,
        stream=True,
        timeout=180,
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload = raw[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            now = time.perf_counter()
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") and ttft is None:
                    ttft = now - t0
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
                prompt_tokens = usage.get("prompt_tokens")

    total = time.perf_counter() - t0
    return {
        "ttft_ms": (ttft or total) * 1000,
        "total_s": total,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def run_condition(
    base_url: str,
    model: str,
    target_prefix_tokens: int,
    n_iterations: int,
    max_tokens: int,
    pause: float,
    mode: str,  # "cached" or "uncached"
    hardware: str,
    ts: str,
) -> list:
    rows = []
    history = []

    # For uncached mode, generate a fresh nonce per iteration so every
    # request has a unique prefix and can never be a cache hit.
    # For cached mode, nonce is fixed (empty) — same prefix every time.
    fixed_nonce = ""
    if mode == "uncached":
        # Pre-generate nonces so uncached iterations always differ
        nonces = ["".join(random.choices(string.hexdigits[:16], k=8))
                  for _ in range(n_iterations)]

    print(f"\n  ── {mode.upper()} condition ──")
    print(f"  {'iter':>4}  {'prompt_tok':>10}  {'TTFT':>8}  {'note'}")
    print(f"  {'-'*50}")

    for i in range(n_iterations):
        turn_idx = i % len(_TURNS)
        user_msg, assistant_msg = _TURNS[turn_idx]

        if mode == "uncached":
            nonce = nonces[i]
        else:
            nonce = fixed_nonce

        prefix = get_prefix(target_prefix_tokens, nonce)

        try:
            r = bench_request(
                base_url, model, prefix, history, user_msg, max_tokens
            )
        except Exception as e:
            print(f"  [{i:>3}] FAILED: {e}")
            continue

        note = "cold" if i == 0 else "warm" if mode == "cached" else "miss"
        print(
            f"  {i:>4}  {r['prompt_tokens'] or '?':>10}  "
            f"{r['ttft_ms']:>7.0f}ms  {note}"
        )

        rows.append({
            "timestamp": ts,
            "hardware": hardware,
            "mode": mode,
            "iteration": i,
            "prefix_tokens_target": target_prefix_tokens,
            "prompt_tokens": r["prompt_tokens"] or "",
            "completion_tokens": r["completion_tokens"] or "",
            "ttft_ms": round(r["ttft_ms"], 1),
            "total_s": round(r["total_s"], 3),
            "url": base_url,
        })

        # Grow the history for next iteration (simulates accumulating context)
        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": assistant_msg})

        if pause > 0 and i < n_iterations - 1:
            time.sleep(pause)

    return rows


CSV_FIELDS = [
    "timestamp", "hardware", "mode", "iteration",
    "prefix_tokens_target", "prompt_tokens", "completion_tokens",
    "ttft_ms", "total_s", "url",
]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--url",           required=True)
    ap.add_argument("--model",         required=True)
    ap.add_argument("--hardware",      required=True, choices=["A100", "H100"])
    ap.add_argument("--out",           default="results/prefix_cache.csv")
    ap.add_argument("--mode",          default="both",
                    choices=["cached", "uncached", "both"],
                    help="which condition(s) to run")
    ap.add_argument("--iterations",    type=int, default=20,
                    help="number of agent loop iterations to simulate")
    ap.add_argument("--prefix-tokens", type=int, default=20000,
                    help="target size of the fixed prefix in tokens")
    ap.add_argument("--max-tokens",    type=int, default=30,
                    help="completion length per request — keep short to focus on TTFT")
    ap.add_argument("--pause",         type=float, default=0.3,
                    help="seconds between requests (lets cache settle)")
    ap.add_argument("--no-warmup",     action="store_true")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_header = not os.path.exists(args.out)

    print(f"\n{'='*60}")
    print(f"Prefix cache benchmark")
    print(f"  URL:           {args.url}")
    print(f"  Model:         {args.model}")
    print(f"  Hardware:      {args.hardware}")
    print(f"  Mode:          {args.mode}")
    print(f"  Iterations:    {args.iterations}")
    print(f"  Prefix tokens: {args.prefix_tokens} (target)")
    print(f"  Output:        {args.out}")
    print(f"{'='*60}")

    # Pre-build prefix (slow for large sizes — do it before warmup)
    print("\nBuilding prefix... ", end="", flush=True)
    _ = get_prefix(args.prefix_tokens)
    est_tokens = len(get_prefix(args.prefix_tokens)) / 3.8
    print(f"done (~{est_tokens:.0f} estimated tokens, {len(get_prefix(args.prefix_tokens))//1024}KB)")

    if not args.no_warmup:
        print("Warmup... ", end="", flush=True)
        try:
            bench_request(args.url, args.model, "Hello.", [], "Hi.", max_tokens=4)
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(1)

    ts = datetime.now().isoformat(timespec="seconds")
    all_rows = []

    modes = ["cached", "uncached"] if args.mode == "both" else [args.mode]
    for mode in modes:
        rows = run_condition(
            base_url=args.url,
            model=args.model,
            target_prefix_tokens=args.prefix_tokens,
            n_iterations=args.iterations,
            max_tokens=args.max_tokens,
            pause=args.pause,
            mode=mode,
            hardware=args.hardware,
            ts=ts,
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No results — is the server running?")
        sys.exit(1)

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n  Appended {len(all_rows)} rows → {args.out}")

    # Quick summary
    for mode in modes:
        mode_rows = [r for r in all_rows if r["mode"] == mode]
        if not mode_rows:
            continue
        ttfts = [r["ttft_ms"] for r in mode_rows]
        first = mode_rows[0]["ttft_ms"]
        steady = [r["ttft_ms"] for r in mode_rows[3:]]  # skip first 3 warm-up iters
        print(f"\n  {mode}: first={first:.0f}ms  "
              f"steady-state median={sum(steady)/len(steady):.0f}ms  "
              f"(across {len(ttfts)} iters)")


if __name__ == "__main__":
    main()
