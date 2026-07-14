"""Gene-first batch investigation — the part that actually looks at genes.

This is the core the batch diagnostic is built around. The UMAP, sample
composition, entropy and ARI/NMI only tell us *where* samples separate; they
never establish *why*. To reason about why, we look at genes, in four steps:

1. **Find sample-enriched regions.** A (cluster, sample) region is interesting
   when it holds far more of that sample's cells than the sample's overall size
   would predict — enrichment over baseline frequency, NOT raw purity. A region
   that is 42% G8 when G8 is only 9% of the data is a strong signal even though
   it is nowhere near "80% one sample".

2. **Within-sample identity DEG — THE PRIMARY GENE EVIDENCE.** For each region,
   compare that cluster's cells against the *rest of the same sample*. Because the
   comparison stays inside one sample, batch is held constant, so the genes it
   returns describe *what this population is* (its identity), uncontaminated by
   sample effects. The two within-sample DEGs of a pair — and how much their top
   genes overlap — are the primary evidence that two regions are the same
   population.

3. **Match regions across samples by their identity genes.** Two regions from
   different samples that share most of their top identity genes are a supported
   identity match (``identity_match_supported``). Shared identity genes SUPPORT
   selecting a pair; they do not establish that the populations are definitively
   identical.

4. **(secondary) Direct cross-sample DEG on matched regions.** As *supporting*
   evidence only, compare the two matched regions to each other and report, per
   side, the genes that differ. This mixes batch back in, so it never outranks the
   within-sample evidence; its job is to surface *how* the regions differ. Here we
   keep *all* genes (stress, mitochondrial, ribosomal, ambient) — they are often
   the most informative about a sample-associated program. We do not interpret
   them; we name them and let the model reason about what they mean.

5. **(secondary) Recurrence.** A sample-associated program that recurs across
   several distinct populations points to a *sample-wide* effect. That is stronger
   than a single pair — but it establishes only "recurring / sample-wide", never
   "technical". Only the experimental design can say whether sample-wide means
   technical or real biology. Nothing here is ever labeled "conclusive".

The default DE engine is scanpy's in-process Wilcoxon (``rank_genes_groups``) —
the identical Mann-Whitney statistic diffxpy's rank test computes, but instant,
whereas diffxpy cold-starts TensorFlow in a fresh subprocess on every call. When
``prefer_diffxpy=True`` this investigation runs the same test through diffxpy's
rank engine (a cross-check / uniform provenance, not a different statistic); the
diffxpy bridge additionally offers the NB Wald count model, but the investigation
does not use it by default. When diffxpy is requested but unavailable, the engine
falls back to Wilcoxon VISIBLY (recorded as ``scanpy_wilcoxon_diffxpy_unavailable``),
and a diffxpy worker *crash* is surfaced, never masked by the fallback.

Cost control: candidate cross-sample pairs are nominated cheaply from mean-
expression profile similarity (no DEG), so the within-sample identity DEGs run
only for the handful of selected pairs — not for every enriched region.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# Generic technical/housekeeping gene patterns, excluded ONLY from the numerical
# identity-overlap calculation so two differently-typed but equally-stressed
# populations don't "match" on shared housekeeping genes. They are NOT removed
# from the full within-sample DEG evidence, and NOT used in the direct comparison
# (where they may be exactly the signal). Deliberately small and generic — it
# encodes "housekeeping genes shouldn't define identity", not cell-type biology,
# and must never grow into a stress/ambient lookup. Hemoglobin genes are NOT
# excluded: they are real identity for erythroid cells, so dropping them without
# context would be wrong.
NUISANCE_PATTERNS = [
    r"^MT-", r"^mt-", r"^RPL", r"^RPS", r"^Rpl", r"^Rps",
    r"^MRPL", r"^MRPS", r"^MALAT1$", r"^Malat1$", r"\.\d+$",
]

_NUISANCE_RE = re.compile("|".join(NUISANCE_PATTERNS))


def is_nuisance_gene(gene: str) -> bool:
    return bool(_NUISANCE_RE.search(str(gene)))


# ---------------------------------------------------------------------------
# Differential-expression engine (diffxpy default, visible Wilcoxon fallback)
# ---------------------------------------------------------------------------

def _wilcoxon_two_group(sub: Any, group: np.ndarray, genes: list[str]) -> pd.DataFrame:
    """In-env Wilcoxon (scanpy ``rank_genes_groups``) fallback.

    Same Mann-Whitney test diffxpy's rank uses; returns the same normalized
    columns so callers never branch on the engine.
    """
    import anndata as ad
    import scanpy as sc

    obs = pd.DataFrame(
        {"group": pd.Categorical(np.where(group != 0, "target", "reference"))}
    )
    var = pd.DataFrame(index=[str(g) for g in genes])
    a = ad.AnnData(X=sub.copy(), obs=obs, var=var)
    sc.tl.rank_genes_groups(a, "group", groups=["target"], reference="reference",
                            method="wilcoxon")
    df = sc.get.rank_genes_groups_df(a, group="target")
    # rank_genes_groups: names, logfoldchanges, pvals_adj (target vs reference).
    return pd.DataFrame(
        {
            "gene": df["names"].astype(str).to_numpy(),
            "qval": df["pvals_adj"].to_numpy(),
            "log2fc": df["logfoldchanges"].to_numpy(),
        }
    )


def two_group_deg(
    matrix: Any,
    genes: list[str],
    target_idx: np.ndarray,
    reference_idx: np.ndarray,
    *,
    test: str = "rank",
    prefer_diffxpy: bool = False,
) -> tuple[pd.DataFrame, str]:
    """One target-vs-reference DEG, returning ``(dataframe, engine)``.

    ``matrix`` is the full cells x genes expression; ``target_idx`` /
    ``reference_idx`` are row indices for the two sides. The result is sorted by
    ascending ``qval`` and always carries ``gene, qval, log2fc, mean_target,
    mean_reference, pct_target, pct_reference`` (log2fc positive = higher in the
    target). ``engine`` is ``"diffxpy_<test>"`` or ``"scanpy_wilcoxon"``.

    The DEFAULT engine is scanpy's in-process Wilcoxon (``prefer_diffxpy=False``):
    for a rank test it is the identical Mann-Whitney statistic diffxpy would
    compute, but instant — whereas diffxpy spawns a fresh subprocess that cold-
    starts TensorFlow (~15-20 s) on EVERY call, which is crippling when many DEGs
    run. When ``prefer_diffxpy=True`` and diffxpy is available, this runs the same
    ``test`` (default ``'rank'``) through diffxpy's engine; ``test='wald'`` would
    select the NB count model, but callers here use rank. If diffxpy is requested
    but unavailable, the engine is recorded as
    ``scanpy_wilcoxon_diffxpy_unavailable`` so the fallback is never mistaken for a
    plain Wilcoxon run; a diffxpy worker crash (``RuntimeError``) propagates.
    """
    target_idx = np.asarray(target_idx).ravel()
    reference_idx = np.asarray(reference_idx).ravel()
    rows = np.concatenate([target_idx, reference_idx])
    sub = matrix[rows]
    group = np.concatenate(
        [np.ones(len(target_idx), dtype=np.int8), np.zeros(len(reference_idx), dtype=np.int8)]
    )

    from ..batch.diffxpy import (
        DiffxpyUnavailable,
        attach_mean_pct,
        diffxpy_available,
        run_two_group_de,
    )

    engine = "scanpy_wilcoxon"
    stats: pd.DataFrame | None = None
    if prefer_diffxpy:
        if diffxpy_available():
            try:
                # run_two_group_de already attaches the shared mean/pct columns.
                stats = run_two_group_de(sub, group, genes, test=test)
                engine = f"diffxpy_{test}"
            except DiffxpyUnavailable:
                # Requested but couldn't run (e.g. interpreter vanished mid-run):
                # fall back VISIBLY. Worker crashes (RuntimeError) still propagate.
                stats = None
                engine = "scanpy_wilcoxon_diffxpy_unavailable"
        else:
            # diffxpy requested but the env is not built: record the fallback so it
            # is never mistaken for a plain Wilcoxon run.
            engine = "scanpy_wilcoxon_diffxpy_unavailable"
    if stats is None:
        # Wilcoxon path: get the SAME mean/pct columns via the shared attach helper.
        stats = attach_mean_pct(_wilcoxon_two_group(sub, group, genes), sub, group, genes)

    return stats.sort_values("qval", kind="mergesort").reset_index(drop=True), engine


def top_positive_genes(
    deg: pd.DataFrame, n: int, *, exclude_nuisance: bool = False, min_effect: float = 0.0
) -> list[str]:
    """Top ``n`` genes higher in the target, ranked by ``expression_effect``.

    Orientation and ranking use ``expression_effect`` (mean_target -
    mean_reference), never the engine's fold-change.
    """
    up = deg[deg["expression_effect"] > min_effect].copy()
    if exclude_nuisance:
        up = up[~up["gene"].map(is_nuisance_gene)]
    up = up.sort_values("expression_effect", ascending=False, kind="mergesort")
    return [str(g) for g in up["gene"].head(n).tolist()]


# ---------------------------------------------------------------------------
# Step 1 — sample-enriched regions
# ---------------------------------------------------------------------------

def find_sample_enriched_regions(
    adata,
    batch_key: str,
    cluster_key: str,
    *,
    min_cells: int = 30,
    min_enrichment: float = 2.0,
) -> list[dict[str, Any]]:
    """(cluster, sample) regions holding more of a sample than its baseline predicts.

    Enrichment = (fraction of the cluster made of this sample) / (this sample's
    fraction of the whole dataset). A region qualifies when it has at least
    ``min_cells`` cells and enrichment at least ``min_enrichment`` — replacing the
    old 80%-purity gate, which missed strongly-enriched-but-not-pure regions.
    """
    batch = adata.obs[batch_key].astype(str).to_numpy()
    cluster = adata.obs[cluster_key].astype(str).to_numpy()
    n_total = len(batch)
    batch_labels, batch_counts = np.unique(batch, return_counts=True)
    baseline = {b: c / n_total for b, c in zip(batch_labels, batch_counts, strict=True)}

    regions: list[dict[str, Any]] = []
    for c in np.unique(cluster):
        in_cluster = cluster == c
        n_cluster = int(in_cluster.sum())
        if n_cluster == 0:
            continue
        sub_batch = batch[in_cluster]
        labs, counts = np.unique(sub_batch, return_counts=True)
        for b, n_bc in zip(labs, counts, strict=True):
            if n_bc < min_cells:
                continue
            frac_in_cluster = n_bc / n_cluster
            base = baseline.get(b, 0.0) or 1e-9
            enrichment = frac_in_cluster / base
            if enrichment < min_enrichment:
                continue
            regions.append(
                {
                    "cluster": str(c),
                    "sample": str(b),
                    "n_cells": int(n_bc),
                    "n_cluster": n_cluster,
                    "frac_of_cluster": round(float(frac_in_cluster), 4),
                    "sample_baseline_frac": round(float(baseline.get(b, 0.0)), 4),
                    "enrichment": round(float(enrichment), 3),
                }
            )
    regions.sort(key=lambda r: r["enrichment"], reverse=True)
    return regions


# ---------------------------------------------------------------------------
# Step 2 — within-sample identity DEG (cached per region)
# ---------------------------------------------------------------------------

def within_sample_identity_deg(
    adata,
    region: dict[str, Any],
    matrix: Any,
    genes: list[str],
    *,
    batch_key: str,
    cluster_key: str,
    prefer_diffxpy: bool = False,
    min_cells: int = 20,
) -> pd.DataFrame | None:
    """One region's cluster vs the rest of its OWN sample — batch held constant.

    The returned genes describe the population's identity, free of sample effects
    (both sides are the same sample). Returns ``None`` when either side is too
    small to test.
    """
    batch = adata.obs[batch_key].astype(str).to_numpy()
    cluster = adata.obs[cluster_key].astype(str).to_numpy()
    same_sample = batch == region["sample"]
    target = np.flatnonzero(same_sample & (cluster == region["cluster"]))
    reference = np.flatnonzero(same_sample & (cluster != region["cluster"]))
    if len(target) < min_cells or len(reference) < min_cells:
        return None
    deg, engine = two_group_deg(
        matrix, genes, target, reference, test="rank", prefer_diffxpy=prefer_diffxpy
    )
    deg.attrs["engine"] = engine
    deg.attrs["n_target"] = int(len(target))
    deg.attrs["n_reference"] = int(len(reference))
    return deg


def identity_signatures(
    identity_degs: dict[tuple[str, str], pd.DataFrame],
    *,
    top_n: int = 50,
) -> dict[tuple[str, str], list[str]]:
    """Top identity genes per region (nuisance excluded — identity, not housekeeping)."""
    sigs: dict[tuple[str, str], list[str]] = {}
    for key, deg in identity_degs.items():
        if deg is None:
            continue
        sigs[key] = top_positive_genes(deg, top_n, exclude_nuisance=True, min_effect=0.0)
    return sigs


# ---------------------------------------------------------------------------
# Step 3 — match regions across samples: cheap profile nomination (below, near
# run_gene_investigation), confirmed by shared within-sample identity genes.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Step 4 (SECONDARY) — direct cross-sample DEG on matched regions (keep ALL genes)
# ---------------------------------------------------------------------------

def direct_pair_deg(
    adata,
    pair: dict[str, Any],
    matrix: Any,
    genes: list[str],
    *,
    batch_key: str,
    cluster_key: str,
    prefer_diffxpy: bool = False,
    top_n: int = 25,
) -> dict[str, Any] | None:
    """SECONDARY evidence: compare the two matched regions directly; report genes
    higher on each side.

    This is supporting, not primary — the two within-sample identity DEGs are the
    primary evidence. A direct region-A-vs-region-B comparison mixes batch back
    in, so it only characterizes *how* the regions differ; it must never outrank
    the within-sample tests. ALL genes are kept here — stress / mitochondrial /
    ribosomal / ambient genes may be the most informative about a sample-associated
    program. We name them (with ``higher_in`` = the sample they are higher in) and
    leave interpretation to the model.
    """
    batch = adata.obs[batch_key].astype(str).to_numpy()
    cluster = adata.obs[cluster_key].astype(str).to_numpy()
    a_idx = np.flatnonzero((batch == pair["sample_a"]) & (cluster == pair["cluster_a"]))
    b_idx = np.flatnonzero((batch == pair["sample_b"]) & (cluster == pair["cluster_b"]))
    if len(a_idx) < 10 or len(b_idx) < 10:
        return None
    deg, engine = two_group_deg(
        matrix, genes, a_idx, b_idx, test="rank", prefer_diffxpy=prefer_diffxpy
    )
    # target = sample_a; positive expression_effect = higher in sample_a.
    higher_a = deg[deg["expression_effect"] > 0].sort_values("expression_effect", ascending=False)
    higher_b = deg[deg["expression_effect"] < 0].sort_values("expression_effect")
    return {
        "cluster_a": pair["cluster_a"], "sample_a": pair["sample_a"],
        "cluster_b": pair["cluster_b"], "sample_b": pair["sample_b"],
        "engine": engine,
        "n_cells_a": int(len(a_idx)), "n_cells_b": int(len(b_idx)),
        "higher_in_a": [str(g) for g in higher_a["gene"].head(top_n)],
        "higher_in_b": [str(g) for g in higher_b["gene"].head(top_n)],
        "deg": deg,
    }


# ---------------------------------------------------------------------------
# Step 5 (SECONDARY) — recurrence of a sample-associated program across populations
# ---------------------------------------------------------------------------

def find_recurrent_programs(
    direct_results: list[dict[str, Any]],
    *,
    min_populations: int = 2,
    top_n: int = 25,
    batch_group_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Genes higher in the SAME sample (or batch group), in the SAME direction,
    across at least ``min_populations`` DISTINCT matched populations.

    A recurring program is scoped to a coherent sample direction. A gene is grouped
    by the sample it is *higher in* — so a gene higher in S2 (vs S1) in the beta
    pair and higher in S2 (vs S3) in the delta pair is one S2-associated recurring
    program across two populations. Two unrelated comparisons (e.g. G8-vs-G1 and
    G6-vs-G5) are NOT pooled unless ``batch_group_map`` explicitly maps their
    associated samples to the same technical batch group.

    Requires: same associated sample/batch group, same direction (higher in that
    sample), and >= ``min_populations`` distinct populations. Each entry names its
    contributing pairs and populations.

    Concludes only that a program is **recurring / sample-wide** — never that it is
    *technical*. A recurring program is equally consistent with a real systemic
    biological difference (e.g. a treated animal); only the design can settle which.
    """
    def _group_of(sample: str) -> str:
        return (batch_group_map or {}).get(sample, sample)

    # key = (associated group, gene) -> {"populations": set, "pairs": set}
    acc: dict[tuple[str, str], dict[str, set]] = {}
    for r in direct_results:
        # A gene in higher_in_a is higher in sample_a; the population is cluster_a
        # (the associated-sample side of this pair). Symmetrically for side b.
        for sample, cluster, other_s, other_c, gene_list in (
            (r["sample_a"], r["cluster_a"], r["sample_b"], r["cluster_b"], r["higher_in_a"]),
            (r["sample_b"], r["cluster_b"], r["sample_a"], r["cluster_a"], r["higher_in_b"]),
        ):
            group = _group_of(sample)
            pair_label = f"{cluster}/{sample} vs {other_c}/{other_s}"
            for gene in gene_list[:top_n]:
                entry = acc.setdefault((group, gene), {"populations": set(), "pairs": set()})
                entry["populations"].add(cluster)
                entry["pairs"].add(pair_label)

    recurrent: list[dict[str, Any]] = []
    for (group, gene), entry in acc.items():
        if len(entry["populations"]) < min_populations:
            continue
        recurrent.append(
            {
                "associated_batch_group": group,
                "gene": gene,
                "direction": f"higher in {group}",
                "n_populations": int(len(entry["populations"])),
                "contributing_populations": sorted(entry["populations"]),
                "contributing_pairs": sorted(entry["pairs"]),
            }
        )
    recurrent.sort(key=lambda r: r["n_populations"], reverse=True)
    return recurrent


# ---------------------------------------------------------------------------
# Orchestration — run the whole gene-first investigation
# ---------------------------------------------------------------------------

def region_mean_profiles(
    adata, matrix: Any, genes: list[str], regions: list[dict[str, Any]],
    *, batch_key: str, cluster_key: str,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Mean expression vector per (cluster, sample) region — the cheap similarity signal."""
    batch = adata.obs[batch_key].astype(str).to_numpy()
    cluster = adata.obs[cluster_key].astype(str).to_numpy()
    keys: list[tuple[str, str]] = []
    profs: list[np.ndarray] = []
    for r in regions:
        mask = (batch == r["sample"]) & (cluster == r["cluster"])
        if not mask.any():
            continue
        m = matrix[mask].mean(axis=0)
        m = np.asarray(m.todense()).ravel() if hasattr(m, "todense") else np.asarray(m).ravel()
        keys.append((r["cluster"], r["sample"]))
        profs.append(m)
    return keys, (np.vstack(profs) if profs else np.zeros((0, len(genes))))


def nominate_cross_sample_pairs(
    keys: list[tuple[str, str]],
    mean_matrix: np.ndarray,
    genes: list[str],
    *,
    min_corr: float = 0.4,
    n_top_variable: int = 2000,
) -> list[dict[str, Any]]:
    """Cheaply nominate "same cell type across different samples" pairs — NO DEG.

    Compares regions by their mean-expression profiles, restricted to the most
    *variable* genes across regions (marker-like; excludes constant housekeeping
    and constant noise that would otherwise dilute the signal), via Pearson
    correlation. Cross-sample pairs above ``min_corr`` are returned, ranked by
    correlation — the "find two similar clusters from two different samples" step.
    The within-sample DEGs that confirm the match run only for the pairs selected
    from here.
    """
    if mean_matrix.shape[0] < 2:
        return []
    nuis = np.array([is_nuisance_gene(g) for g in genes], dtype=bool)
    m = mean_matrix[:, ~nuis] if nuis.any() else mean_matrix
    # Keep the most variable genes across regions (marker-like), so the correlation
    # reflects cell-type identity rather than the shared housekeeping/noise baseline.
    var = m.var(axis=0)
    n_informative = int((var > 1e-8).sum())
    if n_informative >= 2:
        k = min(n_top_variable, n_informative)
        top = np.argsort(var)[::-1][:k]
        m = m[:, top]
    corr = np.corrcoef(m)
    pairs: list[dict[str, Any]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (ca, sa), (cb, sb) = keys[i], keys[j]
            if sa == sb:
                continue
            c = float(corr[i, j])
            if c >= min_corr:
                pairs.append(
                    {"cluster_a": ca, "sample_a": sa, "cluster_b": cb, "sample_b": sb,
                     "_corr": c, "profile_correlation": round(c, 3)}
                )
    # Rank by the FULL-precision correlation (rounding only sets the displayed value,
    # so ties that differ past 3 dp still order correctly).
    pairs.sort(key=lambda p: p["_corr"], reverse=True)
    return pairs


def _confirm_identity_match(
    pair: dict[str, Any],
    signatures: dict[tuple[str, str], list[str]],
    *,
    top25: int = 25,
    top50: int = 50,
    min_shared_top25: int = 5,
) -> dict[str, Any]:
    """Confirm a nominated pair with the two within-sample identity DEGs' overlap."""
    sig_a = signatures.get((pair["cluster_a"], pair["sample_a"]), [])
    sig_b = signatures.get((pair["cluster_b"], pair["sample_b"]), [])
    b25 = set(sig_b[:top25])
    a50, b50 = set(sig_a[:top50]), set(sig_b[:top50])
    shared25 = [g for g in sig_a[:top25] if g in b25]
    shared50 = [g for g in sig_a[:top50] if g in b50]
    union50 = a50 | b50
    clean = {k: v for k, v in pair.items() if k != "_corr"}  # drop the sort-only raw corr
    return {
        **clean,
        "n_shared_top25": len(shared25),
        "n_shared_top50": len(shared50),
        "jaccard_top50": round(len(a50 & b50) / len(union50), 3) if union50 else 0.0,
        "shared_top25_genes": shared25,
        "shared_top50_genes": shared50[:30],
        "identity_match_supported": bool(len(shared25) >= min_shared_top25),
    }


def run_gene_investigation(
    adata,
    matrix: Any,
    genes: list[str],
    *,
    batch_key: str,
    cluster_key: str,
    prefer_diffxpy: bool = False,
    min_cells: int = 30,
    min_enrichment: float = 2.0,
    identity_min_cells: int = 20,
    top_n_signature: int = 50,
    min_shared_top25: int = 5,
    min_profile_corr: float = 0.4,
    max_pairs: int = 3,
    max_attempts: int = 10,
    min_populations: int = 2,
    batch_group_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run the gene-first investigation and return the structured results.

    Fast by construction: candidate cross-sample pairs are nominated CHEAPLY from
    mean-expression profile similarity (no DEG), then the expensive within-sample
    DEGs run ONLY for the candidates we actually investigate — not for every
    enriched region. Candidates are tried in correlation order and BACKFILLED: if a
    candidate fails identity confirmation, the next-best candidate is tried, until
    ``max_pairs`` confirmed pairs are found (or ``max_attempts`` is reached). The
    verdict and design gate live one layer up in the diagnostic.
    """
    regions = find_sample_enriched_regions(
        adata, batch_key, cluster_key, min_cells=min_cells, min_enrichment=min_enrichment
    )
    # Cheap: nominate "similar clusters across samples" by profile correlation.
    keys, profiles = region_mean_profiles(
        adata, matrix, genes, regions, batch_key=batch_key, cluster_key=cluster_key
    )
    candidates = nominate_cross_sample_pairs(keys, profiles, genes, min_corr=min_profile_corr)

    region_info = {(r["cluster"], r["sample"]): r for r in regions}
    identity_degs: dict[tuple[str, str], pd.DataFrame | None] = {}

    def _identity(key: tuple[str, str]):
        if key not in identity_degs:
            identity_degs[key] = within_sample_identity_deg(
                adata, region_info[key], matrix, genes,
                batch_key=batch_key, cluster_key=cluster_key,
                prefer_diffxpy=prefer_diffxpy, min_cells=identity_min_cells,
            )
        return identity_degs[key]

    # Try candidates in correlation order; DEG only the two regions of each, confirm,
    # and backfill from the next candidate when one fails. Stop at max_pairs confirmed.
    population_pairs: list[dict[str, Any]] = []
    selected_pairs: list[dict[str, Any]] = []
    used_regions: set = set()
    attempts = 0
    for cand in candidates:
        if len(selected_pairs) >= max_pairs or attempts >= max_attempts:
            break
        ra = (cand["cluster_a"], cand["sample_a"])
        rb = (cand["cluster_b"], cand["sample_b"])
        if ra in used_regions and rb in used_regions:
            continue  # both regions already belong to a confirmed pair
        attempts += 1
        _identity(ra)
        _identity(rb)
        sigs = identity_signatures(
            {k: v for k, v in identity_degs.items() if v is not None}, top_n=top_n_signature
        )
        confirmed = _confirm_identity_match(cand, sigs, min_shared_top25=min_shared_top25)
        population_pairs.append(confirmed)
        if confirmed["identity_match_supported"]:
            selected_pairs.append(confirmed)
            used_regions.add(ra)
            used_regions.add(rb)

    signatures = identity_signatures(
        {k: v for k, v in identity_degs.items() if v is not None}, top_n=top_n_signature
    )

    # Direct comparison (secondary) only for confirmed same-population pairs.
    direct_results = []
    for p in selected_pairs:
        res = direct_pair_deg(
            adata, p, matrix, genes, batch_key=batch_key, cluster_key=cluster_key,
            prefer_diffxpy=prefer_diffxpy,
        )
        if res is not None:
            direct_results.append(res)
    recurrent = find_recurrent_programs(
        direct_results, min_populations=min_populations, batch_group_map=batch_group_map
    )

    engines = sorted(
        {str(v.attrs.get("engine")) for v in identity_degs.values() if v is not None}
        | {r["engine"] for r in direct_results}
    )
    return {
        "regions": regions,
        "n_candidate_pairs": len(candidates),
        "identity_degs": identity_degs,
        "identity_signatures": signatures,
        "population_pairs": population_pairs,
        "selected_pairs": selected_pairs,
        "direct_results": direct_results,
        "recurrent_programs": recurrent,
        "engines_used": engines,
    }


# ---------------------------------------------------------------------------
# Two-axis verdict — gene evidence vs design interpretation (mutually exclusive)
# ---------------------------------------------------------------------------

def derive_gene_evidence(investigation: dict[str, Any]) -> str:
    """One of: ``none``, ``localized``, ``recurring_sample_associated``.

    - ``none`` — no supported identity-match pairs / no direct gene evidence.
    - ``recurring_sample_associated`` — a program recurs across >= 2 populations.
    - ``localized`` — direct differences exist but do not recur.
    """
    if not investigation["direct_results"]:
        return "none"
    if investigation["recurrent_programs"]:
        return "recurring_sample_associated"
    return "localized"


def derive_design_interpretation(
    confounding_rows: list[dict[str, Any]],
    *,
    condition_columns_present: bool,
    technical_batch_documented: bool = False,
) -> str:
    """One of: ``unknown``, ``confounded_with_biology``,
    ``orthogonal_but_not_known_technical``, ``documented_technical_batch``.

    A non-confounded condition column does NOT prove sample-wide differences are
    technical (donor and other biological effects can remain) — hence the guarded
    ``orthogonal_but_not_known_technical``. ``documented_technical_batch`` is only
    reachable when the caller asserts a technical batch variable separable from
    biology; the harness never infers "technical" on its own.
    """
    if technical_batch_documented:
        return "documented_technical_batch"
    if any(row.get("confounded_with_batch") for row in confounding_rows):
        return "confounded_with_biology"
    if condition_columns_present:
        return "orthogonal_but_not_known_technical"
    return "unknown"


def derive_verdict(gene_evidence: str, design_interpretation: str) -> dict[str, str]:
    """Map the two independent axes to a single, mutually-exclusive recommendation."""
    if gene_evidence == "none":
        return {
            "recommendation": "do_not_integrate_based_on_current_evidence",
            "reason": (
                "The gene investigation did not find gene-level support for integration "
                "(no supported cross-sample population matches). This does not assert that "
                "sample-associated differences exist — only that none were established here."
            ),
        }
    if gene_evidence == "localized":
        return {
            "recommendation": "do_not_integrate_based_on_current_evidence",
            "reason": (
                "Sample-associated differences were localized to individual populations and "
                "did not recur across populations, so a dataset-wide correction is not justified."
            ),
        }
    # gene_evidence == "recurring_sample_associated"
    if design_interpretation in ("unknown", "confounded_with_biology"):
        why = (
            "no experimental-design metadata is available"
            if design_interpretation == "unknown"
            else "sample is confounded with a biological condition"
        )
        return {
            "recommendation": "cannot_determine_technical_vs_biological",
            "reason": (
                f"A sample-associated program recurs across populations, but {why}, so its "
                "technical-versus-biological origin cannot be determined. Do not auto-integrate; "
                "present the evidence and ask whether the samples are comparable replicates."
            ),
        }
    if design_interpretation == "orthogonal_but_not_known_technical":
        return {
            "recommendation": "integration_optional_if_replicates",
            "reason": (
                "A sample-associated program recurs across populations and the available "
                "condition metadata is not confounded with sample — but that alone does not make "
                "the differences technical (donor and other biological effects can remain). "
                "Integration may be reasonable ONLY if the samples are intended as comparable "
                "replicates; otherwise it risks erasing real per-sample biology."
            ),
        }
    return {  # documented_technical_batch
        "recommendation": "integration_supported",
        "reason": (
            "A sample-associated program recurs across populations and the design documents a "
            "technical batch variable separable from biological condition, so the recurring "
            "sample-wide differences are attributable to a technical batch effect."
        ),
    }


# ---------------------------------------------------------------------------
# Readable, deterministic narrative — built from structured results only
# ---------------------------------------------------------------------------

def _fmt_q(q: float) -> str:
    """Render a q-value: scientific notation for tiny values, never a bare 0.0."""
    try:
        q = float(q)
    except (TypeError, ValueError):
        return "n/a"
    if q <= 0:
        return "<1e-300"
    if q < 1e-3:
        return f"{q:.1e}"
    return f"{q:.3f}"


def _top_identity(deg: pd.DataFrame | None, n: int = 8) -> list[str]:
    if deg is None:
        return []
    return top_positive_genes(deg, n, exclude_nuisance=True, min_effect=0.0)


def build_pair_narrative(
    pair: dict[str, Any],
    identity_degs: dict[tuple[str, str], pd.DataFrame | None],
    direct: dict[str, Any] | None,
    recurrent_programs: list[dict[str, Any]],
) -> list[str]:
    """A readable, deterministic account of one tested pair (gene names from data).

    States, in order: which two regions and samples; why they were a candidate
    match; each region vs its within-sample reference; shared identity genes; the
    direct differences; whether the same genes recurred elsewhere; and exactly what
    this does and does not establish. No hardcoded biology.
    """
    ra = (pair["cluster_a"], pair["sample_a"])
    rb = (pair["cluster_b"], pair["sample_b"])
    lines = [
        f"Pair: cluster {ra[0]} in sample {ra[1]}  vs  cluster {rb[0]} in sample {rb[1]}.",
        (
            f"  Candidate match: their within-sample identity genes overlap "
            f"({pair['n_shared_top25']} shared in the top 25) — enough to justify comparing "
            f"them, not to declare them definitively the same population."
        ),
        (
            f"  {ra[0]}/{ra[1]} was compared against all other {ra[1]} cells; its identity "
            f"genes include: {', '.join(_top_identity(identity_degs.get(ra))) or 'n/a'}."
        ),
        (
            f"  {rb[0]}/{rb[1]} was compared against all other {rb[1]} cells; its identity "
            f"genes include: {', '.join(_top_identity(identity_degs.get(rb))) or 'n/a'}."
        ),
        f"  Shared identity genes: {', '.join(pair['shared_top25_genes'][:8]) or 'n/a'}.",
    ]
    if direct is not None:
        lines.append(
            f"  Direct comparison (secondary): higher in {direct['sample_a']}: "
            f"{', '.join(direct['higher_in_a'][:6]) or 'n/a'}; higher in {direct['sample_b']}: "
            f"{', '.join(direct['higher_in_b'][:6]) or 'n/a'}."
        )
        # Only recurrence belonging to THIS pair's own samples counts — a program
        # recurring in G8 says nothing about a G5-vs-G4 pair. Match each side's
        # higher-in genes against recurrence associated with that same sample.
        rec_by_group: dict[str, set] = {}
        for r in recurrent_programs:
            rec_by_group.setdefault(r["associated_batch_group"], set()).add(r["gene"])
        recurred_a = [g for g in direct["higher_in_a"][:25] if g in rec_by_group.get(direct["sample_a"], set())]
        recurred_b = [g for g in direct["higher_in_b"][:25] if g in rec_by_group.get(direct["sample_b"], set())]
        if recurred_a or recurred_b:
            bits = []
            if recurred_a:
                bits.append(f"higher in {direct['sample_a']}: {', '.join(recurred_a[:6])}")
            if recurred_b:
                bits.append(f"higher in {direct['sample_b']}: {', '.join(recurred_b[:6])}")
            lines.append(
                "  Some of these genes recur across this pair's own samples' other populations "
                f"({'; '.join(bits)})."
            )
        else:
            lines.append(
                "  These direct differences did not recur across other populations of the "
                "same samples."
            )
    lines.append(
        "  This establishes a shared-identity candidate match and describes how the two "
        "regions differ. It does NOT establish that the difference is technical."
    )
    return lines


def build_terminal_summary(
    investigation: dict[str, Any],
    gene_evidence: str,
    design_interpretation: str,
    verdict: dict[str, str],
) -> list[str]:
    """Deterministic, already-readable terminal summary from the structured results.

    The model may expand this into prose for the Markdown report, but basic
    readability never depends on the model.
    """
    ident = investigation["identity_degs"]
    direct_by_pair = {
        (d["cluster_a"], d["sample_a"], d["cluster_b"], d["sample_b"]): d
        for d in investigation["direct_results"]
    }
    lines: list[str] = []
    n_regions = len(investigation["regions"])
    n_selected = len(investigation["selected_pairs"])
    lines.append(
        f"Gene-first batch investigation: {n_regions} sample-enriched region(s), "
        f"{n_selected} cross-sample population pair(s) examined."
    )
    for p in investigation["selected_pairs"]:
        key = (p["cluster_a"], p["sample_a"], p["cluster_b"], p["sample_b"])
        lines += build_pair_narrative(p, ident, direct_by_pair.get(key), investigation["recurrent_programs"])
    if investigation["recurrent_programs"]:
        by_group: dict[str, list[str]] = {}
        for r in investigation["recurrent_programs"]:
            by_group.setdefault(r["associated_batch_group"], []).append(r["gene"])
        for group, gs in by_group.items():
            lines.append(
                f"Recurring {group}-associated program across >= 2 populations: "
                f"{', '.join(gs[:10])} (sample-wide signal; technical vs biological unresolved by genes alone)."
            )
    lines.append(f"Gene evidence: {gene_evidence}. Design: {design_interpretation}.")
    lines.append(f"Recommendation: {verdict['recommendation']} — {verdict['reason']}")
    return lines

