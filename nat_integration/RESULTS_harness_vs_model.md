# Harness-vs-model test: Qwen3.6-27B vs GLM-5.2 vs Nemotron-3-Ultra (NAT, LuCA atlas)

**Date:** 2026-06-23 · **Dataset:** LUNG_T06 (LuCA NSCLC atlas ground truth;
effectively single-sample) · **Reps:** 4 each (standardized n=4) · **Harness:**
scagent wrapped in NeMo Agent Toolkit (NAT), identical editable code for all runs ·
**Vision:** shared Nemotron-Nano-12B-v2-VL NIM.

**Thesis under test:** *if the agent harness is well built, the choice of main
reasoning model matters less.* We run the **identical** scagent QC+annotation
workload 4× on each model and compare accuracy, reliability, and cost/latency.

Serving (all OpenAI-compatible vLLM):
- **Qwen3.6-27B** — single GPU
- **GLM-5.2** (744B/40B MoE, FP8) — TP=8, expert-parallel + eager, iscp001
- **Nemotron-3-Ultra** (550B/55B MoE, NVFP4) — TP=8, iscp001

---

## Headline: accuracy converges across all three; only completion-reliability differs

| Axis | Qwen3.6-27B | GLM-5.2 (744B) | Nemotron-3-Ultra (550B) |
|---|---|---|---|
| **Completed** (reps with a scoreable annotation) | **4 / 4** | **1 / 4** | **4 / 4** |
| **Accuracy** (major cell-type vs atlas, mean) | 0.833 | 0.810 (n=1) | **0.844** |
| coarse accuracy | 0.940 | 0.944 (n=1) | 0.942 |

Two clean findings:

1. **Accuracy converges hard.** On runs that completed, all three land in a narrow
   band — major **0.833 / 0.810 / 0.844**, coarse ~**0.94** for all. The structured
   tool layer (scimilarity + CellTypist + PanglaoDB/Cytopus consensus) drives the
   result; **model choice barely moves accuracy.** This is the Layer-1 thesis,
   confirmed across a 27B, a 744B, and a 550B.
2. **Completion-reliability is stochastic and model-specific — not a size effect.**
   Both large models are *not* uniformly unreliable: Nemotron (550B) completed 4/4
   and scored highest. **GLM-5.2 specifically** failed to converge on 3/4 reps via a
   verified over-deliberation mechanism (§2). So the story is **not** "bigger ⇒ worse";
   it's "a real, model-dependent non-convergence failure mode exists, and the harness
   has no floor to catch it regardless of which model trips it."

> **Revision note (supersedes an earlier draft):** an initial cut framed this as
> "the 744B is less reliable than the 27B," based on GLM 1/4 + a Nemotron n=3 (1/3)
> pilot. The standardized **Nemotron n=4 = 4/4** shows that pilot was small-n noise.
> The honest headline is the two points above: accuracy converges; completion is
> stochastic + GLM-specific. The over-deliberation *mechanism* (§2) is verified from
> GLM logs; its *rate across models* is still undersampled (GLM accuracy is n=1).

---

## 1. Accuracy & consistency  (`accuracy_consistency.png`)

| | coarse acc | major acc (primary) |
|---|---|---|
| **Qwen3.6-27B** | mean 0.9402, sd 0.0019, spread 0.0043 (n=4) | mean 0.8326, sd 0.0167, spread 0.043 (n=4) |
| **Nemotron-3-Ultra** | mean 0.9416, sd 0.0049, spread 0.013 (n=4) | **mean 0.8437**, sd 0.0214, spread 0.052 (n=4) |
| **GLM-5.2** | 0.9435 (n=1) | 0.8098 (n=1) |

- Per-rep major — Qwen: **0.861, 0.818, 0.826, 0.826**; Nemotron: **0.858, 0.871,
  0.828, 0.818**. Both tight, overlapping bands.
- GLM produced only one scoreable rep (0.810); the other three failed to
  converge (§2). Its single completed rep sits just below the other two means.
- Coarse accuracy is ~0.94 for **all three** — the structured pipeline pins the
  broad strokes regardless of model.

## 2. Reliability — GLM's over-deliberation / non-convergence failure

3 of 4 GLM reps ended with `annotated_h5ad: null`, `rc=0`, `status=completed`.
Verified mechanism from the run manifests + `agent.py` (not the earlier
"prose-bail" reading):

- **Root cause = over-deliberation.** All three failures reached
  `prepare_annotation`, then spiralled gathering and re-adjudicating PanglaoDB
  markers — forward *and* reverse queries for every competing label (NK /
  smooth-muscle / fibroblast / pericyte; then TAGLN/ACTA2/MYL9/CALD1/TPM2) — and
  **never crossed `stage_annotation_evidence → finalize_annotation`**. Every tool
  call returned `status: ok` — the execution/retrieval harness was flawless.
- **Not a budget exhaustion.** Tool-exec counts: failed reps **36 / 29 / 27**;
  the rep that **finished** did **39** (ended in `finalize_annotation` →
  `write_report`). All under `max_iterations=75`, all `status=completed`. More work
  → success; less → failure — the inverse of a cap.
- **Proximate trigger = the stop branch.** The run ends at `agent.py:5234`
  `finish_reason=="stop"` → lines 5266–5290: GLM emits a *prose planning turn with
  no tool call* (announcing its next batch of queries), `_maybe_continue_after_failure`
  doesn't fire, and scagent calls `_complete_run` + returns **without checking
  `world_state.annotation_validation`** → silent exit while staged-but-unfinalized.

In this n=4 it hit GLM on 3/4 reps and **never** happened for Qwen or Nemotron
(both 4/4). It is **model-specific, not size-driven** — Nemotron is also a large
reasoner (550B) and converged every time. The rate is undersampled (one n=4 run
per model), so treat "3/4" as evidence the mode is real and reproducible *for GLM*,
not as a calibrated cross-model probability. scagent's coordination harness is
**passive**: it enforces an *end gate* (`ready_to_finalize` needs every cluster to
pass; `save_data` is blocked before finalize) but has **no floor** (a no-tool-call
stop ends the run mid-annotation) and **no deliberation budget** (nothing curbs the
upstream marker-query spiral). When a model burns its latitude upstream, it walks
off the path — and nothing pulls it back.

**That gap is the most actionable finding.** The fix is two complementary, in the
four-file annotation path:
1. *(primary, ~surgical)* In the `finish_reason=="stop"` branch, before returning,
   if `annotation_validation` is in-progress/staged-but-not-finalized, **re-prompt**
   instead of completing (call `stage → finalize` now, or `save_data(allow_unvalidated=true)`),
   bounded by an attempt counter. → ends the **silent** failure for all 3 (worst
   case becomes an explicit `incomplete`, never a silent `null`).
2. *(secondary)* **Active convergence pressure** in `annotation_validation` /
   `prompts.py`: after N marker queries / M annotation-phase turns, inject "you have
   enough evidence — stage now," so #1's nudge actually lands. → gets real
   convergence (4/4 finalize).

## 3. Latency & throughput  (profiler / `inference_optimization.json`)

| Metric | Nemotron-3-Ultra (550B, NVFP4) | Qwen3.6-27B | GLM-5.2 (744B, FP8) |
|---|---|---|---|
| Wall time / rep (mean) | **653 s** (~11 min) | 857 s (~14 min) | 1288 s (~21 min) |
| Wall p95 | **745 s** | 930 s | 1685 s |
| Per-LLM-call latency (mean) | **9.1 s** | 20.0 s | 49.0 s |
| call p95 | **19.0 s** | 49.8 s | 165 s |
| call p99 | **29.3 s** | 102 s | 384 s |

**Nemotron is the fastest on every latency metric** — fastest wall time, ~2× faster
per call than Qwen and **~5× faster than GLM**, with the tightest tail (p99 29 s vs
GLM's 384 s) — *while also scoring the highest accuracy* (§1). It does this despite
making the **most** LLM calls per run (47, §5): per-call latency is so low that total
wall time is still lowest.

**Like-for-like serving note (NVIDIA-collab relevant).** Nemotron and GLM ran on the
**same 8×H200 node (iscp001), both TP=8** — so the Nemotron-vs-GLM comparison is a
*clean* same-hardware serving benchmark (unlike Qwen, which used a different GPU
footprint). On identical hardware, **NVFP4 Nemotron-3-Ultra is ~2× faster wall and
~5× faster per call than FP8 GLM-5.2** — a concrete NVFP4 + Nemotron-architecture
serving win. (GLM's eager-mode requirement contributes — CUDA-graph capture
deadlocks on GLM today, see `start_vllm.sh` — so part of the gap is GLM's current
vLLM serving constraints, not pure model/precision. Worth stating both ways.)

## 4. Where time goes — bottleneck ranking  (`bottleneck.png`)

- **Nemotron:** ① `run_scimilarity` 16.5 s → ② `run_neighbors` 14.7 s → ③ `run_umap` 10.1 s — **the LLM is not even top-3.**
- **Qwen:** ① **LLM 20 s** → ② `run_scimilarity` 19 s → ③ `run_neighbors` 15 s
- **GLM:** ① `run_scimilarity` 68 s → ② **LLM 49 s** → ③ `run_neighbors` 15 s

This is the cleanest statement of the regime the thesis targets: with a fast model
(Nemotron), the **science tools fully dominate** and the LLM drops out of the
bottleneck list entirely — the harness/tooling, not the model, is the limiting
factor. (Caveat: `run_scimilarity` measured 68 s under GLM vs ~16–19 s under
Nemotron/Qwen for the *same* tool → node/IO variance, not a model effect; don't
over-read absolute tool times across runs.)

## 5. Trajectory cost  (`trajectory_cost.png`)

| | LLM calls / run | prompt tokens / run |
|---|---|---|
| **Nemotron-3-Ultra** | mean 47, spread 18 (n=4) | mean 6.31M, spread 3.08M |
| **Qwen3.6-27B** | mean 33.2, spread 8 (n=4) | mean 3.89M, spread 1.02M |
| **GLM-5.2** | mean 21.5, spread 11 (n=4) | mean 2.09M, spread 2.15M |

Nemotron does the **most** work per run (47 calls, 6.3M prompt tokens) yet finishes
**fastest** (§3) — low per-call latency outweighs call count. GLM's lower counts are
deflated by its **3 non-converged reps** (ended before finalizing), so its row is not
a fair efficiency comparison. Token-uniqueness ≈ 0 and `common_prefixes` are highly
cacheable for all three → consistent with the earlier **prefill-bound** finding
(~147K avg prompt; latency dominated by prefill, which is also why Nemotron's high
call count stays cheap — repeated context is cache-served).

## 6. Concurrency

Reps ran at `max_concurrency: 1`, so the profiler's p50 concurrency of 2.0 is
workflow+nested spans, not parallel reps. No intra-run parallelism to exploit
(the agent is inherently sequential: reason → tool → reason).

---

## Figures (`nat_eval_figs/`)
- `accuracy_consistency.png` — per-rep coarse & major accuracy, all three models
- `trajectory_cost.png` — LLM calls + prompt tokens per rep
- `bottleneck.png` — per-op avg duration, side by side
- `gantt_qwen.png`, `gantt_glm.png`, `gantt_nemotron.png` — per-rep timelines
- `summary.json`, `summary.md` — aggregated numbers

## Related work & positioning — Anthropic + NCBI, "Paving the Way for Agents in Biology"

[https://www.anthropic.com/research/agents-in-biology](https://www.anthropic.com/research/agents-in-biology)
is the closest published work to this experiment and is our framing anchor.

**Their thesis:** the bottleneck for biology agents is not model reasoning but the
absence of *deterministic execution layers* for querying biological data.
- **VirBench** (120 viral-sequence-retrieval queries; SOTA agents incl. Claude
  Opus 4.7, GPT-5.55, Biomni): without deterministic tools, mean accuracy
  16.9–91.3%, and the *same model gave different answers across runs* (Claude
  Sonnet 4: 106 → 15 → 5 sequences for one identical query; truth 266).
- Downstream harm: agent-retrieved data yielded 3 different phylogenetic trees
  (outbreak origin Jan-2014 vs 1922 vs Apr-2014) — plausible but silently wrong.
- **Fix = `gget virus`** (deterministic retrieval layer co-built with NCBI): all
  agents >90% (peak 99.7%), run-to-run variability eliminated. Quote: *"Adding a
  deterministic retrieval layer made model choice much less important… reliable
  dataset construction should not depend on access to the newest or most expensive
  model."*

**How our result relates:**
1. **Validation.** Our GLM-5.2 (744B, expensive) ≤ Qwen3.6-27B (small, cheap)
   inside the same harness is an independent, *single-cell* instance of their
   "cheaper model + proper tools = as reliable as the expensive one."
2. **Extension (the more interesting half) — there are two harness layers.**
   Their variability lives at the **data-retrieval** layer; ours surfaces a second,
   higher layer the VirBench work didn't test:
   - **Layer 1 — deterministic tools** (their `gget virus` ≈ our `run_scimilarity`
     + PanglaoDB/Cytopus consensus). Governs whether the agent can *get correct
     evidence → accuracy*. Make this deterministic and **model choice stops
     mattering for accuracy**.
   - **Layer 2 — deterministic orchestration / control flow** (the coordination
     harness: entry + completion floors). Governs whether the model is *forced to use
     the tools and complete the protocol → reliability*. GLM *had* Layer 1 and still
     failed by over-deliberating and never crossing `stage → finalize` (§2); Nemotron
     also skipped a required *branch* (batch-effect) on multi-sample data despite
     being heavily prompted. Layer 1 is necessary but **not sufficient**; Layer 2 is
     what we are fixing next (`docs/coordination_harness_design.md`).
3. **Gap we fill.** They explicitly scope to viral sequences and note
   genomics/single-cell is *not* addressed. scagent is single-cell; our
   `biocontext` MCP (PanglaoDB, Cell Ontology) is the same *kind* of object as
   `gget virus`.

**The two layers, read off our data — the headline of this experiment:**
- **Accuracy converges → Layer 1 already works.** On runs that *completed*, all three
  models score nearly the same: major **0.833 / 0.810 / 0.844** (Qwen / GLM /
  Nemotron), ~**0.94** coarse for all. A good deterministic tool layer gets a 27B, a
  744B, and a 550B to the same place — exactly the article's claim, now in
  single-cell. *(Caveat: GLM is n=1 completed; the fix confirms it at n=4.)*
- **Reliability is stochastic + model-specific → Layer 2 is the open problem.**
  Completion: **4/4 Qwen, 4/4 Nemotron, 1/4 GLM**. It is *not* a size effect (Nemotron
  is also large and converged every time) — it's a real, model-dependent failure mode
  that **only GLM tripped here**, via the verified over-deliberation mechanism (§2).
  Accuracy can't even *register* it (non-converged runs produce no score); it shows up
  only in completion / trajectory adherence — the dimension Layer 2 controls and NAT's
  per-step tracing surfaces.

So the clean two-sentence story: **a good tool layer makes the model matter less for
accuracy; a good orchestration layer makes it matter less for reliability.** The
article nailed the first; our coordination-harness fix demonstrates the second
(expected: GLM completion → 4/4, accuracy gap stays ~0, reliability floor holds for
*any* model that trips the failure).

**Sharper one-liner than theirs:** *harness quality determines how much the model
matters* — not simply "the model doesn't matter." Where the harness has a soft spot,
some model will fall through it (here, GLM's over-deliberation) — **independent of
size**, since the comparably-large Nemotron did not. The fix is a model-agnostic
floor, not "pick the right model." Their forward-looking "future models may make such
tools obsolete" risk is countered by "too expensive / slow / hard to audit" — which
our prefill-bound + latency numbers (§3) substantiate.

**Seeded TODO:** a downstream-consequence demo = single-cell twin of their
phylo-tree figure (show annotation variability shifts cell composition / DE).

## Caveats
- GLM accuracy is n=1 completed; the fair n=4 comparison needs the coordination fix first.
- Completion rate is undersampled (one n=4 run/model). "GLM 1/4" shows the failure
  mode is real and reproducible for GLM; it is *not* a calibrated cross-model rate.
- **Like-for-like only for Nemotron vs GLM** (both TP=8 on iscp001, §3) — that pair is
  a clean same-hardware serving benchmark. Qwen used a different GPU footprint, so its
  latency is descriptive of *its* setup, not a head-to-head. Part of the Nemotron–GLM
  gap is GLM's forced eager-mode (vLLM CUDA-graph deadlock), not pure NVFP4-vs-FP8 —
  the NIM-vs-vLLM experiment isolates stack from model/precision.
- `run_scimilarity` timing varied by node; treat absolute tool-time cross-run deltas as noise.

## Next
1. **Make the coordination harness active — general spine + agency** (design spec:
   `docs/coordination_harness_design.md`). Not an annotation-only patch: add a general
   **Obligations layer** in `world_state` (the missing symmetric half of the existing
   `blocked_actions` prerequisite gate), enforced at terminal/entry points, with
   explicit model agency (headroom) as its complement. Two obligations register now:
   - **`annotation_finalize`** (completion / 2a) — terminal gate can't `_complete_run`
     while `annotation_validation` is staged-but-unfinalized → re-prompt
     (`stage → finalize`, or `save_data(allow_unvalidated)`), bounded counter, safe
     fallback; plus convergence pressure once marker-query budget is exceeded.
   - **`batch_decision`** (entry / 2b) — auto-trigger the existing `multi_sample_strategy`
     selector on inspect-time multi-sample detection; convert advisory → enforced.
     **Moot for the LuCA eval (single-sample)** — general-robustness hardening, not a
     confound in the annotation numbers.

   Then re-run **all three** models at n=4 on the identical fixed harness.
   *Predicted:* GLM completion 1/4 → 4/4; accuracy band unchanged (it's a floor);
   Qwen/Nemotron stay 4/4 (never bound). Demonstrates the thesis *by intervention*.
2. **Nail down completion rate.** GLM accuracy is n=1-completed and the non-convergence
   rate is undersampled (one n=4 run/model). After the fix, more reps to estimate the
   pre/post completion rate per model with confidence.
3. **NVFP4 vs FP8 serving (NVIDIA-collab).** §3 already gives a clean same-hardware
   point (Nemotron NVFP4 vs GLM FP8, both TP=8 on iscp001: ~2× wall, ~5× per-call).
   Extend with the NIM-vs-vLLM axis (same model, two stacks) to isolate stack from
   model/precision; resolve how much of the GLM gap is its forced eager-mode.
4. **Downstream-consequence demo** — single-cell twin of the VirBench phylo-tree
   figure: show annotation/trajectory variability shifts cell composition / DE.
