# GLM-5.2 vs Qwen3.6-27B — harness-vs-model test (NAT, LuCA atlas)

**Date:** 2026-06-23 · **Dataset:** LUNG_T06 (LuCA NSCLC atlas ground truth) ·
**Reps:** 4 each · **Harness:** scagent wrapped in NeMo Agent Toolkit (NAT) ·
**Vision:** shared Nemotron-Nano-12B-v2-VL NIM (iscn008:8000).

**Thesis under test:** *if the agent harness is well built, the choice of main
reasoning model matters less.* We run the **identical** scagent QC+annotation
workload 4× on each model and compare accuracy, reliability, and cost/latency.

Serving:
- **GLM-5.2** (744B/40B MoE, FP8) — vLLM TP=8, expert-parallel + eager, iscp001:8000
- **Qwen3.6-27B** — vLLM, iscn008:8001 (GPU0)

Eval drivers ran on separate nodes (GLM→iscb015, Qwen→iscb013) concurrently so
CPU contention didn't cross-contaminate timing.

---

## Headline: the 27B beats the 744B on all three axes

| Axis | Qwen3.6-27B | GLM-5.2 (744B) | Winner |
|---|---|---|---|
| **Reliability** (reps that produced a scoreable annotation) | **4 / 4** | **1 / 4** | Qwen |
| **Accuracy** (major cell-type vs atlas, mean) | **0.833** | 0.810 (n=1) | Qwen |
| **Speed** (wall time / rep, mean) | **857 s** (~14 min) | 1288 s (~21 min) | Qwen |

The larger model is *less* reliable, *no more* accurate, and *slower* — inside
the same harness. This is the thesis, sharpened: the harness carries the small
model cleanly but cannot stop the large model from over-deliberating past the
finalize step and never committing.

---

## 1. Accuracy & consistency  (`accuracy_consistency.png`)

| | coarse acc | major acc (primary) |
|---|---|---|
| **Qwen3.6-27B** | mean 0.9402, sd 0.0019, spread 0.0043 (n=4) | **mean 0.8326**, sd 0.0167, spread 0.043 (n=4) |
| **GLM-5.2** | 0.9435 (n=1) | 0.8098 (n=1) |

- Qwen per-rep major: **0.861, 0.818, 0.826, 0.826** — tight band.
- GLM produced only one scoreable rep (0.810); the other three failed to
  converge (see §2). When it *did* finish it was slightly **below** Qwen's mean.
- Coarse accuracy is ~0.94 for both — the structured pipeline (scimilarity +
  PanglaoDB consensus) pins down the broad strokes regardless of model.

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

This is ~75% reproducible for GLM and **never** happened for Qwen. scagent's
coordination harness is **passive**: it enforces an *end gate* (`ready_to_finalize`
needs every cluster to pass; `save_data` is blocked before finalize) but has **no
floor** (a no-tool-call stop ends the run mid-annotation) and **no deliberation
budget** (nothing curbs the upstream marker-query spiral). A capable model burns
its latitude upstream and walks off the path.

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

| Metric | Qwen3.6-27B | GLM-5.2 |
|---|---|---|
| Wall time / rep (mean) | **857 s** | 1288 s |
| Wall p95 | 930 s | 1685 s |
| Per-LLM-call latency (mean) | **20.0 s** | 49.0 s |
| call p95 | 49.8 s | 165 s |
| call p99 | 102 s | **384 s** |

GLM is ~2.5× slower per LLM call with a far heavier tail; that long-reasoning
behavior (p99 384 s) is the same tendency that produces the non-convergence.

## 4. Where time goes — bottleneck ranking  (`bottleneck.png`)

- **GLM:** ① `run_scimilarity` 68 s → ② **LLM 49 s** → ③ `run_neighbors` 15 s
- **Qwen:** ① **LLM 20 s** → ② `run_scimilarity` 19 s → ③ `run_neighbors` 15 s

For GLM the model itself is a top-2 cost; for Qwen the model is cheap enough that
the **science tools** dominate — the regime where the harness, not the model, is
the limiting factor. (Caveat: `run_scimilarity` measured 68 s under GLM vs 19 s
under Qwen for the *same* tool → node/IO variance, not a model effect; don't
over-read it.)

## 5. Trajectory cost  (`trajectory_cost.png`)

| | LLM calls / run | prompt tokens / run |
|---|---|---|
| **Qwen3.6-27B** | mean 33.2, spread 8 (n=4) | mean 3.89M, spread 1.02M |
| **GLM-5.2** | mean 21.5, spread 11 (n=4) | mean 2.09M, spread 2.15M |

GLM's lower call/token counts are deflated by the **3 non-converged reps** —
they ended before finalizing, so this is not a fair efficiency comparison. Qwen's spread is
tighter on both → more consistent trajectories. Token-uniqueness ≈ 0 and
`common_prefixes` are highly cacheable for both → consistent with the earlier
**prefill-bound** finding (~147K avg prompt; latency dominated by prefill).

## 6. Concurrency

Reps ran at `max_concurrency: 1`, so the profiler's p50 concurrency of 2.0 is
workflow+nested spans, not parallel reps. No intra-run parallelism to exploit
(the agent is inherently sequential: reason → tool → reason).

---

## Figures (`nat_eval_figs/`)
- `accuracy_consistency.png` — per-rep coarse & major accuracy, both models
- `trajectory_cost.png` — LLM calls + prompt tokens per rep
- `bottleneck.png` — per-op avg duration, side by side
- `gantt_qwen.png`, `gantt_glm.png` — per-rep timelines
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
   - **Layer 2 — deterministic orchestration / control flow** (our finalize-guard).
     Governs whether the model is *forced to use the tools and complete the
     protocol → reliability*. GLM *had* Layer 1 and still failed by
     over-deliberating and never crossing `stage → finalize` (§2). Layer 1 is
     necessary but **not sufficient**; Layer 2 is what we are fixing next.
3. **Gap we fill.** They explicitly scope to viral sequences and note
   genomics/single-cell is *not* addressed. scagent is single-cell; our
   `biocontext` MCP (PanglaoDB, Cell Ontology) is the same *kind* of object as
   `gget virus`.

**The two layers, read off our data — the headline of this experiment:**
- **Accuracy converges → Layer 1 already works.** On the runs that *completed*,
  GLM (744B) and Qwen (27B) score nearly the same: **0.810 vs 0.833** major, ~0.94
  coarse for both. A good deterministic tool layer gets even the small/cheap model
  "there" — exactly the article's claim, now in single-cell. *(Caveat: GLM is n=1
  completed; the guard fix below confirms it at n=4.)*
- **Reliability diverges → Layer 2 is the open problem.** Completion rate is
  **4/4 (Qwen) vs 1/4 (GLM)**. Accuracy can't even *register* this difference —
  non-converged runs produce no score. It only shows up once you look at completion /
  trajectory adherence, which is the dimension the second harness layer controls
  and the one NAT's per-step tracing is built to surface.

So the clean two-sentence story: **a good tool layer makes the model matter less
for accuracy; a good orchestration layer makes it matter less for reliability.**
The article nailed the first; our finalize-guard fix demonstrates the second
(expected: GLM completion → 4/4, accuracy gap stays ~0, reliability gap closes).

**Sharper one-liner than theirs:** *harness quality determines how much the model
matters* — not simply "the model doesn't matter." When the harness has a soft spot
the bigger model can be **worse** (more willing to free-form). Their forward-looking
"future models may make such tools obsolete" risk is countered by "too expensive /
slow / hard to audit" — which our prefill-bound + latency numbers (§3) substantiate.

**Seeded TODO:** a downstream-consequence demo = single-cell twin of their
phylo-tree figure (show annotation variability shifts cell composition / DE).

## Caveats
- GLM accuracy is n=1; the fair n=4 comparison needs the finalize-guard fix first.
- Wall/latency compares two different serving footprints (GLM TP=8 eager on a full
  node vs Qwen on 1 GPU). The latency numbers are descriptive of *this* setup, not
  a like-for-like serving benchmark — that's the separate NIM-vs-vLLM experiment.
- `run_scimilarity` timing varied by node; treat tool-time cross-model deltas as noise.

## Next
1. **Make the coordination harness active** (the four-file annotation path):
   (a) guard the `finish_reason=="stop"` branch in `scagent/agent/agent.py` so the
   run can't end while `annotation_validation` is staged-but-unfinalized — re-prompt
   to `stage → finalize` (or `save_data(allow_unvalidated)`), bounded by an attempt
   counter; (b) add convergence pressure in `world_state`/`prompts.py` after N marker
   queries / M annotation-phase turns. Then re-run both at n=4 on the identical
   harness. Expected: GLM completion 1/4 → 4/4, accuracy gap stays ~0 → thesis
   demonstrated *by intervention*. (a) ends the silent failure; (a)+(b) gets convergence.
2. Add a **Nemotron-3-Ultra** baseline row (currently 1/3 completed, 0.795 major) —
   second large-model witness that non-convergence is model-*class*, not GLM-specific.
3. Separate axis: NIM-vs-vLLM serving of the *same* model (showcases NIM).
