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
model cleanly but cannot stop the large model from talking its way out of the
structured workflow.

---

## 1. Accuracy & consistency  (`accuracy_consistency.png`)

| | coarse acc | major acc (primary) |
|---|---|---|
| **Qwen3.6-27B** | mean 0.9402, sd 0.0019, spread 0.0043 (n=4) | **mean 0.8326**, sd 0.0167, spread 0.043 (n=4) |
| **GLM-5.2** | 0.9435 (n=1) | 0.8098 (n=1) |

- Qwen per-rep major: **0.861, 0.818, 0.826, 0.826** — tight band.
- GLM produced only one scoreable rep (0.810); the other three **bailed**
  (see §4). When it *did* finish it was slightly **below** Qwen's mean.
- Coarse accuracy is ~0.94 for both — the structured pipeline (scimilarity +
  PanglaoDB consensus) pins down the broad strokes regardless of model.

## 2. Reliability — GLM's prose-bail failure mode

3 of 4 GLM reps ended with `"workflow produced no annotated_h5ad (rc=0)"`. Root
cause (from the trajectory logs): GLM's final turn was **prose, not a tool
call** — e.g. *"PanglaoDB results are very revealing… this is definitively
AT2…"* — instead of calling `finalize_annotation`. scagent's loop sees an
assistant message with no tool calls, treats the task as done, and exits
mid-annotation → nothing saved.

This is ~75% reproducible for GLM and **never** happened for Qwen. The harness's
finalize/save guard is strong enough to carry the 27B but **not** strong enough
to force the 744B back into the tool protocol. **That gap is the most
actionable finding** — strengthening the guard is the "improve the harness →
model gap shrinks" experiment.

## 3. Latency & throughput  (profiler / `inference_optimization.json`)

| Metric | Qwen3.6-27B | GLM-5.2 |
|---|---|---|
| Wall time / rep (mean) | **857 s** | 1288 s |
| Wall p95 | 930 s | 1685 s |
| Per-LLM-call latency (mean) | **20.0 s** | 49.0 s |
| call p95 | 49.8 s | 165 s |
| call p99 | 102 s | **384 s** |

GLM is ~2.5× slower per LLM call with a far heavier tail; that long-reasoning
behavior (p99 384 s) is the same tendency that produces the prose-bail.

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

GLM's lower call/token counts are inflated by the **3 truncated (bailed) reps** —
they ended early, so this is not a fair efficiency comparison. Qwen's spread is
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
2. **Extension (the more interesting half).** Their variability lives at the
   **data-retrieval** layer. Ours lives at the **control-flow** layer: GLM *had*
   the deterministic tools (`run_scimilarity` + PanglaoDB/Cytopus consensus) and
   still failed by talking its way out of the protocol (prose instead of
   `finalize_annotation`, §2). So a good harness is **two** deterministic layers:
   (a) deterministic *tools* (their gget virus ≈ our consensus enforcement) **and**
   (b) deterministic *orchestration* that forces the model to use them and finish.
   (a) is necessary but **not sufficient** — our finalize-guard fix is the
   intervention form of their argument.
3. **Gap we fill.** They explicitly scope to viral sequences and note
   genomics/single-cell is *not* addressed. scagent is single-cell; our
   `biocontext` MCP (PanglaoDB, Cell Ontology) is the same *kind* of object as
   `gget virus`.

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
1. **Strengthen the finalize guard** (`scagent/agent/agent.py` + prompts/tools/
   world_state — all four files) so a verbose model can't exit on prose mid-
   annotation; re-run both at n=4 on the identical harness. Expected: GLM accuracy
   recovered, gap narrows → thesis demonstrated *by intervention*.
2. Separate axis: NIM-vs-vLLM serving of the *same* model (showcases NIM).
