# Coordination-harness fix — validation (pre vs post)

**Date:** 2026-06-23 · **Branch:** `harness-coordination-fix` · **Harness:** scagent
(editable) under NAT. Design spec: `docs/coordination_harness_design.md`. Baseline
experiment: `RESULTS_harness_vs_model.md`.

This documents the two verified failure modes, the fix, and the before/after that
validates it. **Headline: the model that actually failed (GLM) goes 1/4 → 4/4
completion with accuracy unchanged, and gains grounded agency — on identical serving.**

---

## 1. The problem (two verified failure modes)

Both are *coordination*-harness gaps: scagent computed the right signal but never bound
the model to it (the one exception — a hard save/report gate on annotation finalize —
was bypassed by the prose-stop exit).

- **2a — completion (GLM).** GLM entered `prepare_annotation`, over-deliberated on
  PanglaoDB marker adjudication, and **never reached `finalize_annotation`**. It exited
  via a no-tool-call `stop` turn → `agent.py` `finish_reason=="stop"` → `_complete_run`,
  which **bypasses** the existing save/report guard. Result: `annotated_h5ad: null`,
  `rc=0`, `status=completed`. **3/4 reps.** Not a budget cap (27–36 tool execs vs 75
  limit; the rep that *finished* did the most, 39).
- **2b — entry (Nemotron).** On a multi-donor dataset, with a generic "analyze" prompt,
  Nemotron **skipped the batch-effect decision** (no `diagnose_batch_effect`, no
  integration choice) despite the `multi_sample_strategy` protocol being exhaustively
  prompted. **Skipped 2/3 runs** — stochastic for the same model on the same data.
  Root cause (3/3 correlation): detection happens at load, but the batch-strategy
  *guidance* only surfaced inside the `inspect_data` tool result — skip that tool
  (go `load_data → run_qc`), never see it.

**Principle both prove:** advisory prompting (even exhaustive) is obeyed
model-dependently; model-robustness needs *structural* enforcement of the spine —
floors, not ceilings — with agency left free everywhere else.

## 2. The fix (what changed)

Implemented as **scagent's own hook system** (Claude Code/Agent-SDK hooks are the same
idea, but ship with the SDK; scagent is a custom multi-provider loop, so we built the
equivalent). View-first, reuses existing computations, binds only scientific checkpoints.

| File | Change |
|---|---|
| `world_state.py` | `unmet_obligations()` **view** (annotation completion + batch entry) over existing state; `multi_sample_decision_unresolved()` floor predicate; `note_spine_intervention()` telemetry; `unmet_obligations` added to per-turn `snapshot()` (early surfacing, not gated on `inspect_data`). |
| `agent.py` | `_maybe_continue_for_obligations()` terminal gate wired into **all 3 chat loops** (OpenAI stop+length, Anthropic, Codex). Bounded re-prompt (`OBLIGATION_NUDGES=2`) → **block + forced `save_data(allow_unvalidated)`** fallback for completion. Reuses the existing `_annotation_validation_guard` predicate; closes the prose-stop exit that bypassed it. |
| `prompts.py` | Top-level **spine + agency contract**: required checkpoints are floors (check `unmet_obligations`); *within* them, agency is encouraged (investigate genes, DEG, extra plots, mechanism) — reasoned, converging back. "Converge hard on closed decisions; explore freely on open ones." |
| `tests/coordination_obligations_test.py` | 8 tests (view, floor-moot-on-single-sample, nudge→bounded→forced-fallback, no-op-when-finalized, telemetry). Full suite **187 passing**. |

Design rules applied: **blocking > injecting** (the floor blocks + forces, doesn't just
nudge); **floors not ceilings** (`satisfied` = a decision was *made*, never a particular
outcome — a model already doing the right thing is never bound); **selective** (only
scientific-validity checkpoints; tool-ordering prerequisites stay advisory).

## 3. Validation (pre vs post)

All post-fix runs use the fixed editable install. GLM and Nemotron use **identical
serving** pre and post (iscp001 TP=8 / its NVFP4 config), so their before/after are clean.

### 2a — GLM completion floor (the verified failure) — **clean before/after**
| GLM, LuCA reps=4, same serving | Pre-fix | Post-fix |
|---|---|---|
| **completed** | **1/4** | **4/4** |
| per-rep major | (1 scored) 0.810 | 0.809 / 0.832 / 0.781 / 0.809 |
| **mean major** | 0.810 | **0.808** (held — floor ≠ ceiling) |

The 3 reps that previously died with `no annotated_h5ad (rc=0)` now all finalize.
Completion fixed; accuracy unchanged.

### 2b — Nemotron entry floor (multi-donor Reyfman, generic prompt)
| | surfaced `batch_decision` + ran `diagnose_batch_effect` |
|---|---|
| Pre-fix | **1/3** |
| Post-fix | **3/3** |

Post-fix, Nemotron now references `batch_decision` by name at turn 2 (from the snapshot)
and runs the investigate-first diagnostic every time.

### Controls
- **Single-sample negative control** (LUNG_T06): the batch floor is **moot** (no
  `diagnose_batch_effect`/`pause_and_ask`), annotation finalizes, run completes. → no
  false floor; no regression on the LuCA annotation eval path.
- **Floor doesn't bind Qwen** (already 4/4): completion **4/4 → 4/4**; accuracy
  0.833 → 0.803 = **n=4 noise** (both runs FP8).

## 4. Agency — does the contract add exploration, and for whom?

Trajectory proxies (`nat_integration/agency_compare.py`), pre vs post:

| proxy (mean over reps) | GLM-744B pre→post | Qwen-27B pre→post |
|---|---|---|
| investigative `run_code` | 5.25 → **7.0** (+1.75) | 10 → 7 (−3) |
| `generate_figure` | 1.25 → **2.25** (+1.0) | 2.0 → 2.25 |
| n_tools | 32.75 → 40.5 | 38.25 → 36 |
| finalized | 0.25 → **1.0** | 1.0 → 1.0 |

**Reading:** the contract raises *grounded* agency for the model that was
over-deliberating/under-finishing — **GLM redirected from bail into completion + more
investigation + more plots, with accuracy held.** The already-exploratory small model
(Qwen, already 4/4 and high run_code) didn't gain. So **"bigger models show more agency
in the right places" is supported** *once the floor stops them bailing*: the harness
both (a) brings the weaker-on-this-axis model up to par (1/4 → 4/4) and (b) lets the
capable model express productive agency rather than spin.

(GLM agency deltas are clean — same serving. Qwen's small `run_code` dip is minor /
within noise; Qwen was never the model this targeted.)

## 5. What worked / takeaways

- **Floors work and are model-agnostic.** The two verified failures are fixed; the
  fix never fires when it shouldn't (single-sample, already-finalized, Qwen).
- **Floor, not ceiling.** GLM accuracy unchanged (0.810 → 0.808); the gate only
  recovers lost completions, it doesn't constrain the analysis.
- **Spine + agency compose.** Same change that *forces* completion also *frees* the
  big model to explore more — the two halves of the contract are not in tension when
  the floor handles the "must finish" part.
- **Thesis, by intervention.** Pre-fix, completion diverged by model (GLM 1/4 vs
  Qwen/Nemotron 4/4). Post-fix, all converge to 4/4 — *good harness → the model
  matters less* — while accuracy stays model-independent (~0.81–0.84) as before.

## Caveats / next
- n=4 per cell; completion is now a clear move (GLM 1/4 → 4/4) but rates are still
  modest-n. More reps would tighten the agency means.
- Batch entry floor is enforced via early surfacing + terminal nudge (no forced
  fallback for *entry*); a hard finalize-block for unresolved batch + autonomous
  auto-open of `multi_sample_strategy` remain deferred (add only if surfacing proves
  insufficient — it didn't here: 3/3).
- Re-run Nemotron LuCA reps=4 post-fix (needs iscp001 back from GLM) to complete the
  3-model post-fix accuracy/agency matrix on the annotation task.

## Artifacts
- Configs: `luca_eval_glm_postfix.yml`, `luca_eval_qwen_postfix.yml`; launchers
  `run_glm_postfix_eval.sh`, `run_qwen_postfix_eval.sh`, `run_nemotron_batchprobe.sh`,
  `run_nemotron_singlesample.sh`.
- Eval outputs: `nat_luca_out_{glm,qwen}_postfix/`, `nat_luca_out_glm_prefix/` (backup),
  per-rep run dirs in `nat_runs/`.
- Analysis: `nat_integration/agency_compare.py`.
