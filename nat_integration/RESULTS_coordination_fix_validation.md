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
- **Nemotron LuCA reps=4** (same serving pre/post, iscp001 TP=8): scoreable
  **4/4 → 3/4**, mean major **0.844 → 0.852** (3 scored, held). The one miss (rep0)
  is **not a regression**: it *attempted* `finalize_annotation`, validation failed,
  and the **pre-existing** escape hatch saved `..._UNVALIDATED.h5ad` (scorer then
  returns no atlas accuracy — the *same* error Nemotron's pre-fix n=3 rep0 threw).
  My completion gate **did not fire** (`spine interventions: []`) because finalize
  *was* attempted — this is a different, stochastic "finalize-validation-fails →
  unvalidated" mode, outside the floor's scope, not caused by the fix.

## 4. Agency — does the contract add exploration, and for whom?

Trajectory proxies (`nat_integration/agency_compare.py`), pre vs post, **all 3 models**:

| proxy (mean over reps) | GLM-744B | Qwen-27B | Nemotron-550B |
|---|---|---|---|
| investigative `run_code` | 5.25 → **7.0** ↑ | 10 → 7 ↓ | 14.25 → 9.75 ↓ |
| `generate_figure` | 1.25 → **2.25** ↑ | 2.0 → 2.25 | 1.0 → 1.0 |
| `panglaodb_queries` | 11 → 11.75 | 6.5 → 6 | 12 → **15.75** ↑ |
| n_tools | 32.75 → 40.5 | 38.25 → 36 | 48.75 → 47 |
| finalized (scoreable) | 0.25 → **1.0** | 1.0 → 1.0 | 1.0 → 0.75* |

**Honest reading — this refines the hypothesis rather than confirming the naive version:**
- **The contract clearly helped only GLM** — the model that was over-deliberating
  *without finishing*. It gained grounded exploration (`run_code` +1.75, figures +1.0)
  **and** completion (1/4 → 4/4), accuracy held. Clean (same serving).
- **For the already-exploratory models (Qwen, Nemotron), it did NOT add agency** —
  `run_code` went *down* for both, and Nemotron shifted toward *more* marker-querying
  (`panglaodb` 12 → 15.75). The "converge hard on required steps" half of the contract
  appears to **dominate** the "explore freely" half for models already exploring plenty.
- So it is **not** a general "bigger ⇒ more agency" effect. It is: *the contract
  rescues the model that was failing; it slightly tightens models that were already
  fine.* Mechanism: GLM was the only one whose capability was leaking into
  non-convergence — the floor converts that leak into completion + investigation.
- **Confound:** the contract also *lengthened* the system prompt, which itself shifts
  behavior; the `run_code` dip for Qwen/Nemotron may be partly that, not the wording.
- (\* Nemotron's 0.75 is the unvalidated-finalize miss in §3 controls, not a bail.)

## 5. What worked / takeaways

**Solid (the floors — the core fix):**
- **Floors work and are model-agnostic.** The two verified failures are fixed; the
  fix never fires when it shouldn't (single-sample, already-finalized, Qwen).
- **Floor, not ceiling.** GLM accuracy unchanged (0.810 → 0.808); Nemotron held
  (0.844 → 0.852 on scored). The gate only recovers lost completions.
- **Thesis, by intervention.** Pre-fix, completion diverged by model (GLM 1/4 vs
  Qwen/Nemotron 4/4). Post-fix, GLM converges to 4/4 — *good harness → the model
  matters less* — accuracy stays model-independent (~0.81–0.85) as before.

**Equivocal (the agency contract):**
- It **helps the failing model** (GLM) but does **not broadly increase exploration**;
  `run_code` dropped for both already-exploratory models (Qwen, Nemotron). The
  "explore freely" half is being out-weighed by the "converge hard" half + the longer
  prompt. **Not** a clean "bigger ⇒ more agency" result — needs rebalancing or is
  model-dependent. This is the part of the design still open.

## 6. Bug fixes found via interactive testing (committed)
Two issues surfaced when running `scagent start` interactively (not in autonomous eval):
- **Interactive pause-guard.** The obligation terminal gate fired *while the agent
  was correctly paused* at a `pause_and_ask` checkpoint (awaiting the user), causing a
  thrash loop. Fix: the gate no-ops when `_pending_checkpoint` is set (the GLM
  completion bug had no checkpoint, so it's still caught). Regression test added.
- **Raw-counts surfacing.** `has_raw_counts` was wired to `has_raw_layer` (separate
  raw layer only), so a raw-count X with no layer (e.g. `*_raw.h5ad`) reported
  `has_raw_counts: false` — confusing the model (it burned reasoning figuring out the
  data state) and wrongly flagging `normalize` as blocked. Fix: `has_raw_counts` =
  available anywhere (layer / `adata.raw` / X-is-counts) + new explicit
  `x_is_raw_counts`. 3 regression tests added. Full suite **191 passing**.

## Caveats / next
- n=4 per cell; completion is a clear move for GLM (1/4 → 4/4) but rates/agency means
  are modest-n — more reps would tighten them.
- **Agency contract needs rebalancing** (or is model-dependent): it didn't increase
  exploration for already-exploratory models. Consider strengthening the "explore"
  half, or accept it as "rescues the failing model" only.
- **Nemotron's stochastic "finalize-validation-fails → unvalidated" mode** is separate
  from the floor (the gate doesn't fire — finalize *was* attempted). Fixing it means
  investigating *why* finalize validation fails on that rep, not the coordination layer.
- Batch entry floor uses early surfacing + terminal nudge (no forced fallback for
  *entry*); a hard finalize-block for unresolved batch + autonomous auto-open of
  `multi_sample_strategy` remain deferred (surfacing was sufficient here: 3/3).

## Artifacts
- Configs: `luca_eval_{glm,qwen,nemotron}_postfix.yml`; launchers
  `run_{glm,qwen,nemotron}_postfix_eval.sh`, `run_nemotron_batchprobe.sh`,
  `run_nemotron_singlesample.sh`.
- Eval outputs: `nat_luca_out_{glm,qwen,nemotron}_postfix/`, `nat_luca_out_glm_prefix/`
  (backup), per-rep run dirs in `nat_runs/`.
- Analysis: `nat_integration/agency_compare.py`.
- Tests: `tests/coordination_obligations_test.py` (incl. paused-checkpoint guard),
  `tests/raw_counts_detection_test.py`.
