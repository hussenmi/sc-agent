# Coordination harness: spine + agency (design spec)

**Status:** proposed · **Date:** 2026-06-23 · **Branch:** `harness-coordination-fix`
(developed in a separate worktree so the eval's editable install is untouched until
applied).

## Motivation (what the evals showed)

Running the identical scagent workload across Qwen3.6-27B, GLM-5.2 (744B), and
Nemotron-3-Ultra (550B) — see `nat_integration/RESULTS_harness_vs_model.md` — surfaced
two failure modes that are **not** about model capability and **not** fixed by
prompting harder:

- **Completion failure (2a).** GLM entered the annotation workflow
  (`prepare_annotation`) and then over-deliberated — re-adjudicating PanglaoDB
  markers forward and reverse for every competing label — and **never crossed
  `finalize_annotation`**. The run ended on a no-tool-call "stop" turn while
  annotation was staged-but-unfinalized; the harness completed silently
  (`annotated_h5ad: null`, `rc=0`). 3/4 GLM reps; 0/4 for Qwen and Nemotron.
- **Entry failure (2b).** On a multi-donor dataset (Reyfman, 8 donors), Nemotron
  **never opened the batch-effect decision** — no `diagnose_batch_effect`, no
  integration choice surfaced — despite the `multi_sample_strategy` protocol being
  *exhaustively* prompted (`prompts.py` L59/L232). GLM, on the same data and same
  prompt, surfaced it immediately and correctly. The branch is maximally prompted
  and still obeyed **model-dependently**.

Both are coordination-harness gaps, on two directions of the same axis. And both
prove the load-bearing principle:

> **Advisory prompting is necessary but not sufficient. Model-robustness requires
> *structural* enforcement of the mandatory spine — floors, not ceilings — while
> value-add is left to model agency (headroom).**

This is the single-cell, control-flow extension of the Anthropic+NCBI
"agents in biology" finding: they made the *tool/retrieval* layer deterministic
(accuracy stops depending on the model); we must also make the *coordination* layer
deterministic (reliability stops depending on the model).

## The codebase already has half the gate

`world_state` computes `available_actions` / `blocked_actions` — a **prerequisite
gate** ("can't `run_deg` before clustering"; "can't `run_pseudobulk_deg` without raw
counts"). That is *forward* gating. The two bugs live in the missing directions:

| Direction | Question | Exists? | Bug |
|---|---|---|---|
| Prerequisite (forward) | "May I start X yet?" | ✅ `blocked_actions` | — |
| **Entry obligation** (2b) | "Data demands X — has it been opened?" | ❌ | batch skip |
| **Completion obligation** (2a) | "X was started — was it finished before exit?" | ❌ | annotation non-finalize |

The fix is the **symmetric other half** of an abstraction the code already trusts —
not a new parallel subsystem.

---

## Part A — the Obligations layer (general spine)

One registry in `world_state`. Each obligation is **data-driven** (a predicate over
world/data state), not hard-coded to a specific tool, so new tools add an obligation
instead of a bespoke guard.

```python
@dataclass
class Obligation:
    key: str                       # "batch_decision", "annotation_finalize", ...
    kind: str                      # "entry" | "completion"
    triggered: Callable[[ws], bool]   # when does this become required?
    satisfied: Callable[[ws], bool]   # when is it met?
    guidance: str                  # what to tell the model to do about it
    blocks_terminal: bool = True   # may the run complete/save/report while unmet?
```

- `world_state.unmet_obligations()` = `[o for o in REGISTRY if o.triggered(ws) and not o.satisfied(ws)]`.
- The registry is **open**: this pass registers only the two we have evidence for;
  QC-decision, integration-scoring-after-correction, etc. are fast-follows that
  *register*, they don't get new plumbing.

**Registered now:**

| key | kind | triggered when | satisfied when | guidance |
|---|---|---|---|---|
| `annotation_finalize` | completion | `annotation_validation.required and entered` | `annotation_validation.finalized` | "You staged annotation but did not finalize. Call `stage_annotation_evidence` → `finalize_annotation` now, or `save_data(allow_unvalidated=true)`. Do not run more marker queries." |
| `batch_decision` | entry | inspect-time finds ≥2 sample-like groups (low-cardinality `donor`/`sample`/`batch` col) and no `multi_sample_strategy` decision resolved | `multi_sample_strategy` resolved (any option) **or** moot (single sample) | "This dataset has N sample-like groups. Resolve `multi_sample_strategy` (investigate / integrate / keep / separate) before clustering+annotation." |

Note both `satisfied` predicates enforce that the **decision is made**, not a
particular *outcome* — so they are floors, never ceilings. GLM (which already
surfaces batch and finalizes annotation) is never bound by either.

## Part B — enforcement points (general, not per-case)

1. **Terminal gate** — before `_complete_run` / `save_data` / `write_report`, in the
   `finish_reason in {"stop","length"}` branches of all three chat loops: if
   `unmet_obligations()` (with `blocks_terminal`) is non-empty → **re-prompt** with
   each obligation's `guidance`, bounded by an attempt counter (mirrors
   `_maybe_continue_after_failure`). After K attempts → **safe fallback**
   (`save_data(allow_unvalidated=true)` for completion obligations) so the run never
   returns a silent null. Catches **2a** and any future early-exit.
2. **Entry surfacing** — when an `entry` obligation becomes `triggered` but unentered,
   push it into the model's live context via the existing `next_action` / decision
   channel, so it is *in front of* the model, not buried in a static prompt. In
   autonomous/NAT mode (no user), the fallback is the harness **opening the decision
   itself** (auto-`investigate_integration` → `diagnose_batch_effect` → default),
   not relying on the model to invoke it. Catches **2b**.
3. **Telemetry** — emit unmet/late obligations as step-log events so NAT surfaces
   **"spine adherence"** as a measurable metric across *any* obligation (this is what
   makes the floor effect quantifiable for the article).

## Part C — the agency contract (general headroom)

The spine is *what must happen*; agency is *freedom in how you get there*. Crucially,
agency lives **inside** each spine step, not only in the gaps between steps.

**The contract (one prompt section, system-wide):**
> Within any phase you have full agency to investigate — look at a suspicious
> cluster's genes, run a DEG, plot a marker, test a hypothesis, generate a figure
> beyond the tool-enforced ones — **provided you reason through each step** (state
> the suspicion, what you checked, what you concluded) and **converge back to the
> required decision**. Exploration is the *means* of justifying a spine checkpoint;
> it may never *replace* or skip one. Converge hard on closed/required steps;
> explore freely on open questions.

**Concrete prompt change:** the current anti-busywork rules (e.g. "after QC, do not
run extra `run_code` unless there's a specific anomaly", `prompts.py` L242/L117) are
*blanket* and would suppress exactly the legitimate in-phase investigation we want.
Convert them from **blanket prohibitions** to **reasoned-exception** rules:
> No aimless `run_code` between pipeline steps — **but** if you have a specific
> suspicion (a cluster looks like doublets/dying/ambiguous), investigate it: look at
> its genes, run a DEG, plot it. State the suspicion and the verdict. "Don't dawdle"
> means "don't dawdle *without a reason*"; with a reason, dig in.

This resolves the tension the evals exposed: we want to *encourage* the capable
model's hypothesis/mechanism/extra-plot behavior (where the article says models add
value), while still guaranteeing the required decisions happen.

---

## File map (per AGENTS.md: annotation logic spans four files; change together)

- **`world_state.py`** — `Obligation` dataclass + `OBLIGATION_REGISTRY`,
  `unmet_obligations()`, `annotation_unfinalized()` / `marker_query_count()` helpers,
  multi-sample detection at inspect-time, `next_action` surfacing for entry obligations.
- **`agent.py`** — `_maybe_continue_for_obligations(messages, attempts)` (mirror of
  the failure-recovery hook); wire into `stop`/`length` branches of all loops; thread
  an `obligation_nudge_attempts` counter; safe fallback; emit telemetry events.
- **`tools.py`** — `convergence_hint` field on the PanglaoDB tool result once
  `marker_query_count()` exceeds a per-required-cluster budget (the soft pressure
  that makes the nudge land); the auto-fallback `save_data(allow_unvalidated)` path.
- **`prompts.py`** — the agency contract section (Part C); convert blanket
  anti-busywork rules to reasoned-exception; keep the targeted convergence pressure.

## Tests (`tests/*_test.py`, suffix per repo convention)

- Mock a model that does `prepare_annotation` → no-tool-call stop: assert the
  terminal gate re-prompts, and after K attempts auto-saves `_UNVALIDATED`
  (**never** a silent null).
- Mock multi-sample data at inspect-time: assert `batch_decision` obligation
  triggers, is surfaced, and blocks the terminal gate until resolved; assert it is
  **moot** (no trigger) on single-sample data (so it can't fire on the LuCA eval).
- Assert obligations are **floors**: a trajectory that already finalizes / already
  surfaces batch sees **no** nudge (GLM-equivalent path unaffected).
- Assert the `convergence_hint` fires only past budget.

## Scope & sequencing

- **This pass:** Part A (2 obligations) + Part B (terminal gate + entry surfacing +
  telemetry) + Part C (agency contract + reasoned-exception). 100% model-agnostic.
- **Validation:** re-run all three models at n=4 on the identical fixed harness.
  Predicted: GLM completion 1/4 → 4/4; accuracy band unchanged (it's a floor);
  Qwen/Nemotron unchanged (already 4/4). The batch gate is **moot for the LuCA eval**
  (single-sample) — it is general-robustness hardening + a second article example,
  not a confound in the annotation numbers.
- **Fast-follows (register only):** QC-decision-made, integration-scoring-after-
  correction, and any new tool that introduces a required decision.
