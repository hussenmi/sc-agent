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
- **Entry failure (2b).** On a multi-donor dataset (Reyfman, 8 donors), with a
  generic "just analyze" prompt, Nemotron **skipped the batch-effect decision in 2 of
  3 runs** — no `diagnose_batch_effect`, no integration choice surfaced — despite the
  `multi_sample_strategy` protocol being *exhaustively* prompted (`prompts.py`
  L59/L232). It is **stochastic for the same model on the same data** (skipped /
  honored / skipped), which is the strongest argument that advisory prompting cannot
  guarantee the floor.

  **Root cause (verified, 3/3 correlation): the floor is gated behind an optional
  step.** The multi-sample checkpoint is surfaced off `state.metadata_candidates`,
  which **`inspect_data` populates**. The one run that honored batch
  (`run_2026_06_23_162002`) called `inspect_data` first; both that skipped
  (`..._235405`, `..._173249`) went straight `load_data → run_qc`, never populating
  the candidates, so the `multi_sample_strategy` reason never surfaced. The "floor"
  is currently contingent on the model choosing to inspect. **Fix: detection must run
  deterministically at `load_data` / `world_state.update`, not depend on
  `inspect_data`.** (All three runs *did* set `batch_key="donor"` on
  `normalize_and_hvg` — the model knew it was multi-donor; it just never hit the
  decision gate.)

Both are coordination-harness gaps, on two directions of the same axis. And both
prove the load-bearing principle:

> **Advisory prompting is necessary but not sufficient. Model-robustness requires
> *structural* enforcement of the mandatory spine — floors, not ceilings — while
> value-add is left to model agency (headroom).**

This is the single-cell, control-flow extension of the Anthropic+NCBI
"agents in biology" finding: they made the *tool/retrieval* layer deterministic
(accuracy stops depending on the model); we must also make the *coordination* layer
deterministic (reliability stops depending on the model).

## The codebase already computes the picture — it just doesn't bind the model to it

The premise is **not** "forward gate works, two directions missing." Verified against
the code, scagent already *computes* the full operational picture in all three
directions and **binds the model to almost none of it**:

| Direction | Signal that exists | Enforced? |
|---|---|---|
| Prerequisite (forward) | `world_state.blocked_actions` (`run_deg` needs clusters, etc.) | **No** — computed + serialized into the snapshot (`world_state.py:369`), shown to the model, but **no consumer rejects a call**. Advisory. |
| **Entry** (2b) | `outstanding_decisions`, `multi_sample_strategy: needs_decision`, `next_action` | **No** — only serialized into context (`agent.py:934`). Advisory. *Also* contingent on `inspect_data` (see 2b root cause above). |
| **Completion** (2a) | `annotation_validation` (staged / finalized) | **Partially YES** — `_annotation_validation_guard` (`agent.py:1024`) fires at tool dispatch (`5715`) and **hard-blocks `save_data`/`write_report` before finalize** (`5740`: returns the block payload instead of executing). |

So the harness has exactly **one** hard gate — completion-on-save — and the GLM bug
slipped through the **one exit that gate doesn't cover**:

> The save/report gate guards the *tool-exit* door. GLM exited via the *prose-stop*
> door: a no-tool-call `stop` turn → `finish_reason=="stop"` (`agent.py:5266–5290`)
> calls `_complete_run` **directly, never invoking `_annotation_validation_guard`**.
> **GLM walked out the one unguarded door.**

This makes the fix smaller and more native than "build the symmetric half":

- **Completion (2a):** we **do not add a new guard** — we route the `stop`/`length`
  exits through the **existing** `_annotation_validation_guard` (the same check
  `save_data` already hits). One call site, reusing trusted logic.
- **Entry (2b) + prerequisites:** genuinely advisory today. Add enforcement here —
  but **selectively** (see "selective enforcement" below).
- **Implementation: view-first, not a new registry.** A thin
  `world_state.unmet_obligations()` **view** aggregates signals that already exist
  (`outstanding_decisions` open+triggered, `annotation_validation` staged-not-final,
  the load-time multi-sample detection). It reuses trusted computations and adds no
  parallel state. Promote to a formal `Obligation` registry (below) only if a
  3rd/4th obligation makes the aggregation ungainly — a deliberate later call, not
  the default.

**Selective enforcement.** Bind only **scientific-validity checkpoints** (batch
decision, annotation finalize; later QC-before-annotation, integration-scoring-after-
correction). Tool-ordering prerequisites in `blocked_actions` **stay advisory** —
hard-blocking them would break legitimate `run_code` use and make the spine brittle.
The spine stays thin and defensible: a few load-bearing scientific floors, not a
straitjacket on sequencing.

### Prior art: Claude Code / Agent-SDK hooks (we are building scagent's own)

Anthropic's own agent harness implements exactly this principle as **hooks** —
deterministic lifecycle callbacks: a `Stop` hook (`continue:false`) forces the agent
to keep going; `PreToolUse` can hard-**deny** a tool; `PostToolUse` injects context.
They added a deterministic coordination layer because advisory prompting wasn't
sufficient for their own agent — independent validation of this design.

We do **not** adopt those hooks directly: they ship with the Claude **Agent SDK**,
whereas scagent is a custom multi-provider loop (`client.chat.completions.create`
against vLLM / OpenAI / Groq / Gemini). Porting to the SDK would be Anthropic-centric
and defeat the open-model serving goal. Instead, **the Obligations layer is scagent's
hook system**, mapped 1:1:

| Claude Code hook | scagent equivalent |
|---|---|
| `PreToolUse` (deny) | the **existing** `_annotation_validation_guard` (hard-blocks `save_data`/`write_report` before finalize — the one real gate today). `blocked_actions` is *not* this — it's computed but advisory. |
| `Stop` (`continue:false`) | terminal obligation gate (Part B.1) — the missing exit |
| `PostToolUse` (`additionalContext`) | convergence hint after marker budget (Part C / B) |
| load/SessionStart hook | load-time `batch_decision` entry trigger |

**Design rule — blocking > injecting.** The hooks guidance is explicit that *injected
context is still model-discretion; only blocking / forced action is deterministic.*
This matches our data (advisory checkpoint skipped 2/3). Therefore the terminal gate
must **block the exit and auto-execute the fallback**, and the entry gate in
autonomous mode must **auto-open the decision** — not merely nudge "please finalize."
Nudges are the first, bounded attempt; the floor is the block + forced fallback.

---

## Part A — the unmet-obligations view (general spine)

**Implementation: view-first.** Start with a thin `world_state.unmet_obligations()`
that *aggregates signals already computed* — no new parallel state. The conceptual
`Obligation` shape below is the **target schema** the view returns (and the registry
we promote to only if a 3rd/4th obligation makes the inline aggregation ungainly):

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

- `unmet_obligations()` returns the obligations whose `triggered` holds and
  `satisfied` does not — aggregated from existing signals, not a separate store.
- **Open set**: this pass covers only the two we have evidence for; QC-decision,
  integration-scoring-after-correction, etc. are fast-follows that slot into the same
  view (or registry, if promoted) — no new plumbing.

**Registered now:**

| key | kind | triggered when | satisfied when | guidance |
|---|---|---|---|---|
| `annotation_finalize` | completion | `annotation_validation.required and entered` | `annotation_validation.finalized` | "You staged annotation but did not finalize. Call `stage_annotation_evidence` → `finalize_annotation` now, or `save_data(allow_unvalidated=true)`. Do not run more marker queries." |
| `batch_decision` | entry | **at `load_data` time** (deterministic detection, NOT gated on `inspect_data`): ≥2 sample-like groups (low-cardinality `donor`/`sample`/`batch` col) and no `multi_sample_strategy` decision resolved | `multi_sample_strategy` resolved (any option) **or** moot (single sample) | "This dataset has N sample-like groups. Resolve `multi_sample_strategy` (investigate / integrate / keep / separate) before clustering+annotation." |

Note both `satisfied` predicates enforce that the **decision is made**, not a
particular *outcome* — so they are floors, never ceilings. GLM (which already
surfaces batch and finalizes annotation) is never bound by either.

## Part B — enforcement points (general, not per-case)

1. **Terminal gate** — the genuinely-missing enforcement. In the
   `finish_reason in {"stop","length"}` branches (all three chat loops), **before**
   `_complete_run`, consult `unmet_obligations()` (completion kind). This is the same
   check `save_data`/`write_report` already hit via `_annotation_validation_guard` —
   we are **closing the one exit (`_complete_run`) that bypasses the existing gate**,
   not adding a new guard. If unmet → **re-prompt** with `guidance`, bounded by an
   attempt counter (mirrors `_maybe_continue_after_failure`). After K attempts →
   **block + safe fallback** (`save_data(allow_unvalidated=true)`) so the run never
   returns a silent null. (Per "blocking > injecting": the re-prompt is the bounded
   first attempt; the floor is the block + forced fallback.)
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

- **`world_state.py`** — `unmet_obligations()` **view** aggregating existing signals
  (`annotation_validation` staged-not-final, `outstanding_decisions`/`multi_sample_strategy`
  triggered-unresolved); `marker_query_count()` helper; **multi-sample detection moved
  to load-time** (in `update`/`load_data`, not gated on `inspect_data`); `next_action`
  surfacing for entry obligations. (No new `Obligation` dataclass/registry yet — view
  first; promote only if a 3rd/4th obligation makes it ungainly.)
- **`agent.py`** — route the `stop`/`length` branches through the **existing**
  `_annotation_validation_guard` logic (the missing exit), via a
  `_maybe_continue_for_obligations(messages, attempts)` wrapper (mirror of the
  failure-recovery hook); wire into all loops; thread
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
- Mock multi-sample data loaded **without** calling `inspect_data` (the skip path):
  assert `batch_decision` obligation
  triggers, is surfaced, and blocks the terminal gate until resolved; assert it is
  **moot** (no trigger) on single-sample data (so it can't fire on the LuCA eval).
- Assert obligations are **floors**: a trajectory that already finalizes / already
  surfaces batch sees **no** nudge (GLM-equivalent path unaffected).
- Assert the `convergence_hint` fires only past budget.

## Implementation status (2026-06-23, branch `harness-coordination-fix`)

**Done + tested** (`tests/coordination_obligations_test.py`, full suite 187 passing):
- **A** — `world_state.unmet_obligations()` view (annotation completion + batch entry),
  with `multi_sample_decision_unresolved()` floor predicate.
- **B1** — terminal completion gate wired into **all three** chat loops (OpenAI stop +
  length, Anthropic, Codex): re-prompt bounded by `OBLIGATION_NUDGES`, then forced
  `save_data(allow_unvalidated=true)` fallback for completion obligations. Reuses the
  same predicate as the existing `_annotation_validation_guard`; closes the prose-stop
  exit that bypassed it.
- **B2 (partial)** — entry surfacing: `unmet_obligations` is emitted in the per-turn
  snapshot, so the batch decision is in front of the model from turn one regardless of
  `inspect_data`. Detection confirmed already at load (`sync_from_adata`). The terminal
  gate also lists `batch_decision` as blocking.
- **B3** — telemetry: `note_spine_intervention()` records nudges / forced fallbacks
  into `recent_events` (→ manifest) for spine-adherence measurement.
- **C** — spine + agency contract added as a top-level prompt principle; the existing
  QC anti-busywork rule was verified already reasoned-exception (left as-is).

**Deliberately deferred** (judgment calls, not omissions):
- **Hard finalize/save block for `batch_decision`** + autonomous auto-open of
  `multi_sample_strategy`. Batch entry currently relies on early surfacing + prompt
  directive + terminal nudge (no forced fallback for *entry* obligations). Add the hard
  block only if the validation run shows surfacing is insufficient — avoids over-rigidity
  near the investigate-first flow.
- **End-to-end validation runs** (below).

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
