# scagent → Claude Agent SDK port — design & rationale

**Status:** design / RFC. Nothing ported yet. A working *prototype* of the core
mechanism (floors-as-hooks + local/non-Claude model) has been validated end-to-end
(see §7). This doc exists so the **Iris session** (and anyone else) can review the
plan and push back **before** we start the port.

**Audience & ask:** read §2 (the problem) and §3 (why not a DAG), then critique §4
(target architecture) and §5 (approach). Open questions where feedback is most
valuable are in §10. Add comments inline or in a `feedback.md` next to this file.

---

## 1. What we're trying to do

Re-platform the scagent harness on top of the **Claude Agent SDK** (`claude-agent-sdk`,
the library form of the Claude Code engine) so that:

- The **agent loop, tool-calling, context management, sub-agents, and ask-user** come
  from a maintained runtime instead of ~7.7k lines of hand-rolled loop.
- Our **"floors" (process obligations)** become **SDK PreToolUse hooks** — enforced at
  the runtime layer, in one place, instead of scattered across the loop.
- The agent can **drive itself** (model-chosen next action within guardrails) rather
  than following a largely pre-sequenced pipeline (the "DAG" we want to move away from).
- We keep everything that makes scagent *scagent*: local model serving, the scientific
  floors, OpenShell sandboxing, MCP/BioContext, the single-cell tools.

This is a **re-platforming of control flow**, not a rewrite of the science. The
tools (QC, clustering, batch integration, annotation, DEG) and the domain logic stay;
what changes is *who orchestrates them and how the process is enforced*.

## 2. The problem we're solving

The current harness (`scagent/agent/agent.py`, 7,671 lines) is a custom `SCAgent`
class that:

- Implements its own multi-provider loop (anthropic / openai / codex / gemini /
  vertex — see `_init_*`), MCP wiring, prompt assembly, and output post-processing.
- Enforces the "scientific spine" through **imperative, cross-cutting mechanisms**:
  bounded **re-prompts when the run tries to end with an unmet obligation**
  (`agent.py` ~line 173), a **decision policy** (`decision_policy.py`:
  `auto_execute` / `recommend_and_confirm` / `must_ask`), **entry obligations** that
  re-fire per reclustering round (cluster-QC), and validation gates.
- Spreads a single conceptual rule across several files. Example (from memory): the
  PanglaoDB/annotation enforcement lives in **four** places — the `tools.py`
  validator, `prompts.py`, the `agent.py` loop, and `world_state.py`. Changing one
  rule means auditing all four.

Consequences:

- **Rigid / DAG-like flow.** Because sequencing and obligations are baked into the
  loop and prompts, the agent effectively walks a mostly fixed pipeline. It handles
  the happy path well but is awkward to extend, reorder, or let the model deviate
  from when the data calls for it.
- **Enforcement is entangled with orchestration.** Floors are implemented as loop
  behavior + re-prompts + prompt text, so they're hard to reason about, test in
  isolation, or reuse.
- **High maintenance surface.** 7.7k lines of loop we own and must keep working
  across five providers.

We are **not** claiming the current design is bad — it works and encodes hard-won
domain rules. The issue is that the *control-flow layer* is doing too much and is too
rigid, and the *enforcement layer* is diffuse.

## 3. Why remove the "DAG" structure (and what we mean by it)

"DAG" here = the analysis path is largely **predetermined by the harness** (a fixed
graph of steps with gates between them), rather than **chosen by the model** turn by
turn within guardrails. The current re-prompt/obligation machinery pushes the run
down a specific spine.

Why move away from it:

- **Real single-cell analysis branches.** Whether to recluster, which resolution,
  whether a cluster is a doublet, whether a batch effect is real — these are
  data-dependent judgments. A fixed graph either can't express the branch or needs
  ever-more special-case code to.
- **The model is now capable of driving.** With a competent model + tools + *floors
  that guarantee correctness*, the model can pick the next action and the harness
  only has to guarantee it doesn't skip a required check. That's more flexible and
  far less code than encoding the graph ourselves.
- **Separation of concerns.** Orchestration (what to do next) → the model. Enforcement
  (what must be true before an action) → hooks. Execution safety (where code runs) →
  OpenShell. Each layer becomes independently testable.

**The distinction that makes this safe:** removing the DAG does **not** mean removing
the guarantees. The floors still make it *impossible* to, e.g., `finalize_annotation`
before cluster QC + marker validation ran — we just enforce that as a hook that
**denies the tool call**, instead of as pre-sequenced pipeline steps. Proven in §7.

### Ways we're thinking about approaching the "not a DAG" goal

Not yet decided — options to discuss:

1. **Model-driven with hook floors (leading candidate).** One agent, full toolset,
   the model chooses the next tool each turn; PreToolUse hooks deny any call whose
   preconditions aren't met and return a reason the model acts on. Minimal control
   code; maximal flexibility. (This is exactly what the prototype validated.)
2. **Phase sub-agents.** SDK sub-agents for coarse phases (load/QC → cluster →
   annotate) with hooks inside each. More structure than (1) without a rigid global
   graph; useful if context or specialization per phase matters.
3. **State-machine-lite.** Keep a small explicit notion of "phase" in `world_state`
   but let the model move within/9across phases freely, with hooks as the hard gates.
   A middle ground if pure (1) proves too loose for reproducibility.

Recommendation: start with (1), because the prototype shows the guarantees hold, and
add (2)/(3) only where a concrete need appears (context size, reproducibility).

## 4. Target architecture

```
                    ┌─────────────────────────────────────────────┐
                    │            Claude Agent SDK runtime          │
                    │   loop · tool-calling · context · ask-user   │
                    │                 · sub-agents                 │
                    ├───────────────┬───────────────┬─────────────┤
   FLOORS  ───────► │ PreToolUse    │  TOOLS/SKILLS │  MODEL       │
   (process          │ hooks         │               │             │
    obligations)     │ (can_use_tool,│  scagent ops  │ local vLLM  │
                    │  Pre/Post     │  + contrib-   │ (qwen3.6 /  │
                    │  ToolUse,     │  skills       │  GLM /      │
                    │  Stop)        │  + BioNeMo    │  Nemotron)  │
                    └──────┬────────┴───────┬───────┴──────┬──────┘
                           │                │              │
                           ▼                ▼              ▼
                    floor LOGIC       execute in      LiteLLM bridge
                    (our Python)      OpenShell        (Anthropic↔OpenAI)
                                      sandbox
```

**Four layers, cleanly separated:**

| Layer | What it is | Where it comes from |
|---|---|---|
| **Runtime** | agent loop, tool dispatch, context mgmt, ask-user, sub-agents | Claude Agent SDK |
| **Floors** | process obligations enforced before/around tool calls | SDK **hooks** (PreToolUse deny; `can_use_tool`; Stop) — *floor logic stays our Python in the callback* |
| **Capabilities** | the actual tools/skills the model can call | scagent's ops + our contrib-skills + BioNeMo skills |
| **Execution sandbox** | where `run_code` and skill scripts actually run, isolated | **OpenShell** (already integrated for `run_code`; see `docs/openshell_integration.md`) |

**Model:** the SDK is Claude-shaped (Anthropic Messages format). To keep our
**local vLLM models first-class**, we bridge via **LiteLLM** (`ANTHROPIC_BASE_URL` →
LiteLLM proxy → OpenAI-compatible vLLM). Validated with the real served
`Qwen3.6-27B-NVFP4` and with hosted Nemotron 3 Ultra/Super (§7). This is the one real
coupling cost of the SDK — documented, tested, acceptable.

## 5. How we plan to approach it (incremental)

1. **Audit the current harness pattern-by-pattern** (needs the code — a task for the
   Iris/impl session). For each mechanism in `agent.py` / `decision_policy.py` /
   `world_state.py` / `tools.py`, decide: **keep / drop / adjust**, and where it lands
   in the new architecture (hook? tool? prompt? sub-agent?). Output: a mapping table.
2. **Port the tools** as SDK tools / MCP tools: load (with symbol conversion), QC,
   clustering/reclustering, batch integration + `diagnose_batch_effect`, DEG,
   annotation, `run_code`. Many already exist as functions in `scagent/*` — this is
   wrapping, not rewriting.
3. **Port the floors as hooks**, one at a time, verifying each composes (see §6).
4. **Wire OpenShell** as the execution path for `run_code` and skill scripts under the
   SDK (not the SDK's built-in Bash).
5. **Bridge the local model** via LiteLLM; confirm tool-calling fidelity on our served
   models.
6. **Bring up context/state**: `world_state` snapshots, the result-slimming that
   prevents context overflow, interrupt/resume.
7. **Differential-test** against the current harness on known runs (e.g. the LuCA /
   pbmc3k cases) — same inputs, compare decisions/labels.

Keep the current harness running the whole time; the SDK version is a parallel track
until it's at parity.

## 6. The floors (process obligations)

Floors are the non-negotiable process guarantees. Known ones (from the code + memory):

- **Cluster-QC entry obligation** — `run_cluster_qc` must run before annotation, and
  **re-fires per reclustering/iterative round** (`cluster-qc-enforcement`).
- **PanglaoDB / marker validation** before finalizing a cell-type label
  (`annotation-panglaodb-enforcement-layers` — currently spread across 4 files).
- **Structure-QC floor**, gated on cluster QC.
- **Within-sample paired-DEG batch check** in `diagnose_batch_effect`
  (`batch-deg-and-investigation-plots`).
- **Gene-symbol conversion at load** (root fix for the ribo/MT cascade).
- **Annotation result slimming** (finalize writes slim; full evidence to
  `adata.uns`/disk) to avoid context overflow.
- **Decision policy**: `auto_execute` vs `recommend_and_confirm` vs `must_ask`
  (`decision_policy.py`) → maps naturally onto hook outcomes
  (`allow` / `ask` / `deny`) + `pause_and_ask`.

**Design principles that must survive the port** (these are load-bearing):

- **No hardcoded domain knowledge in the harness.** Floors enforce *process* and
  *reasoning*, never bake in biology/marker tables. Domain knowledge lives in the
  model or a swappable pack. (`no-hardcoded-domain-knowledge-in-harness`)
- **No regex user-intent parsing.** Never pattern-match the user's words to infer
  intent; interpreting intent is the model's job, the harness only enforces process
  floors. (`no-regex-user-intent-in-harness`)

**Tested so far:** two floors (a QC gate and a marker-validation gate) implemented as
real SDK PreToolUse hooks, with per-cluster state — see §7.

## 7. The hook tests we ran, and how they worked

Prototype lives at **`~/projects/sdk-floor-proto/`** (separate from this repo; has its
own memory). It is a **thin proof of the mechanism with STUB tools** — not scagent
logic — built to answer "is this architecture viable?" cheaply.

Setup: Claude Agent SDK (Python) → Claude Code CLI runtime → **LiteLLM proxy** →
model. Two stub tools (`run_cluster_qc`, `validate_markers`, `finalize_annotation`)
and floors as PreToolUse hooks that **deny** finalize until prerequisites ran, using
the SDK's real deny schema:
`{"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": "..."}}`.

**Results:**

- **#1 — floors compose + re-fire (PASS).** Two independent hooks both gate
  `finalize_annotation`, with **per-cluster** state. Annotating two clusters showed:
  each cluster's finalize denied until *both* QC and markers ran for *that* cluster;
  cluster 3 being satisfied did not let cluster 5 through. Nothing was ever finalized
  without its prerequisites (`all_gated=True`), across every run.
- **Model behavior scales with capability (finding).** A weak 8B looped/stalled on
  the denials (floors still held — nothing slipped through). The local
  **Qwen3.6-27B-NVFP4** recovered halfway then needed the process stated in the
  prompt. Hosted **Qwen3.5-122B** and **Nemotron-3-Ultra-550B** recovered fully and
  cleanly. Lesson: **floors enforce regardless of model; completing multi-obligation
  recovery scales with model strength** → in production, state the process in the
  prompt *and* enforce with floors (belt-and-suspenders), which is how scagent already
  operates.
- **#3 — real local model through the bridge (PASS).** Pointed the LiteLLM bridge at
  the actual served `Qwen3.6-27B-NVFP4` on `localhost:8001` and at hosted
  `nvidia/nemotron-3-ultra-550b-a55b`. Both drove the agent end-to-end; Ultra ran
  QC + markers for both clusters (even generating correct markers — CD8A/CD8B/CD3D for
  CD8 T, CD19/CD79A/MS4A1 for B) then finalized both. Clean PASS.

**What this proves:** the SDK runtime + floors-as-real-hooks + local/non-Claude model
via LiteLLM is a **sound substrate**. What it does **not** prove: that porting *all*
scagent floors so they compose correctly, plus state/context/OpenShell, is done —
that's the actual project.

**Gotchas learned (documented for the impl session):**

- LiteLLM's Anthropic `/v1/messages` pass-through defaults to the OpenAI *Responses*
  API → 404 on vLLM/NIM. Fix: `use_chat_completions_url_for_anthropic_messages: true`.
- Hosted NIM large MoE models can cold-start for minutes (qwen3-next-80b ≈ 10 min);
  Nemotron Ultra/Super and Qwen3.5-122B were ~1s warm. Local vLLM avoids this — a
  reason our low-latency serving matters.

## 8. Skills we plan to integrate

Skills = portable capability packages (a `SKILL.md` + optional `scripts/`/`references/`),
the same format Claude Code and the BioNeMo Agent Toolkit use. The SDK agent loads them
as tools. Two sources:

**Ours** (authored in `~/projects/bionemo-lab/contrib-skills/`, verified on real
pbmc3k; these are also our proposed contributions to NVIDIA's toolkit — see §9):

- `library-skills/rapids-singlecell/` — GPU-accelerated scanpy (RAPIDS drop-in).
- `open-models-skills/scimilarity/` — reference-based annotation (CPU/GPU).
- `workflows/single-cell/` — umbrella workflow: `preprocess/` sub-skill (CPU-first,
  GPU-optional) → scimilarity → per-cluster reconciliation evidence table.

Note scagent already has native equivalents (`scagent/annotation/scimilarity.py`,
`celltypist.py`; `scagent/batch/*`) — the port should decide per capability whether to
expose the **existing function as a tool** or the **skill**; likely tools internally,
skills for portability/sharing.

**BioNeMo Agent Toolkit** (hosted NIM + local model/library skills; see §9): protein/
genomics/chemistry skills the agent could call when a task needs them, plus the
single-cell library-skill NVIDIA announced but hasn't shipped (which our
rapids-singlecell contribution fills).

## 9. The BioNeMo connection

NVIDIA's **BioNeMo Agent Toolkit** (announced BIO 2026) is a catalog of agent-callable
**skills** (SKILL.md + scripts) over BioNeMo models, exposed via **MCP + NIM**, and
explicitly harness-agnostic. It sits on the same primitives we're adopting:

- **Same skill format** we're using for our contrib-skills → our single-cell skills
  drop straight into its categories (`library-skills` / `open-models-skills` /
  `workflows`).
- **NAT** (NeMo Agent Toolkit) is the orchestration/memory layer NVIDIA pairs with it;
  we already have a scagent↔NAT integration (`scagent-nat-integration`).
- **OpenShell / NemoClaw** are the announcement's secure-execution/blueprint layer —
  which is exactly the role OpenShell plays under our SDK runtime.
- **White space:** no single-cell / cell-annotation skill ships in v0.1.0 → our
  contributions fill the gap, and cell-type annotation reconciliation (scagent's core)
  is uncovered.

So the SDK-ported scagent is naturally positioned to **consume** BioNeMo skills and
**contribute** single-cell ones — the port and the NVIDIA collaboration reinforce each
other. (Entitlement note: hosted biology NIMs need a build.nvidia.com biology-page key;
ESMFold verified working. Full detail in the `bionemo-agent-toolkit` memory.)

## 10. Open questions (feedback wanted)

1. **Control-flow shape** (§3): pure model-driven-with-hooks (1), phase sub-agents (2),
   or state-machine-lite (3)? Start with (1)?
2. **Model coupling:** accept the LiteLLM bridge for local models, or is the
   Claude-shaped SDK too much of a constraint given scagent's multi-provider design
   today (anthropic/openai/codex/gemini/vertex)? Do we lose providers we care about?
3. **Reproducibility:** the current spine makes runs fairly deterministic. Does
   model-driven flow hurt reproducibility for the LuCA benchmark, and if so does
   state-machine-lite (3) mitigate it enough?
4. **Tools vs skills** per capability: which scagent ops become internal tools vs
   portable skills?
5. **NAT relationship:** does the SDK runtime replace or complement the NAT
   integration for tracing/eval?
6. **Scope of first milestone:** smallest end-to-end slice worth building (e.g.
   load → QC → cluster → annotate on pbmc3k with 2–3 real floors as hooks)?

## 11. Pointers

- Prototype: `~/projects/sdk-floor-proto/` (`floor_agent.py`, `config.yaml`, `run.sh`)
  — run notes and gotchas in its memory (`sdk-floor-hook-prototype`).
- Contrib-skills: `~/projects/bionemo-lab/contrib-skills/` (+ vendored toolkit clone).
- Current harness: `scagent/agent/{agent,tools,decision_policy,world_state,sandbox}.py`.
- Related docs: `docs/openshell_integration.md`, `docs/coordination_harness_design.md`,
  `docs/nvidia_collab.md`, `docs/serving_findings.md`.
- Memory (authoritative, sc-agent project): `sdk-floor-hook-prototype`,
  `bionemo-agent-toolkit`, `openshell-scagent-integration`, `scagent-nat-integration`,
  the enforcement/floor memories, `no-hardcoded-domain-knowledge-in-harness`,
  `no-regex-user-intent-in-harness`.
