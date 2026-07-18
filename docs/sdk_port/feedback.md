# SDK port — feedback & discussion

Space for the Iris session (and others) to react to `README.md`. Add dated entries;
reference section numbers (e.g. "§3 option 2", "§10 Q3").

Priority questions to weigh in on are in `README.md` §10:
1. Control-flow shape — model-driven-with-hooks (1) / phase sub-agents (2) / state-machine-lite (3)?
2. Model coupling — is the Claude-shaped SDK + LiteLLM bridge acceptable vs. today's multi-provider design?
3. Reproducibility of model-driven flow for the LuCA benchmark.
4. Tools vs. skills per capability.
5. NAT: replaced or complemented by the SDK runtime?
6. Scope of the first end-to-end milestone.

---

## Entries

<!-- Iris session: add your feedback below, e.g.

### 2026-07-1X — <who/where>
- §3: ...
- §10 Q2: ...

-->

### 2026-07-18 — Iris session (Claude, after reading the full RFC + both Spark prototypes)

Read `README.md` end to end, plus the two Spark prototypes it points at:
`bionemo-lab/contrib-skills/` and `sdk-floor-proto/` (both now backed up to their own
private repos under `hussenmi/`, previously Spark-disk-only).

**Framing that helped me: the two prototypes are the two halves of §4, validated
separately but never yet together.**
- `contrib-skills/` = the **Capabilities** row (library/open-model/workflow skills;
  verified on PBMC3k).
- `sdk-floor-proto/` = the **Runtime + Floors** rows (floors as real PreToolUse
  deny-hooks, per-cluster re-fire, non-Claude model via LiteLLM).
- The **missing middle** = real skills executing *as the tools* under real floors *as
  the hooks*. Nothing runs that fusion yet (proto uses stub tools; skills run outside
  any floor). That fusion is exactly Q6's first milestone — see below.

**§10 answers**

- **Q1 (control-flow):** Start with **(1) model-driven + hook floors** — the proto
  already shows the guarantees hold. But treat it as *coupled to Q3*, not independent:
  option (1) is itself the reproducibility knob. Keep a `world_state` phase notion
  (3-lite) in reserve as the lever, don't build it up front.
- **Q2 (model coupling) — the biggest real risk.** Today's harness is genuinely
  5-provider (anthropic/openai/codex/gemini/vertex). The bridge is only proven on
  OpenAI-compat vLLM + hosted Nemotron. **codex/gemini/vertex tool-calling fidelity
  through LiteLLM's Anthropic passthrough is unproven.** Accept the coupling, but make
  "confirm tool-calling per provider we still care about" an explicit gate, and decide
  now which of the 5 we're willing to drop. Upside: a *uniform* bridge actually helps
  the harness-vs-model ablation (same interface across models).
- **Q3 (reproducibility / LuCA):** This is the crux and it ties straight to the
  ablation "money plot" (enforcement's effect on annotation consistency, R3−R2).
  Model-driven flow trades determinism for flexibility — but floors-as-hooks is the
  *cleanest instantiation* of "harness enforces process, model drives," which is the
  thesis. Recommendation: go (1), **measure consistency on LuCA**, escalate to 3-lite
  only if it regresses. Don't pre-optimize determinism.
- **Q4 (tools vs skills):** Floor-gated core (QC/cluster/annotate/DEG) = **internal
  tools** (hooks must name the tool → tight coupling). Portable/shareable capability
  (rapids-singlecell, scimilarity) = **skills**, which double as the BioNeMo
  contributions. The `contrib-skills` already draw this line correctly.
- **Q5 (NAT):** **Complement, not replace.** NAT is eval/tracing (per-step
  tokens/latency + accuracy/consistency); the SDK is orchestration — orthogonal.
  Better: re-point the existing `SCAGENT_STEP_LOG → NAT` bridge at SDK **hook events**
  (Pre/PostToolUse), which are cleaner instrumentation points than the manual bridge.
- **Q6 (first milestone):** load→QC→cluster→annotate on pbmc3k — but pick the
  **hardest** floors, not the easy ones: the **re-firing per-round cluster-QC entry
  obligation** and the **4-file PanglaoDB validation**. The proto already did easy
  stub gates; the milestone only de-risks anything if it proves a *cross-cutting,
  re-firing* floor collapses into one hook, with **a real `contrib-skill` as the tool
  underneath** (this is what closes the "missing middle").

**Q7 I'd add — the loop does more than orchestration.** §6 lists result-slimming as a
floor but omits **GPU→CPU handoff boundary wrapping**, the **enforced decision-yield
resolver**, and **interrupt/resume** — several aren't obviously "hooks." Sharp unknown:
the SDK now *owns* context/loop, so **can we still intercept a tool *result* before it
hits context** (does `PostToolUse` allow rewriting a result) to preserve slimming? If
not, the annotation-result context-overflow bug we already fixed comes back. This is a
"could-sink-the-port" item and belongs in §10.

**On the thesis (explicitly): the SDK+skills framing strengthens it, doesn't dilute
it.** The floors-as-hooks artifact *is* the thesis extracted and made legible (today
it's buried in 7.7k lines across 4 files). The proto's "floors hold regardless of
model; recovery scales with capability" finding *is* the harness-vs-model ablation in
miniature — and running it on a neutral substrate (SDK) across models (LiteLLM) makes
it **more** credible, not less. One risk to manage: keep the answer to "what's left
that's yours?" crisp — **the floors (process obligations + deny semantics) and the
reconciliation logic are ours; the runtime and skill packaging are commodity substrate
we stand on.** Keep the floors the star and show they're substrate-independent.

**Concrete gaps I noticed in the prototypes (all fine for "beginnings"):** no skill
manifests yet (`skills.sh.json`/`marketplace.json`, needed for the `npx skills`
installer — RFC §5 defers these); the `evals.json` are good behavioral specs but have
no *runner* wired; no formal `annotate/` reconciliation sub-skill yet (workflow notes
it as next); and **no batch-effect skill** — scagent already has that logic
(`diagnose_batch_effect`), and the toolkit ships nothing for it, so it's a natural
next contribution.
