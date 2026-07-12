"""Regression tests: scagent runs headless (no TTY) in smart-autonomous mode.

Two fixes are covered:

1. `scagent.cli.run_analyze` previously had an UNCONDITIONAL TTY guard that
   returned 1 on any non-interactive stdin, blocking batch/eval/CI use even in
   smart-autonomous mode. It must now only block *collaborative* mode without a
   TTY (mirroring the guard in SCAgent.run); smart mode must proceed.

2. `scagent.terminal.prompt_for_decision` previously blocked on `read_text` at
   decision points, crashing on non-TTY stdin. It must now auto-select the
   checkpoint's default (recommended) option when there is no TTY, preferring a
   concrete option over the synthetic "custom" free-text choice.

These are what let the NeMo Agent Toolkit eval drive scagent autonomously.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import scagent.agent as agent_pkg
import scagent.cli as cli
import scagent.terminal as terminal
from scagent.terminal import DecisionChoice, prompt_for_decision


# ── helpers ──────────────────────────────────────────────────────────────────

def _no_tty(monkeypatch, module):
    """Make `module.sys.stdin.isatty()` return False for the test."""
    monkeypatch.setattr(module.sys, "stdin", SimpleNamespace(isatty=lambda: False))


def _analyze_args(**over):
    base = dict(request="run qc and cluster", data="x.h5ad", output=".", name=None,
                provider="openai", model="m", max_iterations=5, quiet=True,
                checkpoints=False, interactive=False, smart_autonomous=True)
    base.update(over)
    return SimpleNamespace(**base)


class _FakeAgent:
    constructed = False

    def __init__(self, **kwargs):
        type(self).constructed = True
        self.provider = kwargs.get("provider") or "openai"
        self.model = kwargs.get("model") or "m"

    def close(self):
        # run_analyze calls agent.close() to tear down the sandbox/MCP connections.
        pass


# ── cli.run_analyze guard ────────────────────────────────────────────────────

def test_smart_mode_runs_headless(monkeypatch):
    """smart_autonomous + no TTY -> passes the guard, builds the agent, returns 0."""
    _no_tty(monkeypatch, cli)
    _FakeAgent.constructed = False
    monkeypatch.setattr(agent_pkg, "SCAgent", _FakeAgent)
    monkeypatch.setattr(cli, "_analyze_with_decisions", lambda agent, **kw: {"ok": True})

    rc = cli.run_analyze(_analyze_args(smart_autonomous=True))

    assert _FakeAgent.constructed is True  # got past the TTY guard
    assert rc == 0


def test_collaborative_mode_still_requires_tty(monkeypatch):
    """collaborative (not smart) + no TTY -> blocked at the guard, returns 1, no agent built."""
    _no_tty(monkeypatch, cli)
    _FakeAgent.constructed = False
    monkeypatch.setattr(agent_pkg, "SCAgent", _FakeAgent)
    monkeypatch.setattr(cli, "_analyze_with_decisions", lambda agent, **kw: {"ok": True})

    rc = cli.run_analyze(_analyze_args(smart_autonomous=False))

    assert rc == 1
    assert _FakeAgent.constructed is False  # never got past the guard


# ── terminal.prompt_for_decision headless auto-default ───────────────────────

def test_decision_auto_selects_default_when_headless(monkeypatch):
    _no_tty(monkeypatch, terminal)
    choices = [DecisionChoice("Keep all cells", "keep"),
               DecisionChoice("Remove flagged cells", "remove")]

    sel = prompt_for_decision("How to proceed?", choices, default_index=1, allow_custom=False)

    assert sel.action == "remove"
    assert sel.index == 1
    assert sel.input_mode == "auto-default"


def test_decision_default_skips_custom_choice(monkeypatch):
    """If the default points at the synthetic 'custom' choice, pick a concrete one."""
    _no_tty(monkeypatch, terminal)
    choices = [DecisionChoice("Type something else...", "custom"),
               DecisionChoice("Use recommended thresholds", "recommended")]

    sel = prompt_for_decision("Q?", choices, default_index=0, allow_custom=False)

    assert sel.action == "recommended"
    assert sel.action != "custom"
