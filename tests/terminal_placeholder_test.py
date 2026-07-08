"""The custom-input placeholder must render dimmed, not like pre-typed text.

`read_user_input` funnels every interactive text prompt (the "Type something
else..." custom option, the "Describe the experiment" free-text action, and the
plain `> ` prompt), so verifying it here covers all of them. prompt_toolkit
shows a bare `str` placeholder in the normal foreground; we wrap it in a dimmed
(grey italic) FormattedText so it reads as a hint.
"""

from __future__ import annotations

import pytest

from scagent import terminal

pytest.importorskip("prompt_toolkit")
from prompt_toolkit.formatted_text import FormattedText  # noqa: E402


class _FakeSession:
    """Captures the placeholder handed to prompt_toolkit's prompt()."""

    def __init__(self):
        self.captured_placeholder = None

    def prompt(self, prompt, placeholder=""):
        self.captured_placeholder = placeholder
        return ""


def test_placeholder_is_dimmed_formatted_text(monkeypatch):
    session = _FakeSession()
    monkeypatch.setattr(terminal, "_get_prompt_session", lambda: session)

    terminal.read_user_input("> ", placeholder="Describe the experiment: samples, donors...")

    ph = session.captured_placeholder
    assert isinstance(ph, FormattedText), "placeholder should be styled, not a bare str"
    (style, text), = ph  # single fragment
    assert "italic" in style and "ansibrightblack" in style  # dimmed hint styling
    assert text == "Describe the experiment: samples, donors..."


def test_empty_placeholder_passed_through_plain(monkeypatch):
    # No placeholder -> nothing to dim; must stay a plain str (no empty fragment).
    session = _FakeSession()
    monkeypatch.setattr(terminal, "_get_prompt_session", lambda: session)

    terminal.read_user_input("> ", placeholder="")

    assert session.captured_placeholder == ""
