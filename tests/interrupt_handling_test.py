"""Interrupting a running turn (Esc / Ctrl+C) must preserve a valid, resumable
conversation.

Ctrl+C already raised KeyboardInterrupt mid-turn; Esc now does the same via a
background listener calling _thread.interrupt_main(). Both hit one handler that
saves the interrupted turn's messages so `continue` resumes cleanly. The OpenAI
loop commits the assistant tool_calls message BEFORE running tools, so an abort
mid-tool leaves dangling tool_calls the API rejects — those must be repaired with
a synthetic tool result. Anthropic/Codex commit only after tools finish, so their
lists stay valid and are left untouched.
"""

from __future__ import annotations

from scagent.agent.agent import SCAgent
from scagent.terminal import EscInterruptListener


def _bare_agent():
    """An SCAgent instance without running __init__ (no provider/client needed)."""
    agent = SCAgent.__new__(SCAgent)
    agent.run_manager = None
    agent._conversation_history = []
    return agent


def _openai_tool_call_msg(call_id="call_1", name="run_qc"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


# --- message helpers ----------------------------------------------------------
def test_message_tool_calls_handles_dict_and_object():
    assert SCAgent._message_tool_calls({"tool_calls": [{"id": "a"}]}) == [{"id": "a"}]
    assert SCAgent._message_tool_calls({"role": "user"}) == []

    class _Msg:
        tool_calls = [type("TC", (), {"id": "x"})()]

    assert SCAgent._tool_call_id(SCAgent._message_tool_calls(_Msg())[0]) == "x"
    assert SCAgent._tool_call_id({"id": "b"}) == "b"


# --- OpenAI dangling repair ---------------------------------------------------
def test_openai_dangling_tool_call_gets_synthetic_result():
    agent = _bare_agent()
    messages = [{"role": "user", "content": "go"}, _openai_tool_call_msg("call_1")]
    agent._preserve_interrupted_turn(messages, "openai")
    # a tool result was appended for the un-answered call
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "call_1"
    assert "Interrupted" in messages[-1]["content"]
    # and it was saved as the conversation history
    assert agent._conversation_history is messages


def test_openai_partial_results_only_fills_missing():
    agent = _bare_agent()
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_qc", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "run_pca", "arguments": "{}"}},
        ],
    }
    messages = [{"role": "user", "content": "go"}, msg, {"role": "tool", "tool_call_id": "c1", "content": "{}"}]
    agent._preserve_interrupted_turn(messages, "openai")
    tool_ids = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
    assert tool_ids == ["c1", "c2"]  # c2 filled, c1 untouched


def test_openai_complete_turn_not_modified():
    agent = _bare_agent()
    messages = [
        {"role": "user", "content": "go"},
        _openai_tool_call_msg("c1"),
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]
    before = len(messages)
    agent._preserve_interrupted_turn(messages, "openai")
    assert len(messages) == before  # nothing dangling → nothing added


def test_anthropic_and_codex_not_repaired_but_saved():
    # Anthropic/Codex commit the assistant message only after tools finish, so an
    # interrupted list has no dangling call to repair. Still saved as history.
    for fmt in ("anthropic", "codex"):
        agent = _bare_agent()
        messages = [{"role": "user", "content": "go"}]
        agent._preserve_interrupted_turn(messages, fmt)
        assert agent._conversation_history is messages
        assert len(messages) == 1


# --- Esc listener no-op safety ------------------------------------------------
def test_esc_listener_is_noop_without_tty():
    # Under pytest stdin is not a TTY, so the listener must be a harmless no-op
    # context manager (no thread, no terminal changes).
    listener = EscInterruptListener()
    assert listener._supported() is False
    with listener as active:
        assert active._thread is None
    # exit is clean and idempotent
    listener.__exit__(None, None, None)


def test_esc_listener_disabled_by_env(monkeypatch):
    monkeypatch.setenv("SCAGENT_NO_ESC_INTERRUPT", "1")
    assert EscInterruptListener()._supported() is False
