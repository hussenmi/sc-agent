"""Reasoning models split each turn into a user-facing `content` channel and a
raw chain-of-thought channel (`reasoning_content` for Gemini/DeepSeek,
`reasoning` for vLLM parsers like glm45/nemotron_v3).

`_split_reasoning_channels` always returns the raw values; the run loop decides
whether to display the CoT inline (SCAGENT_SHOW_THINKING) and whether to persist
it to <run_dir>/reasoning.log (SCAGENT_SAVE_THINKING). These tests pin both."""

from types import SimpleNamespace

from scagent.agent.agent import SCAgent


def _agent():
    agent = object.__new__(SCAgent)
    agent.provider = "openai"
    agent.model = "glm-5.2"
    agent.run_manager = None
    return agent


def _msg(content=None, reasoning_content=None, reasoning=None):
    extra = {}
    if reasoning_content is not None:
        extra["reasoning_content"] = reasoning_content
    if reasoning is not None:
        extra["reasoning"] = reasoning
    return SimpleNamespace(content=content, model_extra=extra)


def test_split_returns_both_channels():
    msg = _msg(content="Let me load the data first.", reasoning_content="The user wants...")
    narration, cot = _agent()._split_reasoning_channels(msg)
    assert narration == "Let me load the data first."
    assert cot == "The user wants..."


def test_split_uses_reasoning_field_for_vllm_parsers():
    # vLLM glm45/nemotron_v3 parsers expose CoT under `reasoning`, not `reasoning_content`.
    msg = _msg(content="ok", reasoning="step-by-step trace")
    _, cot = _agent()._split_reasoning_channels(msg)
    assert cot == "step-by-step trace"


def test_split_drops_empty_narration():
    # Model that goes straight from thinking to a tool call (empty content).
    msg = _msg(content="   ", reasoning_content="thinking...")
    narration, cot = _agent()._split_reasoning_channels(msg)
    assert narration is None
    assert cot == "thinking..."


def test_save_thinking_writes_to_run_dir(tmp_path, monkeypatch):
    monkeypatch.delenv("SCAGENT_SAVE_THINKING", raising=False)  # default = on
    a = _agent()
    a.run_manager = SimpleNamespace(run_dir=tmp_path)
    path = a._save_thinking("the model's reasoning trace", iteration=2)
    assert path == tmp_path / "reasoning.log"
    text = path.read_text()
    assert "the model's reasoning trace" in text
    assert "iteration 2" in text and "glm-5.2" in text


def test_save_thinking_appends_across_turns(tmp_path, monkeypatch):
    monkeypatch.delenv("SCAGENT_SAVE_THINKING", raising=False)
    a = _agent()
    a.run_manager = SimpleNamespace(run_dir=tmp_path)
    a._save_thinking("first", iteration=1)
    a._save_thinking("second", iteration=2)
    text = (tmp_path / "reasoning.log").read_text()
    assert "first" in text and "second" in text


def test_save_thinking_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCAGENT_SAVE_THINKING", "0")
    a = _agent()
    a.run_manager = SimpleNamespace(run_dir=tmp_path)
    assert a._save_thinking("trace", iteration=1) is None
    assert not (tmp_path / "reasoning.log").exists()


def test_save_thinking_noop_without_run_dir(monkeypatch):
    monkeypatch.delenv("SCAGENT_SAVE_THINKING", raising=False)
    a = _agent()  # run_manager is None
    assert a._save_thinking("trace", iteration=1) is None
