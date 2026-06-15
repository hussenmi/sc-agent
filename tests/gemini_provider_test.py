from scagent.agent.agent import SCAgent


def _gemini_agent():
    agent = object.__new__(SCAgent)
    agent.provider = "gemini"
    agent.model = "gemini-3.5-flash"
    return agent


def test_gemini_thinking_uses_openai_compatible_reasoning_effort(monkeypatch):
    monkeypatch.setenv("SCAGENT_THINKING", "1")
    monkeypatch.delenv("SCAGENT_THINKING_EFFORT", raising=False)
    monkeypatch.delenv("SCAGENT_THINKING_BUDGET", raising=False)

    assert _gemini_agent()._thinking_extra() == {"reasoning_effort": "medium"}


def test_gemini_thinking_honors_explicit_effort(monkeypatch):
    monkeypatch.setenv("SCAGENT_THINKING", "1")
    monkeypatch.setenv("SCAGENT_THINKING_EFFORT", "high")

    assert _gemini_agent()._thinking_extra() == {"reasoning_effort": "high"}


def test_gemini_thinking_maps_legacy_budget(monkeypatch):
    monkeypatch.setenv("SCAGENT_THINKING", "1")
    monkeypatch.delenv("SCAGENT_THINKING_EFFORT", raising=False)
    monkeypatch.setenv("SCAGENT_THINKING_BUDGET", "24000")

    assert _gemini_agent()._thinking_extra() == {"reasoning_effort": "high"}
