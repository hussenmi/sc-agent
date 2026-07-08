"""Tests for context-window resolution across serving backends.

Covers the pure helpers (_model_get, _coerce_positive_int, _server_context_limit)
and the full _resolve_context_limit priority chain, with particular attention to
the vLLM (max_model_len) vs llama.cpp (meta.n_ctx) split — the two self-hosted
backends advertise the per-request limit under different field names.
"""

from types import SimpleNamespace

import pytest

from scagent.agent.agent import (
    SCAgent,
    _coerce_positive_int,
    _model_get,
    _server_context_limit,
)

# ── _model_get: dict / attribute / model_extra / missing ─────────────────────

def test_model_get_from_dict():
    assert _model_get({"max_model_len": 262144}, "max_model_len") == 262144


def test_model_get_from_attribute():
    obj = SimpleNamespace(max_model_len=262144)
    assert _model_get(obj, "max_model_len") == 262144


def test_model_get_from_model_extra_fallback():
    # Mimics an OpenAI SDK pydantic object whose extra fields live in model_extra
    # and are NOT exposed as plain attributes.
    class _Pydanticish:
        model_extra = {"meta": {"n_ctx": 32768}}

    assert _model_get(_Pydanticish(), "meta") == {"n_ctx": 32768}


def test_model_get_missing_returns_none():
    assert _model_get({"id": "x"}, "max_model_len") is None
    assert _model_get(SimpleNamespace(id="x"), "max_model_len") is None


def test_model_get_none_object():
    assert _model_get(None, "anything") is None


# ── _coerce_positive_int: tolerant parse, rejects non-positive / garbage ─────

@pytest.mark.parametrize(
    "value,expected",
    [
        (262144, 262144),
        ("262144", 262144),   # some servers stringify
        (32768.0, 32768),     # float
        (0, None),            # zero is not a usable window
        (-1, None),           # negative
        (None, None),
        ("", None),
        ("abc", None),        # garbage must never widen the window
        ([], None),
    ],
)
def test_coerce_positive_int(value, expected):
    assert _coerce_positive_int(value) == expected


# ── _server_context_limit: backend detection + precedence ────────────────────

def test_server_limit_vllm_max_model_len():
    model = {"id": "Qwen3.6-27B", "max_model_len": 262144}
    assert _server_context_limit(model) == (262144, "max_model_len")


def test_server_limit_llamacpp_meta_n_ctx_dict():
    model = {"id": "GLM-5.2", "meta": {"n_ctx": 32768, "n_ctx_train": 1048576}}
    assert _server_context_limit(model) == (32768, "meta.n_ctx")


def test_server_limit_llamacpp_meta_as_object():
    # meta may arrive as a nested object rather than a dict.
    model = SimpleNamespace(id="GLM-5.2", meta=SimpleNamespace(n_ctx=262144))
    assert _server_context_limit(model) == (262144, "meta.n_ctx")


def test_server_limit_prefers_max_model_len_over_meta():
    model = {"max_model_len": 262144, "meta": {"n_ctx": 32768}}
    assert _server_context_limit(model) == (262144, "max_model_len")


def test_server_limit_none_when_absent():
    assert _server_context_limit({"id": "x"}) == (None, None)


def test_server_limit_skips_zero_max_model_len_uses_meta():
    # A bogus max_model_len=0 must not win; fall through to a valid meta.n_ctx.
    model = {"max_model_len": 0, "meta": {"n_ctx": 32768}}
    assert _server_context_limit(model) == (32768, "meta.n_ctx")


def test_server_limit_stringified_values():
    assert _server_context_limit({"max_model_len": "131072"}) == (131072, "max_model_len")
    assert _server_context_limit({"meta": {"n_ctx": "65536"}}) == (65536, "meta.n_ctx")


# ── _resolve_context_limit: full priority chain via a fake client ────────────

class _FakeModels:
    def __init__(self, data):
        self._data = data

    def list(self):
        return SimpleNamespace(data=self._data)


class _FakeClient:
    def __init__(self, base_url, data):
        self.base_url = base_url
        self.models = _FakeModels(data)


def _agent(model, client):
    """Minimal SCAgent stand-in carrying just what _resolve_context_limit reads."""
    return SimpleNamespace(model=model, client=client, _print=lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("SCAGENT_CONTEXT_LIMIT", raising=False)


def test_resolve_llamacpp_uses_meta_n_ctx():
    client = _FakeClient(
        "http://localhost:8001/v1",
        [SimpleNamespace(id="GLM-5.2", meta={"n_ctx": 262144, "n_ctx_train": 1048576})],
    )
    assert SCAgent._resolve_context_limit(_agent("GLM-5.2", client)) == 262144


def test_resolve_vllm_uses_max_model_len():
    client = _FakeClient(
        "http://localhost:8000/v1",
        [SimpleNamespace(id="Qwen3.6-27B", max_model_len=262144)],
    )
    assert SCAgent._resolve_context_limit(_agent("Qwen3.6-27B", client)) == 262144


def test_resolve_env_override_wins(monkeypatch):
    monkeypatch.setenv("SCAGENT_CONTEXT_LIMIT", "50000")
    client = _FakeClient(
        "http://localhost:8001/v1",
        [SimpleNamespace(id="GLM-5.2", meta={"n_ctx": 262144})],
    )
    assert SCAgent._resolve_context_limit(_agent("GLM-5.2", client)) == 50000


def test_resolve_falls_through_when_server_silent():
    # Server reachable but advertises no limit for our model → name/default path.
    # "GLM-5.2" has no K-token in the name and isn't in the cloud table, so it
    # lands on the hard 128K default rather than crashing.
    client = _FakeClient(
        "http://localhost:8001/v1",
        [SimpleNamespace(id="GLM-5.2")],
    )
    assert SCAgent._resolve_context_limit(_agent("GLM-5.2", client)) == 128_000


def test_resolve_server_query_failure_is_caught():
    # A backend that raises on models.list() must not crash resolution.
    class _Boom:
        base_url = "http://localhost:8001/v1"

        class models:
            @staticmethod
            def list():
                raise RuntimeError("connection refused")

    assert SCAgent._resolve_context_limit(_agent("GLM-5.2", _Boom())) == 128_000
