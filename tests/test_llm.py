"""Unit tests for Gemini retry behavior, timeouts, model chains, hedging, and tool calls."""

from __future__ import annotations

import threading
import types

from google.genai import errors
import httpx
import pytest

from app import llm


@pytest.fixture
def no_sleep(monkeypatch):
    """Disable retry sleep delays during unit tests."""
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)


def _counting(outcomes: list):
    """Build a callable that records each invocation and returns or raises from outcomes."""
    calls: list[int] = []
    it = iter(outcomes)

    def fn():
        calls.append(1)
        item = next(it)
        if isinstance(item, BaseException):
            raise item
        return item

    return fn, calls


def _fake_client(monkeypatch, fail: bool = False):
    """Stub llm.client and llm._record with a fake models object capturing configs."""

    class _Models:
        def __init__(self):
            self.configs = []

        def generate_content(self, *, model, contents, config):
            self.configs.append(config)
            if fail:
                raise httpx.ReadTimeout("slow")
            return types.SimpleNamespace(text='{"ok": true}', usage_metadata=None)

    models = _Models()
    fake = types.SimpleNamespace(models=models)
    monkeypatch.setattr(llm, "client", lambda: fake)
    monkeypatch.setattr(llm, "_record", lambda *a, **kw: {})
    return models


def test_timeout_is_retried_once_then_succeeds(no_sleep):
    fn, calls = _counting([httpx.ReadTimeout("stalled"), "ok"])
    assert llm._with_retry(fn) == "ok" and len(calls) == 2


def test_timeout_twice_raises_llm_error(no_sleep):
    fn, calls = _counting([httpx.ReadTimeout("stalled"), httpx.ReadTimeout("stalled again")])
    with pytest.raises(llm.LLMError):
        llm._with_retry(fn)
    assert len(calls) == 2


def test_rate_limit_is_retried_up_to_three_attempts(no_sleep):
    busy = errors.APIError(429, {"error": {"message": "busy"}})
    fn, calls = _counting([busy, busy, "ok"])
    assert llm._with_retry(fn) == "ok" and len(calls) == 3


def test_exhausted_attempts_raise_instead_of_returning_none(no_sleep):
    busy = errors.APIError(429, {"error": {"message": "busy"}})
    fn, calls = _counting([busy, busy, httpx.ReadTimeout("stalled")])
    with pytest.raises(llm.LLMError):
        llm._with_retry(fn)
    assert len(calls) == 3


def test_bad_request_is_not_retried(no_sleep):
    fn, calls = _counting([errors.APIError(400, {"error": {"message": "bad"}})])
    with pytest.raises(errors.APIError):
        llm._with_retry(fn)
    assert len(calls) == 1


def test_generate_json_sets_per_call_timeout(monkeypatch, no_sleep):
    models = _fake_client(monkeypatch, fail=False)
    assert llm.generate_json("vision1", "m", ["x"], {"type": "object"}, timeout_ms=15000) == {"ok": True}
    assert models.configs[0].http_options.timeout == 15000


def test_timeout_without_retry_fails_fast(monkeypatch, no_sleep):
    models = _fake_client(monkeypatch, fail=True)
    with pytest.raises(llm.LLMError):
        llm.generate_json(
            "vision1",
            "m",
            ["x"],
            {"type": "object"},
            timeout_ms=15000,
            retry_timeouts=False,
        )
    assert len(models.configs) == 1


def test_client_is_built_with_http_timeout(monkeypatch):
    seen = {}
    vals = {"EODI_VERTEX_PROJECT": "p", "EODI_VERTEX_ADC": "adc.json"}
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm.config, "get", lambda k, default=None: vals.get(k, default))
    monkeypatch.setattr(
        llm.google.auth,
        "load_credentials_from_file",
        lambda path, **kw: ("creds", "p"),
    )
    monkeypatch.setattr(llm.genai, "Client", lambda **kw: (seen.update(kw) or "client"))
    assert llm.client() == "client"
    assert seen["http_options"].timeout == 60000


def test_generate_json_chain_falls_back_to_next_model(monkeypatch):
    calls = []

    def fake(stage, model, contents, schema, **kw):
        calls.append(model)
        if model == "gemini-4-argon":
            raise llm.LLMError("argon down")
        return {"ok": True}

    monkeypatch.setattr(llm, "generate_json", fake)
    out = llm.generate_json_chain(
        "parse",
        ["gemini-4-argon", "gemini-3.8-flash", "gemini-3.8-flash"],
        ["x"],
        {},
    )
    assert out == {"ok": True} and calls == ["gemini-4-argon", "gemini-3.8-flash"]


def test_generate_json_chain_raises_last_error(monkeypatch):
    def fake(stage, model, contents, schema, **kw):
        raise llm.LLMError(f"{model} down")

    monkeypatch.setattr(llm, "generate_json", fake)
    with pytest.raises(llm.LLMError, match="gemini-3.8-flash down"):
        llm.generate_json_chain("parse", ["gemini-4-argon", "gemini-3.8-flash"], ["x"], {})


def test_generate_json_chain_hedges_a_slow_first_model(monkeypatch):
    order = []

    def fake(stage, model, contents, schema, **kw):
        order.append(model)
        if model == "gemini-4-argon":
            threading.Event().wait(0.4)
            return {"from": "argon"}
        return {"from": "flash"}

    monkeypatch.setattr(llm, "generate_json", fake)
    out = llm.generate_json_chain(
        "judge",
        ["gemini-4-argon", "gemini-3.8-flash"],
        ["x"],
        {},
        hedge_ms=60,
    )
    assert out == {"from": "flash"} and order == ["gemini-4-argon", "gemini-3.8-flash"]


def test_generate_json_chain_keeps_the_first_model_when_it_is_quick(monkeypatch):
    calls = []

    def fake(stage, model, contents, schema, **kw):
        calls.append(model)
        return {"from": model}

    monkeypatch.setattr(llm, "generate_json", fake)
    out = llm.generate_json_chain(
        "parse",
        ["gemini-4-argon", "gemini-3.8-flash"],
        ["x"],
        {},
        hedge_ms=2000,
    )
    assert out == {"from": "gemini-4-argon"} and calls == ["gemini-4-argon"]


def test_cancelled_retry_does_not_call_again():
    ev = threading.Event()
    ev.set()
    fn, calls = _counting(["ok"])
    with pytest.raises(llm.LLMError, match="취소"):
        llm._with_retry(fn, cancelled=ev)
    assert len(calls) == 0


def test_generate_tools_sets_per_call_timeout_and_no_retry(monkeypatch):
    seen = {}

    def fake_gen(stage, model, contents, cfg, timeout_ms=None, retry_timeouts=True, **kw):
        seen["timeout"] = timeout_ms
        seen["retry"] = retry_timeouts
        return "resp"

    monkeypatch.setattr(llm, "_generate", fake_gen)
    assert (
        llm.generate_tools(
            "google_place",
            "m",
            ["x"],
            [],
            None,
            timeout_ms=12000,
            retry_timeouts=False,
        )
        == "resp"
    )
    assert seen == {"timeout": 12000, "retry": False}
