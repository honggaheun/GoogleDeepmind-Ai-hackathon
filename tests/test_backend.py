"""Unit tests for Gemini backend selection and client initialization."""

from __future__ import annotations

import importlib
import threading
import time

import pytest

from app import config, live, llm


def _cfg(monkeypatch, values: dict, backend_env: str | None = None) -> None:
    """Stub config.get with values and optionally set EODI_GEMINI_BACKEND."""
    monkeypatch.delenv("K_SERVICE", raising=False)
    if backend_env is None:
        monkeypatch.delenv("EODI_GEMINI_BACKEND", raising=False)
    else:
        monkeypatch.setenv("EODI_GEMINI_BACKEND", backend_env)
    monkeypatch.setattr(config, "get", lambda name, default=None: values.get(name, default))


def test_backend_prefers_vertex_when_project_and_adc_exist(monkeypatch):
    _cfg(
        monkeypatch,
        {
            "EODI_VERTEX_PROJECT": "p",
            "EODI_VERTEX_ADC": "adc.json",
            "GEMINI_API_KEY": "k",
        },
    )
    assert config.gemini_backend() == "vertex"


def test_backend_falls_back_to_aistudio_key(monkeypatch):
    _cfg(monkeypatch, {"GEMINI_API_KEY": "k"})
    assert config.gemini_backend() == "aistudio"


def test_backend_env_forces_aistudio_even_with_vertex(monkeypatch):
    _cfg(
        monkeypatch,
        {
            "EODI_VERTEX_PROJECT": "p",
            "EODI_VERTEX_ADC": "adc.json",
            "GEMINI_API_KEY": "k",
        },
        backend_env="aistudio",
    )
    assert config.gemini_backend() == "aistudio"


def test_backend_empty_without_any_setting(monkeypatch):
    _cfg(monkeypatch, {})
    assert config.gemini_backend() == ""


def test_client_uses_api_key_on_aistudio(monkeypatch):
    _cfg(monkeypatch, {"GEMINI_API_KEY": "k"})
    seen = {}
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm.genai, "Client", lambda **kw: (seen.update(kw) or "client"))
    assert llm.client() == "client"
    assert seen["api_key"] == "k" and "vertexai" not in seen and (seen["http_options"].timeout == 60000)
    assert "AI Studio" in llm.backend()


def test_client_is_created_once_even_when_threads_ask_at_the_same_time(monkeypatch):
    _cfg(monkeypatch, {"GEMINI_API_KEY": "k"})
    made = []
    got = []

    def _build(**kw):
        time.sleep(0.02)
        obj = object()
        made.append(obj)
        return obj

    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm.genai, "Client", _build)
    threads = [threading.Thread(target=lambda: got.append(llm.client())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made) == 1 and len(got) == 8 and all((g is made[0] for g in got))


def test_client_without_setting_raises_a_clear_error(monkeypatch):
    _cfg(monkeypatch, {})
    monkeypatch.setattr(llm, "_client", None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        llm.client()


def test_live_client_uses_api_key_on_aistudio(monkeypatch):
    _cfg(monkeypatch, {"GEMINI_API_KEY": "k"})
    seen = {}
    monkeypatch.setattr(live.genai, "Client", lambda **kw: (seen.update(kw) or "client"))
    assert live._client("us-central1") == "client"
    assert seen["api_key"] == "k" and "vertexai" not in seen


def test_cloud_run_uses_the_service_account_without_an_adc_file(monkeypatch):
    _cfg(monkeypatch, {"EODI_VERTEX_PROJECT": "p"})
    monkeypatch.setenv("K_SERVICE", "eodi")
    seen = {}
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm.google.auth, "default", lambda **kw: ("sa-creds", "p"))
    monkeypatch.setattr(llm.genai, "Client", lambda **kw: (seen.update(kw) or "client"))
    monkeypatch.setattr(live.google.auth, "default", lambda **kw: ("sa-creds", "p"))
    monkeypatch.setattr(live.genai, "Client", lambda **kw: (seen.update(kw) or "client"))
    assert config.gemini_backend() == "vertex"
    assert llm.client() == "client" and seen["vertexai"] is True and (seen["credentials"] == "sa-creds") and (seen["project"] == "p")
    seen.clear()
    assert live._client("us-central1") == "client" and seen["credentials"] == "sa-creds" and (seen["location"] == "us-central1")


def test_local_pc_without_adc_file_never_falls_back_to_default_credentials(monkeypatch):
    _cfg(monkeypatch, {"EODI_VERTEX_PROJECT": "p", "GEMINI_API_KEY": "k"})
    monkeypatch.delenv("K_SERVICE", raising=False)
    assert config.gemini_backend() == "aistudio"


def test_osm_db_path_can_be_moved_by_environment(monkeypatch):
    monkeypatch.delenv("EODI_OSM_DB", raising=False)
    importlib.reload(config)
    assert str(config.OSM_DB).replace("\\", "/").endswith("data/osm_kr/korea.sqlite")
    monkeypatch.setenv("EODI_OSM_DB", "/tmp/korea.sqlite")
    assert str(importlib.reload(config).OSM_DB).replace("\\", "/").endswith("/tmp/korea.sqlite")
    monkeypatch.delenv("EODI_OSM_DB", raising=False)
    importlib.reload(config)
