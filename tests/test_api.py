"""Tests for FastAPI endpoints, background sessions, TTS, and warmup."""

import time
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import llm, main


def _poll(client: TestClient, sid: str, timeout: float = 5.0) -> dict:
    """Poll session endpoint until background job completes."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/session/{sid}")
        assert r.status_code == 200
        body = r.json()
        if body.get("done"):
            return body
        time.sleep(0.02)
    raise AssertionError("Session did not complete in time")


@pytest.fixture
def client(monkeypatch):
    """TestClient with stubbed pipeline and isolated state."""
    monkeypatch.setenv("EODI_WARMUP", "0")
    main._SESSIONS.clear()
    main.pipeline.SESSIONS.clear()
    main._tts_cache.clear()
    seen = {}

    def _fake_locate(
        *,
        text=None,
        audio=None,
        mime=None,
        region=None,
        current=None,
        photo=None,
        photo_role="target",
        session_id=None,
        on_progress=None,
    ):
        seen["text"] = text
        seen["audio"] = audio
        seen["mime"] = mime
        seen["region"] = region
        seen["current"] = current
        seen["photo"] = photo
        seen["photo_role"] = photo_role
        res = {
            "session_id": session_id,
            "done": True,
            "decision": {
                "kind": "answer",
                "candidate_id": "C1",
                "confidence": "high",
                "reasons": [],
            },
        }
        if on_progress:
            on_progress(res)
        return res

    def _fake_answer(sid, *, option_index=None, text=None, on_progress=None):
        seen["answer"] = (option_index, text)
        res = {
            "session_id": sid,
            "done": True,
            "decision": {
                "kind": "answer",
                "candidate_id": "C2",
                "confidence": "high",
                "reasons": [],
            },
        }
        if on_progress:
            on_progress(res)
        return res

    monkeypatch.setattr(main.pipeline, "locate", _fake_locate)
    monkeypatch.setattr(main.pipeline, "answer", _fake_answer)

    c = TestClient(main.app)
    c.seen = seen
    return c


def test_locate_text_returns_running_then_done(client):
    start = client.post(
        "/api/locate_text",
        json={"text": "오토웨이 타워", "region": None, "lat": 37.5, "lng": 127.0},
    )
    body = _poll(client, start.json()["session_id"])
    assert start.status_code == 200 and start.json()["status"] == "running"
    assert body["decision"]["candidate_id"] == "C1" and client.seen["current"].lat == 37.5


def test_locate_text_rejects_empty_text(client):
    assert client.post("/api/locate_text", json={"text": "  "}).status_code == 400


def test_unknown_session_is_404(client):
    assert client.get("/api/session/nope").status_code == 404


def test_answer_reuses_session(client):
    sid = client.post("/api/locate_text", json={"text": "오토웨이 타워"}).json()["session_id"]
    _poll(client, sid)
    main.pipeline.SESSIONS[sid] = {}
    start = client.post("/api/answer", json={"session_id": sid, "option_index": 1})
    assert start.json() == {"session_id": sid, "status": "running"}
    assert _poll(client, sid)["decision"]["candidate_id"] == "C2" and client.seen["answer"] == (1, None)


def test_answer_unknown_session_is_404(client):
    assert client.post("/api/answer", json={"session_id": "nope", "option_index": 0}).status_code == 404


def test_audio_upload_passes_bytes_and_plain_mime(client):
    files = {"audio": ("voice.webm", b"\x1aE\xdf\xa3", "audio/webm;codecs=opus")}
    sid = client.post("/api/locate", data={"text": "", "region": "삼성동"}, files=files).json()["session_id"]
    _poll(client, sid)
    assert (
        client.seen["audio"],
        client.seen["mime"],
        client.seen["text"],
        client.seen["region"],
    ) == (b"\x1aE\xdf\xa3", "audio/webm", None, "삼성동")


def test_tts_returns_wav_and_caches(client, monkeypatch):
    calls = []
    main._tts_cache.clear()
    monkeypatch.setattr(main.llm, "tts", lambda text: (calls.append(text), b"RIFF0000WAVE")[1])
    first = client.get("/api/tts", params={"text": "오토웨이타워로 보여요."})
    second = client.get("/api/tts", params={"text": "오토웨이타워로 보여요."})
    assert first.headers["content-type"] == "audio/wav" and first.content == b"RIFF0000WAVE"
    assert second.content == first.content and calls == ["오토웨이타워로 보여요."]


def test_ws_live_route_starts_and_waits_for_pipeline(client, monkeypatch):
    async def _fake_relay(ws, start_session=None, wait_session=None, **kw):
        await ws.accept()
        sid = start_session("오토웨이 타워")
        res = wait_session(sid)
        await ws.send_json(
            {
                "type": "done",
                "kind": res["decision"]["kind"],
                "sid_known": sid in main._SESSIONS,
            }
        )
        await ws.close()

    monkeypatch.setattr(main.live, "relay", _fake_relay)
    with client.websocket_connect("/ws/live") as ws:
        assert ws.receive_json() == {"type": "done", "kind": "answer", "sid_known": True}
    assert client.seen["text"] == "오토웨이 타워"


def test_eval_missing_is_404(client, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(main, "EVAL_RESULTS", Path(__file__).resolve().parent / "__missing_eval__.json")
    assert client.get("/api/eval").status_code == 404


def test_warmup_runs_in_background_and_reports(client, monkeypatch):
    main._WARM.update(state="idle", started=None, finished=None, results={})
    monkeypatch.setattr(main.llm, "warmup", lambda *a, **kw: {"main": {"ok": True}})
    assert client.get("/api/warmup").json()["state"] == "idle"
    assert client.post("/api/warmup").json()["state"] in ("running", "done")
    deadline = time.time() + 5.0
    out = client.get("/api/warmup").json()
    while out["state"] != "done" and time.time() < deadline:
        time.sleep(0.02)
        out = client.get("/api/warmup").json()
    assert out["state"] == "done" and out["results"]["main"]["ok"] is True


def test_warmup_swallows_failures_and_keeps_going(monkeypatch):
    monkeypatch.setattr(
        llm,
        "generate_json",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(llm, "embed", lambda *a, **kw: [[0.1]])
    monkeypatch.setattr(llm, "tts", lambda *a, **kw: b"RIFF")
    out = llm.warmup(image=False)
    assert out["embed"]["ok"] is True and out["tts"]["ok"] is True
    assert out["main"]["ok"] is False and "RuntimeError" in out["main"]["error"]


def test_locate_form_accepts_a_target_photo_and_normalizes_it(monkeypatch):
    monkeypatch.setenv("EODI_WARMUP", "0")
    main._SESSIONS.clear()
    seen = {}

    def _fake_locate(**kw):
        seen.update(kw)
        res = {
            "session_id": kw.get("session_id"),
            "done": True,
            "decision": {"kind": "answer", "candidate_id": "C1"},
        }
        if kw.get("on_progress"):
            kw["on_progress"](res)
        return res

    monkeypatch.setattr(main.pipeline, "locate", _fake_locate)
    c = TestClient(main.app)

    buf = BytesIO()
    Image.new("RGBA", (2048, 512), (10, 20, 30, 255)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    r = c.post(
        "/api/locate",
        data={"text": "문정역 3번 출구 고기집", "photo_role": "landmark"},
        files={"photo": ("test.png", png_bytes, "image/png")},
    )
    assert r.status_code == 200
    _poll(c, r.json()["session_id"])
    assert (
        seen["text"] == "문정역 3번 출구 고기집"
        and seen["photo"][:2] == b"\xff\xd8"
        and (seen["photo_role"] == "landmark")
    )
    assert Image.open(BytesIO(seen["photo"])).size == (1024, 256)

    r2 = c.post(
        "/api/locate",
        data={"text": "문정역 3번 출구 고기집", "photo_role": "invalid"},
        files={"photo": ("test.png", png_bytes, "image/png")},
    )
    _poll(c, r2.json()["session_id"])
    assert seen["photo_role"] == "target"

    bad = c.post(
        "/api/locate",
        data={"text": "문정역"},
        files={"photo": ("bad.bin", b"not-an-image", "application/octet-stream")},
    )
    assert bad.status_code == 400


def test_static_files_are_not_cached(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers.get("cache-control") == "no-store"
    assert "cache-control" not in {k.lower() for k in client.get("/api/warmup").headers.keys()}


def test_locate_form_accepts_a_photo_without_text(monkeypatch):
    monkeypatch.setenv("EODI_WARMUP", "0")
    main._SESSIONS.clear()
    seen = {}

    def _fake_locate(**kw):
        seen.update(kw)
        res = {
            "session_id": kw.get("session_id"),
            "done": True,
            "decision": {"kind": "answer", "candidate_id": "C1"},
        }
        if kw.get("on_progress"):
            kw["on_progress"](res)
        return res

    monkeypatch.setattr(main.pipeline, "locate", _fake_locate)
    c = TestClient(main.app)

    buf = BytesIO()
    Image.new("RGB", (200, 100), (10, 20, 30)).save(buf, format="JPEG")
    jpg_bytes = buf.getvalue()

    r = c.post(
        "/api/locate",
        data={"text": "", "region": "송파구"},
        files={"photo": ("pic.jpg", jpg_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    _poll(c, r.json()["session_id"])
    assert (
        seen["text"] is None
        and seen["region"] == "송파구"
        and (seen["photo"][:2] == b"\xff\xd8")
    )
    assert c.post("/api/locate", data={"text": "", "region": ""}).status_code == 400
