"""FastAPI server exposing localization endpoints, session polling, warmup, TTS, eval, and live relay."""

from __future__ import annotations

from contextlib import asynccontextmanager
import copy
from io import BytesIO
import inspect
import json
import os
from pathlib import Path
import threading
import time
import uuid

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from app import config, live, llm, pipeline
from app.contracts import LatLng

EVAL_RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

_SESSIONS: dict[str, dict] = {}
_LOCK = threading.Lock()
_tts_cache: dict[str, bytes] = {}
_WARM: dict = {
    "state": "idle",
    "started": None,
    "finished": None,
    "results": {},
    "backend": "",
}


class LocateTextRequest(BaseModel):
    text: str = ""
    region: str | None = None
    lat: float | None = None
    lng: float | None = None


class AnswerRequest(BaseModel):
    session_id: str
    option_index: int | None = None
    text: str | None = None


def _backend_label() -> str:
    """Return human-readable Gemini backend label."""
    fn = getattr(config, "backend_label", None)
    if callable(fn):
        return fn()
    b = config.gemini_backend() if hasattr(config, "gemini_backend") else ""
    if b == "vertex":
        loc = config.get("EODI_VERTEX_LOCATION", "global") or "global"
        return f"Vertex AI · {loc}"
    if b == "aistudio":
        return "Google AI Studio"
    return b or "none"


def _start_warmup() -> dict:
    """Start the background model warmup thread if not already running."""
    with _LOCK:
        if _WARM["state"] == "running":
            return dict(_WARM)
        _WARM["state"] = "running"
        _WARM["started"] = time.time()
        _WARM["finished"] = None
        _WARM["results"] = {}
        _WARM["backend"] = _backend_label()

    def _run() -> None:
        try:
            res = llm.warmup()
        except Exception as exc:
            res = {"error": f"{type(exc).__name__}: {exc}"}
        with _LOCK:
            _WARM["results"] = res
            _WARM["finished"] = time.time()
            _WARM["state"] = "done"
            _WARM["backend"] = _backend_label()

    threading.Thread(target=_run, daemon=True).start()
    with _LOCK:
        return dict(_WARM)


@asynccontextmanager
async def _lifespan(app_: FastAPI):
    """Trigger background model warmup on server startup unless disabled by EODI_WARMUP=0."""
    if os.environ.get("EODI_WARMUP", "1") != "0":
        _start_warmup()
    yield


app = FastAPI(title="Eodi", lifespan=_lifespan)


@app.middleware("http")
async def _no_store_static(request: Request, call_next):
    """Attach Cache-Control: no-store to non-API static responses."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def _normalize_photo(raw_bytes: bytes) -> bytes:
    """Convert uploaded image bytes to RGB JPEG with longest side <= 1024px."""
    try:
        with Image.open(BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            longest = max(w, h)
            if longest > 1024:
                scale = 1024.0 / float(longest)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                rgb = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
            out = BytesIO()
            rgb.save(out, format="JPEG", quality=85)
            return out.getvalue()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc


def _spawn_locate(
    *,
    text: str | None = None,
    audio: bytes | None = None,
    mime: str | None = None,
    region: str | None = None,
    current: LatLng | None = None,
    photo: bytes | None = None,
    photo_role: str = "target",
) -> str:
    """Initialize a server polling session and run pipeline.locate in a background thread."""
    sid = uuid.uuid4().hex[:8]
    ev = threading.Event()
    init_resp = {
        "session_id": sid,
        "done": False,
        "progress": [],
        "decision": None,
    }
    with _LOCK:
        _SESSIONS[sid] = {"resp": init_resp, "done": False, "event": ev}

    def _on_progress(r: dict) -> None:
        snap = copy.deepcopy(r)
        with _LOCK:
            if sid in _SESSIONS:
                _SESSIONS[sid]["resp"] = snap
                if snap.get("done"):
                    _SESSIONS[sid]["done"] = True

    def _worker() -> None:
        try:
            sig = inspect.signature(pipeline.locate)
            params = sig.parameters
            has_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            kw: dict = {
                "text": text,
                "audio": audio,
                "mime": mime,
                "region": region,
                "current": current,
                "on_progress": _on_progress,
                "session_id": sid,
            }
            if "photo" in params or has_var_kw:
                kw["photo"] = photo
            if "photo_role" in params or has_var_kw:
                kw["photo_role"] = photo_role
            final = pipeline.locate(**kw)
            snap = copy.deepcopy(final) if isinstance(final, dict) else init_resp
            snap["session_id"] = sid
            snap["done"] = True
            with _LOCK:
                _SESSIONS[sid]["resp"] = snap
                _SESSIONS[sid]["done"] = True
        except Exception as exc:
            with _LOCK:
                err_resp = copy.deepcopy(_SESSIONS[sid]["resp"])
                err_resp["error"] = f"{type(exc).__name__}: {exc}"
                err_resp["done"] = True
                _SESSIONS[sid]["resp"] = err_resp
                _SESSIONS[sid]["done"] = True
        finally:
            ev.set()

    threading.Thread(target=_worker, daemon=True).start()
    return sid


def _wait_session(sid: str, timeout: float = 60.0) -> dict:
    """Wait for a background session to finish and return its final response dict."""
    with _LOCK:
        entry = _SESSIONS.get(sid)
    if not entry:
        return {"session_id": sid, "done": True, "error": "Session not found"}
    entry["event"].wait(timeout=timeout)
    with _LOCK:
        resp = copy.deepcopy(_SESSIONS[sid]["resp"])
        resp["done"] = bool(_SESSIONS[sid]["done"])
        return resp


@app.post("/api/locate_text")
def api_locate_text(req: LocateTextRequest):
    """Start an asynchronous text localization session and return its session_id."""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    current = (
        LatLng(float(req.lat), float(req.lng))
        if (req.lat is not None and req.lng is not None)
        else None
    )
    clean_region = req.region.strip() if (req.region and req.region.strip()) else None
    sid = _spawn_locate(
        text=req.text.strip(),
        region=clean_region,
        current=current,
    )
    return {"session_id": sid, "status": "running"}


@app.post("/api/locate")
async def api_locate(
    text: str | None = Form(default=None),
    region: str | None = Form(default=None),
    lat: float | None = Form(default=None),
    lng: float | None = Form(default=None),
    photo_role: str = Form(default="target"),
    audio: UploadFile | None = File(default=None),
    photo: UploadFile | None = File(default=None),
):
    """Start a multimodal localization session from form text, audio, and/or photo."""
    clean_text = text.strip() if (text and text.strip()) else None
    clean_region = region.strip() if (region and region.strip()) else None

    audio_bytes: bytes | None = None
    audio_mime: str | None = None
    if audio is not None:
        raw_audio = await audio.read()
        if raw_audio:
            audio_bytes = raw_audio
            ct = (audio.content_type or "audio/webm").split(";")[0].strip()
            audio_mime = ct or "audio/webm"

    photo_bytes: bytes | None = None
    if photo is not None:
        raw_photo = await photo.read()
        if raw_photo:
            photo_bytes = _normalize_photo(raw_photo)

    if clean_text is None and audio_bytes is None and photo_bytes is None:
        raise HTTPException(status_code=400, detail="At least one of text, audio, or photo is required")

    role = photo_role if photo_role in ("target", "landmark") else "target"
    if clean_text is None and audio_bytes is None:
        role = "target"
    current = LatLng(float(lat), float(lng)) if (lat is not None and lng is not None) else None

    sid = _spawn_locate(
        text=clean_text,
        audio=audio_bytes,
        mime=audio_mime,
        region=clean_region,
        current=current,
        photo=photo_bytes,
        photo_role=role,
    )
    return {"session_id": sid, "status": "running"}


@app.post("/api/answer")
def api_answer(req: AnswerRequest):
    """Submit a follow-up answer for an existing session and re-evaluate in the background."""
    sid = req.session_id
    if sid not in pipeline.SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")

    ev = threading.Event()
    with _LOCK:
        prev_resp = copy.deepcopy((_SESSIONS.get(sid) or {}).get("resp") or {"session_id": sid})
        prev_resp["decision"] = None
        prev_resp["done"] = False
        _SESSIONS[sid] = {"resp": prev_resp, "done": False, "event": ev}

    def _on_progress(r: dict) -> None:
        snap = copy.deepcopy(r)
        with _LOCK:
            if sid in _SESSIONS:
                _SESSIONS[sid]["resp"] = snap
                if snap.get("done"):
                    _SESSIONS[sid]["done"] = True

    def _worker() -> None:
        try:
            final = pipeline.answer(
                sid,
                option_index=req.option_index,
                text=req.text,
                on_progress=_on_progress,
            )
            snap = copy.deepcopy(final) if isinstance(final, dict) else prev_resp
            snap["session_id"] = sid
            snap["done"] = True
            with _LOCK:
                _SESSIONS[sid]["resp"] = snap
                _SESSIONS[sid]["done"] = True
        except Exception as exc:
            with _LOCK:
                err_resp = copy.deepcopy(_SESSIONS[sid]["resp"])
                err_resp["error"] = f"{type(exc).__name__}: {exc}"
                err_resp["done"] = True
                _SESSIONS[sid]["resp"] = err_resp
                _SESSIONS[sid]["done"] = True
        finally:
            ev.set()

    threading.Thread(target=_worker, daemon=True).start()
    return {"session_id": sid, "status": "running"}


@app.get("/api/session/{sid}")
def api_session(sid: str):
    """Return the current progress or completed result for session sid."""
    with _LOCK:
        entry = _SESSIONS.get(sid)
        if entry is None:
            raise HTTPException(status_code=404, detail="Session not found")
        resp = copy.deepcopy(entry["resp"])
        resp["done"] = bool(entry["done"])
    return resp


@app.get("/api/warmup")
def api_get_warmup():
    """Return current model warmup status and results."""
    with _LOCK:
        out = dict(_WARM)
        if not out.get("backend"):
            out["backend"] = _backend_label()
        return out


@app.post("/api/warmup")
def api_post_warmup():
    """Start model warmup in the background and return warmup status."""
    return _start_warmup()


@app.get("/api/tts")
def api_tts(text: str = Query(..., min_length=1)):
    """Synthesize speech WAV for text and cache the result in memory."""
    clean = text.strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Empty text")
    if clean in _tts_cache:
        return Response(content=_tts_cache[clean], media_type="audio/wav")
    wav = llm.tts(clean)
    _tts_cache[clean] = wav
    return Response(content=wav, media_type="audio/wav")


@app.get("/api/eval")
def api_eval():
    """Return offline evaluation results from eval/results.json if present."""
    if not EVAL_RESULTS.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found")
    data = json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))
    return JSONResponse(content=data)


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """Relay Gemini Live WebSocket sessions and bridge to the localization pipeline."""
    def _start_from_live(text: str, region: str | None = None, lat: float | None = None, lng: float | None = None) -> str:
        cur = LatLng(float(lat), float(lng)) if (lat is not None and lng is not None) else None
        return _spawn_locate(text=text, region=region, current=cur)

    def _wait_from_live(sid: str) -> dict:
        return _wait_session(sid, timeout=60.0)

    await ws.accept()

    async def _noop_accept(*a, **kw):
        return None

    ws.accept = _noop_accept
    await live.relay(ws, start_session=_start_from_live, wait_session=_wait_from_live)


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
