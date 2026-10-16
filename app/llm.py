"""Gemini client wrapper providing structured JSON generation, hedging chains, embeddings, and TTS."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import math
import os
import queue
import sys
import threading
import time

import google.auth
from google import genai
from google.genai import errors, types
import httpx

from app import config

TIMEOUT_MS = 60_000
HEDGE_MS = int(os.environ.get("EODI_HEDGE_MS", "8000"))

_RETRY_CODES = {429, 502, 503, 504}
_RETRY_WAITS = (3, 7)

_client = None
_backend_label = ""
_client_lock = threading.Lock()

_usage_log: list[dict] = []
_usage_lock = threading.Lock()
_thread_ctx = threading.local()


class LLMError(Exception):
    """Raised when an LLM request times out, is cancelled, or exhausts retries."""


def make_client():
    """Create a configured google-genai Client and human-readable backend label."""
    b = config.gemini_backend()
    http_options = types.HttpOptions(timeout=TIMEOUT_MS)
    if b == "vertex":
        project = config.get("EODI_VERTEX_PROJECT")
        location = config.get("EODI_VERTEX_LOCATION") or "global"
        adc = config.get("EODI_VERTEX_ADC")
        if adc:
            creds, _ = google.auth.load_credentials_from_file(
                adc, scopes=config.SCOPES, quota_project_id=project
            )
        elif config.on_cloud_run():
            creds, _ = google.auth.default(
                scopes=config.SCOPES, quota_project_id=project
            )
        else:
            raise RuntimeError("Vertex credentials require EODI_VERTEX_ADC or Cloud Run")
        c = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=creds,
            http_options=http_options,
        )
        return c, f"Vertex AI ({project}/{location})"
    if b == "aistudio":
        api_key = config.get("GEMINI_API_KEY")
        c = genai.Client(api_key=api_key, http_options=http_options)
        return c, "Google AI Studio"
    raise RuntimeError(
        "Gemini backend not configured: set EODI_VERTEX_PROJECT + EODI_VERTEX_ADC or GEMINI_API_KEY"
    )


def client():
    """Return the singleton google-genai Client, creating it under a lock if needed."""
    global _client, _backend_label
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:
            return _client
        _client, _backend_label = make_client()
        return _client


def backend() -> str:
    """Return a human-readable label for the active Gemini backend."""
    if _backend_label and _client is not None:
        return _backend_label
    b = config.gemini_backend()
    if b == "vertex":
        project = config.get("EODI_VERTEX_PROJECT") or ""
        location = config.get("EODI_VERTEX_LOCATION") or "global"
        return f"Vertex AI ({project}/{location})"
    if b == "aistudio":
        return "Google AI Studio"
    return "unconfigured"


def _log(msg: str) -> None:
    """Write a diagnostic retry line to stderr with secrets masked."""
    try:
        sys.stderr.write(config.mask(msg) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _with_retry(
    fn,
    stage: str = "llm",
    retry_timeouts: bool = True,
    cancelled: threading.Event | None = None,
):
    """Execute fn with retries on transient HTTP errors and at most one timeout retry."""
    timeouts = 0
    for attempt in range(3):
        if cancelled is not None and cancelled.is_set():
            raise LLMError(f"{stage}: 취소")
        try:
            return fn()
        except (httpx.TimeoutException, TimeoutError) as exc:
            timeouts += 1
            if not retry_timeouts or timeouts >= 2 or attempt >= 2:
                raise LLMError(f"{stage}: timeout: {exc}") from exc
            _log(f"[llm] {stage}: timeout -> retry (1/1)")
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code not in _RETRY_CODES:
                raise
            if attempt >= 2:
                raise LLMError(f"{stage}: HTTP {code} after 3 attempts") from exc
            wait = _RETRY_WAITS[attempt]
            _log(f"[llm] {stage}: HTTP {code} -> retry in {wait}s ({attempt + 1}/2)")
            if cancelled is not None and cancelled.is_set():
                raise LLMError(f"{stage}: 취소")
            time.sleep(wait)
    raise LLMError(f"{stage}: exhausted retries")


def _record(stage: str, model: str, ms: int, resp=None, wasted: bool = False) -> dict:
    """Record token counts and latency for a completed Gemini call without prompt text."""
    meta = getattr(resp, "usage_metadata", None)
    entry: dict = {
        "stage": stage,
        "model": model,
        "ms": int(ms),
        "input_tokens": int(getattr(meta, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(meta, "candidates_token_count", 0) or 0),
        "thought_tokens": int(getattr(meta, "thoughts_token_count", 0) or 0),
    }
    cancel_ev = getattr(_thread_ctx, "cancel_ev", None)
    if wasted or (cancel_ev is not None and cancel_ev.is_set()):
        entry["wasted"] = True
    box = getattr(_thread_ctx, "usage_box", None)
    if isinstance(box, list):
        box.append(entry)
    with _usage_lock:
        _usage_log.append(entry)
        try:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            with (config.LOG_DIR / "gemini_usage.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return entry


def usage() -> list[dict]:
    """Return a copy of recorded Gemini usage entries."""
    with _usage_lock:
        return [dict(row) for row in _usage_log]


def reset_usage() -> None:
    """Clear in-memory usage entries."""
    with _usage_lock:
        _usage_log.clear()


def _thinking_config(thinking: str | None) -> types.ThinkingConfig | None:
    """Build a ThinkingConfig for the requested level, or None when disabled."""
    if not thinking:
        return None
    lv = thinking.strip().upper()
    level_map = {
        "LOW": types.ThinkingLevel.LOW,
        "MEDIUM": types.ThinkingLevel.MEDIUM,
        "HIGH": types.ThinkingLevel.HIGH,
    }
    return types.ThinkingConfig(thinking_level=level_map.get(lv, types.ThinkingLevel.LOW))


def _generate(
    stage: str,
    model: str,
    contents,
    cfg: types.GenerateContentConfig,
    retry_timeouts: bool = True,
    cancelled: threading.Event | None = None,
):
    """Run generate_content with retry policy and usage tracking."""
    t0 = time.monotonic()
    resp = _with_retry(
        lambda: client().models.generate_content(model=model, contents=contents, config=cfg),
        stage=stage,
        retry_timeouts=retry_timeouts,
        cancelled=cancelled,
    )
    ms = int((time.monotonic() - t0) * 1000)
    _record(stage, model, ms, resp)
    return resp


def generate_json(
    stage: str,
    model: str,
    contents,
    schema: dict,
    thinking: str | None = "low",
    system: str | None = None,
    timeout_ms: int | None = None,
    retry_timeouts: bool = True,
    cancelled: threading.Event | None = None,
) -> dict:
    """Generate structured JSON matching schema and return the parsed dictionary."""
    cfg_kwargs: dict = {
        "response_mime_type": "application/json",
        "response_json_schema": schema,
    }
    if system is not None:
        cfg_kwargs["system_instruction"] = system
    tc = _thinking_config(thinking)
    if tc is not None:
        cfg_kwargs["thinking_config"] = tc
    if timeout_ms is not None:
        cfg_kwargs["http_options"] = types.HttpOptions(timeout=timeout_ms)
    cfg = types.GenerateContentConfig(**cfg_kwargs)
    resp = _generate(
        stage,
        model,
        contents,
        cfg,
        retry_timeouts=retry_timeouts,
        cancelled=cancelled,
    )
    text = getattr(resp, "text", None)
    if not text:
        raise LLMError(f"{stage}: empty JSON response from {model}")
    try:
        return json.loads(text)
    except Exception as exc:
        raise LLMError(f"{stage}: invalid JSON from {model}: {exc}") from exc


def _call_generate_json(stage: str, model: str, contents, schema: dict, cancel_ev, kw: dict) -> dict:
    """Invoke generate_json, forwarding cancelled only if supported by the callable."""
    sig = inspect.signature(generate_json)
    params = sig.parameters
    call_kw = dict(kw)
    if "cancelled" in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    ):
        call_kw["cancelled"] = cancel_ev
    return generate_json(stage, model, contents, schema, **call_kw)


def _chain_sequential(stage: str, chain: list[str], contents, schema: dict, **kw) -> dict:
    """Try models in sequence, returning the first success or raising the last error."""
    last_err: Exception | None = None
    for model in chain:
        try:
            return _call_generate_json(stage, model, contents, schema, None, kw)
        except (LLMError, errors.APIError) as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    raise LLMError(f"{stage}: no models in chain")


def generate_json_chain(
    stage: str,
    models: list[str],
    contents,
    schema: dict,
    hedge_ms: int | None = None,
    **kw,
) -> dict:
    """Run a model chain with hedging when earlier models exceed hedge_ms."""
    chain = list(dict.fromkeys(m for m in models if m))
    if not chain:
        raise LLMError(f"{stage}: empty model chain")
    wait_ms = HEDGE_MS if hedge_ms is None else hedge_ms
    if len(chain) == 1 or wait_ms <= 0:
        return _chain_sequential(stage, chain, contents, schema, **kw)

    q: queue.Queue = queue.Queue()
    workers: list[tuple[int, threading.Event, list[dict]]] = []
    next_idx = 0
    in_flight = 0
    last_err: Exception | None = None

    def launch(idx: int) -> None:
        nonlocal in_flight
        cancel_ev = threading.Event()
        usage_box: list[dict] = []
        workers.append((idx, cancel_ev, usage_box))
        in_flight += 1

        def runner() -> None:
            _thread_ctx.cancel_ev = cancel_ev
            _thread_ctx.usage_box = usage_box
            t0 = time.monotonic()
            try:
                res = _call_generate_json(stage, chain[idx], contents, schema, cancel_ev, kw)
                ms = int((time.monotonic() - t0) * 1000)
                q.put((idx, True, res, ms, cancel_ev, usage_box))
            except Exception as exc:
                ms = int((time.monotonic() - t0) * 1000)
                q.put((idx, False, exc, ms, cancel_ev, usage_box))
            finally:
                _thread_ctx.cancel_ev = None
                _thread_ctx.usage_box = None

        t = threading.Thread(target=runner, daemon=True)
        t.start()

    launch(0)
    next_idx = 1

    while in_flight > 0 or next_idx < len(chain):
        if in_flight == 0 and next_idx < len(chain):
            launch(next_idx)
            next_idx += 1
        timeout_s = (wait_ms / 1000.0) if next_idx < len(chain) else None
        try:
            idx, ok, payload, ms, cancel_ev, usage_box = q.get(timeout=timeout_s)
        except queue.Empty:
            if next_idx < len(chain):
                launch(next_idx)
                next_idx += 1
            continue

        in_flight -= 1
        if ok:
            for other_idx, other_ev, other_box in workers:
                if other_idx != idx:
                    other_ev.set()
                    for row in other_box:
                        row["wasted"] = True
            return payload

        last_err = payload
        if in_flight == 0 and next_idx < len(chain):
            launch(next_idx)
            next_idx += 1

    if last_err is not None:
        raise last_err
    raise LLMError(f"{stage}: model chain failed")


def generate_tools(
    stage: str,
    model: str,
    contents,
    tools: list,
    tool_config=None,
    timeout_ms: int | None = None,
    retry_timeouts: bool = True,
):
    """Run generate_content with tool definitions and return the raw response object."""
    cfg_kwargs: dict = {"tools": tools, "tool_config": tool_config}
    if timeout_ms is not None:
        cfg_kwargs["http_options"] = types.HttpOptions(timeout=timeout_ms)
    cfg = types.GenerateContentConfig(**cfg_kwargs)
    sig = inspect.signature(_generate)
    params = sig.parameters
    if "timeout_ms" in params:
        return _generate(
            stage,
            model,
            contents,
            cfg,
            timeout_ms=timeout_ms,
            retry_timeouts=retry_timeouts,
        )
    if "retry_timeouts" in params:
        return _generate(stage, model, contents, cfg, retry_timeouts=retry_timeouts)
    return _generate(stage, model, contents, cfg, retry_timeouts)


def embed(item: str | bytes, mime: str | None = None) -> list[float]:
    """Compute a 3,072-dimensional embedding vector for a single text or image input."""
    if isinstance(item, (bytes, bytearray)):
        content = types.Part.from_bytes(data=bytes(item), mime_type=mime or "image/jpeg")
    else:
        content = str(item)
    t0 = time.monotonic()
    resp = _with_retry(
        lambda: client().models.embed_content(model=config.MODEL_EMBED, contents=content),
        stage="embed",
    )
    ms = int((time.monotonic() - t0) * 1000)
    _record("embed", config.MODEL_EMBED, ms, resp)
    return list(resp.embeddings[0].values)


def cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def tts(text: str, voice: str = "Kore") -> bytes:
    """Synthesize speech audio bytes (WAV) from text using the TTS model."""
    cfg = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
        http_options=types.HttpOptions(timeout=20_000),
    )
    resp = _generate("tts", config.MODEL_TTS, text, cfg, retry_timeouts=False)
    part = resp.candidates[0].content.parts[0]
    return part.inline_data.data


def warmup(image: bool = True) -> dict:
    """Warm up active Gemini models in parallel and return per-model latency and status."""
    tiny_schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    tasks: list[tuple[str, callable]] = [
        (
            "main",
            lambda: generate_json(
                "warmup",
                config.MODEL_MAIN,
                ["ping"],
                tiny_schema,
                thinking="low",
                timeout_ms=15_000,
                retry_timeouts=False,
            ),
        ),
        (
            "light",
            lambda: generate_json(
                "warmup",
                config.MODEL_LIGHT,
                ["ping"],
                tiny_schema,
                thinking=None,
                timeout_ms=15_000,
                retry_timeouts=False,
            ),
        ),
    ]
    if config.MODEL_PARSE not in (config.MODEL_MAIN, config.MODEL_LIGHT):
        tasks.append(
            (
                "argon",
                lambda: generate_json(
                    "warmup",
                    config.MODEL_PARSE,
                    ["ping"],
                    tiny_schema,
                    thinking="low",
                    timeout_ms=15_000,
                    retry_timeouts=False,
                ),
            )
        )
    tasks.append(("embed", lambda: embed("예열")))
    tasks.append(("tts", lambda: tts("예열")))

    results: dict = {}

    def _run_one(name: str, fn) -> tuple[str, dict]:
        t0 = time.monotonic()
        try:
            fn()
            return name, {"ms": int((time.monotonic() - t0) * 1000), "ok": True, "error": None}
        except Exception as exc:
            return name, {
                "ms": int((time.monotonic() - t0) * 1000),
                "ok": False,
                "error": config.mask(f"{type(exc).__name__}: {exc}"),
            }

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futs = [pool.submit(_run_one, name, fn) for name, fn in tasks]
        for fut in futs:
            name, info = fut.result()
            results[name] = info
    return results


if __name__ == "__main__":
    schema = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    out = generate_json("smoke", config.MODEL_MAIN, ["Return ok=true"], schema)
    print("generate_json OK", out)
    vec = embed("오토웨이타워")
    print("embed OK", len(vec))
    wav = tts("안녕하세요")
    mime = "audio/wav" if wav[:4] == b"RIFF" else "application/octet-stream"
    print("tts OK", mime, len(wav))
