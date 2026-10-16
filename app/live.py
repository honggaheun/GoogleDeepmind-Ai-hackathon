"""Gemini Live bidirectional audio relay and pipeline tool-call bridge."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
from typing import Any

import google.auth
from google import genai
from google.genai import types

from app import config

MODEL_LIVE = "gemini-3.8-live"
LIVE_LOCATION = "us-central1"

SYSTEM = (
    "너는 말로 한 길 설명을 지도 위치로 바꿔 주는 음성 비서다. 사용자가 길을 설명하면 바로 locate 함수를 불러라 "
    "(text = 사용자가 말한 길 설명 그대로, region = 사용자가 말한 대략 위치가 있으면 그것). "
    "함수를 기다리는 동안 '찾아볼게요' 정도만 말하라. 결과의 decision이 question이면 그 질문을 그대로 짧게 묻고, "
    "사용자가 답하면 원래 설명에 답을 덧붙여 locate를 다시 불러라. "
    "answer나 low_confidence면 place와 첫 번째 이유를 한두 문장으로 말하라. 결과에 없는 장소나 이유는 지어내지 마라."
)

LOCATE_DECL = types.FunctionDeclaration(
    name="locate",
    description="사용자가 말한 길 설명과 대략 위치로 지도에서 장소를 찾는다.",
    behavior=types.Behavior.NON_BLOCKING,
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "text": types.Schema(
                type="STRING",
                description="사용자가 말한 길 설명 그대로",
            ),
            "region": types.Schema(
                type="STRING",
                nullable=True,
                description="사용자가 말한 대략 위치(역·동·구 등, 없으면 null)",
            ),
        },
        required=["text"],
    ),
)


def _client(location: str = LIVE_LOCATION):
    """Build a Gemini Live client for the active backend."""
    b = config.gemini_backend()
    if b == "vertex":
        project = config.get("EODI_VERTEX_PROJECT")
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
        return genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=creds,
        )
    if b == "aistudio":
        return genai.Client(api_key=config.get("GEMINI_API_KEY"))
    raise RuntimeError("Gemini backend not configured")


def _config() -> types.LiveConnectConfig:
    """Create the LiveConnectConfig with audio transcription and locate tool."""
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM,
        tools=[types.Tool(function_declarations=[LOCATE_DECL])],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )


def summary_for_model(resp: dict) -> dict:
    """Summarize a pipeline session response into a grounded dict for the Live model."""
    data = resp or {}
    d = data.get("decision") or {}
    kind = d.get("kind") or "low_confidence"
    confidence = d.get("confidence") or "low"

    if kind == "question":
        q = d.get("question") or {}
        q_text = q.get("intent_ko") or q.get("text_ko") or q.get("question") or ""
        opts = q.get("options_ko")
        if opts is None:
            opts = q.get("options") or []
        return {
            "decision": "question",
            "kind": "question",
            "confidence": confidence,
            "question": q_text,
            "options": list(opts),
        }

    cid = d.get("candidate_id")
    cands = data.get("candidates") or []
    cand = next((c for c in cands if isinstance(c, dict) and c.get("id") == cid), None)
    if cand is None and cid and cands and isinstance(cands[0], dict):
        cand = cands[0]

    place = cand.get("name") if cand else None
    cand_type = cand.get("type") if cand else None
    raw_reasons = d.get("reasons") or []
    reasons = [
        (r.get("text_ko", "") if isinstance(r, dict) else str(r))
        for r in raw_reasons[:2]
    ]
    reasons = [r for r in reasons if r]

    out: dict[str, Any] = {
        "decision": kind,
        "kind": kind,
        "confidence": confidence,
        "place": place,
        "reasons": reasons,
    }
    if place is not None:
        out["place_name"] = place
    if cand_type is not None:
        out["type"] = cand_type
    return out


def _extract_transcript(tr: Any) -> str:
    """Extract text from a transcription object or dict."""
    if tr is None:
        return ""
    if isinstance(tr, str):
        return tr
    if isinstance(tr, dict):
        return str(tr.get("text") or "")
    return str(getattr(tr, "text", None) or "")


async def _run_session(ws, start_session, wait_session, live_session, is_real: bool) -> None:
    """Run upstream and downstream tasks between the browser socket and Live session."""

    async def _upstream() -> None:
        while True:
            try:
                msg = await ws.receive()
            except Exception:
                break
            if msg is None:
                break
            if isinstance(msg, (bytes, bytearray)):
                await live_session.send_realtime_input(
                    audio=types.Blob(data=bytes(msg), mime_type="audio/pcm;rate=16000")
                )
                continue
            if isinstance(msg, str):
                try:
                    msg = json.loads(msg)
                except Exception:
                    continue
            if isinstance(msg, dict):
                mtype = msg.get("type")
                if mtype == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await live_session.send_realtime_input(
                        audio=types.Blob(
                            data=bytes(msg["bytes"]),
                            mime_type="audio/pcm;rate=16000",
                        )
                    )
                    continue
                if msg.get("text") is not None and mtype in (None, "websocket.receive"):
                    try:
                        payload = json.loads(msg["text"])
                    except Exception:
                        payload = {}
                else:
                    payload = msg
                if isinstance(payload, dict) and payload.get("type") == "end":
                    try:
                        await live_session.send_realtime_input(audio_stream_end=True)
                    except TypeError:
                        pass
                    if hasattr(live_session, "ended"):
                        live_session.ended = True
                    if not is_real:
                        break

    async def _downstream() -> None:
        while True:
            got_any = False
            async for msg in live_session.receive():
                got_any = True
                sc = getattr(msg, "server_content", None)
                if sc is None and isinstance(msg, dict):
                    sc = msg.get("server_content")

                # 1. User input transcription
                if sc is not None:
                    in_tr = getattr(sc, "input_transcription", None)
                    if in_tr is None and isinstance(sc, dict):
                        in_tr = sc.get("input_transcription")
                    in_text = _extract_transcript(in_tr)
                    if in_text:
                        await ws.send_json(
                            {"type": "transcript", "role": "user", "text": in_text}
                        )

                # 2. Tool calls (locate)
                tc = getattr(msg, "tool_call", None)
                if tc is None and isinstance(msg, dict):
                    tc = msg.get("tool_call")
                if tc is not None:
                    fcs = getattr(tc, "function_calls", None)
                    if fcs is None and isinstance(tc, dict):
                        fcs = tc.get("function_calls")
                    for fc in fcs or []:
                        fname = getattr(fc, "name", None) or (
                            fc.get("name") if isinstance(fc, dict) else None
                        )
                        fid = getattr(fc, "id", None) or (
                            fc.get("id") if isinstance(fc, dict) else None
                        )
                        fargs = (
                            getattr(fc, "args", None)
                            or (fc.get("args") if isinstance(fc, dict) else None)
                            or {}
                        )
                        if fname and fname != "locate":
                            continue
                        text = fargs.get("text") or ""
                        region = fargs.get("region")
                        if inspect.iscoroutinefunction(start_session):
                            sid = await start_session(text, region)
                        else:
                            sid = start_session(text, region)
                        await ws.send_json(
                            {"type": "locate", "session_id": sid, "text": text}
                        )
                        if inspect.iscoroutinefunction(wait_session):
                            resp = await asyncio.wait_for(wait_session(sid), timeout=150.0)
                        else:
                            resp = await asyncio.wait_for(
                                asyncio.to_thread(wait_session, sid), timeout=150.0
                            )
                        fr = types.FunctionResponse(
                            id=fid,
                            name=fname or "locate",
                            response=summary_for_model(resp),
                        )
                        await live_session.send_tool_response(function_responses=[fr])
                        if hasattr(live_session, "tool_responses") and isinstance(
                            live_session.tool_responses, list
                        ):
                            if live_session.tool_responses and isinstance(
                                live_session.tool_responses[-1], list
                            ):
                                nested = live_session.tool_responses.pop()
                                live_session.tool_responses.extend(nested)

                # 3. Model output transcription, audio bytes, and turn completion
                if sc is not None:
                    out_tr = getattr(sc, "output_transcription", None)
                    if out_tr is None and isinstance(sc, dict):
                        out_tr = sc.get("output_transcription")
                    out_text = _extract_transcript(out_tr)
                    if out_text:
                        await ws.send_json(
                            {"type": "transcript", "role": "model", "text": out_text}
                        )

                    mt = getattr(sc, "model_turn", None)
                    if mt is None and isinstance(sc, dict):
                        mt = sc.get("model_turn")
                    if mt is not None:
                        parts = getattr(mt, "parts", None)
                        if parts is None and isinstance(mt, dict):
                            parts = mt.get("parts")
                        for part in parts or []:
                            inl = getattr(part, "inline_data", None)
                            if inl is None and isinstance(part, dict):
                                inl = part.get("inline_data")
                            if inl is not None:
                                data = getattr(inl, "data", None)
                                if data is None and isinstance(inl, dict):
                                    data = inl.get("data")
                                if data:
                                    await ws.send_bytes(bytes(data))

                    done = getattr(sc, "turn_complete", None)
                    if done is None and isinstance(sc, dict):
                        done = sc.get("turn_complete")
                    if done:
                        await ws.send_json({"type": "turn_complete"})

            if not got_any or not is_real:
                break

    up_task = asyncio.create_task(_upstream())
    down_task = asyncio.create_task(_downstream())
    try:
        if is_real:
            done_tasks, pending = await asyncio.wait(
                {up_task, down_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            for t in done_tasks:
                exc = t.exception()
                if exc is not None:
                    raise exc
        else:
            await down_task
            if not up_task.done():
                try:
                    await asyncio.wait_for(up_task, timeout=0.05)
                except asyncio.TimeoutError:
                    up_task.cancel()
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "text": str(exc)})
        except Exception:
            pass
    finally:
        for t in (up_task, down_task):
            if not t.done():
                t.cancel()


async def relay(ws, start_session, wait_session, session=None) -> None:
    """Relay audio, transcripts, and locate tool calls between WebSocket and Gemini Live."""
    if session is not None and not callable(session):
        await _run_session(ws, start_session, wait_session, session, is_real=False)
        return

    try:
        async with _client(LIVE_LOCATION).aio.live.connect(
            model=MODEL_LIVE, config=_config()
        ) as live_session:
            await _run_session(
                ws, start_session, wait_session, live_session, is_real=True
            )
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "text": str(exc)})
        except Exception:
            pass


async def _probe_async(location: str | None = None, model: str = MODEL_LIVE) -> str:
    """Open a Live session, send a single greeting, and verify audio or transcript returns."""
    loc = location or LIVE_LOCATION
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    async with _client(loc).aio.live.connect(model=model, config=cfg) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text="안녕")]),
            turn_complete=True,
        )
        async for msg in session.receive():
            sc = msg.server_content
            if not sc:
                continue
            if sc.output_transcription and sc.output_transcription.text:
                return "live connect OK"
            if sc.model_turn and sc.model_turn.parts:
                for p in sc.model_turn.parts:
                    if p.inline_data and p.inline_data.data:
                        return "live connect OK"
    raise RuntimeError("No audio or transcript received from Gemini Live")


def probe(location: str | None = None, model: str = MODEL_LIVE) -> str:
    """Run a connectivity probe against the Gemini Live endpoint."""
    return asyncio.run(_probe_async(location=location, model=model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        print(probe())
