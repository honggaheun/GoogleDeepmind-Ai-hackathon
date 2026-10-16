"""Unit tests for Gemini Live audio, tool-call, and transcript relay."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app import live

TEXT = "삼성역 3번 출구에서 직진, 오토웨이 타워"


class FakeWS:
    """Fake WebSocket recording sent frames and supplying client input frames."""

    def __init__(self):
        self.incoming = [
            {"bytes": b"\x01\x02"},
            {"type": "end"},
        ]
        self.sent: list = []

    async def receive(self):
        if not self.incoming:
            return {"type": "websocket.disconnect"}
        return self.incoming.pop(0)

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append({"type": "audio", "data": data})


class FakeSession:
    """Fake Gemini Live session recording audio input and tool responses."""

    def __init__(self):
        self.audio: list[bytes] = []
        self.ended: bool = False
        self.tool_responses: list = []

    async def send_realtime_input(self, *, audio=None, audio_stream_end=None, **kw) -> None:
        if audio is not None:
            self.audio.append(audio.data)
        if audio_stream_end:
            self.ended = True

    async def send_tool_response(self, *, function_responses) -> None:
        if isinstance(function_responses, list):
            self.tool_responses.extend(function_responses)
        else:
            self.tool_responses.append(function_responses)

    async def receive(self):
        yield SimpleNamespace(
            server_content=SimpleNamespace(
                input_transcription=SimpleNamespace(text=TEXT),
                output_transcription=None,
                model_turn=None,
                turn_complete=False,
            ),
            tool_call=None,
        )
        yield SimpleNamespace(
            server_content=None,
            tool_call=SimpleNamespace(
                function_calls=[
                    SimpleNamespace(
                        id="fc1",
                        name="locate",
                        args={"text": TEXT, "region": None},
                    )
                ]
            ),
        )
        yield SimpleNamespace(
            server_content=SimpleNamespace(
                input_transcription=None,
                output_transcription=SimpleNamespace(text="오토웨이타워로 보여요."),
                model_turn=SimpleNamespace(
                    parts=[SimpleNamespace(inline_data=SimpleNamespace(data=b"\x03\x04"))]
                ),
                turn_complete=True,
            ),
            tool_call=None,
        )


def test_relay_bridges_audio_tool_call_and_transcripts():
    ws, session, started = (FakeWS(), FakeSession(), [])

    def start_session(text, region=None):
        started.append((text, region))
        return "s1"

    def wait_session(sid):
        assert sid == "s1"
        return {
            "decision": {
                "kind": "answer",
                "candidate_id": "C1",
                "confidence": "high",
                "reasons": [{"text_ko": "삼성역 3번 출구에서 직진 경로와 일치합니다."}],
            },
            "candidates": [{"id": "C1", "name": "오토웨이타워", "type": "named_place"}],
        }

    asyncio.run(live.relay(ws, start_session, wait_session, session=session))
    response = session.tool_responses[0].response
    kinds = [m["type"] if isinstance(m, dict) else "audio" for m in ws.sent]

    assert session.audio == [b"\x01\x02"] and session.ended is True
    assert started == [(TEXT, None)]
    assert (response["decision"], response["place"], response["confidence"]) == (
        "answer",
        "오토웨이타워",
        "high",
    )
    assert kinds == ["transcript", "locate", "transcript", "audio", "turn_complete"]
    assert ws.sent[1] == {"type": "locate", "session_id": "s1", "text": TEXT}


def test_summary_for_question_carries_question_and_options():
    resp = {
        "decision": {
            "kind": "question",
            "candidate_id": None,
            "confidence": "low",
            "question": {"intent_ko": "대략 어느 동네예요?", "options_ko": []},
        },
        "candidates": [],
    }
    out = live.summary_for_model(resp)
    assert (out["decision"], out["question"], out["options"]) == (
        "question",
        "대략 어느 동네예요?",
        [],
    )
