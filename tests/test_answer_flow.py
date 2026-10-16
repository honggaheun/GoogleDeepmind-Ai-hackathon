"""Tests for multi-round follow-up answer flows across pipeline and API session stores."""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from app import judge, main, pipeline
from app.contracts import Candidate, Decision, ParseResult, Question


def _poll(client: TestClient, sid: str, timeout: float = 5.0) -> dict:
    """Poll GET /api/session/{sid} until done is True."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/session/{sid}")
        assert r.status_code == 200
        body = r.json()
        if body.get("done"):
            return body
        time.sleep(0.01)
    raise TimeoutError(f"Session {sid} did not finish within {timeout}s")


def _state(sid: str, q: Question, **extra) -> dict:
    """Populate pipeline.SESSIONS[sid] with a test session state."""
    resp = {
        "session_id": sid,
        "done": True,
        "progress": [{"stage": "judge", "status": "done", "ms": 1, "note": "", "features": [], "sub": ""}],
        "candidates": [],
        "evidence": [],
        "decision": {
            "kind": "question",
            "candidate_id": "C1",
            "confidence": "low",
            "reasons": [],
            "question": {
                "id": q.id,
                "intent_ko": q.intent_ko,
                "options_ko": list(q.options_ko or []),
                "source": q.source,
            },
            "vetoed": [],
        },
        "safeguards": [],
        "tts_text": "",
    }
    st = {
        "pr": ParseResult(
            raw={"target": {"kind": "building"}, "landmarks": [], "steps": [], "ambiguities": []},
            source_text="x",
            language="ko",
            invented=[],
            needs_anchor_question=False,
        ),
        "cands": [],
        "ev": [],
        "scores": {},
        "regs": {},
        "question": q,
        "asked": [],
        "text": "x",
        "region": None,
        "current": None,
        "photo": None,
        "photo_role": "target",
        "resp": resp,
    }
    st.update(extra)
    pipeline.SESSIONS[sid] = st
    return st


def test_api_answer_clears_the_stale_question_immediately(monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(main.pipeline, "SESSIONS", {})

    def _fake_locate(text=None, on_progress=None, session_id=None, **kw):
        main.pipeline.SESSIONS[session_id] = {"resp": {"session_id": session_id}}
        out = {
            "session_id": session_id,
            "done": True,
            "decision": {
                "kind": "question",
                "candidate_id": "C1",
                "confidence": "low",
                "question": {"id": "Q-side", "intent_ko": "좌우?", "options_ko": ["왼쪽", "오른쪽"], "source": "attribute"},
            },
        }
        if on_progress:
            on_progress(out)
        return out

    def _fake_answer(sid, option_index=None, text=None, on_progress=None):
        gate.wait(timeout=2.0)
        out = {
            "session_id": sid,
            "done": True,
            "decision": {"kind": "answer", "candidate_id": "C2", "confidence": "high", "reasons": []},
        }
        if on_progress:
            on_progress(out)
        return out

    monkeypatch.setattr(main.pipeline, "locate", _fake_locate)
    monkeypatch.setattr(main.pipeline, "answer", _fake_answer)

    client = TestClient(main.app)
    sid = client.post("/api/locate_text", json={"text": "x"}).json()["session_id"]
    assert _poll(client, sid)["decision"]["kind"] == "question"
    assert client.post("/api/answer", json={"session_id": sid, "option_index": 0}).status_code == 200
    mid = client.get(f"/api/session/{sid}").json()
    assert mid["done"] is False and mid.get("decision") is None
    gate.set()
    assert _poll(client, sid)["decision"]["candidate_id"] == "C2"


def test_dunno_to_region_question_gives_up_instead_of_reasking():
    q = Question(id="Q-region", intent_ko="대략 어디인가요?", options_ko=[], source="region", mapping=None)
    _state("s-region", q)
    out = pipeline.answer("s-region", text="모르겠어요")
    d = out["decision"]
    assert d["kind"] == "low_confidence" and d["question"] is None and (d["candidate_id"] is None)
    assert pipeline.SESSIONS["s-region"]["question"] is None and "Q-region" in pipeline.SESSIONS["s-region"]["asked"]


def test_answer_clears_the_old_decision_before_rejudging(monkeypatch):
    q = Question(
        id="Q-floors",
        intent_ko="몇 층?",
        options_ko=["3층", "5층"],
        source="attribute",
        mapping={0: ["C1"], 1: ["C2"]},
    )
    pr = ParseResult(
        raw={"target": {"kind": "building"}, "landmarks": [], "steps": [], "ambiguities": []},
        source_text="x",
        language="ko",
        invented=[],
        needs_anchor_question=False,
        distance_m={},
    )
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    _state("s-attr", q, pr=pr, cands=cands, scores={"C1": 1.0, "C2": 1.0})

    monkeypatch.setattr(judge, "apply_answer", lambda q_in, idx, cs, sc: {"C1": 2.0, "C2": 0.0})
    monkeypatch.setattr(
        judge,
        "decide",
        lambda *a, **kw: (
            Decision(kind="answer", candidate_id="C1", confidence="high", reasons=[]),
            {"vetoed": [], "unknown_ids": 0},
        ),
    )

    seen = []
    out = pipeline.answer("s-attr", option_index=0, on_progress=lambda r: seen.append(r))
    assert seen and seen[0].get("decision") is None
    assert out["decision"]["kind"] == "answer" and out["decision"]["candidate_id"] == "C1"
    assert "Q-floors" in pipeline.SESSIONS["s-attr"]["asked"]


def test_already_asked_question_is_never_asked_again_by_the_judge():
    pr = ParseResult(
        raw={"target": {"kind": "building", "name": "고기집", "specific": False, "category_code": "FD6"}, "landmarks": [], "steps": [], "ambiguities": []},
        source_text="x",
        language="ko",
        invented=[],
        needs_anchor_question=False,
        distance_m={},
    )
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    d, _ = judge.decide(pr, cands, [], {"C1": 2.0, "C2": 1.0}, asked=["Q-name"], use_gemini=False)
    assert not (d.kind == "question" and d.question and (d.question.id == "Q-name"))


def test_asked_ids_accumulate_across_rounds(monkeypatch):
    q1 = Question(
        id="Q-side",
        intent_ko="좌우?",
        options_ko=["왼쪽", "오른쪽"],
        source="attribute",
        mapping={0: ["C1"], 1: ["C2"]},
    )
    q2 = Question(
        id="Q-floors",
        intent_ko="몇 층?",
        options_ko=["3층", "5층"],
        source="attribute",
        mapping={0: ["C1"], 1: ["C2"]},
    )
    pr = ParseResult(
        raw={"target": {"kind": "building"}, "landmarks": [], "steps": [], "ambiguities": []},
        source_text="x",
        language="ko",
        invented=[],
        needs_anchor_question=False,
        distance_m={},
    )
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    _state("s-acc", q1, pr=pr, cands=cands, scores={"C1": 1.0, "C2": 1.0}, asked=["Q-name"])

    got = []

    def _fake_decide(pr_in, cs, ev, sc, asked, **kw):
        got.append(list(asked))
        if len(got) == 1:
            return Decision(kind="question", candidate_id="C1", confidence="low", question=q2), {"vetoed": []}
        return Decision(kind="answer", candidate_id="C1", confidence="high", reasons=[]), {"vetoed": []}

    monkeypatch.setattr(judge, "apply_answer", lambda q_in, idx, cs, sc: dict(sc))
    monkeypatch.setattr(judge, "phrase_question", lambda q_in, lang="ko": q_in)
    monkeypatch.setattr(judge, "decide", _fake_decide)

    pipeline.answer("s-acc", option_index=0)
    pipeline.answer("s-acc", option_index=0)
    assert got[0] == ["Q-name", "Q-side"] and got[1] == ["Q-name", "Q-side", "Q-floors"]


def test_two_answer_rounds_through_the_api_keep_the_session_serializable(monkeypatch):
    q1 = Question(
        id="Q-side",
        intent_ko="좌우?",
        options_ko=["왼쪽", "오른쪽"],
        source="attribute",
        mapping={0: ["C1"], 1: ["C2"]},
    )
    q2 = Question(
        id="Q-floors",
        intent_ko="몇 층?",
        options_ko=["3층", "5층"],
        source="attribute",
        mapping={0: ["C1"], 1: ["C2"]},
    )
    pr = ParseResult(
        raw={"target": {"kind": "building"}, "landmarks": [], "steps": [], "ambiguities": []},
        source_text="x",
        language="ko",
        invented=[],
        needs_anchor_question=False,
        distance_m={},
    )
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]

    monkeypatch.setattr(main.pipeline, "SESSIONS", {})

    def _fake_locate(text=None, on_progress=None, session_id=None, **kw):
        st = _state(session_id, q1, pr=pr, cands=cands, scores={"C1": 1.0, "C2": 1.0})
        if on_progress:
            on_progress(st["resp"])
        return st["resp"]

    got = []

    def _fake_decide(pr_in, cs, ev, sc, asked, **kw):
        got.append(list(asked))
        if len(got) == 1:
            return Decision(kind="question", candidate_id="C1", confidence="low", question=q2), {"vetoed": []}
        return Decision(kind="answer", candidate_id="C1", confidence="high", reasons=[]), {"vetoed": []}

    monkeypatch.setattr(main.pipeline, "locate", _fake_locate)
    monkeypatch.setattr(judge, "apply_answer", lambda q_in, idx, cs, sc: dict(sc))
    monkeypatch.setattr(judge, "phrase_question", lambda q_in, lang="ko": q_in)
    monkeypatch.setattr(judge, "decide", _fake_decide)

    client = TestClient(main.app)
    sid = client.post("/api/locate_text", json={"text": "x"}).json()["session_id"]
    assert _poll(client, sid)["decision"]["question"]["id"] == "Q-side"
    assert client.post("/api/answer", json={"session_id": sid, "option_index": 0}).status_code == 200
    assert _poll(client, sid)["decision"]["question"]["id"] == "Q-floors"
    assert client.post("/api/answer", json={"session_id": sid, "option_index": 0}).status_code == 200
    final = _poll(client, sid)
    assert final["decision"]["kind"] == "answer" and final["decision"]["candidate_id"] == "C1"
    assert got == [["Q-side"], ["Q-side", "Q-floors"]]
