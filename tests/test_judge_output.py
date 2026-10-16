"""Unit tests for app.judge output validation, target evidence checks, and multimodal contents."""

from __future__ import annotations

from google.genai import types
import pytest

from app import judge
from app.contracts import Candidate, Evidence, ParseResult


def _pr(ambiguities=()) -> ParseResult:
    """Create a generic building ParseResult fixture."""
    return ParseResult(
        raw={
            "target": {"kind": "building"},
            "landmarks": [],
            "steps": [],
            "ambiguities": list(ambiguities),
        },
        source_text="테스트",
        language="ko",
        invented=[],
        needs_anchor_question=False,
    )


def _ev(cid: str, kind: str = "vision", supports: bool = True, weight: float = 0.5, eid: str = "E1") -> Evidence:
    """Create an Evidence item with an explicit id."""
    return Evidence(
        id=eid,
        kind=kind,
        candidate_id=cid,
        supports=supports,
        weight=weight,
        summary_ko=f"{cid} {kind}",
        data={},
    )


@pytest.fixture
def fake_judge(monkeypatch):
    """Install a configurable fake for judge.judge_with_gemini."""
    def _install(choice: str = "C1", confidence: str = "high", ids=("E1", "E2")):
        monkeypatch.setattr(
            judge,
            "judge_with_gemini",
            lambda *a, **kw: {
                "choice": choice,
                "confidence": confidence,
                "reasons": [{"text_ko": "근거 일치", "evidence_ids": list(ids)}],
                "counter_evidence_ids": [],
            },
        )
    return _install


def _meat_pr() -> ParseResult:
    """Create a ParseResult targeting a barbecue restaurant across from a landmark."""
    return ParseResult(
        raw={
            "target": {
                "kind": "building",
                "name": "고기집",
                "specific": False,
                "category_code": "FD6",
                "landmark_ref": "L1",
                "relation": "across",
            },
            "landmarks": [{"id": "L1", "evidence": "돈가스집"}],
            "steps": [],
            "ambiguities": [],
        },
        source_text="돈가스집 맞은편 고기집",
        language="ko",
        invented=[],
        needs_anchor_question=False,
    )


def _target_hit() -> Evidence:
    """Return a target business match Evidence item E3 on C1."""
    return Evidence(
        id="E3",
        kind="kakao",
        candidate_id="C1",
        supports=True,
        weight=0.5,
        summary_ko="카카오 목표 가게 일치",
        data={"target": True, "place": "명륜진사갈비"},
    )


def _relation(ok: bool) -> Evidence:
    """Return a spatial relation Evidence item E4 on C1."""
    return Evidence(
        id="E4",
        kind="relation",
        candidate_id="C1",
        supports=ok,
        weight=0.5 if ok else -0.5,
        summary_ko="맞은편 관계 확인" if ok else "맞은편 아님",
        data={"confirmed": ok},
    )


def _dept_pr() -> ParseResult:
    """Create a ParseResult for 'a building across from Hyundai Department Store' (non-business target)."""
    return ParseResult(
        raw={
            "target": {
                "kind": "building",
                "name": "",
                "specific": False,
                "category_code": "NONE",
                "landmark_ref": "L1",
                "relation": "across",
            },
            "landmarks": [{"id": "L1", "evidence": "현대백화점"}],
            "steps": [],
            "ambiguities": [],
        },
        source_text="코엑스몰 지나서 현대백화점 건너편에 있는 건물",
        language="ko",
        invented=[],
        needs_anchor_question=False,
    )


def test_validate_drops_unknown_ids_and_falls_back_when_no_reason_left():
    out = judge.validate_judge(
        {"choice": "C1", "confidence": "high", "reasons": [{"text_ko": "a", "evidence_ids": ["E9"]}]},
        known_ids={"E1"},
        cand_ids={"C1", "C2"},
        code_top="C2",
    )
    assert out["choice"] == "C2" and out["fallback"] is True and (out["unknown_ids"] == 1)


def test_validate_keeps_reason_and_cleans_ids():
    out = judge.validate_judge(
        {"choice": "C1", "confidence": "high", "reasons": [{"text_ko": "a", "evidence_ids": ["E1", "E9"]}]},
        known_ids={"E1"},
        cand_ids={"C1"},
        code_top="C1",
    )
    assert out["reasons"] == [{"text_ko": "a", "evidence_ids": ["E1"]}] and (not out.get("fallback"))


def test_validate_unknown_candidate_falls_back():
    out = judge.validate_judge(
        {"choice": "C7", "confidence": "high", "reasons": [{"text_ko": "a", "evidence_ids": ["E1"]}]},
        known_ids={"E1"},
        cand_ids={"C1", "C2"},
        code_top="C1",
    )
    assert out["choice"] == "C1" and out["fallback"] is True


def test_validate_none_choice_is_allowed():
    out = judge.validate_judge(
        {"choice": "none", "confidence": "low", "reasons": [{"text_ko": "a", "evidence_ids": ["E1"]}]},
        known_ids={"E1"},
        cand_ids={"C1"},
        code_top="C1",
    )
    assert out["choice"] == "none" and (not out.get("fallback"))


def test_decide_high_confidence_answers(fake_judge):
    fake_judge("C1")
    cands = [Candidate(id="C1"), Candidate(id="C2")]
    ev = [_ev("C1", eid="E1"), _ev("C1", kind="link", eid="E2")]
    decision, _ = judge.decide(_pr(), cands, ev, {"C1": 3.0, "C2": 1.0}, asked=[])
    assert (decision.kind, decision.candidate_id, decision.confidence) == ("answer", "C1", "high")


def test_decide_skips_vetoed_choice(fake_judge):
    fake_judge("C1")
    cands = [Candidate(id="C1"), Candidate(id="C2")]
    ev = [
        _ev("C1", kind="registry", supports=False, weight=-1.5, eid="E1"),
        _ev("C2", eid="E2"),
        _ev("C2", kind="link", eid="E3"),
    ]
    decision, meta = judge.decide(_pr(), cands, ev, {"C1": 1.0, "C2": 3.0}, asked=[])
    assert decision.candidate_id == "C2" and meta["vetoed"] == ["C1"]


def test_decide_low_confidence_asks_question(fake_judge, monkeypatch):
    fake_judge("C1", confidence="low")
    q = judge.Question(
        id="Q-side",
        intent_ko="왼쪽이에요?",
        options_ko=["왼쪽", "오른쪽"],
        source="attribute",
        mapping={"C1": 0, "C2": 1},
    )
    monkeypatch.setattr(judge, "pick_question", lambda *a, **kw: q)
    decision, _ = judge.decide(_pr(), [Candidate(id="C1"), Candidate(id="C2")], [], {"C1": 2.0, "C2": 1.9}, asked=[])
    assert decision.kind == "question" and decision.question.id == "Q-side"


def test_decide_after_three_questions_gives_low_confidence(fake_judge):
    fake_judge("C1", confidence="low")
    cands = [Candidate(id="C1"), Candidate(id="C2"), Candidate(id="C3")]
    decision, meta = judge.decide(_pr(), cands, [], {"C1": 2.0, "C2": 1.9, "C3": 1.8}, asked=["a", "b", "c"])
    assert decision.kind == "low_confidence" and meta["alternatives"] == ["C1", "C2", "C3"]


def test_decide_asks_name_when_target_business_not_evidenced(fake_judge):
    fake_judge("C1", ids=("E1", "E2"))
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    ev = [_ev("C1", kind="link", eid="E1"), _ev("C1", kind="vision", eid="E2")]
    decision, meta = judge.decide(_meat_pr(), cands, ev, {"C1": 3.0, "C2": 1.0}, asked=[])
    assert decision.kind == "question" and decision.question.id == "Q-name" and (decision.question.source == "name")
    assert "고기집" in decision.question.intent_ko and meta["target_unverified"] is True


def test_decide_answers_when_target_business_and_relation_evidenced(fake_judge):
    fake_judge("C1", ids=("E1", "E3"))
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    ev = [_ev("C1", kind="link", eid="E1"), _ev("C1", kind="vision", eid="E2"), _target_hit(), _relation(True)]
    decision, _ = judge.decide(_meat_pr(), cands, ev, {"C1": 3.5, "C2": 1.0}, asked=[])
    assert decision.kind == "answer"


def test_decide_asks_name_when_relation_not_verified(fake_judge):
    fake_judge("C1", ids=("E1", "E3"))
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    ev = [_ev("C1", kind="link", eid="E1"), _ev("C1", kind="vision", eid="E2"), _target_hit(), _relation(False)]
    decision, meta = judge.decide(_meat_pr(), cands, ev, {"C1": 3.5, "C2": 1.0}, asked=[])
    assert decision.kind == "question" and decision.question.id == "Q-name" and (meta["target_unverified"] is True)


def test_named_place_candidate_counts_as_evidenced(fake_judge):
    fake_judge("C1", ids=("E1", "E2"))
    cands = [Candidate(id="C1", type="named_place"), Candidate(id="C2", type="building")]
    ev = [_ev("C1", kind="link", eid="E1"), _ev("C1", kind="vision", eid="E2")]
    decision, _ = judge.decide(_meat_pr(), cands, ev, {"C1": 3.0, "C2": 1.0}, asked=[])
    assert decision.kind == "answer"


def test_name_question_is_not_repeated(fake_judge):
    fake_judge("C1", ids=("E1", "E2"))
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    ev = [_ev("C1", kind="link", eid="E1"), _ev("C1", kind="vision", eid="E2")]
    decision, _ = judge.decide(_meat_pr(), cands, ev, {"C1": 3.0, "C2": 1.0}, asked=["Q-name"])
    assert decision.kind != "question" or decision.question.id != "Q-name"


def test_decide_judge_none_lowers_confidence(fake_judge):
    fake_judge("none", confidence="low")
    cands = [Candidate(id="C1"), Candidate(id="C2")]
    ev = [_ev("C1", eid="E1"), _ev("C1", kind="link", eid="E2")]
    decision, _ = judge.decide(_pr(), cands, ev, {"C1": 3.0, "C2": 1.0}, asked=["a", "b", "c"])
    assert decision.confidence == "low" and decision.candidate_id == "C1"


def test_generic_building_across_a_landmark_is_evidenced_by_relation_alone(fake_judge):
    fake_judge("C1", ids=("E1", "E4"))
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    ev = [_ev("C1", kind="link", eid="E1"), _relation(True)]
    decision, meta = judge.decide(_dept_pr(), cands, ev, {"C1": 3.0, "C2": 1.0}, asked=[])
    assert decision.kind == "answer" and meta.get("target_unverified") is not True


def test_generic_building_still_asks_when_relation_not_verified(fake_judge):
    fake_judge("C1", ids=("E1", "E4"))
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    ev = [_ev("C1", kind="link", eid="E1"), _relation(False)]
    decision, _ = judge.decide(_dept_pr(), cands, ev, {"C1": 3.0, "C2": 1.0}, asked=[])
    assert decision.kind == "question" and decision.question.id == "Q-name"


def test_alley_target_never_asks_for_a_shop_name():
    pr = _pr()
    pr.raw["target"] = {"kind": "alley", "name": "골목"}
    assert judge.target_evidenced(pr, Candidate(id="C1", type="alley"), []) is None


def test_judge_contents_attach_street_photos_with_evidence_labels():
    jpg = b"\xff\xd8" + b"x" * 10
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    ev = [
        Evidence(id="E1", kind="route", candidate_id="C1", supports=True, weight=3.0, summary_ko="경로"),
        Evidence(
            id="E2",
            kind="vision",
            candidate_id="C1",
            supports=True,
            weight=0.5,
            summary_ko="외관",
            capture_date="2018-05",
            image_jpeg=jpg,
        ),
        Evidence(
            id="E3",
            kind="link",
            candidate_id="C1",
            supports=True,
            weight=0.5,
            summary_ko="연결",
            capture_date="2018-05",
            image_jpeg=jpg,
        ),
        Evidence(
            id="E4",
            kind="vision",
            candidate_id="C2",
            supports=False,
            weight=-0.5,
            summary_ko="C2 외관",
            capture_date="2020-09",
            image_jpeg=jpg + b"2",
        ),
        Evidence(id="E5", kind="registry", candidate_id="C1", supports=None, weight=0.0, summary_ko="대장"),
    ]
    parts = judge.judge_contents(_pr(), cands, ev, {"C1": 4.0, "C2": 1.0})
    images = [p for p in parts if isinstance(p, types.Part)]
    labels = [p for p in parts[1:] if isinstance(p, str)]
    assert len(images) == 2 and labels[0].startswith("사진 1: 근거 E2, E3, 후보 C1, 촬영 2018-05") and ("후보 C2" in labels[1])
    assert parts[0].startswith("{") and "E5" in parts[0]


def test_judge_contents_put_the_user_photo_first_with_its_label():
    cands = [Candidate(id="C1", type="building")]
    ev = [Evidence(id="E1", kind="route", candidate_id="C1", supports=True, weight=3.0, summary_ko="경로")]
    parts = judge.judge_contents(_pr(), cands, ev, {"C1": 3.0}, photo=b"\xff\xd8user")
    assert parts[1].startswith("사용자 사진:") and isinstance(parts[2], types.Part) and (len(parts) == 3)
