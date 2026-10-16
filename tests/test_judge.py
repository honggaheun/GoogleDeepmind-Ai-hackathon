"""Unit tests for app.judge scoring, registry evidence, confidence, question selection, and validation."""

from __future__ import annotations

from google.genai import types
import pytest

from app import config, geo, judge
from app.contracts import Candidate, Evidence, LatLng, ParseResult, RegistryRecord

VILLA = {
    "id": "L1",
    "evidence": "3층짜리 빨간 벽돌 빌라",
    "side": "left",
    "visual": {"en": "three-story red brick villa", "floors": 3, "structural": True},
}


def _ll(x: float, y: float) -> LatLng:
    """Convert planar meter offsets (x=east, y=north) from VENUE into a LatLng."""
    return geo.to_ll(x, y, config.VENUE)


def _pr(landmarks=(), target=None, steps=(), ambiguities=()) -> ParseResult:
    """Build a ParseResult test fixture."""
    raw = {
        "landmarks": list(landmarks),
        "target": target or {"kind": "building", "landmark_ref": "L1"},
        "steps": list(steps),
        "ambiguities": list(ambiguities),
    }
    return ParseResult(
        raw=raw,
        source_text="테스트 안내",
        language="ko",
        invented=[],
        needs_anchor_question=False,
        distance_m={},
    )


def _ev(cid: str, kind: str = "vision", supports: bool = True, weight: float = 0.5, data: dict | None = None) -> Evidence:
    """Create a test Evidence object."""
    return Evidence(
        id="E1",
        kind=kind,
        candidate_id=cid,
        supports=supports,
        weight=weight,
        summary_ko=f"{cid} {kind}",
        data=data or {},
    )


def _villa_pr() -> ParseResult:
    """Build a ParseResult describing a 3-story red brick villa target."""
    return _pr(landmarks=[VILLA], target={"kind": "building", "landmark_ref": "L1"})


def _three_buildings() -> tuple[list[Candidate], list[Evidence]]:
    """Build 3 building candidates along a northward path (C1 west/left, C2 & C3 east/right; floors 3, 4, 5)."""
    path = [_ll(0, 0), _ll(0, 100)]
    cands = [
        Candidate(id="C1", type="building", centroid=_ll(-20, 100), path=path, route_score=1.0),
        Candidate(id="C2", type="building", centroid=_ll(20, 100), path=path, route_score=1.0),
        Candidate(id="C3", type="building", centroid=_ll(25, 100), path=path, route_score=1.0),
    ]
    ev = [
        Evidence(id="E1", kind="registry", candidate_id="C1", supports=None, weight=0.0, data={"grnd_floors": 3}),
        Evidence(id="E2", kind="registry", candidate_id="C2", supports=None, weight=0.0, data={"grnd_floors": 4}),
        Evidence(id="E3", kind="registry", candidate_id="C3", supports=None, weight=0.0, data={"grnd_floors": 5}),
    ]
    return cands, ev


def _e(
    eid: str,
    cid: str,
    kind: str,
    supports: bool = True,
    weight: float = 0.5,
    data: dict | None = None,
    text: str = "",
) -> Evidence:
    """Build an Evidence item with an explicit id and summary text."""
    return Evidence(
        id=eid,
        kind=kind,
        candidate_id=cid,
        supports=supports,
        weight=weight,
        summary_ko=text or f"{cid} {kind}",
        data=data or {},
    )


@pytest.mark.parametrize("floors,expected", [(3, 1.0), (4, 0.0), (5, -1.5)])
def test_registry_floors_rule(floors: int, expected: float):
    rec = RegistryRecord(grnd_floors=floors, main_purpose="업무시설")
    ev = judge.registry_evidence(Candidate(id="C1", type="building"), rec, _villa_pr())
    assert ev.kind == "registry" and ev.weight == pytest.approx(expected)


def test_registry_villa_matches_residential_purpose():
    rec = RegistryRecord(grnd_floors=3, main_purpose="공동주택")
    assert judge.registry_evidence(Candidate(id="C1", type="building"), rec, _villa_pr()).weight == pytest.approx(1.5)


def test_registry_none_without_record():
    assert judge.registry_evidence(Candidate(id="C1", type="building"), None, _villa_pr()) is None


def test_code_scores_adds_evidence_but_not_route_twice():
    cands = [Candidate(id="C1", route_score=2.0), Candidate(id="C2", route_score=1.0)]
    ev = [
        _ev("C1", kind="route", weight=2.0),
        _ev("C1", weight=0.5),
        _ev("C2", kind="registry", weight=-1.5),
    ]
    assert judge.code_scores(cands, ev) == {"C1": 2.5, "C2": -0.5}


def test_confidence_high():
    cands = [Candidate(id="C1"), Candidate(id="C2")]
    ev = [_ev("C1"), _ev("C1", kind="link")]
    assert judge.confidence({"C1": 3.0, "C2": 1.0}, cands, ev) == "high"


def test_confidence_low_when_close():
    assert judge.confidence({"C1": 2.0, "C2": 1.8}, [Candidate(id="C1"), Candidate(id="C2")], []) == "low"


def test_confidence_route_evidence_alone_is_not_support():
    cands = [Candidate(id="C1"), Candidate(id="C2")]
    ev = [_ev("C1", kind="route"), _ev("C1")]
    assert judge.confidence({"C1": 3.0, "C2": 1.0}, cands, ev) == "medium"


def test_veto_on_floor_mismatch():
    ev = [_ev("C1", kind="registry", supports=False, weight=-1.5)]
    assert judge.veto("C1", ev) is True and judge.veto("C2", ev) is False


def test_side_question_follows_walking_direction_not_compass():
    path = [_ll(0, 100), _ll(0, 0)]
    cands = [
        Candidate(id="C1", type="building", centroid=_ll(-20, 0), path=path, route_score=1.0),
        Candidate(id="C2", type="building", centroid=_ll(20, 0), path=path, route_score=1.0),
    ]
    side = judge.pick_question(_pr(), cands, [], {"C1": 1.0, "C2": 1.0}, asked=[])
    assert side.id == "Q-side" and side.options_ko == ["왼쪽", "오른쪽"]
    assert side.mapping == {"C1": 1, "C2": 0}


def test_pick_question_prefers_higher_information_gain():
    cands, ev = _three_buildings()
    scores = {c.id: 1.0 for c in cands}
    q = judge.pick_question(_pr(), cands, ev, scores, asked=[])
    side = judge.pick_question(_pr(), cands, [], scores, asked=[])
    assert q.id == "Q-floors" and q.mapping == {"C1": 0, "C2": 1, "C3": 2}
    assert side.id == "Q-side" and judge.info_gain(side, scores) == pytest.approx(0.918, abs=0.001)
    assert judge.info_gain(q, scores) == pytest.approx(1.585, abs=0.001)


def test_relaxation_question_uses_spoken_ordinal():
    steps = [{"order": 1, "action": "turn_right", "turn_target": {"type": "alley", "ordinal": 2}}]
    cands = [
        Candidate(id="C1", relaxations=["ordinal-1"], route_score=1.0),
        Candidate(id="C2", route_score=1.0),
    ]
    q = judge.pick_question(_pr(steps=steps), cands, [], {"C1": 1.0, "C2": 1.0}, asked=[])
    assert q.source == "relaxation" and "2번째" in q.intent_ko and ("1번째" in q.intent_ko)
    assert q.mapping == {"C1": 0, "C2": 1}


def test_apply_answer_rewards_matching_option():
    cands, ev = _three_buildings()
    q = judge.pick_question(_pr(), cands, ev, {c.id: 1.0 for c in cands}, asked=[])
    assert judge.apply_answer(q, 0, cands, {"C1": 1.0, "C2": 1.0, "C3": 1.0}) == {
        "C1": 2.0,
        "C2": 0.0,
        "C3": 0.0,
    }


def test_apply_answer_unknown_keeps_scores():
    cands, ev = _three_buildings()
    q = judge.pick_question(_pr(), cands, ev, {c.id: 1.0 for c in cands}, asked=[])
    assert judge.apply_answer(q, -1, cands, {"C1": 1.0, "C2": 1.0, "C3": 1.0}) == {
        "C1": 1.0,
        "C2": 1.0,
        "C3": 1.0,
    }


def test_registry_name_match_supports_a_named_target():
    pr = ParseResult(
        raw={
            "target": {"kind": "named_place", "name": "오토웨이 타워", "specific": True, "landmark_ref": "L1"},
            "landmarks": [{"id": "L1", "evidence": "오토웨이 타워", "visual": {"en": "office tower"}}],
            "steps": [],
        },
        source_text="오토웨이 타워",
        language="ko",
        invented=[],
        needs_anchor_question=False,
    )
    cand = Candidate(id="C1", type="named_place", name="오토웨이타워")
    e = judge.registry_evidence(cand, RegistryRecord(name="오토웨이타워", grnd_floors=9), pr)
    other = judge.registry_evidence(cand, RegistryRecord(name="현대타워", grnd_floors=9), pr)
    assert e.supports is True and e.weight == 1.0 and (e.data["name_match"] is True) and ("= 목표 이름" in e.summary_ko)
    assert other.supports is None and other.weight == 0.0 and (other.data["name_match"] is False)


def test_confidence_high_with_one_decisive_evidence():
    cands = [Candidate(id="C1"), Candidate(id="C2")]
    route = Evidence(id="E1", kind="route", candidate_id="C1", supports=True, weight=3.0, summary_ko="경로")
    reg = Evidence(
        id="E2",
        kind="registry",
        candidate_id="C1",
        supports=True,
        weight=1.0,
        summary_ko="이름 일치",
        data={"name_match": True},
    )
    plain = Evidence(id="E2", kind="vision", candidate_id="C1", supports=True, weight=0.5, summary_ko="외관")
    assert judge.confidence({"C1": 4.0, "C2": 1.0}, cands, [route, reg]) == "high"
    assert judge.confidence({"C1": 4.0, "C2": 1.0}, cands, [route, plain]) == "medium"


def test_registry_vetoes_non_shop_purpose_for_business_target():
    pr = ParseResult(
        raw={
            "target": {
                "kind": "building",
                "name": "고기집",
                "specific": False,
                "category_code": "FD6",
                "landmark_ref": "L1",
            },
            "landmarks": [{"id": "L1", "evidence": "돈가스집"}],
            "steps": [],
        },
        source_text="고기집",
        language="ko",
        invented=[],
        needs_anchor_question=False,
    )
    e = judge.registry_evidence(
        Candidate(id="C1", type="building"),
        RegistryRecord(name="한림연예예술고등학교", main_purpose="교육연구시설"),
        pr,
    )
    ok = judge.registry_evidence(
        Candidate(id="C2", type="building"),
        RegistryRecord(name="상가", main_purpose="제2종근린생활시설"),
        pr,
    )
    assert e.weight <= -1.5 and judge.veto("C1", [e]) and ("가게가 있을 건물이 아님" in e.summary_ko)
    assert ok.weight == 0.0


def test_confirmed_relation_neutralizes_the_landmark_near_end_relaxation():
    c = Candidate(id="C1", route_score=3.9, relaxations=["landmark_near_end"])
    ev = [_ev("C1", kind="relation", data={"confirmed": True}), _ev("C1", kind="kakao", data={"target": True})]
    assert judge.confidence({"C1": 3.9}, [c], ev) == "high"
    assert judge.confidence({"C1": 3.9}, [c], [_ev("C1", kind="kakao", data={"target": True})]) == "high"
    assert judge.confidence({"C1": 3.9}, [c], [_ev("C1", kind="kakao", data={}), _ev("C1", kind="link")]) == "medium"


def test_registry_blank_name_never_matches_the_target():
    pr = ParseResult(
        raw={"target": {"kind": "named_place", "name": "동촌아귀찜", "specific": True}, "landmarks": [], "steps": []},
        source_text="x",
        language="ko",
        invented=[],
        needs_anchor_question=False,
    )
    e = judge.registry_evidence(
        Candidate(id="C1", type="named_place", name="동촌 본점"),
        RegistryRecord(name=" ", grnd_floors=1),
        pr,
    )
    assert e.data["name_match"] is False and e.weight == 0.0


def test_judge_contents_labels_the_user_photo_by_its_role():
    pr = ParseResult(
        raw={
            "target": {
                "kind": "building",
                "name": "고기집",
                "specific": False,
                "landmark_ref": "L1",
                "relation": "across",
            },
            "landmarks": [{"id": "L1", "evidence": "돈가스집"}],
        },
        source_text="x",
        language="ko",
        invented=[],
        needs_anchor_question=False,
    )
    cands = [Candidate(id="C1", type="building", route_score=3.9)]
    ev_item = _ev("", kind="user_photo", data={"place": "동촌 본점", "signboard": "동촌", "target": False})
    ev_item.id = "E4"
    ev = [ev_item]
    parts_lm = judge.judge_contents(pr, cands, ev, {"C1": 3.9}, photo=b"\xff\xd8user", photo_role="landmark")
    label = next(t for t in parts_lm if isinstance(t, str) and t.startswith("사용자 사진:"))
    assert "지나는 가게" in label and "동촌 본점" in label and ("E4" in label) and ("찾는 곳이 아니다" in label)

    parts_tg = judge.judge_contents(pr, cands, ev, {"C1": 3.9}, photo=b"\xff\xd8user", photo_role="target")
    texts = [t for t in parts_tg if isinstance(t, str)]
    assert "찾는 곳" in next((t for t in texts if t.startswith("사용자 사진:")))
    assert any(isinstance(p, types.Part) for p in parts_tg)


def test_decide_never_answers_confidently_when_every_candidate_is_vetoed():
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    ev = [
        _e("E1", "C1", "registry", supports=False, weight=-1.5),
        _e("E2", "C2", "registry", supports=False, weight=-1.5),
        _e("E3", "C1", "kakao", data={"target": True}),
    ]
    d, meta = judge.decide(_pr(), cands, ev, {"C1": 3.0, "C2": 1.0}, asked=["Q-a", "Q-b", "Q-c"], use_gemini=False)
    assert meta["vetoed"] == ["C1", "C2"] and meta["all_vetoed"] is True
    assert d.kind == "low_confidence" and d.confidence == "low" and (d.candidate_id == "C1")


def test_decide_drops_judge_reasons_written_for_a_vetoed_choice(monkeypatch):
    cands = [Candidate(id="C1", type="building"), Candidate(id="C2", type="building")]
    ev = [
        _e("E1", "C1", "kakao", data={"target": True}, text="C1 가게 자료 일치"),
        _e("E2", "C1", "link", text="C1 건물 연결"),
        _e("E3", "C2", "registry", supports=False, weight=-1.5, text="C2 대장 층수 불일치"),
    ]
    monkeypatch.setattr(
        judge,
        "judge_with_gemini",
        lambda *a, **kw: {
            "choice": "C2",
            "confidence": "high",
            "reasons": [{"text_ko": "C2가 가장 적합합니다.", "evidence_ids": ["E3"]}],
        },
    )
    d, meta = judge.decide(_pr(), cands, ev, {"C1": 3.0, "C2": 1.0}, asked=[], use_gemini=True)
    assert d.candidate_id == "C1" and meta["judge_vetoed"] == "C2"
    assert d.reasons and all(("C2가" not in r["text_ko"] for r in d.reasons))
    assert all((set(r["evidence_ids"]) <= {"E1", "E2"} for r in d.reasons))


def test_validate_judge_keeps_only_reasons_backed_by_the_chosen_candidate():
    out = {
        "choice": "C1",
        "confidence": "high",
        "reasons": [
            {"text_ko": "C1 가게 자료", "evidence_ids": ["E1"]},
            {"text_ko": "C2는 층수가 다르다", "evidence_ids": ["E3"]},
            {"text_ko": "기준 가게 간판", "evidence_ids": ["E9"]},
        ],
    }
    owners = {"E1": "C1", "E2": "C1", "E3": "C2", "E9": ""}
    res = judge.validate_judge(out, set(owners), {"C1", "C2"}, "C1", owners=owners)
    assert [r["text_ko"] for r in res["reasons"]] == ["C1 가게 자료", "기준 가게 간판"] and res["foreign_ids"] == 1
    assert res["choice"] == "C1" and (not res.get("fallback"))


def test_validate_judge_falls_back_when_every_reason_is_about_other_candidates():
    out = {"choice": "C2", "confidence": "high", "reasons": [{"text_ko": "C1 가게 자료", "evidence_ids": ["E1"]}]}
    owners = {"E1": "C1", "E3": "C2"}
    res = judge.validate_judge(out, set(owners), {"C1", "C2"}, "C1", owners=owners)
    assert res["choice"] == "C1" and res["fallback"] is True and (res["reasons"] == [])


def test_shared_target_evidence_is_not_decisive():
    cands = [Candidate(id="C1"), Candidate(id="C2")]
    t1 = _ev("C1", kind="kakao", data={"target": True, "place": "커리A"})
    t2 = _ev("C2", kind="kakao", data={"target": True, "place": "커리B"})
    reg = _ev("C1", kind="registry", data={"name_match": True})
    assert judge.confidence({"C1": 4.0, "C2": 2.5}, cands, [t1, t2]) == "medium"
    assert judge.confidence({"C1": 4.0, "C2": 2.5}, cands, [t1]) == "high"
    assert judge.confidence({"C1": 4.0, "C2": 2.5}, cands, [t1, t2, reg]) == "high"
