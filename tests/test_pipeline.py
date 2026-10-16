"""Unit tests for app.pipeline orchestration, progress reporting, evidence spans, and answer flow."""

from __future__ import annotations

from types import SimpleNamespace
import pytest

from app import geo, pipeline, progress
from app.contracts import (
    Anchor,
    Candidate,
    Decision,
    Evidence,
    LatLng,
    ParseResult,
    Place,
    Question,
    VisionStats,
    Window,
)

TEXT = "지하철 2호선 삼성역 3번 출구 하차 후 직진, 오토웨이 타워"


def _raw() -> dict:
    """Return a structured parse dict for the Samsung Station -> Autoway Tower example."""
    return {
        "region": {"area": "삼성동", "query": "삼성역", "kind": "station", "evidence": "삼성역"},
        "anchor": {"query": "삼성역 3번 출구", "kind": "subway_exit", "evidence": "지하철 2호선 삼성역 3번 출구"},
        "steps": [
            {"order": 1, "action": "go_straight", "evidence": "직진", "distance_class": "medium"},
        ],
        "landmarks": [
            {
                "id": "L1",
                "evidence": "오토웨이 타워",
                "visual": {"en": "glass office tower", "floors": 9, "structural": True},
            }
        ],
        "target": {"kind": "named_place", "name": "오토웨이 타워", "specific": True, "landmark_ref": "L1"},
        "ambiguities": [],
    }


def _pr(invented=()) -> ParseResult:
    """Create a ParseResult fixture from _raw()."""
    return ParseResult(
        raw=_raw(),
        source_text=TEXT,
        language="ko",
        invented=list(invented),
        needs_anchor_question=False,
        distance_m={1: 300},
    )


def fakes_found(cand: Candidate | None = None):
    """Build a fake search result namespace containing one or zero candidates."""
    c = cand or Candidate(
        id="C1",
        type="named_place",
        osm_id=475633202,
        name="오토웨이타워",
        geometry={
            "type": "Polygon",
            "coordinates": [
                [
                    [127.0632653, 37.506504],
                    [127.0638001, 37.5066898],
                    [127.0641203, 37.5061097],
                    [127.0635855, 37.5059239],
                    [127.0632653, 37.506504],
                ]
            ],
        },
        centroid=LatLng(37.5063069, 127.0636928),
        path=[LatLng(37.5082553, 127.0631267), LatLng(37.5063069, 127.0636928)],
        route_score=3.0,
    )
    return SimpleNamespace(
        window=Window(center=LatLng(37.5082553, 127.0631267), radius_m=500, source="text", label="삼성역"),
        anchor=Anchor(name="삼성역 3번 출구", at=LatLng(37.5082553, 127.0631267), source="kakao"),
        candidates=[c],
        evidence=[
            Evidence(
                id="E1",
                kind="route",
                candidate_id="C1",
                supports=True,
                weight=3.0,
                summary_ko="삼성역 3번 출구에서 직진 경로 일치",
            )
        ],
        notes=[],
        places=[Place(name="스타벅스", at=LatLng(37.507, 127.063), category="카페")],
        question=None,
    )


@pytest.fixture
def fakes(monkeypatch):
    """Patch network/model boundaries in pipeline with fast deterministic fakes."""
    monkeypatch.setattr(pipeline.parse, "parse_text", lambda text, region=None: _pr())
    monkeypatch.setattr(pipeline.search, "find_candidates", lambda pr, region, current: fakes_found())
    monkeypatch.setattr(pipeline.vision, "registries", lambda cands: {})
    monkeypatch.setattr(
        pipeline.vision,
        "verify",
        lambda *a, **kw: (
            [
                Evidence(
                    id="E2",
                    kind="vision",
                    candidate_id="C1",
                    supports=True,
                    weight=0.5,
                    summary_ko="외관 일치",
                    capture_date="2022-05",
                    image_jpeg=b"\xff\xd8",
                )
            ],
            VisionStats(images=1, first_present=1),
        ),
    )
    monkeypatch.setattr(
        pipeline.judge,
        "decide",
        lambda pr, cands, ev, scores, asked, **kw: (
            Decision(
                kind="answer",
                candidate_id="C1",
                confidence="high",
                reasons=[{"text_ko": "경로 일치", "evidence_ids": ["E1"]}],
            ),
            {"vetoed": [], "unknown_ids": 0, "alternatives": ["C1"]},
        ),
    )
    monkeypatch.setattr(pipeline.judge, "phrase_question", lambda q, lang="ko": q)
    monkeypatch.setattr(
        pipeline,
        "baseline",
        lambda text, region: {
            "model": "3.8 Flash",
            "address": "서울 강남구 영동대로 417",
            "lnglat": [127.062, 37.507],
            "confidence": "medium",
            "distance_to_result_m": None,
        },
    )


def test_evidence_spans_locate_each_kind():
    spans = pipeline.evidence_spans(TEXT, _raw())
    got = {(TEXT[s["start"] : s["end"]], s["kind"]) for s in spans}
    assert {("지하철 2호선 삼성역 3번 출구", "anchor"), ("직진", "step"), ("오토웨이 타워", "target")} <= got


def test_same_range_keeps_most_specific_kind():
    raw = _raw()
    raw["steps"].append({"order": 2, "action": "arrive", "evidence": "오토웨이 타워", "distance_class": "none"})
    spans = [s for s in pipeline.evidence_spans(TEXT, raw) if TEXT[s["start"] : s["end"]] == "오토웨이 타워"]
    assert [s["kind"] for s in spans] == ["target"]


def test_distance_to_polygon_is_zero_inside_and_metres_outside():
    poly = {
        "type": "Polygon",
        "coordinates": [
            [
                [127.0632653, 37.506504],
                [127.0638001, 37.5066898],
                [127.0641203, 37.5061097],
                [127.0635855, 37.5059239],
                [127.0632653, 37.506504],
            ]
        ],
    }
    north = geo.to_ll(0, 150, LatLng(37.5066898, 127.0638001))
    assert pipeline.distance_to_geometry(LatLng(37.50631, 127.063695), poly) == 0
    assert pipeline.distance_to_geometry(north, poly) == pytest.approx(150, abs=3)


def test_safeguards_only_nonzero():
    stats = VisionStats(images=2, first_present=1, flipped=1, skipped_stale=0)
    items = pipeline.safeguards(_pr(invented=["GS25 지나서"]), stats, {"vetoed": [], "unknown_ids": 0})
    assert [(i["kind"], i["count"]) for i in items] == [("invented_removed", 1), ("vision_flipped", 1)]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("오토웨이타워", "오토웨이타워로 보여요."),
        ("삼성빌딩", "삼성빌딩으로 보여요."),
        ("서울", "서울로 보여요."),
    ],
)
def test_tts_text_uses_right_particle(name: str, expected: str):
    d = Decision(
        kind="answer",
        candidate_id="C1",
        confidence="high",
        reasons=[{"text_ko": "대장 층수가 맞음", "evidence_ids": ["E1"]}],
    )
    text = pipeline.tts_text(d, Candidate(id="C1", type="named_place", name=name), "ko")
    assert text.startswith(expected) and text.endswith("확신도는 높음이에요.")


def test_locate_runs_stages_in_order_and_reports_progress(fakes, monkeypatch):
    monkeypatch.setattr(pipeline.vision, "PHOTO_POLICY", "always")
    seen = []
    resp = pipeline.locate(TEXT, on_progress=lambda r: seen.append([p["stage"] for p in r["progress"]]))
    assert [p["stage"] for p in resp["progress"]] == ["parse", "search", "verify", "judge"]
    assert all((p["status"] == "done" for p in resp["progress"]))
    assert seen[0] == ["parse"] and len(seen) >= 8
    assert resp["decision"]["kind"] == "answer" and resp["decision"]["candidate_id"] == "C1"
    assert [e["id"] for e in resp["evidence"]] == ["E1", "E2"] and resp["evidence"][1]["image_b64"] == "/9g="
    assert resp["candidates"][0]["osm_id"] == 475633202 and resp["candidates"][0]["rank"] == 1


def test_locate_search_question_skips_later_stages(fakes, monkeypatch):
    q = Question(id="Q-region", intent_ko="대략 어느 동네(역)예요?", options_ko=[], source="region")
    monkeypatch.setattr(
        pipeline.search,
        "find_candidates",
        lambda pr, region, current: SimpleNamespace(
            window=None, anchor=None, candidates=[], evidence=[], notes=[], places=[], question=q
        ),
    )
    monkeypatch.setattr(pipeline.judge, "phrase_question", lambda q_in, lang="ko": q_in)
    resp = pipeline.locate("2호선 3번 출구에서 직진")
    assert resp["decision"]["kind"] == "question" and resp["decision"]["question"]["source"] == "region"
    assert [p["status"] for p in resp["progress"]] == ["done", "done", "skipped", "skipped"]
    assert resp["baseline"]["lnglat"] == [127.062, 37.507] and resp["baseline"]["distance_to_result_m"] is None


def test_answer_region_question_reruns_with_region(fakes, monkeypatch):
    q = Question(id="Q-region", intent_ko="대략 어느 동네(역)예요?", options_ko=[], source="region")
    calls = []

    def _fake_search(pr, region, current):
        calls.append(region)
        if not region:
            return SimpleNamespace(
                window=None, anchor=None, candidates=[], evidence=[], notes=[], places=[], question=q
            )
        return fakes_found()

    monkeypatch.setattr(pipeline.search, "find_candidates", _fake_search)
    monkeypatch.setattr(pipeline.judge, "phrase_question", lambda q_in, lang="ko": q_in)
    first = pipeline.locate("2호선 3번 출구에서 직진")
    resp = pipeline.answer(first["session_id"], text="삼성동")
    assert calls == [None, "삼성동"] and resp["session_id"] == first["session_id"]
    assert resp["decision"]["kind"] == "answer"


def test_answer_name_question_reruns_with_name_in_text(fakes, monkeypatch):
    q = Question(id="Q-name", intent_ko="그 고기집의 상호(가게 이름)를 아세요?", options_ko=[], source="name")
    texts = []
    meta = {"vetoed": [], "unknown_ids": 0, "alternatives": ["C1"]}

    def _fake_parse(t, region=None):
        texts.append(t)
        return _pr()

    decisions = iter(
        [
            (Decision(kind="question", candidate_id="C1", confidence="low", question=q), meta),
            (Decision(kind="answer", candidate_id="C1", confidence="high", reasons=[]), meta),
        ]
    )
    monkeypatch.setattr(pipeline.parse, "parse_text", _fake_parse)
    monkeypatch.setattr(pipeline.judge, "decide", lambda *a, **kw: next(decisions))
    monkeypatch.setattr(pipeline.judge, "phrase_question", lambda q_in, lang="ko": q_in)

    first = pipeline.locate(TEXT)
    resp = pipeline.answer(first["session_id"], text="명륜진사갈비")
    assert texts[-1] == f"{TEXT} (찾는 곳 이름: 명륜진사갈비)" and resp["decision"]["kind"] == "answer"
    assert resp["session_id"] == first["session_id"]


def test_answer_attribute_question_redecides_without_rerun(fakes, monkeypatch):
    q = Question(
        id="Q-side",
        intent_ko="왼쪽이에요?",
        options_ko=["왼쪽", "오른쪽"],
        source="attribute",
        mapping={"C1": 0},
    )
    decisions = iter(
        [
            (
                Decision(kind="question", candidate_id="C1", confidence="low", question=q),
                {"vetoed": [], "unknown_ids": 0, "alternatives": ["C1"]},
            ),
            (
                Decision(kind="answer", candidate_id="C1", confidence="high", reasons=[]),
                {"vetoed": [], "unknown_ids": 0, "alternatives": ["C1"]},
            ),
        ]
    )
    monkeypatch.setattr(pipeline.judge, "decide", lambda *a, **kw: next(decisions))
    monkeypatch.setattr(pipeline.judge, "phrase_question", lambda q_in, lang="ko": q_in)
    first = pipeline.locate(TEXT)
    searched = []
    monkeypatch.setattr(pipeline.search, "find_candidates", lambda *a, **kw: searched.append(1))
    resp = pipeline.answer(first["session_id"], option_index=0)
    assert first["decision"]["kind"] == "question"
    assert resp["decision"]["kind"] == "answer" and searched == []
    assert resp["candidates"][0]["score"] == pytest.approx(first["candidates"][0]["score"] + 1.0)


def test_progress_note_reaches_the_running_stage(fakes, monkeypatch):
    seen = []

    def _fake_search(pr, region, current):
        progress.note("경로 펼치는 중")
        return fakes_found()

    monkeypatch.setattr(pipeline.search, "find_candidates", _fake_search)
    pipeline.locate(TEXT, on_progress=lambda r: seen.append([(p["stage"], p.get("sub", "")) for p in r["progress"]]))
    assert any((("search", "경로 펼치는 중") in snap for snap in seen))


def test_safeguards_show_the_old_photo_discount():
    items = pipeline.safeguards(_pr(), VisionStats(images=3, old_photos=2), {"vetoed": [], "unknown_ids": 0})
    assert [(i["kind"], i["count"]) for i in items] == [("old_photo_discount", 2)]
    assert "5년 넘음" in items[0]["text_ko"] and "가중치 절반" in items[0]["text_ko"]


def test_locate_skips_photo_check_without_visual_cue_and_with_one_candidate(fakes, monkeypatch):
    plain = _pr()
    plain.raw["landmarks"] = [{"id": "L1", "evidence": "오토웨이 타워", "visual": {"en": "office tower"}}]
    monkeypatch.setattr(pipeline.parse, "parse_text", lambda text, region=None: plain)
    monkeypatch.setattr(
        pipeline.vision,
        "verify",
        lambda *a, **kw: pytest.fail("vision.verify should not run when skipped"),
    )
    resp = pipeline.locate(TEXT)
    verify = next(p for p in resp["progress"] if p["stage"] == "verify")
    assert verify["status"] == "skipped" and "사진 확인 생략" in verify["note"] and ("사용자 사진 없음" in verify["note"])
    assert resp["decision"]["kind"] == "answer" and [e["kind"] for e in resp["evidence"]] == ["route"]


def test_answer_dunno_to_name_question_gives_best_effort_answer(fakes, monkeypatch):
    q = Question(id="Q-name", intent_ko="그 고기집의 상호를 아세요?", options_ko=[], source="name")
    meta = {"vetoed": [], "unknown_ids": 0, "alternatives": ["C1"]}
    seen_asked = []

    def _fake_decide(pr, cands, ev, scores, asked, **kw):
        seen_asked.append(list(asked))
        if not asked:
            return Decision(kind="question", candidate_id="C1", confidence="low", question=q), meta
        return Decision(kind="low_confidence", candidate_id="C1", confidence="low", reasons=[]), meta

    monkeypatch.setattr(pipeline.judge, "decide", _fake_decide)
    monkeypatch.setattr(pipeline.judge, "phrase_question", lambda q_in, lang="ko": q_in)
    first = pipeline.locate(TEXT)
    resp = pipeline.answer(first["session_id"], option_index=-1)
    assert first["decision"]["kind"] == "question"
    assert resp["decision"]["kind"] == "low_confidence" and resp["decision"]["candidate_id"] == "C1"
    assert seen_asked[-1] == ["Q-name"] and resp["progress"][-1]["note"].endswith("모르겠어요")


def test_candidate_json_carries_building_levels_for_3d():
    c = Candidate(id="C1", building_way_id=10)
    md = type("MD", (), {"buildings": {10: pipeline.Building(way_id=10, levels=9)}})()
    assert (
        pipeline._cand_json(c, {"C1": 3.0}, 1, md)["levels"] == 9
        and pipeline._cand_json(Candidate(id="C2"), {}, 2, md)["levels"] is None
    )


def test_feature_chips_name_the_model_that_actually_answered():
    assert pipeline._features("parse", [("parse", "gemini-4-argon")]) == [
        "Gemini 4 Argon",
        "구조화 출력",
    ]
    assert pipeline._features("parse", [("parse", "gemini-3.8-flash")], audio=True) == [
        "3.8 Flash",
        "구조화 출력",
        "오디오 입력",
    ]
    assert pipeline._features(
        "judge",
        [("judge", "gemini-4-argon"), ("question", "gemini-3.5-flash-lite")],
    ) == ["Gemini 4 Argon 판관", "Flash-Lite 질문"]
    assert pipeline._features(
        "verify",
        [("embed", "gemini-embedding-2"), ("vision1", "gemini-3.8-flash")],
    ) == ["Embedding 2", "3.8 Flash 박스"]


def test_photo_only_without_region_asks_q_region_then_resumes_on_answer(fakes, monkeypatch):
    monkeypatch.setattr(
        pipeline.signboard,
        "read",
        lambda photo: {
            "names": ["동촌 아귀찜"],
            "queries": ["동촌"],
            "kind": "식당",
            "traits": "노란 간판",
            "confidence": "high",
        },
    )
    monkeypatch.setattr(pipeline.judge, "phrase_question", lambda q_in, lang="ko": q_in)

    first = pipeline.locate(None, photo=b"\xff\xd8\xff\xd9")
    assert first["decision"]["kind"] == "question"
    assert first["decision"]["question"]["id"] == "Q-region"
    assert [s["status"] for s in first["progress"]] == ["skipped", "skipped", "skipped", "skipped"]
    assert first["progress"][0]["features"] == ["사진 간판 읽기"]

    second = pipeline.answer(first["session_id"], text="송파구")
    assert second["decision"]["kind"] == "answer"
    assert second["input"]["signboard"]["applied"] == "__target__"
    assert second["progress"][0]["features"] == ["사진 간판 읽기"]

