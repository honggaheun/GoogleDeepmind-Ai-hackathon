"""Unit tests for visual verification policy, registry lookup, and 2-pass street-view verification."""

from __future__ import annotations

import datetime

import pytest

from app import config, geo, mapdata, vision
from app.contracts import (
    Building,
    Candidate,
    Evidence,
    LatLng,
    Pano,
    ParseResult,
    RegistryRecord,
    ViewPoint,
)

GATE = {
    "id": "L1",
    "evidence": "파란 대문 집",
    "side": "left",
    "visual": {"en": "house with a blue gate", "colors": ["blue"], "structural": True},
}
TREE = {
    "id": "L2",
    "evidence": "큰 은행나무 지나서",
    "side": "unknown",
    "visual": {"en": "big ginkgo tree", "structural": True},
}


def _ll(x: float, y: float) -> LatLng:
    """Convert tangent-plane meters around config.VENUE to LatLng."""
    return geo.to_ll(x, y, config.VENUE)


def _rect(way_id: int, x0: float, x1: float, y0: float, y1: float) -> Building:
    """Construct a rectangular Building in local meter coordinates."""
    ring = [_ll(x0, y0), _ll(x1, y0), _ll(x1, y1), _ll(x0, y1), _ll(x0, y0)]
    centroid = _ll((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return Building(way_id=way_id, ring=ring, centroid=centroid)


def _pr(target: dict, landmarks: list[dict]) -> ParseResult:
    """Create a ParseResult with the given target and landmarks."""
    return ParseResult(
        raw={"target": target, "landmarks": landmarks, "steps": [], "ambiguities": []},
        source_text="파란 대문 집",
        language="ko",
    )


def _pr_with(target: dict, landmarks: list[dict]) -> ParseResult:
    """Create a minimal ParseResult for should_verify policy tests."""
    return _pr(target, landmarks)


def _ev(cid: str, attr: str, vp: int, weight: float) -> Evidence:
    """Create a vision Evidence item for consistency tests."""
    supports = True if weight > 0 else (False if weight < 0 else None)
    return Evidence(
        id=f"V-{cid}-{vp}",
        kind="vision",
        candidate_id=cid,
        supports=supports,
        weight=weight,
        summary_ko=attr,
        data={"attribute": attr, "vp": vp},
    )


@pytest.fixture
def scene():
    """Build a test scene with a north-south alley at x=150 and east building 10."""
    nodes = {1: _ll(150, -85), 2: _ll(150, 0)}
    edges = [(1, 2, "service", 100)]
    b10 = _rect(10, 162, 182, -55, -30)
    md = mapdata.MapData.from_parts(config.VENUE, nodes, edges, [b10])
    vp = ViewPoint(at=_ll(150, -42.5), heading_deg=90.0, purpose="target")
    cand = Candidate(
        id="C1",
        type="building",
        osm_id=10,
        centroid=b10.centroid,
        geometry={
            "type": "Polygon",
            "coordinates": [[[p.lng, p.lat] for p in b10.ring]],
        },
        building_way_id=10,
        view_points=[vp],
        route_score=3.0,
    )
    pr = _pr({"kind": "building", "landmark_ref": "L1"}, [GATE, TREE])
    return md, cand, pr


def test_box_bearing_center_box_keeps_heading():
    assert vision.box_bearing(90, [100, 400, 300, 600]) == pytest.approx(90.0, abs=0.1)


def test_stale_photo_skips_visual_check(monkeypatch, scene):
    md, cand, pr = scene
    monkeypatch.setattr(
        vision.streetview,
        "metadata",
        lambda at, radius_m=50: Pano(at=at, date="2016-06"),
    )
    monkeypatch.setattr(
        vision.streetview,
        "image",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("image should not be fetched")),
    )
    evidence, stats = vision.verify([cand], pr, md, {"C1": RegistryRecord(use_apr_day="20170510")})
    assert [e.kind for e in evidence] == ["staleness"] and stats.skipped_stale == 1


def test_present_with_bad_box_becomes_uncertain():
    cleaned = vision.clean_attributes(
        {
            "attributes": [{"name": "L1", "status": "present", "box_2d": [0, 0, 1200, 50]}],
            "buildings": [],
        }
    )
    assert cleaned["L1"]["status"] == "uncertain"


@pytest.mark.parametrize(
    "first,second,structural,sign,old,expected",
    [
        ("present", "yes", True, False, False, 1.0),
        ("present", "yes", False, True, False, 0.5),
        ("present", "yes", False, True, True, 0.25),
        ("present", "uncertain", True, False, False, 0.3),
        ("present", "no", True, False, False, -0.5),
        ("present", "no", False, True, True, -0.25),
        ("absent", None, True, False, False, 0.0),
    ],
)
def test_attribute_weight(first, second, structural, sign, old, expected):
    assert vision.attribute_weight(first, second, structural, sign, old) == pytest.approx(expected)


def test_attributes_for_checks_target_landmark_and_floors_not_route_landmarks(scene):
    _, cand, pr = scene
    names = [a["name"] for a in vision.attributes_for(pr, cand)]
    assert names == ["L1", "target_floors"]


def test_verify_confirms_structural_attribute_links_building_and_floors(monkeypatch, scene):
    md, cand, pr = scene
    monkeypatch.setattr(
        vision.streetview,
        "metadata",
        lambda at, radius_m=50: Pano(at=at, date="2024-05"),
    )
    monkeypatch.setattr(
        vision.streetview,
        "image",
        lambda at, heading_deg, fov_deg=90, pitch=10: b"\xff\xd8jpeg",
    )
    monkeypatch.setattr(vision.llm, "embed", lambda item, mime=None: [1.0, 0.0])

    def fake_gen(stage, model, contents, schema, **kw):
        if stage == "vision_1":
            return {
                "attributes": [
                    {"name": "L1", "status": "present", "box_2d": [200, 400, 600, 600]}
                ],
                "buildings": [{"box_2d": [100, 350, 700, 650], "floors_estimate": 3}],
            }
        return {"answer": "yes"}

    monkeypatch.setattr(vision.llm, "generate_json", fake_gen)
    evidence, stats = vision.verify(
        [cand], pr, md, {"C1": RegistryRecord(grnd_floors=3, use_apr_day="20100101")}
    )
    by_kind = {(e.kind, e.data.get("attribute")): e for e in evidence}
    assert (
        by_kind["vision", "L1"].weight == pytest.approx(1.0)
        and by_kind["vision", "L1"].capture_date == "2024-05"
    )
    assert by_kind["link", "L1"].weight == pytest.approx(0.5)
    assert by_kind["vision", "target_floors"].weight == pytest.approx(0.5)
    assert (stats.images, stats.first_present, stats.flipped) == (2, 1, 0)


def test_gemini_timeout_becomes_failure_evidence_not_crash(monkeypatch, scene):
    md, cand, pr = scene
    monkeypatch.setattr(
        vision.streetview,
        "metadata",
        lambda at, radius_m=50: Pano(at=at, date="2024-05"),
    )
    monkeypatch.setattr(
        vision.streetview,
        "image",
        lambda at, heading_deg, fov_deg=90, pitch=10: b"\xff\xd8jpeg",
    )
    monkeypatch.setattr(vision.llm, "embed", lambda item, mime=None: [1.0, 0.0])

    def fail_gen(*a, **kw):
        raise vision.llm.LLMError("timeout")

    monkeypatch.setattr(vision.llm, "generate_json", fail_gen)
    evidence, _ = vision.verify([cand], pr, md, {"C1": None})
    assert any("1차 판독 실패" in e.summary_ko and e.supports is None for e in evidence)


def test_consistency_two_views_confirmed_adds_bonus():
    evs = [_ev("C1", "L1", 0, 1.0), _ev("C1", "L1", 1, 1.0)]
    vision.apply_consistency(evs, {("C1", "L1", 0): "yes", ("C1", "L1", 1): "yes"})
    assert sum(e.weight for e in evs) == pytest.approx(2.3)


def test_consistency_yes_and_no_cancel_to_zero():
    evs = [_ev("C1", "L1", 0, 1.0), _ev("C1", "L1", 1, -0.5)]
    vision.apply_consistency(evs, {("C1", "L1", 0): "yes", ("C1", "L1", 1): "no"})
    assert [e.weight for e in evs] == [0.0, 0.0] and all(e.supports is None for e in evs)


def test_user_photo_similarity_marks_the_closest_candidate(monkeypatch, scene):
    md, cand, pr = scene
    other = Candidate(
        id="C2",
        type="building",
        osm_id=cand.osm_id,
        centroid=cand.centroid,
        geometry=cand.geometry,
        building_way_id=cand.building_way_id,
        view_points=cand.view_points,
        route_score=2.8,
    )
    counter = {"n": 0}
    monkeypatch.setattr(
        vision.streetview,
        "metadata",
        lambda at, radius_m=50: Pano(at=at, date="2024-05"),
    )
    monkeypatch.setattr(
        vision.streetview,
        "image",
        lambda at, heading_deg, fov_deg=90, pitch=10: b"\xff\xd8jpeg",
    )

    def fake_embed(item, mime=None):
        if isinstance(item, str) or item == b"user":
            return [1.0, 0.0]
        counter["n"] += 1
        return [1.0, 0.0] if counter["n"] == 1 else [0.5, 0.5]

    monkeypatch.setattr(vision.llm, "embed", fake_embed)
    monkeypatch.setattr(
        vision.llm,
        "generate_json",
        lambda *a, **kw: {"attributes": [], "buildings": []},
    )
    evidence, _ = vision.verify([cand, other], pr, md, {}, photo=b"user")
    up = {e.candidate_id: e for e in evidence if e.kind == "user_photo"}
    assert (
        up["C1"].supports is True
        and up["C1"].weight == 0.5
        and (up["C1"].data["similarity"] > up["C2"].data["similarity"])
    )
    assert up["C2"].supports is None and "닮음" in up["C2"].summary_ko


def test_should_verify_default_policy_is_user_photo_only(monkeypatch):
    monkeypatch.setattr(vision, "PHOTO_POLICY", "user_photo")
    cue = _pr_with(
        {"kind": "building", "landmark_ref": "L1"},
        [
            {
                "id": "L1",
                "evidence": "파란 대문 집",
                "visual": {"en": "house with a blue gate", "structural": True},
            }
        ],
    )
    ok, why = vision.should_verify(cue, [Candidate(id="C1", route_score=3.0)])
    assert ok is False and why.startswith("사용자 사진 없음")
    assert vision.should_verify(
        cue, [Candidate(id="C1", route_score=3.0)], photo=b"user"
    ) == (True, "사용자 사진 대조")


def test_should_verify_only_when_photos_can_discriminate(monkeypatch):
    monkeypatch.setattr(vision, "PHOTO_POLICY", "cues")
    one = [Candidate(id="C1", route_score=3.0)]
    close = [Candidate(id="C1", route_score=3.0), Candidate(id="C2", route_score=2.5)]
    far = [Candidate(id="C1", route_score=3.0), Candidate(id="C2", route_score=1.0)]

    named = _pr_with({"kind": "named_place", "name": "오토웨이타워", "specific": True}, [])
    cue = _pr_with(
        {"kind": "building", "landmark_ref": "L1"},
        [
            {
                "id": "L1",
                "evidence": "파란 대문 집",
                "visual": {"en": "house with a blue gate", "structural": True},
            }
        ],
    )
    nocue = _pr_with(
        {"kind": "building", "landmark_ref": "L1"},
        [{"id": "L1", "evidence": "건물", "visual": {"en": "building"}}],
    )
    sign_only = _pr_with(
        {"kind": "named_place", "landmark_ref": "L1"},
        [
            {
                "id": "L1",
                "evidence": "김밥천국",
                "visual": {"en": "restaurant", "sign_text": "김밥천국"},
            }
        ],
    )

    assert vision.should_verify(named, one) == (False, "시각 단서 없음 · 후보 1곳")
    assert vision.should_verify(named, close)[0] is True
    assert (
        vision.should_verify(named, far)[0] is False
        and "점수 차 충분" in vision.should_verify(named, far)[1]
    )
    assert vision.should_verify(named, one, photo=b"user") == (True, "사용자 사진 대조")
    assert vision.should_verify(cue, one) == (True, "시각 단서 1개")
    assert vision.should_verify(nocue, one)[0] is False
    assert vision.should_verify(sign_only, one)[0] is False


def test_user_photo_skips_street_photos_older_than_five_years(monkeypatch, scene):
    md, cand, pr = scene
    monkeypatch.setattr(
        vision.streetview,
        "metadata",
        lambda at, radius_m=50: Pano(at=at, date="2018-04"),
    )
    monkeypatch.setattr(
        vision.streetview,
        "image",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not fetch image")),
    )
    monkeypatch.setattr(
        vision,
        "_select",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not select views")),
    )
    ev, st = vision.verify([cand], pr, md, {}, photo=b"user")
    assert st.skipped_stale == 1 and st.images == 0
    assert any(e.kind == "staleness" and "사용자 사진" in e.summary_ko for e in ev)


def test_photo_age_is_counted_from_today(monkeypatch, scene):
    md, cand, pr = scene
    today = datetime.date.today()
    six_years_ago = f"{today.year - 6:04d}-{today.month:02d}"
    monkeypatch.setattr(
        vision.streetview,
        "metadata",
        lambda at, radius_m=50: Pano(at=at, date=six_years_ago),
    )
    monkeypatch.setattr(
        vision.streetview,
        "image",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not fetch image")),
    )
    ev, st = vision.verify([cand], pr, md, {}, photo=b"user")
    assert st.skipped_stale == 1 and any(e.kind == "staleness" for e in ev)
