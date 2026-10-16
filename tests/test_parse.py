"""Unit tests for route description parsing validation and grounding."""

from __future__ import annotations

import pytest

from app import parse


def _raw(
    anchor_evidence: str = "",
    anchor_kind: str = "none",
    steps: list | None = None,
    landmarks: list | None = None,
    target_kind: str = "building",
) -> dict:
    """Build a minimal raw parser output dictionary for validation tests."""
    return {
        "language": "ko",
        "transcript": "",
        "region": {"evidence": "", "query": "", "kind": "none"},
        "anchor": {
            "evidence": anchor_evidence,
            "query": anchor_evidence,
            "kind": anchor_kind,
        },
        "steps": steps or [],
        "landmarks": landmarks or [],
        "target": {"kind": target_kind},
        "uncertainty": {"hedges": [], "staleness": "unknown"},
        "ambiguities": [],
    }


@pytest.mark.parametrize(
    "cls,minutes,expected",
    [
        ("short", None, (30, 150)),
        ("minutes", 3, (180, 270)),
        ("none", None, (0, 600)),
    ],
)
def test_distance_range(cls, minutes, expected):
    assert parse.distance_range(cls, minutes) == expected


def test_anchor_inside_source_is_kept():
    text = "강남역 11번 출구로 나와서 직진하면 파란 건물"
    pr = parse.validate(_raw("강남역 11번 출구로 나와서", "station_exit"), text)
    assert pr.raw["anchor"]["kind"] == "station_exit" and pr.invented == []


def test_landmark_not_in_source_is_dropped_and_counted():
    text = "강남역 11번 출구로 나와서 직진하면 파란 건물"
    landmark = {
        "id": "L1",
        "evidence": "GS25 지나서",
        "side": "unknown",
        "visual": {"en": "GS25 sign"},
    }
    pr = parse.validate(
        _raw("강남역 11번 출구로 나와서", "station_exit", landmarks=[landmark]),
        text,
    )
    assert pr.raw["landmarks"] == [] and pr.invented == ["GS25 지나서"]


def test_missing_anchor_needs_question_for_building_target():
    pr = parse.validate(_raw(target_kind="building"), "파란 대문 3층 빌라")
    assert pr.needs_anchor_question is True


def test_missing_anchor_is_fine_for_named_place():
    pr = parse.validate(_raw(target_kind="named_place"), "오토웨이 타워")
    assert pr.needs_anchor_question is False


def test_region_evidence_from_region_hint_is_not_invented():
    raw = _raw()
    raw["region"] = {"evidence": "삼성동", "query": "삼성동", "kind": "neighborhood"}
    pr = parse.validate(raw, "2호선 3번 출구에서 직진", region_hint="삼성동")
    assert pr.raw["region"]["kind"] == "neighborhood" and pr.invented == []


def test_region_evidence_outside_text_and_hint_is_invented():
    raw = _raw()
    raw["region"] = {"evidence": "송파구", "query": "송파구", "kind": "district"}
    pr = parse.validate(raw, "2호선 3번 출구에서 직진", region_hint="삼성동")
    assert pr.raw["region"]["kind"] == "none" and pr.invented == ["송파구"]


def test_whitespace_is_normalized_when_matching_evidence():
    step = {
        "order": 1,
        "action": "go_straight",
        "evidence": "쭉 직진",
        "distance_class": "far_walk",
    }
    pr = parse.validate(_raw(steps=[step]), "쭉  직진")
    assert len(pr.raw["steps"]) == 1 and pr.invented == []
    assert pr.distance_m[1] == (100, 600)


def test_validate_keeps_step_toward_only_when_that_landmark_exists():
    raw = _raw(
        anchor_evidence="신촌 유플렉스 앞에서",
        anchor_kind="place",
        steps=[
            {
                "order": 1,
                "action": "go_straight",
                "evidence": "연세대 방향으로 가다가",
                "distance_class": "none",
                "toward": "L1",
            },
            {
                "order": 2,
                "action": "go_straight",
                "evidence": "쭉 가면",
                "distance_class": "far_walk",
                "toward": "L99",
            },
        ],
        landmarks=[
            {
                "id": "L1",
                "evidence": "연세대 방향으로 가다가",
                "side": "unknown",
                "poi": {"name": "연세대학교", "category_code": "SC4", "specific": True},
                "visual": {"en": "university"},
            }
        ],
    )
    pr = parse.validate(raw, "신촌 유플렉스 앞에서 연세대 방향으로 가다가 쭉 가면 파리바게뜨")
    assert pr.raw["steps"][0].get("toward") == "L1" and "toward" not in pr.raw["steps"][1]


def test_normalize_target_promotes_specific_landmark_front():
    text = "사람이 쓰러졌어요! 여기 강남역 11번 출구로 나와서 직진하면 왼쪽에 교보문고 건물 있잖아요, 그 건물 앞이에요"
    raw = _raw(
        anchor_evidence="강남역 11번 출구로 나와서",
        anchor_kind="station_exit",
        landmarks=[
            {
                "id": "L1",
                "evidence": "왼쪽에 교보문고 건물",
                "side": "left",
                "poi": {"name": "교보문고", "category_code": "CT1", "specific": True},
                "visual": {"en": "Kyobo Book Centre building"},
            }
        ],
        target_kind="point",
    )
    raw["target"] = {"kind": "point", "landmark_ref": "L1", "relation": "at"}
    pr = parse.validate(raw, text)
    assert pr.raw["target"]["kind"] == "named_place"
    assert pr.raw["target"]["name"] == "교보문고"
    assert pr.raw["target"]["specific"] is True

