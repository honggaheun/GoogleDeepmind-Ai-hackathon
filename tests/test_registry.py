"""Unit tests for building registry parameter building, row selection, and record conversion."""

from __future__ import annotations

import pytest

from app import registry
from app.contracts import Jibun


def _body(items):
    """Wrap an items payload into a standard registry response envelope."""
    return {"response": {"header": {"resultCode": "00"}, "body": {"items": items}}}


def test_params_pads_bun_ji_and_flat_land():
    p = registry.params(Jibun("1168010600", "948", "0", False, "서울 강남구 대치동 948"))
    assert p == {
        "sigunguCd": "11680",
        "bjdongCd": "10600",
        "platGbCd": "0",
        "bun": "0948",
        "ji": "0000",
    }


def test_params_mountain_lot():
    p = registry.params(Jibun("1171010100", "12", "3", True, ""))
    assert (p["platGbCd"], p["bun"], p["ji"]) == ("1", "0012", "0003")


@pytest.mark.parametrize(
    "capture,approved,expected",
    [
        ("2016-06", "20140207", False),
        ("2015-03", "20170510", True),
        ("2018", "20180601", True),
        ("2016-06", None, False),
    ],
)
def test_photo_predates_building(capture, approved, expected):
    assert registry.photo_predates_building(capture, approved) is expected


def test_pick_prefers_main_building_row():
    rows = [
        {"mainAtchGbCdNm": "부속건축물", "bldNm": "창고"},
        {"mainAtchGbCdNm": "주건축물", "bldNm": "본관"},
    ]
    assert registry.pick(_body({"item": rows}))["bldNm"] == "본관"


def test_pick_accepts_single_item_dict():
    assert (
        registry.pick(_body({"item": {"mainAtchGbCdNm": "주건축물", "bldNm": "오토웨이타워"}}))[
            "bldNm"
        ]
        == "오토웨이타워"
    )


def test_pick_returns_none_when_no_items():
    assert registry.pick(_body("")) is None


def test_record_converts_numbers():
    rec = registry.to_record(
        {
            "grndFlrCnt": 9,
            "ugrndFlrCnt": "5",
            "heit": "51.44",
            "mainPurpsCdNm": "업무시설",
            "etcPurps": "",
            "bldNm": "오토웨이타워",
            "useAprDay": 20140207,
        }
    )
    assert (
        rec.grnd_floors,
        rec.ugrnd_floors,
        rec.height_m,
        rec.main_purpose,
        rec.etc_purpose,
        rec.use_apr_day,
    ) == (9, 5, 51.44, "업무시설", None, "20140207")
