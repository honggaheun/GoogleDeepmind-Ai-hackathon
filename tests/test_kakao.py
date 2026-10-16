"""Unit tests for kakao module."""

from __future__ import annotations

import pytest

from app import config, kakao


def test_keyword_passes_category_filter_and_page(monkeypatch):
    seen = {}

    def fake_get(path, params):
        seen["path"] = path
        seen["params"] = params
        return {"documents": []}

    monkeypatch.setattr(kakao, "_get", fake_get)
    kakao.keyword("2호선", near=config.VENUE, radius_m=3300, category="SW8", page=2)
    assert seen["path"] == "/search/keyword.json"
    assert (
        seen["params"]["category_group_code"],
        seen["params"]["page"],
        seen["params"]["radius"],
    ) == ("SW8", 2, 3300)


def test_category_passes_page(monkeypatch):
    seen = {}

    def fake_get(path, params):
        seen["path"] = path
        seen["params"] = params
        return {"documents": []}

    monkeypatch.setattr(kakao, "_get", fake_get)
    kakao.category("FD6", config.VENUE, 700, page=3)
    assert (
        seen["path"] == "/search/category.json"
        and seen["params"]["page"] == 3
        and (seen["params"]["sort"] == "distance")
    )


def test_parse_places_keeps_category_name():
    body = {
        "documents": [
            {
                "place_name": "명륜진사갈비 본사",
                "x": "127.1",
                "y": "37.5",
                "category_group_code": "",
                "category_name": "서비스,산업 > 기업",
                "address_name": "서울 송파구 가락동 113",
                "distance": "1348.0",
            }
        ]
    }
    place = kakao.parse_places(body)[0]
    assert (place.category_name, place.distance_m) == ("서비스,산업 > 기업", 1348.0)


def test_keyword_without_category_sends_no_filter(monkeypatch):
    seen = {}

    def fake_get(path, params):
        seen["path"] = path
        seen["params"] = params
        return {"documents": []}

    monkeypatch.setattr(kakao, "_get", fake_get)
    kakao.keyword("오토웨이 타워")
    assert "category_group_code" not in seen["params"] and "x" not in seen["params"]
