"""Unit tests for data contracts, JSON schemas, config helpers, and basic utilities."""

from __future__ import annotations

import dataclasses

import pytest

from app import config, contracts, kakao, llm

SCHEMAS = [
    "PARSE_SCHEMA",
    "VISION_SCHEMA",
    "VERIFY_SCHEMA",
    "JUDGE_SCHEMA",
    "QUESTION_SCHEMA",
    "ANSWER_SCHEMA",
    "B0_SCHEMA",
]

DATACLASSES = [
    "LatLng",
    "ParseResult",
    "Place",
    "Jibun",
    "Window",
    "Anchor",
    "Building",
    "ViewPoint",
    "Candidate",
    "Evidence",
    "Question",
    "Decision",
    "RegistryRecord",
    "Pano",
    "VisionStats",
]


def _walk(schema: dict, path: str = "$"):
    """Yield (path, node) pairs recursively across a JSON schema tree."""
    yield path, schema
    for k, sub in (schema.get("properties") or {}).items():
        if isinstance(sub, dict):
            yield from _walk(sub, f"{path}.{k}")
    items = schema.get("items")
    if isinstance(items, dict):
        yield from _walk(items, f"{path}[]")


@pytest.mark.parametrize("name", SCHEMAS)
def test_schema_required_keys_exist_in_properties(name: str):
    schema = getattr(contracts, name)
    for _, node in _walk(schema):
        for key in node.get("required", []):
            assert key in node.get("properties", {})


@pytest.mark.parametrize("name", SCHEMAS)
def test_schema_enums_are_non_empty(name: str):
    schema = getattr(contracts, name)
    for _, node in _walk(schema):
        if "enum" in node:
            assert node["enum"]


@pytest.mark.parametrize("name", DATACLASSES)
def test_dataclass_builds_with_defaults(name: str):
    cls = getattr(contracts, name)
    cls()
    assert dataclasses.is_dataclass(cls)


def test_parse_schema_category_codes_are_kakao_18_plus_none():
    landmark = contracts.PARSE_SCHEMA["properties"]["landmarks"]["items"]
    codes = landmark["properties"]["poi"]["properties"]["category_code"]["enum"]
    assert len(codes) == 19 and "NONE" in codes and ("CS2" in codes) and ("SW8" in codes)


def test_config_get_prefers_process_env_and_strips(monkeypatch):
    monkeypatch.setenv("EODI_TEST_ONLY_VALUE", "  from-env  ")
    assert config.get("EODI_TEST_ONLY_VALUE") == "from-env"


def test_config_get_missing_returns_none():
    assert config.get("EODI_DEFINITELY_MISSING_NAME_12345") is None


def test_config_mask_hides_secret_values(monkeypatch):
    monkeypatch.setenv("KAKAO_REST_API_KEY", "test-secret-kakao-key")
    value = config.get("KAKAO_REST_API_KEY")
    assert config.mask(f"url?key={value}&x=1") == "url?key=***&x=1"


def test_kakao_parse_places_reads_keyword_documents():
    body = {
        "documents": [
            {
                "place_name": "오토웨이타워",
                "x": "127.0636",
                "y": "37.5059",
                "category_group_code": "",
                "address_name": "서울 강남구 대치동 948",
                "distance": "120",
            }
        ]
    }
    places = kakao.parse_places(body)
    p = places[0]
    assert len(places) == 1
    assert p.name == "오토웨이타워" and p.at.lat == pytest.approx(37.5059) and (p.at.lng == pytest.approx(127.0636))
    assert p.address == "서울 강남구 대치동 948" and p.distance_m == pytest.approx(120.0)


def test_kakao_parse_places_without_distance():
    body = {
        "documents": [
            {
                "place_name": "a",
                "x": "127",
                "y": "37",
                "category_group_code": "SW8",
                "address_name": "",
                "distance": "",
            }
        ]
    }
    assert kakao.parse_places(body)[0].distance_m is None


def test_llm_cosine():
    assert llm.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert llm.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
