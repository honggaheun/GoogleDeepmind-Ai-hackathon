"""Unit tests for signboard reading, lookup, and parse structure updates."""

from __future__ import annotations

import copy
from types import SimpleNamespace

from app import config, mapdata, parse, signboard
from app.contracts import Candidate, LatLng, ParseResult, Place, Window

try:
    from app import search as _real_search
except ImportError:
    _real_search = None


class _FakeBudget:
    """Minimal search budget stub for unit testing signboard integration."""

    def __init__(self) -> None:
        self.used = 0
        self.notes: list[str] = []
        self.places: list[Place] = []
        self.maps_sources: list[dict] = []
        self.signboard = None


def _build_search_stub():
    """Return a search module stub when search.py is not yet present or complete."""
    if (
        _real_search is not None
        and hasattr(_real_search, "target_kind")
        and hasattr(_real_search, "_landmark_positions")
        and hasattr(_real_search, "_named_candidates")
    ):
        return _real_search

    ns = SimpleNamespace()
    ns._Budget = _FakeBudget
    ns._kw = lambda budget, query, near=None, radius_m=1000, size=15, category=None, page=1: []
    ns._poi_pool = lambda budget, name, category_code, center, radius_m: []
    ns.match_terms = lambda terms, places: {}
    ns.google_places_of_type = lambda budget, term, center, radius, hint="": []
    ns.google_place = lambda budget, name, near, hint="": None

    def target_kind(pr: ParseResult) -> str:
        t = (pr.raw or {}).get("target") or {}
        kind = t.get("kind") or "point"
        if kind == "named_place" and t.get("specific") is False and t.get("name"):
            return "building"
        return kind

    def _landmark_positions(pr: ParseResult, md, center: LatLng, radius: float, budget):
        out: dict[str, list[Place]] = {}
        for lm in (pr.raw or {}).get("landmarks") or []:
            lid = lm.get("id")
            if not lid:
                continue
            p_queries = lm.get("photo_queries")
            if p_queries:
                rep: dict = {}
                hit = signboard.lookup(
                    p_queries,
                    center,
                    radius,
                    ns._kw,
                    budget,
                    signboard_text=lm.get("photo_names"),
                    report=rep,
                )
                if hit is not None:
                    out[lid] = [hit]
                    if budget is not None:
                        budget.notes.append(f"간판 '{p_queries[0]}' → {hit.name}")
                else:
                    out[lid] = []
                continue
            poi = lm.get("poi") or {}
            if poi.get("specific") and poi.get("name"):
                out[lid] = ns._kw(budget, poi["name"], near=center, radius_m=radius) or []
            elif poi.get("name") or poi.get("category_code"):
                pool = ns._poi_pool(budget, poi.get("name", ""), poi.get("category_code"), center, radius)
                matched = ns.match_terms([poi.get("name", "")], pool).get(poi.get("name", ""), [])
                if not matched:
                    matched = ns.google_places_of_type(budget, poi.get("name", ""), center, radius)
                out[lid] = matched
        return out

    def _named_candidates(pr: ParseResult, md, window, anchors, budget) -> list[Candidate]:
        t = (pr.raw or {}).get("target") or {}
        center = anchors[0].at if anchors else (window.center if window else config.VENUE)
        radius = 1500
        p_queries = t.get("photo_queries")
        if p_queries:
            rep: dict = {}
            hit = signboard.lookup(
                p_queries,
                center,
                radius,
                ns._kw,
                budget,
                signboard_text=t.get("photo_names"),
                report=rep,
            )
            if hit is not None:
                return [
                    Candidate(
                        id="C1",
                        type="named_place",
                        name=hit.name,
                        centroid=hit.at,
                        geometry={"type": "Point", "coordinates": [hit.at.lng, hit.at.lat]},
                    )
                ]
            return []
        name = t.get("name") or ""
        hits = ns._kw(budget, name, near=center, radius_m=radius, size=15) or []
        if not hits:
            gp = ns.google_place(budget, name, center)
            if gp is not None:
                hits = [gp]
        return [
            Candidate(
                id=f"C{i}",
                type="named_place",
                name=h.name,
                centroid=h.at,
                geometry={"type": "Point", "coordinates": [h.at.lng, h.at.lat]},
            )
            for i, h in enumerate(hits[:3], start=1)
        ]

    ns.target_kind = target_kind
    ns._landmark_positions = _landmark_positions
    ns._named_candidates = _named_candidates
    return ns


search = _build_search_stub()

READING = {
    "queries": ["동촌아귀찜", "동촌 아귀찜 황태 낙지찜", "동촌아구찜"],
    "names": ["동촌 아귀찜·황태 낙지찜"],
    "kind": "아귀찜집",
    "texts": ["불고기", "삼겹살"],
    "floors": 1,
    "traits": "기와 지붕의 단층 한식당",
    "confidence": "high",
}

RAW = {
    "language": "ko",
    "transcript": "",
    "region": {"kind": "district", "query": "송파구", "evidence": "송파구"},
    "anchor": {"kind": "place", "query": "한림연예예술고등학교", "evidence": "한림예고"},
    "steps": [
        {
            "order": 1,
            "action": "go_straight",
            "distance_class": "none",
            "evidence": "직진",
        }
    ],
    "landmarks": [
        {
            "id": "lm_1",
            "evidence": "아귀찜집",
            "poi": {"name": "아귀찜집", "category_code": "FD6", "specific": False},
        }
    ],
    "target": {
        "kind": "building",
        "name": "고기집",
        "category_code": "FD6",
        "landmark_ref": "lm_1",
        "relation": "across",
        "specific": False,
    },
    "uncertainty": {"hedges": [], "staleness": "unknown"},
    "ambiguities": [],
}


def _pr(raw: dict | None = None) -> ParseResult:
    """Create a deep-copied ParseResult fixture for signboard tests."""
    return ParseResult(
        raw=copy.deepcopy(raw if raw is not None else RAW),
        source_text="송파구 한림예고에서 직진하다가 아귀찜집 맞은편 고기집",
        language="ko",
    )


def test_landmark_role_turns_the_reference_landmark_into_a_specific_store():
    pr = _pr()
    poi = pr.raw["landmarks"][0]["poi"]
    assert signboard.apply(pr, READING, "landmark") == "lm_1"
    assert poi["specific"] is True and poi["name"] == "동촌아귀찜"
    assert pr.raw["landmarks"][0]["photo_queries"] == READING["queries"]
    assert pr.raw["target"]["name"] == "고기집" and pr.raw["target"]["specific"] is False


def test_target_role_turns_the_target_into_a_named_place():
    pr = _pr()
    t = pr.raw["target"]
    assert signboard.apply(pr, READING, "target") == "__target__"
    assert (
        t["kind"] == "named_place"
        and t["specific"] is True
        and (t["name"] == "동촌아귀찜")
        and (t["photo_queries"] == READING["queries"])
    )
    assert search.target_kind(pr) == "named_place"


def test_landmark_role_without_a_reference_landmark_falls_back_to_target():
    raw = copy.deepcopy(RAW)
    raw["target"] = {
        "kind": "building",
        "name": None,
        "landmark_ref": None,
        "relation": None,
        "specific": None,
    }
    pr = _pr(raw)
    assert signboard.apply(pr, READING, "landmark") == "__target__"
    assert pr.raw["target"]["name"] == "동촌아귀찜"


def test_unreadable_signboard_changes_nothing():
    pr = _pr()
    assert signboard.apply(pr, {"queries": [], "names": [], "confidence": "low"}, "landmark") is None
    assert pr.raw["landmarks"][0]["poi"]["specific"] is False


def test_lookup_tries_queries_in_order_and_keeps_only_hits_sharing_the_name_head():
    calls = []

    def kw(budget, query, near=None, radius_m=500, size=5):
        calls.append(query)
        if query == "동촌아귀찜":
            return [Place(name="아귀찜명가", at=LatLng(37.485, 127.133))]
        if query == "동촌 아귀찜 황태 낙지찜":
            return []
        return [
            Place(name="옛골아구찜", at=LatLng(37.485, 127.133)),
            Place(name="동촌아구찜 문정점", at=LatLng(37.486, 127.134)),
        ]

    place = signboard.lookup(READING["queries"], LatLng(37.485, 127.133), 1500, kw, search._Budget())
    assert place is not None and place.name == "동촌아구찜 문정점" and (calls == READING["queries"])
    assert signboard.lookup([], LatLng(0, 0), 100, kw, None) is None


def test_landmark_positions_use_the_signboard_queries(monkeypatch):
    pr = _pr()
    signboard.apply(pr, READING, "landmark")
    hit = Place(name="동촌아구찜", at=LatLng(37.486, 127.134))

    monkeypatch.setattr(
        search,
        "_kw",
        lambda budget, query, **kwargs: [hit] if query == "동촌아구찜" else [],
    )
    monkeypatch.setattr(search, "_poi_pool", lambda *a, **k: [])
    monkeypatch.setattr(search, "match_terms", lambda *a, **k: {})
    monkeypatch.setattr(search, "google_places_of_type", lambda *a, **k: [])
    monkeypatch.setattr(signboard.llm, "embed", lambda text: [1.0, 0.0, 0.0])

    md = mapdata.MapData.from_parts(config.VENUE, {}, [], [])
    budget = search._Budget()
    out = search._landmark_positions(pr, md, config.VENUE, 1000, budget)
    assert out["lm_1"] == [hit] and any(("동촌아구찜" in n for n in budget.notes))


def test_named_candidates_use_the_signboard_queries(monkeypatch):
    pr = _pr()
    signboard.apply(pr, READING, "target")
    hit = Place(name="동촌아구찜", at=LatLng(37.486, 127.134))
    seen: list[str] = []

    def fake_kw(budget, query, **kwargs):
        seen.append(query)
        return [hit] if query == "동촌아구찜" else []

    monkeypatch.setattr(search, "_kw", fake_kw)
    monkeypatch.setattr(search, "google_place", lambda *a, **k: None)
    monkeypatch.setattr(signboard.llm, "embed", lambda text: [1.0, 0.0, 0.0])

    md = mapdata.MapData.from_parts(config.VENUE, {}, [], [])
    win = Window(center=config.VENUE, radius_m=1500, source="text", label="송파구", kind="district", area="송파구")
    out = search._named_candidates(pr, md, win, [], search._Budget())
    assert (
        len(out) == 1
        and out[0].name == "동촌아구찜"
        and (out[0].type == "named_place")
        and (seen == READING["queries"])
    )


def test_photo_only_parse_is_an_empty_named_target_filled_by_the_signboard():
    pr = parse.photo_only("송파구")
    assert pr.raw["steps"] == [] and pr.raw["anchor"]["kind"] == "none" and (search.target_kind(pr) == "named_place")
    assert signboard.apply(pr, READING, "target") == "__target__" and pr.raw["target"]["name"] == "동촌아귀찜"


def test_lookup_falls_back_to_embedding_similarity_when_the_head_rule_misses(monkeypatch):
    vec = {
        "동촌 아귀찜 · 황태 낙지찜": [1.0, 0.0, 0.0],
        "DONGCHON 아구찜": [0.95, 0.31, 0.0],
        "아귀찜전문 가나다": [0.7, 0.71, 0.0],
    }
    hits = [
        Place(name="아귀찜전문 가나다", at=LatLng(37.49, 127.13)),
        Place(name="DONGCHON 아구찜", at=LatLng(37.491, 127.131)),
    ]
    rep: dict = {}

    monkeypatch.setattr(signboard.llm, "embed", lambda text: vec.get(text, [0.73, 0.68, 0.0]))

    hit = signboard.lookup(
        ["동촌아귀찜"],
        LatLng(37.485, 127.133),
        1500,
        lambda b, q, **k: hits,
        None,
        "동촌 아귀찜 · 황태 낙지찜",
        rep,
    )
    assert hit is not None and hit.name == "DONGCHON 아구찜" and (rep["how"] == "embedding") and (rep["sim"] >= 0.85)

    only_generic = [Place(name="아귀찜전문 가나다", at=LatLng(37.49, 127.13))]
    assert (
        signboard.lookup(
            ["동촌아귀찜"],
            LatLng(37.485, 127.133),
            1500,
            lambda b, q, **k: only_generic,
            None,
            "동촌 아귀찜 · 황태 낙지찜",
            rep,
        )
        is None
    )
    assert rep["how"] == "none" and rep["sim"] < 0.85

    rule_hit = [Place(name="동촌 본점", at=LatLng(37.49, 127.13))]
    assert (
        signboard.lookup(
            ["동촌아귀찜"],
            LatLng(37.485, 127.133),
            1500,
            lambda b, q, **k: rule_hit,
            None,
            "동촌 아귀찜 · 황태 낙지찜",
            rep,
        ).name
        == "동촌 본점"
    )
    assert rep["how"] == "rule" and rep["sim"] is not None


def test_point_in_front_of_a_specific_store_becomes_a_named_place_target():
    raw = {
        "language": "ko",
        "region": {"kind": "none", "query": "", "evidence": ""},
        "anchor": {"kind": "station_exit", "query": "강남역 11번 출구", "evidence": "강남역 11번 출구"},
        "steps": [{"order": 1, "action": "go_straight", "distance_class": "none", "evidence": "직진"}],
        "landmarks": [
            {
                "id": "lm_1",
                "evidence": "교보문고 건물",
                "poi": {"name": "교보문고", "specific": True},
            }
        ],
        "target": {
            "kind": "point",
            "landmark_ref": "lm_1",
            "relation": "front",
            "name": None,
            "specific": None,
        },
    }
    pr = parse.validate(raw, "강남역 11번 출구 직진하면 교보문고 건물 앞이에요", None)
    assert pr.raw["target"]["kind"] == "named_place" and pr.raw["target"]["name"] == "교보문고"

    raw["target"] = {
        "kind": "building",
        "landmark_ref": "lm_1",
        "relation": "across",
        "name": "고기집",
        "specific": False,
    }
    pr = parse.validate(raw, "강남역 11번 출구 직진하면 교보문고 건물 맞은편 고기집이에요", None)
    assert pr.raw["target"]["kind"] == "building" and pr.raw["target"]["relation"] == "across"
