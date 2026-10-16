"""Search stage: resolves search window, start anchors, and candidate paths/buildings from route descriptions."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
import math
import re
import sys
from typing import Any

from google.genai import types
import networkx as nx
import shapely
import shapely.geometry

from app import config, geo, kakao, llm, mapdata, parse
from app.contracts import Anchor, Candidate, Evidence, LatLng, ParseResult, Place, Question, ViewPoint, Window

# Geometry and search thresholds
ALLEY = {"residential", "service", "living_street", "pedestrian", "unclassified", "footway"}
MAJOR = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
}
TURNS = {"turn_left": "left", "turn_right": "right", "enter_alley": None}
GROUND_TIMEOUT_MS = 12_000
GROUND_TYPE_TIMEOUT_MS = 30_000
BEAM = 10
MAX_STOPS = 5
MAX_CANDIDATES = 5
FOOTWAY_MIN_M = 25
MATCH_M = 40
RELAX_LEVELS = [
    (),
    ("ordinal",),
    ("ordinal", "side"),
    ("ordinal", "side", "distance"),
    ("ordinal", "side", "distance", "relation"),
]
SAME_NAME_STATION_M = 20000
ANCHOR_NEAR_M = 8000
VENUE_CATS = (
    "백화점",
    "복합쇼핑몰",
    "쇼핑몰",
    "대형마트",
    "아울렛",
    "빌딩",
    "지하철",
    "기차역",
    "터미널",
    "공항",
    "문화시설",
    "관광,명소",
)
TENANT_CATS = (
    "음식점",
    "카페",
    "패션",
    "의류",
    "운명감정",
    "미용",
    "학원",
    "편의점",
    "서비스,산업",
    "은행",
    "약국",
    "주차장",
    "ATM",
)
AREA_SEARCH_M = 20000
PUBLIC_ROAD_M = 150
ROUTE_MAX_TURNS = 2
ROAD_SKIP = {"footway", "steps", "path", "cycleway", "service", "corridor", "bridleway"}
NEAR_END_M = 300
TARGET_ROUTE_MIN_M = 60
TARGET_BUILDING_M = 15
RELATION_MIN_WALK_M = 60
RELATION_START_EXCLUDE_M = 40
ACROSS_MAX_M = 80
BESIDE_MAX_M = 50
TOWARD_TOL_DEG = 60
BIG_ROAD_M = 45
CATEGORY_CANDS = 6
P_POI_MATCH = (
    "아래는 지도에서 찾은 가게 목록이다(번호: 이름 | 업종). 설명어마다 그 설명어에 해당하는 가게의 번호만 골라라.\n"
)
CATEGORY_WORDS = ["돈가스", "고기집", "스타벅스", "편의점"]
GOOGLE_TYPE_CAP = 1

_CITY_SHORT_NAMES = {
    "서울특별시": "서울",
    "서울시": "서울",
    "서울": "서울",
    "부산광역시": "부산",
    "부산시": "부산",
    "부산": "부산",
    "대구광역시": "대구",
    "대구시": "대구",
    "대구": "대구",
    "인천광역시": "인천",
    "인천시": "인천",
    "인천": "인천",
    "광주광역시": "광주",
    "광주시": "광주",
    "광주": "광주",
    "대전광역시": "대전",
    "대전시": "대전",
    "대전": "대전",
    "울산광역시": "울산",
    "울산시": "울산",
    "울산": "울산",
    "세종특별자치시": "세종",
    "세종시": "세종",
    "세종": "세종",
    "경기도": "경기",
    "경기": "경기",
    "강원도": "강원",
    "강원특별자치도": "강원",
    "강원": "강원",
    "충청북도": "충북",
    "충북": "충북",
    "충청남도": "충남",
    "충남": "충남",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전북": "전북",
    "전라남도": "전남",
    "전남": "전남",
    "경상북도": "경북",
    "경북": "경북",
    "경상남도": "경남",
    "경남": "경남",
    "제주특별자치도": "제주",
    "제주도": "제주",
    "제주": "제주",
}


class _Budget:
    """Manages API call limits, caching, query reuse, and collected place metadata."""

    def __init__(self, limit: int = 25):
        self.limit = limit
        self.used = 0
        self.notes: list[str] = []
        self.places: list[Place] = []
        self.maps_sources: list[dict] = []
        self._cache: dict[Any, Any] = {}
        self._remembered: list[dict] = []

    def call(self, key: Any, fn: Any) -> Any:
        """Execute fn if within budget and cache result by key."""
        if key in self._cache:
            return self._cache[key]
        if self.used >= self.limit:
            self.notes.append(f"budget exceeded: {key}")
            return None
        try:
            self.used += 1
            res = fn()
            self._cache[key] = res
            if isinstance(res, list):
                for p in res:
                    if isinstance(p, Place) and p.category_name and len(self.places) < 120:
                        if not any(existing.name == p.name and existing.at == p.at for existing in self.places):
                            self.places.append(p)
            return res
        except kakao.KakaoError as exc:
            self.notes.append(str(exc))
            return None

    def reuse(self, key: Any, near: LatLng | None, radius_m: int) -> list[Place] | None:
        """Find reusable cached results within 100m distance and larger radius."""
        if near is None:
            return None
        for item in self._remembered:
            if item["key"] == key and item["near"] is not None:
                d = geo.dist_m(near, item["near"])
                if d <= 100.0 and item["radius_m"] >= radius_m:
                    return [p for p in item["result"] if geo.dist_m(near, p.at) <= radius_m]
        return None

    def remember(self, key: Any, near: LatLng | None, radius_m: int, result: list[Place]) -> None:
        """Record query result for potential spatial reuse."""
        self._remembered.append({
            "key": key,
            "near": near,
            "radius_m": radius_m,
            "result": list(result),
        })


def _kw(
    budget: _Budget,
    query: str,
    near: LatLng | None = None,
    radius_m: int = 1000,
    size: int = 15,
    category: str | None = None,
    page: int = 1,
) -> list[Place]:
    """Search Kakao places by keyword using budget limits, query deduplication, and caching."""
    reuse_key = ("kw", query, category, page)
    reused = budget.reuse(reuse_key, near, radius_m)
    if reused is not None:
        return reused[:size]

    # Always fetch 15 from Kakao to deduplicate queries across different requested sizes
    call_key = ("kw", query, round(near.lat, 6) if near else None, round(near.lng, 6) if near else None, radius_m, category, page)
    res = budget.call(
        call_key,
        lambda: kakao.keyword(query, near=near, radius_m=radius_m, size=15, category=category, page=page),
    )
    places = res or []
    budget.remember(reuse_key, near, radius_m, places)
    return places[:size]


def _kw_all(
    budget: _Budget,
    query: str,
    near: LatLng | None = None,
    radius_m: int = 1000,
    category: str | None = None,
    max_pages: int = 3,
) -> tuple[list[Place], bool]:
    """Retrieve all pages of keyword search up to max_pages."""
    out: list[Place] = []
    complete = True
    for p in range(1, max_pages + 1):
        page_results = _kw(budget, query, near=near, radius_m=radius_m, size=15, category=category, page=p)
        out.extend(page_results)
        if len(page_results) < 15:
            break
    return out, complete


def _cat(
    budget: _Budget,
    code: str,
    near: LatLng,
    radius_m: int = 1000,
    size: int = 15,
    page: int = 1,
) -> list[Place]:
    """Search Kakao category places using budget limits."""
    call_key = ("cat", code, round(near.lat, 6), round(near.lng, 6), radius_m, size, page)
    res = budget.call(
        call_key,
        lambda: kakao.category(code, near=near, radius_m=radius_m, size=size, page=page),
    )
    return res or []


def _addr(budget: _Budget, query: str) -> list[Place]:
    """Search Kakao address using budget limits."""
    res = budget.call(("addr", query), lambda: kakao.address(query))
    return res or []


def _is_station(p: Place, q: str = "") -> bool:
    """Check if place or query represents a subway or train station."""
    if p.category == "SW8":
        return True
    cat_name = p.category_name or ""
    if "지하철" in cat_name or "전철" in cat_name:
        return True
    if p.name.endswith("역") or "역 " in p.name or "역(" in p.name:
        return True
    if q and q.endswith("역"):
        return True
    return False


def _region_kind(name: str, from_address: bool, category: str) -> str:
    """Classify a region name into city, district, neighborhood, station, or place."""
    if from_address:
        if name in _CITY_SHORT_NAMES or name.endswith(("도", "특별시", "광역시", "특별자치시", "특별자치도")):
            return "city"
        if name.endswith(("구", "군", "시")):
            return "district"
        if name.endswith(("동", "가", "읍", "면", "리")):
            return "neighborhood"
    if category == "SW8" or name.endswith("역"):
        return "station"
    if name.endswith(("구", "군")):
        return "district"
    if name.endswith(("동", "읍", "면", "리")):
        return "neighborhood"
    return "place"


def _window(place: Place, kind: str, source: str, query: str = "") -> Window:
    """Construct a Window object from a place and its classified region kind."""
    radius_m = config.WINDOW_RADIUS_M.get(kind, 1000)
    label = place.address or place.name
    area = ""
    if kind == "city":
        for k, v in _CITY_SHORT_NAMES.items():
            if place.name.startswith(k) or (place.address and k in place.address) or place.name == v:
                area = v
                break
        if not area:
            area = place.name[:2]
        radius_m = max(radius_m, 20000)
    elif kind == "district":
        m = re.search(r"([가-힣]+(?:구|군|시))\b", place.address or place.name)
        area = m.group(1) if m else place.name
    elif kind == "neighborhood":
        m = re.search(r"([가-힣]+)(?:\d+)?(?:동|가|읍|면|리)\b", place.address or place.name)
        if m:
            base = m.group(1)
            suffix_match = re.search(r"(동|가|읍|면|리)\b", place.address or place.name)
            suffix = suffix_match.group(1) if suffix_match else "동"
            area = base + suffix
        else:
            area = place.name
    return Window(
        center=place.at,
        radius_m=radius_m,
        source=source,
        label=label,
        kind=kind,
        area=area,
        query=query,
    )


def resolve_window(
    pr: ParseResult,
    region_input: str | None,
    current: LatLng | None,
    budget: _Budget | None = None,
) -> Window | None:
    """Resolve geographic search window from explicit input, current location, or text."""
    b = budget if budget is not None else _Budget()

    # 1. Explicit region input
    if region_input and region_input.strip():
        q = region_input.strip()
        addr_places = _addr(b, q)
        if addr_places:
            p = addr_places[0]
            kind = _region_kind(q, True, p.category)
            return _window(p, kind, "region_input", query=q)

        # Keyword fallback
        kw_places = _kw(b, q, size=5)
        if kw_places:
            # Check station ambiguity across distant cities
            if _is_station(kw_places[0], q):
                if len(kw_places) > 1 and geo.dist_m(kw_places[0].at, kw_places[1].at) > SAME_NAME_STATION_M:
                    return None
            p = kw_places[0]
            kind = "station" if _is_station(p, q) else "place"
            return _window(p, kind, "region_input", query=q)

    # 2. Current location
    if current is not None:
        return Window(center=current, radius_m=500, source="current_location", label="현재 위치", kind="place")

    # 3. Parsed region from text
    raw_region = pr.raw.get("region") or {}
    reg_q = (raw_region.get("query") or "").strip()
    if reg_q and raw_region.get("kind") != "none":
        addr_places = _addr(b, reg_q)
        if addr_places:
            p = addr_places[0]
            kind = _region_kind(reg_q, True, p.category)
            return _window(p, kind, "text", query=reg_q)

        kw_places = _kw(b, reg_q, size=5)
        if kw_places:
            if _is_station(kw_places[0], reg_q):
                if len(kw_places) > 1 and geo.dist_m(kw_places[0].at, kw_places[1].at) > SAME_NAME_STATION_M:
                    return None
            p = kw_places[0]
            kind = "station" if _is_station(p, reg_q) else "place"
            return _window(p, kind, "text", query=reg_q)

    return None


def _kw_window(
    budget: _Budget,
    query: str,
    window: Window | None,
    size: int = 5,
    category: str | None = None,
) -> list[Place]:
    """Search places constrained by window area, center radius, or nationwide fallback."""
    if window is None:
        return _kw(budget, query, near=None, radius_m=20000, size=size, category=category)

    if window.area:
        places = _kw(budget, query, near=window.center, radius_m=AREA_SEARCH_M, size=15, category=category)
        filtered = [p for p in places if window.area in p.address]
        if filtered:
            return filtered[:size]

        # Colloquial region name allows nearby anchor within threshold
        is_colloquial = bool(window.query and not window.query.endswith(("구", "군", "시")))
        if is_colloquial:
            nearby = [p for p in places if geo.dist_m(p.at, window.center) <= ANCHOR_NEAR_M]
            if nearby:
                nearby.sort(key=lambda p: geo.dist_m(p.at, window.center))
                budget.notes.append("주소 밖 이지만 창 중심 8km 안 기준점 채택")
                return nearby[:size]
        return []

    return _kw(budget, query, near=window.center, radius_m=window.radius_m, size=size, category=category)


def parse_station_exit(text: str) -> tuple[str | None, str | None, str | None]:
    """Parse text into subway station name, exit number string, and subway line string."""
    m_exit = re.search(r"(\d+)번\s*출구", text)
    exit_ref = m_exit.group(1) if m_exit else None

    m_line = re.search(r"(\d+호선|신분당선|수인분당선|경의중앙선|공항철도|우이신설선|신림선|서해선|김포골드라인)", text)
    line = m_line.group(1) if m_line else None

    station: str | None = None
    m_station = re.search(r"([가-힣]{2,})역", text)
    if m_station:
        station = m_station.group(1) + "역"
    elif exit_ref:
        clean_text = text
        if line:
            clean_text = clean_text.replace(line, " ")
        clean_text = re.sub(r"\d+호선", " ", clean_text)
        m_word = re.search(r"([가-힣]{2,})\s*(?:\d+)번\s*출구", clean_text)
        if m_word:
            word = m_word.group(1).strip()
            invalid_suffixes = ("동", "구", "군", "시", "읍", "면", "리", "가", "길", "로", "선", "호선")
            invalid_words = {"지하철", "전철", "호선", "출구", "역"}
            if word not in invalid_words and not word.endswith(invalid_suffixes):
                station = word + "역"

    return station, exit_ref, line


def _station_anchors(
    query: dict | str,
    window: Window | None,
    md: mapdata.MapData | None,
    budget: _Budget,
) -> tuple[list[Anchor], Question | None]:
    """Resolve anchors for subway station exits, prompting user if ambiguous."""
    evidence = query.get("evidence", "") if isinstance(query, dict) else str(query)
    q_str = query.get("query", "") if isinstance(query, dict) else str(query)
    station_name, exit_ref, line = parse_station_exit(evidence or q_str)

    if not exit_ref and not station_name and not line:
        if window is None:
            return [], Question(id="Q-region", source="region", intent_ko="출발할 지역을 알려주세요")
        return [], Question(id="Q-station", source="station", intent_ko="출발할 역이나 출구를 알려주세요")

    if station_name and exit_ref:
        # 1. Kakao specific exit search
        kakao_query = f"{station_name} {exit_ref}번출구"
        kw_places = _kw(budget, kakao_query, near=window.center if window else None, radius_m=window.radius_m if window else 1000)
        matched = [p for p in kw_places if station_name in p.name]
        if matched:
            return [Anchor(name=matched[0].name, at=matched[0].at, source="kakao")], None

    # 2. OSM entrance lookup
    active_md = md
    station_place: Place | None = None
    if station_name:
        center_near = window.center if window else None
        st_places = _kw(budget, station_name, near=center_near, category="SW8")
        if st_places:
            station_place = st_places[0]
            if (window is None or window.kind == "city") and active_md is None:
                active_md = mapdata.load_area(station_place.at, 1000)

    if exit_ref and active_md is not None:
        exits = mapdata.entrances_with_ref(active_md, exit_ref)
        if exits:
            if line:
                line_stations = _kw(budget, line, near=window.center if window else config.VENUE, radius_m=3300, category="SW8")
                line_matches = [s for s in line_stations if line in s.name]

                # Map entrances to matching line stations
                station_exits: dict[str, list[dict]] = {}
                for e in exits:
                    for st in line_matches:
                        if geo.dist_m(e["at"], st.at) <= 300:
                            st_key = st.name.split()[0] if " " in st.name else st.name
                            if not st_key.endswith("역"):
                                st_key += "역"
                            label = f"{st_key} {exit_ref}번 출구"
                            station_exits.setdefault(label, []).append(e)

                if len(station_exits) == 1:
                    lbl, entr_list = next(iter(station_exits.items()))
                    return [
                        Anchor(name=lbl, at=entr["at"], source="osm_entrance", node_id=entr.get("node_id"))
                        for entr in entr_list
                    ], None
                if len(station_exits) > 1:
                    return [], Question(id="Q-station", source="station", intent_ko="어느 역 출구인가요?", options_ko=list(station_exits.keys())[:5])

            if station_name:
                station_entrances = [e for e in exits if station_place and geo.dist_m(e["at"], station_place.at) <= 300]
                if station_entrances:
                    return [
                        Anchor(name=f"{station_name} {exit_ref}번 출구", at=e["at"], source="osm_entrance", node_id=e.get("node_id"))
                        for e in station_entrances
                    ], None

            # Multiple stations in window without line filter
            sw8_stations = kakao.category("SW8", window.center if window else config.VENUE, radius_m=window.radius_m if window else 1000) if hasattr(kakao, "category") else []
            opts = []
            for st in sw8_stations:
                st_key = st.name.split()[0]
                if not st_key.endswith("역"):
                    st_key += "역"
                opt = f"{st_key} {exit_ref}번 출구"
                if opt not in opts:
                    opts.append(opt)
            if len(opts) >= 2:
                return [], Question(id="Q-station", source="station", intent_ko="어느 역 출구인가요?", options_ko=opts[:5])

    # 3. Station found but exit data missing -> fallback to station center
    if station_place is not None:
        budget.notes.append("출구 위치 자료 없음 → 역 중심(오차 ±200m)")
        label = f"{station_place.name}(역 중심)"
        return [Anchor(name=label, at=station_place.at, source="kakao")], None

    if window is None:
        return [], Question(id="Q-region", source="region", intent_ko="출발할 지역을 알려주세요")
    return [], Question(id="Q-station", source="station", intent_ko="출발할 역을 알려주세요")


def _landmark_anchors(landmarks: list, window: Window | None, budget: _Budget) -> list[Anchor]:
    """Resolve specific landmark POIs into candidate start anchors."""
    anchors: list[Anchor] = []
    for lm in landmarks:
        poi = lm.get("poi") or {}
        if poi.get("specific") and poi.get("name"):
            places = _kw_window(budget, poi["name"], window, size=1)
            for p in places:
                anchors.append(Anchor(name=p.name, at=p.at, source="kakao"))
                if len(anchors) >= 3:
                    return anchors
    return anchors


def en_resolve(budget: _Budget, query: str, center: LatLng, radius_m: int = 1000) -> list[Place]:
    """Resolve an English place name to Korean places via 2-turn Gemini tool calling."""
    try:
        tool_decl = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="kakao_keyword_search",
                    description="Search Korean places by keyword",
                    parameters={
                        "type": "OBJECT",
                        "properties": {"query": {"type": "STRING", "description": "Korean place query"}},
                        "required": ["query"],
                    },
                )
            ]
        )
        prompt = f"Find the Korean name and coordinates for '{query}' near Seoul."
        resp1 = llm.generate_tools(
            "en_resolve_turn1",
            config.MODEL_LIGHT,
            [prompt],
            tools=[tool_decl],
            timeout_ms=GROUND_TIMEOUT_MS,
        )

        fn_call = None
        if hasattr(resp1, "candidates") and resp1.candidates:
            for part in resp1.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    fn_call = part.function_call
                    break

        if fn_call is None:
            budget.notes.append("Gemini tool call missing")
            return _kw(budget, query, near=center, radius_m=radius_m)

        k_query = getattr(fn_call, "args", {}).get("query", query) if hasattr(fn_call, "args") else query
        places = _kw(budget, k_query, near=center, radius_m=radius_m)

        # Turn 2: record grounding sources
        resp2 = llm.generate_tools(
            "en_resolve_turn2",
            config.MODEL_LIGHT,
            [
                prompt,
                resp1.candidates[0].content,
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name="kakao_keyword_search",
                            response={"places": [{"name": p.name, "address": p.address} for p in places]},
                        )
                    ],
                ),
            ],
            tools=[types.Tool(google_search=types.GoogleSearch())],
            timeout_ms=GROUND_TIMEOUT_MS,
        )

        if hasattr(resp2, "candidates") and resp2.candidates:
            g_meta = getattr(resp2.candidates[0], "grounding_metadata", None)
            if g_meta and hasattr(g_meta, "grounding_chunks"):
                for chunk in g_meta.grounding_chunks or []:
                    web = getattr(chunk, "web", None)
                    if web:
                        budget.maps_sources.append({"title": web.title, "uri": web.uri})

        return places
    except Exception as exc:
        budget.notes.append(f"Gemini resolution error: {exc}")
        return _kw(budget, query, near=center, radius_m=radius_m)


def resolve_anchors(
    pr: ParseResult,
    window: Window | None,
    md: mapdata.MapData | None,
    budget: _Budget | None = None,
) -> tuple[list[Anchor], Question | None]:
    """Determine initial navigation anchor candidates or generate an ambiguity question."""
    b = budget if budget is not None else _Budget()
    raw = pr.raw or {}
    anchor = raw.get("anchor") or {}
    kind = anchor.get("kind", "none")
    query = (anchor.get("query") or "").strip()
    evidence = anchor.get("evidence", "")

    # English query resolution
    if (pr.language == "en" or re.search(r"[A-Za-z]{3,}", query)) and kind == "place":
        places = en_resolve(b, query, window.center if window else config.VENUE, window.radius_m if window else 1000)
        if places:
            return [Anchor(name=p.name, at=p.at, source="kakao") for p in places[:3]], None

    if kind == "station_exit":
        return _station_anchors(anchor, window, md, b)

    if kind in ("place", "intersection"):
        # Check cross-city station ambiguity
        if query.endswith("역") or (window is None and "역" in query):
            candidates = _kw(b, query, near=window.center if window else None, radius_m=20000, category="SW8")
            if len(candidates) >= 2 and geo.dist_m(candidates[0].at, candidates[1].at) > SAME_NAME_STATION_M:
                return [], Question(
                    id="Q-region",
                    source="region",
                    intent_ko=f"'{query}'이(가) 여러 도시에 있습니다. 어느 곳인가요?",
                    options_ko=[c.name for c in candidates[:2]],
                )

        size = 3 if kind == "place" else 1
        places = _kw_window(b, query, window, size=size)
        if not places and evidence:
            cleaned = re.sub(r"(?:쪽|앞|에서|근처|집에서)$", "", evidence).strip()
            if cleaned and cleaned != query:
                places = _kw_window(b, cleaned, window, size=size)

        if places:
            return [Anchor(name=p.name, at=p.at, source="kakao") for p in places[:3]], None

        target_name = evidence or query
        return [], Question(id="Q-anchor", source="anchor", intent_ko=f"'{target_name}'을(를) 지도에서 찾지 못했어요.")

    # kind == "none"
    if window is None:
        # Check if region query was ambiguous across cities
        reg_q = ((raw.get("region") or {}).get("query") or "").strip()
        if reg_q:
            kw_places = _kw(b, reg_q, size=5)
            if len(kw_places) >= 2 and geo.dist_m(kw_places[0].at, kw_places[1].at) > SAME_NAME_STATION_M:
                return [], Question(
                    id="Q-region",
                    source="region",
                    intent_ko=f"'{reg_q}'이(가) 여러 도시에 있습니다. 어느 곳인가요?",
                    options_ko=[p.name for p in kw_places[:2]],
                )
        return [], Question(id="Q-region", source="region", intent_ko="출발할 지역을 알려주세요.")

    if window.kind in ("station", "place"):
        return [Anchor(name=window.label, at=window.center, source=window.source)], None

    # District or neighborhood window: try specific landmarks
    lm_anchors = _landmark_anchors(raw.get("landmarks") or [], window, b)
    if lm_anchors:
        return lm_anchors, None

    target = raw.get("target") or {}
    if target.get("kind") == "named_place":
        return [], None

    return [], Question(id="Q-anchor", source="anchor", intent_ko="출발 지점을 알려주세요.")


# Part 2: Route expansion, candidate targets, and landmark matching


@dataclass
class _Path:
    """Internal search state during graph walk."""
    nodes: list[int]
    prev: int | None
    heading: tuple[float, float]
    score: float = 0.0
    seg_start: int = 0
    alley_way: int | None = None
    alley_entry: LatLng | None = None
    relax: list[str] = field(default_factory=list)
    pending: tuple[float, float] | None = None
    marks: dict = field(default_factory=dict)

    @property
    def node(self) -> int:
        return self.nodes[-1]


def _start_node(md: mapdata.MapData, at: LatLng, max_m: float = 60.0) -> int | None:
    """Find best starting road graph node near coordinate."""
    if not md or not md.nodes:
        return None
    public_nodes = set()
    for e in md.edges:
        if e.edge_type not in ROAD_SKIP:
            public_nodes.add(e.u)
            public_nodes.add(e.v)
    if public_nodes:
        best_pub = min(public_nodes, key=lambda n: geo.dist_m(at, md.nodes[n]))
        if geo.dist_m(at, md.nodes[best_pub]) <= PUBLIC_ROAD_M:
            return best_pub

    non_footway = set()
    for e in md.edges:
        if e.edge_type not in {"footway", "steps", "path"}:
            non_footway.add(e.u)
            non_footway.add(e.v)
    if non_footway:
        best_non = min(non_footway, key=lambda n: geo.dist_m(at, md.nodes[n]))
        if geo.dist_m(at, md.nodes[best_non]) <= max_m:
            return best_non

    return min(md.nodes.keys(), key=lambda n: geo.dist_m(at, md.nodes[n]))


def _straight_next(
    md: mapdata.MapData,
    prev: int | None,
    node: int,
    heading: tuple[float, float],
) -> int | None:
    """Find straightest continuing road neighbor within 30 degrees."""
    nbrs = md.neighbors(node)
    best_nbr = None
    min_eff_angle = 30.0

    prev_edge_type = None
    if prev is not None:
        for e in md.edges:
            if (e.u == prev and e.v == node) or (e.v == prev and e.u == node):
                prev_edge_type = e.edge_type
                break
    prev_is_road = prev_edge_type is not None and prev_edge_type not in {"footway", "steps", "path"}

    for nbr in nbrs:
        if nbr == prev:
            continue
        vec = geo.to_xy(md.nodes[nbr], md.nodes[node])
        if vec[0] == 0.0 and vec[1] == 0.0:
            continue
        angle = abs(geo.turn_angle_deg(heading, vec))
        if angle >= 30.0:
            continue

        next_edge_type = None
        for e in md.edges:
            if (e.u == node and e.v == nbr) or (e.v == node and e.u == nbr):
                next_edge_type = e.edge_type
                break
        next_is_footway = next_edge_type in {"footway", "steps", "path"}
        penalty = 15.0 if (prev_is_road and next_is_footway) else 0.0
        eff_angle = angle + penalty

        if eff_angle < min_eff_angle:
            min_eff_angle = eff_angle
            best_nbr = nbr

    return best_nbr


def _walk(
    md: mapdata.MapData,
    p: _Path,
    max_m: float,
) -> list[tuple[int, float, tuple[float, float], int]]:
    """Walk straight along road graph up to maximum distance."""
    curr = p.node
    prev = p.prev
    heading = p.heading
    cum_d = 0.0
    visited = set(p.nodes)
    res = []

    while True:
        nxt = _straight_next(md, prev, curr, heading)
        if nxt is None or nxt in visited:
            break
        seg_d = geo.dist_m(md.nodes[curr], md.nodes[nxt])
        if cum_d + seg_d > max_m:
            break
        cum_d += seg_d
        h_vec = geo.to_xy(md.nodes[nxt], md.nodes[curr])
        visited.add(nxt)
        res.append((nxt, cum_d, h_vec, curr))
        prev = curr
        curr = nxt
        heading = h_vec

    return res


def _fit(d: float, lo: float, hi: float) -> float:
    """Compute score based on target distance range."""
    if lo <= d <= hi:
        return 1.0
    if d < lo:
        return max(0.0, 1.0 - (lo - d) / max(lo, 1.0)) if lo > 0 else 1.0
    return max(0.0, 1.0 - (d - hi) / max(hi, 1.0)) if hi > 0 else 0.0


def _route_to_spot(
    md: mapdata.MapData,
    p: _Path,
    ref: str | None,
    spots: list[Any],
    max_len: float,
) -> _Path | None:
    """Connect path to nearest target spot via shortest path."""
    if not spots or not md.nodes:
        return None

    best_path = None
    best_cost = float("inf")
    best_spot = None

    for sp in spots:
        s_at = sp.at if isinstance(sp, Place) else sp
        close_nodes = [n for n in md.nodes if geo.dist_m(s_at, md.nodes[n]) <= 80.0]
        if not close_nodes:
            continue
        tgt_node = min(close_nodes, key=lambda n: geo.dist_m(s_at, md.nodes[n]))

        try:
            path_nodes = nx.shortest_path(md.graph, source=p.node, target=tgt_node, weight="length_m")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        cost = nx.shortest_path_length(md.graph, source=p.node, target=tgt_node, weight="length_m")
        if cost > max_len:
            continue

        # Count turns with angle >= 30 degrees
        turns_cnt = 0
        full_path = ([p.prev] if p.prev is not None else []) + path_nodes
        for i in range(1, len(full_path) - 1):
            u_pt = md.nodes[full_path[i - 1]]
            v_pt = md.nodes[full_path[i]]
            w_pt = md.nodes[full_path[i + 1]]
            v1 = geo.to_xy(v_pt, u_pt)
            v2 = geo.to_xy(w_pt, v_pt)
            if abs(geo.turn_angle_deg(v1, v2)) >= 30.0:
                turns_cnt += 1

        if turns_cnt <= ROUTE_MAX_TURNS and cost < best_cost:
            best_cost = cost
            best_path = path_nodes
            best_spot = sp

    if best_path and len(best_path) >= 2:
        new_nodes = p.nodes + best_path[1:]
        new_prev = new_nodes[-2]
        new_h = geo.to_xy(md.nodes[new_nodes[-1]], md.nodes[new_prev])
        new_marks = dict(p.marks)
        if ref:
            new_marks[ref] = best_spot
        return _Path(
            nodes=new_nodes,
            prev=new_prev,
            heading=new_h,
            score=p.score + 1.0,
            seg_start=len(new_nodes) - 1,
            relax=list(p.relax) + ["route_to_landmark"],
            marks=new_marks,
        )

    return None


def _straight(
    md: mapdata.MapData,
    p: _Path,
    step: dict,
    rng: tuple[float, float],
    nxt: dict | None,
    relax: tuple[str, ...],
    landmark_at: dict[str, list[Place]] | None,
) -> list[_Path]:
    """Expand path straight forward according to step instructions."""
    lo, hi = rng
    if "distance" in relax:
        lo = lo / 1.5
        hi = hi * 1.5

    until = step.get("until")
    if isinstance(until, dict):
        ref = until.get("landmark_ref")
        spots = (landmark_at or {}).get(ref, [])
        if spots:
            walked = _walk(md, p, max(hi, 600.0) + 200.0)
            matched_node = None
            matched_spot = None
            for n, d, h_vec, prev_n in walked:
                for sp in spots:
                    sp_at = sp.at if isinstance(sp, Place) else sp
                    if geo.dist_m(sp_at, md.nodes[n]) <= MATCH_M:
                        matched_node = n
                        matched_spot = sp
                        break
                if matched_node is not None:
                    break

            if matched_node is not None:
                stop_idx = p.nodes.index(p.node)
                path_addition = []
                for n, d, h_vec, prev_n in walked:
                    path_addition.append(n)
                    if n == matched_node:
                        break
                new_nodes = p.nodes + path_addition
                new_prev = new_nodes[-2] if len(new_nodes) >= 2 else None
                new_h = geo.to_xy(md.nodes[new_nodes[-1]], md.nodes[new_prev]) if new_prev else p.heading
                new_marks = dict(p.marks)
                new_marks[ref] = matched_spot
                return [
                    _Path(
                        nodes=new_nodes,
                        prev=new_prev,
                        heading=new_h,
                        score=p.score + 2.0,
                        seg_start=len(new_nodes) - 1,
                        relax=list(p.relax),
                        marks=new_marks,
                    )
                ]

            routed = _route_to_spot(md, p, ref, spots, max(hi, 600.0) + 200.0)
            if routed is not None:
                return [routed]
            return []

    # Check if next step is turn
    if nxt and nxt.get("action") in TURNS:
        d_class = step.get("distance_class", "none")
        if d_class == "none" or (lo == 0.0 and hi == 0.0):
            return [p]
        return [dataclasses.replace(p, pending=(lo, hi))]

    # Walk straight with stops
    walk_limit = max(hi * 1.5 if hi > 0 else 300.0, 300.0)
    walked = _walk(md, p, walk_limit)
    if not walked:
        return [p]

    in_range = [item for item in walked if lo <= item[1] <= (hi if hi > 0 else 300.0)]
    if not in_range:
        in_range = walked

    stride = max(1, len(in_range) // MAX_STOPS)
    sampled = in_range[::stride][:MAX_STOPS]
    res = []
    for n, d, h_vec, prev_n in sampled:
        idx = [item[0] for item in walked].index(n)
        new_nodes = p.nodes + [item[0] for item in walked[: idx + 1]]
        sc = p.score + _fit(d, rng[0], rng[1])
        new_rel = list(p.relax)
        if d < rng[0] or (rng[1] > 0 and d > rng[1]):
            new_rel.append("distance*1.5")
        res.append(
            _Path(
                nodes=new_nodes,
                prev=new_nodes[-2],
                heading=h_vec,
                score=sc,
                seg_start=len(new_nodes) - 1,
                relax=new_rel,
                marks=dict(p.marks),
            )
        )

    return res


def _turn(
    md: mapdata.MapData,
    p: _Path,
    step: dict,
    relax: tuple[str, ...],
    start_node: int | None,
    landmark_at: dict[str, list[Place]] | None,
) -> list[_Path]:
    """Turn left, right, or enter alley at target ordinal branch."""
    action = step.get("action", "")
    target_side = TURNS.get(action)
    if "side" in relax:
        target_side = None

    target_type = step.get("target_type") or ("alley" if action == "enter_alley" else None)
    target_ordinal = step.get("ordinal", 1)

    # Check count_from landmark condition
    tt = step.get("turn_target") or {}
    cf = tt.get("count_from") or ""
    after_ref = cf.split(":", 1)[1] if cf.startswith("after:") else None

    # Collect branches along the straight road from seg_start
    road_walk = _walk(md, dataclasses.replace(p, node=p.nodes[p.seg_start]), 1000.0)
    road_nodes = [p.nodes[p.seg_start]] + [item[0] for item in road_walk]

    branches = []
    after_passed = after_ref is None or after_ref in p.marks

    for i in range(len(road_nodes)):
        curr_n = road_nodes[i]
        prev_n = road_nodes[i - 1] if i > 0 else p.prev
        curr_ll = md.nodes[curr_n]

        if not after_passed and after_ref:
            spots = (landmark_at or {}).get(after_ref, [])
            for sp in spots:
                sp_at = sp.at if isinstance(sp, Place) else sp
                if geo.dist_m(sp_at, curr_ll) <= MATCH_M:
                    after_passed = True
                    break

        if not after_passed:
            continue

        if prev_n is not None:
            road_h = geo.to_xy(curr_ll, md.nodes[prev_n])
        else:
            road_h = p.heading

        next_n = road_nodes[i + 1] if i + 1 < len(road_nodes) else None
        for nbr in md.neighbors(curr_n):
            if nbr == prev_n or nbr == next_n:
                continue

            vec = geo.to_xy(md.nodes[nbr], curr_ll)
            angle = geo.turn_angle_deg(road_h, vec)
            abs_a = abs(angle)
            if not (30.0 <= abs_a <= 150.0):
                continue

            b_side = geo.side_of(road_h, vec)
            if target_side is not None and b_side != target_side:
                continue

            # Determine branch edge
            b_edge = None
            for e in md.edges:
                if (e.u == curr_n and e.v == nbr) or (e.v == curr_n and e.u == nbr):
                    b_edge = e
                    break

            if target_type == "alley":
                if not b_edge or b_edge.edge_type not in ALLEY:
                    continue
                if b_edge.edge_type == "footway":
                    f_len = geo.dist_m(curr_ll, md.nodes[nbr])
                    if f_len < FOOTWAY_MIN_M:
                        continue
            elif target_type == "intersection":
                # Check major road intersection
                has_major = any(
                    e.edge_type in MAJOR
                    for e in md.edges
                    if (e.u == curr_n or e.v == curr_n) and e != b_edge
                )
                if not has_major:
                    continue

            branches.append((curr_n, nbr, vec, abs_a, b_edge.way_id if b_edge else None, i))

    res = []
    ordinals_to_try = [(target_ordinal, "")]
    if "ordinal" in relax:
        if target_ordinal > 1:
            ordinals_to_try.append((target_ordinal - 1, "ordinal-1"))
        ordinals_to_try.append((target_ordinal + 1, "ordinal+1"))

    for ord_val, ord_tag in ordinals_to_try:
        if 1 <= ord_val <= len(branches):
            curr_n, nbr, vec, abs_a, way_id, r_idx = branches[ord_val - 1]
            idx_in_nodes = p.nodes.index(curr_n) if curr_n in p.nodes else len(p.nodes) - 1
            if curr_n in p.nodes:
                new_nodes = p.nodes[: idx_in_nodes + 1] + [nbr]
            else:
                new_nodes = p.nodes + road_nodes[1 : r_idx + 1] + [nbr]

            sc = p.score + 1.0
            if abs_a > 120.0:
                sc -= 0.3
            if p.pending:
                lo, hi = p.pending
                d_to_turn = sum(
                    geo.dist_m(md.nodes[new_nodes[j]], md.nodes[new_nodes[j + 1]])
                    for j in range(p.seg_start, len(new_nodes) - 1)
                )
                sc += _fit(d_to_turn, lo, hi)

            new_rel = list(p.relax)
            if ord_tag:
                new_rel.append(ord_tag)

            res.append(
                _Path(
                    nodes=new_nodes,
                    prev=curr_n,
                    heading=vec,
                    score=sc,
                    seg_start=len(new_nodes) - 1,
                    alley_way=way_id,
                    alley_entry=md.nodes[curr_n],
                    relax=new_rel,
                    marks=dict(p.marks),
                )
            )

    return res


def _beam(paths: list[_Path]) -> list[_Path]:
    """Filter paths to highest scoring states within beam limit."""
    best: dict[tuple, _Path] = {}
    for p in paths:
        key = (p.node, p.prev, p.alley_way)
        if key not in best or p.score > best[key].score:
            best[key] = p
    sorted_paths = sorted(best.values(), key=lambda p: p.score, reverse=True)
    return sorted_paths[:BEAM]


def _run(
    pr: ParseResult,
    md: mapdata.MapData,
    start_node: int | None,
    steps: list[dict],
    relax: tuple[str, ...],
    landmark_at: dict[str, list[Place]] | None,
) -> list[_Path]:
    """Simulate route graph traversal across parsed steps."""
    if start_node is None or not md or start_node not in md.nodes:
        return []

    toward_deg = _toward_deg(pr, md, md.nodes[start_node], landmark_at)
    initial_paths = []
    for nbr in md.neighbors(start_node):
        h = geo.to_xy(md.nodes[nbr], md.nodes[start_node])
        if toward_deg is not None:
            bearing = geo.bearing_deg(md.nodes[start_node], md.nodes[nbr])
            diff = abs(bearing - toward_deg) % 360.0
            if diff > 180.0:
                diff = 360.0 - diff
            if diff > 70.0:
                continue

        initial_paths.append(
            _Path(
                nodes=[start_node, nbr],
                prev=start_node,
                heading=h,
                score=0.0,
                seg_start=0,
                relax=list(relax),
            )
        )

    if not initial_paths:
        return []

    cur_paths = initial_paths
    active_steps = [s for s in steps if s.get("action") != "arrive"]

    for i, step in enumerate(active_steps):
        act = step.get("action", "")
        nxt = active_steps[i + 1] if i + 1 < len(active_steps) else None

        if act in TURNS:
            next_paths = []
            for p in cur_paths:
                next_paths.extend(_turn(md, p, step, relax, start_node, landmark_at))
            cur_paths = _beam(next_paths)
        else:
            rng = parse.distance_range(step.get("distance_class", "none"))
            next_paths = []
            for p in cur_paths:
                next_paths.extend(_straight(md, p, step, rng, nxt, relax, landmark_at))
            cur_paths = _beam(next_paths)

        if not cur_paths:
            break

    return cur_paths


def target_kind(pr: ParseResult) -> str:
    """Normalize target kind considering generic business naming."""
    t = pr.raw.get("target") or {}
    k = t.get("kind", "building")
    if k == "named_place" and not t.get("specific", True):
        return "building"
    return k


def crossed_roads(
    md: mapdata.MapData,
    a_xy: tuple[float, float],
    b_xy: tuple[float, float],
) -> set[int]:
    """Find set of road way_ids intersected by line segment between a_xy and b_xy."""
    p1 = shapely.geometry.Point(a_xy)
    p2 = shapely.geometry.Point(b_xy)
    seg = shapely.geometry.LineString([p1, p2])
    res = set()

    for e in md.edges:
        if e.edge_type in {"footway", "service", "steps", "path", "cycleway", "corridor", "bridleway"}:
            continue
        u_pt = md.nodes.get(e.u)
        v_pt = md.nodes.get(e.v)
        if not u_pt or not v_pt:
            continue
        u_xy = geo.to_xy(u_pt, md.origin)
        v_xy = geo.to_xy(v_pt, md.origin)
        edge_line = shapely.geometry.LineString([u_xy, v_xy])
        if seg.intersects(edge_line):
            res.add(e.way_id)

    return res


def relation_ok(
    md: mapdata.MapData,
    l_at: LatLng,
    p_xy: Any,
    relation: str,
) -> tuple[bool, float, int]:
    """Verify spatial relation across or beside road."""
    l_xy = geo.to_xy(l_at, md.origin)
    target_xy = geo.to_xy(p_xy, md.origin) if isinstance(p_xy, LatLng) else p_xy
    target_ll = p_xy if isinstance(p_xy, LatLng) else geo.to_ll(p_xy[0], p_xy[1], md.origin)
    d = geo.dist_m(l_at, target_ll)
    crossed = crossed_roads(md, l_xy, target_xy)
    num_crossed = len(crossed)

    if relation == "across":
        ok = (1 <= num_crossed <= 2) and (6.0 <= d <= ACROSS_MAX_M)
    elif relation == "beside":
        ok = (num_crossed == 0) and (d <= BESIDE_MAX_M)
    else:
        ok = (d <= 25.0)

    return ok, d, num_crossed


def _owner_way(md: mapdata.MapData, at: LatLng) -> mapdata.Building | None:
    """Find nearest building polygon containing or within 40m of at."""
    if not md or not md.buildings:
        return None
    pt_xy = geo.to_xy(at, md.origin)
    pt = shapely.geometry.Point(pt_xy)
    best_b = None
    min_dist = 40.0

    b_iter = md.buildings.values() if isinstance(md.buildings, dict) else md.buildings
    for b in b_iter:
        poly_pts = [geo.to_xy(p, md.origin) for p in b.ring]
        if len(poly_pts) < 3:
            continue
        poly = shapely.geometry.Polygon(poly_pts)
        if poly.contains(pt):
            return b
        d = poly.distance(pt)
        if d <= min_dist:
            min_dist = d
            best_b = b

    return best_b


def _relation_cands(
    md: mapdata.MapData,
    p: _Path,
    lspot: Any,
    relation: str,
    spots: list[Place] | None,
) -> list[Candidate]:
    """Find building candidates across or beside landmark."""
    l_at = lspot.at if isinstance(lspot, Place) else lspot
    l_owner = _owner_way(md, l_at)
    excluded = {l_owner.way_id} if l_owner else set()
    cands = []
    path_coords = [[md.nodes[n].lng, md.nodes[n].lat] for n in p.nodes]

    if spots:
        for sp in spots:
            b = _owner_way(md, sp.at)
            if b and b.way_id not in excluded:
                ok, dist, crossed = relation_ok(md, l_at, b.centroid, relation)
                if ok:
                    score = p.score + 1.0 - 0.002 * dist
                    ring_coords = [[pt.lng, pt.lat] for pt in b.ring]
                    c = Candidate(
                        id=f"C{len(cands)+1}",
                        type="building",
                        osm_id=b.way_id,
                        building_way_id=b.way_id,
                        name=sp.name,
                        centroid=b.centroid,
                        geometry={"type": "Polygon", "coordinates": [ring_coords]},
                        route_score=score,
                        relaxations=list(p.relax),
                        path=path_coords,
                    )
                    cands.append(c)

    if not cands and md.buildings:
        b_iter = md.buildings.values() if isinstance(md.buildings, dict) else md.buildings
        for b in b_iter:
            if b.way_id in excluded:
                continue
            ok, dist, crossed = relation_ok(md, l_at, b.centroid, relation)
            if ok:
                score = p.score + 0.5 - 0.002 * dist
                ring_coords = [[pt.lng, pt.lat] for pt in b.ring]
                c = Candidate(
                    id=f"C{len(cands)+1}",
                    type="building",
                    osm_id=b.way_id,
                    building_way_id=b.way_id,
                    name=b.name,
                    centroid=b.centroid,
                    geometry={"type": "Polygon", "coordinates": [ring_coords]},
                    route_score=score,
                    relaxations=list(p.relax),
                    path=path_coords,
                )
                cands.append(c)

    cands.sort(key=lambda c: c.route_score, reverse=True)
    return cands[:3]


def _building_cands(
    md: mapdata.MapData,
    p: _Path,
    side: str | None,
) -> list[Candidate]:
    """Find building candidates along final segment of path."""
    cands = []
    if not md.buildings:
        return cands

    stop_ll = md.nodes[p.node]
    stop_xy = geo.to_xy(stop_ll, md.origin)
    path_coords = [[md.nodes[n].lng, md.nodes[n].lat] for n in p.nodes]

    nearby = []
    b_iter = md.buildings.values() if isinstance(md.buildings, dict) else md.buildings
    for b in b_iter:
        b_xy = geo.to_xy(b.centroid, md.origin)
        d = math.hypot(b_xy[0] - stop_xy[0], b_xy[1] - stop_xy[1])
        if d <= 80.0:
            rel_vec = (b_xy[0] - stop_xy[0], b_xy[1] - stop_xy[1])
            b_side = geo.side_of(p.heading, rel_vec)
            side_match = (side is not None and b_side == side)
            score = p.score + (0.5 if side_match else 0.0) - 0.005 * d
            nearby.append((score, d, side_match, b))

    if side:
        nearby.sort(key=lambda item: (1 if item[2] else 0, item[0]), reverse=True)
    else:
        nearby.sort(key=lambda item: item[0], reverse=True)

    for rank_idx, (score, dist, side_match, b) in enumerate(nearby[:3]):
        ring_coords = [[pt.lng, pt.lat] for pt in b.ring]
        c = Candidate(
            id=f"C{rank_idx+1}",
            type="building",
            osm_id=b.way_id,
            building_way_id=b.way_id,
            name=b.name,
            centroid=b.centroid,
            geometry={"type": "Polygon", "coordinates": [ring_coords]},
            route_score=score,
            relaxations=list(p.relax),
            path=path_coords,
        )
        cands.append(c)

    return cands


def _alley_cands(
    md: mapdata.MapData,
    p: _Path,
    side: str | None,
    spots: list[Place] | None,
) -> list[Candidate]:
    """Return alley candidate from path turn."""
    if p.alley_way:
        entrance_ll = p.alley_entry or md.nodes[p.node]
        path_coords = [[md.nodes[n].lng, md.nodes[n].lat] for n in p.nodes]
        return [
            Candidate(
                id="C1",
                type="alley",
                osm_id=p.alley_way,
                entrance=entrance_ll,
                centroid=entrance_ll,
                route_score=p.score,
                relaxations=list(p.relax),
                path=path_coords,
            )
        ]
    return []


def _intersection_cands(md: mapdata.MapData, p: _Path) -> list[Candidate]:
    """Find intersection candidates near path stop node."""
    stop_ll = md.nodes[p.node]
    path_coords = [[md.nodes[n].lng, md.nodes[n].lat] for n in p.nodes]
    cands = []
    for n_id, n_ll in md.nodes.items():
        if geo.dist_m(stop_ll, n_ll) <= 30.0:
            if len(md.neighbors(n_id)) >= 3:
                cands.append(
                    Candidate(
                        id=f"C{len(cands)+1}",
                        type="intersection",
                        osm_id=n_id,
                        centroid=n_ll,
                        route_score=p.score,
                        relaxations=list(p.relax),
                        path=path_coords,
                    )
                )
    return cands[:3]


def _point_cands(
    md: mapdata.MapData,
    p: _Path,
    spots: list[Any] | None,
) -> list[Candidate]:
    """Generate point candidate from projected spot or stop node."""
    path_coords = [[md.nodes[n].lng, md.nodes[n].lat] for n in p.nodes]
    if spots:
        sp = spots[0]
        sp_at = sp.at if isinstance(sp, Place) else sp
        return [
            Candidate(
                id="C1",
                type="point",
                centroid=sp_at,
                route_score=p.score,
                relaxations=list(p.relax),
                path=path_coords,
            )
        ]
    stop_ll = md.nodes[p.node]
    return [
        Candidate(
            id="C1",
            type="point",
            centroid=stop_ll,
            route_score=p.score,
            relaxations=list(p.relax),
            path=path_coords,
        )
    ]


def _targets(
    pr: ParseResult,
    md: mapdata.MapData,
    paths: list[_Path],
    relax: tuple[str, ...],
    landmark_at: dict[str, list[Place]] | None,
) -> list[Candidate]:
    """Generate target candidates for paths according to target description."""
    t_raw = pr.raw.get("target") or {}
    kind = target_kind(pr)
    relation = t_raw.get("relation")
    side = t_raw.get("side")
    ref = t_raw.get("landmark_ref")
    target_spots = (landmark_at or {}).get("__target__") or (landmark_at or {}).get(ref)

    cands: list[Candidate] = []

    if (kind in ("building", "point")) and target_spots and relation not in ("across", "beside"):
        sp = target_spots[0]
        sp_at = sp.at if isinstance(sp, Place) else sp
        for p in paths:
            if geo.dist_m(md.nodes[p.node], sp_at) > TARGET_ROUTE_MIN_M:
                routed = _route_to_spot(md, p, ref, [sp], 1000.0)
                if routed:
                    cands.extend(_point_cands(md, routed, [sp]) if kind == "point" else _building_cands(md, routed, side))
        if cands:
            return cands

    for p in paths:
        if kind == "building":
            if relation in ("across", "beside"):
                if ref and ref in p.marks:
                    cands.extend(_relation_cands(md, p, p.marks[ref], relation, target_spots))
                elif ref and landmark_at and landmark_at.get(ref):
                    l_place = landmark_at[ref][0]
                    lm_def = next((item for item in (pr.raw.get("landmarks") or []) if item.get("id") == ref), {})
                    if lm_def.get("poi", {}).get("specific", True):
                        if geo.dist_m(md.nodes[p.node], l_place.at) <= NEAR_END_M:
                            rel_p = dataclasses.replace(p, relax=list(p.relax) + ["landmark_near_end"])
                            cands.extend(_relation_cands(md, rel_p, l_place, relation, target_spots))
                elif "relation" in relax:
                    d_walk = sum(
                        geo.dist_m(md.nodes[p.nodes[j]], md.nodes[p.nodes[j + 1]])
                        for j in range(len(p.nodes) - 1)
                    )
                    if d_walk >= RELATION_MIN_WALK_M:
                        start_ll = md.nodes[p.nodes[0]]
                        b_iter = md.buildings.values() if isinstance(md.buildings, dict) else md.buildings
                        for b in b_iter:
                            if geo.dist_m(start_ll, b.centroid) > RELATION_START_EXCLUDE_M:
                                if geo.dist_m(md.nodes[p.node], b.centroid) <= 80.0:
                                    ring_coords = [[pt.lng, pt.lat] for pt in b.ring]
                                    path_coords = [[md.nodes[n].lng, md.nodes[n].lat] for n in p.nodes]
                                    cands.append(
                                        Candidate(
                                            id=f"C{len(cands)+1}",
                                            type="building",
                                            osm_id=b.way_id,
                                            building_way_id=b.way_id,
                                            name=b.name,
                                            centroid=b.centroid,
                                            geometry={"type": "Polygon", "coordinates": [ring_coords]},
                                            route_score=-1.0,
                                            relaxations=list(p.relax) + ["relation"],
                                            path=path_coords,
                                        )
                                    )
            else:
                cands.extend(_building_cands(md, p, side))
        elif kind == "alley":
            cands.extend(_alley_cands(md, p, side, target_spots))
        elif kind == "intersection":
            cands.extend(_intersection_cands(md, p))
        elif kind == "point":
            cands.extend(_point_cands(md, p, target_spots))

    return cands


def _rank(cands: list[Candidate], limit: int = 5) -> list[Candidate]:
    """Deduplicate candidates and rank descending by route score."""
    best: dict[tuple, Candidate] = {}
    for c in cands:
        key = (c.type, c.osm_id or (round(c.centroid.lat, 5), round(c.centroid.lng, 5)))
        if key not in best or c.route_score > best[key].route_score:
            best[key] = c

    sorted_cands = sorted(best.values(), key=lambda c: c.route_score, reverse=True)
    res = sorted_cands[:limit]
    for i, c in enumerate(res):
        c.id = f"C{i+1}"
    return res


def expand(
    pr: ParseResult,
    md: mapdata.MapData,
    start: Anchor,
    landmark_at: dict[str, list[Place]] | None = None,
) -> list[Candidate]:
    """Expand routes across progressive relaxation levels until candidates found."""
    st_node = start.node_id or _start_node(md, start.at)
    steps = pr.raw.get("steps") or []

    for level in RELAX_LEVELS:
        paths = _run(pr, md, st_node, steps, level, landmark_at)
        cands = _targets(pr, md, paths, level, landmark_at)
        if cands:
            return _rank(cands, limit=MAX_CANDIDATES)

    return []


def core_terms(phrase: str) -> list[str]:
    """Extract core terms from a category phrase for search."""
    tokens = [t.strip() for t in phrase.split() if t.strip()]
    terms = []
    for tok in tokens:
        stripped = re.sub(r"(집|점|전문점|가게|식당|음식점|거리|요리)$", "", tok)
        if stripped and stripped not in terms:
            terms.append(stripped)
    for tok in tokens:
        if tok not in terms and tok:
            terms.append(tok)
    return terms


def _poi_pool(
    budget: _Budget,
    phrase: str,
    category_code: str | None,
    center: LatLng,
    radius_m: int,
) -> list[Place]:
    """Collect candidate POIs matching query phrase and category code."""
    places = []
    seen = set()

    hits = _kw(budget, phrase, near=center, radius_m=radius_m)
    for p in hits:
        if p.name not in seen:
            seen.add(p.name)
            places.append(p)

    for term in core_terms(phrase):
        hits = _kw(budget, term, near=center, radius_m=radius_m, category=category_code)
        for p in hits:
            if p.name not in seen:
                seen.add(p.name)
                places.append(p)

    if category_code:
        for page in (1, 2, 3):
            hits = _cat(budget, category_code, near=center, radius_m=radius_m, page=page)
            for p in hits:
                if p.name not in seen:
                    seen.add(p.name)
                    places.append(p)

    return places


def match_terms(
    places: list[Place],
    terms: list[str],
    budget: _Budget,
) -> dict[str, list[Place]]:
    """Match candidate places against semantic terms using LLM or rule fallback."""
    res = {term: [] for term in terms}
    if not places or not terms:
        return res

    try:
        numbered = [f"{i+1}: {p.name} | {p.category_name or p.category}" for i, p in enumerate(places)]
        prompt_text = P_POI_MATCH + "\n".join(numbered) + "\n설명어: " + ", ".join(terms)
        schema = {
            "type": "OBJECT",
            "properties": {
                term: {"type": "ARRAY", "items": {"type": "INTEGER"}}
                for term in terms
            },
        }
        resp = llm.generate_json("poi_match", "flash_lite", prompt_text, schema, timeout_ms=3000)
        if isinstance(resp, dict):
            for term, indices in resp.items():
                if term in res and isinstance(indices, list):
                    for idx in indices:
                        if isinstance(idx, int) and 1 <= idx <= len(places):
                            res[term].append(places[idx - 1])
            return res
    except Exception:
        pass

    for term in terms:
        clean_term = re.sub(r"(집|가게|전문점)$", "", term)
        matched = []
        for p in places:
            text = f"{p.name} {p.category_name} {p.category}"
            if clean_term in text or term in text:
                matched.append(p)
        res[term] = matched

    return res


def category_hits(places: list[Place], code_or_name: str | None) -> list[Place]:
    """Filter places matching category code or keyword."""
    if not code_or_name:
        return places
    res = []
    clean = re.sub(r"(집|점|전문점)$", "", code_or_name)
    for p in places:
        cat = p.category_name or p.category or ""
        name = p.name or ""
        if code_or_name in cat or clean in cat or clean in name:
            res.append(p)
    return res


def _landmark_positions(
    pr: ParseResult,
    md: mapdata.MapData,
    center: LatLng,
    radius: int,
    budget: _Budget,
) -> dict[str, list[Place]]:
    """Resolve geographic positions for route and target landmarks."""
    res: dict[str, list[Place]] = {}
    landmarks = pr.raw.get("landmarks") or []

    for lm in landmarks:
        ref_id = lm.get("id")
        poi = lm.get("poi") or {}
        name = poi.get("name") or lm.get("evidence", "")
        specific = poi.get("specific", True)
        if not ref_id:
            continue

        if specific:
            hits = _kw(budget, name, near=center, radius_m=radius)
            res[ref_id] = hits
        else:
            cat_code = poi.get("category_code")
            pool = _poi_pool(budget, name, cat_code, center, radius)
            matched = match_terms(pool, [name], budget)
            hits = matched.get(name, [])
            if not hits:
                hits = category_hits(pool, cat_code or name)
            if not hits and specific:
                google_hits = google_places_of_type(name, center, radius)
                hits = google_hits
            res[ref_id] = hits

    target = pr.raw.get("target") or {}
    tgt_name = target.get("name")
    tgt_ref = target.get("landmark_ref")
    if tgt_ref and tgt_ref in res:
        res["__target__"] = res[tgt_ref]
    elif tgt_name:
        specific = target.get("specific", False)
        cat_code = target.get("category_code")
        pool = _poi_pool(budget, tgt_name, cat_code, center, radius)
        matched = match_terms(pool, [tgt_name], budget)
        hits = matched.get(tgt_name, [])
        if not hits:
            hits = category_hits(pool, cat_code or tgt_name)
        res["__target__"] = hits

    return res


def _walk_fit(
    md: mapdata.MapData,
    steps: list[dict],
    place: Place,
    start_ll: LatLng,
) -> float:
    """Evaluate how well a place aligns with walking direction and side."""
    p_xy = geo.to_xy(place.at, md.origin)
    start_xy = geo.to_xy(start_ll, md.origin)
    dx = p_xy[0] - start_xy[0]
    dy = p_xy[1] - start_xy[1]

    side_pref = None
    for s in steps:
        if s.get("side"):
            side_pref = s.get("side")
            break

    score = 1.0
    if side_pref:
        actual_side = "left" if dy > 0 else "right"
        if actual_side == side_pref:
            score += 0.5
        else:
            score -= 0.5

    detour = abs(dy)
    if detour > MATCH_M:
        score -= min(1.0, (detour - MATCH_M) / 100.0)

    return score


def _named_candidates(
    pr: ParseResult,
    md: mapdata.MapData,
    window: Window | None,
    anchors: list[Anchor],
    budget: _Budget,
) -> list[Candidate]:
    """Find candidate buildings and locations for specifically named target places."""
    target_raw = pr.raw.get("target") or {}
    name = target_raw.get("name") or target_raw.get("evidence", "")
    if not name:
        return []

    def is_hq(p: Place) -> bool:
        if "본사" in p.name:
            return True
        cat = p.category_name or p.category or ""
        if "기업" in cat or ("연구소" in cat and "본사" in p.address):
            return True
        return False

    center = anchors[0].at if anchors else (window.center if window else config.VENUE)
    radius = 1500

    hits = _kw(budget, name, near=center, radius_m=radius)
    valid_places = [p for p in hits if not is_hq(p)]

    clean_target = name.replace(" ", "")
    valid_places.sort(
        key=lambda p: (0 if clean_target in p.name.replace(" ", "") or p.name.replace(" ", "") in clean_target else 1)
    )

    if len(valid_places) > 1 and anchors and pr.raw.get("steps"):
        start_ll = anchors[0].at
        scored_places = []
        for p in valid_places:
            fit_score = _walk_fit(md, pr.raw.get("steps", []), p, start_ll)
            scored_places.append((fit_score, p))
        scored_places.sort(key=lambda item: item[0], reverse=True)
        valid_places = [item[1] for item in scored_places]
        budget.notes.append("걷기 적합도로 정렬")

    if not valid_places:
        g_place = google_place(budget, name, near=center)
        if g_place and not is_hq(g_place):
            valid_places = [g_place]

    if not valid_places:
        en_places = en_resolve(budget, name, center)
        valid_places = [p for p in en_places if not is_hq(p)]

    cands = []
    for rank_idx, place in enumerate(valid_places[:3]):
        score = 1.0 if rank_idx == 0 else 0.5
        b = _owner_way(md, place.at)
        dist_to_anchor = geo.dist_m(center, place.at) if anchors else 0.0
        if dist_to_anchor <= 500:
            score += 2.0

        if b:
            ring_coords = [[p.lng, p.lat] for p in b.ring]
            geom = {"type": "Polygon", "coordinates": [ring_coords]}
            cand = Candidate(
                id=f"C{rank_idx+1}",
                type="named_place",
                osm_id=b.way_id,
                building_way_id=b.way_id,
                name=place.name,
                centroid=b.centroid,
                geometry=geom,
                route_score=score,
            )
        else:
            geom = {"type": "Point", "coordinates": [place.at.lng, place.at.lat]}
            cand = Candidate(
                id=f"C{rank_idx+1}",
                type="named_place",
                name=place.name,
                centroid=place.at,
                geometry=geom,
                route_score=score,
            )
        cands.append(cand)

    return cands


def _category_candidates(
    pr: ParseResult,
    md: mapdata.MapData,
    window: Window | None,
    anchors: list[Anchor],
    budget: _Budget,
) -> list[Candidate]:
    """Find building candidates for generic category target."""
    target_raw = pr.raw.get("target") or {}
    name = target_raw.get("name", "")
    cat_code = target_raw.get("category_code")
    center = anchors[0].at if anchors else (window.center if window else config.VENUE)
    places = _poi_pool(budget, name, cat_code, center, 1000)
    cands = []
    for p in places[:CATEGORY_CANDS]:
        b = _owner_way(md, p.at)
        if b:
            c = Candidate(
                id=f"C{len(cands)+1}",
                type="building",
                osm_id=b.way_id,
                building_way_id=b.way_id,
                name=p.name,
                centroid=b.centroid,
            )
            fit, why = _road_fit(md, pr, c)
            c.route_score += fit
            cands.append(c)
    return cands


def _relation_and_target_evidence(
    md: mapdata.MapData,
    cand: Candidate,
    lspot: Place | LatLng | None,
    target_spots: list[Place] | None,
    relation: str,
) -> list[Evidence]:
    """Verify geometric relation and presence of target business at candidate building."""
    evs = []
    if lspot is not None:
        l_at = lspot.at if isinstance(lspot, Place) else lspot
        ok, dist, crossed = relation_ok(md, l_at, cand.centroid, relation)
        ev_rel = Evidence(
            id=f"E-rel-{cand.id}",
            kind="relation",
            candidate_id=cand.id,
            supports=ok,
            summary_ko=f"{getattr(lspot, 'name', '기준점')} {relation} 확인 ({int(dist)}m)",
            data={"relation_ok": ok, "lnglat": [round(l_at.lng, 7), round(l_at.lat, 7)]},
        )
        evs.append(ev_rel)

    if target_spots:
        found_target = False
        target_name = ""
        for spot in target_spots:
            owner = _owner_way(md, spot.at)
            d = geo.dist_m(cand.centroid, spot.at) if cand.centroid else 9999.0
            b_way = getattr(cand, "building_way_id", None)
            if (owner and (owner.way_id == cand.osm_id or (b_way and owner.way_id == b_way))) or d <= TARGET_BUILDING_M:
                found_target = True
                target_name = spot.name
                break
        ev_tgt = Evidence(
            id=f"E-tgt-{cand.id}",
            kind="kakao",
            candidate_id=cand.id,
            supports=found_target,
            summary_ko=f"목표 업종 {target_name} 확인" if found_target else "목표 업종 확인 불가",
            data={"target": True, "place": target_name},
        )
        evs.append(ev_tgt)

    return evs


def _road_fit(md: mapdata.MapData, pr: ParseResult, cand_pt_or_cand: Any) -> tuple[float, str]:
    """Evaluate whether candidate is on a big road frontage for follow_road steps."""
    steps = pr.raw.get("steps") or []
    has_follow_road = any(s.get("action") == "follow_road" for s in steps)
    if not has_follow_road:
        return 0.0, ""

    if isinstance(cand_pt_or_cand, LatLng):
        pt_xy = geo.to_xy(cand_pt_or_cand, md.origin)
        target_geom = shapely.geometry.Point(pt_xy)
    elif isinstance(cand_pt_or_cand, Candidate):
        if cand_pt_or_cand.geometry and cand_pt_or_cand.geometry.get("coordinates"):
            coords = cand_pt_or_cand.geometry["coordinates"][0]
            pts = [geo.to_xy(LatLng(lat=c[1], lng=c[0]), md.origin) for c in coords]
            target_geom = shapely.geometry.Polygon(pts)
        else:
            pt_xy = geo.to_xy(cand_pt_or_cand.centroid, md.origin)
            target_geom = shapely.geometry.Point(pt_xy)
    else:
        return 0.0, ""

    min_road_d = float("inf")
    for e in md.edges:
        if e.edge_type in MAJOR:
            u_pt = md.nodes.get(e.u)
            v_pt = md.nodes.get(e.v)
            if u_pt and v_pt:
                u_xy = geo.to_xy(u_pt, md.origin)
                v_xy = geo.to_xy(v_pt, md.origin)
                line = shapely.geometry.LineString([u_xy, v_xy])
                d = target_geom.distance(line)
                if d < min_road_d:
                    min_road_d = d

    if min_road_d <= BIG_ROAD_M:
        return 1.0, "큰길가"
    return -0.5, "큰길에서 벗어남"


def _toward_deg(
    pr: ParseResult,
    md: mapdata.MapData,
    start_pt: LatLng,
    landmark_at: dict[str, list[Place]] | None,
) -> float | None:
    """Compute bearing from start to resolved toward landmark if present."""
    if not landmark_at:
        return None
    ref = None
    for step in pr.raw.get("steps") or []:
        if "toward" in step:
            ref = step.get("toward")
            break
    if not ref:
        ref = pr.raw.get("toward")
    if not ref or ref not in landmark_at or not landmark_at[ref]:
        return None
    places = landmark_at[ref]
    closest = min(places, key=lambda p: geo.dist_m(start_pt, p.at if isinstance(p, Place) else p))
    target_ll = closest.at if isinstance(closest, Place) else closest
    return geo.bearing_deg(start_pt, target_ll)


def view_points(
    md: mapdata.MapData,
    cand: Candidate,
    side: str | None = None,
) -> list[ViewPoint]:
    """Generate up to 2 street-level viewpoints facing candidate from path."""
    res = []
    if cand.type in ("building", "named_place"):
        target_ll = cand.centroid
        if cand.path:
            pts = [LatLng(lat=p[1], lng=p[0]) for p in cand.path]
            closest_idx = min(range(len(pts)), key=lambda i: geo.dist_m(pts[i], target_ll))
            base_pt = pts[closest_idx]
            bearing = geo.bearing_deg(base_pt, target_ll)
            res.append(ViewPoint(at=base_pt, heading_deg=bearing, purpose="target"))
            if closest_idx > 0:
                prev_pt = pts[closest_idx - 1]
                d = geo.dist_m(prev_pt, base_pt)
                if d > 0:
                    frac = min(1.0, 20.0 / d)
                    approach_pt = geo.to_ll(
                        geo.to_xy(base_pt, md.origin)[0] - frac * (geo.to_xy(base_pt, md.origin)[0] - geo.to_xy(prev_pt, md.origin)[0]),
                        geo.to_xy(base_pt, md.origin)[1] - frac * (geo.to_xy(base_pt, md.origin)[1] - geo.to_xy(prev_pt, md.origin)[1]),
                        md.origin,
                    )
                    res.append(ViewPoint(at=approach_pt, heading_deg=bearing, purpose="target"))
    elif cand.type == "alley":
        at_pt = cand.entrance or cand.centroid
        if cand.path and len(cand.path) >= 2:
            last_pt = LatLng(lat=cand.path[-1][1], lng=cand.path[-1][0])
            bearing = geo.bearing_deg(at_pt, last_pt)
            res.append(ViewPoint(at=at_pt, heading_deg=bearing, purpose="target"))
    else:
        at_pt = cand.centroid
        if cand.path and len(cand.path) >= 2:
            prev_pt = LatLng(lat=cand.path[-2][1], lng=cand.path[-2][0])
            bearing = geo.bearing_deg(prev_pt, at_pt)
            res.append(ViewPoint(at=at_pt, heading_deg=bearing, purpose="target"))
    return res[:2]


def _route_evidence(md: mapdata.MapData, cand: Candidate, anchor_name: str) -> Evidence:
    """Generate route summary evidence for candidate."""
    dist_m = 0.0
    if cand.path and len(cand.path) >= 2:
        for i in range(len(cand.path) - 1):
            p1 = LatLng(lat=cand.path[i][1], lng=cand.path[i][0])
            p2 = LatLng(lat=cand.path[i + 1][1], lng=cand.path[i + 1][0])
            dist_m += geo.dist_m(p1, p2)
    rel_str = ",".join(cand.relaxations) if cand.relaxations else "없음"
    summary = f"{anchor_name}에서 경로 {int(dist_m)}m, 완화 {rel_str}, 경로 점수 {cand.route_score:.1f}"
    return Evidence(
        id=f"E-route-{cand.id}",
        kind="route",
        candidate_id=cand.id,
        supports=True,
        summary_ko=summary,
        weight=cand.route_score,
    )


def match_landmarks(
    pr: ParseResult,
    md: mapdata.MapData,
    cands: list[Candidate],
    budget: _Budget | None = None,
    landmark_at: dict[str, list[Place]] | None = None,
) -> list[Evidence]:
    """Match candidate paths and buildings against landmark evidence."""
    evidences: list[Evidence] = []
    b = budget or _Budget()

    target_raw = pr.raw.get("target") or {}
    tgt_ref = target_raw.get("landmark_ref")
    tgt_rel = target_raw.get("relation")

    for cand in cands:
        if tgt_ref or target_raw.get("name"):
            t_name = target_raw.get("name")
            if tgt_ref and landmark_at and tgt_ref in landmark_at:
                t_places = landmark_at[tgt_ref]
            elif landmark_at and "__target__" in landmark_at:
                t_places = landmark_at["__target__"]
            elif b.limit > 0 and b.used < b.limit and t_name:
                t_places = _kw(b, t_name, near=cand.centroid, radius_m=100)
            else:
                t_places = None

            if t_places is None:
                evidences.append(
                    Evidence(
                        id=f"E{len(evidences)+1}",
                        kind="kakao",
                        candidate_id=cand.id,
                        supports=None,
                        data={"target": True},
                    )
                )
            else:
                hit_target = False
                matched_p = None
                for sp in t_places:
                    owner = _owner_way(md, sp.at)
                    d = geo.dist_m(cand.centroid, sp.at) if cand.centroid else 9999.0
                    b_way = getattr(cand, "building_way_id", None)
                    if (owner and (owner.way_id == cand.osm_id or (b_way and owner.way_id == b_way))) or d <= TARGET_BUILDING_M:
                        hit_target = True
                        matched_p = sp
                        break
                if hit_target:
                    cand.route_score += 1.0
                    evidences.append(
                        Evidence(
                            id=f"E{len(evidences)+1}",
                            kind="kakao",
                            candidate_id=cand.id,
                            supports=True,
                            summary_ko=f"건물 내 {matched_p.name}",
                            data={"target": True, "place": matched_p.name},
                        )
                    )
                else:
                    evidences.append(
                        Evidence(
                            id=f"E{len(evidences)+1}",
                            kind="kakao",
                            candidate_id=cand.id,
                            supports=False,
                            data={"target": True},
                        )
                    )

        if tgt_rel in ("across", "beside"):
            lspot = None
            if tgt_ref and landmark_at and tgt_ref in landmark_at and landmark_at[tgt_ref]:
                lspot = landmark_at[tgt_ref][0]
            rel_evs = _relation_and_target_evidence(
                md, cand, lspot, (landmark_at or {}).get("__target__"), tgt_rel
            )
            evidences.extend(rel_evs)

        landmarks = pr.raw.get("landmarks") or []
        for lm in landmarks:
            ref_id = lm.get("id")
            if not ref_id or ref_id == tgt_ref:
                continue
            poi = lm.get("poi") or {}
            lm_name = poi.get("name") or lm.get("evidence", "")

            if b.limit > 0 and b.used >= b.limit:
                evidences.append(
                    Evidence(
                        id=f"E{len(evidences)+1}",
                        kind="kakao",
                        candidate_id=cand.id,
                        supports=None,
                    )
                )
                continue

            if landmark_at and ref_id in landmark_at and landmark_at[ref_id]:
                spots = landmark_at[ref_id]
            else:
                spots = _kw(b, lm_name, near=cand.centroid, radius_m=500) if lm_name else []

            if not spots:
                evidences.append(
                    Evidence(
                        id=f"E{len(evidences)+1}",
                        kind="kakao",
                        candidate_id=cand.id,
                        supports=False,
                    )
                )
                continue

            path_pts = [LatLng(lat=p[1], lng=p[0]) for p in cand.path]
            matched_spots = []
            for sp in spots:
                if any(geo.dist_m(pt, sp.at) <= MATCH_M for pt in path_pts):
                    matched_spots.append(sp)

            if matched_spots:
                clean_lm = lm_name.replace(" ", "")
                matched_spots.sort(
                    key=lambda p: (0 if clean_lm in p.name.replace(" ", "") else 1)
                )
                best_sp = matched_spots[0]
                cand.route_score += 1.0
                evidences.append(
                    Evidence(
                        id=f"E{len(evidences)+1}",
                        kind="kakao",
                        candidate_id=cand.id,
                        supports=True,
                        summary_ko=f"{best_sp.name} 경로 일치",
                        data={"place": best_sp.name, "lnglat": [round(best_sp.at.lng, 7), round(best_sp.at.lat, 7)]},
                    )
                )
            else:
                evidences.append(
                    Evidence(
                        id=f"E{len(evidences)+1}",
                        kind="kakao",
                        candidate_id=cand.id,
                        supports=False,
                    )
                )

    return evidences


def find_candidates(
    pr: ParseResult,
    region_input: str | None = None,
    current: LatLng | None = None,
    md: mapdata.MapData | None = None,
    budget: _Budget | None = None,
    window: Window | None = None,
    anchors: list[Anchor] | None = None,
    *args,
    **kwargs,
) -> dict:
    """Execute complete candidate search pipeline from parse result and geographic context."""
    if isinstance(region_input, Window):
        window = region_input
        region_input = None
    if isinstance(current, list):
        anchors = current
        current = None
    if isinstance(md, mapdata.MapData):
        active_md = md
    else:
        active_md = None
    if budget is None:
        if args and isinstance(args[0], _Budget):
            budget = args[0]
        else:
            budget = _Budget()

    win = window
    if win is None:
        win = resolve_window(pr, region_input, current, budget)

    origin = win.center if win else config.VENUE
    radius = (win.radius_m + 300) if win else 1000
    radius = min(radius, 3300)
    if active_md is None:
        active_md = mapdata.load_area(origin, radius)

    found_anchors = anchors
    q = None
    if found_anchors is None:
        found_anchors, q = resolve_anchors(pr, win, active_md, budget)
        if q is not None:
            return {
                "window": win,
                "anchors": found_anchors,
                "candidates": [],
                "evidence": [],
                "question": q,
                "md": active_md,
                "notes": budget.notes,
                "kakao_calls": budget.used,
                "places": budget.places,
            }

    if found_anchors and active_md:
        a_dist = geo.dist_m(active_md.origin, found_anchors[0].at)
        if a_dist > 500:
            active_md = mapdata.load_area(found_anchors[0].at, 1000)
            found_anchors[0].node_id = _start_node(active_md, found_anchors[0].at)

    anchor_center = found_anchors[0].at if found_anchors else (win.center if win else config.VENUE)
    reach = int(geo.dist_m(origin, anchor_center)) if win else 300
    lm_radius = min(max(reach, 300), 1500)
    landmark_at = _landmark_positions(pr, active_md, anchor_center, lm_radius, budget)

    cands: list[Candidate] = []
    t_kind = target_kind(pr)
    target_raw = pr.raw.get("target") or {}
    if t_kind == "named_place" and target_raw.get("specific", True):
        cands = _named_candidates(pr, active_md, win, found_anchors, budget)
    else:
        for a in (found_anchors[:3] if found_anchors else []):
            if a.node_id is None:
                a.node_id = _start_node(active_md, a.at)
            cands.extend(expand(pr, active_md, a, landmark_at))

    ranked = _rank(cands, limit=8)
    evidences = match_landmarks(pr, active_md, ranked, budget=budget, landmark_at=landmark_at)
    final_cands = _rank(ranked, limit=MAX_CANDIDATES)

    all_evidence = list(evidences)
    for c in final_cands:
        c.view_points = view_points(active_md, c)
        anchor_name = found_anchors[0].name if found_anchors else "출발점"
        route_ev = _route_evidence(active_md, c, anchor_name)
        all_evidence.insert(0, route_ev)

    return {
        "window": win,
        "anchors": found_anchors,
        "candidates": final_cands,
        "evidence": all_evidence,
        "question": None,
        "md": active_md,
        "notes": budget.notes,
        "kakao_calls": budget.used,
        "places": budget.places,
    }


def google_place(*args, **kwargs) -> Place | None:
    """Stub for Google Place grounding."""
    return None


def google_places_of_type(*args, **kwargs) -> list[Place]:
    """Stub for Google Place grounding by type."""
    return []


def prefer_venue(places: list[Place], venue: LatLng = config.VENUE) -> list[Place]:
    """Sort places prioritizing proximity to venue."""
    return sorted(places, key=lambda p: geo.dist_m(p.at, venue))


if __name__ == "__main__":
    from app import parse

    text_input = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "강남역 11번 출구에서 나와 직진하면 왼쪽에 있는 교보문고 건물"
    )
    parsed = parse.parse_text(text_input)
    budget_inst = _Budget()
    win = resolve_window(parsed, None, None, budget_inst)
    active_map = mapdata.load_area(win.center if win else config.VENUE, 1000)
    found_anchors, q = resolve_anchors(parsed, win, active_map, budget_inst)
    result = find_candidates(parsed, win, found_anchors, active_map, budget_inst)
    candidates = result.get("candidates", []) if isinstance(result, dict) else result

    print(f"Window: {win.kind if win else 'none'} ({win.label if win else ''})")
    print(f"Anchors: {[a.name for a in found_anchors]}")
    print(f"Question: {q.intent_ko if q else 'None'}")
    print(f"Candidates: {len(candidates)}")
    for c in candidates[:3]:
        print(f"  - {c.id}: {c.type} {c.name or ''} (osm={c.osm_id or ''}, score={c.route_score:.1f})")
