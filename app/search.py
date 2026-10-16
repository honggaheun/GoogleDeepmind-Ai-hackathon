"""Search stage: resolves search window, start anchors, and candidate paths/buildings from route descriptions."""

from __future__ import annotations

import re
import sys
from typing import Any

from google.genai import types

from app import config, geo, kakao, llm, mapdata
from app.contracts import Anchor, Candidate, LatLng, ParseResult, Place, Question, Window

# Geometry and search thresholds
ALLEY = {"residential", "service", "living_street", "pedestrian", "unclassified", "footway"}
MAJOR = {"motorway", "trunk", "primary", "secondary", "tertiary"}
TURNS = {"turn_left": "left", "turn_right": "right", "enter_alley": None}
GROUND_TIMEOUT_MS = 12_000
GROUND_TYPE_TIMEOUT_MS = 30_000
FOOTWAY_MIN_M = 25
MATCH_M = 40
RELAX_LEVELS = [(), ("ordinal",), ("ordinal", "side"), ("ordinal", "side", "distance")]
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


# Stubs for subsequent sprint integration
def _poi_pool(*args, **kwargs):
    return []


def google_place(*args, **kwargs):
    return None


def google_places_of_type(*args, **kwargs):
    return []


def match_terms(*args, **kwargs):
    return []


def _category_candidates(*args, **kwargs):
    return []


def _landmark_positions(*args, **kwargs):
    return {}


def _named_candidates(*args, **kwargs):
    return []


def _relation_and_target_evidence(*args, **kwargs):
    return "", ""


def _road_fit(*args, **kwargs):
    return 0.0


def _run(*args, **kwargs):
    return []


def _start_node(*args, **kwargs):
    return None


def _targets(*args, **kwargs):
    return []


def _toward_deg(*args, **kwargs):
    return None


def _walk_fit(*args, **kwargs):
    return 0.0


def category_hits(*args, **kwargs):
    return []


def core_terms(*args, **kwargs):
    return []


def expand(*args, **kwargs):
    return []


def find_candidates(
    pr: ParseResult,
    window: Window | None,
    anchors: list[Anchor],
    md: mapdata.MapData | None,
    budget: _Budget | None = None,
) -> list[Candidate]:
    """Find candidate buildings and paths from resolved window and start anchors."""
    return []


def match_landmarks(*args, **kwargs):
    return []


def prefer_venue(*args, **kwargs):
    return []


def view_points(*args, **kwargs):
    return []


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
    candidates = find_candidates(parsed, win, found_anchors, active_map, budget_inst)

    print(f"Window: {win.kind if win else 'none'} ({win.label if win else ''})")
    print(f"Anchors: {[a.name for a in found_anchors]}")
    print(f"Question: {q.intent_ko if q else 'None'}")
    print(f"Candidates: {len(candidates)}")
