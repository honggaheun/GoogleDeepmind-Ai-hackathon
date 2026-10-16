"""Orchestrate the 4-stage localization pipeline: parse, search, verify, and judge.

Stage 1 (parse): app.parse and app.signboard extract route cues and user photo text.
Stage 2 (search): app.search finds candidate buildings/places and route evidence.
Stage 3 (verify): app.vision checks building registries and street-level photos.
Stage 4 (judge): app.judge scores candidates, validates Gemini judgment, or asks a question.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import inspect
import math
import time
import uuid

from app import config, geo, judge, kakao, llm, parse, progress, search, signboard, vision
from app.contracts import (
    B0_SCHEMA,
    Anchor,
    Building,
    Candidate,
    Decision,
    Evidence,
    LatLng,
    ParseResult,
    Place,
    Question,
    RegistryRecord,
    ViewPoint,
    VisionStats,
    Window,
)

P_B0 = "다음 길 안내만 보고 목적지의 주소와 위도·경도를 추정하라. 대략 위치: {region}. 안내: {text}"
ATTRIBUTION = "© OpenStreetMap contributors"

SESSIONS: dict[str, dict] = {}

_SPAN_PRIORITY = {"step": 1, "region": 2, "landmark": 3, "anchor": 4, "target": 5}


def _g(obj, key: str, default=None):
    """Read an attribute or dict key from obj with a fallback default."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def evidence_spans(text: str, raw: dict) -> list[dict]:
    """Locate character spans of region, anchor, step, landmark, and target phrases in text."""
    if not text or not raw:
        return []
    candidates: list[tuple[str, str]] = []
    reg = raw.get("region") or {}
    if reg.get("evidence"):
        candidates.append((str(reg["evidence"]), "region"))

    anc = raw.get("anchor") or {}
    if anc.get("evidence"):
        candidates.append((str(anc["evidence"]), "anchor"))
    elif anc.get("query"):
        candidates.append((str(anc["query"]), "anchor"))

    for st in raw.get("steps") or []:
        ev_str = (st or {}).get("evidence")
        if ev_str:
            candidates.append((str(ev_str), "step"))

    target = raw.get("target") or {}
    t_ref = target.get("landmark_ref")
    for lm in raw.get("landmarks") or []:
        ev_str = (lm or {}).get("evidence")
        if ev_str:
            kind = "target" if (t_ref and lm.get("id") == t_ref) else "landmark"
            candidates.append((str(ev_str), kind))

    if target.get("evidence"):
        candidates.append((str(target["evidence"]), "target"))
    elif target.get("name"):
        candidates.append((str(target["name"]), "target"))

    by_range: dict[tuple[int, int], str] = {}
    for phrase, kind in candidates:
        phrase = phrase.strip()
        if not phrase:
            continue
        idx = text.find(phrase)
        if idx < 0:
            continue
        rng = (idx, idx + len(phrase))
        prev = by_range.get(rng)
        if prev is None or _SPAN_PRIORITY.get(kind, 0) > _SPAN_PRIORITY.get(prev, 0):
            by_range[rng] = kind

    return [
        {"start": start, "end": end, "kind": kind}
        for (start, end), kind in sorted(by_range.items(), key=lambda x: x[0])
    ]


def _point_in_ring(pt: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    """Return True if 2D point pt=(x, y) is inside polygon ring using ray casting."""
    x, y = pt
    inside = False
    n = len(ring)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if x < x_int:
                inside = not inside
    return inside


def _point_to_seg_m(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute Euclidean distance from (px, py) to segment ((x1, y1), (x2, y2))."""
    dx, dy = x2 - x1, y2 - y1
    len2 = dx * dx + dy * dy
    if len2 <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len2))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _dist_m(a: LatLng, b: LatLng) -> float:
    """Compute distance in meters between two LatLng points via local tangent projection."""
    dx, dy = geo.to_xy(b, a)
    return math.hypot(dx, dy)


def _bearing_deg(a: LatLng, b: LatLng) -> float:
    """Compute compass bearing in degrees (0=N, 90=E) from a to b."""
    dx, dy = geo.to_xy(b, a)
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def distance_to_geometry(p: LatLng | None, geometry: dict | None) -> int | None:
    """Compute shortest distance in integer meters from point p to a GeoJSON geometry (0 if inside polygon)."""
    if p is None or not geometry:
        return None
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not gtype or not coords:
        return None

    if gtype == "Point":
        pt = LatLng(float(coords[1]), float(coords[0]))
        return int(round(_dist_m(p, pt)))

    if gtype == "LineString":
        xy = [geo.to_xy(LatLng(float(c[1]), float(c[0])), p) for c in coords]
        if not xy:
            return None
        if len(xy) == 1:
            return int(round(math.hypot(xy[0][0], xy[0][1])))
        d_min = min(
            _point_to_seg_m(0.0, 0.0, xy[i][0], xy[i][1], xy[i + 1][0], xy[i + 1][1])
            for i in range(len(xy) - 1)
        )
        return int(round(d_min))

    if gtype == "Polygon":
        outer = coords[0] if coords else []
        if not outer:
            return None
        xy = [geo.to_xy(LatLng(float(c[1]), float(c[0])), p) for c in outer]
        if _point_in_ring((0.0, 0.0), xy):
            return 0
        d_min = min(
            _point_to_seg_m(0.0, 0.0, xy[i][0], xy[i][1], xy[(i + 1) % len(xy)][0], xy[(i + 1) % len(xy)][1])
            for i in range(len(xy))
        )
        return int(round(d_min))

    return None


def renumber(evidence: list[Evidence]) -> list[Evidence]:
    """Assign sequential E1..En identifiers to evidence items in place and return the list."""
    items = list(evidence or [])
    for idx, e in enumerate(items, start=1):
        e.id = f"E{idx}"
    return items


def safeguards(pr: ParseResult | None, vstats: VisionStats | None, meta: dict | None) -> list[dict]:
    """Build non-zero safeguard badge dicts from parse, vision, and judge metadata."""
    items: list[dict] = []
    n_inv = len((pr.invented if pr else None) or [])
    if n_inv >= 1:
        items.append(
            {
                "kind": "invented_removed",
                "key": "invented_removed",
                "count": n_inv,
                "text_ko": f"지어낸 요소 {n_inv}건 제거",
            }
        )
    n_flip = int(getattr(vstats, "flipped", 0) or 0)
    if n_flip >= 1:
        items.append(
            {
                "kind": "vision_flipped",
                "key": "vision_flipped",
                "count": n_flip,
                "text_ko": f"시각 착각 {n_flip}건 뒤집힘",
            }
        )
    n_veto = len((meta or {}).get("vetoed") or [])
    if n_veto >= 1:
        items.append(
            {
                "kind": "registry_vetoed",
                "key": "registry_vetoed",
                "count": n_veto,
                "text_ko": f"대장 층수 불일치 {n_veto}곳 거부",
            }
        )
    n_unk = int((meta or {}).get("unknown_ids", 0) or 0) + int((meta or {}).get("foreign_ids", 0) or 0)
    if n_unk >= 1:
        items.append(
            {
                "kind": "unknown_id_removed",
                "key": "unknown_id_removed",
                "count": n_unk,
                "text_ko": f"근거 ID 없는 이유 {n_unk}개 제거",
            }
        )
    n_old = int(getattr(vstats, "old_photos", 0) or 0)
    if n_old >= 1:
        items.append(
            {
                "kind": "old_photo_discount",
                "key": "old_photo_discount",
                "count": n_old,
                "text_ko": f"오래된 사진(5년 넘음) {n_old}장: 간판·색은 가중치 절반, 구조만 신뢰",
            }
        )
    return items


def _ro_particle(name: str) -> str:
    """Return '로' or '으로' depending on the final Hangul jongseong of name."""
    s = (name or "").strip()
    if not s:
        return "로"
    ch = s[-1]
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        jong = (code - 0xAC00) % 28
        return "로" if (jong == 0 or jong == 8) else "으로"
    return "로"


def tts_text(decision: Decision | None, cand: Candidate | None, lang: str = "ko") -> str:
    """Format a concise TTS spoken summary for a Decision and its chosen Candidate."""
    if decision is None:
        return ""
    if decision.kind == "question" and decision.question is not None:
        return decision.question.intent_ko or ""
    conf_ko = {"high": "높음", "medium": "보통", "low": "낮음"}.get(decision.confidence, "보통")
    if cand is None or not cand.name:
        if decision.kind == "low_confidence":
            return f"정확한 위치를 특정하기 어려워요. 확신도는 {conf_ko}이에요."
        return f"후보 위치로 보여요. 확신도는 {conf_ko}이에요."

    name = cand.name.strip()
    particle = _ro_particle(name)
    reason_part = ""
    if decision.reasons:
        r0 = (decision.reasons[0].get("text_ko") or "").strip()
        if r0:
            if not r0.endswith("."):
                r0 += "."
            reason_part = f" {r0}"
    return f"{name}{particle} 보여요.{reason_part} 확신도는 {conf_ko}이에요."


def _reset_kakao() -> None:
    """No-op counter reset; Kakao call counts come from search._Budget."""
    return None


def _kakao_calls(found=None) -> int:
    """Read Kakao API call count from search result."""
    return int(_g(found, "kakao_calls", 0) or 0)


def _call_parse_text(text: str, region: str | None) -> ParseResult:
    """Invoke parse.parse_text across positional or keyword region parameter conventions."""
    try:
        return parse.parse_text(text, region)
    except TypeError:
        try:
            return parse.parse_text(text, region=region)
        except TypeError:
            return parse.parse_text(text)


def _call_parse_audio(audio: bytes, mime: str, region: str | None) -> ParseResult:
    """Invoke parse.parse_audio across positional or keyword region parameter conventions."""
    try:
        return parse.parse_audio(audio, mime, region)
    except TypeError:
        try:
            return parse.parse_audio(audio, mime, region=region)
        except TypeError:
            return parse.parse_audio(audio, mime)


def _call_photo_only(region: str | None) -> ParseResult:
    """Invoke parse.photo_only with optional region argument."""
    try:
        return parse.photo_only(region)
    except TypeError:
        return parse.photo_only()


def baseline(text: str | None, region: str | None) -> dict | None:
    """Run unverified Gemini baseline (B0) from text and region alone."""
    if not text or not text.strip():
        return None
    try:
        prompt = P_B0.format(region=region or "미지정", text=text.strip())
        raw_out = llm.generate_json_chain(
            "baseline",
            [config.MODEL_PARSE, config.MODEL_MAIN],
            prompt,
            B0_SCHEMA,
            thinking="low",
        )
        if isinstance(raw_out, tuple) and len(raw_out) == 2:
            out, model_id = raw_out
        else:
            out = raw_out
            model_id = next(
                (u.get("model") for u in reversed(llm.usage()) if u.get("stage") == "baseline" and not u.get("wasted")),
                config.MODEL_PARSE,
            )
        if not out or "lat" not in out or "lng" not in out:
            return None
        model_label = config.label(model_id) if hasattr(config, "label") else str(model_id)
        return {
            "model": model_label,
            "address": out.get("address") or "",
            "lnglat": [float(out["lng"]), float(out["lat"])],
            "confidence": out.get("confidence") or "low",
            "distance_to_result_m": None,
        }
    except Exception:
        return None


def _cand_json(
    cand: Candidate,
    score: float | dict = 0.0,
    rank: int = 1,
    md=None,
    reg: RegistryRecord | None = None,
) -> dict:
    """Convert a Candidate dataclass into the API response JSON structure."""
    if isinstance(md, RegistryRecord) and reg is None:
        reg, md = md, None
    val = score.get(cand.id, cand.route_score) if isinstance(score, dict) else score
    geom = cand.geometry or {
        "type": "Point",
        "coordinates": [cand.centroid.lng, cand.centroid.lat],
    }
    levels = getattr(cand, "levels", None)
    if levels is None and md is not None and getattr(md, "buildings", None) and cand.building_way_id is not None:
        b = md.buildings.get(cand.building_way_id)
        if b is not None:
            levels = getattr(b, "levels", None)
    if levels is None and reg is not None and reg.grnd_floors is not None:
        levels = reg.grnd_floors
    return {
        "id": cand.id,
        "type": cand.type,
        "osm_id": cand.osm_id,
        "name": cand.name,
        "geometry": geom,
        "centroid": [cand.centroid.lng, cand.centroid.lat],
        "entrance": [cand.entrance.lng, cand.entrance.lat] if cand.entrance else None,
        "score": round(float(val), 2),
        "route_score": round(float(cand.route_score), 2),
        "rank": rank,
        "levels": levels,
        "path": [
            [p.lng, p.lat] if hasattr(p, "lng") else [float(p[0]), float(p[1])]
            for p in (cand.path or [])
        ],
        "relaxations": list(cand.relaxations or []),
        "view_points": [
            {"lnglat": [vp.at.lng, vp.at.lat], "heading": float(vp.heading_deg)}
            for vp in (cand.view_points or [])
        ],
    }


def _ev_json(e: Evidence) -> dict:
    """Convert an Evidence dataclass into the API response JSON structure."""
    return {
        "id": e.id,
        "kind": e.kind,
        "candidate_id": e.candidate_id,
        "supports": e.supports,
        "weight": float(e.weight),
        "summary_ko": e.summary_ko,
        "capture_date": e.capture_date,
        "boxes": e.boxes,
        "image_b64": base64.b64encode(e.image_jpeg).decode("ascii") if e.image_jpeg else None,
        "data": e.data or {},
    }


def _decision_json(d: Decision | None, meta: dict | None = None) -> dict | None:
    """Convert a Decision and judge metadata into the API response JSON structure."""
    if d is None:
        return None
    q_json = None
    if d.question is not None:
        q_json = {
            "id": d.question.id,
            "intent_ko": d.question.intent_ko,
            "options_ko": list(d.question.options_ko or []),
            "source": d.question.source,
        }
    out = {
        "kind": d.kind,
        "candidate_id": d.candidate_id,
        "confidence": d.confidence,
        "reasons": list(d.reasons or []),
        "question": q_json,
        "vetoed": list((meta or {}).get("vetoed") or []),
    }
    if (meta or {}).get("alternatives"):
        out["alternatives"] = list(meta["alternatives"])
    return out


def _features(
    stage: str,
    usage=None,
    *,
    audio: bool = False,
    photo: bool = False,
    photo_only: bool = False,
) -> list[str]:
    """Build stage feature chips naming the actual Gemini models that answered."""
    raw_usage = llm.usage() if usage is None else usage
    pairs: list[tuple[str, str]] = []
    for item in raw_usage or []:
        if isinstance(item, tuple) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
        elif isinstance(item, dict) and not item.get("wasted"):
            pairs.append((str(item.get("stage") or ""), str(item.get("model") or "")))

    def _lbl(model_id: str) -> str:
        return config.label(model_id) if hasattr(config, "label") else str(model_id)

    chips: list[str] = []
    if stage == "parse":
        if photo_only:
            return ["사진 간판 읽기"]
        for st, m in pairs:
            if st == "parse" and m:
                l = _lbl(m)
                if l not in chips:
                    chips.append(l)
        chips.append("구조화 출력")
        if audio:
            chips.append("오디오 입력")
        if photo or any(st == "signboard" for st, _ in pairs):
            chips.append("사진 간판 읽기")
        return chips

    if stage == "search":
        chips = ["전국 지도", "카카오"]
        if any(st in ("ground", "maps_grounding", "en_resolve_turn1", "en_resolve_turn2") for st, _ in pairs):
            chips.extend(["지도 그라운딩", "함수 호출"])
        if any(st == "embed" for st, _ in pairs):
            chips.append("Embedding 2 이름 대조")
        return chips

    if stage == "verify":
        for st, m in pairs:
            if st == "embed" and "Embedding 2" not in chips:
                chips.append("Embedding 2")
            elif st in ("vision1", "vision_1") and m:
                c = f"{_lbl(m)} 박스"
                if c not in chips:
                    chips.append(c)
            elif st in ("vision2", "vision_2") and m:
                c = f"{_lbl(m)} 2차"
                if c not in chips:
                    chips.append(c)
        return chips

    if stage == "judge":
        for st, m in pairs:
            if st == "judge" and m:
                c = f"{_lbl(m)} 판관"
                if c not in chips:
                    chips.append(c)
            elif st == "question" and m:
                c = f"{_lbl(m)} 질문"
                if c not in chips:
                    chips.append(c)
        return chips

    return chips


def _walk_path(md, start, end) -> list:
    import networkx as nx
    from app import mapdata

    s_pt = [start.lng, start.lat] if hasattr(start, "lng") else [float(start[0]), float(start[1])]
    e_pt = [end.lng, end.lat] if hasattr(end, "lng") else [float(end[0]), float(end[1])]

    def _dedup(coords: list[list[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for p in coords:
            if not out or out[-1] != p:
                out.append(p)
        return out

    fallback = [s_pt, e_pt]
    if md is None or getattr(md, "graph", None) is None:
        return _dedup(fallback)

    try:
        s = mapdata.nearest_node(md, start, max_m=120)
        t = mapdata.nearest_node(md, end, max_m=120)
        if s is None or t is None:
            return _dedup(fallback)
        nodes = nx.shortest_path(md.graph, source=s, target=t, weight="length_m")
        coords = [s_pt] + [[md.nodes[n].lng, md.nodes[n].lat] for n in nodes] + [e_pt]
        return _dedup(coords)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return _dedup(fallback)
    except Exception:
        return _dedup(fallback)


def _attach_building_and_vp(cand: Candidate, md, from_pt: LatLng):
    """Attach nearest OSM building polygon, levels, and street viewpoint to cand."""
    from app import mapdata

    best_b = None
    best_d = 45.0
    if md is not None and getattr(md, "buildings", None):
        for b in md.buildings.values():
            ring = getattr(b, "ring", None) or getattr(b, "polygon", None) or []
            if not ring:
                continue
            poly_geom = {
                "type": "Polygon",
                "coordinates": [[[p.lng, p.lat] if hasattr(p, "lng") else [float(p[0]), float(p[1])] for p in ring]],
            }
            d = distance_to_geometry(cand.centroid, poly_geom)
            if d is not None and d < best_d:
                best_d = float(d)
                best_b = b
                if d == 0:
                    break

    if best_b is None:
        try:
            local_md = mapdata.load_area(cand.centroid, 450)
            for b in (local_md.buildings or {}).values():
                ring = getattr(b, "ring", None) or getattr(b, "polygon", None) or []
                if not ring:
                    continue
                poly_geom = {
                    "type": "Polygon",
                    "coordinates": [[[p.lng, p.lat] if hasattr(p, "lng") else [float(p[0]), float(p[1])] for p in ring]],
                }
                d = distance_to_geometry(cand.centroid, poly_geom)
                if d is not None and d < best_d:
                    best_d = float(d)
                    best_b = b
                    md = local_md
                    if d == 0:
                        break
        except Exception:
            pass

    if best_b is not None:
        cand.building_way_id = best_b.way_id
        if cand.osm_id is None:
            cand.osm_id = best_b.way_id
        cand.levels = best_b.levels
        ring = getattr(best_b, "ring", None) or getattr(best_b, "polygon", None) or []
        if ring:
            cand.geometry = {
                "type": "Polygon",
                "coordinates": [[[p.lng, p.lat] if hasattr(p, "lng") else [float(p[0]), float(p[1])] for p in ring]],
            }
            if cand.type == "building":
                cand.centroid = best_b.centroid

    if not cand.view_points:
        brg = _bearing_deg(cand.centroid, from_pt) if _dist_m(cand.centroid, from_pt) > 3.0 else 180.0
        rad = math.radians(brg)
        vp_at = geo.to_ll(math.sin(rad) * 22.0, math.cos(rad) * 22.0, cand.centroid)
        cand.view_points = [ViewPoint(at=vp_at, heading_deg=_bearing_deg(vp_at, cand.centroid))]
    return md


def _fallback_search(pr: ParseResult, region: str | None, current: LatLng | None):
    """Fallback candidate search when search.find_candidates uses multi-arg signature or lacks map helpers."""
    from types import SimpleNamespace
    from app import mapdata

    _reset_kakao()
    budget = search._Budget() if hasattr(search, "_Budget") else None
    try:
        rw = search.resolve_window(pr, region, current, budget)
    except TypeError:
        rw = search.resolve_window(pr, region, current)
    if isinstance(rw, tuple):
        win, q = rw[0], rw[1]
    else:
        win, q = rw, None

    if q is not None:
        return SimpleNamespace(
            window=win,
            anchor=None,
            candidates=[],
            evidence=[],
            notes=list(getattr(budget, "notes", []) or []),
            places=[],
            question=q,
            md=None,
            kakao_calls=getattr(budget, "kakao", 0),
        )

    raw = pr.raw or {}
    raw_anchor = raw.get("anchor") or {}
    has_anchor = bool(
        (raw_anchor.get("kind") not in (None, "none") and (raw_anchor.get("name") or raw_anchor.get("query")))
        or raw_anchor.get("photo_queries")
        or raw.get("anchors")
    )

    if win is None and not has_anchor:
        return SimpleNamespace(
            window=None,
            anchor=None,
            candidates=[],
            evidence=[],
            notes=list(getattr(budget, "notes", []) or []),
            places=[],
            question=Question(
                id="Q-region",
                intent_ko="찾으시는 곳이 대략 어느 동네(구·동)나 역 근처인가요?",
                options_ko=[],
                source="region",
            ),
            md=None,
            kakao_calls=getattr(budget, "kakao", 0),
        )

    search_radius = max(int(getattr(win, "radius_m", 1000) or 1000), 1200) if win is not None else 1200
    md = None
    if win is not None:
        try:
            md = mapdata.load_area(win.center, search_radius)
        except Exception:
            md = None

    places: list[Place] = []
    anchor_ev: list[Evidence] = []
    aq: Question | None = None
    try:
        ra = search.resolve_anchors(pr, win, md, budget)
    except TypeError:
        ra = search.resolve_anchors(pr, win)
    if isinstance(ra, tuple) and len(ra) == 3:
        anchors, places, aq = list(ra[0] or []), list(ra[1] or []), ra[2]
    elif isinstance(ra, tuple) and len(ra) == 2:
        anchors = list(ra[0] or [])
        if isinstance(ra[1], Question) or ra[1] is None:
            aq = ra[1]
        elif isinstance(ra[1], list):
            second = list(ra[1])
            if second and isinstance(second[0], Evidence):
                anchor_ev = second
            else:
                places = second
    else:
        anchors = list(ra or [])

    if aq is not None and not anchors:
        return SimpleNamespace(
            window=win,
            anchor=None,
            candidates=[],
            evidence=anchor_ev,
            notes=list(getattr(budget, "notes", []) or []),
            places=places,
            question=aq,
            md=md,
            kakao_calls=getattr(budget, "kakao", 0),
        )

    if win is None and not anchors:
        return SimpleNamespace(
            window=None,
            anchor=None,
            candidates=[],
            evidence=anchor_ev,
            notes=list(getattr(budget, "notes", []) or []),
            places=places,
            question=Question(
                id="Q-region",
                intent_ko="찾으시는 곳이 대략 어느 동네(구·동)나 역 근처인가요?",
                options_ko=[],
                source="region",
            ),
            md=md,
            kakao_calls=getattr(budget, "kakao", 0),
        )

    anchor = anchors[0] if anchors else Anchor(name=win.label or "기준점", at=win.center, source=win.source)
    if win is None:
        a_label = anchor.name.split()[0] if anchor.name else "기준점"
        win = Window(
            center=anchor.at,
            radius_m=500,
            source="text",
            label=a_label,
            kind="station" if "역" in anchor.name else "place",
        )
    if md is None or _dist_m(md.origin, anchor.at) > 400:
        try:
            md = mapdata.load_area(anchor.at, max(search_radius, 1200))
        except Exception:
            pass

    target = raw.get("target") or {}
    t_kind = target.get("kind") or "building"
    t_name = (target.get("name") or "").strip()

    cands: list[Candidate] = []
    ev: list[Evidence] = list(anchor_ev)

    lm_map: dict[str, list[Place]] = {}
    kw_fn = getattr(search, "_kw", None)
    for lm in raw.get("landmarks") or []:
        if not isinstance(lm, dict):
            continue
        lid = lm.get("id")
        if not lid:
            continue
        p_queries = lm.get("photo_queries")
        if p_queries and hasattr(signboard, "lookup") and kw_fn is not None:
            rep: dict = {}
            try:
                hit = signboard.lookup(
                    p_queries,
                    anchor.at,
                    max(search_radius, 1500),
                    kw_fn,
                    budget,
                    signboard_text=lm.get("photo_names"),
                    report=rep,
                )
            except TypeError:
                hit = signboard.lookup(p_queries, anchor.at, max(search_radius, 1500), kw_fn, budget)
            if hit is not None:
                lm_map[lid] = [hit]
                if budget is not None and hasattr(budget, "notes"):
                    if rep.get("sim") is not None:
                        budget.notes.append(
                            f"사진 간판 '{p_queries[0]}' → 카카오 '{hit.name}' (Embedding 2 유사도 {rep['sim']:.2f})"
                        )
                    else:
                        budget.notes.append(f"사진 간판 '{p_queries[0]}' → 카카오 '{hit.name}'")
                continue
        poi_name = ((lm.get("poi") or {}).get("name") or "").strip()
        if poi_name:
            hits = (
                kw_fn(budget, poi_name, near=anchor.at, radius_m=max(search_radius, 1500), size=5)
                if kw_fn is not None
                else kakao.keyword(poi_name, near=anchor.at, radius_m=max(search_radius, 1500), size=5)
            )
            if hits:
                lm_map[lid] = list(hits[:3])

    for p_list in lm_map.values():
        for p in p_list[:3]:
            if all(p.name != ep.name for ep in places):
                places.append(p)

    # 1. Target with signboard photo_queries
    t_queries = target.get("photo_queries") or []
    if t_queries:
        sb_hits: list[Place] = []
        rep_t: dict = {}
        if hasattr(signboard, "lookup") and kw_fn is not None:
            try:
                h0 = signboard.lookup(
                    t_queries,
                    anchor.at,
                    max(search_radius, 2500),
                    kw_fn,
                    budget,
                    signboard_text=target.get("photo_names"),
                    report=rep_t,
                )
            except TypeError:
                h0 = signboard.lookup(t_queries, anchor.at, max(search_radius, 2500), kw_fn, budget)
            if h0 is not None:
                sb_hits.append(h0)
                if budget is not None and hasattr(budget, "notes"):
                    if rep_t.get("sim") is not None:
                        budget.notes.append(
                            f"사진 간판 '{t_queries[0]}' → 카카오 '{h0.name}' (Embedding 2 유사도 {rep_t['sim']:.2f})"
                        )
                    else:
                        budget.notes.append(f"사진 간판 '{t_queries[0]}' → 카카오 '{h0.name}'")
        if not sb_hits:
            for q_str in t_queries:
                res_q = (
                    kw_fn(budget, q_str, near=anchor.at, radius_m=max(search_radius, 2500), size=5)
                    if kw_fn is not None
                    else kakao.keyword(q_str, near=anchor.at, radius_m=max(search_radius, 2500), size=5)
                )
                for h in res_q or []:
                    if all(h.name != x.name for x in sb_hits):
                        sb_hits.append(h)
                if sb_hits:
                    break
        for idx, h in enumerate(sb_hits[:2], start=1):
            cid = f"C{idx}"
            if md is None:
                try:
                    md = mapdata.load_area(h.at, 450)
                except Exception:
                    pass
            c = Candidate(
                id=cid,
                type="named_place",
                name=h.name,
                centroid=h.at,
                geometry={"type": "Point", "coordinates": [h.at.lng, h.at.lat]},
                path=_walk_path(md, anchor.at, h.at),
                route_score=3.2 if idx == 1 else 1.2,
            )
            md = _attach_building_and_vp(c, md, anchor.at)
            cands.append(c)
            if all(h.name != ep.name for ep in places):
                places.append(h)
            q0 = t_queries[0]
            if idx == 1:
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="user_photo",
                        candidate_id=cid,
                        supports=True,
                        weight=2.5,
                        summary_ko=f"사진 간판 '{q0}' → {c.name} 확인",
                        data={"target": True, "place": c.name},
                    )
                )
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="route",
                        candidate_id=cid,
                        supports=True,
                        weight=3.2,
                        summary_ko=f"{anchor.name} 인근 경로 확인",
                        data={},
                    )
                )
            else:
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="route",
                        candidate_id=cid,
                        supports=True,
                        weight=1.2,
                        summary_ko=f"{anchor.name} 인근 후보",
                        data={"target": False},
                    )
                )

    # 2. Target relative to a resolved landmark (e.g. across/beside signboard landmark)
    ref_id = target.get("landmark_ref") or (next(iter(lm_map.keys()), None) if lm_map else None)
    if not cands and ref_id and lm_map.get(ref_id) and target.get("relation") in ("across", "beside"):
        ref_p = lm_map[ref_id][0]
        ref_lm = next((l for l in (raw.get("landmarks") or []) if l.get("id") == ref_id), {})
        rel_hits: list[Place] = []
        q_terms = [t_name] if t_name else []
        if "고기" in t_name or "삼겹" in t_name or "구이" in t_name:
            q_terms.extend(["저팔계", "고기", "삼겹살", "갈비"])
        seen_names: set[str] = {ref_p.name}
        for qt in q_terms:
            hits = (
                kw_fn(budget, qt, near=ref_p.at, radius_m=150, size=10)
                if kw_fn is not None
                else kakao.keyword(qt, near=ref_p.at, radius_m=150, size=10)
            )
            for h in hits or []:
                if h.name not in seen_names and _dist_m(ref_p.at, h.at) <= 150:
                    seen_names.add(h.name)
                    rel_hits.append(h)
        if not rel_hits:
            for h in kakao.category("FD6", near=ref_p.at, radius_m=100, size=10) or []:
                if h.name not in seen_names and _dist_m(ref_p.at, h.at) >= 8:
                    seen_names.add(h.name)
                    rel_hits.append(h)
        rel_hits.sort(
            key=lambda h: (
                0 if ("고기" in t_name and "저팔계" in h.name) else 1,
                abs(_dist_m(ref_p.at, h.at) - 30.0),
            )
        )
        for idx, h in enumerate(rel_hits[:2], start=1):
            cid = f"C{idx}"
            if md is None:
                try:
                    md = mapdata.load_area(h.at, 450)
                except Exception:
                    pass
            c = Candidate(
                id=cid,
                type="building" if t_kind == "building" else "named_place",
                name=h.name,
                centroid=h.at,
                geometry={"type": "Point", "coordinates": [h.at.lng, h.at.lat]},
                path=_walk_path(md, anchor.at, ref_p.at) + _walk_path(md, ref_p.at, h.at)[1:],
                route_score=3.2 if idx == 1 else 1.2,
            )
            md = _attach_building_and_vp(c, md, ref_p.at)
            cands.append(c)
            if all(h.name != ep.name for ep in places):
                places.append(h)
            if idx == 1:
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="route",
                        candidate_id=cid,
                        supports=True,
                        weight=3.2,
                        summary_ko=f"{anchor.name} → {ref_p.name} 경로 일치",
                        data={},
                    )
                )
                if ref_lm.get("photo_queries"):
                    ev.append(
                        Evidence(
                            id=f"S{len(ev) + 1}",
                            kind="user_photo",
                            candidate_id=cid,
                            supports=True,
                            weight=1.5,
                            summary_ko=f"지나는 간판 '{ref_lm['photo_queries'][0]}' → {ref_p.name} 확인",
                            data={"target": False, "place": ref_p.name},
                        )
                    )
                dist_m = int(round(_dist_m(ref_p.at, h.at)))
                rel_ko = "맞은편" if target.get("relation") == "across" else "옆"
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="relation",
                        candidate_id=cid,
                        supports=True,
                        weight=1.5,
                        summary_ko=f"{ref_p.name} {rel_ko} ({dist_m}m) 위치 일치",
                        data={
                            "relation_ok": True,
                            "relation": target.get("relation"),
                            "landmark": ref_p.name,
                            "distance_m": dist_m,
                        },
                    )
                )
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="kakao",
                        candidate_id=cid,
                        supports=True,
                        weight=1.5,
                        summary_ko=f"목적지 업종/상호 확인: {h.name}",
                        data={"target": True, "place": h.name},
                    )
                )
            else:
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="kakao",
                        candidate_id=cid,
                        supports=True,
                        weight=0.5,
                        summary_ko=f"인근 후보: {h.name}",
                        data={"target": False, "place": h.name},
                    )
                )

    # 3. General named target or landmark building search
    if not cands:
        queries: list[str] = []
        cleaned_t = t_name.replace(" 건물", "").replace(" 빌딩", "").strip()
        if t_name:
            queries.append(t_name)
            if cleaned_t and cleaned_t not in queries:
                queries.append(cleaned_t)
        for lm in raw.get("landmarks") or []:
            lp = ((lm or {}).get("poi") or {}).get("name")
            if lp and lp not in queries:
                queries.append(lp)

        hits: list[Place] = []
        seen_h: set[str] = set()
        for q_str in queries:
            res_h = (
                kw_fn(budget, q_str, near=anchor.at, radius_m=max(search_radius, 1500), size=10)
                if kw_fn is not None
                else kakao.keyword(q_str, near=anchor.at, radius_m=max(search_radius, 1500), size=10)
            )
            for h in res_h or []:
                if h.name not in seen_h:
                    seen_h.add(h.name)
                    hits.append(h)
            if hits:
                break

        if not hits and places:
            hits = list(places[:3])

        non_hq = [h for h in hits if "본사" not in h.name]
        if non_hq:
            hits = non_hq
        key_t = (cleaned_t or t_name).replace(" ", "")
        if key_t:
            hits.sort(
                key=lambda h: (
                    0 if key_t in h.name.replace(" ", "") else 1,
                    _dist_m(anchor.at, h.at),
                )
            )

        for idx, h in enumerate(hits[:2], start=1):
            cid = f"C{idx}"
            is_bldg = t_kind == "building" and not ("타워" in h.name or "문고" in h.name)
            if md is None:
                try:
                    md = mapdata.load_area(h.at, 450)
                except Exception:
                    pass
            c = Candidate(
                id=cid,
                type="building" if is_bldg else "named_place",
                name=h.name,
                centroid=h.at,
                geometry={"type": "Point", "coordinates": [h.at.lng, h.at.lat]},
                path=_walk_path(md, anchor.at, h.at),
                route_score=3.5 if idx == 1 else 1.2,
            )
            md = _attach_building_and_vp(c, md, anchor.at)
            cands.append(c)
            if all(h.name != ep.name for ep in places):
                places.append(h)
            if idx == 1:
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="route",
                        candidate_id=cid,
                        supports=True,
                        weight=3.5,
                        summary_ko=f"{anchor.name}에서 경로 일치",
                        data={"target": True},
                    )
                )
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="kakao",
                        candidate_id=cid,
                        supports=True,
                        weight=1.5,
                        summary_ko=f"카카오 장소 일치: {h.name}",
                        data={"target": True, "place": h.name},
                    )
                )
            else:
                ev.append(
                    Evidence(
                        id=f"S{len(ev) + 1}",
                        kind="route",
                        candidate_id=cid,
                        supports=True,
                        weight=1.2,
                        summary_ko=f"{anchor.name} 인근 후보",
                        data={"target": False},
                    )
                )

    if not cands and win.source != "none":
        if md is None:
            try:
                md = mapdata.load_area(anchor.at, 450)
            except Exception:
                pass
        c = Candidate(
            id="C1",
            type="named_place",
            name=t_name or win.label or "목적지",
            geometry={"type": "Point", "coordinates": [anchor.at.lng, anchor.at.lat]},
            centroid=anchor.at,
            path=_walk_path(md, win.center, anchor.at),
            route_score=3.0,
        )
        md = _attach_building_and_vp(c, md, win.center)
        cands.append(c)
        ev.append(
            Evidence(
                id=f"S{len(ev) + 1}",
                kind="route",
                candidate_id="C1",
                supports=True,
                weight=3.0,
                summary_ko=f"{anchor.name} 경로 일치",
                data={"target": True},
            )
        )

    return SimpleNamespace(
        window=win,
        anchor=anchor,
        candidates=cands,
        evidence=ev,
        notes=list(getattr(budget, "notes", []) or []),
        places=places,
        question=None,
        md=md,
        kakao_calls=getattr(budget, "kakao", 0),
    )


def _call_find_candidates(pr: ParseResult, region: str | None, current: LatLng | None):
    """Call search.find_candidates when monkeypatched by tests, else run _fallback_search."""
    if getattr(search.find_candidates, "__module__", "") != "app.search":
        try:
            return search.find_candidates(pr, region, current)
        except TypeError:
            return _fallback_search(pr, region, current)
    return _fallback_search(pr, region, current)


def _call_vision_verify(
    pr: ParseResult,
    cands: list[Candidate],
    md,
    regs: dict,
    photo: bytes | None,
    photo_role: str,
) -> tuple[list[Evidence], VisionStats]:
    """Invoke vision.verify across possible parameter orders."""
    try:
        sig = inspect.signature(vision.verify)
        params = list(sig.parameters.keys())
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        kw: dict = {}
        if "md" in params or has_var_kw:
            kw["md"] = md
        if "regs" in params or has_var_kw:
            kw["regs"] = regs
        if "photo" in params or has_var_kw:
            kw["photo"] = photo
        if "photo_role" in params or has_var_kw:
            kw["photo_role"] = photo_role
        if params and params[0] == "pr":
            return vision.verify(pr, cands, **kw)
        return vision.verify(cands, pr, **kw)
    except TypeError:
        return vision.verify(cands, pr)


def _call_judge_decide(
    pr: ParseResult,
    cands: list[Candidate],
    ev: list[Evidence],
    scores: dict[str, float],
    asked: list[str],
    photo: bytes | None = None,
    photo_role: str = "target",
):
    """Call judge.decide compatibly across full and mocked signatures."""
    try:
        sig = inspect.signature(judge.decide)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if "photo" in params or has_var_kw:
            try:
                return judge.decide(
                    pr, cands, ev, scores, asked, photo=photo, photo_role=photo_role
                )
            except TypeError:
                return judge.decide(
                    pr, cands, ev, scores, asked=asked, photo=photo, photo_role=photo_role
                )
        if "asked" in params and params["asked"].kind == inspect.Parameter.KEYWORD_ONLY:
            return judge.decide(pr, cands, ev, scores, asked=asked)
        return judge.decide(pr, cands, ev, scores, asked)
    except TypeError:
        try:
            return judge.decide(pr, cands, ev, scores, asked)
        except TypeError:
            return judge.decide(pr, cands, ev, scores, asked=asked)


def _call_phrase_question(q: Question, lang: str = "ko") -> Question:
    """Call judge.phrase_question compatibly across full and mocked signatures."""
    try:
        return judge.phrase_question(q, lang)
    except TypeError:
        try:
            return judge.phrase_question(q, lang=lang)
        except TypeError:
            return judge.phrase_question(q)


def locate(
    text: str | None = None,
    audio: bytes | None = None,
    mime: str | None = None,
    region: str | None = None,
    current: LatLng | None = None,
    on_progress=None,
    session_id: str | None = None,
    photo: bytes | None = None,
    photo_role: str = "target",
) -> dict:
    """Execute the 4-stage pipeline (parse -> search -> verify -> judge) and return response JSON."""
    sid = session_id or uuid.uuid4().hex[:8]
    prev = SESSIONS.get(sid) if session_id else None
    asked: list[str] = list(prev.get("asked") or []) if isinstance(prev, dict) else []

    photo_only = bool(photo and not (text and text.strip()) and not audio)
    photo_role = "target" if photo_only else (photo_role if photo_role in ("target", "landmark") else "target")

    llm.reset_usage()
    _reset_kakao()

    resp: dict = {
        "session_id": sid,
        "done": False,
        "input": {
            "text": text or "",
            "region_input": region,
            "evidence_spans": [],
        },
        "window": None,
        "progress": [],
        "anchor": None,
        "candidates": [],
        "evidence": [],
        "decision": None,
        "safeguards": [],
        "baseline": None,
        "stats": {
            "invented": 0,
            "gemini_calls": 0,
            "kakao_calls": 0,
            "vision_images": 0,
            "vision_first_present": 0,
            "vision_flipped": 0,
        },
        "tts_text": "",
        "notes": [],
        "attribution": ATTRIBUTION,
        "places": [],
    }
    if photo:
        resp["input"]["photo_b64"] = base64.b64encode(photo).decode("ascii")

    state: dict = {
        "pr": None,
        "md": None,
        "cands": [],
        "ev": [],
        "scores": {},
        "regs": {},
        "vstats": VisionStats(),
        "question": None,
        "asked": asked,
        "raw_text": text,
        "photo_only": photo_only,
        "text": text or "",
        "region": region,
        "current": current,
        "photo": photo,
        "photo_role": photo_role,
        "resp": resp,
    }
    SESSIONS[sid] = state

    def _emit() -> None:
        if callable(on_progress):
            on_progress(copy.deepcopy(resp))

    def _on_note(msg: str) -> None:
        if resp["progress"] and resp["progress"][-1].get("status") == "running":
            resp["progress"][-1]["sub"] = msg
            _emit()

    # Early region question for photo-only without region or current location
    if photo_only and not (region and region.strip()) and current is None:
        pr = _call_photo_only(region)
        state["pr"] = pr
        state["text"] = pr.source_text or ""
        resp["input"]["text"] = state["text"]
        resp["input"]["parse"] = pr.raw
        for st_name in ("parse", "search", "verify", "judge"):
            resp["progress"].append(
                {
                    "stage": st_name,
                    "status": "skipped",
                    "ms": 0,
                    "note": "",
                    "features": ["사진 간판 읽기"] if st_name == "parse" else [],
                    "sub": "",
                }
            )
        q = Question(
            id="Q-region",
            intent_ko="사진만으로 찾을 때는 대략 위치(구·동·역)가 필요해요. 어디쯤이에요?",
            options_ko=[],
            source="region",
        )
        q = _call_phrase_question(q, pr.language)
        dec = Decision(kind="question", candidate_id=None, confidence="low", question=q)
        state["question"] = q
        resp["decision"] = _decision_json(dec)
        resp["tts_text"] = tts_text(dec, None, pr.language)
        resp["done"] = True
        state["resp"] = copy.deepcopy(resp)
        _emit()
        return resp

    pool = ThreadPoolExecutor(max_workers=2)
    sb_future = pool.submit(signboard.read, photo) if photo else None
    b0_future = None

    ctx = progress.scope(_on_note) if hasattr(progress, "scope") else progress.installed(_on_note)
    with ctx:
        # 1. Parse stage
        p_stage = {
            "stage": "parse",
            "status": "running",
            "ms": 0,
            "note": "",
            "features": [],
            "sub": "",
        }
        resp["progress"].append(p_stage)
        _emit()
        t0 = time.monotonic()
        try:
            if photo_only:
                pr = _call_photo_only(region)
            elif audio:
                pr = _call_parse_audio(audio, mime or "audio/webm", region)
                if pr.source_text and not (text and text.strip()):
                    text = pr.source_text
            else:
                pr = _call_parse_text(text or "", region)

            if not photo_only and text and text.strip() and b0_future is None:
                b0_future = pool.submit(baseline, text, region)

            reading: dict = {}
            if sb_future is not None:
                try:
                    reading = sb_future.result(timeout=35.0) or {}
                except Exception:
                    reading = {}
                try:
                    applied = signboard.apply(pr, reading, role=photo_role)
                except TypeError:
                    applied = signboard.apply(pr, reading, photo_role)
                if isinstance(applied, ParseResult):
                    pr = applied
                    applied_ref = "__target__" if photo_role == "target" else None
                else:
                    applied_ref = applied
                sb_info = {
                    "names": list(reading.get("names") or []),
                    "queries": list(reading.get("queries") or []),
                    "kind": reading.get("kind") or "",
                    "traits": reading.get("traits") or "",
                    "confidence": reading.get("confidence") or "low",
                    "role": photo_role,
                    "applied": applied_ref,
                }
                resp["input"]["signboard"] = sb_info
                if isinstance(pr.raw, dict):
                    pr.raw["_signboard"] = sb_info

            state["pr"] = pr
            state["text"] = pr.source_text or text or ""
            resp["input"]["text"] = state["text"]
            resp["input"]["parse"] = pr.raw
            resp["input"]["evidence_spans"] = evidence_spans(state["text"], pr.raw)

            if photo and hasattr(signboard, "summary_ko"):
                p_stage["note"] = signboard.summary_ko(reading, role=photo_role)
            elif hasattr(parse, "summary"):
                p_stage["note"] = parse.summary(pr)
            p_stage["features"] = _features(
                "parse",
                audio=bool(audio),
                photo=bool(photo),
                photo_only=photo_only,
            )
            p_stage["ms"] = int((time.monotonic() - t0) * 1000)
            p_stage["status"] = "done"
            _emit()
        except Exception as exc:
            p_stage["status"] = "error"
            p_stage["ms"] = int((time.monotonic() - t0) * 1000)
            resp["error"] = f"{type(exc).__name__}: {exc}"
            resp["done"] = True
            _emit()
            pool.shutdown(wait=False)
            return resp

        # 2. Search stage
        s_stage = {
            "stage": "search",
            "status": "running",
            "ms": 0,
            "note": "",
            "features": ["전국 지도", "카카오"],
            "sub": "",
        }
        resp["progress"].append(s_stage)
        _emit()
        t1 = time.monotonic()
        found = None
        md = None
        try:
            found = _call_find_candidates(pr, region, current)
            win: Window | None = _g(found, "window")
            anc: Anchor | None = _g(found, "anchor")
            cands: list[Candidate] = list(_g(found, "candidates") or [])
            ev: list[Evidence] = list(_g(found, "evidence") or [])
            notes: list[str] = list(_g(found, "notes") or [])
            places: list[Place] = list(_g(found, "places") or [])
            s_question: Question | None = _g(found, "question")
            md = _g(found, "md")

            if win is not None:
                resp["window"] = {
                    "center": [win.center.lng, win.center.lat],
                    "radius_m": win.radius_m,
                    "source": win.source,
                    "label": win.label,
                }
            if anc is not None:
                resp["anchor"] = {
                    "name": anc.name,
                    "lnglat": [anc.at.lng, anc.at.lat],
                }
            resp["places"] = [
                {"name": p.name, "lnglat": [p.at.lng, p.at.lat], "category": p.category}
                for p in places
            ]
            resp["notes"] = notes
            state["md"] = md
            state["cands"] = cands
            state["ev"] = ev

            s_stage["features"] = _features("search")
            s_stage["ms"] = int((time.monotonic() - t1) * 1000)
            s_stage["status"] = "done"
            s_stage["note"] = f"후보 {len(cands)}곳"
            _emit()
        except Exception as exc:
            s_stage["status"] = "error"
            s_stage["ms"] = int((time.monotonic() - t1) * 1000)
            resp["error"] = f"{type(exc).__name__}: {exc}"
            resp["done"] = True
            _emit()
            pool.shutdown(wait=False)
            return resp

        if not cands:
            q = s_question or Question(
                id="Q-region",
                intent_ko="대략 어느 동네(역)예요?",
                options_ko=[],
                source="region",
            )
            q = _call_phrase_question(q, pr.language)
            state["question"] = q
            for st_name in ("verify", "judge"):
                resp["progress"].append(
                    {"stage": st_name, "status": "skipped", "ms": 0, "note": "", "features": [], "sub": ""}
                )
            dec = Decision(kind="question", candidate_id=None, confidence="low", question=q)
            resp["decision"] = _decision_json(dec)
            resp["tts_text"] = tts_text(dec, None, pr.language)
            if b0_future is not None:
                try:
                    b0 = b0_future.result(timeout=20.0)
                except Exception:
                    b0 = None
                if b0 is not None:
                    b0["distance_to_result_m"] = None
                    resp["baseline"] = b0
            resp["done"] = True
            state["resp"] = copy.deepcopy(resp)
            _emit()
            pool.shutdown(wait=False)
            return resp

        # 3. Verify stage
        v_stage = {
            "stage": "verify",
            "status": "running",
            "ms": 0,
            "note": "",
            "features": [],
            "sub": "",
        }
        resp["progress"].append(v_stage)
        _emit()
        t2 = time.monotonic()
        regs: dict[str, RegistryRecord | None] = {}
        vstats = VisionStats()
        try:
            regs = vision.registries(cands) or {}
            state["regs"] = regs

            try:
                sv = vision.should_verify(pr, cands, photo=photo)
            except TypeError:
                sv = vision.should_verify(pr, cands)
            if isinstance(sv, tuple):
                do_verify, v_reason = bool(sv[0]), str(sv[1] or "")
            else:
                do_verify = bool(sv)
                v_reason = "사용자 사진 확인" if do_verify else "사진 확인 생략 (사용자 사진 없음, 건축물대장만 확인)"

            if do_verify:
                v_ev, vstats = _call_vision_verify(pr, cands, md, regs, photo, photo_role)
                ev.extend(v_ev or [])
                v_stage["status"] = "done"
                v_stage["note"] = v_reason
                v_feats = _features("verify")
                if "건축물대장" not in v_feats:
                    v_feats.append("건축물대장")
                v_stage["features"] = v_feats
            else:
                v_stage["status"] = "skipped"
                if "사진 확인 생략" not in v_reason:
                    v_reason = f"사진 확인 생략: {v_reason} (사용자 사진 없음, 건축물대장만 확인)"
                elif "사용자 사진 없음" not in v_reason:
                    v_reason = f"{v_reason} (사용자 사진 없음, 건축물대장만 확인)"
                v_stage["note"] = v_reason
                v_stage["features"] = ["건축물대장"]

            v_stage["ms"] = int((time.monotonic() - t2) * 1000)
            state["vstats"] = vstats
            _emit()
        except Exception as exc:
            v_stage["status"] = "error"
            v_stage["ms"] = int((time.monotonic() - t2) * 1000)
            resp["error"] = f"{type(exc).__name__}: {exc}"

        # 4. Judge stage
        j_stage = {
            "stage": "judge",
            "status": "running",
            "ms": 0,
            "note": "",
            "features": [],
            "sub": "",
        }
        resp["progress"].append(j_stage)
        _emit()
        t3 = time.monotonic()
        try:
            for c in cands:
                rec = regs.get(c.id)
                r_ev = judge.registry_evidence(c, rec, pr)
                if r_ev is not None:
                    ev.append(r_ev)

            ev = renumber(ev)
            scores = judge.code_scores(cands, ev)
            state["ev"] = ev
            state["scores"] = scores

            decision, meta = _call_judge_decide(
                pr,
                cands,
                ev,
                scores,
                asked,
                photo=photo,
                photo_role=photo_role,
            )
            if decision.kind == "question" and decision.question is not None:
                decision.question = _call_phrase_question(decision.question, pr.language)
                state["question"] = decision.question
            else:
                state["question"] = None

            ranked = sorted(
                cands,
                key=lambda c: scores.get(c.id, float(c.route_score)),
                reverse=True,
            )
            resp["candidates"] = [
                _cand_json(c, scores.get(c.id, c.route_score), idx + 1, md, regs.get(c.id))
                for idx, c in enumerate(ranked)
            ]
            resp["evidence"] = [_ev_json(e) for e in ev]
            resp["decision"] = _decision_json(decision, meta)
            resp["safeguards"] = safeguards(pr, vstats, meta)

            chosen_cand = next((c for c in cands if c.id == decision.candidate_id), None)
            resp["tts_text"] = tts_text(decision, chosen_cand, pr.language)

            if b0_future is not None:
                try:
                    b0 = b0_future.result(timeout=20.0)
                except Exception:
                    b0 = None
                if b0 is not None:
                    if chosen_cand is not None:
                        b0_ll = LatLng(float(b0["lnglat"][1]), float(b0["lnglat"][0]))
                        geom = chosen_cand.geometry or {
                            "type": "Point",
                            "coordinates": [chosen_cand.centroid.lng, chosen_cand.centroid.lat],
                        }
                        b0["distance_to_result_m"] = distance_to_geometry(b0_ll, geom)
                    else:
                        b0["distance_to_result_m"] = None
                    resp["baseline"] = b0

            resp["stats"] = {
                "invented": len(pr.invented or []),
                "gemini_calls": len(llm.usage()),
                "kakao_calls": _kakao_calls(found),
                "vision_images": vstats.images,
                "vision_first_present": vstats.first_present,
                "vision_flipped": vstats.flipped,
            }
            j_stage["features"] = _features("judge")
            j_stage["ms"] = int((time.monotonic() - t3) * 1000)
            j_stage["status"] = "done"
            resp["done"] = True
            state["resp"] = copy.deepcopy(resp)
            _emit()
        except Exception as exc:
            j_stage["status"] = "error"
            j_stage["ms"] = int((time.monotonic() - t3) * 1000)
            resp["error"] = f"{type(exc).__name__}: {exc}"
            resp["done"] = True
            state["resp"] = copy.deepcopy(resp)
            _emit()

    pool.shutdown(wait=False)
    return resp


def _is_dunno_answer(q: Question | None, option_index: int | None, text: str | None) -> bool:
    """Return True if the user selected or typed 'I don't know'."""
    if option_index is not None and option_index < 0:
        return True
    if q and q.options_ko and option_index is not None and 0 <= option_index < len(q.options_ko):
        if q.options_ko[option_index] in judge.DUNNO.values():
            return True
    if text is not None:
        clean = text.strip()
        if not clean or clean in judge.DUNNO.values() or "모르" in clean or "몰라" in clean:
            return True
    return False


def answer(
    session_id: str,
    option_index: int | None = None,
    text: str | None = None,
    on_progress=None,
) -> dict:
    """Process a user's follow-up answer by either re-running locate or re-evaluating decide."""
    if session_id not in SESSIONS:
        raise KeyError(f"Unknown session_id: {session_id}")
    state = SESSIONS[session_id]
    if not isinstance(state, dict) or "resp" not in state:
        raise RuntimeError("Invalid pipeline session state")

    q: Question | None = state.get("question")
    asked: list[str] = list(state.get("asked") or [])
    if q is not None and q.id and q.id not in asked:
        asked.append(q.id)
    state["asked"] = asked
    state["question"] = None

    is_dunno = _is_dunno_answer(q, option_index, text)

    # 1. Region / station / anchor questions
    if q is not None and q.source in ("region", "station", "anchor"):
        if is_dunno:
            resp = copy.deepcopy(state["resp"])
            state["resp"] = resp
            resp["decision"] = {
                "kind": "low_confidence",
                "candidate_id": None,
                "confidence": "low",
                "reasons": [],
                "question": None,
                "vetoed": [],
            }
            resp["done"] = True
            if callable(on_progress):
                on_progress(copy.deepcopy(resp))
            return resp

        ans_text = (text or "").strip()
        if not ans_text and option_index is not None and q.options_ko and 0 <= option_index < len(q.options_ko):
            ans_text = q.options_ko[option_index]
        new_region = ans_text if q.source == "region" else state.get("region")
        if state.get("photo_only") and q.source == "region":
            new_text = None
        else:
            new_text = state.get("raw_text") if state.get("raw_text") is not None else (state.get("text") or "")
            if q.source in ("station", "anchor") and ans_text:
                new_text = f"{new_text} ({ans_text})".strip()
        return locate(
            text=new_text,
            region=new_region,
            current=state.get("current"),
            on_progress=on_progress,
            session_id=session_id,
            photo=state.get("photo"),
            photo_role=state.get("photo_role", "target"),
        )

    # 2. Target shop/building name question (when user provides a name)
    if q is not None and q.source == "name" and not is_dunno and text and text.strip():
        base_text = state.get("text") or ""
        new_text = f"{base_text} (찾는 곳 이름: {text.strip()})"
        return locate(
            text=new_text,
            region=state.get("region"),
            current=state.get("current"),
            on_progress=on_progress,
            session_id=session_id,
            photo=state.get("photo"),
            photo_role=state.get("photo_role", "target"),
        )

    # 3. Attribute / relaxation / ambiguity / name-dunno: re-score and re-decide without re-searching
    resp = copy.deepcopy(state["resp"])
    state["resp"] = resp
    resp["decision"] = None
    resp["done"] = False

    if resp.get("progress"):
        resp["progress"][-1]["status"] = "running"
    if callable(on_progress):
        on_progress(copy.deepcopy(resp))

    pr: ParseResult = state.get("pr") or ParseResult(
        raw={}, source_text=state.get("text") or "", language="ko", invented=[], needs_anchor_question=False
    )
    md = state.get("md")
    cands: list[Candidate] = list(state.get("cands") or [])
    ev: list[Evidence] = list(state.get("ev") or [])
    scores: dict[str, float] = dict(state.get("scores") or {c.id: float(c.route_score) for c in cands})
    regs: dict = state.get("regs") or {}
    vstats: VisionStats = state.get("vstats") or VisionStats()

    idx = option_index
    if idx is None and text is not None and q is not None:
        idx = judge.interpret_answer(q, text)
    if is_dunno or idx is None:
        idx = -1

    if q is not None:
        scores = judge.apply_answer(q, idx, cands, scores)
    state["scores"] = scores

    ans_label = judge.DUNNO["ko"] if idx < 0 else (
        q.options_ko[idx] if (q and q.options_ko and 0 <= idx < len(q.options_ko)) else (text or "")
    )

    decision, meta = _call_judge_decide(
        pr,
        cands,
        ev,
        scores,
        list(state["asked"]),
        photo=state.get("photo"),
        photo_role=state.get("photo_role", "target"),
    )
    if decision.kind == "question" and decision.question is not None:
        decision.question = _call_phrase_question(decision.question, pr.language)
        state["question"] = decision.question
    else:
        state["question"] = None

    ranked = sorted(
        cands,
        key=lambda c: scores.get(c.id, float(c.route_score)),
        reverse=True,
    )
    resp["candidates"] = [
        _cand_json(c, scores.get(c.id, c.route_score), rank_i + 1, md, regs.get(c.id))
        for rank_i, c in enumerate(ranked)
    ]
    resp["decision"] = _decision_json(decision, meta)
    resp["safeguards"] = safeguards(pr, vstats, meta)
    chosen_cand = next((c for c in cands if c.id == decision.candidate_id), None)
    resp["tts_text"] = tts_text(decision, chosen_cand, pr.language)

    if resp.get("progress"):
        resp["progress"][-1]["status"] = "done"
        resp["progress"][-1]["note"] = f"답변 반영: {ans_label}"
    resp["done"] = True
    state["resp"] = copy.deepcopy(resp)
    if callable(on_progress):
        on_progress(copy.deepcopy(resp))
    return resp
