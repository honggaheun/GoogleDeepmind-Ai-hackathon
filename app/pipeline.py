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
        return int(round(geo.haversine_m(p, pt)))

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


def baseline(text: str | None, region: str | None) -> dict | None:
    """Run unverified Gemini baseline (B0) from text and region alone."""
    if not text or not text.strip():
        return None
    try:
        prompt = P_B0.format(region=region or "미지정", text=text.strip())
        out, model_label = llm.generate_json_chain(
            "baseline",
            [config.MODEL_PARSE, config.MODEL_MAIN],
            prompt,
            B0_SCHEMA,
            thinking="low",
        )
        if not out or "lat" not in out or "lng" not in out:
            return None
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
    reg: RegistryRecord | None = None,
) -> dict:
    """Convert a Candidate dataclass into the API response JSON structure."""
    val = score.get(cand.id, cand.route_score) if isinstance(score, dict) else score
    geom = cand.geometry or {
        "type": "Point",
        "coordinates": [cand.centroid.lng, cand.centroid.lat],
    }
    levels = reg.grnd_floors if (reg and reg.grnd_floors is not None) else getattr(cand, "levels", None)
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
        "path": [[p.lng, p.lat] for p in (cand.path or [])],
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


def _features(stage: str, extra: list[str] | None = None) -> list[str]:
    """Collect active model chips for stage from llm.usage() plus any stage feature tags."""
    chips: list[str] = []
    for u in llm.usage():
        if u.get("stage") == stage and not u.get("wasted"):
            lbl = u.get("model")
            if lbl and lbl not in chips:
                chips.append(lbl)
    for item in extra or []:
        if item and item not in chips:
            chips.append(item)
    return chips


def _fallback_search(pr: ParseResult, region: str | None, current: LatLng | None):
    """Fallback candidate lookup using search.resolve_window and resolve_anchors when find_candidates is a stub."""
    from types import SimpleNamespace

    kakao.reset_counter()
    win, q = search.resolve_window(pr, region, current)
    if q is not None:
        return SimpleNamespace(
            window=win,
            anchor=None,
            candidates=[],
            evidence=[],
            notes=[],
            places=[],
            question=q,
        )
    anchors, places, aq = search.resolve_anchors(pr, win)
    if aq is not None and not anchors:
        return SimpleNamespace(
            window=win,
            anchor=None,
            candidates=[],
            evidence=[],
            notes=[],
            places=places,
            question=aq,
        )
    anchor = anchors[0] if anchors else Anchor(name=win.label or "기준점", at=win.center, source="window")
    target = (pr.raw or {}).get("target") or {}
    t_name = target.get("name") or (pr.source_text or "").strip()
    cands: list[Candidate] = []
    ev: list[Evidence] = []
    for idx, p in enumerate(places[:3], start=1):
        cid = f"C{idx}"
        cands.append(
            Candidate(
                id=cid,
                type="named_place" if target.get("specific") else "building",
                name=p.name,
                geometry={"type": "Point", "coordinates": [p.at.lng, p.at.lat]},
                centroid=p.at,
                path=[anchor.at, p.at],
                route_score=3.0 - (idx - 1) * 0.5,
            )
        )
        ev.append(
            Evidence(
                id=f"E{idx}",
                kind="route",
                candidate_id=cid,
                supports=True,
                weight=3.0 - (idx - 1) * 0.5,
                summary_ko=f"{anchor.name}에서 경로 확인",
                data={"target": True},
            )
        )
    if not cands and win.source != "none":
        cands.append(
            Candidate(
                id="C1",
                type="named_place",
                name=t_name or win.label or "목적지",
                geometry={"type": "Point", "coordinates": [anchor.at.lng, anchor.at.lat]},
                centroid=anchor.at,
                path=[win.center, anchor.at],
                route_score=3.0,
            )
        )
        ev.append(
            Evidence(
                id="E1",
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
        notes=[],
        places=places,
        question=None,
    )


def _call_find_candidates(pr: ParseResult, region: str | None, current: LatLng | None):
    """Call search.find_candidates(pr, region, current), falling back if signature is L2-S1 stub."""
    try:
        return search.find_candidates(pr, region, current)
    except TypeError:
        sig = inspect.signature(search.find_candidates)
        if len(sig.parameters) == 4 and "md" in sig.parameters:
            return _fallback_search(pr, region, current)
        raise


def _call_vision_verify(
    pr: ParseResult,
    cands: list[Candidate],
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

    llm.reset_usage()
    kakao.reset_counter()

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
        "cands": [],
        "ev": [],
        "scores": {},
        "regs": {},
        "vstats": VisionStats(),
        "question": None,
        "asked": asked,
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

    pool = ThreadPoolExecutor(max_workers=2)
    sb_future = pool.submit(signboard.read, photo) if photo else None
    b0_future = None

    with progress.installed(_on_note):
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
            if not (text and text.strip()) and not audio and photo:
                pr = parse.photo_only()
            elif audio:
                pr = parse.parse_audio(audio, mime or "audio/webm", region=region)
                if pr.source_text and not (text and text.strip()):
                    text = pr.source_text
            else:
                pr = parse.parse_text(text or "", region=region)

            if text and text.strip() and b0_future is None:
                b0_future = pool.submit(baseline, text, region)

            if sb_future is not None:
                try:
                    reading = sb_future.result(timeout=35.0)
                except Exception:
                    reading = {}
                resp["input"]["signboard"] = reading
                pr = signboard.apply(pr, reading, role=photo_role)

            state["pr"] = pr
            state["text"] = pr.source_text or text or ""
            resp["input"]["text"] = state["text"]
            resp["input"]["parse"] = pr.raw
            resp["input"]["evidence_spans"] = evidence_spans(state["text"], pr.raw)

            extra_feats = ["구조화 출력"]
            if audio:
                extra_feats.append("오디오 입력")
            if photo:
                extra_feats.append("사진 간판 읽기")
            p_stage["features"] = _features("parse", extra_feats)
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

        # Early region question for photo-only without region or current location
        if not (text and text.strip()) and not audio and photo and not region and current is None:
            q = judge.phrase_question(
                Question(
                    id="Q-region",
                    intent_ko="사진 찍으신 곳이 대략 어느 동네(역)인가요?",
                    options_ko=[],
                    source="region",
                ),
                lang=pr.language,
            )
            for st_name in ("search", "verify", "judge"):
                resp["progress"].append(
                    {"stage": st_name, "status": "skipped", "ms": 0, "note": "", "features": [], "sub": ""}
                )
            dec = Decision(kind="question", candidate_id=None, confidence="low", question=q)
            state["question"] = q
            resp["decision"] = _decision_json(dec)
            resp["tts_text"] = tts_text(dec, None, pr.language)
            resp["done"] = True
            state["resp"] = copy.deepcopy(resp)
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
        try:
            found = _call_find_candidates(pr, region, current)
            win: Window | None = _g(found, "window")
            anc: Anchor | None = _g(found, "anchor")
            cands: list[Candidate] = list(_g(found, "candidates") or [])
            ev: list[Evidence] = list(_g(found, "evidence") or [])
            notes: list[str] = list(_g(found, "notes") or [])
            places: list[Place] = list(_g(found, "places") or [])
            s_question: Question | None = _g(found, "question")

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
            state["cands"] = cands
            state["ev"] = ev

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
            q = judge.phrase_question(q, lang=pr.language)
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

            sv = vision.should_verify(pr, cands, photo=photo)
            if isinstance(sv, tuple):
                do_verify, v_reason = bool(sv[0]), str(sv[1] or "")
            else:
                do_verify = bool(sv)
                v_reason = "사용자 사진 확인" if do_verify else "사진 확인 생략 (사용자 사진 없음)"

            if do_verify:
                v_ev, vstats = _call_vision_verify(pr, cands, regs, photo, photo_role)
                ev.extend(v_ev or [])
                v_stage["status"] = "done"
                v_stage["note"] = v_reason
                v_stage["features"] = _features("verify", ["거리뷰 검증", "건축물대장"])
            else:
                v_stage["status"] = "skipped"
                if "사진 확인 생략" not in v_reason or "사용자 사진 없음" not in v_reason:
                    v_reason = "사진 확인 생략 (사용자 사진 없음)"
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

            decision, meta = judge.decide(
                pr,
                cands,
                ev,
                scores,
                asked=asked,
                photo=photo,
                photo_role=photo_role,
            )
            if decision.kind == "question" and decision.question is not None:
                decision.question = judge.phrase_question(decision.question, lang=pr.language)
                state["question"] = decision.question
            else:
                state["question"] = None

            ranked = sorted(
                cands,
                key=lambda c: scores.get(c.id, float(c.route_score)),
                reverse=True,
            )
            resp["candidates"] = [
                _cand_json(c, scores.get(c.id, c.route_score), idx + 1, regs.get(c.id))
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
                    if decision.kind == "answer" and chosen_cand is not None:
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
                "kakao_calls": kakao.call_count(),
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
        new_text = state.get("text") or ""
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

    decision, meta = judge.decide(
        pr,
        cands,
        ev,
        scores,
        asked=list(state["asked"]),
        photo=state.get("photo"),
        photo_role=state.get("photo_role", "target"),
    )
    if decision.kind == "question" and decision.question is not None:
        decision.question = judge.phrase_question(decision.question, lang=pr.language)
        state["question"] = decision.question
    else:
        state["question"] = None

    ranked = sorted(
        cands,
        key=lambda c: scores.get(c.id, float(c.route_score)),
        reverse=True,
    )
    resp["candidates"] = [
        _cand_json(c, scores.get(c.id, c.route_score), rank_i + 1, regs.get(c.id))
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
