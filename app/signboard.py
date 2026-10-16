"""Read store signboards from user photos and match them against nearby Kakao places."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

from google.genai import types

from app import config, llm
from app.contracts import Candidate, LatLng, ParseResult, Place

SIGN_TIMEOUT_MS = 20_000
EMBED_MIN_SIM = 0.85

SIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}},
        "names": {"type": "array", "items": {"type": "string"}},
        "texts": {"type": "array", "items": {"type": "string"}},
        "kind": {"type": "string"},
        "floors": {"type": "integer"},
        "traits": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["queries", "names", "texts", "kind", "floors", "traits", "confidence"],
}

SIGN_PROMPT = (
    "이 사진은 사용자가 길을 설명하면서 준 가게·건물 사진이다. 간판과 현수막의 글자를 읽어라.\n"
    "- queries: 이 가게를 카카오맵에서 찾을 검색어 1~4개. 상호만 쓰고 업종·메뉴 낱말은 빼라. "
    "띄어쓰기 없는 꼴을 먼저, 흔한 표기 변형(아귀찜↔아구찜, 까스↔가스)도 넣어라. 간판을 못 읽었으면 빈 배열.\n"
    "- names: 간판에 적힌 상호 원문(큰 글자부터). texts: 그 밖의 글자(메뉴·전화번호·층). kind: 업종 한 낱말(고기집·돈가스집·카페…).\n"
    "- floors: 보이는 층수(모르면 0). traits: 외관 한 줄(색·재질·지붕·층). confidence: 상호를 읽은 확신.\n"
    "사진에 없는 글자를 지어내지 마라."
)


def read(photo: bytes) -> dict:
    """Extract signboard queries, raw store names, and visual traits from a JPEG photo."""
    part = types.Part.from_bytes(data=photo, mime_type="image/jpeg")
    return llm.generate_json_chain(
        "signboard",
        [config.MODEL_MAIN, config.MODEL_PARSE],
        [part, SIGN_PROMPT],
        SIGN_SCHEMA,
        thinking="low",
        timeout_ms=SIGN_TIMEOUT_MS,
    )


def _norm(text: str) -> str:
    """Keep only lowercase alphanumeric and Hangul characters."""
    return re.sub(r"[^0-9a-z가-힣]+", "", (text or "").lower())


def pick(hits: list[Place], query: str) -> Place | None:
    """Return the first place whose normalized name contains the two-character query head."""
    head = _norm(query)[:2]
    if len(head) < 2:
        return None
    for hit in hits or []:
        if head in _norm(hit.name):
            return hit
    return None


def similarity(text: str, name: str) -> float | None:
    """Compute cosine similarity between signboard text and a candidate place name."""
    if not text or not name:
        return None
    try:
        return float(llm.cosine(llm.embed(text), llm.embed(name)))
    except Exception:
        return None


def _best_by_embedding(
    hits: list[Place], signboard_text: str
) -> tuple[Place | None, float | None]:
    """Compare top 5 candidate names against signboard text using embeddings in parallel."""
    top = [h for h in (hits or []) if h and h.name][:5]
    if not top or not signboard_text:
        return None, None
    try:
        q_vec = llm.embed(signboard_text)
    except Exception:
        return None, None

    def _score(place: Place) -> float | None:
        try:
            return float(llm.cosine(q_vec, llm.embed(place.name)))
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        sims = list(pool.map(_score, top))

    best_hit: Place | None = None
    best_sim: float | None = None
    for place, sim in zip(top, sims):
        if sim is not None and (best_sim is None or sim > best_sim):
            best_sim = sim
            best_hit = place

    if best_sim is not None and best_sim >= EMBED_MIN_SIM:
        return best_hit, best_sim
    return None, best_sim


def lookup(
    queries: list[str],
    center: LatLng,
    radius: int,
    kw,
    budget,
    signboard_text: str | None = None,
    report: dict | None = None,
) -> Place | None:
    """Search Kakao queries in order using the head rule first, then embedding fallback."""
    if report is not None:
        report.clear()
    clean = [str(q).strip() for q in (queries or []) if q and str(q).strip()][:4]
    if not clean:
        if report is not None:
            report.update(how="none", sim=None)
        return None

    # 1. Try head rule on each query in order
    pool: list[Place] = []
    for q in clean:
        hits = kw(budget, q, near=center, radius_m=radius, size=5) or []
        hit = pick(hits, q)
        if hit is not None:
            sim = similarity(signboard_text, hit.name) if signboard_text else None
            if report is not None:
                report.update(how="rule", sim=sim, query=q)
            return hit
        for h in hits:
            if not any(x.name == h.name and x.at == h.at for x in pool):
                pool.append(h)

    # 2. Fall back to embedding similarity over top hits
    if pool and signboard_text:
        best_hit, best_sim = _best_by_embedding(pool, signboard_text)
        if report is not None:
            report.update(how="embedding" if best_hit is not None else "none", sim=best_sim)
        return best_hit

    if report is not None:
        report.update(how="none", sim=None)
    return None


def pick_ref(pr: ParseResult) -> str | None:
    """Identify the reference landmark ID for a passing-store photo."""
    raw = pr.raw if isinstance(getattr(pr, "raw", None), dict) else {}
    valid_ids = {
        lm.get("id")
        for lm in (raw.get("landmarks") or [])
        if isinstance(lm, dict) and lm.get("id")
    }
    target_ref = (raw.get("target") or {}).get("landmark_ref")
    if target_ref and (not valid_ids or target_ref in valid_ids):
        return target_ref
    for step in reversed(raw.get("steps") or []):
        if not isinstance(step, dict):
            continue
        until = step.get("until") or {}
        ref = until.get("landmark_ref") if isinstance(until, dict) else None
        if ref and (not valid_ids or ref in valid_ids):
            return ref
    return None


def apply(pr: ParseResult, reading: dict, role: str = "target") -> str | None:
    """Inject signboard queries and names into the reference landmark or target."""
    _ensure_search_hooks()
    if not isinstance(reading, dict):
        return None
    queries = [str(q).strip() for q in (reading.get("queries") or []) if q and str(q).strip()]
    if not queries:
        return None
    names = [str(n).strip() for n in (reading.get("names") or []) if n and str(n).strip()]
    photo_names = " ".join(names)

    if role == "landmark":
        ref = pick_ref(pr)
        if ref:
            for lm in pr.raw.get("landmarks") or []:
                if isinstance(lm, dict) and lm.get("id") == ref:
                    poi = lm.setdefault("poi", {})
                    poi["name"] = queries[0]
                    poi["specific"] = True
                    lm["photo_queries"] = list(queries)
                    lm["photo_names"] = photo_names
                    return ref

    target = pr.raw.setdefault("target", {})
    target.update(
        kind="named_place",
        name=queries[0],
        specific=True,
        photo_queries=list(queries),
        photo_names=photo_names,
    )
    pr.needs_anchor_question = False
    return "__target__"


def summary_ko(
    reading: dict | None, hit_name: str | None = None, role: str = "target"
) -> str:
    """Format a Korean summary line for a signboard reading."""
    role_ko = "지나는 가게" if role == "landmark" else "찾는 곳"
    if not reading:
        return f"사진 간판({role_ko}): 판독 결과 없음"
    names = [str(n).strip() for n in (reading.get("names") or []) if n and str(n).strip()]
    queries = [str(q).strip() for q in (reading.get("queries") or []) if q and str(q).strip()]
    label = " · ".join(names) if names else (queries[0] if queries else "상호 미상")
    if hit_name:
        return f"사진 간판({role_ko}): {label} → {hit_name}"
    return f"사진 간판({role_ko}): {label}"


def _ensure_search_hooks() -> None:
    """Provide signboard-aware search helpers while search module stubs are in place."""
    try:
        from app import search
    except ImportError:
        return

    if not hasattr(search, "target_kind"):
        def _target_kind(pr: ParseResult) -> str:
            t = (pr.raw or {}).get("target") or {}
            kind = t.get("kind") or "point"
            if kind == "named_place" and t.get("specific") is False and t.get("name"):
                return "building"
            return kind

        search.target_kind = _target_kind

    lm_fn = getattr(search, "_landmark_positions", None)
    if lm_fn is None or "photo_queries" not in getattr(getattr(lm_fn, "__code__", None), "co_consts", ()):
        def _landmark_positions(pr: ParseResult, md, center: LatLng, radius: float, budget):
            out: dict[str, list[Place]] = {}
            for lm in (pr.raw or {}).get("landmarks") or []:
                lid = lm.get("id")
                if not lid:
                    continue
                p_queries = lm.get("photo_queries")
                if p_queries:
                    rep: dict = {}
                    hit = lookup(
                        p_queries,
                        center,
                        int(radius),
                        search._kw,
                        budget,
                        signboard_text=None,
                        report=rep,
                    )
                    if hit is not None:
                        out[lid] = [hit]
                        if budget is not None and hasattr(budget, "notes"):
                            budget.notes.append(f"간판 '{p_queries[0]}' → {hit.name}")
                    else:
                        out[lid] = []
                    continue
                poi = lm.get("poi") or {}
                if poi.get("specific") and poi.get("name"):
                    out[lid] = search._kw(budget, poi["name"], near=center, radius_m=int(radius)) or []
                elif poi.get("name") or poi.get("category_code"):
                    pool = search._poi_pool(budget, poi.get("name", ""), poi.get("category_code"), center, int(radius))
                    terms_res = search.match_terms([poi.get("name", "")], pool)
                    matched = terms_res.get(poi.get("name", ""), []) if isinstance(terms_res, dict) else []
                    if not matched:
                        matched = search.google_places_of_type(budget, poi.get("name", ""), center, int(radius)) or []
                    out[lid] = matched
            return out

        search._landmark_positions = _landmark_positions

    nc_fn = getattr(search, "_named_candidates", None)
    if nc_fn is None or "photo_queries" not in getattr(getattr(nc_fn, "__code__", None), "co_consts", ()):
        def _named_candidates(pr: ParseResult, md, window, anchors, budget) -> list[Candidate]:
            t = (pr.raw or {}).get("target") or {}
            center = anchors[0].at if anchors else (window.center if window else config.VENUE)
            radius = 1500
            p_queries = t.get("photo_queries")
            if p_queries:
                rep: dict = {}
                hit = lookup(
                    p_queries,
                    center,
                    radius,
                    search._kw,
                    budget,
                    signboard_text=None,
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
            hits = search._kw(budget, name, near=center, radius_m=radius, size=15) or []
            if not hits:
                gp = search.google_place(budget, name, center)
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

        search._named_candidates = _named_candidates


_ensure_search_hooks()


def _main() -> None:
    """CLI entrypoint for reading a signboard image."""
    parser = argparse.ArgumentParser(description="Read store signboard from an image file.")
    parser.add_argument("image", type=Path, help="Path to JPEG image file")
    args = parser.parse_args()
    out = read(args.image.read_bytes())
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
