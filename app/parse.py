"""Parse spoken or written route descriptions into validated structural JSON."""

from __future__ import annotations

import copy
import json
import re
import sys

from google.genai import types

from app import config, contracts, llm
from app.contracts import ParseResult

PARSE_TIMEOUT_MS = 30_000
PHOTO_ONLY_TEXT = "사진 속 가게 찾기"
AUDIO_INSTRUCTION = "첨부한 음성을 먼저 transcript에 그대로 받아 적고, evidence는 transcript의 연속 구간으로 하라."

SYSTEM_PROMPT = """너는 사람이 말로 한 길 안내를 구조화하는 해석기다. 규칙:
1) 모든 evidence는 원문(오디오면 transcript)에서 그대로 복사한 연속 구간이다. 원문에 없는 내용은 만들지 않는다.
2) 거리는 숫자로 바꾸지 말고 distance_class로만 분류한다: 바로=immediate, 조금=short, 쭉=far_walk, 한참=long, N분=minutes(minutes에 N), 언급 없음=none.
3) 랜드마크마다 카카오 검색용 poi(상호·건물명과 카테고리 코드, 없으면 생략)와 거리 사진용 visual(영어 묘사, 간판 글자, 색, 층수,
   structural = 대문·나무·건물 형태처럼 잘 안 바뀌는 특징이면 true)을 채운다. 상호는 poi로, 구조 특징은 visual로 적는다.
   카카오 검색어(anchor.query, region.query, poi.name)는 한국어로 적는다. 영어 입력이면 한국어 이름으로 옮긴다(evidence는 원문 그대로).
   poi.specific: 특정 가게·건물 이름(고유명사, 예: 스타벅스 삼성점·오토웨이타워)이면 true, 업종·종류(예: 돈가스집·고기집·편의점·카페)면 false.
   "X 쪽에서·X에서 나와서·X 앞에서 직진"처럼 출발하는 곳이면 X를 anchor로 적는다(랜드마크로 두지 않는다).
   목적지가 "A 맞은편 고기집"처럼 업종이면 target.kind=building, target.name에 업종(고기집), target.specific=false,
   target.category_code에 카카오 카테고리 코드(음식점 FD6·카페 CE7·편의점 CS2 등), landmark_ref에 A, relation에 across·beside를 적는다.
   목적지가 특정 상호·건물명이면 target.kind=named_place, target.specific=true.
   문장 끝에 "(찾는 곳 이름: X)"가 붙어 있으면 사용자가 되묻기에 답한 목적지 상호다 → target.kind=named_place, target.name=X로 하고,
   원래 설명의 업종·관계(A 맞은편 등)는 landmark_ref·relation에 그대로 남긴다.
4) 좌우는 걸어가는 방향 기준 left/right/across/unknown.
   "X 방향으로 가다가·X 쪽으로 걸으면"처럼 걷는 방향을 장소 이름으로 말하면 그 단계(go_straight)에 toward = X의 landmark id를 적고,
   X를 landmarks에 poi(name=X, specific=true)로 넣는다(evidence는 원문 구간). X에 도착한다는 뜻이 아니므로 until에는 넣지 않는다.
5) "송파구에서 만나기로 했는데"처럼 범위를 말하면 region에 적는다(구=district, 동=neighborhood, 역=station, 장소=place, 없으면 none).
6) 목적지가 건물 이름이면 target.kind=named_place와 target.name, 골목이면 alley, 교차로·모퉁이면 intersection, 대문 앞·나무 앞 같은 곳이면 point,
   묘사된 건물이면 building. 기준 랜드마크가 있으면 target.landmark_ref와 relation을 적는다.
7) 해석이 둘 이상 가능한 부분은 ambiguities에 짧은 한국어 질문과 선택지로 남긴다.
8) 글자 입력이면 transcript는 빈 문자열이다."""

_DISTANCE_RANGES: dict[str, tuple[float, float]] = {
    "immediate": (0, 30),
    "short": (30, 150),
    "far_walk": (100, 600),
    "long": (300, 1500),
    "none": (0, 600),
}

_DIRECT_RELATIONS = {None, "", "at", "front", "entrance", "arrive", "near", "in"}


def distance_range(cls: str, minutes: int | None = None) -> tuple[float, float]:
    """Convert a distance class and optional walk minutes into a meter interval."""
    if cls == "minutes" and minutes:
        return (minutes * 60, minutes * 90)
    return _DISTANCE_RANGES.get(cls or "none", (0, 600))


def _norm(text: str | None) -> str:
    """Normalize text by stripping whitespace and punctuation for substring checks."""
    if not text:
        return ""
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE).lower()


def _in_source(evidence: str | None, *sources: str | None) -> bool:
    """Return True when normalized evidence is a non-empty substring of any source."""
    ev = _norm(evidence)
    if not ev:
        return False
    for src in sources:
        if src and ev in _norm(src):
            return True
    return False


def _normalize_target(target: dict, by_id: dict[str, dict]) -> None:
    """Promote point or building targets referencing a specific POI into named_place."""
    if target.get("kind") not in ("point", "building"):
        return
    ref = target.get("landmark_ref")
    if not ref or ref not in by_id:
        return
    if target.get("relation") not in _DIRECT_RELATIONS:
        return
    poi = by_id[ref].get("poi") or {}
    if poi.get("specific") and poi.get("name"):
        target["kind"] = "named_place"
        target["name"] = poi["name"]
        target["specific"] = True


def validate(raw: dict, source_text: str, region_hint: str | None = None) -> ParseResult:
    """Remove hallucinated elements not grounded in source_text and normalize target."""
    data = copy.deepcopy(raw)
    invented: list[str] = []

    # 1. Validate region and anchor against source_text and region_hint
    region = data.get("region") or {"evidence": "", "query": "", "kind": "none"}
    if region.get("kind", "none") != "none" or region.get("evidence"):
        ev = region.get("evidence", "")
        if not _in_source(ev, source_text, region_hint):
            if ev:
                invented.append(ev)
            region = {"evidence": "", "query": "", "kind": "none"}
    data["region"] = region

    anchor = data.get("anchor") or {"evidence": "", "query": "", "kind": "none"}
    if anchor.get("kind", "none") != "none" or anchor.get("evidence"):
        ev = anchor.get("evidence", "")
        if not _in_source(ev, source_text, region_hint):
            if ev:
                invented.append(ev)
            anchor = {"evidence": "", "query": "", "kind": "none"}
    data["anchor"] = anchor

    # 2. Filter landmarks against source_text
    kept_landmarks: list[dict] = []
    for lm in data.get("landmarks") or []:
        ev = lm.get("evidence", "")
        if _in_source(ev, source_text):
            kept_landmarks.append(lm)
        elif ev:
            invented.append(ev)
    data["landmarks"] = kept_landmarks
    by_id = {lm["id"]: lm for lm in kept_landmarks if lm.get("id")}

    # 3. Filter steps and prune dangling landmark references
    kept_steps: list[dict] = []
    for step in data.get("steps") or []:
        ev = step.get("evidence", "")
        if not _in_source(ev, source_text):
            if ev:
                invented.append(ev)
            continue
        until = step.get("until")
        if isinstance(until, dict):
            ref = until.get("landmark_ref")
            if not ref or ref not in by_id:
                step.pop("until", None)
        if "toward" in step:
            toward = step.get("toward")
            if not toward or toward not in by_id:
                step.pop("toward", None)
        tt = step.get("turn_target")
        if isinstance(tt, dict):
            cf = tt.get("count_from")
            if isinstance(cf, str) and cf.startswith("after:"):
                ref_id = cf.split(":", 1)[1]
                if ref_id not in by_id:
                    tt.pop("count_from", None)
        kept_steps.append(step)
    data["steps"] = kept_steps

    # 4. Clean target reference and normalize specific POI targets
    target = data.get("target") or {"kind": "point"}
    ref = target.get("landmark_ref")
    if ref and ref not in by_id:
        target.pop("landmark_ref", None)
    _normalize_target(target, by_id)
    data["target"] = target

    needs_anchor_question = (
        data["anchor"].get("kind", "none") == "none"
        and data["target"].get("kind") != "named_place"
    )
    distance_m = {
        step["order"]: distance_range(step.get("distance_class", "none"), step.get("minutes"))
        for step in kept_steps
        if "order" in step
    }

    return ParseResult(
        raw=data,
        source_text=source_text,
        language=data.get("language") or "ko",
        invented=invented,
        needs_anchor_question=needs_anchor_question,
        distance_m=distance_m,
    )


def _hint_suffix(region_hint: str | None) -> str:
    """Format an optional region hint suffix for the parser prompt."""
    if region_hint and region_hint.strip():
        return f"\n대략 위치 힌트: {region_hint.strip()}"
    return ""


def parse_text(text: str, region_hint: str | None = None) -> ParseResult:
    """Parse a text route description via Gemini and validate grounding."""
    prompt = f"원문: {text}" + _hint_suffix(region_hint)
    raw = llm.generate_json_chain(
        "parse",
        [config.MODEL_PARSE, config.MODEL_MAIN],
        [prompt],
        contracts.PARSE_SCHEMA,
        thinking="low",
        system=SYSTEM_PROMPT,
        timeout_ms=PARSE_TIMEOUT_MS,
    )
    return validate(raw, text, region_hint=region_hint)


def parse_audio(data: bytes, mime: str, region_hint: str | None = None) -> ParseResult:
    """Transcribe and parse an audio route description in a single Gemini call."""
    part = types.Part.from_bytes(data=data, mime_type=mime)
    instruction = AUDIO_INSTRUCTION + _hint_suffix(region_hint)
    raw = llm.generate_json_chain(
        "parse",
        [config.MODEL_PARSE, config.MODEL_MAIN],
        [part, instruction],
        contracts.PARSE_SCHEMA,
        thinking="low",
        system=SYSTEM_PROMPT,
        timeout_ms=PARSE_TIMEOUT_MS,
    )
    transcript = raw.get("transcript") or ""
    return validate(raw, transcript, region_hint=region_hint)


def photo_only(region_hint: str | None = None) -> ParseResult:
    """Build an empty ParseResult for photo-only queries without calling the LLM."""
    if region_hint and region_hint.strip():
        h = region_hint.strip()
        region = {"evidence": h, "query": h, "kind": "neighborhood"}
    else:
        region = {"evidence": "", "query": "", "kind": "none"}
    raw = {
        "language": "ko",
        "transcript": "",
        "region": region,
        "anchor": {"evidence": "", "query": "", "kind": "none"},
        "steps": [],
        "landmarks": [],
        "target": {"kind": "named_place", "name": "", "specific": True},
        "uncertainty": {"hedges": [], "staleness": "unknown"},
        "ambiguities": [],
    }
    return ParseResult(
        raw=raw,
        source_text=PHOTO_ONLY_TEXT,
        language="ko",
        invented=[],
        needs_anchor_question=False,
        distance_m={},
    )


def summary(pr: ParseResult) -> str:
    """Return a concise single-line summary of anchor, steps, landmarks, and target."""
    raw = pr.raw or {}
    anchor = raw.get("anchor") or {}
    target = raw.get("target") or {}
    steps = [f"{s.get('action')}({s.get('distance_class', 'none')})" for s in raw.get("steps") or []]
    lms = [
        (lm.get("poi") or {}).get("name") or (lm.get("visual") or {}).get("en") or lm.get("id", "")
        for lm in raw.get("landmarks") or []
    ]
    return (
        f"anchor={anchor.get('kind', 'none')}:{anchor.get('query', '')} | "
        f"steps={steps} | landmarks={lms} | "
        f"target={target.get('kind', '')}:{target.get('name', '')} | "
        f"invented={len(pr.invented)}"
    )


if __name__ == "__main__":
    sample = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "지하철 2호선 삼성역 3번 출구 하차 후 직진, 오토웨이 타워"
    )
    res = parse_text(sample)
    out = {
        "anchor": res.raw.get("anchor"),
        "steps": res.raw.get("steps"),
        "landmarks": res.raw.get("landmarks"),
        "target": res.raw.get("target"),
        "invented": res.invented,
        "needs_anchor_question": res.needs_anchor_question,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
