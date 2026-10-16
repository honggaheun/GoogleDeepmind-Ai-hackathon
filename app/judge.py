"""Stage 4: score candidates, enforce building registry vetoes, and decide or ask a follow-up question."""

from __future__ import annotations

import json
import math
import re

from google.genai import types

from app import config, geo, llm
from app.contracts import (
    ANSWER_SCHEMA,
    JUDGE_SCHEMA,
    QUESTION_SCHEMA,
    Candidate,
    Decision,
    Evidence,
    LatLng,
    ParseResult,
    Question,
    RegistryRecord,
)

JUDGE_PHOTOS = 3
JUDGE_TIMEOUT_MS, LIGHT_TIMEOUT_MS = 25_000, 15_000
DUNNO = {"ko": "모르겠어요", "en": "I don't know"}

P_JUDGE = (
    "너는 후보 위치 중 하나를 고르는 판정관이다. 아래 '정답 지도' JSON과 첨부된 거리 사진만 근거로 판단하라.\n"
    "좌우·순서·거리 같은 공간 관계는 이미 코드가 계산했으니 다시 추론하지 말고 그대로 믿어라.\n"
    "'사용자 사진'의 역할(찾는 곳 / 지나는 가게)은 사용자가 정했다 — 사진 라벨에 적힌 역할대로만 쓰고, "
    "사진 속 메뉴·현수막·업종을 근거로 역할을 바꾸거나 코드의 대응을 '잘못 매칭'이라 하지 마라. "
    "찾는 곳 사진이면 거리 사진·대장과 같은 건물인지 비교해 판정에 반영하라.\n"
    "첨부 거리 사진은 그 사진으로 만든 시각 근거(사진 라벨에 적힌 ID)를 네가 직접 확인하는 용도다. 사진에서 본 것을 이유로 쓸 때는\n"
    "그 사진 라벨의 근거 ID를 붙이고, 사진 판독 근거와 네 관찰이 다르면 counter_evidence_ids에 그 ID를 적어라.\n"
    "모든 이유에는 근거 ID(E로 시작)를 붙여야 하고, 목록에 없는 ID는 쓰지 마라.\n"
    "근거가 부족하면 choice를 \"none\", confidence를 low로 하라. 사진은 오래되었을 수 있으니 촬영 연월을 고려하라.\n"
    "거리 사진이 없는 것은 정책(사용자 사진이 있을 때만 봄)이지 결함이 아니다 — 사진이 없다는 이유로 확신도를 낮추지 마라. "
    "건축물대장 건물명이 목표 이름과 맞고 경로도 맞으면 high가 타당하다."
)

P_QUESTION = (
    "다음 질문 의도를 사용자가 바로 답할 수 있는 짧은 {lang} 질문 하나와 선택지 2~4개로 바꿔라. "
    "마지막 선택지는 \"{dunno}\".\n"
    "질문 의도: {intent}\n기본 선택지: {options}"
)

P_ANSWER = (
    "질문: {question} / 선택지(0부터): {options} / 사용자의 답: {answer}\n"
    "사용자의 답이 어느 선택지 번호에 해당하는지 정수로 골라라. 모름·해당 없음이면 -1."
)

_GENERIC_TARGET_NAMES = {
    "",
    "건물",
    "곳",
    "골목",
    "교차로",
    "지점",
    "목적지",
    "가게",
    "상가",
    "빌딩",
    "집",
    "매장",
    "장소",
}
_NON_SHOP_PURPOSES = ("교육연구", "종교", "공공", "의료", "공장")
_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def _norm_name(s: str | None) -> str:
    """Normalize a place or building name by removing whitespace and punctuation."""
    if not s:
        return ""
    return re.sub(r"[\s\W_]+", "", str(s)).lower()


def _target_landmark(pr: ParseResult) -> dict:
    """Return the parsed landmark dict corresponding to the target when it is not a cross/beside reference."""
    raw = (pr.raw if pr else None) or {}
    target = raw.get("target") or {}
    if target.get("relation") in ("across", "beside"):
        return {}
    ref = target.get("landmark_ref")
    landmarks = raw.get("landmarks") or []
    if ref:
        for lm in landmarks:
            if lm.get("id") == ref:
                return lm
    return landmarks[-1] if landmarks else {}


def _is_business_target(pr: ParseResult) -> bool:
    """Return True if the parsed target describes a shop, business category, or specific named place."""
    raw = (pr.raw if pr else None) or {}
    target = raw.get("target") or {}
    kind = target.get("kind") or ""
    if kind in ("alley", "intersection", "point"):
        return False
    cat = target.get("category_code")
    if cat and cat != "NONE":
        return True
    name = (target.get("name") or "").strip()
    if name and name not in _GENERIC_TARGET_NAMES:
        return True
    t_lm = _target_landmark(pr)
    poi = t_lm.get("poi") or {}
    if (poi.get("name") or "").strip() or (poi.get("category_code") or "") not in ("", "NONE"):
        return True
    return False


def registry_evidence(cand: Candidate, rec: RegistryRecord | None, pr: ParseResult) -> Evidence | None:
    """Compare a building registry record with target cues and return registry Evidence."""
    if rec is None:
        return None
    raw = (pr.raw if pr else None) or {}
    target = raw.get("target") or {}
    t_lm = _target_landmark(pr)
    visual = t_lm.get("visual") or {}
    lm_text = f"{t_lm.get('evidence') or ''} {visual.get('en') or ''}".lower()

    weight = 0.0
    notes: list[str] = []
    data: dict = {
        "grnd_floors": rec.grnd_floors,
        "ugrnd_floors": rec.ugrnd_floors,
        "main_purpose": rec.main_purpose,
        "name": rec.name,
    }

    # 1. Floor comparison
    desc_floors = visual.get("floors")
    if desc_floors is not None and rec.grnd_floors is not None:
        diff = abs(int(desc_floors) - int(rec.grnd_floors))
        if diff == 0:
            weight += 1.0
            notes.append(f"층수 일치({rec.grnd_floors}층)")
        elif diff == 1:
            notes.append(f"층수 1층 차이(설명 {desc_floors}층 vs 대장 {rec.grnd_floors}층)")
        else:
            weight -= 1.5
            notes.append(f"층수 불일치(설명 {desc_floors}층 vs 대장 {rec.grnd_floors}층)")

    # 2. Residential purpose match
    purpose = rec.main_purpose or ""
    if any(k in lm_text for k in ("빌라", "주택", "villa", "house")):
        if "공동주택" in purpose or "단독주택" in purpose:
            weight += 0.5
            notes.append(f"주용도 일치({purpose})")

    # 3. Named target match
    is_named = target.get("kind") == "named_place" or bool(target.get("specific"))
    t_name = _norm_name(target.get("name") or cand.name)
    r_name = _norm_name(rec.name)
    if is_named and target.get("name"):
        t_only = _norm_name(target.get("name"))
        if len(t_only) >= 2 and len(r_name) >= 2 and (t_only in r_name or r_name in t_only):
            weight += 1.0
            data["name_match"] = True
            notes.append(f"대장 건물명 '{rec.name}' = 목표 이름")
        else:
            data["name_match"] = False
    elif "name_match" not in data and rec.name is not None:
        if len(t_name) >= 2 and len(r_name) >= 2 and (t_name in r_name or r_name in t_name):
            data["name_match"] = True
        else:
            data["name_match"] = False

    # 4. Non-shop purpose veto for business targets
    if _is_business_target(pr) and not data.get("name_match"):
        if any(p in purpose for p in _NON_SHOP_PURPOSES):
            weight -= 1.5
            data["purpose_veto"] = True
            notes.append(f"주용도 '{purpose}' — 가게가 있을 건물이 아님")

    base_parts = [
        rec.name or "이름 없음",
        f"지상 {rec.grnd_floors or '?'}층·지하 {rec.ugrnd_floors or 0}층",
        purpose or "용도 미상",
    ]
    if rec.use_apr_day:
        base_parts.append(f"사용승인 {rec.use_apr_day}")
    summary = "대장: " + ", ".join(base_parts)
    if notes:
        summary += " (" + ", ".join(notes) + ")"

    supports: bool | None = None
    if weight > 0:
        supports = True
    elif weight < 0:
        supports = False

    return Evidence(
        id="",
        kind="registry",
        candidate_id=cand.id,
        supports=supports,
        weight=weight,
        summary_ko=summary,
        data=data,
    )


def code_scores(cands: list[Candidate], ev: list[Evidence]) -> dict[str, float]:
    """Sum candidate route_score and non-route evidence weights per candidate."""
    scores: dict[str, float] = {}
    for c in cands or []:
        extra = sum(
            float(e.weight)
            for e in (ev or [])
            if e.candidate_id == c.id and e.kind != "route"
        )
        scores[c.id] = float(c.route_score) + extra
    return scores


def veto(cand_id: str, ev: list[Evidence]) -> bool:
    """Return True if the candidate has a registry evidence item with weight <= -1.5."""
    return any(
        e.candidate_id == cand_id and e.kind == "registry" and float(e.weight) <= -1.5
        for e in (ev or [])
    )


def _has_decisive_evidence(cand_id: str, active_ids: set[str], ev: list[Evidence]) -> bool:
    """Check whether a candidate has a unique decisive evidence item among active candidates."""
    cand_ev = [e for e in (ev or []) if e.candidate_id == cand_id and e.supports is True]
    for e in cand_ev:
        if e.kind == "registry" and (e.data or {}).get("name_match") is True:
            return True
        if e.kind == "user_photo":
            return True
    has_target_hit = any(
        e.kind in ("kakao", "google") and (e.data or {}).get("target") is True
        for e in cand_ev
    )
    if has_target_hit:
        other_target_hit = any(
            e.candidate_id in active_ids
            and e.candidate_id != cand_id
            and e.supports is True
            and e.kind in ("kakao", "google")
            and (e.data or {}).get("target") is True
            for e in (ev or [])
        )
        if not other_target_hit:
            return True
    return False


def confidence(scores: dict[str, float], cands: list[Candidate], ev: list[Evidence]) -> str:
    """Compute rule-based confidence ('high', 'medium', or 'low') from scores and evidence."""
    active = [c for c in (cands or []) if not veto(c.id, ev)]
    if not active:
        return "low"
    ranked = sorted(active, key=lambda c: scores.get(c.id, float(c.route_score)), reverse=True)
    top = ranked[0]
    top_score = float(scores.get(top.id, top.route_score))
    second_score = float(scores.get(ranked[1].id, ranked[1].route_score)) if len(ranked) >= 2 else 0.0
    gap = top_score - second_score if len(ranked) >= 2 else top_score

    active_ids = {c.id for c in active}
    top_supports = [
        e
        for e in (ev or [])
        if e.candidate_id == top.id and e.kind != "route" and e.supports is True
    ]
    has_relation = any(e.kind == "relation" for e in top_supports)
    has_decisive = _has_decisive_evidence(top.id, active_ids, ev)

    relaxations = list(top.relaxations or [])
    if has_relation or has_decisive:
        relaxations = [
            r for r in relaxations if r not in ("landmark_near_end", "route_to_landmark")
        ]

    if gap < 0.3 or len(relaxations) >= 2 or top_score < 1.0:
        return "low"
    if gap >= 1.0 and (len(top_supports) >= 2 or has_decisive) and len(relaxations) == 0:
        return "high"
    return "medium"


def _cand_to_opt(mapping: dict | None) -> dict[str, int]:
    """Normalize a question mapping into {candidate_id: option_index}."""
    if not mapping:
        return {}
    out: dict[str, int] = {}
    for k, v in mapping.items():
        if isinstance(v, list):
            opt_idx = int(k)
            for cid in v:
                out[str(cid)] = opt_idx
        else:
            out[str(k)] = int(v)
    return out


def info_gain(q: Question, scores: dict[str, float]) -> float:
    """Compute expected entropy reduction in bits for a question over candidate softmax probabilities."""
    if not scores or not q or not q.mapping:
        return 0.0
    cids = list(scores.keys())
    if len(cids) <= 1:
        return 0.0
    max_s = max(float(scores[c]) for c in cids)
    exps = {c: math.exp(float(scores[c]) - max_s) for c in cids}
    total = sum(exps.values())
    probs = {c: exps[c] / total for c in cids}

    def _entropy(dist: list[float]) -> float:
        return -sum(p * math.log2(p) for p in dist if p > 1e-12)

    h_before = _entropy(list(probs.values()))
    c2o = _cand_to_opt(q.mapping)
    opts = sorted(set(c2o.values()))
    if len(opts) <= 1:
        return 0.0

    unmapped_p = sum(probs[c] for c in cids if c not in c2o)
    opt_weights: dict[int, float] = {}
    for o in opts:
        mapped_p = sum(probs[c] for c in cids if c2o.get(c) == o)
        opt_weights[o] = mapped_p + unmapped_p / len(opts)

    h_after = 0.0
    for o in opts:
        p_o = opt_weights[o]
        if p_o <= 1e-12:
            continue
        post_scores = {
            c: float(scores[c]) + (1.0 if c2o.get(c) == o else (-1.0 if c in c2o else 0.0))
            for c in cids
        }
        # When every candidate is mapped and answered cleanly, partition directly for exact split entropy
        if unmapped_p <= 1e-12:
            group = [probs[c] / p_o for c in cids if c2o.get(c) == o]
            h_after += p_o * _entropy(group)
        else:
            m_s = max(post_scores.values())
            p_exps = [math.exp(post_scores[c] - m_s) for c in cids]
            p_tot = sum(p_exps)
            h_after += p_o * _entropy([x / p_tot for x in p_exps])

    return max(0.0, h_before - h_after)


def _candidate_side(c: Candidate) -> str | None:
    """Compute whether candidate centroid lies to the left or right of the final walking segment."""
    if not c.path or len(c.path) < 2:
        return None
    p_prev, p_end = c.path[-2], c.path[-1]
    hx, hy = geo.to_xy(p_end, p_prev)
    bx, by = geo.to_xy(c.centroid, p_end)
    if (hx * hx + hy * hy) < 1e-6 or (bx * bx + by * by) < 1e-6:
        return None
    side = geo.side_of((hx, hy), (bx, by))
    return side if side in ("left", "right") else None


def pick_question(
    pr: ParseResult,
    cands: list[Candidate],
    ev: list[Evidence],
    scores: dict[str, float],
    asked: list[str],
) -> Question | None:
    """Select the highest-information-gain unasked question across candidate attributes and relaxations."""
    asked_set = set(asked or [])
    active = [c for c in (cands or []) if not veto(c.id, ev)]
    if not active:
        active = list(cands or [])

    questions: list[Question] = []

    # 1. Floor question from registry evidence
    if "Q-floors" not in asked_set:
        floors_by_cid: dict[str, int] = {}
        for e in ev or []:
            if e.kind == "registry" and (e.data or {}).get("grnd_floors") is not None:
                floors_by_cid[e.candidate_id] = int(e.data["grnd_floors"])
        for c in active:
            if c.id not in floors_by_cid and getattr(c, "levels", None):
                floors_by_cid[c.id] = int(c.levels)
        distinct_floors = sorted({floors_by_cid[c.id] for c in active if c.id in floors_by_cid})
        if len(distinct_floors) >= 2:
            opts = [f"{f}층" for f in distinct_floors[:4]]
            f_to_idx = {f: i for i, f in enumerate(distinct_floors[:4])}
            mapping = {
                c.id: f_to_idx[floors_by_cid[c.id]]
                for c in active
                if floors_by_cid.get(c.id) in f_to_idx
            }
            questions.append(
                Question(
                    id="Q-floors",
                    intent_ko="찾으시는 건물이 전체 몇 층짜리인가요?",
                    options_ko=opts,
                    source="attribute",
                    mapping=mapping,
                )
            )

    # 2. Walking direction left/right question
    if "Q-side" not in asked_set:
        sides: dict[str, int] = {}
        for c in active:
            s = _candidate_side(c)
            if s == "left":
                sides[c.id] = 0
            elif s == "right":
                sides[c.id] = 1
        if len(set(sides.values())) >= 2:
            questions.append(
                Question(
                    id="Q-side",
                    intent_ko="걸어가시는 방향에서 그 건물이 길 왼쪽에 있었나요, 오른쪽에 있었나요?",
                    options_ko=["왼쪽", "오른쪽"],
                    source="attribute",
                    mapping=sides,
                )
            )

    # 3. Ordinal relaxation question
    if "Q-ordinal" not in asked_set:
        raw = (pr.raw if pr else None) or {}
        steps = raw.get("steps") or []
        spoken_ord = None
        for st in steps:
            tt = (st or {}).get("turn_target") or {}
            if tt.get("ordinal"):
                spoken_ord = int(tt["ordinal"])
                break
        has_minus = any("ordinal-1" in (c.relaxations or []) for c in active)
        has_plus = any("ordinal+1" in (c.relaxations or []) for c in active)
        if spoken_ord is not None and (has_minus or has_plus):
            alt_ord = spoken_ord - 1 if has_minus else spoken_ord + 1
            mapping = {}
            for c in active:
                rels = c.relaxations or []
                if "ordinal-1" in rels or "ordinal+1" in rels:
                    mapping[c.id] = 0
                else:
                    mapping[c.id] = 1
            if len(set(mapping.values())) >= 2:
                questions.append(
                    Question(
                        id="Q-ordinal",
                        intent_ko=f"골목이 {spoken_ord}번째가 아니라 혹시 {alt_ord}번째 골목이었을까요?",
                        options_ko=[f"{alt_ord}번째 골목", f"{spoken_ord}번째 골목"],
                        source="relaxation",
                        mapping=mapping,
                    )
                )

    # 4. High-impact parsed ambiguities
    raw = (pr.raw if pr else None) or {}
    for idx, amb in enumerate(raw.get("ambiguities") or []):
        if (amb or {}).get("impact") != "high":
            continue
        qid = f"Q-amb-{idx + 1}"
        if qid in asked_set:
            continue
        opts = list((amb.get("options_ko") or [])[:4])
        if len(opts) < 2:
            opts = ["네, 맞아요", "아니요, 달라요"]
        mapping = {c.id: i % len(opts) for i, c in enumerate(active)}
        questions.append(
            Question(
                id=qid,
                intent_ko=amb.get("question_ko") or "어느 쪽에 더 가까운가요?",
                options_ko=opts,
                source="ambiguity",
                mapping=mapping,
            )
        )

    if not questions:
        return None

    best_q = max(questions, key=lambda q: info_gain(q, scores))
    if info_gain(best_q, scores) <= 0.0 and best_q.source != "ambiguity":
        return None
    return best_q


def apply_answer(
    q: Question,
    option_index: int,
    cands: list[Candidate],
    scores: dict[str, float],
) -> dict[str, float]:
    """Update candidate scores (+1.0 for matching option, -1.0 for other mapped options)."""
    updated = {k: float(v) for k, v in (scores or {}).items()}
    if option_index is None or option_index < 0 or not q or not q.mapping:
        return updated
    c2o = _cand_to_opt(q.mapping)
    if option_index not in set(c2o.values()):
        return updated
    for c in cands or []:
        if c.id not in c2o:
            continue
        if c2o[c.id] == option_index:
            updated[c.id] = updated.get(c.id, float(c.route_score)) + 1.0
        else:
            updated[c.id] = updated.get(c.id, float(c.route_score)) - 1.0
    return updated


def answer_map(
    pr: ParseResult,
    cands: list[Candidate],
    ev: list[Evidence],
    scores: dict[str, float],
    top_n: int = 3,
) -> dict:
    """Build the structured answer map dictionary for the Gemini judge."""
    ranked = sorted(
        cands or [],
        key=lambda c: scores.get(c.id, float(c.route_score)),
        reverse=True,
    )
    cand_items = []
    for c in ranked[:top_n]:
        cand_items.append(
            {
                "id": c.id,
                "type": c.type,
                "name": c.name,
                "code_score": round(float(scores.get(c.id, c.route_score)), 2),
                "route_score": round(float(c.route_score), 2),
                "path_points": len(c.path or []),
                "side": _candidate_side(c),
                "relaxations": list(c.relaxations or []),
                "vetoed": veto(c.id, ev),
            }
        )
    ev_items = [
        {
            "id": e.id,
            "kind": e.kind,
            "candidate_id": e.candidate_id,
            "supports": e.supports,
            "weight": e.weight,
            "summary_ko": e.summary_ko,
            "capture_date": e.capture_date,
        }
        for e in (ev or [])
    ]
    raw = (pr.raw if pr else None) or {}
    return {
        "note": "좌우·순서·거리는 이미 코드가 계산한 값이므로 재추론하지 마라.",
        "source_text": pr.source_text if pr else "",
        "target": raw.get("target") or {},
        "candidates": cand_items,
        "evidence": ev_items,
    }


def judge_contents(
    pr: ParseResult,
    cands: list[Candidate],
    ev: list[Evidence],
    scores: dict[str, float],
    photo: bytes | None = None,
    photo_role: str = "target",
) -> list:
    """Assemble multimodal prompt contents (answer map JSON, user photo, and up to 3 street photos)."""
    amap = answer_map(pr, cands, ev, scores)
    parts: list = [json.dumps(amap, ensure_ascii=False)]

    if photo:
        if photo_role == "landmark":
            up_ev = next((e for e in (ev or []) if e.kind == "user_photo"), None)
            eid = (up_ev.id if up_ev and up_ev.id else "E4")
            place = ((up_ev.data or {}).get("place") if up_ev else None) or "기준 가게"
            label = (
                f"사용자 사진: 사용자가 '지나는 가게(기준 랜드마크)'로 지정 — 찾는 곳이 아니다. "
                f"간판을 읽어 카카오 '{place}'로 확정({eid}). "
                f"사진 속 메뉴·현수막으로 역할을 뒤집지 마라."
            )
        else:
            label = "사용자 사진: 사용자가 '찾는 곳'으로 지정한 목표의 최신 모습"
        parts.append(label)
        parts.append(types.Part.from_bytes(data=photo, mime_type="image/jpeg"))

    # Group street photos by candidate and capture viewpoint (up to JUDGE_PHOTOS)
    ranked_ids = [
        c.id
        for c in sorted(
            cands or [],
            key=lambda x: scores.get(x.id, float(x.route_score)),
            reverse=True,
        )
    ]
    rank_order = {cid: idx for idx, cid in enumerate(ranked_ids)}

    groups: list[dict] = []
    seen_groups: dict[tuple, dict] = {}
    for e in ev or []:
        if not e.image_jpeg:
            continue
        key = (e.candidate_id, e.capture_date or "", id(e.image_jpeg) if not e.capture_date else e.image_jpeg[:32])
        if key not in seen_groups:
            g = {
                "candidate_id": e.candidate_id,
                "capture_date": e.capture_date or "미상",
                "image_jpeg": e.image_jpeg,
                "eids": [e.id] if e.id else [],
            }
            seen_groups[key] = g
            groups.append(g)
        else:
            if e.id and e.id not in seen_groups[key]["eids"]:
                seen_groups[key]["eids"].append(e.id)

    groups.sort(key=lambda g: rank_order.get(g["candidate_id"], 999))
    for idx, g in enumerate(groups[:JUDGE_PHOTOS], start=1):
        eids_str = ", ".join(g["eids"])
        label = f"사진 {idx}: 근거 {eids_str}, 후보 {g['candidate_id']}, 촬영 {g['capture_date']}"
        parts.append(label)
        parts.append(types.Part.from_bytes(data=g["image_jpeg"], mime_type="image/jpeg"))

    return parts


def judge_with_gemini(
    pr: ParseResult,
    cands: list[Candidate],
    ev: list[Evidence],
    scores: dict[str, float],
    photo: bytes | None = None,
    photo_role: str = "target",
) -> dict:
    """Call the Gemini judge model chain with structured output and no timeout retries."""
    contents = judge_contents(pr, cands, ev, scores, photo=photo, photo_role=photo_role)
    out, model_label = llm.generate_json_chain(
        "judge",
        [config.MODEL_JUDGE, config.MODEL_MAIN],
        contents,
        JUDGE_SCHEMA,
        thinking="medium",
        system=P_JUDGE,
        timeout_ms=JUDGE_TIMEOUT_MS,
        retry_timeouts=False,
    )
    res = dict(out or {})
    res["_model"] = model_label
    return res


def validate_judge(
    out: dict,
    known_ids: set[str],
    cand_ids: set[str],
    code_top: str,
    owners: dict[str, str] | None = None,
) -> dict:
    """Strip unknown or foreign-candidate evidence IDs from judge reasons and fall back if needed."""
    res = dict(out or {})
    choice = res.get("choice") or ""
    unknown_ids = 0
    foreign_ids = 0
    cleaned_reasons: list[dict] = []

    valid_choice_for_owners = choice if choice in cand_ids else code_top
    for r in res.get("reasons") or []:
        kept: list[str] = []
        for eid in r.get("evidence_ids") or []:
            if eid not in known_ids:
                unknown_ids += 1
                continue
            if owners is not None and choice in cand_ids:
                owner = owners.get(eid, "")
                if owner not in ("", valid_choice_for_owners):
                    foreign_ids += 1
                    continue
            kept.append(eid)
        if kept:
            cleaned_reasons.append({"text_ko": r.get("text_ko") or "", "evidence_ids": kept})

    fallback = False
    if choice not in cand_ids and choice != "none":
        choice = code_top
        fallback = True
    elif choice != "none" and not cleaned_reasons:
        choice = code_top
        fallback = True

    res["choice"] = choice
    res["reasons"] = cleaned_reasons
    res["unknown_ids"] = unknown_ids
    res["foreign_ids"] = foreign_ids
    res["fallback"] = fallback
    return res


def target_evidenced(pr: ParseResult, cand: Candidate | None, ev: list[Evidence]) -> bool | None:
    """Check whether a candidate has required target business or spatial relation evidence."""
    if cand is None:
        return None
    raw = (pr.raw if pr else None) or {}
    target = raw.get("target") or {}
    t_kind = target.get("kind") or ""
    if cand.type in ("alley", "intersection", "point") or t_kind in ("alley", "intersection", "point"):
        return None
    if cand.type == "named_place":
        return True

    cand_ev = [e for e in (ev or []) if e.candidate_id == cand.id]
    relation = target.get("relation")
    has_rel_Need = relation in ("across", "beside")
    if has_rel_Need:
        if any(e.kind == "relation" and e.supports is False for e in cand_ev):
            return False

    if _is_business_target(pr):
        has_biz = any(
            e.supports is True
            and (
                (e.kind in ("kakao", "google", "user_photo") and (e.data or {}).get("target") is True)
                or (e.kind == "registry" and (e.data or {}).get("name_match") is True)
                or (e.kind == "vision" and bool((e.data or {}).get("sign")))
            )
            for e in cand_ev
        )
        if not has_biz:
            return False
        if has_rel_Need and any(e.kind == "relation" for e in cand_ev):
            return any(e.kind == "relation" and e.supports is True for e in cand_ev)
        return True

    if has_rel_Need:
        if any(e.kind == "relation" and e.supports is True for e in cand_ev):
            return True
        if any(e.kind == "relation" for e in cand_ev):
            return False

    return None


def _name_question(pr: ParseResult) -> Question:
    """Build a follow-up question asking for the shop or building name of the target."""
    raw = (pr.raw if pr else None) or {}
    target = raw.get("target") or {}
    label = (target.get("name") or "").strip() or "찾으시는 곳"
    return Question(
        id="Q-name",
        intent_ko=f"그 {label}의 상호(가게 이름)를 아시나요?",
        options_ko=[],
        source="name",
        mapping=None,
    )


def _fallback_reasons(cand_id: str, ev: list[Evidence]) -> list[dict]:
    """Build up to 2 fallback reasons from the highest-weight supporting evidence for cand_id."""
    supp = [
        e
        for e in (ev or [])
        if e.candidate_id == cand_id and e.supports is not False and e.id
    ]
    supp.sort(key=lambda e: float(e.weight), reverse=True)
    out = []
    for e in supp[:2]:
        if e.summary_ko:
            out.append({"text_ko": e.summary_ko, "evidence_ids": [e.id]})
    return out


def decide(
    pr: ParseResult,
    cands: list[Candidate],
    ev: list[Evidence],
    scores: dict[str, float],
    asked: list[str],
    use_gemini: bool = True,
    photo: bytes | None = None,
    photo_role: str = "target",
) -> tuple[Decision, dict]:
    """Combine code scores, registry vetoes, Gemini judgment, and target checks into a Decision."""
    asked_list = list(asked or [])
    asked_set = set(asked_list)
    vetoed_ids = [c.id for c in (cands or []) if veto(c.id, ev)]
    active = [c for c in (cands or []) if c.id not in set(vetoed_ids)]
    all_vetoed = bool(cands) and not active

    ranked_all = sorted(
        cands or [],
        key=lambda c: scores.get(c.id, float(c.route_score)),
        reverse=True,
    )
    ranked_active = sorted(
        active,
        key=lambda c: scores.get(c.id, float(c.route_score)),
        reverse=True,
    )
    code_top = (ranked_active[0].id if ranked_active else (ranked_all[0].id if ranked_all else None))
    code_conf = "low" if all_vetoed else confidence(scores, cands, ev)

    meta: dict = {
        "vetoed": vetoed_ids,
        "all_vetoed": all_vetoed,
        "unknown_ids": 0,
        "foreign_ids": 0,
        "fallback": False,
        "alternatives": [c.id for c in (ranked_active or ranked_all)[:3]],
    }

    if not cands:
        return Decision(kind="low_confidence", candidate_id=None, confidence="low", reasons=[]), meta

    choice = code_top
    final_conf = code_conf
    reasons: list[dict] = []

    if use_gemini and not all_vetoed and code_top is not None:
        try:
            raw_j = judge_with_gemini(pr, cands, ev, scores, photo=photo, photo_role=photo_role)
            known_ids = {e.id for e in (ev or []) if e.id}
            cand_ids = {c.id for c in (cands or [])}
            owners = {e.id: (e.candidate_id or "") for e in (ev or []) if e.id}
            val = validate_judge(raw_j, known_ids, cand_ids, code_top, owners=owners)
            meta["unknown_ids"] = val.get("unknown_ids", 0)
            meta["foreign_ids"] = val.get("foreign_ids", 0)
            meta["fallback"] = bool(val.get("fallback"))

            j_choice = val.get("choice")
            j_conf = val.get("confidence") or code_conf
            if j_choice == "none":
                choice = code_top
                final_conf = "low"
                reasons = val.get("reasons") or []
            elif j_choice in set(vetoed_ids):
                meta["judge_vetoed"] = j_choice
                choice = code_top
                final_conf = code_conf
                reasons = _fallback_reasons(choice, ev)
            else:
                choice = j_choice or code_top
                if _CONF_RANK.get(j_conf, 1) < _CONF_RANK.get(code_conf, 1):
                    final_conf = j_conf
                reasons = val.get("reasons") or []
        except Exception as exc:
            meta["judge_error"] = f"{type(exc).__name__}: {exc}"
            meta["fallback"] = True
            choice = code_top
            final_conf = code_conf

    if not reasons and choice:
        reasons = _fallback_reasons(choice, ev)

    if all_vetoed:
        return (
            Decision(
                kind="low_confidence",
                candidate_id=choice,
                confidence="low",
                reasons=reasons,
            ),
            meta,
        )

    chosen_cand = next((c for c in (cands or []) if c.id == choice), None)
    t_ev = target_evidenced(pr, chosen_cand, ev)
    if t_ev is False:
        meta["target_unverified"] = True
        if "Q-name" not in asked_set and len(asked_list) < 3:
            nq = _name_question(pr)
            return (
                Decision(
                    kind="question",
                    candidate_id=choice,
                    confidence="low",
                    reasons=reasons,
                    question=nq,
                ),
                meta,
            )
        final_conf = "low"

    raw = (pr.raw if pr else None) or {}
    has_high_amb = any((a or {}).get("impact") == "high" for a in (raw.get("ambiguities") or []))

    if final_conf == "high" and t_ev is not False:
        return (
            Decision(
                kind="answer",
                candidate_id=choice,
                confidence="high",
                reasons=reasons,
            ),
            meta,
        )

    if final_conf == "medium" and not has_high_amb and t_ev is not False:
        return (
            Decision(
                kind="answer",
                candidate_id=choice,
                confidence="medium",
                reasons=reasons,
            ),
            meta,
        )

    if len(asked_list) < 3:
        q = pick_question(pr, cands, ev, scores, asked=asked_list)
        if q is not None and q.id not in asked_set:
            return (
                Decision(
                    kind="question",
                    candidate_id=choice,
                    confidence=final_conf,
                    reasons=reasons,
                    question=q,
                ),
                meta,
            )

    return (
        Decision(
            kind="low_confidence",
            candidate_id=choice,
            confidence="low",
            reasons=reasons,
        ),
        meta,
    )


def phrase_question(q: Question, lang: str = "ko") -> Question:
    """Rephrase a Question into natural language using Flash-Lite while keeping valid options."""
    if q is None:
        return q
    dunno = DUNNO.get(lang, DUNNO["ko"])
    if not q.options_ko:
        return q
    try:
        prompt = P_QUESTION.format(
            lang="한국어" if lang == "ko" else "English",
            dunno=dunno,
            intent=q.intent_ko,
            options=", ".join(q.options_ko),
        )
        out = llm.generate_json(
            "question",
            config.MODEL_LIGHT,
            prompt,
            QUESTION_SCHEMA,
            thinking="minimal",
            timeout_ms=LIGHT_TIMEOUT_MS,
        )
        new_q = (out or {}).get("question") or q.intent_ko
        new_opts = list((out or {}).get("options") or [])
        base_without_dunno = [o for o in q.options_ko if o != dunno]
        filtered_new = [o for o in new_opts if o != dunno]
        if len(filtered_new) == len(base_without_dunno):
            opts = filtered_new + [dunno]
        else:
            opts = base_without_dunno + [dunno]
        return Question(
            id=q.id,
            intent_ko=new_q,
            options_ko=opts,
            source=q.source,
            mapping=q.mapping,
        )
    except Exception:
        base_without_dunno = [o for o in q.options_ko if o != dunno]
        return Question(
            id=q.id,
            intent_ko=q.intent_ko,
            options_ko=base_without_dunno + [dunno],
            source=q.source,
            mapping=q.mapping,
        )


def interpret_answer(q: Question, text: str) -> int:
    """Map free-form user answer text to an option index (or -1 for unknown/dunno)."""
    if not text or not text.strip():
        return -1
    clean = text.strip()
    if clean in DUNNO.values() or "모르" in clean or "몰라" in clean:
        return -1
    if not q or not q.options_ko:
        return -1
    for idx, opt in enumerate(q.options_ko):
        if opt in DUNNO.values():
            continue
        if clean == opt or opt in clean or clean in opt:
            return idx
    try:
        prompt = P_ANSWER.format(
            question=q.intent_ko,
            options=json.dumps(q.options_ko, ensure_ascii=False),
            answer=clean,
        )
        out = llm.generate_json(
            "answer",
            config.MODEL_LIGHT,
            prompt,
            ANSWER_SCHEMA,
            thinking="minimal",
            timeout_ms=LIGHT_TIMEOUT_MS,
        )
        idx = int((out or {}).get("option_index", -1))
        if 0 <= idx < len(q.options_ko):
            if q.options_ko[idx] in DUNNO.values():
                return -1
            return idx
    except Exception:
        pass
    return -1
