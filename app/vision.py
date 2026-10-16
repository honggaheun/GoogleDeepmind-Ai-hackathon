"""Verify route candidates using building registry records, user photos, and two-pass street-level imagery."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import datetime
import json
from pathlib import Path

from google.genai import types

from app import config, geo, llm, mapdata, registry, streetview
from app.contracts import (
     VERIFY_SCHEMA,
    VISION_SCHEMA,
    Candidate,
    Evidence,
    ParseResult,
    Pano,
    RegistryRecord,
    VisionStats,
)

PHOTO_POLICY = config.get("EODI_PHOTO_POLICY", "user_photo") or "user_photo"
GEMINI_WORKERS = 3
VISION_TIMEOUT_MS = 15_000
OLD_MONTHS = 60
MAX_IMAGES = 8
PHOTO_GAP = 0.03
AMBIGUOUS_GAP = 1.0

P_VISION_1 = (
    "You are checking a street-level photo (captured {capture_date}) for visual attributes of a described place.\n"
    "Attributes to check: {attributes}.\n"
    "For each attribute answer present, absent, or uncertain. If present, give box_2d as [ymin, xmin, ymax, xmax] normalized to 0-1000\n"
    "around the evidence. Also list the visible buildings with box_2d and the estimated number of above-ground floors.\n"
    "Do not guess: if the view is blocked or unclear, answer uncertain. Shop signs may be outdated because the photo is old;\n"
    "report sign text exactly as visible."
)

P_VISION_2 = (
    "Look at this zoomed street-level photo. Is there {description} near the center of the image? "
    "Answer yes, no, or uncertain."
)


def _target_landmark(pr: ParseResult) -> dict | None:
    """Return the landmark dict referenced by target.landmark_ref when it describes the target itself."""
    raw = pr.raw if isinstance(getattr(pr, "raw", None), dict) else {}
    target = raw.get("target") or {}
    if target.get("relation") in ("across", "beside"):
        return None
    ref = target.get("landmark_ref")
    if not ref:
        return None
    for lm in raw.get("landmarks") or []:
        if isinstance(lm, dict) and lm.get("id") == ref:
            return lm
    return None


def _unchanging_cues(pr: ParseResult) -> list[dict]:
    """Return target landmarks that have invariant visual cues (structural, colors, or floors)."""
    lm = _target_landmark(pr)
    if not lm:
        return []
    vis = lm.get("visual") or {}
    has_sign = bool(vis.get("sign_text"))
    if vis.get("floors") or (not has_sign and (vis.get("structural") or vis.get("colors"))):
        return [lm]
    return []


def should_verify(
    pr: ParseResult,
    cands: list[Candidate],
    photo: bytes | None = None,
    gap: float = AMBIGUOUS_GAP,
) -> tuple[bool, str]:
    """Decide whether street-level photo verification should run and return (should_run, reason_ko)."""
    if photo is not None:
        return True, "사용자 사진 대조"
    policy = (PHOTO_POLICY or "user_photo").strip().lower()
    if policy == "user_photo":
        return False, "사용자 사진 없음 → 거리 사진 생략"
    if not cands:
        return False, "후보 없음"
    if policy == "always":
        return True, "항상 확인"

    cues = _unchanging_cues(pr)
    if cues:
        return True, f"시각 단서 {len(cues)}개"
    if len(cands) == 1:
        return False, "시각 단서 없음 · 후보 1곳"

    scores = sorted((float(c.route_score) for c in cands), reverse=True)
    diff = scores[0] - scores[1]
    if diff < gap:
        return True, f"후보 점수 차 좁음({diff:.1f})"
    return False, f"시각 단서 없음 · 점수 차 충분({diff:.1f})"


def registries(cands: list[Candidate]) -> dict[str, RegistryRecord | None]:
    """Look up building registry records for building and named_place candidates by centroid."""
    out: dict[str, RegistryRecord | None] = {}
    for c in cands or []:
        if c.type in ("building", "named_place"):
            try:
                out[c.id] = registry.for_point(c.centroid)
            except Exception:
                out[c.id] = None
        else:
            out[c.id] = None
    return out


def attributes_for(pr: ParseResult, cand: Candidate) -> list[dict]:
    """Build the list of visual attributes to check for a candidate from ParseResult."""
    attrs: list[dict] = []
    lm = _target_landmark(pr)
    if lm:
        vis = lm.get("visual") or {}
        en = vis.get("en")
        if en:
            structural = bool(vis.get("structural"))
            sign = bool(vis.get("sign_text")) and not structural
            attrs.append(
                {
                    "name": lm["id"],
                    "desc": str(en),
                    "structural": structural,
                    "sign": sign,
                    "label": str(lm.get("evidence") or en),
                }
            )

    if cand.type == "named_place" and cand.name:
        attrs.append(
            {
                "name": "target_sign",
                "desc": f"sign or storefront for '{cand.name}'",
                "structural": False,
                "sign": True,
                "label": f"'{cand.name}' 간판",
            }
        )

    if cand.type in ("building", "named_place"):
        floors = ((lm or {}).get("visual") or {}).get("floors")
        attrs.append(
            {
                "name": "target_floors",
                "desc": (
                    f"building with {floors} above-ground floors"
                    if floors
                    else "number of above-ground floors of the building"
                ),
                "structural": True,
                "sign": False,
                "label": f"{floors}층 건물" if floors else "건물 층수",
            }
        )

    return attrs


def _valid_box(box) -> bool:
    """Return True if box is [ymin, xmin, ymax, xmax] within 0..1000."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if not all(isinstance(v, (int, float)) for v in box):
        return False
    ymin, xmin, ymax, xmax = box
    return 0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000


def clean_attributes(resp: dict) -> dict:
    """Validate 1st-pass attribute boxes and demote invalid present boxes to uncertain."""
    out: dict[str, dict] = {}
    for item in (resp or {}).get("attributes") or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        name = str(item["name"])
        status = str(item.get("status") or "uncertain")
        box = item.get("box_2d")
        if status == "present" and not _valid_box(box):
            status = "uncertain"
            box = None
        out[name] = {
            "name": name,
            "status": status,
            "box_2d": [int(v) for v in box] if (status == "present" and box) else None,
            "note": item.get("note"),
        }
    return out


def attribute_weight(
    first: str,
    second: str | None,
    structural: bool,
    sign: bool,
    old: bool,
) -> float:
    """Compute evidence weight for a visual attribute from 1st/2nd pass results and photo age."""
    if first != "present":
        return 0.0
    if second == "yes":
        w = 1.0 if structural else 0.5
    elif second == "no":
        w = -0.5
    else:
        w = 0.3
    if old and (sign or not structural):
        w *= 0.5
    return w


def box_bearing(
    heading_deg: float,
    box_2d: list[int | float],
    fov_deg: float = 90.0,
) -> float:
    """Return compass bearing of the horizontal center of box_2d [ymin, xmin, ymax, xmax]."""
    return geo.ray_bearing(heading_deg, fov_deg, float(box_2d[1]), float(box_2d[3]))


def apply_consistency(
    evidence: list[Evidence],
    outcomes: dict[tuple[str, str, int], str],
) -> None:
    """Reward attributes confirmed from two views (+0.3) and zero out contradictory views."""
    groups: dict[tuple[str, str], list[str]] = {}
    for (cid, attr, _vp), val in (outcomes or {}).items():
        groups.setdefault((cid, attr), []).append(val)

    for (cid, attr), vals in groups.items():
        evs = [
            e
            for e in evidence
            if e.kind == "vision"
            and e.candidate_id == cid
            and (e.data or {}).get("attribute") == attr
        ]
        if not evs:
            continue
        if "yes" in vals and "no" in vals:
            for e in evs:
                e.weight = 0.0
                e.supports = None
        elif vals.count("yes") >= 2:
            evs[0].weight = round(evs[0].weight + 0.3, 4)


def _months_old(capture_date: str | None) -> int:
    """Compute age of a YYYY-MM capture date in months relative to today."""
    if not capture_date:
        return 0
    today = datetime.date.today()
    try:
        parts = str(capture_date).strip().split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 1
        return (today.year - year) * 12 + (today.month - month)
    except (ValueError, IndexError):
        return 0


def _is_old(capture_date: str | None) -> bool:
    """Return True when capture_date is strictly older than OLD_MONTHS."""
    return _months_old(capture_date) > OLD_MONTHS


def _fetch_image(at, heading_deg: float, fov_deg: float = 90) -> bytes | None:
    """Fetch a street-view image using positional args with 2-arg fallback for test doubles."""
    try:
        if fov_deg == 90:
            return streetview.image(at, heading_deg)
        return streetview.image(at, heading_deg, fov_deg)
    except TypeError:
        try:
            return streetview.image(at, heading_deg)
        except Exception:
            return None
    except Exception:
        return None


def _call_generate_json(
    stage: str,
    model: str,
    contents: list,
    schema: dict,
    thinking: str | None = None,
) -> dict:
    """Call llm.generate_json with fallback for positional-only test doubles."""
    try:
        return llm.generate_json(
            stage,
            model,
            contents,
            schema,
            thinking=thinking,
            timeout_ms=VISION_TIMEOUT_MS,
            retry_timeouts=False,
        )
    except TypeError:
        return llm.generate_json(stage, model, contents, schema)


def _select(
    views: list[dict],
    pr: ParseResult,
    photo: bytes | None = None,
    limit: int = 3,
) -> list[dict]:
    """Rank wide street-view images by embedding similarity to description and optional user photo."""
    if not views:
        return []
    q_text = pr.source_text or (pr.raw.get("transcript") if isinstance(pr.raw, dict) else "") or "building"
    q_emb = llm.embed(q_text)
    p_emb = llm.embed(photo) if photo is not None else None

    for v in views:
        emb = llm.embed(v["image"])
        v["emb"] = emb
        t_sim = llm.cosine(q_emb, emb)
        if p_emb is not None:
            p_sim = llm.cosine(p_emb, emb)
            v["photo_sim"] = p_sim
            v["sim_score"] = (t_sim + p_sim) / 2.0
        else:
            v["sim_score"] = t_sim

    ranked = sorted(views, key=lambda item: item.get("sim_score", 0.0), reverse=True)
    return ranked[:limit]


def verify(
    cands: list[Candidate],
    pr: ParseResult,
    md: mapdata.MapData | None,
    regs: dict[str, RegistryRecord | None],
    max_images: int = MAX_IMAGES,
    photo: bytes | None = None,
) -> tuple[list[Evidence], VisionStats]:
    """Run street-view metadata checks, embedding selection, and 2-pass Gemini verification."""
    evidence: list[Evidence] = []
    stats = VisionStats()
    regs = regs or {}

    # 1. Check panorama metadata and filter stale captures
    vp_tasks: list[tuple[Candidate, int, object]] = []
    for cand in cands or []:
        for vp_idx, vp in enumerate((cand.view_points or [])[:2]):
            vp_tasks.append((cand, vp_idx, vp))

    if not vp_tasks:
        return evidence, stats

    panos: list[Pano | None] = []
    for _cand, _vp_idx, vp in vp_tasks:
        try:
            panos.append(streetview.metadata(vp.at))
        except Exception:
            panos.append(None)

    views: list[dict] = []
    for (cand, vp_idx, vp), pano in zip(vp_tasks, panos):
        if pano is None:
            continue
        reg = regs.get(cand.id)
        if reg and registry.photo_predates_building(pano.date, reg.use_apr_day):
            stats.skipped_stale += 1
            evidence.append(
                Evidence(
                    id=f"V{len(evidence) + 1}",
                    kind="staleness",
                    candidate_id=cand.id,
                    supports=None,
                    weight=0.0,
                    summary_ko=f"거리 사진({pano.date})이 건물 사용승인일({reg.use_apr_day}) 이전이라 생략",
                    data={"capture_date": pano.date, "use_apr_day": reg.use_apr_day},
                    capture_date=pano.date,
                )
            )
            continue

        old = _is_old(pano.date)
        if photo is not None and old:
            stats.skipped_stale += 1
            evidence.append(
                Evidence(
                    id=f"V{len(evidence) + 1}",
                    kind="staleness",
                    candidate_id=cand.id,
                    supports=None,
                    weight=0.0,
                    summary_ko=f"거리 사진({pano.date})이 5년 넘어 사용자 사진과 대조하지 않음",
                    data={"capture_date": pano.date},
                    capture_date=pano.date,
                )
            )
            continue

        if stats.images >= max_images:
            break
        img = _fetch_image(vp.at, vp.heading_deg, 90)
        if not img:
            continue
        stats.images += 1
        if old:
            stats.old_photos += 1
        views.append(
            {
                "cand": cand,
                "vp_idx": vp_idx,
                "vp": vp,
                "pano": pano,
                "old": old,
                "image": img,
            }
        )

    if not views:
        return evidence, stats

    # 2. Rank views with Embedding 2 and record user photo similarity
    selected = _select(views, pr, photo)

    if photo is not None:
        cand_best_sim: dict[str, float] = {}
        for v in views:
            cid = v["cand"].id
            sim = float(v.get("photo_sim", 0.0))
            if cid not in cand_best_sim or sim > cand_best_sim[cid]:
                cand_best_sim[cid] = sim

        if cand_best_sim:
            ranked_sims = sorted(cand_best_sim.items(), key=lambda kv: kv[1], reverse=True)
            top_cid, top_sim = ranked_sims[0]
            second_sim = ranked_sims[1][1] if len(ranked_sims) >= 2 else -1.0
            top_wins = (top_sim - second_sim) >= PHOTO_GAP

            for cid, sim in cand_best_sim.items():
                is_winner = cid == top_cid and top_wins
                evidence.append(
                    Evidence(
                        id=f"V{len(evidence) + 1}",
                        kind="user_photo",
                        candidate_id=cid,
                        supports=True if is_winner else None,
                        weight=0.5 if is_winner else 0.0,
                        summary_ko=(
                            f"사용자 사진과 가장 닮음 (유사도 {sim:.2f})"
                            if is_winner
                            else f"사용자 사진 닮음 비교 (유사도 {sim:.2f})"
                        ),
                        data={"similarity": sim},
                    )
                )

    # 3. First-pass visual check on selected wide views (max 3 concurrent Gemini calls)
    def _run_first_pass(v: dict) -> tuple[dict, dict | None, Exception | None]:
        cand = v["cand"]
        pano = v["pano"]
        attrs = attributes_for(pr, cand)
        check_attrs = [a for a in attrs if a["name"] != "target_floors"]
        attr_desc = (
            ", ".join(f"{a['name']} ({a['desc']})" for a in check_attrs)
            if check_attrs
            else "building facade and stories"
        )
        prompt = P_VISION_1.format(capture_date=pano.date, attributes=attr_desc)
        part = types.Part.from_bytes(data=v["image"], mime_type="image/jpeg")
        try:
            resp = _call_generate_json(
                "vision_1",
                config.MODEL_MAIN,
                [part, prompt],
                VISION_SCHEMA,
                thinking="low",
            )
            return v, resp, None
        except Exception as exc:
            return v, None, exc

    with ThreadPoolExecutor(max_workers=min(GEMINI_WORKERS, len(selected))) as pool:
        first_results = list(pool.map(_run_first_pass, selected))

    # 4. Second-pass verification on present attributes, ray-cast linking, and floor comparison
    outcomes: dict[tuple[str, str, int], str] = {}

    for v, resp, err in first_results:
        cand: Candidate = v["cand"]
        vp_idx: int = v["vp_idx"]
        vp = v["vp"]
        pano: Pano = v["pano"]
        old: bool = v["old"]

        if err is not None or resp is None:
            evidence.append(
                Evidence(
                    id=f"V{len(evidence) + 1}",
                    kind="vision",
                    candidate_id=cand.id,
                    supports=None,
                    weight=0.0,
                    summary_ko=f"거리 사진 1차 판독 실패 ({pano.date})",
                    data={"error": str(err), "vp": vp_idx},
                    capture_date=pano.date,
                )
            )
            continue

        cleaned = clean_attributes(resp)
        attrs = attributes_for(pr, cand)

        for attr in attrs:
            name = attr["name"]
            if name == "target_floors":
                continue
            info = cleaned.get(name)
            if not info or info["status"] != "present":
                continue

            stats.first_present += 1
            box = info["box_2d"]
            b_deg = box_bearing(vp.heading_deg, box)

            second_ans: str | None = None
            if stats.images < max_images:
                narrow_img = _fetch_image(pano.at, b_deg, 50)
                if narrow_img:
                    stats.images += 1
                    narrow_part = types.Part.from_bytes(data=narrow_img, mime_type="image/jpeg")
                    prompt2 = P_VISION_2.format(description=attr["desc"])
                    try:
                        r2 = _call_generate_json(
                            "vision_2",
                            config.MODEL_LIGHT,
                            [narrow_part, prompt2],
                            VERIFY_SCHEMA,
                            thinking=None,
                        )
                        second_ans = str((r2 or {}).get("answer") or "uncertain")
                    except Exception:
                        second_ans = "uncertain"
                    if second_ans == "no":
                        stats.flipped += 1

            w = attribute_weight("present", second_ans, attr["structural"], attr["sign"], old)
            outcomes[(cand.id, name, vp_idx)] = second_ans or "uncertain"
            supports = True if w > 0 else (False if w < 0 else None)
            evidence.append(
                Evidence(
                    id=f"V{len(evidence) + 1}",
                    kind="vision",
                    candidate_id=cand.id,
                    supports=supports,
                    weight=w,
                    summary_ko=f"{attr['label']} 시각 확인 ({second_ans or '1차'})",
                    data={
                        "attribute": name,
                        "first": "present",
                        "second": second_ans,
                        "vp": vp_idx,
                    },
                    capture_date=pano.date,
                    image_jpeg=v["image"],
                    boxes=[box],
                )
            )

            if second_ans != "no" and md is not None and cand.building_way_id is not None:
                hit = mapdata.first_hit(md, pano.at, b_deg)
                matched = hit is not None and hit[0].way_id == cand.building_way_id
                link_w = 0.5 if matched else -0.5
                evidence.append(
                    Evidence(
                        id=f"V{len(evidence) + 1}",
                        kind="link",
                        candidate_id=cand.id,
                        supports=matched,
                        weight=link_w,
                        summary_ko=(
                            f"{attr['label']} 방향이 후보 건물과 일치"
                            if matched
                            else f"{attr['label']} 방향이 다른 건물을 가리킴"
                        ),
                        data={
                            "attribute": name,
                            "bearing_deg": round(b_deg, 1),
                            "hit_way_id": hit[0].way_id if hit else None,
                        },
                        capture_date=pano.date,
                    )
                )

        # Compare estimated building floors against registry record
        reg = regs.get(cand.id)
        bldgs = (resp or {}).get("buildings") or []
        est_floors: int | None = None
        for b in bldgs:
            if isinstance(b, dict) and b.get("floors_estimate") is not None:
                est_floors = int(b["floors_estimate"])
                break
        if est_floors is not None and reg is not None and reg.grnd_floors:
            diff_fl = abs(est_floors - int(reg.grnd_floors))
            fl_ok = diff_fl <= 1
            fl_w = 0.5 if fl_ok else -0.5
            evidence.append(
                Evidence(
                    id=f"V{len(evidence) + 1}",
                    kind="vision",
                    candidate_id=cand.id,
                    supports=fl_ok,
                    weight=fl_w,
                    summary_ko=(
                        f"사진 추정 {est_floors}층 = 대장 지상 {reg.grnd_floors}층"
                        if fl_ok
                        else f"사진 추정 {est_floors}층 ≠ 대장 지상 {reg.grnd_floors}층"
                    ),
                    data={
                        "attribute": "target_floors",
                        "floors_estimate": est_floors,
                        "registry_floors": int(reg.grnd_floors),
                    },
                    capture_date=pano.date,
                )
            )

    apply_consistency(evidence, outcomes)
    return evidence, stats


def _main() -> None:
    """CLI entrypoint for testing visual verification policy and registry lookup on a query."""
    from app import parse, search

    parser = argparse.ArgumentParser(description="Run visual verification stage on a route query.")
    parser.add_argument("text", help="Route description in Korean or English")
    parser.add_argument("--region", default=None, help="Optional region hint")
    parser.add_argument("--photo", default=None, help="Optional path to user photo")
    args = parser.parse_args()

    photo_bytes = Path(args.photo).read_bytes() if args.photo else None
    pr = (
        parse.parse_text(args.text, region_hint=args.region)
        if hasattr(parse, "parse_text")
        else parse.parse(text=args.text, region_hint=args.region)
    )
    md = None
    cands: list[Candidate] = []
    try:
        found = search.find_candidates(pr, region=args.region)
        cands = getattr(found, "candidates", None) or (
            found.get("candidates") if isinstance(found, dict) else list(found or [])
        )
        md = getattr(found, "md", None) or (found.get("md") if isinstance(found, dict) else None)
    except TypeError:
        budget = search._Budget() if hasattr(search, "_Budget") else None
        win = search.resolve_window(pr, args.region, None, budget)
        md = mapdata.load_area(win.center if win else config.VENUE, 1000)
        anchors, _ = search.resolve_anchors(pr, win, md, budget)
        res = search.find_candidates(pr, win, anchors, md, budget)
        cands = res[0] if isinstance(res, tuple) else list(res or [])

    if not cands:
        from app import pipeline

        found = pipeline._call_find_candidates(pr, args.region, None)
        cands = list(getattr(found, "candidates", None) or [])
        md = getattr(found, "md", None) or md

    regs = registries(cands)
    ok, reason = should_verify(pr, cands, photo=photo_bytes)
    ev, stats = verify(cands, pr, md, regs, photo=photo_bytes) if ok else ([], VisionStats())
    print(
        json.dumps(
            {
                "candidates": len(cands),
                "registries": {
                    k: asdict(v) if v is not None else None for k, v in regs.items()
                },
                "should_verify": [ok, reason],
                "evidence_count": len(ev),
                "stats": asdict(stats),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
