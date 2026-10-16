"""Evaluate end-to-end localization accuracy, baseline error, and safeguard rates across test cases."""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
from pathlib import Path
import statistics
import time

import shapely.geometry

from app import config, geo, mapdata, pipeline
from app.contracts import LatLng

BUILDING_LIKE = {"building", "named_place"}


def _locate(text: str, region: str | None = None) -> dict:
    """Run pipeline.locate for a single query text and optional region hint."""
    try:
        return pipeline.locate(text, region=region)
    except (AttributeError, TypeError):
        b0 = pipeline.baseline(text, region) if hasattr(pipeline, "baseline") else None
        if b0 is None and text and text.strip():
            try:
                from app import llm
                from app.contracts import B0_SCHEMA

                out = llm.generate_json(
                    "baseline",
                    config.MODEL_MAIN,
                    [pipeline.P_B0.format(region=region or "미지정", text=text.strip())],
                    B0_SCHEMA,
                    thinking="low",
                )
                if isinstance(out, dict) and "lat" in out and "lng" in out:
                    b0 = {
                        "model": config.MODEL_MAIN,
                        "address": out.get("address") or "",
                        "lnglat": [float(out["lng"]), float(out["lat"])],
                        "confidence": out.get("confidence") or "low",
                        "distance_to_result_m": None,
                    }
            except Exception:
                b0 = None
        return {
            "session_id": "s1",
            "decision": {"kind": "low_confidence", "candidate_id": None, "confidence": "low"},
            "candidates": [],
            "evidence": [],
            "baseline": b0,
            "stats": {},
        }


def _answer(session_id: str, text: str) -> dict:
    """Submit a follow-up answer string to pipeline.answer for session_id."""
    sig = inspect.signature(pipeline.answer)
    if "text" in sig.parameters:
        return pipeline.answer(session_id, text=text)
    if "answer_text" in sig.parameters:
        return pipeline.answer(session_id, answer_text=text)
    return pipeline.answer(session_id, None, text)


def _first_latlng(geom: dict) -> LatLng:
    """Extract a reference LatLng origin from a GeoJSON geometry dict."""
    gtype = (geom or {}).get("type")
    coords = (geom or {}).get("coordinates") or []
    if gtype == "Point" and len(coords) >= 2:
        return LatLng(lat=float(coords[1]), lng=float(coords[0]))
    if gtype == "LineString" and coords and len(coords[0]) >= 2:
        return LatLng(lat=float(coords[0][1]), lng=float(coords[0][0]))
    if gtype == "Polygon" and coords and coords[0] and len(coords[0][0]) >= 2:
        return LatLng(lat=float(coords[0][0][1]), lng=float(coords[0][0][0]))
    return config.VENUE


def _shape(geom: dict, origin: LatLng):
    """Convert a GeoJSON geometry into a local planar shapely shape in meters around origin."""
    gtype = (geom or {}).get("type")
    coords = (geom or {}).get("coordinates") or []
    if gtype == "Point":
        p = LatLng(lat=float(coords[1]), lng=float(coords[0]))
        return shapely.geometry.Point(geo.to_xy(p, origin))
    if gtype == "LineString":
        pts = [geo.to_xy(LatLng(lat=float(c[1]), lng=float(c[0])), origin) for c in coords]
        if len(pts) == 1:
            return shapely.geometry.Point(pts[0])
        return shapely.geometry.LineString(pts)
    if gtype == "Polygon":
        ring = coords[0] if coords else []
        pts = [geo.to_xy(LatLng(lat=float(c[1]), lng=float(c[0])), origin) for c in ring]
        if len(pts) >= 3:
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            return shapely.geometry.Polygon(pts)
        if len(pts) == 2:
            return shapely.geometry.LineString(pts)
        if len(pts) == 1:
            return shapely.geometry.Point(pts[0])
    raise ValueError(f"Unsupported GeoJSON geometry: {geom}")


def geom_distance(g1: dict, g2: dict) -> int:
    """Compute shortest distance in integer meters between two GeoJSON geometries (0 if overlapping)."""
    origin = _first_latlng(g1) if g1 else (_first_latlng(g2) if g2 else config.VENUE)
    s1 = _shape(g1, origin)
    s2 = _shape(g2, origin)
    if s1.intersects(s2):
        return 0
    return int(round(s1.distance(s2)))


def answer_geometry(ans: dict) -> dict | None:
    """Resolve expected target GeoJSON geometry from a case answer dictionary."""
    if not ans:
        return None
    if ans.get("geometry"):
        return ans["geometry"]
    way_id = ans.get("osm_way_id") if ans.get("osm_way_id") is not None else ans.get("expect_way")
    if way_id is not None:
        try:
            geom = mapdata.way_geometry(int(way_id))
            if geom is not None:
                return geom
        except Exception:
            pass
    ll = ans.get("lnglat") if ans.get("lnglat") is not None else ans.get("expect_lnglat")
    if ll is not None and len(ll) >= 2:
        return {"type": "Point", "coordinates": [float(ll[0]), float(ll[1])]}
    return None


def _decision_kind(resp: dict) -> str:
    """Extract decision kind ('answer', 'question', 'low_confidence') from a pipeline response."""
    dec = (resp or {}).get("decision") or {}
    return str(dec.get("kind") or dec.get("action") or "low_confidence")


def _cand_geom(cand: dict) -> dict | None:
    """Extract GeoJSON geometry from a candidate dict in a pipeline response."""
    if not cand:
        return None
    if cand.get("geometry"):
        return cand["geometry"]
    c = cand.get("centroid") or cand.get("lnglat")
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return {"type": "Point", "coordinates": [float(c[0]), float(c[1])]}
    if isinstance(c, dict) and "lat" in c and "lng" in c:
        return {"type": "Point", "coordinates": [float(c["lng"]), float(c["lat"])]}
    return None


def _score(
    text: str,
    region: str | None,
    ans: dict,
    ans_geom: dict | None,
    followups: list[str],
    tolerance_m: int = 30,
) -> dict:
    """Run one evaluation pass (with optional follow-up answers) and score against ans_geom."""
    t0 = time.monotonic()
    resp = _locate(text, region)
    first_decision = _decision_kind(resp)

    b0_error_m: int | None = None
    b0 = (resp or {}).get("baseline")
    if isinstance(b0, dict) and b0.get("lnglat") and ans_geom is not None:
        ll = b0["lnglat"]
        b0_geom = {"type": "Point", "coordinates": [float(ll[0]), float(ll[1])]}
        b0_error_m = geom_distance(b0_geom, ans_geom)

    first_pass_hit = False
    if first_decision != "question" and ans_geom is not None:
        cands0 = (resp or {}).get("candidates") or []
        dec0 = (resp or {}).get("decision") or {}
        cid0 = dec0.get("candidate_id") or dec0.get("choice_id")
        chosen0 = next((c for c in cands0 if c.get("id") == cid0), cands0[0] if cands0 else None)
        cg0 = _cand_geom(chosen0) if chosen0 else None
        if cg0 is not None:
            first_pass_hit = geom_distance(cg0, ans_geom) <= tolerance_m

    questions_asked = 0
    remaining = list(followups or [])
    while _decision_kind(resp) == "question" and remaining:
        q_ans = remaining.pop(0)
        sid = (resp or {}).get("session_id") or "s1"
        questions_asked += 1
        resp = _answer(sid, q_ans)

    elapsed_s = round(time.monotonic() - t0, 2)
    decision = _decision_kind(resp)
    ended_with_question = decision == "question"
    dec = (resp or {}).get("decision") or {}
    confidence = dec.get("confidence") or "low"

    error_m: int | None = None
    hit30_at1 = False
    hit30_at3 = False
    after_question_hit = False
    type_match = False

    if not ended_with_question and ans_geom is not None:
        cands = (resp or {}).get("candidates") or []
        cid = dec.get("candidate_id") or dec.get("choice_id")
        chosen = next((c for c in cands if c.get("id") == cid), cands[0] if cands else None)
        if chosen is not None:
            cg = _cand_geom(chosen)
            if cg is not None:
                error_m = geom_distance(cg, ans_geom)
                hit30_at1 = error_m <= 30
                after_question_hit = error_m <= tolerance_m
            top3_geoms = [g for g in (_cand_geom(c) for c in cands[:3]) if g is not None]
            hit30_at3 = hit30_at1 or any(geom_distance(g, ans_geom) <= 30 for g in top3_geoms)

            c_type = chosen.get("type") or chosen.get("kind")
            a_type = (ans or {}).get("type") or (ans or {}).get("expect_kind")
            if c_type and a_type:
                type_match = bool(
                    c_type == a_type or (c_type in BUILDING_LIKE and a_type in BUILDING_LIKE)
                )

    stats = (resp or {}).get("stats") or {}
    evs = (resp or {}).get("evidence") or []
    floor_evs = [
        e
        for e in evs
        if isinstance(e, dict) and (e.get("data") or {}).get("attribute") == "target_floors"
    ]
    floors_checked = int(
        stats.get("floors_checked", len(floor_evs))
        if "floors_checked" in stats
        else len(floor_evs)
    )
    floors_wrong = int(
        stats.get(
            "floors_wrong",
            sum(
                1
                for e in floor_evs
                if (e.get("data") or {}).get("match") is False or e.get("supports") is False
            ),
        )
    )

    return {
        "first_decision": first_decision,
        "decision": decision,
        "ended_with_question": ended_with_question,
        "questions_asked": questions_asked,
        "confidence": confidence,
        "elapsed_s": elapsed_s,
        "error_m": error_m,
        "b0_error_m": b0_error_m,
        "hit30_at1": hit30_at1,
        "hit30_at3": hit30_at3,
        "first_pass_hit": first_pass_hit,
        "after_question_hit": after_question_hit,
        "type_match": type_match,
        "invented": int(stats.get("invented", 0) or 0),
        "flipped": int(stats.get("vision_flipped", stats.get("flipped", 0)) or 0),
        "first_present": int(stats.get("vision_first_present", stats.get("first_present", 0)) or 0),
        "floors_checked": floors_checked,
        "floors_wrong": floors_wrong,
    }


def run_case(case: dict) -> dict:
    """Evaluate a single benchmark case, re-running without region when a region hint is present."""
    text = case.get("text") or ""
    region = case.get("region_input") if "region_input" in case else case.get("region")
    tolerance_m = int(case.get("tolerance_m") or 30)

    ans = case.get("answer")
    if not isinstance(ans, dict):
        ans = {
            "type": case.get("expect_kind"),
            "osm_way_id": case.get("expect_way"),
            "lnglat": case.get("expect_lnglat"),
        }
    ans_geom = answer_geometry(ans)

    followups = list(case.get("followups") or [])
    if not followups and case.get("answer_if_asked"):
        followups = [str(case["answer_if_asked"])]

    row = _score(text, region, ans, ans_geom, followups, tolerance_m=tolerance_m)
    row["id"] = case.get("id")
    if region is not None:
        row["no_region"] = _score(text, None, ans, ans_geom, followups, tolerance_m=tolerance_m)
    else:
        row["no_region"] = None
    return row


def summarize(rows: list[dict]) -> tuple[dict, dict]:
    """Aggregate case rows into English and Korean summary metrics."""
    n = len(rows)
    sys_errs = [r["error_m"] for r in rows if r.get("error_m") is not None]
    b0_errs = [r["b0_error_m"] for r in rows if r.get("b0_error_m") is not None]

    sys_med = int(round(statistics.median(sys_errs))) if sys_errs else None
    b0_med = int(round(statistics.median(b0_errs))) if b0_errs else None

    hit1 = sum(1 for r in rows if r.get("hit30_at1"))
    hit3 = sum(1 for r in rows if r.get("hit30_at3"))
    tmatch = sum(1 for r in rows if r.get("type_match"))

    flipped = sum(int(r.get("flipped", 0) or 0) for r in rows)
    first_present = sum(int(r.get("first_present", 0) or 0) for r in rows)
    v_halluc_rate = round(flipped / first_present, 4) if first_present > 0 else 0.0

    floors_wrong = sum(int(r.get("floors_wrong", 0) or 0) for r in rows)
    floors_checked = sum(int(r.get("floors_checked", 0) or 0) for r in rows)
    f_err_rate = round(floors_wrong / floors_checked, 4) if floors_checked > 0 else 0.0

    invented_total = sum(int(r.get("invented", 0) or 0) for r in rows)

    reg_rows = [r for r in rows if r.get("no_region") is not None]
    paired_with_errs = [
        r["error_m"]
        for r in reg_rows
        if r.get("error_m") is not None and r["no_region"].get("error_m") is not None
    ]
    if not paired_with_errs:
        paired_with_errs = [r["error_m"] for r in reg_rows if r.get("error_m") is not None]
    without_errs = [
        r["no_region"]["error_m"]
        for r in reg_rows
        if r["no_region"].get("error_m") is not None
    ]
    reg_with_med = int(round(statistics.median(paired_with_errs))) if paired_with_errs else None
    reg_without_med = int(round(statistics.median(without_errs))) if without_errs else None
    reg_without_q = sum(1 for r in reg_rows if r["no_region"].get("ended_with_question"))

    s = {
        "cases": n,
        "system_error_median_m": sys_med,
        "b0_error_median_m": b0_med,
        "hit30_at1": f"{hit1}/{n}",
        "hit30_at3": f"{hit3}/{n}",
        "type_match": f"{tmatch}/{n}",
        "vision_hallucination_rate": v_halluc_rate,
        "floors_error_rate": f_err_rate,
        "invented_total": invented_total,
        "region_with_error_median_m": reg_with_med,
        "region_without_error_median_m": reg_without_med,
        "region_without_questions": reg_without_q,
    }
    ko = {
        "사례 수": n,
        "시스템 중앙 오차(m)": sys_med,
        "기준선(B0) 중앙 오차(m)": b0_med,
        "30m 이내 적중(@1)": s["hit30_at1"],
        "30m 이내 적중(@3)": s["hit30_at3"],
        "유형 일치": s["type_match"],
        "시각 착각 비율": v_halluc_rate,
        "층수 판독 오류율": f_err_rate,
        "지어낸 단서 제거 수": invented_total,
        "대략 위치 있음 중앙 오차(m)": reg_with_med,
        "대략 위치 없음 중앙 오차(m)": reg_without_med,
        "대략 위치 없을 때 되묻기 수": reg_without_q,
    }
    return s, ko


def main(argv: list[str] | None = None) -> int:
    """CLI entry point to run evaluation cases and write eval/results.json."""
    parser = argparse.ArgumentParser(description="Run localization evaluation suite")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of cases to run")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parent / "cases.json",
        help="Path to evaluation cases JSON file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results.json",
        help="Path to write evaluation results JSON",
    )
    args = parser.parse_args(argv)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.limit is not None:
        cases = cases[: args.limit]

    rows: list[dict] = []
    for c in cases:
        row = run_case(c)
        rows.append(row)

    s, ko = summarize(rows)
    n = len(rows)
    first_pass_acc = round(sum(1 for r in rows if r.get("first_pass_hit")) / n, 4) if n else 0.0
    after_q_acc = round(sum(1 for r in rows if r.get("after_question_hit")) / n, 4) if n else 0.0

    out_data = {
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n": n,
        "first_pass_acc": first_pass_acc,
        "after_question_acc": after_q_acc,
        "median_error_m": s["system_error_median_m"],
        "baseline_median_error_m": s["b0_error_median_m"],
        "summary": s,
        "summary_ko": ko,
        "per_case": rows,
        **s,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"id\tdecision\terror_m\tb0_error_m\thit30_at1\ttype_match")
    for r in rows:
        print(
            f"{r.get('id')}\t{r.get('decision')}\t{r.get('error_m')}\t"
            f"{r.get('b0_error_m')}\t{r.get('hit30_at1')}\t{r.get('type_match')}"
        )
    print(json.dumps(ko, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
