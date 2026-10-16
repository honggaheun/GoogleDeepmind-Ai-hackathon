"""Unit tests for evaluation runner geometry distance, case scoring, and summary statistics."""

from __future__ import annotations

import pytest

from app import geo
from app.contracts import LatLng
from eval import run_eval

TOWER = {
    "type": "Polygon",
    "coordinates": [
        [
            [127.0632653, 37.506504],
            [127.0638001, 37.5066898],
            [127.0641203, 37.5061097],
            [127.0635855, 37.5059239],
            [127.0632653, 37.506504],
        ]
    ],
}
NORTH_VERTEX = LatLng(37.5066898, 127.0638001)


def _point(p: LatLng) -> dict:
    """Create a GeoJSON Point dictionary from a LatLng."""
    return {"type": "Point", "coordinates": [p.lng, p.lat]}


def _resp(chosen_at: LatLng, baseline_at: LatLng, kind: str = "answer") -> dict:
    """Create a fake pipeline response dictionary with a chosen candidate and baseline."""
    return {
        "session_id": "s1",
        "decision": {"kind": kind, "candidate_id": "C1" if kind == "answer" else None, "confidence": "high"},
        "candidates": [
            {
                "id": "C1",
                "type": "building",
                "geometry": _point(chosen_at),
                "centroid": [chosen_at.lng, chosen_at.lat],
            }
        ],
        "baseline": {"lnglat": [baseline_at.lng, baseline_at.lat]},
        "stats": {"invented": 0, "vision_flipped": 0, "vision_first_present": 0},
    }


def test_overlapping_geometries_are_zero():
    assert run_eval.geom_distance(TOWER, TOWER) == 0


def test_point_30m_from_polygon():
    p = geo.to_ll(0, 30, NORTH_VERTEX)
    assert run_eval.geom_distance(_point(p), TOWER) == pytest.approx(30, abs=2)


def test_answer_geometry_from_lnglat():
    assert run_eval.answer_geometry({"lnglat": [127.0, 37.5]}) == {
        "type": "Point",
        "coordinates": [127.0, 37.5],
    }


def test_run_case_scores_against_answer_and_reruns_without_region(monkeypatch):
    target = LatLng(37.50631, 127.063695)
    calls = []

    def fake_locate(text, region=None):
        calls.append(region)
        b0 = geo.to_ll(100, 0, target)
        return _resp(target, b0, kind="answer")

    monkeypatch.setattr(run_eval, "_locate", fake_locate)
    row = run_eval.run_case(
        {
            "id": 1,
            "text": "t",
            "region_input": "삼성동",
            "lang": "ko",
            "answer": {"type": "named_place", "lnglat": [target.lng, target.lat]},
        }
    )
    assert (row["error_m"], row["hit30_at1"], row["hit30_at3"], row["type_match"]) == (
        0,
        True,
        True,
        True,
    )
    assert row["b0_error_m"] == pytest.approx(100, abs=2) and calls == ["삼성동", None]
    assert row["no_region"]["error_m"] == 0


def test_question_case_has_no_error_but_keeps_b0(monkeypatch):
    target = LatLng(37.50631, 127.063695)
    monkeypatch.setattr(run_eval, "_locate", lambda text, region=None: _resp(target, target, kind="question"))
    row = run_eval.run_case(
        {
            "id": 2,
            "text": "t",
            "region_input": None,
            "lang": "ko",
            "answer": {"type": "point", "lnglat": [target.lng, target.lat]},
        }
    )
    assert row["ended_with_question"] is True and row["error_m"] is None and (row["b0_error_m"] == 0)


def test_followup_answers_question_and_scores_final(monkeypatch):
    target = LatLng(37.50631, 127.063695)
    asked = []
    monkeypatch.setattr(run_eval, "_locate", lambda text, region=None: _resp(target, target, kind="question"))

    def fake_answer(sid, text):
        asked.append((sid, text))
        return _resp(target, target, kind="answer")

    monkeypatch.setattr(run_eval, "_answer", fake_answer)
    row = run_eval.run_case(
        {
            "id": 3,
            "text": "t",
            "region_input": None,
            "lang": "ko",
            "followups": ["명륜진사갈비"],
            "answer": {"type": "named_place", "lnglat": [target.lng, target.lat]},
        }
    )
    assert asked == [("s1", "명륜진사갈비")] and row["questions_asked"] == 1
    assert (row["first_decision"], row["decision"], row["error_m"], row["hit30_at1"]) == (
        "question",
        "answer",
        0,
        True,
    )


def test_summarize_medians_hits_and_rates():
    rows = [
        {
            "error_m": 0,
            "hit30_at1": True,
            "hit30_at3": True,
            "type_match": True,
            "b0_error_m": 172,
            "invented": 1,
            "flipped": 1,
            "first_present": 2,
            "floors_checked": 1,
            "floors_wrong": 1,
            "no_region": {"error_m": 40, "ended_with_question": False},
        },
        {
            "error_m": 60,
            "hit30_at1": False,
            "hit30_at3": True,
            "type_match": False,
            "b0_error_m": 30,
            "invented": 0,
            "flipped": 0,
            "first_present": 2,
            "floors_checked": 1,
            "floors_wrong": 0,
            "no_region": {"error_m": None, "ended_with_question": True},
        },
    ]
    s, ko = run_eval.summarize(rows)
    assert (s["cases"], s["system_error_median_m"], s["b0_error_median_m"]) == (2, 30, 101)
    assert (s["hit30_at1"], s["hit30_at3"], s["type_match"]) == ("1/2", "2/2", "1/2")
    assert (s["vision_hallucination_rate"], s["floors_error_rate"], s["invented_total"]) == (0.25, 0.5, 1)
    assert (
        s["region_with_error_median_m"],
        s["region_without_error_median_m"],
        s["region_without_questions"],
    ) == (0, 40, 1)
    assert ko["사례 수"] == 2
