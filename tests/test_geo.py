"""Unit tests for geo functions."""

from __future__ import annotations

import pytest

from app import config, geo


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def test_side_left_when_branch_is_north_while_heading_east():
    assert geo.side_of((1, 0), (0, 1)) == "left"


def test_side_right_when_branch_is_south_while_heading_east():
    assert geo.side_of((1, 0), (0, -1)) == "right"


def test_turn_angle_left_is_positive():
    assert geo.turn_angle_deg((1, 0), (1, 1)) == pytest.approx(45.0, abs=0.1)


def test_turn_angle_right_is_negative():
    assert geo.turn_angle_deg((1, 0), (1, -1)) == pytest.approx(-45.0, abs=0.1)


def test_bearing_north():
    north = geo.to_ll(0, 100, config.VENUE)
    assert _angle_diff(geo.bearing_deg(config.VENUE, north), 0) <= 1


def test_bearing_east():
    east = geo.to_ll(100, 0, config.VENUE)
    assert _angle_diff(geo.bearing_deg(config.VENUE, east), 90) <= 1


def test_ray_bearing_center_box_keeps_heading():
    assert geo.ray_bearing(90, 90, 400, 600) == pytest.approx(90.0, abs=0.1)


def test_ray_bearing_left_box_turns_left():
    assert geo.ray_bearing(90, 90, 0, 200) == pytest.approx(51.34, abs=0.1)


def test_to_xy_inverts_to_ll():
    x, y = geo.to_xy(geo.to_ll(123.4, -56.7, config.VENUE), config.VENUE)
    assert (x, y) == (pytest.approx(123.4, abs=1e-06), pytest.approx(-56.7, abs=1e-06))


def test_dist_m_agrees_with_local_offset():
    assert geo.dist_m(config.VENUE, geo.to_ll(60, 80, config.VENUE)) == pytest.approx(100, abs=0.5)
