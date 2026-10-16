"""Unit tests for mapdata spatial indexing and queries."""

from __future__ import annotations

import time

import pytest

from app import config, geo, mapdata
from app.contracts import Building, LatLng

VENUE_WAY = 475633202


def _ll(x: float, y: float) -> LatLng:
    return geo.to_ll(x, y, config.VENUE)


def _rect(way_id: int, x0: float, x1: float, y0: float, y1: float) -> Building:
    ring = [_ll(x0, y0), _ll(x1, y0), _ll(x1, y1), _ll(x0, y1), _ll(x0, y0)]
    centroid = _ll((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return Building(way_id=way_id, ring=ring, centroid=centroid)


@pytest.fixture(scope="module")
def venue():
    t0 = time.monotonic()
    md = mapdata.load_area(config.VENUE, 1000)
    sec = time.monotonic() - t0
    return md, sec


@pytest.fixture
def toy():
    nodes = {
        1: _ll(0, 0),
        2: _ll(50, 0),
        3: _ll(100, 0),
        4: _ll(100, 50),
    }
    edges = [
        (1, 2, "tertiary", 1),
        (2, 3, "tertiary", 1),
        (3, 4, "residential", 2),
    ]
    buildings = [
        _rect(10, 20, 40, 5, 20),
        _rect(11, 20, 40, -20, -5),
        _rect(12, 20, 40, 40, 55),
    ]
    return mapdata.MapData.from_parts(config.VENUE, nodes, edges, buildings)


def test_load_area_counts_match_index_check(venue):
    md, _ = venue
    assert 850 <= len(md.roads) <= 860
    assert 2360 <= len(md.buildings) <= 2370
    assert len(md.entrances) == 29


def test_venue_building_attributes(venue):
    md, _ = venue
    b = md.buildings[VENUE_WAY]
    assert (b.name, b.levels, b.housenumber) == ("오토웨이타워", 9, "417")


def test_first_hit_from_10m_west_of_venue_building(venue):
    md, _ = venue
    west_x, west_y = min((geo.to_xy(p, md.origin) for p in md.buildings[VENUE_WAY].ring))
    hit = mapdata.first_hit(md, geo.to_ll(west_x - 10, west_y, md.origin), 90)
    assert hit is not None
    building, dist = hit
    assert building.way_id == VENUE_WAY
    assert dist == pytest.approx(10, abs=3)


def test_load_area_under_one_second(venue):
    _, seconds = venue
    assert seconds < 1.0


def test_entrances_with_ref_finds_four_exit_3_near_venue(venue):
    md, _ = venue
    exits = mapdata.entrances_with_ref(md, "3")
    assert len(exits) == 4 and {e["ref"] for e in exits} == {"3"}
    assert 3404873908 in {e["node_id"] for e in exits}


def test_entrances_with_ref_accepts_beon_suffix(venue):
    md, _ = venue
    assert len(mapdata.entrances_with_ref(md, "3번")) == 4


def test_trees_near_finds_named_protected_tree(venue):
    md, _ = venue
    trees = mapdata.trees_near(md, LatLng(37.5005521, 127.063058), 30)
    assert [(t["node_id"], t["name"]) for t in trees] == [(5687393790, "대치동 보호수")]


def test_from_parts_edge_attributes(toy):
    data = toy.graph.edges[1, 2]
    assert (data["highway"], data["way_id"]) == ("tertiary", 1)
    assert data["length_m"] == pytest.approx(50, abs=0.5)


def test_from_parts_keeps_road_node_order(toy):
    assert toy.roads[1]["node_ids"] == [1, 2, 3]


def test_nearest_node_within_limit(toy):
    assert mapdata.nearest_node(toy, _ll(97, 4)) == 3


def test_nearest_node_none_when_too_far(toy):
    assert mapdata.nearest_node(toy, _ll(300, 300)) is None


def test_buildings_on_side_left_is_north_when_heading_east(toy):
    assert [b.way_id for b in mapdata.buildings_on_side(toy, _ll(0, 0), _ll(100, 0), "left")] == [10]


def test_buildings_on_side_right_is_south_when_heading_east(toy):
    assert [b.way_id for b in mapdata.buildings_on_side(toy, _ll(0, 0), _ll(100, 0), "right")] == [11]


def test_buildings_at_point_inside(toy):
    assert [b.way_id for b in mapdata.buildings_at(toy, _ll(30, 10))] == [10]


def test_buildings_near_finds_building_just_outside(toy):
    assert [b.way_id for b in mapdata.buildings_near(toy, _ll(30, 25), 10)] == [10]


def test_buildings_near_respects_radius(toy):
    assert mapdata.buildings_near(toy, _ll(30, 25), 3) == []


def test_first_hit_returns_nearest_of_two(toy):
    hit = mapdata.first_hit(toy, _ll(30, -40), 0)
    assert hit is not None
    building, dist = hit
    assert building.way_id == 11 and dist == pytest.approx(20, abs=0.5)


def test_first_hit_none_beyond_max(toy):
    assert mapdata.first_hit(toy, _ll(30, -40), 0, max_m=10) is None
