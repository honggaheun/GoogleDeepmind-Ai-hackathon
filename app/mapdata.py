"""Load and index OpenStreetMap roads, buildings, and point landmarks from SQLite into a spatial graph."""

from __future__ import annotations

import array
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import sqlite3

import networkx as nx
import numpy as np
import shapely
import shapely.geometry
from shapely.strtree import STRtree

from app import config, geo
from app.contracts import Building, LatLng


@dataclass
class MapEdge:
    u: int
    v: int
    edge_type: str
    way_id: int


@dataclass
class MapData:
    origin: LatLng
    graph: nx.Graph
    roads: dict
    buildings: dict[int, Building]
    building_polys: dict
    building_tree: STRtree
    entrances: list[dict]
    trees: list[dict]
    _node_ids: np.ndarray
    _node_xy: np.ndarray
    _poly_way_ids: list[int] = field(default_factory=list)
    nodes: dict[int, LatLng] = field(default_factory=dict)
    edges: list[MapEdge] = field(default_factory=list)

    def neighbors(self, node: int) -> list[int]:
        if self.graph and node in self.graph:
            return list(self.graph.neighbors(node))
        return []

    @classmethod
    def from_parts(
        cls,
        origin: LatLng,
        nodes: dict[int, LatLng],
        edges: list[tuple[int, int, str, int]],
        buildings: list[Building],
    ) -> MapData:
        """Construct an in-memory MapData instance from raw node, edge, and building parts."""
        g = nx.Graph()
        for nid, p in nodes.items():
            x, y = geo.to_xy(p, origin)
            g.add_node(nid, lat=p.lat, lng=p.lng, x=x, y=y)

        roads_dict: dict[int, dict] = {}
        for a, b, highway, way_id in edges:
            length_m = geo.dist_m(nodes[a], nodes[b])
            g.add_edge(a, b, highway=highway, way_id=way_id, length_m=length_m)
            if way_id not in roads_dict:
                roads_dict[way_id] = {"way_id": way_id, "highway": highway, "node_ids": [a, b]}
            else:
                nlist = roads_dict[way_id]["node_ids"]
                if nlist[-1] == a:
                    nlist.append(b)
                elif nlist[0] == b:
                    nlist.insert(0, a)
                elif b not in nlist:
                    nlist.append(b)

        b_dict: dict[int, Building] = {b.way_id: b for b in buildings}
        b_polys: dict[int, shapely.geometry.Polygon] = {}
        poly_list: list[shapely.geometry.Polygon] = []
        poly_way_ids: list[int] = []

        for b in buildings:
            pts = [geo.to_xy(p, origin) for p in b.ring]
            if len(pts) >= 3:
                if pts[0] != pts[-1]:
                    pts.append(pts[0])
                poly = shapely.geometry.Polygon(pts)
                b_polys[b.way_id] = poly
                poly_list.append(poly)
                poly_way_ids.append(b.way_id)

        tree = STRtree(poly_list)
        node_ids = np.array(list(g.nodes()), dtype=np.int64)
        node_xy = (
            np.array([[g.nodes[nid]["x"], g.nodes[nid]["y"]] for nid in node_ids], dtype=np.float64)
            if len(node_ids) > 0
            else np.empty((0, 2), dtype=np.float64)
        )

        edge_objs = [MapEdge(u=a, v=b, edge_type=highway, way_id=way_id) for a, b, highway, way_id in edges]

        return cls(
            origin=origin,
            graph=g,
            roads=roads_dict,
            buildings=b_dict,
            building_polys=b_polys,
            building_tree=tree,
            entrances=[],
            trees=[],
            _node_ids=node_ids,
            _node_xy=node_xy,
            _poly_way_ids=poly_way_ids,
            nodes=nodes,
            edges=edge_objs,
        )


def _parse_levels(levels_raw: str | None) -> int | None:
    """Extract the first integer floor number from a levels tag string."""
    if not levels_raw:
        return None
    m = re.search(r"\d+", str(levels_raw))
    return int(m.group(0)) if m else None


def load_area(center: LatLng, radius_m: float, db_path=None) -> MapData:
    """Query OSM spatial index within a radius and construct graph, buildings, and point landmarks."""
    path = db_path or config.OSM_DB
    lat0 = center.lat
    dlat = radius_m / 110540.0
    dlng = radius_m / (111320.0 * math.cos(math.radians(lat0)))
    min_lat, max_lat = lat0 - dlat, lat0 + dlat
    min_lng, max_lng = center.lng - dlng, center.lng + dlng

    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # 1. Roads
    road_rows = con.execute(
        """
        SELECT roads.way_id, roads.highway, roads.name, roads.node_ids, roads.coords
        FROM roads
        JOIN roads_rtree ON roads.way_id = roads_rtree.id
        WHERE roads_rtree.min_lng <= ? AND roads_rtree.max_lng >= ?
          AND roads_rtree.min_lat <= ? AND roads_rtree.max_lat >= ?
        """,
        (max_lng, min_lng, max_lat, min_lat),
    ).fetchall()

    g = nx.Graph()
    roads_dict: dict[int, dict] = {}
    nodes_dict: dict[int, LatLng] = {}
    edges_list: list[MapEdge] = []
    for r in road_rows:
        way_id = r["way_id"]
        highway = r["highway"]
        name = r["name"]
        n_arr = array.array("q")
        n_arr.frombytes(r["node_ids"])
        c_arr = array.array("d")
        c_arr.frombytes(r["coords"])

        node_list = list(n_arr)
        roads_dict[way_id] = {
            "way_id": way_id,
            "highway": highway,
            "name": name,
            "node_ids": node_list,
        }

        # Add nodes and edges
        num_pts = len(node_list)
        for i in range(num_pts):
            nid = node_list[i]
            lng = c_arr[2 * i]
            lat = c_arr[2 * i + 1]
            if nid not in g:
                p = LatLng(lat=lat, lng=lng)
                nodes_dict[nid] = p
                x, y = geo.to_xy(p, center)
                g.add_node(nid, lat=lat, lng=lng, x=x, y=y)
            if i > 0:
                prev_id = node_list[i - 1]
                prev_node = g.nodes[prev_id]
                curr_node = g.nodes[nid]
                length_m = geo.dist_m(
                    LatLng(lat=prev_node["lat"], lng=prev_node["lng"]),
                    LatLng(lat=curr_node["lat"], lng=curr_node["lng"]),
                )
                g.add_edge(prev_id, nid, highway=highway, way_id=way_id, length_m=length_m)
                edges_list.append(MapEdge(u=prev_id, v=nid, edge_type=highway, way_id=way_id))

    # 2. Buildings
    bldg_rows = con.execute(
        """
        SELECT buildings.way_id, buildings.building, buildings.name, buildings.levels,
               buildings.housenumber, buildings.coords
        FROM buildings
        JOIN buildings_rtree ON buildings.way_id = buildings_rtree.id
        WHERE buildings_rtree.min_lng <= ? AND buildings_rtree.max_lng >= ?
          AND buildings_rtree.min_lat <= ? AND buildings_rtree.max_lat >= ?
        """,
        (max_lng, min_lng, max_lat, min_lat),
    ).fetchall()

    b_dict: dict[int, Building] = {}
    b_polys: dict[int, shapely.geometry.Polygon] = {}
    poly_list: list[shapely.geometry.Polygon] = []
    poly_way_ids: list[int] = []

    for b_row in bldg_rows:
        way_id = b_row["way_id"]
        c_arr = array.array("d")
        c_arr.frombytes(b_row["coords"])
        ring = [
            LatLng(lat=c_arr[2 * i + 1], lng=c_arr[2 * i])
            for i in range(len(c_arr) // 2)
        ]
        if not ring:
            continue
        c_lat = sum(p.lat for p in ring) / len(ring)
        c_lng = sum(p.lng for p in ring) / len(ring)
        levels = _parse_levels(b_row["levels"])
        bldg = Building(
            way_id=way_id,
            name=b_row["name"],
            levels=levels,
            housenumber=b_row["housenumber"],
            ring=ring,
            centroid=LatLng(lat=c_lat, lng=c_lng),
        )
        b_dict[way_id] = bldg

        pts = [geo.to_xy(p, center) for p in ring]
        if len(pts) >= 3:
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            poly = shapely.geometry.Polygon(pts)
            b_polys[way_id] = poly
            poly_list.append(poly)
            poly_way_ids.append(way_id)

    # 3. Points
    pt_rows = con.execute(
        """
        SELECT points.node_id, points.kind, points.ref, points.name, points.species,
               points.lng, points.lat
        FROM points
        JOIN points_rtree ON points.node_id = points_rtree.id
        WHERE points_rtree.min_lng <= ? AND points_rtree.max_lng >= ?
          AND points_rtree.min_lat <= ? AND points_rtree.max_lat >= ?
        """,
        (max_lng, min_lng, max_lat, min_lat),
    ).fetchall()

    entrances: list[dict] = []
    trees: list[dict] = []
    for pr in pt_rows:
        kind = pr["kind"]
        at = LatLng(lat=pr["lat"], lng=pr["lng"])
        if kind == "subway_entrance":
            entrances.append(
                {"node_id": pr["node_id"], "ref": pr["ref"], "name": pr["name"], "at": at}
            )
        elif kind == "tree":
            trees.append(
                {
                    "node_id": pr["node_id"],
                    "name": pr["name"],
                    "species": pr["species"],
                    "at": at,
                }
            )

    con.close()

    tree = STRtree(poly_list)
    node_ids = np.array(list(g.nodes()), dtype=np.int64)
    node_xy = (
        np.array([[g.nodes[nid]["x"], g.nodes[nid]["y"]] for nid in node_ids], dtype=np.float64)
        if len(node_ids) > 0
        else np.empty((0, 2), dtype=np.float64)
    )

    return MapData(
        origin=center,
        graph=g,
        roads=roads_dict,
        buildings=b_dict,
        building_polys=b_polys,
        building_tree=tree,
        entrances=entrances,
        trees=trees,
        _node_ids=node_ids,
        _node_xy=node_xy,
        _poly_way_ids=poly_way_ids,
        nodes=nodes_dict,
        edges=edges_list,
    )


def nearest_node(md: MapData, p: LatLng, max_m: float = 60) -> int | None:
    """Find the nearest road graph node within a distance threshold."""
    if len(md._node_xy) == 0:
        return None
    px, py = geo.to_xy(p, md.origin)
    dists = np.hypot(md._node_xy[:, 0] - px, md._node_xy[:, 1] - py)
    min_idx = int(np.argmin(dists))
    if dists[min_idx] <= max_m:
        return int(md._node_ids[min_idx])
    return None


def buildings_at(md: MapData, p: LatLng) -> list[Building]:
    """Return all buildings containing the given coordinate."""
    px, py = geo.to_xy(p, md.origin)
    pt = shapely.geometry.Point(px, py)
    indices = md.building_tree.query(pt, predicate="intersects")
    res: list[Building] = []
    for idx in indices:
        way_id = md._poly_way_ids[idx]
        poly = md.building_polys[way_id]
        if poly.intersects(pt):
            res.append(md.buildings[way_id])
    return res


def buildings_near(md: MapData, p: LatLng, radius_m: float) -> list[Building]:
    """Return all buildings within radius_m ordered by distance from closest to farthest."""
    px, py = geo.to_xy(p, md.origin)
    pt = shapely.geometry.Point(px, py)
    buf = pt.buffer(radius_m)
    indices = md.building_tree.query(buf, predicate="intersects")
    matched: list[tuple[float, Building]] = []
    for idx in indices:
        way_id = md._poly_way_ids[idx]
        poly = md.building_polys[way_id]
        d = poly.distance(pt)
        if d <= radius_m:
            matched.append((d, md.buildings[way_id]))
    matched.sort(key=lambda item: item[0])
    return [b for _, b in matched]


def buildings_on_side(
    md: MapData, a: LatLng, b: LatLng, side: str, band_m: float = 25
) -> list[Building]:
    """선분 a→b의 말한 쪽 띠와 겹치는 건물(가까운 순), side 밖 값은 양쪽."""
    ax, ay = geo.to_xy(a, md.origin)
    bx, by = geo.to_xy(b, md.origin)
    vx, vy = bx - ax, by - ay
    len_v = math.hypot(vx, vy)
    if len_v == 0:
        return []
    ux, uy = vx / len_v, vy / len_v

    if side == "left":
        nx, ny = -uy * band_m, ux * band_m
        pts = [(ax, ay), (bx, by), (bx + nx, by + ny), (ax + nx, ay + ny), (ax, ay)]
    elif side == "right":
        nx, ny = uy * band_m, -ux * band_m
        pts = [(ax, ay), (bx, by), (bx + nx, by + ny), (ax + nx, ay + ny), (ax, ay)]
    else:
        lx, ly = -uy * band_m, ux * band_m
        rx, ry = uy * band_m, -ux * band_m
        pts = [
            (ax + rx, ay + ry),
            (bx + rx, by + ry),
            (bx + lx, by + ly),
            (ax + lx, ay + ly),
            (ax + rx, ay + ry),
        ]

    band_poly = shapely.geometry.Polygon(pts)
    indices = md.building_tree.query(band_poly, predicate="intersects")
    start_pt = shapely.geometry.Point(ax, ay)

    res: list[tuple[float, Building]] = []
    for idx in indices:
        way_id = md._poly_way_ids[idx]
        poly = md.building_polys[way_id]
        if poly.intersects(band_poly):
            d = poly.distance(start_pt)
            res.append((d, md.buildings[way_id]))

    res.sort(key=lambda item: item[0])
    return [bldg for _, bldg in res]


def first_hit(
    md: MapData, origin: LatLng, bearing_deg: float, max_m: float = 60
) -> tuple[Building, float] | None:
    """선 쏘기."""
    ox, oy = geo.to_xy(origin, md.origin)
    rad = math.radians(bearing_deg)
    dx = math.sin(rad) * max_m
    dy = math.cos(rad) * max_m
    ray = shapely.geometry.LineString([(ox, oy), (ox + dx, oy + dy)])
    start_pt = shapely.geometry.Point(ox, oy)

    indices = md.building_tree.query(ray, predicate="intersects")
    best_b: Building | None = None
    min_dist = float("inf")

    for idx in indices:
        way_id = md._poly_way_ids[idx]
        b = md.buildings[way_id]
        poly = md.building_polys[way_id]
        inter = poly.intersection(ray)
        if inter.is_empty:
            continue
        dist = 0.0 if poly.contains(start_pt) else start_pt.distance(inter)
        if dist < min_dist and dist <= max_m:
            min_dist = dist
            best_b = b

    if best_b is not None:
        return (best_b, min_dist)
    return None


def entrances_with_ref(md: MapData, ref: str) -> list[dict]:
    """Find subway entrances matching exit number reference."""
    clean = str(ref).rstrip("번")
    return [e for e in md.entrances if str(e.get("ref") or "").rstrip("번") == clean]


def trees_near(md: MapData, p: LatLng, radius_m: float) -> list[dict]:
    """Find trees within radius_m ordered by distance from closest to farthest."""
    matched: list[tuple[float, dict]] = []
    for t in md.trees:
        d = geo.dist_m(p, t["at"])
        if d <= radius_m:
            matched.append((d, t))
    matched.sort(key=lambda item: item[0])
    return [t for _, t in matched]


def way_geometry(way_id: int, db_path=None) -> dict | None:
    """Fetch GeoJSON geometry for a building polygon or road line string."""
    path = db_path or config.OSM_DB
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    row = con.execute("SELECT coords FROM buildings WHERE way_id = ?", (way_id,)).fetchone()
    if row:
        c_arr = array.array("d")
        c_arr.frombytes(row[0])
        pts = [[c_arr[2 * i], c_arr[2 * i + 1]] for i in range(len(c_arr) // 2)]
        con.close()
        return {"type": "Polygon", "coordinates": [pts]}

    row = con.execute("SELECT coords FROM roads WHERE way_id = ?", (way_id,)).fetchone()
    if row:
        c_arr = array.array("d")
        c_arr.frombytes(row[0])
        pts = [[c_arr[2 * i], c_arr[2 * i + 1]] for i in range(len(c_arr) // 2)]
        con.close()
        return {"type": "LineString", "coordinates": pts}

    con.close()
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MapData area query tool")
    parser.add_argument("--radius", type=float, default=1000.0, help="Search radius in meters")
    parser.parse_args()
