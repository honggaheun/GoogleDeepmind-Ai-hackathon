"""Local plane coordinates, spherical geodesics, turning angles, and field-of-view ray projection."""

from __future__ import annotations

import math

from app.contracts import LatLng


def to_xy(p: LatLng, origin: LatLng) -> tuple[float, float]:
    """Convert latitude and longitude coordinates to meters on a tangent plane relative to origin."""
    lat0_rad = math.radians(origin.lat)
    x = (p.lng - origin.lng) * math.cos(lat0_rad) * 111320.0
    y = (p.lat - origin.lat) * 110540.0
    return (x, y)


def to_ll(x: float, y: float, origin: LatLng) -> LatLng:
    """Convert local plane meter coordinates back to latitude and longitude coordinates."""
    lat = origin.lat + (y / 110540.0)
    lat0_rad = math.radians(origin.lat)
    cos_lat0 = math.cos(lat0_rad)
    lng = origin.lng + (x / (cos_lat0 * 111320.0)) if cos_lat0 != 0 else origin.lng
    return LatLng(lat=lat, lng=lng)


def dist_m(a: LatLng, b: LatLng) -> float:
    """Compute great-circle distance in meters between two coordinates using haversine formula."""
    r = 6371000.0
    phi1 = math.radians(a.lat)
    phi2 = math.radians(b.lat)
    delta_phi = math.radians(b.lat - a.lat)
    delta_lambda = math.radians(b.lng - a.lng)
    h = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1.0 - h))
    return r * c


def bearing_deg(a: LatLng, b: LatLng) -> float:
    """나침반 방위(북 0, 동 90, 시계 방향), 0~360."""
    phi1 = math.radians(a.lat)
    phi2 = math.radians(b.lat)
    delta_lambda = math.radians(b.lng - a.lng)
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    theta = math.atan2(y, x)
    return (math.degrees(theta) + 360.0) % 360.0


def _cross(h: tuple[float, float], b: tuple[float, float]) -> float:
    """진행 방향 h 기준으로 b가 어느 쪽인지. z > 0 왼쪽, z < 0 오른쪽."""
    return h[0] * b[1] - h[1] * b[0]


def side_of(h: tuple[float, float], b: tuple[float, float]) -> str:
    """Determine whether vector b lies to the left or right of heading vector h."""
    z = _cross(h, b)
    return "left" if z > 0 else "right"


def turn_angle_deg(h: tuple[float, float], b: tuple[float, float]) -> float:
    """h에서 b로 꺾는 각도(도). 왼쪽이 양수, -180~180."""
    angle_h = math.atan2(h[1], h[0])
    angle_b = math.atan2(b[1], b[0])
    diff = angle_b - angle_h
    diff = (diff + math.pi) % (2.0 * math.pi) - math.pi
    return math.degrees(diff)


def ray_bearing(heading_deg: float, fov_deg: float, xmin: float, xmax: float) -> float:
    """사진 속 박스(0~1000 정규화 가로 범위)의 중심 방위. 사진 중심 = heading."""
    x_mid = (xmin + xmax) / 2.0
    f = 500.0 / math.tan(math.radians(fov_deg / 2.0))
    offset_deg = math.degrees(math.atan2(x_mid - 500.0, f))
    return (heading_deg + offset_deg + 360.0) % 360.0
