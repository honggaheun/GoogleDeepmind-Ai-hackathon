"""Kakao Local REST API client for keyword, category, address, and coord2address lookups."""

from __future__ import annotations

import httpx

from app import config
from app.contracts import Jibun, LatLng, Place

BASE = "https://dapi.kakao.com/v2/local"


class KakaoError(Exception):
    """Raised when Kakao REST API requests fail or return client/server error codes."""


def _get(path: str, params: dict) -> dict:
    """category = 카테고리 그룹 코드로 거르기(예: SW8 지하철역), page = 1~45(쪽당 size개)."""
    key = config.get("KAKAO_REST_API_KEY")
    if not key:
        raise KakaoError("KAKAO_REST_API_KEY is not configured")
    url = f"{BASE}{path}"
    headers = {"Authorization": f"KakaoAK {key}"}
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=10.0)
    except Exception as exc:
        raise KakaoError(f"Kakao request error: {exc}") from exc
    if resp.status_code >= 400:
        raise KakaoError(f"Kakao HTTP {resp.status_code}: {resp.text}")
    return resp.json()


def parse_places(body: dict) -> list[Place]:
    """Convert Kakao local search documents into Place objects."""
    places: list[Place] = []
    for doc in body.get("documents") or []:
        dist_raw = doc.get("distance")
        dist = float(dist_raw) if dist_raw not in (None, "") else None
        name = doc.get("place_name") or doc.get("address_name") or ""
        places.append(
            Place(
                name=name,
                at=LatLng(lat=float(doc.get("y", 0.0)), lng=float(doc.get("x", 0.0))),
                category=doc.get("category_group_code", ""),
                address=doc.get("address_name", ""),
                distance_m=dist,
                category_name=doc.get("category_name", ""),
            )
        )
    return places


def keyword(
    query: str,
    near: LatLng | None = None,
    radius_m: int = 1000,
    size: int = 15,
    category: str | None = None,
    page: int = 1,
) -> list[Place]:
    """Search places matching keyword query with optional location and category filters."""
    params: dict = {"query": query, "size": size, "page": page}
    if near is not None:
        params["x"] = str(near.lng)
        params["y"] = str(near.lat)
        params["radius"] = radius_m
    if category:
        params["category_group_code"] = category
    data = _get("/search/keyword.json", params)
    return parse_places(data)


def category(
    code: str,
    near: LatLng,
    radius_m: int = 1000,
    size: int = 15,
    page: int = 1,
) -> list[Place]:
    """Search places of a specific category group around a location ordered by distance."""
    params: dict = {
        "category_group_code": code,
        "x": str(near.lng),
        "y": str(near.lat),
        "radius": radius_m,
        "size": size,
        "page": page,
        "sort": "distance",
    }
    data = _get("/search/category.json", params)
    return parse_places(data)


def address(query: str) -> list[Place]:
    """Search parcel and road-name addresses and return matching locations."""
    params = {"query": query}
    data = _get("/search/address.json", params)
    return parse_places(data)


def jibun(at: LatLng) -> Jibun | None:
    """Convert coordinates into legal dong administrative code and parcel numbers."""
    params = {"x": str(at.lng), "y": str(at.lat)}
    data = _get("/geo/coord2address.json", params)
    docs = data.get("documents") or []
    if not docs:
        return None
    addr = docs[0].get("address")
    if not addr:
        return None
    return Jibun(
        b_code=addr.get("b_code", ""),
        main_no=addr.get("main_address_no", ""),
        sub_no=addr.get("sub_address_no") or "0",
        mountain=addr.get("mountain_yn") == "Y",
        address=addr.get("address_name", ""),
    )
