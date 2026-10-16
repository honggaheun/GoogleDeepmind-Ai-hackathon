"""Fetch Google Street View panorama metadata and 640x640 images in memory only."""

from __future__ import annotations

import argparse
import json

import requests

from app import config
from app.contracts import LatLng, Pano

BASE = "https://maps.googleapis.com/maps/api/streetview"


class StreetViewError(RuntimeError):
    """Raised when Street View API rejects the request or returns an unexpected error."""


def _key() -> str:
    """Return the configured Google Maps API key or raise StreetViewError."""
    key = config.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise StreetViewError("GOOGLE_MAPS_API_KEY is not set")
    return key


def metadata(at: LatLng, radius_m: int = 50) -> Pano | None:
    """Query Street View metadata near a point and return Pano(at, date) or None."""
    params = {
        "location": f"{at.lat},{at.lng}",
        "radius": radius_m,
        "source": "outdoor",
        "key": _key(),
    }
    try:
        resp = requests.get(f"{BASE}/metadata", params=params, timeout=10)
    except Exception as exc:
        raise StreetViewError(f"Street View metadata request failed: {exc}") from exc
    if resp.status_code != 200:
        raise StreetViewError(f"Street View metadata HTTP {resp.status_code}")
    body = resp.json()
    status = body.get("status")
    if status in ("ZERO_RESULTS", "NOT_FOUND"):
        return None
    if status != "OK":
        msg = body.get("error_message") or status or "unknown error"
        raise StreetViewError(f"Street View metadata {status}: {msg}")
    loc = body.get("location") or {}
    # Street View panorama IDs are not stored (terms)
    return Pano(
        at=LatLng(lat=float(loc["lat"]), lng=float(loc["lng"])),
        date=str(body.get("date") or ""),
    )


def image(
    at: LatLng,
    heading_deg: float,
    fov_deg: float = 90,
    pitch: float = 10,
) -> bytes | None:
    """Fetch a 640x640 JPEG in memory only; return None when no panorama exists (404)."""
    params = {
        "size": "640x640",
        "location": f"{at.lat},{at.lng}",
        "heading": round(float(heading_deg), 1),
        "fov": round(float(fov_deg), 1),
        "pitch": round(float(pitch), 1),
        "source": "outdoor",
        "return_error_code": "true",
        "key": _key(),
    }
    try:
        resp = requests.get(BASE, params=params, timeout=15)
    except Exception as exc:
        raise StreetViewError(f"Street View image request failed: {exc}") from exc
    if resp.status_code == 404:
        return None
    if resp.status_code != 200:
        raise StreetViewError(f"Street View image HTTP {resp.status_code}")
    ctype = (resp.headers.get("Content-Type") or resp.headers.get("content-type") or "").lower()
    if ctype and "image" not in ctype:
        return None
    return resp.content


def _main() -> None:
    """CLI entrypoint for inspecting Street View metadata at a coordinate."""
    parser = argparse.ArgumentParser(description="Check Street View metadata at a coordinate.")
    parser.add_argument("--lat", type=float, default=config.VENUE.lat, help="Latitude")
    parser.add_argument("--lng", type=float, default=config.VENUE.lng, help="Longitude")
    parser.add_argument("--radius", type=int, default=50, help="Search radius in meters")
    args = parser.parse_args()
    pano = metadata(LatLng(args.lat, args.lng), radius_m=args.radius)
    if pano is None:
        print(json.dumps({"status": "ZERO_RESULTS"}, ensure_ascii=False))
    else:
        print(
            json.dumps(
                {"lat": pano.at.lat, "lng": pano.at.lng, "date": pano.date},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    _main()
