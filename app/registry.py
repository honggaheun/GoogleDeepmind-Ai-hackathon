"""Query the Korean Building Registry title ledger API and cache records by parcel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import re
import threading
from urllib.parse import unquote

import httpx

from app import config, kakao
from app.contracts import Jibun, LatLng, RegistryRecord

URL = "https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo"
CACHE_PATH = config.CACHE_DIR / "registry.json"
TIMEOUT_S = 5.0
_CACHE_LOCK = threading.Lock()


class RegistryError(Exception):
    """Raised when the building registry API key is missing or the upstream request fails."""


def params(j: Jibun) -> dict:
    """Build parcel query parameters with 4-digit zero-padded lot numbers."""
    b_code = str(j.b_code or "")
    return {
        "sigunguCd": b_code[:5],
        "bjdongCd": b_code[5:10],
        "platGbCd": "1" if j.mountain else "0",
        "bun": str(j.main_no or "0").zfill(4),
        "ji": str(j.sub_no or "0").zfill(4),
    }


def pick(body: dict) -> dict | None:
    """Select the main building row from a registry API response body."""
    if not isinstance(body, dict):
        return None
    resp = body.get("response") if "response" in body else body
    if not isinstance(resp, dict):
        return None
    inner = resp.get("body") if "body" in resp else resp
    if not isinstance(inner, dict):
        return None
    items = inner.get("items")
    if not isinstance(items, dict):
        return None
    item = items.get("item")
    if isinstance(item, dict):
        rows = [item]
    elif isinstance(item, list):
        rows = [r for r in item if isinstance(r, dict)]
    else:
        return None
    if not rows:
        return None
    for row in rows:
        if str(row.get("mainAtchGbCdNm") or "").strip() == "주건축물":
            return row
    return rows[0]


def _num(val, typ=int):
    """Convert a registry field to int or float, returning None on blank or invalid input."""
    if val is None or str(val).strip() == "":
        return None
    try:
        return int(float(val)) if typ is int else float(val)
    except (ValueError, TypeError):
        return None


def to_record(row: dict) -> RegistryRecord:
    """Convert a raw title ledger row dict into a normalized RegistryRecord."""
    name = str(row.get("bldNm") or "").strip()
    main_purpose = str(row.get("mainPurpsCdNm") or "").strip() or None
    etc_purpose = str(row.get("etcPurps") or "").strip() or None
    apr_raw = str(row.get("useAprDay") or "").strip()
    use_apr_day = apr_raw if apr_raw and apr_raw != "0" else None
    return RegistryRecord(
        grnd_floors=_num(row.get("grndFlrCnt"), int),
        ugrnd_floors=_num(row.get("ugrndFlrCnt"), int),
        height_m=_num(row.get("heit"), float),
        main_purpose=main_purpose,
        etc_purpose=etc_purpose,
        name=name,
        use_apr_day=use_apr_day,
    )


def photo_predates_building(capture: str | None, use_apr_day: str | None) -> bool:
    """Return True if the photo capture month is earlier than the building approval month."""
    if not capture or not use_apr_day:
        return False
    cap = str(capture).strip()
    apr = re.sub(r"\D+", "", str(use_apr_day))
    if len(apr) < 4:
        return False
    try:
        if "-" in cap:
            parts = cap.split("-")
            cap_ym = (int(parts[0]), int(parts[1]))
        else:
            cap_ym = (int(cap[:4]), 1)
        apr_y = int(apr[:4])
        apr_m = int(apr[4:6]) if len(apr) >= 6 and int(apr[4:6]) >= 1 else 1
        return cap_ym < (apr_y, apr_m)
    except (ValueError, IndexError):
        return False


def _load_cache() -> dict:
    """Load cached registry records from disk if present."""
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    """Persist registry cache dictionary to disk."""
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def title(j: Jibun) -> RegistryRecord | None:
    """Fetch the primary building title ledger record for a parcel, using disk cache."""
    if not j or not j.b_code or len(j.b_code) < 10:
        return None
    p = params(j)
    cache_key = f"{p['sigunguCd']}-{p['bjdongCd']}-{p['platGbCd']}-{p['bun']}-{p['ji']}"

    with _CACHE_LOCK:
        cache = _load_cache()
        if cache_key in cache:
            cached = cache[cache_key]
            return RegistryRecord(**cached) if isinstance(cached, dict) else None

    key = config.get("DATA_GO_KR_KEY")
    if not key:
        raise RegistryError("Missing DATA_GO_KR_KEY")

    query = {
        **p,
        "serviceKey": unquote(key),
        "_type": "json",
        "numOfRows": 10,
    }
    try:
        resp = httpx.get(URL, params=query, timeout=TIMEOUT_S)
        resp.raise_for_status()
        body = resp.json()
    except Exception as exc:
        raise RegistryError(config.mask(str(exc))) from exc

    if not isinstance(body, dict):
        raise RegistryError("Unexpected registry response format")
    header = ((body.get("response") or {}).get("header") or {})
    if isinstance(header, dict):
        code = str(header.get("resultCode") or "00").strip()
        if code and code != "00":
            raise RegistryError(str(header.get("resultMsg") or f"resultCode={code}"))

    row = pick(body)
    rec = to_record(row) if row is not None else None

    with _CACHE_LOCK:
        cache = _load_cache()
        cache[cache_key] = asdict(rec) if rec is not None else None
        _save_cache(cache)

    return rec


def for_point(at: LatLng) -> RegistryRecord | None:
    """Resolve a coordinate to a parcel via Kakao reverse geocoding and fetch its title ledger."""
    j = kakao.jibun(at)
    if j is None:
        return None
    if not j.b_code and j.address and hasattr(kakao, "_get"):
        try:
            docs = (kakao._get("/search/address.json", {"query": j.address}) or {}).get("documents") or []
            b_code = ((docs[0].get("address") or {}).get("b_code") if docs else "") or ""
            if b_code:
                j = Jibun(
                    b_code=b_code,
                    main_no=j.main_no,
                    sub_no=j.sub_no,
                    mountain=j.mountain,
                    address=j.address,
                )
        except Exception:
            pass
    return title(j)


def _main() -> None:
    """CLI entrypoint for querying building registry by latitude and longitude."""
    parser = argparse.ArgumentParser(description="Fetch building registry title info for a coordinate.")
    parser.add_argument("lat", type=float, help="Latitude")
    parser.add_argument("lng", type=float, help="Longitude")
    args = parser.parse_args()
    rec = for_point(LatLng(lat=args.lat, lng=args.lng))
    print(json.dumps(asdict(rec) if rec else None, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
