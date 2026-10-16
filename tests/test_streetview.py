"""Unit tests for Street View metadata and image fetching."""

from __future__ import annotations

import pytest

from app import config, streetview
from app.contracts import LatLng


class _Resp:
    """Minimal HTTP response stub for Street View tests."""

    def __init__(
        self,
        body: dict,
        status: int = 200,
        ctype: str = "application/json",
        content: bytes = b"\xff\xd8\xff",
    ) -> None:
        self._body = body
        self.status_code = status
        self.headers = {"Content-Type": ctype}
        self.content = content

    def json(self) -> dict:
        return self._body


@pytest.fixture
def fake_get(monkeypatch):
    """Monkeypatch streetview._key and streetview.requests.get with a fixed response."""
    monkeypatch.setattr(streetview, "_key", lambda: "test-key")

    def _set(resp: _Resp):
        monkeypatch.setattr(streetview.requests, "get", lambda *a, **k: resp)

    return _set


def test_metadata_ok_returns_capture_point_and_date(fake_get):
    fake_get(
        _Resp(
            {
                "status": "OK",
                "date": "2016-06",
                "location": {"lat": 37.5064, "lng": 127.0632},
                "pano_id": "x",
            }
        )
    )
    pano = streetview.metadata(config.VENUE)
    assert (pano.at, pano.date) == (LatLng(37.5064, 127.0632), "2016-06")
    assert not hasattr(pano, "pano_id")


def test_metadata_zero_results_is_none(fake_get):
    fake_get(_Resp({"status": "ZERO_RESULTS"}))
    assert streetview.metadata(config.VENUE) is None


def test_metadata_denied_is_error_not_none(fake_get):
    fake_get(_Resp({"status": "REQUEST_DENIED", "error_message": "key invalid"}))
    with pytest.raises(streetview.StreetViewError):
        streetview.metadata(config.VENUE)


def test_image_missing_is_none(fake_get):
    fake_get(_Resp({}, status=404, ctype="text/plain"))
    assert streetview.image(config.VENUE, 90) is None
