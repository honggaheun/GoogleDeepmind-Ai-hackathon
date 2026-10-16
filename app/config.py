"""Runtime configuration, secret masking, and Gemini backend selection."""

from __future__ import annotations

import os
from pathlib import Path

import google.auth

from app.contracts import LatLng

try:
    import winreg
except ImportError:  # pragma: no cover
    winreg = None

ROOT = Path(__file__).resolve().parent.parent
SECRET_NAMES = ("KAKAO_REST_API_KEY", "MAPS_API_KEY", "DATA_GO_KR_KEY", "GEMINI_API_KEY")
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

VENUE = LatLng(37.50631, 127.063695)

MODEL_MAIN = "gemini-3.8-flash"
MODEL_LIGHT = "gemini-3.5-flash-lite"
MODEL_EMBED = "gemini-embedding-2"
MODEL_TTS = "gemini-3.8-flash-tts"
MODEL_ARGON = "gemini-4-argon"

CACHE_DIR = ROOT / "cache"
LOG_DIR = ROOT / "logs"
OSM_DB = Path(os.environ.get("EODI_OSM_DB") or (ROOT / "data" / "osm_kr" / "korea.sqlite"))

WINDOW_RADIUS_M = {
    "city": 20000,
    "district": 3000,
    "neighborhood": 1000,
    "station": 500,
    "place": 300,
    "current_location": 500,
}

_MODEL_LABELS = {
    MODEL_ARGON: "Gemini 4 Argon",
    MODEL_MAIN: "3.8 Flash",
    MODEL_LIGHT: "Flash-Lite",
    MODEL_EMBED: "Embedding 2",
    MODEL_TTS: "3.8 Flash TTS",
    "gemini-3.8-live": "3.8 Live",
}


def get(name: str, default: str | None = None) -> str | None:
    """Read a setting from process environment first, falling back to HKCU Environment."""
    val = os.environ.get(name)
    if val is not None and val.strip():
        return val.strip()
    if winreg is not None:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                reg_val, _ = winreg.QueryValueEx(key, name)
                if isinstance(reg_val, str) and reg_val.strip():
                    return reg_val.strip()
        except OSError:
            pass
    return default


MODEL_PARSE = get("EODI_MODEL_PARSE") or MODEL_ARGON
MODEL_JUDGE = get("EODI_MODEL_JUDGE") or MODEL_ARGON


def mask(text: str) -> str:
    """Replace any configured secret key values in text with '***'."""
    if not text:
        return text
    out = str(text)
    for name in SECRET_NAMES:
        secret = get(name)
        if secret:
            out = out.replace(secret, "***")
    return out


def label(model: str) -> str:
    """Return a short human-readable display name for a Gemini model ID."""
    return _MODEL_LABELS.get(model, model)


def on_cloud_run() -> bool:
    """Return True when running inside a Cloud Run service container."""
    return bool(os.environ.get("K_SERVICE"))


def gemini_backend() -> str:
    """Return 'vertex', 'aistudio', or '' according to available credentials."""
    mode = (os.environ.get("EODI_GEMINI_BACKEND") or get("EODI_GEMINI_BACKEND") or "auto").strip().lower()
    has_vertex = bool(get("EODI_VERTEX_PROJECT") and (get("EODI_VERTEX_ADC") or on_cloud_run()))
    has_key = bool(get("GEMINI_API_KEY"))
    if mode == "vertex":
        return "vertex" if has_vertex else ""
    if mode == "aistudio":
        return "aistudio" if has_key else ""
    if has_vertex:
        return "vertex"
    if has_key:
        return "aistudio"
    return ""


def vertex_credentials(project: str | None = None, auth_mod=None):
    """Load Vertex credentials from ADC file or Cloud Run service account."""
    auth = auth_mod or google.auth
    proj = project or get("EODI_VERTEX_PROJECT")
    adc = get("EODI_VERTEX_ADC")
    if adc:
        creds, _ = auth.load_credentials_from_file(adc, scopes=SCOPES, quota_project_id=proj)
        return creds
    if on_cloud_run():
        creds, _ = auth.default(scopes=SCOPES, quota_project_id=proj)
        return creds
    raise RuntimeError("Vertex credentials require EODI_VERTEX_ADC or Cloud Run environment")
