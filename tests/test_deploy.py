"""Tests for Cloud Run deployment artifacts and README structure."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_requirements_pinned() -> None:
    """Verify requirements.txt contains all pinned runtime and test dependencies."""
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [
        "google-genai==2.29.0",
        "google-auth==2.61.0",
        "fastapi==0.143.0",
        "uvicorn[standard]==0.54.0",
        "python-multipart==0.0.32",
        "shapely==2.2.0",
        "networkx==3.7",
        "pillow==12.3.0",
        "requests==2.34.2",
        "pytest==9.1.1",
    ]
    assert lines == expected


def test_dockerfile_configuration() -> None:
    """Verify Dockerfile base image, environment, layer caching, and entrypoint command."""
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim" in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "PYTHONUTF8=1" in text
    assert "WORKDIR /srv" in text
    assert "COPY requirements.txt" in text
    assert "pip install --no-cache-dir" in text
    assert "COPY app ./app" in text
    assert "COPY eval ./eval" in text
    assert "COPY start.sh ./start.sh" in text
    assert 'CMD ["sh", "start.sh"]' in text


def test_start_sh_posix_and_lf() -> None:
    """Verify start.sh uses LF line endings, copies SQLite from GCS mount, and starts uvicorn."""
    raw = (ROOT / "start.sh").read_bytes()
    assert raw.count(b"\r") == 0
    assert raw.endswith(b"\n")
    text = raw.decode("utf-8")
    assert "set -e" in text
    assert "/tmp/korea.sqlite" in text
    assert 'cp "$EODI_OSM_GCS_MOUNT/osm_kr/korea.sqlite" "$EODI_OSM_DB"' in text
    assert 'exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}" --proxy-headers --forwarded-allow-ips=\'*\'' in text


def test_ignore_files_match() -> None:
    """Verify .dockerignore and .gcloudignore contain the same exclusion list."""
    docker_ignore = [
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gcloud_ignore = [
        line.strip()
        for line in (ROOT / ".gcloudignore").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {
        ".venv/",
        "data/",
        "cache/",
        "logs/",
        "tests/",
        "**/__pycache__/",
        "*.sqlite",
        "*.jsonl",
        ".git/",
        "deploy/",
    }
    assert docker_ignore == gcloud_ignore
    assert required.issubset(set(docker_ignore))


def test_deploy_ps1_bom_and_commands() -> None:
    """Verify deploy/deploy.ps1 starts with UTF-8 BOM and contains Cloud Run flags and smoke steps."""
    raw = (ROOT / "deploy" / "deploy.ps1").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert '$ErrorActionPreference = "Continue"' in text
    assert "asia-northeast3" in text
    assert "whereisit" in text
    assert "--execution-environment gen2" in text
    assert "--cpu 2 --memory 4Gi --cpu-boost --min-instances 1 --max-instances 1 --session-affinity" in text
    assert "--allow-unauthenticated" in text
    assert "--no-invoker-iam-check" in text
    assert "/api/warmup" in text
    assert "/api/locate_text" in text
    assert "/api/session/" in text


def test_readme_sections_and_models() -> None:
    """Verify README.md includes required headings, Gemini models, module list, and attribution."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "# 어디예요? (Eodiyeyo?) — spoken directions → a place on the map" in text
    for section in (
        "## Overview",
        "## How it works",
        "## Gemini models used",
        "## Project structure",
        "## Run locally",
        "## Deploy to Cloud Run",
        "## Data & attribution",
        "## Limitations",
    ):
        assert section in text
    for model in (
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.8-live",
        "gemini-3.8-flash-tts",
        "gemini-embedding-2",
    ):
        assert model in text
    assert "© OpenStreetMap contributors" in text
