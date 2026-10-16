#!/bin/sh
set -e

EODI_OSM_DB="${EODI_OSM_DB:-/tmp/korea.sqlite}"
export EODI_OSM_DB

if [ ! -f "$EODI_OSM_DB" ] && [ -n "${EODI_OSM_GCS_MOUNT:-}" ]; then
    cp "$EODI_OSM_GCS_MOUNT/osm_kr/korea.sqlite" "$EODI_OSM_DB"
    ls -lh "$EODI_OSM_DB"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8080}" --proxy-headers --forwarded-allow-ips='*'
