# 어디예요? (Eodiyeyo?) — spoken directions → a place on the map

## Overview

**어디예요? (Eodiyeyo?)** is a web application that turns spoken, written, or photographed route descriptions into a single verified building, alley, or point on the map. Instead of guessing coordinates directly from free-form text, it walks the road network from the described starting point, checks physical cues against building records and street-level imagery, and shows the matched evidence and confidence score alongside the map. When multiple candidates remain tied on evidence, it asks one targeted clarifying question to resolve the ambiguity.

## How it works

1. **Parse (`Stage 1`)** — `gemini-4-argon` parses spoken or written route instructions into structured JSON with verbatim `evidence` spans, qualitative distance classes, landmarks, and target attributes, while user-uploaded storefront photos are read for signboard text.
2. **Search (`Stage 2`)** — Kakao Local API resolves the search window and starting anchors (such as subway station exits or named landmarks), and a beam search walks the local OpenStreetMap road and alley graph (`networkx` + `shapely`) to collect candidate buildings, places, and alleys.
3. **Verify (`Stage 3`)** — Top candidates are checked against Korean Building Registry records (above-ground floor count and main building use), user photos, signboard embeddings, and two-pass Google Street View imagery (`gemini-3.8-flash`).
4. **Judge (`Stage 4`)** — `gemini-4-argon` evaluates the candidate evidence table and street-level photos, citing specific `Evidence.id` values to select the winning location or ask a follow-up question.

**Verification loop and safeguards:**
- **Invented elements removed**: Any parsed region, anchor, landmark, step, or target cue whose `evidence` string is not a contiguous substring of the user's input transcript is removed before search.
- **Registry floor mismatch vetoes**: When the user specifies an explicit building floor count and the official Korean Building Registry record disagrees, the candidate receives a floor-mismatch veto penalty and cannot be promoted as a confident match.
- **Reasons without evidence IDs dropped**: Every reason returned by the judge model must cite valid `Evidence.id` entries attached to the chosen candidate; uncited claims are dropped, and the deterministic evidence score breaks ties if no cited reasons remain.
- **Clarifying question when evidence is tied (human in the loop)**: When top candidates remain within the tie margin or lack decisive verification, the pipeline generates a single clarifying question (asking about floor count, side of the street, or store name) and re-judges once the user replies.

## Gemini models used

| Model / Tool | Role in pipeline |
| :--- | :--- |
| `gemini-4-argon` | Stage 1 structured route parsing (`parse.py`) and Stage 4 evidence-grounded final judgment & question generation (`judge.py`) |
| `gemini-3.8-flash` | Hedged fallback for parse/judge, baseline coordinate comparison, user photo signboard OCR (`signboard.py`), and two-pass Street View visual verification (`vision.py`) |
| `gemini-3.5-flash-lite` | Fast shop-type and category matching during candidate filtering and tie-break question phrasing |
| `gemini-3.8-live` | Bidirectional WebSocket voice relay (`live.py`) for real-time spoken route descriptions |
| `gemini-3.8-flash-tts` | Korean and English text-to-speech synthesis (`/api/tts`) for spoken results and clarifying questions |
| `gemini-embedding-2` | Semantic similarity matching between OCR signboard text and nearby Kakao place names (`signboard.py`) |
| Google Maps grounding | Fallback place resolution (`types.GoogleMaps`) when local keyword search returns no anchor in the search window (`search.py`) |

## Project structure

- `app/config.py` — Runtime configuration, secret masking, and Gemini backend selection (Vertex AI or AI Studio).
- `app/contracts.py` — Shared dataclasses (`LatLng`, `ParseResult`, `Candidate`, `Evidence`, `Decision`, `Question`) and JSON schemas across all pipeline stages.
- `app/llm.py` — Gemini client wrapper providing structured JSON generation, hedged fallback chains, embeddings, and TTS.
- `app/geo.py` — Local tangent-plane meter coordinates, spherical geodesics, turning angles, and field-of-view ray projection.
- `app/parse.py` — Stage 1: parses spoken or written route descriptions into validated structural JSON and strips ungrounded cues.
- `app/signboard.py` — Stage 1 & 3: reads store signboards from user photos and matches them against nearby Kakao places using normalized text and `gemini-embedding-2`.
- `app/kakao.py` — Stage 2: Kakao Local REST API client for keyword, category, address, and coordinate-to-parcel lookups in memory.
- `app/mapdata.py` — Stage 2: loads and indexes OpenStreetMap roads, buildings, and point landmarks from SQLite into a spatial graph.
- `app/search.py` — Stage 2: resolves the search window, start anchors, and candidate paths/buildings by walking the road graph.
- `app/registry.py` — Stage 3: queries the Korean Building Registry title ledger API (`getBrTitleInfo`) and caches parcel records.
- `app/streetview.py` — Stage 3: fetches Google Street View panorama metadata and 640×640 images in memory only.
- `app/vision.py` — Stage 3: verifies route candidates using building registry records, user photos, and two-pass street-level imagery.
- `app/judge.py` — Stage 4: scores candidates, enforces building registry vetoes, validates cited evidence IDs, and decides or asks a follow-up question.
- `app/pipeline.py` — Orchestrates the 4-stage localization pipeline (`parse` → `search` → `verify` → `judge`), baseline comparison, and follow-up answer handling.
- `app/progress.py` — Thread-local progress message sink for streaming stage updates to the session state.
- `app/live.py` — Bidirectional WebSocket relay between the browser and `gemini-3.8-live`.
- `app/main.py` — FastAPI server exposing localization endpoints, session polling, warmup, TTS, evaluation results, and static web UI.

## Run locally

1. Install Python 3.13 and the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide the spatial SQLite database at `data/osm_kr/korea.sqlite` (or set `EODI_OSM_DB` to its path).
3. Configure environment variables for external APIs and Gemini access:
   - `KAKAO_REST_API_KEY` — Kakao Local REST API key
   - `DATA_GO_KR_KEY` — Public Data Portal key for the Korean Building Registry service
   - `MAPS_API_KEY` — Google Street View Static API key
   - `EODI_VERTEX_PROJECT`, `EODI_VERTEX_LOCATION` (`global`), and `EODI_VERTEX_ADC` (path to authorized user ADC JSON) for Vertex AI, or `GEMINI_API_KEY` for AI Studio
4. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8080
   ```
5. Open `http://127.0.0.1:8080` in a browser.

## Deploy to Cloud Run

The service deploys to Google Cloud Run (`asia-northeast3`, Gen 2 execution environment, 2 vCPU, 4 GiB memory, `min-instances=1`, `max-instances=1`, `--session-affinity`, `--cpu-boost`, `concurrency=30`, `timeout=3600`). The OpenStreetMap SQLite database (`osm_kr/korea.sqlite`) is mounted read-only from a Cloud Storage bucket at `/gcs` and copied into `/tmp/korea.sqlite` by `start.sh` before starting `uvicorn`. API keys are injected from Secret Manager (`kakao-rest-api-key`, `data-go-kr-key`, `maps-api-key`), and Gemini calls authenticate via the Cloud Run service account (`google.auth.default`).

Run the deployment and smoke-test script from Windows PowerShell:

```powershell
.\deploy\deploy.ps1 -Project <GCP_PROJECT_ID> -Bucket <GCS_BUCKET_NAME>
```

## Data & attribution

- **Map data & tiles**: © OpenStreetMap contributors, licensed under the Open Data Commons Open Database License (ODbL).
- **Local POI & address search**: Kakao Local API (`dapi.kakao.com`), kept in server memory only during request handling.
- **Building ledger data**: Korea building registry open data (`apis.data.go.kr`, `BldRgstHubService/getBrTitleInfo`).
- **Street-level imagery**: Google Street View Static API; Street View imagery is never stored and never shown next to non-Google maps.

## Limitations

- **Geographic coverage**: Search and verification cover South Korea only, relying on the Korea OpenStreetMap extract (`korea.sqlite`), Kakao Local API, and the Korean Building Registry API.
- **In-memory session store**: Active localization sessions (`_SESSIONS`) are stored in process memory, so Cloud Run deployments must run with `--max-instances 1` and `--session-affinity`.
- **Verification budget per query**: Visual verification inspects at most the top 3 ranked candidates (`TOP_VERIFY = 3`) and forwards at most 3 Street View photos (`JUDGE_PHOTOS = 3`) to the final judge call.
- **Single follow-up turn**: When candidates remain tied after initial verification, the pipeline asks one clarifying question and produces a final decision upon receiving the user's answer.
