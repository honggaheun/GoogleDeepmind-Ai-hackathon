"""Static UI asset and contract validation tests."""

from pathlib import Path
import json
import re

STATIC_DIR = Path(__file__).resolve().parent.parent / "app" / "static"

REQUIRED_IDS = [
    "map",
    "stageBadge",
    "leftCol",
    "dock",
    "spoken",
    "text",
    "legend",
    "region",
    "photoBtn",
    "photoFile",
    "here",
    "hereLabel",
    "photoChip",
    "photoThumb",
    "photoText",
    "photoRole",
    "photoClear",
    "rec",
    "live",
    "run",
    "liveMeter",
    "liveState",
    "liveLog",
    "chipsToggle",
    "chips",
    "timelinePanel",
    "tlTitle",
    "tlSummary",
    "totalTime",
    "timeline",
    "safeguards",
    "rightCol",
    "evTitle",
    "evCount",
    "evHint",
    "evidence",
    "answer",
    "viewer",
    "viewerMeta",
    "viewerClose",
    "viewerPhoto",
    "viewerSide",
    "tabs",
    "tabTimeline",
    "tabEvidence",
    "tabEvCount",
    "tools",
    "warmBadge",
    "mockBadge",
    "baselineToggle",
    "replayBtn",
    "evalBtn",
    "errorBar",
    "osmAttr",
    "placesAttr",
    "evalPanel",
    "evalClose",
    "evalBody",
]

REQUIRED_FUNCTIONS = [
    "el",
    "svg",
    "showError",
    "circle",
    "bearingDeg",
    "sector",
    "points",
    "fit",
    "markerEl",
    "baseHeightAt",
    "marker",
    "camMarker",
    "clearMap",
    "drawWindow",
    "animatePaths",
    "opacityFor",
    "setCands",
    "drawLandmarks",
    "drawCands",
    "setBaselineVisible",
    "clearDecisionDrawing",
    "drawDecision",
    "renderSpans",
    "stageSummary",
    "renderTimeline",
    "renderSafeguards",
    "photoKey",
    "photoGroups",
    "isOld",
    "photoBox",
    "verdictPill",
    "evCard",
    "renderEvidence",
    "highlight",
    "openViewer",
    "closeViewer",
    "setAnswerHeight",
    "showPanel",
    "setState",
    "runExample",
    "fitStrip",
    "renderIdle",
    "renderRunning",
    "ring",
    "evidenceBadges",
    "emphasize",
    "rankMap",
    "chainStrip",
    "renderAnswer",
    "renderQuestion",
    "toast",
    "copyShare",
    "speak",
    "reveal",
    "drawMap",
    "renderSignboard",
    "discovery",
    "fitBadge",
    "renderBadge",
    "render",
    "resetAll",
    "setBusy",
    "poll",
    "startPolling",
    "stopPolling",
    "post",
    "runText",
    "sendAudio",
    "sendAnswer",
    "meterStart",
    "meterStop",
    "toggleRecord",
    "runMock",
    "pollWarm",
    "loadChips",
    "openEval",
]


def test_static_files_exist():
    # Verify all 5 static files exist
    expected = ["index.html", "style.css", "app.js", "examples.json", "mock_locate.json"]
    for fname in expected:
        fpath = STATIC_DIR / fname
        assert fpath.exists(), f"Missing static file: {fname}"
        assert fpath.stat().st_size > 0, f"Empty static file: {fname}"


def test_index_html_elements():
    # Verify index.html contains all 58 required element IDs and CDN dependencies
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for elem_id in REQUIRED_IDS:
        pattern = rf'id=["\']{elem_id}["\']'
        assert re.search(pattern, html), f"Missing element ID in index.html: {elem_id}"

    assert "maplibre-gl" in html
    assert "pretendard" in html.lower()
    assert "style.css?v=" in html
    assert "app.js?v=" in html


def test_examples_json_structure():
    # Validate examples.json syntax and content
    raw = (STATIC_DIR / "examples.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert isinstance(data, list)
    assert len(data) >= 8

    card_items = [item for item in data if "card" in item]
    assert len(card_items) >= 3
    for item in data:
        assert "text" in item
        assert "region" in item


def test_mock_locate_json_contract():
    # Validate mock_locate.json conforms to locate session schema
    raw = (STATIC_DIR / "mock_locate.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data.get("session_id") == "mock"
    assert data.get("done") is True

    assert "input" in data
    assert "window" in data
    assert "progress" in data
    assert "anchor" in data
    assert "candidates" in data
    assert "evidence" in data
    assert "decision" in data

    candidates = data["candidates"]
    assert len(candidates) >= 1
    top = candidates[0]
    assert "centroid" in top
    assert "geometry" in top
    assert "levels" in top

    evidence = data["evidence"]
    assert len(evidence) >= 1
    assert all("kind" in e and "supports" in e for e in evidence)


def test_style_css_rules():
    # Validate style.css tokens and responsive breakpoints
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    assert "--left-w: 392px" in css
    assert "--right-w: 368px" in css
    assert "--panel:" in css
    assert "--brand:" in css
    assert "--accent:" in css
    assert "@media (max-width: 1100px)" in css
    assert "@media (max-width: 700px)" in css


def test_app_js_declarations():
    # Validate app.js contains all 76 functions and key constants
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    constants = [
        "STAGE_KO",
        "STAGE_HINT",
        "STAGE_WHO",
        "STAGE_WHO_SHORT",
        "KIND_KO",
        "KIND_ICON",
        "KIND_ORDER",
        "TYPE_KO",
        "CONF_KO",
        "CONF_PCT",
        "EMPTY",
        "VENUE",
        "METER",
        "METER_BARS",
    ]
    for c in constants:
        assert c in js, f"Missing constant in app.js: {c}"

    for fn in REQUIRED_FUNCTIONS:
        pattern = rf"(function\s+{fn}\b|const\s+{fn}\s*=|let\s+{fn}\s*=)"
        assert re.search(pattern, js), f"Missing function in app.js: {fn}"


def test_body_state_classes():
    # Verify presence of required body status and layout classes
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    state_classes = [
        "st-answered",
        "st-asking",
        "st-head",
        "st-idle",
        "st-ms",
        "st-name",
        "st-note",
        "st-running",
        "st-sum",
        "ran",
        "editing",
    ]
    for cls in state_classes:
        assert cls in css or cls in js, f"Missing body state class: {cls}"


def test_api_endpoints_and_links():
    # Verify integration endpoints and external map providers
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    endpoints = [
        "/api/locate_text",
        "/api/locate",
        "/api/session/",
        "/api/answer",
        "/api/warmup",
        "/api/eval",
        "/api/tts",
    ]
    for ep in endpoints:
        assert ep in js, f"Missing endpoint: {ep}"

    urls = [
        "https://map.kakao.com/link/map/",
        "https://map.kakao.com/link/roadview/",
        "https://tiles.openfreemap.org/styles/positron",
        "https://www.openstreetmap.org/",
    ]
    for url in urls:
        assert url in js, f"Missing external URL pattern: {url}"


def test_korean_ui_strings():
    # Verify critical Korean user messages and strings
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    strings = [
        "길 설명을 적거나 사진을 넣어 주세요.",
        "확신이 낮을 때만 가장 많이 걸러 내는 질문 하나를 드려요.",
        "주소를 몰라도, 설명으로 찾습니다.",
        "검증 없는 Gemini",
        "모의 화면에서는 음성을 만들지 않아요.",
    ]
    for s in strings:
        assert s in js, f"Missing Korean UI string: {s}"


def test_hidden_tab_and_safety_timeout():
    # Verify background tab optimization and reveal safety timer
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "document.hidden" in js
    assert "5000" in js


def test_photo_chip_specification():
    # Verify photo chip elements, styling, and text update logic
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="photoChip"' in html
    assert 'id="photoThumb"' in html
    assert 'id="photoText"' in html
    assert 'id="photoRole"' in html
    assert 'id="photoClear"' in html

    assert ".photo-chip" in css
    assert "#photoText" in css
    assert "#photoClear" in css
    assert "24px" in css

    assert "간판: ${names} → 지도 검색" in js
    assert "간판을 읽지 못했어요 — 닮음 대조만" in js
    assert 'S.photoRole = "target"' in js


def test_stage_badge_specification():
    # Verify stage badge DOM hierarchy, model mapping, and display conditions
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="stageBadge"' in html
    assert "STAGE_WHO = {" in js
    assert "Gemini 4 Argon 판관" in js

    assert "resp.decision || resp.done" in js
    assert 'b.hidden = true' in js
    assert 'class: "spinner"' in js
    assert 'class: "m"' in js

    assert "#stageBadge" in css
    assert "transform: translateX(-50%)" in css


def test_map_store_layers_specification():
    # Verify OSM POI and Kakao places layer definitions and filter exclusion
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert '"poi-dot"' in js
    assert '"poi-label"' in js
    assert '"places-dot"' in js
    assert '"places-label"' in js

    assert '"bus"' in js
    assert '"atm"' in js
    assert '"drinking_water"' in js
    assert '"playground"' in js
    assert '"cemetery"' in js
    assert '"toilets"' in js

    assert "placesSrc.setData" in js
    assert "장소 자료: 카카오" in js


def test_replay_button_flow_specification():
    # Verify replay button trigger, session ID reuse, and polling dispatch
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="replayBtn"' in html
    assert "S.lastSid = S.sid" in js
    assert "replayBtn.hidden = false" in js
    assert "resetAll()" in js
    assert "S.sid = lastSid" in js
    assert "S.replaying = true" in js
    assert "마지막 결과를 다시 재생합니다 — API 호출 없음" in js
    assert "startPolling()" in js
    assert "`/api/session/${S.sid}`" in js


def test_live_meter_specification():
    # Verify audio level meter calculations, speaking state, and cleanup
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'id="liveMeter"' in html
    assert 'id="liveState"' in html
    assert "window.meterStart = meterStart" in js
    assert "window.meterStop = meterStop" in js
    assert "fftSize = 1024" in js
    assert "rms * 7" in js
    assert "METER_BARS = 12" in js
    assert "Gemini 말하는 중" in js
    assert "말소리 감지" in js
    assert 'label: () => "녹음 중"' in js

    assert "#liveMeter.speaking" in css
    assert "meterPulse" in css


def test_collapsible_dock_specification():
    # Verify busy state marking, responsive dock collapse, and example chips toggle
    css = (STATIC_DIR / "style.css").read_text(encoding="utf-8")
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert 'document.body.classList.add("ran")' in js
    assert "body.ran .legend" in css
    assert "body.ran #chips" in css
    assert "body.ran .brand p" in css
    assert "body.ran #chipsToggle" in css
    assert "body.ran #chips.open" in css
    assert "chips.classList.toggle(\"open\")" in js


