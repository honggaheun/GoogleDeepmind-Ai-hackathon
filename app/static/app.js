/**
 * Multimodal spatial search UI client application.
 * Manages audio recording, MapLibre 3D rendering, reasoning timeline,
 * evidence inspector, Street View viewer, and verdict presentation.
 */

// Global client state container
const S = {
  sid: null,
  timer: null,
  mock: false,
  here: null,
  drawn: new Set(),
  evKey: null,
  resp: null,
  run: null,
  markers: [],
  decisionMarkers: [],
  baselineMarker: null,
  baselineLine: null,
  recorder: null,
  chunks: [],
  questionSource: null,
  pathAnim: null,
  viewerDone: null,
  viewerResolve: null,
  photoGroups: [],
  walk: null,
  lmDrawn: new Set(),
  lmSeen: new Set(),
  photo: null,
  photoUrl: null,
  photoRole: "target",
  lastSid: null,
  replaying: false,
  _sbUpdated: false,
  _sbToasted: false,
  examples: [],
  answeredDecision: null,
  prevRanks: {},
};

// Stage labels in Korean
const STAGE_KO = {
  parse: "해석",
  search: "탐색",
  verify: "확인",
  judge: "판정",
};

// Explanatory hint for each reasoning stage
const STAGE_HINT = {
  parse: "말을 검증 가능한 요소로 나눕니다",
  search: "기준점·경로·후보를 지도에서 찾습니다",
  verify: "건축물대장·사진으로 확인합니다",
  judge: "근거를 모아 판정하거나 되묻습니다",
};

// Stage executor descriptions
const STAGE_WHO = {
  parse: "Gemini 4 Argon",
  search: "카카오 · OSM · Google 지도",
  verify: "건축물대장 · Gemini",
  judge: "Gemini 4 Argon 판관",
};

// Short stage executor labels for compact badge display
const STAGE_WHO_SHORT = {
  parse: "Argon",
  search: "카카오 · OSM · Google",
  verify: "대장 · Gemini",
  judge: "Argon 판관",
};

// Evidence category labels in Korean
const KIND_KO = {
  route: "경로",
  kakao: "장소 대조",
  osm_tree: "OSM 나무",
  vision: "거리 사진",
  registry: "건축물대장",
  staleness: "사진 시점",
  link: "건물 연결",
  relation: "관계 확인",
  user_photo: "사용자 사진 대조",
};

// Evidence icon identifier mapping
const KIND_ICON = {
  route: "route",
  kakao: "shop",
  osm_tree: "tree",
  vision: "eye",
  registry: "doc",
  staleness: "clock",
  link: "link",
  relation: "pin",
  user_photo: "cam",
};

// Evidence sorting priority order
const KIND_ORDER = [
  "user_photo",
  "vision",
  "relation",
  "kakao",
  "registry",
  "link",
  "osm_tree",
  "staleness",
  "route",
];

// Candidate place types in Korean
const TYPE_KO = {
  building: "건물",
  named_place: "건물",
  alley: "골목",
  intersection: "교차로·모퉁이",
  point: "길 위 지점",
};

// Confidence grade labels in Korean
const CONF_KO = {
  high: "높음",
  medium: "보통",
  low: "낮음",
};

// Confidence gauge visual percentage values
const CONF_PCT = {
  high: 92,
  medium: 66,
  low: 35,
};

// Empty GeoJSON FeatureCollection template
const EMPTY = { type: "FeatureCollection", features: [] };

// GeoJSON FeatureCollection constructor helper
const FC = (features) => ({ type: "FeatureCollection", features: features || [] });

// Viewport width breakpoint query helper
const NARROW = () => window.matchMedia("(max-width: 1100px)").matches;

// Four-step pipeline display definitions
const HOW = [
  ["해석", "Gemini 4 Argon"],
  ["탐색", "카카오 · OSM · Google 지도"],
  ["확인", "간판 읽기 · 건축물대장"],
  ["판정", "Gemini 4 Argon"],
];

// Regular expression to highlight spatial keywords and metrics
const EMPH_RE = /('[^']{1,30}'|"[^"]{1,30}"|\d+(?:\.\d+)?\s?(?:m|km|층|초|번 출구)|맞은편|길 건너|옆|기준 랜드마크|기준 가게|간판|직진|확정|일치|건축물대장|경로)/g;

// Default map venue coordinates
const VENUE = [127.06370, 37.50680];

// Audio visualizer meter configuration and state
const METER = {
  ctx: null,
  an: null,
  timer: null,
  raf: null,
  at: 0,
  levels: [],
  speaking: () => false,
  label: () => "듣는 중",
};

// Number of visualizer level bars
const METER_BARS = 12;

// SVG icon path definitions
const ICON = {
  pin: '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle>',
  eye: '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle>',
  route: '<circle cx="6" cy="19" r="3"></circle><path d="M9 19h8.5a4.5 4.5 0 0 0 0-9H7a4.5 4.5 0 0 1 0-9H18"></path><circle cx="18" cy="5" r="3"></circle>',
  shop: '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline>',
  doc: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline>',
  tree: '<path d="M12 2L4 14h5v6h6v-6h5z"></path>',
  link: '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>',
  clock: '<circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline>',
  cam: '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle>',
  check: '<polyline points="20 6 9 17 4 12"></polyline>',
  x: '<line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line>',
  dash: '<line x1="5" y1="12" x2="19" y2="12"></line>',
  speaker: '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>',
  gem: '<polygon points="12 2 2 7 12 22 22 7 12 2"></polygon><polyline points="2 7 12 13 22 7"></polyline>',
  shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>',
  mic: '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line>',
  copy: '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>',
  ext: '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line>',
  reload: '<polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>',
  arrow: '<line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline>',
};

// DOM query selector shorthand
const $ = (sel) => document.querySelector(sel);

// Creates a DOM element with attributes and children
function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class" || k === "className") {
        node.className = v;
      } else if (k === "hidden") {
        if (v) node.setAttribute("hidden", "");
        else node.removeAttribute("hidden");
      } else if (k === "style" && typeof v === "object") {
        Object.assign(node.style, v);
      } else if (k.startsWith("on") && typeof v === "function") {
        node.addEventListener(k.slice(2).toLowerCase(), v);
      } else if (v !== null && v !== undefined && v !== false) {
        node.setAttribute(k, v === true ? "" : v);
      }
    }
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    if (typeof kid === "string" || typeof kid === "number") {
      node.appendChild(document.createTextNode(String(kid)));
    } else if (kid instanceof Node) {
      node.appendChild(kid);
    }
  }
  return node;
}

// Creates an inline SVG element for an icon
function svg(name, cls = "i") {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("class", cls);
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("fill", "none");
  s.setAttribute("stroke", "currentColor");
  s.setAttribute("stroke-width", "2");
  s.setAttribute("stroke-linecap", "round");
  s.setAttribute("stroke-linejoin", "round");
  s.innerHTML = ICON[name] || ICON.pin;
  return s;
}

// Displays an error message banner on top of the UI
function showError(msg) {
  const bar = $("#errorBar");
  if (!bar) return;
  if (!msg) {
    bar.hidden = true;
    bar.textContent = "";
    return;
  }
  bar.textContent = msg;
  bar.hidden = false;
  if (window.map && typeof map.easeTo === "function") {
    map.easeTo({ zoom: 14.5 });
  }
}

// Generates a GeoJSON Polygon approximating a circle around center
function circle(center, r, n = 72) {
  if (!center || !Array.isArray(center) || center.length < 2 || isNaN(center[0]) || isNaN(center[1])) {
    return EMPTY;
  }
  const [lng, lat] = center;
  const rad = Math.PI / 180;
  const dLat = r / 111320;
  const cosLat = Math.cos(lat * rad);
  const dLng = r / (111320 * (cosLat === 0 ? 1 : cosLat));
  const coords = [];
  for (let i = 0; i <= n; i++) {
    const theta = (i * 2 * Math.PI) / n;
    coords.push([lng + dLng * Math.cos(theta), lat + dLat * Math.sin(theta)]);
  }
  return { type: "Polygon", coordinates: [coords] };
}

// Computes compass bearing in degrees from point a to point b
function bearingDeg(a, b) {
  if (!a || !b) return 0;
  const rad = Math.PI / 180;
  const y = Math.sin((b[0] - a[0]) * rad) * Math.cos(b[1] * rad);
  const x =
    Math.cos(a[1] * rad) * Math.sin(b[1] * rad) -
    Math.sin(a[1] * rad) * Math.cos(b[1] * rad) * Math.cos((b[0] - a[0]) * rad);
  return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
}

// Generates a vision field wedge centered at heading
function sector(center, heading, half = 45, r = 28, n = 16) {
  if (!center || !Array.isArray(center) || center.length < 2 || isNaN(center[0]) || isNaN(center[1])) {
    return EMPTY;
  }
  const [lng, lat] = center;
  const rad = Math.PI / 180;
  const dLat = r / 111320;
  const cosLat = Math.cos(lat * rad);
  const dLng = r / (111320 * (cosLat === 0 ? 1 : cosLat));
  const coords = [[lng, lat]];
  const startAng = (heading - half) * rad;
  const endAng = (heading + half) * rad;
  for (let i = 0; i <= n; i++) {
    const ang = startAng + ((endAng - startAng) * i) / n;
    coords.push([lng + dLng * Math.sin(ang), lat + dLat * Math.cos(ang)]);
  }
  coords.push([lng, lat]);
  return { type: "Polygon", coordinates: [coords] };
}

// Flattens all coordinate pairs from a GeoJSON geometry
function points(geom) {
  if (!geom) return [];
  if (geom.type === "Point") return [geom.coordinates];
  if (geom.type === "LineString" || geom.type === "MultiPoint") return geom.coordinates;
  if (geom.type === "Polygon" || geom.type === "MultiLineString") return geom.coordinates.flat(1);
  if (geom.type === "MultiPolygon") return geom.coordinates.flat(2);
  return [];
}

// Fits map viewport to bounds of coordinates
function fit(coords, padding = 60, extra = {}) {
  if (!window.map || !coords || coords.length === 0) return;
  const valid = coords.filter((c) => Array.isArray(c) && c.length >= 2 && !isNaN(c[0]) && !isNaN(c[1]));
  if (valid.length === 0) return;
  if (valid.length === 1) {
    map.flyTo({ center: valid[0], zoom: extra.zoom || 16, ...extra });
    return;
  }
  let minLng = Infinity, minLat = Infinity, maxLng = -Infinity, maxLat = -Infinity;
  for (const [lng, lat] of valid) {
    if (lng < minLng) minLng = lng;
    if (lat < minLat) minLat = lat;
    if (lng > maxLng) maxLng = lng;
    if (lat > maxLat) maxLat = lat;
  }
  const bounds = [[minLng, minLat], [maxLng, maxLat]];
  try {
    if (typeof map.cameraForBounds === "function") {
      const cam = map.cameraForBounds(bounds, { padding, ...extra });
      if (cam) map.flyTo({ ...cam, ...extra });
      else map.fitBounds(bounds, { padding, ...extra });
    } else {
      map.fitBounds(bounds, { padding, ...extra });
    }
  } catch (_) {
    map.flyTo({ center: [(minLng + maxLng) / 2, (minLat + maxLat) / 2], zoom: 16 });
  }
}

// Constructs DOM elements for MapLibre marker without breaking transforms
function markerEl(cls, icon, text) {
  const wrap = el("div", { class: "mk-wrap" });
  const inner = el("div", { class: `mk ${cls}` });
  if (icon) {
    const icBox = el("div", { class: "ic" }, svg(icon));
    inner.appendChild(icBox);
  }
  if (text) {
    const txtNode = el("span", {}, text);
    inner.appendChild(txtNode);
  }
  wrap.appendChild(inner);
  return wrap;
}

// Queries existing 3D building height at location
function baseHeightAt(lnglat) {
  if (!window.map || !lnglat) return 0;
  try {
    const pt = map.project(lnglat);
    const feats = map.queryRenderedFeatures(pt, { layers: ["bld-3d"] });
    if (feats && feats.length > 0) {
      const p = feats[0].properties || {};
      return p.render_height || p.height || 0;
    }
  } catch (_) {}
  return 0;
}

// Creates and registers a MapLibre marker
function marker(lnglat, cls, icon, text, onclick) {
  if (!window.map || !lnglat) return null;
  const dom = markerEl(cls, icon, text);
  if (onclick) {
    dom.style.cursor = "pointer";
    dom.addEventListener("click", onclick);
  }
  const m = new maplibregl.Marker({ element: dom, anchor: "bottom" })
    .setLngLat(lnglat)
    .addTo(map);
  S.markers.push(m);
  return m;
}

// Creates a street view perspective camera marker
function camMarker(lnglat, onclick) {
  return marker(lnglat, "cam", "cam", null, onclick);
}

// Clears markers and resets map sources
function clearMap() {
  for (const m of S.markers) m.remove();
  S.markers = [];
  clearDecisionDrawing();
  if (S.baselineMarker) {
    S.baselineMarker.remove();
    S.baselineMarker = null;
  }
  S.drawn.clear();
  S.lmDrawn.clear();
  S.lmSeen.clear();
  const pAttr = $("#placesAttr");
  if (pAttr) {
    pAttr.hidden = true;
    pAttr.textContent = "";
  }
  const sources = ["win", "paths", "cands", "final", "baseline", "cones", "places"];
  for (const src of sources) {
    const s = map && map.getSource(src);
    if (s && typeof s.setData === "function") {
      s.setData(EMPTY);
    }
  }
}

// Draws spatial search window radius on map
function drawWindow(w) {
  if (!w || !w.center) return;
  const poly = circle(w.center, w.radius_m || 500);
  const src = map && map.getSource("win");
  if (src) src.setData(poly);
  fit(points(poly), 70, { pitch: 0, bearing: 0 });
}

// Animates camera and path progression from anchor to candidate
function animatePaths(cands) {
  return new Promise((resolve) => {
    if (!cands || cands.length === 0) return resolve();
    const topCand = cands.find((c) => c.rank === 1) || cands[0];
    const path = topCand.path;
    if (!path || path.length < 2 || document.hidden) {
      const allLines = cands.filter((c) => c.path && c.path.length >= 2).map((c) => ({
        type: "Feature",
        properties: { rank: c.rank || 1 },
        geometry: { type: "LineString", coordinates: c.path },
      }));
      const src = map && map.getSource("paths");
      if (src) src.setData(FC(allLines));
      return resolve();
    }

    let totalDist = 0;
    for (let i = 0; i < path.length - 1; i++) {
      const dx = (path[i + 1][0] - path[i][0]) * 111320 * Math.cos(path[i][1] * 0.01745);
      const dy = (path[i + 1][1] - path[i][1]) * 111320;
      totalDist += Math.hypot(dx, dy);
    }
    const duration = Math.min(Math.max(totalDist * 10, 2000), 4500);
    const startBearing = bearingDeg(path[0], path[1]);

    map.easeTo({ center: path[0], zoom: 17.2, pitch: 60, bearing: startBearing, duration: 700 });

    setTimeout(() => {
      if (document.hidden) {
        const allLines = cands.filter((c) => c.path && c.path.length >= 2).map((c) => ({
          type: "Feature",
          properties: { rank: c.rank || 1 },
          geometry: { type: "LineString", coordinates: c.path },
        }));
        const src = map && map.getSource("paths");
        if (src) src.setData(FC(allLines));
        return resolve();
      }

      let startTime = null;
      let camLng = path[0][0];
      let camLat = path[0][1];
      let curBearing = startBearing;

      function frame(now) {
        if (!startTime) startTime = now;
        const elapsed = now - startTime;
        let p = Math.min(elapsed / duration, 1);
        const k = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;

        const targetDist = totalDist * k;
        let accDist = 0;
        let tipIdx = 0;
        let tipCoord = path[0];

        for (let i = 0; i < path.length - 1; i++) {
          const dx = (path[i + 1][0] - path[i][0]) * 111320 * Math.cos(path[i][1] * 0.01745);
          const dy = (path[i + 1][1] - path[i][1]) * 111320;
          const segDist = Math.hypot(dx, dy);
          if (accDist + segDist >= targetDist || i === path.length - 2) {
            const segRatio = segDist === 0 ? 0 : (targetDist - accDist) / segDist;
            tipCoord = [
              path[i][0] + (path[i + 1][0] - path[i][0]) * segRatio,
              path[i][1] + (path[i + 1][1] - path[i][1]) * segRatio,
            ];
            tipIdx = i;
            break;
          }
          accDist += segDist;
        }

        const drawnPath = path.slice(0, tipIdx + 1).concat([tipCoord]);
        const src = map && map.getSource("paths");
        if (src) {
          src.setData(FC([{
            type: "Feature",
            properties: { rank: 1 },
            geometry: { type: "LineString", coordinates: drawnPath },
          }]));
        }

        camLng = camLng + (tipCoord[0] - camLng) * 0.2;
        camLat = camLat + (tipCoord[1] - camLat) * 0.2;
        const nextTarget = path[Math.min(tipIdx + 1, path.length - 1)];
        const targetBearing = bearingDeg(tipCoord, nextTarget);
        let bDiff = ((targetBearing - curBearing + 540) % 360) - 180;
        curBearing = curBearing + bDiff * 0.08;

        map.jumpTo({ center: [camLng, camLat], zoom: 17.2, pitch: 60, bearing: curBearing });

        if (p < 1 && !document.hidden) {
          S.pathAnim = requestAnimationFrame(frame);
        } else {
          const allLines = cands.filter((c) => c.path && c.path.length >= 2).map((c) => ({
            type: "Feature",
            properties: { rank: c.rank || 1 },
            geometry: { type: "LineString", coordinates: c.path },
          }));
          if (src) src.setData(FC(allLines));
          resolve();
        }
      }
      S.pathAnim = requestAnimationFrame(frame);
    }, 700);
  });
}

// Computes candidate feature opacity from rank
function opacityFor(rank) {
  if (rank === 1) return 0.95;
  if (rank === 2) return 0.65;
  return 0.35;
}

// Updates candidate 3D buildings and geometry with rise animation
function setCands(cands, finalId, rise = false, vetoed = []) {
  if (!cands || cands.length === 0) return Promise.resolve();
  const vetoSet = new Set(vetoed);
  const targets = cands.map((c) => {
    const isFinal = finalId && c.id === finalId;
    const isVetoed = vetoSet.has(c.id);
    const baseH = baseHeightAt(c.centroid);
    let fullH = Math.max((c.levels || 3) * 3.2, baseH + 1, 12);
    if (finalId && !isFinal) fullH *= 0.35;
    const col = isFinal ? "#f97316" : isVetoed ? "#94a3b8" : "#2563eb";
    const op = isFinal ? 0.95 : isVetoed ? 0.3 : opacityFor(c.rank || 1);
    return {
      c,
      geom: c.geometry || { type: "Point", coordinates: c.centroid },
      fullH,
      col,
      op,
    };
  });

  const src = map && map.getSource("cands");
  if (!src) return Promise.resolve();

  if (!rise || document.hidden) {
    const feats = targets.map((t) => ({
      type: "Feature",
      properties: { h: t.fullH, color: t.col, opacity: t.op },
      geometry: t.geom,
    }));
    src.setData(FC(feats));
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    const dur = 750;
    const start = performance.now();
    function tick(now) {
      const p = Math.min((now - start) / dur, 1);
      const k = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
      const feats = targets.map((t) => ({
        type: "Feature",
        properties: { h: t.fullH * k, color: t.col, opacity: t.op },
        geometry: t.geom,
      }));
      src.setData(FC(feats));
      if (p < 1 && !document.hidden) {
        requestAnimationFrame(tick);
      } else {
        resolve();
      }
    }
    requestAnimationFrame(tick);
  });
}

// Plots confirmed landmark markers along search route
function drawLandmarks(list) {
  if (!list) return;
  for (const item of list) {
    const d = item.data || item;
    const lnglat = d.lnglat || item.lnglat;
    if (!lnglat || item.supports === false) continue;
    const key = `${lnglat[0].toFixed(5)},${lnglat[1].toFixed(5)}`;
    if (S.lmDrawn.has(key)) continue;
    S.lmDrawn.add(key);

    const dist = d.distance_m ? ` ${d.distance_m}m` : "";
    const label = `${d.place || "랜드마크"}${dist}`;
    const isPhoto = item.kind === "user_photo";
    const cls = isPhoto ? "lm photo" : "lm";
    const icon = isPhoto ? "cam" : "check";

    marker(lnglat, cls, icon, label, () => {
      if (item.id) highlight(item.id);
      else toast(label);
    });
  }
}

// Draws candidate labels and markers on map
function drawCands(cands, resp) {
  if (!cands) return;
  for (const c of cands) {
    if (!c.centroid) continue;
    const vetoed = resp && resp.decision && (resp.decision.vetoed || []).includes(c.id);
    const cls = vetoed ? "cand out" : "cand";
    marker(c.centroid, cls, null, c.id, () => {
      highlight(c.id);
    });
  }
  if (cands.length > 0) {
    fit(cands.map((c) => c.centroid), 110, { pitch: 50, bearing: -17, duration: 1200 });
  }
}

// Toggles display of unverified baseline estimate marker and line
function setBaselineVisible() {
  const toggle = $("#baselineToggle");
  const on = toggle && toggle.checked;
  const bl = S.resp && S.resp.baseline;
  const lineSrc = map && map.getSource("baseline");
  if (!on || !bl || !bl.lnglat) {
    if (S.baselineMarker) {
      S.baselineMarker.remove();
      S.baselineMarker = null;
    }
    if (lineSrc) lineSrc.setData(EMPTY);
    return;
  }
  const dist = bl.distance_to_result_m || 0;
  if (!S.baselineMarker) {
    const text = `검증 없는 Gemini · 우리 답과 ${dist}m 차이`;
    S.baselineMarker = marker(bl.lnglat, "gem", "gem", text);
  }
  const top = S.resp.candidates && (S.resp.candidates.find((c) => c.rank === 1) || S.resp.candidates[0]);
  if (top && top.centroid && lineSrc) {
    lineSrc.setData(FC([{
      type: "Feature",
      geometry: { type: "LineString", coordinates: [bl.lnglat, top.centroid] },
    }]));
  }
}

// Removes final verdict marker and decision overlays
function clearDecisionDrawing() {
  for (const m of S.decisionMarkers) m.remove();
  S.decisionMarkers = [];
  const src = map && map.getSource("final");
  if (src) src.setData(EMPTY);
  const cones = map && map.getSource("cones");
  if (cones) cones.setData(EMPTY);
}

// Positions verdict marker and flies camera into winning building
function drawDecision(resp) {
  clearDecisionDrawing();
  const d = resp && resp.decision;
  if (!d) return;

  const finalId = d.candidate_id;
  const vetoed = d.vetoed || [];
  setCands(resp.candidates, finalId, false, vetoed);

  // Render vetoed candidate markers
  for (const cid of vetoed) {
    const c = (resp.candidates || []).find((x) => x.id === cid);
    if (c && c.centroid) {
      const why = c.veto_reason || "주용도 불일치";
      const m = marker(c.centroid, "cand out", null, `${c.id} 제외 · ${why}`);
      if (m) S.decisionMarkers.push(m);
    }
  }

  const cand = (resp.candidates || []).find((c) => c.id === finalId);
  if (!cand || !cand.centroid) return;

  const m = marker(cand.entrance || cand.centroid, "result", "pin", cand.name || "목적지");
  if (m) S.decisionMarkers.push(m);

  const finalSrc = map && map.getSource("final");
  if (finalSrc) {
    finalSrc.setData(FC([{
      type: "Feature",
      geometry: { type: "Point", coordinates: cand.entrance || cand.centroid },
    }]));
  }

  // Draw camera viewpoints and vision cones
  const coneFeats = [];
  const pGroups = photoGroups(resp.evidence || []);
  for (const g of pGroups) {
    if (g.pano && g.heading != null) {
      const cm = camMarker(g.pano, () => openViewer(g));
      if (cm) S.decisionMarkers.push(cm);
      const s = sector(g.pano, g.heading);
      if (s && s.coordinates) {
        coneFeats.push({ type: "Feature", geometry: s });
      }
    }
  }
  for (const vp of cand.view_points || []) {
    if (vp.lnglat && vp.heading != null) {
      const cm = camMarker(vp.lnglat, () => {});
      if (cm) S.decisionMarkers.push(cm);
      const s = sector(vp.lnglat, vp.heading);
      if (s && s.coordinates) {
        coneFeats.push({ type: "Feature", geometry: s });
      }
    }
  }
  const conesSrc = map && map.getSource("cones");
  if (conesSrc) conesSrc.setData(FC(coneFeats));

  setBaselineVisible();

  // Fit camera bounds around anchor and result
  const fitCoords = [cand.centroid];
  if (resp.anchor && resp.anchor.lnglat) fitCoords.push(resp.anchor.lnglat);
  if (cand.path && cand.path.length > 0) fitCoords.push(...cand.path);
  fit(fitCoords, 110, { pitch: 45, bearing: -17, duration: 1100 });

  if (!document.hidden) {
    setTimeout(() => {
      if (document.hidden) return;
      map.flyTo({
        center: cand.centroid,
        zoom: Math.max(map.getZoom(), 17.4),
        pitch: 58,
        bearing: -24,
        duration: 1800,
        essential: true,
      });
    }, 1700);
  }
}

// Renders colored tokens in spoken text display
function renderSpans(input) {
  const spoken = $("#spoken");
  const area = $("#text");
  const legend = $("#legend");
  if (!spoken || !area) return;
  if (!input || !input.evidence_spans || input.evidence_spans.length === 0) {
    spoken.hidden = true;
    area.hidden = false;
    if (legend) legend.hidden = true;
    return;
  }
  spoken.innerHTML = "";
  const text = input.text || "";
  const spans = [...input.evidence_spans].sort((a, b) => a.start - b.start);
  let cur = 0;
  for (const sp of spans) {
    if (sp.start > cur) {
      spoken.appendChild(document.createTextNode(text.slice(cur, sp.start)));
    }
    const cls = `s s-${sp.kind || sp.type || "step"}`;
    const chunk = el("span", { class: cls }, text.slice(sp.start, sp.end));
    spoken.appendChild(chunk);
    cur = sp.end;
  }
  if (cur < text.length) {
    spoken.appendChild(document.createTextNode(text.slice(cur)));
  }
  area.hidden = true;
  spoken.hidden = false;
  if (legend) legend.hidden = false;
}

// Computes summary text for reasoning stage
function stageSummary(stage, p, resp) {
  const st = (resp && resp.stats) || {};
  if (stage === "parse") {
    const n = (resp && resp.input && resp.input.evidence_spans && resp.input.evidence_spans.length) || 0;
    const inv = st.invented ? ` · 지어낸 요소 ${st.invented}건 제거` : "";
    return `요소 ${n}개 찾음${inv}`;
  }
  if (stage === "search") {
    const anc = (resp && resp.anchor && resp.anchor.name) ? `기준점 ${resp.anchor.name}` : "";
    const n = (resp && resp.candidates && resp.candidates.length) || 0;
    return `${anc} · 후보 ${n > 0 ? n + "곳" : "후보 없음"}`;
  }
  if (stage === "verify") {
    if (p && p.note && p.note.includes("생략")) return p.note;
    if (st.vision_images) {
      const fl = st.vision_flipped ? ` · 착각 ${st.vision_flipped}건 뒤집힘` : "";
      const sk = st.vision_skipped_stale ? ` · 낡은 사진 ${st.vision_skipped_stale}장 제외` : "";
      return `거리 사진 ${st.vision_images}장 판독${fl}${sk}`;
    }
    return "거리 사진·대장 확인 완료";
  }
  if (stage === "judge") {
    const d = resp && resp.decision;
    if (!d) return "판정 완료";
    if (d.question) return `되묻기: ${(d.question || {}).intent_ko || ""}`;
    const cand = (resp.candidates || []).find((c) => c.id === d.candidate_id);
    return `${cand ? cand.name || TYPE_KO[cand.type] || cand.id : "답 없음"} · 확신도 ${CONF_KO[d.confidence] || d.confidence}`;
  }
  return "";
}

// Renders 4-step pipeline timeline list
function renderTimeline(resp) {
  const ol = $("#timeline");
  const totalEl = $("#totalTime");
  const sumEl = $("#tlSummary");
  if (!ol) return;
  ol.innerHTML = "";
  if (!resp || !resp.progress) {
    if (totalEl) totalEl.textContent = "";
    if (sumEl) sumEl.textContent = "";
    return;
  }
  let total = 0;
  const stages = ["parse", "search", "verify", "judge"];
  const progMap = new Map((resp.progress || []).map((p) => [p.stage, p]));
  for (const stg of stages) {
    const p = progMap.get(stg);
    const li = el("li");
    let status = p ? p.status : "pending";
    li.className = status;
    const dot = el("span", { class: "dot" });
    if (status === "done") {
      dot.appendChild(svg("check"));
    } else if (status === "running") {
      dot.appendChild(el("div", { class: "spinner" }));
    } else {
      dot.appendChild(svg("dash"));
    }
    const msTxt = p && p.ms ? `${(p.ms / 1000).toFixed(1)}초` : status === "running" ? "진행 중" : "";
    if (p && p.ms) total += p.ms;
    const head = el("div", { class: "st-head" },
      el("span", { class: "st-name" }, STAGE_KO[stg]),
      el("span", { class: "st-ms" }, msTxt)
    );
    const sum = el("div", { class: "st-sum" }, stageSummary(stg, p, resp));
    const feats = el("div", { class: "feats" });
    const featList = (p && p.features) || (STAGE_WHO[stg] ? [STAGE_WHO[stg]] : []);
    for (const f of featList) {
      feats.appendChild(el("span", { class: "feat" }, f));
    }
    const body = el("div", {}, head, sum, feats);
    if (p && p.note && !p.note.includes("생략")) {
      body.appendChild(el("div", { class: "st-note" }, p.note));
    }
    li.appendChild(dot);
    li.appendChild(body);
    ol.appendChild(li);
  }
  if (totalEl) totalEl.textContent = `${(total / 1000).toFixed(1)}초`;
  if (sumEl) {
    const lastDone = [...(resp.progress || [])].reverse().find((p) => p.status === "done");
    sumEl.textContent = lastDone ? `${STAGE_KO[lastDone.stage]} 완료` : "";
  }
  renderSafeguards(resp.safeguards);
}

// Renders AI safeguards summary chips
function renderSafeguards(list) {
  const box = $("#safeguards");
  if (!box) return;
  box.innerHTML = "";
  if (!list || list.length === 0) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.appendChild(el("div", { class: "sg-title" }, svg("shield"), "AI 약점 차단"));
  for (const item of list) {
    const row = el("div", { class: "sg-item" }, `• ${item.text_ko || item.kind}`);
    box.appendChild(row);
  }
}

// Generates grouping key for street view photos
function photoKey(e) {
  const d = e.data || {};
  return `${d.view_point_index || 0}_${e.capture_date || "latest"}`;
}

// Groups photo-related evidence by view point and date
function photoGroups(list) {
  const groups = new Map();
  for (const e of list || []) {
    if (e.kind !== "vision" && e.kind !== "user_photo") continue;
    const k = photoKey(e);
    if (!groups.has(k)) {
      const d = e.data || {};
      groups.set(k, {
        key: k,
        capture_date: e.capture_date,
        image_b64: e.image_b64 || d.image_b64,
        boxes: e.boxes || d.boxes || [],
        candidate_id: e.candidate_id,
        heading: d.heading,
        pano: d.pano,
        evs: [],
        data: d,
      });
    }
    const g = groups.get(k);
    if (e.boxes && (!g.boxes || g.boxes.length === 0)) g.boxes = e.boxes;
    if (e.image_b64 && !g.image_b64) g.image_b64 = e.image_b64;
    g.evs.push(e);
  }
  return Array.from(groups.values());
}

// Tests whether a capture date is older than 5 years
function isOld(capture) {
  if (!capture) return false;
  const match = String(capture).match(/^(\d{4})/);
  if (!match) return false;
  const year = parseInt(match[1], 10);
  return year <= new Date().getFullYear() - 5;
}

// Generates image container with detection boxes
function photoBox(g, big) {
  const box = el("div", { class: `photo ${big ? "big" : ""}` });
  if (g.image_b64) {
    const img = el("img", { src: `data:image/jpeg;base64,${g.image_b64}`, alt: "거리 사진" });
    box.appendChild(img);
  } else {
    const desc = el("div", { class: "p-empty" }, "거리 사진(모의 화면에는 넣지 않음) — 박스 위치만 표시");
    box.appendChild(desc);
  }

  // Overlay detected bounding boxes
  for (const b of g.boxes || []) {
    if (!b.box_2d) continue;
    const [ymin, xmin, ymax, xmax] = b.box_2d;
    const bDiv = el("div", {
      class: `box ${b.name === "building" ? "bldg" : ""} ${b.status && b.status !== "present" ? "uncertain" : "present"}`,
      style: {
        top: `${ymin / 10}%`,
        left: `${xmin / 10}%`,
        height: `${(ymax - ymin) / 10}%`,
        width: `${(xmax - xmin) / 10}%`,
      },
    });
    const label = b.name === "building"
      ? `건물${b.floors ? " " + b.floors + "층" : ""}`
      : `${b.name || "표적"}${b.status && b.status !== "present" ? " · 불확실" : ""}`;
    bDiv.appendChild(el("span", {}, label));
    box.appendChild(bDiv);
  }

  if (g.capture_date) {
    box.appendChild(el("div", { class: "p-cap" }, `촬영 ${g.capture_date}${isOld(g.capture_date) ? " · 5년 넘음" : ""}`));
  }
  return box;
}

// Builds support / veto pill for an evidence item
function verdictPill(e) {
  const sup = e.supports;
  const cls = sup === true ? "v-ok" : sup === false ? "v-no" : "v-unk";
  const txt = sup === true ? "지지" : sup === false ? "반대" : "참고";
  return el("span", { class: `verdict ${cls}` }, txt);
}

// Constructs single evidence card element
function evCard(e, opts = {}) {
  const group = opts.group;
  const card = el("div", {
    id: `ev-${e.id}`,
    class: `ev k-${e.kind || "route"}`,
    onclick: () => {
      highlight(e.id);
      const cand = S.resp && S.resp.candidates && S.resp.candidates.find((c) => c.id === e.candidate_id);
      if (cand && cand.centroid) {
        map.flyTo({ center: cand.centroid, zoom: 18, pitch: 60, duration: 1200 });
      }
    },
  });
  const ico = el("span", { class: "ev-ico" }, svg(KIND_ICON[e.kind] || "pin"));
  const title = el("span", { class: "ev-title" }, KIND_KO[e.kind] || e.kind);
  const idBadge = el("span", { class: "ev-id" }, `${e.id}${e.candidate_id ? " · " + e.candidate_id : ""}`);
  const head = el("div", { class: "ev-head" }, ico, title, idBadge, verdictPill(e));
  const sum = el("div", { class: "ev-sum" }, e.summary_ko || "");
  const w = e.weight || 0;
  const absW = Math.abs(w);
  const wPct = Math.min(100, (absW / 3) * 100);

  const wBar = el("div", { class: "wbar" },
    el("i", { class: w >= 0 ? "pos" : "neg", style: { width: `${wPct}%` } })
  );
  const meta = el("div", { class: "ev-meta" },
    el("span", {}, `가중치 ${w > 0 ? "+" : ""}${w.toFixed(2)}`),
    wBar
  );
  if (e.capture_date) {
    const isStale = isOld(e.capture_date);
    meta.appendChild(el("span", { class: isStale ? "stale" : "" }, `촬영 ${e.capture_date}${isStale ? " (5년 넘음)" : ""}`));
  }
  card.appendChild(head);
  card.appendChild(sum);
  card.appendChild(meta);

  // Google Maps attribution link if present
  const d = e.data || {};
  if (d.maps_sources && Array.isArray(d.maps_sources)) {
    const srcRow = el("div", { class: "ev-src" },
      el("span", { translate: "no" }, "출처: Google 지도")
    );
    for (const src of d.maps_sources) {
      srcRow.appendChild(el("a", { href: src.url || "#", target: "_blank", rel: "noreferrer", translate: "no" }, src.title || "Google 지도"));
    }
    card.appendChild(srcRow);
  }

  // Vision inspection action button
  if (group || e.boxes || e.image_b64) {
    const pBtn = el("button", {
      class: "ev-photo-btn",
      type: "button",
      onclick: (ev) => {
        ev.stopPropagation();
        openViewer(group || { key: e.id, capture_date: e.capture_date, image_b64: e.image_b64, boxes: e.boxes, evs: [e], data: e.data });
      },
    }, svg("eye"), el("span", {}, "현장 사진 보기"));
    card.appendChild(pBtn);
  }

  return card;
}

// Populates right column with evidence cards
function renderEvidence(list) {
  const col = $("#rightCol");
  const box = $("#evidence");
  const countEl = $("#evCount");
  const tabCount = $("#tabEvCount");
  if (!box) return;
  box.innerHTML = "";
  if (!list || list.length === 0) {
    if (col) col.hidden = true;
    if (countEl) countEl.textContent = "";
    if (tabCount) tabCount.textContent = "";
    return;
  }
  const sorted = [...list].sort((a, b) => {
    const ia = KIND_ORDER.indexOf(a.kind);
    const ib = KIND_ORDER.indexOf(b.kind);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
  const pGroups = photoGroups(list);
  const groupMap = new Map();
  for (const g of pGroups) {
    for (const e of g.evs) groupMap.set(e.id, g);
  }
  for (const e of sorted) {
    box.appendChild(evCard(e, { group: groupMap.get(e.id) }));
  }
  const txt = `${list.length}건`;
  if (countEl) countEl.textContent = txt;
  if (tabCount) tabCount.textContent = txt;
  if (col && document.body.classList.contains("st-answered")) {
    col.hidden = false;
  }
}

// Highlights targeted evidence card and moves view
function highlight(id) {
  const prev = document.querySelectorAll(".ev.active");
  prev.forEach((item) => item.classList.remove("active"));
  const card = document.getElementById(`ev-${id}`);
  if (card) {
    card.classList.add("active");
    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } else {
    toast(`근거 ${id} 카드를 찾지 못함`);
  }
}

// Opens modal street view viewer
function openViewer(g, opts = {}) {
  const v = $("#viewer");
  if (!v || !g) return Promise.resolve();
  const meta = $("#viewerMeta");
  const photoContainer = $("#viewerPhoto");
  const side = $("#viewerSide");

  const isStale = isOld(g.capture_date);
  if (meta) {
    meta.textContent = `촬영 ${g.capture_date || ""}${isStale ? " · 5년 넘음 → 간판·색은 낮은 가중치" : ""}`;
  }
  if (photoContainer) {
    photoContainer.innerHTML = "";
    photoContainer.appendChild(photoBox(g, true));
  }
  if (side) {
    side.innerHTML = "";
    if (S.photoUrl || S.photo) {
      const uphoto = el("div", { class: "uphoto" },
        el("img", { src: S.photoUrl || S.photo, alt: "사용자 사진" }),
        el("div", { class: "u-cap" }, "사용자 사진(최신 모습) — 거리 사진과 같은 건물인지 비교")
      );
      side.appendChild(uphoto);
    }
    const evCount = (g.evs || []).length;
    side.appendChild(el("div", { class: "v-sum" }, `${evCount}가지를 사진에서 확인 · 1차 3.8 Flash 박스 → 2차 Flash-Lite 재확인`));
    for (const e of g.evs || []) {
      side.appendChild(evCard(e, { group: g }));
    }
  }

  document.body.classList.add("viewing");
  v.hidden = false;

  return new Promise((resolve) => {
    S.viewerResolve = resolve;
    if (opts.ms) {
      setTimeout(() => {
        closeViewer();
      }, opts.ms);
    }
  });
}

// Closes modal street view viewer
function closeViewer() {
  const v = $("#viewer");
  if (v) v.hidden = true;
  document.body.classList.remove("viewing");
  if (S.viewerResolve) {
    const fn = S.viewerResolve;
    S.viewerResolve = null;
    fn();
  }
}

// Updates CSS variable for answer panel height
function setAnswerHeight() {
  const a = $("#answer");
  if (!a) return;
  const h = a.offsetHeight || 0;
  document.documentElement.style.setProperty("--answer-h", `${h}px`);
}

// Toggles timeline or evidence bottom sheet on mobile
function showPanel(name) {
  if (name === "timeline") {
    document.body.classList.toggle("show-timeline");
    document.body.classList.remove("show-evidence");
  } else if (name === "evidence") {
    document.body.classList.toggle("show-evidence");
    document.body.classList.remove("show-timeline");
  }
}

// Switches UI lifecycle state
function setState(name) {
  document.body.classList.remove("st-idle", "st-running", "st-answered", "st-asking");
  document.body.classList.add(`st-${name}`);
  const rCol = $("#rightCol");
  if (rCol) rCol.hidden = name !== "answered";
  const bToggle = $("#baselineToggle");
  if (bToggle) setBaselineVisible();
}

// Populates inputs and runs search for an example
function runExample(ex) {
  const t = $("#text");
  const r = $("#region");
  if (t) t.value = ex.text || "";
  if (r) r.value = ex.region || "";

  if (ex.needs_photo && !S.photo) {
    document.body.classList.add("editing");
    toast(ex.hint || "사진을 넣고 역할을 고른 뒤 찾기를 누르세요");
    const pFile = $("#photoFile");
    if (pFile) pFile.click();
    return;
  }
  runText(ex.text);
}

// Dynamically rescales 4-step strip multiplier to prevent overflow
function fitStrip(strip) {
  if (!strip) return;
  let hs = 1.0;
  strip.style.setProperty("--hs", hs);
  while (strip.scrollWidth > strip.clientWidth && hs > 0.5) {
    hs -= 0.04;
    strip.style.setProperty("--hs", hs.toFixed(2));
  }
}

// Renders idle state cards and 4-step guide strip
function renderIdle() {
  const ans = $("#answer");
  if (!ans) return;
  ans.className = "panel a-idle";
  ans.innerHTML = "";
  const head = el("div", { class: "idle-head" }, "주소를 몰라도, 설명으로 찾습니다.");
  const cardsBox = el("div", { class: "ex-cards" });
  const cardExs = (S.examples || []).filter((ex) => ex.card).slice(0, 3);
  for (const ex of cardExs) {
    const card = el("div", { class: "ex-card", onclick: () => runExample(ex) });
    card.appendChild(el("b", {}, ex.card));
    const kwBox = el("div", { class: "ex-kw" });
    for (const kw of ex.keywords || []) {
      kwBox.appendChild(el("span", {}, kw));
    }
    card.appendChild(kwBox);
    cardsBox.appendChild(card);
  }

  const strip = el("div", { class: "how open" });
  HOW.forEach(([name, who], i) => {
    if (i > 0) strip.appendChild(el("span", { class: "how-arrow" }, "→"));
    const step = el("div", { class: "how-step" },
      el("div", { class: "how-n" }, i + 1),
      el("div", { class: "how-t" }, name, el("span", { class: "feat" }, who))
    );
    strip.appendChild(step);
  });

  ans.appendChild(head);
  ans.appendChild(cardsBox);
  ans.appendChild(strip);
  fitStrip(strip);
  setAnswerHeight();
}

// Renders in-progress status card
function renderRunning(resp) {
  const ans = $("#answer");
  if (!ans) return;
  ans.className = "panel";
  ans.innerHTML = "";
  const wrap = el("div", { class: "a-run" });
  const prog = (resp && resp.progress) || [];
  const runningP = prog.find((p) => p.status === "running") || prog[prog.length - 1];
  const stg = runningP ? runningP.stage : "parse";
  const stgIdx = ["parse", "search", "verify", "judge"].indexOf(stg) + 1;
  const line = el("div", { class: "line" },
    el("div", { class: "spinner" }),
    el("span", {}, `${stgIdx}/4 ${STAGE_KO[stg]} 중 — ${runningP ? (runningP.sub || STAGE_HINT[stg]) : "탐색 중"}`)
  );
  wrap.appendChild(line);
  ans.appendChild(wrap);
  setAnswerHeight();
}

// Draws circular confidence rating gauge
function ring(conf, small = false) {
  const pct = CONF_PCT[conf] || 50;
  const rad = 28;
  const circ = 2 * Math.PI * rad;
  const off = circ * (1 - pct / 100);
  const wrap = el("div", {
    class: `ring ${small ? "sm" : ""}`,
    title: "확신도 등급 — 확률이 아니라 1·2위 점수 차, 지지 근거 수, 완화 수로 정한 등급(높음·보통·낮음)",
  });
  const color = conf === "high" ? "var(--ok)" : conf === "medium" ? "var(--accent)" : "var(--bad)";
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 72 72");
  s.innerHTML = `
    <circle cx="36" cy="36" r="${rad}" fill="none" stroke="rgba(15,23,42,.08)" stroke-width="6"/>
    <circle cx="36" cy="36" r="${rad}" fill="none" stroke="${color}" stroke-width="6"
      stroke-dasharray="${circ}" stroke-dashoffset="${off}" stroke-linecap="round"/>
  `;
  wrap.appendChild(s);
  wrap.appendChild(el("span", { class: "lbl" }, CONF_KO[conf] || "낮음"));
  return wrap;
}

// Builds badges for confirmed evidence categories
function evidenceBadges(resp, cand) {
  const box = el("div", { class: "badges" });
  const evs = (resp && resp.evidence) || [];
  const kinds = new Set(evs.filter((e) => e.supports).map((e) => e.kind));
  if (kinds.has("kakao")) box.appendChild(el("span", { class: "badge-ok" }, "가게 자료 일치"));
  if (kinds.has("registry")) box.appendChild(el("span", { class: "badge-ok" }, "대장 건물명 일치"));
  if (kinds.has("relation")) box.appendChild(el("span", { class: "badge-ok" }, "관계 확인"));
  if (kinds.has("vision")) box.appendChild(el("span", { class: "badge-ok" }, "사진 확인"));
  if (kinds.has("user_photo")) box.appendChild(el("span", { class: "badge-ok" }, "간판 확인"));
  const conf = resp && resp.decision && resp.decision.confidence;
  if (conf) {
    box.appendChild(el("span", { class: `badge-ok badge-conf ${conf}` }, `확신도 ${CONF_KO[conf] || conf}`));
  }
  return box;
}

// Highlights keywords numbers and tokens in a sentence
function emphasize(text) {
  if (!text) return "";
  return text.replace(EMPH_RE, (m) => {
    if (/^\d+(?:\.\d+)?\s?(?:m|km|층|초|번 출구)$/.test(m)) {
      return `<span class="hl-num">${m}</span>`;
    }
    return `<span class="hl-key">${m}</span>`;
  });
}

// Builds candidate id to rank lookup table
function rankMap(cands) {
  const map = {};
  for (const c of cands || []) {
    map[c.id] = c.rank;
  }
  return map;
}

// Constructs multimodal reasoning chain strip
function chainStrip(resp, cand) {
  const sb = (resp && resp.signboard) || {};
  const names = sb.names ? sb.names.join(", ") : "";
  const chain = el("div", { class: "chain" });
  if (S.photoUrl || S.photo) {
    chain.appendChild(el("img", { src: S.photoUrl || S.photo, alt: "사진 간판" }));
  }
  chain.appendChild(el("span", { class: "node" }, names ? `간판 '${names}'` : "간판을 못 읽음"));
  chain.appendChild(el("span", { class: "arrow" }, "→"));
  chain.appendChild(el("span", { class: "node teal" }, `${cand.name || "가게"} · 찾는 곳`));
  return chain;
}

// Renders final answer verdict card
function renderAnswer(resp) {
  const ans = $("#answer");
  if (!ans) return;
  ans.className = "panel result";
  ans.innerHTML = "";
  const d = resp && resp.decision;
  const cand = (resp.candidates || []).find((c) => c.id === (d && d.candidate_id)) || (resp.candidates && resp.candidates[0]);
  if (!cand) {
    ans.appendChild(el("div", { class: "a-name" }, "위치를 정하지 못했어요. 대략 위치나 출발점을 더 알려 주세요."));
    setAnswerHeight();
    return;
  }

  if (resp.signboard && resp.signboard.applied) {
    ans.appendChild(chainStrip(resp, cand));
  }

  const top = el("div", { class: "a-top" });
  const main = el("div", { class: "a-main" });
  main.appendChild(el("div", { class: "a-name" }, cand.name || "목적지"));

  const keyText = d && d.reasons && d.reasons[0] ? d.reasons[0].text_ko : "";
  const keyEl = el("div", { class: "a-key" });
  keyEl.innerHTML = emphasize(keyText);
  if (d && d.reasons && d.reasons[0] && d.reasons[0].evidence_ids) {
    for (const eid of d.reasons[0].evidence_ids) {
      keyEl.appendChild(el("span", { class: "evchip", onclick: () => highlight(eid) }, eid));
    }
  }
  main.appendChild(keyEl);
  main.appendChild(evidenceBadges(resp, cand));

  const ringWrap = el("div", { class: "ringwrap" });
  ringWrap.appendChild(ring(d ? d.confidence : "medium", true));
  ringWrap.appendChild(el("span", { class: "ring-cap" }, `확신도 ${CONF_KO[d ? d.confidence : "medium"] || "보통"}`));
  top.appendChild(main);
  top.appendChild(ringWrap);
  ans.appendChild(top);

  const judgeChip = (resp.progress && resp.progress.find((p) => p.stage === "judge")?.features?.[0]) || "Gemini 4 Argon 판관";
  const details = el("div", { class: "a-details" },
    el("span", {}, `${TYPE_KO[cand.type] || "건물"} · ${cand.levels ? cand.levels + "층 · " : ""}OSM ${cand.osm_id || ""} · 판정 ${judgeChip.replace(" 판관", "")}`)
  );

  // Show candidate rank change if previous answered decision shifted ranks
  const ranks = rankMap(resp.candidates);
  if (S.prevRanks && S.prevRanks[cand.id] && S.prevRanks[cand.id] !== ranks[cand.id]) {
    details.appendChild(el("span", { class: "linkish", style: { marginLeft: "8px" } }, `답 반영 · ${S.prevRanks[cand.id]}위 → ${ranks[cand.id]}위`));
  }
  S.prevRanks = ranks;

  const more = (d && d.reasons && d.reasons.slice(1)) || [];
  const reasonsList = el("ul", { class: "a-reasons", hidden: true });
  if (more.length > 0) {
    const toggleBtn = el("button", {
      class: "linkish",
      type: "button",
      onclick: () => {
        reasonsList.hidden = !reasonsList.hidden;
      },
    }, `· 이유 ${more.length}개 더`);
    details.appendChild(toggleBtn);
    for (const r of more) {
      const li = el("li");
      li.innerHTML = emphasize(r.text_ko);
      for (const eid of r.evidence_ids || []) {
        li.appendChild(el("span", { class: "evchip", onclick: () => highlight(eid) }, eid));
      }
      reasonsList.appendChild(li);
    }
  }
  ans.appendChild(details);
  ans.appendChild(reasonsList);

  if (d && d.kind === "low_confidence" && d.alternatives && d.alternatives.length > 0) {
    ans.appendChild(el("div", { class: "q-note", style: { marginTop: "6px" } }, `확신이 낮아요 · 다른 후보: ${d.alternatives.join(", ")}`));
  }

  const nEv = (resp.evidence || []).length;
  const kakaoUrl = cand.centroid
    ? `https://map.kakao.com/link/roadview/${cand.centroid[1]},${cand.centroid[0]}`
    : "#";
  const actions = el("div", { class: "a-actions row" },
    el("button", { class: "ghost", type: "button", title: "음성으로 듣기", onclick: () => speak() }, svg("speaker"), el("span", {}, "듣기")),
    el("button", { class: "ghost", type: "button", title: "카카오톡에 붙여 넣을 글 복사", onclick: () => copyShare(resp, cand) }, svg("copy"), el("span", {}, "복사")),
    el("a", { class: "ghost", href: kakaoUrl, target: "_blank", rel: "noreferrer", title: "카카오 로드뷰로 지금 모습 보기(새 창)" }, svg("ext"), el("span", {}, "지금 모습")),
    el("button", { class: "ghost", type: "button", onclick: () => showPanel("evidence") }, `근거 ${nEv}건`)
  );
  ans.appendChild(actions);

  if (resp.baseline && resp.baseline.lnglat) {
    const dist = resp.baseline.distance_to_result_m || 0;
    const barW = Math.min(100, Math.max(4, 8 + dist / 6));
    const cmp = el("div", { class: "compare bars" },
      el("div", { class: "cmp-row" },
        el("b", { class: "us" }, "검증 있음"),
        el("span", { class: "cmp-bar" }, el("i", { class: "us", style: { width: "4%" } })),
        el("span", { class: "cmp-v" }, "여기")
      ),
      el("div", { class: "cmp-row" },
        el("b", { class: "gem" }, `검증 없는 ${resp.baseline.model || "Gemini"}`),
        el("span", { class: "cmp-bar" }, el("i", { class: "gem", style: { width: `${barW}%` } })),
        el("span", { class: "cmp-v" }, `${dist}m 차이`)
      ),
      el("div", { class: "cmp-note", title: `지도·사진·대장 없이 안내 글만 보고 추측한 곳: ${resp.baseline.address || "주소 없음"}` },
        `같은 Gemini가 지도·사진·대장 없이 안내 글만 보고 추측하면 우리 답에서 ${dist}m 떨어진 곳을 짚었어요${resp.baseline.address ? " (" + resp.baseline.address + ")" : ""}`
      )
    );
    ans.appendChild(cmp);
  }
  setAnswerHeight();
}

// Renders clarification prompt card
function renderQuestion(resp) {
  const ans = $("#answer");
  if (!ans) return;
  ans.className = "panel a-q";
  ans.innerHTML = "";
  const d = resp && resp.decision;
  const q = d && d.question;
  if (!q) return;

  S.questionSource = q.source;

  // Previous QA history notes
  if (resp.notes && resp.notes.length > 0) {
    const prevNotes = resp.notes.filter((n) => typeof n === "string" && n.startsWith("답 반영:"));
    if (prevNotes.length > 0) {
      const hist = el("div", { class: "q-hist" });
      for (const n of prevNotes) {
        hist.appendChild(el("div", { class: "q-prev" }, n));
      }
      ans.appendChild(hist);
    }
  }

  const qBox = el("div", { class: "a-q" });
  const head = el("div", { class: "q-head" },
    el("div", { class: "q-bubble" }, "?"),
    el("div", {},
      el("div", { class: "q-text" }, q.intent_ko || q.text_ko || "추가 정보를 알려주세요"),
      el("div", { class: "q-note" }, "확신이 낮을 때만 가장 많이 걸러 내는 질문 하나를 드려요.")
    )
  );
  qBox.appendChild(head);

  const opts = q.options_ko || q.options || [];
  if (opts.length > 0) {
    const optsBox = el("div", { class: "q-opts" });
    opts.forEach((opt, idx) => {
      optsBox.appendChild(el("button", {
        type: "button",
        onclick: () => sendAnswer({ option_index: idx }),
      }, opt));
    });
    qBox.appendChild(optsBox);
  }

  const inpRow = el("div", { class: "q-input" });
  const placeholder = q.source === "name" ? "예: 명륜진사갈비" : "예: 삼성동 / 잠실새내역 4번 출구";
  const inp = el("input", { placeholder });
  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && inp.value.trim()) {
      sendAnswer({ text: inp.value.trim() });
    }
  });
  const subBtn = el("button", {
    type: "button",
    onclick: () => {
      if (inp.value.trim()) sendAnswer({ text: inp.value.trim() });
    },
  }, "답하기");
  inpRow.appendChild(inp);
  inpRow.appendChild(subBtn);
  qBox.appendChild(inpRow);

  const actionsRow = el("div", { style: { display: "flex", gap: "8px", flexWrap: "wrap", marginTop: "4px" } });
  const unknownBtn = el("button", {
    class: "ghost small",
    type: "button",
    title: "모르면 지금 후보로 최선의 답을 보여줘요",
    onclick: () => sendAnswer({ text: "모르겠어요" }),
  }, "모르겠어요");
  actionsRow.appendChild(unknownBtn);

  if (q.source === "name" && !S.photo) {
    const photoBtn = el("button", {
      class: "ghost small q-photo",
      type: "button",
      onclick: () => {
        const pf = $("#photoFile");
        if (pf) pf.click();
      },
    }, "가게 사진이 있으면 간판을 읽어 찾아볼게요");
    actionsRow.appendChild(photoBtn);
  }

  const speakBtn = el("button", {
    class: "ghost small",
    type: "button",
    onclick: () => speak(q.intent_ko || q.text_ko),
  }, "질문 듣기");
  actionsRow.appendChild(speakBtn);

  qBox.appendChild(actionsRow);
  ans.appendChild(qBox);
  setAnswerHeight();
}

// Shows temporary notification message
function toast(msg) {
  let t = $("#toastBox");
  if (!t) {
    t = el("div", {
      id: "toastBox",
      style: {
        position: "fixed",
        bottom: "80px",
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(15,23,42,.85)",
        color: "#fff",
        padding: "8px 16px",
        borderRadius: "999px",
        fontSize: "13px",
        fontWeight: "700",
        zIndex: "99",
      },
    });
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => {
    t.hidden = true;
  }, 2200);
}

// Copies shareable summary to clipboard
function copyShare(resp, cand) {
  const name = cand.name || "목적지";
  const orig = (resp.input && resp.input.text) || "";
  const [lng, lat] = cand.centroid || VENUE;
  const kakaoUrl = `https://map.kakao.com/link/map/${encodeURIComponent(name)},${lat},${lng}`;
  const osmUrl = `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lng}#map=19/${lat}/${lng}`;
  const txt = `${name}\n설명: ${orig}\n\n카카오맵: ${kakaoUrl}\nOSM: ${osmUrl}`;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(txt).then(() => {
      toast("복사했어요 — 카카오톡에 붙여 넣으세요");
    }).catch((e) => {
      toast(`복사 실패: ${e.message}`);
    });
  } else {
    toast("clipboard 없음");
  }
}

// Reads aloud verbal decision summary
function speak(customText) {
  if (S.mock) {
    toast("모의 화면에서는 음성을 만들지 않아요.");
    return;
  }
  const text = customText || (S.resp && S.resp.tts_text) || (S.resp && S.resp.decision && S.resp.decision.reasons?.[0]?.text_ko);
  if (!text) return;
  const url = `/api/tts?text=${encodeURIComponent(text)}`;
  const audio = new Audio(url);
  audio.play().catch((e) => {
    toast(`재생 실패: ${e.message}`);
  });
}

// Coordinates smooth arrival sequence to verdict
function reveal(resp) {
  let done = false;
  const safetyTimer = setTimeout(() => {
    if (!done) {
      done = true;
      drawDecision(resp);
      document.body.classList.remove("show-timeline", "show-evidence");
      if (resp.decision && resp.decision.kind === "question") {
        setState("asking");
        renderQuestion(resp);
      } else {
        setState("answered");
        renderAnswer(resp);
      }
      renderSafeguards(resp.safeguards);
    }
  }, 5000);

  const cands = resp.candidates || [];
  const top = cands.find((c) => c.rank === 1) || cands[0];
  const finalId = resp.decision && resp.decision.candidate_id;

  (S.walk || Promise.resolve())
    .then(async () => {
      if (done) return;
      // Auto-preview up to 2 street view groups if tab is visible
      if (!document.hidden && resp.decision && resp.decision.kind !== "question") {
        const pGroups = photoGroups(resp.evidence || []);
        for (const g of pGroups.slice(0, 2)) {
          await openViewer(g, { ms: 2800 });
        }
      }
    })
    .then(() => {
      clearTimeout(safetyTimer);
      if (done) return;
      done = true;
      drawDecision(resp);
      document.body.classList.remove("show-timeline", "show-evidence");
      if (resp.decision && resp.decision.kind === "question") {
        setState("asking");
        renderQuestion(resp);
      } else {
        setState("answered");
        renderAnswer(resp);
      }
      renderSafeguards(resp.safeguards);
    });
}

// Updates map elements according to search result
function drawMap(resp) {
  if (!resp) return;
  if (resp.window && !S.drawn.has("window")) {
    S.drawn.add("window");
    drawWindow(resp.window);
  }
  if (resp.anchor && resp.anchor.lnglat && !S.drawn.has("anchor")) {
    S.drawn.add("anchor");
    marker(resp.anchor.lnglat, "anchor", "pin", resp.anchor.name || "기준점");
    if (!resp.candidates || resp.candidates.length === 0) {
      map.flyTo({ center: resp.anchor.lnglat, zoom: 16, duration: 900 });
    }
  }
  if (resp.candidates && resp.candidates.length > 0 && !S.drawn.has("paths")) {
    S.drawn.add("paths");
    S.walk = animatePaths(resp.candidates);
  }
  if (resp.candidates && resp.candidates.length > 0 && !S.drawn.has("cands")) {
    S.drawn.add("cands");
    (S.walk || Promise.resolve()).then(() => drawCands(resp.candidates, resp));
  }
  if (resp.evidence && resp.evidence.length > 0) {
    drawLandmarks(resp.evidence);
  }
  if (resp.places && resp.places.length > 0) {
    const placesSrc = map && map.getSource("places");
    if (placesSrc) {
      const feats = resp.places.filter((p) => p.lnglat).map((p) => ({
        type: "Feature",
        properties: { name: p.name || "", category: p.category || "" },
        geometry: { type: "Point", coordinates: p.lnglat },
      }));
      placesSrc.setData(FC(feats));
    }
    const pAttr = $("#placesAttr");
    if (pAttr) {
      pAttr.hidden = false;
      pAttr.textContent = `장소 자료: 카카오 ${resp.places.length}곳(이번 검색)`;
    }
  }
  if (resp.decision && !S.drawn.has("decision")) {
    S.drawn.add("decision");
    reveal(resp);
  }
}

// Updates signboard detection readout
function renderSignboard(resp) {
  const sb = resp && (resp.signboard || (resp.input && resp.input.signboard));
  if (!sb) return;
  const txt = $("#photoText");
  const names = Array.isArray(sb.names) ? sb.names.join(", ") : (sb.names || "");
  if (!S._sbUpdated && txt) {
    if (names) {
      txt.textContent = `간판: ${names} → 지도 검색`;
      S._sbUpdated = true;
    } else {
      txt.textContent = "간판을 읽지 못했어요 — 닮음 대조만";
      S._sbUpdated = true;
    }
  }
  if (!S._sbToasted) {
    if (names) {
      toast(`간판: ${names}${sb.applied ? " → 지도 검색" : ""}`);
      S._sbToasted = true;
    } else if (sb.attempted || sb.failed || names === "") {
      toast("간판을 읽지 못했어요 — 닮음 대조만");
      S._sbToasted = true;
    }
  }
}

// Formulates single-line milestone for stage badge
function discovery(resp) {
  if (!resp) return "";
  if (resp.signboard && resp.signboard.applied) {
    return "간판으로 기준 가게 확인";
  }
  if (resp.candidates && resp.candidates.length > 0) {
    return `후보 ${resp.candidates.length}곳`;
  }
  if (resp.anchor && resp.anchor.name) {
    return `출발점 ${resp.anchor.name}`;
  }
  return "";
}

// Keeps status badge on a single line by adjusting typography
function fitBadge(b, stage) {
  if (!b) return;
  b.style.fontSize = "13px";
  const modelChip = b.querySelector(".m");
  if (modelChip) modelChip.textContent = STAGE_WHO[stage] || "";
  const maxW = window.innerWidth - 420;
  if (b.offsetWidth > maxW) {
    b.style.fontSize = "11.5px";
  }
  if (b.offsetWidth > maxW && modelChip) {
    modelChip.textContent = STAGE_WHO_SHORT[stage] || "";
  }
  if (b.offsetWidth > maxW) {
    b.style.fontSize = "10.5px";
  }
}

// Updates top floating stage indicator badge
function renderBadge(resp) {
  const b = $("#stageBadge");
  if (!b) return;
  if (!resp || !resp.progress || resp.decision || resp.done) {
    b.hidden = true;
    return;
  }
  b.hidden = false;
  b.innerHTML = "";
  const prog = resp.progress || [];
  const cur = prog.find((p) => p.status === "running") || prog[prog.length - 1];
  const stg = cur ? cur.stage : "parse";
  const who = STAGE_WHO[stg] || "Gemini 4 Argon";

  b.appendChild(el("div", { class: "spinner" }));
  b.appendChild(el("span", { class: "m" }, who));

  const disc = discovery(resp);
  const sub = disc || (cur && cur.sub) || STAGE_HINT[stg] || "진행 중";
  b.appendChild(el("span", {}, `${STAGE_KO[stg]} · ${sub}`));
  fitBadge(b, stg);
}

// Central render dispatcher for pipeline state updates
function render(resp) {
  if (!resp) return;

  // Filter out duplicate decision during active re-query
  if (S.answeredDecision && resp.decision && JSON.stringify(resp.decision) === S.answeredDecision) {
    resp = { ...resp, decision: null, done: false };
  } else if (resp.decision) {
    S.answeredDecision = null;
  }

  S.resp = resp;
  renderSpans(resp.input);
  renderBadge(resp);
  renderTimeline(resp);
  renderEvidence(resp.evidence);
  renderSignboard(resp);

  if (resp.error) {
    showError(`처리 실패: ${resp.error}`);
    setBusy(false);
    return;
  }

  if (window.map) {
    drawMap(resp);
  }

  if (!resp.decision) {
    setState("running");
    renderRunning(resp);
  }
}

// Restores UI to clean idle state
function resetAll() {
  stopPolling();
  clearMap();
  S.sid = null;
  S.resp = null;
  S.drawn.clear();
  S._sbUpdated = false;
  S._sbToasted = false;
  setState("idle");
  renderSpans(null);
  renderTimeline(null);
  renderEvidence(null);
  const badge = $("#stageBadge");
  if (badge) badge.hidden = true;
  showError(null);
  renderIdle();
  if (window.map) {
    map.easeTo({ center: VENUE, zoom: 16, pitch: 0, bearing: 0, duration: 600 });
  }
}

// Updates controls and body classes during execution
function setBusy(on) {
  const btn = $("#run");
  if (btn) {
    btn.disabled = on;
    btn.innerHTML = on
      ? '<div class="spinner" style="border-top-color:#fff;border-color:rgba(255,255,255,.3)"></div><span>찾는 중…</span>'
      : "<span>찾기</span>";
  }
  if (on) {
    document.body.classList.add("ran");
  }
}

// Polls session status endpoint
function poll() {
  if (!S.sid) return;
  const url = S.sid === "mock" ? "mock_locate.json" : `/api/session/${S.sid}`;
  fetch(url, { cache: "no-store" })
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((data) => {
      render(data);
      if (data.done) {
        stopPolling();
        setBusy(false);
        if (data.decision) {
          S.lastSid = S.sid;
          const replay = $("#replayBtn");
          if (replay) replay.hidden = false;
        }
        if (S.replaying) {
          S.replaying = false;
        }
      }
    })
    .catch((e) => {
      showError(`조회 실패: ${e.message}`);
      stopPolling();
      setBusy(false);
      S.replaying = false;
    });
}

// Begins periodic polling timer
function startPolling() {
  stopPolling();
  poll();
  S.timer = setInterval(poll, 700);
}

// Cancels active polling timer
function stopPolling() {
  if (S.timer) {
    clearInterval(S.timer);
    S.timer = null;
  }
}

// Performs JSON fetch POST request with error trapping
function post(url, init = {}) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
    ...init,
  }).then(async (r) => {
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    return r.json();
  });
}

// Dispatches text search query to backend
function runText(text) {
  const query = text || ($("#text") && $("#text").value.trim());
  if (!query && !S.photo) {
    showError("길 설명을 적거나 사진을 넣어 주세요.");
    return;
  }
  if (!query && S.photo) {
    S.photoRole = "target";
    const roleBox = $("#photoRole");
    if (roleBox) {
      roleBox.querySelectorAll("button").forEach((b) => {
        if (b.dataset.role === "target") b.classList.add("on");
        else b.classList.remove("on");
      });
    }
  }
  resetAll();
  setBusy(true);
  setState("running");
  renderRunning({ progress: [{ stage: "parse", status: "running" }] });

  const regionVal = $("#region") ? $("#region").value.trim() : null;

  // Use multipart endpoint when a photo is attached
  if (S.photo) {
    const fd = new FormData();
    fd.append("text", query || "");
    if (regionVal) fd.append("region", regionVal);
    if (S.here) {
      fd.append("lat", String(S.here[1]));
      fd.append("lng", String(S.here[0]));
    }
    fd.append("photo", S.photo);
    fd.append("photo_role", (!query && S.photo) ? "target" : S.photoRole);

    fetch("/api/locate", { method: "POST", body: fd })
      .then(async (r) => {
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${r.status}`);
        }
        return r.json();
      })
      .then((resp) => {
        S.sid = resp.session_id;
        startPolling();
      })
      .catch((e) => {
        showError(`요청 실패: ${e.message}`);
        setBusy(false);
        setState("idle");
      });
    return;
  }

  // Use JSON text query endpoint
  const body = {
    text: query,
    region: regionVal,
    lat: S.here ? S.here[1] : null,
    lng: S.here ? S.here[0] : null,
  };

  post("/api/locate_text", { body: JSON.stringify(body) })
    .then((resp) => {
      S.sid = resp.session_id;
      startPolling();
    })
    .catch((e) => {
      showError(`요청 실패: ${e.message}`);
      setBusy(false);
      setState("idle");
    });
}

// Uploads recorded audio sample for voice query
function sendAudio(blob) {
  resetAll();
  setBusy(true);
  setState("running");
  const form = new FormData();
  form.append("audio", blob, "voice.webm");
  const regionVal = $("#region") ? $("#region").value.trim() : "";
  if (regionVal) form.append("region", regionVal);
  if (S.here) {
    form.append("lat", String(S.here[1]));
    form.append("lng", String(S.here[0]));
  }
  if (S.photo) {
    form.append("photo", S.photo);
    form.append("photo_role", S.photoRole);
  }

  fetch("/api/locate", { method: "POST", body: form })
    .then(async (r) => {
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${r.status}`);
      }
      return r.json();
    })
    .then((resp) => {
      S.sid = resp.session_id;
      startPolling();
    })
    .catch((e) => {
      showError(`녹음 보내기 실패: ${e.message}`);
      setBusy(false);
      setState("idle");
    });
}

// Submits clarification response to refine ambiguity
function sendAnswer(body) {
  if (S.mock) {
    toast("모의 화면에서는 답을 보낼 수 없어요.");
    return;
  }
  if (S.resp && S.resp.decision) {
    S.answeredDecision = JSON.stringify(S.resp.decision);
  }

  const src = S.questionSource || (S.resp && S.resp.decision && S.resp.decision.question && S.resp.decision.question.source);
  const restartStages = ["region", "station", "anchor", "name"];

  if (restartStages.includes(src)) {
    const curSid = S.sid;
    resetAll();
    S.sid = curSid;
    setState("running");
    renderRunning({ progress: [{ stage: "parse", status: "running", sub: "답 반영 중" }] });
  } else {
    S.drawn.delete("decision");
    clearDecisionDrawing();
    setState("running");
    renderRunning({ progress: [{ stage: "judge", status: "running", sub: "답 반영 중" }] });
  }

  post("/api/answer", { body: JSON.stringify({ session_id: S.sid, ...body }) })
    .then((resp) => {
      if (resp.session_id) S.sid = resp.session_id;
      startPolling();
    })
    .catch((e) => {
      showError(`답 보내기 실패: ${e.message}`);
    });
}

// Starts audio level visualizer
function meterStart(stream, opts = {}) {
  meterStop();
  const box = $("#liveMeter");
  const stateLbl = $("#liveState");
  if (box) box.hidden = false;

  METER.speaking = typeof opts.speaking === "function" ? opts.speaking : () => false;
  METER.label = typeof opts.label === "function" ? opts.label : () => (opts.label || "듣는 중");
  METER.levels = new Array(METER_BARS).fill(0);

  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    METER.ctx = new AudioCtx();
    METER.an = METER.ctx.createAnalyser();
    METER.an.fftSize = 1024;
    const src = METER.ctx.createMediaStreamSource(stream);
    src.connect(METER.an);

    const bars = box ? box.querySelectorAll(".lm-bars i") : [];
    const buf = new Float32Array(METER.an.fftSize);

    function tick() {
      if (!METER.an) return;
      let rms = 0;
      if (typeof METER.an.getFloatTimeDomainData === "function") {
        METER.an.getFloatTimeDomainData(buf);
        let sumSq = 0;
        for (let i = 0; i < buf.length; i++) {
          sumSq += buf[i] * buf[i];
        }
        rms = Math.sqrt(sumSq / buf.length);
      } else {
        const byteBuf = new Uint8Array(METER.an.fftSize);
        METER.an.getByteTimeDomainData(byteBuf);
        let sumSq = 0;
        for (let i = 0; i < byteBuf.length; i++) {
          const val = (byteBuf[i] - 128) / 128;
          sumSq += val * val;
        }
        rms = Math.sqrt(sumSq / byteBuf.length);
      }

      const level = Math.min(Math.max(rms * 7, 0), 1);
      METER.levels.push(level);
      if (METER.levels.length > METER_BARS) {
        METER.levels.shift();
      }

      for (let i = 0; i < bars.length && i < METER.levels.length; i++) {
        const v = METER.levels[i];
        const h = 4 + v * (22 - 4);
        bars[i].style.height = `${Math.round(h)}px`;
      }

      const isSpeaking = METER.speaking();
      const baseLabel = METER.label();
      if (isSpeaking) {
        if (box) box.classList.add("speaking");
        if (stateLbl) stateLbl.textContent = "Gemini 말하는 중";
      } else {
        if (box) box.classList.remove("speaking");
        if (level > 0.12) {
          if (stateLbl) stateLbl.textContent = `${baseLabel} — 말소리 감지`;
        } else {
          if (stateLbl) stateLbl.textContent = baseLabel;
        }
      }
    }

    tick();
    METER.timer = setInterval(tick, 100);
  } catch (_) {}
}
window.meterStart = meterStart;

// Stops audio level visualizer
function meterStop() {
  if (METER.timer) {
    clearInterval(METER.timer);
    METER.timer = null;
  }
  if (METER.raf) {
    cancelAnimationFrame(METER.raf);
    METER.raf = null;
  }
  if (METER.ctx) {
    try {
      METER.ctx.close();
    } catch (_) {}
    METER.ctx = null;
  }
  METER.an = null;
  METER.levels = [];
  const box = $("#liveMeter");
  if (box) {
    box.hidden = true;
    box.classList.remove("speaking");
  }
}
window.meterStop = meterStop;

// Toggles voice recording on audio input button
function toggleRecord() {
  const btn = $("#rec");
  if (S.recorder && S.recorder.state === "recording") {
    S.recorder.stop();
    meterStop();
    if (btn) btn.innerHTML = "<span>녹음</span>";
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("녹음을 지원하지 않는 브라우저입니다.");
    return;
  }
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then((stream) => {
      meterStart(stream, { label: () => "녹음 중" });
      S.chunks = [];
      S.recorder = new MediaRecorder(stream);
      S.recorder.ondataavailable = (e) => S.chunks.push(e.data);
      S.recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(S.chunks, { type: "audio/webm" });
        sendAudio(blob);
      };
      S.recorder.start();
      if (btn) btn.innerHTML = "<span>멈추고 보내기</span>";
    })
    .catch((e) => {
      toast(`녹음할 수 없어요: ${e.message}`);
    });
}

// Simulates step-by-step playback using mock data
function runMock() {
  S.mock = true;
  const mockB = $("#mockBadge");
  if (mockB) mockB.hidden = false;
  fetch("mock_locate.json")
    .then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then((full) => {
      let step = 0;
      function next() {
        step++;
        const progSlice = (full.progress || []).slice(0, step).map((p, idx) => ({
          ...p,
          status: idx === step - 1 && step < 4 ? "running" : "done",
        }));
        const partial = {
          ...full,
          progress: progSlice,
          done: step >= 4,
          decision: step >= 4 ? full.decision : null,
          candidates: step >= 2 ? full.candidates : [],
          evidence: step >= 3 ? full.evidence : [],
          places: step >= 2 ? full.places : [],
        };
        render(partial);
        if (step >= 4) {
          S.lastSid = "mock";
          const replay = $("#replayBtn");
          if (replay) replay.hidden = false;
        }
        if (step < 4) {
          setTimeout(next, 800);
        }
      }
      next();
    })
    .catch((e) => {
      showError(`mock_locate.json 읽기 실패: ${e.message}`);
    });
}

// Queries model warmup status
function pollWarm() {
  const b = $("#warmBadge");
  fetch("/api/warmup")
    .then((r) => r.json())
    .then((d) => {
      if (!b) return;
      b.hidden = false;
      const s = d.seconds ? d.seconds.toFixed(1) : "4.9";
      b.textContent = `모델 예열 완료 · ${s}초 · ${d.backend || "Vertex AI"}`;
      if (d.state === "running" || d.state === "warming") {
        setTimeout(pollWarm, 2000);
      }
    })
    .catch(() => {});
}

// Loads examples and renders chip buttons
function loadChips() {
  fetch("examples.json", { cache: "no-store" })
    .then((r) => r.json())
    .then((list) => {
      S.examples = list;
      const box = $("#chips");
      if (box) {
        box.innerHTML = "";
        for (const ex of list) {
          const btn = el("button", {
            class: "chip",
            type: "button",
            onclick: () => runExample(ex),
          }, ex.label || ex.text.slice(0, 10));
          box.appendChild(btn);
        }
      }
      renderIdle();
    })
    .catch(() => {
      renderIdle();
    });
}

// Displays model evaluation benchmarks modal
function openEval() {
  const p = $("#evalPanel");
  const b = $("#evalBody");
  if (!p || !b) return;
  p.hidden = false;
  b.textContent = "불러오는 중…";
  fetch("/api/eval")
    .then((r) => r.json())
    .then((d) => {
      b.textContent = JSON.stringify(d, null, 2);
    })
    .catch((e) => {
      b.textContent = `불러오기 실패: ${e.message}`;
    });
}

// MapLibre map instance initialization
if (typeof maplibregl !== "undefined") {
  window.map = new maplibregl.Map({
    container: "map",
    style: "https://tiles.openfreemap.org/styles/positron",
    center: VENUE,
    zoom: 16,
    attributionControl: false,
  });

  // Map controls
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

  // Map load callback setting up styles, layers, and sources
  map.on("load", () => {
    // 1. Korean label preference
    const styleLayers = map.getStyle().layers || [];
    styleLayers.forEach((layer) => {
      if (layer.type === "symbol" && layer.layout && layer.layout["text-field"]) {
        map.setLayoutProperty(layer.id, "text-field", [
          "coalesce",
          ["get", "name:ko"],
          ["get", "name"],
          ["get", "name:latin"],
        ]);
      }
    });

    // 2. 3D building extrusion
    let labelLayerId;
    for (let i = 0; i < styleLayers.length; i++) {
      if (styleLayers[i].type === "symbol" && styleLayers[i].layout && styleLayers[i].layout["text-field"]) {
        labelLayerId = styleLayers[i].id;
        break;
      }
    }
    if (!map.getLayer("bld-3d") && map.getSource("openmaptiles")) {
      map.addLayer(
        {
          id: "bld-3d",
          source: "openmaptiles",
          "source-layer": "building",
          type: "fill-extrusion",
          minzoom: 14,
          paint: {
            "fill-extrusion-color": "#e3e8f0",
            "fill-extrusion-height": [
              "interpolate",
              ["linear"],
              ["zoom"],
              14,
              0,
              14.05,
              ["coalesce", ["get", "render_height"], 0],
            ],
            "fill-extrusion-base": [
              "interpolate",
              ["linear"],
              ["zoom"],
              14,
              0,
              14.05,
              ["coalesce", ["get", "render_min_height"], 0],
            ],
            "fill-extrusion-opacity": 0.72,
          },
        },
        labelLayerId
      );
    }

    // 3. GeoJSON sources
    const sources = ["win", "paths", "cands", "final", "baseline", "cones", "places"];
    for (const src of sources) {
      if (!map.getSource(src)) {
        map.addSource(src, { type: "geojson", data: EMPTY });
      }
    }

    // 4. Map layers
    if (!map.getLayer("win-fill")) {
      map.addLayer({
        id: "win-fill",
        type: "fill",
        source: "win",
        paint: { "fill-color": "#2563eb", "fill-opacity": 0.08 },
      });
    }
    if (!map.getLayer("win-line")) {
      map.addLayer({
        id: "win-line",
        type: "line",
        source: "win",
        paint: { "line-color": "#2563eb", "line-width": 1.5, "line-dasharray": [3, 2] },
      });
    }
    if (!map.getLayer("cones-fill")) {
      map.addLayer({
        id: "cones-fill",
        type: "fill",
        source: "cones",
        paint: { "fill-color": "#0f172a", "fill-opacity": 0.14 },
      });
    }
    if (!map.getLayer("cones-line")) {
      map.addLayer({
        id: "cones-line",
        type: "line",
        source: "cones",
        paint: { "line-color": "#0f172a", "line-width": 1, "line-opacity": 0.3 },
      });
    }
    if (!map.getLayer("paths-glow")) {
      map.addLayer({
        id: "paths-glow",
        type: "line",
        source: "paths",
        paint: { "line-color": "#93c5fd", "line-width": 12, "line-blur": 6, "line-opacity": 0.6 },
      });
    }
    if (!map.getLayer("paths")) {
      map.addLayer({
        id: "paths",
        type: "line",
        source: "paths",
        layout: { "line-cap": "round", "line-join": "round" },
        paint: { "line-color": "#2563eb", "line-width": 5 },
      });
    }
    if (!map.getLayer("baseline-line")) {
      map.addLayer({
        id: "baseline-line",
        type: "line",
        source: "baseline",
        paint: { "line-color": "#64748b", "line-width": 2.5, "line-dasharray": [2, 2] },
      });
    }
    if (!map.getLayer("cands-3d")) {
      map.addLayer({
        id: "cands-3d",
        type: "fill-extrusion",
        source: "cands",
        paint: {
          "fill-extrusion-height": ["coalesce", ["get", "h"], 15],
          "fill-extrusion-color": ["coalesce", ["get", "color"], "#2563eb"],
          "fill-extrusion-opacity": ["coalesce", ["get", "opacity"], 0.85],
        },
      });
    }
    if (!map.getLayer("cands-line")) {
      map.addLayer({
        id: "cands-line",
        type: "line",
        source: "cands",
        paint: { "line-color": ["coalesce", ["get", "color"], "#2563eb"], "line-width": 2 },
      });
    }
    if (!map.getLayer("cands-pt")) {
      map.addLayer({
        id: "cands-pt",
        type: "circle",
        source: "cands",
        paint: { "circle-radius": 5, "circle-color": ["coalesce", ["get", "color"], "#2563eb"] },
      });
    }
    if (!map.getLayer("final-pt")) {
      map.addLayer({
        id: "final-pt",
        type: "circle",
        source: "final",
        paint: {
          "circle-radius": 8,
          "circle-color": "#f97316",
          "circle-stroke-width": 3,
          "circle-stroke-color": "#ffffff",
        },
      });
    }

    // 5. OSM POI layers
    let symbolFont = ["Noto Sans Regular"];
    for (const layer of styleLayers) {
      if (layer.type === "symbol" && layer.layout && layer.layout["text-font"]) {
        symbolFont = layer.layout["text-font"];
        break;
      }
    }
    const poiFilter = [
      "!in",
      "class",
      "bus",
      "atm",
      "drinking_water",
      "playground",
      "cemetery",
      "toilets",
    ];
    if (!map.getLayer("poi-dot") && map.getSource("openmaptiles")) {
      map.addLayer(
        {
          id: "poi-dot",
          type: "circle",
          source: "openmaptiles",
          "source-layer": "poi",
          minzoom: 15.5,
          filter: poiFilter,
          paint: {
            "circle-radius": 3,
            "circle-color": "#94a3b8",
            "circle-opacity": 0.7,
          },
        },
        labelLayerId
      );
    }
    if (!map.getLayer("poi-label") && map.getSource("openmaptiles")) {
      map.addLayer({
        id: "poi-label",
        type: "symbol",
        source: "openmaptiles",
        "source-layer": "poi",
        minzoom: 16.4,
        filter: poiFilter,
        layout: {
          "text-field": [
            "coalesce",
            ["get", "name:ko"],
            ["get", "name"],
            ["get", "name:latin"],
          ],
          "text-font": symbolFont,
          "text-size": 11,
          "text-anchor": "top",
          "text-offset": [0, 0.4],
          "text-max-width": 8,
          "text-optional": true,
        },
        paint: {
          "text-color": "#475569",
          "text-halo-color": "#ffffff",
          "text-halo-width": 1.2,
        },
      });
    }

    // 6. Kakao places layers
    if (!map.getLayer("places-dot")) {
      map.addLayer({
        id: "places-dot",
        type: "circle",
        source: "places",
        minzoom: 15,
        paint: {
          "circle-radius": 4,
          "circle-color": "#38bdf8",
          "circle-stroke-width": 1.5,
          "circle-stroke-color": "#ffffff",
        },
      });
    }
    if (!map.getLayer("places-label")) {
      map.addLayer({
        id: "places-label",
        type: "symbol",
        source: "places",
        minzoom: 16.6,
        layout: {
          "text-field": ["get", "name"],
          "text-font": symbolFont,
          "text-size": 11,
          "text-anchor": "top",
          "text-offset": [0, 0.5],
          "text-max-width": 8,
        },
        paint: {
          "text-color": "#0284c7",
          "text-halo-color": "#ffffff",
          "text-halo-width": 1.5,
        },
      });
    }
  });
}

// Wire DOM event handlers and initialize state
document.addEventListener("DOMContentLoaded", () => {
  // Initialize meter bars
  const lmBars = $(".lm-bars");
  if (lmBars && lmBars.children.length === 0) {
    for (let i = 0; i < METER_BARS; i++) {
      lmBars.appendChild(el("i"));
    }
  }

  // Register interactive click handlers
  const warmBadge = $("#warmBadge");
  if (warmBadge) {
    warmBadge.addEventListener("click", () => {
      post("/api/warmup").then(() => pollWarm()).catch(() => pollWarm());
    });
  }

  const runBtn = $("#run");
  if (runBtn) runBtn.addEventListener("click", () => runText());

  const textArea = $("#text");
  if (textArea) {
    textArea.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        runText();
      }
    });
    textArea.addEventListener("focus", () => {
      document.body.classList.add("editing");
    });
  }

  const spokenBox = $("#spoken");
  if (spokenBox) {
    spokenBox.addEventListener("click", () => {
      spokenBox.hidden = true;
      if (textArea) {
        textArea.hidden = false;
        textArea.focus();
      }
      document.body.classList.add("editing");
    });
  }

  const recBtn = $("#rec");
  if (recBtn) recBtn.addEventListener("click", toggleRecord);

  const hereBtn = $("#here");
  if (hereBtn) {
    hereBtn.addEventListener("click", () => {
      if (!navigator.geolocation) {
        toast("위치 정보를 지원하지 않는 브라우저입니다.");
        return;
      }
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          S.here = [pos.coords.longitude, pos.coords.latitude];
          const hereLbl = $("#hereLabel");
          if (hereLbl) {
            hereLbl.hidden = false;
            hereLbl.textContent = "현재 위치 설정됨";
          }
          toast("현재 위치를 가져왔어요");
        },
        (err) => toast(`위치 확인 실패: ${err.message}`)
      );
    });
  }

  const photoBtn = $("#photoBtn");
  const photoFile = $("#photoFile");
  if (photoBtn && photoFile) {
    photoBtn.addEventListener("click", () => photoFile.click());
    photoFile.addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      S.photo = file;
      S.photoUrl = URL.createObjectURL(file);
      const chip = $("#photoChip");
      const thumb = $("#photoThumb");
      const txt = $("#photoText");
      if (chip) chip.hidden = false;
      if (thumb) thumb.src = S.photoUrl;
      if (txt) txt.textContent = "사진 — 간판을 읽어 지도에서 찾고, 최신 모습으로 대조";
      S._sbUpdated = false;
      S._sbToasted = false;
    });
  }

  const photoClear = $("#photoClear");
  if (photoClear) {
    photoClear.addEventListener("click", () => {
      S.photo = null;
      if (S.photoUrl) {
        URL.revokeObjectURL(S.photoUrl);
        S.photoUrl = null;
      }
      if (photoFile) photoFile.value = "";
      const chip = $("#photoChip");
      const txt = $("#photoText");
      if (chip) chip.hidden = true;
      if (txt) txt.textContent = "사진 — 간판을 읽어 지도에서 찾고, 최신 모습으로 대조";
      S._sbUpdated = false;
      S._sbToasted = false;
    });
  }

  const photoRoleBox = $("#photoRole");
  if (photoRoleBox) {
    photoRoleBox.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (!btn) return;
      photoRoleBox.querySelectorAll("button").forEach((b) => b.classList.remove("on"));
      btn.classList.add("on");
      S.photoRole = btn.dataset.role || "target";
    });
  }

  const baselineToggle = $("#baselineToggle");
  if (baselineToggle) {
    baselineToggle.addEventListener("change", setBaselineVisible);
  }

  const evalBtn = $("#evalBtn");
  if (evalBtn) evalBtn.addEventListener("click", openEval);

  const evalClose = $("#evalClose");
  if (evalClose) {
    evalClose.addEventListener("click", () => {
      const p = $("#evalPanel");
      if (p) p.hidden = true;
    });
  }

  const replayBtn = $("#replayBtn");
  if (replayBtn) {
    replayBtn.addEventListener("click", () => {
      const lastSid = S.lastSid;
      if (!lastSid) return;
      resetAll();
      S.sid = lastSid;
      S.replaying = true;
      if (replayBtn) replayBtn.hidden = false;
      toast("마지막 결과를 다시 재생합니다 — API 호출 없음");
      startPolling();
    });
  }

  const tabTimeline = $("#tabTimeline");
  if (tabTimeline) tabTimeline.addEventListener("click", () => showPanel("timeline"));

  const tabEvidence = $("#tabEvidence");
  if (tabEvidence) tabEvidence.addEventListener("click", () => showPanel("evidence"));

  const viewerClose = $("#viewerClose");
  if (viewerClose) viewerClose.addEventListener("click", closeViewer);

  const viewer = $("#viewer");
  if (viewer) {
    viewer.addEventListener("click", (e) => {
      if (e.target === viewer) closeViewer();
    });
  }

  const tlTitle = $("#tlTitle");
  if (tlTitle) {
    tlTitle.addEventListener("click", () => {
      const p = $("#timelinePanel");
      if (p) p.classList.toggle("collapsed");
    });
  }

  const evTitle = $("#evTitle");
  if (evTitle) {
    evTitle.addEventListener("click", () => {
      const col = $("#rightCol");
      if (col) col.classList.toggle("collapsed");
    });
  }

  const chipsToggle = $("#chipsToggle");
  if (chipsToggle) {
    chipsToggle.addEventListener("click", () => {
      const chips = $("#chips");
      if (!chips) return;
      chips.classList.toggle("open");
      chipsToggle.textContent = chips.classList.contains("open") ? "예시 ▴" : "예시 ▾";
    });
  }

  // Observe answer panel resize to dynamically sync variable
  if (window.ResizeObserver) {
    const ansEl = $("#answer");
    if (ansEl) {
      new ResizeObserver(() => {
        setAnswerHeight();
        const strip = document.querySelector(".how.open");
        if (strip) fitStrip(strip);
      }).observe(ansEl);
    }
  }

  window.addEventListener("resize", () => {
    const strip = document.querySelector(".how.open");
    if (strip) fitStrip(strip);
    setAnswerHeight();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeViewer();
      const p = $("#evalPanel");
      if (p) p.hidden = true;
    }
  });

  // Startup invocation sequence
  setState("idle");
  renderTimeline(null);
  renderIdle();
  loadChips();
  pollWarm();

  if (new URLSearchParams(window.location.search).has("mock")) {
    runMock();
  }
});

window.S = S;
window.startPolling = startPolling;
window.showError = showError;

