"""Shared data contracts and JSON schemas across parse, search, verify, and judge stages."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LatLng:
    lat: float = 0.0
    lng: float = 0.0


@dataclass
class ParseResult:
    raw: dict = field(default_factory=dict)
    source_text: str = ""
    language: str = "ko"
    invented: list = field(default_factory=list)
    needs_anchor_question: bool = False
    distance_m: dict = field(default_factory=dict)


@dataclass
class Place:
    name: str = ""
    at: LatLng = field(default_factory=LatLng)
    category: str = ""
    address: str = ""
    distance_m: float | None = None
    category_name: str = ""


@dataclass
class Jibun:
    b_code: str = ""
    main_no: str = ""
    sub_no: str = "0"
    mountain: bool = False
    address: str = ""


@dataclass
class Window:
    center: LatLng = field(default_factory=LatLng)
    radius_m: int = 1000
    source: str = "none"
    label: str = ""
    kind: str = ""
    area: str = ""
    query: str = ""


@dataclass
class Anchor:
    name: str = ""
    at: LatLng = field(default_factory=LatLng)
    source: str = "kakao"
    node_id: int | None = None


@dataclass
class Building:
    way_id: int = 0
    name: str | None = None
    levels: int | None = None
    housenumber: str | None = None
    ring: list = field(default_factory=list)
    centroid: LatLng = field(default_factory=LatLng)


@dataclass
class ViewPoint:
    at: LatLng = field(default_factory=LatLng)
    heading_deg: float = 0.0
    purpose: str = "target"


@dataclass
class Candidate:
    id: str = ""
    type: str = "point"
    osm_id: int | None = None
    name: str | None = None
    geometry: dict = field(default_factory=dict)
    centroid: LatLng = field(default_factory=LatLng)
    entrance: LatLng | None = None
    path: list = field(default_factory=list)
    route_score: float = 0.0
    relaxations: list = field(default_factory=list)
    view_points: list = field(default_factory=list)
    building_way_id: int | None = None


@dataclass
class Evidence:
    id: str = ""
    kind: str = "route"
    candidate_id: str = ""
    supports: bool | None = None
    weight: float = 0.0
    summary_ko: str = ""
    data: dict = field(default_factory=dict)
    capture_date: str | None = None
    image_jpeg: bytes | None = None
    boxes: list | None = None


@dataclass
class Question:
    id: str = ""
    intent_ko: str = ""
    options_ko: list = field(default_factory=list)
    source: str = "ambiguity"
    mapping: dict | None = None


@dataclass
class Decision:
    kind: str = "low_confidence"
    candidate_id: str | None = None
    confidence: str = "low"
    reasons: list = field(default_factory=list)
    question: Question | None = None


@dataclass
class RegistryRecord:
    grnd_floors: int | None = None
    ugrnd_floors: int | None = None
    height_m: float | None = None
    main_purpose: str | None = None
    etc_purpose: str | None = None
    name: str | None = None
    use_apr_day: str | None = None


@dataclass
class Pano:
    at: LatLng = field(default_factory=LatLng)
    date: str = ""


@dataclass
class VisionStats:
    images: int = 0
    first_present: int = 0
    flipped: int = 0
    skipped_stale: int = 0
    old_photos: int = 0


_KAKAO_CATEGORY_CODES = [
    "MT1",
    "CS2",
    "PS3",
    "SC4",
    "AC5",
    "PK6",
    "OL7",
    "SW8",
    "BK9",
    "CT1",
    "AG2",
    "PO3",
    "AT4",
    "AD5",
    "FD6",
    "CE7",
    "HP8",
    "PM9",
    "NONE",
]

PARSE_SCHEMA = {
    "type": "object",
    "required": [
        "language",
        "transcript",
        "region",
        "anchor",
        "steps",
        "landmarks",
        "target",
        "uncertainty",
        "ambiguities",
    ],
    "properties": {
        "language": {
            "type": "string",
            "enum": ["ko", "en", "other"],
        },
        "transcript": {
            "type": "string",
        },
        "region": {
            "type": "object",
            "required": ["evidence", "query", "kind"],
            "properties": {
                "evidence": {"type": "string"},
                "query": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["district", "neighborhood", "station", "place", "none"],
                },
            },
        },
        "anchor": {
            "type": "object",
            "required": ["evidence", "kind", "query"],
            "properties": {
                "evidence": {"type": "string"},
                "query": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["station_exit", "place", "intersection", "none"],
                },
            },
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["order", "action", "evidence", "distance_class"],
                "properties": {
                    "order": {"type": "integer"},
                    "action": {
                        "type": "string",
                        "enum": [
                            "go_straight",
                            "turn_left",
                            "turn_right",
                            "cross",
                            "enter_alley",
                            "follow_road",
                            "go_uphill",
                            "go_downhill",
                            "pass",
                            "arrive",
                        ],
                    },
                    "evidence": {"type": "string"},
                    "distance_class": {
                        "type": "string",
                        "enum": ["immediate", "short", "far_walk", "long", "minutes", "none"],
                    },
                    "minutes": {"type": "integer"},
                    "toward": {"type": "string"},
                    "until": {
                        "type": "object",
                        "properties": {
                            "landmark_ref": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "enum": ["pass", "arrive", "see"],
                            },
                        },
                    },
                    "turn_target": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["alley", "intersection", "road", "unknown"],
                            },
                            "ordinal": {"type": "integer"},
                            "count_from": {"type": "string"},
                        },
                    },
                },
            },
        },
        "landmarks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "evidence", "side", "visual"],
                "properties": {
                    "id": {"type": "string"},
                    "evidence": {"type": "string"},
                    "poi": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "category_code": {
                                "type": "string",
                                "enum": list(_KAKAO_CATEGORY_CODES),
                            },
                            "specific": {"type": "boolean"},
                        },
                    },
                    "visual": {
                        "type": "object",
                        "required": ["en"],
                        "properties": {
                            "en": {"type": "string"},
                            "sign_text": {"type": "string"},
                            "colors": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "floors": {"type": "integer"},
                            "structural": {"type": "boolean"},
                        },
                    },
                    "side": {
                        "type": "string",
                        "enum": ["left", "right", "across", "unknown"],
                    },
                },
            },
        },
        "target": {
            "type": "object",
            "required": ["kind"],
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["building", "named_place", "alley", "intersection", "point"],
                },
                "landmark_ref": {"type": "string"},
                "name": {"type": "string"},
                "specific": {"type": "boolean"},
                "category_code": {
                    "type": "string",
                    "enum": list(_KAKAO_CATEGORY_CODES),
                },
                "relation": {
                    "type": "string",
                    "enum": ["at", "beside", "entrance", "across"],
                },
            },
        },
        "uncertainty": {
            "type": "object",
            "properties": {
                "hedges": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "staleness": {
                    "type": "string",
                    "enum": ["recent", "old", "unknown"],
                },
            },
        },
        "ambiguities": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "about", "impact", "question_ko", "options"],
                "properties": {
                    "id": {"type": "string"},
                    "about": {"type": "string"},
                    "impact": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "question_ko": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
    },
}

VISION_SCHEMA = {
    "type": "object",
    "required": ["attributes", "buildings"],
    "properties": {
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "status"],
                "properties": {
                    "name": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": ["present", "absent", "uncertain"],
                    },
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "note": {"type": "string"},
                },
            },
        },
        "buildings": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["box_2d"],
                "properties": {
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "floors_estimate": {"type": "integer"},
                    "sign_text": {"type": "string"},
                },
            },
        },
    },
}

VERIFY_SCHEMA = {
    "type": "object",
    "required": ["answer"],
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["yes", "no", "uncertain"],
        },
    },
}

JUDGE_SCHEMA = {
    "type": "object",
    "required": ["choice", "confidence", "reasons"],
    "properties": {
        "choice": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["text_ko", "evidence_ids"],
                "properties": {
                    "text_ko": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "counter_evidence_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

QUESTION_SCHEMA = {
    "type": "object",
    "required": ["question_ko", "options_ko"],
    "properties": {
        "question_ko": {"type": "string"},
        "options_ko": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}

ANSWER_SCHEMA = {
    "type": "object",
    "required": ["option_index"],
    "properties": {
        "option_index": {"type": "integer"},
        "note": {"type": "string"},
    },
}

B0_SCHEMA = {
    "type": "object",
    "required": ["address", "lat", "lng", "confidence"],
    "properties": {
        "address": {"type": "string"},
        "lat": {"type": "number"},
        "lng": {"type": "number"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
}
