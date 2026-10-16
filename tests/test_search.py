"""Unit tests for geographic window resolution and start anchor detection."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from app import config, geo, kakao, llm, mapdata, search
from app.contracts import Anchor, LatLng, ParseResult, Place, Question, Window


def _ll(x: float, y: float) -> LatLng:
    return geo.to_ll(x, y, config.VENUE)


def _rect(way_id: int, x0: float, x1: float, y0: float, y1: float) -> mapdata.Building:
    ring = [_ll(x0, y0), _ll(x1, y0), _ll(x1, y1), _ll(x0, y1), _ll(x0, y0)]
    centroid = _ll((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    return mapdata.Building(way_id=way_id, ring=ring, centroid=centroid)


N = {
    'N0': (0, 0),
    'N1': (50, 0),
    'N2': (100, 0),
    'N3': (150, 0),
    'N4': (200, 0),
    'A1': (50, 60),
    'A2': (100, -80),
    'A3': (150, -80),
}
ID = {name: i for i, name in enumerate(N, start=1)}
START = Anchor(name='N0', at=_ll(0, 0), source='kakao')
SONGPA = Window(
    center=config.VENUE,
    radius_m=3000,
    source='text',
    label='서울 송파구',
    kind='district',
    area='송파구',
)
GANGNAM_COLLOQUIAL = Window(
    center=config.VENUE,
    radius_m=3000,
    source='text',
    label='서울 강남구',
    kind='district',
    area='강남구',
    query='강남',
)
CUTLET = Place(name='돈가스집', at=_ll(100, 14), category_name='음식점 > 일식 > 돈까스,우동')
MEAT_ACROSS = Place(name='고기집A', at=_ll(100, -14), category_name='음식점 > 한식 > 육류,고기')
MEAT_FAR = Place(name='고기집B', at=_ll(160, -14), category_name='음식점 > 한식 > 육류,고기')
MEAT_BESIDE = Place(name='고기집C', at=_ll(130, 14), category_name='음식점 > 한식 > 육류,고기')
MALL = Place(name='코엑스몰', at=_ll(50, 14), category_name='쇼핑')
DEPT = Place(name='현대백화점', at=_ll(150, 14), category_name='백화점')


def _pr(steps: list[dict], target: dict, landmarks=(), anchor=None) -> ParseResult:
    raw = {
        'steps': steps,
        'target': target,
        'landmarks': list(landmarks),
    }
    if anchor is not None:
        raw['anchor'] = anchor
    return ParseResult(raw=raw, language='ko')


def _with_landmark(specific: bool) -> ParseResult:
    poi = {
        'name': '카츠젠 문정직영점' if specific else '돈가스집',
        'specific': specific,
    }
    return _pr([], {'kind': 'building'}, landmarks=[{'poi': poi}], anchor={'kind': 'none', 'evidence': '', 'query': ''})


@pytest.fixture
def toy():
    nodes = {ID[k]: _ll(*v) for k, v in N.items()}
    edges = [
        (ID['N0'], ID['N1'], 'tertiary', 1),
        (ID['N1'], ID['N2'], 'tertiary', 1),
        (ID['N2'], ID['N3'], 'tertiary', 1),
        (ID['N3'], ID['N4'], 'tertiary', 1),
        (ID['N1'], ID['A1'], 'residential', 2),
        (ID['N2'], ID['A2'], 'residential', 3),
        (ID['N3'], ID['A3'], 'residential', 4),
    ]
    buildings = [
        _rect(10, 155, 175, -60, -40),
        _rect(11, 125, 145, -60, -40),
    ]
    return mapdata.MapData.from_parts(config.VENUE, nodes, edges, buildings)


@pytest.fixture
def two_exits():
    nodes = {
        1: _ll(-200, 0),
        2: _ll(200, 0),
    }
    edges = [(1, 2, 'tertiary', 1)]
    md = mapdata.MapData.from_parts(config.VENUE, nodes, edges, [])
    md.entrances = [
        {'node_id': 101, 'ref': '3', 'name': None, 'at': _ll(-200, 0)},
        {'node_id': 102, 'ref': '3', 'name': None, 'at': _ll(200, 0)},
    ]
    window = Window(center=config.VENUE, radius_m=1000, source='region_input', label='강남구')
    return md, window


@pytest.fixture(autouse=True)
def fake_kakao(monkeypatch):
    def fake_address(query: str) -> list[Place]:
        q = query.strip()
        if q in ('송파구', '강남구', '서초구', '마포구', '종로구'):
            return [Place(name=q, at=config.VENUE, address=f"서울 {q}", category="")]
        if q in ('삼성동', '역삼동', '대치동', '문정동'):
            return [Place(name=q, at=config.VENUE, address=f"서울 강남구 {q}", category="")]
        if q in ('부산', '경기도', '충청북도', '대구', '대전'):
            return [Place(name=q, at=config.VENUE, address=q, category="")]
        return []

    def fake_keyword(query: str, near=None, radius_m=1000, size=15, category=None, page=1) -> list[Place]:
        q = query.strip()
        if '삼성역' in q:
            return [Place(name='삼성역 2호선', at=_ll(-200, 0), category='SW8', category_name='교통,수송 > 지하철,전철 > 2호선', address='서울 강남구 삼성동')]
        if '봉은사역' in q:
            return [Place(name='봉은사역 9호선', at=_ll(200, 0), category='SW8', category_name='교통,수송 > 지하철,전철 > 9호선', address='서울 강남구 삼성동')]
        if '코엑스' in q:
            return [Place(name='코엑스', at=_ll(50, 14), category='', address='서울 강남구 삼성동')]
        if '오토웨이' in q:
            return [Place(name='오토웨이타워', at=_ll(0, 0), category='', address='서울 강남구 대치동 417')]
        if '2호선' in q:
            return [Place(name='삼성역 2호선', at=_ll(-200, 0), category='SW8', category_name='교통,수송 > 지하철,전철 > 2호선', address='서울 강남구 삼성동')]
        if '9호선' in q:
            return [Place(name='봉은사역 9호선', at=_ll(200, 0), category='SW8', category_name='교통,수송 > 지하철,전철 > 9호선', address='서울 강남구 삼성동')]
        if '한림연예예술고등학교' in q:
            return [Place(name='한림연예예술고등학교', at=_ll(0, -4076), address='서울 송파구 장지동 850')]
        if '카츠젠 문정직영점' in q:
            return [Place(name='카츠젠 문정직영점', at=_ll(50, 50), address='서울 송파구 문정동 10')]
        if '미례국밥' in q:
            return [Place(name='미례국밥 강남삼성타운점', at=_ll(0, -4000), address='서울 서초구 서초동 1327')]
        return []

    def fake_category(code: str, near, radius_m=1000, size=15, page=1) -> list[Place]:
        if code == 'SW8':
            diff = near.lng - config.VENUE.lng if near else 0.0
            if diff < -1e-6:
                return [Place(name='삼성역 2호선', at=_ll(-200, 0), category='SW8', category_name='2호선')]
            elif diff > 1e-6:
                return [Place(name='봉은사역 9호선', at=_ll(200, 0), category='SW8', category_name='9호선')]
            else:
                return [
                    Place(name='삼성역 2호선', at=_ll(-200, 0), category='SW8', category_name='2호선'),
                    Place(name='봉은사역 9호선', at=_ll(200, 0), category='SW8', category_name='9호선'),
                ]
        return []

    monkeypatch.setattr(search.kakao, 'address', fake_address)
    monkeypatch.setattr(search.kakao, 'keyword', fake_keyword)
    monkeypatch.setattr(search.kakao, 'category', fake_category)


def _fake_turns(monkeypatch, fail=False):
    asked = []

    def fake_generate_tools(model_role, model, contents, tools=None, timeout_ms=None):
        if fail:
            raise llm.LLMError("Gemini error")
        if model_role == "en_resolve_turn1":
            part = MagicMock()
            part.function_call.name = "kakao_keyword_search"
            part.function_call.args = {"query": "오토웨이타워"}
            candidate = MagicMock()
            candidate.content.parts = [part]
            resp = MagicMock()
            resp.candidates = [candidate]
            return resp
        elif model_role == "en_resolve_turn2":
            chunk = MagicMock()
            chunk.web.title = "Spaces Autoway Tower - Google Maps"
            chunk.web.uri = "https://maps.google.com/?cid=1"
            meta = MagicMock()
            meta.grounding_chunks = [chunk]
            candidate = MagicMock()
            candidate.grounding_metadata = meta
            resp = MagicMock()
            resp.candidates = [candidate]
            return resp
        raise ValueError(f"Unexpected call: {model_role}")

    def fake_kw(budget, query, near=None, radius_m=1000, size=15, category=None, page=1):
        asked.append(query)
        if query == "오토웨이타워":
            return [Place(name="오토웨이타워", at=config.VENUE, address="서울 강남구 대치동 417")]
        return []

    monkeypatch.setattr(search.llm, "generate_tools", fake_generate_tools)
    monkeypatch.setattr(search, "_kw", fake_kw)
    return asked


@pytest.mark.parametrize(
    'text,expected',
    [
        ('지하철 2호선 삼성역 3번 출구', ('삼성역', '3', '2호선')),
        ('2호선 3번 출구', (None, '3', '2호선')),
        ('강남역11번출구', ('강남역', '11', None)),
        ('삼성 3번 출구', ('삼성역', '3', None)),
        ('삼성동 2호선 3번 출구', (None, '3', '2호선')),
    ],
)
def test_parse_station_exit(text, expected):
    assert search.parse_station_exit(text) == expected


@pytest.mark.parametrize('region,kind', [('송파구', 'district'), ('삼성동', 'neighborhood'), ('삼성역', 'station'), ('코엑스', 'place')])
def test_window_radius_follows_region_kind(region, kind):
    w = search.resolve_window(_pr([], {'kind': 'point'}), region, None)
    assert (w.source, w.radius_m) == ('region_input', config.WINDOW_RADIUS_M[kind])


def test_window_from_current_location():
    w = search.resolve_window(_pr([], {'kind': 'point'}), None, config.VENUE)
    assert (w.source, w.radius_m) == ('current_location', 500)


def test_exit_without_station_or_window_asks_region():
    exit_only = {'evidence': '2호선 3번 출구', 'query': '2호선 3번 출구', 'kind': 'station_exit'}
    anchors, question = search.resolve_anchors(_pr([], {'kind': 'point'}, anchor=exit_only), None, None)
    assert anchors == [] and question.source == 'region'


def test_no_anchor_and_no_window_asks_region():
    none = {'evidence': '', 'query': '', 'kind': 'none'}
    anchors, question = search.resolve_anchors(_pr([], {'kind': 'alley'}, anchor=none), None, None)
    assert anchors == [] and question.source == 'region'


def test_district_window_carries_area_name():
    w = search.resolve_window(_pr([], {'kind': 'point'}), '송파구', None)
    assert (w.kind, w.area) == ('district', '송파구')


def test_anchor_inside_district_found_beyond_window_radius():
    school = Place(name='한림연예예술고등학교', at=_ll(0, -4076), address='서울 송파구 장지동 850')
    anchor = {'evidence': '한림예고 쪽', 'query': '한림연예예술고등학교', 'kind': 'place'}
    anchors, question = search.resolve_anchors(_pr([], {'kind': 'building'}, anchor=anchor), SONGPA, None)
    assert question is None and [a.name for a in anchors] == ['한림연예예술고등학교']


def test_anchor_outside_district_is_rejected_and_asked_by_name(monkeypatch):
    other = Place(name='한림연예예술고등학교', at=_ll(0, -2000), address='서울 강남구 대치동 1')
    anchor = {'evidence': '한림예고 쪽', 'query': '한림연예예술고등학교', 'kind': 'place'}
    monkeypatch.setattr(search.kakao, 'keyword', lambda *args, **kwargs: [other])
    anchors, question = search.resolve_anchors(_pr([], {'kind': 'building'}, anchor=anchor), SONGPA, None)
    assert anchors == [] and question.source == 'anchor' and ('한림예고' in question.intent_ko)


def test_colloquial_region_accepts_nearby_anchor_outside_the_district():
    shop = Place(name='미례국밥 강남삼성타운점', at=_ll(0, -4000), address='서울 서초구 서초동 1327')
    anchor = {'evidence': '미례국밥 집에서', 'query': '미례국밥', 'kind': 'place'}
    budget = search._Budget()
    anchors, question = search.resolve_anchors(
        _pr([], {'kind': 'building'}, anchor=anchor),
        GANGNAM_COLLOQUIAL,
        None,
        budget,
    )
    assert question is None and [a.name for a in anchors] == ['미례국밥 강남삼성타운점']
    assert any(('주소 밖' in n for n in budget.notes))


def test_explicit_district_still_rejects_anchor_outside_it():
    shop = Place(name='미례국밥 강남삼성타운점', at=_ll(0, -4000), address='서울 서초구 서초동 1327')
    strict = Window(center=config.VENUE, radius_m=3000, source='text', label='서울 강남구', kind='district', area='강남구', query='강남구')
    anchors, question = search.resolve_anchors(
        _pr([], {'kind': 'building'}, anchor={'evidence': '미례국밥 집에서', 'query': '미례국밥', 'kind': 'place'}),
        strict,
        None,
    )
    assert anchors == [] and question.source == 'anchor'


def test_colloquial_region_does_not_accept_a_far_away_same_name_place(monkeypatch):
    far = Place(name='미례국밥', at=_ll(0, -15000), address='부산 부산진구 전포동 356-10')
    monkeypatch.setattr(search.kakao, 'keyword', lambda *args, **kwargs: [far])
    anchors, question = search.resolve_anchors(
        _pr([], {'kind': 'building'}, anchor={'evidence': '미례국밥 집에서', 'query': '미례국밥', 'kind': 'place'}),
        GANGNAM_COLLOQUIAL,
        None,
    )
    assert anchors == [] and question.source == 'anchor'


def test_generic_landmark_is_never_used_as_anchor():
    anchors, question = search.resolve_anchors(_with_landmark(False), SONGPA, None)
    assert anchors == [] and question.source == 'anchor'


def test_specific_landmark_can_stand_in_as_anchor():
    anchors, question = search.resolve_anchors(_with_landmark(True), SONGPA, None)
    assert question is None and [a.name for a in anchors] == ['카츠젠 문정직영점']


def test_line_filter_picks_single_exit(two_exits):
    md, window = two_exits
    query = {'evidence': '2호선 3번 출구', 'query': '2호선 3번 출구', 'kind': 'station_exit'}
    anchors, question = search.resolve_anchors(_pr([], {'kind': 'point'}, anchor=query), window, md)
    assert question is None and [(a.name, a.source) for a in anchors] == [('삼성역 3번 출구', 'osm_entrance')]


def test_two_stations_without_line_ask_station(two_exits):
    md, window = two_exits
    query = {'evidence': '3번 출구', 'query': '3번 출구', 'kind': 'station_exit'}
    anchors, question = search.resolve_anchors(_pr([], {'kind': 'point'}, anchor=query), window, md)
    assert question.source == 'station' and question.options_ko == ['삼성역 3번 출구', '봉은사역 3번 출구']


def test_line_filter_checks_every_exit_not_only_nearest(toy, monkeypatch):
    toy.entrances = [{'node_id': 900 + i, 'ref': '3', 'name': None, 'at': _ll(100 * i, 0)} for i in range(9)]
    line_stations = [
        Place(name='선릉역 2호선', at=_ll(0, 10), category='SW8'),
        Place(name='역삼역 2호선', at=_ll(800, 10), category='SW8'),
    ]
    window = Window(center=config.VENUE, radius_m=1000, source='region_input', label='강남구')
    monkeypatch.setattr(search.kakao, 'keyword', lambda *args, **kwargs: line_stations)
    monkeypatch.setattr(search.kakao, 'category', lambda *args, **kwargs: line_stations)

    pr = _pr([], {'kind': 'point'}, anchor={'evidence': '2호선 3번 출구', 'query': '2호선 3번 출구', 'kind': 'station_exit'})
    anchors, question = search.resolve_anchors(pr, window, toy)
    assert anchors == [] and question.options_ko == ['선릉역 3번 출구', '역삼역 3번 출구']


def test_en_resolve_uses_korean_query_from_function_call(monkeypatch):
    asked = _fake_turns(monkeypatch, fail=False)
    budget = search._Budget()
    places = search.en_resolve(budget, 'Autoway Tower', config.VENUE, 1000)
    assert [p.name for p in places] == ['오토웨이타워'] and asked == ['오토웨이타워']
    assert budget.maps_sources == [{'title': 'Spaces Autoway Tower - Google Maps', 'uri': 'https://maps.google.com/?cid=1'}]


def test_en_resolve_falls_back_to_kakao_with_original_name(monkeypatch):
    asked = []

    def fake_generate_tools(model_role, model, contents, tools=None, timeout_ms=None):
        resp = MagicMock()
        resp.candidates = [MagicMock(content=MagicMock(parts=[]))]
        return resp

    def fake_kw(b, q, *args, **kwargs):
        asked.append(q)
        return []

    monkeypatch.setattr(search.llm, "generate_tools", fake_generate_tools)
    monkeypatch.setattr(search, "_kw", fake_kw)

    budget = search._Budget()
    assert search.en_resolve(budget, 'Autoway Tower', config.VENUE, 1000) == [] and asked == ['Autoway Tower']
    assert any(('Gemini' in n for n in budget.notes))


def test_en_resolve_falls_back_on_gemini_timeout(monkeypatch):
    asked = []

    def fake_generate_tools(*args, **kwargs):
        raise search.llm.LLMError("Gemini timeout")

    def fake_kw(b, q, *args, **kwargs):
        asked.append(q)
        return []

    monkeypatch.setattr(search.llm, "generate_tools", fake_generate_tools)
    monkeypatch.setattr(search, "_kw", fake_kw)

    budget = search._Budget()
    assert search.en_resolve(budget, 'Autoway Tower', config.VENUE, 1000) == [] and asked == ['Autoway Tower']


def test_english_anchor_goes_through_en_resolve(monkeypatch):
    _fake_turns(monkeypatch, fail=False)
    pr = _pr([], {'kind': 'point'}, anchor={'evidence': 'Autoway Tower', 'query': 'Autoway Tower', 'kind': 'place'})
    anchors, question = search.resolve_anchors(pr, None, None)
    assert question is None and [(a.name, a.source) for a in anchors] == [('오토웨이타워', 'kakao')]


def test_station_exit_missing_falls_back_to_station_centre(monkeypatch):
    station = Place(name='서면역 1호선', at=_ll(0, 0), category_name='교통,수송 > 지하철,전철 > 부산1호선')
    md = mapdata.MapData.from_parts(config.VENUE, {1: _ll(0, 0), 2: _ll(50, 0)}, [(1, 2, 'tertiary', 1)], [])
    budget = search._Budget()

    def fake_kw(b, q, *args, **kwargs):
        if '번출구' in q:
            return []
        return [station]

    monkeypatch.setattr(search, '_kw', fake_kw)
    pr = _pr([], {'kind': 'point'}, anchor={'evidence': '서면역 7번 출구', 'query': '서면역 7번 출구', 'kind': 'station_exit'})
    anchors, question = search.resolve_anchors(pr, None, md, budget)
    assert question is None and anchors and ('역 중심' in anchors[0].name) and (anchors[0].at == station.at)
    assert any(('출구 위치 자료 없음' in n for n in budget.notes))


def test_city_region_is_a_city_window_filtered_by_short_name():
    w = search._window(Place(name='부산', at=_ll(0, 0)), 'city', 'text')
    assert search._region_kind('부산', True, '') == 'city'
    assert search._region_kind('경기도', True, '') == 'city'
    assert search._region_kind('송파구', True, '') == 'district'
    assert w.kind == 'city' and w.radius_m >= 20000 and (w.area == '부산')
    assert search._window(Place(name='충청북도', at=_ll(0, 0)), 'city', 'text').area == '충북'


def test_city_window_station_search_recentres_on_the_station(monkeypatch):
    station = Place(name='서면역 1호선', at=_ll(2700, 0), category_name='교통,수송 > 지하철,전철 > 부산1호선')
    loads = []
    window = search._window(Place(name='부산', at=_ll(0, 0)), 'city', 'text')

    def fake_kw(b, q, *args, **kwargs):
        if '번출구' in q:
            return []
        return [station]

    def fake_load_area(center, radius_m, db_path=None):
        loads.append((center, radius_m))
        return mapdata.MapData.from_parts(center, {1: station.at}, [], [])

    monkeypatch.setattr(search, '_kw', fake_kw)
    monkeypatch.setattr(search.mapdata, 'load_area', fake_load_area)

    pr = _pr([], {'kind': 'point'}, anchor={'evidence': '서면역 7번 출구', 'query': '서면역 7번 출구', 'kind': 'station_exit'})
    anchors, question = search.resolve_anchors(pr, window, None)
    assert question is None and anchors and (anchors[0].at == station.at)
    assert loads and loads[-1][1] <= 1000 and (geo.dist_m(loads[-1][0], station.at) < 1)


def test_kw_dedupes_the_same_query_across_sizes(monkeypatch):
    calls = []
    b = search._Budget()

    def fake_keyword(query, near=None, radius_m=1000, size=15, category=None, page=1):
        calls.append(size)
        return [Place(name=f"p{i}", at=config.VENUE) for i in range(15)]

    monkeypatch.setattr(search.kakao, 'keyword', fake_keyword)

    assert len(search._kw(b, '오토웨이타워', near=config.VENUE, radius_m=500, size=5)) == 5
    assert len(search._kw(b, '오토웨이타워', near=config.VENUE, radius_m=500)) == 15
    assert calls == [15] and b.used == 1


def test_kakao_results_are_reused_for_a_nearby_centre_and_smaller_radius(monkeypatch):
    calls = []
    b = search._Budget()

    def fake_keyword(query, near=None, radius_m=1000, size=15, category=None, page=1):
        calls.append((query, radius_m))
        return [Place(name='가까운 집', at=_ll(100, 0)), Place(name='먼 집', at=_ll(1200, 0))]

    monkeypatch.setattr(search.kakao, 'keyword', fake_keyword)

    first = search._kw(b, '오토웨이타워', near=_ll(0, 0), radius_m=1500)
    second = search._kw(b, '오토웨이타워', near=_ll(50, 0), radius_m=800)

    assert [p.name for p in first] == ['가까운 집', '먼 집'] and [p.name for p in second] == ['가까운 집']
    assert calls == [('오토웨이타워', 1500)] and b.used == 1

    search._kw(b, '오토웨이타워', near=_ll(300, 0), radius_m=1500)
    search._kw(b, '오토웨이타워', near=_ll(0, 0), radius_m=2000)
    assert len(calls) == 3


def test_station_vicinity_without_start_uses_the_station_as_start():
    pr = _pr(
        [{'order': 1, 'action': 'arrive', 'evidence': '앞이에요', 'distance_class': 'none'}],
        {'kind': 'building', 'name': '국밥집', 'specific': False, 'category_code': 'FD6'},
        anchor={'evidence': '', 'query': '', 'kind': 'none'},
    )
    station = Window(center=_ll(0, 0), radius_m=500, source='text', label='불광역 3호선', kind='station')
    anchors, q = search.resolve_anchors(pr, station, None, search._Budget())
    assert q is None and [a.name for a in anchors] == ['불광역 3호선'] and (anchors[0].at == station.center)

    anchors_district, q_district = search.resolve_anchors(pr, SONGPA, None, search._Budget())
    assert anchors_district == [] and q_district is not None and (q_district.source == 'anchor')


def test_same_station_name_in_two_cities_asks_which_city(monkeypatch):
    daejeon = Place(name='중앙로역 대전1호선', at=LatLng(36.329, 127.428), category_name='교통,수송 > 지하철,전철')
    daegu = Place(name='중앙로역 대구1호선', at=LatLng(35.8713, 128.5941), category_name='교통,수송 > 지하철,전철')
    pr = _pr([{'order': 1, 'action': 'arrive', 'evidence': '앞이에요', 'distance_class': 'none'}], {'kind': 'building', 'name': '국밥집', 'specific': False}, anchor={'evidence': '', 'query': '', 'kind': 'none'})
    pr.raw['region'] = {'evidence': '중앙로역 근처', 'query': '중앙로역', 'kind': 'station'}

    def fake_addr(budget, query):
        return []

    def fake_kw(budget, query, near=None, radius_m=1000, size=15, category=None, page=1):
        if '중앙로역' in query:
            return [daejeon, daegu]
        if '시청역' in query:
            return [
                Place(name='시청역 1호선', at=LatLng(37.565, 126.977), category='SW8'),
                Place(name='시청역 2호선', at=LatLng(37.564, 126.976), category='SW8'),
            ]
        return []

    monkeypatch.setattr(search, '_addr', fake_addr)
    monkeypatch.setattr(search, '_kw', fake_kw)

    budget = search._Budget()
    assert search.resolve_window(pr, None, None, budget) is None
    anchors, q = search.resolve_anchors(pr, None, None, budget)
    assert anchors == [] and q.id == 'Q-region' and (q.options_ko[:2] == ['중앙로역 대전1호선', '중앙로역 대구1호선'])

    pr.raw['region'] = {'evidence': '시청역 근처', 'query': '시청역', 'kind': 'station'}
    assert search.resolve_window(pr, None, None, search._Budget()).label == '시청역 1호선'


def test_start_point_in_two_cities_asks_which_city(monkeypatch):
    daejeon = Place(name='중앙로역 대전1호선', at=LatLng(36.329, 127.428), category_name='교통,수송 > 지하철,전철')
    daegu = Place(name='중앙로역 대구1호선', at=LatLng(35.8713, 128.5941), category_name='교통,수송 > 지하철,전철')
    pr = _pr(
        [{'order': 1, 'action': 'arrive', 'evidence': '앞이에요', 'distance_class': 'none'}],
        {'kind': 'building', 'name': '국밥집', 'specific': False},
        anchor={'evidence': '중앙로역 근처', 'query': '중앙로역', 'kind': 'place'},
    )

    def fake_kw(budget, query, near=None, radius_m=1000, size=15, category=None, page=1):
        if '중앙로역' in query:
            return [daejeon, daegu]
        return []

    monkeypatch.setattr(search, '_kw', fake_kw)

    anchors, q = search.resolve_anchors(pr, None, None)
    assert anchors == [] and q.id == 'Q-region' and (set(q.options_ko[:2]) == {'중앙로역 대전1호선', '중앙로역 대구1호선'})
