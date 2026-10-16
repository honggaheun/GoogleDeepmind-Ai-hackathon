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


# --- Part 2 Tests ---

from app.contracts import Candidate


@pytest.fixture
def street():
    nodes = {
        1: _ll(0, 0),
        2: _ll(50, 0),
        3: _ll(100, 0),
        4: _ll(150, 0),
        5: _ll(200, 0),
    }
    edges = [
        (1, 2, 'tertiary', 1),
        (2, 3, 'tertiary', 1),
        (3, 4, 'tertiary', 1),
        (4, 5, 'tertiary', 1),
    ]
    buildings = [
        _rect(20, 90, 110, 10, 30),
        _rect(21, 90, 110, -30, -10),
        _rect(22, 150, 170, -30, -10),
        _rect(23, 120, 140, 10, 30),
    ]
    return mapdata.MapData.from_parts(config.VENUE, nodes, edges, buildings)


def _chain_pr(relation: str, target_name: str = '고기집') -> ParseResult:
    steps = [
        {'order': 1, 'action': 'go_straight', 'distance_class': 'none'},
    ]
    target = {
        'kind': 'building',
        'name': target_name,
        'specific': False,
        'relation': {'landmark_ref': 'L1', 'relation': relation},
    }
    landmarks = [
        {'id': 'L1', 'evidence': '돈가스집', 'poi': {'name': '돈가스집', 'specific': False}},
    ]
    return _pr(steps, target, landmarks, anchor={'kind': 'none', 'evidence': '', 'query': ''})


def _category_pr(action: str = 'follow_road') -> ParseResult:
    steps = [
        {'order': 1, 'action': action, 'distance_class': 'none', 'evidence': '대로변'},
    ]
    target = {'kind': 'building', 'name': '가게', 'specific': False}
    return _pr(steps, target)


def _dept_pr(specific: bool = True) -> ParseResult:
    steps = [
        {'order': 1, 'action': 'go_straight', 'distance_class': 'none', 'until': {'landmark_ref': 'L2', 'relation': 'pass'}},
    ]
    target = {
        'kind': 'building',
        'name': '건물',
        'specific': False,
        'relation': {'landmark_ref': 'L1', 'relation': 'across'},
    }
    landmarks = [
        {'id': 'L2', 'evidence': '코엑스몰', 'poi': {'name': '코엑스몰', 'specific': True}},
        {'id': 'L1', 'evidence': '현대백화점' if specific else '백화점', 'poi': {'name': '현대백화점' if specific else '백화점', 'specific': specific}},
    ]
    return _pr(steps, target, landmarks)


def _building_route(side: str, ordinal: int = 2) -> ParseResult:
    steps = [
        {'order': 1, 'action': 'turn_right', 'direction': 'right', 'ordinal': ordinal, 'evidence': f'{ordinal}번째 골목 우회전'},
        {'order': 2, 'action': 'arrive', 'side': side, 'evidence': '왼쪽 건물' if side == 'left' else '오른쪽 건물'},
    ]
    target = {'kind': 'building', 'side': side}
    return _pr(steps, target)


def _route(walk_in: bool = False) -> list[dict]:
    return [
        {'order': 1, 'action': 'turn_right', 'direction': 'right', 'ordinal': 2, 'evidence': '두 번째 골목 우회전'},
    ]


def _gs25_route() -> ParseResult:
    steps = [
        {'order': 1, 'action': 'go_straight', 'evidence': '직진', 'distance_class': 'none'},
    ]
    target = {'kind': 'building', 'name': 'GS25', 'specific': True}
    landmarks = [
        {'id': 'L1', 'evidence': 'GS25', 'poi': {'name': 'GS25', 'specific': True}},
    ]
    return _pr(steps, target, landmarks)


def _starbucks_building_route() -> ParseResult:
    steps = [
        {'order': 1, 'action': 'turn_right', 'direction': 'right', 'ordinal': 2, 'evidence': '두 번째 골목 우회전'},
    ]
    target = {'kind': 'building', 'name': '스타벅스', 'specific': True}
    return _pr(steps, target)


def _toward_pr(toward: str = 'L9') -> ParseResult:
    steps = [
        {'order': 1, 'action': 'go_straight', 'evidence': '연세대 방향으로 직진', 'distance_class': 'none', 'toward': toward},
    ]
    target = {'kind': 'building'}
    landmarks = [
        {'id': toward, 'evidence': '연세대학교', 'poi': {'name': '연세대학교', 'specific': True}},
    ]
    return _pr(steps, target, landmarks)


def test_building_on_left_after_second_right_alley(toy):
    assert search.expand(_building_route('left'), toy, START)[0].osm_id == 10


def test_alley_target_is_last_turned_alley(toy):
    top = search.expand(_pr(_route(walk_in=False), {'kind': 'alley'}), toy, START)[0]
    assert (top.type, top.osm_id, top.entrance) == ('alley', 4, _ll(150, 0))


def test_missing_third_right_alley_relaxes_ordinal(toy):
    cands = search.expand(_building_route('left', ordinal=3), toy, START)
    assert cands and 'ordinal-1' in cands[0].relaxations


def test_building_on_right_side(toy):
    assert search.expand(_building_route('right'), toy, START)[0].osm_id == 11


def test_ordinal_counts_from_start_even_after_long_straight():
    road = {
        1: (0, 0),
        2: (50, 0),
        3: (100, 0),
        4: (150, 0),
        5: (200, 0),
        12: (50, -80),
        13: (100, -80),
        14: (150, -80),
    }
    nodes = {k: _ll(*xy) for k, xy in road.items()}
    edges = [
        (1, 2, 'tertiary', 1),
        (2, 3, 'tertiary', 1),
        (3, 4, 'tertiary', 1),
        (4, 5, 'tertiary', 1),
        (2, 12, 'residential', 12),
        (3, 13, 'residential', 13),
        (4, 14, 'residential', 14),
    ]
    md = mapdata.MapData.from_parts(config.VENUE, nodes, edges, [])
    steps = [
        {'order': 1, 'action': 'go_straight', 'evidence': '쭉 직진하다가', 'distance_class': 'long'},
        {'order': 2, 'action': 'turn_right', 'direction': 'right', 'ordinal': 2, 'evidence': '두 번째 골목에서 우회전'},
    ]
    pr = _pr(steps, {'kind': 'alley'})
    top = search.expand(pr, md, START)[0]
    assert (top.osm_id, top.entrance) == (13, _ll(100, 0))


def test_find_candidates_reloads_map_around_far_anchor(monkeypatch):
    far = geo.to_ll(4000, 0, config.VENUE)
    steps = [{'order': 1, 'action': 'go_straight', 'evidence': '직진', 'distance_class': 'short'}]
    pr = _pr(steps, {'kind': 'point'}, anchor={'evidence': '먼 학교', 'query': '먼 학교', 'kind': 'place'})
    monkeypatch.setattr(search.kakao, 'address', lambda *a, **k: [])
    monkeypatch.setattr(
        search.kakao,
        'keyword',
        lambda budget, query, **kw: [Place(name='먼 학교', at=far, category_name='교육,학문 > 학교')] if '학교' in query else [],
    )
    out = search.find_candidates(pr)
    assert out['anchors'][0].node_id is not None and geo.dist_m(out['md'].origin, far) < 1


def test_across_picks_nearest_target_spot_across_the_road(street):
    la = {'L1': [CUTLET], '__target__': [MEAT_FAR, MEAT_ACROSS, MEAT_BESIDE]}
    cands = search.expand(_chain_pr('across'), street, START, la)
    assert [c.osm_id for c in cands][:2] == [21, 22] and cands[0].name == '고기집A'


def test_beside_picks_same_side_neighbour(street):
    la = {'L1': [CUTLET], '__target__': [MEAT_FAR, MEAT_ACROSS, MEAT_BESIDE]}
    assert search.expand(_chain_pr('beside'), street, START, la)[0].osm_id == 23


def test_across_without_target_spots_uses_geometry(street):
    cands = search.expand(_chain_pr('across'), street, START, {'L1': [CUTLET]})
    assert cands[0].osm_id == 21


def test_across_with_unconfirmed_landmark_falls_back_to_path_end_buildings(street):
    cands = search.expand(_chain_pr('across'), street, START, {'L1': []})
    assert cands and all(('relation' in c.relaxations for c in cands)) and (cands[0].route_score < 1.0)


def test_no_candidates_for_business_target_asks_name(monkeypatch):
    pr = _chain_pr('across')
    pr.raw['anchor'] = {'evidence': '출발 학교 쪽', 'query': '출발 학교', 'kind': 'place'}
    monkeypatch.setattr(
        search.kakao,
        'keyword',
        lambda *a, **k: [Place(name='출발 학교', at=config.VENUE, category_name='교육,학문 > 학교')],
    )
    monkeypatch.setattr(search.kakao, 'category', lambda *a, **k: [])
    monkeypatch.setattr(search, 'google_places_of_type', lambda *a, **k: [])
    out = search.find_candidates(pr)
    assert out['candidates'] and all(('relation' in c.relaxations for c in out['candidates'])) and (out['question'] is None)


def test_match_terms_filters_places_by_meaning(monkeypatch):
    places = [CUTLET, MEAT_ACROSS, Place(name='자담치킨', at=_ll(0, 0), category_name='음식점 > 치킨')]
    def fake_llm(schema, prompt, **kw):
        return {'matches': [{'term': '고기집', 'name': '고기집A'}, {'term': '돈가스', 'name': '돈가스집'}]}
    monkeypatch.setattr(search.llm, 'generate_json', fake_llm)
    out = search.match_terms(places, ['고기집', '돈가스'], search._Budget())
    assert [p.name for p in out['고기집']] == ['고기집A'] and [p.name for p in out['돈가스']] == ['돈가스집']


def test_landmark_positions_uses_category_rule_before_google(monkeypatch):
    import types
    pool = [
        Place(name='밥향기', at=_ll(0, 0), category_name='음식점 > 중식 > 중국요리'),
        Place(name='자담치킨', at=_ll(20, 0), category_name='음식점 > 치킨'),
    ]
    pr = _pr(
        [{'order': 1, 'action': 'arrive', 'evidence': '식당이 있어', 'distance_class': 'none', 'until': {'landmark_ref': 'lm2', 'relation': 'arrive'}}],
        {'kind': 'building', 'category_name': '중식', 'name': '식당'},
        landmarks=[{'id': 'lm2', 'evidence': '식당', 'poi': {'name': '식당', 'specific': False}}],
    )
    monkeypatch.setattr(search, '_poi_pool', lambda *a, **k: pool)
    monkeypatch.setattr(search, 'match_terms', lambda places, terms, budget: {t: [] for t in terms})
    google_called = False
    def fake_google(*a, **k):
        nonlocal google_called
        google_called = True
        return []
    monkeypatch.setattr(search, 'google_places_of_type', fake_google)
    out = search._landmark_positions(pr, types.SimpleNamespace(trees=[]), _ll(0, 0), 300, search._Budget())
    assert [p.name for p in out['__target__']] == ['밥향기'] and [p.name for p in out['lm2']] == ['밥향기']
    assert not google_called


def test_road_fit_rewards_big_road_frontage_only_for_follow_road_steps(toy):
    fit, why = search._road_fit(toy, _category_pr(), _ll(100, -200))
    assert search._road_fit(toy, _category_pr(), _ll(163, -42)) == (1.0, '큰길가')
    assert fit == -0.5 and '큰길에서' in why
    assert search._road_fit(toy, _category_pr(action='go_straight'), _ll(163, -42)) == (0.0, '')


def test_relation_and_target_evidence_for_across_candidate(street):
    la = {'L1': [CUTLET], '__target__': [MEAT_ACROSS]}
    pr = _chain_pr('across')
    cands = search.expand(pr, street, START, la)
    top = search._relation_and_target_evidence(street, pr, cands[0], la)
    assert any((e.data.get('relation_ok') is True for e in top)) and any((e.data.get('target') is True and e.supports for e in top))


def test_view_point_faces_building_from_path(toy):
    top = search.expand(_building_route('left'), toy, START)[0]
    views = search.view_points(toy, top)
    assert views and abs(views[0].heading_deg - 90) <= 3


def test_view_point_looks_into_alley(toy):
    top = search.expand(_pr(_route(walk_in=False), {'kind': 'alley'}), toy, START)[0]
    views = search.view_points(toy, top)
    assert views[0].at == _ll(150, 0) and abs(views[0].heading_deg - 180) <= 3


def test_match_landmarks_adds_kakao_evidence_and_bonus(monkeypatch, toy):
    pr = _gs25_route()
    cands = search.expand(pr, toy, START)
    before = cands[0].route_score
    monkeypatch.setattr(search.kakao, 'keyword', lambda budget, query, **kw: [Place(name='GS25', at=_ll(100, 10))])
    evidence = search.match_landmarks(pr, toy, cands)
    assert any((e.kind == 'kakao' and e.supports is True and (e.candidate_id == cands[0].id) for e in evidence))
    assert cands[0].route_score == pytest.approx(before + 1.0)


def test_match_landmarks_prefers_name_match_over_nearer_tenant(monkeypatch, toy):
    pr = _gs25_route()
    cands = search.expand(pr, toy, START)
    results = [Place(name='5G연구개발센터', at=_ll(100, 3)), Place(name='GS25 삼성점', at=_ll(100, 8))]
    monkeypatch.setattr(search.kakao, 'keyword', lambda budget, query, **kw: results)
    evidence = search.match_landmarks(pr, toy, cands)
    assert 'GS25 삼성점' in next((e.summary_ko for e in evidence if e.supports))


def test_target_landmark_is_checked_at_candidate_building(monkeypatch, toy):
    pr = _starbucks_building_route()
    cands = search.expand(pr, toy, START)
    before = cands[0].route_score
    monkeypatch.setattr(search.kakao, 'keyword', lambda budget, query, **kw: [Place(name='스타벅스', at=_ll(160, -50))])
    evidence = search.match_landmarks(pr, toy, cands)
    hit = next(e for e in evidence if e.candidate_id == cands[0].id)
    assert hit.supports is True and hit.data.get('target') is True
    assert cands[0].route_score == pytest.approx(before + 1.0)


def test_target_landmark_on_route_but_away_from_building_does_not_match(monkeypatch, toy):
    pr = _starbucks_building_route()
    cands = search.expand(pr, toy, START)
    before = cands[0].route_score
    monkeypatch.setattr(search.kakao, 'keyword', lambda budget, query, **kw: [Place(name='스타벅스', at=_ll(0, 0))])
    evidence = search.match_landmarks(pr, toy, cands)
    assert all((e.supports is not True for e in evidence)) and cands[0].route_score == pytest.approx(before)


def test_match_landmarks_marks_unsearched_as_unknown(toy):
    pr = _gs25_route()
    cands = search.expand(pr, toy, START)
    evidence = search.match_landmarks(pr, toy, cands, budget=search._Budget(limit=0))
    assert evidence and all((e.supports is None for e in evidence))


def test_named_target_skips_headquarters_and_falls_back_to_google(monkeypatch, toy):
    hq = Place(name='명륜진사갈비 본사', at=_ll(60, 0), category_name='서비스,산업 > 기업', address='서울 송파구 가락동 113')
    pr = _pr([], {'kind': 'named_place', 'name': '명륜진사갈비'})
    monkeypatch.setattr(search.kakao, 'keyword', lambda *a, **k: [hq])
    monkeypatch.setattr(search.kakao, 'address', lambda *a, **k: [])
    monkeypatch.setattr(
        search,
        'google_place',
        lambda name, near, rad: Place(name='명륜진사갈비', at=_ll(160, -50), category_name='음식점 > 한식 > 육류,고기'),
    )
    cands = search._named_candidates(pr, toy, None, [START], search._Budget())
    assert [(c.osm_id, c.name) for c in cands] == [(10, '명륜진사갈비')]


def test_find_candidates_named_place_from_station_exit(monkeypatch):
    places = {
        '삼성역 3번출구': Place(name='삼성역 3번출구', at=LatLng(37.50826, 127.06313)),
        '오토웨이 타워': Place(name='오토웨이타워', at=LatLng(37.50631, 127.063695)),
    }
    exit3 = {'evidence': '삼성역 3번 출구', 'query': '삼성역 3번 출구', 'kind': 'station_exit'}
    steps = [{'order': 1, 'action': 'go_straight', 'evidence': '직진', 'distance_class': 'none'}]
    pr = _pr(steps, {'kind': 'named_place', 'name': '오토웨이 타워', 'specific': True}, anchor=exit3)
    def fake_kw(budget, query, **kw):
        for k, v in places.items():
            if k in query or query in k:
                return [v]
        return []
    monkeypatch.setattr(search.kakao, 'keyword', fake_kw)
    out = search.find_candidates(pr)
    assert out['question'] is None and [a.name for a in out['anchors']] == ['삼성역 3번 출구']
    top = out['candidates'][0]
    assert (top.id, top.type, top.osm_id) == ('C1', 'named_place', 475633202) and top.view_points


def test_across_landmark_named_only_in_target_is_located_near_path_end(street):
    cands = search.expand(_dept_pr(True), street, START, {'L2': [MALL], 'L1': [DEPT]})
    assert cands and cands[0].osm_id == 22 and ('landmark_near_end' in cands[0].relaxations)


def test_across_generic_landmark_off_path_still_gives_no_candidates(street):
    assert search.expand(_dept_pr(False), street, START, {'L2': [MALL], 'L1': [DEPT]}) == []


def test_named_candidate_snaps_to_a_building_up_to_40m_away(monkeypatch, street):
    pr = _pr([], {'kind': 'named_place', 'name': 'Autoway Tower', 'specific': True})
    monkeypatch.setattr(search, 'en_resolve', lambda name, window, budget: Place(name='Autoway Tower', at=_ll(100, 38)))
    cands = search._named_candidates(pr, street, None, [], search._Budget())
    assert cands and cands[0].osm_id == 20 and (cands[0].geometry['type'] == 'Polygon')


def test_landmark_evidence_carries_the_place_position_for_the_map(street):
    la = {'L1': [CUTLET], '__target__': [MEAT_FAR, MEAT_ACROSS, MEAT_BESIDE]}
    cands = search.expand(_chain_pr('across'), street, START, la)
    ev = search.match_landmarks(_chain_pr('across'), street, cands, landmark_at=la)
    rel = [e for e in ev if e.candidate_id == cands[0].id and e.data.get('relation_ok')]
    assert rel and rel[0].data['lnglat'] == [round(CUTLET.at.lng, 7), round(CUTLET.at.lat, 7)]


def test_start_node_prefers_a_public_road_over_a_campus_service_dead_end():
    md = mapdata.MapData.from_parts(
        config.VENUE,
        {1: _ll(0, 10), 2: _ll(0, 70), 3: _ll(50, 0), 4: _ll(50, -100), 5: _ll(50, 100)},
        [(1, 2, 'service', 1), (3, 4, 'tertiary', 2), (3, 5, 'tertiary', 3)],
        [],
    )
    md2 = mapdata.MapData.from_parts(
        config.VENUE,
        {1: _ll(0, 10), 2: _ll(0, 70)},
        [(1, 2, 'service', 1)],
        [],
    )
    assert search._start_node(md, config.VENUE) == 3
    assert search._start_node(md2, config.VENUE) == 1


def test_straight_routes_to_a_known_landmark_off_the_straight_line():
    md = mapdata.MapData.from_parts(
        config.VENUE,
        {1: _ll(0, 0), 2: _ll(100, 0), 3: _ll(600, 0), 4: _ll(100, 120)},
        [(1, 2, 'residential', 1), (2, 3, 'residential', 1), (2, 4, 'residential', 2)],
        [],
    )
    spot = Place(name='동촌 본점', at=_ll(110, 115))
    raw = {
        'steps': [{'order': 1, 'action': 'go_straight', 'distance_class': 'none', 'until': {'landmark_ref': 'lm_1', 'relation': 'see'}}],
        'landmarks': [{'id': 'lm_1', 'evidence': '돈가스집', 'poi': {'name': '동촌 본점', 'specific': True}}],
    }
    pr = ParseResult(raw=raw, language='ko')
    hit = [p for p in search._run(pr, md, 1, raw['steps'], (), {'lm_1': [spot]}) if p.marks]
    assert hit and hit[0].nodes == [1, 2, 4] and ('route_to_landmark' in hit[0].relax)
    far = Place(name='동촌', at=_ll(500, 500))
    assert not [p for p in search._run(pr, md, 1, raw['steps'], (), {'lm_1': [far]}) if p.marks]


def test_point_target_at_a_known_store_routes_to_it_when_the_walk_misses():
    md = mapdata.MapData.from_parts(
        config.VENUE,
        {1: _ll(0, 0), 2: _ll(100, 0), 3: _ll(600, 0), 4: _ll(100, 150)},
        [(1, 2, 'residential', 1), (2, 3, 'residential', 1), (2, 4, 'residential', 2)],
        [],
    )
    store = Place(name='교보문고 강남점', at=_ll(108, 150))
    raw = {
        'steps': [{'order': 1, 'action': 'go_straight', 'distance_class': 'none'}],
        'landmarks': [{'id': 'L1', 'evidence': '교보문고 건물', 'poi': {'name': '교보문고', 'specific': True}}],
        'target': {'kind': 'point', 'name': '교보문고 앞'},
    }
    pr = ParseResult(raw=raw, language='ko')
    cands = search.expand(pr, md, START, {'L1': [store]})
    best = cands[0]
    assert best.type == 'point' and geo.dist_m(best.centroid, store.at) < 30 and ('route_to_landmark' in best.relaxations)


def test_named_chain_store_prefers_the_branch_matching_the_walk_and_side(monkeypatch):
    md = mapdata.MapData.from_parts(
        config.VENUE,
        {1: _ll(0, 0), 2: _ll(100, 0), 3: _ll(300, 0), 4: _ll(100, 160)},
        [(1, 2, 'residential', 1), (2, 3, 'residential', 1), (2, 4, 'residential', 2)],
        [],
    )
    off = Place(name='스타벅스 코엑스사거리점', at=_ll(100, 170))
    on = Place(name='스타벅스 봉은사역점', at=_ll(280, 20))
    monkeypatch.setattr(search, '_kw', lambda budget, q, near=None, radius_m=1000, size=15, category=None, page=1: [off, on])
    budget = search._Budget()
    pr = _pr([{'order': 1, 'action': 'go_straight', 'evidence': '직진', 'distance_class': 'none'}], {'kind': 'named_place', 'name': '스타벅스', 'side': 'left', 'specific': True})
    best = search._walk_fit(pr, md, START, [off, on], budget)
    assert best.name == '스타벅스 봉은사역점' and any(('걷기 적합도' in n for n in budget.notes))


def test_toward_bearing_comes_from_the_resolved_landmark(toy):
    east = {'L9': [Place(name='연세대학교', at=_ll(500, 0))]}
    north = {'L9': [Place(name='연세대학교', at=_ll(100, 400))]}
    assert abs(search._toward_deg(_toward_pr(), toy, _ll(100, 0), east) - 90) < 1
    assert abs(search._toward_deg(_toward_pr(), toy, _ll(100, 0), north)) < 1
    assert search._toward_deg(_toward_pr(), toy, _ll(100, 0), {}) is None
    assert search._toward_deg(_pr(_route(), {'kind': 'building'}), toy, _ll(100, 0), east) is None


def test_expand_starts_only_in_the_named_direction(toy):
    start = Anchor(name='N3', at=_ll(150, 0), source='kakao')
    south = {'L9': [Place(name='연세대학교', at=_ll(150, -500))]}
    east = {'L9': [Place(name='연세대학교', at=_ll(500, 0))]}
    assert {c.osm_id for c in search.expand(_toward_pr(), toy, start, south)} >= {10, 11}
    assert all((c.osm_id not in (10, 11) for c in search.expand(_toward_pr(), toy, start, east)))


def test_poi_pool_also_asks_kakao_for_each_core_term_within_the_category(monkeypatch):
    asked = []
    def fake_kw(budget, q, near=None, radius_m=1000, size=15, category=None, page=1):
        asked.append((q, category))
        return [Place(name=f'{q}_{category}_{page}', at=_ll(0, 0))]
    def fake_cat(budget, code, near=None, radius_m=1000, size=15, page=1):
        return [Place(name=f'cat_{code}_{page}', at=_ll(0, 0))]
    monkeypatch.setattr(search, '_kw', fake_kw)
    monkeypatch.setattr(search, '_cat', fake_cat)
    pool = search._poi_pool(search._Budget(), '인도 커리집', 'FD6', _ll(0, 0), 600)
    assert ('인도 커리집', None) in asked and ('커리', 'FD6') in asked and (('인도', 'FD6') in asked)
    assert len({p.name for p in pool}) == len(pool) >= 3


def test_shop_belongs_to_its_nearest_building_even_outside_15m(toy):
    spot = Place(name='커리146', at=_ll(163, -15), category_name='음식점 > 아시아음식 > 인도음식')
    pr = _pr([{'order': 1, 'action': 'go_straight', 'evidence': '쭉', 'distance_class': 'none'}], {'kind': 'building', 'name': '인도 커리집', 'specific': False, 'category_code': 'FD6'})
    cands = [Candidate(id='C1', type='building', building_way_id=10), Candidate(id='C2', type='building', building_way_id=11)]
    evs = []
    bonus = {}
    got = {}
    for c in cands:
        e = search._relation_and_target_evidence(toy, pr, c, None, target_spots=[spot])
        evs.extend(e)
        t_hit = any(x.supports for x in e if x.data.get('target'))
        got[c.id] = t_hit
        bonus[c.id] = 1.0 if t_hit else 0.0
    assert got == {'C1': True, 'C2': False} and bonus == {'C1': 1.0, 'C2': 0.0}


def test_named_target_prefers_the_place_whose_name_matches_over_tenants(monkeypatch, toy):
    tenant = Place(name='5G연구개발센터', at=_ll(163, -42), category_name='서비스,산업 > 연구소', address='오토웨이타워 9층')
    tower = Place(name='오토웨이타워', at=_ll(160, -40), category_name='부동산 > 빌딩')
    pr = _pr([{'order': 1, 'action': 'go_straight', 'evidence': '직진', 'distance_class': 'none'}], {'kind': 'named_place', 'name': '오토웨이 타워', 'specific': True})
    monkeypatch.setattr(search, '_kw', lambda *a, **k: [tenant, tower])
    monkeypatch.setattr(search, 'google_place', lambda *a, **k: None)
    out = search._named_candidates(pr, toy, None, [START], search._Budget())
    assert out[0].name == '오토웨이타워'
