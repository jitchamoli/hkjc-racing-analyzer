"""Place odds must actually reach the race card.

Bug (found 2026-09-14): 0 of 48,031 stored odds snapshots since May carried a
place price. Both legacy odds URLs are dead:

    racing/information/English/Racing/DisplayOdds.aspx  → 404 (moved to en-us route, also 404)
    racing/information/English/Odds/DisplayOdds.aspx    → 200 maintenance page (2,854 bytes)

`fetch_odds_html` requires >5,000 bytes, so it returned None for WIN and PLA on
every call. Win odds only survived because `_apply_tips_data` backfills them
from the Tips Index page — which has no place column — so `place_odds` stayed
0.0 on every runner, every meeting.

HKJC now serves pool odds from the bet.hkjc.com GraphQL endpoint. It rejects
any query text it has not whitelisted ("Internal server error -
WHITELIST_ERROR"); the query below is the one bet.hkjc.com itself sends,
captured 2026-09-14, and it succeeds from plain `requests`.

ASSUMPTION (unverified until pools open): `oddsNodes[].combString` is the
zero-padded runner number ("01") and `oddsValue` is a decimal string ("3.4").
The captured response had empty oddsNodes (pools INITIAL / STOP_SELL two days
before the meeting). Confirm on the first live card.
"""
import asyncio

import pytest

from core.fetcher import HKJCApiClient, HKJCDataFetcher, HKJCParser, HorseEntry, RaceCard


def _pool(odds_type, nodes):
    return {
        "id": f"MTG_20260916_0001{odds_type}1", "status": "START_SELL",
        "sellStatus": "START_SELL", "oddsType": odds_type,
        "leg": {"number": -1, "races": [1]}, "cWinSelections": [],
        "oddsNodes": [{"combString": c, "oddsValue": v, "hotFavourite": False,
                       "oddsDropValue": 0, "bankerOdds": []} for c, v in nodes],
    }


WIN_AND_PLA = {"data": {"raceMeetings": [{"pmPools": [
    _pool("PLA", [("01", "1.4"), ("02", "3.1"), ("03", "SCR"), ("04", "---")]),
    _pool("WIN", [("01", "2.2"), ("02", "9.5")]),
]}]}}


class TestParsePoolOdds:
    def test_reads_the_requested_pool_not_the_other(self):
        assert HKJCParser.parse_pool_odds(WIN_AND_PLA, "PLA") == {1: 1.4, 2: 3.1}
        assert HKJCParser.parse_pool_odds(WIN_AND_PLA, "WIN") == {1: 2.2, 2: 9.5}

    def test_scratched_and_unpriced_runners_are_skipped(self):
        pla = HKJCParser.parse_pool_odds(WIN_AND_PLA, "PLA")
        assert 3 not in pla and 4 not in pla

    @pytest.mark.parametrize("payload", [
        None,
        {},
        {"errors": [{"message": "Internal server error - WHITELIST_ERROR"}]},
        {"data": {"raceMeetings": [{"pmPools": [_pool("PLA", [])]}]}},   # pool not open yet
        {"data": {"raceMeetings": []}},
    ])
    def test_empty_or_error_payload_gives_no_odds(self, payload):
        assert HKJCParser.parse_pool_odds(payload, "PLA") == {}


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = "x"

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _PostSession:
    def __init__(self, payload):
        self.payload = payload
        self.posts = []

    def post(self, url, json=None, **_kw):
        self.posts.append((url, json))
        return _Resp(self.payload)


class TestFetchPoolOddsJson:
    def test_posts_the_whitelisted_racing_query_with_dashed_date(self, monkeypatch):
        c = HKJCApiClient()
        s = _PostSession(WIN_AND_PLA)
        monkeypatch.setattr(c, "session", s)

        out = c.fetch_pool_odds_json("2026/09/16", 3, "HV", ["PLA"])

        assert out == WIN_AND_PLA
        url, body = s.posts[0]
        assert url == "https://info.cld.hkjc.com/graphql/base/"
        assert body["operationName"] == "racing"
        assert body["variables"] == {"date": "2026-09-16", "venueCode": "HV",
                                     "raceNo": 3, "oddsTypes": ["PLA"]}
        # The server only accepts its own query text — guard the key parts.
        q = body["query"]
        assert "query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int)" in q
        assert "pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo)" in q
        assert "oddsNodes" in q and "combString" in q and "oddsValue" in q

    def test_graphql_error_returns_none(self, monkeypatch):
        c = HKJCApiClient()
        monkeypatch.setattr(c, "session", _PostSession(
            {"errors": [{"message": "Internal server error - WHITELIST_ERROR"}]}))
        assert c.fetch_pool_odds_json("2026/09/16", 1, "HV", ["PLA"]) is None

    def test_network_failure_returns_none(self, monkeypatch):
        c = HKJCApiClient()

        class _Boom:
            def post(self, *_a, **_kw):
                raise ConnectionError("down")

        monkeypatch.setattr(c, "session", _Boom())
        assert c.fetch_pool_odds_json("2026/09/16", 1, "HV", ["PLA"]) is None


class TestLiveFetchFillsPlaceOdds:
    def _race(self):
        return RaceCard(race_number=1, horses=[
            HorseEntry(number=1, name="A", win_odds=2.5),
            HorseEntry(number=2, name="B", win_odds=8.0),
        ])

    def test_place_odds_come_from_graphql_when_legacy_pages_are_dead(self, monkeypatch):
        f = HKJCDataFetcher(use_playwright=False)
        monkeypatch.setattr(f.api, "fetch_odds_html", lambda *a, **k: None)   # dead legacy URLs
        s = _PostSession(WIN_AND_PLA)
        monkeypatch.setattr(f.api, "session", s)
        race = self._race()

        asyncio.run(f._apply_odds_via_requests(race, "2026/09/16", 1, venue="HV"))

        assert [h.place_odds for h in race.horses] == [1.4, 3.1]
        assert s.posts and s.posts[0][1]["variables"]["venueCode"] == "HV"

    def test_win_odds_are_left_alone(self, monkeypatch):
        """Scope guard: this fix restores PLACE odds only. Win odds keep coming
        from their current source; switching that is a separate decision."""
        f = HKJCDataFetcher(use_playwright=False)
        monkeypatch.setattr(f.api, "fetch_odds_html", lambda *a, **k: None)
        monkeypatch.setattr(f.api, "session", _PostSession(WIN_AND_PLA))
        race = self._race()

        asyncio.run(f._apply_odds_via_requests(race, "2026/09/16", 1, venue="HV"))

        assert [h.win_odds for h in race.horses] == [2.5, 8.0]

    def test_get_race_card_passes_its_venue_through(self, monkeypatch):
        f = HKJCDataFetcher(use_playwright=False)
        seen = {}

        async def _spy(race, race_date, race_no, venue=""):
            seen["venue"] = venue

        async def _no_tips(*_a, **_k):
            return None

        monkeypatch.setattr(f.api, "fetch_race_card_html", lambda *a, **k: "<html/>")
        monkeypatch.setattr(HKJCParser, "parse_race_card",
                            staticmethod(lambda *a, **k: self._race()))
        monkeypatch.setattr(f, "_apply_odds_via_requests", _spy)
        monkeypatch.setattr(f, "_apply_tips_data", _no_tips)

        asyncio.run(f.get_race_card("2026/09/16", 1, venue="HV"))

        assert seen.get("venue") == "HV"
