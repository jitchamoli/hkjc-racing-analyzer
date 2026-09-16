"""get_live_odds / get_place_odds must survive the dead DisplayOdds pages too.

Sibling of test_place_odds_graphql_2026_09_14. The race-card path was fixed
there; these two helpers still asked the same 404/maintenance URLs and then
fell through to a Playwright fallback that navigates the SAME dead URL, so
both returned {} every time. Callers: the interactive odds refresh in
core/analyzer.py and data.odds_tracker.poll_odds_snapshot.
"""
import asyncio

from core.fetcher import HKJCDataFetcher

PAYLOAD = {"data": {"raceMeetings": [{"pmPools": [
    {"oddsType": "WIN", "oddsNodes": [{"combString": "01", "oddsValue": "2.2"},
                                      {"combString": "07", "oddsValue": "15.0"}]},
    {"oddsType": "PLA", "oddsNodes": [{"combString": "01", "oddsValue": "1.3"},
                                      {"combString": "07", "oddsValue": "4.1"}]},
]}]}}


class _Resp:
    status_code = 200
    text = "x"

    def json(self):
        return PAYLOAD

    def raise_for_status(self):
        pass


class _PostSession:
    def __init__(self):
        self.posts = []

    def post(self, url, json=None, **_kw):
        self.posts.append((url, json))
        return _Resp()


def _fetcher(monkeypatch):
    f = HKJCDataFetcher(use_playwright=False)
    monkeypatch.setattr(f.api, "fetch_odds_html", lambda *a, **k: None)   # dead pages
    s = _PostSession()
    monkeypatch.setattr(f.api, "session", s)
    return f, s


def test_live_win_odds_fall_back_to_the_pool_api(monkeypatch):
    f, s = _fetcher(monkeypatch)
    assert asyncio.run(f.get_live_odds("2026/09/16", 1, venue="HV")) == {1: 2.2, 7: 15.0}
    assert s.posts[0][1]["variables"]["oddsTypes"] == ["WIN"]


def test_place_odds_fall_back_to_the_pool_api(monkeypatch):
    f, s = _fetcher(monkeypatch)
    assert asyncio.run(f.get_place_odds("2026/09/16", 1, venue="HV")) == {1: 1.3, 7: 4.1}
    assert s.posts[0][1]["variables"]["oddsTypes"] == ["PLA"]


def test_venue_is_detected_when_the_caller_does_not_pass_one(monkeypatch):
    """Existing callers pass no venue; the query needs one to return a meeting."""
    f, s = _fetcher(monkeypatch)
    monkeypatch.setattr(f.api, "detect_venue", lambda _d: "ST")

    asyncio.run(f.get_live_odds("2026/09/13", 4))

    assert s.posts[0][1]["variables"]["venueCode"] == "ST"


def test_working_html_page_is_still_preferred(monkeypatch):
    """If HKJC ever restores the HTML page, use it and skip the API call."""
    f, s = _fetcher(monkeypatch)
    monkeypatch.setattr(f.api, "fetch_odds_html",
                        lambda *a, **k: '<table><tr><td>3</td><td>9.9</td></tr></table>')

    assert asyncio.run(f.get_live_odds("2026/09/16", 1, venue="HV")) == {3: 9.9}
    assert s.posts == []
