"""The odds fetch must always identify WHICH meeting it wants.

Bug (code review 2026-08-03): `fetch_odds_html` accepted `race_date` but the
PRIMARY URL carried only `RaceNo`:

    DisplayOdds.aspx?RaceNo={race_no}&OddsType={odds_type}     # no RaceDate

and returned as soon as that responded 200 with >5000 bytes, so the dated
fallback below it was never reached. On any day other than the meeting being
scored — an offline re-render, a `--date` backfill, `generate_predictions.py
'2026/04/15'` run later — HKJC serves the CURRENT meeting's odds and those get
stamped onto the card. From there they feed implied_prob, the dynamic market
blend, pure_prob edge math and the frozen bet ticket, with nothing anywhere
asserting the date matched.

`fetch_race_card_html` and `fetch_results_html` both pass the date; only the
odds endpoint omitted it. The dated URL must be tried FIRST.
"""
import pytest

from core.fetcher import HKJCApiClient


class _Resp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        pass


class _RecordingSession:
    """Records every URL requested; returns a plausible full-size page."""
    def __init__(self):
        self.urls = []

    def get(self, url, **_kw):
        self.urls.append(url)
        return _Resp("x" * 6000)


@pytest.fixture
def client(monkeypatch):
    c = HKJCApiClient()
    session = _RecordingSession()
    monkeypatch.setattr(c, "session", session)
    return c, session


class TestOddsUrlCarriesRaceDate:
    def test_first_request_includes_the_requested_race_date(self, client):
        c, session = client
        c.fetch_odds_html("2026/04/15", 3, "WIN")

        assert session.urls, "no request was made"
        assert "RaceDate=2026/04/15" in session.urls[0], (
            f"first odds URL does not identify the meeting: {session.urls[0]}"
        )

    def test_race_no_and_odds_type_still_present(self, client):
        c, session = client
        c.fetch_odds_html("2026/04/15", 3, "PLA")

        assert "RaceNo=3" in session.urls[0]
        assert "OddsType=PLA" in session.urls[0]

    def test_every_attempted_url_is_dated(self, client, monkeypatch):
        """Even the fallbacks must not silently fetch a different meeting."""
        c, _ = client

        class _AllFail(_RecordingSession):
            def get(self, url, **_kw):
                self.urls.append(url)
                return _Resp("", status=404)

        session = _AllFail()
        monkeypatch.setattr(c, "session", session)
        c.fetch_odds_html("2026/04/15", 3, "WIN")

        assert session.urls, "no request was made"
        undated = [u for u in session.urls if "RaceDate=" not in u]
        assert not undated, f"undated odds URLs would fetch the wrong meeting: {undated}"
