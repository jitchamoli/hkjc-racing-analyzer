"""Regression: race-card parser must populate race.course (the running rail).

Real HKJC race-info format is 'Turf, "C" Course, 1600M, Good To Firm' — comma-
separated, quoted rail (incl. e.g. "C+3"). The old regex expected a dash
('Turf - "A"'), so it captured only the surface and left race.course = '',
silently disabling the rail-aware DrawBiasAnalyzer (it fell back to the basic
draw scorer). This pins that the rail is parsed into race.course AND flows
through to DrawBiasAnalyzer._extract_rail. (2026-07-01 expert re-review.)
"""
import pytest

from core.fetcher import HKJCParser, HorseEntry
from analytics.draw_bias import DrawBiasAnalyzer


def _card_html(info_line: str) -> str:
    # Minimal page: the race-info text is read from the whole-page get_text().
    return f"<html><body><p>{info_line}</p></body></html>"


@pytest.fixture(autouse=True)
def _stub_horse_table(monkeypatch):
    # parse_race_card returns None unless horses parse; stub one so we can test
    # the race-info (rail) parsing in isolation from the 27-cell table logic.
    monkeypatch.setattr(HKJCParser, "_parse_horse_table",
                        staticmethod(lambda soup: [HorseEntry(number=1, name="TESTER")]))


@pytest.mark.parametrize("info,expected_rail", [
    ('July 01, 2026, Sha Tin, 16:00 Turf, "C" Course, 1600M, Good To Firm', "C"),
    ('Sha Tin, Turf, "A" Course, 1200M, Good', "A"),
    ('Happy Valley, Turf, "C+3" Course, 1650M, Good To Firm', "C+3"),
    ('Sha Tin, Turf, "A+2" Course, 1400M, Good', "A+2"),
])
def test_rail_parsed_into_course(info, expected_rail):
    rc = HKJCParser.parse_race_card(_card_html(info), 1, "2026/07/01")
    assert rc is not None
    assert rc.track == "Turf"
    assert rc.course == expected_rail, f"rail not parsed from {info!r}"
    # And it must be consumable by the draw analyzer.
    assert DrawBiasAnalyzer._extract_rail(rc.course) == expected_rail


def test_awt_has_no_rail():
    rc = HKJCParser.parse_race_card(
        _card_html('Sha Tin, All Weather Track, 1200M, Good'), 1, "2026/07/01")
    assert rc is not None
    assert rc.course == "", "AWT should carry no running rail"
