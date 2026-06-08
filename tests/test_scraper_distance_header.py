"""Regression: results-page header parse must read distance even when venue
and distance share the SAME header cell.

The old parser broke out of the header-row loop on the venue match BEFORE the
distance regex ran, so headers like "Sha Tin, 1650M, Good, Class 4" left
distance at its 1200 default — silently corrupting whole meetings to 1200
(data-accuracy audit 2026-06-08). These lock the fix in data/scraper.py.
"""
from data.scraper import _parse_tables


def _results_table(extra_header_cell):
    """A minimal valid HKJC table_data: header table + one results table."""
    header_table = [[extra_header_cell]]
    results_table = [
        ["Pla", "Horse No", "Horse", "Jockey", "Trainer", "Act.Wt",
         "Horse Wt", "Dr", "LBW", "RunPos", "Time", "WinOdds"],
        ["1", "5", "TEST HORSE", "Z Purton", "J Size", "126",
         "1100", "4", "-", "5 4 1", "1:38.50", "3.2"],
        ["2", "7", "OTHER HORSE", "J Moreira", "C Fownes", "124",
         "1090", "2", "1-1/2", "3 2 2", "1:38.74", "5.5"],
    ]
    return [header_table, results_table]


def test_distance_parsed_when_venue_and_distance_same_cell():
    td = _results_table("Sha Tin, 1650M, Good, Class 4")
    race = _parse_tables(td, "2026/06/08", 1)
    assert race is not None
    assert race["venue"] == "Sha Tin"
    assert race["distance"] == 1650          # NOT the 1200 default
    assert race["going"] == "Good"


def test_distance_parsed_happy_valley_same_cell():
    td = _results_table("Happy Valley, 1200M, Good To Firm, Class 3")
    race = _parse_tables(td, "2026/06/08", 1)
    assert race["venue"] == "Happy Valley"
    assert race["distance"] == 1200
    assert race["going"] == "Good To Firm"


def test_stray_four_digit_does_not_override_distance():
    # A prize-money / year figure without a trailing 'M' must not be read as
    # distance, and an out-of-range value is rejected by the sanity bound.
    td = _results_table("Sha Tin, 2000M, Good, Class 2, HK$1,500,000")
    race = _parse_tables(td, "2026/06/08", 1)
    assert race["distance"] == 2000
