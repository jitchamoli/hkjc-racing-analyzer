"""Venue detection for the barrier-trial scraper (2026-06-07 fix).

The first backfill over-tagged Sha Tin trials as Conghua because every page
carries "Conghua Racecourse" nav links. Venue must come from the
"<Venue> Barrier Trial" header only.
"""
from scripts.scrape_trials import _detect_venue

_CONGHUA_NAV = ('<a href="https://crc.hkjc.com/en-US/" rel="Conghua Racecourse" '
                'target="_blank">Conghua Racecourse</a>') * 7


def test_sha_tin_header_beats_conghua_nav():
    html = f'<div class="rcheader"> Sha Tin Barrier Trial </div>{_CONGHUA_NAV}'
    assert _detect_venue(html) == "ST"


def test_conghua_header():
    html = f'<div class="rcheader"> Conghua Barrier Trial </div>{_CONGHUA_NAV}'
    assert _detect_venue(html) == "Conghua"


def test_happy_valley_header():
    html = f'<div class="rcheader"> Happy Valley Barrier Trial </div>{_CONGHUA_NAV}'
    assert _detect_venue(html) == "HV"


def test_no_header_returns_blank():
    assert _detect_venue(_CONGHUA_NAV) == ""   # nav links alone must NOT match
    assert _detect_venue("") == ""
    assert _detect_venue(None) == ""
