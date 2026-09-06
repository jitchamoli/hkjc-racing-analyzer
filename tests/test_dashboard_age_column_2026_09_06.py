"""The PER-HORSE DETAIL table must show horse age.

Age is fetched for every runner on a live card (120/120 on 2026/09/06) and is
already consumed by the model — `parse_age_score()` feeds the ML features and
`parse_age_int()` feeds the rule engine. It was simply absent from the
dashboard's detail-table column spec, so a field we collect and trust was
invisible to the person reading the card.

Two properties beyond "the column exists", both of which come from the way age
actually arrives:

1. FORMAT. The live HKJC race card writes `'4'`; Renavon rows in the DB write
   `'5.0'`. Both reach the template unchanged, so a naive render prints "5.0"
   for a five-year-old whenever a meeting is re-rendered from a DB-backed path.
   The cell must normalise to a plain integer, matching what `parse_age_int()`
   does for the rule engine.

2. COVERAGE. Age is 100% on live cards but only ~68% across the historical DB,
   degrading backwards (2021 = 20%). A horse with no age must render the same
   em-dash placeholder every other detail column uses, and — via the existing
   `activeCols` filter — the whole column must disappear when NO horse in the
   race has an age, rather than printing a column of dashes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.template import generate_html

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402


def _horse(number: int, rank: int, age):
    h = {
        'number': number, 'name': f'HORSE {number}', 'rank': rank,
        'win_odds': 5.0, 'place_odds': 2.0,
        'calibrated_prob': 0.2, 'pure_prob': 0.2, 'market_prob': 0.2,
        'ml_prob': 0.2, 'raw_ml_prob': 0.2, 'place_prob': 0.5,
        'rule_score': 60.0, 'edge_pp': 0.0, 'ev_win': 0.0, 'overlay': 1.0,
        'expected_return': 0.0, 'model_rank': rank, 'confidence': 'MED',
        'bet_recommendation': 'SKIP', 'edge_fires': False,
        'jockey': 'J', 'trainer': 'T', 'draw': number,
        # one always-present detail column so the table renders even when
        # every age is missing
        'last_6': '123456',
    }
    if age is not None:
        h['age'] = age
    return h


def _race(horses):
    return {
        'race_number': 1, 'time': '13:00', 'distance': 1200,
        'going': 'Good', 'class_type': 'Class 3', 'venue': 'Sha Tin',
        'course': 'A', 'conviction_tier': 'MED', 'has_edge': False,
        'results_available': False, 'horses': horses,
        'bet_ticket': {'version': 1, 'bets': [], 'total_combos': 0,
                       'allocated_amount': 0.0},
    }


def _detail_table(page, race_html_url):
    page.goto(race_html_url)
    page.wait_for_load_state('networkidle')
    page.click('[data-testid="race-card-R1"]')
    page.wait_for_selector('.detail-section', timeout=10000)
    for sec in page.query_selector_all('.detail-section'):
        label = sec.query_selector('.section-label')
        if label and 'PER-HORSE DETAIL' in label.inner_text():
            return sec
    raise AssertionError('PER-HORSE DETAIL section not rendered')


def _render(races, tmp_path: Path) -> str:
    html = generate_html(json.dumps(races), '2026-09-06 16:00:00',
                         '2026/09/06', venue='Sha Tin')
    out = tmp_path / 'dashboard.html'
    out.write_text(html)
    return f'file://{out.resolve()}'


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def _headers_and_rows(sec):
    heads = [th.inner_text().strip() for th in sec.query_selector_all('thead th')]
    rows = []
    for tr in sec.query_selector_all('tbody tr'):
        rows.append([td.inner_text().strip() for td in tr.query_selector_all('td')])
    return heads, rows


def test_age_column_is_rendered(browser, tmp_path):
    races = [_race([_horse(1, 1, '4'), _horse(2, 2, '6')])]
    page = browser.new_page()
    try:
        sec = _detail_table(page, _render(races, tmp_path))
        heads, rows = _headers_and_rows(sec)
        assert 'AGE' in heads, f'no AGE column in detail table: {heads}'
        col = heads.index('AGE')
        assert [r[col] for r in rows] == ['4', '6']
    finally:
        page.close()


def test_renavon_float_age_renders_as_an_integer(browser, tmp_path):
    """`'5.0'` from the DB must print as `5`, not `5.0`."""
    races = [_race([_horse(1, 1, '5.0'), _horse(2, 2, 7.0)])]
    page = browser.new_page()
    try:
        sec = _detail_table(page, _render(races, tmp_path))
        heads, rows = _headers_and_rows(sec)
        col = heads.index('AGE')
        assert [r[col] for r in rows] == ['5', '7']
    finally:
        page.close()


def test_missing_age_renders_the_placeholder(browser, tmp_path):
    """68% historical coverage — a horse with no age must not blank the cell."""
    races = [_race([_horse(1, 1, '4'), _horse(2, 2, None)])]
    page = browser.new_page()
    try:
        sec = _detail_table(page, _render(races, tmp_path))
        heads, rows = _headers_and_rows(sec)
        col = heads.index('AGE')
        assert [r[col] for r in rows] == ['4', '—']
    finally:
        page.close()


def test_column_disappears_when_no_horse_has_an_age(browser, tmp_path):
    """Re-rendering a 2021 meeting must not print a column of dashes."""
    races = [_race([_horse(1, 1, None), _horse(2, 2, None)])]
    page = browser.new_page()
    try:
        sec = _detail_table(page, _render(races, tmp_path))
        heads, _ = _headers_and_rows(sec)
        assert 'AGE' not in heads, f'empty AGE column still rendered: {heads}'
    finally:
        page.close()
