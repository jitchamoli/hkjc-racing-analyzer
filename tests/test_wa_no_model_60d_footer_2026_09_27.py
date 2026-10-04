"""WA copy must not carry the `Model 60d` trust footer (2026-09-27).

The footer reported db.get_accuracy_summary(60) for the race's conviction
tier. That number is a SYNTHETIC one-bet-per-race measure taken from
prediction_log's rank-1 `bet_recommendation` in 1-unit-stake terms — it is
not the frozen `race_bet_ticket` the user actually stakes, so it contradicts
`validate_hk --replay` on the same meeting. On small buckets it also renders
figures like "EDGE tier: 3/3 (100%) · +319% ROI" straight into the message.

User asked for it gone. These tests pin its absence from every WA payload.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from dashboard.template import generate_html

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright, Browser  # noqa: E402


# An accuracy summary whose tier bucket is populated — this is what makes the
# footer render at all (it is silent when the bucket is missing or has 0 bets).
ACCURACY = {
    'by_conviction': {
        'EDGE': {'bets': 3, 'hits': 3, 'stake': 5.0, 'profit': 15.9},
        'HIGH': {'bets': 32, 'hits': 6, 'stake': 94.0, 'profit': -33.4},
    },
    'by_confidence': {},
}


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def _race(tier: str = 'EDGE') -> dict:
    horses = []
    for n in range(1, 11):
        horses.append({
            'number': n, 'name': f'HORSE{n}', 'rank': n,
            'win_odds': 2.0 + n * 0.5, 'place_odds': 1.5 + n * 0.2,
            'pure_prob': 0.10, 'market_prob': 0.10, 'calibrated_prob': 0.10,
            'edge_pp': 0.0, 'ev_win': 0.0, 'overlay': 1.0, 'edge_fires': False,
            'bet_recommendation': 'SKIP', 'actual_position': None,
            'jockey': f'J{n}', 'trainer': f'T{n}', 'draw': n, 'weight': 130,
        })
    bets = [{
        'pool': 'QIN', 'label': 'BOX QIN-4 PROTECTION', 'combos': 6,
        'amount': 60.0, 'reason': 'test',
        'legs': [{'rank': i, 'number': i, 'name': f'HORSE{i}'} for i in (1, 2, 3, 4)],
    }]
    return {
        'race_number': 8, 'time': '13:00', 'venue': 'Sha Tin', 'distance': 1200,
        'class_type': 'Class 4', 'going': 'GOOD', 'horses': horses,
        'conviction_tier': tier, 'has_edge': tier == 'EDGE',
        'top_2_share': 0.40, 'top_3_share': 0.55, 'bankroll': 10000,
        'suggested_stake': {'amount': 100.0, 'pct': 1.0, 'method': 'test'},
        'results_available': False,
        'bet_ticket': {
            'version': 1, 'race_date': '2026/09/27', 'race_number': 8,
            'generated_at': '2026-09-27 12:00:00', 'model_version': 'test',
            'total_combos': 6, 'unit_stake': 10.0, 'allocated_amount': 60.0,
            'conviction_tier': tier, 'has_edge': tier == 'EDGE', 'bets': bets,
        },
        'actual_results': None, 'actual_winner': None, 'actual_top3': None,
    }


def _render(tmp_path: Path, tier: str = 'EDGE') -> str:
    html = generate_html(
        json.dumps([_race(tier)]), '2026-09-27 14:00:00', '2026/09/27',
        venue='Sha Tin', accuracy_summary=ACCURACY,
    )
    out = tmp_path / 'dashboard.html'
    out.write_text(html)
    return f"file://{out.resolve()}"


@pytest.mark.parametrize('tier', ['EDGE', 'HIGH'])
def test_full_wa_payload_has_no_model_60d_footer(browser: Browser, tmp_path, tier):
    page = browser.new_page()
    page.goto(_render(tmp_path, tier))
    page.wait_for_load_state('networkidle')
    payload = page.locator('[data-testid="copy-wa-R8"]').first.get_attribute('data-payload') or ''
    page.close()

    assert payload, "full WA payload was empty — test would pass vacuously"
    assert 'Model 60d' not in payload, (
        f"full WA still carries the 60d trust footer: {payload[-300:]!r}"
    )
    assert '60d' not in payload, (
        f"full WA still reports a 60d figure: {payload[-300:]!r}"
    )


def test_compact_wa_ticket_payload_has_no_model_60d_footer(browser: Browser, tmp_path):
    page = browser.new_page()
    page.goto(_render(tmp_path))
    page.wait_for_load_state('networkidle')
    payload = page.locator('[data-testid="copy-wa-ticket-R8"]').first.get_attribute('data-payload') or ''
    page.close()

    assert payload, "compact WA payload was empty — test would pass vacuously"
    assert 'Model 60d' not in payload, (
        f"compact WA carries the 60d trust footer: {payload[-300:]!r}"
    )
