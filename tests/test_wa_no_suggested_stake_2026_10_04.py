"""WA copy must not suggest a stake size (2026-10-04).

User decides how much to put on each race from the Tier, so the message
carries no budget recommendation: no "Suggested stake" banner in the Full
WA, no "STAKE / Budget · Ticket · Unit" block in the compact ticket copy.

What stays: the Tier on the header line, the per-bet dollar amounts that
describe the ticket as priced, and the RARE EDGE banner on EDGE races.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from dashboard.template import generate_html

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright, Browser  # noqa: E402


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


def _race(tier: str) -> dict:
    horses = []
    for n in range(1, 9):
        horses.append({
            'number': n, 'name': f'HORSE{n}', 'rank': n,
            'win_odds': 2.0 + n, 'place_odds': 1.5 + n * 0.2,
            'pure_prob': 0.30 if n == 1 else 0.10,
            'market_prob': 0.10, 'calibrated_prob': 0.30 if n == 1 else 0.10,
            'edge_pp': 20.0 if n == 1 else 0.0,
            'ev_win': 0.5 if n == 1 else 0.0,
            'overlay': 1.2, 'edge_fires': (tier == 'EDGE' and n == 1),
            'bet_recommendation': 'SKIP', 'actual_position': None,
            'jockey': f'J{n}', 'trainer': f'T{n}', 'draw': n, 'weight': 130,
        })
    bets = [
        {'pool': 'QIN', 'label': 'BOX QIN-4 PROTECTION', 'combos': 6, 'amount': 60.0,
         'reason': 'recommender',
         'legs': [{'rank': i, 'number': i, 'name': f'HORSE{i}'} for i in (1, 2, 3, 4)]},
        {'pool': 'TRIO', 'label': 'BOX TRIO-5', 'combos': 10, 'amount': 100.0,
         'reason': 'recommender',
         'legs': [{'rank': i, 'number': i, 'name': f'HORSE{i}'} for i in (1, 2, 3, 4, 5)]},
    ]
    return {
        'race_number': 3, 'time': '13:45', 'venue': 'Sha Tin', 'distance': 1200,
        'class_type': 'Class 4', 'going': 'GOOD', 'horses': horses,
        'conviction_tier': tier, 'has_edge': tier == 'EDGE',
        'top_2_share': 0.40, 'top_3_share': 0.55, 'bankroll': 10000,
        'suggested_stake': {'amount': 300.0, 'pct': 3.0, 'method': f'tier-fraction ({tier})'},
        'results_available': False,
        'bet_ticket': {
            'version': 1, 'race_date': '2026/10/04', 'race_number': 3,
            'generated_at': '2026-10-04 13:00:00', 'model_version': 'test',
            'total_combos': 16, 'unit_stake': 10.0, 'allocated_amount': 160.0,
            'over_budget': False, 'over_amount': 0.0,
            'conviction_tier': tier, 'has_edge': tier == 'EDGE', 'bets': bets,
        },
        'actual_results': None, 'actual_winner': None, 'actual_top3': None,
    }


def _payloads(browser, tmp_path, tier):
    html = generate_html(json.dumps([_race(tier)]), '2026-10-04 13:00:00',
                         '2026/10/04', venue='Sha Tin')
    out = tmp_path / f'dash_{tier}.html'
    out.write_text(html)
    page = browser.new_page()
    page.goto(f"file://{out.resolve()}")
    page.wait_for_load_state('networkidle')
    full = page.locator('[data-testid="copy-wa-R3"]').first.get_attribute('data-payload') or ''
    compact = page.locator('[data-testid="copy-wa-ticket-R3"]').first.get_attribute('data-payload') or ''
    page.close()
    return full, compact


@pytest.mark.parametrize('tier', ['HIGH', 'EDGE'])
def test_full_wa_has_no_suggested_stake_banner(browser, tmp_path, tier):
    full, _ = _payloads(browser, tmp_path, tier)
    assert full, 'full WA payload empty — test would pass vacuously'
    assert 'Suggested stake' not in full, f"stake banner still in Full WA: {full[:400]!r}"
    assert 'HKD' not in full, f"stake banner remnant in Full WA: {full[:400]!r}"


@pytest.mark.parametrize('tier', ['HIGH', 'EDGE'])
def test_compact_wa_has_no_stake_block(browser, tmp_path, tier):
    _, compact = _payloads(browser, tmp_path, tier)
    assert compact, 'compact WA payload empty — test would pass vacuously'
    assert '*STAKE*' not in compact, f"STAKE block still in compact WA: {compact[:400]!r}"
    assert 'Budget $' not in compact, f"Budget line still in compact WA: {compact[:400]!r}"
    assert 'Unit $' not in compact, f"Unit line still in compact WA: {compact[:400]!r}"


@pytest.mark.parametrize('tier', ['HIGH', 'EDGE'])
def test_tier_and_ticket_amounts_survive(browser, tmp_path, tier):
    """What the user keeps: the Tier to size from, and the ticket as priced."""
    full, compact = _payloads(browser, tmp_path, tier)
    assert f'Tier: {tier}' in full, f"Tier line lost from Full WA: {full[:300]!r}"
    assert '$60' in compact and '$100' in compact, (
        f"per-bet amounts lost from compact WA: {compact[:400]!r}"
    )
    assert '6 combos' in compact and '10 combos' in compact


def test_edge_banner_survives_on_edge_race(browser, tmp_path):
    full, _ = _payloads(browser, tmp_path, 'EDGE')
    assert 'RARE EDGE' in full, f"EDGE banner lost: {full[:500]!r}"


def test_no_edge_banner_on_non_edge_race(browser, tmp_path):
    full, _ = _payloads(browser, tmp_path, 'HIGH')
    assert 'RARE EDGE' not in full
