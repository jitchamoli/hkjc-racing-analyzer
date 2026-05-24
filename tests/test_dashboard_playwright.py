"""Browser-level regression tests for the betting-critical dashboard UI.

Added 2026-05-24 (GPT-5.5 Playwright audit). Locks the three contracts
that matter for real-money correctness:

1. STRIKE badge reflects the actual frozen bet_ticket, not synthetic
   top-N rank slices. The R8/R11 2026-05-24 phantom-hit class of bug
   manifested in the Past view as `STRIKE-F4-5` badges on races where
   F4 BOX-5 was gated out and never staked.
2. Visual BET TICKET shows exactly what's in race.bet_ticket — no
   client-side fabrication of bet rows.
3. Copy WA payload contains the same bet labels as the on-screen
   ticket (WA-vs-visual drift has bitten before).

Skipped automatically when Playwright's chromium isn't installed locally.
Run: `python3 -m pytest tests/test_dashboard_playwright.py -v`
"""
from __future__ import annotations
import json
import re
import tempfile
from pathlib import Path

import pytest

from dashboard.template import generate_html


pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright, Browser  # noqa: E402


def _render_to_file(races: list, tmp_path: Path) -> str:
    """Write a self-contained dashboard HTML to a tmp file and return file://."""
    html = generate_html(
        json.dumps(races),
        '2026-05-24 14:00:00',
        '2026/05/24',
        venue='Sha Tin',
    )
    out = tmp_path / 'dashboard.html'
    out.write_text(html)
    return f"file://{out.resolve()}"


def _make_race(
    *,
    race_number: int,
    bet_ticket_bets: list[dict],
    horses_top4_positions: list[int] | None = None,
    results_available: bool = True,
    conviction_tier: str = 'EDGE',
    field_size: int = 10,
) -> dict:
    """Build a minimal race dict.

    horses_top4_positions: list of horse numbers in finishing order (1st..4th)
    by horse `number`. The horse with `number=N` will be placed at rank N
    (rank 1 = horse #1, etc.) so the test can vary actual_position
    independently from predicted rank.
    """
    horses = []
    placements = horses_top4_positions or []
    for n in range(1, field_size + 1):
        actual_pos = placements.index(n) + 1 if n in placements else None
        horses.append({
            'number': n,
            'name': f'HORSE{n}',
            'rank': n,
            'win_odds': 2.0 + n * 0.5,
            'pure_prob': 0.10,
            'market_prob': 0.10,
            'calibrated_prob': 0.10,
            'edge_pp': 0.0,
            'ev_win': 0.0,
            'overlay': 1.0,
            'edge_fires': False,
            'bet_recommendation': 'SKIP',
            'actual_position': actual_pos,
            'jockey': f'J{n}',
            'trainer': f'T{n}',
            'draw': n,
            'weight': 130,
        })
    return {
        'race_number': race_number,
        'time': '13:00',
        'venue': 'Sha Tin',
        'distance': 1200,
        'class_type': 'Class 4',
        'going': 'GOOD',
        'horses': horses,
        'conviction_tier': conviction_tier,
        'has_edge': conviction_tier == 'EDGE',
        'top_2_share': 0.40,
        'top_3_share': 0.55,
        'bankroll': 10000,
        'suggested_stake': {'amount': 100.0, 'pct': 1.0, 'method': 'test'},
        'results_available': results_available,
        'bet_ticket': {
            'version': 1,
            'race_date': '2026/05/24',
            'race_number': race_number,
            'generated_at': '2026-05-24 12:00:00',
            'model_version': 'test',
            'total_combos': sum(int(b.get('combos', 0)) for b in bet_ticket_bets),
            'unit_stake': 10.0,
            'allocated_amount': sum(int(b.get('combos', 0)) for b in bet_ticket_bets) * 10.0,
            'conviction_tier': conviction_tier,
            'has_edge': conviction_tier == 'EDGE',
            'bets': bet_ticket_bets,
        },
        'actual_results': placements if results_available else None,
        'actual_winner': placements[0] if results_available and placements else None,
        'actual_top3': placements[:3] if results_available else None,
    }


def _qin_bet(numbers: list[int]) -> dict:
    return {
        'label': f'BOX QIN-{len(numbers)}' if len(numbers) > 2 else 'STRAIGHT QIN',
        'pool': 'QUINELLA',
        'combos': len(numbers) * (len(numbers) - 1) // 2 if len(numbers) > 2 else 1,
        'legs': [{'rank': i + 1, 'number': n, 'name': f'HORSE{n}'} for i, n in enumerate(numbers)],
        'reason': 'recommender',
        'unit_stake': 10.0,
        'amount': 10.0 * (len(numbers) * (len(numbers) - 1) // 2 if len(numbers) > 2 else 1),
    }


def _f4_box5_bet(numbers: list[int]) -> dict:
    return {
        'label': 'F4 BOX-5',
        'pool': 'FIRST 4',
        'combos': 5,
        'legs': [{'rank': i + 1, 'number': n, 'name': f'HORSE{n}'} for i, n in enumerate(numbers)],
        'reason': 'main_f4_gate',
        'unit_stake': 10.0,
        'amount': 50.0,
    }


@pytest.fixture(scope='module')
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:
            pytest.skip(f'chromium not installed: {e}')
        yield b
        b.close()


def test_dashboard_renders_race_cards_without_js_errors(browser: Browser, tmp_path):
    """Smoke test: page must render race cards without throwing JS errors.

    Caught a regression 2026-05-24 where a leftover `mainBetTypes` reference
    (variable removed during the WA refactor but one usage missed) crashed
    renderHero(), leaving the dashboard with only static chrome and no race
    cards. Playwright contract tests didn't catch it because their fixtures
    didn't exercise the t6 chaos path. This pins it."""
    race = _make_race(
        race_number=8,
        bet_ticket_bets=[_qin_bet([1, 2, 3, 4])],
        results_available=False,
        conviction_tier='HIGH',
        field_size=11,
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.goto(url)
    page.wait_for_load_state('networkidle')
    page.wait_for_timeout(300)
    assert errors == [], f"JS errors on render: {errors}"
    assert page.locator('[data-testid="race-card-R8"]').count() == 1, (
        'race card must render'
    )
    page.close()


def test_strike_badge_reads_from_bet_ticket_not_synthetic_top_n(browser: Browser, tmp_path):
    """R8/R11 2026-05-24 class of bug at the UI level: top-4 finishers fall
    inside top-5 ranks, but F4 BOX-5 is NOT in the bet_ticket. Old buggy
    _didBetHit() showed STRIKE-F4-5. Fixed version must show STRIKE-QIN
    (since BOX QIN-3 was the only ticket entry and the QIN pair hit)."""
    race = _make_race(
        race_number=8,
        bet_ticket_bets=[_qin_bet([1, 2, 3])],
        horses_top4_positions=[1, 2, 4, 5],
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    # Strike badge renders inside the bet-card header in the default
    # Race Order view — no tab switching needed.
    badge = page.locator('[data-testid="strike-badge-R8"]')
    assert badge.count() == 1, 'expected exactly one STRIKE badge for R8'
    badge_text = badge.inner_text()
    assert 'QIN' in badge_text, f"expected STRIKE-QIN, got: {badge_text!r}"
    assert 'F4' not in badge_text, (
        f"phantom STRIKE-F4-5 leaked — bet_ticket had no F4 BOX-5 but badge says: {badge_text!r}"
    )
    page.close()


def test_visual_bet_ticket_shows_frozen_ticket_bets(browser: Browser, tmp_path):
    """The on-screen BET TICKET must AT LEAST display every bet in
    race.bet_ticket.bets. (The strict converse — that the rendered ticket
    contains ONLY ticket-derived bets and no client-side fabrications — is
    asserted by the xfail test below; the detail-panel ticket currently
    builds its own ticketLines in JS instead of iterating race.bet_ticket.)
    """
    race = _make_race(
        race_number=8,
        bet_ticket_bets=[_qin_bet([1, 2, 3, 4])],
        results_available=False,
        conviction_tier='EDGE',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    page.click('[data-testid="race-card-R8"]')
    page.wait_for_selector('[data-testid="bet-ticket-R8"]', timeout=3000)
    ticket_text = page.locator('[data-testid="bet-ticket-R8"]').first.inner_text()
    assert 'BOX QIN-4' in ticket_text, f"ticket missing BOX QIN-4: {ticket_text!r}"
    page.close()


def test_visual_bet_ticket_has_no_client_side_fabrication(browser: Browser, tmp_path):
    """STRICT contract: rendered ticket must NOT include MAIN bets that
    aren't in race.bet_ticket. Locked in by 2026-05-24 ticket-driven
    refactor of _buildRaceDetailPanel — JS now iterates bet_ticket.bets
    with the gated extras suppressed when not in the ticket."""
    race = _make_race(
        race_number=8,
        bet_ticket_bets=[_qin_bet([1, 2, 3, 4])],
        results_available=False,
        conviction_tier='EDGE',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    page.click('[data-testid="race-card-R8"]')
    page.wait_for_selector('[data-testid="bet-ticket-R8"]', timeout=3000)
    ticket_text = page.locator('[data-testid="bet-ticket-R8"]').first.inner_text()
    assert '★ MAIN · F4 BOX-5' not in ticket_text
    assert '+ BOX TRIO-5' not in ticket_text
    page.close()


def test_copy_wa_payload_includes_bet_ticket_labels(browser: Browser, tmp_path):
    """Copy WA must AT LEAST include the bet labels from race.bet_ticket.
    (The strict converse — WA contains ONLY ticket labels — is asserted
    by the xfail test below, since the WA generator currently does its
    own gate logic; see test_copy_wa_payload_has_no_client_side_fabrication.)
    """
    race = _make_race(
        race_number=8,
        bet_ticket_bets=[
            _qin_bet([1, 2, 3, 4]),
            _f4_box5_bet([1, 2, 3, 4, 5]),
        ],
        results_available=False,
        conviction_tier='EDGE',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    payload = page.locator('[data-testid="copy-wa-R8"]').first.get_attribute('data-payload') or ''
    assert 'BOX QIN-4' in payload, f"WA payload missing BOX QIN-4: {payload[:300]!r}"
    assert 'F4 BOX-5' in payload, f"WA payload missing F4 BOX-5: {payload[:300]!r}"
    for n in (1, 2, 3, 4, 5):
        assert f'#{n}' in payload, f"WA payload missing horse #{n}: {payload[:300]!r}"
    page.close()


def test_copy_wa_payload_has_no_client_side_fabrication(browser: Browser, tmp_path):
    """STRICT contract: WA payload must NOT include MAIN bet labels that
    aren't in race.bet_ticket. Locked in by 2026-05-24 ticket-driven
    refactor of the WA generator — every MAIN/PROTECTION/SAVER emitter
    now gates on `_waTicketHas(label)`."""
    race = _make_race(
        race_number=8,
        bet_ticket_bets=[_qin_bet([1, 2, 3, 4])],  # ONLY QIN, no MAIN bets
        results_available=False,
        conviction_tier='EDGE',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    payload = page.locator('[data-testid="copy-wa-R8"]').first.get_attribute('data-payload') or ''
    assert 'BOX TRIO-5' not in payload, (
        f"WA fabricated BOX TRIO-5 not in ticket: {payload[:300]!r}"
    )
    assert 'F4 BOX-5' not in payload, (
        f"WA fabricated F4 BOX-5 not in ticket: {payload[:300]!r}"
    )
    page.close()
