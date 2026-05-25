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


# ---------------------------------------------------------------------------
# STRIKE-PLACE / STRIKE-QPL — added 2026-05-25 (GPT-5.5 R8 Finding #1).
# _didBetHit() ignored PLACE and QIN PLACE pools; replay scored those bets
# but the dashboard badge silently said MISS on a real hit.
# ---------------------------------------------------------------------------

def _place_bet(number: int) -> dict:
    return {
        'label': 'PLACE',
        'pool': 'PLACE',
        'combos': 1,
        'legs': [{'rank': 1, 'number': number, 'name': f'HORSE{number}'}],
        'reason': 'recommender',
        'unit_stake': 10.0,
        'amount': 10.0,
    }


def _qpl_banker_bet(pivot: int, bankers: list[int]) -> dict:
    legs = [{'rank': 1, 'number': pivot, 'name': f'HORSE{pivot}'}]
    legs += [{'rank': i + 2, 'number': n, 'name': f'HORSE{n}'}
             for i, n in enumerate(bankers)]
    return {
        'label': 'QIN PLACE BANKER',
        'pool': 'QIN PLACE',
        'combos': len(bankers),
        'legs': legs,
        'reason': 'qin_place_banker',
        'unit_stake': 10.0,
        'amount': 10.0 * len(bankers),
    }


def test_strike_place_badge_renders_when_pick_places(browser: Browser, tmp_path):
    """PLACE bet on #1; horse #1 finishes 2nd (top-3). Badge must show
    STRIKE-PLACE — prior to 2026-05-25 _didBetHit had no PLACE branch and
    silently returned MISS even though validate_hk replay priced the win."""
    race = _make_race(
        race_number=8,
        bet_ticket_bets=[_place_bet(1)],
        horses_top4_positions=[3, 1, 5, 7],  # #1 finishes 2nd
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    badge = page.locator('[data-testid="strike-badge-R8"]')
    assert badge.count() == 1, 'expected exactly one STRIKE badge for R8'
    badge_text = badge.inner_text()
    assert 'PLACE' in badge_text, (
        f"expected STRIKE-PLACE on placed pick, got: {badge_text!r}"
    )
    assert 'MISS' not in badge_text
    page.close()


def test_strike_qpl_badge_renders_when_pivot_and_banker_both_place(browser: Browser, tmp_path):
    """QPL BANKER pivot=#1 + bankers=[#2, #3, #4]. Top-3 = [#1, #2, #5].
    Pivot (1) and banker (2) both placed → ≥1 hit pair → STRIKE-QPL."""
    race = _make_race(
        race_number=9,
        bet_ticket_bets=[_qpl_banker_bet(pivot=1, bankers=[2, 3, 4])],
        horses_top4_positions=[1, 2, 5, 6],
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    badge = page.locator('[data-testid="strike-badge-R9"]')
    assert badge.count() == 1
    badge_text = badge.inner_text()
    assert 'QPL' in badge_text, (
        f"expected STRIKE-QPL on placed QPL pair, got: {badge_text!r}"
    )
    page.close()


def test_strike_qpl_miss_when_pivot_off_board(browser: Browser, tmp_path):
    """Pivot=#1 falls off-board (5th); no pair can pay. Badge = MISS."""
    race = _make_race(
        race_number=10,
        bet_ticket_bets=[_qpl_banker_bet(pivot=1, bankers=[2, 3, 4])],
        horses_top4_positions=[2, 3, 4, 5],  # #1 off podium
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    badge_text = page.locator('[data-testid="strike-badge-R10"]').inner_text()
    assert 'QPL' not in badge_text, (
        f"QPL must NOT fire when pivot misses top-3, got: {badge_text!r}"
    )
    page.close()


def test_strike_multi_hit_badge_lists_all_with_priority_order(browser: Browser, tmp_path):
    """Multi-hit (2+ pool types hitting on the same race) now collapses
    to a single chip `★ STRIKE × N · A/B/C` with labels ordered by
    priority (WIN > QIN > QPL > F4-5 > BOX4 > TRIO5 > TRIO6 > PLACE).

    Previously only the highest-priority single STRIKE badge rendered
    and lower hits were invisible on the card header. After 2026-05-25
    (multi-hit support) all hits surface; priority still drives the
    order so QIN leads PLACE on this race."""
    race = _make_race(
        race_number=11,
        bet_ticket_bets=[_qin_bet([1, 2]), _place_bet(1)],
        horses_top4_positions=[1, 2, 5, 6],
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    badge_text = page.locator('[data-testid="strike-badge-R11"]').inner_text()
    # Multi-hit chip: must report count + both labels in priority order
    # (QIN first, PLACE after).
    assert '× 2' in badge_text, (
        f"expected multi-hit count `× 2`, got: {badge_text!r}"
    )
    qin_idx = badge_text.find('QIN')
    place_idx = badge_text.find('PLACE')
    assert qin_idx >= 0 and place_idx >= 0, (
        f"expected both QIN and PLACE in multi-hit chip, got: {badge_text!r}"
    )
    assert qin_idx < place_idx, (
        f"priority order violated — QIN must precede PLACE, got: {badge_text!r}"
    )
    page.close()


def test_strike_single_hit_keeps_simple_badge(browser: Browser, tmp_path):
    """Single-hit case must still render the simple `★ STRIKE · X` badge
    (not the multi-hit chip), so the visual scan stays compact when only
    one bet hits — backward compatibility with the pre-multi-hit UX."""
    race = _make_race(
        race_number=12,
        bet_ticket_bets=[_qin_bet([1, 2])],
        horses_top4_positions=[1, 2, 5, 6],
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    badge_text = page.locator('[data-testid="strike-badge-R12"]').inner_text()
    assert badge_text.strip() == '★ STRIKE · QIN', (
        f"single-hit should keep simple badge, got: {badge_text!r}"
    )
    page.close()


def test_per_bet_hit_marks_in_detail_panel(browser: Browser, tmp_path):
    """Detail panel BET TICKET must show ✓ HIT / ✗ MISS next to each
    bet label after results. R13 ticket: QIN [1,2] hits, F4 BOX-5
    [1,2,3,4,5] misses (top-4 = [1,2,5,9] — #9 outside). User can scan
    multi-bet tickets without manually checking each bet's legs."""
    race = _make_race(
        race_number=13,
        bet_ticket_bets=[_qin_bet([1, 2]), _f4_box5_bet([1, 2, 3, 4, 5])],
        horses_top4_positions=[1, 2, 5, 9],  # QIN hits, F4 misses (#9 not in legs)
        conviction_tier='EDGE',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    page.click('[data-testid="race-card-R13"]')
    page.wait_for_selector('[data-testid="bet-ticket-R13"]', timeout=3000)
    ticket_text = page.locator('[data-testid="bet-ticket-R13"]').first.inner_text()
    assert '✓ HIT' in ticket_text, (
        f"expected ✓ HIT mark in ticket, got: {ticket_text!r}"
    )
    assert '✗ MISS' in ticket_text, (
        f"expected ✗ MISS mark in ticket, got: {ticket_text!r}"
    )
    page.close()


def test_per_bet_hit_marks_in_wa_payload(browser: Browser, tmp_path):
    """WA copy mirrors the detail-panel marks in plain text. Same R13
    scenario — QIN hit, F4 BOX-5 missed — must reflect in payload."""
    race = _make_race(
        race_number=14,
        bet_ticket_bets=[_qin_bet([1, 2]), _f4_box5_bet([1, 2, 3, 4, 5])],
        horses_top4_positions=[1, 2, 5, 9],
        conviction_tier='EDGE',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    payload = page.locator('[data-testid="copy-wa-R14"]').first.get_attribute('data-payload') or ''
    assert '✓ HIT' in payload, (
        f"WA payload missing ✓ HIT, got: {payload[:400]!r}"
    )
    assert '✗ MISS' in payload, (
        f"WA payload missing ✗ MISS, got: {payload[:400]!r}"
    )
    page.close()


# ---------------------------------------------------------------------------
# Compact "Copy WA Ticket" payload — added 2026-05-25 (GPT-5.5 R14).
# Race-day bet-entry mode: ticket-first, HKJC pool names, frozen-ticket
# legs + amounts only. Must NEVER re-derive legs from rank order.
# ---------------------------------------------------------------------------

def test_copy_wa_ticket_button_renders_alongside_full(browser: Browser, tmp_path):
    """Both buttons (compact + full) must appear on every bet card."""
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
    assert page.locator('[data-testid="copy-wa-ticket-R8"]').count() == 1, (
        "Compact Copy WA Ticket button missing"
    )
    assert page.locator('[data-testid="copy-wa-R8"]').count() == 1, (
        "Full Copy WA button missing (back-compat)"
    )
    page.close()


def test_copy_wa_ticket_payload_renders_exact_frozen_legs_and_amounts(
        browser: Browser, tmp_path):
    """Compact ticket payload (frozen-bets branch) must:
       - Use HKJC-derived pool names (QIN BOX-4, FIRST 4 BOX-5)
       - List leg horse-numbers from frozen bet_ticket (not rank slice)
       - Show per-bet $ amount from bet.amount (not recomputed)
       - Show Budget / Ticket / Unit from frozen ticket fields
       - NOT include TOP PICK or NOTES (those bias entry).
    """
    bets = [
        _qin_bet([1, 3, 7, 10]),     # BOX QIN-4, amount 60, combos 6
        _f4_box5_bet([1, 3, 7, 10, 12]),  # F4 BOX-5, amount 50, combos 5
    ]
    race = _make_race(
        race_number=6,
        bet_ticket_bets=bets,
        results_available=False,
        conviction_tier='EDGE',
        field_size=12,
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    payload = page.locator(
        '[data-testid="copy-wa-ticket-R6"]'
    ).first.get_attribute('data-payload') or ''

    # Header + status guard
    assert payload.startswith('HKJC R6'), (
        f"compact payload missing HKJC R6 header: {payload[:200]!r}"
    )
    assert 'STATUS: LIVE' in payload, (
        f"missing STATUS: LIVE guard: {payload[:200]!r}"
    )

    # Stake block (frozen ticket has bets → STAKE must appear)
    assert 'STAKE' in payload, payload[:400]
    assert 'Budget $' in payload and 'Ticket $' in payload and 'Unit $' in payload, (
        f"missing Budget/Ticket/Unit: {payload[:400]!r}"
    )

    # BET TICKET block: numbered list, HKJC-derived pool names, exact legs
    assert 'BET TICKET' in payload, payload[:400]
    # Derived pool naming (R15 spec: not just BOX QIN-4 PROTECTION — every
    # regular box should use HKJC-style names too).
    assert 'QIN BOX-4' in payload, (
        f"compact must derive QIN BOX-4 from pool=QUINELLA, n=4: {payload[:400]!r}"
    )
    assert 'FIRST 4 BOX-5' in payload, (
        f"compact must derive FIRST 4 BOX-5 from pool=FIRST 4, n=5: {payload[:400]!r}"
    )
    # Per-bet amounts come from bet.amount
    assert '$60' in payload, f"missing $60 QIN amount: {payload[:400]!r}"
    assert '$50' in payload, f"missing $50 F4 amount: {payload[:400]!r}"
    # Exact frozen legs (not rank slices — note #10 and #12 are NOT in rank 1-5)
    for n in (1, 3, 7, 10, 12):
        assert f'#{n}' in payload, (
            f"compact payload missing leg #{n}: {payload[:500]!r}"
        )
    # #2/#4/#5 are NOT legs in this ticket, so they must be absent.
    for n in (2, 4, 5):
        assert f'#{n}' not in payload, (
            f"compact payload leaked non-ticket horse #{n}: {payload[:500]!r}"
        )

    # R15 spec: compact normal-bet payload must NOT carry TOP PICK / NOTES
    # (those belong in Full WA only — they bias entry at the terminal).
    assert 'TOP PICK' not in payload, (
        f"compact normal-bet payload must not carry TOP PICK: {payload[:500]!r}"
    )
    assert 'NOTES' not in payload, (
        f"compact normal-bet payload must not carry NOTES: {payload[:500]!r}"
    )
    page.close()


def test_copy_wa_ticket_qin_box4_label_derived_from_pool_and_legs(
        browser: Browser, tmp_path):
    """R15 #4 — the compact payload must DERIVE 'QIN BOX-4' from
    pool='QUINELLA' + n=4 legs, not from a hardcoded label table.
    Mirror assertion lives in a static contract test."""
    race = _make_race(
        race_number=12,
        bet_ticket_bets=[_qin_bet([5, 8, 11, 14])],  # BOX QIN-4
        results_available=False,
        conviction_tier='EDGE',
        field_size=14,
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')
    payload = page.locator(
        '[data-testid="copy-wa-ticket-R12"]'
    ).first.get_attribute('data-payload') or ''
    assert 'QIN BOX-4' in payload, (
        f"compact must render QIN BOX-4 (derived): {payload[:400]!r}"
    )
    # Exact legs surface verbatim
    for n in (5, 8, 11, 14):
        assert f'#{n}' in payload, (
            f"missing leg #{n} in QIN BOX-4 payload: {payload[:400]!r}"
        )
    page.close()


def test_copy_wa_ticket_2leg_qin_renders_as_straight_qin_not_qin_straight(
        browser: Browser, tmp_path):
    """Repo-wide convention is 'STRAIGHT QIN' / 'STRAIGHT TRIO' (adjective
    first) — generator at dashboard/generator.py:506,510 + WA fallback at
    template.py:942,945,1876,1878 all use that order. _hkjcPoolLabel must
    match so visual, generator-emitted, and compact-derived labels stay
    consistent in copy."""
    race = _make_race(
        race_number=13,
        bet_ticket_bets=[_qin_bet([4, 9])],  # 2-leg → STRAIGHT QIN
        results_available=False,
        conviction_tier='EDGE',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')
    payload = page.locator(
        '[data-testid="copy-wa-ticket-R13"]'
    ).first.get_attribute('data-payload') or ''
    assert 'STRAIGHT QIN' in payload, (
        f"compact must render 'STRAIGHT QIN' (adjective first): {payload[:400]!r}"
    )
    assert 'QIN STRAIGHT' not in payload, (
        f"compact must NOT render 'QIN STRAIGHT' (wrong order): {payload[:400]!r}"
    )
    page.close()


def test_copy_wa_ticket_no_system_bet_and_force_pick_when_empty(
        browser: Browser, tmp_path):
    """R15 #2 — when frozen bet_ticket is empty:
       - NO STAKE block
       - NO BET TICKET block
       - NO dollar amounts
       - NO SYSTEM BET banner
       - OPTIONAL FORCE PICK with top-3 horses
       - 'Manual only — not counted as model ROI ticket.' footer.
    """
    race = _make_race(
        race_number=7,
        bet_ticket_bets=[],
        results_available=False,
        conviction_tier='SKIP',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')
    payload = page.locator(
        '[data-testid="copy-wa-ticket-R7"]'
    ).first.get_attribute('data-payload') or ''

    # Positive: NO SYSTEM BET + OPTIONAL FORCE PICK + manual footer
    assert 'NO SYSTEM BET' in payload, (
        f"missing NO SYSTEM BET banner: {payload[:400]!r}"
    )
    assert 'OPTIONAL FORCE PICK' in payload, (
        f"missing OPTIONAL FORCE PICK block: {payload[:400]!r}"
    )
    assert 'Manual only' in payload and 'not counted as model ROI' in payload, (
        f"missing manual-only footer: {payload[:400]!r}"
    )
    # Top-3 horses present in force-pick block
    for n in (1, 2, 3):
        assert f'#{n}' in payload, (
            f"force-pick missing horse #{n}: {payload[:400]!r}"
        )

    # Negative: no STAKE block, no BET TICKET block, no stake-$ labels.
    # The OPTIONAL FORCE PICK block legitimately carries horse ODDS in
    # `$odds` form ("#X · $2.5 · rank · prob"), so we cannot blanket-reject
    # the `$` character — only the stake-context tokens.
    assert 'STAKE' not in payload, (
        f"no-bet payload must not carry STAKE block: {payload[:400]!r}"
    )
    assert 'BET TICKET' not in payload, (
        f"no-bet payload must not carry BET TICKET block: {payload[:400]!r}"
    )
    for stake_tag in ('Budget $', 'Ticket $', 'Unit $'):
        assert stake_tag not in payload, (
            f"no-bet payload leaked stake-context token {stake_tag!r}: "
            f"{payload[:400]!r}"
        )
    page.close()


def test_copy_wa_full_says_no_system_bet_when_ticket_empty(
        browser: Browser, tmp_path):
    """R15 #3 — Full WA payload (rationale-rich) must clearly say
    'NO SYSTEM BET' before any horse suggestions when the frozen
    ticket is empty. Old wording 'SKIP — no bet recommended' was
    ambiguous (user could read it as 'low conviction but still bet')."""
    race = _make_race(
        race_number=14,
        bet_ticket_bets=[],
        results_available=False,
        conviction_tier='SKIP',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')
    payload = page.locator(
        '[data-testid="copy-wa-R14"]'
    ).first.get_attribute('data-payload') or ''
    assert 'NO SYSTEM BET' in payload, (
        f"Full WA must say NO SYSTEM BET when ticket empty: {payload[:500]!r}"
    )


def test_copy_wa_full_uses_frozen_ticket_legs_not_rank_slices(
        browser: Browser, tmp_path):
    """R15 #5 — Full WA Playwright regression. Construct a ticket whose
    F4 BOX-5 legs include horse numbers OUTSIDE the top-5 rank slice
    (e.g. #11 instead of #5). The Full payload must list #11 (from
    frozen legs) and NOT #5 (which would only appear if the renderer
    reverted to race.horses.slice(0, 5))."""
    # Make a 12-horse field with the F4 BOX-5 legs intentionally
    # diverging from default top-5 rank order: numbers 1,3,7,10,11
    # are the FROZEN legs; #2/#4/#5 are NOT in the ticket.
    bets = [_f4_box5_bet([1, 3, 7, 10, 11])]
    race = _make_race(
        race_number=15,
        bet_ticket_bets=bets,
        results_available=False,
        conviction_tier='EDGE',
        field_size=12,
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')
    payload = page.locator(
        '[data-testid="copy-wa-R15"]'
    ).first.get_attribute('data-payload') or ''
    # Frozen legs must show up
    for n in (1, 3, 7, 10, 11):
        assert f'#{n}' in payload, (
            f"Full WA missing frozen leg #{n} of F4 BOX-5: {payload[:500]!r}"
        )
    # #2/#4/#5 must NOT show up in the F4 BOX-5 line. Look for them
    # specifically inside an F4 BOX-5 segment (anywhere in payload is
    # fine — but the simple field-wide check is OK because the test
    # fixture's other bets are absent: only F4 BOX-5 is in the ticket).
    for n in (2, 4, 5):
        assert f'#{n}' not in payload, (
            f"Full WA leaked non-leg horse #{n} (rank-slice fallback): "
            f"{payload[:500]!r}"
        )
    page.close()


def test_copy_wa_ticket_locked_when_results_available(browser: Browser, tmp_path):
    """LOCKED guard replaces LIVE once the race is resolved."""
    race = _make_race(
        race_number=9,
        bet_ticket_bets=[_qin_bet([1, 2, 3])],
        horses_top4_positions=[1, 2, 3, 4],
        results_available=True,
        conviction_tier='EDGE',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')
    payload = page.locator(
        '[data-testid="copy-wa-ticket-R9"]'
    ).first.get_attribute('data-payload') or ''
    assert 'STATUS: LOCKED' in payload, (
        f"missing LOCKED guard: {payload[:300]!r}"
    )
    page.close()


def test_hit_marks_absent_before_results(browser: Browser, tmp_path):
    """Live-prep messages (race not yet resolved) must NOT show ✗ MISS
    chips. Empty hit-mark when results_available is False."""
    race = _make_race(
        race_number=15,
        bet_ticket_bets=[_qin_bet([1, 2, 3])],
        results_available=False,
        conviction_tier='EDGE',
    )
    url = _render_to_file([race], tmp_path)
    page = browser.new_page()
    page.goto(url)
    page.wait_for_load_state('networkidle')

    page.click('[data-testid="race-card-R15"]')
    page.wait_for_selector('[data-testid="bet-ticket-R15"]', timeout=3000)
    ticket_text = page.locator('[data-testid="bet-ticket-R15"]').first.inner_text()
    payload = page.locator('[data-testid="copy-wa-R15"]').first.get_attribute('data-payload') or ''

    for marker in ('✓ HIT', '✗ MISS'):
        assert marker not in ticket_text, (
            f"unresolved race must not carry {marker} mark in ticket"
        )
        assert marker not in payload, (
            f"unresolved race must not carry {marker} mark in WA payload"
        )
    page.close()
