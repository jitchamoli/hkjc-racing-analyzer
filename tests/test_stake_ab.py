"""scripts/stake_ab — _pnl() formula + firing_bets() triple gate.

Test-engineer expert review 2026-05-23 priority #1: this is the core P&L
math for every forward-validation stake variant. Dividend-indexing bugs
(div-per-$10 vs div-per-$1) would silently 10×-scale every ROI.
"""
import csv
from unittest.mock import patch

from scripts.stake_ab import _append_log, _pnl, firing_bets


# ---------------------------------------------------------------------------
# _pnl() — stake distributed across N combos, dividend per $1 on winning combo
# ---------------------------------------------------------------------------

def test_pnl_zero_stake_returns_zero():
    assert _pnl(stake=0, combos=10, hit=True, div_per_1=5.0) == 0.0
    assert _pnl(stake=0, combos=10, hit=False, div_per_1=5.0) == 0.0


def test_pnl_miss_loses_full_stake():
    assert _pnl(stake=100, combos=10, hit=False, div_per_1=5.0) == -100.0


def test_pnl_hit_box_trio5_at_known_dividend():
    # BOX TRIO-5: stake $100 across 10 combos = $10/combo.
    # Dividend $39.60 per $1 → winning combo returns $10 × $39.60 = $396.
    # Net P&L = $396 - $100 stake = +$296.
    # (This is the R6 2026/05/20 scenario from the gate_impact backfill.)
    pnl = _pnl(stake=100, combos=10, hit=True, div_per_1=39.6)
    assert abs(pnl - 296.0) < 0.01


def test_pnl_hit_straight_qin():
    # STRAIGHT QIN: stake $10, 1 combo, dividend $8 per $1 → return $80.
    # P&L = $80 - $10 = +$70.
    pnl = _pnl(stake=10, combos=1, hit=True, div_per_1=8.0)
    assert abs(pnl - 70.0) < 0.01


def test_pnl_hit_dividend_below_breakeven_still_negative():
    # BOX TRIO-5 hit but div is small ($5/$1 stake) → return per combo $5 ×
    # ($100/10) = $50 → P&L = $50 - $100 = -$50 (a "hit" but still a loss).
    pnl = _pnl(stake=100, combos=10, hit=True, div_per_1=5.0)
    assert abs(pnl - (-50.0)) < 0.01


def test_pnl_zero_dividend_treated_as_miss():
    # If dividend lookup fails (returns 0), we shouldn't pretend the bet hit.
    assert _pnl(stake=100, combos=10, hit=True, div_per_1=0.0) == -100.0


# ---------------------------------------------------------------------------
# firing_bets() — triple gate (heavy fav / LOW or SKIP / small field)
# ---------------------------------------------------------------------------

def _race(top1_odds: float, tier: str, n_horses: int, with_qin_horses: int = 4):
    """Build a synthetic race that has `with_qin_horses` horses with
    bet_recommendation=QIN so firing_bets has a non-empty fires dict to
    expand on top of."""
    horses = [{'win_odds': top1_odds}]
    horses += [{'win_odds': top1_odds + i + 1} for i in range(n_horses - 1)]
    for h in horses[:with_qin_horses]:
        h['bet_recommendation'] = 'QIN'
    return {'horses': horses, 'conviction_tier': tier}


def test_firing_bets_normal_med_race_fires_both():
    # Normal MED race, 10 horses, top1 $3.0 — both BOX TRIO-5 and F4 BOX-5 fire.
    fires = firing_bets(_race(top1_odds=3.0, tier='MED', n_horses=10))
    assert 'BOX TRIO-5' in fires
    assert 'F4 BOX-5' in fires
    assert fires['BOX TRIO-5'] == 10
    assert fires['F4 BOX-5'] == 5


def test_firing_bets_fires_on_heavy_fav_2026_05_24():
    """After 2026-05-24 ungate (heavy-fav-skip-ungated-2026-05-24), heavy
    favorites no longer suppress BOX TRIO-5 / F4 BOX-5. Tier + field-size
    gates still apply; heavy-fav clause is gone."""
    fires = firing_bets(_race(top1_odds=2.2, tier='HIGH', n_horses=10))
    assert 'BOX TRIO-5' in fires
    assert 'F4 BOX-5' in fires


def test_firing_bets_skips_on_low_conviction():
    fires = firing_bets(_race(top1_odds=5.0, tier='LOW', n_horses=10))
    assert 'BOX TRIO-5' not in fires
    assert 'F4 BOX-5' not in fires


def test_firing_bets_skips_on_skip_conviction():
    fires = firing_bets(_race(top1_odds=5.0, tier='SKIP', n_horses=10))
    assert 'BOX TRIO-5' not in fires
    assert 'F4 BOX-5' not in fires


def test_firing_bets_skips_on_small_field():
    fires = firing_bets(_race(top1_odds=5.0, tier='MED', n_horses=7))
    assert 'BOX TRIO-5' not in fires
    assert 'F4 BOX-5' not in fires


def test_firing_bets_boundary_at_field_8():
    # Field of exactly 8 should fire (gate is `< 8`).
    fires = firing_bets(_race(top1_odds=5.0, tier='MED', n_horses=8))
    assert 'BOX TRIO-5' in fires


def test_firing_bets_boundary_at_top1_odds_2_5():
    # top1 of exactly $2.50 should fire (gate is `< 2.5`).
    fires = firing_bets(_race(top1_odds=2.5, tier='MED', n_horses=10))
    assert 'BOX TRIO-5' in fires


def test_firing_bets_edge_tier_fires():
    fires = firing_bets(_race(top1_odds=5.0, tier='EDGE', n_horses=10))
    assert 'BOX TRIO-5' in fires


def test_firing_bets_high_tier_fires():
    fires = firing_bets(_race(top1_odds=5.0, tier='HIGH', n_horses=10))
    assert 'BOX TRIO-5' in fires


# ---------------------------------------------------------------------------
# _append_log — atomic write (no torn file on crash mid-rewrite)
# ---------------------------------------------------------------------------

def _row(date='2026/05/24', variant='flat_min'):
    return {
        'race_date': date, 'race_number': 1, 'variant': variant,
        'bet_label': 'BOX TRIO-5', 'combos': 10, 'stake_hkd': 100,
        'hit': 0, 'dividend_per_1': 0, 'pnl_hkd': -100,
    }


def test_append_log_writes_via_tmp_then_replace(tmp_path, monkeypatch):
    """Atomic-write contract: the function must write to a .tmp sibling
    and then os.replace it, so a crash mid-rewrite cannot zero the file."""
    log_path = tmp_path / 'stake_ab_log.csv'
    monkeypatch.setattr('scripts.stake_ab.LOG_PATH', log_path)

    captured_tmp_paths = []
    real_replace = type(log_path).replace

    def spy_replace(self, target):
        captured_tmp_paths.append(str(self))
        return real_replace(self, target)

    monkeypatch.setattr(type(log_path), 'replace', spy_replace)
    _append_log([_row()], '2026/05/24')

    assert log_path.exists(), 'final log file should exist after _append_log'
    assert any(p.endswith('.csv.tmp') for p in captured_tmp_paths), (
        f'expected a .csv.tmp intermediate; got {captured_tmp_paths!r}. '
        f'A direct open(LOG_PATH, "w") truncates before write — non-atomic.'
    )


def test_append_log_no_tmp_left_behind_on_success(tmp_path, monkeypatch):
    log_path = tmp_path / 'stake_ab_log.csv'
    monkeypatch.setattr('scripts.stake_ab.LOG_PATH', log_path)
    _append_log([_row()], '2026/05/24')
    assert not (tmp_path / 'stake_ab_log.csv.tmp').exists(), (
        'tmp file should be moved into place, not left as a sibling'
    )


def test_append_log_idempotent_re_run_drops_prior_rows_for_date(tmp_path, monkeypatch):
    log_path = tmp_path / 'stake_ab_log.csv'
    monkeypatch.setattr('scripts.stake_ab.LOG_PATH', log_path)
    _append_log([_row(date='2026/05/24', variant='flat_min')], '2026/05/24')
    _append_log([_row(date='2026/05/24', variant='flat_min')], '2026/05/24')
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f'expected idempotent re-run, got {len(rows)} rows'


def test_append_log_partial_backfill_preserves_other_variants(tmp_path, monkeypatch):
    """If a single variant is re-run for an old date (e.g. backfilling
    flat_min_pre_chaos after it was added), the existing rows for OTHER
    variants on that date must survive. Drop only rows where (date, variant)
    matches the new rows."""
    log_path = tmp_path / 'stake_ab_log.csv'
    monkeypatch.setattr('scripts.stake_ab.LOG_PATH', log_path)
    _append_log([
        _row(date='2026/05/24', variant='flat_min'),
        _row(date='2026/05/24', variant='flat_1pct'),
        _row(date='2026/05/24', variant='invert'),
    ], '2026/05/24')

    # Partial backfill: only flat_min_pre_chaos for the same date.
    _append_log([_row(date='2026/05/24', variant='flat_min_pre_chaos')],
                '2026/05/24')

    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    variants = sorted(r['variant'] for r in rows)
    assert variants == ['flat_1pct', 'flat_min', 'flat_min_pre_chaos', 'invert'], (
        f'partial backfill stripped sibling variants; got {variants!r}'
    )


# ---------------------------------------------------------------------------
# Ticket-driven firing (GPT-5.5 R9 2026-05-25)
# firing_bets() must prefer race.bet_ticket.bets when present; only fall
# back to legacy reconstruction for pre-bet_ticket JSONs. This closes the
# drift that had stake_ab over-firing BOX TRIO-5 + F4 BOX-5 (and missing
# BOX QIN-4 PROTECTION + F4 SAVER) vs the actual frozen ticket.
# ---------------------------------------------------------------------------

def _race_with_ticket(bets: list[dict], horses: int = 10,
                       conviction_tier: str = 'MED') -> dict:
    """Build a race that has BOTH a frozen bet_ticket AND horse-level
    bet_recommendations that would drive a DIFFERENT legacy reconstruction.
    Used to prove ticket wins over reconstruction."""
    horse_list = [{'win_odds': 3.0 + i, 'rank': i + 1, 'number': i + 1,
                   'name': f'H{i+1}',
                   # All horses have rec=QIN so legacy reconstruction would
                   # produce a giant BOX QIN-N spanning the whole field.
                   'bet_recommendation': 'QIN'} for i in range(horses)]
    return {
        'horses': horse_list,
        'conviction_tier': conviction_tier,
        'bet_ticket': {'bets': bets},
    }


def test_firing_bets_prefers_frozen_ticket_over_legacy_reconstruction():
    """When bet_ticket exists, firing_bets must report exactly the ticket
    labels — even when horse-level bet_recommendation would re-derive
    something completely different. This closes the drift identified by
    GPT-5.5 R9 (over-firing BOX TRIO-5/F4 BOX-5 on real meetings).

    Without the fix: 10 horses all rec=QIN reconstructs as BOX QIN-10
    (45 combos) + BOX TRIO-5 + F4 BOX-5 because MED+field≥8 triple gate
    passes. With the fix: just whatever's actually in the ticket."""
    race = _race_with_ticket(
        bets=[{'label': 'BOX QIN-3', 'combos': 3}],
    )
    fires = firing_bets(race)
    assert fires == {'BOX QIN-3': 3}, (
        f"ticket-first failed; got {fires!r}"
    )


def test_firing_bets_empty_ticket_returns_empty_not_legacy():
    """A ticket with no bets means the recommender chose to skip the race
    — that's the ground truth, not a signal to fall back to legacy. Key
    distinction: empty BET LIST in a dict-typed ticket = skip; missing
    bet_ticket key entirely = legacy JSON. This guards SKIP-tier races
    from being silently bet by the fallback path."""
    race = _race_with_ticket(bets=[])
    fires = firing_bets(race)
    assert fires == {}, (
        f"empty ticket must NOT trigger legacy reconstruction; got {fires!r}"
    )


def test_firing_bets_missing_ticket_falls_back_to_legacy():
    """No bet_ticket key (legacy JSON pre-ticket-system) → legacy
    reconstruction. Preserves backward compatibility for old race JSONs
    that were saved before build_bet_ticket existed."""
    race = _race_with_ranks(top1_odds=3.0, tier='MED', n_horses=10)
    # No bet_ticket key. Legacy path should fire MED/field=10 triple gate.
    fires = firing_bets(race)
    assert 'BOX TRIO-5' in fires
    assert 'F4 BOX-5' in fires


def test_firing_bets_ticket_first_preserves_chaos_filter():
    """CHAOS filter must still apply even when ticket is present —
    otherwise the flat_min_pre_chaos counterfactual variant breaks."""
    race = _race_with_ticket(
        bets=[{'label': 'BOX QIN-3', 'combos': 3}],
    )
    # SOFT going + Class 4 should trigger CHAOS detect → no fires regardless
    # of what's in the ticket. (CHAOS suppression sits outside the recommender;
    # it's a stake_ab forward-validation overlay.)
    race['going'] = 'SOFT'
    race['class_type'] = 'Class 4'
    fires_chaos_on = firing_bets(race, apply_chaos=True)
    fires_chaos_off = firing_bets(race, apply_chaos=False)
    assert fires_chaos_on == {}, (
        f"CHAOS must suppress even with ticket; got {fires_chaos_on!r}"
    )
    assert fires_chaos_off == {'BOX QIN-3': 3}, (
        f"CHAOS-off must read ticket; got {fires_chaos_off!r}"
    )


def test_firing_bets_ticket_first_captures_server_side_addons():
    """The whole point of the fix: server-side add-ons emitted by
    build_bet_ticket (BOX QIN-4 PROTECTION, F4 SAVER, QIN PLACE BANKER)
    must surface in firing_bets even though they have no horse-level
    bet_recommendation backing. Legacy reconstruction silently dropped
    these — ticket-first picks them up correctly."""
    race = _race_with_ticket(
        bets=[
            {'label': 'BOX QIN-3', 'combos': 3},
            {'label': 'BOX QIN-4 PROTECTION', 'combos': 6},
            {'label': 'F4 SAVER', 'combos': 1},
            {'label': 'QIN PLACE BANKER', 'combos': 3},
        ],
    )
    fires = firing_bets(race)
    expected = {
        'BOX QIN-3': 3,
        'BOX QIN-4 PROTECTION': 6,
        'F4 SAVER': 1,
        'QIN PLACE BANKER': 3,
    }
    assert fires == expected, f"server-side add-ons dropped; got {fires!r}"


# ---------------------------------------------------------------------------
# QIN PLACE BANKER firing (GPT-5.5 R8 Finding #2 2026-05-25)
# ---------------------------------------------------------------------------

def _race_with_ranks(top1_odds: float, tier: str, n_horses: int):
    """Build a synthetic race with explicit rank numbers (1..n) so the
    rank-based sort in firing_bets() picks the right top1. Required for
    QIN PLACE BANKER firing tests."""
    horses = [
        {'win_odds': top1_odds + i,
         'rank': i + 1,
         'number': i + 1,
         'name': f'H{i+1}'}
        for i in range(n_horses)
    ]
    return {'horses': horses, 'conviction_tier': tier}


def test_firing_bets_emits_qin_place_banker_on_short_fav():
    """Mirrors dashboard/generator.py:432-439: top1 ≤ $1.8 AND ≥4 horses
    triggers a 3-combo QIN PLACE BANKER. Prior to 2026-05-25 stake_ab
    didn't emit this and silently undercounted ROI on heavy-fav cards."""
    fires = firing_bets(_race_with_ranks(top1_odds=1.5, tier='HIGH', n_horses=10))
    assert fires.get('QIN PLACE BANKER') == 3, (
        f"QIN PLACE BANKER must fire 3 combos on top1=$1.5; got {fires!r}"
    )


def test_firing_bets_no_qpl_banker_above_1_8_threshold():
    fires = firing_bets(_race_with_ranks(top1_odds=1.9, tier='HIGH', n_horses=10))
    assert 'QIN PLACE BANKER' not in fires, (
        f"QPL BANKER must NOT fire when top1 > $1.8; got {fires!r}"
    )


def test_firing_bets_no_qpl_banker_with_under_4_horses():
    fires = firing_bets(_race_with_ranks(top1_odds=1.5, tier='HIGH', n_horses=3))
    assert 'QIN PLACE BANKER' not in fires, (
        f"QPL BANKER needs ≥4 horses (pivot + 3 bankers); got {fires!r}"
    )


def test_firing_bets_qpl_banker_at_1_8_boundary():
    """top1 EXACTLY $1.8 fires (gate is `≤ 1.8`)."""
    fires = firing_bets(_race_with_ranks(top1_odds=1.8, tier='HIGH', n_horses=10))
    assert fires.get('QIN PLACE BANKER') == 3


# ---------------------------------------------------------------------------
# hit_and_dividend — PLACE + QIN PLACE BANKER pricing
# (GPT-5.5 R8 Finding #2 2026-05-25)
# ---------------------------------------------------------------------------

import sqlite3
from pathlib import Path

import pytest

import config


def _conn_or_skip():
    if not Path(config.DB_PATH).exists():
        pytest.skip("No horse_history.db found")
    return sqlite3.connect(str(config.DB_PATH))


def test_hit_and_dividend_place_priced_from_race_dividends():
    """PLACE pricing was 0.0 (MVP stub) until 2026-05-25. Verify a top pick
    that placed top-3 now returns a non-zero per-$1 dividend matched from
    race_dividends pool='PLACE' for the horse's number."""
    from scripts.stake_ab import hit_and_dividend
    conn = _conn_or_skip()
    # Find a (race, placed horse_number) pair we can stage.
    row = conn.execute("""
        SELECT race_date, race_number, combination
        FROM race_dividends WHERE pool='PLACE' LIMIT 1
    """).fetchone()
    if row is None:
        conn.close()
        pytest.skip("no PLACE dividend rows")
    date, rno, num_str = row
    placed_num = int(num_str)

    # Stage a synthetic race dict with the placed horse as top-1 pick.
    race = {
        'race_number': rno,
        'horses': [
            {'name': 'TOPPICK', 'number': placed_num, 'rank': 1,
             'actual_position': 2, 'win_odds': 5.0,
             'bet_recommendation': 'PLACE'},
            {'name': 'SECOND', 'number': 99, 'rank': 2,
             'actual_position': 1},
            {'name': 'THIRD', 'number': 98, 'rank': 3,
             'actual_position': 3},
        ],
    }
    hit, div = hit_and_dividend('PLACE', race, conn, date)
    conn.close()
    assert hit, "PLACE must register hit when top pick finishes top-3"
    assert div > 0, (
        f"PLACE per-$1 dividend must be > 0 after 2026-05-25 fix; got {div}"
    )


def test_hit_and_dividend_qpl_banker_sums_hit_pairs():
    """QIN PLACE BANKER returns sum of per-$1 dividends across all hit
    (pivot, banker) pairs. A single placed pivot + banker → ≥1 priced pair."""
    from scripts.stake_ab import hit_and_dividend
    conn = _conn_or_skip()
    row = conn.execute("""
        SELECT race_date, race_number, combination
        FROM race_dividends WHERE pool='QUINELLA PLACE' LIMIT 1
    """).fetchone()
    if row is None:
        conn.close()
        pytest.skip("no QPL dividend rows")
    date, rno, combo = row
    n1, n2 = sorted(int(x) for x in combo.split(','))

    # Stage so pivot=n1 (rank 1), bankers include n2 (rank 2). Both placed.
    race = {
        'race_number': rno,
        'horses': [
            {'name': 'PIVOT', 'number': n1, 'rank': 1,
             'actual_position': 1, 'win_odds': 1.5,
             'bet_recommendation': 'WIN'},
            {'name': 'BK1', 'number': n2, 'rank': 2,
             'actual_position': 3},
            {'name': 'BK2', 'number': 97, 'rank': 3,
             'actual_position': 5},
            {'name': 'BK3', 'number': 96, 'rank': 4,
             'actual_position': 6},
        ],
    }
    hit, div = hit_and_dividend('QIN PLACE BANKER', race, conn, date)
    conn.close()
    assert hit, "QPL BANKER hit when pivot + ≥1 banker both top-3"
    assert div > 0, f"QPL per-$1 dividend sum must be > 0; got {div}"


def test_score_bet_from_ticket_box_qin4_protection_does_not_crash():
    """Old hit_and_dividend split `BOX QIN-4 PROTECTION` as a BOX QIN-N
    label and crashed on `int('4 PROTECTION')`. New _score_bet_from_ticket
    reads pool='QUINELLA' + legs directly; the suffix is irrelevant
    (GPT-5.5 R10 2026-05-25 — silent ticket-scoring regression after R9
    made firing_bets ticket-first but kept label parsing in scoring)."""
    from scripts.stake_ab import _score_bet_from_ticket
    conn = _conn_or_skip()
    race = {
        'race_number': 8,
        'horses': [
            {'name': f'H{i}', 'number': i, 'rank': i,
             'actual_position': i if i <= 4 else None}
            for i in range(1, 8)
        ],
    }
    bet = {
        'label': 'BOX QIN-4 PROTECTION', 'pool': 'QUINELLA', 'combos': 6,
        'legs': [{'name': f'H{i}', 'number': i} for i in range(1, 5)],
    }
    # Should not raise — and should report hit (top-2 ⊂ {H1..H4}).
    hit, div = _score_bet_from_ticket(bet, race, conn, '2026/05/24')
    conn.close()
    assert hit, "BOX QIN-4 PROTECTION must hit when top-2 inside legs"


def test_score_bet_from_ticket_f4_saver_scores():
    """F4 SAVER had NO branch in old hit_and_dividend → (False, 0.0)
    for every staked F4 SAVER bet. Scoring now treats it as pool='FIRST 4'
    and checks top-4 ⊂ leg names."""
    from scripts.stake_ab import _score_bet_from_ticket
    conn = _conn_or_skip()
    race = {
        'race_number': 8,
        'horses': [
            {'name': f'H{i}', 'number': i, 'rank': i,
             'actual_position': i if i <= 4 else None}
            for i in range(1, 8)
        ],
    }
    bet = {
        'label': 'F4 SAVER', 'pool': 'FIRST 4', 'combos': 1,
        'legs': [{'name': f'H{i}', 'number': i} for i in range(1, 5)],
    }
    hit, div = _score_bet_from_ticket(bet, race, conn, '2026/05/24')
    conn.close()
    assert hit, "F4 SAVER hits when top-4 finishers all in legs (any order)"


def test_score_bet_from_ticket_trio6_uses_legs_not_top_picks():
    """TRIO-6 with manual/auto rank-6 swap: ticket legs include the swapped
    horse (e.g. rank-9 #99) but old scoring used `_top_picks(race, 6)`
    which returned rank-1..6 (without the swap). Result: false-miss when
    the swap-in horse placed top-3. Now scoring iterates ticket legs."""
    from scripts.stake_ab import _score_bet_from_ticket
    conn = _conn_or_skip()
    # Top-3 finishers are #1, #2, #99 (the swapped-in horse). Default
    # top-6 ranks would have been #1..#6 — without the swap legs, #99
    # wouldn't be in scoring, so the bet false-misses.
    race = {
        'race_number': 9,
        'horses': [
            {'name': 'H1', 'number': 1, 'rank': 1, 'actual_position': 1},
            {'name': 'H2', 'number': 2, 'rank': 2, 'actual_position': 2},
            {'name': 'H99', 'number': 99, 'rank': 9, 'actual_position': 3},  # swap-in placed 3rd
            {'name': 'H3', 'number': 3, 'rank': 3, 'actual_position': 4},
            {'name': 'H4', 'number': 4, 'rank': 4, 'actual_position': 5},
            {'name': 'H5', 'number': 5, 'rank': 5, 'actual_position': 6},
            {'name': 'H6', 'number': 6, 'rank': 6, 'actual_position': 7},
        ],
    }
    # Ticket legs: rank-1..5 + the manual-override #99 (NOT rank-6 #6).
    bet = {
        'label': 'BOX TRIO-6', 'pool': 'TRIO', 'combos': 20,
        'legs': [
            {'name': 'H1', 'number': 1}, {'name': 'H2', 'number': 2},
            {'name': 'H3', 'number': 3}, {'name': 'H4', 'number': 4},
            {'name': 'H5', 'number': 5}, {'name': 'H99', 'number': 99},
        ],
    }
    hit, div = _score_bet_from_ticket(bet, race, conn, '2026/05/24')
    conn.close()
    assert hit, (
        "TRIO-6 swap hit must read ticket legs, not _top_picks rank-1..6"
    )


def test_score_bet_from_ticket_qpl_banker_pivot_off_board():
    """Same miss case as the legacy test, now via the ticket-leg scorer."""
    from scripts.stake_ab import _score_bet_from_ticket
    conn = _conn_or_skip()
    race = {
        'race_number': 99,
        'horses': [
            {'name': 'PIVOT', 'number': 1, 'rank': 1, 'actual_position': 5},
            {'name': 'BK1', 'number': 2, 'rank': 2, 'actual_position': 1},
            {'name': 'BK2', 'number': 3, 'rank': 3, 'actual_position': 2},
            {'name': 'BK3', 'number': 4, 'rank': 4, 'actual_position': 3},
        ],
    }
    bet = {
        'label': 'QIN PLACE BANKER', 'pool': 'QIN PLACE', 'combos': 3,
        'legs': [
            {'name': 'PIVOT', 'number': 1}, {'name': 'BK1', 'number': 2},
            {'name': 'BK2', 'number': 3}, {'name': 'BK3', 'number': 4},
        ],
    }
    hit, div = _score_bet_from_ticket(bet, race, conn, '2026/05/24')
    conn.close()
    assert not hit
    assert div == 0.0


def test_append_log_writes_lf_line_endings(tmp_path, monkeypatch):
    """csv.DictWriter defaults to '\\r\\n' even with newline='' on the
    file open. Force LF-only so `git show --check` doesn't flag every
    appended row as trailing whitespace (GPT-5.5 R10 Low)."""
    from scripts.stake_ab import _append_log
    log_path = tmp_path / 'stake_ab_log.csv'
    monkeypatch.setattr('scripts.stake_ab.LOG_PATH', log_path)
    _append_log([_row()], '2026/05/24')
    raw = log_path.read_bytes()
    assert b'\r\n' not in raw, (
        f"CSV must use LF-only line endings, found CRLF: {raw[:120]!r}"
    )
    assert raw.endswith(b'\n')


def test_hit_and_dividend_qpl_banker_miss_when_pivot_off_board():
    """If pivot doesn't place top-3, no pair can pay → miss."""
    from scripts.stake_ab import hit_and_dividend
    conn = _conn_or_skip()
    race = {
        'race_number': 99,
        'horses': [
            {'name': 'PIVOT', 'number': 1, 'rank': 1,
             'actual_position': 5},
            {'name': 'BK1', 'number': 2, 'rank': 2,
             'actual_position': 1},
            {'name': 'BK2', 'number': 3, 'rank': 3,
             'actual_position': 2},
            {'name': 'BK3', 'number': 4, 'rank': 4,
             'actual_position': 3},
        ],
    }
    hit, div = hit_and_dividend('QIN PLACE BANKER', race, conn, '1900/01/01')
    conn.close()
    assert not hit
    assert div == 0.0


def test_append_log_full_rerun_still_replaces_same_variant(tmp_path, monkeypatch):
    """The full-rerun case (same set of variants for same date) must still
    overwrite — otherwise duplicate rows accumulate."""
    log_path = tmp_path / 'stake_ab_log.csv'
    monkeypatch.setattr('scripts.stake_ab.LOG_PATH', log_path)
    _append_log([
        _row(date='2026/05/24', variant='flat_min'),
        _row(date='2026/05/24', variant='flat_1pct'),
    ], '2026/05/24')
    _append_log([
        _row(date='2026/05/24', variant='flat_min'),
        _row(date='2026/05/24', variant='flat_1pct'),
    ], '2026/05/24')
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2, f'expected 2 rows after full re-run, got {len(rows)}'
