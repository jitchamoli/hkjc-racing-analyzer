"""scripts/stake_ab — _pnl() formula + firing_bets() triple gate.

Test-engineer expert review 2026-05-23 priority #1: this is the core P&L
math for every forward-validation stake variant. Dividend-indexing bugs
(div-per-$10 vs div-per-$1) would silently 10×-scale every ROI.
"""
from scripts.stake_ab import _pnl, firing_bets


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


def test_firing_bets_skips_on_heavy_fav():
    fires = firing_bets(_race(top1_odds=2.2, tier='HIGH', n_horses=10))
    assert 'BOX TRIO-5' not in fires
    assert 'F4 BOX-5' not in fires


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
