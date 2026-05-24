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
