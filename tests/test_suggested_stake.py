"""Bankroll-fraction stake suggestions for the dashboard's per-card $ display."""
import pytest

from betting.optimizer import compute_suggested_stake


class TestTierFractionExoticBoxes:
    """Exotic boxes use the tier fraction directly (no Kelly — multi-combo)."""

    def test_high_qin_is_2pct(self):
        out = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='QIN',
            top_prob=0.30, top_odds=3.5, bankroll=10000,
        )
        assert out['amount'] == 200.0
        assert out['pct'] == 2.0
        assert 'HIGH' in out['method']

    def test_med_box_is_1pct(self):
        out = compute_suggested_stake(
            conviction_tier='MED', bet_type='QIN',
            top_prob=0.20, top_odds=5.0, bankroll=10000,
        )
        assert out['amount'] == 100.0
        assert out['pct'] == 1.0

    def test_low_is_half_pct(self):
        out = compute_suggested_stake(
            conviction_tier='LOW', bet_type='QIN',
            top_prob=0.18, top_odds=6.0, bankroll=10000,
        )
        assert out['amount'] == 50.0
        assert out['pct'] == 0.5

    def test_edge_is_3pct(self):
        out = compute_suggested_stake(
            conviction_tier='EDGE', bet_type='QIN',
            top_prob=0.25, top_odds=5.0, bankroll=10000,
        )
        assert out['amount'] == 300.0
        assert out['pct'] == 3.0


class TestSkipZeroExposure:
    """SKIP-related cases must produce zero stake — never suggest money on a SKIP."""

    def test_skip_tier_returns_zero(self):
        out = compute_suggested_stake(
            conviction_tier='SKIP', bet_type='WIN',
            top_prob=0.30, top_odds=3.5, bankroll=10000,
        )
        assert out['amount'] == 0.0

    def test_top_bet_skip_overrides_tier(self):
        # Tier HIGH but the top horse's bet was SKIP'd by the recommender
        # (race-shape gate failed downstream). Must not suggest money.
        out = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='SKIP',
            top_prob=0.30, top_odds=3.5, bankroll=10000,
        )
        assert out['amount'] == 0.0
        assert 'skip' in out['method'].lower()

    def test_empty_bet_type_returns_zero(self):
        out = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='',
            top_prob=0.30, top_odds=3.5, bankroll=10000,
        )
        assert out['amount'] == 0.0

    def test_zero_bankroll_returns_zero(self):
        out = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='QIN',
            top_prob=0.30, top_odds=3.5, bankroll=0,
        )
        assert out['amount'] == 0.0


class TestHalfKellyForWin:
    """WIN/BANKER WIN bets use half-Kelly, capped at the tier fraction."""

    def test_high_win_half_kelly_below_cap(self):
        # 30% prob @ $3.5 → b=2.5, p=.3, q=.7 → kelly=(0.75-0.7)/2.5=0.02
        # half = 0.01 = 1.0%. Below HIGH cap of 2%.
        out = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='WIN',
            top_prob=0.30, top_odds=3.5, bankroll=10000,
        )
        assert out['amount'] == 100.0
        assert out['pct'] == 1.0
        assert 'half-Kelly' in out['method']

    def test_longshot_kelly_capped_at_tier(self):
        # 20% prob @ $8 → b=7, kelly=(1.4-0.8)/7=0.086, half=4.3%
        # Capped at EDGE 3% — don't blow bankroll on one longshot.
        out = compute_suggested_stake(
            conviction_tier='EDGE', bet_type='WIN',
            top_prob=0.20, top_odds=8.0, bankroll=10000,
        )
        assert out['amount'] == 300.0
        assert out['pct'] == 3.0

    def test_negative_kelly_floors_at_zero(self):
        # 10% prob @ $3 → b=2, kelly=(0.2-0.9)/2 < 0 → no edge → 0
        out = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='WIN',
            top_prob=0.10, top_odds=3.0, bankroll=10000,
        )
        assert out['amount'] == 0.0

    def test_banker_win_uses_kelly_path(self):
        # 45% prob @ $3.0 → b=2, kelly=(0.9-0.55)/2=0.175, half=8.75%
        # Capped at HIGH 2%. (40%/2.5 is breakeven; pick something with real edge.)
        out = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='BANKER WIN',
            top_prob=0.45, top_odds=3.0, bankroll=10000,
        )
        assert out['amount'] > 0
        assert 'half-Kelly' in out['method']


class TestBankrollScales:
    def test_amount_scales_linearly_with_bankroll(self):
        a = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='QIN',
            top_prob=0.30, top_odds=3.5, bankroll=5000,
        )
        b = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='QIN',
            top_prob=0.30, top_odds=3.5, bankroll=10000,
        )
        # Bankroll doubled → amount doubled (both 2% of bankroll)
        assert b['amount'] == 2 * a['amount']
        assert a['pct'] == b['pct']
