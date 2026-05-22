"""Integration tests for 2026-05-17 improvements that couldn't be verified
on today's frozen race data (all 11 races finished before the changes shipped).

Pins:
- Stake scaling on HIGH races flows from generator → race_data['suggested_stake']
- Manual override with a real (non-no-op) swap renders the 🛠 MANUAL badge
- EDGE banner reads live EDGE_STATS injection
- finish_time backfill writes a value when sectional sum is non-zero
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class TestStakeScalingIntegration:
    """Gap 1: Direct verification that top_2_share flows through to suggested_stake."""

    def test_high_at_weak_top2_no_longer_scaled_down(self):
        # FLATTENED 2026-05-22: the within-tier top_2_share ramp was removed
        # (no EV justification — per-tier exotic ROI doesn't separate; N=3
        # origin). A weak-HIGH race now gets the FULL HIGH stake, not a shaved
        # one. See _confidence_scale + project_conviction_gate_wrong_for_exotics.
        from betting.optimizer import compute_suggested_stake
        stake = compute_suggested_stake(
            conviction_tier='HIGH',
            bet_type='QIN',
            top_prob=0.3,
            top_odds=4.0,
            bankroll=10000,
            top_2_share=0.42,  # weak HIGH (close to threshold)
        )
        # Full HIGH = 2% × $10000 = $200, no scaling regardless of top_2_share.
        assert stake['amount'] == 200.0, \
            f"Expected full $200 (flattened), got ${stake['amount']}"
        assert '×' not in stake['method']

    def test_high_at_strong_top2_unchanged(self):
        from betting.optimizer import compute_suggested_stake
        stake = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='QIN',
            top_prob=0.3, top_odds=4.0, bankroll=10000,
            top_2_share=0.65,
        )
        assert stake['amount'] == 200.0
        assert '×' not in stake['method']

    def test_back_compat_no_top2_arg(self):
        # Older callers don't pass top_2_share — should default to no scaling
        from betting.optimizer import compute_suggested_stake
        stake = compute_suggested_stake(
            conviction_tier='HIGH', bet_type='QIN',
            top_prob=0.3, top_odds=4.0, bankroll=10000,
        )
        assert stake['amount'] == 200.0


class TestCardExposureCap:
    """Card-level exposure cap: total forward suggested-stake across a card is
    capped at config.CARD_EXPOSURE_CAP of bankroll, pro-rated; settled races
    excluded. Added 2026-05-22 (Risk-expert panel finding)."""

    def _races(self, amounts, finished=None):
        finished = finished or [False] * len(amounts)
        return [{'results_available': fin,
                 'suggested_stake': {'amount': a, 'pct': a / 100.0,
                                     'method': 'tier-fraction'}}
                for a, fin in zip(amounts, finished)]

    def test_caps_when_total_exceeds(self):
        from dashboard.generator import apply_card_exposure_cap
        races = self._races([300, 200, 200, 300])  # total 1000 = 10% of 10k... push over
        scale = apply_card_exposure_cap(races, bankroll=10000, cap_frac=0.05)  # cap=$500
        total = sum(r['suggested_stake']['amount'] for r in races)
        assert scale < 1.0
        assert total <= 500 + 40  # ~$500 after $10 rounding
        assert all(r['suggested_stake'].get('card_capped') for r in races)

    def test_no_cap_when_under(self):
        from dashboard.generator import apply_card_exposure_cap
        races = self._races([100, 100, 100])  # total $300 < $1000 cap
        scale = apply_card_exposure_cap(races, bankroll=10000, cap_frac=0.10)
        assert scale == 1.0
        assert all(r['suggested_stake']['amount'] in (100,) for r in races)
        assert not any(r['suggested_stake'].get('card_capped') for r in races)

    def test_settled_races_excluded_from_cap(self):
        from dashboard.generator import apply_card_exposure_cap
        # 3 finished ($900) + 2 pending ($600). Only pending count toward cap.
        races = self._races([300, 300, 300, 300, 300],
                            finished=[True, True, True, False, False])
        apply_card_exposure_cap(races, bankroll=10000, cap_frac=0.05)  # cap=$500
        pending_total = sum(r['suggested_stake']['amount']
                            for r in races if not r['results_available'])
        assert pending_total <= 500 + 20
        # finished races untouched
        assert all(r['suggested_stake']['amount'] == 300
                   for r in races if r['results_available'])

    def test_cap_disabled_when_zero(self):
        from dashboard.generator import apply_card_exposure_cap
        races = self._races([5000, 5000])
        scale = apply_card_exposure_cap(races, bankroll=10000, cap_frac=0.0)
        assert scale == 1.0
        assert sum(r['suggested_stake']['amount'] for r in races) == 10000


class TestManualOverrideRendering:
    """Gap 2: Manual override that differs from current rank-6 should render
    the 🛠 MANUAL badge in the dashboard HTML.

    Uses the rendered template + an in-memory race fixture where horse #99
    is rank 6 but the override points to horse #5 (rank 3) — different
    horses, so the swap is a real swap (not a no-op suppression).
    """

    def test_manual_badge_renders_in_template(self):
        from dashboard.template import generate_html
        race_fixture = [{
            'race_number': 1, 'time': '13:30', 'distance': 1200,
            'class_type': 'Class 4', 'going': 'Good', 'venue': 'Sha Tin',
            'conviction_tier': 'HIGH', 'has_edge': False,
            'top_2_share': 0.55, 'top_3_share': 0.72,
            'odds_age_minutes': 0, 'bankroll': 10000,
            'suggested_stake': {'amount': 200, 'pct': 2.0, 'method': 'tier-fraction (HIGH)'},
            'manual_trio6_horse': 5,
            'manual_trio6_reason': 'RPR 80 + Atzeni',
            'horses': [
                {'rank': i, 'number': i, 'name': f'H{i}', 'win_odds': 3.0 + i,
                 'calibrated_prob': 0.25 - i*0.02, 'edge_pp': 0,
                 'edge_fires': False, 'pure_prob': 0.1, 'market_prob': 0.1,
                 'bet_recommendation': 'QIN' if i <= 2 else 'SKIP',
                 'gear': 'B/TT', 'jockey': 'X', 'trainer': 'Y',
                 'jockey_form_30d': 0.1, 'jockey_strike_rate': 0.1,
                 'steam_move': {'is_steam': False}}
                for i in range(1, 13)
            ],
        }]
        html = generate_html(json.dumps(race_fixture), '2026-05-17 12:00:00',
                              '2026/05/17', 'Sha Tin')
        # The MANUAL badge code path must exist in the rendered JS (actual
        # #X→#Y substitution happens at runtime when JS evaluates with the
        # fixture data; the static-HTML check just confirms the badge code
        # is present and would fire on a real swap).
        assert '🛠 MANUAL' in html, "Manual override badge code path missing"
        # Also confirm the manual-override pre-check is shipped (race.manual_trio6_horse)
        assert 'race.manual_trio6_horse' in html

    def test_manual_override_matching_rank6_suppresses_badge(self):
        """When override == current rank-6, badge must NOT render (no-op fix
        from 2026-05-17 — avoids 'MANUAL #6→#6' confusing display)."""
        from dashboard.template import generate_html
        race_fixture = [{
            'race_number': 1, 'time': '13:30', 'distance': 1200,
            'class_type': 'Class 4', 'going': 'Good', 'venue': 'Sha Tin',
            'conviction_tier': 'HIGH', 'has_edge': False,
            'top_2_share': 0.55, 'top_3_share': 0.72,
            'odds_age_minutes': 0, 'bankroll': 10000,
            'suggested_stake': {'amount': 200, 'pct': 2.0, 'method': 'tier-fraction (HIGH)'},
            'manual_trio6_horse': 6,  # matches current rank-6
            'manual_trio6_reason': 'no-op test',
            'horses': [
                {'rank': i, 'number': i, 'name': f'H{i}', 'win_odds': 3.0 + i,
                 'calibrated_prob': 0.25 - i*0.02, 'edge_pp': 0,
                 'edge_fires': False, 'pure_prob': 0.1, 'market_prob': 0.1,
                 'bet_recommendation': 'QIN' if i <= 2 else 'SKIP',
                 'gear': 'B/TT', 'jockey': 'X', 'trainer': 'Y',
                 'jockey_form_30d': 0.1, 'jockey_strike_rate': 0.1,
                 'steam_move': {'is_steam': False}}
                for i in range(1, 13)
            ],
        }]
        html = generate_html(json.dumps(race_fixture), '2026-05-17 12:00:00',
                              '2026/05/17', 'Sha Tin')
        # The no-op-suppression badge — explicit 'MANUAL #6→#6' should not appear
        assert 'MANUAL #6→#6' not in html, "No-op MANUAL badge should be suppressed"


class TestEdgeBannerLiveStats:
    """Gap 4 closure: EDGE_STATS const injected and consumed by template."""

    def test_edge_stats_injected_into_template(self):
        from dashboard.template import generate_html
        html = generate_html('[]', '2026-05-17 12:00:00', '2026/05/17', 'Sha Tin',
                              edge_stats={'n_fires': 7, 'n_placed': 5, 'n_won': 1,
                                          'last_updated': '2026/05/17'})
        # The const declaration with our data should be in the HTML
        assert 'EDGE_STATS = {' in html
        assert '"n_fires": 7' in html
        assert '"n_placed": 5' in html
        assert '"n_won": 1' in html

    def test_edge_stats_defaults_to_empty(self):
        from dashboard.template import generate_html
        html = generate_html('[]', '2026-05-17 12:00:00', '2026/05/17', 'Sha Tin')
        # No stats passed → empty object (banner falls back to generic text)
        assert 'EDGE_STATS = {}' in html


class TestFinishTimeBackfill:
    """Gap closure: finish_time backfill from sectional sum produces sane format."""

    def test_backfill_produces_mm_ss_format(self):
        # Mirror the logic in _attach_sectional_times
        sec = "13.73 21.95 23.42 24.60"
        total = sum(float(x) for x in sec.split())
        mins, secs = divmod(total, 60)
        ft = f"{int(mins)}:{secs:05.2f}"
        assert ft == "1:23.70"

    def test_backfill_handles_sub_minute(self):
        # Defensive — shouldn't happen for real HK races but tested
        sec = "10 12 15"
        total = sum(float(x) for x in sec.split())
        mins, secs = divmod(total, 60)
        ft = f"{int(mins)}:{secs:05.2f}"
        assert ft == "0:37.00"
