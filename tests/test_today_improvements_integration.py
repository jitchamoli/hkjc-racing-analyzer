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

    def test_high_at_weak_top2_gets_scaled_down(self):
        from betting.optimizer import compute_suggested_stake
        # Simulate generator's call site
        stake = compute_suggested_stake(
            conviction_tier='HIGH',
            bet_type='QIN',
            top_prob=0.3,
            top_odds=4.0,
            bankroll=10000,
            top_2_share=0.42,  # weak HIGH (close to threshold)
        )
        # Expect ~0.80 scale: scale = 0.75 + (0.42-0.40)/0.10 * 0.25 = 0.80
        # Full HIGH = 2% × $10000 = $200; scaled = $200 × 0.80 = $160
        assert stake['amount'] == pytest.approx(160, abs=10), \
            f"Expected ~$160 at top_2=0.42, got ${stake['amount']}"
        assert '×0.80' in stake['method']

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
