"""`edge_filtered` forward-A/B variant — bet only where we think it's mispriced.

The idea: how far the model's top pick disagrees with its market price might be
a good RACE SELECTOR. It is NOT a good leg orderer — ranking by the pure model
collapses top-2 capture to 6.3% vs 13.6% for the production blend — so this
variant changes WHICH races we bet, never the order of horses within them.

The motivating "+44.1% on n=109" split is WITHDRAWN (2026-08-03): it compared
the analyzer's UNNORMALISED model_probability (per-race sums 0.70-2.32) against
a normalised market probability, so the difference was not an edge. Re-run on
the calibrated pure_prob the 634-race holdout is non-monotone and the positive
cells are 6/44 and 2/16 bets — noise. The floor is therefore left at its
original +5.0 rather than tuned on burned data; see the sweep table in
scripts/stake_ab.py. Forward log is the verdict; ships only at posterior >= 0.95.

Because the edge_pp stored in old meeting JSONs is on the pre-calibration
scale, the floor also refuses meetings before PURE_CALIBRATION_DATE — see
tests/test_stake_ab_edge_scale_guard_2026_08_03.py.
"""
import pytest

from scripts.stake_ab import (
    VARIANTS, race_meets_edge_floor, variant_stakes, MEETING_DATE_KEY,
)


def _race(edge_pp, tier='HIGH'):
    return {
        'conviction_tier': tier,
        # Post-recalibration meeting — otherwise the scale guard refuses it.
        MEETING_DATE_KEY: '2026/08/10',
        'horses': [
            {'rank': 1, 'name': 'A', 'edge_pp': edge_pp},
            {'rank': 2, 'name': 'B', 'edge_pp': -3.0},
            {'rank': 3, 'name': 'C', 'edge_pp': -8.0},
        ],
    }


class TestVariantConfig:
    def test_variant_is_registered(self):
        assert 'edge_filtered' in VARIANTS

    def test_stakes_like_flat_min(self):
        """Isolate race SELECTION, not stake size."""
        cfg = VARIANTS['edge_filtered']
        assert cfg['mode'] == 'flat_per_combo'
        assert cfg['stake_per_combo'] == VARIANTS['flat_min']['stake_per_combo']

    def test_floor_is_five_points(self):
        assert VARIANTS['edge_filtered']['min_edge_pp'] == 5.0


class TestEdgeFloor:
    def test_race_at_or_above_the_floor_qualifies(self):
        assert race_meets_edge_floor(_race(5.0), 5.0) is True
        assert race_meets_edge_floor(_race(12.4), 5.0) is True

    def test_race_below_the_floor_is_skipped(self):
        assert race_meets_edge_floor(_race(4.9), 5.0) is False
        assert race_meets_edge_floor(_race(-6.0), 5.0) is False

    def test_only_the_rank_1_horse_counts(self):
        """A big edge on a rank-5 outsider must not qualify the race."""
        race = {
            'conviction_tier': 'HIGH',
            MEETING_DATE_KEY: '2026/08/10',
            'horses': [
                {'rank': 1, 'name': 'A', 'edge_pp': -2.0},
                {'rank': 5, 'name': 'E', 'edge_pp': 40.0},
            ],
        }
        assert race_meets_edge_floor(race, 5.0) is False

    def test_missing_edge_field_does_not_qualify(self):
        race = {'conviction_tier': 'HIGH', MEETING_DATE_KEY: '2026/08/10',
                'horses': [{'rank': 1, 'name': 'A'}]}
        assert race_meets_edge_floor(race, 5.0) is False

    def test_empty_race_does_not_qualify(self):
        assert race_meets_edge_floor(
            {MEETING_DATE_KEY: '2026/08/10', 'horses': []}, 5.0) is False


class TestStaking:
    def test_qualifying_race_is_staked_normally(self):
        fires = {'BOX QIN-4': 6}
        stakes = variant_stakes('edge_filtered', _race(9.0), fires, bankroll=10000)
        assert stakes == {'BOX QIN-4': 60.0}

    def test_non_qualifying_race_is_staked_zero(self):
        fires = {'BOX QIN-4': 6}
        stakes = variant_stakes('edge_filtered', _race(1.0), fires, bankroll=10000)
        assert stakes == {'BOX QIN-4': 0.0}

    def test_flat_min_is_unaffected_by_the_edge_floor(self):
        """Guard: the baseline must keep betting every race."""
        fires = {'BOX QIN-4': 6}
        stakes = variant_stakes('flat_min', _race(1.0), fires, bankroll=10000)
        assert stakes == {'BOX QIN-4': 60.0}
