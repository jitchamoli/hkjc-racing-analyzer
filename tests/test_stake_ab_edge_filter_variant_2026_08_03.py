"""`edge_filtered` forward-A/B variant — bet only where we think it's mispriced.

Motivation (holdout diagnostic 2026-08-03, 634 races): splitting races by how
far the model's top pick disagrees with its market price,

    model edge >= +5pp   n=109   top-2 hit 18.3%   ROI  +44.1%
    model edge <  0      n=404   top-2 hit 14.6%   ROI   -2.0%

i.e. mispricing looks like a good RACE SELECTOR. Note it is NOT a good leg
orderer: ranking by the pure model inside those same races collapses top-2
capture from ~19% to ~4%, and wholesale pure-model ranking captures 6.3% vs
13.6% for the production blend. So this variant changes WHICH races we bet,
never the order of the horses within them.

Held as a hypothesis, not a rule: the edge buckets are non-monotonic (the
+0..+5pp bucket comes in at -38.2%), which is what noise looks like, and this
is the burned holdout. Forward-logged alongside straight_qin_only; ships only
at posterior >= 0.95.
"""
import pytest

from scripts.stake_ab import VARIANTS, race_meets_edge_floor, variant_stakes


def _race(edge_pp, tier='HIGH'):
    return {
        'conviction_tier': tier,
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
            'horses': [
                {'rank': 1, 'name': 'A', 'edge_pp': -2.0},
                {'rank': 5, 'name': 'E', 'edge_pp': 40.0},
            ],
        }
        assert race_meets_edge_floor(race, 5.0) is False

    def test_missing_edge_field_does_not_qualify(self):
        race = {'conviction_tier': 'HIGH', 'horses': [{'rank': 1, 'name': 'A'}]}
        assert race_meets_edge_floor(race, 5.0) is False

    def test_empty_race_does_not_qualify(self):
        assert race_meets_edge_floor({'horses': []}, 5.0) is False


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
