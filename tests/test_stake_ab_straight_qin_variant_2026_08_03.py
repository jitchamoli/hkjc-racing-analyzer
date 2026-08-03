"""`straight_qin_only` forward-A/B variant.

Motivation (holdout diagnostic 2026-08-03, 634 races, real HKJC dividends):
box widening destroys quinella ROI monotonically —

    straight top-2 (1 combo)   hit 13.6%  need 13.7%  ROI  -1.0%
    box top-3      (3 combos)  hit 26.3%  need 30.2%  ROI -12.9%
    box top-4      (6 combos)  hit 40.1%  need 47.6%  ROI -15.9%
    box top-5      (10 combos) hit 51.1%  need 62.8%  ROI -18.6%

Each widening adds combos faster than it adds hit-rate x dividend. A paired
bootstrap puts P(straight beats box-4) at 0.899 — suggestive but BELOW the
0.95 ship bar, so this is a forward-logged variant, NOT a production change.
The un-burned forward sample agrees independently (straight_QIN +2.0% while
every box runs -19.8% to -49.3%).

The variant collapses every WIN-quinella-pool bet to a single 1-combo
STRAIGHT QIN on our top 2, and stakes nothing on any other pool. Everything
else about staking is identical to flat_min (HK$10/combo), so the comparison
isolates bet STRUCTURE, not stake size.
"""
import pytest

from scripts.stake_ab import VARIANTS, restrict_to_straight_qin, variant_stakes


class TestStraightQinOnlyVariantConfig:
    def test_variant_is_registered(self):
        assert 'straight_qin_only' in VARIANTS

    def test_it_stakes_like_flat_min(self):
        """Isolate structure, not stake size — same HK$10/combo as the baseline."""
        cfg = VARIANTS['straight_qin_only']
        assert cfg['mode'] == 'flat_per_combo'
        assert cfg['stake_per_combo'] == VARIANTS['flat_min']['stake_per_combo']


class TestRestrictToStraightQin:
    def test_box_qin_collapses_to_one_combo_straight(self):
        fires = {'BOX QIN-4': 6}
        assert restrict_to_straight_qin(fires) == {'STRAIGHT QIN': 1}

    def test_box_qin_protection_also_collapses(self):
        fires = {'BOX QIN-4 PROTECTION': 6}
        assert restrict_to_straight_qin(fires) == {'STRAIGHT QIN': 1}

    def test_existing_straight_qin_is_kept_as_is(self):
        fires = {'STRAIGHT QIN': 1}
        assert restrict_to_straight_qin(fires) == {'STRAIGHT QIN': 1}

    def test_non_quinella_pools_are_dropped(self):
        """TRIO/F4/QPL are different pools — this variant does not bet them."""
        fires = {'BOX TRIO-5': 10, 'F4 BOX-5': 5, 'F4 SAVER': 4,
                 'BOX TRIO-6': 20, 'QIN PLACE BANKER': 3}
        assert restrict_to_straight_qin(fires) == {}

    def test_mixed_card_keeps_only_the_straight_quinella(self):
        fires = {'BOX QIN-4': 6, 'BOX TRIO-5': 10, 'F4 BOX-5': 5,
                 'QIN PLACE BANKER': 3}
        assert restrict_to_straight_qin(fires) == {'STRAIGHT QIN': 1}

    def test_two_quinella_bets_do_not_double_up(self):
        """A card firing both a box and its protection must still bet ONE combo."""
        fires = {'BOX QIN-3': 3, 'BOX QIN-4 PROTECTION': 6}
        assert restrict_to_straight_qin(fires) == {'STRAIGHT QIN': 1}

    def test_empty_stays_empty(self):
        assert restrict_to_straight_qin({}) == {}

    def test_input_is_not_mutated(self):
        fires = {'BOX QIN-4': 6, 'BOX TRIO-5': 10}
        restrict_to_straight_qin(fires)
        assert fires == {'BOX QIN-4': 6, 'BOX TRIO-5': 10}


class TestStakingTheVariant:
    def test_one_combo_at_ten_dollars(self):
        race = {'conviction_tier': 'HIGH'}
        fires = restrict_to_straight_qin({'BOX QIN-4': 6})
        stakes = variant_stakes('straight_qin_only', race, fires, bankroll=10000)
        assert stakes == {'STRAIGHT QIN': 10.0}

    def test_stakes_far_less_than_flat_min_on_the_same_card(self):
        """The whole point: 1 combo instead of 6."""
        race = {'conviction_tier': 'HIGH'}
        raw = {'BOX QIN-4': 6, 'BOX TRIO-5': 10}
        base = variant_stakes('flat_min', race, raw, bankroll=10000)
        lean = variant_stakes('straight_qin_only', race,
                              restrict_to_straight_qin(raw), bankroll=10000)
        assert sum(base.values()) == pytest.approx(160.0)
        assert sum(lean.values()) == pytest.approx(10.0)
