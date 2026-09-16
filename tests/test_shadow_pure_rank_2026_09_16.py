"""Shadow the model-only ranking (pure_prob) against production.

User (2026-09-16): "I don't always want to rely on market voted horses —
sometimes horses not ranked well come in the top 3." Measured over 236 races:
148 podium finishers were market rank 8+; production's top 5 covered 16 of
them, the model-only order covered 26, at the same top-3 capture and a better
TRIO-5 ROI (-15.1% vs -21.3%). The bootstrap CI spans zero, so this SHADOWS —
it changes no bet until the gap holds.
"""
import pytest

from scripts.shadow_pure_rank import box_result, market_outsiders, pure_order


def _h(num, pure, odds, **kw):
    return dict(number=num, pure_prob=pure, win_odds=odds, **kw)


class TestPureOrder:
    def test_sorts_by_the_models_own_opinion_not_the_odds(self):
        horses = [_h(1, 0.10, 2.0), _h(2, 0.35, 18.0), _h(3, 0.22, 5.0)]
        assert pure_order(horses) == [2, 3, 1]

    def test_ties_break_by_production_rank_not_input_order(self):
        """Mirrors the recommender's deterministic tiebreak: never incidental order."""
        horses = [_h(9, 0.20, 12.0), _h(4, 0.20, 3.0)]          # production rank: 9 then 4
        assert pure_order(horses) == [9, 4]
        assert pure_order(list(reversed(horses))) == [4, 9]

    def test_missing_pure_prob_sinks_to_the_bottom(self):
        horses = [_h(1, None, 2.0), _h(2, 0.05, 30.0)]
        assert pure_order(horses) == [2, 1]


class TestMarketOutsiders:
    def test_finds_podium_horses_the_market_rated_8th_or_worse(self):
        horses = [_h(n, 0.1, float(n)) for n in range(1, 13)]    # odds ascend with number
        assert market_outsiders({1, 2, 11}, horses) == {11}

    def test_unpriced_runners_are_not_counted_as_outsiders(self):
        horses = [_h(1, 0.1, 2.0), _h(2, 0.1, 0.0)]
        assert market_outsiders({2}, horses) == set()


class TestBoxResult:
    ORDER = [10, 20, 30, 40, 50, 60]

    def test_hit_pays_the_dividend_once(self):
        assert box_result(self.ORDER, {20, 30, 50}, 175.0) == (100.0, 175.0, True)

    def test_miss_costs_the_stake(self):
        assert box_result(self.ORDER, {20, 30, 60}, 175.0) == (100.0, 0.0, False)

    @pytest.mark.parametrize("dividend", [None, 0.0])
    def test_unpriced_race_is_skipped_not_booked_as_a_loss(self, dividend):
        assert box_result(self.ORDER, {20, 30, 50}, dividend) is None

    def test_short_field_cannot_fill_the_box(self):
        assert box_result([1, 2, 3], {1, 2, 3}, 99.0) is None
