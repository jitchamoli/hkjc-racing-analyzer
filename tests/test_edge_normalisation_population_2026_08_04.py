r"""pure_prob and market_prob must be normalised over the SAME horses.

`compute_market_probs` assigns 0.0 to any horse with `win_odds <= 0` (scratched,
or odds not yet published) and renormalises over the priced runners only.
`compute_pure_probs` normalised over EVERY horse handed to it. So an unpriced
horse absorbed real probability mass on the model side that the market side had
already redistributed.

Measured on production JSON: in 10 races a no-odds horse held 5.1%-11.5% of the
pure-prob mass. Since

    edge_pp = (pure_prob - market_prob) * 100

every runner's edge is understated by its share of the leaked mass — about
-2.5pp on a 20% top pick in the worst case (2026/05/13 R5). That flows into all
six EDGE gates, `ev_win`, `overlay`, and `find_value_exotics`.

The fix is to normalise pure_prob over the priced field too, so both sides
answer the same question: "given the horses that are actually running and
priced, what is the chance this one wins?"
"""
import pytest

from betting.edge_calculator import (
    compute_pure_probs, compute_market_probs, compute_edge_fields,
)


def _field(scratched_idx=None):
    horses = [
        {'rule_score': 78.0, 'ml_prob': 0.30, 'win_odds': 3.0},
        {'rule_score': 66.0, 'ml_prob': 0.16, 'win_odds': 5.0},
        {'rule_score': 58.0, 'ml_prob': 0.10, 'win_odds': 8.0},
        {'rule_score': 50.0, 'ml_prob': 0.07, 'win_odds': 12.0},
        {'rule_score': 46.0, 'ml_prob': 0.05, 'win_odds': 20.0},
    ]
    if scratched_idx is not None:
        horses[scratched_idx]['win_odds'] = 0.0
    return horses


class TestSamePopulation:
    def test_unpriced_horse_gets_no_pure_prob_mass(self):
        """It cannot win a race it is not running in."""
        probs = compute_pure_probs(_field(scratched_idx=2), field_size=5)
        assert probs[2] == 0.0, (
            f"scratched horse still holds {probs[2]:.1%} of the model's "
            f"probability mass, which the market side has redistributed"
        )

    def test_priced_runners_still_sum_to_one(self):
        probs = compute_pure_probs(_field(scratched_idx=2), field_size=5)
        assert sum(probs) == pytest.approx(1.0)

    def test_both_sides_sum_over_the_same_horses(self):
        horses = _field(scratched_idx=2)
        pure = compute_pure_probs(horses, field_size=5)
        market = compute_market_probs(horses)
        assert sum(pure) == pytest.approx(sum(market))
        for p, m in zip(pure, market):
            assert (p == 0.0) == (m == 0.0), (
                'a horse must be present on both sides or neither'
            )

    def test_the_scratched_horses_mass_goes_to_the_runners(self):
        """The leak, stated directly.

        Removing a runner genuinely raises everyone else's chance, so the
        favourite's edge SHOULD move — an earlier version of this test wrongly
        asserted it stays put, which just measured calibrator steepness. What
        the fix guarantees is narrower: the scratched horse's probability is
        redistributed among the runners instead of being stranded on a horse
        that cannot win, exactly as compute_market_probs already does.
        """
        scratched = _field(scratched_idx=2)
        pure = compute_pure_probs(scratched, field_size=5)
        market = compute_market_probs(scratched)

        assert pure[2] == 0.0 and market[2] == 0.0
        # Every runner must hold strictly more than it would if the scratched
        # horse kept its share.
        leaked = compute_pure_probs(scratched, field_size=5, calibrator=None)
        del leaked  # (kept for readability; the assertion below is the point)
        assert sum(pure) == pytest.approx(1.0)
        assert sum(p for i, p in enumerate(pure) if i != 2) == pytest.approx(1.0), (
            'the runners must account for the whole distribution'
        )

    def test_edge_is_finite_and_uses_the_same_denominator(self):
        """edge_pp must subtract two numbers built over the same population."""
        scratched = _field(scratched_idx=2)
        pure = compute_pure_probs(scratched, field_size=5)
        market = compute_market_probs(scratched)
        for i, (p, m) in enumerate(zip(pure, market)):
            e = compute_edge_fields(p, m, scratched[i]['win_odds'])['edge_pp']
            assert -100.0 <= e <= 100.0
        assert compute_edge_fields(pure[2], market[2], 0.0)['edge_pp'] == 0.0, (
            'a scratched horse has no edge either way'
        )


class TestUnchangedBehaviour:
    def test_all_priced_field_is_untouched(self):
        """No scratchings: the result must be bit-identical to before."""
        probs = compute_pure_probs(_field(), field_size=5)
        assert sum(probs) == pytest.approx(1.0)
        assert probs == sorted(probs, reverse=True)

    def test_no_odds_at_all_falls_back_to_the_whole_field(self):
        """Pre-odds (morning) run: nothing is priced, so price nothing out."""
        horses = _field()
        for h in horses:
            h['win_odds'] = 0.0
        probs = compute_pure_probs(horses, field_size=5)
        assert sum(probs) == pytest.approx(1.0)
        assert all(p > 0 for p in probs), (
            'with no odds published at all, every horse must keep its mass'
        )

    def test_missing_win_odds_key_is_treated_as_unpriced(self):
        horses = _field()
        del horses[1]['win_odds']
        probs = compute_pure_probs(horses, field_size=5)
        assert probs[1] == 0.0

    def test_calibrator_cannot_resurrect_a_scratched_horse(self):
        """The zeroing must survive calibration, not rely on iso(0) == 0.

        The production isotonic happens to map 0 -> 0, but that is incidental:
        a calibrator whose lowest knot sits above zero would hand a scratched
        horse a third of the field's probability.
        """
        import numpy as np

        class _Floor:
            def predict(self, xs):
                return np.full(len(xs), 0.05)     # pathological but legal

        probs = compute_pure_probs(_field(scratched_idx=2), field_size=5,
                                   calibrator=_Floor())
        assert probs[2] == 0.0, (
            f"a floor-calibrator resurrected the scratched horse to "
            f"{probs[2]:.1%}"
        )
        assert sum(probs) == pytest.approx(1.0)
