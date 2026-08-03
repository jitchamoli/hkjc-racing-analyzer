"""The blend calibrator must not collapse distinct horses onto identical probs.

Problem (measured 2026-08-03 on the refit calibrator): the post-blend isotonic
is a coarse step function — only 48 distinct output levels across [0, 0.30],
and every input below 0.0202 maps to EXACTLY 0.0. Two consequences on the
634-race holdout:

  * 4.4% of runners ended with calibrated_prob == 0.0 despite being live
    runners with real odds (expected_return pinned at -1.0, 0% displayed).
  * Exact ties straddled the BOX TRIO-5 leg cutoff in ~45% of races (and the
    BOX Q4 cutoff in 40%). Where that happens `calibrated_prob` has no
    resolving power at all, so the leg choice falls through to the
    `pre_blend_rank` tiebreak — i.e. the PURE model silently picks the legs.

That last point matters because the pure model is measurably the WORST
ranking source available: ranking by it captures only 6.3% of quinellas vs
13.6% for the blend and 13.9% for the market. So in ~45% of box races the
cutoff was being decided by the weakest available signal.

Fix: add a vanishing multiple of the (market-blended) input to the isotonic
output. Monotone, so calibration is preserved to ~1e-6, but it restores a
strict ordering inside each isotonic plateau — ties now break on the BLENDED
probability, which includes the market, rather than on the pure model.
"""
import numpy as np
import pytest

from core.scorer import calibrate_probabilities


class _Horse:
    def __init__(self, win_odds):
        self.win_odds = win_odds
        self.place_odds = 0.0


class _An:
    def __init__(self, prob, odds):
        self.horse = _Horse(odds)
        self.combined_probability = prob
        self.model_probability = None
        self.pre_blend_rank = None
        self.rank = None
        self.expected_return = 0.0


class _FlatIso:
    """Worst-case isotonic: maps everything to one plateau, and small to zero."""
    def predict(self, xs):
        return np.array([0.0 if x < 0.05 else 0.20 for x in np.asarray(xs)])


def _field():
    # Clearly different model opinions and clearly different market prices.
    return [_An(0.40, 2.0), _An(0.25, 4.0), _An(0.20, 9.0),
            _An(0.10, 20.0), _An(0.05, 50.0)]


class TestNoCollapsedProbabilities:
    def test_live_runners_never_get_exactly_zero(self):
        probs = calibrate_probabilities(_field(), field_size=12, distance=1200,
                                        calibrator=_FlatIso())
        assert all(p > 0 for p in probs), (
            f"a priced runner was assigned probability 0: {probs}"
        )

    def test_plateau_does_not_produce_exact_ties(self):
        """Horses on the same isotonic step must still be strictly ordered."""
        probs = calibrate_probabilities(_field(), field_size=12, distance=1200,
                                        calibrator=_FlatIso())
        assert len(set(probs)) == len(probs), (
            f"isotonic plateau collapsed distinct horses to equal probs: {probs}"
        )

    def test_plateau_ties_break_on_the_pure_model_opinion(self):
        """Inside a plateau the MODEL's ordering must win, not the market's.

        Measured 2026-08-03: breaking plateau ties on the blended value made
        every headline metric worse (top-pick Brier 0.1920 -> 0.1963, straight
        QIN -0.81% -> -2.05%, TRIO-5 -15.6% -> -19.8%). Globally the blend
        outranks the pure model, but where the isotonic has flattened them the
        blend has said "indistinguishable", and the model's independent view is
        the only remaining signal — the behaviour pre_blend_rank already gave.
        """
        # Model likes index 0 most; the MARKET likes index 1 most (odds 1.5).
        field = [_An(0.50, 8.0), _An(0.10, 1.5), _An(0.08, 30.0)]
        probs = calibrate_probabilities(field, field_size=12, distance=1200,
                                        calibrator=_FlatIso())
        assert probs[0] > probs[1], (
            f"plateau tie broke toward the market pick instead of the model: {probs}"
        )

    def test_still_normalised(self):
        probs = calibrate_probabilities(_field(), field_size=12, distance=1200,
                                        calibrator=_FlatIso())
        assert sum(probs) == pytest.approx(1.0)

    def test_calibration_is_essentially_unchanged(self):
        """The tie-break term must be vanishing, not a re-weighting.

        A CONSTANT calibrator says "every horse is equally likely". The
        epsilon may order them, but must not reintroduce real spread.
        """
        class _ConstIso:
            def predict(self, xs):
                return np.full(len(np.asarray(xs)), 0.20)

        probs = calibrate_probabilities(_field(), field_size=12, distance=1200,
                                        calibrator=_ConstIso())
        assert max(probs) - min(probs) < 0.01, (
            f"tie-break term is too large, it is re-ranking: {probs}"
        )
        assert len(set(probs)) == len(probs), "constant iso still produced ties"

    def test_no_calibrator_path_unaffected(self):
        probs = calibrate_probabilities(_field(), field_size=12, distance=1200,
                                        calibrator=None)
        assert sum(probs) == pytest.approx(1.0)
        assert all(p > 0 for p in probs)


class TestProductionCalibratorGranularity:
    def test_production_calibrator_no_longer_zeroes_live_runners(self):
        """End-to-end with the real signed artifact."""
        probs = calibrate_probabilities(_field(), field_size=12, distance=1200)
        assert all(p > 0 for p in probs), f"production zeroed a runner: {probs}"
        assert len(set(probs)) == len(probs), f"production tied runners: {probs}"
