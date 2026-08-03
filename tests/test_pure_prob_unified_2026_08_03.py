r"""There must be exactly ONE "the model's own opinion" probability.

Until now there were two, and they disagreed three different ways:

                        analyzer model_probability   compute_pure_probs
    ML input            isotonic-CALIBRATED ml_prob   RAW XGBoost ml_prob
    normalisation       none (per-race sum 0.70-2.32) sums to 1.0
    calibration         none                          pure_calibrator.pkl

Both claim to be "the pre-market blend model probability". Measured over 15
production meetings the analyzer number gave rank-1 a mean of 0.198 while
compute_pure_probs gave 0.133 — for the same horses in the same races.

This is not cosmetic. Any analysis that reaches for "what did the model think"
gets a different answer depending on which one it grabs, and a per-race sum of
1.29 is not a probability distribution at all — subtracting a properly-scaled
market probability from it produces a number that cannot be interpreted as an
edge. (That mis-scaling produced a spurious "+44% edge" reading during the
2026-08-03 review, which evaporated once both sides were normalised.)

So: `model_probability` must BE `compute_pure_probs`, not a parallel
re-derivation of it. The F-19 invariant it was introduced for — snapshot once,
never let a rank swap overwrite it — is unchanged.

Ranking is untouched: `model_probability` is a diagnostic snapshot that nothing
in the recommender, ticket, or dashboard leg-selection path reads.
"""
import pytest

from core.analyzer import HorseAnalysis
from core.scorer import apply_calibrated_probs
from betting.edge_calculator import compute_pure_probs


def _analysis(rule_score: float, raw_ml: float, cal_ml: float,
              win_odds: float) -> HorseAnalysis:
    """A field member whose RAW and CALIBRATED ml probs deliberately differ.

    The gap is the point: it is what distinguishes the two competing
    definitions of "the model's opinion".
    """
    from ml.model import MLPrediction

    class _Horse:
        def __init__(self, o):
            self.win_odds = o
            self.name = "Test"

    ml_pred = MLPrediction(
        horse_number=1, horse_name="Test",
        win_probability=cal_ml,
        place_probability=min(0.95, cal_ml * 2.5),
        ml_confidence="MEDIUM",
        feature_importance={},
        model_version="test",
        raw_win_probability=raw_ml,
    )
    return HorseAnalysis(
        horse=_Horse(win_odds),
        rank=1,
        rule_score=rule_score,
        rule_confidence="MEDIUM",
        ml_prediction=ml_pred,
        combined_probability=0.6 * 0.5 + 0.4 * cal_ml,   # placeholder pre-blend
        combined_confidence="MEDIUM",
        value_score=0.0,
        expected_return=0.0,
        bet_recommendation="",
        reasoning=[],
        prediction_breakdown={},
        model_probability=None,
    )


def _field():
    """A 5-runner field. raw_ml and cal_ml differ per horse."""
    return [
        _analysis(rule_score=78.0, raw_ml=0.34, cal_ml=0.26, win_odds=3.0),
        _analysis(rule_score=66.0, raw_ml=0.19, cal_ml=0.16, win_odds=5.0),
        _analysis(rule_score=58.0, raw_ml=0.11, cal_ml=0.12, win_odds=8.0),
        _analysis(rule_score=48.0, raw_ml=0.06, cal_ml=0.09, win_odds=15.0),
        _analysis(rule_score=44.0, raw_ml=0.04, cal_ml=0.08, win_odds=25.0),
    ]


def _calibrated(n):
    """Plausible post-blend probs — irrelevant to model_probability."""
    raw = [0.30, 0.22, 0.18, 0.16, 0.14][:n]
    return [r / sum(raw) for r in raw]


class TestModelProbabilityIsPureProb:
    def test_model_probability_is_a_distribution(self):
        """A per-race sum of 1.29 is not a probability — edges built on it lie."""
        field = _field()
        apply_calibrated_probs(field, _calibrated(len(field)))

        total = sum(a.model_probability for a in field)
        assert total == pytest.approx(1.0), (
            f"model_probability must sum to 1.0 across the field, got {total:.4f}"
        )

    def test_model_probability_equals_compute_pure_probs(self):
        """The definitional claim: ONE function, not two implementations."""
        field = _field()
        expected = compute_pure_probs(
            [{'rule_score': a.rule_score,
              'ml_prob': a.ml_prediction.raw_win_probability,
              'win_odds': a.horse.win_odds} for a in field],
            field_size=len(field),
        )

        apply_calibrated_probs(field, _calibrated(len(field)))

        got = [a.model_probability for a in field]
        assert got == pytest.approx(expected), (
            "model_probability diverges from compute_pure_probs — the two "
            f"definitions are still separate.\n  pure : {expected}\n  model: {got}"
        )

    def test_uses_raw_ml_prob_not_the_calibrated_one(self):
        """pure_prob is the PRE-calibration opinion (docs spec + generator).

        Two fields identical except for raw_win_probability must produce
        different model_probability. If the analyzer path is still reading
        MLPrediction.win_probability, this cannot move.
        """
        base = _field()
        apply_calibrated_probs(base, _calibrated(len(base)))

        bumped = _field()
        bumped[0].ml_prediction.raw_win_probability = 0.60   # raw only
        apply_calibrated_probs(bumped, _calibrated(len(bumped)))

        assert bumped[0].model_probability > base[0].model_probability, (
            "raising RAW ml_prob did not move model_probability — the analyzer "
            "snapshot is still built from the calibrated ml_prob"
        )


class TestF19InvariantSurvives:
    def test_snapshot_is_not_overwritten_by_a_rank_swap(self):
        """F-19, unchanged: apply twice, the snapshot holds."""
        field = _field()
        apply_calibrated_probs(field, _calibrated(len(field)))
        first = [a.model_probability for a in field]

        swapped = list(reversed(_calibrated(len(field))))
        apply_calibrated_probs(field, swapped)

        assert [a.model_probability for a in field] == pytest.approx(first)
        assert [a.combined_probability for a in field] == pytest.approx(swapped)

    def test_combined_probability_still_becomes_the_calibrated_value(self):
        field = _field()
        cal = _calibrated(len(field))
        apply_calibrated_probs(field, cal)
        assert [a.combined_probability for a in field] == pytest.approx(cal)


class TestDegradedInputs:
    def test_missing_ml_prediction_does_not_crash(self):
        """Rule-only degradation (ML unavailable) must still snapshot."""
        field = _field()
        for a in field:
            a.ml_prediction = None
        apply_calibrated_probs(field, _calibrated(len(field)))
        assert sum(a.model_probability for a in field) == pytest.approx(1.0)

    def test_zero_raw_ml_falls_back_to_the_calibrated_ml_prob(self):
        """Legacy MLPrediction objects predate raw_win_probability (default 0.0).

        Mirrors dashboard/generator's `h.get('raw_ml_prob') or h['ml_prob']`.
        """
        field = _field()
        for a in field:
            a.ml_prediction.raw_win_probability = 0.0
        apply_calibrated_probs(field, _calibrated(len(field)))

        expected = compute_pure_probs(
            [{'rule_score': a.rule_score,
              'ml_prob': a.ml_prediction.win_probability,
              'win_odds': a.horse.win_odds} for a in field],
            field_size=len(field),
        )
        assert [a.model_probability for a in field] == pytest.approx(expected)
