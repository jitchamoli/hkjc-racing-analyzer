"""F-19: verify model_probability is preserved through BANKER/market-fav rank swaps.

combined_probability is mutated by _resync_ranks_and_probs() when the recommender
promotes a horse. model_probability must retain the pre-swap value so accuracy.py
can measure the model, not the human overrides.

2026-08-03: the snapshotted VALUE changed — model_probability is now the shared
`compute_pure_probs` (normalised across the field, built from the RAW ml_prob,
run through the pure calibrator) instead of the raw unnormalised
combined_probability, which was a second divergent definition of the same thing.
See tests/test_pure_prob_unified_2026_08_03.py. The invariant these tests exist
for — snapshot once, never let a rank swap overwrite it — is unchanged, so they
assert it against the shared definition rather than a hardcoded number.
"""
from dataclasses import dataclass, field
from typing import List, Dict
from unittest.mock import patch

import pytest

from core.analyzer import HorseAnalysis
from core.scorer import apply_calibrated_probs


def _make_horse(win_odds: float):
    """Return a minimal Horse-like object with a win_odds attribute."""

    @dataclass
    class FakeHorse:
        win_odds: float = 0.0
        name: str = "Test"

    return FakeHorse(win_odds=win_odds)


def _make_analysis(combined_probability: float, win_odds: float) -> HorseAnalysis:
    """Build a minimal HorseAnalysis for testing swap logic."""
    from ml.model import MLPrediction

    ml_pred = MLPrediction(
        horse_number=1, horse_name="Test",
        win_probability=combined_probability,
        place_probability=combined_probability * 2.5,
        ml_confidence="MEDIUM",
        feature_importance={},
        model_version="test",
    )
    return HorseAnalysis(
        horse=_make_horse(win_odds),
        rank=1,
        rule_score=60.0,
        rule_confidence="MEDIUM",
        ml_prediction=ml_pred,
        combined_probability=combined_probability,
        combined_confidence="MEDIUM",
        value_score=combined_probability * 100,
        expected_return=0.0,
        bet_recommendation="QIN",
        reasoning=[],
        prediction_breakdown={},
        model_probability=None,
    )


class TestModelProbabilityPreservation:
    def test_apply_calibrated_probs_sets_model_probability_on_first_call(self):
        """F-19: the first call must snapshot the pre-blend model opinion —
        i.e. the shared pure_prob — before combined_probability is overwritten."""
        from betting.edge_calculator import pure_probs_from_analyses

        a = _make_analysis(combined_probability=0.30, win_odds=4.0)
        assert a.model_probability is None
        expected = pure_probs_from_analyses([a])[0]

        apply_calibrated_probs([a], [0.25])

        assert a.model_probability == pytest.approx(expected), (
            "model_probability must be the shared pure_prob, not a re-derivation"
        )
        assert a.combined_probability == pytest.approx(0.25)

    def test_model_probability_not_overwritten_on_subsequent_calls(self):
        """F-19: once set, repeated apply_calibrated_probs must not change it."""
        a = _make_analysis(combined_probability=0.30, win_odds=4.0)
        apply_calibrated_probs([a], [0.25])   # first call — snapshots
        snapshot = a.model_probability
        assert snapshot is not None

        # Simulate rank swap: apply again with swapped probs
        apply_calibrated_probs([a], [0.18])

        assert a.model_probability == pytest.approx(snapshot), (
            "model_probability must not be overwritten after the initial snapshot"
        )
        assert a.combined_probability == pytest.approx(0.18), (
            "combined_probability should reflect the swap"
        )

    def test_model_probability_preserved_across_field(self):
        """F-19: multiple horses in a field each snapshot their own model_probability."""
        from betting.edge_calculator import pure_probs_from_analyses

        horses = [
            _make_analysis(combined_probability=0.40, win_odds=2.5),
            _make_analysis(combined_probability=0.25, win_odds=4.5),
            _make_analysis(combined_probability=0.15, win_odds=8.0),
        ]
        expected = pure_probs_from_analyses(horses)
        original_probs = [0.40, 0.25, 0.15]
        apply_calibrated_probs(horses, original_probs)

        # Confirm initial snapshot — and that it IS a distribution
        assert [a.model_probability for a in horses] == pytest.approx(expected)
        assert sum(a.model_probability for a in horses) == pytest.approx(1.0)

        # Simulate rank swap (swap horse 0 and 1)
        swapped_probs = [0.25, 0.40, 0.15]
        apply_calibrated_probs(horses, swapped_probs)

        # combined_probability updated, model_probability preserved
        assert [a.combined_probability for a in horses] == pytest.approx(swapped_probs)
        assert [a.model_probability for a in horses] == pytest.approx(expected)
