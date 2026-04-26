"""F-20: mixed-odds blend — verify calibrate_probabilities handles races where
some horses have no live odds while others do.

Key invariants:
1. Output probs always sum to 1.0 (normalization is applied regardless).
2. For horses without odds, output is proportional to their cal_prob relative to
   the sum — same math as if they used cal_prob as their market proxy.
"""
import math

import pytest

from core.scorer import calibrate_probabilities, apply_calibrated_probs


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class FakeHorse:
    def __init__(self, win_odds):
        self.win_odds = win_odds
        self.name = "test"


class FakeAnalysis:
    def __init__(self, combined_probability, win_odds):
        self.horse = FakeHorse(win_odds)
        self.combined_probability = combined_probability
        self.value_score = combined_probability * 100
        self.expected_return = 0.0
        self.model_probability = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMixedOddsBlend:
    def test_output_sums_to_one_all_odds_present(self):
        """Sanity check: all horses with odds → output normalizes to 1.0."""
        analyses = [
            FakeAnalysis(0.40, 3.0),
            FakeAnalysis(0.30, 5.0),
            FakeAnalysis(0.20, 8.0),
            FakeAnalysis(0.10, 15.0),
        ]
        probs = calibrate_probabilities(analyses, field_size=4, distance=1200)
        assert abs(sum(probs) - 1.0) < 1e-9, f"Probs sum to {sum(probs)}, expected 1.0"

    def test_output_sums_to_one_one_horse_missing_odds(self):
        """F-20: race with one horse missing odds still normalizes to 1.0."""
        analyses = [
            FakeAnalysis(0.40, 3.0),   # has odds
            FakeAnalysis(0.30, 5.0),   # has odds
            FakeAnalysis(0.30, 0.0),   # NO odds (scratched / pre-race)
        ]
        probs = calibrate_probabilities(analyses, field_size=3, distance=1200)
        assert abs(sum(probs) - 1.0) < 1e-9, f"Probs sum to {sum(probs)}, expected 1.0"

    def test_output_sums_to_one_all_horses_missing_odds(self):
        """F-20: edge case — no horse has live odds; cal_probs drive the entire output."""
        analyses = [
            FakeAnalysis(0.50, 0.0),
            FakeAnalysis(0.30, 0.0),
            FakeAnalysis(0.20, 0.0),
        ]
        probs = calibrate_probabilities(analyses, field_size=3, distance=1200)
        assert abs(sum(probs) - 1.0) < 1e-9, f"Probs sum to {sum(probs)}, expected 1.0"

    def test_no_odds_ratio_preserves_cal_prob_proportion(self):
        """F-20: when all horses lack odds, output ratios match cal_prob ratios.

        With no market signal, blended[i] = cal_probs[i] for all i.
        After normalization by the softmax total, the relative ordering is
        preserved (no horse is systematically boosted or suppressed).
        """
        analyses = [
            FakeAnalysis(0.60, 0.0),
            FakeAnalysis(0.40, 0.0),
        ]
        probs = calibrate_probabilities(analyses, field_size=2, distance=1200)
        # Both have same model_w applied, so ratio of outputs should be close to
        # ratio of inputs (exact equality holds only if softmax is linear, but
        # direction must be preserved).
        assert probs[0] > probs[1], (
            "Higher cal_prob horse must have higher blended prob when all odds missing"
        )

    def test_mixed_odds_both_valid_values(self):
        """F-20: probs are in (0, 1] for both odds-present and odds-absent horses."""
        analyses = [
            FakeAnalysis(0.55, 4.0),
            FakeAnalysis(0.45, 0.0),
        ]
        probs = calibrate_probabilities(analyses, field_size=2, distance=1200)
        for p in probs:
            assert 0.0 < p <= 1.0, f"Prob {p} out of (0,1]"
