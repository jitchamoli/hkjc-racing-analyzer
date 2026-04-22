"""Tests for score combination and probability calibration."""

import pytest

from ml.model import create_combined_prediction
import config


def _expected_rule_prob(rule_score: float, field_size: int = 12) -> float:
    """Recompute the anchored-sigmoid formula used by create_combined_prediction."""
    import math
    base_prob = 1.0 / max(field_size, 2)
    logit_base = math.log(base_prob / (1.0 - base_prob))
    return 1.0 / (1.0 + math.exp(-(0.06 * (rule_score - 50) + logit_base)))


class TestCombinedPrediction:
    def test_base_rate_anchored_blend(self):
        """Rule score sigmoid is anchored to field-size base rate before blending."""
        prob, conf, breakdown = create_combined_prediction(
            rule_score=60.0,
            rule_confidence="MEDIUM",
            ml_prob=0.20,
            ml_confidence="MEDIUM",
            ml_available=True,
            field_size=12,
        )
        expected = _expected_rule_prob(60.0, 12) * config.RULE_WEIGHT + 0.20 * config.ML_WEIGHT
        assert abs(prob - expected) < 0.001

    def test_average_horse_returns_base_rate(self):
        """rule_score=50 should output approximately 1/field_size (base rate)."""
        prob, _, _ = create_combined_prediction(
            rule_score=50.0,
            rule_confidence="MEDIUM",
            ml_prob=1.0 / 12,
            ml_confidence="MEDIUM",
            ml_available=True,
            field_size=12,
        )
        # At rule_score=50 the anchored sigmoid outputs exactly 1/12 ≈ 0.083
        assert abs(prob - 1.0 / 12) < 0.01

    def test_ml_unavailable_uses_rule_only(self):
        """When ML unavailable, weight should be 100% rule (via anchored sigmoid)."""
        prob, conf, breakdown = create_combined_prediction(
            rule_score=80.0,
            rule_confidence="HIGH",
            ml_prob=0.50,
            ml_confidence="HIGH",
            ml_available=False,
            field_size=12,
        )
        assert abs(prob - _expected_rule_prob(80.0, 12)) < 0.001
        assert breakdown['weights']['ml'] == 0.0

    def test_confidence_high_high(self):
        _, conf, _ = create_combined_prediction(
            rule_score=80, rule_confidence="HIGH",
            ml_prob=0.3, ml_confidence="HIGH", ml_available=True,
        )
        assert conf == "VERY_HIGH"

    def test_confidence_low_low(self):
        _, conf, _ = create_combined_prediction(
            rule_score=30, rule_confidence="LOW",
            ml_prob=0.05, ml_confidence="LOW", ml_available=True,
        )
        assert conf == "LOW"

    def test_breakdown_structure(self):
        _, _, breakdown = create_combined_prediction(
            rule_score=50, rule_confidence="MEDIUM",
            ml_prob=0.1, ml_confidence="LOW", ml_available=True,
        )
        assert "rule_based" in breakdown
        assert "ml_based" in breakdown
        assert "final_probability" in breakdown
        assert "weights" in breakdown
        assert breakdown["rule_based"]["score"] == 50


class TestWeightedScore:
    def test_compute_weighted_score_basic(self):
        from core.scorer import compute_weighted_score

        factors = {
            "rating": 80.0,
            "form": 70.0,
            "odds": 60.0,
            "jockey": 75.0,
            "trainer": 65.0,
            "draw": 50.0,
        }
        score = compute_weighted_score(factors)
        assert 50 < score < 80  # should be a weighted average in this range

    def test_zero_factors_excluded(self):
        from core.scorer import compute_weighted_score

        # tips=0 should be excluded from weight normalization
        factors = {
            "rating": 80.0,
            "form": 70.0,
            "odds": 60.0,
            "jockey": 75.0,
            "trainer": 65.0,
            "draw": 50.0,
            "tips": 0.0,  # should be excluded
        }
        score = compute_weighted_score(factors)
        assert 50 < score < 80

    def test_always_included_factors_kept_even_at_zero(self):
        """Class_movement, gear, weight_carried are always included."""
        from core.scorer import compute_weighted_score

        factors = {
            "rating": 80.0,
            "form": 70.0,
            "odds": 60.0,
            "jockey": 75.0,
            "trainer": 65.0,
            "draw": 50.0,
            "class_movement": 0.0,  # always included
            "gear": 0.0,  # always included
            "weight_carried": 0.0,  # always included
        }
        # These zeroes should pull the score down
        score = compute_weighted_score(factors)
        assert score < 70
