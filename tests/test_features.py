"""Tests for ML feature extraction."""

import numpy as np
import pytest

from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES, FEATURE_NAMES, validate_features
from ml.features import FeatureEngineering


class TestFeatureSchema:
    def test_schema_has_correct_feature_count(self):
        # v6.8 (2026-05-14): added early_speed (#22), late_kick (#23), running_style (#24)
        assert NUM_FEATURES == 24

    def test_feature_names_match_schema(self):
        assert len(FEATURE_NAMES) == NUM_FEATURES
        for i, f in enumerate(FEATURE_SCHEMA):
            assert FEATURE_NAMES[i] == f["name"]

    def test_all_defaults_within_range(self):
        for f in FEATURE_SCHEMA:
            lo, hi = f["range"]
            assert lo <= f["default"] <= hi, (
                f"{f['name']}: default {f['default']} not in range {f['range']}"
            )

    def test_validate_features_correct_length(self):
        features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
        assert validate_features(features) is True

    def test_validate_features_wrong_length(self):
        features = np.full(10, 0.5, dtype=np.float32)
        with pytest.raises(AssertionError, match=f"Expected {NUM_FEATURES}"):
            validate_features(features)

    def test_validate_features_rejects_nan(self):
        features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
        features[0] = np.nan
        with pytest.raises(AssertionError, match="NaN"):
            validate_features(features)

    def test_validate_features_rejects_inf(self):
        features = np.full(NUM_FEATURES, 0.5, dtype=np.float32)
        features[5] = np.inf
        with pytest.raises(AssertionError, match="Inf"):
            validate_features(features)


class TestFeatureEngineering:
    def test_parse_last_6_slash_format(self):
        result = FeatureEngineering.parse_last_6("1/3/10/2/5")
        assert result == [1, 3, 10, 2, 5]

    def test_parse_last_6_legacy_format(self):
        result = FeatureEngineering.parse_last_6("13425")
        assert result == [1, 3, 4, 2, 5]

    def test_parse_last_6_empty(self):
        result = FeatureEngineering.parse_last_6("")
        assert result == []

    def test_form_score_all_wins(self):
        score = FeatureEngineering.calculate_form_score("1/1/1")
        assert score == 100.0

    def test_form_score_empty(self):
        score = FeatureEngineering.calculate_form_score("")
        assert score == 50.0

    def test_draw_advantage_ideal_sprint(self):
        # Draw 4 is ideal for <= 1000m
        score = FeatureEngineering.calculate_draw_advantage(4, 1000)
        assert score == 1.0

    def test_draw_advantage_wide(self):
        score = FeatureEngineering.calculate_draw_advantage(14, 1200)
        assert score < 0.5

    def test_extract_features_returns_correct_count(self, sample_horse_data):
        features = FeatureEngineering.extract_features(
            sample_horse_data, [], "Sha Tin"
        )
        assert len(features) == NUM_FEATURES

    def test_extract_features_no_nan(self, sample_horse_data):
        features = FeatureEngineering.extract_features(
            sample_horse_data, [], "Sha Tin"
        )
        assert not np.any(np.isnan(features))

    def test_extract_features_no_inf(self, sample_horse_data):
        features = FeatureEngineering.extract_features(
            sample_horse_data, [], "Sha Tin"
        )
        assert not np.any(np.isinf(features))

    def test_extract_features_within_ranges(self, sample_horse_data):
        features = FeatureEngineering.extract_features(
            sample_horse_data, [], "Sha Tin"
        )
        for i, f in enumerate(FEATURE_SCHEMA):
            lo, hi = f["range"]
            assert lo - 0.01 <= features[i] <= hi + 0.01, (
                f"Feature {f['name']} = {features[i]} outside range {f['range']}"
            )

    def test_extract_features_minimal_data(self):
        """Minimal horse data should still produce correct number of features with defaults."""
        minimal = {"name": "Unknown", "distance": 1200, "field_size": 10}
        features = FeatureEngineering.extract_features(minimal, [], "")
        assert len(features) == NUM_FEATURES
        assert not np.any(np.isnan(features))

    def test_field_rel_draw_clipped_to_one(self):
        """F-16: field_rel_draw must not exceed 1.0.

        14-gate races with field_size=12 (inference default) produce draw/field_size > 1.
        The clip ensures the schema range (0,1) is always respected.
        """
        # draw=14, field_size=12 → raw = 14/12 ≈ 1.167, clipped → 1.0
        horse_over_gate = {"name": "Test", "distance": 1200, "field_size": 12, "draw": 14}
        features = FeatureEngineering.extract_features(horse_over_gate, [], "Sha Tin")
        field_rel_draw_idx = FEATURE_NAMES.index("field_rel_draw")
        assert features[field_rel_draw_idx] <= 1.0, (
            f"field_rel_draw={features[field_rel_draw_idx]} exceeds 1.0 for draw=14, field_size=12"
        )

    def test_field_rel_draw_normal_case(self):
        """F-16: field_rel_draw stays within [0,1] for normal cases."""
        for draw in range(1, 15):
            horse = {"name": "Test", "distance": 1200, "field_size": 14, "draw": draw}
            features = FeatureEngineering.extract_features(horse, [], "Sha Tin")
            idx = FEATURE_NAMES.index("field_rel_draw")
            assert 0.0 <= features[idx] <= 1.0, (
                f"field_rel_draw={features[idx]} out of [0,1] for draw={draw}"
            )

    @pytest.mark.parametrize("age_raw,expected_score", [
        ('5.0', 4/6),    # DB float-string format — 5yo → (9-5)/6 = 0.667
        ('5',   4/6),    # race-card int-string format — same horse, same score
        ('3.0', 1.0),    # young horse → 1.0
        ('3',   1.0),
        ('9.0', 0.0),    # old horse → 0.0
        ('9',   0.0),
        ('10.0', 0.0),   # 9yo+ floor at 0.0
        ('-',    0.5),   # missing → 0.5 fallback
        ('',     0.5),
        (0,      0.5),
        (None,   0.5),
    ])
    def test_age_parser_handles_db_and_card_formats(self, age_raw, expected_score):
        """F-17: parse '5.0' (DB) and '5' (race card) to the same score.

        Regression: the prior `[:2]` slice on '5.0' produced '5.' →
        int('5.') raised → silent fallback to 0.5. 94% of training rows
        ended up stuck at 0.5 while live inference parsed correctly,
        creating a train/predict mismatch on a real feature.
        """
        horse = {"name": "Test", "distance": 1200, "field_size": 12, "age": age_raw}
        features = FeatureEngineering.extract_features(horse, [], "Sha Tin")
        idx = FEATURE_NAMES.index("horse_age")
        assert abs(features[idx] - expected_score) < 1e-6, (
            f"age={age_raw!r} → {features[idx]}, expected {expected_score}"
        )
