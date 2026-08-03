"""`validate_features` must enforce the per-feature ranges the schema declares.

Bug (code review 2026-08-03): every entry in FEATURE_SCHEMA declares a
`range` tuple, but `validate_features` only checked vector LENGTH and
NaN/Inf. The ranges were decorative.

That matters because this repo's #1 documented bug class is train/predict
feature drift, and the navigational comments in ml/features.py and
ml/training.py are themselves off-by-one from the schema (a leftover from the
v6.11 prune). A permuted or mis-scaled feature vector is invisible: it has the
right length and no NaN, so it sails through validation and XGBoost happily
returns a confident, meaningless probability.

Range checking is the cheap guard that catches exactly that class of mistake,
and the correct bounds are already written down in the schema.
"""
import numpy as np
import pytest

from ml.feature_schema import FEATURE_SCHEMA, NUM_FEATURES, validate_features


def _valid_vector():
    """A vector of each feature's declared default — must always validate."""
    return np.array([f["default"] for f in FEATURE_SCHEMA], dtype=np.float32)


class TestRangeValidation:
    def test_default_vector_is_valid(self):
        assert validate_features(_valid_vector()) is True

    def test_value_above_declared_range_is_rejected(self):
        feats = _valid_vector()
        lo, hi = FEATURE_SCHEMA[0]["range"]
        feats[0] = hi + 1.0
        with pytest.raises(AssertionError, match="prior_win_rate"):
            validate_features(feats)

    def test_value_below_declared_range_is_rejected(self):
        feats = _valid_vector()
        lo, hi = FEATURE_SCHEMA[0]["range"]
        feats[0] = lo - 1.0
        with pytest.raises(AssertionError, match="prior_win_rate"):
            validate_features(feats)

    def test_boundary_values_are_accepted(self):
        """Inclusive bounds — a feature legitimately at its limit must pass."""
        feats = _valid_vector()
        for i, spec in enumerate(FEATURE_SCHEMA):
            feats[i] = spec["range"][0]
        assert validate_features(feats) is True

        for i, spec in enumerate(FEATURE_SCHEMA):
            feats[i] = spec["range"][1]
        assert validate_features(feats) is True

    def test_length_and_nan_checks_still_apply(self):
        with pytest.raises(AssertionError):
            validate_features(np.zeros(NUM_FEATURES - 1, dtype=np.float32))

        feats = _valid_vector()
        feats[3] = np.nan
        with pytest.raises(AssertionError, match="NaN"):
            validate_features(feats)

    def test_real_extracted_features_pass_range_validation(self):
        """The guard must not fire on genuine production feature vectors."""
        import sqlite3
        import config
        from ml.features import FeatureEngineering

        if not config.DB_PATH.exists():
            pytest.skip("production DB not present")

        conn = sqlite3.connect(str(config.DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            # Take the most recent meeting rather than a hardcoded date: a DB
            # restored from a dump that predates 2026/07/15 used to fail this
            # HARD (`assert rows`) rather than adapting.
            rows = conn.execute(
                "SELECT horse_name, race_date, distance, going, jockey, trainer "
                "FROM race_results ORDER BY race_date DESC LIMIT 25"
            ).fetchall()
            assert rows, "race_results is empty"
            for row in rows:
                data = dict(row)
                data["name"] = data.pop("horse_name")
                data["field_size"] = 12
                feats = FeatureEngineering.extract_features(data, [], "HV", conn=conn)
                assert validate_features(feats) is True
        finally:
            conn.close()
