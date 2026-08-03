"""A DB failure loading prior history must fail CLOSED, not silently default.

Bug (code review 2026-08-03): `extract_features` wrapped the prior-history
query in

    except Exception as e:
        logger.warning(f"Failed to load prior history for {horse_name}: {e}")

leaving `prior = []`. That silently collapses NINE features (prior win rate,
recent form, class, distance, going, jockey/trainer pairing, pace, late kick)
to their schema defaults. `validate_features()` still passes (length and
NaN/Inf only), `predict()` returns a confident-looking probability, and the
dashboard renders a perfectly normal card built on defaults.

`sqlite3.OperationalError: database is locked` is entirely plausible here —
the monitor holds a long-lived connection while auto_dump.sh and auto-learn
write mid-meeting.

This contradicts the repo's fail-closed posture: F-17 makes untrained
`predict()` raise, `CombinedRaceAnalyzer(fail_closed_ml=True)` is the
production default, and 20 lines above this handler an empty `race_date`
alongside a live connection already raises ValueError for exactly the same
reason — refuse rather than quietly produce a plausible wrong answer.

The DB-less path (conn=None, no race_date) is a documented supported default
for novel horses and unit tests and must keep working.
"""
import sqlite3

import pytest

from ml.features import FeatureEngineering


class _ExplodingConn:
    """A live-looking connection whose queries fail."""
    def execute(self, *_a, **_k):
        raise sqlite3.OperationalError("database is locked")


def _horse_data():
    return {
        "name": "TEST HORSE",
        "race_date": "2026/07/15",
        "distance": 1200,
        "going": "GOOD",
        "jockey": "Z Purton",
        "trainer": "J Size",
        "field_size": 12,
    }


class TestPriorHistoryFailsClosed:
    def test_db_error_loading_prior_history_raises(self):
        """A locked DB must surface, not silently yield default features."""
        with pytest.raises(RuntimeError, match="prior history"):
            FeatureEngineering.extract_features(_horse_data(), [], "ST",
                                                conn=_ExplodingConn())

    def test_dbless_path_still_returns_defaults(self):
        """Supported contract: no conn and no date -> schema defaults, no raise."""
        from ml.feature_schema import NUM_FEATURES

        data = _horse_data()
        data.pop("race_date")
        feats = FeatureEngineering.extract_features(data, [], "ST", conn=None)

        assert len(feats) == NUM_FEATURES

    def test_healthy_db_is_unaffected(self):
        """Guard: a working connection still extracts a full vector."""
        import config
        from ml.feature_schema import NUM_FEATURES

        if not config.DB_PATH.exists():
            pytest.skip("production DB not present")

        conn = sqlite3.connect(str(config.DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            feats = FeatureEngineering.extract_features(_horse_data(), [], "ST", conn=conn)
            assert len(feats) == NUM_FEATURES
        finally:
            conn.close()
