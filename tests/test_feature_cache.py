"""Backtest feature-cache invariants — opt-in, schema-aware, atomic."""
import numpy as np
import pytest

from data import feature_cache as fc
from ml.feature_schema import NUM_FEATURES


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Each test gets its own cache dir + clean state."""
    monkeypatch.setattr(fc, "CACHE_DIR", tmp_path)
    # Force a known db row count so cache_path is deterministic per-test
    monkeypatch.setattr(fc, "_db_row_count", lambda: 999)
    yield
    fc.disable()


class TestCacheOptIn:
    def test_disabled_by_default(self):
        assert fc.is_enabled() is False

    def test_get_returns_none_when_disabled(self):
        assert fc.get("2026/05/13", 1, "PODIUM") is None

    def test_put_is_noop_when_disabled(self):
        # Should not raise; should not store
        fc.put("2026/05/13", 1, "PODIUM", np.zeros(NUM_FEATURES, dtype=np.float32))
        fc.enable()
        assert fc.get("2026/05/13", 1, "PODIUM") is None


class TestCacheRoundtrip:
    def test_put_then_get_returns_same_vector(self):
        fc.enable()
        vec = np.arange(NUM_FEATURES, dtype=np.float32)
        fc.put("2026/05/13", 1, "PODIUM", vec)
        out = fc.get("2026/05/13", 1, "PODIUM")
        assert out is not None
        np.testing.assert_array_equal(out, vec)

    def test_miss_returns_none(self):
        fc.enable()
        assert fc.get("2026/05/13", 99, "MISSING") is None

    def test_wrong_shape_is_rejected(self):
        fc.enable()
        bad = np.zeros(NUM_FEATURES + 5, dtype=np.float32)
        fc.put("2026/05/13", 1, "BAD", bad)
        assert fc.get("2026/05/13", 1, "BAD") is None


class TestCachePersistence:
    def test_save_and_reload_preserves_entries(self, tmp_path):
        fc.enable()
        vecs = {
            ("2026/05/13", 1, "PODIUM"): np.arange(NUM_FEATURES, dtype=np.float32),
            ("2026/05/13", 2, "SETANTA"): np.arange(NUM_FEATURES, dtype=np.float32) * 0.5,
        }
        for (rd, rn, hn), v in vecs.items():
            fc.put(rd, rn, hn, v)
        fc.save()
        # Re-enable on the same dir → loads the file
        fc.disable()
        fc.enable()
        for (rd, rn, hn), v in vecs.items():
            out = fc.get(rd, rn, hn)
            assert out is not None
            np.testing.assert_array_equal(out, v)

    def test_save_is_atomic_no_partial_files_left(self, tmp_path):
        fc.enable()
        fc.put("2026/05/13", 1, "X", np.zeros(NUM_FEATURES, dtype=np.float32))
        fc.save()
        # Only the final .npz file, no .tmp leftovers
        files = sorted(p.name for p in tmp_path.glob("*"))
        assert files == ["features_v24_n999.npz"], f"unexpected files: {files}"


class TestCacheInvalidation:
    def test_cache_path_includes_schema_and_row_count(self, monkeypatch):
        monkeypatch.setattr(fc, "_db_row_count", lambda: 12345)
        path = fc._cache_path()
        assert "v24" in path.name  # NUM_FEATURES
        assert "n12345" in path.name  # row count

    def test_different_row_count_writes_to_different_file(self, monkeypatch, tmp_path):
        # Run 1: row count 100
        monkeypatch.setattr(fc, "_db_row_count", lambda: 100)
        fc.enable()
        fc.put("2026/05/13", 1, "X", np.zeros(NUM_FEATURES, dtype=np.float32))
        fc.save()
        # Run 2: row count 200 (simulates new data)
        monkeypatch.setattr(fc, "_db_row_count", lambda: 200)
        fc.disable()
        fc.enable()
        # Cache for the new row count should be empty — different file
        assert fc.get("2026/05/13", 1, "X") is None
        files = sorted(p.name for p in tmp_path.glob("*.npz"))
        assert "features_v24_n100.npz" in files
