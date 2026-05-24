"""4-fold split invariant: races must not cross train/calib/test windows.

The data scientist's nightmare: training and evaluation windows accidentally
overlap, the calibrator sees the same races the backtest uses to evaluate it,
and metrics look great in CI but collapse in production. This test asserts
the disjointness structurally so it can't drift.

Split shape (from ml.folds + scripts/backtest.py + scripts/calibrate_blend.py):
  race groups[0:60%]    model train
  race groups[60:70%]   earlystop validation
  race groups[70:85%]   calibrator fit
  race groups[85:100%]  backtest holdout

The chronological invariant we require:
  no (race_date, race_number) appears in more than one fold
"""
import sqlite3

import pytest

import config


@pytest.fixture(scope="module")
def conn():
    if not config.DB_PATH.exists():
        pytest.skip("DB not present (CI env without tracking_data/horse_history.db)")
    c = sqlite3.connect(str(config.DB_PATH))
    c.row_factory = sqlite3.Row
    yield c
    c.close()


def _ordered_rows(conn):
    from scripts.backtest import _load_training_window_rows
    return _load_training_window_rows(conn)


class TestHoldoutDisjointness:
    def test_calibrator_window_ends_before_backtest_starts(self, conn):
        from ml.folds import split_rows_by_race
        rows = _ordered_rows(conn)
        folds = split_rows_by_race(rows)
        calib_dates = [r["race_date"] for r in folds["calib"]]
        test_dates = [r["race_date"] for r in folds["test"]]

        assert calib_dates, "calibrator window is empty"
        assert test_dates, "backtest holdout is empty"

        max_calib_date = max(calib_dates)
        min_test_date = min(test_dates)

        assert max_calib_date <= min_test_date, (
            f"Calibrator window ends {max_calib_date} but backtest starts "
            f"{min_test_date} — windows overlap, breaks ML rule #1 (no leakage)."
        )

    def test_train_window_ends_before_calibrator_starts(self, conn):
        from ml.folds import split_rows_by_race
        rows = _ordered_rows(conn)
        folds = split_rows_by_race(rows)
        train_dates = [r["race_date"] for r in folds["train"]]
        calib_dates = [r["race_date"] for r in folds["calib"]]

        assert train_dates and calib_dates

        # Calibrator fits on rows[70:85%] — train is rows[0:60%] with a
        # 10% earlystop buffer at [60:70%]. By the time we reach the calibrator
        # window, training has been over for ~10% of rows. Strict-ancestor.
        max_train_date = max(train_dates)
        min_calib_date = min(calib_dates)
        assert max_train_date <= min_calib_date, (
            f"Train window ends {max_train_date} but calibrator starts "
            f"{min_calib_date} — training data leaks into calibrator fit."
        )

    def test_window_sizes_match_expected_proportions(self, conn):
        from ml.folds import split_rows_by_race
        rows = _ordered_rows(conn)
        n = len(rows)
        assert n > 1000, f"DB too small ({n}) for meaningful proportions test"
        folds = split_rows_by_race(rows)
        train_n = len(folds["train"])
        val_n = len(folds["val"])
        calib_n = len(folds["calib"])
        test_n = len(folds["test"])
        # ±1% slack on each fold for off-by-one from int() truncation
        for name, got, expected in [
            ("train", train_n, 0.60),
            ("earlystop", val_n, 0.10),
            ("calibrator", calib_n, 0.15),
            ("backtest", test_n, 0.15),
        ]:
            actual = got / n
            assert abs(actual - expected) < 0.01, (
                f"{name} fold = {actual:.3f} of {n}, expected {expected:.2f}"
            )

    def test_no_race_key_crosses_folds(self, conn):
        from ml.folds import split_rows_by_race
        rows = _ordered_rows(conn)
        folds = split_rows_by_race(rows)
        seen = {}
        for fold_name, fold_rows in folds.items():
            for r in fold_rows:
                key = (r["race_date"], r["race_number"])
                assert key not in seen or seen[key] == fold_name, (
                    f"Race {key} appears in both {seen[key]} and {fold_name}"
                )
                seen[key] = fold_name
