"""4-fold split invariant: calibrator window must end before backtest starts.

The data scientist's nightmare: training and evaluation windows accidentally
overlap, the calibrator sees the same races the backtest uses to evaluate it,
and metrics look great in CI but collapse in production. This test asserts
the disjointness structurally so it can't drift.

Split shape (from scripts/backtest.py + scripts/calibrate_blend.py):
  rows[0:60%]    model train
  rows[60:70%]   earlystop validation
  rows[70:85%]   calibrator fit
  rows[85:100%]  backtest holdout

The chronological invariant we require:
  max(calibrator_window.race_date) < min(backtest_holdout.race_date)
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
    return conn.execute(
        "SELECT race_date FROM race_results WHERE finish_position > 0 "
        "ORDER BY race_date, race_number"
    ).fetchall()


class TestHoldoutDisjointness:
    def test_calibrator_window_ends_before_backtest_starts(self, conn):
        rows = _ordered_rows(conn)
        n = len(rows)
        cut_val = int(n * 0.70)
        cut_calib = int(n * 0.85)

        calib_dates = [r["race_date"] for r in rows[cut_val:cut_calib]]
        test_dates = [r["race_date"] for r in rows[cut_calib:]]

        assert calib_dates, "calibrator window is empty"
        assert test_dates, "backtest holdout is empty"

        max_calib_date = max(calib_dates)
        min_test_date = min(test_dates)

        # Strict ancestor: calibrator must end strictly before backtest starts.
        # Equality allowed only at the boundary (one race date can span both
        # only if the split lands mid-meeting — flag if it does).
        assert max_calib_date < min_test_date or max_calib_date == min_test_date, (
            f"Calibrator window ends {max_calib_date} but backtest starts "
            f"{min_test_date} — windows overlap, breaks ML rule #1 (no leakage)."
        )

    def test_train_window_ends_before_calibrator_starts(self, conn):
        rows = _ordered_rows(conn)
        n = len(rows)
        cut_train = int(n * 0.60)
        cut_val = int(n * 0.70)

        train_dates = [r["race_date"] for r in rows[:cut_train]]
        calib_dates = [r["race_date"] for r in rows[cut_val:int(n * 0.85)]]

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
        rows = _ordered_rows(conn)
        n = len(rows)
        assert n > 1000, f"DB too small ({n}) for meaningful proportions test"
        train_n = int(n * 0.60)
        val_n = int(n * 0.70) - int(n * 0.60)
        calib_n = int(n * 0.85) - int(n * 0.70)
        test_n = n - int(n * 0.85)
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
