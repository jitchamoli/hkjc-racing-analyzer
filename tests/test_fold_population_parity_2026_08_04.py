r"""Training and backtest must cut folds over the SAME race population.

`ml/training.build_training_data_from_db` drops unusable races (missing venue,
absurd field) BEFORE building `groups`, so the model's folds are cut over one
list of race groups. `scripts/backtest._load_training_window_rows` applies no
such filter, so IT cuts over a different, longer list. The two boundaries then
drift apart:

    fold        model pickle              backtest
    train end   2024/09/18-R4             2024/09/15-R8
    test start  2025/12/07-R2             2025/12/03-R7

Measured consequence at the time of writing: 3 races / 38 horse-rows inside
backtest's "truly-virgin" test fold were in the model's CALIB fold — in-sample
for the win/place isotonics. That is 0.48% of the test fold, so the numeric
inflation is small today.

**The problem is that the invariant is unenforced.** `tests/test_holdout_
disjointness.py` looks like it guards this but cannot: all four of its tests
call `scripts.backtest._load_training_window_rows` and re-split it with
`ml.folds.split_rows_by_race`, then assert the result is internally disjoint —
a tautology of `grouped_temporal_indices`. It never reads the model's
`fold_metadata`, so it passes with the drift present and would pass with any
future drift too.

Given this repo's history of leak-inflated results, "small today" is not a
defence. This test compares backtest's first test group against the boundary
the model actually recorded.
"""
import sqlite3

import pytest

import config


def _model_fold_metadata():
    import pickle
    path = config.MODEL_DIR / 'hkjc_racing_model.pkl'
    if not path.exists():
        pytest.skip('no production model')
    return pickle.load(open(path, 'rb')).get('fold_metadata') or {}


class TestFoldBoundariesAgree:
    def test_backtest_test_fold_starts_where_the_model_says(self):
        meta = _model_fold_metadata()
        folds = meta.get('folds') or {}
        model_first = (folds.get('test') or {}).get('first_group')
        if not model_first:
            pytest.skip('model pickle carries no fold boundary metadata')

        if not config.DB_PATH.exists():
            pytest.skip('no local DB')
        conn = sqlite3.connect(str(config.DB_PATH))
        conn.row_factory = sqlite3.Row
        from scripts.backtest import load_held_out_races, group_by_race
        races = group_by_race(load_held_out_races(conn))
        conn.close()
        if not races:
            pytest.skip('no held-out races')

        # Match the pickle's own key format: "YYYY/MM/DD-<race_number>".
        bt_first = min(races, key=lambda k: (k[0], int(k[1])))
        bt_first = f"{bt_first[0]}-{int(bt_first[1])}"
        assert bt_first == model_first, (
            f"backtest's test fold starts at {bt_first} but the model recorded "
            f"{model_first} — the two are cutting folds over different race "
            f"populations, so backtest is scoring races the model calibrated on"
        )

    def test_group_counts_agree(self):
        meta = _model_fold_metadata()
        n_model = meta.get('n_groups')
        if not n_model:
            pytest.skip('model pickle carries no n_groups')

        if not config.DB_PATH.exists():
            pytest.skip('no local DB')
        conn = sqlite3.connect(str(config.DB_PATH))
        conn.row_factory = sqlite3.Row
        from scripts.backtest import _load_training_window_rows
        rows = _load_training_window_rows(conn)
        conn.close()
        n_bt = len({(r['race_date'], r['race_number']) for r in rows})
        assert n_bt == n_model, (
            f"backtest sees {n_bt} race groups, the model trained on {n_model}. "
            f"Fold boundaries are computed from this count, so any difference "
            f"slides backtest's test fold into the model's calib fold."
        )


class TestLastNFlagIsSafe:
    def test_last_n_counts_races_not_horse_rows(self):
        """`--last 500` used to mean ~42 races: `rows` are HORSE rows."""
        if not config.DB_PATH.exists():
            pytest.skip('no local DB')
        conn = sqlite3.connect(str(config.DB_PATH))
        conn.row_factory = sqlite3.Row
        from scripts.backtest import load_held_out_races, group_by_race
        races = group_by_race(load_held_out_races(conn, last_n=20))
        conn.close()
        assert len(races) == 20, (
            f"--last 20 produced {len(races)} races; the flag is documented as "
            f"'last N races' but was slicing horse rows"
        )

    def test_last_n_cannot_reach_outside_the_test_fold(self):
        """A large N must clamp, not silently read the training fold."""
        if not config.DB_PATH.exists():
            pytest.skip('no local DB')
        conn = sqlite3.connect(str(config.DB_PATH))
        conn.row_factory = sqlite3.Row
        from scripts.backtest import load_held_out_races, group_by_race
        full = group_by_race(load_held_out_races(conn))
        huge = group_by_race(load_held_out_races(conn, last_n=10 ** 6))
        conn.close()
        assert set(huge) <= set(full), (
            '--last with a large N reached back through the calibration fold '
            'and into training data'
        )
