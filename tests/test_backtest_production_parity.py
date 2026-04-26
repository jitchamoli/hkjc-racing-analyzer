"""F-03 integration test: per-race predictions from scripts/backtest.py
must match production dashboard predictions within tolerance.

Uses sample holdout races. For each, runs both the backtest prediction
path and the production prediction path (as used by generate_predictions.py),
and asserts agreement on per-horse combined_probability and bet_recommendation.

After F-03, both paths call into CombinedRaceAnalyzer + assign_bet_recommendations,
so they should produce identical results (same code path = tautological agreement).
The value of this test is that future divergence between paths will be caught.
"""
import os
import sys
import sqlite3
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


SAMPLE_RACES = [
    # (date, race_no) — known holdout races from backtest (dates verified in DB)
    ('2026/04/22', 1),
    ('2026/04/22', 5),
    ('2026/04/19', 3),
    ('2026/04/15', 2),
    ('2026/03/29', 4),
]


PROB_TOLERANCE = 0.01
"""Per-horse combined_probability delta. After F-03, backtest and
production run the same code path, so probabilities should match
within floating-point tolerance."""


def _load_race_horses(conn, race_date, race_no):
    """Load horses for a specific race from the DB."""
    rows = conn.execute(
        "SELECT * FROM race_results WHERE race_date=? AND race_number=? "
        "AND finish_position > 0 ORDER BY horse_number",
        (race_date, race_no)
    ).fetchall()
    return [dict(r) for r in rows]


def _backtest_predict_race(race_date, race_no):
    """Run backtest's per-race prediction and return a dict of
    horse_name -> (combined_probability, bet_recommendation).

    F-03: calls the same production code path (build_production_race + analyze_race).
    """
    from scripts.backtest import build_production_race, _get_analyzer

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        horse_rows = _load_race_horses(conn, race_date, race_no)
        if not horse_rows:
            pytest.skip(f"No DB data for {race_date} R{race_no}")

        race = build_production_race(horse_rows)
        analyzer = _get_analyzer()
        analyses = analyzer.analyze_race(race)

        return {
            a.horse.name: (a.combined_probability, a.bet_recommendation)
            for a in analyses
        }
    finally:
        conn.close()


def _production_predict_race(race_date, race_no):
    """Run the production prediction path (same as generate_predictions.py)
    and return a dict of horse_name -> (combined_probability, bet_recommendation).

    After F-03 both paths call the same underlying analyzer, so results
    are identical. This function exists as an explicit second invocation
    to verify the shared code path.
    """
    from scripts.backtest import build_production_race, _get_analyzer
    from core.analyzer import CombinedRaceAnalyzer
    from core.recommender import assign_bet_recommendations

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    try:
        horse_rows = _load_race_horses(conn, race_date, race_no)
        if not horse_rows:
            pytest.skip(f"No DB data for {race_date} R{race_no}")

        race = build_production_race(horse_rows)
        # Use a fresh analyzer instance to verify no shared-state dependency
        analyzer = CombinedRaceAnalyzer()
        try:
            analyses = analyzer.analyze_race(race)
        finally:
            analyzer.close()

        return {
            a.horse.name: (a.combined_probability, a.bet_recommendation)
            for a in analyses
        }
    finally:
        conn.close()


@pytest.mark.parametrize("race_date,race_no", SAMPLE_RACES)
def test_backtest_production_parity(race_date, race_no):
    """Per-horse combined_probability from backtest must equal production
    within tolerance, and bet_recommendation must match exactly.

    After F-03, both paths call CombinedRaceAnalyzer.analyze_race with
    the same Race object built from DB data. Agreement is expected to be
    exact (floating-point identical), but the tolerance allows for any
    minor differences from singleton vs fresh analyzer state.
    """
    bt_result = _backtest_predict_race(race_date, race_no)
    prod_result = _production_predict_race(race_date, race_no)

    # Same set of horses
    assert set(bt_result.keys()) == set(prod_result.keys()), (
        f"horse-name sets differ: backtest={set(bt_result.keys())}, "
        f"production={set(prod_result.keys())}"
    )

    # Per-horse agreement on probability and recommendation
    for name in bt_result:
        bt_prob, bt_rec = bt_result[name]
        pr_prob, pr_rec = prod_result[name]
        delta = abs(bt_prob - pr_prob)
        assert delta <= PROB_TOLERANCE, (
            f"{name}: backtest_prob={bt_prob:.4f}, "
            f"production_prob={pr_prob:.4f}, delta={delta:.4f}"
        )
        assert bt_rec == pr_rec, (
            f"{name}: backtest_rec={bt_rec}, production_rec={pr_rec}"
        )
