"""Regression guard for the rule-engine date-leak class (2026-05-22).

The rule engine queries the horse's history during backtest, but race_results
CONTAINS the race being scored — so without a `before_date` filter these
lookups returned the current race's own result (finish_position/is_winner/
margin), a direct outcome leak. This bug was found and fixed TWICE (the second
time it had 6 siblings). These tests assert every leak-prone DB method respects
`before_date` so it can never silently return again.
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.database import HorseDatabase


@pytest.fixture
def db(tmp_path):
    """A horse 'TESTER' with three runs: 2026/01/01, 02/01, 03/01 (the last is
    the 'current' race being scored). before_date='2026/03/01' must exclude it."""
    d = HorseDatabase(db_path=str(tmp_path / "leak.db"))
    runs = [
        ("2026/01/01", 1, 5, 0, 0, 3.0),   # date, race_no, finish, is_winner, is_top3, margin
        ("2026/02/01", 1, 1, 1, 1, 0.0),
        ("2026/03/01", 1, 8, 0, 0, 6.0),   # the CURRENT race — must be excluded
    ]
    for date, rno, fin, win, top3, margin in runs:
        d.conn.execute(
            "INSERT INTO race_results (race_date, race_number, horse_name, jockey, "
            "trainer, finish_position, is_winner, is_top3, margin, win_odds, going, "
            "venue, distance) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (date, rno, "TESTER", "J WIN", "T WIN", fin, win, top3, margin, 4.0, "Good",
             "Sha Tin", 1200),
        )
    d.conn.commit()
    return d

CUR = "2026/03/01"  # the race being scored


def test_get_horse_history_excludes_current_and_later(db):
    leaked = db.get_horse_history("TESTER", limit=10)              # no filter (live)
    clean = db.get_horse_history("TESTER", limit=10, before_date=CUR)
    assert len(leaked) == 3 and len(clean) == 2
    assert all(r["race_date"] < CUR for r in clean)


def test_get_rolling_stats_excludes_current_race_result(db):
    # Leaking: 3 runs incl. the 8th-place current race. Clean: 2 prior runs.
    leaked = db.get_rolling_stats("TESTER", n_races=5)
    clean = db.get_rolling_stats("TESTER", n_races=5, before_date=CUR)
    assert leaked["runs"] == 3 and clean["runs"] == 2
    # the current race (8th) must not be in the clean average
    assert clean["avg_pos"] == pytest.approx((5 + 1) / 2)


def test_get_beaten_margins_excludes_current(db):
    clean = db.get_beaten_margins("TESTER", n_races=5, before_date=CUR)
    assert clean["runs"] == 2  # current race's margin=6.0 excluded


def test_get_horse_record_excludes_current(db):
    clean = db.get_horse_record("TESTER", before_date=CUR)
    assert clean["total_starts"] == 2  # not 3


def test_get_last_jockey_is_prior_not_current(db):
    # all rides are 'J WIN' here; the point is the count/window excludes current
    clean_hist = db.get_horse_history("TESTER", limit=1, before_date=CUR)
    assert clean_hist[0]["race_date"] == "2026/02/01"  # not 03/01


def test_get_horse_cd_history_excludes_current(db):
    # 3rd-round leak: C&D history counted the current race's own result.
    leaked = db.get_horse_cd_history("TESTER", "Sha Tin", 1200)
    clean = db.get_horse_cd_history("TESTER", "Sha Tin", 1200, before_date=CUR)
    assert len(leaked) == 3 and len(clean) == 2
    assert all(r["date"] < CUR for r in clean)


def test_season_aggregate_stats_accept_before_date(db):
    # season-aggregate leak: get_jockey_stats/get_trainer_stats read tables
    # rebuilt over the whole season incl. future; before_date computes
    # season-to-date from race_results (current race 03/01 excluded).
    j = db.get_jockey_stats("J WIN", before_date=CUR)
    t = db.get_trainer_stats("T WIN", before_date=CUR)
    assert j["rides"] == 2 and j["wins"] == 1      # 02/01 win counted, 03/01 excluded
    assert t["runners"] == 2 and t["wins"] == 1


def test_recent_form_methods_accept_before_date(db):
    # smoke: the jockey/trainer recent-form + going-record methods must accept
    # before_date and filter (regression against the 6-sibling miss)
    jf = db.get_jockey_recent_form("J WIN", n_rides=10, before_date=CUR)
    tf = db.get_trainer_recent_form("T WIN", n_runners=10, before_date=CUR)
    gr = db.get_horse_going_record("TESTER", "Good", before_date=CUR)
    assert jf["rides"] == 2 and tf["runners"] == 2 and gr["runs"] == 2


# ---------------------------------------------------------------------------
# PARITY / LEAK-CLASS GENERALIZATION GUARD
# ---------------------------------------------------------------------------
# The per-method tests above pin the 10 known methods. This end-to-end test
# generalizes: it runs the WHOLE rule engine and asserts that adding the current
# race's OWN result row to the DB does not change any horse's rule_score. ANY
# rule factor that leaks the current race (a future un-dated lookup) fails this.

def _insert(conn, date, rno, horse, num, jockey, trainer, fin, win, top3, margin):
    conn.execute(
        "INSERT INTO race_results (race_date, race_number, horse_name, horse_number, "
        "jockey, trainer, finish_position, is_winner, is_top3, margin, win_odds, going, "
        "venue, distance, draw) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (date, rno, horse, num, jockey, trainer, fin, win, top3, margin, 4.0, "Good",
         "Sha Tin", 1200, num),
    )


def test_adding_current_and_future_races_does_not_change_rule_score(tmp_path):
    """Score a race, then insert that race's OWN result + several FUTURE races
    into the DB and score again — rule_score must be IDENTICAL. The real leak
    pulled current+future rows into "recent history", tripping the multi-run
    factors, so the test injects future races to be SENSITIVE (verified by a
    mutation test: dropping a before_date filter makes this diverge). Any factor
    that leaks the current/future race fails this — generalizes the per-method
    guards to the whole rule engine."""
    try:
        from scripts.backtest import build_production_race
        from core.analyzer import CombinedRaceAnalyzer
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"analyzer/backtest import unavailable: {e}")

    CUR = "2026/04/01"
    db = HorseDatabase(db_path=str(tmp_path / "parity.db"))
    # Prior history (strictly before CUR) — the only data a leak-free score may use
    prior = [
        ("2026/01/04", 1, "ALPHA", 1, "J A", "T A", 4, 0, 0, 2.0),
        ("2026/02/01", 1, "ALPHA", 1, "J A", "T A", 5, 0, 0, 3.5),
        ("2026/01/04", 1, "BETA", 2, "J B", "T B", 6, 0, 0, 4.0),
        ("2026/02/01", 1, "BETA", 2, "J B", "T B", 7, 0, 0, 5.0),
    ]
    for r in prior:
        _insert(db.conn, *r)
    db.conn.commit()

    cur_rows = [
        {"race_date": CUR, "race_number": 1, "horse_name": "ALPHA", "horse_number": 1,
         "jockey": "J A", "trainer": "T A", "draw": 1, "win_odds": 3.5, "distance": 1200,
         "venue": "Sha Tin", "going": "Good", "finish_position": 1, "is_winner": 1,
         "is_top3": 1, "margin": 0.0},
        {"race_date": CUR, "race_number": 1, "horse_name": "BETA", "horse_number": 2,
         "jockey": "J B", "trainer": "T B", "draw": 7, "win_odds": 5.0, "distance": 1200,
         "venue": "Sha Tin", "going": "Good", "finish_position": 2, "is_winner": 0,
         "is_top3": 1, "margin": 1.0},
    ]
    race = build_production_race(cur_rows)
    analyzer = CombinedRaceAnalyzer()
    analyzer._horse_db = db                       # point the rule engine at the fixture

    def rule_scores():
        analyzer._par_times_by_date.clear()       # avoid cross-run cache state
        return {a.horse.name: round(a.rule_score, 6) for a in analyzer.analyze_race(race)}

    live = rule_scores()                          # DB has only PRIOR races (leak-free)

    # The current race's OWN result + 3 FUTURE wins each — what an un-dated
    # lookup would wrongly fold into "recent form" during backtest.
    leaky_rows = []
    for c in cur_rows:
        leaky_rows.append((c["race_date"], 1, c["horse_name"], c["horse_number"],
                           c["jockey"], c["trainer"], c["finish_position"],
                           c["is_winner"], c["is_top3"], c["margin"]))
        for i, fdate in enumerate(("2026/04/15", "2026/05/01", "2026/05/15")):
            leaky_rows.append((fdate, 1, c["horse_name"], c["horse_number"],
                               c["jockey"], c["trainer"], 1, 1, 1, 0.0))  # future WINS
    for r in leaky_rows:
        _insert(db.conn, *r)
    db.conn.commit()
    backtest = rule_scores()                      # DB now has current + future races too

    assert live == backtest, (
        f"rule_score changed when current+future races were added to the DB → a "
        f"rule factor is LEAKING the future.\n  live={live}\n  bt={backtest}")
