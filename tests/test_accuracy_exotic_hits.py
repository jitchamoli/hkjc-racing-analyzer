"""Race-level hit detection for QIN, TRIO, WIN, PLACE, BANKER bets.

Locks the get_accuracy_summary / get_bet_type_roi rewrite (2026-05-17):
- Bets are evaluated at race-level, not per-horse (a single QIN race used
  to inflate bets_placed by 3× and never register a hit because the win
  matcher only knew 'WIN' / 'BANKER WIN' / 'PLACE').
- QIN hits when ≥2 of the model's top-3 picks finish in top 2.
- TRIO hits when all 3 of the model's top-3 picks finish in top 3.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from data.database import HorseDatabase


def _make_db():
    """Spin up a fresh on-disk SQLite (HorseDatabase auto-creates schema)."""
    tmpfile = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    tmpfile.close()
    path = Path(tmpfile.name)
    db = HorseDatabase(db_path=str(path))
    return db, path


def _insert_race(conn, date, race_no, picks):
    """picks = [(rank, horse_name, bet, actual_pos, actual_won, actual_top3, profit_loss), ...]

    horse_number is set = rank for deterministic dividend-combo lookups in tests.
    """
    for rank, horse, bet, pos, won, t3, pnl in picks:
        conn.execute("""
            INSERT INTO prediction_log
            (race_date, race_number, horse_name, horse_number, predicted_rank,
             bet_recommendation, calibrated_prob, win_odds,
             actual_position, actual_won, actual_top3, profit_loss,
             confidence, results_filled, predicted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,datetime('now'))
        """, (date, race_no, horse, rank, rank, bet, 0.1, 5.0,
              pos, won, t3, pnl, 'HIGH'))
    conn.commit()


def _insert_dividend(conn, date, race_no, pool, combination, dividend):
    """Seed race_dividends so _real_exotic_pnl can look up exotic payouts."""
    conn.execute("""
        INSERT INTO race_dividends (race_date, race_number, pool, combination, dividend)
        VALUES (?,?,?,?,?)
    """, (date, race_no, pool, combination, dividend))
    conn.commit()


class TestRaceLevelHitDetection:
    def test_qin_hits_when_two_of_top3_finish_top2(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            # Model picks A/B/C as top-3; A finishes 1st, B finishes 2nd → QIN hit
            _insert_race(conn, '2026/01/01', 1, [
                (1, 'A', 'QIN', 1, 1, 1, -10),
                (2, 'B', 'QIN', 2, 0, 1, -10),
                (3, 'C', 'QIN', 5, 0, 0, -10),
                (4, 'D', 'SKIP', 3, 0, 1, 0),
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_placed'] == 1, "should count race-level, not 3 per horse"
            assert s['bets_won'] == 1, "QIN with 2 top-3 picks in top 2 = hit"
        finally:
            path.unlink(missing_ok=True)

    def test_qin_misses_when_only_one_of_top3_in_top2(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            # Only A finishes top 2; B and C don't → QIN miss
            _insert_race(conn, '2026/01/02', 1, [
                (1, 'A', 'QIN', 1, 1, 1, -10),
                (2, 'B', 'QIN', 4, 0, 0, -10),
                (3, 'C', 'QIN', 8, 0, 0, -10),
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_placed'] == 1
            assert s['bets_won'] == 0
        finally:
            path.unlink(missing_ok=True)

    def test_trio_hits_when_all_top3_finish_top3(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            # All 3 picks finish in top 3 (positions 2, 1, 3) → TRIO hit
            _insert_race(conn, '2026/01/03', 1, [
                (1, 'A', 'TRIO', 2, 0, 1, -10),
                (2, 'B', 'TRIO', 1, 1, 1, -10),
                (3, 'C', 'TRIO', 3, 0, 1, -10),
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_placed'] == 1
            assert s['bets_won'] == 1
        finally:
            path.unlink(missing_ok=True)

    def test_trio_misses_when_one_pick_outside_top3(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            # C finishes 4th → only 2 of top-3 picks in top 3 → TRIO miss
            _insert_race(conn, '2026/01/04', 1, [
                (1, 'A', 'TRIO', 1, 1, 1, -10),
                (2, 'B', 'TRIO', 2, 0, 1, -10),
                (3, 'C', 'TRIO', 4, 0, 0, -10),
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_placed'] == 1
            assert s['bets_won'] == 0
        finally:
            path.unlink(missing_ok=True)

    def test_box_q4_qin_hits_when_rank1_and_rank4_in_top2(self):
        """Regression for 2026-05-17 audit fix.

        BOX Q4 stakes 6u across 4 legs (all 4 horses marked 'QIN'). If rank-1
        and rank-4 finish 1-2, the ticket hits. Pre-fix code only checked
        horses[:3], missing rank-4 entirely → false MISS + false -6u P&L.
        """
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/01/15', 1, [
                (1, 'A', 'QIN', 1, 1, 1, -10),  # rank-1 wins
                (2, 'B', 'QIN', 5, 0, 0, -10),
                (3, 'C', 'QIN', 7, 0, 0, -10),
                (4, 'D', 'QIN', 2, 0, 1, -10),  # rank-4 finishes 2nd
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_placed'] == 1
            assert s['bets_won'] == 1, "BOX Q4 covers rank-1 + rank-4 → hit"
        finally:
            path.unlink(missing_ok=True)

    def test_box_q4_trio_hits_when_three_of_four_in_top3(self):
        """BOX TRIO-4 (4 legs, 4 combos) hits when any 3 of the 4 finish top 3."""
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/01/16', 1, [
                (1, 'A', 'TRIO', 1, 1, 1, -10),
                (2, 'B', 'TRIO', 5, 0, 0, -10),  # rank-2 misses
                (3, 'C', 'TRIO', 3, 0, 1, -10),
                (4, 'D', 'TRIO', 2, 0, 1, -10),  # rank-4 places 2nd → still 3-of-4 in top 3
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_won'] == 1, "rank-1,3,4 all in top 3 → TRIO-4 hit"
        finally:
            path.unlink(missing_ok=True)

    def test_win_bet_hits_only_when_rank1_wins(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/01/05', 1, [
                (1, 'A', 'WIN', 1, 1, 1, 50),
                (2, 'B', 'WIN', 5, 0, 0, 0),
            ])
            _insert_race(conn, '2026/01/05', 2, [
                (1, 'X', 'WIN', 3, 0, 1, -10),
                (2, 'Y', 'WIN', 1, 1, 1, 0),
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_placed'] == 2
            assert s['bets_won'] == 1, "rank-1 won in race 1 only"
        finally:
            path.unlink(missing_ok=True)

    def test_place_bet_hits_when_rank1_top3(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/01/06', 1, [
                (1, 'A', 'PLACE', 3, 0, 1, 5),
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_won'] == 1
        finally:
            path.unlink(missing_ok=True)

    def test_skip_excluded_from_bets_placed(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/01/07', 1, [
                (1, 'A', 'SKIP', 1, 1, 1, 0),
                (2, 'B', 'SKIP', 2, 0, 1, 0),
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_placed'] == 0
            assert s['rank1_wins'] == 1, "SKIP doesn't suppress rank-1 win tracking"
        finally:
            path.unlink(missing_ok=True)

    def test_bet_type_roi_uses_race_level_grouping(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            # 2 QIN races (1 hit, 1 miss), 1 TRIO race (hit)
            _insert_race(conn, '2026/01/10', 1, [
                (1, 'A', 'QIN', 1, 1, 1, -10),
                (2, 'B', 'QIN', 2, 0, 1, -10),
                (3, 'C', 'QIN', 7, 0, 0, -10),
            ])
            _insert_race(conn, '2026/01/10', 2, [
                (1, 'D', 'QIN', 5, 0, 0, -10),
                (2, 'E', 'QIN', 6, 0, 0, -10),
                (3, 'F', 'QIN', 7, 0, 0, -10),
            ])
            _insert_race(conn, '2026/01/10', 3, [
                (1, 'G', 'TRIO', 1, 1, 1, -10),
                (2, 'H', 'TRIO', 2, 0, 1, -10),
                (3, 'I', 'TRIO', 3, 0, 1, -10),
            ])
            conn.close()
            by_type = {b['bet_type']: b for b in db.get_bet_type_roi(days=9999)}
            assert by_type['QIN']['count'] == 2, "race-level, not 6"
            assert by_type['QIN']['hits'] == 1
            assert by_type['TRIO']['count'] == 1
            assert by_type['TRIO']['hits'] == 1
        finally:
            path.unlink(missing_ok=True)

    def test_qin_hit_pnl_uses_real_dividend(self):
        """Hit QIN should credit (dividend - stake) using real race_dividends row."""
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            # Straight QIN (2 legs only) — rank 3 is SKIP, not part of bet
            _insert_race(conn, '2026/02/15', 1, [
                (1, 'A', 'QIN', 1, 1, 1, -10),
                (2, 'B', 'QIN', 2, 0, 1, -10),
                (3, 'C', 'SKIP', 7, 0, 0, 0),
            ])
            # QUINELLA combination is sorted "low,high" — picks were horse_numbers 1+2
            _insert_dividend(conn, '2026/02/15', 1, 'QUINELLA', '1,2', 47.5)
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            # 1 bet (straight QIN, 2 legs → cost 1u), hit pays 47.5u, net +46.5u
            assert s['bets_won'] == 1
            assert abs(s['total_profit'] - 46.5) < 1e-6, f"got {s['total_profit']}"
            assert abs(s['total_stake'] - 1.0) < 1e-6
        finally:
            path.unlink(missing_ok=True)

    def test_qin_box_q3_uses_3u_stake(self):
        """BOX QIN with 3 legs costs 3u (3 combos) and uses real dividend."""
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/02/16', 1, [
                (1, 'A', 'QIN', 1, 1, 1, -10),
                (2, 'B', 'QIN', 2, 0, 1, -10),
                (3, 'C', 'QIN', 8, 0, 0, -10),
            ])
            _insert_dividend(conn, '2026/02/16', 1, 'QUINELLA', '1,2', 60.0)
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            # Hit: dividend 60 - stake 3 = +57u
            assert abs(s['total_profit'] - 57.0) < 1e-6
            assert abs(s['total_stake'] - 3.0) < 1e-6
        finally:
            path.unlink(missing_ok=True)

    def test_qin_miss_charges_stake(self):
        """Missed QIN charges -stake regardless of dividend table contents."""
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/02/17', 1, [
                (1, 'A', 'QIN', 5, 0, 0, -10),
                (2, 'B', 'QIN', 6, 0, 0, -10),
            ])
            # No dividend seeded; doesn't matter for a miss
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert abs(s['total_profit'] - (-1.0)) < 1e-6
            assert abs(s['total_stake'] - 1.0) < 1e-6
        finally:
            path.unlink(missing_ok=True)

    def test_qin_hit_no_dividend_falls_back_to_stored_pnl(self):
        """If dividend lookup fails, fall back to stored profit_loss (/10 normalization)."""
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/02/18', 1, [
                (1, 'A', 'QIN', 1, 1, 1, 50),    # stored pnl in $10 units
                (2, 'B', 'QIN', 2, 0, 1, 50),
            ])
            # No dividend seeded → fallback path
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            # rank1 stored profit_loss = 50 → normalized 5.0u
            assert abs(s['total_profit'] - 5.0) < 1e-6
        finally:
            path.unlink(missing_ok=True)

    def test_trio_hit_uses_real_dividend(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/02/19', 1, [
                (1, 'A', 'TRIO', 1, 1, 1, -10),
                (2, 'B', 'TRIO', 2, 0, 1, -10),
                (3, 'C', 'TRIO', 3, 0, 1, -10),
            ])
            _insert_dividend(conn, '2026/02/19', 1, 'TRIO', '1,2,3', 500.0)
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_won'] == 1
            # Straight TRIO: 1 combo, dividend 500 - stake 1 = +499
            assert abs(s['total_profit'] - 499.0) < 1e-6
        finally:
            path.unlink(missing_ok=True)

    def test_win_pnl_normalized_to_1u(self):
        """WIN bets: stored $10-stake pnl divided by 10 → 1u-stake pnl."""
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            # Hit WIN @ 5.0 odds: stored profit_loss = (5*10-10) = 40 → 1u: +4
            _insert_race(conn, '2026/02/20', 1, [
                (1, 'A', 'WIN', 1, 1, 1, 40),
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert abs(s['total_profit'] - 4.0) < 1e-6
            assert abs(s['total_stake'] - 1.0) < 1e-6
        finally:
            path.unlink(missing_ok=True)

    def test_unknown_bet_type_does_not_crash(self):
        db, path = _make_db()
        try:
            conn = sqlite3.connect(str(path))
            _insert_race(conn, '2026/01/11', 1, [
                (1, 'A', 'FUTURESBET', 1, 1, 1, -10),
            ])
            conn.close()
            s = db.get_accuracy_summary(days=9999)
            assert s['bets_placed'] == 1
            assert s['bets_won'] == 0, "unknown bet type counts as placed but never a hit"
        finally:
            path.unlink(missing_ok=True)
