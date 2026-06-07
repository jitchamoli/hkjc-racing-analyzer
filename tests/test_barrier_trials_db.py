"""barrier_trials table + HorseDatabase.ingest_barrier_trials (2026-06-07)."""
from data.database import HorseDatabase


def _row(name, pos, heat=1, date="2026/05/29", **kw):
    r = {"date": date, "horse_name": name, "position": pos, "heat": heat,
         "margin": "", "time": "1.10.0", "running_position": "1 1 1",
         "jockey": "Z Purton", "trainer": "J Size", "draw": "5", "gear": ""}
    r.update(kw)
    return r


def test_ingest_and_readback():
    db = HorseDatabase(":memory:")
    try:
        n = db.ingest_barrier_trials(
            [_row("AEROVOLANIC", 1, margin=""), _row("WITH A SMILE", 2, margin="3/4L")],
            venue="ST")
        assert n == 2
        rows = db.conn.execute(
            "SELECT horse_name, finish_position, margin, venue FROM barrier_trials "
            "ORDER BY finish_position").fetchall()
        assert [r[0] for r in rows] == ["AEROVOLANIC", "WITH A SMILE"]
        assert rows[1][2] == "3/4L" and rows[0][3] == "ST"
    finally:
        db.close()


def test_idempotent_reingest():
    db = HorseDatabase(":memory:")
    try:
        db.ingest_barrier_trials([_row("AEROVOLANIC", 1)], venue="ST")
        db.ingest_barrier_trials([_row("AEROVOLANIC", 1, time="9.99.99")], venue="ST")  # re-scrape
        rows = db.conn.execute("SELECT finish_time FROM barrier_trials").fetchall()
        assert len(rows) == 1 and rows[0][0] == "9.99.99"  # replaced, not duplicated
    finally:
        db.close()


def test_skips_rows_missing_keys():
    db = HorseDatabase(":memory:")
    try:
        n = db.ingest_barrier_trials([
            {"horse_name": "NO DATE", "position": 1, "heat": 1},   # no date
            {"date": "2026/05/29", "position": 2, "heat": 1},      # no name
            _row("VALID", 1),
        ], venue="ST")
        assert n == 1
        assert db.conn.execute("SELECT COUNT(*) FROM barrier_trials").fetchone()[0] == 1
    finally:
        db.close()
