"""A failed `rebuild_stats()` must not leave a pending DELETE on the connection.

Bug (code review 2026-08-03): `_ingest_race_day_inner` calls `rebuild_stats()`
inside `try/except Exception: logger.error(...)` with NO rollback. But
`rebuild_stats` opens with `DELETE FROM jockey_stats` / `DELETE FROM
trainer_stats` and only commits at the end. If it raises in between, the
DELETEs stay pending on the shared connection — and the next unrelated
`commit()` (e.g. `backfill_card_fields()` during auto-learn, which runs
immediately afterwards on the same connection) silently finalises them.

Result: `jockey_stats` and `trainer_stats` are permanently emptied while the
log claims only that stats "may be stale". `get_jockey_stats()` /
`get_trainer_stats()` then return None for every rider and trainer, zeroing
the jockey and trainer rule-engine factors for the whole race day.
"""
import pytest

from data.database import HorseDatabase


@pytest.fixture
def db(tmp_path):
    d = HorseDatabase(db_path=str(tmp_path / "t.db"))
    d.conn.execute(
        "INSERT INTO jockey_stats (name, season, wins, rides) VALUES ('Z Purton', '2026', 10, 40)"
    )
    d.conn.execute(
        "INSERT INTO trainer_stats (name, season, wins, runners) VALUES ('J Size', '2026', 5, 30)"
    )
    d.conn.commit()
    yield d
    d.close()


def _counts(d):
    return (
        d.conn.execute("SELECT COUNT(*) FROM jockey_stats").fetchone()[0],
        d.conn.execute("SELECT COUNT(*) FROM trainer_stats").fetchone()[0],
    )


class TestRebuildStatsRollback:
    def test_failed_rebuild_leaves_no_pending_transaction(self, db, monkeypatch):
        """After a swallowed failure the connection must be clean."""
        def boom(*a, **k):
            db.conn.execute("DELETE FROM jockey_stats")
            db.conn.execute("DELETE FROM trainer_stats")
            raise RuntimeError("renavon ranking tables missing")

        monkeypatch.setattr(db, "rebuild_stats", boom)
        db._rebuild_stats_after_ingest()

        assert not db.conn.in_transaction, (
            "a pending DELETE survived the swallowed rebuild_stats failure"
        )

    def test_later_unrelated_commit_does_not_wipe_stats(self, db, monkeypatch):
        """The real-world consequence: auto-learn's next commit must be safe."""
        def boom(*a, **k):
            db.conn.execute("DELETE FROM jockey_stats")
            db.conn.execute("DELETE FROM trainer_stats")
            raise RuntimeError("renavon ranking tables missing")

        monkeypatch.setattr(db, "rebuild_stats", boom)
        db._rebuild_stats_after_ingest()

        # Something else on the same connection commits later (backfill_card_fields).
        db.conn.execute(
            "INSERT INTO jockey_stats (name, season, wins, rides) VALUES ('K Teetan', '2026', 3, 20)"
        )
        db.conn.commit()

        jockeys, trainers = _counts(db)
        assert jockeys == 2, f"jockey_stats was wiped by a later commit ({jockeys} rows)"
        assert trainers == 1, f"trainer_stats was wiped by a later commit ({trainers} rows)"

    def test_successful_rebuild_is_unaffected(self, db, monkeypatch):
        """Guard: the happy path must still commit its work."""
        def ok(*a, **k):
            db.conn.execute("DELETE FROM jockey_stats")
            db.conn.execute(
                "INSERT INTO jockey_stats (name, season, wins, rides) "
                "VALUES ('C Y Ho', '2026', 7, 50)"
            )
            db.conn.commit()

        monkeypatch.setattr(db, "rebuild_stats", ok)
        db._rebuild_stats_after_ingest()

        rows = db.conn.execute(
            "SELECT name FROM jockey_stats"
        ).fetchall()
        assert [r[0] for r in rows] == ["C Y Ho"]
