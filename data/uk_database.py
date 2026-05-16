"""UK racing database — schema + idempotent ingestion.

Stores The Racing API result payloads (UK + IRE + global if user chooses)
in a separate SQLite file from HK so the schemas, ingestion paths, and
future ML models stay fully isolated. See config.UK_DB_PATH for the
file location.

Schema notes:
- Race-level fields mirror the API exactly so we never lose information
  on ingestion. Field name parity with the API makes debugging easy.
- Per-runner row uses the API's stable IDs (horse_id, jockey_id,
  trainer_id) as the join keys — names can collide across years and
  across regions.
- INSERT OR REPLACE on (race_id, horse_id) for idempotency. Re-running
  ingestion for the same date is safe — newer payload overwrites older.
- `rpr` and `topspeed` stored as TEXT because the API returns '–' for
  accounts without the higher-tier subscription; '–' is not parseable
  as int. Cast to int at feature-extraction time if/when available.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import config


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS uk_races (
    race_id        TEXT PRIMARY KEY,
    date           TEXT NOT NULL,
    region         TEXT,
    course         TEXT,
    course_id      TEXT,
    off_time       TEXT,
    off_dt         TEXT,
    race_name      TEXT,
    race_type      TEXT,
    race_class     TEXT,
    pattern        TEXT,
    rating_band    TEXT,
    age_band       TEXT,
    sex_rest       TEXT,
    dist_str       TEXT,
    dist_yards     INTEGER,
    dist_meters    INTEGER,
    dist_furlongs  TEXT,
    going          TEXT,
    surface        TEXT,
    is_jumps       INTEGER DEFAULT 0,
    runner_count   INTEGER,
    ingested_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_uk_races_date ON uk_races(date);
CREATE INDEX IF NOT EXISTS idx_uk_races_course ON uk_races(course);
CREATE INDEX IF NOT EXISTS idx_uk_races_region ON uk_races(region);

CREATE TABLE IF NOT EXISTS uk_race_results (
    race_id              TEXT NOT NULL,
    horse_id             TEXT NOT NULL,
    horse                TEXT NOT NULL,
    finish_position      INTEGER,
    finish_position_raw  TEXT,
    saddle_number        INTEGER,
    draw                 INTEGER,
    sp_str               TEXT,
    sp_decimal           REAL,
    bsp                  REAL,
    is_favourite         INTEGER DEFAULT 0,
    beaten_distance      TEXT,
    overall_btn          TEXT,
    age                  INTEGER,
    sex                  TEXT,
    weight_str           TEXT,
    weight_lbs           INTEGER,
    headgear             TEXT,
    finish_time          TEXT,
    official_rating      INTEGER,
    rpr                  TEXT,
    topspeed             TEXT,
    prize                INTEGER,
    jockey               TEXT,
    jockey_id            TEXT,
    jockey_claim_lbs     INTEGER,
    trainer              TEXT,
    trainer_id           TEXT,
    owner                TEXT,
    owner_id             TEXT,
    sire                 TEXT,
    sire_id              TEXT,
    dam                  TEXT,
    dam_id               TEXT,
    damsire              TEXT,
    damsire_id           TEXT,
    comment              TEXT,
    silk_url             TEXT,
    PRIMARY KEY (race_id, horse_id),
    FOREIGN KEY (race_id) REFERENCES uk_races(race_id)
);

CREATE INDEX IF NOT EXISTS idx_uk_results_horse_id ON uk_race_results(horse_id);
CREATE INDEX IF NOT EXISTS idx_uk_results_horse ON uk_race_results(horse);
CREATE INDEX IF NOT EXISTS idx_uk_results_jockey_id ON uk_race_results(jockey_id);
CREATE INDEX IF NOT EXISTS idx_uk_results_trainer_id ON uk_race_results(trainer_id);
CREATE INDEX IF NOT EXISTS idx_uk_results_race_pos ON uk_race_results(race_id, finish_position);
"""


# ---------------------------------------------------------------------------
# Field coercion helpers
# ---------------------------------------------------------------------------

def _to_int(v: Any) -> Optional[int]:
    """Best-effort int coercion. API returns numeric values as strings; some
    fields (position, prize, weight_lbs) need int. '–' / '' / None → None."""
    if v is None or v == "" or v == "–":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "" or v == "–":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm_text(v: Any) -> Optional[str]:
    """Empty string → None for nullable text columns; preserves '–' which
    is meaningful in the API (subscription gate marker)."""
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# Database wrapper
# ---------------------------------------------------------------------------

class UKHorseDatabase:
    """SQLite store for The Racing API result payloads.

    Minimal first-cut: schema + idempotent ingestion + basic counts.
    Read methods that conform to core.ports.HorseHistoryStore will be
    added as the analyzer side of the UK pipeline gets built — YAGNI
    until a real call-site exists.
    """

    def __init__(self, db_path: Optional[str] = None):
        path = str(db_path or config.UK_DB_PATH)
        # Make sure parent dir exists (tracking_data/ should but be safe).
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ----- Ingestion -----

    def ingest_results(self, races: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        """Insert / overwrite race + runner rows for an iterable of API race dicts.

        Returns: {'races': N_races_written, 'runners': N_runners_written}.
        Idempotent — re-running for the same race_id replaces the prior rows.
        """
        n_races = 0
        n_runners = 0
        now = datetime.now(timezone.utc).isoformat(timespec='seconds')
        for race in races:
            runners = race.get('runners', []) or []
            self.conn.execute(
                """INSERT OR REPLACE INTO uk_races (
                    race_id, date, region, course, course_id,
                    off_time, off_dt, race_name, race_type, race_class,
                    pattern, rating_band, age_band, sex_rest,
                    dist_str, dist_yards, dist_meters, dist_furlongs,
                    going, surface, is_jumps, runner_count, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    race.get('race_id'),
                    race.get('date'),
                    race.get('region'),
                    race.get('course'),
                    race.get('course_id'),
                    race.get('off'),
                    race.get('off_dt'),
                    race.get('race_name'),
                    race.get('type'),
                    race.get('class'),
                    race.get('pattern'),
                    race.get('rating_band'),
                    race.get('age_band'),
                    race.get('sex_rest'),
                    race.get('dist'),
                    _to_int(race.get('dist_y')),
                    _to_int(race.get('dist_m')),
                    race.get('dist_f'),
                    race.get('going'),
                    race.get('surface'),
                    1 if (race.get('jumps') or '').strip() else 0,
                    len(runners),
                    now,
                ),
            )
            n_races += 1

            for r in runners:
                sp_str = r.get('sp') or ''
                is_fav = 1 if 'F' in sp_str.upper() else 0
                self.conn.execute(
                    """INSERT OR REPLACE INTO uk_race_results (
                        race_id, horse_id, horse,
                        finish_position, finish_position_raw,
                        saddle_number, draw,
                        sp_str, sp_decimal, bsp, is_favourite,
                        beaten_distance, overall_btn,
                        age, sex, weight_str, weight_lbs, headgear,
                        finish_time, official_rating, rpr, topspeed, prize,
                        jockey, jockey_id, jockey_claim_lbs,
                        trainer, trainer_id, owner, owner_id,
                        sire, sire_id, dam, dam_id, damsire, damsire_id,
                        comment, silk_url
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        race.get('race_id'),
                        r.get('horse_id'),
                        r.get('horse'),
                        _to_int(r.get('position')),
                        _norm_text(r.get('position')),
                        _to_int(r.get('number')),
                        _to_int(r.get('draw')),
                        _norm_text(r.get('sp')),
                        _to_float(r.get('sp_dec')),
                        _to_float(r.get('bsp')),
                        is_fav,
                        _norm_text(r.get('btn')),
                        _norm_text(r.get('ovr_btn')),
                        _to_int(r.get('age')),
                        _norm_text(r.get('sex')),
                        _norm_text(r.get('weight')),
                        _to_int(r.get('weight_lbs')),
                        _norm_text(r.get('headgear')),
                        _norm_text(r.get('time')),
                        _to_int(r.get('or')),
                        _norm_text(r.get('rpr')),
                        _norm_text(r.get('tsr')),
                        _to_int(r.get('prize')),
                        _norm_text(r.get('jockey')),
                        _norm_text(r.get('jockey_id')),
                        _to_int(r.get('jockey_claim_lbs')),
                        _norm_text(r.get('trainer')),
                        _norm_text(r.get('trainer_id')),
                        _norm_text(r.get('owner')),
                        _norm_text(r.get('owner_id')),
                        _norm_text(r.get('sire')),
                        _norm_text(r.get('sire_id')),
                        _norm_text(r.get('dam')),
                        _norm_text(r.get('dam_id')),
                        _norm_text(r.get('damsire')),
                        _norm_text(r.get('damsire_id')),
                        _norm_text(r.get('comment')),
                        _norm_text(r.get('silk_url')),
                    ),
                )
                n_runners += 1
        self.conn.commit()
        return {'races': n_races, 'runners': n_runners}

    # ----- Read helpers -----

    def get_summary(self) -> Dict[str, int]:
        cur = self.conn.cursor()
        races = cur.execute("SELECT COUNT(*) FROM uk_races").fetchone()[0]
        runners = cur.execute("SELECT COUNT(*) FROM uk_race_results").fetchone()[0]
        horses = cur.execute("SELECT COUNT(DISTINCT horse_id) FROM uk_race_results").fetchone()[0]
        jockeys = cur.execute("SELECT COUNT(DISTINCT jockey_id) FROM uk_race_results").fetchone()[0]
        trainers = cur.execute("SELECT COUNT(DISTINCT trainer_id) FROM uk_race_results").fetchone()[0]
        first = cur.execute("SELECT MIN(date) FROM uk_races").fetchone()[0]
        last = cur.execute("SELECT MAX(date) FROM uk_races").fetchone()[0]
        return {
            'races': races, 'runners': runners,
            'horses': horses, 'jockeys': jockeys, 'trainers': trainers,
            'first_date': first, 'last_date': last,
        }
