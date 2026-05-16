"""JP racing database — schema + idempotent ingestion.

Stores The Racing API result payloads for Japanese racing (JRA stakes only
at this plan tier) in a separate SQLite file from UK/AU/HK. See
config.JP_DB_PATH for the file location.

Schema is identical to UK/AU (same upstream API, same field names). Kept
as a separate file rather than a shared parameterised class — see
data/au_database.py header for the same reasoning.

Coverage caveat: on the standard plan tier, JP racing comes through with
OR (official rating) and RPR (Racing Post Rating) as '–'. Stored as TEXT
so the marker survives; feature extraction would need to handle the gap
if a JP model were ever attempted (it isn't planned — too few races).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import config


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jp_races (
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

CREATE INDEX IF NOT EXISTS idx_jp_races_date ON jp_races(date);
CREATE INDEX IF NOT EXISTS idx_jp_races_course ON jp_races(course);
CREATE INDEX IF NOT EXISTS idx_jp_races_region ON jp_races(region);

CREATE TABLE IF NOT EXISTS jp_race_results (
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
    FOREIGN KEY (race_id) REFERENCES jp_races(race_id)
);

CREATE INDEX IF NOT EXISTS idx_jp_results_horse_id ON jp_race_results(horse_id);
CREATE INDEX IF NOT EXISTS idx_jp_results_horse ON jp_race_results(horse);
CREATE INDEX IF NOT EXISTS idx_jp_results_jockey_id ON jp_race_results(jockey_id);
CREATE INDEX IF NOT EXISTS idx_jp_results_trainer_id ON jp_race_results(trainer_id);
CREATE INDEX IF NOT EXISTS idx_jp_results_race_pos ON jp_race_results(race_id, finish_position);
"""


def _to_int(v: Any) -> Optional[int]:
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
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


class JPHorseDatabase:
    """SQLite store for The Racing API JP result payloads.

    Schema-compatible with UK/AU stores at the column level so future
    cross-jurisdiction queries can UNION ALL across all three.
    """

    def __init__(self, db_path: Optional[str] = None):
        path = str(db_path or config.JP_DB_PATH)
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

    def ingest_results(self, races: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        n_races = 0
        n_runners = 0
        now = datetime.now(timezone.utc).isoformat(timespec='seconds')
        for race in races:
            runners = race.get('runners', []) or []
            self.conn.execute(
                """INSERT OR REPLACE INTO jp_races (
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
                    """INSERT OR REPLACE INTO jp_race_results (
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

    def get_summary(self) -> Dict[str, int]:
        cur = self.conn.cursor()
        races = cur.execute("SELECT COUNT(*) FROM jp_races").fetchone()[0]
        runners = cur.execute("SELECT COUNT(*) FROM jp_race_results").fetchone()[0]
        horses = cur.execute("SELECT COUNT(DISTINCT horse_id) FROM jp_race_results").fetchone()[0]
        jockeys = cur.execute("SELECT COUNT(DISTINCT jockey_id) FROM jp_race_results").fetchone()[0]
        trainers = cur.execute("SELECT COUNT(DISTINCT trainer_id) FROM jp_race_results").fetchone()[0]
        first = cur.execute("SELECT MIN(date) FROM jp_races").fetchone()[0]
        last = cur.execute("SELECT MAX(date) FROM jp_races").fetchone()[0]
        return {
            'races': races, 'runners': runners,
            'horses': horses, 'jockeys': jockeys, 'trainers': trainers,
            'first_date': first, 'last_date': last,
        }
