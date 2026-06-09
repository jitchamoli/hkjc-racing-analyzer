"""One-time data repair for the 2026-06-08 audit findings.

Fixes corruption caused by the results-page header-parse bug (distance defaulted
to 1200), plus a few isolated ingest defects. Idempotent — safe to re-run.

Parts:
  A. distance/class_type/venue ← authoritative prediction JSON, for every
     results/race_data_*_with_results.json (uses the fixed backfill_card_fields).
  B. finish_time ← sum(sectional_times) for finished rows that lost finish_time
     but kept sectionals (2026/05/03 R7-R11).
  C. delete the 14 phantom rows merged under 2025/11/15 R9 (dot-format finish_time
     set whose winner does NOT match the WIN dividend; verified before deleting).
  D. null the corrupt running_position on 2026/05/03 R7 (column-bleed: tokens far
     exceed field size, e.g. '1 50 86').

The root cause is fixed in data/scraper.py (header parse) + data/database.py
(JSON-authoritative backfill + ingest distance guards). This script only repairs
already-stored rows. Run scripts.dump_db afterwards to persist.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.database import HorseDatabase, _VALID_HK_DISTANCES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fname_to_date(path: str) -> str:
    m = re.search(r"race_data_(\d{4})(\d{2})(\d{2})_with_results\.json", path)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else ""


def part_a_card_fields():
    print("=== PART A: distance/class/venue ← prediction JSON (authoritative) ===")
    db = HorseDatabase()
    c = db.conn.cursor()
    total = 0
    try:
        for path in sorted(glob.glob(os.path.join(REPO_ROOT, "results",
                                                   "race_data_*_with_results.json"))):
            date = _fname_to_date(path)
            if not date:
                continue
            try:
                data = json.load(open(path))
            except Exception as e:
                print(f"  skip {path}: {e}"); continue
            races = data["races"] if isinstance(data, dict) and "races" in data else data
            # snapshot distances before
            before = dict(c.execute(
                "SELECT race_number, distance FROM race_results WHERE race_date=? "
                "GROUP BY race_number", (date,)).fetchall())
            db.backfill_card_fields(date, races)
            after = dict(c.execute(
                "SELECT race_number, distance FROM race_results WHERE race_date=? "
                "GROUP BY race_number", (date,)).fetchall())
            changed = [(rn, before.get(rn), after.get(rn))
                       for rn in after if before.get(rn) != after.get(rn)]
            if changed:
                total += len(changed)
                print(f"  {date}: distance fixed on {len(changed)} race(s): "
                      + ", ".join(f"R{rn} {b}->{a}" for rn, b, a in sorted(changed)))
    finally:
        db.close()
    print(f"  Part A: {total} race-distance corrections\n")


def part_b_finish_time_from_sectionals():
    print("=== PART B: finish_time ← sum(sectional_times) for blank-but-sectional rows ===")
    conn = sqlite3.connect(str(config.DB_PATH))
    c = conn.cursor()
    rows = c.execute(
        "SELECT rowid, sectional_times FROM race_results "
        "WHERE finish_position > 0 AND IFNULL(finish_time,'')='' "
        "AND IFNULL(sectional_times,'')!=''").fetchall()
    fixed = 0
    for rowid, sec in rows:
        try:
            total = sum(float(x) for x in sec.split())
        except ValueError:
            continue
        if not (40 < total < 200):   # sane race-time bound
            continue
        m = int(total // 60)
        s = total - 60 * m
        ft = f"{m}:{s:05.2f}"        # e.g. 1:39.85 ; parse_finish_time handles it
        c.execute("UPDATE race_results SET finish_time=? WHERE rowid=?", (ft, rowid))
        fixed += 1
    conn.commit(); conn.close()
    print(f"  Part B: {fixed} finish_time values derived from sectionals\n")


def part_c_delete_phantom_2025_11_15_r9():
    print("=== PART C: delete phantom rows merged under 2025/11/15 R9 ===")
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    D, R = "2025/11/15", 9
    rows = c.execute("SELECT id, horse_number, horse_name, finish_position, finish_time, win_odds "
                     "FROM race_results WHERE race_date=? AND race_number=?", (D, R)).fetchall()
    if len(rows) != 28:
        print(f"  R9 has {len(rows)} rows (expected 28) — already repaired or changed; skipping.\n")
        conn.close(); return
    colon = [r for r in rows if ":" in (r["finish_time"] or "")]
    dot = [r for r in rows if ":" not in (r["finish_time"] or "")]
    win = c.execute("SELECT combination FROM race_dividends WHERE race_date=? AND race_number=? "
                    "AND pool='WIN'", (D, R)).fetchone()
    win_no = str(win["combination"]).strip() if win else None
    colon_winner = next((str(r["horse_number"]) for r in colon if r["finish_position"] == 1), None)
    dot_winner = next((str(r["horse_number"]) for r in dot if r["finish_position"] == 1), None)
    # Safety: only delete the dot set if (a) split is exactly 14/14, (b) the colon
    # winner matches the official WIN dividend, and (c) the dot winner does NOT.
    if len(colon) == 14 and len(dot) == 14 and colon_winner == win_no and dot_winner != win_no:
        ids = [r["id"] for r in dot]
        c.executemany("DELETE FROM race_results WHERE id=?", [(i,) for i in ids])
        conn.commit()
        print(f"  Deleted {len(ids)} phantom rows (winner #{dot_winner}, not the WIN-div #{win_no}). "
              f"Kept the real race (winner #{colon_winner} = WIN-div).\n")
    else:
        print(f"  ABORT: safety check failed (colon={len(colon)} dot={len(dot)} "
              f"colon_winner={colon_winner} dot_winner={dot_winner} win_div={win_no}). "
              f"No rows deleted — inspect manually.\n")
    conn.close()


def part_d_null_corrupt_running_position():
    print("=== PART D: null corrupt running_position on 2026/05/03 R7 ===")
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    D, R, MAXFIELD = "2026/05/03", 7, 14
    rows = c.execute("SELECT id, running_position FROM race_results "
                     "WHERE race_date=? AND race_number=?", (D, R)).fetchall()
    fixed = 0
    for r in rows:
        rp = (r["running_position"] or "").strip()
        # space-separated only (concatenated '665' is a legit single token)
        if " " in rp and any(tok.isdigit() and int(tok) > MAXFIELD + 6 for tok in rp.split()):
            c.execute("UPDATE race_results SET running_position='' WHERE id=?", (r["id"],))
            fixed += 1
    conn.commit(); conn.close()
    print(f"  Part D: nulled {fixed} corrupt running_position values (re-scrape to refill)\n")


def main():
    part_a_card_fields()
    part_b_finish_time_from_sectionals()
    part_c_delete_phantom_2025_11_15_r9()
    part_d_null_corrupt_running_position()
    print("Repair complete. Run `python3 -m scripts.dump_db` to persist the DB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
