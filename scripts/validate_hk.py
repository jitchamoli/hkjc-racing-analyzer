"""HK forward-validation log — the ONLY un-burned evidence we can generate.

WHY THIS EXISTS
---------------
The 668-race backtest holdout is BURNED: dozens of ship/kill decisions have
been made against it, so its ROI figures are upper bounds shaped by selection,
not honest out-of-sample edge. And a severe rule-engine date-leak (fixed
2026-05-22) inflated those figures further. Leak-free, the exotic boxes show
NEGATIVE ROI on that holdout — but we can't fully trust the negative either,
because the holdout is burned and the dividends are heavy-tailed.

The only way to know the truth is to FREEZE a cutoff and score every FUTURE
meeting exactly once, never tuning on it. This script does that.

HOW IT WORKS
------------
- It scores the ACTUAL LIVE predictions stored in prediction_log (predicted_rank
  per horse at race time) — NOT a re-run of the current model. That's what you
  would really have bet.
- A frozen cutoff date lives in tracking_data/hk_forward_cutoff.txt. Meetings
  STRICTLY AFTER the cutoff are the clean forward sample. On first run the cutoff
  defaults to the latest meeting already in prediction_log (so all prior, burned
  data is excluded).
- Per-meeting results are appended to tracking_data/hk_forward_validation.csv
  (idempotent — a date already logged is skipped). The CSV is the permanent record.

STRATEGIES SCORED (real HKJC dividends, per-$1 basis):
  straight QIN  — our top-2 == actual top-2          (stake 1)
  BOX-Q4        — actual top-2 ⊂ our top-4            (stake 6)
  TRIO-5        — actual top-3 ⊂ our top-5            (stake 10)
  TRIO-6        — actual top-3 ⊂ our top-6            (stake 20)
  F4 BOX-5      — actual top-4 ⊂ our top-5            (stake 5)

Usage:
  python3 -m scripts.validate_hk                 # log new clean meetings + report
  python3 -m scripts.validate_hk --report        # cumulative report only (no scan)
  python3 -m scripts.validate_hk --include-burned # ALSO show the pre-cutoff meetings,
                                                  #   clearly labelled (NOT written to CSV)
  python3 -m scripts.validate_hk --set-cutoff 2026/05/20
"""
from __future__ import annotations
import argparse
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import config
from betting.dividends import hkjc_declared_to_unit

CUTOFF_RE = re.compile(r'^\d{4}/\d{2}/\d{2}$')

CUTOFF_FILE = Path(config.DB_PATH).parent / "hk_forward_cutoff.txt"
LOG_FILE = Path(config.DB_PATH).parent / "hk_forward_validation.csv"

# (label, leg_n, box_n, n_combos, pool)  — leg/box define capture; pool the dividend
STRATS = [
    ("straight_QIN", 2, 2, 1, "QUINELLA"),
    ("BOX_Q4",       2, 4, 6, "QUINELLA"),
    ("TRIO_5",       3, 5, 10, "TRIO"),
    ("TRIO_6",       3, 6, 20, "TRIO"),
    ("F4_BOX_5",     4, 5, 5,  "FIRST 4"),
]
CSV_COLS = ["race_date", "races"] + [f"{s[0]}_{k}" for s in STRATS for k in ("stake", "ret", "hits")]


def _quin_div(conn, d, rno, a, b):
    lo, hi = sorted([int(a), int(b)])
    row = conn.execute(
        "SELECT dividend FROM race_dividends WHERE race_date=? AND race_number=? "
        "AND pool='QUINELLA' AND combination=?", (d, rno, f"{lo},{hi}")).fetchone()
    return hkjc_declared_to_unit(row[0]) if row else None


def _exotic_div(conn, d, rno, pool, combination):
    # combination is the winning permutation/set as stored in race_dividends:
    # TRIO + FIRST 4 = sorted horse-numbers comma-joined ("1,3,7", "1,3,7,8");
    # TIERCE + QUARTET = finish-order comma-joined ("7,8,1", "7,8,1,3").
    # Without this filter, fetchone() picks whichever row SQLite returns first
    # — fine today (one row per race/pool) but the May-22 dividend-basis bug
    # was the same family of silent-fetch mistake. Filter explicitly.
    row = conn.execute(
        "SELECT dividend FROM race_dividends WHERE race_date=? AND race_number=? "
        "AND pool=? AND combination=?", (d, rno, pool, combination)).fetchone()
    return hkjc_declared_to_unit(row[0]) if row else None


def _load_cutoff(conn) -> str:
    if CUTOFF_FILE.exists():
        cutoff = CUTOFF_FILE.read_text().strip()
        # An empty/whitespace/corrupted cutoff would silently make every
        # row pass `race_date > ''`, contaminating un-burned evidence with
        # all burned data. Refuse to proceed.
        if not CUTOFF_RE.match(cutoff):
            raise SystemExit(
                f"refusing to run: {CUTOFF_FILE} contains {cutoff!r}, "
                f"not a YYYY/MM/DD date. Fix the file or re-init with "
                f"`python3 -m scripts.validate_hk --set-cutoff YYYY/MM/DD`."
            )
        return cutoff
    row = conn.execute("SELECT MAX(race_date) FROM prediction_log WHERE results_filled=1").fetchone()
    cutoff = row[0] or "2026/05/20"
    _atomic_write_text(CUTOFF_FILE, cutoff)
    print(f"[init] froze forward-validation cutoff at {cutoff} "
          f"(meetings AFTER this date are the clean sample)")
    return cutoff


def _meeting_dates(conn, after: str | None, only_burned_through: str | None = None):
    q = "SELECT DISTINCT race_date FROM prediction_log WHERE results_filled=1"
    params = []
    if after:
        q += " AND race_date > ?"; params.append(after)
    if only_burned_through:
        q += " AND race_date <= ?"; params.append(only_burned_through)
    q += " ORDER BY race_date"
    return [r[0] for r in conn.execute(q, params).fetchall()]


def _score_meeting(conn, date) -> dict:
    """Return {strat: [stake, ret, hits]} + race count for one meeting."""
    rows = conn.execute(
        "SELECT race_number, horse_number, horse_name, predicted_rank, actual_position "
        "FROM prediction_log WHERE race_date=? AND results_filled=1", (date,)).fetchall()
    races = defaultdict(list)
    for r in rows:
        races[r["race_number"]].append(dict(r))
    acc = {s[0]: [0.0, 0.0, 0] for s in STRATS}
    n_races = 0
    for rno, horses in races.items():
        ranked = sorted([h for h in horses if h["predicted_rank"]], key=lambda h: h["predicted_rank"])
        if len(ranked) < 6:
            continue
        our = {n: set(h["horse_name"] for h in ranked[:n]) for n in (2, 3, 4, 5, 6)}
        finishers = {pos: [h for h in horses if h["actual_position"] == pos] for pos in (1, 2, 3, 4)}
        if not all(len(finishers[p]) == 1 for p in (1, 2)):
            continue  # need a clean top-2 to price anything
        n_races += 1
        actual = {p: {finishers[p][0]["horse_name"]} for p in (1, 2)}
        a3 = set().union(*[ {finishers[p][0]["horse_name"]} for p in (1,2,3) if finishers[p] ]) \
            if all(len(finishers[p]) == 1 for p in (1,2,3)) else None
        a4 = set().union(*[ {finishers[p][0]["horse_name"]} for p in (1,2,3,4) if finishers[p] ]) \
            if all(len(finishers[p]) == 1 for p in (1,2,3,4)) else None
        top2_actual = actual[1] | actual[2]
        top2_nums = sorted(int(finishers[p][0]["horse_number"]) for p in (1, 2))

        for label, leg, box, ncombo, pool in STRATS:
            if leg == 2:
                hit = (our[box] >= top2_actual) if box > 2 else (our[2] == top2_actual)
            elif leg == 3:
                hit = a3 is not None and a3 <= our[box]
            else:
                hit = a4 is not None and a4 <= our[box]
            # Fetch dividend only on hit — saves SQL round-trips and avoids
            # needing top-3/top-4 finishers in the no-hit branch.
            div = None
            if hit:
                if pool == "QUINELLA":
                    div = _quin_div(conn, date, rno, *top2_nums)
                elif pool == "TRIO":
                    combo = ",".join(str(n) for n in sorted(
                        int(finishers[p][0]["horse_number"]) for p in (1, 2, 3)))
                    div = _exotic_div(conn, date, rno, pool, combo)
                elif pool == "FIRST 4":
                    combo = ",".join(str(n) for n in sorted(
                        int(finishers[p][0]["horse_number"]) for p in (1, 2, 3, 4)))
                    div = _exotic_div(conn, date, rno, pool, combo)
            if hit and div is None:
                continue  # can't price a win — skip (rare; ~99.6% coverage)
            acc[label][0] += ncombo
            acc[label][1] += div if hit else 0.0
            acc[label][2] += 1 if hit else 0
    acc["_races"] = n_races
    return acc


def _read_csv() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as f:
        return list(csv.DictReader(f))


def _atomic_write_text(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _append_csv(date, acc):
    # Atomic read-modify-rewrite (NOT append): a torn append would corrupt the
    # only un-burned evidence file, and a non-atomic write races backups/concurrent
    # runs. Rebuild the full table and os.replace it in.
    rows = _read_csv()
    newrow = {"race_date": date, "races": acc["_races"]}
    for s in STRATS:
        newrow[f"{s[0]}_stake"] = acc[s[0]][0]
        newrow[f"{s[0]}_ret"] = round(acc[s[0]][1], 2)
        newrow[f"{s[0]}_hits"] = acc[s[0]][2]
    rows.append(newrow)
    tmp = LOG_FILE.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    tmp.replace(LOG_FILE)


def _report(logged: list[dict], title: str):
    print(f"\n{'='*68}\n  {title}\n{'='*68}")
    if not logged:
        print("  (no meetings logged yet — run after the next HK meeting completes)")
        return
    dates = sorted(r["race_date"] for r in logged)
    n_m = len(dates)
    print(f"  meetings: {n_m}  ({dates[0]} .. {dates[-1]})")
    gate = "" if n_m >= 6 else "   [WARN] N<6 — DIRECTIONAL ONLY (not a ship/kill signal yet)"
    print(f"  {'strategy':<14}{'hits':>6}{'stake':>9}{'P&L':>11}{'ROI':>9}{gate}")
    print("  " + "-" * 50)
    for label, *_ in [(s[0],) for s in STRATS]:
        stake = sum(float(r[f"{label}_stake"]) for r in logged)
        ret = sum(float(r[f"{label}_ret"]) for r in logged)
        hits = sum(int(r[f"{label}_hits"]) for r in logged)
        if stake <= 0:
            continue
        roi = 100 * (ret - stake) / stake
        print(f"  {label:<14}{hits:>6}{stake:>9.0f}{ret-stake:>+11.1f}{roi:>+9.1f}%")
    print("\n  NOTE: real HKJC dividends, per-$1 basis. This is the un-burned forward")
    print("  sample — never used for tuning. Trust it over the burned backtest.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", action="store_true", help="cumulative report only")
    p.add_argument("--include-burned", action="store_true",
                   help="also show pre-cutoff meetings (labelled, NOT written to CSV)")
    p.add_argument("--set-cutoff", metavar="YYYY/MM/DD", help="set the frozen cutoff date")
    args = p.parse_args()

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    if args.set_cutoff:
        new_cutoff = args.set_cutoff.strip()
        if not CUTOFF_RE.match(new_cutoff):
            raise SystemExit(
                f"--set-cutoff expects YYYY/MM/DD, got {new_cutoff!r}"
            )
        _atomic_write_text(CUTOFF_FILE, new_cutoff)
        print(f"cutoff set to {new_cutoff}")
    cutoff = _load_cutoff(conn)
    print(f"forward cutoff: {cutoff}  (clean sample = meetings AFTER this)")

    logged = _read_csv()
    already = {r["race_date"] for r in logged}

    if not args.report:
        new_dates = [d for d in _meeting_dates(conn, after=cutoff) if d not in already]
        for d in new_dates:
            acc = _score_meeting(conn, d)
            _append_csv(d, acc)
            print(f"  logged {d}: {acc['_races']} races")
        if not new_dates:
            print("  no new clean meetings to log.")
        logged = _read_csv()

    _report(logged, "HK FORWARD VALIDATION (clean, un-burned)")

    if args.include_burned:
        burned = []
        for d in _meeting_dates(conn, after=None, only_burned_through=cutoff):
            acc = _score_meeting(conn, d)
            row = {"race_date": d}
            for s in STRATS:
                row[f"{s[0]}_stake"], row[f"{s[0]}_ret"], row[f"{s[0]}_hits"] = acc[s[0]]
            burned.append(row)
        _report(burned, "PROVISIONAL — pre-cutoff meetings (BURNED window, directional only)")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
