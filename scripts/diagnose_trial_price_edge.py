#!/usr/bin/env python3
"""Beats-the-price test for the barrier-trial signal.

The cheap diagnostic showed first-up trial winners PLACE at 35.6% vs 16.9%.
But trials are public, so the market likely prices them. This asks the decisive
question: backing first-up trial-winners at their ACTUAL odds/dividends, is the
ROI positive (edge) or ~ -takeout (already priced)?

WIN: stake 1u, return = win_odds if won else 0  (win_odds = SP/closing).
PLACE: stake 1u, return = per-$1 place dividend if it placed-and-paid else 0.
Leak-free trial lookup (trial_date < race_date, 90d). First-up = >=60d gap.
"""
import bisect
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from betting.dividends import hkjc_declared_to_unit

WINDOW_DAYS = 90
FRESH_GAP_DAYS = 60
TAKEOUT = 0.175  # ~HK pool takeout, for reference


def _dt(s):
    return datetime.strptime(s.replace("-", "/"), "%Y/%m/%d")


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    trials = {}
    for r in conn.execute("SELECT horse_name, trial_date, finish_position "
                          "FROM barrier_trials WHERE finish_position >= 1"):
        trials.setdefault(r["horse_name"], []).append((_dt(r["trial_date"]), r["finish_position"]))
    for v in trials.values():
        v.sort()

    # per-$1 place dividend keyed by (date, race_no, horse_number)
    place_div = {}
    for r in conn.execute("SELECT race_date, race_number, combination, dividend "
                          "FROM race_dividends WHERE pool='PLACE'"):
        place_div[(r["race_date"], r["race_number"], str(r["combination"]).strip())] = \
            hkjc_declared_to_unit(r["dividend"])

    races = conn.execute(
        "SELECT race_date, race_number, horse_name, horse_number, finish_position, win_odds "
        "FROM race_results WHERE finish_position > 0 ORDER BY race_date").fetchall()
    conn.close()

    race_dates = {}
    for r in races:
        race_dates.setdefault(r["horse_name"], []).append(_dt(r["race_date"]))
    for v in race_dates.values():
        v.sort()

    groups = {}  # name -> list of (won, placed, win_odds, place_ret, implied)
    for r in races:
        odds = r["win_odds"] or 0
        if odds <= 0:
            continue
        h, rd = r["horse_name"], _dt(r["race_date"])
        fp = r["finish_position"]
        th = [(d, p) for (d, p) in trials.get(h, []) if d < rd and (rd - d).days <= WINDOW_DAYS]
        if not th:
            continue
        ds = race_dates.get(h, [])
        i = bisect.bisect_left(ds, rd)
        gap = (rd - ds[i - 1]).days if i > 0 else 9999
        first_up = gap >= FRESH_GAP_DAYS
        trial_won = any(p == 1 for _, p in th)

        won = 1 if fp == 1 else 0
        placed = 1 if fp <= 3 else 0
        pret = place_div.get((r["race_date"], r["race_number"], str(r["horse_number"]).strip()), 0.0)
        rec = (won, placed, odds, pret, 1.0 / odds)

        groups.setdefault("all_with_trial", []).append(rec)
        if trial_won:
            groups.setdefault("trial_WINNER", []).append(rec)
        else:
            groups.setdefault("trial_non_winner", []).append(rec)
        if first_up:
            groups.setdefault("firstup_all", []).append(rec)
            groups.setdefault("firstup_trial_WINNER" if trial_won
                              else "firstup_trial_non_winner", []).append(rec)

    def report(name):
        g = groups.get(name, [])
        if len(g) < 30:
            print(f"  {name:28} n={len(g):<5} (too small)")
            return
        n = len(g)
        win_rate = sum(x[0] for x in g) / n
        place_rate = sum(x[1] for x in g) / n
        win_roi = sum((o if w else 0) for w, p, o, pr, im in g) / n - 1
        # place ROI only over horses where a place dividend was declared in their
        # race (fields >=7 pay 3, 4-6 pay 2; no row => place pool didn't pay)
        place_roi = sum(pr if pl else 0 for w, pl, o, pr, im in g) / n - 1
        implied = sum(x[4] for x in g) / n  # mean 1/odds (raw, ~+overround)
        print(f"  {name:28} n={n:<5} win {win_rate*100:4.1f}% (impl~{implied*100:4.1f}%)  "
              f"place {place_rate*100:4.1f}%  | WIN ROI {win_roi*100:+6.1f}%  PLACE ROI {place_roi*100:+6.1f}%")

    print(f"Backing at actual odds/dividends. Takeout ~{TAKEOUT*100:.0f}% "
          f"(efficient-market ROI floor ~ -{TAKEOUT*100:.0f}%).\n")
    print("GROUP                          N      WIN% (implied)   PLACE%   ROI (win / place)")
    for name in ["all_with_trial", "trial_WINNER", "trial_non_winner",
                 "firstup_all", "firstup_trial_WINNER", "firstup_trial_non_winner"]:
        report(name)

    print("\nVERDICT:")
    g = groups.get("firstup_trial_WINNER", [])
    if len(g) >= 30:
        n = len(g)
        win_roi = sum((o if w else 0) for w, p, o, pr, im in g) / n - 1
        place_roi = sum(pr if pl else 0 for w, pl, o, pr, im in g) / n - 1
        best = max(win_roi, place_roi)
        if best > 0:
            print(f"  POSITIVE — first-up trial-winners beat the price "
                  f"(best ROI {best*100:+.1f}% > 0). Real edge candidate → proceed.")
        elif best > -TAKEOUT:
            print(f"  PRICED-BUT-CLOSE — best ROI {best*100:+.1f}% beats the -{TAKEOUT*100:.0f}% "
                  f"takeout floor but isn't profitable. Marginal; needs forward CLV.")
        else:
            print(f"  PRICED OUT — best ROI {best*100:+.1f}% <= -takeout. Market already "
                  f"prices the trial signal; high place rate but no value. KILL for betting.")
    else:
        print("  INCONCLUSIVE — first-up trial-winner sample too small.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
