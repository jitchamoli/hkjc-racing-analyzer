#!/usr/bin/env python3
"""Cheap diagnostic: does recent barrier-trial form carry race-outcome signal?

Leak-free — for each race entry we only look at trials with trial_date <
race_date (within a 90-day window). The orthogonality-relevant test is the
FIRST-UP subset (horse returning from >=60 days off / debut): there the model's
recent-race-form features are blank, so any trial signal is genuinely new.

Gates (mirrors the project's cheap-diagnostic convention on is_top3):
    |r(signal, is_top3)| < 0.02 -> KILL (no signal / market already prices it)
    |r(signal, is_top3)| > 0.03 -> PROCEED to re-rank
    in between                  -> MARGINAL
"""
import bisect
import os
import sqlite3
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

WINDOW_DAYS = 90      # trial counts as "recent" if within this many days
FRESH_GAP_DAYS = 60   # horse is "first-up/fresh" if last race was >= this ago


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def _dt(s):
    return datetime.strptime(s.replace("-", "/"), "%Y/%m/%d")


def main():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    # trials by horse -> sorted [(date, finish_position)]
    trials = {}
    for r in conn.execute("SELECT horse_name, trial_date, finish_position "
                          "FROM barrier_trials WHERE finish_position >= 1"):
        trials.setdefault(r["horse_name"], []).append((_dt(r["trial_date"]), r["finish_position"]))
    for v in trials.values():
        v.sort()

    races = conn.execute(
        "SELECT race_date, horse_name, finish_position FROM race_results "
        "WHERE finish_position > 0 ORDER BY race_date").fetchall()
    conn.close()

    race_dates = {}
    for r in races:
        race_dates.setdefault(r["horse_name"], []).append(_dt(r["race_date"]))
    for v in race_dates.values():
        v.sort()

    data = []  # dict per race entry
    for r in races:
        h, rd = r["horse_name"], _dt(r["race_date"])
        fp = r["finish_position"]
        th = [(d, p) for (d, p) in trials.get(h, []) if d < rd and (rd - d).days <= WINDOW_DAYS]
        ds = race_dates.get(h, [])
        i = bisect.bisect_left(ds, rd)
        gap = (rd - ds[i - 1]).days if i > 0 else 9999  # 9999 = debut
        data.append({
            "is_top3": 1 if fp <= 3 else 0,
            "is_winner": 1 if fp == 1 else 0,
            "has_trial": 1 if th else 0,
            "trial_won": 1 if any(p == 1 for _, p in th) else 0,
            "best_pos": min((p for _, p in th), default=None),
            "first_up": 1 if gap >= FRESH_GAP_DAYS else 0,
        })

    n = len(data)
    n_trial = sum(d["has_trial"] for d in data)
    n_fu = sum(d["first_up"] for d in data)
    n_fu_trial = sum(1 for d in data if d["first_up"] and d["has_trial"])
    print(f"race entries: {n:,} | with recent trial: {n_trial:,} ({100*n_trial/n:.1f}%)")
    print(f"first-up entries: {n_fu:,} | first-up WITH recent trial: {n_fu_trial:,}")
    print(f"base rates: is_top3={sum(d['is_top3'] for d in data)/n:.3f}  "
          f"is_winner={sum(d['is_winner'] for d in data)/n:.3f}\n")

    def corrs(subset, label):
        if len(subset) < 50:
            print(f"  [{label}] n={len(subset)} too small")
            return None
        t3 = [d["is_top3"] for d in subset]
        wn = [d["is_winner"] for d in subset]
        won = [d["trial_won"] for d in subset]
        pos = [d["best_pos"] for d in subset]
        r_won_t3 = pearson(won, t3)
        r_pos_t3 = pearson(pos, t3)   # expect NEGATIVE (lower pos = better)
        r_won_wn = pearson(won, wn)
        base_t3 = sum(t3) / len(t3)
        print(f"  [{label}] n={len(subset)}  base is_top3={base_t3:.3f}")
        print(f"     r(trial_won,  is_top3)   = {r_won_t3:+.4f}")
        print(f"     r(best_pos,   is_top3)   = {r_pos_t3:+.4f}  (neg = better trial -> places)")
        print(f"     r(trial_won,  is_winner) = {r_won_wn:+.4f}")
        # placed rate split by trial winner vs not
        tw = [d for d in subset if d["trial_won"]]
        ntw = [d for d in subset if not d["trial_won"]]
        if tw and ntw:
            print(f"     placed%: trial-winner {100*sum(d['is_top3'] for d in tw)/len(tw):.1f}% "
                  f"vs non-winner {100*sum(d['is_top3'] for d in ntw)/len(ntw):.1f}%")
        return r_won_t3

    print("=== ALL entries with a recent trial ===")
    r_all = corrs([d for d in data if d["has_trial"]], "all+trial")
    print("\n=== FIRST-UP entries with a recent trial (orthogonality test) ===")
    r_fu = corrs([d for d in data if d["has_trial"] and d["first_up"]], "firstup+trial")

    print("\n=== VERDICT (Gate B on first-up subset, the orthogonal case) ===")
    key = r_fu if r_fu is not None else r_all
    if key is None:
        print("  INCONCLUSIVE — sample too small")
    elif abs(key) < 0.02:
        print(f"  KILL — |r|={abs(key):.4f} < 0.02: no usable trial signal "
              f"(market already prices visible trials)")
    elif abs(key) > 0.03:
        print(f"  PROCEED — |r|={abs(key):.4f} > 0.03: trial form carries signal; "
              f"move to re-rank / trained A/B")
    else:
        print(f"  MARGINAL — |r|={abs(key):.4f} in [0.02,0.03]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
