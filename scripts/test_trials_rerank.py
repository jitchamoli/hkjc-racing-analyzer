"""Signal-additivity (re-rank) test for the barrier-trial candidate.

Mirrors test_sire_going_rerank.py. Re-ranks production predictions by blending
calibrated_prob with a first-up trial-quality nudge, sweeps blend weight lambda,
reports top-N capture. Does NOT modify the model.

The diagnostic put the signal in the FIRST-UP subset (model blind there), so the
nudge is applied ONLY to first-up horses (>=60d since last race) that have a
recent trial (<=90d). All other horses get a neutral 0 nudge.

Caveats: BURNED data (model trained on/near these races); sparse signal (only
first-up-trial horses are affected), so few races change — read the
"races affected" count. A negative here is strong evidence a retrain won't help;
a positive is necessary-not-sufficient (forward log is the only honest gate).
"""
from __future__ import annotations

import bisect
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

WINDOW_DAYS = 90
FRESH_GAP_DAYS = 60
N_MEETINGS = 30


def _dt(s):
    return datetime.strptime(s.replace("-", "/"), "%Y/%m/%d")


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    # trials by horse + race history (for first-up) ------------------------
    trials = defaultdict(list)
    for r in conn.execute("SELECT horse_name, trial_date, finish_position "
                          "FROM barrier_trials WHERE finish_position >= 1"):
        trials[r["horse_name"]].append((_dt(r["trial_date"]), r["finish_position"]))
    for v in trials.values():
        v.sort()
    race_dates = defaultdict(list)
    for r in conn.execute("SELECT horse_name, race_date FROM race_results "
                          "WHERE finish_position > 0"):
        race_dates[r["horse_name"]].append(_dt(r["race_date"]))
    for v in race_dates.values():
        v.sort()

    def trial_quality(horse, before_date):
        """1/best_trial_pos for a first-up horse with a recent trial, else None."""
        rd = _dt(before_date)
        th = [p for (d, p) in trials.get(horse, []) if d < rd and (rd - d).days <= WINDOW_DAYS]
        if not th:
            return None
        ds = race_dates.get(horse, [])
        i = bisect.bisect_left(ds, rd)
        gap = (rd - ds[i - 1]).days if i > 0 else 9999
        if gap < FRESH_GAP_DAYS:
            return None  # not first-up — signal lives in the first-up subset
        return 1.0 / min(th)

    meetings = [r[0] for r in conn.execute(
        "SELECT DISTINCT race_date FROM prediction_log WHERE results_filled=1 "
        "ORDER BY race_date DESC LIMIT ?", (N_MEETINGS,)).fetchall()]
    print(f"Testing on {len(meetings)} meetings: {meetings[-1]} .. {meetings[0]}\n")

    races = []
    for date in meetings:
        rno_horses = defaultdict(list)
        for r in conn.execute("""
            SELECT race_number, horse_name, calibrated_prob, actual_position
            FROM prediction_log
            WHERE race_date=? AND results_filled=1 AND calibrated_prob IS NOT NULL
        """, (date,)).fetchall():
            rno_horses[r["race_number"]].append({
                "horse_name": r["horse_name"],
                "calibrated_prob": r["calibrated_prob"],
                "tq": trial_quality(r["horse_name"], date),
                "actual_position": r["actual_position"],
            })
        for rno, horses in rno_horses.items():
            if len(horses) < 5:
                continue
            top3 = {h["horse_name"] for h in horses if h["actual_position"] in (1, 2, 3)}
            if len(top3) != 3:
                continue
            races.append({"horses": horses, "top3": top3,
                          "winner": next((h["horse_name"] for h in horses
                                          if h["actual_position"] == 1), None)})

    affected = sum(1 for r in races if any(h["tq"] is not None for h in r["horses"]))
    n_fu_trial = sum(1 for r in races for h in r["horses"] if h["tq"] is not None)
    print(f"{len(races)} clean-podium races | races with a first-up trial horse: "
          f"{affected} | first-up trial horses: {n_fu_trial}")
    if affected < 10:
        print("  WARNING: very few affected races — result will be dominated by noise.\n")

    # z-score the nudge within each race (neutral 0 where no signal)
    for race in races:
        vals = [h["tq"] for h in race["horses"] if h["tq"] is not None]
        mu = (sum(vals) / len(vals)) if vals else 0.0
        sd = (statistics.pstdev(vals) if len(vals) >= 2 else 1.0) or 1.0
        for h in race["horses"]:
            h["tz"] = ((h["tq"] - mu) / sd) if h["tq"] is not None else 0.0

    scale = sum(h["calibrated_prob"] for r in races for h in r["horses"]) \
        / max(1, sum(len(r["horses"]) for r in races))

    def measure(lm):
        win = qin = t4 = t5 = 0
        for race in races:
            for h in race["horses"]:
                h["_s"] = h["calibrated_prob"] + lm * h["tz"] * scale
            rk = sorted(race["horses"], key=lambda h: -h["_s"])
            top2 = {h["horse_name"] for h in rk[:2]}
            top4 = {h["horse_name"] for h in rk[:4]}
            top5 = {h["horse_name"] for h in rk[:5]}
            a2 = {next((h["horse_name"] for h in race["horses"]
                        if h["actual_position"] == p), None) for p in (1, 2)} - {None}
            win += rk[0]["horse_name"] == race["winner"]
            qin += bool(a2) and a2 <= top2
            t4 += race["top3"] <= top4
            t5 += race["top3"] <= top5
        return win, qin, t4, t5

    n = len(races)
    print("\n" + "=" * 70)
    print(f"RE-RANK — {n} races, score = cal_prob + λ × trial_z × scale")
    print("=" * 70)
    print(f"{'λ':>6} {'WIN':>12} {'QIN':>12} {'TOP-4':>12} {'TOP-5':>12}")
    base = None
    for lm in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        w, q, t4, t5 = measure(lm)
        if base is None:
            base = (w, q, t4, t5)
            tag = " (baseline)"
        else:
            tag = ""
        d = lambda x, b: " " if x == b else ("+" if x > b else "-")
        print(f"{lm:>6.2f} {w:>3}/{n} {d(w,base[0])}  {q:>3}/{n} {d(q,base[1])}  "
              f"{t4:>3}/{n} {d(t4,base[2])}  {t5:>3}/{n} {d(t5,base[3])}{tag}")
    print("\n  HONEST READ: burned data; the only honest gate is forward log.")
    print("  Apply small-sample discount: deltas <3 hits are noise.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
