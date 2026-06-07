"""Decisive trained A/B for the barrier-trial TIME signal (the untested one).

The prior investigation (scripts/ab_trials_trained.py) tested only trial
FINISHING POSITION (1/best_pos) — the weakest, possibly-inverted HK trial signal
(trainers routinely instruct horses not to win in trials). Independent review
(2026-06-07) flagged that the column that actually carries trial alpha — the
CLOCK — was never tested, the first-up gate was loose (60d), and Conghua was
pooled in.

This script fixes all three:
  * SIGNAL = trial TIME vs par. Par = fastest clock in the same-distance heat
    cluster on the same (venue, trial_date). A horse posting a fast ABSOLUTE
    time even when it finished mid-heat is exactly what placing misses.
  * FIRST-UP GATE = >=90d since last actual race (true HK first-up), trial
    within 90d before the race.
  * ST/HV ONLY — Conghua trials (different surface/purpose) excluded.
  * SCORED ONLY ON FIRST-UP-AFFECTED HOLDOUT RACES (not diluted over all races).
  * MULTI-SEED — single-seed capture deltas are variance (cf. the prior script's
    false +14 on seed 42).

Leak-free: trial_date < race_date; feature built from prior trials only. Does
NOT touch production schema/model — fresh heads for the relative A/B.
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402

import config  # noqa: E402
from ml.training import build_training_data_from_db  # noqa: E402

WINDOW_DAYS = 90       # trial must be within 90d before the race
FRESH_GAP_DAYS = 90    # true HK first-up = >=90d since last actual race
CLUSTER_GAP = 2.5      # seconds; heats whose median times differ by more start a new distance cluster
CAP = 6.0              # seconds; trial_speed = max(0, CAP - time_behind_best)
SEEDS = [0, 1, 7, 21, 42]


def _dt(s):
    return datetime.strptime(s.replace("-", "/"), "%Y/%m/%d")


def _to_sec(t):
    """'M.SS.HH' or 'SS.HH' -> seconds, else None."""
    try:
        p = t.strip().split(".")
        if len(p) == 3:
            return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 100.0
        if len(p) == 2:
            return int(p[0]) + int(p[1]) / 100.0
    except Exception:
        return None
    return None


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def build_trial_time_index(conn):
    """horse -> sorted [(trial_date, time_behind_best_seconds)] for ST/HV trials.

    Par = min time within a same-distance heat cluster on the same (venue, date).
    Heats clustered by gap-linkage on their median time.
    """
    # gather rows: (venue, date) -> heat -> [(horse, sec)]
    by_meet = defaultdict(lambda: defaultdict(list))
    for r in conn.execute(
        "SELECT venue, trial_date, heat, horse_name, finish_time "
        "FROM barrier_trials WHERE venue IN ('ST','HV') "
        "AND finish_time IS NOT NULL AND TRIM(finish_time)!=''"
    ):
        sec = _to_sec(r["finish_time"])
        if sec is None or not (45 < sec < 200):
            continue
        by_meet[(r["venue"], r["trial_date"])][r["heat"]].append((r["horse_name"], sec))

    horse_idx = defaultdict(list)
    for (venue, date), heats in by_meet.items():
        # per-heat median, then cluster heats by gap on median time
        hmed = sorted(((np.median([s for _, s in rows]), h) for h, rows in heats.items()))
        clusters = []  # list of (par_time, set_of_heats)
        cur_heats, cur_times = [], []
        prev = None
        for med, h in hmed:
            if prev is not None and (med - prev) > CLUSTER_GAP:
                clusters.append((cur_heats, cur_times))
                cur_heats, cur_times = [], []
            cur_heats.append(h)
            cur_times.extend(s for _, s in heats[h])
            prev = med
        if cur_heats:
            clusters.append((cur_heats, cur_times))
        # par per cluster = fastest clock in the cluster
        heat_to_par = {}
        for chs, ctimes in clusters:
            par = min(ctimes)
            for h in chs:
                heat_to_par[h] = par
        for h, rows in heats.items():
            par = heat_to_par[h]
            for horse, sec in rows:
                horse_idx[horse].append((_dt(date), sec - par))  # time_behind >=0
    for v in horse_idx.values():
        v.sort()
    return horse_idx


def main():
    print("Building training matrix from DB...")
    X, y, sw, y_place = build_training_data_from_db()
    if X is None:
        print("No training data."); return 1
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    y_place = np.asarray(y_place, dtype=int)
    print(f"  X={X.shape}, winners={y.sum()}, placers={y_place.sum()}")

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    keys = conn.execute(
        "SELECT horse_name, race_date, race_number, is_winner, is_top3 "
        "FROM race_results WHERE finish_position > 0 "
        "ORDER BY race_date, race_number"
    ).fetchall()
    # permutation-safe alignment check (not just length)
    assert len(keys) == len(X), f"alignment broke: {len(keys)} keys vs {len(X)} rows"
    kw = np.array([int(k["is_winner"] or 0) for k in keys])
    kp = np.array([int(k["is_top3"] or 0) for k in keys])
    assert np.array_equal(kw, y) and np.array_equal(kp, y_place), \
        "alignment broke: key labels do not match training labels (row order drift)"
    print("  alignment OK (labels match elementwise)")

    # first-up race history
    race_dates = defaultdict(list)
    for r in conn.execute("SELECT horse_name, race_date FROM race_results WHERE finish_position > 0"):
        race_dates[r["horse_name"]].append(_dt(r["race_date"]))
    for v in race_dates.values():
        v.sort()

    trial_idx = build_trial_time_index(conn)
    conn.close()

    import bisect

    def trial_speed(horse, rd):
        """max(0, CAP - time_behind) for a first-up horse w/ a recent ST/HV trial, else 0."""
        th = [tb for (d, tb) in trial_idx.get(horse, []) if d < rd and (rd - d).days <= WINDOW_DAYS]
        if not th:
            return 0.0
        ds = race_dates.get(horse, [])
        i = bisect.bisect_left(ds, rd)
        gap = (rd - ds[i - 1]).days if i > 0 else 9999
        if gap < FRESH_GAP_DAYS:
            return 0.0  # not first-up — signal lives in the first-up subset
        return max(0.0, CAP - min(th))  # best (fastest) trial that window

    dates = [k["race_date"] for k in keys]
    rnos = [k["race_number"] for k in keys]
    ts = np.array([trial_speed(k["horse_name"], _dt(k["race_date"])) for k in keys])
    nz = int((ts > 0).sum())
    print(f"  trial_speed nonzero (first-up + ST/HV trial) on {nz}/{len(ts)} rows ({100*nz/len(ts):.2f}%)\n")

    # temporal split
    uniq = sorted(set(dates))
    cut = uniq[int(0.85 * len(uniq))]
    tr = np.array([d < cut for d in dates])
    ho = ~tr
    print(f"split @ {cut}: train {tr.sum()}, holdout {ho.sum()}")

    Xt = np.column_stack([X, ts])

    # build holdout races, keep only CLEAN (>=5 runners, 1 winner, full podium)
    ho_idx = np.where(ho)[0]
    races = defaultdict(list)
    for i in ho_idx:
        races[(dates[i], rnos[i])].append(i)
    clean, affected = [], []
    for grp in races.values():
        if len(grp) < 5 or sum(int(y[i] == 1) for i in grp) != 1:
            continue
        pod3 = {i for i in grp if y_place[i] == 1}
        if len(pod3) != 3:
            continue
        clean.append((grp, pod3))
        if any(ts[i] > 0 for i in grp):
            affected.append((grp, pod3))
    print(f"clean holdout races: {len(clean)} | FIRST-UP-AFFECTED: {len(affected)}\n")
    if len(affected) < 20:
        print("  WARNING: <20 affected races — result dominated by noise; treat as inconclusive.\n")

    def fit_pred(Xmat, ylab, seed):
        m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              random_state=seed, n_jobs=4)
        m.fit(Xmat[tr], ylab[tr])
        return m.predict_proba(Xmat[ho])[:, 1]

    # map global row index -> position within ho slice (predict_proba is over ho)
    pos = {i: j for j, i in enumerate(ho_idx)}

    def capture(pp, race_set):
        win = t4 = t5 = 0
        for grp, pod3 in race_set:
            ranked = sorted(grp, key=lambda i: -pp[pos[i]])
            win += y[ranked[0]] == 1
            t4 += pod3 <= set(ranked[:4])
            t5 += pod3 <= set(ranked[:5])
        return win, t4, t5

    print("Multi-seed trained A/B, scored ONLY on first-up-affected races")
    print(f"(feature: trial time-vs-par, first-up >={FRESH_GAP_DAYS}d, ST/HV only)\n")
    print(f"{'seed':>5} | {'WIN Δ':>6} {'TOP4 Δ':>7} {'TOP5 Δ':>7} | {'winBrier Δ':>11} {'plcBrier Δ':>11}")
    print("-" * 64)
    dW, d4, d5, dWB, dPB = [], [], [], [], []
    yb_w, yb_p = y[ho], y_place[ho]
    for s in SEEDS:
        pwb = fit_pred(X, y, s)
        pwt = fit_pred(Xt, y, s)
        ppb = fit_pred(X, y_place, s)
        ppt = fit_pred(Xt, y_place, s)
        wb, t4b, t5b = capture(ppb, affected)
        wt, t4t, t5t = capture(ppt, affected)
        wbd = brier(pwt, yb_w) - brier(pwb, yb_w)
        pbd = brier(ppt, yb_p) - brier(ppb, yb_p)
        dW.append(wt - wb); d4.append(t4t - t4b); d5.append(t5t - t5b)
        dWB.append(wbd); dPB.append(pbd)
        print(f"{s:>5} | {wt-wb:>+6} {t4t-t4b:>+7} {t5t-t5b:>+7} | {wbd:>+11.5f} {pbd:>+11.5f}")

    mean = lambda a: sum(a) / len(a)
    print("-" * 64)
    print(f"{'MEAN':>5} | {mean(dW):>+6.1f} {mean(d4):>+7.1f} {mean(d5):>+7.1f} | "
          f"{mean(dWB):>+11.5f} {mean(dPB):>+11.5f}")
    print(f"  per-seed TOP-4 Δ: {d4}\n  per-seed TOP-5 Δ: {d5}\n  per-seed WIN Δ: {dW}")

    t4c = all(x > 0 for x in d4) or all(x < 0 for x in d4)
    t5c = all(x > 0 for x in d5) or all(x < 0 for x in d5)
    print("\nVERDICT (scored on first-up-affected races only):")
    if len(affected) < 20:
        print("  INCONCLUSIVE — too few affected races.")
    elif mean(dPB) < -0.0003 and mean(d5) >= 3 and t5c:
        print("  POSITIVE & STABLE — place Brier improves and TOP-5 capture gains hold "
              "across all seeds on the first-up population. Real candidate → forward log.")
    elif not t4c and not t5c:
        print(f"  NOISE — capture deltas flip sign across seeds "
              f"(TOP-4 {d4}, TOP-5 {d5}); mean place Brier Δ={mean(dPB):+.5f}. "
              f"The trial CLOCK adds nothing the model can use. Kill the trial angle.")
    else:
        print(f"  MARGINAL — mean TOP-5 Δ={mean(d5):+.1f}, place Brier Δ={mean(dPB):+.5f}; "
              f"signs {'stable' if t5c else 'flip'}. Not a clean win; forward-validate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
