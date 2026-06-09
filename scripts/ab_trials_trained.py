"""Direct trained-model A/B for the barrier-trial feature (ML head only).

Trains win + place XGB heads WITH vs WITHOUT a recent_trial_quality feature on
the same temporal split (train < 85th-pct date, holdout >= 85th pct), compares
Brier (win, place) + top-N capture on the holdout. Does NOT touch production
schema/model — self-contained.

recent_trial_quality = 1/best_trial_finish_position within 90d before the race
(0 if none). NOT first-up-gated: the model has days-since features and can learn
the interaction (the first-up-gated version was already tested via the re-rank,
which was null). Leak-free: only trials with trial_date < race_date.

Caveats: holdout overlaps the production model's training window (this is a
fresh head, so that's fine for the relative A/B), but capture/Brier deltas of a
few hits are noise — apply the small-sample discount.
"""
from __future__ import annotations

import bisect
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

WINDOW_DAYS = 90


def _dt(s):
    return datetime.strptime(s.replace("-", "/"), "%Y/%m/%d")


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


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
    keys = conn.execute("SELECT horse_name, race_date, race_number FROM race_results "
                        "WHERE finish_position > 0 ORDER BY race_date, race_number").fetchall()
    assert len(keys) == len(X), f"alignment broke: {len(keys)} keys vs {len(X)} rows"

    # trial signal, leak-free
    trials = defaultdict(list)
    for r in conn.execute("SELECT horse_name, trial_date, finish_position "
                          "FROM barrier_trials WHERE finish_position >= 1"):
        trials[r["horse_name"]].append((_dt(r["trial_date"]), r["finish_position"]))
    for v in trials.values():
        v.sort()
    conn.close()

    def trial_q(horse, rd):
        th = [p for (d, p) in trials.get(horse, []) if d < rd and (rd - d).days <= WINDOW_DAYS]
        return (1.0 / min(th)) if th else 0.0

    tq = np.array([trial_q(k["horse_name"], _dt(k["race_date"])) for k in keys])
    dates = [k["race_date"] for k in keys]
    rnos = [k["race_number"] for k in keys]
    nz = int((tq > 0).sum())
    print(f"  recent_trial_quality nonzero on {nz} / {len(tq)} rows ({100*nz/len(tq):.1f}%)\n")

    # temporal split: train < 85th-pct date, holdout >= 85th pct
    uniq = sorted(set(dates))
    cut = uniq[int(0.85 * len(uniq))]
    tr = np.array([d < cut for d in dates])
    ho = ~tr
    print(f"split @ {cut}: train {tr.sum()}, holdout {ho.sum()}\n")

    Xt = np.column_stack([X, tq])  # with-trial matrix
    yb_w, yb_p = y[ho], y_place[ho]

    ho_idx = np.where(ho)[0]
    races = defaultdict(list)
    for j, i in enumerate(ho_idx):
        races[(dates[i], rnos[i])].append((j, y[i], y_place[i]))
    # pre-filter to clean races (>=5 runners, exactly 1 winner, full podium)
    clean = []
    for grp in races.values():
        if len(grp) < 5 or sum(1 for _, yy, _ in grp if yy == 1) != 1:
            continue
        pod3 = {j for j, yy, yp in grp if yp == 1}
        if len(pod3) == 3:
            clean.append((grp, pod3))
    n = len(clean)

    def fit_pred(Xmat, ylab, seed):
        m = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              random_state=seed, n_jobs=4)
        m.fit(Xmat[tr], ylab[tr])
        return m.predict_proba(Xmat[ho])[:, 1]

    def capture(pp):
        win = t4 = t5 = 0
        for grp, pod3 in clean:
            ranked = sorted(grp, key=lambda t: -pp[t[0]])
            win += ranked[0][1] == 1
            t4 += pod3 <= {t[0] for t in ranked[:4]}
            t5 += pod3 <= {t[0] for t in ranked[:5]}
        return win, t4, t5

    SEEDS = [0, 1, 7, 21, 42]
    print(f"Multi-seed A/B over {len(SEEDS)} seeds, {n} clean holdout races...\n")
    print(f"{'seed':>5} | {'WIN Δ':>6} {'TOP4 Δ':>7} {'TOP5 Δ':>7} | {'placeBrier Δ':>13}")
    print("-" * 52)
    dW, d4, d5, dPB = [], [], [], []
    for s in SEEDS:
        pwb = fit_pred(X, y, s)
        pwt = fit_pred(Xt, y, s)
        ppb = fit_pred(X, y_place, s)
        ppt = fit_pred(Xt, y_place, s)
        wb, t4b, t5b = capture(ppb)
        wt, t4t, t5t = capture(ppt)
        pbd = brier(ppt, yb_p) - brier(ppb, yb_p)
        dW.append(wt - wb); d4.append(t4t - t4b); d5.append(t5t - t5b); dPB.append(pbd)
        print(f"{s:>5} | {wt-wb:>+6} {t4t-t4b:>+7} {t5t-t5b:>+7} | {pbd:>+13.5f}")

    mean = lambda a: sum(a) / len(a)
    print("-" * 52)
    print(f"{'MEAN':>5} | {mean(dW):>+6.1f} {mean(d4):>+7.1f} {mean(d5):>+7.1f} | {mean(dPB):>+13.5f}")
    print(f"  TOP-4 Δ per seed: {d4}   TOP-5 Δ per seed: {d5}")

    print("\nVERDICT:")
    # Real signal: capture gain consistent in SIGN across seeds AND mean place
    # Brier improves. Seed-flipping signs = noise.
    t4_consistent = all(x > 0 for x in d4) or all(x < 0 for x in d4)
    t5_consistent = all(x > 0 for x in d5) or all(x < 0 for x in d5)
    if mean(dPB) < -0.0003 and mean(d5) >= 3 and t5_consistent:
        print("  POSITIVE & STABLE — place Brier improves and TOP-5 capture gains "
              "hold across all seeds. Real candidate; proceed to forward log.")
    elif (not t4_consistent and not t5_consistent):
        print(f"  NOISE — capture deltas flip sign across seeds "
              f"(TOP-4 {d4}, TOP-5 {d5}); mean place Brier Δ={mean(dPB):+.5f}. "
              f"The single-seed gain was seed variance. Do not ship.")
    else:
        print(f"  MARGINAL — mean TOP-5 Δ={mean(d5):+.1f}, place Brier Δ={mean(dPB):+.5f}; "
              f"signs {'stable' if t5_consistent else 'flip'}. Not a clean win; "
              f"forward-validate before any ship.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
