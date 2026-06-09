"""Kill-or-keep ROI backtest for FR MVP model. Mirror of scripts/backtest_uk.py."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ml.fr_features import FEATURE_COLS, TARGET, build_training_table
from ml.fr_model import FRRacingModel
from scripts.oos_common import resolve_oos_cutoff


def main():
    print("Loading data + model...")
    df = build_training_table()
    df = df.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    model = FRRacingModel.load()
    # Slice the test fold by the cutoff FROZEN at train time, not int(len*0.85)
    # on the live row count — the latter drifts as new data is ingested (audit M1).
    cutoff, cutoff_src = resolve_oos_cutoff(model, df)
    test = df[df["date"] > cutoff].copy()
    if test.empty:
        print(f"  No OOS rows after cutoff {cutoff.date()} [{cutoff_src}] — "
              f"the model was trained through the latest data. Ingest newer races "
              f"or retrain on an earlier cutoff.")
        return 1
    print(f"  Test fold: {len(test):,} rows, "
          f"{test['date'].min().date()} → {test['date'].max().date()}  "
          f"(cutoff {cutoff.date()} [{cutoff_src}])")

    test["model_p"] = model.predict_proba(test[FEATURE_COLS])

    test["model_rank"] = test.groupby("race_id")["model_p"].rank(method="first", ascending=False)
    test["sp_rank_check"] = test.groupby("race_id")["sp_dec"].rank(method="first", ascending=True)
    races = test.groupby("race_id")

    def flat_win_roi(pick_mask, label):
        picks = test[pick_mask].copy()
        picks = picks[picks["sp_dec"].notna() & (picks["sp_dec"] > 0)]
        picks["pnl"] = np.where(picks[TARGET] == 1, picks["sp_dec"] - 1, -1)
        n = len(picks)
        stake = n * 1.0
        pnl = picks["pnl"].sum()
        wins = picks[TARGET].sum()
        roi = (pnl / stake * 100) if stake > 0 else 0
        strike = (wins / n * 100) if n > 0 else 0
        avg_winner_sp = picks.loc[picks[TARGET]==1, "sp_dec"].mean() if wins > 0 else 0
        print(f"  {label:<22}  n={n:>5}  strike={strike:>5.1f}%  "
              f"P&L={pnl:>+8.1f}u  ROI={roi:>+7.2f}%  avg_win_sp={avg_winner_sp:.2f}")
        return roi, strike

    print(f"\n=== Flat WIN ROI on test fold ({races.ngroups} races) ===")
    print(f"  {'Strategy':<22}  {'n':>5}  {'strike':>7}  {'P&L':>8}    {'ROI':>7}  {'avg_sp':>10}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*8}    {'-'*7}  {'-'*10}")

    flat_win_roi(test["model_rank"] == 1, "Model top-1")
    flat_win_roi(test["sp_rank_check"] == 1, "Market fav (top-1)")
    flat_win_roi(test["model_rank"] == 2, "Model rank-2")
    flat_win_roi(test["sp_rank_check"] == 2, "Market rank-2")

    disagree = test[(test["model_rank"] == 1) & (test["sp_rank_check"] > 1)]
    print(f"\n=== Edge probe: model top-1 WHERE model ≠ market fav ===")
    print(f"  Disagreements: {len(disagree)} of {races.ngroups} races "
          f"({len(disagree)/races.ngroups*100:.1f}%)")
    flat_win_roi(test.index.isin(disagree.index), "Model TOP1 vs market")

    return 0


if __name__ == "__main__":
    sys.exit(main())
