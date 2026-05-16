"""Quick kill-or-keep ROI backtest for UK MVP model.

For each race in the held-out test fold:
  - Model top-1 pick: bet 1 unit WIN. Profit = sp_dec - 1 if it wins, else -1.
  - Market favourite (lowest SP): same comparison.
  - Random: average over 1000 random picks per race.

Reports ROI %, strike rate, total P&L. If model can't beat market favourite
flat-WIN ROI on N>=200 races, it has no edge over the simplest possible
baseline and we kill the build.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ml.uk_features import FEATURE_COLS, TARGET, build_training_table
from ml.uk_model import UKRacingModel


def main():
    print("Loading data + model...")
    df = build_training_table()
    df = df.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    cut_cal = int(len(df) * 0.85)
    test = df.iloc[cut_cal:].copy()
    print(f"  Test fold: {len(test):,} rows, "
          f"{test['date'].min().date()} → {test['date'].max().date()}")

    model = UKRacingModel.load()
    test["model_p"] = model.predict_proba(test[FEATURE_COLS])

    # Per-race: rank by model prob (top1 = highest model_p), and by SP (top1 = lowest SP)
    test["model_rank"] = test.groupby("race_id")["model_p"].rank(method="first",
                                                                 ascending=False)
    test["sp_rank_check"] = test.groupby("race_id")["sp_dec"].rank(method="first",
                                                                    ascending=True)
    races = test.groupby("race_id")

    def flat_win_roi(pick_mask, label):
        picks = test[pick_mask].copy()
        # Drop picks with no SP (NaN) — can't compute P&L
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

    # Strategy: model top-1 ONLY when it disagrees with market favourite
    disagree = test[(test["model_rank"] == 1) & (test["sp_rank_check"] > 1)]
    print(f"\n=== Edge probe: model top-1 WHERE model ≠ market fav ===")
    print(f"  Disagreements: {len(disagree)} of {races.ngroups} races "
          f"({len(disagree)/races.ngroups*100:.1f}%)")
    flat_win_roi(test.index.isin(disagree.index), "Model TOP1 vs market")

    return 0


if __name__ == "__main__":
    sys.exit(main())
