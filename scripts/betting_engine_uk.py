"""UK/GB value-betting engine — OOS edge-threshold backtest.

The honest question this answers: does value-betting the GB model (bet WIN when
our de-vigged edge >= threshold) beat the market + overround on out-of-sample
races with REAL SP? Sweeps edge thresholds and staking, reports ROI / strike /
P&L per bucket. This is the gate BEFORE any live recommender — no edge here =
no live betting.

Usage:
    python3 -m scripts.betting_engine_uk                 # OOS (date > frozen cutoff)
    python3 -m scripts.betting_engine_uk --in-sample     # whole DB (optimistic; sanity only)
    python3 -m scripts.betting_engine_uk --kelly         # add fractional-Kelly staking column

Caveats baked into the print-out:
- ROI uses ACTUAL SP for both features and settlement (validate_uk caveat).
  Real bets lock earlier/BSP prices that drift — a separate question.
- The model is win-only MVP; no place/each-way. No validated edge yet.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from betting.uk_edge import add_edge_columns, kelly_fraction, select_value_bets
from ml.uk_features import FEATURE_COLS, TARGET, build_training_table
from ml.uk_model import UKRacingModel
from scripts.oos_common import resolve_oos_cutoff

THRESHOLDS = [0.0, 0.02, 0.05, 0.08, 0.12]


def _report(picks: pd.DataFrame, label: str, kelly: bool) -> dict:
    picks = picks[picks["sp_dec"].notna() & (picks["sp_dec"] > 1.0)].copy()
    n = len(picks)
    if n == 0:
        print(f"  {label:<34}  n=   0   (no qualifying bets)")
        return {"n": 0}
    picks["pnl"] = np.where(picks[TARGET] == 1, picks["sp_dec"] - 1.0, -1.0)
    wins = int(picks[TARGET].sum())
    pnl = float(picks["pnl"].sum())
    roi = pnl / n * 100
    strike = wins / n * 100
    avg_sp = float(picks.loc[picks[TARGET] == 1, "sp_dec"].mean()) if wins else 0.0
    extra = ""
    if kelly:
        picks["kf"] = [kelly_fraction(p, s) for p, s in
                       zip(picks["model_p_norm"], picks["sp_dec"])]
        # Kelly P&L per unit-bankroll: stake=kf, return=kf*(sp-1) on win, -kf on loss.
        # NOTE: Kelly only stakes f*>0 bets, so this is a positive-EV-SELECTED
        # subset — its ROI is a selection effect, not independent validation.
        kret = float(np.where(picks[TARGET] == 1, picks["kf"] * (picks["sp_dec"] - 1.0),
                              -picks["kf"]).sum())
        kstake = float(picks["kf"].sum())
        kroi = (kret / kstake * 100) if kstake > 0 else 0.0
        extra = f"   kelly_ROI={kroi:>+6.1f}% (stake {kstake:.1f}u)"
    print(f"  {label:<34}  n={n:>4}  strike={strike:>5.1f}%  "
          f"P&L={pnl:>+8.1f}u  ROI={roi:>+7.2f}%  avg_win_sp={avg_sp:>5.2f}{extra}")
    return {"n": n, "roi": roi, "strike": strike, "pnl": pnl}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-sample", action="store_true",
                    help="Backtest the whole DB (optimistic — sanity only).")
    ap.add_argument("--kelly", action="store_true", help="Add fractional-Kelly ROI column.")
    args = ap.parse_args()

    print("Loading data + model...")
    df = build_training_table().sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    model = UKRacingModel.load()
    cutoff, src = resolve_oos_cutoff(model, df)
    if args.in_sample:
        test = df; scope = "WHOLE DB (in-sample, optimistic)"
    else:
        test = df[df["date"] > cutoff].copy()
        scope = f"OOS (date > {cutoff.date()} [{src}])"
    if test.empty:
        print(f"  No rows in scope {scope}. Ingest newer races or retrain.")
        return 1

    test["model_p"] = model.predict_proba(test[FEATURE_COLS])
    test = add_edge_columns(test)
    n_races = test["race_id"].nunique()
    print(f"  Scope: {scope}  |  {n_races:,} races, {len(test):,} runners "
          f"({test['date'].min().date()} → {test['date'].max().date()})")

    # Baselines
    print(f"\n=== Baselines (flat 1u WIN) ===")
    test["model_rank"] = test.groupby("race_id")["model_p"].rank(method="first", ascending=False)
    test["sp_rank"] = test.groupby("race_id")["sp_dec"].rank(method="first", ascending=True)
    _report(test[test["sp_rank"] == 1], "Market favourite", args.kelly)
    _report(test[test["model_rank"] == 1], "Model top-1 (every race)", args.kelly)

    # Value-betting threshold sweep
    print(f"\n=== Value betting: bet WIN where edge >= threshold (flat 1u) ===")
    for t in THRESHOLDS:
        picks = select_value_bets(test, min_edge=t, top1_only=False)
        _report(picks, f"edge >= {t*100:>4.1f}pp  (all overlays)", args.kelly)
    print(f"\n=== Value betting, model top-1 only (edge >= threshold) ===")
    for t in THRESHOLDS:
        picks = select_value_bets(test, min_edge=t, top1_only=True)
        _report(picks, f"top-1 & edge >= {t*100:>4.1f}pp", args.kelly)

    print("\nVERDICT GUIDE: a bucket is tradeable only if ROI > 0 on n>=200. "
          "ROI uses settle-at-SP (optimistic vs real early/BSP prices).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
