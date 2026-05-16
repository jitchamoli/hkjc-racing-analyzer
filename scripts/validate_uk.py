"""Forward-validation: score the saved UK model on a date range.

Runs the existing model (no retrain, no feature changes) against UK races
in the specified window and reports flat-WIN ROI for three strategies:
  - Model top-1 pick
  - Market favourite (lowest SP)
  - Model top-1 WHERE model disagrees with market favourite (the edge probe)

The disagreement strategy is the one we're validating. The MVP backtest
showed +1.00% ROI on n=95 disagreements (test fold). Forward-validation
asks: does that edge hold on races the model has never seen?

Usage:
    # Score last weekend's UK racing
    python3 -m scripts.validate_uk --start 2026-05-16 --end 2026-05-17

    # Score everything since training cutoff (true OOS)
    python3 -m scripts.validate_uk --since-cutoff

    # Score a single date
    python3 -m scripts.validate_uk --date 2026-05-17

Notes:
- Requires races in the window to be present in uk_history.db (ingest first).
- If the window overlaps the model's training window, script warns —
  that's in-sample noise, not forward-validation.
- We use ACTUAL SP for both feature computation and P&L. This means the
  ROI is "what would we have made if we'd bet the model picks at SP."
  Real-time betting would lock in earlier prices (morning odds, BSP),
  which can drift — that gap is a separate question to validate later.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from ml.uk_features import FEATURE_COLS, TARGET, build_training_table
from ml.uk_model import UKRacingModel


def _training_cutoff(df: pd.DataFrame) -> pd.Timestamp:
    """Reproduce the 85% temporal split used by scripts.train_uk and return
    the last date inside the training window (i.e. the highest in-sample date).
    Anything strictly greater is true out-of-sample."""
    df_sorted = df.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    cut = int(len(df_sorted) * 0.85)
    return df_sorted.iloc[cut - 1]["date"]


def _flat_win(picks: pd.DataFrame, label: str) -> dict:
    picks = picks[picks["sp_dec"].notna() & (picks["sp_dec"] > 0)].copy()
    if picks.empty:
        print(f"  {label:<28}  no picks with valid SP")
        return {"n": 0, "roi": 0.0, "strike": 0.0}
    picks["pnl"] = np.where(picks[TARGET] == 1, picks["sp_dec"] - 1, -1)
    n = len(picks)
    wins = int(picks[TARGET].sum())
    pnl = float(picks["pnl"].sum())
    roi = pnl / n * 100
    strike = wins / n * 100
    avg_w = float(picks.loc[picks[TARGET] == 1, "sp_dec"].mean()) if wins > 0 else 0.0
    print(f"  {label:<28}  n={n:>4}  strike={strike:>5.1f}%  "
          f"P&L={pnl:>+8.1f}u  ROI={roi:>+7.2f}%  avg_win_sp={avg_w:.2f}")
    return {"n": n, "wins": wins, "pnl": pnl, "roi": roi, "strike": strike}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--date", help="Single date YYYY-MM-DD")
    grp.add_argument("--range", nargs=2, metavar=("START", "END"),
                     help="Date range YYYY-MM-DD YYYY-MM-DD (inclusive)")
    grp.add_argument("--since-cutoff", action="store_true",
                     help="Score everything strictly after the training cutoff (true OOS)")
    p.add_argument("--start", help="(alt) range start, paired with --end")
    p.add_argument("--end", help="(alt) range end")
    args = p.parse_args()

    print("Loading data + features...")
    df = build_training_table()
    df = df.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    cutoff = _training_cutoff(df)
    print(f"  Total rows: {len(df):,}  "
          f"date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Training cutoff (in-sample max): {cutoff.date()}")

    # Resolve window
    if args.since_cutoff:
        window = df[df["date"] > cutoff].copy()
        label = f"OOS (date > {cutoff.date()})"
    elif args.date:
        d = pd.to_datetime(args.date)
        window = df[df["date"] == d].copy()
        label = args.date
    else:
        if args.range:
            start, end = args.range
        else:
            start, end = args.start, args.end
        if not (start and end):
            print("ERROR: must pass --date, --range S E, or --since-cutoff", file=sys.stderr)
            return 2
        start, end = pd.to_datetime(start), pd.to_datetime(end)
        window = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        label = f"{start.date()} → {end.date()}"

    if window.empty:
        print(f"\nNo races found in window: {label}")
        print("(If you expected races, run `python3 -m scripts.ingest_uk` first.)")
        return 1

    overlap = window["date"] <= cutoff
    if overlap.any():
        n_overlap = int(overlap.sum())
        print(f"\nWARNING: {n_overlap:,}/{len(window):,} rows in window are within the "
              f"training window. ROI on those rows is IN-SAMPLE noise.")

    n_races = window["race_id"].nunique()
    print(f"\nWindow: {label}  |  {n_races:,} races, {len(window):,} runners")

    # Predict
    model = UKRacingModel.load()
    window["model_p"] = model.predict_proba(window[FEATURE_COLS])

    # Per-race ranks
    window["model_rank"] = window.groupby("race_id")["model_p"].rank(
        method="first", ascending=False)
    window["sp_rank_check"] = window.groupby("race_id")["sp_dec"].rank(
        method="first", ascending=True)

    print("\n=== Flat WIN ROI ===")
    print(f"  {'Strategy':<28}  {'n':>4}  {'strike':>7}  {'P&L':>8}    {'ROI':>7}  {'avg_win':>8}")
    print(f"  {'-'*28}  {'-'*4}  {'-'*7}  {'-'*8}    {'-'*7}  {'-'*8}")

    top1 = _flat_win(window[window["model_rank"] == 1], "Model top-1")
    fav = _flat_win(window[window["sp_rank_check"] == 1], "Market fav (top-1)")
    disagree_mask = (window["model_rank"] == 1) & (window["sp_rank_check"] > 1)
    disagree = _flat_win(window[disagree_mask], "Model TOP1 vs market (edge)")

    # Guidance
    n_dis = disagree["n"]
    print(f"\nDisagreements: {n_dis} of {n_races} races "
          f"({n_dis/n_races*100:.1f}%)" if n_races else "")
    if n_dis < 50:
        print(f"  → n<50: too small to draw conclusions. Need ≥200 for a real read.")
    elif n_dis < 200:
        print(f"  → n={n_dis}: directional only, not conclusive.")
    else:
        verdict = "EDGE CONFIRMED" if disagree["roi"] > 0 else "EDGE NOT CONFIRMED"
        print(f"  → n={n_dis}: {verdict} (ROI {disagree['roi']:+.2f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
