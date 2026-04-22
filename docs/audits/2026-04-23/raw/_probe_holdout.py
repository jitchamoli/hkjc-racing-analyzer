#!/usr/bin/env python3
"""
Forensic probe: re-run the backtest holdout loop and emit per-race failure flags.

Read-only. Never imports from any production module in a modifying way.
Output: docs/audits/2026-04-23/raw/holdout-failures.json

Usage:
    python3 docs/audits/2026-04-23/raw/_probe_holdout.py
"""

import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on path (probe lives 3 dirs deep)
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from scripts.backtest import (
    load_held_out_races,
    group_by_race,
    build_horse_data,
    predict_race,
    get_real_quinella_dividend,
)
from ml.model import HKJCRacingMLModel
from ml.features import FeatureEngineering


OUTPUT_PATH = Path(__file__).parent / "holdout-failures.json"


def _safe_rank(preds_sorted, horse_name):
    """Return 1-based rank of horse_name in sorted predictions, or None."""
    for i, (h, ml, prob, mkt) in enumerate(preds_sorted):
        if h["horse_name"] == horse_name:
            return i + 1
    return None


def run_probe():
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    rows = load_held_out_races(conn)
    races = group_by_race(rows)
    print(f"Held-out races: {len(races)}  |  horses: {len(rows)}")

    model = HKJCRacingMLModel()
    fe = FeatureEngineering()
    if not model.is_trained:
        print("ERROR: model not loaded — cannot run probe.")
        sys.exit(1)

    output_rows = []

    for (date, race_no) in sorted(races.keys()):
        horses = races[(date, race_no)]

        # Skip races with no odds (same gate as backtest)
        if not any((h.get("win_odds") or 0) > 0 for h in horses):
            continue

        field_size = len(horses)

        # Get predictions sorted by final probability descending
        preds = predict_race(horses, model, fe, conn)
        sorted_idx = sorted(range(len(preds)), key=lambda i: preds[i][2], reverse=True)
        preds_sorted = [preds[i] for i in sorted_idx]  # list of (horse, ml, final, market)

        # Collect ranked picks (up to 4)
        def _pick(rank):
            """0-based rank → (horse_dict, ml_prob, final_prob, market_prob) or Nones."""
            if rank < len(preds_sorted):
                return preds_sorted[rank]
            return None, None, None, None

        r1h, r1ml, r1prob, r1mkt = _pick(0)
        r2h, r2ml, r2prob, r2mkt = _pick(1)
        r3h, r3ml, r3prob, r3mkt = _pick(2)
        r4h, r4ml, r4prob, r4mkt = _pick(3)

        # Actual results
        finish_map = {h["horse_name"]: h.get("finish_position") for h in horses}
        winner = next((h for h in horses if h.get("finish_position") == 1), None)
        runnerup = next((h for h in horses if h.get("finish_position") == 2), None)

        winner_name = winner["horse_name"] if winner else None
        runnerup_name = runnerup["horse_name"] if runnerup else None

        winner_our_rank = _safe_rank(preds_sorted, winner_name) if winner_name else None
        runnerup_our_rank = _safe_rank(preds_sorted, runnerup_name) if runnerup_name else None

        # Quinella dividend (actual HKJC payout, None if not backfilled)
        qin_dividend = None
        if winner and runnerup:
            qin_dividend = get_real_quinella_dividend(
                conn, date, race_no,
                winner["horse_number"], runnerup["horse_number"]
            )

        # Shares
        top_2_share = (r1prob or 0) + (r2prob or 0)
        top_3_share = top_2_share + (r3prob or 0)
        top_4_share = top_3_share + (r4prob or 0)

        # --- Build row ---
        row = {
            "race_date": date,
            "race_no": race_no,
            "field_size": field_size,
            # Rank-1
            "rank1_horse": r1h["horse_name"] if r1h else None,
            "rank1_prob": float(r1prob) if r1prob is not None else None,
            "rank1_finish": finish_map.get(r1h["horse_name"]) if r1h else None,
            "rank1_market_implied_prob": float(r1mkt) if r1mkt is not None else None,
            # Rank-2
            "rank2_horse": r2h["horse_name"] if r2h else None,
            "rank2_prob": float(r2prob) if r2prob is not None else None,
            "rank2_finish": finish_map.get(r2h["horse_name"]) if r2h else None,
            # Rank-3
            "rank3_horse": r3h["horse_name"] if r3h else None,
            "rank3_prob": float(r3prob) if r3prob is not None else None,
            "rank3_finish": finish_map.get(r3h["horse_name"]) if r3h else None,
            # Rank-4
            "rank4_horse": r4h["horse_name"] if r4h else None,
            "rank4_prob": float(r4prob) if r4prob is not None else None,
            "rank4_finish": finish_map.get(r4h["horse_name"]) if r4h else None,
            # Actual top finishers
            "winner_name": winner_name,
            "winner_our_rank": winner_our_rank,
            "runnerup_name": runnerup_name,
            "runnerup_our_rank": runnerup_our_rank,
            # Probability shares
            "top_2_share": float(top_2_share),
            "top_3_share": float(top_3_share),
            "top_4_share": float(top_4_share),
            # Quinella dividend
            "qin_pair_dividend_actual": float(qin_dividend) if qin_dividend is not None else None,
        }

        # --- Failure flags ---
        data_incomplete = (
            row["rank1_finish"] is None
            or row["winner_our_rank"] is None
            or row["runnerup_our_rank"] is None
        )

        if data_incomplete:
            row["rank1_miss"] = False
            row["top2_miss"] = False
            row["winner_deep"] = False
            row["qin_box3_miss"] = False
            row["qin_box4_miss"] = False
        else:
            r1f = row["rank1_finish"]
            r2f = row["rank2_finish"]
            w_rank = row["winner_our_rank"]
            ru_rank = row["runnerup_our_rank"]

            row["rank1_miss"] = r1f >= 4
            row["top2_miss"] = r1f >= 4 and (r2f is None or r2f >= 4)
            row["winner_deep"] = w_rank >= 5
            row["qin_box3_miss"] = not (w_rank <= 3 and ru_rank <= 3)
            row["qin_box4_miss"] = not (w_rank <= 4 and ru_rank <= 4)

        row["data_incomplete"] = data_incomplete

        output_rows.append(row)

    conn.close()
    return output_rows


def main():
    rows = run_probe()
    with open(OUTPUT_PATH, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")
    byte_size = OUTPUT_PATH.stat().st_size
    print(f"File size: {byte_size:,} bytes")


if __name__ == "__main__":
    main()
