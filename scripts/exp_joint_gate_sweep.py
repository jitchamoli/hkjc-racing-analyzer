"""EXPERIMENT: joint 3D gate sweep — (min_top2, min_top3, min_top4).

Today the gate cascade has three independent thresholds:
    MIN_TOP2_CONVICTION = 0.25  (current — only swept implicitly)
    MIN_TOP3_CONVICTION = 0.30  (current — never swept)
    MIN_TOP4_CONVICTION = 0.52  (current — swept in 1D by eval_topn)

The 4 prior model experiments all hit the Brier/hit-rate-improves-but-
ROI-degrades wall because we tune model first then gates. This script
attacks the OTHER half of that coupling: jointly optimize all three
gate thresholds at once with the CURRENT model.

If the 3D maximum beats current — ship.
If it ties — current was already optimal; tells us the gate coupling
is binding even more tightly than we thought.

Cheap replay (no retrain, no model change) — uses the same per-race
gate-input capture as scripts/eval_topn.py.
"""
import argparse
import itertools
import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from core.recommender import _choose_race_bet_type
from scripts.backtest import load_held_out_races, group_by_race
from scripts.eval_topn import collect

logging.basicConfig(level=logging.WARNING, format="%(message)s")


# Current production values — included as a baseline anchor in the grid.
CUR_TOP2 = 0.25
CUR_TOP3 = 0.30
CUR_TOP4 = 0.52


def evaluate_combo(rows, min_top2, min_top3, min_top4):
    """Replay the gate cascade for one (min_top2, min_top3, min_top4) combo.

    Returns dict: hits, stake, pnl, roi, bets, distribution-by-bet-type.
    """
    n_bets = 0
    n_hits = 0
    stake = 0.0
    pnl = 0.0
    by_type = {"QIN": 0, "BOX Q": 0, "BOX Q4": 0, "TRIO": 0, "WIN": 0, "SKIP": 0}

    for r in rows:
        bt = _choose_race_bet_type(
            has_live_odds=True, n=r["n"], top_is_banker=r["top_is_banker"],
            market_prob_top=r["market_prob_top"],
            gap_1_2=r["gap_1_2"], gap_2_3=r["gap_2_3"],
            top_2_share=r["top_2_share"], top_3_share=r["top_3_share"],
            top_4_share=r["top_4_share"],
            rank3_strong=r["rank3_strong"], rank4_strong=r["rank4_strong"],
            min_top2=min_top2, min_top3=min_top3, min_top4=min_top4,
        )
        by_type[bt] = by_type.get(bt, 0) + 1
        if bt in ("SKIP", "WIN", "TRIO"):
            continue
        if r["q_div"] is None:
            continue
        if bt == "BOX Q4":
            cost, hit = 6.0, r["box4_hit"]
        elif bt == "BOX Q":
            cost, hit = 3.0, r["box3_hit"]
        else:  # QIN
            cost, hit = 1.0, r["straight_hit"]
        stake += cost
        n_bets += 1
        if hit:
            n_hits += 1
            pnl += r["q_div"] - cost
        else:
            pnl -= cost
    roi = pnl / stake * 100 if stake else 0
    return {
        "min_top2": min_top2, "min_top3": min_top3, "min_top4": min_top4,
        "bets": n_bets, "hits": n_hits, "stake": stake, "pnl": pnl, "roi": roi,
        "by_type": dict(by_type),
    }


def grid_search(rows, label):
    """Sweep the 3D grid. Print top combos by ROI + baseline."""
    grid_t2 = [0.20, 0.22, 0.25, 0.28, 0.30]
    grid_t3 = [0.26, 0.28, 0.30, 0.33, 0.36]
    grid_t4 = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58]
    total = len(grid_t2) * len(grid_t3) * len(grid_t4)
    print(f"\n=== {label} — {total} combos × {len(rows)} races ===")

    results = []
    for t2, t3, t4 in itertools.product(grid_t2, grid_t3, grid_t4):
        results.append(evaluate_combo(rows, t2, t3, t4))

    # Baseline = current production thresholds
    baseline = evaluate_combo(rows, CUR_TOP2, CUR_TOP3, CUR_TOP4)

    results.sort(key=lambda r: -r["roi"])
    top10 = results[:10]

    print(f"\n  Baseline (current: top2={CUR_TOP2} top3={CUR_TOP3} top4={CUR_TOP4}):")
    print(f"    bets={baseline['bets']}  hits={baseline['hits']}  "
          f"stake=${baseline['stake']:.0f}  P&L=${baseline['pnl']:+.0f}  "
          f"ROI={baseline['roi']:+.1f}%")

    print(f"\n  Top 10 combos by ROI (sweep over {total} settings):")
    print(f"  {'#':>3} {'min2':>6} {'min3':>6} {'min4':>6}  "
          f"{'bets':>5} {'hits':>5} {'stake':>7} {'P&L':>9} {'ROI':>9}  vs base")
    print("  " + "-" * 84)
    for i, r in enumerate(top10, 1):
        roi_delta = r["roi"] - baseline["roi"]
        marker = ""
        if (abs(r["min_top2"] - CUR_TOP2) < 1e-6
                and abs(r["min_top3"] - CUR_TOP3) < 1e-6
                and abs(r["min_top4"] - CUR_TOP4) < 1e-6):
            marker = "  <- current"
        print(f"  {i:>3} {r['min_top2']:>6.2f} {r['min_top3']:>6.2f} {r['min_top4']:>6.2f}  "
              f"{r['bets']:>5} {r['hits']:>5} {r['stake']:>7.0f} "
              f"{r['pnl']:>+9.0f} {r['roi']:>+8.1f}%  "
              f"{roi_delta:>+6.1f}pp{marker}")

    # Rank of current in the sorted list
    cur_idx = next(
        (i for i, r in enumerate(results, 1)
         if abs(r['min_top2']-CUR_TOP2) < 1e-6 and abs(r['min_top3']-CUR_TOP3) < 1e-6
         and abs(r['min_top4']-CUR_TOP4) < 1e-6), None)
    if cur_idx is not None:
        print(f"\n  Current production gates rank: #{cur_idx} / {total}")

    return baseline, top10[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=None,
                    help="Use last N rows (~740 = last 6 meetings)")
    ap.add_argument("--validate-both", action="store_true",
                    help="After full-holdout sweep, re-evaluate top combo on last-740 (both-windows test)")
    args = ap.parse_args()

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    rows_full = load_held_out_races(conn, last_n=args.last)
    races_full = group_by_race(rows_full)
    print(f"\nCollecting per-race gate inputs from {len(races_full)} races...")
    captured_full = collect(races_full, conn)
    print(f"  captured {len(captured_full)} races with valid data")

    baseline_full, best_full = grid_search(captured_full, "FULL HOLDOUT")

    if args.validate_both and args.last is None:
        # Re-run on last 740 rows (~60 races) — both-windows test
        rows_6 = load_held_out_races(conn, last_n=740)
        races_6 = group_by_race(rows_6)
        captured_6 = collect(races_6, conn)
        print(f"\n=== Last-6-meetings re-evaluation of FULL-WINNER combo ===")
        b = evaluate_combo(captured_6, best_full["min_top2"],
                            best_full["min_top3"], best_full["min_top4"])
        b_base = evaluate_combo(captured_6, CUR_TOP2, CUR_TOP3, CUR_TOP4)
        print(f"  Best combo from full sweep on last-6: "
              f"ROI={b['roi']:+.1f}%  (baseline {b_base['roi']:+.1f}%, "
              f"Δ {b['roi']-b_base['roi']:+.1f}pp)")
        if b["roi"] > b_base["roi"] and best_full["roi"] > baseline_full["roi"]:
            print(f"  ✅ BOTH-WINDOWS POSITIVE — candidate for production.")
        elif b["roi"] > b_base["roi"]:
            print(f"  ⚠ Only last-6 positive; full-holdout neutral or worse.")
        else:
            print(f"  ❌ Last-6 regresses — DO NOT SHIP.")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
