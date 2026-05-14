"""Top-N ranking evaluation + BOX Q4 gate sweep.

Two questions this answers:

  #1  How good is the model at RANKING — do the actual top finishers land
      in the model's top-4 / top-5 picks? (the bet objective, not Brier)

  #2  Where should the BOX Q4 conviction gate (min_top4) sit — what's the
      hit-rate vs volume vs ROI tradeoff as we move the threshold?

Both run off ONE pass over the held-out races: we capture per-race the gate
inputs + hit booleans + dividend, then replay the gate cascade in-memory at
each candidate threshold. No retrain, no model change.

Usage:
  python3 -m scripts.eval_topn                 # full held-out test fold
  python3 -m scripts.eval_topn --last 740      # ~last 6 meetings
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from core.recommender import _rank_has_strong_history, _choose_race_bet_type
from scripts.backtest import (
    load_held_out_races, group_by_race, predict_race_production,
    get_real_quinella_dividend, estimate_quinella_dividend,
)


def collect(races, conn):
    """One pass: per-race gate inputs + hit booleans + dividend."""
    rows = []
    for (date, race_no), horses in sorted(races.items()):
        if not any((h.get('win_odds') or 0) > 0 for h in horses):
            continue
        preds = predict_race_production(horses, conn)
        if len(preds) < 2:
            continue

        analyses = [p[3] for p in preds]          # ranked rank-1 first
        n = len(analyses)
        probs = [a.combined_probability or 0.0 for a in analyses]

        top_prob = probs[0]
        p2 = probs[1] if n >= 2 else 0.0
        p3 = probs[2] if n >= 3 else 0.0
        p4 = probs[3] if n >= 4 else 0.0
        p5 = probs[4] if n >= 5 else 0.0

        gap_1_2 = top_prob - p2
        gap_2_3 = p2 - p3
        top_2_share = top_prob + p2
        top_3_share = top_2_share + p3
        top_4_share = top_3_share + p4
        top_5_share = top_4_share + p5

        top_is_banker = any("BANKER" in r for r in analyses[0].reasoning)
        top_odds = preds[0][0].get('win_odds') or 0
        market_prob_top = (1.0 / top_odds) if top_odds > 0 else 0.0
        rank3_strong = n >= 3 and _rank_has_strong_history(analyses[2])
        rank4_strong = n >= 4 and _rank_has_strong_history(analyses[3])

        # Model's top-N picks (by name)
        names = [p[0]['horse_name'] for p in preds]
        model_top4 = set(names[:4])
        model_top5 = set(names[:5])

        # Actual finishers
        actual_top2 = {h['horse_name'] for h in horses if h.get('finish_position') in (1, 2)}
        actual_top3 = {h['horse_name'] for h in horses if h.get('finish_position') in (1, 2, 3)}
        actual_top4 = {h['horse_name'] for h in horses if h.get('finish_position') in (1, 2, 3, 4)}

        # Ranking-quality hits (these are ALWAYS-ON dashboard layers — no gate)
        box4_hit = len(actual_top2) == 2 and actual_top2.issubset(model_top4)
        box3_hit = len(actual_top2) == 2 and actual_top2.issubset(set(names[:3]))
        straight_hit = len(actual_top2) == 2 and actual_top2 == set(names[:2])
        trio5_hit = len(actual_top3) == 3 and actual_top3.issubset(model_top5)
        f4_5_hit = len(actual_top4) == 4 and actual_top4.issubset(model_top5)

        # Quinella dividend (real HKJC, else Harville estimate)
        winners = [h for h in horses if h.get('finish_position') in (1, 2)]
        q_div = None
        if len(winners) == 2:
            q_div = get_real_quinella_dividend(
                conn, date, race_no,
                winners[0]['horse_number'], winners[1]['horse_number'])
            if q_div is None:
                q_div = estimate_quinella_dividend(horses, winners[0], winners[1])

        rows.append({
            'race_key': f"{date}-R{race_no}", 'n': n,
            'gap_1_2': gap_1_2, 'gap_2_3': gap_2_3,
            'top_2_share': top_2_share, 'top_3_share': top_3_share,
            'top_4_share': top_4_share, 'top_5_share': top_5_share,
            'top_is_banker': top_is_banker, 'market_prob_top': market_prob_top,
            'rank3_strong': rank3_strong, 'rank4_strong': rank4_strong,
            'box4_hit': box4_hit, 'box3_hit': box3_hit, 'straight_hit': straight_hit,
            'trio5_hit': trio5_hit, 'f4_5_hit': f4_5_hit,
            'q_div': q_div,
        })
    return rows


def pct(num, den):
    return f"{num/den*100:.1f}%" if den else "  — "


def report_ranking_quality(rows):
    """#1: how good is the model at putting actual finishers in top-4 / top-5."""
    n = len(rows)
    print("=" * 70)
    print(f"  #1  MODEL RANKING QUALITY  ({n} races)")
    print("      (these are always-on dashboard layers — no gate involved)")
    print("=" * 70)
    metrics = [
        ("Straight QIN  (actual top-2 == model top-2)", 'straight_hit', 3.0),
        ("BOX-3 QIN     (actual top-2 in model top-3)", 'box3_hit', 9.1),
        ("BOX-4 QIN     (actual top-2 in model top-4)", 'box4_hit', 18.2),
        ("TRIO-5        (actual top-3 in model top-5)", 'trio5_hit', None),
        ("F4 BOX-5      (actual top-4 in model top-5)", 'f4_5_hit', None),
    ]
    print(f"  {'Metric':<46} {'Hits':>6} {'Rate':>8} {'Random':>9}")
    print("  " + "-" * 66)
    for label, key, baseline in metrics:
        hits = sum(r[key] for r in rows)
        base = f"~{baseline:.1f}%" if baseline else "   —"
        print(f"  {label:<46} {hits:>6} {pct(hits, n):>8} {base:>9}")
    print()


def report_share_buckets(rows):
    """#2a: bucket races by top_4_share — does higher conviction = higher hit?"""
    print("=" * 70)
    print("  #2a  BOX-4 HIT RATE vs top_4_share BUCKET")
    print("       (if hit rate climbs with the bucket, a higher gate helps)")
    print("=" * 70)
    buckets = [(0.0, 0.32), (0.32, 0.36), (0.36, 0.40),
               (0.40, 0.45), (0.45, 0.55), (0.55, 1.01)]
    print(f"  {'top_4_share':<16} {'Races':>7} {'Box-4 hits':>12} {'Hit rate':>10}")
    print("  " + "-" * 48)
    for lo, hi in buckets:
        sub = [r for r in rows if lo <= r['top_4_share'] < hi]
        hits = sum(r['box4_hit'] for r in sub)
        print(f"  [{lo:.2f}, {hi:.2f})     {len(sub):>7} {hits:>12} {pct(hits, len(sub)):>10}")
    print()


def report_gate_sweep(rows):
    """#2b: replay the gate cascade at each candidate min_top4 threshold."""
    print("=" * 70)
    print("  #2b  BOX Q4 GATE SWEEP  (min_top4 threshold)")
    print("       current production value = 0.36")
    print("=" * 70)
    print(f"  {'min_top4':>9} {'Bets':>6} {'BOXQ4':>7} {'Hits':>6} {'HitRate':>9} "
          f"{'Stake':>8} {'P&L':>11} {'ROI':>9}")
    print("  " + "-" * 68)

    for thr in [0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48, 0.52]:
        n_bets = n_boxq4 = n_hits = 0
        stake = pnl = 0.0
        for r in rows:
            bt = _choose_race_bet_type(
                has_live_odds=True, n=r['n'], top_is_banker=r['top_is_banker'],
                market_prob_top=r['market_prob_top'],
                gap_1_2=r['gap_1_2'], gap_2_3=r['gap_2_3'],
                top_2_share=r['top_2_share'], top_3_share=r['top_3_share'],
                top_4_share=r['top_4_share'],
                rank3_strong=r['rank3_strong'], rank4_strong=r['rank4_strong'],
                min_top4=thr,
            )
            if bt in ('SKIP', 'WIN', 'TRIO'):
                continue  # not a quinella-box bet — out of scope for this sweep
            if r['q_div'] is None:
                continue
            n_bets += 1
            if bt == 'BOX Q4':
                cost, hit = 6.0, r['box4_hit']
                n_boxq4 += 1
            elif bt == 'BOX Q':
                cost, hit = 3.0, r['box3_hit']
            else:  # QIN
                cost, hit = 1.0, r['straight_hit']
            stake += cost
            if hit:
                n_hits += 1
                pnl += r['q_div'] - cost
            else:
                pnl -= cost
        roi = pnl / stake * 100 if stake else 0
        marker = "  <- current" if abs(thr - 0.36) < 1e-6 else ""
        print(f"  {thr:>9.2f} {n_bets:>6} {n_boxq4:>7} {n_hits:>6} "
              f"{pct(n_hits, n_bets):>9} {stake:>8.0f} {pnl:>+11.0f} "
              f"{roi:>+8.1f}%{marker}")
    print()
    print("  Note: WIN/TRIO/SKIP races are excluded — this sweep only moves the")
    print("  BOX Q4 / BOX Q / QIN boundary. Higher min_top4 pushes marginal")
    print("  races down to BOX Q (3 combos) or QIN (1 combo), not out entirely.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--last', type=int, default=None,
                    help='Use last N rows (~740 = last 6 meetings)')
    args = ap.parse_args()

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = load_held_out_races(conn, last_n=args.last)
    races = group_by_race(rows)
    print(f"\nEvaluating {len(races)} races "
          f"({'last ' + str(args.last) + ' rows' if args.last else 'full held-out fold'})\n")

    data = collect(races, conn)
    conn.close()

    if not data:
        print("No races with predictions.")
        return 1

    report_ranking_quality(data)
    report_share_buckets(data)
    report_gate_sweep(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
