#!/usr/bin/env python3
"""ROI/CLV backtest on held-out races — the only metric that matters.

Walks the last 20% of historical races (never seen by the model during training
or validation) and simulates betting at actual historical closing odds. Reports
PnL / ROI / CLV / Brier / calibration curve + quinella/trifecta hit-rates.

Limitations:
  - Backtest uses the ML model + market blend only (no live rule-engine signals
    like tips/opening odds — those aren't stored per historical race).
  - Flat unit stakes — no Kelly sizing.
  - Quinella/trifecta P&L needs dividend lookups we don't have; we report
    HIT RATE only (did top-2 model picks finish top-2?).

Usage:
    python3 -m scripts.backtest                    # Default blend settings
    python3 -m scripts.backtest --last 500         # Last 500 races only
    python3 -m scripts.backtest --sweep            # Sweep blend weights
    python3 -m scripts.backtest --temp 2.0 --model-w 0.8
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.model import HKJCRacingMLModel
from ml.features import FeatureEngineering
from core.scorer import _dynamic_blend_weights


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_held_out_races(conn, last_n=None):
    rows = conn.execute(
        "SELECT * FROM race_results WHERE finish_position > 0 "
        "ORDER BY race_date, race_number"
    ).fetchall()
    rows = [dict(r) for r in rows]
    if last_n is not None:
        return rows[-last_n:]
    cut_val = int(len(rows) * 0.8)
    return rows[cut_val:]


def group_by_race(rows):
    races = defaultdict(list)
    for r in rows:
        key = (r['race_date'], r['race_number'])
        races[key].append(r)
    return races


def build_horse_data(h, field_size):
    return {
        'name': h['horse_name'],
        'jockey': h.get('jockey') or '',
        'trainer': h.get('trainer') or '',
        'draw': h.get('draw') or 0,
        'distance': h.get('distance') or 1200,
        'going': h.get('going') or '',
        'field_size': field_size,
        'class_type': h.get('class_type') or '',
        'race_date': h['race_date'],
        'weight': h.get('weight') or 0,
        'horse_weight': h.get('horse_weight') or 0,
        'days_since_last_run': h.get('days_since_last_run') or 0,
        'last_6': '',
    }


# ---------------------------------------------------------------------------
# Prediction + bet simulation
# ---------------------------------------------------------------------------

def estimate_quinella_dividend(horses, horse_a, horse_b, takeout=0.175):
    """Estimate HK quinella dividend for an unordered pair using Harville's model.

    P(quinella{A,B}) ≈ P(A)*P(B)/(1-P(A)) + P(B)*P(A)/(1-P(B))

    where P(X) is market-implied win probability (odds-based, normalized).
    Dividend per $1 = (1 - takeout) / P(quinella).

    Harville assumes independence — real pace-correlated pairs diverge, but
    this is the standard approximation used in industry.
    """
    odds = {h['horse_name']: (h.get('win_odds') or 0) for h in horses}
    # Normalize market probs (remove overround)
    imp = {n: (1.0 / o) for n, o in odds.items() if o > 0}
    s = sum(imp.values())
    if s <= 0:
        return None
    p = {n: v / s for n, v in imp.items()}

    pa = p.get(horse_a.get('horse_name') if isinstance(horse_a, dict) else horse_a, 0)
    pb = p.get(horse_b.get('horse_name') if isinstance(horse_b, dict) else horse_b, 0)
    if pa <= 0 or pb <= 0 or pa >= 1 or pb >= 1:
        return None

    p_quinella = pa * pb / (1 - pa) + pb * pa / (1 - pb)
    if p_quinella <= 0:
        return None
    return (1.0 - takeout) / p_quinella


def predict_race(horses, model, fe, conn, temperature=None,
                 model_w_override=None):
    """Return list of per-horse (horse, ml_prob, final_prob, market_prob).

    temperature: softmax T (default config.SOFTMAX_TEMPERATURE)
    model_w_override: if set, use fixed model weight instead of dynamic blend
    """
    field_size = len(horses)
    ml_probs = []
    for h in horses:
        horse_data = build_horse_data(h, field_size)
        features = fe.extract_features(horse_data, [], '', conn=conn)
        prob, _ = model.predict(features, field_size=field_size)
        ml_probs.append(prob)
    ml_arr = np.array(ml_probs, dtype=float)

    temp = temperature if temperature is not None else config.SOFTMAX_TEMPERATURE
    sm = np.exp((ml_arr - ml_arr.max()) * temp)
    sm = sm / sm.sum() if sm.sum() > 0 else np.ones_like(sm) / field_size

    odds = np.array([h.get('win_odds') or 0 for h in horses], dtype=float)
    if np.all(odds > 0):
        market = 1.0 / odds
        market = market / market.sum()
        if model_w_override is not None:
            model_w = float(model_w_override)
            market_w = 1.0 - model_w
        else:
            dist = horses[0].get('distance') or 1200
            model_w, market_w = _dynamic_blend_weights(field_size, dist)
        final = model_w * sm + market_w * market
        final = final / final.sum()
    else:
        market = np.zeros_like(ml_arr)
        final = sm

    return [(horses[i], ml_arr[i], final[i], market[i]) for i in range(field_size)]


def simulate_bets(races, model, fe, conn, temperature=None,
                  model_w_override=None):
    """Simulate betting — returns per-race records."""
    records = []
    for (date, race_no), horses in sorted(races.items()):
        if not any((h.get('win_odds') or 0) > 0 for h in horses):
            continue
        preds = predict_race(horses, model, fe, conn,
                             temperature=temperature,
                             model_w_override=model_w_override)

        sorted_by_prob = sorted(range(len(preds)),
                                key=lambda i: preds[i][2], reverse=True)
        top_idx = sorted_by_prob[0]
        top_h, top_ml, top_prob, top_mkt = preds[top_idx]
        top_odds = top_h.get('win_odds') or 0
        if top_odds <= 0:
            continue

        # Top-2 horses for quinella tracking
        rank2_idx = sorted_by_prob[1] if len(sorted_by_prob) >= 2 else None
        rank3_idx = sorted_by_prob[2] if len(sorted_by_prob) >= 3 else None
        rank2_h = preds[rank2_idx][0] if rank2_idx is not None else None
        rank3_h = preds[rank3_idx][0] if rank3_idx is not None else None

        # Actual finishers
        actual_top2 = {h['horse_name'] for h in horses
                       if h.get('finish_position') in (1, 2)}
        actual_top3 = {h['horse_name'] for h in horses
                       if h.get('finish_position') in (1, 2, 3)}

        won = bool(top_h.get('is_winner'))
        ev = top_odds * top_prob - 1
        pnl_flat = (top_odds - 1) if won else -1

        # Market favourite baseline
        mkt_idx = int(np.argmax([p[3] for p in preds]))
        mkt_h = preds[mkt_idx][0]
        mkt_won = bool(mkt_h.get('is_winner'))
        mkt_odds = mkt_h.get('win_odds') or 0
        mkt_pnl = (mkt_odds - 1) if mkt_won else -1

        # Quinella/trifecta hit detection
        our_top2_names = {top_h['horse_name']}
        if rank2_h:
            our_top2_names.add(rank2_h['horse_name'])
        our_top3_names = set(our_top2_names)
        if rank3_h:
            our_top3_names.add(rank3_h['horse_name'])

        # Top 4 for box quinella analysis
        rank4_idx = sorted_by_prob[3] if len(sorted_by_prob) >= 4 else None
        rank4_h = preds[rank4_idx][0] if rank4_idx is not None else None
        our_top4_names = set(our_top3_names)
        if rank4_h:
            our_top4_names.add(rank4_h['horse_name'])

        quinella_hit = our_top2_names == actual_top2
        trifecta_hit = our_top3_names == actual_top3
        # Box-3: both top-2 actual horses are in our top-3 picks (C(3,2) = 3 combos)
        box3_q_hit = actual_top2.issubset(our_top3_names)
        # Box-4: both top-2 actual horses are in our top-4 picks (C(4,2) = 6 combos)
        box4_q_hit = actual_top2.issubset(our_top4_names)
        # Banker leg: our rank-1 finishes top-2
        rank1_in_top2 = top_h['horse_name'] in actual_top2
        # Top-pick in top-3
        rank1_in_top3 = top_h['horse_name'] in actual_top3

        # Simulated quinella dividend (Harville model from market odds)
        # Only valid if both winning horses have live odds
        actual_winners = [h for h in horses if h.get('finish_position') in (1, 2)]
        q_dividend = None
        if len(actual_winners) == 2:
            q_dividend = estimate_quinella_dividend(
                horses, actual_winners[0], actual_winners[1])

        # Simulated P&L for different strategies ($1 per combo)
        # Straight quinella: 1 combo (our top-2). Pays if our top-2 = actual top-2.
        pnl_q_straight = None
        if quinella_hit and q_dividend is not None:
            pnl_q_straight = q_dividend - 1.0
        elif q_dividend is not None:
            pnl_q_straight = -1.0

        # Box-3: 3 combos @ $1 each. Pays dividend (1 winning combo) if actual top-2 ⊂ our top-3.
        pnl_box3 = None
        if box3_q_hit and q_dividend is not None:
            pnl_box3 = q_dividend - 3.0
        elif q_dividend is not None:
            pnl_box3 = -3.0

        # Box-4: 6 combos @ $1 each
        pnl_box4 = None
        if box4_q_hit and q_dividend is not None:
            pnl_box4 = q_dividend - 6.0
        elif q_dividend is not None:
            pnl_box4 = -6.0

        # Short-odds banker proxy: top pick at odds <3.5 with high model conviction
        banker_proxy = (top_odds <= 3.5 and top_prob >= 0.25 and ev > 0)

        records.append({
            'race_key': f"{date}-R{race_no}",
            'horse': top_h['horse_name'],
            'ml_prob': top_ml,
            'final_prob': top_prob,
            'market_prob': top_mkt,
            'odds': top_odds,
            'ev': ev,
            'won': won,
            'pnl': pnl_flat,
            'mkt_fav_pnl': mkt_pnl,
            'mkt_fav_won': mkt_won,
            'same_pick': top_idx == mkt_idx,
            'quinella_hit': quinella_hit,
            'trifecta_hit': trifecta_hit,
            'box3_q_hit': box3_q_hit,
            'box4_q_hit': box4_q_hit,
            'rank1_in_top2': rank1_in_top2,
            'rank1_in_top3': rank1_in_top3,
            'banker_proxy': banker_proxy,
            'top2_names': our_top2_names,
            'top3_names': our_top3_names,
            'q_dividend': q_dividend,
            'pnl_q_straight': pnl_q_straight,
            'pnl_box3': pnl_box3,
            'pnl_box4': pnl_box4,
        })
    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _strategy_summary(name, mask, records):
    bets = [r for i, r in enumerate(records) if mask[i]]
    n = len(bets)
    if n == 0:
        print(f"\n[{name}]  no qualifying bets")
        return
    pnl = np.array([r['pnl'] for r in bets])
    won = np.array([r['won'] for r in bets])
    print(f"\n[{name}]")
    print(f"  Bets:   {n:<5} Wins: {int(won.sum())} ({won.mean()*100:.1f}%)   "
          f"Total PnL: {pnl.sum():+.2f}u   ROI: {pnl.sum()/n*100:+.2f}%")


def print_strategies(records, min_ev):
    n = len(records)
    all_mask = np.ones(n, dtype=bool)
    ev_mask = np.array([r['ev'] >= min_ev for r in records])
    high_ev_mask = np.array([r['ev'] >= 0.15 for r in records])
    disagree_mask = np.array([not r['same_pick'] for r in records])
    banker_mask = np.array([r['banker_proxy'] for r in records])
    underdog_mask = np.array([r['odds'] >= 8 for r in records])

    print("\n" + "=" * 70)
    print(f"  WIN strategies ({n} races total)")
    print("=" * 70)
    _strategy_summary("A: Flat bet model top-pick every race", all_mask, records)
    _strategy_summary(f"B: EV-filtered (EV ≥ {min_ev*100:.0f}%)", ev_mask, records)
    _strategy_summary("C: High-conviction (EV ≥ 15%)", high_ev_mask, records)
    _strategy_summary("D: Model disagrees with market fav", disagree_mask, records)
    _strategy_summary("E: Short-odds banker proxy (odds≤3.5, prob≥25%)",
                      banker_mask, records)
    _strategy_summary("F: Underdog bets (model top-pick @ odds≥8)",
                      underdog_mask, records)

    # Baseline
    mkt_records = [{'pnl': r['mkt_fav_pnl'], 'won': r['mkt_fav_won']} for r in records]
    _strategy_summary("Baseline: Bet market favourite every race",
                      np.ones(n, dtype=bool), mkt_records)


def print_exotic_hits(records):
    n = len(records)
    q = sum(1 for r in records if r['quinella_hit'])
    t = sum(1 for r in records if r['trifecta_hit'])
    box3 = sum(1 for r in records if r['box3_q_hit'])
    box4 = sum(1 for r in records if r['box4_q_hit'])
    r1_top2 = sum(1 for r in records if r['rank1_in_top2'])
    r1_top3 = sum(1 for r in records if r['rank1_in_top3'])

    print("\n" + "=" * 70)
    print(f"  EXOTIC-BET HIT RATES ({n} races, 12-horse avg field)")
    print("=" * 70)
    print(f"  {'Strategy':<44}{'Hits':<10}{'Rate':<10}{'Baseline'}")
    print("  " + "-" * 68)
    print(f"  {'Quinella straight (top-2 model = actual)':<44}"
          f"{q:<10}{q/n*100:5.1f}%{'':<5}~3.0% ({q/n/0.03:.1f}× random)")
    print(f"  {'Box-3 quinella (actual top-2 ⊂ top-3 model)':<44}"
          f"{box3:<10}{box3/n*100:5.1f}%{'':<5}~9.1% ({box3/n/0.091:.1f}× random)")
    print(f"  {'Box-4 quinella (actual top-2 ⊂ top-4 model)':<44}"
          f"{box4:<10}{box4/n*100:5.1f}%{'':<5}~18.2% ({box4/n/0.182:.1f}× random)")
    print(f"  {'Trifecta straight (top-3 model = actual)':<44}"
          f"{t:<10}{t/n*100:5.1f}%{'':<5}~2.7% ({t/n/0.027:.1f}× random)")
    print(f"  {'Rank-1 pick finishes top-2':<44}"
          f"{r1_top2:<10}{r1_top2/n*100:5.1f}%{'':<5}~16.7% ({r1_top2/n/0.167:.1f}× random)")
    print(f"  {'Rank-1 pick finishes top-3':<44}"
          f"{r1_top3:<10}{r1_top3/n*100:5.1f}%{'':<5}~25.0% ({r1_top3/n/0.25:.1f}× random)")

    # SIMULATED P&L using Harville-model quinella dividends (from market odds)
    print("\n" + "-" * 70)
    print("  SIMULATED QUINELLA P&L (Harville dividend estimate from market odds)")
    print("  Note: assumes independence (no pace correlation) + HK 17.5% takeout")
    print("-" * 70)

    with_div = [r for r in records if r.get('q_dividend') is not None]
    n_div = len(with_div)
    if n_div == 0:
        print("  No races with valid dividend estimate.")
        return

    # Dividend stats (on winning pairs only — what we'd collect when hit)
    hit_divs_straight = [r['q_dividend'] for r in with_div if r['quinella_hit']]
    hit_divs_box3 = [r['q_dividend'] for r in with_div if r['box3_q_hit']]

    print(f"\n  Simulated dividend on winning pairs (when we hit):")
    if hit_divs_straight:
        divs = np.array(hit_divs_straight)
        print(f"    Straight Q hits (n={len(divs)}): "
              f"avg=${divs.mean():.2f}, median=${np.median(divs):.2f}, "
              f"min=${divs.min():.2f}, max=${divs.max():.2f}")
    if hit_divs_box3:
        divs = np.array(hit_divs_box3)
        print(f"    Box-3 Q hits (n={len(divs)}): "
              f"avg=${divs.mean():.2f}, median=${np.median(divs):.2f}, "
              f"min=${divs.min():.2f}, max=${divs.max():.2f}")

    for label, pnl_key, cost in [
        ("Straight quinella (1 unit/race)", 'pnl_q_straight', 1),
        ("Box-3 quinella (3 units/race)", 'pnl_box3', 3),
        ("Box-4 quinella (6 units/race)", 'pnl_box4', 6),
    ]:
        pnls = np.array([r[pnl_key] for r in with_div if r.get(pnl_key) is not None])
        if len(pnls) == 0:
            continue
        total_stake = len(pnls) * cost
        total_pnl = pnls.sum()
        roi = total_pnl / total_stake * 100
        print(f"\n  [{label}]")
        print(f"    Bets: {len(pnls)} races  |  Stake: ${total_stake}  "
              f"|  P&L: ${total_pnl:+.2f}")
        print(f"    SIMULATED ROI: {roi:+.2f}%")


def print_calibration(records):
    probs = np.array([r['final_prob'] for r in records])
    won = np.array([r['won'] for r in records], dtype=float)
    brier = float(np.mean((probs - won) ** 2))
    print("\n" + "=" * 70)
    print("  CALIBRATION")
    print("=" * 70)
    print(f"  Brier score (top-pick): {brier:.4f}")
    bins = [0.0, 0.08, 0.12, 0.18, 0.25, 1.0]
    print(f"  {'bin':<14}{'n':<6}{'predicted':<12}{'actual':<10}{'diff'}")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        k = int(mask.sum())
        if k > 0:
            pred = float(probs[mask].mean())
            act = float(won[mask].mean())
            diff = act - pred
            print(f"  [{lo*100:4.1f}-{hi*100:4.1f}%] "
                  f"{k:<6}{pred*100:5.2f}%{'':<6}{act*100:5.2f}%{'':<4}{diff*100:+5.2f}pp")


def print_clv(records):
    probs = np.array([r['final_prob'] for r in records])
    mkt = np.array([r['market_prob'] for r in records])
    won = np.array([r['won'] for r in records], dtype=float)
    mask = (mkt > 0) & (probs > 0)
    if mask.sum() == 0:
        return
    log_ratio = np.log(probs[mask] / mkt[mask])
    print("\n" + "=" * 70)
    print("  CLV / MARKET-AGREEMENT")
    print("=" * 70)
    print(f"  Avg log(model/market):     {log_ratio.mean():+.4f}")
    edge = probs[mask] - mkt[mask]
    if won[mask].std() > 0 and edge.std() > 0:
        corr = np.corrcoef(edge, won[mask])[0, 1]
        print(f"  Corr(edge, actual win):    {corr:+.4f}  "
              f"(positive = our disagreements are predictive)")
    n_disagree = sum(1 for r in records if not r['same_pick'])
    print(f"  Disagreement with market:  {n_disagree}/{len(records)} "
          f"({n_disagree/len(records)*100:.1f}%)")


# ---------------------------------------------------------------------------
# Sweep mode
# ---------------------------------------------------------------------------

def run_sweep(races, model, fe, conn, min_ev):
    """Sweep (temperature, model_weight) combos and report ROI + disagreement."""
    print("\n" + "=" * 70)
    print("  BLEND-WEIGHT SWEEP")
    print("=" * 70)
    print(f"  {'temp':<7}{'model_w':<10}{'disagree':<12}"
          f"{'win%':<8}{'ROI_flat':<11}{'ROI_ev':<10}{'Qhit%'}")
    print("  " + "-" * 66)

    temps = [1.0, 1.5, 2.5, 4.0]
    model_ws = [0.4, 0.6, 0.8, 1.0]
    for temp in temps:
        for mw in model_ws:
            records = simulate_bets(races, model, fe, conn,
                                    temperature=temp, model_w_override=mw)
            n = len(records)
            if n == 0:
                continue
            pnl = np.array([r['pnl'] for r in records])
            won = np.array([r['won'] for r in records])
            disagree = sum(1 for r in records if not r['same_pick'])
            ev_bets = [r for r in records if r['ev'] >= min_ev]
            roi_ev = (sum(r['pnl'] for r in ev_bets) / len(ev_bets) * 100
                      if ev_bets else float('nan'))
            qhit = sum(1 for r in records if r['quinella_hit'])
            print(f"  {temp:<7.1f}{mw:<10.2f}{disagree:<5}/{n:<5}"
                  f"{won.mean()*100:<8.1f}{pnl.sum()/n*100:+7.2f}%   "
                  f"{roi_ev:+7.2f}%" if ev_bets else
                  f"  {temp:<7.1f}{mw:<10.2f}{disagree:<5}/{n:<5}"
                  f"{won.mean()*100:<8.1f}{pnl.sum()/n*100:+7.2f}%   "
                  f"{'n/a':<10}", end='')
            print(f"   {qhit/n*100:5.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--last', type=int, default=None,
                        help='Use last N races (default: true held-out 20%)')
    parser.add_argument('--min-ev', type=float, default=0.05)
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep softmax temperature × model weight')
    parser.add_argument('--temp', type=float, default=None,
                        help='Override softmax temperature')
    parser.add_argument('--model-w', type=float, default=None,
                        help='Override model weight (fixed, bypasses dynamic blend)')
    parser.add_argument('--exclude-date', action='append', default=[],
                        help='Exclude specific race dates (YYYY/MM/DD) — can repeat')
    args = parser.parse_args()

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    rows = load_held_out_races(conn, last_n=args.last)
    if args.exclude_date:
        excluded = set(args.exclude_date)
        rows = [r for r in rows if r['race_date'] not in excluded]
        print(f"Excluded dates: {sorted(excluded)}")
    races = group_by_race(rows)
    print(f"Held-out races: {len(races)}  |  horses: {len(rows)}")

    model = HKJCRacingMLModel()
    fe = FeatureEngineering()
    if not model.is_trained:
        print("ERROR: model not loaded.")
        return 1

    if args.sweep:
        run_sweep(races, model, fe, conn, args.min_ev)
        conn.close()
        return 0

    records = simulate_bets(races, model, fe, conn,
                            temperature=args.temp,
                            model_w_override=args.model_w)
    if not records:
        print("No bets produced.")
        return 1

    print(f"\nConfig: temp={args.temp or config.SOFTMAX_TEMPERATURE}  "
          f"model_w={args.model_w or 'dynamic'}")

    print_strategies(records, args.min_ev)
    print_exotic_hits(records)
    print_clv(records)
    print_calibration(records)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
