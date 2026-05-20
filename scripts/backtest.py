#!/usr/bin/env python3
"""ROI/CLV backtest on held-out races — the only metric that matters.

Walks the last 20% of historical races (never seen by the model during training
or validation) and simulates betting at actual historical closing odds. Reports
PnL / ROI / CLV / Brier / calibration curve + quinella/trifecta hit-rates.

F-03 (2026-04-23): per-race prediction now routes through the same
CombinedRaceAnalyzer + assign_bet_recommendations pipeline that
generate_predictions.py uses on race day. Bet counts are lower than the
pre-fix baseline because the recommender now applies conviction gates
(BOX Q4 / BOX Q / QIN / SKIP) instead of unconditionally boxing every race.

Limitations:
  - Historical DB lacks tips data, gear from scraper, last_6, age, and
    season_stakes for most rows. These fields default to neutral values
    (score=0 excluded from rule engine, or 50 for always-included factors).
    A fallback warning is logged per race when this occurs (see _FALLBACK_FIELDS).
  - Flat unit stakes — no Kelly sizing.
  - Quinella/trifecta dividend needs lookup; we use REAL HKJC payouts from
    race_dividends table, with Harville estimate as fallback.

Usage:
    python3 -m scripts.backtest                    # Default blend settings
    python3 -m scripts.backtest --last 500         # Last 500 races only
    python3 -m scripts.backtest --sweep            # Sweep blend weights (ML-only path)
    python3 -m scripts.backtest --temp 2.0 --model-w 0.8
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from ml.model import HKJCRacingMLModel
from ml.features import FeatureEngineering
from core.scorer import _dynamic_blend_weights

logger = logging.getLogger(__name__)

# Fields that the production path uses but the DB often lacks for historical rows.
# When these are missing, the rule engine excludes them (score=0), which slightly
# deflates rule scores vs live race cards. This is the correct degraded-input behavior.
_FALLBACK_FIELDS = {'last_6', 'rating', 'gear', 'age', 'season_stakes'}


# ---------------------------------------------------------------------------
# Production analyzer singleton (F-03: shared with parity test)
# ---------------------------------------------------------------------------

_analyzer_singleton = None


def _get_analyzer():
    """Return a lazily-constructed CombinedRaceAnalyzer singleton."""
    global _analyzer_singleton
    if _analyzer_singleton is None:
        from core.analyzer import CombinedRaceAnalyzer
        _analyzer_singleton = CombinedRaceAnalyzer()
    return _analyzer_singleton


def build_production_race(horse_rows):
    """Build a production Race object from DB horse rows.

    Uses fields available in race_results. Missing fields (tips, last_6,
    gear, rating, age, season_stakes) fall back to neutral defaults,
    causing the rule engine to exclude those factors. A fallback warning
    is emitted when any FALLBACK_FIELDS value is absent.
    """
    from core.analyzer import Race, Horse

    if not horse_rows:
        return None

    # Race-level fields come from the first row (same for all horses in a race)
    ref = horse_rows[0]
    date = ref.get('race_date', '')
    distance_raw = ref.get('distance') or 1200
    distance = str(distance_raw)
    class_type = ref.get('class_type') or ''
    going = ref.get('going') or ''
    venue = ref.get('venue') or ''
    course = ref.get('course') or ''
    race_number = ref.get('race_number', 0)

    # Detect if any fallback fields are missing across the field
    missing_fields = set()
    for h in horse_rows:
        for f in _FALLBACK_FIELDS:
            if not h.get(f):
                missing_fields.add(f)
    if missing_fields:
        logger.debug(
            "Race %s R%s: using neutral fallbacks for %s",
            date, race_number, sorted(missing_fields)
        )

    horses = []
    for h in horse_rows:
        win_odds = h.get('win_odds') or 0.0

        # Estimate place odds from win odds (same logic as convert_race in analyzer)
        if win_odds > 0:
            if win_odds < 3:
                place_odds = 1.1
            elif win_odds < 5:
                place_odds = 1.3
            elif win_odds < 10:
                place_odds = 1.8
            elif win_odds < 20:
                place_odds = 3.0
            else:
                place_odds = 5.0
        else:
            place_odds = 0.0

        horse = Horse(
            number=h.get('horse_number') or 0,
            name=h.get('horse_name') or '',
            draw=h.get('draw') or 0,
            jockey=h.get('jockey') or '',
            trainer=h.get('trainer') or '',
            weight=str(h.get('weight') or ''),
            rating=float(h.get('rating') or 0.0),
            win_odds=win_odds,
            place_odds=place_odds,
            last_6=h.get('last_6') or '',     # often '' in DB — neutral fallback
            gear=h.get('gear') or '',
            distance=distance,
            days_since_last_run=h.get('days_since_last_run') or 0,
            rating_change=h.get('rating_change') or 0,
            horse_weight=h.get('horse_weight') or 0,
            age=h.get('age') or '',            # often '' in DB — neutral fallback
            # Fields not in DB — neutral defaults
            tips_odds=0.0,
            initial_tips_index=0.0,
            raceday_tips_index=0.0,
            is_top_pick=False,
            running_style='',
            last_400m_avg=0.0,
            cd_wins=0,
            cd_runs=0,
            trial_position=0,
            opening_odds=0.0,
            weight_change=0,
            season_stakes='',
            import_cat='',
            sex='',
        )
        horses.append(horse)

    return Race(
        race_number=race_number,
        venue=venue,
        date=date,
        time='',
        distance=distance,
        class_type=class_type,
        going=going,
        track='',
        course=course,
        horses=horses,
        odds_updated_at='',
    )


def predict_race_production(horse_rows, conn):
    """F-03: run the production prediction pipeline on a historical race.

    Returns list of (horse_row_dict, combined_probability, market_prob, HorseAnalysis)
    sorted by rank (rank-1 first). Uses the same CombinedRaceAnalyzer +
    assign_bet_recommendations path as generate_predictions.py.
    """
    if not any((h.get('win_odds') or 0) > 0 for h in horse_rows):
        return []

    race = build_production_race(horse_rows)
    if not race:
        return []

    analyzer = _get_analyzer()
    analyses = analyzer.analyze_race(race)

    # Build a name -> horse_row lookup for result data
    row_by_name = {h['horse_name']: h for h in horse_rows}

    result = []
    for a in analyses:
        h_row = row_by_name.get(a.horse.name)
        if not h_row:
            continue
        odds = a.horse.win_odds
        market_prob = (1.0 / odds) if odds and odds > 0 else 0.0
        result.append((h_row, a.combined_probability, market_prob, a))

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_held_out_races(conn, last_n=None):
    """Load truly-held-out test set: 85-100% of chronologically-sorted rows.

    Matches the new 4-fold split in ml/model.py.train(): 0-60% train,
    60-70% earlystop_val, 70-85% calib, 85-100% test. Backtest reads
    the test fold; the calib fold is reserved for fitting the blend
    calibrator (load_calibration_window below).
    """
    rows = conn.execute(
        "SELECT * FROM race_results WHERE finish_position > 0 "
        "ORDER BY race_date, race_number"
    ).fetchall()
    rows = [dict(r) for r in rows]
    if last_n is not None:
        return rows[-last_n:]
    cut_calib = int(len(rows) * 0.85)
    return rows[cut_calib:]


def load_calibration_window(conn):
    """Load the 70-85% slice — disjoint from both model train and backtest test.

    The blend calibrator (scripts/calibrate_blend.py) was previously fit on
    the same 80-100% the model treated as test. With the new 4-fold split,
    the calibrator now fits on the dedicated 70-85% calib fold so the
    blended-prob → actual-win mapping is learned on data that the upstream
    isotonic calibrator (in ml/model.py) ALSO fit on, but that backtest
    never sees. Backtest then evaluates on the truly-virgin 85-100% test.
    """
    rows = conn.execute(
        "SELECT * FROM race_results WHERE finish_position > 0 "
        "ORDER BY race_date, race_number"
    ).fetchall()
    rows = [dict(r) for r in rows]
    cut_val = int(len(rows) * 0.70)
    cut_calib = int(len(rows) * 0.85)
    return rows[cut_val:cut_calib]


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
        'race_number': h.get('race_number') or 0,  # cache key (data/feature_cache.py)
        'weight': h.get('weight') or 0,
        'horse_weight': h.get('horse_weight') or 0,
        'days_since_last_run': h.get('days_since_last_run') or 0,
        'last_6': '',
    }


# ---------------------------------------------------------------------------
# Dividend helpers
# ---------------------------------------------------------------------------

def estimate_quinella_dividend(horses, horse_a, horse_b, takeout=0.175):
    """Estimate HK quinella dividend for an unordered pair using Harville's model.

    Used as a fallback when real dividend data is unavailable.
    """
    odds = {h['horse_name']: (h.get('win_odds') or 0) for h in horses}
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


def get_real_quinella_dividend(conn, race_date, race_no, horse_num_a, horse_num_b):
    """Look up actual HKJC quinella dividend from race_dividends table.

    HKJC stores quinella combo as sorted comma-separated horse numbers, e.g. "1,4".
    Returns dividend in dollars, or None if not found.
    """
    lo, hi = sorted([int(horse_num_a), int(horse_num_b)])
    combo = f"{lo},{hi}"
    row = conn.execute(
        "SELECT dividend FROM race_dividends "
        "WHERE race_date = ? AND race_number = ? AND pool = 'QUINELLA' AND combination = ?",
        (race_date, race_no, combo)
    ).fetchone()
    return float(row[0]) if row else None


# ---------------------------------------------------------------------------
# ML-only prediction (kept for --sweep mode only)
# ---------------------------------------------------------------------------

def predict_race_ml_only(horses, model, fe, conn, temperature=None,
                         model_w_override=None):
    """ML-only prediction path — used only by --sweep mode.

    Returns list of per-horse (horse, ml_prob, final_prob, market_prob).
    """
    field_size = len(horses)
    ml_probs = []
    for h in horses:
        horse_data = build_horse_data(h, field_size)
        features = fe.extract_features(horse_data, [], '', conn=conn)
        prob, _, _ = model.predict(features, field_size=field_size)
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


# ---------------------------------------------------------------------------
# Main simulation (F-03: production pipeline)
# ---------------------------------------------------------------------------

def simulate_bets(races, conn):
    """Simulate betting using the production prediction pipeline.

    F-03: routes each race through CombinedRaceAnalyzer + assign_bet_recommendations,
    the same path used by generate_predictions.py on race day.

    Returns per-race records. Bet P&L for quinella strategies tracks both:
      - Unconditional boxes (for comparison with pre-fix baseline)
      - Recommender-gated bets (what the dashboard actually recommends)
    """
    records = []
    n_fallback_races = 0

    for (date, race_no), horses in sorted(races.items()):
        if not any((h.get('win_odds') or 0) > 0 for h in horses):
            continue

        preds = predict_race_production(horses, conn)
        if not preds:
            continue

        # preds is sorted by rank (rank-1 first)
        top_h_row, top_prob, top_mkt, top_analysis = preds[0]
        top_odds = top_h_row.get('win_odds') or 0
        if top_odds <= 0:
            continue

        # Check for fallback fields usage
        ref = horses[0]
        if not any([ref.get('last_6'), ref.get('rating'), ref.get('gear'),
                    ref.get('age'), ref.get('season_stakes')]):
            n_fallback_races += 1

        # Rank 2-4 horse rows
        rank2_h_row = preds[1][0] if len(preds) >= 2 else None
        rank3_h_row = preds[2][0] if len(preds) >= 3 else None
        rank4_h_row = preds[3][0] if len(preds) >= 4 else None

        # Actual finishers
        actual_top2 = {h['horse_name'] for h in horses
                       if h.get('finish_position') in (1, 2)}
        actual_top3 = {h['horse_name'] for h in horses
                       if h.get('finish_position') in (1, 2, 3)}

        won = bool(top_h_row.get('is_winner'))
        ev = top_odds * top_prob - 1
        pnl_flat = (top_odds - 1) if won else -1

        # Market favourite baseline (by market_prob)
        mkt_idx = max(range(len(preds)), key=lambda i: preds[i][2])
        mkt_h_row = preds[mkt_idx][0]
        mkt_won = bool(mkt_h_row.get('is_winner'))
        mkt_odds = mkt_h_row.get('win_odds') or 0
        mkt_pnl = (mkt_odds - 1) if mkt_won else -1

        # Named sets for hit detection
        our_top2_names = {top_h_row['horse_name']}
        if rank2_h_row:
            our_top2_names.add(rank2_h_row['horse_name'])
        our_top3_names = set(our_top2_names)
        if rank3_h_row:
            our_top3_names.add(rank3_h_row['horse_name'])
        our_top4_names = set(our_top3_names)
        if rank4_h_row:
            our_top4_names.add(rank4_h_row['horse_name'])

        quinella_hit = our_top2_names == actual_top2
        trifecta_hit = our_top3_names == actual_top3
        box3_q_hit = actual_top2.issubset(our_top3_names)
        box4_q_hit = actual_top2.issubset(our_top4_names)
        rank1_in_top2 = top_h_row['horse_name'] in actual_top2
        rank1_in_top3 = top_h_row['horse_name'] in actual_top3

        # Quinella dividend: prefer REAL HKJC payout; fall back to Harville estimate
        actual_winners = [h for h in horses if h.get('finish_position') in (1, 2)]
        q_dividend = None
        q_dividend_source = None
        if len(actual_winners) == 2:
            real_div = get_real_quinella_dividend(
                conn, date, race_no,
                actual_winners[0]['horse_number'], actual_winners[1]['horse_number'])
            if real_div is not None:
                q_dividend = real_div
                q_dividend_source = 'real'
            else:
                q_dividend = estimate_quinella_dividend(
                    horses, actual_winners[0], actual_winners[1])
                q_dividend_source = 'harville' if q_dividend else None

        # Unconditional quinella P&L (for historical comparison — all races)
        pnl_q_straight = None
        if quinella_hit and q_dividend is not None:
            pnl_q_straight = q_dividend - 1.0
        elif q_dividend is not None:
            pnl_q_straight = -1.0

        pnl_box3 = None
        if box3_q_hit and q_dividend is not None:
            pnl_box3 = q_dividend - 3.0
        elif q_dividend is not None:
            pnl_box3 = -3.0

        pnl_box4 = None
        if box4_q_hit and q_dividend is not None:
            pnl_box4 = q_dividend - 6.0
        elif q_dividend is not None:
            pnl_box4 = -6.0

        # Recommender-gated quinella P&L (F-03: what the dashboard actually bets)
        # Determine what the recommender chose for this race
        rec_bet_type = _infer_race_bet_type(preds)
        pnl_rec_q = _simulate_recommender_qin(
            rec_bet_type, q_dividend,
            actual_top2, our_top2_names, our_top3_names, our_top4_names,
            box3_q_hit, box4_q_hit, quinella_hit
        )

        # Short-odds banker proxy
        banker_proxy = (top_odds <= 3.5 and top_prob >= 0.25 and ev > 0)
        same_pick = (mkt_h_row['horse_name'] == top_h_row['horse_name'])

        records.append({
            'race_key': f"{date}-R{race_no}",
            'horse': top_h_row['horse_name'],
            # Top-pick conviction tier (HIGH/MED/LOW) — used for the
            # tier-conditional edge-correlation breakdown in print_clv.
            'conviction_tier': getattr(top_analysis, 'combined_confidence', '') or '',
            'final_prob': top_prob,
            'market_prob': top_mkt,
            'odds': top_odds,
            'ev': ev,
            'won': won,
            'pnl': pnl_flat,
            'mkt_fav_pnl': mkt_pnl,
            'mkt_fav_won': mkt_won,
            'same_pick': same_pick,
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
            'q_dividend_source': q_dividend_source,
            'pnl_q_straight': pnl_q_straight,
            'pnl_box3': pnl_box3,
            'pnl_box4': pnl_box4,
            'rec_bet_type': rec_bet_type,
            'pnl_rec_q': pnl_rec_q,
        })

    if n_fallback_races > 0:
        total = len(records)
        pct = n_fallback_races / total * 100 if total else 0
        print(f"\n[F-03 note] {n_fallback_races}/{total} races ({pct:.0f}%) used neutral fallbacks "
              f"for missing DB fields (last_6/rating/gear/age/season_stakes). "
              f"Rule engine excluded these factors; other factors scored normally.")

    return records


def _infer_race_bet_type(preds):
    """Infer the race-level bet type from per-horse bet_recommendations.

    The recommender assigns the same race_bet_type across relevant horses.
    We recover it by scanning the top-4 analyses.
    """
    for _, _, _, analysis in preds[:4]:
        rec = analysis.bet_recommendation
        if rec == 'BANKER WIN':
            return 'WIN'
        if rec == 'WIN':
            return 'WIN'
        if rec == 'TRIO':
            return 'TRIO'
    # Check if top-4 are all QIN — indicates BOX Q4
    qin_count = sum(1 for _, _, _, a in preds[:4] if a.bet_recommendation == 'QIN')
    skip_count = sum(1 for _, _, _, a in preds[4:] if a.bet_recommendation == 'SKIP')
    if qin_count == 4 and (len(preds) <= 4 or skip_count > 0):
        return 'BOX Q4'
    if qin_count == 3:
        return 'BOX Q'
    if qin_count == 2:
        return 'QIN'
    # Any non-SKIP in top picks → check further
    top_rec = preds[0][3].bet_recommendation if preds else 'SKIP'
    if top_rec == 'SKIP':
        return 'SKIP'
    return 'SKIP'


def _simulate_recommender_qin(rec_bet_type, q_dividend,
                               actual_top2, our_top2, our_top3, our_top4,
                               box3_hit, box4_hit, straight_hit):
    """Simulate quinella P&L for the recommender-gated bet.

    Returns (pnl, cost) or None if race is skipped.
    Cost is the stake per combo × number of combos.
    """
    if q_dividend is None or rec_bet_type == 'SKIP':
        return None
    if rec_bet_type == 'BOX Q4':
        cost = 6.0
        hit = box4_hit
    elif rec_bet_type == 'BOX Q':
        cost = 3.0
        hit = box3_hit
    elif rec_bet_type == 'QIN':
        cost = 1.0
        hit = straight_hit
    else:
        return None
    return (q_dividend - cost) if hit else (-cost)


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

    # Multiple-comparisons reminder. With 6 strategies (A-F) compared
    # against the baseline at α=0.05, you expect ~0.3 false discoveries
    # by chance. The strategies above are NOT pre-registered — treat
    # them as exploratory and discount apparent edges accordingly.
    print(f"\n  NB: 6 strategies A-F are exploratory (not pre-registered). With α=0.05")
    print(f"      Bonferroni-corrected threshold is α=0.0083 — adjust apparent significance.")


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

    # QUINELLA P&L — unconditional (all races)
    print("\n" + "-" * 70)
    n_real = sum(1 for r in records if r.get('q_dividend_source') == 'real')
    n_harv = sum(1 for r in records if r.get('q_dividend_source') == 'harville')
    if n_real > 0:
        print(f"  QUINELLA P&L — UNCONDITIONAL ({n_real}/{n_real+n_harv} races use REAL HKJC dividends)")
    else:
        print("  QUINELLA P&L — UNCONDITIONAL (Harville estimate from market odds, HK 17.5% takeout)")
    print("-" * 70)

    with_div = [r for r in records if r.get('q_dividend') is not None]
    n_div = len(with_div)
    if n_div == 0:
        print("  No races with valid dividend estimate.")
    else:
        hit_divs_straight = [r['q_dividend'] for r in with_div if r['quinella_hit']]
        hit_divs_box3 = [r['q_dividend'] for r in with_div if r['box3_q_hit']]

        print(f"\n  Dividend on winning pairs (when we hit):")
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

    # RECOMMENDER-GATED quinella P&L (F-03: production conviction gates applied)
    print("\n" + "-" * 70)
    print("  QUINELLA P&L — RECOMMENDER-GATED (conviction gates applied)")
    print("  (BOX Q4 / BOX Q / QIN per race, SKIP when no conviction)")
    print("-" * 70)

    bet_type_counts = defaultdict(int)
    for r in records:
        bet_type_counts[r.get('rec_bet_type', 'SKIP')] += 1
    print(f"\n  Bet-type distribution:")
    for bt, cnt in sorted(bet_type_counts.items(), key=lambda x: -x[1]):
        print(f"    {bt:<12}: {cnt} races ({cnt/n*100:.0f}%)")

    gated = [r for r in records if r.get('pnl_rec_q') is not None]
    n_gated = len(gated)
    if n_gated == 0:
        print("\n  No recommender-gated quinella bets placed (all SKIP).")
    else:
        # Stake varies by bet type
        total_stake = sum(
            6.0 if r['rec_bet_type'] == 'BOX Q4'
            else 3.0 if r['rec_bet_type'] == 'BOX Q'
            else 1.0
            for r in gated
        )
        total_pnl = sum(r['pnl_rec_q'] for r in gated)
        roi = total_pnl / total_stake * 100
        hits = sum(1 for r in gated if r['pnl_rec_q'] > 0)
        print(f"\n  [{n_gated} races bet]")
        print(f"    Bets: {n_gated}  |  Stake: ${total_stake:.0f}  "
              f"|  P&L: ${total_pnl:+.2f}  |  Hits: {hits} ({hits/n_gated*100:.1f}%)")
        print(f"    SIMULATED ROI: {roi:+.2f}%")


def print_calibration(records):
    """Calibration report — Brier, BSS, ECE, and per-bin diff.

    Brier is a raw score; useful for tracking but uninterpretable in
    isolation. BSS = 1 − Brier/Brier_climatology says how much better
    than 'always predict base rate' we are (positive = better, 0 = same,
    negative = worse). ECE = Σ |pred − actual| × bin_weight collapses
    the per-bin diffs into one number a stats person actually tracks.
    """
    probs = np.array([r['final_prob'] for r in records])
    won = np.array([r['won'] for r in records], dtype=float)
    n_total = len(won)
    brier = float(np.mean((probs - won) ** 2))
    p_bar = float(won.mean())
    brier_climatology = p_bar * (1 - p_bar)
    bss = 1.0 - brier / brier_climatology if brier_climatology > 0 else float('nan')

    print("\n" + "=" * 70)
    print("  CALIBRATION")
    print("=" * 70)
    print(f"  Brier score (top-pick): {brier:.4f}   base rate p̄={p_bar:.4f}")
    print(f"  Brier Skill Score:      {bss:+.4f}   (>0 beats 'always predict p̄'; ~+0.05 typical)")

    bins = [0.0, 0.08, 0.12, 0.18, 0.25, 1.0]
    MIN_BIN_N = 20  # bins below this are too thin to make reliable claims about
    print(f"\n  {'bin':<14}{'n':<6}{'predicted':<12}{'actual':<10}{'diff':<10}{'note'}")
    ece_num = 0.0
    ece_denom = 0
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        k = int(mask.sum())
        if k == 0:
            continue
        pred = float(probs[mask].mean())
        act = float(won[mask].mean())
        diff = act - pred
        note = ""
        if k >= MIN_BIN_N:
            # Only thick-enough bins contribute to ECE — thin bins are noise.
            ece_num += abs(diff) * k
            ece_denom += k
        else:
            note = f"[thin: n<{MIN_BIN_N}, treat % as noisy]"
        print(f"  [{lo*100:4.1f}-{hi*100:4.1f}%] "
              f"{k:<6}{pred*100:5.2f}%{'':<6}{act*100:5.2f}%{'':<4}{diff*100:+5.2f}pp   {note}")
    ece = ece_num / ece_denom if ece_denom > 0 else float('nan')
    print(f"\n  Expected Calibration Error (ECE, thick bins only): {ece:.4f}")
    print(f"    (single-number calibration metric — lower is better, 0 = perfectly calibrated)")


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

    # Tier-conditional breakdown. Answers "is the negative correlation
    # structural OR concentrated in low-conviction races?" Stats people
    # always want this when the headline correlation looks bad.
    print(f"\n  ── by conviction tier (top-pick) ──")
    print(f"  {'tier':<10}{'n':<7}{'corr(edge, win)':<20}{'mean edge':<14}{'win rate'}")
    print(f"  {'-'*10}{'-'*7}{'-'*20}{'-'*14}{'-'*10}")
    tiers = sorted({r.get('conviction_tier', '') or 'NONE' for r in records})
    for tier in tiers:
        t_idx = np.array([(r.get('conviction_tier') or 'NONE') == tier for r in records])
        m_t = mask & t_idx
        k = int(m_t.sum())
        if k < 10:
            print(f"  {tier:<10}{k:<7}{'(n<10, skip)':<20}")
            continue
        edge_t = probs[m_t] - mkt[m_t]
        won_t = won[m_t]
        if edge_t.std() > 0 and won_t.std() > 0:
            corr_t = np.corrcoef(edge_t, won_t)[0, 1]
        else:
            corr_t = float('nan')
        print(f"  {tier:<10}{k:<7}{corr_t:+.4f}{'':<13}{edge_t.mean()*100:+.2f}pp{'':<5}{won_t.mean()*100:.1f}%")


# ---------------------------------------------------------------------------
# Sweep mode (ML-only path — for blend weight research)
# ---------------------------------------------------------------------------

def run_sweep(races, model, fe, conn, min_ev):
    """Sweep (temperature, model_weight) combos and report ROI + disagreement.

    NOTE: sweep uses the ML-only prediction path (not the production pipeline)
    because varying temperature/model_w is not supported by CombinedRaceAnalyzer.
    This is intentional — sweep is a research tool, not a production simulation.
    """
    print("\n" + "=" * 70)
    print("  BLEND-WEIGHT SWEEP (ML-only path, not production pipeline)")
    print("=" * 70)
    print(f"  {'temp':<7}{'model_w':<10}{'disagree':<12}"
          f"{'win%':<8}{'ROI_flat':<11}{'ROI_ev':<10}{'Qhit%'}")
    print("  " + "-" * 66)

    temps = [1.0, 1.5, 2.5, 4.0]
    model_ws = [0.4, 0.6, 0.8, 1.0]
    for temp in temps:
        for mw in model_ws:
            # ML-only simulation for sweep
            sweep_records = []
            for (date, race_no), horses in sorted(races.items()):
                if not any((h.get('win_odds') or 0) > 0 for h in horses):
                    continue
                preds = predict_race_ml_only(horses, model, fe, conn,
                                             temperature=temp, model_w_override=mw)
                sorted_by_prob = sorted(range(len(preds)),
                                        key=lambda i: preds[i][2], reverse=True)
                top_idx = sorted_by_prob[0]
                top_h, top_ml, top_prob, top_mkt = preds[top_idx]
                top_odds = top_h.get('win_odds') or 0
                if top_odds <= 0:
                    continue
                mkt_idx = int(np.argmax([p[3] for p in preds]))
                mkt_h = preds[mkt_idx][0]
                won = bool(top_h.get('is_winner'))
                ev = top_odds * top_prob - 1
                pnl = (top_odds - 1) if won else -1
                same_pick = (mkt_h['horse_name'] == top_h['horse_name'])
                actual_top2 = {h['horse_name'] for h in horses
                               if h.get('finish_position') in (1, 2)}
                top2_names = {top_h['horse_name']}
                if len(sorted_by_prob) >= 2:
                    top2_names.add(preds[sorted_by_prob[1]][0]['horse_name'])
                qhit = top2_names == actual_top2
                sweep_records.append({
                    'pnl': pnl, 'won': won, 'ev': ev,
                    'same_pick': same_pick, 'quinella_hit': qhit
                })

            n = len(sweep_records)
            if n == 0:
                continue
            pnl_arr = np.array([r['pnl'] for r in sweep_records])
            won_arr = np.array([r['won'] for r in sweep_records])
            disagree = sum(1 for r in sweep_records if not r['same_pick'])
            ev_bets = [r for r in sweep_records if r['ev'] >= min_ev]
            roi_ev = (sum(r['pnl'] for r in ev_bets) / len(ev_bets) * 100
                      if ev_bets else float('nan'))
            qhit = sum(1 for r in sweep_records if r['quinella_hit'])
            print(f"  {temp:<7.1f}{mw:<10.2f}{disagree:<5}/{n:<5}"
                  f"{won_arr.mean()*100:<8.1f}{pnl_arr.sum()/n*100:+7.2f}%   "
                  f"{roi_ev:+7.2f}%" if ev_bets else
                  f"  {temp:<7.1f}{mw:<10.2f}{disagree:<5}/{n:<5}"
                  f"{won_arr.mean()*100:<8.1f}{pnl_arr.sum()/n*100:+7.2f}%   "
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
                        help='Sweep softmax temperature × model weight (ML-only path)')
    parser.add_argument('--temp', type=float, default=None,
                        help='(sweep only) Override softmax temperature')
    parser.add_argument('--model-w', type=float, default=None,
                        help='(sweep only) Override model weight')
    parser.add_argument('--exclude-date', action='append', default=[],
                        help='Exclude specific race dates (YYYY/MM/DD) — can repeat')
    parser.add_argument('--pnl-json', default=None,
                        help='Path to write per-race P&L JSON (for bootstrap CI)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable on-disk feature cache (forces fresh extraction)')
    args = parser.parse_args()

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    from data.database import apply_perf_pragmas
    apply_perf_pragmas(conn)

    # On-disk feature cache — saves ~5-7s of extract_features time on
    # repeated runs against the same DB. Cache key encodes NUM_FEATURES
    # and total row count so schema/data changes auto-invalidate.
    if not args.no_cache:
        from data import feature_cache as _fc
        _fc.enable()
        import atexit
        atexit.register(_fc.save)

    rows = load_held_out_races(conn, last_n=args.last)
    if args.exclude_date:
        excluded = set(args.exclude_date)
        rows = [r for r in rows if r['race_date'] not in excluded]
        print(f"Excluded dates: {sorted(excluded)}")
    races = group_by_race(rows)
    print(f"Held-out races: {len(races)}  |  horses: {len(rows)}")
    print(f"Pipeline: production (CombinedRaceAnalyzer + assign_bet_recommendations)")

    # IID check: how many backtest horses also appear in the calibrator
    # window (or earlier training data)? Each individual PREDICTION is
    # leakage-free (extract_features filters race_date < ?), but the test
    # set is NOT IID — the same horse can appear in both windows, so
    # bootstrap CIs that assume independent samples will under-state
    # uncertainty. Use the per-race P&L JSON + block-bootstrap by horse
    # if you need an honest CI.
    test_horses = {r['horse_name'] for r in rows}
    pre_window_count = conn.execute(
        f"SELECT COUNT(DISTINCT horse_name) FROM race_results "
        f"WHERE finish_position > 0 AND horse_name IN ({','.join(['?']*len(test_horses))}) "
        f"AND race_date < ?",
        list(test_horses) + [min(r['race_date'] for r in rows)],
    ).fetchone()[0]
    overlap_pct = (100 * pre_window_count / len(test_horses)) if test_horses else 0
    print(f"IID check: {pre_window_count}/{len(test_horses)} test horses ({overlap_pct:.0f}%) "
          f"also appear in pre-backtest data — test set is NOT IID by horse.")

    if args.sweep:
        model = HKJCRacingMLModel()
        fe = FeatureEngineering()
        if not model.is_trained:
            print("ERROR: model not loaded.")
            conn.close()
            return 1
        run_sweep(races, model, fe, conn, args.min_ev)
        conn.close()
        return 0

    records = simulate_bets(races, conn)
    if not records:
        print("No bets produced.")
        conn.close()
        return 1

    print(f"\nConfig: production pipeline (temp={config.SOFTMAX_TEMPERATURE}, "
          f"dynamic market blend)")

    print_strategies(records, args.min_ev)
    print_exotic_hits(records)
    print_clv(records)
    print_calibration(records)

    # Emit per-race P&L JSON for bootstrap CI computation
    if args.pnl_json:
        pnl_out_path = Path(args.pnl_json)
        pnl_out_path.parent.mkdir(parents=True, exist_ok=True)
        pnl_rows = []
        # Recommender-gated stake/return re-derived from the per-race bet type +
        # hit flags so the PRODUCTION strategy also gets a bootstrap CI.
        _gated_cost = {'BOX Q4': 6.0, 'BOX Q': 3.0, 'QIN': 1.0}
        _gated_hit_field = {'BOX Q4': 'box4_q_hit', 'BOX Q': 'box3_q_hit', 'QIN': 'quinella_hit'}
        for r in records:
            with_div = r.get('q_dividend') is not None
            rec = r.get('rec_bet_type')
            g_stake = _gated_cost.get(rec, 0.0) if (with_div and rec not in (None, 'SKIP')) else 0.0
            g_hit = bool(r.get(_gated_hit_field.get(rec, ''), False))
            g_return = (r['q_dividend'] if g_hit else 0.0) if g_stake > 0 else 0.0
            pnl_rows.append({
                'race_key': r['race_key'],
                'straightq_stake': 1.0 if with_div else 0.0,
                'straightq_return': (r['q_dividend'] if r.get('quinella_hit') else 0.0) if with_div else 0.0,
                'box3_stake': 3.0 if with_div else 0.0,
                'box3_return': (r['q_dividend'] if r.get('box3_q_hit') else 0.0) if with_div else 0.0,
                'box4_stake': 6.0 if with_div else 0.0,
                'box4_return': (r['q_dividend'] if r.get('box4_q_hit') else 0.0) if with_div else 0.0,
                'gated_stake': g_stake,
                'gated_return': g_return,
            })
        with open(pnl_out_path, 'w') as f:
            json.dump(pnl_rows, f, indent=2)
        print(f"\nPer-race P&L written to: {pnl_out_path}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
