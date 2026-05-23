#!/usr/bin/env python3
"""
Generate race predictions and dashboard - non-interactive
"""

import asyncio
import itertools
import re
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fetcher import HKJCDataFetcher
from core.analyzer import CombinedRaceAnalyzer
from dashboard.template import generate_html
from data.database import HorseDatabase
from data.odds_tracker import OddsDatabase, OddsMovementAnalyzer
from analytics.pace import PaceAnalyzer
from betting.exotic_calculator import find_value_exotics

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _atomic_write(path: str, content: str) -> None:
    # Write to a sibling temp file then os.replace so concurrent readers
    # (the dashboard server, a browser polling the file) never see a
    # half-written file. os.replace is atomic on POSIX and Windows ≥ Vista.
    # try/finally cleans up the tmp on write failure so .tmp.{pid} files
    # don't accumulate after disk-full / SIGKILL / etc.
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, 'w') as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_OVERRIDES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tracking_data', 'manual_overrides.json',
)


def _load_manual_overrides() -> dict:
    """Tactical per-race overrides for the rank-6 TRIO-6 swap. When the auto
    rank-6 swap picks a poor candidate (e.g., horse off 168-day layoff) but
    racing-API form/RPR signal flags a better hedge, the user can set the
    TRIO-6 rank-6 horse here. Schema documented at the top of the JSON file.
    Returns {} if file missing or unreadable so a misfiled override never
    breaks dashboard generation.
    """
    if not os.path.exists(_OVERRIDES_PATH):
        return {}
    try:
        with open(_OVERRIDES_PATH) as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"manual_overrides.json unreadable, skipping: {e}")
        return {}


# Result fields set by results_fetcher that should survive prediction re-runs
_RESULT_KEYS_RACE = {'results_available', 'actual_results', 'actual_winner', 'actual_top3', 'actual_win_odds', 'accuracy'}
_RESULT_KEYS_HORSE = {'actual_position'}

# Pre-race prediction fields that should be FROZEN once a race is finished.
# Re-running generate_predictions after a race ends pulls post-race "odds"
# (HKJC swaps in final SP / dividend prices once the race is settled), which
# corrupts the model blend and produces fake EDGE fires on losing horses.
# Audit 2026-05-17: R1 flipped MED → EDGE post-race because #10's odds
# moved $7.40 → $3.10 in the post-race feed, dragging cal_probs.
_FROZEN_PREDICTION_KEYS_RACE = {'conviction_tier', 'has_edge', 'suggested_stake',
                                 'bankroll', 'odds_age_minutes',
                                 # Shape stats — must move in lockstep with the
                                 # frozen horse-level calibrated_prob (audit
                                 # 2026-05-17, ML expert finding HIGH-1)
                                 'top_2_share', 'top_3_share',
                                 # EDGE banner identifiers — must point at the
                                 # horse whose frozen edge_fires=True
                                 'edge_horse_rank', 'edge_horse_number',
                                 'edge_horse_name',
                                 # Tactical manual override for TRIO-6 rank-6
                                 # slot (set via tracking_data/manual_overrides.json)
                                 'manual_trio6_horse', 'manual_trio6_reason'}
_FROZEN_PREDICTION_KEYS_HORSE = {'rank', 'win_odds', 'calibrated_prob',
                                  'edge_pp', 'ev_win', 'overlay', 'edge_fires',
                                  'pure_prob', 'market_prob', 'model_rank',
                                  'bet_recommendation', 'confidence',
                                  'expected_return', 'value_score'}


def _merge_with_existing(json_file: str, new_races: list) -> list:
    """Preserve result data from a prior prediction file when re-generating.

    If the existing file has actual_position / accuracy fields (set by
    results_fetcher), carry them forward into the new predictions so a
    re-run for fresher odds doesn't wipe scored results.
    """
    # Prefer _with_results variant (has scored results), fall back to base file
    results_file = json_file.replace('.json', '_with_results.json')
    source = None
    for candidate in [results_file, json_file]:
        if os.path.exists(candidate):
            source = candidate
            break

    if not source:
        return new_races

    try:
        with open(source) as f:
            old_data = json.load(f)
        # _with_results wraps races in {"races": [...]}
        old_races = old_data.get('races', old_data) if isinstance(old_data, dict) else old_data
    except (json.JSONDecodeError, OSError):
        return new_races

    # Build lookup: race_number -> old race data
    old_by_num = {r['race_number']: r for r in old_races if isinstance(r, dict)}

    # Sanity check: if all races with results share the same winner, data is likely bogus
    winners_with_results = [
        r.get('actual_winner') for r in old_by_num.values()
        if r.get('results_available') and r.get('actual_winner')
    ]
    if len(winners_with_results) >= 3 and len(set(winners_with_results)) == 1:
        logger.warning(f"  Skipping result merge — all {len(winners_with_results)} races "
                       f"have same winner #{winners_with_results[0]} (likely bogus data)")
        return new_races

    for race in new_races:
        old_race = old_by_num.get(race['race_number'])
        if not old_race or not old_race.get('results_available'):
            continue

        # Copy race-level result fields
        for key in _RESULT_KEYS_RACE:
            if key in old_race:
                race[key] = old_race[key]

        # Freeze pre-race prediction fields on finished races (audit 2026-05-17).
        # HKJC's post-race odds feed corrupts the model blend on re-run,
        # producing fake EDGE fires on horses that finished outside the money.
        for key in _FROZEN_PREDICTION_KEYS_RACE:
            if key in old_race:
                race[key] = old_race[key]

        # Copy horse-level result fields + freeze pre-race predictions
        old_horses = {h['number']: h for h in old_race.get('horses', []) if 'number' in h}
        # Build a horse-number → predicted_rank-ordered list so we can also
        # restore the original rank ordering (post-race feed can reshuffle).
        old_horses_by_rank = sorted(
            (h for h in old_race.get('horses', []) if 'rank' in h),
            key=lambda h: h.get('rank', 99)
        )
        for horse in race.get('horses', []):
            old_h = old_horses.get(horse['number'])
            if old_h:
                for key in _RESULT_KEYS_HORSE:
                    if key in old_h:
                        horse[key] = old_h[key]
                for key in _FROZEN_PREDICTION_KEYS_HORSE:
                    if key in old_h:
                        horse[key] = old_h[key]
        # Re-sort horses by the frozen rank so downstream renderers see the
        # original pre-race ordering, not the post-race shuffled one.
        if old_horses_by_rank:
            race['horses'] = sorted(race.get('horses', []),
                                     key=lambda h: h.get('rank', 99))

        logger.info(f"  Race {race['race_number']}: FROZEN — preserved pre-race "
                    f"predictions + results from prior run")

    return new_races


def _load_previous_race_data(json_file: str) -> list:
    """Return race list from the most recent prior JSON at this path, or []."""
    if not os.path.exists(json_file):
        return []
    try:
        with open(json_file) as f:
            data = json.load(f)
        return data.get('races', data) if isinstance(data, dict) else data
    except (json.JSONDecodeError, OSError):
        return []


def _compute_odds_age_minutes(odds_db, race_date: str, race_number: int):
    """Return minutes elapsed since the most recent odds snapshot for this race.

    Returns None when no snapshot exists OR on any error — caller treats
    None as 'no fresh odds' and FAILS G5. Previously this returned 0, which
    passed the `< 30` gate even when there were zero snapshots in the DB,
    inverting the gate's stated purpose ('stale odds invalidate EV').
    """
    try:
        conn = odds_db._get_conn()
        try:
            row = conn.execute(
                "SELECT MAX(timestamp) FROM odds_snapshots "
                "WHERE race_date = ? AND race_number = ?",
                (race_date, race_number),
            ).fetchone()
            if row and row[0]:
                last_ts = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                return max(0, int((datetime.now() - last_ts).total_seconds() / 60))
        finally:
            conn.close()
    except Exception:
        pass
    return None


def apply_card_exposure_cap(all_races: list, bankroll: float,
                            cap_frac: float = 0.10) -> float:
    """Cap the day's TOTAL forward suggested-stake exposure at cap_frac of
    bankroll, pro-rating every UNFINISHED race's stake down proportionally.

    Per-race stakes are computed independently, so a busy card can put 15-20%
    of bankroll at risk at once (and the box bets within a race are correlated
    — they win/lose together). Settled races aren't at risk so they're excluded
    from the cap. Mutates each capped race's 'suggested_stake' in place and
    returns the applied scale factor (1.0 = no cap needed).
    """
    if not cap_frac or cap_frac <= 0:
        return 1.0
    pending = [r for r in all_races
               if not r.get('results_available') and r.get('suggested_stake')]
    total_stake = sum((r['suggested_stake'].get('amount') or 0.0) for r in pending)
    cap_amount = bankroll * cap_frac
    if not (total_stake > cap_amount > 0):
        return 1.0
    scale = cap_amount / total_stake
    for r in pending:
        s = r['suggested_stake']
        s['amount'] = round((s.get('amount') or 0.0) * scale, -1)
        if s.get('pct') is not None:
            s['pct'] = round(s['pct'] * scale, 2)
        s['card_capped'] = True
        s['method'] = (s.get('method') or '') + \
            f" · card-capped ×{scale:.2f} ({int(cap_frac*100)}% day limit)"
    return scale


def build_race_data(race, analyses, race_date, db, odds_db, analyzer=None):
    """Build the JSON-serializable race data dict from analyses.

    Shared between generator and monitor to avoid duplication.
    Returns race_data dict with horses, speed_map, and exotic_value.
    analyzer: optional CombinedRaceAnalyzer instance — used for ML integrity check (G6).
    """
    # Reset per-run caches so backtests across multiple race dates don't
    # bleed stale jockey/trainer rates between dates.
    from ml.features import clear_feature_caches
    clear_feature_caches()
    race_going = race.going or ''
    race_distance = int(race.distance) if race.distance and str(race.distance).isdigit() else 1200

    # Speed map
    speed_map_input = [
        {'number': h.number, 'name': h.name, 'draw': h.draw,
         'running_style': getattr(h, 'running_style', ''),
         'last_6': h.last_6}
        for h in race.horses
    ]
    speed_map = PaceAnalyzer.generate_speed_map(speed_map_input, race_distance)

    # Odds movement
    odds_summary = odds_db.get_movement_summary(race_date, race.race_number)

    # Loop-invariant: same venue+distance for all horses in this race
    dist_record = db.get_distance_record(race.venue or '', race_distance)

    race_data = {
        "race_number": race.race_number,
        "venue": race.venue,
        "time": race.time,
        "distance": race.distance,
        "class_type": race.class_type,
        "going": race.going,
        "course": race.course,
        "horses": [],
    }

    for i, a in enumerate(analyses):
        h = a.horse
        ml_prob = a.ml_prediction.win_probability if a.ml_prediction else 0
        # Raw (pre-calibration) ML prob — feeds pure_prob per CLAUDE.md spec.
        # Mixing calibrated ML into pure_prob biased edge_pp toward smaller
        # magnitudes; isotonic calibration shifts toward base rate, attenuating
        # strong ML signals on top picks.
        raw_ml_prob = a.ml_prediction.raw_win_probability if a.ml_prediction else 0
        place_prob = a.ml_prediction.place_probability if a.ml_prediction else 0
        calibrated_prob = a.combined_probability if a.combined_probability else 0

        going_form = db.get_horse_going_form(h.name, race_going, race_date)
        jh_stats = db.get_jockey_horse_stats(h.jockey, h.name, race_date)
        dist_pb = db.get_distance_best_time(h.name, race_distance, race_date)
        # Same-format history: prior runs at this exact venue+distance
        # so the dashboard can show "this horse has run here before, here is what happened"
        format_history = db.get_horse_format_history(
            h.name, race.venue or '', race_distance,
            going='',  # going is variable; loose match for now
            before_date=race_date, limit=5,
        )
        # Jockey strike rate (Bayesian-smoothed, prior-only). Reuse the same
        # cached helper the ML feature pipeline uses so we don't re-query.
        from ml.features import FeatureEngineering
        jockey_strike_rate = FeatureEngineering.calculate_jockey_win_rate(
            h.jockey or '', [], conn=db.conn, before_date=race_date,
        )
        jockey_form_30d = FeatureEngineering.calculate_jockey_form_window(
            h.jockey or '', conn=db.conn, before_date=race_date, window_days=30,
        )

        horse_odds_move = odds_summary.get(h.number, {})
        is_steam, steam_severity, steam_label = OddsMovementAnalyzer.detect_steam_move(
            horse_odds_move.get('opening', 0), h.win_odds or 0
        )
        is_cold, cold_label = OddsMovementAnalyzer.detect_cold_horse(
            horse_odds_move.get('opening', 0), h.win_odds or 0, a.rule_score
        )

        race_data["horses"].append({
            "rank": i + 1,
            "number": h.number,
            "name": h.name,
            "jockey": h.jockey,
            "trainer": h.trainer,
            "draw": h.draw,
            "weight": h.weight,
            "gear": h.gear,
            "rating": h.rating,
            "win_odds": h.win_odds,
            "place_odds": h.place_odds,
            "last_6": h.last_6,
            "age": h.age,
            "horse_weight": h.horse_weight,
            "days_since_last_run": h.days_since_last_run,
            "rating_change": h.rating_change,
            "rule_score": a.rule_score,
            "ml_prob": ml_prob,
            "raw_ml_prob": raw_ml_prob,
            "place_prob": place_prob,
            "calibrated_prob": calibrated_prob,
            "expected_return": a.expected_return,
            "bet_recommendation": a.bet_recommendation,
            "confidence": a.combined_confidence,
            "reasoning": a.reasoning,
            "going_form": going_form,
            "jockey_horse_stats": jh_stats,
            "distance_pb": dist_pb,
            "distance_record": dist_record,
            "format_history": format_history,
            "jockey_strike_rate": round(jockey_strike_rate, 4),
            "jockey_form_30d": round(jockey_form_30d, 4),
            "odds_movement": horse_odds_move,
            "steam_move": {"is_steam": is_steam, "severity": steam_severity, "label": steam_label},
            "cold_horse": {"is_cold": is_cold, "label": cold_label},
        })

    race_data["speed_map"] = speed_map

    # === Edge math (pure_prob, market_prob, edge fields, edge_fires) ===
    from betting.edge_calculator import (
        compute_pure_probs, compute_market_probs, compute_edge_fields,
        apply_edge_gates, EdgeGates,
    )
    try:
        from ml.integrity import check_model_integrity
    except ImportError:
        check_model_integrity = None

    # Use raw_ml_prob (pre-calibration) for pure_prob — CLAUDE.md spec compliance.
    # Falls back to ml_prob if raw is missing (legacy JSON, model load failure).
    horses_for_edge = [{
        'rule_score': h['rule_score'],
        'ml_prob': h.get('raw_ml_prob') or h['ml_prob'],
        'win_odds': h['win_odds'] or 0,
    } for h in race_data['horses']]

    pure_probs_list = compute_pure_probs(horses_for_edge, field_size=len(horses_for_edge))
    market_probs_list = compute_market_probs(horses_for_edge)

    # Model integrity — used by G6.
    integrity_ok = False
    try:
        if check_model_integrity is not None:
            ml_model = getattr(analyzer, 'ml_analyzer', None)
            ml_model = getattr(ml_model, 'ml_model', None) if ml_model else None
            metadata = getattr(ml_model, '_metadata', {}) if ml_model else {}
            check_model_integrity(metadata)
            integrity_ok = True
    except Exception:
        integrity_ok = False

    # Persist the just-fetched odds before computing age — otherwise
    # _compute_odds_age_minutes returns None (no snapshot in DB) and G5
    # fails for every EDGE candidate even though the dashboard is showing
    # the freshest possible odds. This snapshot is only used downstream by
    # the EDGE freshness gate; nothing else reads it on the predict path.
    # Audit fix 2026-05-17.
    # Anti-poison: skip the snapshot if the race already has results in
    # race_results — HKJC post-race odds (final SP / settled) would
    # otherwise overwrite the legitimate pre-race odds we captured earlier
    # and corrupt any future backtest that reads odds_snapshots.
    try:
        race_finished = False
        if db is not None and hasattr(db, 'conn'):
            row = db.conn.execute(
                "SELECT 1 FROM race_results WHERE race_date=? AND race_number=? LIMIT 1",
                (race_date, race.race_number),
            ).fetchone()
            race_finished = bool(row)
        if not race_finished:
            odds_horses = [
                {'number': h.number, 'name': h.name,
                 'win_odds': h.win_odds or 0, 'place_odds': getattr(h, 'place_odds', 0) or 0}
                for h in race.horses if (h.win_odds or 0) > 0
            ]
            if odds_horses:
                odds_db.record_snapshot(race_date, race.race_number,
                                        odds_horses, source='generator')
    except Exception:
        pass  # snapshot failure must not block prediction generation

    # G5 odds freshness — compute once for the race, reuse per-horse
    odds_age_min = _compute_odds_age_minutes(odds_db, race_date, race.race_number)

    gates = EdgeGates()
    has_edge_race = False
    for i, h in enumerate(race_data['horses']):
        pure = pure_probs_list[i]
        market = market_probs_list[i]
        odds = h['win_odds'] or 0
        fields = compute_edge_fields(pure, market, odds)

        model_rank = h['rank']  # already 1-indexed in the horses loop

        ok, _failed = apply_edge_gates(
            ev_win=fields['ev_win'], pure_prob=pure, edge_pp=fields['edge_pp'],
            model_rank=model_rank, odds_age_min=odds_age_min,
            integrity_ok=integrity_ok, gates=gates,
        )

        h['pure_prob'] = round(pure, 4)
        h['market_prob'] = round(market, 4)
        h['edge_pp'] = round(fields['edge_pp'], 2)
        h['ev_win'] = round(fields['ev_win'], 3)
        h['overlay'] = round(fields['overlay'], 3)
        h['edge_fires'] = ok
        h['model_rank'] = model_rank
        if ok:
            has_edge_race = True

    race_data['has_edge'] = has_edge_race
    race_data['odds_age_minutes'] = odds_age_min

    # Manual TRIO-6 rank-6 override — tactical hedge when racing-API form/RPR
    # signal beats the auto rank-6 swap (e.g., auto-swap picks horse with
    # 168-day layoff). Surface as race-level fields so the template can apply
    # the override without recomputing form data client-side.
    _overrides = _load_manual_overrides()
    _race_overrides = (_overrides.get(race_date) or {}).get(str(race.race_number)) or {}
    if _race_overrides.get('trio6_horse'):
        race_data['manual_trio6_horse'] = _race_overrides['trio6_horse']
        race_data['manual_trio6_reason'] = _race_overrides.get('reason', '')

    # Identify which horse fired EDGE — alert text was misleadingly hardcoded
    # to "the top pick", but G4 allows max_model_rank=2 so rank-2 can fire EDGE
    # while rank-1 doesn't. Surface the lowest-rank firing horse so the
    # template can name it correctly.
    edge_firers = [h for h in race_data['horses'] if h.get('edge_fires')]
    if edge_firers:
        edge_firers.sort(key=lambda h: h['rank'])
        eh = edge_firers[0]
        race_data['edge_horse_rank'] = eh['rank']
        race_data['edge_horse_number'] = eh['number']
        race_data['edge_horse_name'] = eh['name']

    # === Post-bet analytics: Harville dividend + conviction tier + EV ===
    from betting.dividends import (
        estimate_quinella_dividend, quinella_bet_ev, normalize_market_probs
    )
    horses_json = race_data['horses']
    qin_indices = [i for i, h in enumerate(horses_json) if h['bet_recommendation'] == 'QIN']
    win_odds_list = [h.get('win_odds') or 0 for h in horses_json]
    model_probs = [h.get('calibrated_prob') or 0 for h in horses_json]
    market_probs = normalize_market_probs(win_odds_list)

    # Conviction tier (replaces strategy_confidence — spec §4)
    from betting.conviction import assign_conviction_tier
    sorted_probs = sorted(model_probs, reverse=True)
    top_2_share = sum(sorted_probs[:2]) if len(sorted_probs) >= 2 else 0
    top_3_share = sum(sorted_probs[:3]) if len(sorted_probs) >= 3 else 0
    race_data['conviction_tier'] = assign_conviction_tier(
        has_edge=race_data['has_edge'],
        top_2_share=top_2_share,
        top_3_share=top_3_share,
    )
    race_data['top_2_share'] = round(top_2_share, 4)
    race_data['top_3_share'] = round(top_3_share, 4)

    # Quinella dividend estimate + EV (only when QIN bet is active)
    if len(qin_indices) >= 2:
        pair_combos = list(itertools.combinations(qin_indices, 2))
        ev_result = quinella_bet_ev(model_probs, market_probs, pair_combos)
        race_data['qin_bet_ev'] = {
            'n_combos': len(pair_combos),
            'total_stake': ev_result['total_stake'],
            'expected_profit': round(ev_result['expected_profit'], 2),
            'ev_pct': round(ev_result['ev_pct'], 1),
            'hit_prob': round(ev_result['hit_prob'], 4),
            'avg_dividend': round(ev_result['avg_dividend'], 2),
        }
        # Top pair dividend — prefer real HKJC payout if race has results
        top_div = None
        top_div_source = 'harville'
        if db and race.date and race.race_number:
            # If results already in, actual_winner + 2nd give us the real pair
            try:
                finishers = db.conn.execute(
                    "SELECT horse_number, finish_position FROM race_results "
                    "WHERE race_date = ? AND race_number = ? AND finish_position IN (1, 2)",
                    (race.date, race.race_number)
                ).fetchall()
                if len(finishers) == 2:
                    h1, h2 = finishers[0][0], finishers[1][0]
                    real_div = db.get_quinella_dividend(race.date, race.race_number, h1, h2)
                    if real_div:
                        top_div = real_div
                        top_div_source = 'real'
            except Exception:
                pass
        if top_div is None:
            # estimate_quinella_dividend returns a per-$1 figure; the real-payout
            # path (db.get_quinella_dividend) returns the raw per-HK$10 HKJC
            # declaration. Scale the estimate ×10 so the DISPLAYED dividend is on
            # the same per-$10 basis punters expect (was silently jumping 10×
            # when real results replaced the estimate).
            est = estimate_quinella_dividend(market_probs, qin_indices[0], qin_indices[1])
            top_div = est * 10 if est else None
        race_data['qin_top_pair_dividend'] = round(top_div, 2) if top_div else None
        race_data['qin_dividend_source'] = top_div_source

    # Exotic bet value detection
    if len(analyses) >= 4:
        # Edge math must use pure_prob (pre-blend); calibrated_prob is
        # market-blended, which mechanically shrinks edge vs the market.
        model_probs = [h['pure_prob'] for h in race_data['horses']]
        market_probs = []
        for h in race_data['horses']:
            wo = h.get('win_odds', 0)
            market_probs.append(1.0 / wo if wo and wo > 0 else 0)
        mp_total = sum(market_probs) or 1.0
        market_probs = [p / mp_total for p in market_probs]
        horse_nums = [h['number'] for h in race_data['horses']]
        place_probs = [h.get('place_prob', 0) for h in race_data['horses']]
        horse_draws = [h.get('draw', 0) for h in race_data['horses']]
        horse_styles = [
            next((r.split(':')[0] for r in (h.get('reasoning') or [])
                  if 'Leader' in r or 'Stalker' in r or 'Midfield' in r or 'Closer' in r), '')
            for h in race_data['horses']
        ]
        exotics = find_value_exotics(
            model_probs, market_probs, horse_nums, min_edge=0.10,
            place_probs=place_probs, draws=horse_draws, styles=horse_styles,
        )
        race_data["exotic_value"] = exotics

    return race_data


async def main():
    race_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y/%m/%d')
    if not re.fullmatch(r'\d{4}/\d{2}/\d{2}', race_date):
        print(f"Invalid date format: {race_date}. Expected YYYY/MM/DD")
        sys.exit(1)

    logger.info(f"Generating predictions for {race_date}")

    fetcher = HKJCDataFetcher()
    analyzer = CombinedRaceAnalyzer()
    # Lazy-started Playwright browser sits on fetcher._pw_fetcher. Without an
    # explicit close() at the end the chromium subprocess lingers and stacks
    # across shadow runs that immediately follow (audit fix 2026-05-17).
    try:
        await _run_generation(fetcher, analyzer, race_date)
    finally:
        await fetcher.close()


async def _run_generation(fetcher, analyzer, race_date: str):

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_races = []

    db = HorseDatabase()
    odds_db = OddsDatabase()

    # Fetch all race cards concurrently (was sequential 1→11 — the main
    # generate_predictions bottleneck). analyze/build stay sequential below
    # since they touch the DB; only the network fetch is parallelized.
    logger.info("Fetching all race cards concurrently...")
    fetched = await asyncio.gather(
        *(fetcher.get_race_card(race_date, rn) for rn in range(1, 12)),
        return_exceptions=True,
    )

    for race_no, race in enumerate(fetched, start=1):
        if isinstance(race, Exception):
            logger.warning(f"Race {race_no} fetch failed: {race}, stopping")
            break
        if not race or not race.horses:
            logger.info(f"No data for Race {race_no}, stopping")
            break

        analyses = analyzer.analyze_race(race)
        race_data = build_race_data(race, analyses, race_date, db, odds_db, analyzer=analyzer)
        all_races.append(race_data)
        logger.info(f"  Race {race_no}: {len(race.horses)} horses, bet: {analyses[0].bet_recommendation if analyses else 'N/A'}")

    date_str = race_date.replace('/', '')
    json_file = os.path.join(results_dir, f"race_data_{date_str}.json")

    # Merge: preserve result fields from prior run (e.g. actual_position, accuracy)
    all_races = _merge_with_existing(json_file, all_races)

    # Tier stats — load from cache (refreshed on results ingestion, Task 8)
    from betting.tier_stats import load_cached_stats
    tier_stats = load_cached_stats()
    for r in all_races:
        r['tier_stats'] = tier_stats

    # Per-card suggested stake — bankroll-fraction (BOX) or half-Kelly
    # (WIN/BANKER WIN), tier-capped. See betting/optimizer.compute_suggested_stake.
    # Skip recompute on finished races — _merge_with_existing already restored
    # the frozen pre-race stake (audit 2026-05-17, Python expert finding HIGH).
    from betting.optimizer import compute_suggested_stake
    import config as _cfg
    for r in all_races:
        if r.get('results_available'):
            continue
        top = (r.get('horses') or [{}])[0]
        stake = compute_suggested_stake(
            conviction_tier=r.get('conviction_tier') or 'SKIP',
            bet_type=top.get('bet_recommendation') or '',
            top_prob=top.get('calibrated_prob') or 0.0,
            top_odds=top.get('win_odds') or 0.0,
            bankroll=_cfg.BANKROLL,
            top_2_share=r.get('top_2_share'),
        )
        r['suggested_stake'] = stake
        r['bankroll'] = _cfg.BANKROLL

    # Card-level exposure cap (see _apply_card_exposure_cap).
    apply_card_exposure_cap(all_races, _cfg.BANKROLL,
                            getattr(_cfg, 'CARD_EXPOSURE_CAP', 0.10))

    # Alerts — diff current vs previous race_data file
    from betting.alerts import compute_alerts, AlertState, load_alert_state, save_alert_state
    prev_json = _load_previous_race_data(json_file)
    state = load_alert_state(race_date)
    alerts = compute_alerts(all_races, prev_json, state)
    save_alert_state(race_date, state)
    for r in all_races:
        r['alerts'] = [a for a in alerts if a['race'] == r['race_number']]

    _atomic_write(json_file, json.dumps(all_races, indent=2))

    # Log predictions to DB for longitudinal accuracy tracking
    model_ver = analyzer.ml_analyzer.ml_model.model_version if hasattr(analyzer, 'ml_analyzer') else ''
    db.log_predictions(race_date, all_races, model_version=model_ver)

    # Snapshot 60-day accuracy for the dashboard's "Recent accuracy" panel +
    # WhatsApp tier-trust footer. Safe-fallback to None when no results yet.
    try:
        accuracy_summary = db.get_accuracy_summary(days=60)
        if 'message' in accuracy_summary:
            accuracy_summary = None
    except Exception:
        accuracy_summary = None

    db.close()

    logger.info(f"Saved predictions to {json_file}")
    logger.info(f"Total races: {len(all_races)}")

    json_data = json.dumps(all_races)
    # Prevent </script> in JSON from breaking the <script> block (XSS defense)
    json_data = json_data.replace('</script>', r'<\/script>').replace('</Script>', r'<\/Script>')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Detect venue from race data
    venue = "Happy Valley" if any(r.get("venue", "").startswith("Happy") for r in all_races) else "Sha Tin"

    # Load EDGE stats (live count from track_edge.py — replaces the
    # hardcoded N=4 banner text). Silent fallback to None if file missing.
    edge_stats = None
    try:
        es_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'tracking_data', 'edge_stats.json')
        if os.path.exists(es_path):
            with open(es_path) as f:
                edge_stats = json.load(f)
    except (json.JSONDecodeError, IOError):
        pass

    html = generate_html(json_data, now, race_date, venue,
                          accuracy_summary=accuracy_summary,
                          edge_stats=edge_stats)

    html_file = os.path.join(results_dir, f"racing_dashboard_{date_str}.html")
    latest_file = os.path.join(results_dir, "racing_dashboard.html")
    _atomic_write(html_file, html)
    _atomic_write(latest_file, html)

    logger.info(f"Saved dashboard to {html_file}")
    logger.info(f"Saved latest dashboard to {latest_file}")
    logger.info("Done!")


if __name__ == '__main__':
    asyncio.run(main())
