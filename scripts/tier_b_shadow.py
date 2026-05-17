"""Tier B forward-validation shadow logger.

Records per-race what the OLD recommender (pre-2026-05-17, with conviction
thresholds 0.33/0.42 and TRIO branch enabled) would have done, so we can
measure whether the Tier B changes (conviction raise + TRIO disable) help
or hurt over forward data.

Also logs EDGE candidates at lower G3 thresholds (+2pp, +3pp) alongside
production (+5pp). Production EDGE has fired twice in ~60d — too rare
to validate the threshold. Forward shadow lets us see what the lower
thresholds would have caught without actually betting on them.

Reads results/race_data_<date>.json (already-generated production
predictions) and writes one row per race to tracking_data/tier_b_log.csv.
No HKJC re-fetch — purely local replay of recommender decisions against
the predicted probabilities the production model already produced.

After the meeting, run `scripts.tier_b_evaluate <date>` to look up actual
results from race_results / race_dividends and score both schemes.

Usage:
    python3 -m scripts.tier_b_shadow '2026/05/17'
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / "tracking_data" / "tier_b_log.csv"

# OLD thresholds (pre-2026-05-17 audit)
OLD_HIGH_T2, OLD_HIGH_T3 = 0.33, 0.42
OLD_MED_T2, OLD_MED_T3 = 0.27, 0.35
# NEW thresholds (current production)
NEW_HIGH_T2, NEW_HIGH_T3 = 0.40, 0.55
NEW_MED_T2, NEW_MED_T3 = 0.30, 0.42
LOW_T2 = 0.22


def _compute_shape(horses: list) -> dict | None:
    """Race shape stats from sorted calibrated_probs. None if <2 horses."""
    probs = sorted(
        [(h.get('calibrated_prob') or 0) for h in horses],
        reverse=True,
    )
    if len(probs) < 2:
        return None
    return {
        'n': len(probs),
        'top_2_share': probs[0] + probs[1],
        'top_3_share': sum(probs[:3]),
        'gap_1_2': probs[0] - probs[1],
        'gap_2_3': (probs[1] - probs[2]) if len(probs) >= 3 else 0.0,
    }


def _tier(top2: float, top3: float, has_edge: bool,
          h2: float, h3: float, m2: float, m3: float) -> str:
    if has_edge:
        return 'EDGE'
    if top2 >= h2 or top3 >= h3:
        return 'HIGH'
    if top2 >= m2 or top3 >= m3:
        return 'MED'
    if top2 >= LOW_T2:
        return 'LOW'
    return 'SKIP'


def _would_old_trio_fire(shape: dict) -> bool:
    """OLD TRIO gate: both gaps tight AND top_3_share strong AND n>=3."""
    return (shape['n'] >= 3
            and shape['gap_1_2'] < 0.025
            and shape['gap_2_3'] < 0.025
            and shape['top_3_share'] > 0.32)


# Production EDGE gates (betting/edge_calculator.py:EdgeGates)
_EDGE_MIN_EV = 0.05
_EDGE_MIN_PROB = 0.10
_EDGE_MAX_RANK = 2
_EDGE_MAX_AGE_MIN = 30


def _edge_fires_at(edge_pp: float, ev_win: float, pure_prob: float,
                    rank: int, odds_age_min, g3_threshold: float) -> bool:
    """All 6 gates with G3 swept. G6 (integrity) trusted true here — if it
    were false the prod recommender would have skipped this race entirely.
    """
    if edge_pp is None or ev_win is None or pure_prob is None:
        return False
    if ev_win < _EDGE_MIN_EV:
        return False
    if pure_prob < _EDGE_MIN_PROB:
        return False
    if edge_pp < g3_threshold:
        return False
    if rank > _EDGE_MAX_RANK:
        return False
    if odds_age_min is None or odds_age_min > _EDGE_MAX_AGE_MIN:
        return False
    return True


def _scan_edge_shadow(horses: list, odds_age_min) -> dict:
    """For each G3 threshold, pick the top-2 horse with the largest edge_pp
    that passes all gates. Returns the horse_number (or empty) per threshold.
    """
    candidates_by_threshold = {2.0: [], 3.0: [], 5.0: []}
    for h in horses[:2]:  # G4: model_rank ≤ 2
        rank = h.get('rank')
        edge_pp = h.get('edge_pp')
        ev_win = h.get('ev_win')
        pure_prob = h.get('pure_prob')
        num = h.get('number')
        for thr in candidates_by_threshold.keys():
            if _edge_fires_at(edge_pp, ev_win, pure_prob, rank, odds_age_min, thr):
                candidates_by_threshold[thr].append((edge_pp, num))
    # Pick best edge_pp per threshold (largest = most disagreement with market)
    out = {}
    for thr, picks in candidates_by_threshold.items():
        if picks:
            best = max(picks, key=lambda p: p[0])
            out[thr] = best[1]
        else:
            out[thr] = ''
    return out


def _current_race_bet(horses: list) -> str:
    """Roll up per-horse bet_recommendation to a single race-level bet type.

    Mirrors the inference in scripts/shadow_generate._infer_bet_type.
    """
    for h in horses[:4]:
        rec = (h.get('bet_recommendation') or '').strip()
        if rec in ('BANKER WIN', 'WIN'):
            return 'WIN'
        if rec == 'TRIO':
            return 'TRIO'
    qin_count = sum(1 for h in horses[:4]
                    if h.get('bet_recommendation') == 'QIN')
    if qin_count == 4:
        return 'BOX Q4'
    if qin_count == 3:
        return 'BOX Q'
    if qin_count == 2:
        return 'QIN'
    return 'SKIP'


def run(race_date: str) -> int:
    date_str = race_date.replace('/', '')
    src = REPO_ROOT / 'results' / f'race_data_{date_str}.json'
    if not src.exists():
        print(f"FAIL: {src} not found. Run generate_predictions.py first.",
              file=sys.stderr)
        return 1

    with open(src) as f:
        races = json.load(f)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fresh = not LOG_PATH.exists()
    fields = [
        'race_date', 'race_number', 'n_horses',
        'top_2_share', 'top_3_share', 'gap_1_2', 'gap_2_3',
        'tier_old', 'tier_new', 'bet_now', 'bet_old',
        'top_3_horse_numbers', 'top_4_horse_numbers',
        # EDGE shadow — horse_number that would have fired at each G3 cutoff
        # (empty if no top-2 horse passes all gates at that threshold). The
        # production cutoff is +5pp; +3 and +2 are observation-only.
        'edge_5pp_horse', 'edge_3pp_horse', 'edge_2pp_horse',
        'odds_age_min',
    ]
    rows_written = 0
    with open(LOG_PATH, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if fresh:
            w.writeheader()
        for r in races:
            horses = r.get('horses') or []
            shape = _compute_shape(horses)
            if not shape:
                continue
            has_edge = bool(r.get('has_edge'))
            tier_old = _tier(shape['top_2_share'], shape['top_3_share'], has_edge,
                             OLD_HIGH_T2, OLD_HIGH_T3, OLD_MED_T2, OLD_MED_T3)
            tier_new = _tier(shape['top_2_share'], shape['top_3_share'], has_edge,
                             NEW_HIGH_T2, NEW_HIGH_T3, NEW_MED_T2, NEW_MED_T3)
            bet_now = _current_race_bet(horses)
            # OLD cascade order: WIN → TRIO → BOX Q4 → BOX Q → QIN → SKIP.
            # TRIO was inserted between WIN and BOX*. So bet_old differs from
            # bet_now only when (a) bet_now is NOT WIN (WIN gate unchanged)
            # AND (b) the OLD TRIO gate would have fired.
            if bet_now in ('WIN', 'BANKER WIN'):
                bet_old = bet_now
            elif _would_old_trio_fire(shape):
                bet_old = 'TRIO'
            else:
                bet_old = bet_now

            top3 = ','.join(str(h.get('number') or '') for h in horses[:3])
            top4 = ','.join(str(h.get('number') or '') for h in horses[:4])

            odds_age = r.get('odds_age_minutes')
            edge_shadow = _scan_edge_shadow(horses, odds_age)

            w.writerow({
                'race_date': race_date,
                'race_number': r.get('race_number'),
                'n_horses': shape['n'],
                'top_2_share': round(shape['top_2_share'], 4),
                'top_3_share': round(shape['top_3_share'], 4),
                'gap_1_2': round(shape['gap_1_2'], 4),
                'gap_2_3': round(shape['gap_2_3'], 4),
                'tier_old': tier_old,
                'tier_new': tier_new,
                'bet_now': bet_now,
                'bet_old': bet_old,
                'top_3_horse_numbers': top3,
                'top_4_horse_numbers': top4,
                'edge_5pp_horse': edge_shadow.get(5.0, ''),
                'edge_3pp_horse': edge_shadow.get(3.0, ''),
                'edge_2pp_horse': edge_shadow.get(2.0, ''),
                'odds_age_min': odds_age if odds_age is not None else '',
            })
            rows_written += 1
            if bet_old != bet_now:
                print(f"  R{r.get('race_number')}: bet_old={bet_old} → bet_now={bet_now} "
                      f"(top_3_share={shape['top_3_share']:.2f}, gap_1_2={shape['gap_1_2']:.3f})")
            if tier_old != tier_new:
                print(f"  R{r.get('race_number')}: tier_old={tier_old} → tier_new={tier_new}")
            if edge_shadow.get(2.0):
                msg = f"  R{r.get('race_number')}: edge candidates — "
                msg += f"5pp={edge_shadow.get(5.0) or '—'}, "
                msg += f"3pp={edge_shadow.get(3.0) or '—'}, "
                msg += f"2pp={edge_shadow.get(2.0) or '—'}"
                print(msg)

    print(f"Wrote {rows_written} rows to {LOG_PATH}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('race_date', help='YYYY/MM/DD')
    args = p.parse_args()
    sys.exit(run(args.race_date))


if __name__ == '__main__':
    main()
