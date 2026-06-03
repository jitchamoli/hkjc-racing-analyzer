"""Evaluate production vs shadow bets against actual results for a race date.

Two variants supported (mirror of scripts/shadow_generate.py):

  rule_w       (default, legacy):
               Reads results/shadow_race_data_YYYYMMDD.json (rule_w=0.9)
               Logs to tracking_data/shadow_log.csv

  jockey_form:
               Reads results/shadow_jockey_race_data_YYYYMMDD.json
                                                  (jockey_form factor disabled)
               Logs to tracking_data/shadow_jockey_log.csv

Run AFTER `results_fetcher.py` has filled in real finishes. Computes hit + P&L
for both prod + shadow variants using real HKJC dividends. Appends one row
per race to the variant's CSV log so we accumulate data across meetings.
After ~6 meetings (~60 races), run with --report for cumulative comparison.

Usage:
    python3 -m scripts.shadow_evaluate '2026/05/17'
    python3 -m scripts.shadow_evaluate '2026/05/17' --variant jockey_form
    python3 -m scripts.shadow_evaluate --report                            # rule_w
    python3 -m scripts.shadow_evaluate --report --variant jockey_form
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from betting.dividends import hkjc_declared_to_unit

REPO_ROOT = Path(__file__).resolve().parent.parent

VARIANTS = {
    'rule_w': {
        'shadow_basename': 'shadow_race_data',
        'log_path': REPO_ROOT / 'tracking_data' / 'shadow_log.csv',
        'prod_label': 'production (rule_w=0.6)',
        'shadow_label': 'shadow (rule_w=0.9)',
        'ship_hint': 'Consider shipping RULE_WEIGHT=0.9.',
        'keep_hint': 'Keep RULE_WEIGHT=0.6.',
    },
    'jockey_form': {
        'shadow_basename': 'shadow_jockey_race_data',
        'log_path': REPO_ROOT / 'tracking_data' / 'shadow_jockey_log.csv',
        'prod_label': 'production (jockey_form on)',
        'shadow_label': 'shadow (jockey_form OFF)',
        'ship_hint': 'Consider shipping JOCKEY_FORM_RIDES_MIN=15 (disable factor).',
        'keep_hint': 'Keep JOCKEY_FORM_RIDES_MIN=3 (factor enabled).',
    },
}

LOG_HEADERS = [
    'race_date', 'race_number',
    'prod_bet', 'prod_legs', 'prod_hit', 'prod_pnl', 'prod_stake',
    'shadow_bet', 'shadow_legs', 'shadow_hit', 'shadow_pnl', 'shadow_stake',
    'q_dividend', 'actual_top2',
]


def _cost(bet_type: str) -> float:
    return {'QIN': 1.0, 'BOX Q': 3.0, 'BOX Q4': 6.0, 'WIN': 1.0,
            'BANKER WIN': 1.0, 'PLACE': 1.0, 'TRIO': 1.0}.get(bet_type, 0.0)


def _name(leg) -> str:
    # Production race_data uses 'name'; shadow race_data uses 'horse_name'.
    # Single accessor lets the evaluator consume both shapes.
    return leg.get('name') or leg.get('horse_name') or ''


def _legs_for_bet(bet_type: str, ranked_legs):
    """ranked_legs already sorted by rank. Return horse names for the bet."""
    if bet_type == 'BOX Q4': return {_name(l) for l in ranked_legs[:4]}
    if bet_type == 'BOX Q':  return {_name(l) for l in ranked_legs[:3]}
    if bet_type == 'QIN':    return {_name(l) for l in ranked_legs[:2]}
    if bet_type == 'TRIO':   return {_name(l) for l in ranked_legs[:3]}
    if bet_type in ('WIN', 'BANKER WIN', 'PLACE'): return {_name(ranked_legs[0])}
    return set()


def _is_hit(bet_type: str, legs: set, actual_top2: set, actual_top3: set, top1_won: bool) -> bool:
    if bet_type in ('BOX Q', 'BOX Q4'): return actual_top2.issubset(legs)
    if bet_type == 'QIN':               return legs == actual_top2
    if bet_type == 'TRIO':              return actual_top3.issubset(legs) and len(legs) >= 3
    if bet_type in ('WIN', 'BANKER WIN'): return top1_won
    # PLACE — top horse hits if it finished top-3 (HKJC PLACE pool pays 1st/2nd/3rd).
    # legs is a 1-element set with rank-1 name; we check it intersects actual_top3.
    if bet_type == 'PLACE': return bool(legs & actual_top3)
    return False


def _pnl(bet_type: str, hit: bool, q_dividend: float | None) -> tuple[float | None, float]:
    """Returns (pnl, stake). pnl is None when no dividend / not a quinella bet."""
    cost = _cost(bet_type)
    if bet_type in ('QIN', 'BOX Q', 'BOX Q4'):
        if q_dividend is None: return None, 0.0
        return (q_dividend - cost) if hit else (-cost), cost
    # WIN/PLACE/TRIO P&L not computed here — out of MVP scope
    return None, 0.0


def evaluate_date(race_date: str, variant: str) -> int:
    cfg = VARIANTS[variant]
    date_str = race_date.replace('/', '')
    results_dir = REPO_ROOT / "results"
    prod_path = results_dir / f"race_data_{date_str}.json"
    shadow_path = results_dir / f"{cfg['shadow_basename']}_{date_str}.json"

    if not prod_path.exists():
        print(f"ERROR: production file missing: {prod_path}", file=sys.stderr)
        return 1
    if not shadow_path.exists():
        print(f"ERROR: shadow file missing: {shadow_path}. "
              f"Run scripts.shadow_generate --variant {variant} first.",
              file=sys.stderr)
        return 1

    prod = json.loads(prod_path.read_text())
    shadow = json.loads(shadow_path.read_text())['races']
    shadow_by_race = {r['race_number']: r for r in shadow}

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    rows_to_append = []
    print(f"Variant: {variant} ({cfg['prod_label']} vs {cfg['shadow_label']})")
    print(f"Race  | prod_bet  prod_hit prod_pnl  | shadow_bet shadow_hit shadow_pnl | dividend")
    print('-' * 90)

    for race in prod:
        rn = race['race_number']
        # Production bet info — derive actual box size from horse count, not
        # just rank-1's bet_recommendation. Rank-1's rec says 'QIN' equally
        # for STRAIGHT QIN (2 horses) and BOX QIN-4 (4 horses) — different
        # stakes (1u vs 6u) and very different hit rates. _infer_bet_type
        # counts horses with QIN/TRIO rec and resolves to STRAIGHT/BOX-N.
        # Falls back to rank-1 rec for WIN/BANKER WIN which aren't box-bets.
        horses_sorted = sorted(race['horses'], key=lambda h: h['rank'])
        prod_bet = _infer_bet_type_from_horses(horses_sorted)
        prod_legs_set = _legs_for_bet(prod_bet, horses_sorted)

        # Shadow
        sh = shadow_by_race.get(rn)
        if not sh:
            shadow_bet = 'SKIP'; shadow_legs_set = set()
        else:
            shadow_bet = sh['bet_type']
            shadow_legs_set = _legs_for_bet(shadow_bet, sh['legs'])

        # Actual finishes from race_results
        actual_rows = conn.execute(
            "SELECT horse_name, horse_number, finish_position, is_winner, is_top3 "
            "FROM race_results WHERE race_date=? AND race_number=?", (race_date, rn)
        ).fetchall()
        if not actual_rows:
            print(f"R{rn:>2}  | (no results yet)")
            continue
        actual_top2 = {r['horse_name'] for r in actual_rows if r['finish_position'] in (1, 2)}
        actual_top3 = {r['horse_name'] for r in actual_rows if r['is_top3']}
        top1_won = any(r['horse_name'] in prod_legs_set and r['is_winner'] for r in actual_rows)

        # Real quinella dividend
        win2 = [r for r in actual_rows if r['finish_position'] in (1, 2)]
        q_div = None
        if len(win2) == 2:
            d = conn.execute(
                "SELECT dividend FROM race_dividends WHERE race_date=? AND race_number=? "
                "AND pool='QUINELLA' AND combination=?",
                (race_date, rn, f"{min(win2[0]['horse_number'], win2[1]['horse_number'])},"
                                f"{max(win2[0]['horse_number'], win2[1]['horse_number'])}"),
            ).fetchone()
            q_div = hkjc_declared_to_unit(d['dividend']) if d else None

        prod_hit = _is_hit(prod_bet, prod_legs_set, actual_top2, actual_top3, top1_won)
        sh_top1_won = any(r['horse_name'] in shadow_legs_set and r['is_winner'] for r in actual_rows)
        shadow_hit = _is_hit(shadow_bet, shadow_legs_set, actual_top2, actual_top3, sh_top1_won)
        prod_pnl, prod_stake = _pnl(prod_bet, prod_hit, q_div)
        shadow_pnl, shadow_stake = _pnl(shadow_bet, shadow_hit, q_div)

        prod_pnl_str = f"{prod_pnl:>+6.1f}u" if prod_pnl is not None else "  -   "
        shadow_pnl_str = f"{shadow_pnl:>+6.1f}u" if shadow_pnl is not None else "  -   "
        div_str = f"${q_div:>5.0f}" if q_div is not None else " - "

        prod_hit_str = '✓' if prod_hit else '✗' if prod_bet != 'SKIP' else '-'
        shadow_hit_str = '✓' if shadow_hit else '✗' if shadow_bet != 'SKIP' else '-'
        print(f"R{rn:>2}  | {prod_bet:<9} {prod_hit_str:<8} {prod_pnl_str}    | "
              f"{shadow_bet:<10} {shadow_hit_str:<10} {shadow_pnl_str}   | {div_str}")

        rows_to_append.append({
            'race_date': race_date, 'race_number': rn,
            'prod_bet': prod_bet, 'prod_legs': '|'.join(sorted(prod_legs_set)),
            'prod_hit': int(prod_hit), 'prod_pnl': prod_pnl if prod_pnl is not None else '',
            'prod_stake': prod_stake,
            'shadow_bet': shadow_bet, 'shadow_legs': '|'.join(sorted(shadow_legs_set)),
            'shadow_hit': int(shadow_hit), 'shadow_pnl': shadow_pnl if shadow_pnl is not None else '',
            'shadow_stake': shadow_stake,
            'q_dividend': q_div if q_div is not None else '',
            'actual_top2': '|'.join(sorted(actual_top2)),
        })

    conn.close()
    _append_log(rows_to_append, race_date, cfg['log_path'])
    print(f"\nAppended {len(rows_to_append)} rows to {cfg['log_path']}")
    return 0


def _infer_bet_type_from_horses(horses_sorted) -> str:
    """Derive race-level bet type from per-horse bet_recommendation values.

    Handles all bet_recommendation values emitted by core/recommender.py:
    WIN, BANKER WIN, PLACE, QIN, TRIO (and SKIP). PLACE is rare but real —
    bug 2026-05-17: this returned SKIP for PLACE races until the explicit
    branch was added.
    """
    for h in horses_sorted[:4]:
        rec = h.get('bet_recommendation', '')
        if rec in ('BANKER WIN', 'WIN'): return 'WIN'
        if rec == 'PLACE': return 'PLACE'
        if rec == 'TRIO': return 'TRIO'
    qin = sum(1 for h in horses_sorted[:4] if h.get('bet_recommendation') == 'QIN')
    if qin == 4: return 'BOX Q4'
    if qin == 3: return 'BOX Q'
    if qin == 2: return 'QIN'
    return 'SKIP'


def _append_log(rows, race_date, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Dedupe: drop any existing rows for this race_date first
    existing = []
    if log_path.exists():
        with open(log_path) as f:
            reader = csv.DictReader(f)
            existing = [r for r in reader if r['race_date'] != race_date]
    with open(log_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LOG_HEADERS)
        w.writeheader()
        for r in existing: w.writerow(r)
        for r in rows: w.writerow(r)


from scripts.forward_common import bayesian_verdict, MIN_MEETINGS


def _per_meeting_winners(rows) -> list[int]:
    """Compute per-meeting binary outcomes from the shadow CSV rows.

    For each meeting in chronological order, returns 1 if shadow CUMULATIVE
    ROI (across all meetings up to and including this one) beats prod
    cumulative ROI, else 0. Cumulative ROI is used (vs per-meeting ROI) so
    early variance from a single noisy meeting doesn't flip the verdict.
    """
    meetings = sorted(set(r['race_date'] for r in rows))
    out = []
    for end_date in meetings:
        through = [r for r in rows if r['race_date'] <= end_date]
        p_stake = sum(float(r['prod_stake'] or 0) for r in through)
        s_stake = sum(float(r['shadow_stake'] or 0) for r in through)
        p_pnl = sum(float(r['prod_pnl']) for r in through if r['prod_pnl'] != '')
        s_pnl = sum(float(r['shadow_pnl']) for r in through if r['shadow_pnl'] != '')
        p_roi = (p_pnl / p_stake) if p_stake else 0
        s_roi = (s_pnl / s_stake) if s_stake else 0
        out.append(1 if s_roi > p_roi else 0)
    return out


def report(variant: str) -> int:
    cfg = VARIANTS[variant]
    log_path = cfg['log_path']
    if not log_path.exists():
        print(f"No {log_path.name} yet. Run scripts.shadow_evaluate --variant {variant} "
              f"<date> after each meeting.")
        return 0
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Log empty.")
        return 0

    meetings = sorted(set(r['race_date'] for r in rows))
    print(f"Cumulative shadow comparison — variant={variant} — "
          f"{len(meetings)} meetings, {len(rows)} races")
    print(f"Meetings: {', '.join(meetings)}")
    print()

    print(f"{'metric':<25} {cfg['prod_label']:>25} {cfg['shadow_label']:>25}")
    print('-' * 80)

    p_bets = sum(1 for r in rows if r['prod_pnl'] != '')
    s_bets = sum(1 for r in rows if r['shadow_pnl'] != '')
    p_hits = sum(int(r['prod_hit']) for r in rows if r['prod_pnl'] != '')
    s_hits = sum(int(r['shadow_hit']) for r in rows if r['shadow_pnl'] != '')
    p_stake = sum(float(r['prod_stake'] or 0) for r in rows)
    s_stake = sum(float(r['shadow_stake'] or 0) for r in rows)
    p_pnl = sum(float(r['prod_pnl']) for r in rows if r['prod_pnl'] != '')
    s_pnl = sum(float(r['shadow_pnl']) for r in rows if r['shadow_pnl'] != '')
    p_roi = (p_pnl / p_stake * 100) if p_stake else 0
    s_roi = (s_pnl / s_stake * 100) if s_stake else 0

    print(f"{'Bets':<25} {p_bets:>25} {s_bets:>25}")
    print(f"{'Hits':<25} {p_hits:>25} {s_hits:>25}")
    print(f"{'Hit rate':<25} {p_hits/p_bets*100 if p_bets else 0:>24.1f}% {s_hits/s_bets*100 if s_bets else 0:>24.1f}%")
    print(f"{'Stake':<25} {f'${p_stake:.0f}':>25} {f'${s_stake:.0f}':>25}")
    print(f"{'P&L':<25} {f'{p_pnl:+.1f}u':>25} {f'{s_pnl:+.1f}u':>25}")
    print(f"{'ROI':<25} {f'{p_roi:+.1f}%':>25} {f'{s_roi:+.1f}%':>25}")
    print()
    print(f"Delta (shadow - production):  P&L {s_pnl-p_pnl:+.1f}u  |  ROI {s_roi-p_roi:+.1f}pp")
    print()

    # Bayesian Beta-Binomial posterior, gated by the MIN_MEETINGS floor.
    # NOT optional-stopping safe: cumulative-through-meeting outcomes are
    # autocorrelated, so the posterior overstates confidence at small N
    # (see forward_common.bayesian_verdict NOTE). The floor is the guard.
    winners = _per_meeting_winners(rows)
    v = bayesian_verdict(winners)
    print(f"  Per-meeting outcomes (1=shadow cum ROI > prod): {winners}")
    print(f"  Bayesian posterior P(shadow > prod): {v['posterior']:.3f}  "
          f"({v['wins']}/{v['n_meetings']} meetings)")
    if v['verdict'] == 'SHIP':
        print(f"  → SHIP. {cfg['ship_hint']}  (posterior >= 0.95)")
    elif v['verdict'] == 'KILL':
        print(f"  → KILL. {cfg['keep_hint']}  (posterior <= 0.05)")
    elif v['n_meetings'] < MIN_MEETINGS:
        print(f"  → WATCH. Only {v['n_meetings']}/{MIN_MEETINGS} meetings — below the "
              f"floor for a SHIP/KILL verdict regardless of posterior "
              f"(posterior={v['posterior']:.3f}; cumulative-ROI outcomes are "
              f"autocorrelated so small-N posteriors overstate confidence).")
    else:
        print(f"  → WATCH. Posterior is in [0.05, 0.95] — keep accruing meetings.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('race_date', nargs='?', help='YYYY/MM/DD (omit if using --report)')
    p.add_argument('--variant', choices=list(VARIANTS), default='rule_w',
                   help='Which shadow experiment to evaluate (default: rule_w)')
    p.add_argument('--report', action='store_true',
                   help='Print cumulative comparison from the variant log')
    args = p.parse_args()
    if args.report:
        return report(args.variant)
    if not args.race_date:
        p.error('race_date required unless --report')
    return evaluate_date(args.race_date, args.variant)


if __name__ == '__main__':
    sys.exit(main())
