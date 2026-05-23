"""CHAOS no-bet filter — diagnostic-first per project pattern.

Detects races where the field is too chaotic for any model signal to be
trusted (soft going, ≥4 horses returning from long breaks, Class 5 with
compact top-3 market). Compares ROI on CHAOS-flagged races vs non-CHAOS
races over all available meetings.

Decision rule (matches the project's existing kill-or-ship pattern):
  * Bayesian posterior P(CHAOS < non-CHAOS ROI by meeting) > 0.80
    AND CHAOS-races ROI ≥ 10pp worse than non-CHAOS → SHIP
  * Posterior < 0.50 → KILL (no evidence the angle exists in our data)
  * Between → SHADOW only

Usage:
    python3 -m scripts.chaos_diagnostic
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from scripts.stake_ab import firing_bets, hit_and_dividend, _pnl  # noqa: E402
from scripts.forward_common import bayesian_verdict  # noqa: E402


SOFT_GOING_KEYWORDS = ('YIELDING', 'SOFT', 'HEAVY', 'WET-SLOW', 'WET SLOW')
LONG_BREAK_DAYS = 60
LONG_BREAK_MIN_COUNT = 4
COMPACT_MARKET_SPREAD = 2.0  # HK$
COMPACT_MARKET_MIN_FIELD = 10


def detect_chaos(race: dict) -> tuple[bool, str]:
    """Returns (is_chaos, trigger_label).
    Triggers OR'd; first match wins for the label."""
    horses = race.get('horses') or []
    going = (race.get('going') or '').upper()
    cls = (race.get('class_type') or '').upper()
    field = len(horses)

    if any(kw in going for kw in SOFT_GOING_KEYWORDS):
        return True, f"soft_going"

    long_breaks = sum(
        1 for h in horses if (h.get('days_since_last_run') or 0) > LONG_BREAK_DAYS
    )
    if long_breaks >= LONG_BREAK_MIN_COUNT:
        return True, f"breaks_{long_breaks}_horses_60d+"

    if 'CLASS 5' in cls and field >= COMPACT_MARKET_MIN_FIELD:
        odds = sorted([
            h.get('win_odds') or 0 for h in horses if (h.get('win_odds') or 0) > 0
        ])
        if len(odds) >= 3 and odds[2] - odds[0] < COMPACT_MARKET_SPREAD:
            return True, f"c5_compact_market"

    return False, ""


def evaluate_race(race: dict, conn, race_date: str,
                  stake_per_combo: float = 10.0) -> tuple[float, float, int]:
    """Returns (total_stake, total_pnl, n_bets_fired) for this race at flat
    $10/combo (the user's actual habit per memory user_stake_habit_10hkd_minimum)."""
    if not race.get('results_available'):
        return 0.0, 0.0, 0
    fires = firing_bets(race)
    if not fires:
        return 0.0, 0.0, 0
    total_stake = 0.0
    total_pnl = 0.0
    for bet_label, combos in fires.items():
        stake = stake_per_combo * combos
        hit, div = hit_and_dividend(bet_label, race, conn, race_date)
        pnl = _pnl(stake, combos, hit, div)
        total_stake += stake
        total_pnl += pnl
    return total_stake, total_pnl, len(fires)


def main() -> int:
    files = sorted((REPO_ROOT / 'results').glob('race_data_*_with_results.json'))
    if not files:
        print("No meeting result files found.", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(config.DB_PATH))

    # Per-meeting aggregates
    per_meeting = defaultdict(lambda: {
        'chaos_stake': 0.0, 'chaos_pnl': 0.0, 'chaos_races': 0, 'chaos_bets': 0,
        'clean_stake': 0.0, 'clean_pnl': 0.0, 'clean_races': 0, 'clean_bets': 0,
    })
    trigger_counts = defaultdict(int)
    trigger_details = []

    for fp in files:
        race_date = fp.stem.replace('race_data_', '').replace('_with_results', '')
        race_date_fmt = f"{race_date[:4]}/{race_date[4:6]}/{race_date[6:]}"

        try:
            payload = json.loads(fp.read_text())
        except json.JSONDecodeError as e:
            print(f"  skip {race_date}: {e}", file=sys.stderr)
            continue
        races = payload['races'] if isinstance(payload, dict) else payload

        for race in races:
            is_chaos, trigger = detect_chaos(race)
            stake, pnl, n_bets = evaluate_race(race, conn, race_date_fmt)
            if stake == 0:
                continue
            agg = per_meeting[race_date_fmt]
            if is_chaos:
                agg['chaos_stake'] += stake
                agg['chaos_pnl'] += pnl
                agg['chaos_races'] += 1
                agg['chaos_bets'] += n_bets
                trigger_counts[trigger] += 1
                trigger_details.append({
                    'date': race_date_fmt, 'race': race.get('race_number'),
                    'trigger': trigger, 'stake': stake, 'pnl': pnl,
                })
            else:
                agg['clean_stake'] += stake
                agg['clean_pnl'] += pnl
                agg['clean_races'] += 1
                agg['clean_bets'] += n_bets

    conn.close()

    if not per_meeting:
        print("No usable races.", file=sys.stderr)
        return 1

    # Print per-meeting + cumulative
    print("=" * 78)
    print("CHAOS DIAGNOSTIC — per meeting (flat HK$10/combo)")
    print("=" * 78)
    print(f"{'Date':<12} {'CHAOS races':>11} {'C-Stake':>9} {'C-PnL':>9} {'C-ROI':>8} | "
          f"{'Clean races':>11} {'Cl-Stake':>9} {'Cl-PnL':>9} {'Cl-ROI':>8}")
    print('-' * 110)

    chaos_meeting_wins = []  # for Bayesian — 1 if CHAOS ROI < Clean ROI
    cum_chaos_stake = cum_chaos_pnl = 0.0
    cum_clean_stake = cum_clean_pnl = 0.0

    for date in sorted(per_meeting.keys()):
        a = per_meeting[date]
        cum_chaos_stake += a['chaos_stake']
        cum_chaos_pnl += a['chaos_pnl']
        cum_clean_stake += a['clean_stake']
        cum_clean_pnl += a['clean_pnl']
        c_roi = (a['chaos_pnl'] / a['chaos_stake'] * 100) if a['chaos_stake'] else 0
        cl_roi = (a['clean_pnl'] / a['clean_stake'] * 100) if a['clean_stake'] else 0
        cum_c_roi = (cum_chaos_pnl / cum_chaos_stake) if cum_chaos_stake else 0
        cum_cl_roi = (cum_clean_pnl / cum_clean_stake) if cum_clean_stake else 0
        # Per-meeting win: cumulative CHAOS ROI < cumulative Clean ROI
        if cum_chaos_stake > 0 and cum_clean_stake > 0:
            chaos_meeting_wins.append(1 if cum_c_roi < cum_cl_roi else 0)
        print(f"{date:<12} {a['chaos_races']:>11} "
              f"{a['chaos_stake']:>9.0f} {a['chaos_pnl']:>+9.0f} {c_roi:>+7.1f}% | "
              f"{a['clean_races']:>11} "
              f"{a['clean_stake']:>9.0f} {a['clean_pnl']:>+9.0f} {cl_roi:>+7.1f}%")

    print('-' * 110)
    total_chaos_races = sum(a['chaos_races'] for a in per_meeting.values())
    total_clean_races = sum(a['clean_races'] for a in per_meeting.values())
    chaos_roi = (cum_chaos_pnl / cum_chaos_stake * 100) if cum_chaos_stake else 0
    clean_roi = (cum_clean_pnl / cum_clean_stake * 100) if cum_clean_stake else 0
    delta_pp = chaos_roi - clean_roi
    print(f"{'CUMULATIVE':<12} {total_chaos_races:>11} "
          f"{cum_chaos_stake:>9.0f} {cum_chaos_pnl:>+9.0f} {chaos_roi:>+7.1f}% | "
          f"{total_clean_races:>11} "
          f"{cum_clean_stake:>9.0f} {cum_clean_pnl:>+9.0f} {clean_roi:>+7.1f}%")
    print()

    print("CHAOS trigger breakdown:")
    for trigger, n in sorted(trigger_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {trigger:<30} {n} races")
    print()

    # Bayesian verdict
    print("=" * 78)
    print("BAYESIAN VERDICT")
    print("=" * 78)
    verdict = bayesian_verdict(chaos_meeting_wins)
    print(f"  N meetings with CHAOS+Clean races: {verdict['n_meetings']}")
    print(f"  Meeting-wins where CHAOS ROI < Clean ROI: {verdict['wins']}")
    print(f"  Posterior P(CHAOS < Clean): {verdict['posterior']:.3f}")
    print(f"  Bayesian verdict: {verdict['verdict']}")
    print()

    # Decision
    print("=" * 78)
    print("DECISION")
    print("=" * 78)
    print(f"  CHAOS-races ROI:  {chaos_roi:+.1f}%  ({total_chaos_races} races, "
          f"stake ${cum_chaos_stake:.0f})")
    print(f"  Clean-races ROI:  {clean_roi:+.1f}%  ({total_clean_races} races, "
          f"stake ${cum_clean_stake:.0f})")
    print(f"  Δ (CHAOS − Clean): {delta_pp:+.1f}pp")
    print()
    if verdict['posterior'] >= 0.80 and delta_pp <= -10.0:
        print("  → SHIP — both gates pass (posterior ≥ 0.80 AND ≥ 10pp worse)")
    elif verdict['posterior'] < 0.50:
        print("  → KILL — posterior < 0.50, no evidence the angle holds")
    else:
        print("  → SHADOW — marginal; ship as forward-validation variant only")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
