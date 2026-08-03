"""Full-DB CHAOS diagnostic — 5 years of history, ~495 meetings.

Bypasses the JSON cache (which only covers 10 recent meetings) and queries
race_results directly. Simulates flat-$10 WIN bets on the market favorite
to proxy "what a bet on this race would have done" without needing model
predictions for old races.

Limitations vs the JSON-cache diagnostic:
- Bets the MARKET favorite, not the model's pick. Reasonable proxy since
  production blends ~50/50 model+market, but it misses the model's edge.
- Only the WIN pool is simulated. No QIN/TRIO/F4 dividend data goes back
  5 years in our race_dividends table.

What this CAN tell us:
- Does soft-going underperform firm-going on plain favorite-WIN bets
  across 5 years? If YES across 50+ wet meetings, the angle is robust.
- It can't validate the model's CHAOS interaction because we don't have
  historical model picks.

Usage:
    python3 -m scripts.chaos_diagnostic_db
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from scripts.forward_common import bayesian_verdict  # noqa: E402


# HKJC uses short-form going codes in the legacy DB (~75% of rows) and
# long-form in recent JSON-fed rows. Normalize so the shipped substring
# detector in betting/conviction.py catches both.
GOING_NORMALIZE = {
    'G': 'GOOD', 'GD': 'GOOD',
    'GF': 'GOOD TO FIRM', 'FT': 'FAST',
    'GY': 'GOOD TO YIELDING',
    'Y': 'YIELDING', 'YS': 'YIELDING TO SOFT',
    'S': 'SOFT', 'H': 'HEAVY',
    'WS': 'WET SLOW', 'WF': 'WET FAST',
    'RA': 'RAIN AFFECTED',  # → contains 'RAIN' — see SOFT_KEYWORDS_DB
    'SE': 'SEALED',
}

# Augmented soft-side keywords for the DB sweep (RAIN added — Rain Affected
# is a HKJC label that the shipped filter doesn't currently catch since it
# never appears in live race-card JSON).
SOFT_KEYWORDS_DB = ('YIELDING', 'SOFT', 'HEAVY', 'WET-SLOW', 'WET SLOW',
                    'WET ', 'RAIN')


def normalize_going(g: str) -> str:
    g = (g or '').strip()
    if not g:
        return ''
    return GOING_NORMALIZE.get(g, g).upper()


def is_chaos_db(going: str) -> bool:
    """Soft-going detection — DELEGATES to the shipped filter.

    This used to keep its own substring match, which fired on
    'Good To Yielding' (and GY) and on 'Wet-Fast'. Production `detect_chaos`
    excludes both — GY was deliberately narrowed out on 2026-07-15 because the
    favourite strikes ~31% on it, indistinguishable from firm. So the
    diagnostic was measuring a strictly WIDER filter than the one deployed,
    and the "~-7pp expected lift" figure in CLAUDE.md came from this script.

    A diagnostic that disagrees with production can't inform a decision about
    production, so it now calls the same function. Use `is_chaos_db_augmented`
    if you specifically want the exploratory wider sweep.
    """
    from betting.conviction import detect_chaos
    return bool(detect_chaos(normalize_going(going))[0])


def is_chaos_db_augmented(going: str) -> bool:
    """Exploratory WIDER sweep — NOT what production ships.

    Adds RAIN / WET labels that never appear in live race-card JSON. Kept for
    hypothesis generation only; never quote its numbers as the shipped
    filter's effect.
    """
    g = normalize_going(going)
    return any(kw in g for kw in SOFT_KEYWORDS_DB)


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    # All races with at least one finisher + market odds + going populated
    races = conn.execute("""
        SELECT race_date, race_number, going, class_type
        FROM race_results
        WHERE finish_position > 0 AND going != '' AND going IS NOT NULL
        GROUP BY race_date, race_number
        ORDER BY race_date, race_number
    """).fetchall()

    print(f"Total races to evaluate: {len(races):,} "
          f"across {len({r['race_date'] for r in races}):,} meetings")
    print()

    per_meeting = defaultdict(lambda: {
        'chaos_n': 0, 'chaos_stake': 0.0, 'chaos_pnl': 0.0,
        'clean_n': 0, 'clean_stake': 0.0, 'clean_pnl': 0.0,
    })
    going_flag_counts = defaultdict(int)
    going_total_counts = defaultdict(int)
    per_year = defaultdict(lambda: {
        'chaos_n': 0, 'chaos_stake': 0.0, 'chaos_pnl': 0.0,
        'clean_n': 0, 'clean_stake': 0.0, 'clean_pnl': 0.0,
    })

    STAKE = 10.0  # HK$10 per race on the favorite

    for race in races:
        date = race['race_date']
        rn = race['race_number']
        going = race['going']
        year = date[:4]

        is_chaos = is_chaos_db(going)
        going_total_counts[going] += 1
        if is_chaos:
            going_flag_counts[going] += 1

        # Pick the market favorite — lowest win_odds, must have odds + finish
        horses = conn.execute("""
            SELECT horse_name, win_odds, is_winner, finish_position
            FROM race_results
            WHERE race_date = ? AND race_number = ? AND win_odds > 0
              AND finish_position > 0
        """, (date, rn)).fetchall()
        if not horses:
            continue
        favorite = min(horses, key=lambda h: h['win_odds'])
        won = bool(favorite['is_winner'])
        odds = favorite['win_odds']
        pnl = STAKE * (odds - 1.0) if won else -STAKE

        agg_m = per_meeting[date]
        agg_y = per_year[year]
        bucket = 'chaos' if is_chaos else 'clean'
        for agg in (agg_m, agg_y):
            agg[f'{bucket}_n'] += 1
            agg[f'{bucket}_stake'] += STAKE
            agg[f'{bucket}_pnl'] += pnl

    conn.close()

    # Per-year summary
    print("=" * 96)
    print("PER-YEAR — flat HK$10 on market favorite, WIN pool only")
    print("=" * 96)
    print(f"{'Year':<6} {'CHAOS races':>12} {'C-Stake':>10} {'C-PnL':>10} "
          f"{'C-ROI':>8} | {'Clean races':>12} {'Cl-Stake':>10} "
          f"{'Cl-PnL':>10} {'Cl-ROI':>8}")
    print('-' * 110)
    cum_c_s = cum_c_p = cum_cl_s = cum_cl_p = 0.0
    for year in sorted(per_year.keys()):
        a = per_year[year]
        cum_c_s += a['chaos_stake']; cum_c_p += a['chaos_pnl']
        cum_cl_s += a['clean_stake']; cum_cl_p += a['clean_pnl']
        c_roi = (a['chaos_pnl'] / a['chaos_stake'] * 100) if a['chaos_stake'] else 0
        cl_roi = (a['clean_pnl'] / a['clean_stake'] * 100) if a['clean_stake'] else 0
        print(f"{year:<6} {a['chaos_n']:>12,} {a['chaos_stake']:>10,.0f} "
              f"{a['chaos_pnl']:>+10,.0f} {c_roi:>+7.1f}% | "
              f"{a['clean_n']:>12,} {a['clean_stake']:>10,.0f} "
              f"{a['clean_pnl']:>+10,.0f} {cl_roi:>+7.1f}%")
    print('-' * 110)
    cum_c_roi = (cum_c_p / cum_c_s * 100) if cum_c_s else 0
    cum_cl_roi = (cum_cl_p / cum_cl_s * 100) if cum_cl_s else 0
    total_chaos_n = sum(a['chaos_n'] for a in per_year.values())
    total_clean_n = sum(a['clean_n'] for a in per_year.values())
    print(f"{'TOTAL':<6} {total_chaos_n:>12,} {cum_c_s:>10,.0f} "
          f"{cum_c_p:>+10,.0f} {cum_c_roi:>+7.1f}% | "
          f"{total_clean_n:>12,} {cum_cl_s:>10,.0f} "
          f"{cum_cl_p:>+10,.0f} {cum_cl_roi:>+7.1f}%")
    print()

    # Going string breakdown — which labels actually flagged
    print("=" * 96)
    print("GOING-LABEL BREAKDOWN — which strings flagged CHAOS")
    print("=" * 96)
    flagged = [(g, n) for g, n in going_flag_counts.items() if n > 0]
    for g, n in sorted(flagged, key=lambda kv: -kv[1]):
        total = going_total_counts[g]
        print(f"  {g!r:<25} {n:>5,} flagged of {total:>5,} total "
              f"(normalized → {normalize_going(g)!r})")
    if not flagged:
        print("  (no soft-going strings flagged — check normalization map)")
    print()

    # Bayesian per-meeting verdict
    chaos_meeting_wins = []  # 1 if CHAOS cumulative ROI < Clean cumulative ROI
    run_c_s = run_c_p = run_cl_s = run_cl_p = 0.0
    n_meetings_with_both = 0
    for date in sorted(per_meeting.keys()):
        a = per_meeting[date]
        if a['chaos_stake'] > 0 and a['clean_stake'] > 0:
            run_c_s += a['chaos_stake']; run_c_p += a['chaos_pnl']
            run_cl_s += a['clean_stake']; run_cl_p += a['clean_pnl']
            c_roi_run = run_c_p / run_c_s if run_c_s else 0
            cl_roi_run = run_cl_p / run_cl_s if run_cl_s else 0
            chaos_meeting_wins.append(1 if c_roi_run < cl_roi_run else 0)
            n_meetings_with_both += 1

    verdict = bayesian_verdict(chaos_meeting_wins)
    print("=" * 96)
    print("BAYESIAN PER-MEETING VERDICT (only meetings with BOTH CHAOS + Clean races)")
    print("=" * 96)
    print(f"  N qualifying meetings: {verdict['n_meetings']:,}")
    print(f"  Meetings where CHAOS cumulative ROI < Clean cumulative: "
          f"{verdict['wins']:,}/{verdict['n_meetings']:,}")
    print(f"  Posterior P(CHAOS < Clean): {verdict['posterior']:.4f}")
    print(f"  Bayesian verdict: {verdict['verdict']}")
    print()

    # Decision
    delta_pp = cum_c_roi - cum_cl_roi
    print("=" * 96)
    print("DECISION (full 5-year history, market favorite proxy)")
    print("=" * 96)
    print(f"  CHAOS races: {total_chaos_n:,} bets, ROI {cum_c_roi:+.2f}%")
    print(f"  Clean races: {total_clean_n:,} bets, ROI {cum_cl_roi:+.2f}%")
    print(f"  Δ:           {delta_pp:+.2f}pp")
    print()
    if verdict['posterior'] >= 0.80 and delta_pp <= -2.0:
        print("  → CHAOS angle ROBUST on 5-year favorite-bet history.")
        print("    The soft-going effect survives outside the 10-meeting sample.")
    elif verdict['posterior'] < 0.50:
        print("  → CHAOS angle WEAK on 5-year history.")
        print("    The 10-meeting result may be sample-specific.")
    else:
        print("  → CHAOS angle MARGINAL on 5-year history. Hold; let forward "
              "evidence accumulate.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
