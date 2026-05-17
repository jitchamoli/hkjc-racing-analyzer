"""Backtest a candidate swap rule across historical meetings.

Process gate built after 2026-05-17 rank-5 swap incident: shipped on N=1
retrospective, failed first forward test, reverted same race. The rule
'do not ship a structural change without ≥6-meeting backtest' is now
testable from the CLI before any swap proposal hits production.

Currently supports rank-N swap rules:
  - "Replace the rank-N horse in TRIO-5/TRIO-6 box when a lower-ranked
     horse meets odds + gear criteria"

Usage:
    python3 scripts/eval_swap_rule.py \
        --rank-out 5 \
        --odds-ratio 0.5 \
        --require-gear \
        --bet trio5

Output: per-meeting + cumulative hit rate WITH vs WITHOUT swap.

Verdict logic (see [[n6_validation_rule]] memory):
  - REGRESSION if WITH-swap hit rate < WITHOUT-swap by >5pp
  - INSUFFICIENT_DATA if <6 meetings have fires
  - CANDIDATE if WITH > WITHOUT by ≥5pp AND ≥6 meetings of fires
  - NEUTRAL otherwise — keep collecting
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_meetings():
    """Load all race_data_*_with_results.json files (only finished meetings)."""
    pattern = str(REPO_ROOT / "results" / "race_data_*_with_results.json")
    meetings = []
    for p in sorted(glob.glob(pattern)):
        try:
            data = json.loads(Path(p).read_text())
            races = data.get('races') if isinstance(data, dict) else data
            if races and any(r.get('results_available') for r in races):
                # Normalize date from filename: race_data_YYYYMMDD_with_results.json
                date_str = Path(p).stem.replace('race_data_', '').replace('_with_results', '')
                meetings.append((date_str, races))
        except (json.JSONDecodeError, IOError):
            continue
    return meetings


def _apply_rank_swap_rule(horses_sorted: list, rank_out: int,
                          odds_ratio: float, require_gear: bool,
                          max_scan_rank: int = 12) -> dict | None:
    """Apply a rank-N swap rule to one race.

    Returns dict {fired: bool, swap_in_num, swap_out_num, new_box_nums}
    where new_box_nums is the top-N picks with the swap applied.
    """
    if len(horses_sorted) < rank_out:
        return None
    rank_out_horse = horses_sorted[rank_out - 1]
    rank_out_odds = rank_out_horse.get('win_odds') or 0
    if rank_out_odds <= 0:
        return None
    threshold = rank_out_odds * odds_ratio
    candidate = None
    for h in horses_sorted[rank_out:max_scan_rank]:
        odds = h.get('win_odds') or 0
        gear = (h.get('gear') or '').strip()
        if odds <= 0 or odds > threshold:
            continue
        if require_gear and not gear:
            continue
        if candidate is None or odds < (candidate.get('win_odds') or 0):
            candidate = h
    if not candidate:
        return None
    return {
        'fired': True,
        'swap_in_num': candidate['number'],
        'swap_out_num': rank_out_horse['number'],
    }


def _box_hit(box_nums: list[int], actual_top3: list[int]) -> bool:
    return bool(actual_top3) and all(n in box_nums for n in actual_top3)


def _evaluate(args):
    meetings = _load_meetings()
    if not meetings:
        print("No completed meetings found in results/")
        return 1

    box_size = 5 if args.bet == 'trio5' else 6
    print(f"\nRule: rank-{args.rank_out} swap | odds ≤ {args.odds_ratio*100:.0f}% "
          f"of rank-{args.rank_out} | gear={'required' if args.require_gear else 'optional'} "
          f"| measuring BOX TRIO-{box_size}")
    print('=' * 80)

    per_meeting = []
    total_fires = 0
    total_with_swap_hits = 0
    total_without_swap_hits = 0

    for date_str, races in meetings:
        m_fires = 0
        m_with = 0
        m_without = 0
        for r in races:
            if not r.get('results_available'):
                continue
            horses_sorted = sorted(r['horses'], key=lambda h: h.get('rank') or 99)
            if len(horses_sorted) < box_size:
                continue
            actual_top3 = r.get('actual_top3') or []
            if not actual_top3:
                continue

            # Baseline: top-N box WITHOUT swap
            default_box = [h['number'] for h in horses_sorted[:box_size]]
            without_hit = _box_hit(default_box, actual_top3)

            # Apply swap rule
            swap = _apply_rank_swap_rule(horses_sorted, args.rank_out,
                                          args.odds_ratio, args.require_gear)
            if not swap:
                continue  # rule didn't fire — not a candidate race

            m_fires += 1
            total_fires += 1
            if without_hit:
                m_without += 1
                total_without_swap_hits += 1

            # With swap: replace rank_out horse
            with_box = default_box[:args.rank_out - 1] + [swap['swap_in_num']]
            if box_size > args.rank_out:
                # Keep the rest of the default box after the swapped position
                with_box += default_box[args.rank_out:]
            with_hit = _box_hit(with_box, actual_top3)
            if with_hit:
                m_with += 1
                total_with_swap_hits += 1

        if m_fires > 0:
            per_meeting.append((date_str, m_fires, m_with, m_without))
            print(f"  {date_str}: {m_fires:>2} fires | WITH swap: {m_with}/{m_fires} | "
                  f"WITHOUT swap: {m_without}/{m_fires}")

    print('=' * 80)
    n_meetings = len(per_meeting)
    print(f"\nMeetings with at least 1 candidate fire: {n_meetings}")
    print(f"Total candidate fires: {total_fires}")

    if total_fires == 0:
        print("\nNo races where the rule fired — try different parameters.")
        return 0

    with_pct = total_with_swap_hits / total_fires * 100
    without_pct = total_without_swap_hits / total_fires * 100
    delta_pp = with_pct - without_pct

    print(f"\nCumulative TRIO-{box_size} hit rate on candidate races:")
    print(f"  WITH swap:    {total_with_swap_hits}/{total_fires} = {with_pct:.1f}%")
    print(f"  WITHOUT swap: {total_without_swap_hits}/{total_fires} = {without_pct:.1f}%")
    print(f"  Delta:        {delta_pp:+.1f}pp (positive = swap helps)")

    print(f"\n--- Verdict ---")
    if n_meetings < 6:
        print(f"⚠️  INSUFFICIENT_DATA — only {n_meetings}/6 meetings of fires.")
        print(f"   Per [[n6_validation_rule]], do not ship.")
    elif delta_pp < -5.0:
        print(f"❌ REGRESSION — swap loses {-delta_pp:.1f}pp hit rate. DO NOT SHIP.")
    elif delta_pp >= 5.0:
        print(f"✅ CANDIDATE — swap gains {delta_pp:.1f}pp on {n_meetings} meetings.")
        print(f"   Consider shipping, but ALSO run a forward-validation period.")
    else:
        print(f"⚖️  NEUTRAL — swap delta {delta_pp:+.1f}pp within noise. Keep collecting.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rank-out', type=int, default=5,
                   help="The rank slot the swap REMOVES (default: 5)")
    p.add_argument('--odds-ratio', type=float, default=0.5,
                   help="Swap-in horse odds ≤ this ratio of rank-N odds (default: 0.5)")
    p.add_argument('--require-gear', action='store_true',
                   help="Require swap-in horse to have gear visible")
    p.add_argument('--bet', choices=['trio5', 'trio6'], default='trio5',
                   help="Which bet to measure hit rate for (default: trio5)")
    return _evaluate(p.parse_args())


if __name__ == '__main__':
    sys.exit(main())
