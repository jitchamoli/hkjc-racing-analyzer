#!/usr/bin/env python3
"""Prediction accuracy report — prints longitudinal stats from prediction_log.

Usage:
    python3 -m scripts.accuracy              # Last 30 days
    python3 -m scripts.accuracy --days 90    # Last 90 days
    python3 -m scripts.accuracy --all        # All time
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import HorseDatabase


def print_summary(summary: dict):
    print("\n" + "=" * 60)
    period = "All time" if summary['period_days'] >= 9999 else f"Last {summary['period_days']} days"
    print(f"  PREDICTION ACCURACY — {period}")
    print("=" * 60)

    print(f"\n  Races tracked:       {summary['total_races']}")
    print(f"  #1 Pick Win Rate:    {summary['rank1_win_rate']}% ({summary['rank1_wins']}/{summary['total_races']})")
    print(f"  Top-3 Overlap:       {summary['top3_overlap_rate']}%")
    print(f"  Avg Winner Rank:     {summary['avg_winner_rank']}")

    print(f"\n  --- Betting Performance ---")
    print(f"  Bets Placed:         {summary['bets_placed']}")
    print(f"  Bets Won:            {summary['bets_won']}")
    print(f"  Hit Rate:            {summary['bet_hit_rate']}%")
    print(f"  Total P&L:           ${summary['total_profit']:+.0f}")
    print(f"  ROI:                 {summary['roi_percent']:+.1f}%")

    if summary['banker_count']:
        print(f"\n  --- BANKER Picks ---")
        print(f"  Total:               {summary['banker_count']}")
        print(f"  Wins:                {summary['banker_wins']}")
        print(f"  Hit Rate:            {summary['banker_hit_rate']}%")

    if summary.get('by_confidence'):
        print(f"\n  --- By Confidence Level ---")
        for conf, data in sorted(summary['by_confidence'].items()):
            if data['total'] > 0:
                rate = round(data['wins'] / data['total'] * 100, 1)
                print(f"  {conf:20s} {data['wins']}/{data['total']} ({rate}%)")


def print_class_breakdown(classes: list):
    if not classes:
        return
    print(f"\n  --- Accuracy by Class ---")
    print(f"  {'Class':<15} {'Races':>6} {'#1 Win%':>8} {'Top3%':>7} {'P&L':>8}")
    print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*7} {'-'*8}")
    for c in classes:
        print(f"  {c['class']:<15} {c['races']:>6} {c['rank1_win_rate']:>7.1f}% {c['top3_hit_rate']:>6.1f}% ${c['profit']:>+7.0f}")


def print_bet_type_roi(bets: list):
    if not bets:
        return
    print(f"\n  --- ROI by Bet Type ---")
    print(f"  {'Type':<15} {'Count':>6} {'Hits':>5} {'Hit%':>6} {'P&L':>8} {'ROI':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*5} {'-'*6} {'-'*8} {'-'*7}")
    for b in bets:
        print(f"  {b['bet_type']:<15} {b['count']:>6} {b['hits']:>5} {b['hit_rate']:>5.1f}% ${b['profit']:>+7.0f} {b['roi']:>+6.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Prediction accuracy report')
    parser.add_argument('--days', type=int, default=30, help='Lookback period in days (default: 30)')
    parser.add_argument('--all', action='store_true', help='Show all-time stats')
    args = parser.parse_args()

    days = 9999 if args.all else args.days

    db = HorseDatabase()

    summary = db.get_accuracy_summary(days)
    if 'message' in summary:
        print(f"\n  {summary['message']}")
        print("  Run results_fetcher.py after races to populate accuracy data.")
        db.close()
        return

    print_summary(summary)
    print_class_breakdown(db.get_accuracy_by_class(days))
    print_bet_type_roi(db.get_bet_type_roi(days))

    print("\n" + "=" * 60)
    db.close()


if __name__ == '__main__':
    main()
