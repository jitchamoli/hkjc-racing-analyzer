#!/usr/bin/env python3
"""CLI: bankroll-guard daily status for a meeting (stop-loss / cap check).

Usage:
    python3 -m scripts.bankroll_status 2026/06/03
    python3 -m scripts.bankroll_status 2026/06/03 --bankroll 1000

Recomputes the day's realised P&L from the frozen bet tickets + results and
reports whether the daily loss / exposure caps have been hit. Read-only.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from betting.bankroll_guard import daily_status


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("date", help="Meeting date YYYY/MM/DD")
    p.add_argument("--bankroll", type=float, default=config.BANKROLL,
                   help=f"Rolling bankroll in HKD (default config.BANKROLL={config.BANKROLL})")
    args = p.parse_args()

    st = daily_status(args.date, bankroll=args.bankroll)
    print(st.banner())
    print(f"  bankroll ${args.bankroll:.0f}  |  "
          f"per-race cap ${args.bankroll * config.MAX_STAKE_PER_RACE_PCT:.0f} "
          f"({config.MAX_STAKE_PER_RACE_PCT * 100:.0f}%)  |  "
          f"exposure cap ${args.bankroll * config.DAILY_EXPOSURE_CAP_PCT:.0f} "
          f"({config.DAILY_EXPOSURE_CAP_PCT * 100:.0f}%)  |  "
          f"loss cap ${args.bankroll * config.DAILY_LOSS_CAP_PCT:.0f} "
          f"({config.DAILY_LOSS_CAP_PCT * 100:.0f}%)")
    if st.halted:
        print("  >>> NO MORE BETS TODAY — daily cap reached <<<")
    return 0


if __name__ == "__main__":
    sys.exit(main())
