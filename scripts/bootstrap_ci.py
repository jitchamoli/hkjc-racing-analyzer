"""Bootstrap 95% confidence intervals for backtest ROI.

The backtest already exports per-race P&L via `--pnl-json`. This script
consumes that file and reports a CI on the headline ROI numbers using
two bootstrap variants:

  * iid:           naive resample of races with replacement. Assumes
                   races are independent, which they aren't (same horse
                   in many races) — UNDER-states uncertainty.
  * block-by-date: resample whole race dates (meetings) with replacement.
                   Preserves within-meeting correlation (same field,
                   same going). More honest, typically wider CIs.

Usage:
  python3 -m scripts.backtest --pnl-json /tmp/pnl.json
  python3 -m scripts.bootstrap_ci /tmp/pnl.json
  python3 -m scripts.bootstrap_ci /tmp/pnl.json --strategy box4  # default: all
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STRATEGIES = [
    ("straightq", "Straight quinella"),
    ("box3",     "Box-3 quinella"),
    ("box4",     "Box-4 quinella"),
]

DEFAULT_N_BOOT = 5000


def _roi(stakes: np.ndarray, returns: np.ndarray) -> float:
    s = stakes.sum()
    if s <= 0:
        return float("nan")
    pnl = returns.sum() - s
    return 100.0 * pnl / s


def iid_bootstrap(stakes, returns, n_boot=DEFAULT_N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(stakes)
    rois = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rois[b] = _roi(stakes[idx], returns[idx])
    return rois


def block_bootstrap_by_date(rows, key, stake_key, return_key,
                             n_boot=DEFAULT_N_BOOT, seed=0):
    """Resample whole race dates (meetings) with replacement."""
    by_date = defaultdict(list)
    for r in rows:
        date = r["race_key"].split("-R")[0]
        by_date[date].append(r)
    dates = list(by_date.keys())
    rng = np.random.default_rng(seed)
    rois = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(dates), len(dates))
        chosen = [by_date[dates[i]] for i in picked]
        flat = [row for grp in chosen for row in grp]
        stakes = np.array([row[stake_key] for row in flat])
        returns = np.array([row[return_key] for row in flat])
        rois[b] = _roi(stakes, returns)
    return rois


def report(name: str, stakes: np.ndarray, returns: np.ndarray, rows, n_boot, seed):
    point = _roi(stakes, returns)
    if np.isnan(point):
        print(f"  {name:<22}  (no bets)")
        return
    iid = iid_bootstrap(stakes, returns, n_boot=n_boot, seed=seed)
    block = block_bootstrap_by_date(
        rows, key="race_key", stake_key=f"{name}_stake", return_key=f"{name}_return",
        n_boot=n_boot, seed=seed,
    )
    iid_lo, iid_hi = np.percentile(iid, [2.5, 97.5])
    blk_lo, blk_hi = np.percentile(block, [2.5, 97.5])
    print(f"  {name:<22}  point={point:+8.1f}%   "
          f"iid 95%=[{iid_lo:+8.1f}%, {iid_hi:+8.1f}%]   "
          f"by-date 95%=[{blk_lo:+8.1f}%, {blk_hi:+8.1f}%]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pnl_json", help="Path to per-race P&L JSON from `backtest --pnl-json`")
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                    help=f"Number of bootstrap resamples (default {DEFAULT_N_BOOT})")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.pnl_json)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1
    rows = json.loads(path.read_text())
    if not rows:
        print("ERROR: empty P&L file", file=sys.stderr)
        return 1

    print(f"Bootstrap CI ({args.n_boot:,} resamples, seed={args.seed})")
    print(f"  Loaded {len(rows):,} per-race P&L records from {path.name}")
    print(f"  by-date bootstrap preserves within-meeting correlation (honest CI).")
    print(f"  iid bootstrap assumes race independence (under-states uncertainty).\n")

    for key, label in STRATEGIES:
        stake_key = f"{key}_stake"
        return_key = f"{key}_return"
        stakes = np.array([r.get(stake_key, 0) for r in rows])
        returns = np.array([r.get(return_key, 0) for r in rows])
        if stakes.sum() <= 0:
            print(f"  {label:<22}  (no stakes recorded)")
            continue
        report(key, stakes, returns, rows, args.n_boot, args.seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
