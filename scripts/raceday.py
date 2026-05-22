"""Race-day phase controller — one tool for the whole meeting rhythm.

Encodes the fixed race-day sequence so steps (especially shadow logging) can
never be skipped by jumping straight to a live refresh. Thin dispatcher over
the existing entry points — no logic is duplicated here.

Phases:
  start   — run ONCE in the morning. Delegates to scripts.race_day:
            generate_predictions + shadow_generate (rule_w=0.9) + tier_b_shadow.
            Locks the forward-validation snapshots before the live loop starts.
  live    — run REPEATEDLY before/after each race. Chains generate_predictions
            (fresh odds) -> results_fetcher (merge results + strike badges).
            Safe to run anytime: results_fetcher no-ops cleanly pre-race.
  finish  — run ONCE after the last race. Delegates to scripts.post_race
            (results ingest + auto-learn + trackers + dashboard regen + DB dump).
            Extra flags (e.g. --skip-dump) are forwarded to post_race.

Usage:
    python3 -m scripts.raceday start  '2026/05/24'
    python3 -m scripts.raceday live   '2026/05/24'   # repeat through the card
    python3 -m scripts.raceday finish '2026/05/24'
    python3 -m scripts.raceday finish '2026/05/24' --skip-dump
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, cmd: list[str]) -> int:
    print(f"\n{'='*70}\n  {label}\n  $ {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.monotonic()
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.monotonic() - t0
    status = "OK" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
    print(f"\n  [{label}] {status} in {elapsed:.1f}s", flush=True)
    return r.returncode


def _start(date: str, _extra: list[str]) -> int:
    # race_day already bundles generate + shadow_generate + tier_b_shadow.
    return _run("START — predictions + shadow + tier-B snapshots",
                [sys.executable, '-m', 'scripts.race_day', date])


def _live(date: str, _extra: list[str]) -> int:
    rc = _run("LIVE 1/2 — fresh odds + dashboard",
              [sys.executable, 'generate_predictions.py', date])
    if rc != 0:
        print("\nABORT: prediction refresh failed. Skipping results merge.",
              file=sys.stderr)
        return rc
    # results_fetcher no-ops cleanly when no race has finished yet, so the same
    # `live` command is correct both before the first race and after each race.
    return _run("LIVE 2/2 — merge results + strike badges",
                [sys.executable, 'results_fetcher.py', date])


def _finish(date: str, extra: list[str]) -> int:
    return _run("FINISH — post_race (ingest + auto-learn + trackers + dump)",
                [sys.executable, '-m', 'scripts.post_race', date, *extra])


PHASES = {'start': _start, 'live': _live, 'finish': _finish}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('phase', choices=PHASES.keys(), help='race-day phase')
    p.add_argument('race_date', help='YYYY/MM/DD')
    args, extra = p.parse_known_args()

    rc = PHASES[args.phase](args.race_date, extra)

    if rc == 0:
        nxt = {
            'start': "Now run `raceday live` before/after each race through the card.",
            'live': "Re-run `raceday live` to refresh; `raceday finish` after the last race.",
            'finish': "Meeting closed. ROI for trackers: python3 -m scripts.backfill_dividends --last 1",
        }[args.phase]
        print(f"\n  {args.phase.upper()} OK — {nxt}")
    return rc


if __name__ == '__main__':
    sys.exit(main())
