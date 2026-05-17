"""Post-race one-command wrapper: results ingest + shadow eval + DB dump.

Runs in sequence (each step depends on the previous):
  1. python3 results_fetcher.py <date> --auto-learn — ingest finishes,
     retrain ML, recalibrate, refresh dashboard with hit/miss badges
  2. python3 -m scripts.shadow_evaluate <date>      — compare prod vs shadow
     vs actual, append to shadow_log.csv
  3. python3 -m scripts.tier_b_evaluate <date>      — score OLD vs NEW
     recommender for Tier B forward-validation
  4. bash scripts/auto_dump.sh                      — refresh DB snapshot,
     commit + push if bytes changed

Skips shadow step if results/shadow_race_data_<date>.json is missing.
Skips tier B eval if tracking_data/tier_b_log.csv has no rows for date.

Usage:
    python3 -m scripts.post_race '2026/05/17'
    python3 -m scripts.post_race '2026/05/17' --skip-dump  # leave DB dump for later
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, cmd: list[str], required: bool = True) -> int:
    print(f"\n{'='*70}\n  {label}\n  $ {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.monotonic()
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.monotonic() - t0
    status = "OK" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
    print(f"\n  [{label}] {status} in {elapsed:.1f}s", flush=True)
    return r.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('race_date', help='YYYY/MM/DD')
    p.add_argument('--skip-dump', action='store_true',
                   help='Skip the DB dump step (run scripts/auto_dump.sh later)')
    args = p.parse_args()

    rc = _run("1/3 results ingest + auto-learn",
              [sys.executable, 'results_fetcher.py', args.race_date, '--auto-learn'])
    if rc != 0:
        print(f"\nABORT: results_fetcher failed. Fix and rerun.", file=sys.stderr)
        return rc

    # Step 2: shadow eval — skip cleanly if shadow file missing
    date_str = args.race_date.replace('/', '')
    shadow_path = REPO_ROOT / 'results' / f'shadow_race_data_{date_str}.json'
    if not shadow_path.exists():
        print(f"\n  [2/3 shadow_evaluate] SKIPPED — no shadow file at {shadow_path}")
    else:
        rc = _run("2/3 shadow_evaluate",
                  [sys.executable, '-m', 'scripts.shadow_evaluate', args.race_date])
        if rc != 0:
            print(f"\nWARN: shadow_evaluate failed but results were ingested. "
                  f"Continuing to dump step.", file=sys.stderr)

    # Step 3: tier B eval — always safe to run (skips internally if no rows)
    _run("3/4 tier B evaluate (OLD vs NEW recommender)",
         [sys.executable, '-m', 'scripts.tier_b_evaluate', args.race_date])

    if args.skip_dump:
        print(f"\n  [4/4 DB dump] SKIPPED — run `bash scripts/auto_dump.sh` later")
    else:
        rc = _run("4/4 DB dump + commit + push",
                  ['bash', 'scripts/auto_dump.sh'])
        if rc != 0:
            print(f"\nWARN: auto_dump exited non-zero. Check logs/auto_dump.log.",
                  file=sys.stderr)

    print(f"\n{'='*70}\n  POST-RACE COMPLETE — {args.race_date}\n{'='*70}")
    print(f"  Cumulative shadow comparison: python3 -m scripts.shadow_evaluate --report")
    print(f"  Cumulative tier B forward:    python3 -m scripts.tier_b_evaluate --report")
    return 0


if __name__ == '__main__':
    sys.exit(main())
