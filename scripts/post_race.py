"""Post-race one-command wrapper: results ingest + analytics + DB dump.

Runs in sequence:
  1. results_fetcher.py <date> --auto-learn   — ingest finishes, retrain ML,
                                                refresh dashboard with badges
  2. scripts.shadow_evaluate <date>           — prod vs rule_w=0.9 shadow
  3. scripts.tier_b_evaluate <date>           — OLD vs NEW recommender
  4. scripts.track_spoilers <date>            — log outside-top-5 placers
  4a. scripts.track_longshot_trio <date>      — paper experiment: top-5 +
                                                steamed rank 6-8 longshot
                                                (NOT wired to dashboard/WA)
  5. scripts.track_edge <date>                — log EDGE fires + outcomes
                                                (feeds dashboard banner)
  6. bash scripts/auto_dump.sh                — refresh DB snapshot, push

Skips shadow step if results/shadow_race_data_<date>.json is missing.
Other steps are always-safe (skip internally if no relevant data).

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

    rc = _run("1/6 results ingest + auto-learn",
              [sys.executable, 'results_fetcher.py', args.race_date, '--auto-learn'])
    if rc != 0:
        print(f"\nABORT: results_fetcher failed. Fix and rerun.", file=sys.stderr)
        return rc

    # Step 2: shadow eval — skip cleanly if shadow file missing
    date_str = args.race_date.replace('/', '')
    shadow_path = REPO_ROOT / 'results' / f'shadow_race_data_{date_str}.json'
    if not shadow_path.exists():
        print(f"\n  [2/6 shadow_evaluate] SKIPPED — no shadow file at {shadow_path}")
    else:
        rc = _run("2/6 shadow_evaluate",
                  [sys.executable, '-m', 'scripts.shadow_evaluate', args.race_date])
        if rc != 0:
            print(f"\nWARN: shadow_evaluate failed but results were ingested. "
                  f"Continuing to remaining steps.", file=sys.stderr)

    # Step 3: tier B eval — always safe (skips internally if no rows)
    _run("3/6 tier B evaluate (OLD vs NEW recommender)",
         [sys.executable, '-m', 'scripts.tier_b_evaluate', args.race_date])

    # Step 4: spoiler tracker — logs rank>5 horses that placed top-3
    _run("4/6 track_spoilers",
         [sys.executable, '-m', 'scripts.track_spoilers', args.race_date])

    # Step 4a: longshot-TRIO paper experiment — accrues toward N>=6 ship gate.
    # Paper-only: nothing rendered to dashboard/WA until the report says SHIP.
    _run("4a/6 track_longshot_trio (paper experiment)",
         [sys.executable, '-m', 'scripts.track_longshot_trio', args.race_date])

    # Step 5: EDGE tracker — logs EDGE fires + outcomes; feeds dashboard banner
    _run("5/6 track_edge (banner stats)",
         [sys.executable, '-m', 'scripts.track_edge', args.race_date])

    # After track_edge updates stats, re-render the dashboard so the EDGE
    # banner reflects this meeting's fires too. Use fetch_results (the
    # results-overlay path) NOT generate.py: generate.py rebuilds from a fresh
    # live card fetch and never sets results_available, so it would wipe the
    # STRIKE/MISS badges that step 1 just merged. fetch_results re-renders WITH
    # badges and reloads edge_stats — and skips the needless live odds fetch.
    _run("5a/6 regenerate dashboard with fresh edge_stats (results overlay)",
         [sys.executable, '-m', 'scripts.fetch_results', args.race_date])

    if args.skip_dump:
        print(f"\n  [6/6 DB dump] SKIPPED — run `bash scripts/auto_dump.sh` later")
    else:
        rc = _run("6/6 DB dump + commit + push",
                  ['bash', 'scripts/auto_dump.sh'])
        if rc != 0:
            print(f"\nWARN: auto_dump exited non-zero. Check logs/auto_dump.log.",
                  file=sys.stderr)

    print(f"\n{'='*70}\n  POST-RACE COMPLETE — {args.race_date}\n{'='*70}")
    print(f"  Cumulative shadow comparison: python3 -m scripts.shadow_evaluate --report")
    print(f"  Cumulative tier B forward:    python3 -m scripts.tier_b_evaluate --report")
    print(f"  Spoiler dataset:              tracking_data/spoilers.csv")
    print(f"  EDGE history:                 tracking_data/edge_log.csv + edge_stats.json")
    print(f"  Longshot-TRIO experiment:     python3 -m scripts.track_longshot_trio --report")
    return 0


if __name__ == '__main__':
    sys.exit(main())
