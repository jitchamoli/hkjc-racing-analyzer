"""Race-day one-command wrapper: production predictions + shadow logging.

Runs in sequence:
  1. python3 generate_predictions.py <date>   — production dashboard + JSON
  2. python3 -m scripts.shadow_generate <date> — shadow predictions (rule_w=0.9)

Both steps are required for the active shadow-logging experiment. After
the meeting finishes, run `scripts.post_race` to ingest results, evaluate
shadow vs production, and refresh the DB dump.

Usage:
    python3 -m scripts.race_day '2026/05/17'
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(label: str, cmd: list[str]) -> int:
    """Run a subcommand, stream output, return exit code."""
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
    args = p.parse_args()

    rc = _run("1/2 production predictions",
              [sys.executable, 'generate_predictions.py', args.race_date])
    if rc != 0:
        print(f"\nABORT: production prediction failed. Skipping shadow.", file=sys.stderr)
        return rc

    rc = _run("2/2 shadow predictions (rule_w=0.9)",
              [sys.executable, '-m', 'scripts.shadow_generate', args.race_date])
    if rc != 0:
        print(f"\nWARN: shadow failed but production succeeded. "
              f"You can bet from the production dashboard; rerun shadow later.",
              file=sys.stderr)
        return rc

    print(f"\n{'='*70}\n  RACE DAY READY — {args.race_date}\n{'='*70}")
    print(f"  Dashboard: results/racing_dashboard.html")
    print(f"  Production JSON: results/race_data_{args.race_date.replace('/','')}.json")
    print(f"  Shadow JSON:     results/shadow_race_data_{args.race_date.replace('/','')}.json")
    print(f"\n  After racing: python3 -m scripts.post_race '{args.race_date}'")
    return 0


if __name__ == '__main__':
    sys.exit(main())
