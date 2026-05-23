"""Race-day one-command wrapper: production predictions + shadow logging.

Runs in sequence:
  1. python3 generate_predictions.py <date>            — production dashboard + JSON
  2. python3 -m scripts.shadow_generate <date>         — shadow rule_w=0.9
  3. python3 -m scripts.shadow_generate <date>
       --variant jockey_form                           — shadow jockey_form OFF
  4. python3 -m scripts.tier_b_shadow <date>           — log OLD-recommender decisions
                                                          for Tier B forward-validation

All four feed independent experiments. After the meeting finishes, run
`scripts.post_race` to ingest results, evaluate all shadows vs production +
tier B old vs new, and refresh the DB dump.

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

    rc = _run("2/4 shadow predictions (rule_w=0.9)",
              [sys.executable, '-m', 'scripts.shadow_generate', args.race_date])
    if rc != 0:
        print(f"\nWARN: rule_w shadow failed but production succeeded. "
              f"You can bet from the production dashboard; rerun shadow later.",
              file=sys.stderr)
        # Don't return — other experiments are independent

    rc = _run("3/4 shadow predictions (jockey_form OFF)",
              [sys.executable, '-m', 'scripts.shadow_generate', args.race_date,
               '--variant', 'jockey_form'])
    if rc != 0:
        print(f"\nWARN: jockey_form shadow failed but production succeeded.",
              file=sys.stderr)

    # Tier B shadow is a pure local replay of race_data JSON (no HKJC fetch)
    rc_b = _run("4/4 tier B shadow (OLD recommender decisions)",
                [sys.executable, '-m', 'scripts.tier_b_shadow', args.race_date])
    if rc_b != 0:
        print(f"\nWARN: tier B shadow failed but production succeeded.",
              file=sys.stderr)

    date_str = args.race_date.replace('/', '')
    print(f"\n{'='*70}\n  RACE DAY READY — {args.race_date}\n{'='*70}")
    print(f"  Dashboard:           results/racing_dashboard.html")
    print(f"  Production JSON:     results/race_data_{date_str}.json")
    print(f"  Shadow JSON (rule):  results/shadow_race_data_{date_str}.json")
    print(f"  Shadow JSON (jock):  results/shadow_jockey_race_data_{date_str}.json")
    print(f"\n  After racing: python3 -m scripts.post_race '{args.race_date}'")
    return 0


if __name__ == '__main__':
    sys.exit(main())
