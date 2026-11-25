"""Race-day one-command wrapper: production predictions + shadow logging.

Runs in sequence:
  1. python3 generate_predictions.py <date>            — production dashboard + JSON
  2. python3 -m scripts.shadow_generate --variant tips — TIPS-off shadow (forward A/B)
  3. python3 -m scripts.tier_b_shadow <date>           — log OLD-recommender decisions
                                                          for Tier B forward-validation
  4. python3 -m scripts.snapshot_racing_api <date>     — optional RacingAPI snapshot

(jockey_form shadow RETIRED 2026-06-21 — SHIP: posterior 0.965, JOCKEY_FORM_RIDES_MIN
 now 15 in production. rule_w=0.9 shadow RETIRED 2026-06-03 — KILL: posterior 0.016.
 Both histories preserved in shadow_jockey_log.csv / shadow_log.csv; shadow_generate
 + shadow_evaluate still callable manually.)

These feed independent experiments. After the meeting finishes, run
`scripts.post_race` to ingest results, evaluate all shadows vs production +
tier B old vs new, and refresh the DB dump.

Usage:
    python3 -m scripts.race_day 'YYYY/MM/DD'
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

    rc = _run("1/4 production predictions",
              [sys.executable, 'generate_predictions.py', args.race_date])
    if rc != 0:
        print(f"\nABORT: production prediction failed. Skipping shadow.", file=sys.stderr)
        return rc

    # jockey_form shadow RETIRED 2026-06-21 — shipped to production
    # (JOCKEY_FORM_RIDES_MIN=15), so the shadow now equals production (no-op).
    # History preserved in tracking_data/shadow_jockey_log.csv.

    # TIPS-off shadow (forward A/B, started 2026-06-27). The TIPS factor only
    # fires on the live card (tips_index never stored in race_results), so it is
    # untestable in backtest — a forward shadow is the only valid measure. This
    # re-fetches the cards and re-scores with the TIPS weight zeroed; post_race
    # then compares it vs production over N>=6 meetings. Non-fatal on failure.
    rc_t = _run("2/4 shadow_generate (TIPS off — forward A/B)",
                [sys.executable, '-m', 'scripts.shadow_generate',
                 args.race_date, '--variant', 'tips'])
    if rc_t != 0:
        print(f"\nWARN: TIPS shadow generation failed but production succeeded.",
              file=sys.stderr)

    # Tier B shadow is a pure local replay of race_data JSON (no HKJC fetch)
    rc_b = _run("3/4 tier B shadow (OLD recommender decisions)",
                [sys.executable, '-m', 'scripts.tier_b_shadow', args.race_date])
    if rc_b != 0:
        print(f"\nWARN: tier B shadow failed but production succeeded.",
              file=sys.stderr)

    # Optional RacingAPI confluence snapshot. This third-party integration is
    # independent of HKJC prediction generation and may fail when credentials
    # or subscription access are unavailable.
    rc_c = _run("OPTIONAL confluence snapshot (RacingAPI → JSON)",
                [sys.executable, '-m', 'scripts.snapshot_racing_api', args.race_date])
    if rc_c != 0:
        print(f"\nWARN: optional RacingAPI snapshot failed (auth/network/quota?). Confluence "
              f"tracking will skip this meeting. Production unaffected.",
              file=sys.stderr)

    date_str = args.race_date.replace('/', '')
    print(f"\n{'='*70}\n  RACE DAY READY — {args.race_date}\n{'='*70}")
    print(f"  Dashboard:           results/racing_dashboard.html")
    print(f"  Production JSON:     results/race_data_{date_str}.json")
    print(f"\n  After racing: python3 -m scripts.post_race '{args.race_date}'")
    return 0


if __name__ == '__main__':
    sys.exit(main())
