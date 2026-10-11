"""Post-race one-command wrapper: results ingest + analytics + DB dump.

Runs in sequence:
  1. results_fetcher.py <date> --auto-learn   — ingest finishes, retrain ML,
                                                refresh dashboard with badges
  2. scripts.shadow_evaluate <date>           — prod vs rule_w=0.9 shadow
  2b. scripts.shadow_evaluate --variant tips  — prod vs TIPS-off shadow (fwd A/B)
  3. scripts.tier_b_evaluate <date>           — OLD vs NEW recommender
  4. scripts.track_spoilers <date>            — log outside-top-5 placers
  4a. scripts.track_longshot_trio <date>      — paper experiment: top-5 +
                                                steamed rank 6-8 longshot
                                                (NOT wired to dashboard/WA)
  5. scripts.track_edge <date>                — log EDGE fires + outcomes
                                                (feeds dashboard banner)
  5d. scripts.clv_report --split-by-market-rank — CLV edge gauge (rank-2-3
                                                disagreement CLV is the signal)
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
    """Run a subcommand, stream output, return exit code.

    When `required=True`, a non-zero exit aborts the whole post-race phase
    (sys.exit). When `required=False`, the failure is warned about but
    swallowed so downstream steps still run. Prior to 2026-05-29 the flag
    was silently ignored — every step was de-facto required AND non-fatal
    in a confusing mix; pinning this contract is locked by
    tests/test_audit_2026_05_29.py::test_post_race_run_aborts_when_required.
    """
    print(f"\n{'='*70}\n  {label}\n  $ {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.monotonic()
    r = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.monotonic() - t0
    status = "OK" if r.returncode == 0 else f"FAIL (exit {r.returncode})"
    print(f"\n  [{label}] {status} in {elapsed:.1f}s", flush=True)
    if r.returncode != 0 and required:
        print(f"\nABORT: required step '{label}' failed (exit {r.returncode}). "
              f"Fix and rerun.", file=sys.stderr)
        sys.exit(r.returncode)
    if r.returncode != 0:
        print(f"\nWARN: non-required step '{label}' failed (exit {r.returncode}). "
              f"Continuing.", file=sys.stderr)
    return r.returncode


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('race_date', help='YYYY/MM/DD')
    p.add_argument('--skip-dump', action='store_true',
                   help='Skip the DB dump step (run scripts/auto_dump.sh later)')
    args = p.parse_args()

    # Step 1 is the ONLY hard-required step: everything downstream consumes
    # the results it ingests. required=True (default) makes _run sys.exit on
    # failure — so the old manual `if rc != 0: return rc` guard here is now
    # dead code and was removed (audit-2026-05-29 finding F).
    _run("1/6 results ingest + auto-learn",
         [sys.executable, 'results_fetcher.py', args.race_date, '--auto-learn'])

    _run("calibration audit (frozen pre-off probabilities)",
         [sys.executable, '-m', 'scripts.calibration_report'], required=False)

    # Step 2: shadow eval — skip cleanly if shadow file missing.
    # Two variants share this step: rule_w (legacy) and jockey_form.
    date_str = args.race_date.replace('/', '')
    rule_w_shadow = REPO_ROOT / 'results' / f'shadow_race_data_{date_str}.json'
    if not rule_w_shadow.exists():
        print(f"\n  [2a/6 shadow_evaluate rule_w] SKIPPED — no file at {rule_w_shadow}")
    else:
        rc = _run("2a/6 shadow_evaluate (rule_w)",
                  [sys.executable, '-m', 'scripts.shadow_evaluate', args.race_date],
                  required=False)
        if rc != 0:
            print(f"\nWARN: shadow_evaluate rule_w failed but results were ingested.",
                  file=sys.stderr)

    # jockey_form shadow eval RETIRED 2026-06-21 — shipped to production
    # (JOCKEY_FORM_RIDES_MIN=15); shadow now equals production (no-op).
    # History preserved in tracking_data/shadow_jockey_log.csv.

    # Step 2b: TIPS-off shadow eval (forward A/B started 2026-06-27). Skip
    # cleanly if the shadow file is missing (e.g. shadow_generate failed AM).
    tips_shadow = REPO_ROOT / 'results' / f'shadow_tips_race_data_{date_str}.json'
    if not tips_shadow.exists():
        print(f"\n  [2b/6 shadow_evaluate tips] SKIPPED — no file at {tips_shadow}")
    else:
        rc = _run("2b/6 shadow_evaluate (TIPS off)",
                  [sys.executable, '-m', 'scripts.shadow_evaluate', args.race_date,
                   '--variant', 'tips'],
                  required=False)
        if rc != 0:
            print(f"\nWARN: shadow_evaluate tips failed but results were ingested.",
                  file=sys.stderr)

    # Step 2c: stake A/B — compares ROI under flat-$10/combo (user habit),
    # production tier-ramp, flat-1%, and inverted ramp. Same picks, different
    # stakes. Reads results/race_data_*_with_results.json so it must run AFTER
    # step 1 (which writes that file). See scripts/stake_ab.py for variants.
    merged_path = REPO_ROOT / 'results' / f'race_data_{date_str}_with_results.json'
    if not merged_path.exists():
        print(f"\n  [2c/6 stake_ab] SKIPPED — no merged race_data at {merged_path}")
    else:
        rc = _run("2c/6 stake_ab (staking variants on same picks)",
                  [sys.executable, '-m', 'scripts.stake_ab', args.race_date],
                  required=False)
        if rc != 0:
            print(f"\nWARN: stake_ab failed but results were ingested.",
                  file=sys.stderr)

    # Steps 3-5 are non-critical trackers/diagnostics — a crash in any of
    # them MUST NOT abort the phase before the DB dump (step 6). required=False
    # so a tracker failure warns and continues (audit-2026-05-29 finding C —
    # before the fix these defaulted to required=True and a tracker crash
    # would sys.exit before the dump, risking same-day DB-dump loss).

    # Step 3: tier B eval — always safe (skips internally if no rows)
    _run("3/6 tier B evaluate (OLD vs NEW recommender)",
         [sys.executable, '-m', 'scripts.tier_b_evaluate', args.race_date],
         required=False)

    # Step 4: spoiler tracker — logs rank>5 horses that placed top-3
    _run("4/6 track_spoilers",
         [sys.executable, '-m', 'scripts.track_spoilers', args.race_date],
         required=False)

    # Step 4a: longshot-TRIO paper experiment — accrues toward N>=6 ship gate.
    # Paper-only: nothing rendered to dashboard/WA until the report says SHIP.
    _run("4a/6 track_longshot_trio (paper experiment)",
         [sys.executable, '-m', 'scripts.track_longshot_trio', args.race_date],
         required=False)

    # Step 4c: QPL (Quinella Place) paper experiment — logs model top-2 as a
    # QPL pair vs straight QIN on real dividends. User observation 2026-06-10
    # (QPL hits more); 16-meeting backfill leans KILL (frequency illusion — QPL
    # hits ~2x but ROI worse). Paper-only; accrues toward the N>=6 forward gate.
    _run("4c/6 track_qpl (paper experiment)",
         [sys.executable, '-m', 'scripts.track_qpl', args.race_date],
         required=False)

    # Step 4b: HK forward-validation — appends this meeting (if after the frozen
    # cutoff) to the un-burned forward log. The only honest, never-tuned-on
    # evidence of real betting performance. Runs after step 1 fills results.
    # --replay added 2026-05-24 (GPT-5.5 Finding #1): also score the actual
    # frozen bet_ticket so the daily report doesn't show synthetic phantom
    # hits for bets that were gated out (R8/R11 2026-05-24 trigger).
    _run("4b/6 validate_hk (forward-validation log + replay)",
         [sys.executable, '-m', 'scripts.validate_hk', '--replay'], required=False)

    # Step 5: EDGE tracker — logs EDGE fires + outcomes; feeds dashboard banner
    _run("5/6 track_edge (banner stats)",
         [sys.executable, '-m', 'scripts.track_edge', args.race_date],
         required=False)

    # Step 5b: gate impact — compares BOX TRIO-5 OLD-vs-NEW gate P&L over the
    # last 8 meetings so user sees the cumulative effect of the gate ship every
    # meeting. Read-only diagnostic; never blocks.
    _run("5b/6 gate_impact (BOX TRIO-5 OLD vs NEW)",
         [sys.executable, '-m', 'scripts.gate_impact'], required=False)

    # Step 5c: confluence tracker — score the AM racing-API snapshot against
    # results to extend tracking_data/confluence_log.csv. WIN-sidebet
    # forward-validation per [[edge-pattern-n4]]. Skips silently if no snapshot
    # (racing API was down at AM, or this meeting predates the tracker).
    _run("5c/6 track_confluence (3-source WIN gate)",
         [sys.executable, 'scripts/track_confluence.py', args.race_date],
         required=False)

    # Step 5d: CLV edge gauge — cumulative closing-line-value on rank-1 picks,
    # split by market agreement. The fast, low-variance read on whether we have
    # a real edge: watch the rank-2-3 (mild-disagreement) CLV line accrue toward
    # N>=10 meetings. Immune to the favourite mirage AND the longshot artifact
    # (the split's results-aware verdict flags both). Read-only; never blocks.
    _run("5d/6 clv_report (market-agreement edge gauge)",
         [sys.executable, '-m', 'scripts.clv_report', '--split-by-market-rank'],
         required=False)

    # After track_edge updates stats, re-render the dashboard so the EDGE
    # banner reflects this meeting's fires too. Use fetch_results (the
    # results-overlay path) NOT generate.py: generate.py rebuilds from a fresh
    # live card fetch and never sets results_available, so it would wipe the
    # STRIKE/MISS badges that step 1 just merged. fetch_results re-renders WITH
    # badges and reloads edge_stats — and skips the needless live odds fetch.
    # required=False: a dashboard re-render failure must not block the DB dump
    # (step 6) — the dump is the data-recovery artifact the user always wants
    # after a meeting. A stale dashboard is recoverable by re-running; a
    # missing dump is not (audit-2026-05-29 finding C).
    _run("5a/6 regenerate dashboard with fresh edge_stats (results overlay)",
         [sys.executable, '-m', 'scripts.fetch_results', args.race_date],
         required=False)

    # Barrier-trials accrual. The trials FEATURE is killed (4-gate
    # investigation, CLAUDE.md) — this keeps the DATA current so the one
    # regime the kill left open (first-up horses, where nothing else measures
    # fitness) stays testable. Nothing called scrape_trials before 2026-09-06,
    # so the table sat 93 days stale and held zero rows for the pre-season
    # block — discovered on the morning of a season opener where all 120
    # runners were first-up. required=False: a network scrape unrelated to
    # this meeting must never abort a phase that has already ingested results.
    # Runs BEFORE the dump so the new rows are in the dump that gets committed.
    # Paper trackers (2026-09-16): neither changes a bet. Placed after the
    # results overlay so both read the finished card.
    _run("5f/6 track_gate_skips (do the gates earn their skips?)",
         [sys.executable, '-m', 'scripts.track_gate_skips', args.race_date],
         required=False)

    _run("5g/6 shadow_pure_rank (model-only vs blended ranking)",
         [sys.executable, '-m', 'scripts.shadow_pure_rank', args.race_date],
         required=False)

    _run("5e/6 scrape_trials (barrier-trial accrual)",
         [sys.executable, '-m', 'scripts.scrape_trials', '--since-last'],
         required=False)

    if args.skip_dump:
        print(f"\n  [6/6 DB dump] SKIPPED — run `bash scripts/auto_dump.sh` later")
    else:
        # required=False so a dump failure surfaces the specific log-path
        # warning below (with required=True _run would sys.exit first, making
        # that warning dead code and skipping the COMPLETE summary).
        rc = _run("6/6 DB dump + commit + push",
                  ['bash', 'scripts/auto_dump.sh'], required=False)
        if rc != 0:
            print(f"\nWARN: auto_dump exited non-zero. Check logs/auto_dump.log.",
                  file=sys.stderr)

    print(f"\n{'='*70}\n  POST-RACE COMPLETE — {args.race_date}\n{'='*70}")
    print(f"  Cumulative shadow rule_w:     python3 -m scripts.shadow_evaluate --report")
    print(f"  Cumulative shadow jockey_form:python3 -m scripts.shadow_evaluate --report --variant jockey_form")
    print(f"  Cumulative shadow TIPS:       python3 -m scripts.shadow_evaluate --report --variant tips")
    print(f"  Cumulative stake A/B:         python3 -m scripts.stake_ab --report")
    print(f"  Gate impact (BOX TRIO-5):     python3 -m scripts.gate_impact")
    print(f"  Cumulative tier B forward:    python3 -m scripts.tier_b_evaluate --report")
    print(f"  Spoiler dataset:              tracking_data/spoilers.csv")
    print(f"  EDGE history:                 tracking_data/edge_log.csv + edge_stats.json")
    print(f"  Longshot-TRIO experiment:     python3 -m scripts.track_longshot_trio --report")
    print(f"  QPL experiment (paper):       python3 -m scripts.track_qpl --report")
    print(f"  Confluence log (WIN gate):    tracking_data/confluence_log.csv")
    print(f"  CLV edge gauge (mkt split):   python3 -m scripts.clv_report --split-by-market-rank")
    return 0


if __name__ == '__main__':
    sys.exit(main())
