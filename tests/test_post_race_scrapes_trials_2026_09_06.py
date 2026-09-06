"""`raceday finish` must keep barrier_trials current.

The table sat 93 days stale (last scrape 2026/06/05) and was discovered empty
for the entire 2026 pre-season block on the morning of the season opener —
2026/09/06, when all 120 runners were first-up off a 53+ day layoff and trial
form was the only fitness evidence available anywhere. `scrape_trials.py`
existed and worked; nothing called it. It was in no automated path at all.

Two properties this pins:

1. post_race INVOKES scripts.scrape_trials with --since-last. The barrier
   trials feature itself is KILLED (see the 4-gate investigation in CLAUDE.md);
   this is data accrual only, so that the hypothesis stays testable on future
   first-up-heavy meetings rather than needing another 93-day backfill.

2. It runs with required=False. A trials scrape is a network call against a
   page unrelated to the meeting being closed — it must never abort the
   post-race phase, which ingests results, refits calibrators and writes the
   DB dump. Failure warns and continues.

It is placed BEFORE the DB dump step so newly scraped trials are included in
the dump that step commits, rather than waiting a meeting to be persisted.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'scripts' / 'post_race.py'


def _run_calls():
    """Every _run(...) call in post_race.main(), in source order."""
    tree = ast.parse(SRC.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Name) and f.id == '_run'):
            continue
        label = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else ''
        cmd = []
        if len(node.args) > 1 and isinstance(node.args[1], ast.List):
            cmd = [e.value for e in node.args[1].elts if isinstance(e, ast.Constant)]
        required = True
        for kw in node.keywords:
            if kw.arg == 'required' and isinstance(kw.value, ast.Constant):
                required = kw.value.value
        out.append((node.lineno, label, cmd, required))
    return sorted(out)


def test_post_race_scrapes_trials():
    calls = _run_calls()
    hits = [c for c in calls if 'scripts.scrape_trials' in c[2]]
    assert hits, (
        'post_race never calls scripts.scrape_trials — barrier_trials will go '
        f'stale again. steps found: {[c[1] for c in calls]}'
    )


def test_trials_scrape_uses_since_last():
    hit = next(c for c in _run_calls() if 'scripts.scrape_trials' in c[2])
    assert '--since-last' in hit[2], (
        f'must scrape incrementally, not re-fetch every date: {hit[2]}'
    )


def test_trials_scrape_is_not_required():
    """A network scrape must never abort the meeting close."""
    hit = next(c for c in _run_calls() if 'scripts.scrape_trials' in c[2])
    assert hit[3] is False, (
        'scrape_trials must run with required=False — a failed trials scrape '
        'would otherwise abort post_race after results are already ingested'
    )


def test_trials_scrape_runs_before_the_db_dump():
    """So the dump that gets committed contains the new trial rows."""
    calls = _run_calls()
    trials = next(c for c in calls if 'scripts.scrape_trials' in c[2])
    dumps = [c for c in calls if 'dump' in c[1].lower()]
    assert dumps, f'no DB dump step found: {[c[1] for c in calls]}'
    assert trials[0] < dumps[0][0], (
        f'scrape_trials (line {trials[0]}) must precede the DB dump '
        f'(line {dumps[0][0]}) or new trials wait a meeting to be persisted'
    )
