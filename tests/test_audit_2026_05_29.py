"""Regression-lock tests for the 12-finding audit of 2026-05-29.

One test per finding from the 10-expert review. Each lock makes the bug
hard to silently reintroduce. Tests are grouped by finding number so the
audit memo (`memory/project_audit_2026-05-29.md`) maps 1:1.

Most tests are CHEAP — they read a source file and assert the fix shape.
A grep-based test catches the regression even when the unit test infrastructure
for that subsystem doesn't exist (JS in template.py, gate logic with real
DB+JSON+dividends). Behavioural tests are used where they're cheap to set
up (Python helpers with no DB dependency).

When a test changes you must update both the lock AND the audit memo.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────
# Finding #1 — post_race.py `_run(required=)` must actually honor the flag
# ─────────────────────────────────────────────────────────────────────

def test_post_race_run_aborts_when_required():
    """Required=True step that fails should sys.exit, not silently continue."""
    sys.path.insert(0, str(REPO / 'scripts'))
    try:
        import post_race
        with patch.object(post_race.subprocess, 'run') as mock_run:
            mock_run.return_value = MagicMock(returncode=2)
            with pytest.raises(SystemExit) as exc_info:
                post_race._run('test step', ['false'], required=True)
            assert exc_info.value.code == 2
    finally:
        sys.path.pop(0)


def test_post_race_run_continues_when_not_required():
    """Required=False step that fails should return the exit code, not abort."""
    sys.path.insert(0, str(REPO / 'scripts'))
    try:
        import post_race
        with patch.object(post_race.subprocess, 'run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            rc = post_race._run('optional step', ['false'], required=False)
            assert rc == 1  # no SystemExit
    finally:
        sys.path.pop(0)


# ─────────────────────────────────────────────────────────────────────
# Finding #2 — track_confluence.py must dedup on re-run
# ─────────────────────────────────────────────────────────────────────

def test_track_confluence_dedup_on_re_run(tmp_path, monkeypatch):
    """Re-running for the same date must not double-write rows."""
    sys.path.insert(0, str(REPO / 'scripts'))
    try:
        import track_confluence
        log_path = tmp_path / 'confluence_log.csv'
        monkeypatch.setattr(track_confluence, 'LOG_PATH', log_path)

        row1 = {h: '' for h in track_confluence.HEADERS}
        row1['race_date'] = '2026/05/27'
        row1['race_number'] = 4
        row1['rank1_name'] = 'BRIGHT DAY'
        row1['won'] = 1

        track_confluence._write_rows([row1], replace_date='2026/05/27')
        track_confluence._write_rows([row1], replace_date='2026/05/27')

        body = log_path.read_text().strip().split('\n')
        data_rows = [r for r in body[1:] if r]  # skip header
        assert len(data_rows) == 1, (
            f"Re-run produced {len(data_rows)} rows; expected dedup to 1"
        )
    finally:
        sys.path.pop(0)


# ─────────────────────────────────────────────────────────────────────
# Finding #3 — _didBetHit must return null for SKIP races, not 'MISS'
# ─────────────────────────────────────────────────────────────────────

def test_did_bet_hit_skip_returns_null():
    """Static check: the both-branches-return-MISS pattern must be gone."""
    src = (REPO / 'dashboard' / 'template.py').read_text()
    # The buggy line was: return bets.length === 0 ? 'MISS' : 'MISS';
    assert "bets.length === 0 ? 'MISS' : 'MISS'" not in src, (
        "_didBetHit still returns 'MISS' in both branches — SKIP races "
        "render the wrong badge"
    )
    # And the fix should be: return null for SKIP/paper-only, 'MISS' for real miss
    assert "realBets.length === 0 ? null : 'MISS'" in src, (
        "_didBetHit fix not present — should return null for empty/paper-only ticket"
    )


# ─────────────────────────────────────────────────────────────────────
# Finding #4 — going_performance must compare via _going_bucket()
# ─────────────────────────────────────────────────────────────────────

def test_going_bucket_groups_synonyms():
    """The bucket helper itself must map all known synonyms onto same bucket."""
    from ml.features import FeatureEngineering
    bucket = FeatureEngineering._going_bucket
    # FIRM family — these are all the same physical surface
    assert bucket('GOOD') == bucket('GOOD TO FIRM') == bucket('GF') == \
           bucket('FIRM') == bucket('GD') == 'FIRM'
    # SOFT family
    assert bucket('HEAVY') == bucket('WET-SLOW') == bucket('SOFT') == 'SOFT'
    # YIELDING family
    assert bucket('YIELDING') == bucket('Y') == 'YIELDING'
    # WET-FAST stays its own bucket (NOT collapsed into SOFT)
    assert bucket('WET-FAST') == bucket('WF') == 'WET-FAST'
    assert bucket('WET-FAST') != bucket('SOFT')


def test_going_performance_uses_bucket_not_raw_eq():
    """Both inference and training sites must use _going_bucket() for the
    same-going comparison. Raw 'going'] == going equality silently nulls
    Feature 10 for the majority of horses because DB stores 'GOOD TO FIRM'
    where today's card may have 'GF'."""
    # extract_features in features.py
    src_f = (REPO / 'ml' / 'features.py').read_text()
    # Find the going-performance block by anchor comment
    f11_idx = src_f.find('Feature 10: going performance')
    assert f11_idx > 0
    f11_block = src_f[f11_idx:f11_idx + 1200]
    assert "_going_bucket(going)" in f11_block, (
        "ml/features.py Feature 10 must bucket today's going via "
        "FeatureEngineering._going_bucket()"
    )
    # Old buggy line was: p.get('going', '') == going
    assert "p.get('going', '') == going" not in f11_block, (
        "ml/features.py Feature 10 still uses raw exact-string match"
    )

    # build_training_data_from_db in training.py
    src_t = (REPO / 'ml' / 'training.py').read_text()
    t11_idx = src_t.find('Feature 10: going performance')
    assert t11_idx > 0
    t11_block = src_t[t11_idx:t11_idx + 1200]
    assert "_going_bucket(going)" in t11_block, (
        "ml/training.py Feature 10 must bucket today's going"
    )
    assert "p.get('going', '') == going" not in t11_block, (
        "ml/training.py Feature 10 still uses raw exact-string match"
    )


# ─────────────────────────────────────────────────────────────────────
# Finding #5 — gate_impact.py PROD arm must apply tier+field gate
# ─────────────────────────────────────────────────────────────────────

def test_gate_impact_prod_arm_applies_tier_and_field_gate():
    """PROD arm must skip the same races production does (LOW/SKIP tier, field<8)."""
    src = (REPO / 'scripts' / 'gate_impact.py').read_text()
    assert 'prod_skips' in src, (
        "gate_impact must compute and apply prod_skips so PROD arm "
        "doesn't bet races production skips"
    )
    # Should gate on tier and field exactly as the rest of the pipeline does
    assert "tier not in ('EDGE', 'HIGH', 'MED')" in src
    assert 'field < 8' in src
    # And the PROD accumulator must be inside a `not prod_skips` branch
    assert 'if not prod_skips' in src


# ─────────────────────────────────────────────────────────────────────
# Finding #6 — rebuild_stats failure must not raise out of ingest_race_day
# ─────────────────────────────────────────────────────────────────────

def test_rebuild_stats_failure_does_not_propagate():
    """ingest_race_day must keep going if rebuild_stats fails — race_results
    is already committed; raising masks the success as failure."""
    src = (REPO / 'data' / 'database.py').read_text()
    rebuild_idx = src.find('self.rebuild_stats()')
    assert rebuild_idx > 0, (
        "Couldn't find self.rebuild_stats() call — has the ingest path changed?"
    )
    # Look at the 30 lines following the call for the except handler
    tail = src[rebuild_idx:rebuild_idx + 2000]
    except_idx = tail.find('except Exception:')
    assert except_idx > 0, (
        "rebuild_stats call is not wrapped in try/except — failure would "
        "abort ingest entirely. Wrap in try and log+swallow."
    )
    # Take just the except block (stop at the next non-indented line or
    # the next blank line followed by less indentation)
    except_tail = tail[except_idx:except_idx + 800]
    # Heuristic: until the next dedent
    lines = except_tail.splitlines()
    in_block = []
    for i, line in enumerate(lines):
        if i == 0:
            in_block.append(line)
            continue
        if not line.strip():
            in_block.append(line)
            continue
        # Inside the except block, indentation should be > the except's
        if line.startswith('            ') or line.startswith('                '):
            in_block.append(line)
        else:
            break
    bare_raise = [l for l in (s.strip() for s in in_block) if l == 'raise']
    assert not bare_raise, (
        "ingest_race_day's rebuild_stats except block re-raises — this masks "
        "a successful ingest as failure. Log + swallow instead."
    )


# ─────────────────────────────────────────────────────────────────────
# Finding #7 — copyWA() must have execCommand + window.prompt fallback
# ─────────────────────────────────────────────────────────────────────

def test_copywa_has_fallback_ladder():
    """Per-card copyWA must mirror the bulk copy handler's fallback ladder
    (execCommand textarea trick + window.prompt last resort) so file://
    contexts and Safari focus rules don't silently no-op."""
    src = (REPO / 'dashboard' / 'template.py').read_text()
    copywa_idx = src.find('function copyWA(text, btn)')
    assert copywa_idx > 0
    block = src[copywa_idx:copywa_idx + 2500]
    # Three signals the fallback ladder is present
    assert "document.execCommand('copy')" in block, (
        "copyWA missing execCommand fallback"
    )
    assert 'window.prompt' in block, (
        "copyWA missing window.prompt last-resort"
    )
    assert '.catch(fallback)' in block, (
        "copyWA missing .catch() chain on clipboard promise"
    )


# ─────────────────────────────────────────────────────────────────────
# Finding #8 — Copy all bets bulk payload must append the disclaimer
# ─────────────────────────────────────────────────────────────────────

def test_copy_all_bets_appends_disclaimer():
    """Per-card copyWA appends the disclaimer; bulk handler bypasses copyWA
    and must append it independently."""
    src = (REPO / 'dashboard' / 'template.py').read_text()
    bulk_idx = src.find("copyAllBtn.addEventListener('click'")
    assert bulk_idx > 0
    block = src[bulk_idx:bulk_idx + 2500]
    # The combined-payload assignment line must include the disclaimer
    combined_idx = block.find("var combined = payloads.join")
    assert combined_idx > 0
    after = block[combined_idx:combined_idx + 400]
    assert 'entertainment only' in after.lower(), (
        "Bulk 'Copy all bets' handler omits the entertainment-only disclaimer"
    )


# ─────────────────────────────────────────────────────────────────────
# Finding #9 — value_score must stay pinned to pre-blend combined_prob
# ─────────────────────────────────────────────────────────────────────

def test_value_score_not_overwritten_in_recommender():
    """CLAUDE.md ML Rule 4: value_score = combined_prob * 100 (pre-blend).
    _resync_ranks_and_probs and apply_calibrated_probs must NOT overwrite it."""
    rec = (REPO / 'core' / 'recommender.py').read_text()
    resync_idx = rec.find('def _resync_ranks_and_probs')
    assert resync_idx > 0
    block = rec[resync_idx:resync_idx + 1500]
    assert 'a.value_score' not in block, (
        "_resync_ranks_and_probs overwrites value_score with the post-blend "
        "calibrated_prob — violates CLAUDE.md ML Rule 4"
    )

    scorer = (REPO / 'core' / 'scorer.py').read_text()
    apply_idx = scorer.find('def apply_calibrated_probs')
    assert apply_idx > 0
    apply_block = scorer[apply_idx:apply_idx + 1500]
    # Allow `value_score` to appear in comments (it should be discussed)
    # but not as an assignment.
    assert 'a.value_score =' not in apply_block, (
        "apply_calibrated_probs overwrites value_score — violates Rule 4"
    )


# ─────────────────────────────────────────────────────────────────────
# Finding #10 — validate_hk synthetic report must caveat that it's pre-gate
# ─────────────────────────────────────────────────────────────────────

def test_validate_hk_synthetic_report_has_caveat():
    """The synthetic --report output must explicitly state it's a pre-gate
    upper bound, not what production would have staked."""
    src = (REPO / 'scripts' / 'validate_hk.py').read_text()
    assert 'pre-gate UPPER BOUND' in src or 'SYNTHETIC = pre-gate' in src, (
        "validate_hk synthetic report missing the pre-gate caveat — "
        "user can confuse this with real production ROI"
    )
    assert '--replay' in src  # caveat should point at the replay alternative


# ─────────────────────────────────────────────────────────────────────
# Finding #11 — monitor must call _merge_with_existing before finalize
# ─────────────────────────────────────────────────────────────────────

def test_monitor_merges_existing_before_finalize():
    """monitor/race_day.py must restore frozen state from prior JSON before
    re-running finalize_tickets_and_stakes — otherwise transient odds drift
    silently rebuilds the ticket every 6 minutes."""
    src = (REPO / 'monitor' / 'race_day.py').read_text()
    assert '_merge_with_existing' in src, (
        "monitor/race_day.py must call _merge_with_existing to restore the "
        "frozen bet_ticket before recomputing — otherwise ticket-is-truth "
        "invariant is violated on every refresh"
    )
    merge_idx = src.find('_merge_with_existing(json_file')
    finalize_idx = src.find('finalize_tickets_and_stakes(all_races')
    assert merge_idx > 0 and finalize_idx > 0
    assert merge_idx < finalize_idx, (
        "_merge_with_existing must be called BEFORE finalize_tickets_and_stakes"
    )


# ─────────────────────────────────────────────────────────────────────
# Finding #12 — BOX TRIO-6 decision must come from single Python source
# ─────────────────────────────────────────────────────────────────────

def test_trio6_decision_single_source():
    """All three sites (server gate, JS detail panel, JS WA copy) must
    consume race.trio6_decision built by _compute_trio6_decision — not
    recompute independently."""
    # Server helper exists
    gen_src = (REPO / 'dashboard' / 'generator.py').read_text()
    assert 'def _compute_trio6_decision' in gen_src
    assert "race['trio6_decision'] = decision" in gen_src

    # JS sites read trio6_decision, not the local rebuild of steam/chaos.
    # Positive check: BOTH JS consumers (detail panel + WA compact copy)
    # must reference race.trio6_decision — so >= 2 reads. A vacuous count
    # of 0 (the prior `<= 1` check) would pass even if both sites had
    # silently dropped the read; this catches that.
    tmpl_src = (REPO / 'dashboard' / 'template.py').read_text()
    n_reads = tmpl_src.count('race.trio6_decision')
    assert n_reads >= 2, (
        f"Expected both JS sites (detail panel + WA copy) to read "
        f"race.trio6_decision; found {n_reads} read(s)."
    )
    # Negative check: the pre-consolidation steam-recount pattern must be
    # fully gone from template.py (was duplicated across WA + detail panel).
    steam_recount = "race.horses.slice(0, 12).filter"
    assert tmpl_src.count(steam_recount) == 0, (
        "Found a site recomputing steam_count from race.horses — must read "
        "race.trio6_decision.steam_count instead (consolidated 2026-05-29)."
    )


def test_trio6_decision_helper_handles_all_branches():
    """_compute_trio6_decision contract: heavy-fav skip, steam take,
    HIGH+big-field take, otherwise YOUR CALL. Chaos is independent."""
    sys.path.insert(0, str(REPO))
    try:
        from dashboard.generator import _compute_trio6_decision
        # Heavy fav → SKIP regardless of other signals
        horses_heavy = [{'win_odds': 2.0, 'steam_move': {'is_steam': True}}] * 6
        d = _compute_trio6_decision(
            horses=horses_heavy, tier='HIGH', field=12, top1_odds=2.0,
            top_2_share=0.30,
        )
        assert d['take'] is False
        assert 'SKIP' in d['verdict_label']

        # 2+ steam horses → TAKE
        horses_steam = [
            {'win_odds': 3.0, 'steam_move': {'is_steam': True}},
            {'win_odds': 5.0, 'steam_move': {'is_steam': True}},
            {'win_odds': 8.0, 'steam_move': None},
        ] + [{'win_odds': 10.0, 'steam_move': None}] * 9
        d = _compute_trio6_decision(
            horses=horses_steam, tier='MED', field=12, top1_odds=3.0,
            top_2_share=0.35,
        )
        assert d['take'] is True
        assert '★ TAKE' in d['verdict_label']

        # HIGH + field≥11 + competitive fav → TAKE
        horses_high = [{'win_odds': 3.5, 'steam_move': None}] * 12
        d = _compute_trio6_decision(
            horses=horses_high, tier='HIGH', field=12, top1_odds=3.5,
            top_2_share=0.40,
        )
        assert d['take'] is True

        # Otherwise → YOUR CALL (and bet NOT added)
        horses_meh = [{'win_odds': 6.0, 'steam_move': None}] * 8
        d = _compute_trio6_decision(
            horses=horses_meh, tier='LOW', field=8, top1_odds=6.0,
            top_2_share=0.30,
        )
        assert d['take'] is False
        assert 'YOUR CALL' in d['verdict_label']
    finally:
        sys.path.pop(0)


def test_trio6_decision_frozen():
    """audit-2026-05-29 finding A: trio6_decision must be in
    _FROZEN_PREDICTION_KEYS_RACE so a post-off re-render does NOT recompute
    the TRIO-6 verdict from drifted odds while the bet_ticket stays frozen.
    Without this freeze, the displayed verdict can disagree with what was
    staked — defeating the fix #12 single-source consolidation."""
    sys.path.insert(0, str(REPO))
    try:
        from dashboard.generator import _FROZEN_PREDICTION_KEYS_RACE
        assert 'trio6_decision' in _FROZEN_PREDICTION_KEYS_RACE, (
            "trio6_decision must be frozen alongside bet_ticket — otherwise "
            "re-renders recompute the verdict from current odds and it drifts "
            "from the staked ticket"
        )
        # bet_ticket must also be frozen (the pair must move together)
        assert 'bet_ticket' in _FROZEN_PREDICTION_KEYS_RACE
    finally:
        sys.path.pop(0)


# ─────────────────────────────────────────────────────────────────────
# Cross-cutting invariant — going strings must NEVER be compared raw
# ─────────────────────────────────────────────────────────────────────

def test_no_raw_going_string_equality_in_ml_layer():
    """Structural lock: every place in ml/ that filters by `going` must do
    so via _going_bucket. Catches a future regression of Finding #4 — and
    catches the same bug being introduced in any new feature.

    Allows commented references and the _going_bucket definition itself.
    """
    forbidden = re.compile(r"\.get\(['\"]going['\"][^)]*\)\s*==\s*going")
    for path in (REPO / 'ml').glob('*.py'):
        src = path.read_text()
        # Strip lines that are comments
        non_comment = '\n'.join(
            line for line in src.splitlines()
            if not line.strip().startswith('#')
        )
        matches = forbidden.findall(non_comment)
        assert not matches, (
            f"{path.name}: found raw `p.get('going',...) == going` equality. "
            f"Use FeatureEngineering._going_bucket() on both sides — see "
            f"audit Finding #4 (2026-05-29) for why."
        )


# ─────────────────────────────────────────────────────────────────────
# Follow-up D2 — going_pref wet-slow surfaces classify as wet, not GOOD
# ─────────────────────────────────────────────────────────────────────

def test_going_pref_wet_surfaces_classified_wet():
    """audit-2026-05-29 follow-up: WET SLOW / RAIN AFFECTED are genuinely wet
    (~616 historical rows) but used to fall through to the wetness=1 GOOD
    default, so the rule engine never awarded the wet-track-specialist bonus.
    WET-FAST must STAY firm (plays fast — no mud bonus)."""
    from analytics.going_pref import GoingPreferenceAnalyzer as G
    assert G._classify_going('WET SLOW') == 4
    assert G._classify_going('Wet Slow') == 4
    assert G._classify_going('WET-SLOW') == 4
    assert G._classify_going('Rain Affected') == 4
    # WET-FAST is a fast surface — must NOT be treated as wet
    assert G._classify_going('WET-FAST') < 3
    assert G._classify_going('Wet Fast') < 3
    # unchanged baselines
    assert G._classify_going('GOOD TO FIRM') == 0
    assert G._classify_going('YIELDING') == 3
    # no duplicate keys snuck in
    keys = list(G.GOING_WETNESS.keys())
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"


def test_going_bucket_awt_maps_to_firm():
    """2026-06-08 review fix: AWT goings MUST bucket to FIRM in BOTH directions.
    The earlier self-map ('Good (AWT)' -> 'GOOD (AWT)') was a bug: 'GOOD (AWT)'
    is not a key of _GOING_REVERSE, so _GOING_REVERSE.get() returned [] and the
    sire-going feature silently fell back to its 0.20 prior for every AWT row.
    Mapping AWT -> FIRM (matching _GOING_REVERSE['FIRM']) lets AWT horses match
    their FIRM priors. The model was retrained with this behaviour, so train and
    inference agree. Empty string is still the only value that returns ''."""
    from ml.features import FeatureEngineering as FE, _GOING_REVERSE
    assert FE._going_bucket('') == ''
    assert FE._going_bucket('Good (AWT)') == 'FIRM'
    assert FE._going_bucket('SEALED') == 'FIRM'
    assert FE._going_bucket('SE') == 'FIRM'
    # round-trip: the reverse list must contain the DB going string (the gap
    # that caused the bug — bucket maps forward, reverse expands back)
    assert 'GOOD (AWT)' in _GOING_REVERSE['FIRM']
    # known surfaces still bucket correctly
    assert FE._going_bucket('GOOD TO FIRM') == 'FIRM'
    assert FE._going_bucket('WF') == 'WET-FAST'


# ─────────────────────────────────────────────────────────────────────
# Follow-up D — monitor injects tier_stats (display parity with generator)
# ─────────────────────────────────────────────────────────────────────

def test_monitor_injects_tier_stats():
    """audit-2026-05-29 follow-up: monitor must inject tier_stats before
    finalize so the dashboard trust panel populates, matching the main
    generator. Must NOT call log_predictions (mid-race re-log corrupts
    predicted_rank) or compute_alerts (sharp-move path already fires them)."""
    src = (REPO / 'monitor' / 'race_day.py').read_text()
    assert 'load_cached_stats' in src and "r['tier_stats']" in src, (
        "monitor must inject tier_stats for trust-panel parity"
    )
    # tier_stats injection must come before finalize_tickets_and_stakes
    ts_idx = src.find("r['tier_stats']")
    fin_idx = src.find('finalize_tickets_and_stakes(all_races')
    assert 0 < ts_idx < fin_idx, "tier_stats must be injected before finalize"
    # log_predictions must NOT be CALLED in the monitor (would corrupt rank).
    # Check for the call syntax, not the bare word — the explanatory comment
    # in monitor/race_day.py legitimately mentions log_predictions.
    assert '.log_predictions(' not in src and 'db.log_predictions' not in src, (
        "monitor must NOT call log_predictions — mid-race re-log corrupts "
        "predicted_rank (the freeze logic guards against exactly this)"
    )
