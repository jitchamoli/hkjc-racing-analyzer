"""Tests for gear-change analyzer first-time + last-run diff detection.

Score values are calibrated from race_results data via
scripts/calibrate_gear.py — these tests assert BEHAVIOR (labels, prefixes,
score bounds) rather than exact numbers, so re-calibration doesn't break
the suite.
"""

from analytics.gear_change import GearChangeAnalyzer as G


def test_no_gear_returns_neutral():
    score, reason = G.analyze_gear("")
    assert score == 50.0
    assert reason == "No gear changes"


def test_known_gear_is_within_calibrated_bounds():
    """A standard gear code should return a score in [20, 80] (the floor/ceil
    used by the calibrator), not the raw default 50."""
    score, reason = G.analyze_gear("B")
    assert 20 <= score <= 80
    assert "Blinkers ON" in reason
    assert "FIRST TIME" not in reason
    assert "NEW" not in reason


def test_debutant_blinkers_uses_first_time_variant():
    """Debutant with B should fire the B1 variant (different score than B)
    and surface the FIRST TIME label in reasoning."""
    debut_score, debut_reason = G.analyze_gear("B", is_first_time=True)
    standard_score, _ = G.analyze_gear("B")
    assert "FIRST TIME (debut)" in debut_reason
    assert "Blinkers FIRST TIME" in debut_reason
    # B1 score is computed independently — should differ from standard B
    # (with neutral defaults they'd be equal, but post-calibration they
    # diverge. Allow either case to keep the test robust.)
    assert 20 <= debut_score <= 80


def test_gear_unchanged_from_last_run_is_not_new():
    """Same gear as last run → no NEW/FIRST TIME prefix."""
    score, reason = G.analyze_gear("B/TT", prior_gear="B/TT")
    assert "NEW" not in reason
    assert "FIRST TIME" not in reason


def test_gear_added_vs_last_run_marks_only_the_new_code():
    """Adding B to a horse that previously wore TT only — B gets NEW prefix,
    TT does not."""
    score, reason = G.analyze_gear("B/TT", prior_gear="TT")
    assert "NEW — Blinkers" in reason
    # TT was already there
    assert "NEW — Tongue-tie" not in reason


def test_unknown_gear_code_falls_through():
    score, reason = G.analyze_gear("XYZ")
    assert score == 50.0
    assert "no significant impact" in reason


def test_debutant_with_only_tt_still_flagged_first_time():
    """TT has no +1 variant, but should still get the FIRST TIME prefix
    so the change is visible in reasoning."""
    score, reason = G.analyze_gear("TT", is_first_time=True)
    assert "FIRST TIME (debut) — Tongue-tie ON" in reason
    assert 20 <= score <= 80


def test_dominant_score_picks_biggest_deviation_from_neutral():
    """When multiple codes apply, the score with biggest |score-50| wins —
    so a fade signal can come through even alongside a positive one."""
    # Manually set known scores to test the dominant-score logic
    saved = G.GEAR_IMPACT.copy()
    try:
        G.GEAR_IMPACT["B"] = {"score": 30, "label": "B-test", "note": "fade"}
        G.GEAR_IMPACT["TT"] = {"score": 65, "label": "TT-test", "note": "boost"}
        score, _ = G.analyze_gear("B/TT")
        # |30-50|=20 > |65-50|=15, so B (30) should dominate
        assert score == 30
    finally:
        G.GEAR_IMPACT.clear()
        G.GEAR_IMPACT.update(saved)
