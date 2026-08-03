r"""Closes the remaining coverage gaps found by the 2026-08-04 test audit.

Each test below names the exact production line whose removal it detects.
Verified by mechanically reverting that line and watching the test fail.

Also replaces three assertions that could not fail:

  * `test_default_uses_the_production_artifact` asserted `sum(probs) == 1.0`,
    which holds whether or not the calibrator loads. Disabling calibration
    entirely left it green.
  * four of five season-determinism tests pass with the wall-clock bug present,
    because today's date happens to agree with the DB's MAX(race_date). They
    stay vacuous until July 2027.
  * `scripts/calibrate_pure.py`'s two cache-disable lines had NO test. They
    were in fact deleted during the audit and nothing caught it — and their
    removal reintroduces the compounding-recalibration bug from June.
"""
import pytest


class TestApprenticeClaimIsWiredIntoTheParser:
    """`apply_apprentice_claim` was unit-tested; its ONE call site was not.

    Reverting `weight = HKJCParser.apply_apprentice_claim(weight, jockey_raw)`
    in `_parse_horse_table` restored the original train/predict skew (547/3,082
    runners had card weight != DB weight) with the full suite green.
    """

    def test_declared_weight_is_reduced_by_the_claim(self):
        import inspect
        from core.fetcher import HKJCParser

        src = inspect.getsource(HKJCParser._parse_horse_table)
        assert 'apply_apprentice_claim' in src, (
            "_parse_horse_table no longer applies the apprentice claim — "
            "carried weight will disagree with the DB for every claiming ride"
        )
        assert HKJCParser.apply_apprentice_claim('133', 'P N Wong(-7)') == '126'
        assert HKJCParser.strip_jockey_claim('P N Wong(-7)') == 'P N Wong'

    def test_non_claiming_ride_is_untouched(self):
        from core.fetcher import HKJCParser
        assert HKJCParser.apply_apprentice_claim('133', 'Z Purton') == '133'


class TestStraightQinVariantIsWired:
    """`restrict_to_straight_qin` was unit-tested; the two-line wiring was not.

    Deleting it makes the variant silently bet the full box while still being
    reported as 'straight QIN only'.
    """

    def test_variant_stakes_one_combo_not_the_box(self):
        from scripts.stake_ab import VARIANTS
        import inspect
        from scripts import stake_ab

        assert VARIANTS['straight_qin_only'].get('qin_straight_only') is True
        src = inspect.getsource(stake_ab.evaluate_date)
        assert 'restrict_to_straight_qin' in src, (
            "evaluate_date no longer restricts the straight_qin_only variant — "
            "it would bet the full box under a label claiming otherwise"
        )

    def test_restriction_collapses_a_box_to_one_combo(self):
        from scripts.stake_ab import restrict_to_straight_qin
        assert restrict_to_straight_qin({'BOX QIN-4': 6}) == {'STRAIGHT QIN': 1}
        assert restrict_to_straight_qin({'BOX TRIO-5': 10}) == {}


class TestCalibratePureFitsOnItsApplyDomain:
    """The two lines that prevent compounding recalibration.

    `collect_pairs` must disable the cached pure calibrator before collecting,
    so the isotonic is fit on the SAME uncalibrated domain it is later applied
    to. Without it each refit stacks a correction on an already-corrected
    output — the exact bug documented in project_calibration_fix_2026-06-27 for
    the blend calibrator.

    Note `reset_pure_calibrator_cache()` is NOT a substitute: it sets
    LOADED=False, which RE-ENABLES loading. The disable must set LOADED=True
    with the calibrator None, short-circuiting the loader.
    """

    def test_collect_pairs_short_circuits_the_loader(self):
        import inspect
        from scripts import calibrate_pure

        src = inspect.getsource(calibrate_pure.collect_pairs)
        assert '_PURE_CALIBRATOR' in src and '_PURE_CALIBRATOR_LOADED' in src, (
            "calibrate_pure.collect_pairs no longer disables the cached "
            "calibrator — every refit will compound the previous correction"
        )
        assert 'reset_pure_calibrator_cache' not in src, (
            "reset_pure_calibrator_cache RE-ENABLES loading (LOADED=False); "
            "it cannot be used to disable calibration during collection"
        )

    def test_disabling_the_cache_makes_the_loader_return_none(self):
        import betting.edge_calculator as ec
        ec.reset_pure_calibrator_cache()
        ec._PURE_CALIBRATOR = None
        ec._PURE_CALIBRATOR_LOADED = True          # the pattern under test
        try:
            assert ec._load_pure_calibrator() is None
        finally:
            ec.reset_pure_calibrator_cache()

    def test_reset_helper_does_the_opposite(self):
        """Guard the trap: reset re-enables, it does not disable."""
        import betting.edge_calculator as ec
        ec.reset_pure_calibrator_cache()
        assert ec._PURE_CALIBRATOR_LOADED is False
        assert ec._load_pure_calibrator() is not None, (
            'after reset the production artifact must load again'
        )


class TestPureCalibrationIsActuallyApplied:
    """Replaces an assertion that held whether or not the calibrator loaded."""

    def test_default_path_differs_from_calibration_disabled(self):
        from betting.edge_calculator import compute_pure_probs
        import betting.edge_calculator as ec

        field = [
            {'rule_score': 78.0, 'ml_prob': 0.30, 'win_odds': 3.0},
            {'rule_score': 66.0, 'ml_prob': 0.16, 'win_odds': 5.0},
            {'rule_score': 58.0, 'ml_prob': 0.10, 'win_odds': 8.0},
        ] + [{'rule_score': 46.0, 'ml_prob': 0.05, 'win_odds': 20.0}
             for _ in range(9)]

        ec.reset_pure_calibrator_cache()
        assert ec._load_pure_calibrator() is not None, (
            'the signed production artifact must load on the default path'
        )
        cal = compute_pure_probs(field, field_size=12)
        raw = compute_pure_probs(field, field_size=12, calibrator=None)
        assert cal != pytest.approx(raw), (
            'calibrated and uncalibrated output are identical — the default '
            'path is silently skipping the calibrator'
        )


class TestSeasonStartIsDataDerivedNotWallClock:
    """Four of five original tests passed WITH the bug, by wall-clock luck.

    Pinning a fake clock in a different season from the DB's MAX(race_date) is
    the only way to prove derivation rather than coincidence.
    """

    def test_ignores_a_wall_clock_in_a_different_season(self, monkeypatch,
                                                        tmp_path):
        import sqlite3
        from datetime import datetime as _real_dt
        import data.database as db

        path = tmp_path / 'h.db'
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE race_results (race_date TEXT)")
        conn.execute("INSERT INTO race_results VALUES ('2026/03/15')")
        conn.commit()

        class _FakeDT(_real_dt):
            @classmethod
            def now(cls, tz=None):
                return _real_dt(2029, 12, 25)      # a wildly different season

        monkeypatch.setattr(db, 'datetime', _FakeDT)

        d = db.HorseDatabase.__new__(db.HorseDatabase)
        d.conn = conn
        got = d._current_season_start()
        conn.close()

        assert got == '2025/07/01', (
            f"season start {got!r} — with the DB's latest race on 2026/03/15 "
            f"the season began 2025/07/01; a 2029 wall clock must not matter"
        )
