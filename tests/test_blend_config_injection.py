"""DI refactor (2026-06-07): the model/market blend weight and the post-blend
calibrator are injectable into calibrate_probabilities / assign_bet_recommendations
via BlendConfig + the calibrator arg, so blend research no longer needs to
monkeypatch module globals. Defaults must reproduce production behaviour exactly.
"""
import numpy as np

from core.scorer import (
    _dynamic_blend_weights, BlendConfig, calibrate_probabilities,
    USE_PRODUCTION_CALIBRATOR, _load_blend_calibrator,
)


class _Horse:
    def __init__(self, odds):
        self.win_odds = odds


class _Analysis:
    def __init__(self, cp, odds):
        self.combined_probability = cp
        self.horse = _Horse(odds)
        self.model_probability = None
        self.expected_return = None


def _mk():
    # model and market disagree: model loves A (5.0), market favours B (2.0)
    return [_Analysis(0.60, 5.0), _Analysis(0.25, 2.0), _Analysis(0.15, 8.0)]


def test_default_cfg_identical_to_no_cfg():
    for fs, dist in [(12, 1200), (8, 1800), (14, 1000), (6, 2400)]:
        assert _dynamic_blend_weights(fs, dist) == \
               _dynamic_blend_weights(fs, dist, BlendConfig())


def test_more_market_config_lowers_model_weight():
    base_mw, _ = _dynamic_blend_weights(12, 1200)
    more_mkt_mw, _ = _dynamic_blend_weights(12, 1200, BlendConfig(base_model_w=0.30))
    assert more_mkt_mw < base_mw


def test_blend_config_shifts_probs_toward_market():
    # calibrator disabled so we observe the raw blend effect
    default = calibrate_probabilities(_mk(), field_size=3, distance=1200,
                                      calibrator=None)
    more_mkt = calibrate_probabilities(_mk(), field_size=3, distance=1200,
                                       blend_config=BlendConfig(base_model_w=0.30),
                                       calibrator=None)
    # B is the market favourite (odds 2.0) → more market weight lifts its prob
    assert more_mkt[1] > default[1]
    assert abs(sum(more_mkt) - 1.0) < 1e-9


def test_injected_calibrator_is_applied():
    class _ConstCal:
        def predict(self, arr):
            return np.full(len(arr), 0.5)
    injected = calibrate_probabilities(_mk(), field_size=3, distance=1200,
                                       calibrator=_ConstCal())
    none_cal = calibrate_probabilities(_mk(), field_size=3, distance=1200,
                                       calibrator=None)
    # Constant calibrator → uniform after renorm; None → non-uniform.
    # Tolerance is 1e-5 rather than 1e-9 because calibrate_probabilities adds
    # _BLEND_CAL_TIEBREAK_EPS (1e-6) x the blended input to the isotonic
    # output, so a constant calibrator now yields a near-uniform distribution
    # that is strictly ordered instead of exactly tied (2026-08-03). Without
    # that term the coarse production isotonic collapsed distinct horses onto
    # identical probabilities and box leg cutoffs fell through to the
    # pre_blend_rank tiebreak. Still ~4 orders of magnitude tighter than the
    # non-calibrated spread asserted below, so a real regression still trips.
    assert max(injected) - min(injected) < 1e-5
    assert max(none_cal) - min(none_cal) > 1e-3


def test_default_args_output_identical():
    # The core invariant of the refactor: default args must be byte-identical
    # to explicitly passing BlendConfig(). calibrator pinned to None to isolate
    # the blend path (independent of whether the prod calibrator file exists).
    default = calibrate_probabilities(_mk(), field_size=12, distance=1200,
                                      calibrator=None)
    explicit = calibrate_probabilities(_mk(), field_size=12, distance=1200,
                                       blend_config=BlendConfig(), calibrator=None)
    assert default == explicit


def test_sentinel_routes_to_production_loader():
    # USE_PRODUCTION_CALIBRATOR (default) must route to _load_blend_calibrator().
    # Asserting equality with the explicitly-loaded calibrator proves the
    # routing regardless of whether that loader returns None or a real object
    # (env-independent — works with or without models/blend_calibrator.pkl).
    default = calibrate_probabilities(_mk(), field_size=3, distance=1200)
    explicit = calibrate_probabilities(_mk(), field_size=3, distance=1200,
                                       calibrator=_load_blend_calibrator())
    assert default == explicit
    assert abs(sum(default) - 1.0) < 1e-9
