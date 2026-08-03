r"""The pure calibrator's upper tail must be clamped like the blend one's.

`models/pure_calibrator.pkl` is fit on ~7,700 samples with a base rate of 8%,
so almost nothing lands above 0.20. Measured on the 2026-08-03 refit, only 4 of
49 knots sit above 0.20 and the top one reads:

    0.2661 -> 1.0000

An isotonic asserting CERTAINTY off roughly one sample. Applied unclamped and
then renormalised, a single horse at pre-cal 0.27 takes essentially the whole
race distribution, which flows straight into `edge_pp`, `ev_win`, `overlay` and
all six EDGE gates.

This is the same pathology `_BLEND_CAL_INPUT_CLIP = 0.30` already guards on the
post-blend calibrator; the pure one was shipped without it. On held-out data 9
runners landed above pre-cal 0.20, the model rated them 40.8% against a market
implying 43.7%, only 3 of 9 actually won — and 5 of the 9 cleared the EDGE gate
on the strength of the unsupported inflation.

HONEST CAVEAT, carried over from the blend-clamp review: clamping is NOT purely
conservative. Because the output is renormalised per race, capping the top horse
redistributes the removed mass to the others. It caps a specific, measured
over-statement at the cost of a small, diffuse inflation elsewhere — the right
trade only because the tail it removes is fit on ~1 sample.
"""
import numpy as np
import pytest

import betting.edge_calculator as ec
from betting.edge_calculator import compute_pure_probs


@pytest.fixture(autouse=True)
def _reset():
    ec.reset_pure_calibrator_cache()
    yield
    ec.reset_pure_calibrator_cache()


class TestClampExists:
    def test_clip_constant_is_defined_within_the_supported_range(self):
        assert hasattr(ec, '_PURE_CAL_INPUT_CLIP')
        assert 0.10 < ec._PURE_CAL_INPUT_CLIP <= 0.30

    def test_clamp_sits_below_the_sparse_tail(self):
        """Most fitted knots must lie BELOW the clamp, or it clamps nothing."""
        iso = ec._load_pure_calibrator()
        if iso is None:
            pytest.skip('no production pure calibrator')
        x = np.asarray(iso.X_thresholds_)
        below = (x < ec._PURE_CAL_INPUT_CLIP).sum()
        assert below / len(x) >= 0.85, (
            f'only {below}/{len(x)} knots are below the clamp — it is set too '
            f'low and is discarding well-supported calibration'
        )


class TestCertaintyIsCapped:
    def test_a_standout_cannot_take_the_whole_race(self):
        """Unclamped, a pre-cal 0.27 favourite mapped to iso 1.0."""
        field = [{'rule_score': 95.0, 'ml_prob': 0.75, 'win_odds': 1.6}] + [
            {'rule_score': 40.0, 'ml_prob': 0.03, 'win_odds': 40.0}
            for _ in range(9)
        ]
        probs = compute_pure_probs(field, field_size=10)
        assert probs[0] < 0.95, (
            f'favourite took {probs[0]:.1%} of the race — the unsupported '
            f'isotonic tail is still being applied'
        )
        assert sum(probs) == pytest.approx(1.0)

    def test_clamping_is_monotone_preserving(self):
        """Clamping must never reorder the field."""
        field = [
            {'rule_score': 92.0, 'ml_prob': 0.60, 'win_odds': 2.0},
            {'rule_score': 80.0, 'ml_prob': 0.35, 'win_odds': 3.5},
            {'rule_score': 66.0, 'ml_prob': 0.16, 'win_odds': 7.0},
            {'rule_score': 50.0, 'ml_prob': 0.06, 'win_odds': 18.0},
        ]
        probs = compute_pure_probs(field, field_size=4)
        assert probs == sorted(probs, reverse=True)


class TestBelowTheClampIsUntouched:
    def test_ordinary_field_is_unchanged_by_the_clamp(self):
        """The 99% of runners below the clamp must be bit-identical."""
        field = [
            {'rule_score': 70.0, 'ml_prob': 0.14, 'win_odds': 6.0},
            {'rule_score': 62.0, 'ml_prob': 0.11, 'win_odds': 8.0},
            {'rule_score': 55.0, 'ml_prob': 0.08, 'win_odds': 12.0},
        ] + [{'rule_score': 46.0, 'ml_prob': 0.05, 'win_odds': 25.0}
             for _ in range(9)]

        iso = ec._load_pure_calibrator()
        if iso is None:
            pytest.skip('no production pure calibrator')

        clamped = compute_pure_probs(field, field_size=12)

        class _NoClamp:
            def predict(self, xs):
                return iso.predict(np.asarray(xs))

        unclamped = compute_pure_probs(field, field_size=12,
                                       calibrator=_NoClamp())
        assert clamped == pytest.approx(unclamped, rel=1e-9), (
            'the clamp altered a field that sits entirely below it'
        )
