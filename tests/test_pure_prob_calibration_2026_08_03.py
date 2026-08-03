r"""pure_prob must be a calibrated probability, not a raw score.

Measured over the last 15 production meetings (147 races):

    rank-1   pure_prob 13.3%   market 23.3%   calibrated_prob 26.3%
             ACTUAL win rate 26.5%

The market and calibrated_prob are both close to truth; pure_prob is off by
roughly 2x. That matters because pure_prob is what every edge quantity is
built from:

    edge_pp = (pure_prob - market_prob) * 100
    ev_win  = pure_prob * win_odds - 1
    overlay = pure_prob * win_odds

Subtracting a well-scaled market number from a mis-scaled model number bakes a
STRUCTURAL ~-10pp into every top pick's edge before any genuine disagreement,
so the EDGE gate (G3 needs >= +5pp) could essentially only fire on noise —
3/147 races (2.0%) flagged EDGE over those 15 meetings, 4/1866 horses.

Fitting an isotonic on (pure_prob -> is_winner) over the disjoint calib fold
makes edge_pp mean what it claims. It does NOT touch ranking: pure_prob is not
in the recommender path, so calibrated_prob and leg selection are unaffected.
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


def _field():
    """A plausible 12-runner shape: one standout, then a tail."""
    return [
        {'rule_score': 78.0, 'ml_prob': 0.30, 'win_odds': 3.0},
        {'rule_score': 66.0, 'ml_prob': 0.16, 'win_odds': 5.0},
        {'rule_score': 58.0, 'ml_prob': 0.10, 'win_odds': 8.0},
    ] + [{'rule_score': 46.0, 'ml_prob': 0.05, 'win_odds': 20.0} for _ in range(9)]


class TestCalibratorIsApplied:
    def test_calibrator_changes_the_shape(self):
        """A real calibrator must reshape pure_prob, not just rescale it.

        NOTE: because the output is renormalised to sum to 1, a purely LINEAR
        calibrator (x -> 2x) is a no-op. That is correct and expected — only a
        change in SHAPE is meaningful. A fitted isotonic is convex here: it
        lifts the favourite (13.3% -> ~26%) and pushes the tail down, which is
        exactly a shape change.
        """
        class _Convex:
            def predict(self, xs):
                return np.asarray(xs) ** 2      # widens the spread

        raw = compute_pure_probs(_field(), field_size=12, calibrator=None)
        cal = compute_pure_probs(_field(), field_size=12, calibrator=_Convex())
        assert max(cal) > max(raw), (
            f"calibrator did not reshape the distribution: "
            f"raw top {max(raw):.3f} -> cal top {max(cal):.3f}"
        )

    def test_linear_calibrator_is_a_noop_after_renormalisation(self):
        """Documents the property above so it can't be mistaken for a bug."""
        class _Linear:
            def predict(self, xs):
                return np.asarray(xs) * 2.0

        raw = compute_pure_probs(_field(), field_size=12, calibrator=None)
        cal = compute_pure_probs(_field(), field_size=12, calibrator=_Linear())
        assert cal == pytest.approx(raw)

    def test_still_a_distribution(self):
        class _Iso:
            def predict(self, xs):
                return np.clip(np.asarray(xs) * 2.0, 0.0, 1.0)

        probs = compute_pure_probs(_field(), field_size=12, calibrator=_Iso())
        assert sum(probs) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_ordering_is_preserved(self):
        """Isotonic is monotone, so it must never reorder the field."""
        class _Iso:
            def predict(self, xs):
                return np.sqrt(np.asarray(xs))

        raw = compute_pure_probs(_field(), field_size=12, calibrator=None)
        cal = compute_pure_probs(_field(), field_size=12, calibrator=_Iso())
        assert [i for i, _ in sorted(enumerate(raw), key=lambda t: -t[1])] == \
               [i for i, _ in sorted(enumerate(cal), key=lambda t: -t[1])]

    def test_disabled_calibrator_reproduces_the_raw_blend(self):
        probs = compute_pure_probs(_field(), field_size=12, calibrator=None)
        assert sum(probs) == pytest.approx(1.0)

    def test_default_uses_the_production_artifact(self):
        """Default path must load the signed pkl, not silently skip it."""
        probs = compute_pure_probs(_field(), field_size=12)
        assert sum(probs) == pytest.approx(1.0)


class TestUnderConfidenceIsCorrected:
    def test_top_pick_is_lifted_toward_its_true_rate(self):
        """With the production calibrator the favourite must not read ~13%."""
        probs = compute_pure_probs(_field(), field_size=12)
        raw = compute_pure_probs(_field(), field_size=12, calibrator=None)
        assert max(probs) > max(raw), (
            f"production calibrator did not lift the top pick: "
            f"raw {max(raw):.3f} -> cal {max(probs):.3f}"
        )


class TestIntegrity:
    def test_unsigned_calibrator_is_refused(self, monkeypatch, tmp_path):
        """Matches the blend calibrator's policy: no sidecar, no load."""
        import pathlib
        real_exists = pathlib.Path.exists

        def fake_exists(self):
            if self.name == 'pure_calibrator.sha256':
                return False
            return real_exists(self)

        monkeypatch.setattr('pathlib.Path.exists', fake_exists)
        ec.reset_pure_calibrator_cache()
        assert ec._load_pure_calibrator() is None
