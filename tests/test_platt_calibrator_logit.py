"""F-01 test: Platt calibrator must be fit on logits, not raw probabilities.
Measured by checking that the calibrator's output span over a realistic
input range is wide enough to preserve signal."""
import numpy as np

from ml.model import HKJCRacingMLModel


def _logit_input(p):
    """Helper: the production code applies logit before calibrator.predict_proba.
    This matches the F-01 inference path in ml/model.py."""
    from scipy.special import logit
    return np.array([[logit(np.clip(p, 1e-6, 1 - 1e-6))]])


def test_calibrator_span_over_realistic_prob_range():
    """Train a model + calibrator on synthetic data, then measure the
    span of calibrated probabilities across the XGBoost output range
    when the calibrator is fed LOGIT inputs (post-F-01 behavior).

    Pre-fix: calibrator fit on raw probs → span ~0.012 (88x compression).
    Post-fix: calibrator fit on logits → span >= 0.10 when given logit inputs."""
    model = HKJCRacingMLModel()

    # Synthetic training data mirroring race-prediction regime:
    # base rate ~8%, feature signal at ~20% win rate for a subset
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.random((n, 23))
    # Signal: y=1 more likely when X[:,0] is high (feature 0 = "strong horse")
    p_true = 0.05 + 0.25 * X[:, 0]
    y = (rng.random(n) < p_true).astype(int)

    # Train (save=False to avoid touching disk)
    result = model.train(X, y, save=False)
    assert result.get('error') is None, f"Training failed: {result}"
    assert model.calibrator is not None, "Calibrator should be fitted"

    # Measure calibrator span across a grid of raw probs using LOGIT inputs
    # (the new F-01 behavior). This test passes iff the calibrator was fit
    # on logits; if it was fit on raw probs, logit inputs will produce
    # a weak/compressed output because logits are far outside [0,1].
    raw_grid = np.linspace(0.02, 0.95, 50)

    calibrated = []
    for rp in raw_grid:
        cp = model.calibrator.predict_proba(_logit_input(rp))[0, 1]
        calibrated.append(cp)

    span = max(calibrated) - min(calibrated)
    assert span >= 0.10, (
        f"Calibrator span {span:.4f} is below 0.10 threshold — "
        f"Platt compression still present (calibrator fit on raw probs, not logits). "
        f"Pre-fix was ~0.012 (88x compression)."
    )
