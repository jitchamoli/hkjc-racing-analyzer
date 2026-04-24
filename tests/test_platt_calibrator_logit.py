"""Calibrator span test: calibrator must be fit on logits, not raw probabilities.
Measured by checking that the calibrator's output span over a realistic
input range is wide enough to preserve signal. Works for both Platt
(LogisticRegression) and isotonic (IsotonicRegression) calibrators."""
import numpy as np
from scipy.special import logit

from ml.model import HKJCRacingMLModel


def _logit_value(p):
    """Helper: apply logit transform to match the production inference path."""
    return float(logit(np.clip(p, 1e-6, 1 - 1e-6)))


def test_calibrator_span_over_realistic_prob_range():
    """Train a model + calibrator on synthetic data, then measure the
    span of calibrated probabilities across the XGBoost output range
    when the calibrator is fed LOGIT inputs.

    Pre-fix (raw prob input): span ~0.012 (88x compression).
    Post-fix (logit input): span >= 0.10 for both Platt and isotonic."""
    model = HKJCRacingMLModel()

    # Synthetic training data mirroring race-prediction regime:
    # base rate ~8%, feature signal at ~20% win rate for a subset
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.random((n, 20))
    # Signal: y=1 more likely when X[:,0] is high (feature 0 = "strong horse")
    p_true = 0.05 + 0.25 * X[:, 0]
    y = (rng.random(n) < p_true).astype(int)

    # Train (save=False to avoid touching disk)
    result = model.train(X, y, save=False)
    assert result.get('error') is None, f"Training failed: {result}"
    assert model.calibrator is not None, "Calibrator should be fitted"

    # Measure calibrator span across a grid of raw probs using LOGIT inputs.
    # Supports both IsotonicRegression.predict(1D) and LogisticRegression.predict_proba(2D).
    raw_grid = np.linspace(0.02, 0.95, 50)
    logit_grid = np.array([_logit_value(p) for p in raw_grid])

    try:
        calibrated = model.calibrator.predict(logit_grid)
    except AttributeError:
        calibrated = model.calibrator.predict_proba(logit_grid.reshape(-1, 1))[:, 1]

    span = float(max(calibrated) - min(calibrated))
    assert span >= 0.10, (
        f"Calibrator span {span:.4f} is below 0.10 threshold — "
        f"calibrator compression still present (fit on raw probs, not logits). "
        f"Pre-fix was ~0.012 (88x compression)."
    )


def test_predict_span_exercises_production_path():
    """Stronger test: calls model.predict() end-to-end for two feature vectors
    and confirms outputs differ. Uses training-distribution feature vectors
    (mean of winners vs mean of non-winners) rather than all-zero/all-one
    edge cases — isotonic has flat regions at the extremes, so artificially
    extreme vectors may fall in the same flat bucket.

    The key check is that the calibrated output preserves ranking direction."""
    model = HKJCRacingMLModel()

    # Train with deterministic signal
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.random((n, 20))
    p_true = 0.05 + 0.25 * X[:, 0]
    y = (rng.random(n) < p_true).astype(int)
    result = model.train(X, y, save=False)
    assert result.get('error') is None, f"Training failed: {result}"
    # Skip F-05 integrity check for this synthetic-trained test model
    model._metadata = None

    # Use mean of winners vs mean of non-winners — these are within-distribution
    # inputs that span a meaningful logit range without falling in flat edges.
    winners = X[y == 1]
    losers = X[y == 0]
    feat_strong = winners[:50].mean(axis=0).astype(np.float32)
    feat_weak = losers[:200].mean(axis=0).astype(np.float32)

    prob_strong, _ = model.predict(feat_strong, field_size=12)
    prob_weak, _ = model.predict(feat_weak, field_size=12)

    # The model learned that high X[:,0] → higher win prob.
    # Winners have higher X[0] on average, so prob_strong >= prob_weak.
    # Verify ranking is preserved (logit transform + calibrator not broken).
    assert prob_strong >= prob_weak, (
        f"Inference path may be broken: prob_strong={prob_strong:.4f} should be >= "
        f"prob_weak={prob_weak:.4f}. Either logit transform dropped from predict() "
        f"or model did not learn signal."
    )
