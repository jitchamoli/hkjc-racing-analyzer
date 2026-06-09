"""Regression tests for the 2026-06-09 pass-3 audit fixes + the test-gaps that
let those bugs ship green.

- Gap 1 / Fix 2: QIN EV must be computed from pure_prob, not calibrated_prob.
- Gap 2: a pickle whose n_features != NUM_FEATURES must be refused at load AND
  the downstream predict() must fail closed (not silently use a base rate).
"""
import hashlib
import pickle

import numpy as np
import pytest

from ml.feature_schema import NUM_FEATURES


# ---------------------------------------------------------------------------
# Gap 1 / Fix 2 — QIN EV uses pure_prob, not calibrated_prob.
# The bug: dashboard/generator.py fed calibrated_prob (market-blended) to
# quinella_bet_ev, dragging P(hit) toward the market and understating EV.
# This pins the CONTRACT that the two prob sources give materially different EV,
# so a regression to calibrated_prob is detectable, and documents the direction.
# ---------------------------------------------------------------------------
def test_quinella_ev_depends_on_which_prob_is_passed():
    from betting.dividends import quinella_bet_ev, normalize_market_probs

    win_odds = [3.0, 4.0, 6.0, 8.0, 12.0, 20.0]
    market_probs = normalize_market_probs(win_odds)
    # Model strongly disagrees with the market on the top pair (we rate them
    # much higher than the market) — this is the case the edge math exists for.
    pure_probs = [0.40, 0.30, 0.10, 0.08, 0.07, 0.05]
    # calibrated = pure blended toward market → attenuated separation
    calibrated_probs = [(p + m) / 2 for p, m in zip(pure_probs, market_probs)]
    pairs = [(0, 1)]

    ev_pure = quinella_bet_ev(pure_probs, market_probs, pairs)
    ev_cal = quinella_bet_ev(calibrated_probs, market_probs, pairs)

    # The two must differ (else the bug would be invisible)...
    assert ev_pure["hit_prob"] != ev_cal["hit_prob"]
    # ...and pure_prob (our true, un-blended opinion on a pair we like) yields a
    # higher P(hit) and EV than the market-dragged calibrated prob.
    assert ev_pure["hit_prob"] > ev_cal["hit_prob"]
    assert ev_pure["ev_pct"] > ev_cal["ev_pct"]


def test_generator_qin_ev_block_uses_pure_prob():
    # Guard the wiring: the QIN EV block must pull pure_prob for the EV call.
    # (model_probs stays calibrated because it drives the conviction-tier shares.)
    import inspect
    import dashboard.generator as g
    src = inspect.getsource(g.build_race_data)
    assert "quinella_bet_ev(pure_probs" in src, \
        "QIN EV must be computed from pure_prob (audit pass-3 Fix 2)"
    assert "quinella_bet_ev(model_probs" not in src, \
        "QIN EV must NOT use calibrated model_probs"


# ---------------------------------------------------------------------------
# Gap 2 — n_features mismatch must fail closed end-to-end.
# ---------------------------------------------------------------------------
def test_n_features_mismatch_refuses_load_and_predict_fails_closed(tmp_path):
    from ml.model import HKJCRacingMLModel

    # Build a pickle that is structurally fine but declares the WRONG n_features.
    payload = {
        "model": object(),
        "place_model": object(),
        "calibrator": None,
        "place_calibrator": None,
        "version": "bad-nfeat",
        "n_features": NUM_FEATURES - 1,   # <-- mismatch
        "feature_names": [f"f{i}" for i in range(NUM_FEATURES - 1)],
        "n_samples": 5000,
        "trained_date": "2026-06-08T00:00:00",
    }
    pkl = tmp_path / "bad_model.pkl"
    raw = pickle.dumps(payload)
    pkl.write_bytes(raw)
    # valid sidecar so we get PAST the checksum gate and reach the n_features gate
    pkl.with_suffix(".sha256").write_text(hashlib.sha256(raw).hexdigest())

    m = HKJCRacingMLModel()
    # Neutralize any real model loaded by __init__, then load the bad pickle.
    m.is_trained = False
    m.model = None
    m.MODEL_FILE = pkl
    m._load_model()

    # Load must be REFUSED (mismatch → stays untrained), not silently accepted.
    assert m.is_trained is False
    # And the downstream predict must fail closed (F-17), not return a base rate.
    with pytest.raises(RuntimeError):
        m.predict(np.zeros(NUM_FEATURES), field_size=12)
