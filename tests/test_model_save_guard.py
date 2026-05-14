"""Tests for F-04 pytest-overwrite guard: synthetic training must not
overwrite the production pickle."""
import hashlib
import numpy as np
import pytest

from ml.model import HKJCRacingMLModel
from ml.feature_schema import NUM_FEATURES


def test_train_with_save_false_does_not_write_pickle(tmp_path, monkeypatch):
    """When train(save=False) is called, the pickle on disk must be untouched."""
    pkl_path = tmp_path / "model.pkl"
    sha_path = tmp_path / "model.sha256"
    monkeypatch.setattr(HKJCRacingMLModel, "MODEL_FILE", pkl_path)
    monkeypatch.setattr(HKJCRacingMLModel, "MODEL_DIR", tmp_path)

    sentinel = b"PREEXISTING_PICKLE_DO_NOT_OVERWRITE"
    pkl_path.write_bytes(sentinel)
    pre_sha = hashlib.sha256(sentinel).hexdigest()
    sha_path.write_text(pre_sha)

    model = HKJCRacingMLModel()
    rng = np.random.default_rng(42)
    # Bumped from 500 to 1500 samples so stratified train/val/test split
    # always contains positives in every fold (sklearn complains otherwise).
    X = rng.random((1500, NUM_FEATURES))
    y = (rng.random(1500) < 0.10).astype(int)

    model.train(X, y, save=False)

    assert pkl_path.read_bytes() == sentinel, "train(save=False) overwrote the pickle"
    assert sha_path.read_text() == pre_sha, "train(save=False) overwrote the sha256"


def test_save_model_refuses_below_min_samples(tmp_path, monkeypatch):
    """_save_model() must refuse to write when the model was trained on too few samples."""
    pkl_path = tmp_path / "model.pkl"
    monkeypatch.setattr(HKJCRacingMLModel, "MODEL_FILE", pkl_path)
    monkeypatch.setattr(HKJCRacingMLModel, "MODEL_DIR", tmp_path)

    sentinel = b"PREEXISTING"
    pkl_path.write_bytes(sentinel)

    # Pre-seed sha file with sentinel too so we can pin both side-effects
    pre_sha = hashlib.sha256(sentinel).hexdigest()
    sha_path = tmp_path / "model.sha256"
    sha_path.write_text(pre_sha)

    model = HKJCRacingMLModel()
    rng = np.random.default_rng(42)
    X = rng.random((300, NUM_FEATURES))  # below 1000 MIN_SAVE_SAMPLES
    y = (rng.random(300) < 0.08).astype(int)

    model.train(X, y, save=True)

    assert pkl_path.read_bytes() == sentinel, "_save_model wrote a pickle trained on <1000 samples"
    assert sha_path.read_text() == pre_sha, "_save_model wrote a sha256 despite <1000-sample guard"
