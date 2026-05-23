"""UK + FR model SHA256 verify-then-load — shipped 2026-05-23 from the
5-expert review's CRITICAL finding (UK/FR pickle.load with no integrity
check while UK is in weekly production use via scripts.validate_uk).

Test-engineer expert review 2026-05-23 priority #4: a regression that
silently bypasses the sidecar check is a pickle-RCE security gap.
"""
import hashlib
import pickle
import tempfile
from pathlib import Path

import pytest

from ml.uk_model import UKRacingModel
from ml.fr_model import FRRacingModel


def _write_valid_pickle_with_sidecar(tmp: Path, model_cls):
    """Write a minimally-valid model pickle + matching .sha256 sidecar."""
    payload = {
        'model': None,  # we don't exercise predict — just load
        'feature_names': ['f1', 'f2'],
        'trained_date': '2026-05-23T00:00:00',
        'n_train_samples': 100,
        'training_metrics': {},
    }
    raw = pickle.dumps(payload)
    tmp.write_bytes(raw)
    sidecar = tmp.with_suffix('.sha256')
    sidecar.write_text(hashlib.sha256(raw).hexdigest())
    return tmp, sidecar


@pytest.mark.parametrize('model_cls', [UKRacingModel, FRRacingModel])
def test_load_succeeds_with_matching_sidecar(model_cls):
    with tempfile.TemporaryDirectory() as d:
        path, _ = _write_valid_pickle_with_sidecar(Path(d) / 'm.pkl', model_cls)
        # Should not raise.
        inst = model_cls.load(path)
        assert inst.feature_names == ['f1', 'f2']


@pytest.mark.parametrize('model_cls', [UKRacingModel, FRRacingModel])
def test_load_refuses_without_sidecar(model_cls):
    with tempfile.TemporaryDirectory() as d:
        path, sidecar = _write_valid_pickle_with_sidecar(
            Path(d) / 'm.pkl', model_cls)
        sidecar.unlink()  # delete the sidecar
        with pytest.raises(RuntimeError, match='checksum file missing'):
            model_cls.load(path)


@pytest.mark.parametrize('model_cls', [UKRacingModel, FRRacingModel])
def test_load_refuses_on_hash_mismatch(model_cls):
    with tempfile.TemporaryDirectory() as d:
        path, sidecar = _write_valid_pickle_with_sidecar(
            Path(d) / 'm.pkl', model_cls)
        # Tamper: rewrite the pickle with different content but keep
        # the original (now stale) sidecar.
        path.write_bytes(pickle.dumps({'model': None, 'feature_names': ['x'],
                                       'trained_date': '2099-01-01',
                                       'n_train_samples': 0,
                                       'training_metrics': {}}))
        with pytest.raises(RuntimeError, match='checksum mismatch'):
            model_cls.load(path)


@pytest.mark.parametrize('model_cls', [UKRacingModel, FRRacingModel])
def test_save_writes_sidecar(model_cls):
    """save() must write the sidecar so the next load() can verify."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'm.pkl'
        inst = model_cls()
        inst.model = None
        inst.feature_names = ['f1', 'f2']
        inst.trained_date = '2026-05-23T00:00:00'
        inst.n_train_samples = 100
        inst.training_metrics = {}
        saved = inst.save(path)
        assert saved.exists()
        assert path.with_suffix('.sha256').exists()
        # And the just-saved model should load cleanly.
        model_cls.load(path)
