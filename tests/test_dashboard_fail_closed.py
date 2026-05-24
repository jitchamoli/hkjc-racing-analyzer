import asyncio
from datetime import datetime, timedelta
import hashlib
import pickle
import sys

import pytest

from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES
from ml.integrity import (
    IntegrityError,
    check_dashboard_ml_integrity,
    check_production_model_artifact,
)


class _FakePredictor:
    def predict(self, values):
        return values


def _write_signed_pickle(path, payload, write_sha=True):
    raw = pickle.dumps(payload)
    path.write_bytes(raw)
    if write_sha:
        path.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest())


def _valid_model_payload():
    trained_date = (datetime.now() - timedelta(days=1)).isoformat(timespec='seconds')
    return {
        'model': object(),
        'place_model': object(),
        'calibrator': object(),
        'place_calibrator': object(),
        'version': 'test-model',
        'n_features': NUM_FEATURES,
        'feature_names': list(FEATURE_NAMES),
        'n_samples': 2000,
        'trained_date': trained_date,
        'fold_metadata': {
            'n_groups': 10,
            'folds': {
                'train': {'n_rows': 60, 'n_groups': 6,
                          'first_group': '2026/01/01-1', 'last_group': '2026/01/06-1'},
                'val': {'n_rows': 10, 'n_groups': 1,
                        'first_group': '2026/01/07-1', 'last_group': '2026/01/07-1'},
                'calib': {'n_rows': 15, 'n_groups': 2,
                          'first_group': '2026/01/08-1', 'last_group': '2026/01/09-1'},
                'test': {'n_rows': 15, 'n_groups': 1,
                         'first_group': '2026/01/10-1', 'last_group': '2026/01/10-1'},
            },
        },
    }


def test_production_model_artifact_requires_grouped_fold_metadata(tmp_path):
    model_path = tmp_path / 'hkjc_racing_model.pkl'
    payload = _valid_model_payload()
    payload.pop('fold_metadata')
    _write_signed_pickle(model_path, payload)

    with pytest.raises(IntegrityError, match='fold_metadata'):
        check_production_model_artifact(model_path)


def test_grouped_fold_metadata_accepts_intra_day_two_digit_race_boundary(tmp_path):
    # ST Sundays have R10/R11. Lex compare would falsely flag '2026/01/06-10'
    # <= '2026/01/06-9' as "folds overlap". The race-number-aware sort key
    # must accept this boundary.
    model_path = tmp_path / 'hkjc_racing_model.pkl'
    payload = _valid_model_payload()
    payload['fold_metadata'] = {
        'n_groups': 4,
        'folds': {
            'train': {'n_rows': 12, 'n_groups': 1,
                      'first_group': '2026/01/06-1', 'last_group': '2026/01/06-9'},
            'val': {'n_rows': 12, 'n_groups': 1,
                    'first_group': '2026/01/06-10', 'last_group': '2026/01/06-10'},
            'calib': {'n_rows': 12, 'n_groups': 1,
                      'first_group': '2026/01/06-11', 'last_group': '2026/01/06-11'},
            'test': {'n_rows': 12, 'n_groups': 1,
                     'first_group': '2026/01/07-1', 'last_group': '2026/01/07-1'},
        },
    }
    _write_signed_pickle(model_path, payload)
    check_production_model_artifact(model_path)  # must not raise


def test_dashboard_ml_integrity_requires_blend_calibrator_sidecar(tmp_path):
    model_path = tmp_path / 'hkjc_racing_model.pkl'
    blend_path = tmp_path / 'blend_calibrator.pkl'
    _write_signed_pickle(model_path, _valid_model_payload())
    _write_signed_pickle(
        blend_path,
        {
            'calibrator': _FakePredictor(),
            'metadata': {
                'n_samples': 2000,
                'n_winners': 160,
                'base_rate': 0.08,
                'trained_date': (datetime.now() - timedelta(days=1)).isoformat(timespec='seconds'),
                'version': 'test-blend',
            },
        },
        write_sha=False,
    )

    with pytest.raises(IntegrityError, match='checksum sidecar missing'):
        check_dashboard_ml_integrity(model_path, blend_path)


def test_dashboard_main_aborts_before_fetch_when_preflight_fails(monkeypatch):
    from dashboard import generator

    constructed = []

    def fail_preflight():
        raise IntegrityError("bad model")

    class FetcherShouldNotStart:
        def __init__(self):
            constructed.append('fetcher')

    class AnalyzerShouldNotStart:
        def __init__(self, *args, **kwargs):
            constructed.append('analyzer')

    monkeypatch.setattr(generator, 'preflight_dashboard_ml', fail_preflight)
    monkeypatch.setattr(generator, 'HKJCDataFetcher', FetcherShouldNotStart)
    monkeypatch.setattr(generator, 'CombinedRaceAnalyzer', AnalyzerShouldNotStart)
    monkeypatch.setattr(sys, 'argv', ['generate_predictions.py', '2026/05/24'])

    with pytest.raises(SystemExit) as exc:
        asyncio.run(generator.main())

    assert exc.value.code == 2
    assert constructed == []
