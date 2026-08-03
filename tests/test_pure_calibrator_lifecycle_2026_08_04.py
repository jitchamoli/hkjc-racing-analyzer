r"""`pure_calibrator.pkl` must be refit on retrain and verified at preflight.

It was added 2026-08-03 and wired into neither lifecycle hook, which left the
EDGE tier able to fail silently in two different ways:

1. **Never refit.** `dashboard/results_fetcher` re-runs `scripts.calibrate_blend`
   after every auto-learn retrain, with a comment explaining exactly why:
   a calibrator frozen against the previous model's probability distribution no
   longer corresponds to current pipeline output. `pure_prob` is a direct
   function of `rule_score` and `raw_win_probability` — both move on every
   retrain — yet `scripts.calibrate_pure` was never invoked.

2. **Not preflighted, and fails open.** `check_dashboard_ml_integrity` validates
   the model and the blend calibrator. A missing sidecar or checksum mismatch on
   the pure calibrator logs one `logger.error` and returns None, and the
   pipeline proceeds UNCALIBRATED — worth ~-10pp on every `edge_pp`, so EDGE
   stops firing all day and conviction tiers drift down, with the dashboard
   looking entirely normal.

The blend calibrator has the same fail-open loader but is backstopped by
`check_blend_calibrator_artifact`. The pure one had no backstop at all.
"""
import hashlib
import pickle

import pytest

from ml.integrity import IntegrityError, check_pure_calibrator_artifact


class TestAutoLearnRefitsIt:
    def test_results_fetcher_reruns_calibrate_pure(self):
        """Whatever reruns calibrate_blend must also rerun calibrate_pure."""
        import inspect
        from dashboard import results_fetcher

        src = inspect.getsource(results_fetcher)
        assert 'scripts.calibrate_blend' in src, 'precondition: blend refit exists'
        assert 'scripts.calibrate_pure' in src, (
            "auto-learn refits the blend calibrator but not the pure one — "
            "pure_prob is stranded against the previous model after every retrain"
        )


class TestPreflightCoversIt:
    def test_dashboard_preflight_reports_the_pure_calibrator(self):
        from ml.integrity import check_dashboard_ml_integrity
        out = check_dashboard_ml_integrity()
        assert 'pure_version' in out, (
            "check_dashboard_ml_integrity does not validate pure_calibrator — "
            "a corrupt artifact would silently disable the EDGE tier"
        )

    def test_missing_sidecar_is_a_hard_failure(self, tmp_path):
        pkl = tmp_path / 'pure_calibrator.pkl'
        pkl.write_bytes(pickle.dumps({'calibrator': None, 'metadata': {}}))
        with pytest.raises(IntegrityError):
            check_pure_calibrator_artifact(pkl)

    def test_checksum_mismatch_is_a_hard_failure(self, tmp_path):
        pkl = tmp_path / 'pure_calibrator.pkl'
        pkl.write_bytes(pickle.dumps({'calibrator': None, 'metadata': {}}))
        pkl.with_suffix('.sha256').write_text('0' * 64 + '\n')
        with pytest.raises(IntegrityError):
            check_pure_calibrator_artifact(pkl)

    def test_calibrator_without_predict_is_rejected(self, tmp_path):
        pkl = tmp_path / 'pure_calibrator.pkl'
        raw = pickle.dumps({'calibrator': object(),
                            'metadata': {'n_samples': 7685, 'n_winners': 629,
                                         'base_rate': 0.082,
                                         'version': 'x', 'trained_date': 'y'}})
        pkl.write_bytes(raw)
        pkl.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest() + '\n')
        with pytest.raises(IntegrityError):
            check_pure_calibrator_artifact(pkl)

    def test_the_live_artifact_passes(self):
        payload = check_pure_calibrator_artifact()
        assert hasattr(payload['calibrator'], 'predict')
        assert payload['metadata']['n_samples'] > 1000


class TestLoaderRefusesTampering:
    def test_checksum_mismatch_refuses_to_load(self, tmp_path, monkeypatch):
        """The loader's mismatch branch had no test at all (probe Y)."""
        import betting.edge_calculator as ec

        real_pkl = (tmp_path / 'pure_calibrator.pkl')
        real_pkl.write_bytes(pickle.dumps({'calibrator': 'tampered'}))
        real_pkl.with_suffix('.sha256').write_text('deadbeef' * 8 + '\n')

        monkeypatch.setattr(
            ec, '__file__', str(tmp_path / 'pkg' / 'edge_calculator.py'))
        (tmp_path / 'pkg').mkdir(exist_ok=True)
        (tmp_path / 'models').mkdir(exist_ok=True)
        (tmp_path / 'models' / 'pure_calibrator.pkl').write_bytes(
            real_pkl.read_bytes())
        (tmp_path / 'models' / 'pure_calibrator.sha256').write_text(
            'deadbeef' * 8 + '\n')

        ec.reset_pure_calibrator_cache()
        assert ec._load_pure_calibrator() is None, (
            'a checksum mismatch must refuse to load, not return the payload'
        )
        ec.reset_pure_calibrator_cache()

    def test_payload_missing_the_calibrator_key_is_not_cached(self, tmp_path,
                                                              monkeypatch):
        """A signed-but-keyless payload used to cache None for the process."""
        import betting.edge_calculator as ec

        (tmp_path / 'pkg').mkdir(exist_ok=True)
        (tmp_path / 'models').mkdir(exist_ok=True)
        raw = pickle.dumps({'metadata': {}})          # no 'calibrator' key
        (tmp_path / 'models' / 'pure_calibrator.pkl').write_bytes(raw)
        (tmp_path / 'models' / 'pure_calibrator.sha256').write_text(
            hashlib.sha256(raw).hexdigest() + '\n')
        monkeypatch.setattr(
            ec, '__file__', str(tmp_path / 'pkg' / 'edge_calculator.py'))

        ec.reset_pure_calibrator_cache()
        assert ec._load_pure_calibrator() is None
        assert ec._PURE_CALIBRATOR_LOADED is False, (
            'a keyless payload must not latch the cache — a repaired artifact '
            'would never be picked up for the life of the process'
        )
        ec.reset_pure_calibrator_cache()
