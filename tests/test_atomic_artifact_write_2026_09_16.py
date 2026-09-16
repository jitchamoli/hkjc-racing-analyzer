"""Model + calibrator saves must not be able to leave a half-written artifact.

Before this, all three writers did:

    path.write_bytes(pickle.dumps(payload))      # <- crash / full disk here
    sha_path.write_text(sha256(...).hexdigest()) # <- never runs

A crash mid-write leaves a TRUNCATED pickle whose sidecar still matches the
previous model, so the integrity guard refuses to load it — on race day that
means no card until a backup is restored. Writing to temp files first and
renaming makes the truncated-pickle case impossible.
"""
import hashlib
import pickle

import pytest

from ml.integrity import _read_verified_pickle, write_signed_bytes


def test_writes_artifact_and_matching_sidecar(tmp_path):
    p = tmp_path / 'model.pkl'
    raw = pickle.dumps({'hello': 'world'})

    write_signed_bytes(p, raw)

    assert p.read_bytes() == raw
    assert p.with_suffix('.sha256').read_text().strip() == hashlib.sha256(raw).hexdigest()
    assert _read_verified_pickle(p, 'test artifact') == {'hello': 'world'}


def test_leaves_no_temp_files_behind(tmp_path):
    p = tmp_path / 'model.pkl'
    write_signed_bytes(p, pickle.dumps({'a': 1}))
    assert sorted(f.name for f in tmp_path.iterdir()) == ['model.pkl', 'model.sha256']


def test_replaces_a_previous_artifact_and_its_sidecar(tmp_path):
    p = tmp_path / 'model.pkl'
    write_signed_bytes(p, pickle.dumps({'v': 1}))
    write_signed_bytes(p, pickle.dumps({'v': 2}))

    assert _read_verified_pickle(p, 'test artifact') == {'v': 2}


def test_a_failed_write_leaves_the_previous_model_loadable(tmp_path, monkeypatch):
    """The point of the fix: a crash during the save must not brick the model."""
    p = tmp_path / 'model.pkl'
    write_signed_bytes(p, pickle.dumps({'v': 'good'}))

    real_write = type(p).write_bytes

    def boom(self, data):
        if self.suffix == '.tmp':
            raise OSError("No space left on device")
        return real_write(self, data)

    monkeypatch.setattr(type(p), 'write_bytes', boom)
    with pytest.raises(OSError):
        write_signed_bytes(p, pickle.dumps({'v': 'half-written'}))

    assert _read_verified_pickle(p, 'test artifact') == {'v': 'good'}
    assert not [f for f in tmp_path.iterdir() if f.suffix == '.tmp']
