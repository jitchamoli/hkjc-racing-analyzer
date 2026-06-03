"""Guard: a retrain that does NOT raise but also does NOT write a new model
(e.g. _save_model silently refuses below MIN_SAVE_SAMPLES on a 200-999 row DB)
must NOT delete the backup — otherwise the production model is permanently lost
with nothing on disk.

Bug (10-expert review 2026-06-03): train_ml_model.main() renamed the model to
.bak, called model.train(), and on the no-exception path unconditionally
unlinked the .bak files. _save_model() returns silently (no raise) when
n_samples < MIN_SAVE_SAMPLES, so the success path deleted the only copy.

Fix: _finalize_model_backup() drops the backup ONLY if a new model exists on
disk; otherwise it restores the backup.
"""
from train_ml_model import _finalize_model_backup


def _mk(p, text="x"):
    p.write_text(text)


def test_backup_removed_when_new_model_written(tmp_path):
    """Normal success: a new model exists -> backup is cleaned up."""
    model_file = tmp_path / "m.pkl"
    sha_file = tmp_path / "m.sha256"
    backup = tmp_path / "m.pkl.bak"
    sha_backup = tmp_path / "m.sha256.bak"
    # Simulate: old model backed up, then a NEW model written by train().
    _mk(backup, "OLD"); _mk(sha_backup, "OLDSHA")
    _mk(model_file, "NEW"); _mk(sha_file, "NEWSHA")

    wrote = _finalize_model_backup(model_file, sha_file, backup, sha_backup)

    assert wrote is True
    assert model_file.read_text() == "NEW"
    assert not backup.exists()
    assert not sha_backup.exists()


def test_exception_after_save_keeps_new_model(tmp_path):
    """Round-2 finding: a post-save audit (corr/constant-feature/log I/O) can
    raise AFTER _save_model() wrote a valid new model. main()'s except block now
    routes through _finalize_model_backup, so a model present at failure time is
    KEPT (not clobbered by the old backup) and the operator isn't silently
    rolled back. Same contract as the success-with-model path."""
    model_file = tmp_path / "m.pkl"
    sha_file = tmp_path / "m.sha256"
    backup = tmp_path / "m.pkl.bak"
    sha_backup = tmp_path / "m.sha256.bak"
    # New model was saved; THEN a post-save step raised. Backup still on disk.
    _mk(backup, "OLD"); _mk(sha_backup, "OLDSHA")
    _mk(model_file, "NEW"); _mk(sha_file, "NEWSHA")

    kept = _finalize_model_backup(model_file, sha_file, backup, sha_backup)

    assert kept is True, "freshly-saved model must be kept when an audit raises after save"
    assert model_file.read_text() == "NEW", "new model was clobbered by the old backup"
    assert not backup.exists() and not sha_backup.exists()


def test_backup_restored_when_no_new_model(tmp_path):
    """Silent-refuse: train() didn't raise but wrote no model -> the backup
    MUST be restored, never deleted (the regression this test pins)."""
    model_file = tmp_path / "m.pkl"
    sha_file = tmp_path / "m.sha256"
    backup = tmp_path / "m.pkl.bak"
    sha_backup = tmp_path / "m.sha256.bak"
    # Simulate: old model backed up; _save_model() silently refused -> no model_file.
    _mk(backup, "OLD"); _mk(sha_backup, "OLDSHA")
    assert not model_file.exists()

    wrote = _finalize_model_backup(model_file, sha_file, backup, sha_backup)

    assert wrote is False
    assert model_file.exists(), "production model was LOST — backup not restored"
    assert model_file.read_text() == "OLD"
    assert sha_file.read_text() == "OLDSHA"
    assert not backup.exists()
    assert not sha_backup.exists()
