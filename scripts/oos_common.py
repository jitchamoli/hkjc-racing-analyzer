"""Shared out-of-sample split helpers for the UK/FR forward-validation pipelines.

Two invariants this module enforces (audit 2026-06-09 pass-2 M1/M2):

1. Fold boundaries snap to a DATE boundary so a race is never split across
   folds — the same in-race-leak fix the HK pipeline got in b0da85a
   (rows→race-grouped folds). A date boundary is also a race boundary, so this
   covers both.

2. The train/test cutoff is FROZEN at train time (persisted in the model
   payload's training_metrics["cutoff_date"]), NOT recomputed from the live DB
   row count. Recomputing `int(len(df) * 0.85)` drifts forward on every
   `ingest --since-last`, silently shrinking the reported OOS window and
   over-counting in-sample rows.
"""
from __future__ import annotations

import pandas as pd


def _snap_to_date_boundary(df: pd.DataFrame, idx: int) -> int:
    """Move idx forward to the first row whose date differs from row idx-1's, so
    a fold boundary never falls in the middle of a race or a date."""
    if idx <= 0 or idx >= len(df):
        return idx
    d = df.iloc[idx - 1]["date"]
    while idx < len(df) and df.iloc[idx]["date"] == d:
        idx += 1
    return idx


def grouped_temporal_folds(df: pd.DataFrame) -> dict:
    """4-fold temporal split (60/10/15/15) with every boundary snapped to a date
    boundary. df must already be sorted by ["date", "race_id"]."""
    n = len(df)
    cut_train = _snap_to_date_boundary(df, int(n * 0.60))
    cut_es = _snap_to_date_boundary(df, int(n * 0.70))
    cut_cal = _snap_to_date_boundary(df, int(n * 0.85))
    return {
        "train": df.iloc[:cut_train],
        "earlystop": df.iloc[cut_train:cut_es],
        "calibrate": df.iloc[cut_es:cut_cal],
        "test": df.iloc[cut_cal:],
    }


def cutoff_date_for(df: pd.DataFrame) -> pd.Timestamp:
    """Highest in-sample date = last date of all non-test (train+earlystop+
    calibrate) folds. Anything strictly greater is true OOS. Call at TRAIN time
    to freeze the cutoff.

    On a tiny/single-date dataset a fold can snap empty, so we take the max over
    all three in-sample folds (not just calibrate) to avoid an empty .max() ->
    NaT, which would crash .strftime() at the train site."""
    folds = grouped_temporal_folds(df)
    in_sample = pd.concat([folds["train"], folds["earlystop"], folds["calibrate"]])
    if in_sample.empty:
        # Degenerate: everything snapped into the test fold (or df empty). Fall
        # back to the global max so callers still get a usable Timestamp.
        return df["date"].max() if not df.empty else pd.NaT
    return in_sample["date"].max()


def resolve_oos_cutoff(model, df: pd.DataFrame):
    """Prefer the cutoff frozen in the model payload at train time; fall back to
    recomputing from df (with a clear source label) for legacy models that
    pre-date the frozen cutoff. Returns (cutoff_timestamp, source_str)."""
    frozen = (getattr(model, "training_metrics", None) or {}).get("cutoff_date")
    if frozen:
        return pd.to_datetime(frozen), "frozen at train time (model payload)"
    return (cutoff_date_for(df),
            "RECOMPUTED from live row count — legacy model, retrain to freeze")
