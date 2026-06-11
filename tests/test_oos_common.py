"""Tests for scripts/oos_common.py — the UK/FR OOS split helpers (audit M1/M2)."""
import types

import pandas as pd

from scripts.oos_common import (
    _snap_to_date_boundary, grouped_temporal_folds, cutoff_date_for,
    resolve_oos_cutoff,
)


def _df(n_dates=10, runners_per_race=8):
    """n_dates meetings, one race each, runners_per_race rows per race."""
    rows = []
    for d in range(n_dates):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=d)
        for r in range(runners_per_race):
            rows.append({"date": date, "race_id": f"R{d}", "n": r})
    return pd.DataFrame(rows).sort_values(["date", "race_id"]).reset_index(drop=True)


def test_snap_moves_to_next_date_boundary():
    df = _df(n_dates=4, runners_per_race=5)  # 20 rows, dates change every 5 rows
    # idx 3 is mid-first-date → snaps forward to 5 (start of 2nd date)
    assert _snap_to_date_boundary(df, 3) == 5
    # idx already on a boundary stays put
    assert _snap_to_date_boundary(df, 5) == 5
    # degenerate bounds
    assert _snap_to_date_boundary(df, 0) == 0
    assert _snap_to_date_boundary(df, len(df)) == len(df)


def test_folds_never_split_a_race_across_boundaries():
    df = _df(n_dates=10, runners_per_race=8)
    folds = grouped_temporal_folds(df)
    # every fold is whole-races: each race_id lives in exactly one fold
    race_to_folds = {}
    for name, fold in folds.items():
        for rid in fold["race_id"].unique():
            race_to_folds.setdefault(rid, set()).add(name)
    assert all(len(v) == 1 for v in race_to_folds.values())
    # folds partition all rows, in order, no overlap
    assert sum(len(f) for f in folds.values()) == len(df)


def test_cutoff_is_calibrate_max_and_test_is_strictly_after():
    df = _df(n_dates=10, runners_per_race=8)
    cutoff = cutoff_date_for(df)
    folds = grouped_temporal_folds(df)
    assert cutoff == folds["calibrate"]["date"].max()
    # test fold is exactly the rows strictly after the cutoff date
    assert (folds["test"]["date"] > cutoff).all()
    assert (df[df["date"] > cutoff].shape[0] == folds["test"].shape[0])


def test_resolve_prefers_frozen_cutoff_over_recompute():
    df = _df(n_dates=10, runners_per_race=8)
    # model with a frozen cutoff in training_metrics
    model = types.SimpleNamespace(training_metrics={"cutoff_date": "2026-01-03"})
    cutoff, src = resolve_oos_cutoff(model, df)
    assert cutoff == pd.Timestamp("2026-01-03")
    assert "frozen" in src
    # legacy model (no cutoff_date) falls back to recompute, clearly labelled
    legacy = types.SimpleNamespace(training_metrics={})
    cutoff2, src2 = resolve_oos_cutoff(legacy, df)
    assert cutoff2 == cutoff_date_for(df)
    assert "RECOMPUTED" in src2


def test_cutoff_date_robust_on_single_date_dataset():
    # audit pass-3: a tiny/single-date dataset can snap the calibrate fold empty.
    # cutoff_date_for must NOT return NaT (which would crash .strftime at the
    # train site) — it falls back to the max over all in-sample folds.
    df = _df(n_dates=1, runners_per_race=8)  # all rows share one date
    cutoff = cutoff_date_for(df)
    assert pd.notna(cutoff)
    assert cutoff == pd.Timestamp("2026-01-01")
    # the train site does this — must not raise
    assert cutoff.strftime("%Y-%m-%d") == "2026-01-01"


def test_frozen_cutoff_does_not_drift_when_more_data_is_appended():
    # The whole point of M1: ingesting more rows must NOT move a frozen cutoff.
    small = _df(n_dates=10, runners_per_race=8)
    model = types.SimpleNamespace(
        training_metrics={"cutoff_date": cutoff_date_for(small).strftime("%Y-%m-%d")})
    big = _df(n_dates=20, runners_per_race=8)  # simulate later --since-last ingests
    cutoff_small, _ = resolve_oos_cutoff(model, small)
    cutoff_big, _ = resolve_oos_cutoff(model, big)
    assert cutoff_small == cutoff_big  # frozen — independent of df size
    # whereas a naive recompute WOULD drift:
    assert cutoff_date_for(small) != cutoff_date_for(big)
