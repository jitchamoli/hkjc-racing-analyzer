"""Chronological race-group split helpers.

Training/evaluation rows are per horse, but leakage boundaries must be per
race. These helpers preserve row order while ensuring every runner from the
same (race_date, race_number) stays in the same fold.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Sequence

import numpy as np


def _unique_groups_in_order(groups: Sequence[str]) -> list[str]:
    return list(OrderedDict((g, None) for g in groups).keys())


def grouped_temporal_indices(groups: Sequence[str]) -> dict[str, np.ndarray]:
    """Return train/val/calib/test row indices with whole races kept together."""
    group_list = list(groups)
    if not group_list:
        empty = np.array([], dtype=int)
        return {"train": empty, "val": empty, "calib": empty, "test": empty}

    unique = _unique_groups_in_order(group_list)
    n = len(unique)
    cut_train = int(n * 0.60)
    cut_val = int(n * 0.70)
    cut_calib = int(n * 0.85)

    # Keep every fold non-empty when there are enough races. Small synthetic or
    # test arrays generally call the legacy row split by passing groups=None.
    if n >= 4:
        cut_train = max(1, min(cut_train, n - 3))
        cut_val = max(cut_train + 1, min(cut_val, n - 2))
        cut_calib = max(cut_val + 1, min(cut_calib, n - 1))

    group_fold = {}
    for i, g in enumerate(unique):
        if i < cut_train:
            fold = "train"
        elif i < cut_val:
            fold = "val"
        elif i < cut_calib:
            fold = "calib"
        else:
            fold = "test"
        group_fold[g] = fold

    out = {k: [] for k in ("train", "val", "calib", "test")}
    for i, g in enumerate(group_list):
        out[group_fold[g]].append(i)
    return {k: np.asarray(v, dtype=int) for k, v in out.items()}


def fold_metadata(groups: Sequence[str]) -> dict:
    """Compact metadata for artifact manifests and audit output."""
    idx = grouped_temporal_indices(groups)
    meta = {"n_groups": len(_unique_groups_in_order(groups)), "folds": {}}
    group_arr = np.asarray(list(groups), dtype=object)
    for name, rows in idx.items():
        fold_groups = _unique_groups_in_order(group_arr[rows].tolist()) if len(rows) else []
        meta["folds"][name] = {
            "n_rows": int(len(rows)),
            "n_groups": int(len(fold_groups)),
            "first_group": fold_groups[0] if fold_groups else None,
            "last_group": fold_groups[-1] if fold_groups else None,
        }
    return meta


def split_rows_by_race(rows: Iterable[dict]) -> dict[str, list[dict]]:
    """Split row dicts into train/val/calib/test by race key."""
    row_list = list(rows)
    groups = [
        f"{r.get('race_date', '')}-R{r.get('race_number', 0)}"
        for r in row_list
    ]
    idx = grouped_temporal_indices(groups)
    return {name: [row_list[i] for i in rows_idx] for name, rows_idx in idx.items()}
