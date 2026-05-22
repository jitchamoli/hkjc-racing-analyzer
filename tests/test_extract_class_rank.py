"""Unit coverage for ml.features.extract_class_rank — the single source of truth
for HKJC class rank, used by BOTH the rule engine (analytics/class_movement.py)
and the ML class_change feature. A G1->class-1 misparse shipped historically;
these pins lock the ladder. Variants (spelled-out Group, Restricted, NR) added
2026-05-22 after a domain review found ~0.9% of rows fell through to None."""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.features import extract_class_rank as r


@pytest.mark.parametrize("s,expected", [
    # Group races sit ABOVE class 1 (rank 0/1), NOT inside the digit ladder
    ("G1", 0), ("Group 1", 0), ("Group One", 0),
    ("G2", 0), ("Group 2", 0), ("Group Two", 0),
    ("G3", 1), ("Group 3", 1), ("Group Three", 1),
    # plain class ladder
    ("1", 1), ("Class 1", 1), ("Class 4", 4), ("5", 5),
    # restricted variants (previously -> None)
    ("Class 4 (Restricted)", 4), ("Class 3 (Restricted)", 3),
    ("4R", 4), ("3R", 3),
    # griffin / maiden / age-restricted
    ("GRIFFIN", 4), ("Maiden", 4), ("4YO", 3), ("4 Year Olds", 3),
    # unparseable
    ("", None), (None, None), ("???", None),
])
def test_ladder(s, expected):
    assert r(s) == expected


def test_griffin_is_not_below_class5():
    # the old bug: GRIFFIN=6 made a Griffin->Class-4 move look like a 2-rung DROP.
    assert r("GRIFFIN") == 4 and r("Class 4") == 4   # => same class, neutral
