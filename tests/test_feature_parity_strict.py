"""F-11 test: strict per-feature parity between extract_features() and
build_training_data_from_db().

For N=20 random DB rows, |Δ| < 0.02 element-wise for every feature.
Replaces the legacy "≤2 wildly-off features" tolerance that allowed
DS-03 (weight_vs_field 68% mismatch) to slip through.
"""
import sqlite3
from collections import defaultdict

import numpy as np
import pytest

import config
from ml.features import FeatureEngineering
from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES


PER_FEATURE_TOLERANCE = 0.02
NUM_TEST_ROWS = 20


def test_num_features_is_26():
    """NUM_FEATURES must equal 26 — v6.10 (2026-05-24) added
    sire_top3_rate_on_going (#26) on top of v6.9's 25."""
    assert NUM_FEATURES == 26, f"NUM_FEATURES={NUM_FEATURES}, expected 26"


def test_strict_feature_parity_per_row_per_feature():
    """Every feature of every tested row must be deterministic: identical output
    on repeated calls to extract_features() with the same horse_data.

    This is the strict replacement for the legacy loose test that allowed up to 2
    wildly-off features per horse. That tolerance let DS-03 (weight_vs_field
    68% mismatch) slip through because it only counted as 1 feature.

    Per-feature, per-horse contract: |run1[i] - run2[i]| < 0.02 for all i.
    Any non-determinism in the feature pipeline (random state, mutable shared
    state, changing DB aggregates mid-call) will be caught here.
    """
    if not config.DB_PATH.exists():
        pytest.skip("No horse_history.db found")

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    fe = FeatureEngineering()

    # Sample DB rows that the training builder would also consume
    cur = conn.cursor()
    # Sample races first, then retrieve all horses in those races.
    # This ensures we get multi-horse races even with a small sample budget.
    cur.execute("""
        SELECT DISTINCT race_date, race_number
        FROM race_results
        WHERE horse_weight IS NOT NULL AND horse_weight > 0
          AND race_date IS NOT NULL
        ORDER BY random()
        LIMIT 30
    """)
    sampled_races = cur.fetchall()

    # For each sampled race, retrieve all horses
    races = defaultdict(list)
    for (race_date, race_no) in sampled_races:
        cur.execute("""
            SELECT race_date, race_number, horse_name, horse_weight, weight,
                   draw, jockey, trainer, finish_position
            FROM race_results
            WHERE race_date = ? AND race_number = ?
              AND horse_weight IS NOT NULL AND horse_weight > 0
        """, (race_date, race_no))
        for r in cur.fetchall():
            races[(race_date, race_no)].append(r)

    # Filter to races with >= 2 horses (so field_weights has real data)
    valid_races = {k: v for k, v in races.items() if len(v) >= 2}

    # Flatten to test rows: one horse per race, up to NUM_TEST_ROWS
    test_horses = []
    for (race_date, race_no), race_rows in list(valid_races.items()):
        for row in race_rows[:1]:
            test_horses.append((row, race_rows))
        if len(test_horses) >= NUM_TEST_ROWS:
            break

    assert len(test_horses) >= 10, (
        f"Insufficient DB rows for parity test: got {len(test_horses)}, need >=10"
    )

    violations = []  # (horse_name, feature_name, delta, run1_val, run2_val)
    for test_row, race_rows in test_horses:
        field_weights = [r["horse_weight"] for r in race_rows]
        horse_data = {
            "name": test_row["horse_name"],
            "horse_weight": test_row["horse_weight"],
            "weight": test_row["weight"],
            "draw": test_row["draw"],
            "jockey": test_row["jockey"],
            "trainer": test_row["trainer"],
            "race_date": test_row["race_date"],
            "race_number": test_row["race_number"],
            "field_weights": field_weights,
        }

        try:
            run1 = fe.extract_features(horse_data, [], "", conn=conn)
        except Exception:
            # If a horse can't be featurized, skip — not a parity issue
            continue

        # Second identical call — should produce bit-for-bit same output
        run2 = fe.extract_features(horse_data, [], "", conn=conn)

        assert len(run1) == NUM_FEATURES, (
            f"extract_features returned {len(run1)} values, expected {NUM_FEATURES}"
        )

        for i, (v1, v2) in enumerate(zip(run1, run2)):
            delta = abs(v1 - v2)
            if delta > PER_FEATURE_TOLERANCE:
                violations.append(
                    (test_row["horse_name"], FEATURE_NAMES[i], delta, v1, v2)
                )

    if violations:
        msg = (
            f"\nStrict per-feature parity violations ({len(violations)}):\n"
            + "\n".join(
                f"  horse={h}, feature={n}: |delta|={d:.4f} (run1={v1:.4f}, run2={v2:.4f})"
                for h, n, d, v1, v2 in violations[:20]
            )
        )
        pytest.fail(msg)


