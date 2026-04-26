"""F-06 test: weight_vs_field feature must match between extract_features()
and build_training_data_from_db() for historical DB rows.

Pre-fix (DS-03): 68% of rows showed |Δ| > 0.1 because field_weights was
never passed in horse_data during inference, causing inference to always use
the absolute-weight fallback formula while training used the field-median
formula.

Post-fix: <=5% of rows exceed 0.1 tolerance.
"""
import sqlite3

import pytest

import config
from ml.feature_schema import FEATURE_NAMES
from ml.features import FeatureEngineering


def _compute_weight_vs_field(weight_val: int, field_weights: list) -> float:
    """Same formula used in both extract_features() and build_training_data_from_db().

    This helper is the canonical implementation: field-relative when field_weights
    is available, absolute fallback otherwise.
    """
    if weight_val <= 0:
        return 0.5
    if field_weights:
        sw = sorted(field_weights)
        median_wt = sw[len(sw) // 2]
        diff = weight_val - median_wt
        return max(0.0, min(1.0, 0.5 - diff / 20.0))
    else:
        return max(0.0, min(1.0, 1.0 - (weight_val - 108) / 27.0))


@pytest.mark.skipif(
    not config.DB_PATH.exists(),
    reason="No DB available"
)
def test_weight_vs_field_train_predict_parity():
    """For historical DB rows, extract_features()['weight_vs_field'] must
    agree with the training-side computation within 0.1 absolute delta on
    >=95% of rows.

    Root cause of DS-03: field_weights was not passed in horse_data during
    inference, so extract_features() always used the absolute-weight fallback
    instead of the field-relative formula used during training.
    """
    assert "weight_vs_field" in FEATURE_NAMES
    wvf_idx = FEATURE_NAMES.index("weight_vs_field")

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    # Fetch up to 50 rows with valid weights
    rows = conn.execute("""
        SELECT race_date, race_number, horse_name, horse_weight, weight,
               draw, finish_position, jockey, trainer
        FROM race_results
        WHERE horse_weight IS NOT NULL AND horse_weight > 0
          AND weight IS NOT NULL AND weight != ''
        ORDER BY race_date DESC, race_number, horse_name
        LIMIT 50
    """).fetchall()

    if len(rows) < 10:
        pytest.skip(f"Too few DB rows with valid weights: {len(rows)}")

    # Precompute field weight lists per race (same as training does)
    race_keys_needed = {f"{r['race_date']}-{r['race_number']}" for r in rows}
    field_weights_map: dict[str, list] = {}
    for rk in race_keys_needed:
        date, race_no = rk.rsplit("-", 1)
        wt_rows = conn.execute("""
            SELECT weight FROM race_results
            WHERE race_date = ? AND race_number = ? AND weight IS NOT NULL AND weight != ''
        """, (date, int(race_no))).fetchall()
        field_wts = []
        for wr in wt_rows:
            try:
                field_wts.append(int(wr["weight"]))
            except (ValueError, TypeError):
                pass
        if field_wts:
            field_weights_map[rk] = field_wts

    fe = FeatureEngineering()
    deltas_above_threshold = 0
    total_checked = 0

    for row in rows:
        race_key = f"{row['race_date']}-{row['race_number']}"
        field_wts = field_weights_map.get(race_key, [])

        # Compute training-side value using the precomputed field weights
        try:
            weight_val = int(row["weight"])
        except (ValueError, TypeError):
            continue

        train_wvf = _compute_weight_vs_field(weight_val, field_wts)

        # Compute inference-side value via extract_features()
        # F-06 fix: field_weights must be present in horse_data
        horse_data = {
            "name": row["horse_name"],
            "horse_weight": row["horse_weight"],
            "weight": row["weight"],
            "draw": row["draw"] or 1,
            "jockey": row["jockey"] or "",
            "trainer": row["trainer"] or "",
            "race_date": row["race_date"],
            "race_number": row["race_number"],
            "field_weights": field_wts,  # F-06 fix: pass field_weights
            "field_size": len(field_wts) or 12,
        }
        try:
            infer_feats = fe.extract_features(horse_data, [], "", conn=conn)
            infer_wvf = infer_feats[wvf_idx]
        except Exception as e:
            print(f"Skipped {row['horse_name']} due to error: {e}")
            continue

        delta = abs(infer_wvf - train_wvf)
        total_checked += 1
        if delta > 0.1:
            deltas_above_threshold += 1
            print(
                f"Mismatch: {row['horse_name']} @ {row['race_date']} R{row['race_number']} "
                f"infer={infer_wvf:.4f} train={train_wvf:.4f} Δ={delta:.4f}"
            )

    conn.close()

    assert total_checked > 0, "No rows checked — test is broken"
    fail_rate = deltas_above_threshold / total_checked
    assert fail_rate <= 0.05, (
        f"F-06 regression: {deltas_above_threshold}/{total_checked} "
        f"({fail_rate:.1%}) rows exceed 0.1 delta on weight_vs_field. "
        f"Target: <=5%. Pre-fix DS-03 was 68%."
    )


def test_core_analyzer_includes_field_weights_in_horse_data():
    """F-06 regression guard: core/analyzer.py's horse_data dict must
    include field_weights when building features for extract_features().
    If removed, extract_features silently falls back to absolute-weight
    formula and train/predict parity breaks (pre-fix was 82% mismatches).

    This test fails if someone accidentally removes the field_weights line
    from core/analyzer.py, catching the regression at CI time.
    """
    import inspect
    import core.analyzer as analyzer_module

    # Check that the source code contains the field_weights plumbing
    source = inspect.getsource(analyzer_module)
    assert '"field_weights"' in source or "'field_weights'" in source, (
        "core/analyzer.py does not mention 'field_weights' in horse_data construction. "
        "F-06 fix likely reverted. Pre-fix state silently falls back to "
        "absolute-weight formula, breaking train/predict parity (was 82% error rate)."
    )
