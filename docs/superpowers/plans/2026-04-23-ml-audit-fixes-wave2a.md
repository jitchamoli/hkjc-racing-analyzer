# ML Audit Fixes — Wave 2a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship six measurement-critical fixes from the 2026-04-23 ML audit — C1 (rating=0 drag in `core/scorer.py`), F-06 (`weight_vs_field` train/predict mismatch), F-15 (drop redundant `jockey_trainer_combo`), F-11 (tightened feature parity test), F-09 (grid-sweep blend weights), F-22 (re-evaluate box-4 pivot on corrected baseline) — in three staged commits with hard validation gates between.

**Architecture:** Stage A ships two code-only fixes (no retrain) that remove the rating=0 drag and fix the weight_vs_field train/predict gap. Stage B drops one feature (`jockey_trainer_combo`, r=0.79 with `jockey_win_rate`) and retrains; the `NUM_FEATURES` guard in `_load_model()` naturally enforces the migration. Stage C adds a per-feature parity test, grid-sweeps blend weights against the corrected baseline, and re-evaluates the box-4 default on the new numbers with explicit CI math. Each stage has a checkpointed rollback tag.

**Tech Stack:** Python 3, XGBoost, scikit-learn, `statsmodels.stats.proportion` (Wilson CI), numpy (bootstrap CI), pytest.

**Spec:** `docs/superpowers/specs/2026-04-23-ml-audit-fixes-wave2a-design.md`

**Upstream:** Wave 1 complete — `docs/audits/2026-04-23/post-fix-baseline.md` is the current authoritative baseline.

---

## File Structure

**To modify (production code):**
- `core/scorer.py` — C1 handling of rating=0/gear='' in `compute_weighted_score`; F-09 hardcoded blend-weight choice in `_dynamic_blend_weights`
- `ml/feature_schema.py` — F-15 drop `jockey_trainer_combo`, NUM_FEATURES → 22
- `ml/features.py` — F-06 weight_vs_field parity fix; F-15 remove `jockey_trainer_combo` from `extract_features()`
- `ml/training.py` — F-06 weight_vs_field parity fix; F-15 remove from `build_training_data_from_db()`
- `core/recommender.py` — F-22 default logic update (conditional on CI evidence)
- `CLAUDE.md` — F-09 blend choice, F-22 default decision, 22-feature schema note

**To create:**
- `tests/test_scorer_missing_rating.py` — C1 unit test
- `tests/test_feature_parity_weight.py` — F-06 parity probe
- `tests/test_feature_parity_strict.py` — F-11 tightened per-feature test
- `docs/audits/2026-04-23/wave2-baseline.md` — new authoritative baseline
- `docs/audits/2026-04-23/raw/wave2-*.txt` — evidence captures (pre-retrain, post-retrain, sweep output, final backtest)

**Backups created:**
- `models/hkjc_racing_model.pkl.wave1-backup` (rollback point for Stage B)

**Tags:**
- `wave2-pre` (before Stage A)
- `wave2-pre-retrain` (Stage A complete, before Stage B)
- `wave2-post-retrain` (Stage B complete, before Stage C)
- `wave2-complete` (Stage C end)

---

## Stage A — Code-only fixes (no retrain)

### Task 1: C1 — Fix `rating=0` drag in `core/scorer.py`

**Context:** `core/scorer.py:63-64` defines `ALWAYS_INCLUDED = {"rating", "form", "jockey", "trainer", "draw", "class_movement", "gear", "weight_carried"}`. The docstring at line 62 says "score of 0 means no data; these use 50 as neutral" — but the code at line 75 actually INCLUDES score=0 in the weighted sum, dragging down every horse's rule_score by ~8 points when `rating == 0`. 86% of holdout races are affected.

**Fix approach: exclusion via dynamic-weight renormalization** (matches approved spec). Change the inclusion test at line 75 from `if key in ALWAYS_INCLUDED or score > 0:` to `if score > 0:` (simpler and treats ALWAYS_INCLUDED factors uniformly with missing factors). The ALWAYS_INCLUDED set is retained for semantic clarity in the module but no longer has load-bearing inclusion logic.

Note: the docstring's alternative intent ("use 50 as neutral") would substitute 50 for missing values. We went with exclusion per the approved spec because it matches how non-ALWAYS_INCLUDED missing factors are already handled.

**Files:**
- Modify: `core/scorer.py` (`compute_weighted_score`, line 67-82)
- Create: `tests/test_scorer_missing_rating.py`

- [ ] **Step 1: Write the failing test**

```python
"""C1 test: missing rating (=0) is excluded from weighted average,
not dragged into it with weight 0.08."""
from core.scorer import compute_weighted_score


def test_missing_rating_excluded_from_weighted_score():
    """A race where all horses have rating=0 (missing data) must produce
    the same ranking as a race where rating is absent entirely from the
    factors dict. Pre-fix: rating=0 contributes 0 × 0.08 = 0 to every
    score, dragging all scores by ~8 points. Post-fix: rating absent from
    weight_map, renormalized out."""
    factors_with_rating_zero = {
        "rating": 0.0,       # missing data, should be excluded
        "form": 60.0,
        "jockey": 55.0,
        "trainer": 50.0,
        "draw": 50.0,
        "pace": 55.0,
    }
    factors_without_rating = {
        "form": 60.0,
        "jockey": 55.0,
        "trainer": 50.0,
        "draw": 50.0,
        "pace": 55.0,
    }
    score_with_zero = compute_weighted_score(factors_with_rating_zero)
    score_without = compute_weighted_score(factors_without_rating)
    assert abs(score_with_zero - score_without) < 0.01, (
        f"rating=0 treated differently than rating absent: "
        f"with_zero={score_with_zero:.4f}, without={score_without:.4f}. "
        f"Pre-fix: with_zero is ~8 points lower due to ALWAYS_INCLUDED drag."
    )


def test_missing_gear_excluded_from_weighted_score():
    """Same assertion for gear (empty string produces gear_score=0)."""
    factors_with_gear_zero = {
        "rating": 65.0,
        "form": 60.0,
        "jockey": 55.0,
        "gear": 0.0,   # missing gear
    }
    factors_without_gear = {
        "rating": 65.0,
        "form": 60.0,
        "jockey": 55.0,
    }
    score_with_zero = compute_weighted_score(factors_with_gear_zero)
    score_without = compute_weighted_score(factors_without_gear)
    assert abs(score_with_zero - score_without) < 0.01, (
        f"gear=0 treated differently: with_zero={score_with_zero:.4f}, "
        f"without={score_without:.4f}"
    )


def test_nonzero_rating_still_contributes():
    """Rating with actual value (e.g. 70) still contributes correctly."""
    factors_with_rating = {
        "rating": 70.0,
        "form": 60.0,
        "jockey": 55.0,
    }
    factors_without_rating = {
        "form": 60.0,
        "jockey": 55.0,
    }
    score_with = compute_weighted_score(factors_with_rating)
    score_without = compute_weighted_score(factors_without_rating)
    # Rating of 70 is above the others' average, so adding it should raise score
    assert score_with > score_without, (
        f"rating=70 should raise score but didn't: "
        f"with={score_with:.4f}, without={score_without:.4f}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer
pytest tests/test_scorer_missing_rating.py -v
```

Expected: `test_missing_rating_excluded_from_weighted_score` and `test_missing_gear_excluded_from_weighted_score` FAIL (current behavior includes them with 0 weight). `test_nonzero_rating_still_contributes` should PASS already (guard against regression).

- [ ] **Step 3: Apply the C1 fix in `core/scorer.py`**

Edit `compute_weighted_score` at lines 67-82. Replace:

```python
def compute_weighted_score(factors: Dict[str, float]) -> float:
    """Compute rule-based score from factor dict with dynamic weight normalization.

    Factors with score == 0 are excluded (except ALWAYS_INCLUDED factors).
    Remaining weights are renormalized to sum to 1.0.
    """
    weight_map = {}
    for key, score in factors.items():
        if key in ALWAYS_INCLUDED or score > 0:
            if key in BettingWeights.WEIGHT_MAP:
                weight_map[key] = BettingWeights.WEIGHT_MAP[key]

    total_weight = sum(weight_map.values())
    if total_weight > 0:
        return sum(factors[k] * (w / total_weight) for k, w in weight_map.items())
    return 50.0
```

With:

```python
def compute_weighted_score(factors: Dict[str, float]) -> float:
    """Compute rule-based score from factor dict with dynamic weight normalization.

    Factors with score <= 0 are excluded as "no data" and weights are
    renormalized across the remaining factors to sum to 1.0.

    Historical note (C1, 2026-04-23 audit): `ALWAYS_INCLUDED` previously
    forced inclusion of rating/gear/etc. even at score=0, dragging rule
    scores down ~8 points in 86% of backtest races where DB was missing
    these fields. The set is retained for documentation but no longer
    has load-bearing inclusion logic — exclusion is now uniform across
    all factors.
    """
    weight_map = {}
    for key, score in factors.items():
        if score > 0:
            if key in BettingWeights.WEIGHT_MAP:
                weight_map[key] = BettingWeights.WEIGHT_MAP[key]

    total_weight = sum(weight_map.values())
    if total_weight > 0:
        return sum(factors[k] * (w / total_weight) for k, w in weight_map.items())
    return 50.0
```

Also update the `ALWAYS_INCLUDED` comment at line 62 to reflect the change:

```python
# Historical set — previously forced inclusion even at score=0.
# As of C1 (2026-04-23), no longer load-bearing; retained for documentation.
# Missing factors (score <= 0) are now uniformly excluded via renormalization.
ALWAYS_INCLUDED = {"rating", "form", "jockey", "trainer", "draw",
                   "class_movement", "gear", "weight_carried"}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_scorer_missing_rating.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Run the full test suite**

```bash
pytest tests/ --tb=short -q
```

Expected: all pre-existing tests still pass (93 from Wave 1 + 3 new = 96). If any pre-existing test asserted on a specific rule_score value that happened to include the drag, it will fail — update it to reflect corrected behavior.

- [ ] **Step 6: Run backtest and capture new Brier**

```bash
python3 -m scripts.backtest 2>&1 | tee docs/audits/2026-04-23/raw/wave2-stage-a-backtest-c1.txt
```

Capture the holdout Brier from the output. **Stage A primary gate:** holdout Brier must improve from 0.168 to < 0.150. If Brier is unchanged or worse, C1 didn't fix the drag and we need to investigate (don't commit).

- [ ] **Step 7: Commit**

```bash
git add core/scorer.py tests/test_scorer_missing_rating.py docs/audits/2026-04-23/raw/wave2-stage-a-backtest-c1.txt
git commit -m "C1: exclude score=0 factors uniformly in compute_weighted_score

Previously ALWAYS_INCLUDED forced inclusion of rating/gear/etc. even
when score=0 (missing data), dragging rule scores down ~8 points in
backtest races lacking DB fields. 86% of Wave 1 holdout races affected.

Fix: drop the ALWAYS_INCLUDED special case in compute_weighted_score;
all missing factors are now uniformly excluded via dynamic-weight
renormalization. Live predictions are unaffected (race cards always
populate rating). Backtest Brier improves from 0.168 to <target>.

Tests: 3 new in tests/test_scorer_missing_rating.py verify the fix."
```

### Task 2: F-06 — Fix `weight_vs_field` train/predict mismatch

**Context:** DS-03 measured 68% of horses with |Δ| > 0.1 on `weight_vs_field` between `extract_features()` (inference) and `build_training_data_from_db()` (training). This is a train/predict distribution mismatch — the model was trained on one feature but predicts on another.

**Fix approach:**
1. Read both `ml/features.py::extract_features()` and `ml/training.py::build_training_data_from_db()` to find where `weight_vs_field` is computed in each.
2. Identify the semantic difference (field-median calculation, unit mismatch, normalization difference, etc.).
3. Pick the semantically correct version and make both compute it identically. **Default choice if unclear:** the training path's implementation — because the current model was trained on that distribution, so the inference side is the one that should adapt.
4. Verify with a parity probe.

**Files:**
- Modify: `ml/features.py` and/or `ml/training.py` (wherever `weight_vs_field` is computed)
- Create: `tests/test_feature_parity_weight.py`

- [ ] **Step 1: Read both computations**

```bash
cd /Users/jitendrachamoli/Documents/Project_Hub/hkjc_racing_analyzer
grep -n "weight_vs_field" ml/features.py ml/training.py
```

Locate the two computation sites. Read the surrounding 20 lines at each site. Write down in 2-3 sentences what each computes and how they differ.

- [ ] **Step 2: Write the failing parity test**

Create `tests/test_feature_parity_weight.py`:

```python
"""F-06 test: weight_vs_field feature must match between extract_features()
and build_training_data_from_db() for historical DB rows."""
import sqlite3
import numpy as np

import config
from ml.features import FeatureEngineering


def test_weight_vs_field_train_predict_parity():
    """For 50 random DB horse rows, extract_features()['weight_vs_field']
    must equal build_training_data_from_db()'s corresponding value within
    0.1 absolute delta for >=95% of rows.

    Pre-fix: DS-03 measured 68% of rows with |Δ| > 0.1.
    Post-fix: <=5% of rows with |Δ| > 0.1.
    """
    from ml.training import build_training_data_from_db
    from ml.feature_schema import FEATURE_NAMES

    assert "weight_vs_field" in FEATURE_NAMES, "weight_vs_field missing from schema"
    wvf_idx = FEATURE_NAMES.index("weight_vs_field")

    # Build training data (which produces per-row feature vectors)
    X_train, y_train, _, _ = build_training_data_from_db(limit=50)
    assert X_train is not None, "Training build returned None"
    assert len(X_train) >= 30, f"Expected >= 30 rows, got {len(X_train)}"

    # For each training row, also run inference path and compare
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    fe = FeatureEngineering()

    # NOTE: to match a training row to its inference call, the test needs
    # horse_data in the same shape build_training_data_from_db used. The
    # implementer adapts this to the actual API; the assertion below is
    # the contract.
    deltas = []
    for i, train_features in enumerate(X_train[:50]):
        # Call extract_features on the same horse's historical context
        # (adapt to actual API). Capture the inference-path weight_vs_field.
        # For this stub, the implementer re-runs build_training_data with
        # the same seed + row, then calls extract_features on the row's
        # horse_data dict, and computes the delta.
        #
        # If building parallel calls is cumbersome, the simpler version:
        # measure the delta on a set of synthetic horse_data dicts where
        # weight is known, and verify both functions produce the same
        # weight_vs_field given identical inputs.
        pass

    # Stub assertion — implementer replaces with real delta computation
    # that reflects the ACTUAL parity check on the 50 rows.
    assert True, "Replace with actual parity delta assertion per implementer"
```

**Important:** The test above is a SCAFFOLD. The implementer must replace it with a real parity check that:
(a) Runs `build_training_data_from_db(limit=50)` (or iterates some subset of DB rows)
(b) For each training row, runs `extract_features(horse_data, ...)` where `horse_data` is the same horse in the same race
(c) Computes |extract_features_result[wvf_idx] - train_features[wvf_idx]|
(d) Asserts: out of 50 rows, fewer than 5 rows have |delta| > 0.1 (i.e., <=10% failure rate, target 5%)

The implementer decides the exact matching strategy — the simplest is to have `build_training_data_from_db` return a parallel list of horse_data dicts it consumed, so the test can zip them.

- [ ] **Step 3: Run the test to verify the starting state**

```bash
pytest tests/test_feature_parity_weight.py -v
```

The test should run (not error). With the stub, it trivially passes. The implementer must tighten the stub to fail on the current pre-fix mismatch before fixing it.

- [ ] **Step 4: Make the test fail on current code**

Replace the `pass` body and `assert True` in the test with the actual parity check (per Step 2 guidance). Re-run:

```bash
pytest tests/test_feature_parity_weight.py -v
```

Expected: FAIL with delta-count >5 (current pre-fix state). If the test PASSES on pre-fix code, the delta is less severe than DS-03 measured — document and proceed anyway (still worth tightening the test).

- [ ] **Step 5: Identify and apply the fix**

Based on Step 1's analysis, determine which function's `weight_vs_field` is wrong. Apply the fix. Options:
- If both use different median calculations: pick one (prefer training path's, since that's what the model learned).
- If units differ (kg vs lbs): standardize on the DB's unit (lbs per CLAUDE.md).
- If one normalizes and the other doesn't: add or remove the normalization consistently.

The fix is likely <10 lines of code. Document the choice in a comment at both sites.

- [ ] **Step 6: Run the test to verify it passes**

```bash
pytest tests/test_feature_parity_weight.py -v
```

Expected: PASS. <=5 of 50 rows exceed 0.1 delta.

- [ ] **Step 7: Run full test suite**

```bash
pytest tests/ --tb=short -q
```

Expected: all pass (96 + 1 new = 97).

- [ ] **Step 8: Commit**

```bash
git add ml/features.py ml/training.py tests/test_feature_parity_weight.py
git commit -m "F-06: fix weight_vs_field train/predict parity (DS-03)

Identified the semantic difference between extract_features() and
build_training_data_from_db()'s weight_vs_field computation:
[DESCRIBE IN 1-2 LINES WHAT DIFFERED].

Aligned both on [CHOSEN APPROACH]. Parity probe measures <=5% of rows
with |delta| > 0.1 (was 68% pre-fix per DS-03).

Test added: tests/test_feature_parity_weight.py enforces the contract
on 50 DB rows going forward."
```

### Stage A Gates

- [ ] **Step 1: Verify all Stage A commits landed**

```bash
git log --oneline 001cbcd..HEAD
```

Expected: C1 and F-06 commits both present.

- [ ] **Step 2: Verify backtest Brier improved**

```bash
python3 -m scripts.backtest 2>&1 | tee docs/audits/2026-04-23/raw/wave2-stage-a-final.txt
```

Inspect the output — Brier (top-pick) must be < 0.150 (down from 0.168 Wave 1 baseline). This is the Stage A primary gate. If Brier is unchanged, C1 or F-06 didn't take effect — investigate before proceeding.

- [ ] **Step 3: Verify test suite green**

```bash
pytest tests/ --tb=short -q
```

Expected: all pass.

- [ ] **Step 4: Tag the pre-retrain checkpoint + backup pickle**

```bash
cp models/hkjc_racing_model.pkl models/hkjc_racing_model.pkl.wave1-backup
cp models/hkjc_racing_model.sha256 models/hkjc_racing_model.sha256.wave1-backup
git tag wave2-pre-retrain
git tag --list | grep wave2
```

Expected: backups exist, tag created.

---

## Stage B — Schema change + retrain

### Task 3: F-15 — Drop `jockey_trainer_combo` and retrain

**Context:** Wave 1's `experiment_log.json` flagged `jockey_win_rate × jockey_trainer_combo = 0.79` — above the 0.7 correlation threshold. Per DS-07, drop the redundant feature. The pickle's `_load_model()` feature-count guard will refuse to load the 23-feature Wave 1 pickle after schema changes to 22 — intentional, forces retrain.

**Files:**
- Modify: `ml/feature_schema.py` (remove entry, update NUM_FEATURES to 22)
- Modify: `ml/features.py` (remove `jockey_trainer_combo` from `extract_features()`)
- Modify: `ml/training.py` (remove from `build_training_data_from_db()`)
- Regenerate: `models/hkjc_racing_model.pkl`, `.sha256`, `experiment_log.json`

- [ ] **Step 1: Remove `jockey_trainer_combo` from schema**

Edit `ml/feature_schema.py`. In `FEATURE_SCHEMA` (lines 12-36), remove the entry:

```python
    {"name": "jockey_trainer_combo", "default": 0.5, "range": (0.0, 1.0)}, # 21
```

Renumber comments for subsequent entries (22 → 21, 23 → 22). `NUM_FEATURES = len(FEATURE_SCHEMA)` auto-updates to 22.

- [ ] **Step 2: Remove from `extract_features()`**

```bash
grep -n "jockey_trainer_combo" ml/features.py
```

Find the line(s) where `jockey_trainer_combo` is computed and added to the feature vector. Delete them. Ensure the feature vector list order still matches `FEATURE_NAMES` exactly.

Verify by running:

```bash
python3 -c "
from ml.features import FeatureEngineering
from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES
print(f'NUM_FEATURES={NUM_FEATURES}')
print(f'FEATURE_NAMES (last 5): {FEATURE_NAMES[-5:]}')
"
```

Expected: `NUM_FEATURES=22`; last 5 names end with `horse_weight_change`, `speed_consistency` (no `jockey_trainer_combo` in the list).

- [ ] **Step 3: Remove from `build_training_data_from_db()`**

```bash
grep -n "jockey_trainer_combo" ml/training.py
```

Find and delete the corresponding computation. Ensure the feature vector built matches `FEATURE_NAMES` order.

- [ ] **Step 4: Run feature-vector validation**

```bash
pytest tests/test_feature_parity_weight.py -v  # from Task 2
pytest tests/test_features.py -v                # pre-existing
```

Expected: both pass. The parity test from F-06 also serves as a post-F-15 smoke test — if feature counts mismatch, it fails.

- [ ] **Step 5: Retrain**

```bash
python3 -m scripts.train 2>&1 | tee docs/audits/2026-04-23/raw/wave2-retrain-output.txt
```

Expected:
- Training completes without traceback
- Log shows `Training with N samples` where N >= 3000
- Log shows `Platt calibrator fitted on K validation samples (logit-input)` (F-01 still live)
- Log shows `Saved model version YYYYMMDD-HHMMSS (n_samples=N)` (F-04 still live)
- **Correlation report shows ZERO pairs with |r| > 0.7** (F-15 took effect)

- [ ] **Step 6: Verify Stage B gates**

```bash
python3 -c "
import pickle, json
import numpy as np
from scipy.special import logit

m = pickle.load(open('models/hkjc_racing_model.pkl','rb'))

print('STAGE B GATE CHECKS (Wave 2):')
print()

# Gate 1: n_samples >= 3000
n = m.get('n_samples')
g1 = n is not None and n >= 3000
print(f'[Gate 1] n_samples >= 3000: n_samples={n} -> {\"PASS\" if g1 else \"FAIL\"}')

# Gate 2: trained_date >= 2026-04-23
td = m.get('trained_date', '')
g2 = td >= '2026-04-23'
print(f'[Gate 2] trained_date: {td} -> {\"PASS\" if g2 else \"FAIL\"}')

# Gate 3: n_features == 22 (schema change took effect)
nf = m.get('n_features')
g3 = nf == 22
print(f'[Gate 3] n_features == 22: n_features={nf} -> {\"PASS\" if g3 else \"FAIL\"}')

# Gate 4: Calibrator span >= 0.10
cal = m.get('calibrator')
if cal is not None:
    grid = logit(np.clip(np.linspace(0.02, 0.95, 50), 1e-6, 1-1e-6)).reshape(-1, 1)
    probs = cal.predict_proba(grid)[:, 1]
    span = float(probs.max() - probs.min())
    g4 = span >= 0.10
    print(f'[Gate 4] span >= 0.10: span={span:.4f} -> {\"PASS\" if g4 else \"FAIL\"}')
else:
    g4 = False
    print('[Gate 4] FAIL (no calibrator)')

# Gate 5: Training-set Brier <= 0.080
log = json.load(open('models/experiment_log.json'))
latest = log[-1]
brier = latest.get('brier_score')
g5 = brier is not None and brier <= 0.080
print(f'[Gate 5] Brier <= 0.080: brier={brier:.4f} -> {\"PASS\" if g5 else \"FAIL\"}')

# Gate 6: BSS >= 0
baseline = (1/12) * (1 - 1/12)
bss = 1 - (brier / baseline) if brier is not None else -1
g6 = bss >= 0.0
print(f'[Gate 6] BSS >= 0: bss={bss:.4f} -> {\"PASS\" if g6 else \"FAIL\"}')

# Gate 7: |Brier(22-feat) - Brier(Wave1 23-feat)| <= 0.01
# Wave 1 Brier was 0.0737 per stage-b-gates.txt
wave1_brier = 0.0737
delta = abs(brier - wave1_brier)
g7 = delta <= 0.01
print(f'[Gate 7] |Brier - Wave1 Brier (0.0737)| <= 0.01: delta={delta:.4f} -> {\"PASS\" if g7 else \"FAIL\"}')

# Gate 8: no correlations > 0.7 in experiment_log
high_corrs = latest.get('high_correlations', {})
g8 = len(high_corrs) == 0
print(f'[Gate 8] no |r|>0.7 pairs: high_corrs={high_corrs} -> {\"PASS\" if g8 else \"FAIL\"}')

all_pass = all([g1, g2, g3, g4, g5, g6, g7, g8])
print()
print(f'OVERALL: {\"ALL GATES PASS\" if all_pass else \"GATE FAILURE — ROLLBACK REQUIRED\"}')
" | tee docs/audits/2026-04-23/raw/wave2-stage-b-gates.txt
```

Expected: all 8 gates PASS.

**If any gate fails, ROLLBACK:**
```bash
cp models/hkjc_racing_model.pkl.wave1-backup models/hkjc_racing_model.pkl
cp models/hkjc_racing_model.sha256.wave1-backup models/hkjc_racing_model.sha256
echo "ROLLED BACK to Wave 1 pickle. Investigate Stage B failure."
# Also revert the ml/feature_schema.py, ml/features.py, ml/training.py changes:
git checkout HEAD~1 -- ml/feature_schema.py ml/features.py ml/training.py
```

Do NOT commit a failed retrain. Report BLOCKED.

- [ ] **Step 7: Verify F-05 integrity check accepts new pickle**

```bash
python3 -c "
from ml.model import HKJCRacingMLModel
from ml.integrity import check_model_integrity
m = HKJCRacingMLModel()
check_model_integrity(m._metadata)
print('OK: integrity check accepts new 22-feature pickle')
"
```

Expected: OK message. If it raises, something in the metadata doesn't align.

- [ ] **Step 8: Run accuracy report**

```bash
python3 -m scripts.accuracy --all 2>&1 | tee docs/audits/2026-04-23/raw/wave2-stage-b-accuracy.txt
```

Expected: runs cleanly, Brier/BSS numbers printed.

- [ ] **Step 9: Commit + tag**

```bash
git add ml/feature_schema.py ml/features.py ml/training.py \
       models/hkjc_racing_model.pkl models/hkjc_racing_model.sha256 \
       models/experiment_log.json \
       docs/audits/2026-04-23/raw/wave2-retrain-output.txt \
       docs/audits/2026-04-23/raw/wave2-stage-b-gates.txt \
       docs/audits/2026-04-23/raw/wave2-stage-b-accuracy.txt
git commit -m "F-15: drop jockey_trainer_combo, retrain on 22 features

Wave 1's feature correlation report flagged jockey_win_rate ×
jockey_trainer_combo = 0.79 — near-duplicate feature. DS-07 classified
as Medium. Drop the redundant feature: FEATURE_SCHEMA now has 22
entries, NUM_FEATURES = 22.

Retrain gates pass (Wave 1 set + new gates):
  - Brier Δ vs 23-feat baseline <= 0.01 (confirms feature was redundant,
    not carrying unique signal)
  - Zero |r|>0.7 pairs in correlation report
  - n_samples, span, BSS all still healthy

The Wave 1 23-feature pickle is refused by _load_model()'s feature-count
guard — correct; forces migration. Backup at
models/hkjc_racing_model.pkl.wave1-backup."

git tag wave2-post-retrain
```

---

## Stage C — Post-retrain validation + tuning + strategic re-eval

### Task 4: F-11 — Tighten feature parity test

**Context:** `tests/test_data_accuracy.py::test_training_and_predict_features_same_distribution` currently allows up to 2 "wildly-off" features per horse. DS-03 slipped through because `weight_vs_field` was wildly-off for 68% of horses but only counted as 1 feature. Tighten to per-feature, per-horse.

**Files:**
- Create: `tests/test_feature_parity_strict.py`
- Possibly modify: `tests/test_data_accuracy.py` (remove the loose test, or leave and add the strict one alongside)

- [ ] **Step 1: Inspect the existing loose test**

```bash
grep -n "test_training_and_predict_features_same_distribution\|wildly" tests/test_data_accuracy.py
```

Read the function. Note the tolerance structure (likely: per-horse max of 2 features with large delta allowed).

- [ ] **Step 2: Write the strict test**

Create `tests/test_feature_parity_strict.py`:

```python
"""F-11 test: strict per-feature parity between extract_features() and
build_training_data_from_db().

For N=20 random DB rows, |Δ| < 0.02 element-wise for every feature.
Replaces the legacy "≤2 wildly-off features" tolerance that allowed
DS-03 (weight_vs_field mismatch) to slip through."""
import numpy as np
import sqlite3

import config
from ml.features import FeatureEngineering
from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES


PER_FEATURE_TOLERANCE = 0.02
NUM_TEST_ROWS = 20


def test_strict_feature_parity_per_row_per_feature():
    """Every feature of every tested row must agree within 0.02 between
    the training-time computation and the inference-time computation.

    This is the gate that F-06 must pass. If weight_vs_field (or any
    other feature) drifts by more than 0.02 on any row, this test fails.
    """
    from ml.training import build_training_data_from_db

    X_train, _, _, _ = build_training_data_from_db(limit=NUM_TEST_ROWS)
    assert X_train is not None and len(X_train) >= NUM_TEST_ROWS // 2, (
        f"Insufficient training data to run parity test: "
        f"got {0 if X_train is None else len(X_train)} rows"
    )

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    fe = FeatureEngineering()

    # Adapt to actual API: build_training_data_from_db should expose the
    # horse_data dicts it consumed, so we can call extract_features on
    # each and compare. If the function doesn't expose them, the
    # implementer adds a parallel return value — this is in service of
    # the strict parity contract.
    violations = []  # list of (row_idx, feature_name, delta)
    for i, train_features in enumerate(X_train[:NUM_TEST_ROWS]):
        horse_data = _get_horse_data_for_training_row(i, conn)  # adapt
        infer_features = fe.extract_features(horse_data, [], '', conn=conn)
        deltas = np.abs(np.asarray(infer_features) - np.asarray(train_features))
        for j, delta in enumerate(deltas):
            if delta > PER_FEATURE_TOLERANCE:
                violations.append((i, FEATURE_NAMES[j], float(delta)))

    if violations:
        msg = (
            f"\nFeature parity violations (>{PER_FEATURE_TOLERANCE}):\n"
            + "\n".join(
                f"  row {r}, feature {n}, delta {d:.4f}"
                for r, n, d in violations[:20]
            )
            + (f"\n  ... and {len(violations)-20} more" if len(violations) > 20 else "")
        )
        assert False, msg


def _get_horse_data_for_training_row(row_idx, conn):
    """Adapt to actual API — returns the horse_data dict that would have
    been passed to extract_features for the idx-th training row."""
    raise NotImplementedError(
        "Implementer adapts this to pull horse_data from the same source "
        "build_training_data_from_db used."
    )
```

**Adapter note:** The implementer must either (a) extend `build_training_data_from_db` to return parallel horse_data dicts, or (b) write a mini-builder in the test that iterates DB race_results and produces horse_data exactly like the training builder does. The test's correctness depends on calling `extract_features` on the same input the training row was built from.

- [ ] **Step 3: Run test — should pass on F-06-corrected code**

```bash
pytest tests/test_feature_parity_strict.py -v
```

Expected: PASS after F-06 is in place. If it FAILS, there's another train/predict mismatch beyond weight_vs_field — document the failing features and either fix them in this task or escalate.

- [ ] **Step 4: Verify test catches regressions**

Temporarily revert F-06 (checkout the pre-F-06 version of whichever file was changed):

```bash
git stash
git show HEAD~2:ml/features.py > /tmp/features_prefix.py  # or whichever file F-06 touched
# Apply the pre-fix version:
cp /tmp/features_prefix.py ml/features.py
pytest tests/test_feature_parity_strict.py -v  # should FAIL
# Then restore:
git checkout HEAD -- ml/features.py
git stash pop  # if we stashed anything
pytest tests/test_feature_parity_strict.py -v  # should PASS again
```

Expected: test fails on reverted code, passes on current code. This confirms the test is doing real work.

- [ ] **Step 5: Decide fate of the loose test**

Either:
- Option A: Remove the loose `test_training_and_predict_features_same_distribution` from `tests/test_data_accuracy.py` since the strict version supersedes it.
- Option B: Keep both (loose as smoke test, strict as gate).

Recommend option A — duplicate tests are noise.

- [ ] **Step 6: Run full suite**

```bash
pytest tests/ --tb=short -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tests/test_feature_parity_strict.py tests/test_data_accuracy.py
git commit -m "F-11: strict per-feature parity test (replaces loose tolerance)

Per-feature, per-row |Δ| < 0.02 assertion over 20 random DB rows.
Legacy test allowed up to 2 wildly-off features per horse, which is
how DS-03 (weight_vs_field 68% mismatch) slipped through.

Test passes on Stage A+B corrected code. Would fail if F-06 or F-15
is reverted (verified by temporary revert during development)."
```

### Task 5: F-09 — Grid-sweep market-blend weights

**Context:** P-07 flagged the dynamic blend weights (35-65% by field_size × distance) as heuristic. Sweep to ground them empirically.

**Files:**
- Possibly modify: `scripts/backtest.py` (add `--blend-override` flag or similar to the sweep path)
- Modify: `core/scorer.py::_dynamic_blend_weights` (hardcode chosen point)
- Modify: `CLAUDE.md` (document choice)
- Create: `docs/audits/2026-04-23/raw/wave2-sweep-output.txt` and `docs/audits/2026-04-23/raw/wave2-sweep-summary.json`

- [ ] **Step 1: Inspect current `--sweep` path**

```bash
grep -n "sweep\|run_sweep\|--sweep" scripts/backtest.py
```

Understand whether `--sweep` already varies blend weights. If it varies something else (e.g., softmax temperature), either:
- Extend it to sweep blend parameters, OR
- Write a new sweep driver that loops over the grid and calls `predict_race_production` with overridden blend weights.

The simpler approach: write the sweep driver inline as a separate script section gated by `--sweep-blend`, independent of the existing `--sweep`.

- [ ] **Step 2: Implement the blend-weight sweep**

Add to `scripts/backtest.py` (or a new script `scripts/sweep_blend.py`) a driver that:

```python
# Pseudocode — implementer adapts to actual codebase structure

BLEND_GRID = [
    {"base_model_w": m, "field_threshold": f, "distance_threshold": d}
    for m in [0.25, 0.35, 0.45, 0.55, 0.65]
    for f in [8, 10, 12]
    for d in [1200, 1400, 1600]
]  # 45 points

results = []
for point in BLEND_GRID:
    # Override _dynamic_blend_weights for this run. Simplest: monkey-patch
    # core.scorer._dynamic_blend_weights for the duration of this iteration,
    # or parameterize the function to accept config.
    patched_weights = make_patched_blend_weights(**point)
    with patch_blend(patched_weights):
        backtest_result = run_holdout_backtest()  # returns dict of metrics
    results.append({
        **point,
        "brier": backtest_result["brier"],
        "box3_roi": backtest_result["box3_roi"],
        "box3_hit_rate": backtest_result["box3_hit_rate"],
        "n_races": backtest_result["n_races"],
    })

# Save full sweep output
json.dump(results, open("docs/audits/2026-04-23/raw/wave2-sweep-summary.json", "w"), indent=2)

# Pick best grid point by BSS (lower Brier = higher BSS)
best = min(results, key=lambda r: r["brier"])
print(f"Best point: {best}")
```

The actual implementation depends on how the production pipeline exposes blend weights. The cleanest path: parameterize `_dynamic_blend_weights` to accept config overrides, then pass overrides through the sweep.

- [ ] **Step 3: Run the sweep**

```bash
python3 -m scripts.backtest --sweep-blend 2>&1 | tee docs/audits/2026-04-23/raw/wave2-sweep-output.txt
```

Expected: 45 grid points run, each produces Brier + ROI numbers. Takes 10-30 minutes depending on hardware.

- [ ] **Step 4: Pick the winning grid point**

```bash
python3 -c "
import json
results = json.load(open('docs/audits/2026-04-23/raw/wave2-sweep-summary.json'))

# Sort by Brier (primary)
results.sort(key=lambda r: r['brier'])
print('Top 5 by Brier:')
for r in results[:5]:
    print(f'  {r}')

# Find the default point (the current heuristic's equivalent)
default = [r for r in results
           if r['base_model_w'] == 0.45 and r['field_threshold'] == 10 and r['distance_threshold'] == 1400]
print()
print(f'Current default equivalent: {default}')

# Check if top pick is statistically distinguishable from default
best_brier = results[0]['brier']
default_brier = default[0]['brier'] if default else None
if default_brier:
    print(f'Δ Brier (best - default): {best_brier - default_brier:.4f}')
    if abs(best_brier - default_brier) < 0.005:
        print('Best and default indistinguishable — keep default (simpler)')
    else:
        print(f'Best grid point meaningfully different — adopt it')
"
```

- [ ] **Step 5: Hardcode the chosen point**

Edit `core/scorer.py::_dynamic_blend_weights`. If the sweep picked:
- Point close to default (±0.005 BSS): keep current function, add a comment citing the sweep confirmed current choice.
- Different point: update the thresholds/weights to match. Keep the same signature so callers don't change.

Example change (if new default is `model_w=0.55, field_threshold=12`):

```python
def _dynamic_blend_weights(field_size: int, distance: int) -> Tuple[float, float]:
    """Dynamic market/model blend weights.

    F-09 empirical grid sweep (2026-04-23, 45 grid points × 73-race holdout):
    best BSS at model_w=0.55 base with field>=12+sprint shift. See
    docs/audits/2026-04-23/raw/wave2-sweep-summary.json.
    """
    model_w = 0.55  # was config.MODEL_BLEND_WEIGHT
    market_w = 1.0 - model_w

    # Large field (12+) + sprint (<=1400m): market sharpens, shift to 0.70 market
    if field_size >= 12 and distance <= 1400:
        market_w = 0.70
        model_w = 0.30
    # Small field (<8) or long (>=1800m): model gets heavier weight
    elif field_size < 8 or distance >= 1800:
        model_w = 0.70
        market_w = 0.30

    return model_w, market_w
```

- [ ] **Step 6: Update CLAUDE.md**

In the "Dynamic Market/Model Blend" section of CLAUDE.md, update the numbers to reflect the sweep-chosen values. Add a line: "Grid-sweep chosen 2026-04-23: see docs/audits/2026-04-23/raw/wave2-sweep-summary.json (45 points). Selection rule: lowest Brier on 73-race holdout, tie-break toward simpler (constant) schedule."

- [ ] **Step 7: Commit**

```bash
git add scripts/backtest.py core/scorer.py CLAUDE.md \
       docs/audits/2026-04-23/raw/wave2-sweep-output.txt \
       docs/audits/2026-04-23/raw/wave2-sweep-summary.json
git commit -m "F-09: grid-sweep blend weights (45 points × 73-race holdout)

Swept model_w ∈ {0.25-0.65}, field_threshold ∈ {8,10,12},
distance_threshold ∈ {1200,1400,1600}. Chose grid point with lowest
Brier on holdout; tie-broken toward simpler schedule.

Result: [chosen point in commit body]. Previously-heuristic weights
now have documented empirical basis. See
docs/audits/2026-04-23/raw/wave2-sweep-summary.json for full sweep
data."
```

### Task 6: F-22 — Re-evaluate box-4 pivot on corrected baseline

**Context:** Wave 1 post-fix numbers suggested box-3 unconditional (+1798%) > box-4 (+1228%). Now that C1 removes the drag and F-15+retrain cleans the model, the picture may have changed. Apply the CI decision rule.

**Files:**
- Possibly modify: `core/recommender.py` (if CI evidence supports switching default)
- Modify: `CLAUDE.md` (document the decision)

- [ ] **Step 1: Run the final backtest with F-09 weights**

```bash
python3 -m scripts.backtest 2>&1 | tee docs/audits/2026-04-23/raw/wave2-final-backtest.txt
```

Capture: straight-Q, box-3, box-4 hit rates and ROI (unconditional AND recommender-gated).

- [ ] **Step 2: Compute Wilson CIs on hit rates**

```bash
python3 -c "
from statsmodels.stats.proportion import proportion_confint

# Fill in from step 1 output:
n_races = 73  # or actual from output
straightq_hits = ...  # from output
box3_hits = ...
box4_hits = ...

for name, hits in [('straight-Q', straightq_hits), ('box-3', box3_hits), ('box-4', box4_hits)]:
    lo, hi = proportion_confint(hits, n_races, alpha=0.05, method='wilson')
    print(f'{name}: hit_rate={hits/n_races:.3f}, 95% Wilson CI=[{lo:.3f}, {hi:.3f}]')
" | tee docs/audits/2026-04-23/raw/wave2-hit-rate-ci.txt
```

- [ ] **Step 3: Compute bootstrap CIs on ROI**

```bash
python3 << 'EOF' | tee docs/audits/2026-04-23/raw/wave2-roi-ci.txt
"""Bootstrap 95% CI on quinella ROI.

For each strategy, the per-race P&L is a list: (stake_spent, dividend_received_if_hit_else_0).
Resample with replacement N times, compute ROI per resample, take 2.5%/97.5% percentiles.
"""
import json
import numpy as np

# Load per-race P&L from backtest output. The implementer extracts this from the
# backtest — if the backtest doesn't emit per-race P&L, add an output file for it.
# For this scaffold, load from a JSON the backtest should write.
per_race = json.load(open('docs/audits/2026-04-23/raw/wave2-per-race-pnl.json'))
# Schema: [{"race_date": ..., "race_no": ..., "straightq_stake": 1, "straightq_return": 0 or dividend, "box3_stake": 3, "box3_return": 0 or dividend, "box4_stake": 6, "box4_return": 0 or dividend}, ...]

N_BOOT = 10000
rng = np.random.default_rng(42)

for strategy in ['straightq', 'box3', 'box4']:
    stakes = np.array([r[f'{strategy}_stake'] for r in per_race])
    returns = np.array([r[f'{strategy}_return'] for r in per_race])
    roi_boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(stakes), len(stakes))
        total_stake = stakes[idx].sum()
        total_return = returns[idx].sum()
        roi = (total_return - total_stake) / total_stake if total_stake > 0 else 0
        roi_boot.append(roi)
    lo = np.percentile(roi_boot, 2.5)
    hi = np.percentile(roi_boot, 97.5)
    mean = np.mean(roi_boot)
    print(f'{strategy}: mean_ROI={mean*100:+.1f}%, 95% bootstrap CI=[{lo*100:+.1f}%, {hi*100:+.1f}%]')
EOF
```

**Note:** If `scripts/backtest.py` doesn't currently emit per-race P&L as JSON, add that output in a small addition. Emit to `docs/audits/2026-04-23/raw/wave2-per-race-pnl.json`.

- [ ] **Step 4: Apply the decision rule**

Based on Steps 2 and 3, decide:

- **Keep box-4 default** iff `box4_hit_CI_low > box3_hit_CI_high` AND `box4_ROI_CI_low > box3_ROI_CI_high`. Both hit rate AND ROI must clearly favor box-4.
- **Switch to box-3 default** iff `box3_hit_CI_low > box4_hit_CI_high` AND `box3_ROI_CI_low > box4_ROI_CI_high`.
- **Switch to straight-Q default** iff recommender-gated straight-Q CI dominates both.
- **Keep current default with caveat** iff CIs overlap (ambiguous evidence).

- [ ] **Step 5: Implement the decision in `core/recommender.py`**

If the decision is "keep box-4," no code change — just documentation.

If switching default, find the logic in `core/recommender.py` that chooses between box-3 and box-4 (likely involves `top_3_share` and `top_4_share` thresholds). Update the default to the new decision while keeping the existing fallback tree.

- [ ] **Step 6: Update CLAUDE.md**

In the "Betting Strategy" section, replace the Apr 22 rationale with the F-22 decision. Include the specific CI numbers for reproducibility:

```
## Betting Strategy — [updated YYYY-MM-DD, F-22]

[Decision]: default to [box-3 | box-4 | straight-Q].

Evidence (73-race holdout, post-Wave-2 corrected baseline):
- Straight-Q: hit rate X% (Wilson CI [Y%, Z%]), ROI +W% (bootstrap CI [A%, B%])
- Box-3: hit rate X% (Wilson CI [Y%, Z%]), ROI +W% (bootstrap CI [A%, B%])
- Box-4: hit rate X% (Wilson CI [Y%, Z%]), ROI +W% (bootstrap CI [A%, B%])

Decision rule: [CI comparison outcome].
```

- [ ] **Step 7: Commit**

```bash
git add core/recommender.py CLAUDE.md \
       docs/audits/2026-04-23/raw/wave2-final-backtest.txt \
       docs/audits/2026-04-23/raw/wave2-hit-rate-ci.txt \
       docs/audits/2026-04-23/raw/wave2-roi-ci.txt \
       docs/audits/2026-04-23/raw/wave2-per-race-pnl.json
git commit -m "F-22: re-evaluate box-4 pivot on corrected baseline

Post-Wave-2 backtest CIs: [paste summary numbers]. Decision: [keep box-4
| switch to box-3 | keep with caveat].

Rationale: [one sentence on CI comparison outcome]. Full numbers in
docs/audits/2026-04-23/raw/wave2-{hit-rate,roi}-ci.txt."
```

### Task 7: Wave 2 baseline doc + user handoff

**Files:**
- Create: `docs/audits/2026-04-23/wave2-baseline.md`

- [ ] **Step 1: Compile the baseline doc**

Create `docs/audits/2026-04-23/wave2-baseline.md`:

```markdown
# Post-Wave-2a Baseline — 2026-04-23

**Supersedes:** `docs/audits/2026-04-23/post-fix-baseline.md` (Wave 1 baseline).

**Scope:** After Wave 2a fixes (C1, F-06, F-15, F-11, F-09, F-22) landed.

## Model State

| Field | Wave 1 | Wave 2a |
|---|---|---|
| n_features | 23 | 22 (F-15 dropped jockey_trainer_combo) |
| n_samples | 4403 | [from new experiment_log] |
| Trained date | 2026-04-23T09:35:46 | [new retrain date] |
| Calibrator span | 0.2604 | [new value] |
| Training Brier | 0.0737 | [new value] |
| BSS (training) | +0.035 | [new value] |
| Holdout Brier (backtest) | 0.168 | [new value — expected < 0.150 post-C1] |
| High-correlation pairs (|r|>0.7) | 1 (jockey×combo) | 0 (F-15 removed the pair) |

## Strategic Decision (F-22)

[Write out the decision: keep/switch/ambiguous, with CI numbers.]

## Blend Weights (F-09)

[Report the grid-sweep outcome and the chosen point.]

## Known Limitations

- 73-race holdout remains small. CIs are wide. F-21 data collection recommended.
- [Any new limitations surfaced during Wave 2a]

## Reproduction

```bash
git checkout wave2-pre          # pre-Wave-2a state
python3 -m scripts.backtest     # old numbers

git checkout wave2-complete     # post-Wave-2a
python3 -m scripts.backtest     # new numbers
```

## Next Steps

1. **Wave 2b hygiene** — F-08, F-12 through F-20 (smaller fixes)
2. **F-10 dividend multiplier** — if EV% still broken
3. **F-21 data collection** — parallel track
4. **Dashboard redesign** — after Wave 2b
```

Fill in all bracketed fields from the evidence files.

- [ ] **Step 2: Run the final end-to-end gate**

```bash
pytest tests/ --tb=short -q
python3 -m scripts.backtest > /tmp/wave2-final-check.txt 2>&1; echo "exit: $?"
python3 -m scripts.accuracy --all > /tmp/wave2-accuracy-check.txt 2>&1; echo "exit: $?"
```

Expected: all green.

- [ ] **Step 3: Commit + tag**

```bash
git add docs/audits/2026-04-23/wave2-baseline.md
git commit -m "Wave 2a complete: baseline documented

Captures post-Wave-2a model state, sweep results, CI-backed strategic
decision, and known limitations. Supersedes post-fix-baseline.md as
authoritative reference.

Fixes shipped: C1, F-06, F-15, F-11, F-09, F-22. Next: Wave 2b
hygiene + F-21 data collection in parallel."

git tag wave2-complete
```

- [ ] **Step 4: Hand off to user**

Post summary message (not a file):

```
Wave 2a complete. Baseline: docs/audits/2026-04-23/wave2-baseline.md

Key numbers:
- Holdout Brier: [X] (was 0.168)
- BSS (training): [X] (was +0.035)
- Strategic default: [decision]
- Blend weights: [chosen point or "default confirmed"]

Tags: wave2-pre, wave2-pre-retrain, wave2-post-retrain, wave2-complete.

Next options:
1. Wave 2b hygiene (F-08, F-12-F-20)
2. F-10 dividend multiplier (if EV still broken)
3. F-21 data collection
4. Dashboard redesign

Your call.
```

---

## Self-Review (performed inline before delivery)

**1. Spec coverage:**
- C1 scorer fix (spec §Stage A): Task 1 ✓
- F-06 weight_vs_field parity (spec §Stage A): Task 2 ✓
- Stage A gates (spec §Stage A Gates): end-of-Task-2 verification ✓
- F-15 drop feature + retrain (spec §Stage B): Task 3 ✓
- Stage B gates with 8 checks (spec §Stage B Gates): Task 3 Step 6 ✓
- F-11 tightened parity test (spec §Stage C): Task 4 ✓
- F-09 grid-sweep (spec §Stage C): Task 5 ✓
- F-22 re-evaluation with CI decision rule (spec §Stage C): Task 6 ✓
- Stage C gates + final end-to-end (spec §Stage C): Task 7 Step 2 ✓
- Wave 2 baseline doc (spec §Post-Deliverables): Task 7 ✓
- Rollback triggers (spec §Known Risks): Task 3 Step 6 rollback commands ✓
- Tags (spec §Checkpoint management): end of Task 2 (pre-retrain), end of Task 3 (post-retrain), end of Task 7 (complete) ✓

**2. Placeholder scan:**
- Task 2 Step 2 test contains a `pass` scaffold that says "implementer replaces" — not a placeholder in the spec-violation sense; it's a conditional-on-discovery step with explicit guidance on what to substitute (a real parity assertion over 50 rows).
- Task 4 Step 2 `_get_horse_data_for_training_row` stub says "adapt to actual API" — same category as above.
- Task 5 Step 2 pseudocode explicitly labels itself pseudocode. Implementer writes real code.
- Task 6 Step 6 CLAUDE.md template has bracketed fields — same pattern as Wave 1 Task 6; filled in Step 6 itself from evidence files.
- No TBD/TODO/implement-later tokens.

**3. Type consistency:**
- `NUM_FEATURES` changes from 23 → 22 in Task 3; all downstream references (Task 4 parity test, Task 7 baseline doc) consistent.
- Tag names consistent: `wave2-pre`, `wave2-pre-retrain`, `wave2-post-retrain`, `wave2-complete`.
- Gate names consistent across tasks (Gate 1-8 in Task 3 Step 6).
- `check_model_integrity`, `HKJCRacingMLModel`, `compute_weighted_score` all match the codebase's current names (verified via reads during design).
- Bootstrap / Wilson terminology used consistently in Task 6.

**Known execution caveats baked into the plan:**
- Task 2 requires the implementer to write real parity logic beyond the scaffold — explicit in the step text.
- Task 3 Step 6 gate-failure path includes full rollback commands.
- Task 5 `scripts/backtest.py --sweep-blend` may need to be created if the existing sweep path is incompatible — step explicitly describes this.
- Task 6 bootstrap computation depends on backtest emitting per-race P&L JSON — if it doesn't, step says to add that output (small addition).
