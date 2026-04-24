# Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the accumulated "show everything" dashboard with an edge-first layout that tells the user which 0-5 races to bet, shows *why* to trust each pick, and flags live odds moves in real time.

**Architecture:** Four new isolated pure-Python modules compute edge math, tier assignment, alerts, and historical per-tier hit-rates. `dashboard/generator.py` wires them into an additive JSON schema. `dashboard/template.py` is rewritten to render the new hero + strip + alerts rail + trust card skeleton. The core ML pipeline (Wave 2c 20-feature isotonic model) is untouched — all new work is downstream of calibration.

**Tech Stack:** Python 3, pytest, sqlite3, `statsmodels.stats.proportion` (Wilson CI), XGBoost pipeline unchanged.

**Spec:** `docs/superpowers/specs/2026-04-24-dashboard-redesign-design.md` (commit `e22d7a2`)

---

## File Structure

### New files (4 modules + 5 tests)
- `betting/edge_calculator.py` — pure_prob, market_prob, edge_pp, ev_win, overlay, edge_fires
- `betting/conviction.py` — tier assignment (EDGE / HIGH / MED / LOW / SKIP)
- `betting/alerts.py` — snapshot diff → alert events with dedup
- `betting/tier_stats.py` — DB-backed hit-rate + Wilson CI per tier, cached to `tracking_data/tier_stats.json`
- `tests/test_edge_calculator.py`
- `tests/test_conviction_tiers.py`
- `tests/test_alerts_dedup.py`
- `tests/test_tier_stats_wilson_ci.py`
- `tests/test_template_empty_state.py`

### Modified files
- `dashboard/generator.py` — call new modules, add fields to JSON
- `dashboard/template.py` — rewrite skeleton (hero, strip, alerts rail, trust card, empty state)
- `monitor/race_day.py` — add auto-rerun trigger on sharp STEAM/COLD over top-pick
- `dashboard/results_fetcher.py` — invalidate tier_stats cache after results ingestion
- `data/database.py` — add `conviction_tier` column to `prediction_log` (additive, nullable)

### Untouched (hard boundary)
- `core/analyzer.py`, `core/scorer.py`, `ml/model.py`, `ml/features.py`, `ml/training.py` — pipeline is stable at Wave 2c

---

## Task 1: `betting/edge_calculator.py` — Pure Math + Gates

**Files:**
- Create: `betting/edge_calculator.py`
- Test: `tests/test_edge_calculator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_edge_calculator.py
"""Edge math + 6-gate logic tests."""
import math
import pytest
from betting.edge_calculator import (
    compute_pure_probs, compute_market_probs, compute_edge_fields,
    apply_edge_gates, EdgeGates,
)


def test_pure_probs_softmax_sums_to_one():
    # 3 horses with rule_score + ml_prob
    horses = [
        {'rule_score': 70, 'ml_prob': 0.30},
        {'rule_score': 50, 'ml_prob': 0.15},
        {'rule_score': 40, 'ml_prob': 0.10},
    ]
    pure = compute_pure_probs(horses)
    assert len(pure) == 3
    assert math.isclose(sum(pure), 1.0, abs_tol=1e-6)
    # Strongest horse should have highest pure_prob
    assert pure[0] > pure[1] > pure[2]


def test_market_probs_normalize_from_odds():
    horses = [{'win_odds': 2.0}, {'win_odds': 4.0}, {'win_odds': 8.0}]
    market = compute_market_probs(horses)
    # 1/2 + 1/4 + 1/8 = 0.875 → normalize → 0.571, 0.286, 0.143
    assert math.isclose(sum(market), 1.0, abs_tol=1e-6)
    assert market[0] > market[1] > market[2]


def test_market_probs_handle_zero_odds():
    horses = [{'win_odds': 2.0}, {'win_odds': 0}, {'win_odds': 4.0}]
    market = compute_market_probs(horses)
    assert market[1] == 0.0
    assert math.isclose(market[0] + market[2], 1.0, abs_tol=1e-6)


def test_edge_fields_positive_gap():
    # pure > market → positive edge
    pure, market, odds = 0.22, 0.12, 8.3
    fields = compute_edge_fields(pure, market, odds)
    assert math.isclose(fields['edge_pp'], 10.0, abs_tol=0.01)
    assert math.isclose(fields['ev_win'], 0.22 * 8.3 - 1, abs_tol=0.01)
    assert math.isclose(fields['overlay'], 0.22 * 8.3, abs_tol=0.01)


def test_edge_fields_negative_gap():
    pure, market, odds = 0.10, 0.20, 5.0
    fields = compute_edge_fields(pure, market, odds)
    assert fields['edge_pp'] < 0
    assert fields['ev_win'] < 0
    assert fields['overlay'] < 1.0


def test_gates_all_pass_fires():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.10, pure_prob=0.22, edge_pp=10.0,
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is True
    assert failed == []


def test_gate_g1_ev_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.02, pure_prob=0.22, edge_pp=10.0,  # ev too low
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is False
    assert 'G1' in failed


def test_gate_g2_prob_floor_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.50, pure_prob=0.05, edge_pp=4.0,  # prob too low (longshot)
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is False
    assert 'G2' in failed


def test_gate_g3_edge_gap_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.06, pure_prob=0.15, edge_pp=3.0,  # gap too small
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is False
    assert 'G3' in failed


def test_gate_g4_rank_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.10, pure_prob=0.15, edge_pp=8.0,
        model_rank=5, odds_age_min=5, integrity_ok=True, gates=gates,  # rank too low
    )
    assert ok is False
    assert 'G4' in failed


def test_gate_g5_odds_stale_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.10, pure_prob=0.22, edge_pp=10.0,
        model_rank=1, odds_age_min=45, integrity_ok=True, gates=gates,  # stale
    )
    assert ok is False
    assert 'G5' in failed


def test_gate_g6_integrity_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.10, pure_prob=0.22, edge_pp=10.0,
        model_rank=1, odds_age_min=5, integrity_ok=False, gates=gates,  # bad model
    )
    assert ok is False
    assert 'G6' in failed


def test_gates_with_custom_thresholds():
    gates = EdgeGates(min_ev=0.10)  # stricter
    ok, _ = apply_edge_gates(
        ev_win=0.07, pure_prob=0.22, edge_pp=10.0,
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_edge_calculator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'betting.edge_calculator'`

- [ ] **Step 3: Write minimal implementation**

```python
# betting/edge_calculator.py
"""Pure-prob and edge math with 6-gate filtering.

pure_prob = softmax(0.6·sigmoid(rule_score) + 0.4·ml_prob)
— our independent opinion BEFORE the market blend.

Feeds edge_pp, ev_win, overlay; gated by EV + prob floor + edge gap + rank
+ odds freshness + model integrity. See
docs/superpowers/specs/2026-04-24-dashboard-redesign-design.md §4.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class EdgeGates:
    """Gate thresholds. Defaults per spec §4."""
    min_ev: float = 0.05
    min_prob: float = 0.10
    min_edge_pp: float = 5.0
    max_model_rank: int = 2
    max_odds_age_min: int = 30


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rule_score_to_prob(rule_score: float) -> float:
    # Base-rate-anchored sigmoid: score 50 → ~0.5, score 100 → ~0.88
    return _sigmoid(0.06 * (rule_score - 50))


def compute_pure_probs(horses: List[Dict]) -> List[float]:
    """pure_prob = softmax(0.6·sigmoid(rule) + 0.4·ml_prob) across the field.

    Each horse dict needs 'rule_score' and 'ml_prob'.
    Returns list of pure_probs summing to 1.0.
    """
    raw = [0.6 * _rule_score_to_prob(h['rule_score']) + 0.4 * h['ml_prob']
           for h in horses]
    max_raw = max(raw) if raw else 0
    exp = [math.exp(r - max_raw) for r in raw]
    total = sum(exp) or 1.0
    return [e / total for e in exp]


def compute_market_probs(horses: List[Dict]) -> List[float]:
    """market_prob = normalize(1 / win_odds) across the field.

    Horses with win_odds == 0 get market_prob 0; remaining normalized to 1.0.
    """
    inv = [(1.0 / h['win_odds']) if h.get('win_odds', 0) > 0 else 0.0
           for h in horses]
    total = sum(inv) or 1.0
    return [v / total for v in inv]


def compute_edge_fields(pure: float, market: float, win_odds: float) -> Dict[str, float]:
    """Per-horse edge signals."""
    return {
        'edge_pp': (pure - market) * 100.0,
        'ev_win': pure * win_odds - 1.0 if win_odds > 0 else -1.0,
        'overlay': pure * win_odds if win_odds > 0 else 0.0,
    }


def apply_edge_gates(
    ev_win: float, pure_prob: float, edge_pp: float,
    model_rank: int, odds_age_min: float, integrity_ok: bool,
    gates: EdgeGates = None,
) -> Tuple[bool, List[str]]:
    """Return (all_pass, [failed_gate_ids]). All 6 gates must pass for ★ EDGE."""
    gates = gates or EdgeGates()
    failed = []
    if ev_win < gates.min_ev: failed.append('G1')
    if pure_prob < gates.min_prob: failed.append('G2')
    if edge_pp < gates.min_edge_pp: failed.append('G3')
    if model_rank > gates.max_model_rank: failed.append('G4')
    if odds_age_min > gates.max_odds_age_min: failed.append('G5')
    if not integrity_ok: failed.append('G6')
    return (len(failed) == 0, failed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_edge_calculator.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add betting/edge_calculator.py tests/test_edge_calculator.py
git commit -m "feat(dashboard): edge calculator — pure_prob + 6 gates

Adds betting/edge_calculator.py with compute_pure_probs (softmax of
0.6·sigmoid(rule) + 0.4·ml_prob), compute_market_probs, compute_edge_fields,
and apply_edge_gates (G1 EV · G2 prob · G3 gap · G4 rank · G5 odds age ·
G6 integrity). 12 unit tests covering the gate matrix."
```

---

## Task 2: `betting/conviction.py` — Tier Assignment

**Files:**
- Create: `betting/conviction.py`
- Test: `tests/test_conviction_tiers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conviction_tiers.py
"""Tier assignment at spec boundaries."""
import pytest
from betting.conviction import assign_conviction_tier, ConvictionThresholds


def test_edge_takes_precedence_over_high():
    # high top_3 share AND has_edge → still EDGE
    tier = assign_conviction_tier(has_edge=True, top_2_share=0.50, top_3_share=0.60)
    assert tier == 'EDGE'


def test_high_tier_on_top_2_share():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.35, top_3_share=0.40)
    assert tier == 'HIGH'


def test_high_tier_on_top_3_share():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.30, top_3_share=0.45)
    assert tier == 'HIGH'


def test_med_tier():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.28, top_3_share=0.36)
    assert tier == 'MED'


def test_low_tier():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.23, top_3_share=0.30)
    assert tier == 'LOW'


def test_skip_tier():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.15, top_3_share=0.22)
    assert tier == 'SKIP'


def test_boundary_high_at_exact_threshold():
    # top_2_share exactly 0.33 → HIGH
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.33, top_3_share=0.30)
    assert tier == 'HIGH'


def test_boundary_med_at_exact_threshold():
    # top_2_share exactly 0.27 → MED
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.27, top_3_share=0.30)
    assert tier == 'MED'


def test_custom_thresholds():
    thresholds = ConvictionThresholds(high_top2=0.50)
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.40,
                                  top_3_share=0.50, thresholds=thresholds)
    # 0.40 < 0.50 HIGH threshold, but 0.50 top_3 ≥ default 0.42 → HIGH via top_3
    assert tier == 'HIGH'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_conviction_tiers.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# betting/conviction.py
"""Conviction tier assignment — replaces strategy_confidence.

Tiers: EDGE | HIGH | MED | LOW | SKIP
EDGE takes precedence over pair-conviction tiers. See spec §4.
"""

from dataclasses import dataclass


@dataclass
class ConvictionThresholds:
    """Spec §4 defaults."""
    high_top2: float = 0.33
    high_top3: float = 0.42
    med_top2: float = 0.27
    med_top3: float = 0.35
    low_top2: float = 0.22


def assign_conviction_tier(
    has_edge: bool, top_2_share: float, top_3_share: float,
    thresholds: ConvictionThresholds = None,
) -> str:
    t = thresholds or ConvictionThresholds()
    if has_edge:
        return 'EDGE'
    if top_2_share >= t.high_top2 or top_3_share >= t.high_top3:
        return 'HIGH'
    if top_2_share >= t.med_top2 or top_3_share >= t.med_top3:
        return 'MED'
    if top_2_share >= t.low_top2:
        return 'LOW'
    return 'SKIP'
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_conviction_tiers.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add betting/conviction.py tests/test_conviction_tiers.py
git commit -m "feat(dashboard): conviction tier assignment

EDGE | HIGH | MED | LOW | SKIP. EDGE takes precedence over pair-conviction
tiers. Thresholds dataclass makes them overridable for sweeps."
```

---

## Task 3: `betting/tier_stats.py` — Historical Hit-Rate + Wilson CI

**Files:**
- Create: `betting/tier_stats.py`
- Test: `tests/test_tier_stats_wilson_ci.py`
- Modify: `data/database.py` — add `conviction_tier` column to `prediction_log`

- [ ] **Step 1: Add `conviction_tier` column to `prediction_log`**

Find the `prediction_log` CREATE TABLE in `data/database.py` and add `conviction_tier TEXT DEFAULT NULL` to the schema. Then add an idempotent ALTER TABLE runner in the database `_migrate()` method (or equivalent startup hook) that executes `ALTER TABLE prediction_log ADD COLUMN conviction_tier TEXT DEFAULT NULL` wrapped in try/except for the OperationalError raised on an already-existing column.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_tier_stats_wilson_ci.py
"""Tier-stats hit-rate + Wilson 95% CI calculation."""
import math
import pytest
import sqlite3
import tempfile
import os
from betting.tier_stats import compute_tier_stats, wilson_ci_95


def test_wilson_ci_known_proportion():
    # 7/10 → Wilson 95% CI roughly [0.39, 0.89]
    low, high = wilson_ci_95(7, 10)
    assert math.isclose(low, 0.397, abs_tol=0.01)
    assert math.isclose(high, 0.893, abs_tol=0.01)


def test_wilson_ci_small_sample():
    # 1/3 has wide CI
    low, high = wilson_ci_95(1, 3)
    assert low < 0.10
    assert high > 0.80


def test_wilson_ci_zero_hits():
    # 0/10
    low, high = wilson_ci_95(0, 10)
    assert low == 0.0
    assert 0.20 < high < 0.35


def test_wilson_ci_zero_total():
    low, high = wilson_ci_95(0, 0)
    assert low == 0.0 and high == 1.0  # degenerate — caller checks for total > 0


def test_compute_tier_stats_end_to_end():
    # Build an in-memory predictions × results dataset, verify hit-rate
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    try:
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE prediction_log (
                race_date TEXT, race_number INTEGER, horse_name TEXT,
                horse_number INTEGER, predicted_rank INTEGER,
                conviction_tier TEXT, bet_recommendation TEXT,
                PRIMARY KEY (race_date, race_number, horse_name)
            );
            CREATE TABLE race_results (
                race_date TEXT, race_number INTEGER, horse_name TEXT,
                horse_number INTEGER, finish_position INTEGER, is_winner INTEGER,
                PRIMARY KEY (race_date, race_number, horse_name)
            );
        """)
        # Tier = HIGH, 3 races, rank-1 pick wins in 2 of them
        for i in range(3):
            conn.execute(
                "INSERT INTO prediction_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f'2026/04/{20+i:02d}', 1, f'H{i}', 1, 1, 'HIGH', 'WIN'),
            )
            conn.execute(
                "INSERT INTO race_results VALUES (?, ?, ?, ?, ?, ?)",
                (f'2026/04/{20+i:02d}', 1, f'H{i}', 1, 1 if i < 2 else 3, 1 if i < 2 else 0),
            )
        conn.commit()

        stats = compute_tier_stats(conn, window=10)
        assert 'HIGH' in stats
        assert stats['HIGH']['hit'] == 2
        assert stats['HIGH']['total'] == 3
        assert math.isclose(stats['HIGH']['rate'], 2 / 3, abs_tol=0.01)
        assert 0 < stats['HIGH']['ci_low'] < stats['HIGH']['rate']
        assert stats['HIGH']['rate'] < stats['HIGH']['ci_high'] < 1.0
    finally:
        conn.close()
        os.unlink(db_path)


def test_compute_tier_stats_sliding_window():
    """Only the last N races should contribute."""
    db_path = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name
    try:
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE prediction_log (
                race_date TEXT, race_number INTEGER, horse_name TEXT,
                horse_number INTEGER, predicted_rank INTEGER,
                conviction_tier TEXT, bet_recommendation TEXT,
                PRIMARY KEY (race_date, race_number, horse_name)
            );
            CREATE TABLE race_results (
                race_date TEXT, race_number INTEGER, horse_name TEXT,
                horse_number INTEGER, finish_position INTEGER, is_winner INTEGER,
                PRIMARY KEY (race_date, race_number, horse_name)
            );
        """)
        # 5 past races, all HIGH tier, first 2 hit, last 3 miss.
        # Window=2 should only see the last 2 (both misses → rate=0)
        for i in range(5):
            is_win = 1 if i < 2 else 0
            conn.execute(
                "INSERT INTO prediction_log VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f'2026/04/{20+i:02d}', 1, f'H{i}', 1, 1, 'HIGH', 'WIN'),
            )
            conn.execute(
                "INSERT INTO race_results VALUES (?, ?, ?, ?, ?, ?)",
                (f'2026/04/{20+i:02d}', 1, f'H{i}', 1, 1 if is_win else 5, is_win),
            )
        conn.commit()

        stats = compute_tier_stats(conn, window=2)
        assert stats['HIGH']['hit'] == 0
        assert stats['HIGH']['total'] == 2
    finally:
        conn.close()
        os.unlink(db_path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_tier_stats_wilson_ci.py -v`
Expected: FAIL — ModuleNotFoundError for `betting.tier_stats`

- [ ] **Step 4: Write minimal implementation**

```python
# betting/tier_stats.py
"""Per-tier hit-rate + Wilson 95% CI, computed from prediction_log × race_results.

Sliding-window over the last N completed races (default 73 — matches the
backtest holdout size). Cached to tracking_data/tier_stats.json; recomputed
on every results ingestion (see dashboard/results_fetcher.py integration).
"""

import json
import math
import os
import sqlite3
from typing import Dict, Tuple

from statsmodels.stats.proportion import proportion_confint

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tracking_data', 'tier_stats.json',
)


def wilson_ci_95(hits: int, total: int) -> Tuple[float, float]:
    """95% Wilson confidence interval on a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    low, high = proportion_confint(hits, total, alpha=0.05, method='wilson')
    return (float(low), float(high))


def compute_tier_stats(conn: sqlite3.Connection, window: int = 73) -> Dict:
    """Join prediction_log × race_results, group by conviction_tier, compute hit-rates.

    "Hit" for a WIN-tier (EDGE/HIGH/MED/LOW) means the rank-1 picked horse
    finished first. Only considers predictions that have a corresponding
    finish_position (race completed).

    window: only include the last `window` completed (race_date, race_number)
    pairs. Uses DISTINCT race_date, race_number sorted DESC and takes top N.
    """
    c = conn.cursor()
    # Step 1: get the window of most-recent completed races
    c.execute("""
        SELECT DISTINCT r.race_date, r.race_number
        FROM race_results r
        WHERE r.finish_position > 0
        ORDER BY r.race_date DESC, r.race_number DESC
        LIMIT ?
    """, (window,))
    window_races = c.fetchall()
    if not window_races:
        return {}

    # Step 2: for each tier, count rank-1 predictions that were winners
    # in the windowed races. Parameterize race filter as tuple of (date, number) IN.
    placeholders = ','.join(['(?, ?)'] * len(window_races))
    flat = [x for pair in window_races for x in pair]

    q = f"""
        SELECT p.conviction_tier,
               SUM(CASE WHEN r.is_winner = 1 THEN 1 ELSE 0 END) AS hits,
               COUNT(*) AS total
        FROM prediction_log p
        JOIN race_results r
          ON p.race_date = r.race_date
         AND p.race_number = r.race_number
         AND p.horse_name = r.horse_name
        WHERE p.predicted_rank = 1
          AND p.conviction_tier IS NOT NULL
          AND (p.race_date, p.race_number) IN ({placeholders})
        GROUP BY p.conviction_tier
    """
    c.execute(q, flat)

    stats = {}
    for tier, hits, total in c.fetchall():
        rate = hits / total if total > 0 else 0.0
        ci_low, ci_high = wilson_ci_95(hits, total)
        stats[tier] = {
            'hit': int(hits),
            'total': int(total),
            'rate': round(rate, 4),
            'ci_low': round(ci_low, 4),
            'ci_high': round(ci_high, 4),
        }
    return stats


def load_cached_stats() -> Dict:
    """Read tracking_data/tier_stats.json, return {} if missing."""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_cached_stats(stats: Dict) -> None:
    """Overwrite tracking_data/tier_stats.json."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(stats, f, indent=2)


def refresh_cache(conn: sqlite3.Connection, window: int = 73) -> Dict:
    """Recompute from DB and overwrite the cache file."""
    stats = compute_tier_stats(conn, window=window)
    write_cached_stats(stats)
    return stats
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_tier_stats_wilson_ci.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add betting/tier_stats.py tests/test_tier_stats_wilson_ci.py data/database.py
git commit -m "feat(dashboard): tier_stats + Wilson CI + prediction_log.conviction_tier

Sliding 73-race window over prediction_log × race_results joined on tier.
Wilson 95% CI via statsmodels. Cached to tracking_data/tier_stats.json;
refresh_cache() is called from results ingestion (next task).

prediction_log gets conviction_tier column (nullable) to let generator.py
populate going forward. Historical rows stay NULL — excluded from stats."
```

---

## Task 4: `betting/alerts.py` — Event Diff with Dedup

**Files:**
- Create: `betting/alerts.py`
- Test: `tests/test_alerts_dedup.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_alerts_dedup.py
"""Alert generation + dedup across snapshots."""
import pytest
from betting.alerts import compute_alerts, AlertState


def _make_race(race_num, horses, has_edge=False, minutes_to_post=30):
    return {
        'race_number': race_num,
        'minutes_to_post': minutes_to_post,
        'has_edge': has_edge,
        'horses': horses,
    }


def _horse(number, odds, edge_pp=0.0, prev_odds=None):
    h = {'number': number, 'win_odds': odds, 'edge_pp': edge_pp}
    if prev_odds is not None:
        h['prev_odds'] = prev_odds
    return h


def test_edge_opened_fires_once():
    prev = _make_race(5, [_horse(1, 8.3, edge_pp=10.0)], has_edge=False)
    curr = _make_race(5, [_horse(1, 8.3, edge_pp=10.0)], has_edge=True)
    state = AlertState()

    alerts_1 = compute_alerts([curr], [prev], state)
    assert any(a['type'] == 'EDGE_OPENED' and a['race'] == 5 for a in alerts_1)

    # Second call with same inputs — should NOT re-fire
    alerts_2 = compute_alerts([curr], [curr], state)
    assert not any(a['type'] == 'EDGE_OPENED' for a in alerts_2)


def test_edge_closed_suppresses_subsequent_edge_opened_same_race():
    # EDGE fires, then closes (odds moved), then briefly re-passes the gate —
    # we should NOT re-open on the same race within the same day.
    state = AlertState()
    r_open = _make_race(5, [_horse(1, 8.3, edge_pp=10.0)], has_edge=True)
    r_closed = _make_race(5, [_horse(1, 5.0, edge_pp=1.0)], has_edge=False)
    r_reopen = _make_race(5, [_horse(1, 8.3, edge_pp=10.0)], has_edge=True)

    compute_alerts([r_open], [], state)
    compute_alerts([r_closed], [r_open], state)  # emits EDGE_CLOSED
    alerts = compute_alerts([r_reopen], [r_closed], state)
    assert not any(a['type'] == 'EDGE_OPENED' for a in alerts)


def test_edge_widened_requires_prior_edge_state():
    # edge_pp grew ≥ 2pp — fires only if previously had edge
    prev = _make_race(5, [_horse(1, 8.3, edge_pp=6.0)], has_edge=True)
    curr = _make_race(5, [_horse(1, 8.3, edge_pp=9.0)], has_edge=True)
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    assert any(a['type'] == 'EDGE_WIDENED' for a in alerts)


def test_steam_fires_on_25pct_shortening():
    prev = _make_race(3, [_horse(4, 14.0)])
    curr = _make_race(3, [_horse(4, 9.5)])  # 14 → 9.5 = -32%, triggers
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    steam = [a for a in alerts if a['type'] == 'STEAM']
    assert len(steam) == 1
    assert steam[0]['race'] == 3
    assert steam[0]['horse'] == 4


def test_steam_does_not_fire_below_threshold():
    prev = _make_race(3, [_horse(4, 10.0)])
    curr = _make_race(3, [_horse(4, 9.0)])  # -10% only
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    assert not any(a['type'] == 'STEAM' for a in alerts)


def test_cold_fires_on_40pct_drift_of_top_pick():
    prev = _make_race(5, [_horse(2, 4.5)])  # assume top pick
    curr = _make_race(5, [_horse(2, 7.2)])  # 4.5 → 7.2 = +60%, triggers
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    cold = [a for a in alerts if a['type'] == 'COLD']
    assert len(cold) == 1


def test_race_upcoming_fires_at_6_min():
    curr = _make_race(3, [_horse(1, 5.0)], minutes_to_post=5)
    state = AlertState()

    alerts = compute_alerts([curr], [], state)
    upcoming = [a for a in alerts if a['type'] == 'RACE_UPCOMING']
    assert len(upcoming) == 1


def test_race_upcoming_dedupes_across_refreshes():
    curr = _make_race(3, [_horse(1, 5.0)], minutes_to_post=5)
    state = AlertState()
    compute_alerts([curr], [], state)
    alerts = compute_alerts([curr], [curr], state)  # second refresh
    assert not any(a['type'] == 'RACE_UPCOMING' for a in alerts)


def test_steam_dedupes_within_day():
    prev = _make_race(3, [_horse(4, 14.0)])
    curr = _make_race(3, [_horse(4, 9.5)])
    state = AlertState()
    compute_alerts([curr], [prev], state)
    alerts = compute_alerts([curr], [curr], state)  # still at 9.5
    assert not any(a['type'] == 'STEAM' for a in alerts)


def test_alerts_sorted_newest_first():
    # Two events same tick — order stable by emission sequence (newest first)
    prev = _make_race(3, [_horse(4, 14.0), _horse(2, 5.0)])
    curr = _make_race(3, [_horse(4, 9.5), _horse(2, 7.5)])  # steam #4, cold #2
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    # All share the same timestamp this tick; both present
    assert len(alerts) == 2
    types = {a['type'] for a in alerts}
    assert 'STEAM' in types and 'COLD' in types
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_alerts_dedup.py -v`
Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# betting/alerts.py
"""Snapshot-diff alert generation for the live rail.

State is carried across ticks in AlertState. Each (race, horse, type)
pair fires at most once per day — once EDGE_CLOSED emits for a race,
we also mark EDGE_OPENED as already-emitted for that race so the window
doesn't re-open on a subsequent tick. See spec §7.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple


@dataclass
class AlertState:
    """Carries dedup state across refresh ticks within a day."""
    emitted: Set[Tuple[int, int, str]] = field(default_factory=set)
    # (race, horse, type) — horse may be 0 for race-level alerts (RACE_UPCOMING)


def _already_emitted(state: AlertState, race: int, horse: int, typ: str) -> bool:
    return (race, horse, typ) in state.emitted


def _mark(state: AlertState, race: int, horse: int, typ: str) -> None:
    state.emitted.add((race, horse, typ))


def compute_alerts(current_races: List[Dict], previous_races: List[Dict],
                   state: AlertState) -> List[Dict]:
    """Produce alert events by diffing current vs previous race snapshots.

    Returned list is newest-first within this tick.
    Fields per alert: type, race, horse, ts, severity, detail.
    """
    prev_by_race = {r['race_number']: r for r in previous_races}
    out = []
    ts = int(time.time())

    for curr in current_races:
        race_num = curr['race_number']
        prev = prev_by_race.get(race_num, {})
        prev_horses = {h['number']: h for h in prev.get('horses', [])}

        # ── EDGE_OPENED / EDGE_CLOSED / EDGE_WIDENED ─────────────
        curr_edge = curr.get('has_edge', False)
        prev_edge = prev.get('has_edge', False)
        if curr_edge and not prev_edge and not _already_emitted(state, race_num, 0, 'EDGE_OPENED'):
            out.append({
                'type': 'EDGE_OPENED', 'race': race_num, 'horse': 0,
                'ts': ts, 'severity': 'strong', 'detail': None,
            })
            _mark(state, race_num, 0, 'EDGE_OPENED')
        elif prev_edge and not curr_edge and not _already_emitted(state, race_num, 0, 'EDGE_CLOSED'):
            out.append({
                'type': 'EDGE_CLOSED', 'race': race_num, 'horse': 0,
                'ts': ts, 'severity': 'mild', 'detail': None,
            })
            _mark(state, race_num, 0, 'EDGE_CLOSED')
            # Once closed, prevent re-open within the day (spec dedup rule)
            _mark(state, race_num, 0, 'EDGE_OPENED')

        if curr_edge and prev_edge:
            # Find the top edge_pp on each side
            curr_best_pp = max((h.get('edge_pp', 0) for h in curr['horses']), default=0)
            prev_best_pp = max((h.get('edge_pp', 0) for h in prev.get('horses', [])), default=0)
            if (curr_best_pp - prev_best_pp) >= 2.0 and not _already_emitted(state, race_num, 0, 'EDGE_WIDENED'):
                out.append({
                    'type': 'EDGE_WIDENED', 'race': race_num, 'horse': 0,
                    'ts': ts, 'severity': 'strong',
                    'detail': f'+{curr_best_pp - prev_best_pp:.1f}pp',
                })
                _mark(state, race_num, 0, 'EDGE_WIDENED')

        # ── STEAM / COLD per horse ───────────────────────────────
        for h in curr['horses']:
            num = h['number']
            curr_odds = h.get('win_odds', 0)
            prev_h = prev_horses.get(num, {})
            prev_odds = prev_h.get('win_odds', 0)
            if not (prev_odds and curr_odds):
                continue

            pct_change = (curr_odds - prev_odds) / prev_odds
            # Steam: shortened by ≥ 25%
            if pct_change <= -0.25 and not _already_emitted(state, race_num, num, 'STEAM'):
                out.append({
                    'type': 'STEAM', 'race': race_num, 'horse': num,
                    'ts': ts, 'severity': 'strong',
                    'detail': f'{prev_odds:.1f} → {curr_odds:.1f}',
                })
                _mark(state, race_num, num, 'STEAM')
            # Cold: drifted ≥ 40% on a top-pick. Simplification: any horse with
            # prev rank info isn't in our state, so just apply to all top-3 by
            # odds. A generator upstream can pass rank in the horse dict; for
            # now we use "was among the two lowest odds" as a proxy for top-pick.
            elif pct_change >= 0.40 and not _already_emitted(state, race_num, num, 'COLD'):
                # Top-pick proxy: was this horse in the two shortest prev odds?
                prev_sorted = sorted(
                    (ph for ph in prev.get('horses', []) if ph.get('win_odds', 0) > 0),
                    key=lambda x: x['win_odds'],
                )
                if prev_sorted and num in {ph['number'] for ph in prev_sorted[:2]}:
                    out.append({
                        'type': 'COLD', 'race': race_num, 'horse': num,
                        'ts': ts, 'severity': 'mild',
                        'detail': f'{prev_odds:.1f} → {curr_odds:.1f}',
                    })
                    _mark(state, race_num, num, 'COLD')

        # ── RACE_UPCOMING ────────────────────────────────────────
        mtp = curr.get('minutes_to_post')
        if mtp is not None and 0 <= mtp <= 6 and not _already_emitted(state, race_num, 0, 'RACE_UPCOMING'):
            out.append({
                'type': 'RACE_UPCOMING', 'race': race_num, 'horse': 0,
                'ts': ts, 'severity': 'mild', 'detail': f'{mtp}m to post',
            })
            _mark(state, race_num, 0, 'RACE_UPCOMING')

    # Newest-first ordering within tick — reverse insertion order
    return list(reversed(out))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_alerts_dedup.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add betting/alerts.py tests/test_alerts_dedup.py
git commit -m "feat(dashboard): alerts.py — 6 event types + dedup

Snapshot-diff across refresh ticks: EDGE_OPENED/CLOSED/WIDENED,
STEAM (odds -25%), COLD (+40% on top-pick proxy), RACE_UPCOMING
(≤6min). Dedup via (race, horse, type) set carried in AlertState.
EDGE_CLOSED marks EDGE_OPENED as emitted to prevent same-day re-open."
```

---

## Task 5: Wire New Modules into `dashboard/generator.py`

**Files:**
- Modify: `dashboard/generator.py` — lines ~185-272 (build_race_data) + lines ~289-334 (main)

- [ ] **Step 1: Add edge calculation after build_race_data's horse loop**

Locate `build_race_data` in `dashboard/generator.py`. After the loop that populates `race_data["horses"]` (just before `race_data["speed_map"] = speed_map`), insert:

```python
    # === Edge math (pure_prob, market_prob, edge fields, edge_fires) ===
    from betting.edge_calculator import (
        compute_pure_probs, compute_market_probs, compute_edge_fields,
        apply_edge_gates, EdgeGates,
    )
    from ml.integrity import check_model_integrity

    horses_for_edge = [{
        'rule_score': h['rule_score'], 'ml_prob': h['ml_prob'],
        'win_odds': h['win_odds'] or 0,
    } for h in race_data['horses']]

    pure_probs = compute_pure_probs(horses_for_edge)
    market_probs_edge = compute_market_probs(horses_for_edge)

    # Model integrity — used by G6. Cache the result per-run, not per-horse.
    try:
        integrity_ok = check_model_integrity(
            getattr(analyzer.ml_analyzer.ml_model, '_metadata', {})
        )
    except Exception:
        integrity_ok = False

    gates = EdgeGates()
    has_edge_race = False
    for i, h in enumerate(race_data['horses']):
        pure = pure_probs[i]
        market = market_probs_edge[i]
        odds = h['win_odds'] or 0
        fields = compute_edge_fields(pure, market, odds)

        # G5 — odds freshness. For now treat all odds in a fresh run as fresh.
        odds_age_min = 0  # generator.py fetches odds during this run

        model_rank = h['rank']  # already 1-indexed
        ok, _failed = apply_edge_gates(
            ev_win=fields['ev_win'], pure_prob=pure, edge_pp=fields['edge_pp'],
            model_rank=model_rank, odds_age_min=odds_age_min,
            integrity_ok=integrity_ok, gates=gates,
        )

        h['pure_prob'] = round(pure, 4)
        h['market_prob'] = round(market, 4)
        h['edge_pp'] = round(fields['edge_pp'], 2)
        h['ev_win'] = round(fields['ev_win'], 3)
        h['overlay'] = round(fields['overlay'], 3)
        h['edge_fires'] = ok
        h['model_rank'] = model_rank
        if ok:
            has_edge_race = True

    race_data['has_edge'] = has_edge_race
    race_data['odds_age_minutes'] = 0  # fresh this run
```

- [ ] **Step 2: Replace `strategy_confidence` block with conviction_tier**

In `build_race_data`, find the block that sets `race_data['strategy_confidence']`. Replace with:

```python
    # Conviction tier (replaces strategy_confidence)
    from betting.conviction import assign_conviction_tier
    race_data['conviction_tier'] = assign_conviction_tier(
        has_edge=race_data['has_edge'],
        top_2_share=top_2_share,
        top_3_share=top_3_share,
    )
    race_data['top_2_share'] = round(top_2_share, 4)
    race_data['top_3_share'] = round(top_3_share, 4)
```

Remove the lines that set `strategy_confidence = 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'`.

- [ ] **Step 3: Remove is_best_ev / is_worst_ev from main()**

In `main()`, find the block:

```python
    races_with_ev = [r for r in all_races if r.get('qin_bet_ev')]
    if len(races_with_ev) >= 2:
        best = max(...)
        worst = min(...)
        ...
        for r in all_races:
            r['is_best_ev'] = ...
            r['is_worst_ev'] = ...
```

Delete the entire block — `is_best_ev`/`is_worst_ev` is killed per spec §5.

- [ ] **Step 4: Add `_load_previous_race_data` helper near `_merge_with_existing`**

Add right after `_merge_with_existing` at module scope:

```python
def _load_previous_race_data(json_file: str) -> list:
    """Return race list from the most recent prior JSON at this path, or []."""
    if not os.path.exists(json_file):
        return []
    try:
        with open(json_file) as f:
            data = json.load(f)
        return data.get('races', data) if isinstance(data, dict) else data
    except (json.JSONDecodeError, OSError):
        return []
```

- [ ] **Step 5: Load tier_stats + alerts into each race**

In `main()`, after `_merge_with_existing(json_file, all_races)` and before `json.dump`:

```python
    # Tier stats — load from cache (refreshed on results ingestion)
    from betting.tier_stats import load_cached_stats
    tier_stats = load_cached_stats()
    for r in all_races:
        r['tier_stats'] = tier_stats  # same payload per race — small JSON

    # Alerts — diff current vs previous race_data file
    from betting.alerts import compute_alerts, AlertState
    prev_json = _load_previous_race_data(json_file)
    state = AlertState()
    alerts = compute_alerts(all_races, prev_json, state)
    # Attach each alert to its race (rail also renders the flattened list)
    for r in all_races:
        r['alerts'] = [a for a in alerts if a['race'] == r['race_number']]
```

- [ ] **Step 6: Log `conviction_tier` to prediction_log**

Find `data/database.py:log_predictions`. In the INSERT statement, add `conviction_tier` to the column list and the corresponding VALUES row so the generator's new tier is persisted. Extract the value via `race.get('conviction_tier')` in the Python loop.

- [ ] **Step 7: Run existing tests + integration smoke**

```bash
pytest tests/ -v -x
python3 generate_predictions.py '2026/04/22'  # race day with data already
python3 -c "import json; d=json.load(open('results/race_data_20260422.json')); r=d[0]; print('conviction_tier=', r.get('conviction_tier')); print('has_edge=', r.get('has_edge')); print('horse keys=', list(r['horses'][0].keys()))"
```

Expected: all tests pass; printed output shows `conviction_tier`, `has_edge`, and new horse-level keys (`pure_prob`, `edge_pp`, `ev_win`, `overlay`, `edge_fires`, `model_rank`).

- [ ] **Step 8: Commit**

```bash
git add dashboard/generator.py data/database.py
git commit -m "feat(dashboard): wire edge/conviction/alerts into race_data JSON

Adds pure_prob, market_prob, edge_pp, ev_win, overlay, edge_fires,
model_rank per horse. Adds has_edge, conviction_tier, odds_age_minutes,
alerts, tier_stats per race. Removes is_best_ev/is_worst_ev and
strategy_confidence. prediction_log now persists conviction_tier."
```

---

## Task 6: Rewrite `dashboard/template.py`

**Files:**
- Rewrite: `dashboard/template.py`
- Test: `tests/test_template_empty_state.py`

The existing `dashboard/template.py` is 68KB of inline JS/HTML. Instead of touching it surgically, this task replaces the rendered skeleton while preserving the race-detail components (horse table, speed map, WhatsApp copy).

- [ ] **Step 1: Write the failing test for empty-state rendering**

```python
# tests/test_template_empty_state.py
"""Template renders honest-day message when no race clears MED+ tier."""
import json
import re
from dashboard.template import generate_html


def _race(num, tier='SKIP', has_edge=False):
    return {
        'race_number': num, 'venue': 'Sha Tin', 'time': '13:00',
        'distance': 1200, 'class_type': 'Class 4', 'going': 'Good',
        'course': 'A', 'conviction_tier': tier, 'has_edge': has_edge,
        'top_2_share': 0.15, 'top_3_share': 0.22,
        'horses': [{'number': 1, 'name': 'Test', 'jockey': 'TJ', 'trainer': 'TT',
                    'draw': 1, 'win_odds': 5.0, 'rank': 1, 'pure_prob': 0.2,
                    'market_prob': 0.2, 'edge_pp': 0, 'ev_win': 0, 'overlay': 1.0,
                    'edge_fires': False, 'model_rank': 1, 'reasoning': [],
                    'calibrated_prob': 0.2, 'rule_score': 50, 'ml_prob': 0.2,
                    'combined_score': 20, 'expected_return': 0,
                    'bet_recommendation': 'SKIP', 'confidence': '',
                    'place_prob': 0.3, 'last_6': '', 'age': 4,
                    'horse_weight': 1100, 'days_since_last_run': 14,
                    'rating_change': 0, 'going_form': None,
                    'jockey_horse_stats': None, 'distance_pb': None,
                    'distance_record': None, 'odds_movement': {},
                    'steam_move': {'is_steam': False}, 'cold_horse': {'is_cold': False},
                    'weight': 130, 'gear': '', 'rating': 60, 'place_odds': 2.5}],
        'speed_map': {}, 'alerts': [], 'tier_stats': {},
    }


def test_empty_state_renders_honest_day_message():
    # All 11 races SKIP tier
    races = [_race(i) for i in range(1, 12)]
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')

    assert 'No bets meet the threshold today' in html
    assert 'skipping the card is the dashboard' in html.lower() or \
           'skip the card' in html.lower()


def test_hero_renders_edge_race():
    races = [_race(1), _race(5, tier='EDGE', has_edge=True)]
    races[1]['horses'][0]['edge_fires'] = True
    races[1]['horses'][0]['edge_pp'] = 10.0
    races[1]['horses'][0]['ev_win'] = 0.83
    races[1]['top_2_share'] = 0.45
    races[1]['top_3_share'] = 0.55
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')

    # EDGE badge appears in hero
    assert 'EDGE' in html
    assert '★' in html or 'edge' in html.lower()


def test_strip_shows_all_11_races():
    races = [_race(i) for i in range(1, 12)]
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')
    # R1 through R11 all appear in the strip
    for i in range(1, 12):
        assert re.search(rf'\bR{i}\b', html), f'R{i} missing from strip'


def test_killed_fields_not_referenced():
    """strategy_confidence, is_best_ev, combined_score should not drive UI."""
    races = [_race(1)]
    # Old-style fields intentionally NOT set
    html = generate_html(json.dumps(races), '2026-04-27 13:00:00', '2026/04/27')
    # Hard assertions that old class names / copy from the removed system are gone.
    # These are the telltale strings from the old template.
    assert 'is_best_ev' not in html.lower()
    # A+/A/B+/B/B- grade system should be gone
    assert 'grade-aplus' not in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_template_empty_state.py -v`
Expected: FAIL — may pass trivially on "not referenced" checks but hero/empty-state won't render the new text.

- [ ] **Step 3: Rewrite `dashboard/template.py`**

Replace the entire file contents with a new implementation. The rewritten file should:

1. Keep the `generate_html(races_json, now, race_date, venue='Sha Tin', auto_refresh=False)` signature — backward-compatible with `dashboard/generator.py`.
2. Keep the XSS-safe `esc()` helper and the `</script>` sanitization on `races_json`.
3. Keep the existing refresh banner, WhatsApp buttons, clock.
4. Replace the old dashboard renderer with: **top bar → hero → strip → race-detail zone → alerts rail** (on ≥ 1024px) or stacked (< 1024px).

Skeleton (trim — full content ~600-800 lines; implementer fills in visual polish):

```python
# dashboard/template.py — rewrite
"""HKJC dashboard HTML generator — edge-first layout.

Renders: top bar, hero (0-5 bet cards with trust cards), 11-race strip,
race detail (click to expand), sticky alerts rail on desktop/iPad landscape.

Legacy grade system (A+/A/B+/B/B-), strategy_confidence, is_best_ev,
and combined_score are NOT rendered. See spec §3 for IA rules and §5
for field hygiene verdicts.
"""

TIER_COLORS = {
    'EDGE': '#fbbf24', 'HIGH': '#10b981', 'MED': '#f59e0b',
    'LOW': '#64748b', 'SKIP': '#475569',
}

ALERT_COLORS = {
    'EDGE_OPENED': '#fbbf24', 'EDGE_WIDENED': '#fbbf24',
    'EDGE_CLOSED': '#94a3b8', 'STEAM': '#ef4444',
    'COLD': '#06b6d4', 'RACE_UPCOMING': '#10b981',
}


def generate_html(races_json: str, now: str, race_date: str,
                  venue: str = 'Sha Tin', auto_refresh: bool = False) -> str:
    # XSS defense — strip </script> from JSON payload
    races_json_safe = races_json.replace('</script>', r'<\/script>') \
                                .replace('</Script>', r'<\/Script>')

    return _HTML_TEMPLATE.format(
        races_json=races_json_safe,
        now=now,
        race_date=race_date,
        venue=venue,
        auto_refresh_meta='<meta http-equiv="refresh" content="60">' if auto_refresh else '',
    )


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head>
  <meta charset="UTF-8">
  <title>HKJC Racing Dashboard — {race_date}</title>
  {auto_refresh_meta}
  <style>
    :root {{ --bg:#0f172a; --panel:#1e293b; --border:#334155; --fg:#e2e8f0; --dim:#94a3b8;
             --gold:#fbbf24; --green:#10b981; --amber:#f59e0b; --red:#ef4444; --cyan:#06b6d4;
             --violet:#8b5cf6; }}
    body {{ margin:0; font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--fg); }}
    .topbar {{ display:flex; justify-content:space-between; padding:12px 20px; border-bottom:1px solid var(--border); }}
    .shell {{ display:grid; grid-template-columns:1fr 280px; gap:16px; padding:16px; max-width:1400px; margin:0 auto; }}
    @media (max-width:1024px) {{ .shell {{ grid-template-columns:1fr; }} .alerts-rail {{ position:static !important; }} }}
    .hero {{ background:rgba(15,23,42,.4); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:12px; }}
    .strip {{ display:grid; grid-template-columns:repeat(11,1fr); gap:4px; padding:10px; background:rgba(15,23,42,.3); border:1px solid var(--border); border-radius:10px; }}
    .strip-tile {{ padding:8px 4px; border-radius:4px; text-align:center; cursor:pointer; transition:transform .1s; }}
    .strip-tile:hover {{ transform:translateY(-2px); }}
    .tier-EDGE {{ background:rgba(251,191,36,.2); border:1px solid rgba(251,191,36,.5); }}
    .tier-HIGH {{ background:rgba(16,185,129,.15); border:1px solid rgba(16,185,129,.35); }}
    .tier-MED  {{ background:rgba(245,158,11,.15); border:1px solid rgba(245,158,11,.35); }}
    .tier-LOW  {{ background:rgba(100,116,139,.2); }}
    .tier-SKIP {{ background:rgba(71,85,105,.2); opacity:.6; }}
    .alerts-rail {{ position:sticky; top:16px; align-self:start; background:rgba(15,23,42,.4); border:1px solid var(--border); border-radius:10px; padding:12px; }}
    .alert {{ padding:8px; border-radius:4px; margin-bottom:6px; border-left:3px solid; }}
    .trust-card {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-top:10px; }}
    .trust-block {{ background:rgba(15,23,42,.5); padding:8px; border-radius:6px; }}
    .empty-hero {{ background:rgba(15,23,42,.3); border:1px dashed var(--border); border-radius:10px; padding:24px; text-align:center; }}
  </style>
</head><body>

<div class="topbar">
  <div><strong>HKJC</strong> <span style="color:var(--dim)">{race_date} · {venue}</span></div>
  <div style="color:var(--dim)">Last refresh {now}</div>
</div>

<div class="shell">
  <div>
    <div id="hero" class="hero"></div>
    <div id="strip" class="strip"></div>
    <div id="race-detail" style="margin-top:12px"></div>
  </div>
  <div class="alerts-rail" id="alerts-rail"></div>
</div>

<script>
const RACES = {races_json};

function esc(s) {{ var d=document.createElement('div'); d.textContent=String(s!=null?s:''); return d.innerHTML; }}

function renderHero() {{
  const hero = document.getElementById('hero');
  const bets = RACES
    .filter(r => ['EDGE', 'HIGH', 'MED'].includes(r.conviction_tier))
    .sort((a, b) => {{
      const order = {{ EDGE: 0, HIGH: 1, MED: 2 }};
      if (order[a.conviction_tier] !== order[b.conviction_tier])
        return order[a.conviction_tier] - order[b.conviction_tier];
      const aPp = Math.max(...a.horses.map(h => h.edge_pp || 0));
      const bPp = Math.max(...b.horses.map(h => h.edge_pp || 0));
      if (aPp !== bPp) return bPp - aPp;
      return (b.top_3_share || 0) - (a.top_3_share || 0);
    }})
    .slice(0, 5);

  if (bets.length === 0) {{
    // Find closest miss among SKIP/LOW
    const closest = RACES
      .filter(r => r.top_3_share != null)
      .sort((a, b) => (b.top_3_share || 0) - (a.top_3_share || 0))[0];
    const closestMsg = closest
      ? `Closest to HIGH: R${{closest.race_number}} (top_3_share ${{(closest.top_3_share * 100).toFixed(0)}}%)`
      : '';
    hero.innerHTML = `<div class="empty-hero">
      <div style="font-size:2rem">🏇</div>
      <div style="font-size:1.1rem;font-weight:700;margin-bottom:6px">No bets meet the threshold today</div>
      <div style="color:var(--dim);max-width:440px;margin:0 auto;line-height:1.6">
        Model and market agree too closely for an edge play, and no race clears HIGH/MED conviction on pair identification. The strip below shows all races if you want to drill in, but <strong>skipping the card is the dashboard's recommendation.</strong>
      </div>
      ${{closestMsg ? `<div style="margin-top:12px;font-size:.8rem;color:var(--amber)">${{esc(closestMsg)}}</div>` : ''}}
    </div>`;
    return;
  }}

  hero.innerHTML = `<div style="font-size:.62rem;color:var(--dim);letter-spacing:.12em;margin-bottom:8px">TODAY'S BETS — ${{bets.length}} OF ${{RACES.length}} RACES</div>` +
    bets.map(renderBetCard).join('');
}}

function renderBetCard(race) {{
  const topHorse = race.horses[0];
  const tier = race.conviction_tier;
  const isEdge = tier === 'EDGE';
  // Why-strong bullets: top 3 of topHorse.reasoning
  const whyBullets = (topHorse.reasoning || []).slice(0, 3)
    .map(r => `<li>${{esc(r)}}</li>`).join('');

  // Tier track record
  const stats = (RACES[0].tier_stats || {{}})[tier];
  const trackRecord = stats
    ? `<div>${{tier}} hit ${{stats.hit}}/${{stats.total}} (${{(stats.rate*100).toFixed(0)}}%)<br>
       <span style="color:var(--dim);font-size:.68rem">Wilson [${{(stats.ci_low*100).toFixed(0)}}-${{(stats.ci_high*100).toFixed(0)}}%]</span></div>`
    : `<div style="color:var(--dim)">Insufficient sample</div>`;

  return `
    <div style="background:${{isEdge ? 'linear-gradient(135deg,rgba(251,191,36,.18),rgba(251,191,36,.06))' : 'rgba(16,185,129,.08)'}};
                border:1px solid ${{isEdge ? 'rgba(251,191,36,.4)' : 'rgba(16,185,129,.25)'}};
                border-radius:10px;padding:14px;margin-bottom:8px">
      <div style="display:flex;justify-content:space-between">
        <div>
          <span style="background:${{TIER_COLORS[tier] || '#64748b'}};color:#000;padding:2px 8px;border-radius:4px;font-size:.6rem;font-weight:800">
            ${{isEdge ? '★ EDGE' : esc(tier)}}
          </span>
          <strong style="margin-left:8px">R${{race.race_number}} · ${{esc(race.time || '')}} · ${{esc(race.class_type || '')}} ${{race.distance}}m</strong>
        </div>
        <div style="color:var(--dim);font-size:.7rem">
          ${{isEdge ? `+${{topHorse.edge_pp.toFixed(1)}}pp gap · ev +${{topHorse.ev_win.toFixed(2)}}` : `top_3 ${{(race.top_3_share*100).toFixed(0)}}%`}}
        </div>
      </div>
      <div style="margin-top:6px">
        <strong>#${{topHorse.number}} ${{esc(topHorse.name)}}</strong>
        · odds ${{topHorse.win_odds}}
        · ${{esc(topHorse.bet_recommendation)}}
      </div>
      <div class="trust-card">
        <div class="trust-block">
          <div style="font-size:.6rem;color:var(--dim);letter-spacing:.1em">WHY STRONG</div>
          <ul style="font-size:.72rem;margin:6px 0 0 14px;padding:0">${{whyBullets || '<li style="color:var(--dim)">No reasoning captured</li>'}}</ul>
        </div>
        <div class="trust-block">
          <div style="font-size:.6rem;color:var(--dim);letter-spacing:.1em">OUR VS MARKET</div>
          <div style="font-size:.72rem;line-height:1.5;margin-top:4px">
            Pure <strong style="color:var(--green)">${{(topHorse.pure_prob*100).toFixed(0)}}%</strong><br>
            Market <strong style="color:var(--dim)">${{(topHorse.market_prob*100).toFixed(0)}}%</strong><br>
            Edge <strong style="color:var(--gold)">${{topHorse.edge_pp.toFixed(1)}}pp</strong>
          </div>
        </div>
        <div class="trust-block">
          <div style="font-size:.6rem;color:var(--dim);letter-spacing:.1em">TIER TRACK RECORD</div>
          <div style="font-size:.72rem;margin-top:4px">${{trackRecord}}</div>
        </div>
      </div>
    </div>`;
}}

function renderStrip() {{
  const strip = document.getElementById('strip');
  strip.innerHTML = RACES.map(r => {{
    const tier = r.conviction_tier || 'SKIP';
    const badge = tier === 'EDGE' ? 'R' + r.race_number + ' ★' : 'R' + r.race_number;
    const subtype = tier === 'SKIP' ? 'skip' : (r.horses[0]?.bet_recommendation || '');
    return `<div class="strip-tile tier-${{tier}}" onclick="showRaceDetail(${{r.race_number}})">
      <div style="font-size:.72rem;font-weight:700">${{esc(badge)}}</div>
      <div style="font-size:.6rem;color:var(--dim)">${{esc(subtype)}}</div>
    </div>`;
  }}).join('');
}}

function renderAlertsRail() {{
  const rail = document.getElementById('alerts-rail');
  const allAlerts = RACES.flatMap(r => r.alerts || []).sort((a, b) => b.ts - a.ts);
  if (allAlerts.length === 0) {{
    rail.innerHTML = `<div style="font-size:.62rem;color:var(--dim);letter-spacing:.12em">LIVE ALERTS</div>
      <div style="color:var(--dim);font-size:.78rem;margin-top:10px">No live events yet.</div>`;
    return;
  }}
  rail.innerHTML = `<div style="font-size:.62rem;color:var(--dim);letter-spacing:.12em;margin-bottom:10px">LIVE ALERTS</div>` +
    allAlerts.map(a => {{
      const color = {{'EDGE_OPENED':'#fbbf24','EDGE_WIDENED':'#fbbf24','EDGE_CLOSED':'#94a3b8','STEAM':'#ef4444','COLD':'#06b6d4','RACE_UPCOMING':'#10b981'}}[a.type] || '#64748b';
      const label = a.type.replace('_', ' ');
      return `<div class="alert" style="border-left-color:${{color}};background:${{color}}14">
        <strong style="font-size:.72rem">${{esc(label)}}</strong>
        <div style="font-size:.66rem;color:var(--dim)">R${{a.race}} ${{a.horse ? '#'+a.horse : ''}} ${{a.detail ? '· '+esc(a.detail) : ''}}</div>
      </div>`;
    }}).join('');
}}

function showRaceDetail(raceNum) {{
  const race = RACES.find(r => r.race_number === raceNum);
  if (!race) return;
  const det = document.getElementById('race-detail');
  // Existing horse table, speed map, reasoning — keep simple table render here.
  // Full polish carried forward from old template in follow-up pass.
  det.innerHTML = `<div style="background:rgba(15,23,42,.3);border:1px solid var(--border);border-radius:10px;padding:16px">
    <h3>R${{race.race_number}} · ${{esc(race.time || '')}} · ${{race.distance}}m</h3>
    <table style="width:100%;border-collapse:collapse;font-size:.82rem">
      <thead><tr style="border-bottom:1px solid var(--border);color:var(--dim);font-size:.66rem">
        <th style="text-align:left;padding:6px">#</th>
        <th style="text-align:left;padding:6px">NAME</th>
        <th style="text-align:right;padding:6px">PURE</th>
        <th style="text-align:right;padding:6px">MARKET</th>
        <th style="text-align:right;padding:6px">EDGE</th>
        <th style="text-align:right;padding:6px">ODDS</th>
        <th style="text-align:left;padding:6px">BET</th>
      </tr></thead>
      <tbody>${{race.horses.map(h => `
        <tr style="border-bottom:1px solid var(--border)">
          <td style="padding:6px">${{h.number}}</td>
          <td style="padding:6px">${{esc(h.name)}}</td>
          <td style="padding:6px;text-align:right">${{(h.pure_prob*100).toFixed(1)}}%</td>
          <td style="padding:6px;text-align:right">${{(h.market_prob*100).toFixed(1)}}%</td>
          <td style="padding:6px;text-align:right;color:${{h.edge_pp > 0 ? 'var(--green)' : 'var(--dim)'}}">${{h.edge_pp.toFixed(1)}}</td>
          <td style="padding:6px;text-align:right">${{h.win_odds || '—'}}</td>
          <td style="padding:6px">${{esc(h.bet_recommendation)}}</td>
        </tr>`).join('')}}</tbody>
    </table>
  </div>`;
  det.scrollIntoView({{behavior: 'smooth'}});
}}

renderHero();
renderStrip();
renderAlertsRail();
</script>
</body></html>"""
```

- [ ] **Step 4: Run tests + smoke**

```bash
pytest tests/test_template_empty_state.py -v
# Plus existing tests to confirm nothing else broke
pytest tests/ -v -x
# Smoke: regenerate and eyeball the output
python3 generate_predictions.py '2026/04/22'
open results/racing_dashboard.html    # macOS — visual inspection
```

Expected: 4 new template tests pass; the dashboard opens in browser and shows hero/strip/alerts-rail layout.

- [ ] **Step 5: Commit**

```bash
git add dashboard/template.py tests/test_template_empty_state.py
git commit -m "feat(dashboard): rewrite template.py with edge-first layout

Replaces the 68KB legacy template with hero + strip + alerts rail +
trust card skeleton. Empty state renders honest-day message when no
race clears MED+ tier. Grade A+/A/B+/B/B- system removed. Backward-
compatible generate_html signature.

Race detail keeps the horse table inline; full race-detail polish
(speed map, reasoning list, WhatsApp copy) carries forward in next
pass. 4 new rendering assertions in test_template_empty_state.py."
```

---

## Task 7: Patch `monitor/race_day.py` — Auto-Rerun on Sharp Move

**Files:**
- Modify: `monitor/race_day.py`

- [ ] **Step 1: Inspect the existing `regenerate_dashboard` call site**

Before editing, open `monitor/race_day.py` and find the `regenerate_dashboard(...)` call inside `run_monitor` (around line 176 per git blame). Note the exact keyword arguments it takes (e.g. `fetcher=fetcher, analyzer=analyzer, odds_db=odds_db`) so you can reproduce them verbatim in the `else` branch of step 2.

- [ ] **Step 2: Add sharp-move detector, gate `regenerate_dashboard` behind it**

In `monitor/race_day.py`, inside `run_monitor` after `snapshot_odds(...)` and before the existing `regenerate_dashboard(...)` call, add:

```python
            # Sharp-move auto-rerun trigger (spec §7): if a top-pick from the
            # most recent race_data moved ≥ 25% (steam) or ≥ 40% (cold), force
            # an unconditional full prediction re-run before re-rendering.
            date_str = race_date.replace('/', '')
            race_data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'results', f'race_data_{date_str}.json',
            )
            sharp_move = False
            if os.path.exists(race_data_path):
                with open(race_data_path) as f:
                    prev = json.load(f)
                for race in prev:
                    for h in race.get('horses', [])[:2]:  # top 2 picks only
                        prev_odds = h.get('win_odds', 0)
                        curr_odds = _look_up_latest_odds(
                            odds_db, race_date, race['race_number'], h['number']
                        )
                        if prev_odds and curr_odds:
                            pct = (curr_odds - prev_odds) / prev_odds
                            if pct <= -0.25 or pct >= 0.40:
                                sharp_move = True
                                logger.info(
                                    f"Sharp move on R{race['race_number']} #{h['number']}: "
                                    f"{prev_odds} → {curr_odds} ({pct:+.1%}) — forcing re-run"
                                )
                                break
                    if sharp_move:
                        break

            if sharp_move:
                # Full re-run in subprocess — re-fetches cards AND odds
                subprocess.run(
                    ['python3', 'generate_predictions.py', race_date],
                    check=False, timeout=300,
                )
            else:
                # Keep the existing call exactly as it was before this patch
                await regenerate_dashboard(
                    race_date, fetcher=fetcher, analyzer=analyzer, odds_db=odds_db,
                )
```

Also add the helper at module scope:

```python
def _look_up_latest_odds(odds_db, race_date, race_num, horse_num):
    """Peek the most recent win_odds for a horse from odds history."""
    try:
        row = odds_db.conn.execute(
            "SELECT win_odds FROM odds_history "
            "WHERE race_date = ? AND race_number = ? AND horse_number = ? "
            "ORDER BY ts DESC LIMIT 1",
            (race_date, race_num, horse_num),
        ).fetchone()
        return row[0] if row else 0
    except Exception:
        return 0
```

Ensure `import subprocess, json, os` are present at the top of the file (add any that are missing).

- [ ] **Step 3: Manual smoke test**

```bash
# Run the monitor against a known date and force odds changes via DB
python3 -m scripts.monitor '2026/04/22' &
MONITOR_PID=$!
sleep 30
# Inspect logs for 'forcing re-run' or 'Sharp move' messages
kill $MONITOR_PID
```

Expected: monitor starts, polls odds, and when a sharp move is detected triggers `generate_predictions.py` in a subprocess. No crash.

- [ ] **Step 4: Commit**

```bash
git add monitor/race_day.py
git commit -m "feat(dashboard): monitor auto-rerun on sharp STEAM/COLD move

When a top-2 pick moves ≥ 25% (steam) or ≥ 40% (cold) between refresh
ticks, force a full generate_predictions.py re-run instead of just
re-rendering the dashboard. Fixes Q2-C 'slow to react' pain."
```

---

## Task 7b: Race-Detail Parity + WhatsApp Copy

The Task 6 rewrite keeps the race-detail table minimal so the hero/strip/alerts can land first. This task restores the remaining race-detail content the spec marks as KEEP/DEMOTE (speed map, reasoning list, per-horse detail fields) and the WhatsApp export (§11).

**Files:**
- Modify: `dashboard/template.py` — expand `showRaceDetail` + add `copyToWhatsApp`

- [ ] **Step 1: Expand `showRaceDetail` in `template.py`**

Find the `showRaceDetail(raceNum)` function in `dashboard/template.py`. Replace its body so that in addition to the horse table it also renders:

```javascript
  // Speed map (demoted from hero, kept in detail per spec §5)
  const speedMap = race.speed_map || {};
  const speedMapHTML = Object.keys(speedMap).length
    ? `<div style="margin-top:16px"><div style="font-size:.62rem;color:var(--dim);letter-spacing:.12em;margin-bottom:6px">SPEED MAP</div>
         <pre style="background:rgba(15,23,42,.4);padding:10px;border-radius:6px;font-size:.72rem;margin:0;white-space:pre-wrap">${esc(JSON.stringify(speedMap, null, 2))}</pre></div>`
    : '';

  // Full reasoning list for each horse
  const reasoningHTML = race.horses.map(h =>
    (h.reasoning || []).length
      ? `<div style="margin-top:10px"><strong>#${h.number} ${esc(h.name)}</strong>
           <ul style="margin:4px 0 0 18px;font-size:.78rem">${(h.reasoning || []).map(r => `<li>${esc(r)}</li>`).join('')}</ul></div>`
      : ''
  ).join('');

  // Per-horse detail row: going_form, jockey_horse_stats, distance_pb, weight change, etc.
  // Render as additional columns on the horse table when present, else omit gracefully.
```

Re-render `det.innerHTML` to append `${speedMapHTML}` and a `<div>${reasoningHTML}</div>` block after the horse table. Additional per-horse columns (last_6, horse_weight, going_form hits, jockey_horse starts) surface as a second table below the primary one — keep columns conditional on non-null values to avoid empty cells on sparse races.

- [ ] **Step 2: Add WhatsApp copy button (§11 — shrunk payload)**

Add inside the hero bet card renderer in `renderBetCard`:

```javascript
  const waBtn = `<button onclick="copyBetToWA(${race.race_number})"
    style="background:linear-gradient(135deg,#25d366,#128c7e);color:#fff;border:none;
           padding:6px 12px;border-radius:6px;font-size:.72rem;font-weight:700;
           cursor:pointer;margin-top:10px">Copy WA</button>`;
```

And the companion function:

```javascript
function copyBetToWA(raceNum) {
  const race = RACES.find(r => r.race_number === raceNum);
  if (!race) return;
  const h = race.horses[0];
  const tier = race.conviction_tier;
  const whyLines = (h.reasoning || []).slice(0, 3).map(r => '• ' + r).join('\n');
  const text = `🏇 R${raceNum} · ${race.class_type} ${race.distance}m\n` +
    `${tier === 'EDGE' ? '★ EDGE' : tier}: ${h.bet_recommendation} #${h.number} ${h.name}\n` +
    `odds ${h.win_odds} · pure ${(h.pure_prob*100).toFixed(0)}% vs market ${(h.market_prob*100).toFixed(0)}%\n` +
    (tier === 'EDGE' ? `edge +${h.edge_pp.toFixed(1)}pp · ev +${h.ev_win.toFixed(2)}\n` : '') +
    whyLines;
  navigator.clipboard.writeText(text);
}
```

Also add a "Copy all bets" button in the hero header that concatenates the above for each bet card in `bets`, separated by `\n\n---\n\n`.

- [ ] **Step 3: Extend `test_template_empty_state.py` with a parity test**

```python
def test_race_detail_renders_reasoning_and_speed_map():
    race = _race(1, tier='HIGH')
    race['horses'][0]['reasoning'] = ['Jockey change to Purton', 'Draw 4 sweet spot']
    race['speed_map'] = {'leaders': [1, 4], 'stalkers': [2, 7]}
    html = generate_html(json.dumps([race]), '2026-04-27 13:00:00', '2026/04/27')
    # Reasoning text shows up somewhere in the rendered HTML
    assert 'Jockey change to Purton' in html
    # Speed map keys present
    assert 'leaders' in html
```

Run: `pytest tests/test_template_empty_state.py -v`
Expected: all 5 tests pass.

- [ ] **Step 4: Commit**

```bash
git add dashboard/template.py tests/test_template_empty_state.py
git commit -m "feat(dashboard): race-detail parity + WhatsApp copy (shrunk)

Restores speed map and full reasoning list in race-detail view (§5
DEMOTE verdict). Adds WhatsApp copy with shrunk payload: hero bets
only, trust-card summary (§11). No longer dumps every horse's stats."
```

---

## Task 8: Invalidate `tier_stats.json` on Results Ingestion

**Files:**
- Modify: `dashboard/results_fetcher.py`

- [ ] **Step 1: Refresh cache after results ingestion**

In `dashboard/results_fetcher.py`, locate the code path that ingests new results into the DB (likely after `db.ingest_race_day()` or after `fill_prediction_results()`). Add:

```python
            # Tier-stats cache refresh — new results shift the windowed hit-rate
            from betting.tier_stats import refresh_cache
            try:
                refresh_cache(db.conn, window=73)
                logger.info("Refreshed tier_stats.json cache")
            except Exception as e:
                logger.warning(f"tier_stats refresh failed: {e}")
```

- [ ] **Step 2: Smoke test**

```bash
python3 results_fetcher.py '2026/04/22'
# Verify
python3 -c "import json; print(json.load(open('tracking_data/tier_stats.json')))"
```

Expected: `tracking_data/tier_stats.json` exists and contains per-tier hit/total/rate/ci_low/ci_high keys.

- [ ] **Step 3: Commit**

```bash
git add dashboard/results_fetcher.py
git commit -m "feat(dashboard): refresh tier_stats.json on results ingestion

Each time new finishing positions land, re-compute the sliding-73-race
tier hit-rates + Wilson CIs and overwrite tracking_data/tier_stats.json.
Generator loads the cache into race_data on the next dashboard run."
```

---

## Final Verification

- [ ] **Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all existing tests + the 5 new test files pass. Count should be the prior total + ~40 new cases.

- [ ] **Regenerate a real race day end-to-end**

```bash
python3 generate_predictions.py '2026/04/22'
open results/racing_dashboard.html
```

Eyeball check:
- Hero renders 0-5 bet cards (or empty-hero message with closest-miss tag)
- 11-race strip is visible, coloured by tier, clickable
- Alerts rail on right (desktop) or below hero (narrow window)
- Trust card renders 3 blocks: Why Strong · Our vs Market · Tier Track Record
- Clicking a strip tile reveals race-detail with horse table + speed map + reasoning list
- WhatsApp "Copy WA" button on each hero bet produces a shrunk payload (bet + trust summary)
- No `combined_score`, no `is_best_ev`, no A+/A grade system visible

- [ ] **Push branch + tag**

```bash
git tag dashboard-redesign-complete
# User decides on push timing — do not push without explicit request
```
