# Enhanced Analytics & Race Day Monitor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize codebase into packages, add enhanced analytics (going form, jockey-horse stats, distance PBs, speed maps, steam/cold detection), extend ML to 18 features, and build a race day monitor with auto-refresh dashboard.

**Architecture:** Two-phase approach. Phase 1 moves existing files into a package structure with zero behavior changes. Phase 2 adds new DB queries, ML features, odds detection, speed maps, race day monitor, and dashboard sections — all as incremental additions to the refactored structure.

**Tech Stack:** Python 3, SQLite, XGBoost, asyncio, Playwright (fetching), HTML/CSS/JS (dashboard)

---

## Phase 1: Package Refactor

### Task 1: Create package directories and `__init__.py` files

**Files:**
- Create: `core/__init__.py`, `analytics/__init__.py`, `data/__init__.py`, `monitor/__init__.py`, `dashboard/__init__.py`, `betting/__init__.py`, `scripts/__init__.py`

- [ ] **Step 1: Create all package directories**

```bash
mkdir -p core analytics data monitor dashboard betting scripts
```

- [ ] **Step 2: Create `core/__init__.py`**

```python
"""Core analysis pipeline: fetcher, analyzer, ML predictor."""
```

- [ ] **Step 3: Create `analytics/__init__.py` with shared utility**

```python
"""Specialized racing analyzers (draw bias, pace, sectionals, etc.)."""

import re
from typing import List


def parse_form_positions(last_6: str) -> List[int]:
    """Parse form string into list of finishing positions.
    Handles both '1/3/10/2/5' (slash-separated) and legacy '13425' (digit-by-digit)."""
    if not last_6:
        return []
    s = str(last_6)
    if "/" in s:
        return [int(p) for p in s.split("/") if p.strip().isdigit()]
    else:
        return [int(ch) for ch in s if ch.isdigit()]
```

- [ ] **Step 4: Create remaining `__init__.py` files**

`data/__init__.py`:
```python
"""Data layer: database, odds tracking, scraping."""
```

`monitor/__init__.py`:
```python
"""Race day monitoring and auto-refresh."""
```

`dashboard/__init__.py`:
```python
"""Dashboard generation: predictions JSON + HTML."""
```

`betting/__init__.py`:
```python
"""Betting optimization and strategy."""
```

`scripts/__init__.py`:
```python
"""CLI entry points."""
```

- [ ] **Step 5: Commit**

```bash
git add core/ analytics/ data/ monitor/ dashboard/ betting/ scripts/
git commit -m "Create package directory structure with __init__.py files"
```

---

### Task 2: Move core modules (fetcher, analyzer, predictor)

**Files:**
- Move: `hkjc_data_fetcher.py` → `core/fetcher.py`
- Move: `hkjc_racing_analyzer.py` → `core/analyzer.py`
- Move: `ml_predictor.py` → `core/predictor.py`
- Create: root-level shims for backwards compatibility

- [ ] **Step 1: Copy files to new locations**

```bash
cp hkjc_data_fetcher.py core/fetcher.py
cp hkjc_racing_analyzer.py core/analyzer.py
cp ml_predictor.py core/predictor.py
```

- [ ] **Step 2: Update imports in `core/analyzer.py`**

Change the import block at top (lines 17-44) from:

```python
from hkjc_data_fetcher import HKJCDataFetcher, HorseEntry, RaceCard

from ml_predictor import (
    HKJCRacingMLModel,
    FeatureEngineering,
    CombinedRacingAnalyzer,
    MLPrediction,
    create_combined_prediction,
    ML_AVAILABLE,
)

from betting_optimizer import BettingOptimizer, BettingReportGenerator

from advanced_analytics import (
    DrawBiasAnalyzer,
    PaceAnalyzer,
    SectionalAnalyzer,
    BarrierTrialAnalyzer,
    CourseDistanceAnalyzer,
    ClassMovementAnalyzer,
    GearChangeAnalyzer,
    WeightAnalyzer,
    GoingPreferenceAnalyzer,
    RollingFormAnalyzer,
    BeatenMarginAnalyzer,
    JockeyChangeAnalyzer,
)
from odds_tracker import OddsMovementAnalyzer
```

To:

```python
from core.fetcher import HKJCDataFetcher, HorseEntry, RaceCard

from core.predictor import (
    HKJCRacingMLModel,
    FeatureEngineering,
    CombinedRacingAnalyzer,
    MLPrediction,
    create_combined_prediction,
    ML_AVAILABLE,
)

from betting.optimizer import BettingOptimizer, BettingReportGenerator

from analytics.draw_bias import DrawBiasAnalyzer
from analytics.pace import PaceAnalyzer
from analytics.sectionals import SectionalAnalyzer
from analytics.barrier_trials import BarrierTrialAnalyzer
from analytics.course_distance import CourseDistanceAnalyzer
from analytics.class_movement import ClassMovementAnalyzer
from analytics.gear_change import GearChangeAnalyzer
from analytics.weight import WeightAnalyzer
from analytics.going_pref import GoingPreferenceAnalyzer
from analytics.rolling_form import RollingFormAnalyzer
from analytics.beaten_margins import BeatenMarginAnalyzer
from analytics.jockey_change import JockeyChangeAnalyzer
from data.odds_tracker import OddsMovementAnalyzer
```

- [ ] **Step 3: Update `core/__init__.py` with re-exports**

```python
"""Core analysis pipeline: fetcher, analyzer, ML predictor."""
from core.fetcher import HKJCDataFetcher, HorseEntry, RaceCard
from core.analyzer import CombinedRaceAnalyzer, HorseAnalysis, Horse, Race, convert_race
from core.predictor import HKJCRacingMLModel, CombinedRacingAnalyzer as MLCombinedAnalyzer, ML_AVAILABLE
```

- [ ] **Step 4: Create root-level shim files for backwards compatibility**

`hkjc_data_fetcher.py` — replace contents with:
```python
"""Backwards-compatible shim — import from core.fetcher."""
from core.fetcher import *  # noqa: F401,F403
```

`hkjc_racing_analyzer.py` — replace contents with:
```python
"""Backwards-compatible shim — import from core.analyzer."""
from core.analyzer import *  # noqa: F401,F403
```

`ml_predictor.py` — replace contents with:
```python
"""Backwards-compatible shim — import from core.predictor."""
from core.predictor import *  # noqa: F401,F403
```

- [ ] **Step 5: Commit**

```bash
git add core/ hkjc_data_fetcher.py hkjc_racing_analyzer.py ml_predictor.py
git commit -m "Move core modules to core/ package with backwards-compatible shims"
```

---

### Task 3: Split advanced_analytics.py into individual analyzer files

**Files:**
- Create: `analytics/draw_bias.py`, `analytics/pace.py`, `analytics/sectionals.py`, `analytics/barrier_trials.py`, `analytics/course_distance.py`, `analytics/class_movement.py`, `analytics/gear_change.py`, `analytics/weight.py`, `analytics/going_pref.py`, `analytics/rolling_form.py`, `analytics/beaten_margins.py`, `analytics/jockey_change.py`
- Modify: `advanced_analytics.py` → shim

- [ ] **Step 1: Extract each analyzer class into its own file**

Each file gets:
1. The module docstring and necessary imports (`re`, `logging`, `typing`)
2. The single class
3. An import of `parse_form_positions` from `analytics` where needed

Example for `analytics/draw_bias.py`:
```python
"""Draw bias analyzer — rail position aware draw scoring."""
import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class DrawBiasAnalyzer:
    # ... entire DrawBiasAnalyzer class from advanced_analytics.py lines 38-199
```

Example for `analytics/pace.py`:
```python
"""Pace analyzer — running style classification + pace scenarios."""
import logging
from typing import Dict, List, Tuple

from analytics import parse_form_positions

logger = logging.getLogger(__name__)


class PaceAnalyzer:
    # ... entire PaceAnalyzer class from advanced_analytics.py lines 206-348
```

Repeat for all 12 analyzers. Each file contains exactly one class. Files that use `parse_form_positions` import it from `analytics`.

Classes that use `parse_form_positions`:
- `PaceAnalyzer` (line 248)
- `SectionalAnalyzer` (line 427)

All other classes are self-contained.

- [ ] **Step 2: Replace `advanced_analytics.py` with shim**

```python
"""Backwards-compatible shim — import from analytics/ package."""
from analytics import parse_form_positions  # noqa: F401
from analytics.draw_bias import DrawBiasAnalyzer  # noqa: F401
from analytics.pace import PaceAnalyzer  # noqa: F401
from analytics.sectionals import SectionalAnalyzer  # noqa: F401
from analytics.barrier_trials import BarrierTrialAnalyzer  # noqa: F401
from analytics.course_distance import CourseDistanceAnalyzer  # noqa: F401
from analytics.class_movement import ClassMovementAnalyzer  # noqa: F401
from analytics.gear_change import GearChangeAnalyzer  # noqa: F401
from analytics.weight import WeightAnalyzer  # noqa: F401
from analytics.going_pref import GoingPreferenceAnalyzer  # noqa: F401
from analytics.rolling_form import RollingFormAnalyzer  # noqa: F401
from analytics.beaten_margins import BeatenMarginAnalyzer  # noqa: F401
from analytics.jockey_change import JockeyChangeAnalyzer  # noqa: F401
```

- [ ] **Step 3: Commit**

```bash
git add analytics/ advanced_analytics.py
git commit -m "Split advanced_analytics.py into individual analyzer modules"
```

---

### Task 4: Move data layer modules

**Files:**
- Move: `horse_database.py` → `data/database.py`
- Move: `odds_tracker.py` → `data/odds_tracker.py`
- Move: `odds_scraper.py` → `data/odds_scraper.py`
- Move: `scrape_history.py` → `data/scraper.py`
- Create: root-level shims

- [ ] **Step 1: Copy files to new locations**

```bash
cp horse_database.py data/database.py
cp odds_tracker.py data/odds_tracker.py
cp odds_scraper.py data/odds_scraper.py
cp scrape_history.py data/scraper.py
```

- [ ] **Step 2: Update path references in moved files**

In `data/database.py`, the DB path uses `Path(__file__).parent / 'tracking_data'`. After move, `__file__` points to `data/database.py`, so the path becomes `data/tracking_data/` which is wrong. Fix:

```python
DB_DIR = Path(__file__).parent.parent / 'tracking_data'
```

Same fix needed in:
- `data/odds_tracker.py`: change `DB_DIR` to use `parent.parent`
- `data/scraper.py`: update any `Path(__file__).parent` references to `parent.parent`
- `core/predictor.py`: the `extract_features()` and `build_training_data_from_db()` functions reference `Path(__file__).parent / 'tracking_data'` — change to `parent.parent`

Pattern: anywhere `Path(__file__).parent / 'tracking_data'` or `Path(__file__).parent / 'models'` appears, add one more `.parent` since files moved one directory deeper.

- [ ] **Step 3: Update imports in `data/odds_tracker.py`**

The `poll_odds_snapshot` function references `fetcher.get_live_odds` — update any imports if present.

- [ ] **Step 4: Create root-level shims**

`horse_database.py`:
```python
"""Backwards-compatible shim — import from data.database."""
from data.database import *  # noqa: F401,F403
```

`odds_tracker.py`:
```python
"""Backwards-compatible shim — import from data.odds_tracker."""
from data.odds_tracker import *  # noqa: F401,F403
```

`odds_scraper.py`:
```python
"""Backwards-compatible shim — import from data.odds_scraper."""
from data.odds_scraper import *  # noqa: F401,F403
```

`scrape_history.py`:
```python
"""Backwards-compatible shim — import from data.scraper."""
from data.scraper import *  # noqa: F401,F403
```

- [ ] **Step 5: Commit**

```bash
git add data/ horse_database.py odds_tracker.py odds_scraper.py scrape_history.py core/predictor.py
git commit -m "Move data layer modules to data/ package"
```

---

### Task 5: Move dashboard and betting modules

**Files:**
- Move: `generate_predictions.py` → split into `dashboard/generator.py` + `dashboard/template.py`
- Move: `betting_optimizer.py` → `betting/optimizer.py`
- Move: `results_fetcher.py` → `dashboard/results_fetcher.py`
- Create: root-level shims

- [ ] **Step 1: Split generate_predictions.py**

`dashboard/generator.py` — gets `main()` function and prediction orchestration (lines 1-122):
```python
#!/usr/bin/env python3
"""Generate race predictions and dashboard - non-interactive."""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add project root to path for package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fetcher import HKJCDataFetcher
from core.analyzer import CombinedRaceAnalyzer
from dashboard.template import generate_html

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # ... lines 23-121 from generate_predictions.py, unchanged
    # except: replace generate_html import (already done above)
    pass


if __name__ == '__main__':
    asyncio.run(main())
```

`dashboard/template.py` — gets `generate_html()` function (lines 124-847):
```python
"""HTML dashboard template generation."""

from datetime import datetime


def generate_html(races_json, now, race_date, venue="Sha Tin", auto_refresh=False):
    # ... entire generate_html function from generate_predictions.py
    # Add auto_refresh parameter: if True, insert <meta http-equiv="refresh" content="60">
    # after the <meta name="viewport"> tag
    pass
```

Note: Add `auto_refresh` parameter to `generate_html()` signature now (defaulting to `False`). When `True`, insert:
```html
<meta http-equiv="refresh" content="60">
```
after the viewport meta tag. This is used by the race day monitor (Task 11).

- [ ] **Step 2: Move betting and results**

```bash
cp betting_optimizer.py betting/optimizer.py
cp results_fetcher.py dashboard/results_fetcher.py
```

Update `Path(__file__).parent` references in both files to use `parent.parent` for project root paths.

Update imports in `dashboard/results_fetcher.py`:
```python
from core.fetcher import HKJCDataFetcher
```

Update imports in `betting/optimizer.py` if it references other modules.

- [ ] **Step 3: Create root-level shims**

`generate_predictions.py`:
```python
"""Backwards-compatible shim — import from dashboard.generator."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dashboard.generator import main
import asyncio
if __name__ == '__main__':
    asyncio.run(main())
```

`betting_optimizer.py`:
```python
"""Backwards-compatible shim — import from betting.optimizer."""
from betting.optimizer import *  # noqa: F401,F403
```

`results_fetcher.py`:
```python
"""Backwards-compatible shim — import from dashboard.results_fetcher."""
from dashboard.results_fetcher import *  # noqa: F401,F403
```

- [ ] **Step 4: Commit**

```bash
git add dashboard/ betting/ generate_predictions.py betting_optimizer.py results_fetcher.py
git commit -m "Move dashboard and betting modules to packages"
```

---

### Task 6: Create CLI entry point scripts

**Files:**
- Create: `scripts/generate.py`, `scripts/scrape.py`, `scripts/train.py`, `scripts/monitor.py`, `scripts/fetch_results.py`

- [ ] **Step 1: Create `scripts/generate.py`**

```python
#!/usr/bin/env python3
"""CLI: Generate predictions and dashboard.

Usage: python3 -m scripts.generate '2026/04/16'
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.generator import main

if __name__ == '__main__':
    asyncio.run(main())
```

- [ ] **Step 2: Create `scripts/scrape.py`**

```python
#!/usr/bin/env python3
"""CLI: Scrape historical results.

Usage: python3 -m scripts.scrape
       python3 -m scripts.scrape --date 2026/04/09
       python3 -m scripts.scrape --rankings
       python3 -m scripts.scrape --days 90
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.scraper import main

if __name__ == '__main__':
    main()
```

Note: `scrape_history.py` uses `if __name__ == '__main__'` with sync code. Check the actual main block and wrap accordingly (may need `asyncio.run()` if async).

- [ ] **Step 3: Create `scripts/train.py`**

```python
#!/usr/bin/env python3
"""CLI: Retrain ML model.

Usage: python3 -m scripts.train
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_ml_model import main

if __name__ == '__main__':
    main()
```

Note: `train_ml_model.py` stays at root for now — it has complex imports. The script just forwards.

- [ ] **Step 4: Create `scripts/fetch_results.py`**

```python
#!/usr/bin/env python3
"""CLI: Fetch results and score predictions.

Usage: python3 -m scripts.fetch_results '2026/04/16'
       python3 -m scripts.fetch_results '2026/04/16' --auto-learn
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.results_fetcher import main

if __name__ == '__main__':
    asyncio.run(main())
```

- [ ] **Step 5: Create `scripts/monitor.py`** (placeholder for Phase 2)

```python
#!/usr/bin/env python3
"""CLI: Race day odds monitor with auto-refresh dashboard.

Usage: python3 -m scripts.monitor '2026/04/16'
"""
import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# monitor.race_day will be implemented in Phase 2
print("Race day monitor not yet implemented. Coming in Phase 2.")
sys.exit(1)
```

- [ ] **Step 6: Commit**

```bash
git add scripts/
git commit -m "Add CLI entry point scripts"
```

---

### Task 7: Verify refactor — everything works identically

**Files:**
- None changed — verification only

- [ ] **Step 1: Test prediction generation**

```bash
python3 generate_predictions.py '2026/04/15'
```

Expected: Produces `results/race_data_20260415.json` and `results/racing_dashboard_20260415.html` with same content as before refactor.

- [ ] **Step 2: Test via new CLI entry point**

```bash
python3 -m scripts.generate '2026/04/15'
```

Expected: Same output as step 1.

- [ ] **Step 3: Test ML training**

```bash
python3 train_ml_model.py
```

Expected: Model trains successfully with 4155+ samples.

- [ ] **Step 4: Test import chain**

```bash
python3 -c "from core.analyzer import CombinedRaceAnalyzer; print('OK')"
python3 -c "from analytics.draw_bias import DrawBiasAnalyzer; print('OK')"
python3 -c "from data.database import HorseDatabase; print('OK')"
python3 -c "from data.odds_tracker import OddsDatabase; print('OK')"
python3 -c "from betting.optimizer import BettingOptimizer; print('OK')"
```

Expected: All print "OK" without errors.

- [ ] **Step 5: Test backwards compatibility shims**

```bash
python3 -c "from hkjc_data_fetcher import HKJCDataFetcher; print('OK')"
python3 -c "from hkjc_racing_analyzer import CombinedRaceAnalyzer; print('OK')"
python3 -c "from advanced_analytics import DrawBiasAnalyzer; print('OK')"
python3 -c "from horse_database import HorseDatabase; print('OK')"
```

Expected: All print "OK" — old import paths still work via shims.

- [ ] **Step 6: Commit verification notes**

```bash
git commit --allow-empty -m "Verify: Phase 1 refactor complete — all imports and commands working"
```

---

## Phase 2: Enhanced Analytics Features

### Task 8: Add going-filtered form + jockey-horse stats + distance PBs to database

**Files:**
- Modify: `data/database.py`

- [ ] **Step 1: Add `get_horse_going_form()` method to `HorseDatabase`**

Add after the existing `get_horse_record()` method:

```python
def get_horse_going_form(self, horse_name: str, going: str, before_date: str = '') -> dict:
    """Get horse's record on a specific going condition.

    Returns: {'runs': int, 'wins': int, 'places': int, 'avg_pos': float}
    """
    params = [horse_name, going]
    date_filter = ''
    if before_date:
        date_filter = ' AND race_date < ?'
        params.append(before_date)

    row = self.conn.execute(f"""
        SELECT COUNT(*) as runs,
               COALESCE(SUM(is_winner), 0) as wins,
               COALESCE(SUM(is_top3), 0) as places,
               COALESCE(AVG(CASE WHEN finish_position > 0 THEN finish_position END), 0) as avg_pos
        FROM race_results
        WHERE horse_name = ? AND going = ?{date_filter}
    """, params).fetchone()

    return {
        'runs': row['runs'] or 0,
        'wins': row['wins'] or 0,
        'places': row['places'] or 0,
        'avg_pos': round(row['avg_pos'] or 0, 1),
    }
```

- [ ] **Step 2: Add `get_jockey_horse_stats()` method**

```python
def get_jockey_horse_stats(self, jockey: str, horse_name: str, before_date: str = '') -> dict:
    """Get jockey's record on a specific horse.

    Returns: {'rides': int, 'wins': int, 'places': int, 'avg_pos': float}
    """
    params = [jockey, horse_name]
    date_filter = ''
    if before_date:
        date_filter = ' AND race_date < ?'
        params.append(before_date)

    row = self.conn.execute(f"""
        SELECT COUNT(*) as rides,
               COALESCE(SUM(is_winner), 0) as wins,
               COALESCE(SUM(is_top3), 0) as places,
               COALESCE(AVG(CASE WHEN finish_position > 0 THEN finish_position END), 0) as avg_pos
        FROM race_results
        WHERE jockey = ? AND horse_name = ?{date_filter}
    """, params).fetchone()

    return {
        'rides': row['rides'] or 0,
        'wins': row['wins'] or 0,
        'places': row['places'] or 0,
        'avg_pos': round(row['avg_pos'] or 0, 1),
    }
```

- [ ] **Step 3: Add `get_distance_best_time()` and `get_distance_record()` methods**

```python
def get_distance_best_time(self, horse_name: str, distance: int, before_date: str = '') -> str:
    """Get horse's personal best finish time at a given distance.

    Returns: time string like '1:09.2' or '' if no data.
    """
    params = [horse_name, distance]
    date_filter = ''
    if before_date:
        date_filter = ' AND race_date < ?'
        params.append(before_date)

    row = self.conn.execute(f"""
        SELECT finish_time FROM race_results
        WHERE horse_name = ? AND distance = ?{date_filter}
        AND finish_time IS NOT NULL AND finish_time != ''
        ORDER BY finish_time ASC
        LIMIT 1
    """, params).fetchone()

    return row['finish_time'] if row else ''

def get_distance_record(self, venue: str, distance: int) -> str:
    """Get track record (fastest time) at venue + distance.

    Returns: time string like '1:08.1' or '' if no data.
    """
    row = self.conn.execute("""
        SELECT finish_time FROM race_results
        WHERE venue LIKE ? AND distance = ?
        AND finish_time IS NOT NULL AND finish_time != ''
        ORDER BY finish_time ASC
        LIMIT 1
    """, (f'%{venue}%', distance)).fetchone()

    return row['finish_time'] if row else ''
```

- [ ] **Step 4: Verify queries against real DB**

```bash
python3 -c "
from data.database import HorseDatabase
db = HorseDatabase()
print('Going form:', db.get_horse_going_form('GOLDEN SIXTY', 'GOOD'))
print('J-H stats:', db.get_jockey_horse_stats('Z Purton', 'GOLDEN SIXTY'))
print('PB 1200m:', db.get_distance_best_time('GOLDEN SIXTY', 1200))
print('Record ST 1200:', db.get_distance_record('Sha Tin', 1200))
"
```

Expected: Returns dicts with real data from the 4155 rows in DB. Some may return zeros if the specific horse isn't in our DB — that's fine, try with other horse names if needed.

- [ ] **Step 5: Commit**

```bash
git add data/database.py
git commit -m "Add going form, jockey-horse stats, and distance PB queries"
```

---

### Task 9: Add 4 new ML features (14 → 18)

**Files:**
- Modify: `core/predictor.py` — `FEATURE_NAMES`, `extract_features()`, `build_training_data_from_db()`, `generate_synthetic_training_data()`

- [ ] **Step 1: Update `FEATURE_NAMES` in `HKJCRacingMLModel`**

In `core/predictor.py`, find the `FEATURE_NAMES` list (line ~468) and add 4 new entries:

```python
FEATURE_NAMES = [
    'prior_win_rate', 'recent_form', 'draw_advantage',
    'jockey_win_rate', 'trainer_win_rate', 'field_size',
    'field_rel_draw', 'prior_pace', 'prior_speed',
    'cd_form_prior', 'odds_implied_prob',
    'class_level', 'freshness', 'going_pref',
    # New enhanced analytics features
    'going_form_rate', 'jockey_horse_coop', 'distance_pb_ratio', 'odds_movement',
]
```

- [ ] **Step 2: Add features 15-18 to `extract_features()`**

After the existing Feature 14 block (line ~460, before `return np.array(features ...)`), add:

```python
        # Feature 15: going form rate (win rate on current going condition)
        going_form_rate = 0.5
        if prior and going:
            prior_same_going = [p for p in prior if p.get('going', '') == going]
            if len(prior_same_going) >= 2:
                going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                going_form_rate = going_wins / len(prior_same_going)
        features.append(going_form_rate)

        # Feature 16: jockey-horse cooperation rate
        jockey_horse_coop = 0.5
        jockey = horse_data.get('jockey', '')
        if prior and jockey:
            prior_same_jockey = [p for p in prior if p.get('jockey', '') == jockey]
            if len(prior_same_jockey) >= 2:
                jh_wins = sum(1 for p in prior_same_jockey if p.get('is_winner', 0))
                jh_places = sum(1 for p in prior_same_jockey if p.get('is_top3', 0))
                jockey_horse_coop = min(1.0, (jh_wins * 2 + jh_places) / (len(prior_same_jockey) * 3))
        features.append(jockey_horse_coop)

        # Feature 17: distance personal best ratio (horse PB / track record)
        distance_pb_ratio = 0.5
        if prior:
            prior_at_dist = [p for p in prior
                             if p.get('distance') == distance
                             and p.get('finish_time', '')
                             and ':' in str(p.get('finish_time', ''))]
            if prior_at_dist:
                def _parse_time(t):
                    try:
                        parts = str(t).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except (ValueError, IndexError):
                        return 999.0

                times = [_parse_time(p['finish_time']) for p in prior_at_dist]
                horse_pb = min(t for t in times if t < 999)
                # Get track record from all prior races at this distance
                all_prior_dist = [p for p in prior
                                  if p.get('distance') == distance
                                  and p.get('finish_time', '')
                                  and ':' in str(p.get('finish_time', ''))]
                # Use DB for track record
                try:
                    db_path_pb = Path(__file__).parent.parent / 'tracking_data' / 'horse_history.db'
                    if db_path_pb.exists():
                        conn_pb = sqlite3.connect(str(db_path_pb))
                        row_pb = conn_pb.execute(
                            "SELECT MIN(finish_time) FROM race_results WHERE distance = ? AND finish_time != '' AND finish_time IS NOT NULL",
                            (distance,)
                        ).fetchone()
                        conn_pb.close()
                        if row_pb and row_pb[0]:
                            track_record = _parse_time(row_pb[0])
                            if track_record < 999 and horse_pb < 999:
                                distance_pb_ratio = min(1.0, track_record / horse_pb)
                except Exception:
                    pass
        features.append(distance_pb_ratio)

        # Feature 18: odds movement (opening → current, from odds_history.db)
        odds_movement = 0.5
        win_odds = horse_data.get('win_odds', 0)
        horse_number = horse_data.get('number', 0)
        race_date_for_odds = horse_data.get('race_date', '')
        race_number = horse_data.get('race_number', 0)
        if win_odds > 0 and horse_number > 0 and race_date_for_odds:
            try:
                odds_db_path = Path(__file__).parent.parent / 'tracking_data' / 'odds_history.db'
                if odds_db_path.exists():
                    conn_odds = sqlite3.connect(str(odds_db_path))
                    row_odds = conn_odds.execute(
                        """SELECT win_odds FROM odds_snapshots
                           WHERE race_date = ? AND race_number = ? AND horse_number = ?
                           ORDER BY timestamp ASC LIMIT 1""",
                        (race_date_for_odds, race_number, horse_number)
                    ).fetchone()
                    conn_odds.close()
                    if row_odds and row_odds[0] > 0:
                        opening = row_odds[0]
                        movement = (opening - win_odds) / opening  # positive = shortening
                        odds_movement = max(0.0, min(1.0, (movement + 1) / 2))  # scale [-1,1] to [0,1]
            except Exception:
                pass
        features.append(odds_movement)
```

- [ ] **Step 3: Add features 15-18 to `build_training_data_from_db()`**

After the existing Feature 14 block (line ~790, before `features = [`), add:

```python
            # Feature 15: going form rate
            if prior and going:
                prior_same_going_f = [p for p in prior if p.get('going', '') == going]
                if len(prior_same_going_f) >= 2:
                    gf_wins = sum(1 for p in prior_same_going_f if p.get('is_winner', 0))
                    feat_going_form = gf_wins / len(prior_same_going_f)
                else:
                    feat_going_form = 0.5
            else:
                feat_going_form = 0.5

            # Feature 16: jockey-horse cooperation
            jockey_name = r.get('jockey', '') or ''
            if prior and jockey_name:
                prior_same_jh = [p for p in prior if p.get('jockey', '') == jockey_name]
                if len(prior_same_jh) >= 2:
                    jh_w = sum(1 for p in prior_same_jh if p.get('is_winner', 0))
                    jh_p = sum(1 for p in prior_same_jh if p.get('is_top3', 0))
                    feat_jh_coop = min(1.0, (jh_w * 2 + jh_p) / (len(prior_same_jh) * 3))
                else:
                    feat_jh_coop = 0.5
            else:
                feat_jh_coop = 0.5

            # Feature 17: distance PB ratio
            feat_dist_pb = 0.5
            prior_dist_times = [p for p in prior
                                if p.get('distance') == distance
                                and p.get('finish_time', '')
                                and ':' in str(p.get('finish_time', ''))]
            if prior_dist_times:
                def _pt(t):
                    try:
                        parts = str(t).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except (ValueError, IndexError):
                        return 999.0
                horse_times = [_pt(p['finish_time']) for p in prior_dist_times]
                horse_pb_t = min(t for t in horse_times if t < 999)
                # Track record: best time at this distance across all data
                # (computed once outside the loop would be more efficient, but
                # keeping it simple and correct — DB query is fast)
                try:
                    tr_row = conn.execute(
                        "SELECT MIN(finish_time) FROM race_results WHERE distance = ? AND finish_time != '' AND finish_time IS NOT NULL AND race_date < ?",
                        (distance, race_date)
                    ).fetchone()
                    if tr_row and tr_row[0]:
                        tr_secs = _pt(tr_row[0])
                        if tr_secs < 999 and horse_pb_t < 999:
                            feat_dist_pb = min(1.0, tr_secs / horse_pb_t)
                except Exception:
                    pass

            # Feature 18: odds movement (not available in historical DB — use fallback)
            feat_odds_move = 0.5
```

Then update the features list:

```python
            features = [
                feat_form_proxy, feat_form, feat_draw,
                feat_jockey, feat_trainer, feat_field_size,
                feat_field_draw, feat_pace, feat_finish, feat_cd, feat_odds,
                feat_class, feat_freshness, feat_going,
                feat_going_form, feat_jh_coop, feat_dist_pb, feat_odds_move,
            ]
```

- [ ] **Step 4: Update `generate_synthetic_training_data()` to produce 18 features**

After the existing Feature 14 block (line ~867, after `going_pref = ...`), add:

```python
        # Feature 15: going form rate
        going_form_rate = np.random.uniform(0.0, 0.5)

        # Feature 16: jockey-horse cooperation
        jockey_horse_coop = np.random.uniform(0.0, 0.5)

        # Feature 17: distance PB ratio
        distance_pb_ratio = np.random.uniform(0.7, 1.0)

        # Feature 18: odds movement
        odds_movement_feat = np.random.uniform(0.3, 0.7)
```

Update the features list:
```python
        features = [
            rating / 100, form_score / 100, draw_adv,
            jockey_rate, trainer_rate, feat_field_size,
            feat_field_draw, prior_pace, prior_speed, cd_form, odds_implied,
            class_level, freshness, going_pref,
            going_form_rate, jockey_horse_coop, distance_pb_ratio, odds_movement_feat,
        ]
```

Update the win_logit formula to include new features:
```python
        win_logit = (
            1.8 * (rating / 100 - 0.6)
            + 1.2 * (form_score / 100 - 0.4)
            + 0.5 * (draw_adv - 0.6)
            + 3.0 * (jockey_rate - 0.10)
            + 2.0 * (trainer_rate - 0.08)
            + 0.3 * (feat_field_size - 0.8)
            + 0.4 * (feat_field_draw - 0.5)
            + 0.5 * (prior_pace - 0.5)
            + 0.3 * (prior_speed - 0.5)
            + 0.5 * (cd_form - 0.5)
            + 1.5 * (odds_implied - 0.1)
            + 0.3 * (class_level - 0.5)
            + 0.2 * (freshness - 0.5)
            + 0.3 * (going_pref - 0.5)
            + 0.4 * (going_form_rate - 0.25)
            + 0.3 * (jockey_horse_coop - 0.25)
            + 0.2 * (distance_pb_ratio - 0.85)
            + 0.1 * (odds_movement_feat - 0.5)
            - 2.3
        )
```

- [ ] **Step 5: Delete old model to force retrain**

```bash
rm -f models/hkjc_racing_model.pkl
```

- [ ] **Step 6: Retrain model**

```bash
python3 train_ml_model.py
```

Expected: Model trains with 18 features on 4155+ samples. Check that feature importance output lists all 18 features.

- [ ] **Step 7: Commit**

```bash
git add core/predictor.py models/
git commit -m "Extend ML model to 18 features: going form, jockey-horse, distance PB, odds movement"
```

---

### Task 10: Add steam/drift/cold horse detection to odds tracker

**Files:**
- Modify: `data/odds_tracker.py`

- [ ] **Step 1: Add `detect_steam_move()` to `OddsMovementAnalyzer`**

```python
    @staticmethod
    def detect_steam_move(
        opening_odds: float, current_odds: float, threshold: float = 0.20
    ) -> Tuple[bool, str, str]:
        """Detect if a horse has steamed (odds shortened significantly).

        Returns (is_steam, severity, label).
        severity: 'STEAM' (>=30%), 'STRONG_SUPPORT' (>=20%), or ''
        """
        if opening_odds <= 0 or current_odds <= 0:
            return False, '', ''

        pct_change = (opening_odds - current_odds) / opening_odds

        if pct_change >= 0.30:
            return True, 'STEAM', f"STEAM -{pct_change:.0%} (${opening_odds:.1f}→${current_odds:.1f})"
        elif pct_change >= threshold:
            return True, 'STRONG_SUPPORT', f"Strong support -{pct_change:.0%} (${opening_odds:.1f}→${current_odds:.1f})"
        return False, '', ''
```

- [ ] **Step 2: Add `detect_cold_horse()` to `OddsMovementAnalyzer`**

```python
    @staticmethod
    def detect_cold_horse(
        opening_odds: float,
        current_odds: float,
        rule_score: float,
        drift_threshold: float = 0.25,
        form_threshold: float = 65.0,
    ) -> Tuple[bool, str]:
        """Detect cold horse: drifting in market but strong fundamentals.

        Returns (is_cold, label).
        """
        if opening_odds <= 0 or current_odds <= 0:
            return False, ''

        pct_change = (opening_odds - current_odds) / opening_odds  # negative = drifting

        if pct_change <= -drift_threshold and rule_score >= form_threshold:
            drift_pct = abs(pct_change)
            return True, f"COLD HORSE: Drifted +{drift_pct:.0%} but rule score {rule_score:.0f}"
        return False, ''
```

- [ ] **Step 3: Add `get_movement_summary()` to `OddsDatabase`**

```python
    def get_movement_summary(
        self, race_date: str, race_number: int
    ) -> Dict[int, Dict]:
        """Get odds movement summary for all horses in a race.

        Returns: {horse_number: {'opening': float, 'current': float, 'change_pct': float}}
        """
        opening = self.get_opening_odds(race_date, race_number)
        current = self.get_latest_odds(race_date, race_number)

        summary = {}
        for num in set(list(opening.keys()) + list(current.keys())):
            o = opening.get(num, 0)
            c = current.get(num, 0)
            change = (o - c) / o * 100 if o > 0 else 0
            summary[num] = {
                'opening': o,
                'current': c,
                'change_pct': round(change, 1),
            }
        return summary
```

- [ ] **Step 4: Commit**

```bash
git add data/odds_tracker.py
git commit -m "Add steam move, cold horse detection, and movement summary to odds tracker"
```

---

### Task 11: Build speed map generator

**Files:**
- Modify: `analytics/pace.py`

- [ ] **Step 1: Add `generate_speed_map()` method to `PaceAnalyzer`**

```python
    @classmethod
    def generate_speed_map(
        cls, horses: List[Dict], distance: int, venue: str = "ST"
    ) -> List[Dict]:
        """Generate predicted running positions for all horses in a race.

        Each horse dict needs: 'number', 'name', 'draw', 'running_style' (or 'last_6').
        Returns list of dicts sorted by predicted early position:
        [{'number': 1, 'name': 'HORSE', 'style': 'Leader', 'predicted_pos': 1, 'lane': 'inside'}, ...]
        """
        entries = []
        for h in horses:
            style = h.get('running_style', '')
            if not style or style == 'Midfield':
                # Try to estimate from form
                style = cls.estimate_style_from_form(
                    h.get('last_6', ''),
                    h.get('draw', 0),
                    distance,
                )

            draw = h.get('draw', 0)
            field_size = len(horses)

            # Predict early position based on style
            style_base = {
                'Leader': 1.5,
                'Stalker': 4.0,
                'Midfield': 7.0,
                'Closer': 10.0,
            }.get(style, 7.0)

            # Draw influence: low draws get slightly better position early
            draw_adj = (draw - field_size / 2) * 0.15 if draw > 0 else 0

            predicted = max(1.0, min(float(field_size), style_base + draw_adj))

            # Lane: inside/middle/outside based on draw
            if draw <= field_size * 0.33:
                lane = 'inside'
            elif draw <= field_size * 0.66:
                lane = 'middle'
            else:
                lane = 'outside'

            entries.append({
                'number': h.get('number', 0),
                'name': h.get('name', ''),
                'style': style,
                'predicted_pos': round(predicted, 1),
                'lane': lane,
                'draw': draw,
            })

        entries.sort(key=lambda x: x['predicted_pos'])
        # Assign integer positions
        for i, e in enumerate(entries):
            e['predicted_pos'] = i + 1

        return entries
```

- [ ] **Step 2: Commit**

```bash
git add analytics/pace.py
git commit -m "Add speed map generator to pace analyzer"
```

---

### Task 12: Add enhanced data to prediction JSON output

**Files:**
- Modify: `dashboard/generator.py`

- [ ] **Step 1: Import new modules in generator**

Add imports at top:
```python
from data.database import HorseDatabase
from data.odds_tracker import OddsDatabase, OddsMovementAnalyzer
from analytics.pace import PaceAnalyzer
```

- [ ] **Step 2: Add enhanced data collection in the race loop**

After `analyses = analyzer.analyze_race(race)` in `main()`, add:

```python
        # Enhanced analytics data
        db = HorseDatabase()
        odds_db = OddsDatabase()
        race_going = race.going or ''
        race_distance = int(race.distance) if race.distance and race.distance.isdigit() else 1200

        # Speed map
        speed_map_input = [
            {'number': h.number, 'name': h.name, 'draw': h.draw,
             'running_style': h.running_style, 'last_6': h.last_6}
            for h in race.horses
        ]
        speed_map = PaceAnalyzer.generate_speed_map(speed_map_input, race_distance)

        # Odds movement
        odds_summary = odds_db.get_movement_summary(race_date, race.race_number)
```

- [ ] **Step 3: Add enhanced fields to each horse's JSON output**

Inside the `for i, a in enumerate(analyses):` loop, after the existing horse data dict, add:

```python
                # Enhanced analytics
                going_form = db.get_horse_going_form(h.name, race_going, race_date)
                jh_stats = db.get_jockey_horse_stats(h.jockey, h.name, race_date)
                dist_pb = db.get_distance_best_time(h.name, race_distance, race_date)
                dist_record = db.get_distance_record(race.venue, race_distance)

                # Odds movement for this horse
                horse_odds_move = odds_summary.get(h.number, {})
                is_steam, steam_severity, steam_label = OddsMovementAnalyzer.detect_steam_move(
                    horse_odds_move.get('opening', 0), h.win_odds
                )
                is_cold, cold_label = OddsMovementAnalyzer.detect_cold_horse(
                    horse_odds_move.get('opening', 0), h.win_odds, a.rule_score
                )
```

Add these to the horse dict in the JSON output:
```python
                "going_form": going_form,
                "jockey_horse_stats": jh_stats,
                "distance_pb": dist_pb,
                "distance_record": dist_record,
                "odds_movement": horse_odds_move,
                "steam_move": {"is_steam": is_steam, "severity": steam_severity, "label": steam_label},
                "cold_horse": {"is_cold": is_cold, "label": cold_label},
```

Add speed map to the race-level data:
```python
        race_data["speed_map"] = speed_map
```

- [ ] **Step 4: Close DB connection after loop**

```python
        db.conn.close()
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/generator.py
git commit -m "Add enhanced analytics data to prediction JSON output"
```

---

### Task 13: Add enhanced sections to HTML dashboard

**Files:**
- Modify: `dashboard/template.py`

- [ ] **Step 1: Add CSS for new dashboard elements**

Add to the existing CSS block in `generate_html()`:

```css
.market-badge{display:inline-flex;align-items:center;gap:3px;padding:2px 8px;border-radius:12px;font-size:.65rem;font-weight:700}
.market-badge.steam{background:rgba(16,185,129,.2);color:var(--green)}
.market-badge.drift{background:rgba(239,68,68,.2);color:var(--red)}
.market-badge.cold{background:rgba(251,191,36,.2);color:var(--gold)}
.enhanced-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:6px;margin-top:8px;padding:8px;background:var(--surface2);border-radius:8px;font-size:.72rem}
.enhanced-stats .es-item{display:flex;flex-direction:column;gap:2px}
.enhanced-stats .es-label{color:var(--dim);font-size:.62rem;text-transform:uppercase}
.enhanced-stats .es-value{font-weight:600}
.speed-map{background:var(--surface2);border-radius:10px;padding:14px;margin:12px 0}
.speed-map-title{font-size:.78rem;font-weight:700;margin-bottom:10px;color:var(--accent)}
.speed-map-track{position:relative;height:180px;background:linear-gradient(180deg,rgba(16,185,129,.05) 0%,rgba(16,185,129,.02) 100%);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.speed-map-zone{position:absolute;left:0;right:0;display:flex;align-items:center;padding:0 10px;font-size:.6rem;color:var(--dim)}
.speed-map-zone.leaders{top:0;height:25%;border-bottom:1px dashed var(--border)}
.speed-map-zone.stalkers{top:25%;height:25%;border-bottom:1px dashed var(--border)}
.speed-map-zone.midfield{top:50%;height:25%;border-bottom:1px dashed var(--border)}
.speed-map-zone.closers{top:75%;height:25%}
.speed-map-horse{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:var(--accent);color:#000;font-size:.68rem;font-weight:800;margin:2px}
.speed-map-horse.inside{border:2px solid var(--green)}
.speed-map-horse.outside{border:2px solid var(--red)}
```

- [ ] **Step 2: Add JavaScript to render enhanced stats in horse detail view**

In the JavaScript section where horse detail cards are rendered, add after the existing reasoning section:

```javascript
// Enhanced analytics section
let enhHtml = '<div class="enhanced-stats">';

// Going form
const gf = horse.going_form;
if (gf && gf.runs > 0) {
    enhHtml += '<div class="es-item"><span class="es-label">Going Form</span><span class="es-value">' +
        esc(raceData.going || '') + ': ' + gf.wins + 'W/' + gf.places + 'P/' + gf.runs + 'R</span></div>';
}

// Jockey-Horse stats
const jh = horse.jockey_horse_stats;
if (jh && jh.rides > 0) {
    enhHtml += '<div class="es-item"><span class="es-label">Jockey-Horse</span><span class="es-value">' +
        esc(horse.jockey) + ': ' + jh.wins + 'W ' + jh.places + 'P / ' + jh.rides + ' rides</span></div>';
}

// Distance PB
if (horse.distance_pb) {
    let pbText = 'PB ' + esc(horse.distance_pb);
    if (horse.distance_record) {
        pbText += ' (record ' + esc(horse.distance_record) + ')';
    }
    enhHtml += '<div class="es-item"><span class="es-label">Distance Best</span><span class="es-value">' + pbText + '</span></div>';
}

enhHtml += '</div>';

// Market move badges
let badgeHtml = '';
if (horse.steam_move && horse.steam_move.is_steam) {
    badgeHtml += '<span class="market-badge steam">\u25B2 ' + esc(horse.steam_move.label) + '</span>';
} else if (horse.odds_movement && horse.odds_movement.change_pct < -15) {
    badgeHtml += '<span class="market-badge drift">\u25BC DRIFT ' + Math.abs(horse.odds_movement.change_pct).toFixed(0) + '%</span>';
}
if (horse.cold_horse && horse.cold_horse.is_cold) {
    badgeHtml += '<span class="market-badge cold">\u2605 COLD HORSE</span>';
}
```

- [ ] **Step 3: Add speed map rendering per race**

```javascript
// Speed map for the race
function renderSpeedMap(speedMap) {
    if (!speedMap || speedMap.length === 0) return '';
    const zones = {leaders: [], stalkers: [], midfield: [], closers: []};
    speedMap.forEach(function(h) {
        const pos = h.predicted_pos;
        const zone = pos <= Math.ceil(speedMap.length * 0.25) ? 'leaders' :
                     pos <= Math.ceil(speedMap.length * 0.5) ? 'stalkers' :
                     pos <= Math.ceil(speedMap.length * 0.75) ? 'midfield' : 'closers';
        zones[zone].push(h);
    });

    let html = '<div class="speed-map"><div class="speed-map-title">Predicted Speed Map</div>';
    html += '<div class="speed-map-track">';
    ['leaders', 'stalkers', 'midfield', 'closers'].forEach(function(zone) {
        html += '<div class="speed-map-zone ' + zone + '"><span>' + zone.charAt(0).toUpperCase() + zone.slice(1) + '</span>';
        zones[zone].forEach(function(h) {
            html += '<span class="speed-map-horse ' + esc(h.lane) + '" title="' + esc(h.name) + ' (Draw ' + h.draw + ')">' + h.number + '</span>';
        });
        html += '</div>';
    });
    html += '</div></div>';
    return html;
}
```

- [ ] **Step 4: Add `auto_refresh` support**

At the top of `generate_html()`, after the viewport meta tag, add:

```python
    refresh_tag = '<meta http-equiv="refresh" content="60">' if auto_refresh else ''
```

Insert `{refresh_tag}` after the viewport meta tag in the HTML template.

- [ ] **Step 5: Commit**

```bash
git add dashboard/template.py
git commit -m "Add enhanced analytics sections and speed map to dashboard"
```

---

### Task 14: Build race day monitor

**Files:**
- Create: `monitor/race_day.py`
- Modify: `scripts/monitor.py`

- [ ] **Step 1: Create `monitor/race_day.py`**

```python
#!/usr/bin/env python3
"""Race day odds monitor.

Wakes up ~6 minutes before each race, fetches odds, detects market moves,
and regenerates the dashboard with auto-refresh enabled.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fetcher import HKJCDataFetcher
from data.odds_tracker import OddsDatabase
from dashboard.generator import main as generate_dashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MONITOR] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

POLL_INTERVAL = 300  # 5 minutes
PRE_RACE_WINDOW = 360  # 6 minutes before race


def parse_race_time(time_str: str, race_date: str) -> datetime:
    """Parse race time string like '13:00' into datetime."""
    try:
        date_part = race_date.replace('/', '-')
        return datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return None


async def fetch_race_times(fetcher: HKJCDataFetcher, race_date: str) -> list:
    """Get race times for all races on a given date.

    Returns list of (race_number, race_time_datetime) sorted by time.
    """
    race_times = []
    for race_no in range(1, 12):
        race = await fetcher.get_race_card(race_date, race_no)
        if not race or not race.horses:
            break
        race_time = parse_race_time(race.time, race_date)
        if race_time:
            race_times.append((race_no, race_time))
            logger.info(f"Race {race_no}: {race.time} ({len(race.horses)} runners)")
        else:
            logger.warning(f"Race {race_no}: no time available")
    return race_times


async def snapshot_odds(fetcher: HKJCDataFetcher, race_date: str, race_no: int):
    """Fetch and store current odds for a race."""
    odds_db = OddsDatabase()
    try:
        race = await fetcher.get_race_card(race_date, race_no)
        if race and race.horses:
            horses = [
                {
                    'number': h.number,
                    'name': h.name,
                    'win_odds': h.win_odds,
                    'place_odds': h.place_odds,
                }
                for h in race.horses
                if h.number > 0 and h.win_odds > 0
            ]
            if horses:
                odds_db.record_snapshot(race_date, race_no, horses, source='monitor')
                logger.info(f"R{race_no}: Snapshotted odds for {len(horses)} horses")
                return len(horses)
    except Exception as e:
        logger.warning(f"R{race_no}: Failed to snapshot odds: {e}")
    return 0


async def regenerate_dashboard(race_date: str):
    """Regenerate dashboard with auto-refresh enabled."""
    try:
        # Import and call dashboard generation with auto_refresh=True
        from core.fetcher import HKJCDataFetcher
        from core.analyzer import CombinedRaceAnalyzer
        from dashboard.template import generate_html
        import json

        fetcher = HKJCDataFetcher()
        analyzer = CombinedRaceAnalyzer()

        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)

        all_races = []
        for race_no in range(1, 12):
            race = await fetcher.get_race_card(race_date, race_no)
            if not race or not race.horses:
                break
            analyses = analyzer.analyze_race(race)

            # Build race data (same as dashboard/generator.py main())
            race_data = {
                "race_number": race.race_number,
                "venue": race.venue,
                "time": race.time,
                "distance": race.distance,
                "class_type": race.class_type,
                "going": race.going,
                "course": race.course,
                "horses": [],
            }

            from data.database import HorseDatabase
            from data.odds_tracker import OddsDatabase as ODB, OddsMovementAnalyzer
            from analytics.pace import PaceAnalyzer

            db = HorseDatabase()
            odds_db = ODB()
            race_going = race.going or ''
            race_distance = int(race.distance) if race.distance and race.distance.isdigit() else 1200
            odds_summary = odds_db.get_movement_summary(race_date, race.race_number)

            speed_map_input = [
                {'number': h.number, 'name': h.name, 'draw': h.draw,
                 'running_style': h.running_style, 'last_6': h.last_6}
                for h in race.horses
            ]
            speed_map = PaceAnalyzer.generate_speed_map(speed_map_input, race_distance)
            race_data["speed_map"] = speed_map

            for i, a in enumerate(analyses):
                h = a.horse
                ml_prob = a.ml_prediction.win_probability if a.ml_prediction else 0
                calibrated_prob = a.combined_probability if a.combined_probability else 0

                going_form = db.get_horse_going_form(h.name, race_going, race_date)
                jh_stats = db.get_jockey_horse_stats(h.jockey, h.name, race_date)
                dist_pb = db.get_distance_best_time(h.name, race_distance, race_date)
                dist_record = db.get_distance_record(race.venue, race_distance)

                horse_odds_move = odds_summary.get(h.number, {})
                is_steam, steam_severity, steam_label = OddsMovementAnalyzer.detect_steam_move(
                    horse_odds_move.get('opening', 0), h.win_odds
                )
                is_cold, cold_label = OddsMovementAnalyzer.detect_cold_horse(
                    horse_odds_move.get('opening', 0), h.win_odds, a.rule_score
                )

                race_data["horses"].append({
                    "rank": i + 1,
                    "number": h.number,
                    "name": h.name,
                    "jockey": h.jockey,
                    "trainer": h.trainer,
                    "draw": h.draw,
                    "weight": h.weight,
                    "gear": h.gear,
                    "rating": h.rating,
                    "win_odds": h.win_odds,
                    "place_odds": h.place_odds,
                    "last_6": h.last_6,
                    "age": getattr(h, 'age', ''),
                    "horse_weight": h.horse_weight,
                    "days_since_last_run": h.days_since_last_run,
                    "rating_change": h.rating_change,
                    "rule_score": a.rule_score,
                    "ml_prob": ml_prob,
                    "combined_score": a.value_score,
                    "calibrated_prob": calibrated_prob,
                    "expected_return": a.expected_return,
                    "bet_recommendation": a.bet_recommendation,
                    "confidence": a.combined_confidence,
                    "reasoning": a.reasoning,
                    "going_form": going_form,
                    "jockey_horse_stats": jh_stats,
                    "distance_pb": dist_pb,
                    "distance_record": dist_record,
                    "odds_movement": horse_odds_move,
                    "steam_move": {"is_steam": is_steam, "severity": steam_severity, "label": steam_label},
                    "cold_horse": {"is_cold": is_cold, "label": cold_label},
                })

            db.conn.close()
            all_races.append(race_data)

        date_str = race_date.replace('/', '')
        json_data = json.dumps(all_races)
        json_data = json_data.replace('</script>', r'<\/script>').replace('</Script>', r'<\/Script>')
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        venue = "Happy Valley" if any(r.get("venue", "").startswith("Happy") for r in all_races) else "Sha Tin"

        html = generate_html(json_data, now, race_date, venue, auto_refresh=True)

        html_file = results_dir / f"racing_dashboard_{date_str}.html"
        latest_file = results_dir / "racing_dashboard.html"
        html_file.write_text(html)
        latest_file.write_text(html)

        logger.info(f"Dashboard regenerated with {len(all_races)} races")

    except Exception as e:
        logger.error(f"Dashboard regeneration failed: {e}")


async def run_monitor(race_date: str):
    """Main monitor loop."""
    logger.info(f"Starting race day monitor for {race_date}")
    logger.info(f"Pre-race window: {PRE_RACE_WINDOW}s, Poll interval: {POLL_INTERVAL}s")

    fetcher = HKJCDataFetcher()

    # Step 1: Get race times
    race_times = await fetch_race_times(fetcher, race_date)
    if not race_times:
        logger.error("No race times found. Waiting 30 minutes then retrying...")
        await asyncio.sleep(1800)
        race_times = await fetch_race_times(fetcher, race_date)
        if not race_times:
            logger.error("Still no race times. Exiting.")
            return

    logger.info(f"Found {len(race_times)} races")

    # Step 2: Process each race
    for race_no, race_time in race_times:
        wake_time = race_time - timedelta(seconds=PRE_RACE_WINDOW)
        now = datetime.now()

        if now < wake_time:
            sleep_secs = (wake_time - now).total_seconds()
            logger.info(f"Sleeping {sleep_secs:.0f}s until R{race_no} window ({wake_time.strftime('%H:%M')})")
            await asyncio.sleep(sleep_secs)
        elif now > race_time:
            logger.info(f"R{race_no} already started, skipping")
            continue

        # Snapshot and regenerate
        logger.info(f"R{race_no}: Pre-race window active")
        await snapshot_odds(fetcher, race_date, race_no)
        await regenerate_dashboard(race_date)

    logger.info("All races processed. Monitor complete.")


def main():
    race_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y/%m/%d')
    asyncio.run(run_monitor(race_date))


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Update `scripts/monitor.py`**

Replace placeholder content with:

```python
#!/usr/bin/env python3
"""CLI: Race day odds monitor with auto-refresh dashboard.

Usage: python3 -m scripts.monitor '2026/04/16'
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.race_day import main

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Commit**

```bash
git add monitor/ scripts/monitor.py
git commit -m "Add race day monitor with pre-race odds polling and auto-refresh dashboard"
```

---

### Task 15: Update CLAUDE.md with new commands and architecture

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add new commands section**

Add to the Commands section:

```markdown
# Race day monitor (auto-polls odds 6min before each race, regenerates dashboard)
python3 -m scripts.monitor '2026/04/16'

# New CLI entry points (equivalent to root scripts)
python3 -m scripts.generate '2026/04/16'
python3 -m scripts.scrape --date 2026/04/09
python3 -m scripts.train
python3 -m scripts.fetch_results '2026/04/16'
```

- [ ] **Step 2: Update Architecture section**

Add package structure info and document the enhanced analytics:

```markdown
### Package Structure

```
core/       — fetcher, analyzer, ML predictor
analytics/  — 12 specialized analyzers (draw bias, pace, sectionals, etc.)
data/       — database, odds tracker, scraper
monitor/    — race day monitor
dashboard/  — prediction generator, HTML template
betting/    — Kelly Criterion optimizer
scripts/    — CLI entry points
```

Root-level shims exist for backwards compatibility (e.g., `hkjc_data_fetcher.py` re-exports from `core/fetcher.py`).
```

Add ML features update:

```markdown
### ML Model Features (18 total)

Features 1-14: unchanged (prior_win_rate through going_pref).
Features 15-18 (new):
- `going_form_rate`: win% on current going condition (DB query)
- `jockey_horse_coop`: win% of this jockey on this horse (DB query)
- `distance_pb_ratio`: horse's PB / track record at distance (DB query)
- `odds_movement`: opening→current odds change (odds_history.db, builds up over time)
```

- [ ] **Step 3: Document race day monitor**

```markdown
### Race Day Monitor

`python3 -m scripts.monitor '2026/04/16'` — polls odds 6 minutes before each race (every 5 min), snapshots to odds_history.db, detects steam/drift/cold horses, regenerates dashboard with auto-refresh (60s). Dashboard at `results/racing_dashboard.html` reloads automatically in browser.
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "Update CLAUDE.md with package structure, new features, and monitor docs"
```

---

### Task 16: End-to-end verification

**Files:**
- None changed — verification only

- [ ] **Step 1: Run full prediction pipeline**

```bash
python3 -m scripts.generate '2026/04/15'
```

Expected: Dashboard generated with enhanced analytics sections (going form, jockey-horse stats, distance PB, speed map). JSON includes new fields.

- [ ] **Step 2: Verify JSON has enhanced fields**

```bash
python3 -c "
import json
with open('results/race_data_20260415.json') as f:
    data = json.load(f)
race = data[0]
horse = race['horses'][0]
print('speed_map:', 'speed_map' in race)
print('going_form:', 'going_form' in horse)
print('jockey_horse_stats:', 'jockey_horse_stats' in horse)
print('distance_pb:', 'distance_pb' in horse)
print('steam_move:', 'steam_move' in horse)
print('cold_horse:', 'cold_horse' in horse)
print('odds_movement:', 'odds_movement' in horse)
"
```

Expected: All print `True`.

- [ ] **Step 3: Verify ML model has 18 features**

```bash
python3 -c "
from core.predictor import HKJCRacingMLModel
m = HKJCRacingMLModel()
print('Features:', len(m.FEATURE_NAMES))
print('Names:', m.FEATURE_NAMES)
print('Trained:', m.is_trained)
print('Importance keys:', len(m.feature_importance))
"
```

Expected: 18 features, all named, trained=True, 18 importance keys.

- [ ] **Step 4: Verify monitor can start (dry run)**

```bash
timeout 10 python3 -m scripts.monitor '2026/04/15' 2>&1 || true
```

Expected: Starts, logs "Starting race day monitor", then either finds race times or waits. Exits cleanly on timeout.

- [ ] **Step 5: Final commit**

```bash
git commit --allow-empty -m "Verify: Phase 2 complete — enhanced analytics, 18 ML features, race day monitor"
```
