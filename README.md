# HKJC Horse Racing Analyzer v6.5

Real-time horse racing analysis for the Hong Kong Jockey Club (HKJC). Combines a 24-factor rule-based engine with an XGBoost ML model (19 features, isotonic-calibrated, post-blend recalibrated) to surface bets the public may be underestimating. The dashboard renders every race as a clickable card stack with two views — **Race Order** (all 11 chronological) and **Top Bets** (curated tier-sorted) — plus a sticky live alerts rail.

## What's New in v6.5

Two compounding improvements on top of v6.4 + 4-expert review fixes.

### Three "data-already-here" features (16 → 19)

Sourced from the Renavon backfill, previously used only by the rule engine:

- **`horse_age`** (3.2% importance) — DB at 99.8% post-Renavon. Younger HK horses (3-5y) systematically improve.
- **`rating_change`** (4.6% importance — beats `prior_win_rate`!) — DB at 62.6%. Captures handicapper's view of recent form change.
- **`jockey_form_30d`** (4.5% importance) — Bayesian-smoothed 30-day jockey strike rate. Captures hot streaks the all-time aggregate (Feature 4) hides.

Schema is now 19 features, all with non-zero importance.

### Post-blend isotonic calibrator

The XGBoost-side isotonic calibrator only sees raw model probs; it doesn't see the post-rule-blend or post-market-blend output. A second isotonic was fit on **71,360 (calibrated_prob, is_winner) pairs** from the held-out 20% via the production pipeline (`scripts/calibrate_blend.py`). Stored as `models/blend_calibrator.pkl` with `.sha256` sidecar. Lazy-loaded by `core/scorer.py:calibrate_probabilities` and applied per-horse with renormalization. On v6.5 the bin shifts are small (0.30 → 0.346 vs v6.4's 0.30 → 0.459) because the new features already pull raw output closer to actual win rates.

### Dashboard restructure

- **Two prominent View tabs** above the card stack: **📋 Race Order** (default — all 11 races as full cards in chronological order) and **⭐ Top Bets** (curated HIGH/MED/EDGE subset, tier-sorted then by edge_pp).
- **Click anywhere on a race card** to expand its detail panel inline (per-horse table, same-format history, speed map, reasoning). Click again to collapse. Top-right chevron flips ▼ → ▲.
- **Same-format history per horse** — for every horse with prior runs at this exact venue + distance, the expanded panel lists their last 5 attempts (date, going, class, finish position colour-coded, odds, margin, jockey).
- **BET TICKET** block at the top of every expanded race — synthesizes per-horse bet recommendations into one race-level instruction (e.g. "STRAIGHT QIN · #5 THE HEIR + #11 PRESTIGE HALL").
- **Plain-English labels** with click-to-toggle ⓘ popovers on Pure / Market / Edge ("Pure" → "Our model").
- **⟳ Hard reload** button in the topbar that bypasses browser cache.
- **Atomic dashboard writes** (write-temp-then-`os.replace`) so a polling browser never sees a half-written file.

### v6.5 Performance (5,966-race holdout, 554 with real HKJC dividends)

| Metric | v6.4 OFF | v6.4 + cal | **v6.5 + cal** |
|---|---|---|---|
| Recommender quinella ROI | +113.05% | +117.97% | **+168.71%** |
| EV ≥ 5% (WIN) ROI | +65.61% | +1.50% | +67.67% |
| EV ≥ 15% (WIN) ROI | +59.20% | -4.32% | +75.20% |
| Bet market favourite (baseline) | -16.67% | — | -16.67% |
| Top-pick Brier | 0.2165 | 0.2045 | 0.2145 |

Recommender ROI jumped **+50pp** (v6.4 → v6.5 + calibrator). Brier on the new 5,966-race holdout is honest about residual under-prediction; old "0.108" was measured on a smaller specific sample. Strategy A (flat WIN on top pick) remains slightly negative across all model versions — market takeout dominates flat staking.

### Previous Versions
- **v6.4**: 16-feature lean schema, exponential recency weighting, isotonic-fit-on-validation, `extract_class_rank()` helper. Dropped `prior_pace`, `adjusted_speed`, `distance_pb_ratio`, `speed_consistency` (all 0.0 importance after audit fixes)
- **v6.3**: 5-year recency cutoff (50K samples), Renavon bulk import (368K legacy rows ingested for backfill)
- **v6.2**: Full 49-year training (360K rows) — regressed by ~30% ROI vs v6.1 / v6.3
- **v6.1**: Wave 2c, 7,540 samples, isotonic calibrator
- **v6.0**: Dashboard redesign — edge-first layout, EDGE gate, conviction tiers, trust card, alerts rail
- **v5.3**: Speed figure weight correction, train/predict consistency, place model wired
- **v5.2**: Beyer speed figures, exotic calculator, dynamic blend, 24-feature ML
- **v5.1**: 22-feature ML, Platt scaling, place model, correlation audit
- **v5.0**: 26-factor engine, real DB stats, 18-feature ML, historical scraper
- **v4.0**: QIN default strategy, BANKER WIN system, market probability blending
- **v3.0**: 14-factor analytics, rail bias, pace map, sectionals, C&D form
- **v2.0**: Tips Index integration, ML prediction, advanced analytics
- **v1.0**: Basic rule-based scoring

## How It Works

### Data Sources (from racing.hkjc.com)

| Source | What It Provides |
|--------|-----------------|
| Race Card page | Horse entries, jockeys, trainers, draw, weight, gear, ratings, last 6 runs |
| Tips Index page | Live odds, Initial Tips Index, Race Day Tips Index, top picks per race |
| Horse Profile page | Running positions, sectional times, class history, C&D record |
| Results page | Finishing positions, margins, odds, real dividend payouts (WIN/PLACE/QIN/QPL/FORECAST/TRIO/TIERCE/FIRST 4/QUARTET) |
| Horse Database (local) | 50,692 training rows (5y window, ~4,200 winners), 6,005+ dividend rows across 63 recent meetings, jockey/trainer season stats sourced from Renavon rankings |

### Prediction Pipeline

```
rule_score (0-100)  → sigmoid → ×0.60
ml_prob (isotonic-calibrated)  → ×0.40
combined_prob = rule + ml (uncalibrated, pre-blend)
→ softmax (temp=1.5, mild normalization)
→ pure_prob (our independent opinion)
→ dynamic blend: model × (30-60%) + market × (40-70%)
→ normalize to sum = 1.0
→ calibrated_prob, expected_return = odds × prob - 1

Edge math (additive, post-calibration):
edge_pp = (pure - market) × 100
ev_win  = pure × win_odds - 1
overlay = pure × win_odds
6 gates → edge_fires bool → conviction_tier
```

The 60/40 rule+ML blend forms the model's "independent opinion" (`pure_prob`); the market is then mixed in. Edge math compares `pure_prob` vs `market_prob` BEFORE the market blend so the gap is meaningful.

### Bet Recommendation Logic (`core/recommender.py`)

- **★ EDGE → WIN bet** when all 6 gates pass (rare — ~1-2 races per card on average)
- **HIGH → BOX Q4** when `top_2_share ≥ 0.33` OR `top_3_share ≥ 0.42` AND rank-3/4 have strong history
- **MED → BOX Q3** fallback when `top_2_share ≥ 0.27` OR `top_3_share ≥ 0.35`
- **straight QIN** when `top_2_share ≥ 0.25` and tier is at least MED
- **TRIO** when top 3 are tightly bunched (gap_1_2 < 0.02 AND gap_2_3 < 0.02)
- **PLACE** when top pick has strong place model probability
- **SKIP** when no live odds, field < 2, or no conviction gate met

### Conviction Tiers (replaces strategy_confidence)

| Tier | Condition | Treatment |
|------|-----------|-----------|
| **★ EDGE** | All 6 EDGE gates pass | Single-horse WIN, hero placement |
| **HIGH** | `top_2_share ≥ 0.33` OR `top_3_share ≥ 0.42` | QIN/BOX-3/BOX-4 hero |
| **MED** | `top_2_share ≥ 0.27` OR `top_3_share ≥ 0.35` | Hero, smaller stake |
| **LOW** | `top_2_share ≥ 0.22` | Strip only, not hero |
| **SKIP** | else | Strip greyed, no recommendation |

EDGE takes precedence over pair-conviction tiers.

## Project Structure

```
hkjc_racing_analyzer/
  core/                      # Core modules
    fetcher.py               #   Real-time data layer (HKJC API + parser)
    analyzer.py              #   Main analyzer — 24-factor engine + ML blend
    scorer.py                #   Score combination + probability calibration
    recommender.py           #   Bet recommendation engine
    predictor.py             #   ML predictor shim
  ml/                        # ML pipeline
    feature_schema.py        #   16-feature schema (v6.4)
    features.py              #   Feature extraction + extract_class_rank helper
    model.py                 #   XGBoost — isotonic-calibrated, integrity-checked
    training.py              #   Training data builders (5y default, exp recency)
    integrity.py             #   F-05 model integrity check (≥1000 samples, <90 days)
    experiments.py           #   JSON-based experiment tracking (production-runs only)
  betting/                   # Betting + edge math
    edge_calculator.py       #   pure_prob, market_prob, edge_pp, 6-gate filter
    conviction.py            #   Tier assignment (EDGE/HIGH/MED/LOW/SKIP)
    alerts.py                #   Snapshot-diff event generation with dedup
    tier_stats.py            #   Per-tier hit-rate + Wilson CI cache
    dividends.py             #   Quinella EV pricing (Harville + place boost)
    exotic_calculator.py     #   Trifecta/First4/Quinella value detection
    optimizer.py             #   Kelly Criterion sizing
  analytics/                 # 12 specialized analyzers (draw bias, pace, sectionals,
                             #   barrier trial, C&D, class, gear, weight, going,
                             #   rolling form, beaten margins, jockey change, speed)
  data/                      # Data layer
    database.py              #   SQLite schema + queries
    odds_tracker.py          #   Timestamped odds_snapshots
    odds_scraper.py          #   Live + history odds fetching
    scraper.py               #   Historical results scraper (Playwright)
    validation.py            #   Schema validation
  dashboard/                 # Dashboard
    generator.py             #   Prediction JSON + HTML dashboard generator
    template.py              #   Edge-first HTML template (hero/strip/alerts/trust)
    results_fetcher.py       #   Fetch results + score predictions + auto-learn
  monitor/                   # Race-day auto-polling monitor
    race_day.py              #   Polls odds, regenerates dashboard, auto-reruns on sharp moves
  scripts/                   # CLI entry points
    generate.py · scrape.py · train.py · monitor.py · backtest.py
    backfill_dividends.py · accuracy.py · serve.py · fetch_results.py
  tests/                     # pytest suite — 170 tests
  config.py                  # Centralized constants
  models/                    # Trained ML models (pickle + .sha256 sidecar)
  results/                   # HTML dashboards and prediction JSON
  tracking_data/             # SQLite (horse_history.db, odds_history.db) + tier_stats.json
  docs/superpowers/          # Specs and implementation plans
```

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### Build Historical Database (recommended first run)

```bash
# Scrape past 60 days of meetings (default)
python3 -m scripts.scrape

# Or a specific date
python3 -m scripts.scrape --date 2026/04/22

# Import full-season jockey/trainer rankings
python3 -m scripts.scrape --rankings

# Backfill real HKJC dividends (idempotent, skips already-present)
python3 -m scripts.backfill_dividends
```

### Generate Dashboard

```bash
python3 generate_predictions.py '2026/04/22'
# Opens results/racing_dashboard.html with edge-first layout
```

### Race-Day Monitor (auto-refresh odds + auto-rerun on sharp moves)

```bash
python3 -m scripts.monitor '2026/04/22'
# Polls odds 6 min before each race, regenerates dashboard,
# triggers full prediction re-run on sharp STEAM/COLD over top-2
```

### Fetch Results & Auto-Learn

```bash
python3 results_fetcher.py '2026/04/22' --auto-learn
# Scrapes results, scores predictions, ingests into DB,
# refreshes tier_stats.json cache, retrains ML model
```

### Backtest

```bash
python3 -m scripts.backtest              # held-out last 20% of DB
python3 -m scripts.backtest --sweep      # grid over softmax temp × model weight
```

### Run Tests

```bash
pytest tests/ -v
# 170 tests covering edge math, gates, tier assignment, alerts dedup,
# Wilson CI, template rendering, JS-syntax validation, end-to-end integration
```

## Performance (last 500 races with real HKJC dividends)

41 races in this window have full real-dividend metadata; the rest backtest with the Harville-estimated quinella payouts.

| Strategy | Bets | Hit Rate | ROI |
|----------|------|----------|-----|
| Recommender-gated (v6.4) | 38 | 39.5% | **+1433%** |
| Straight quinella | 41 | — | +1813% |
| Box-3 quinella | 41 | — | +1918% |
| Box-4 quinella | 41 | — | +1283% |

### Calibration

- **Brier score (top-pick): 0.108** (was 0.176 in v6.3; UK ML benchmark ~0.18)
- **All 16 features have 4–16% importance** — no dead weight after schema cuts

### Strategy Decision

All quinella strategies have overlapping ROI confidence intervals at this sample size. **No strategy is statistically dominant.** The recommender's conviction-gated mode (BOX Q4 / BOX Q3 / QIN / SKIP per race) is the operational default — it auto-downgrades when conviction tiers aren't met.

### Audit Trail

Bug fixes shipped between v6.3 and v6.4:
1. Field clipping: training extraction now clips `field_size` and `field_rel_draw` to [0,1] (matched inference)
2. Class regex: G1/G2/G3/GRIFFIN/4YO no longer mishandled by digit-only extraction
3. Freshness fallback: dslr computed from prior race date when DB column is null
4. Exponential recency: replaced linear 0.3→1.0 weighting
5. experiment_log: tests with `save=False` no longer pollute the production training ledger

Issues NOT addressed (worth tracking):
- Calibrator validation slice spans 2024-04 → 2025-04 (1y stale on field strength)
- Distance imbalance (1200m has 19,555 rows vs 14 at 1050m)
- `Corr(edge_pp, actual win)` is mildly negative on disagreements (n still small)

## Requirements

- Python 3.10+
- Internet connection (fetches live data from racing.hkjc.com)
- Playwright + Chromium (for historical scraping and JS-rendered pages)
- `node` (optional — only used by `tests/test_template_js_syntax.py` to validate rendered JS; tests skip gracefully if absent)

## Key Dependencies

- `xgboost` — ML prediction engine
- `scikit-learn` — Model training, isotonic calibration
- `statsmodels` — Wilson 95% CIs for tier track record
- `numpy` — Numerical computations
- `playwright` — Browser automation for JS-rendered pages
- `beautifulsoup4` — HTML parsing
- `requests` — HTTP requests

## Disclaimer

For entertainment and educational purposes only. Not financial advice. Past performance does not guarantee future results. Gamble responsibly.
