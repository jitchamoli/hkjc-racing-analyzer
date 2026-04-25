# HKJC Horse Racing Analyzer v6.0

Real-time horse racing analysis for the Hong Kong Jockey Club (HKJC). Combines a 24-factor rule-based engine with an XGBoost ML model (20 features, isotonic-calibrated) to surface bets the public may be underestimating. The dashboard uses an edge-first layout: a hero block of 0–5 high-conviction picks with a trust card showing per-tier historical hit-rates, a live alerts rail for sharp odds moves, and an honest "no bets today" empty state when the model and market agree too closely.

## What's New in v6.0

- **Dashboard redesign — edge-first layout**: hero (0–5 bets) + 11-tile race strip + sticky alerts rail + race-detail with per-horse tables
- **★ EDGE gate**: 6-gate filter (EV ≥ +5%, prob ≥ 10%, edge gap ≥ 5pp, model rank ≤ 2, odds age < 30min, model integrity) flags only races where model-vs-market disagreement is statistically backed
- **Conviction tiers**: EDGE / HIGH / MED / LOW / SKIP replaces the old A+/A/B+/B/B- grade system
- **Trust Card**: per-tier hit-rate with Wilson 95% CI from `prediction_log` × `race_results`, computed offline and cached
- **Live alerts rail**: STEAM (-25%), COLD (+40% on top-pick), EDGE_OPENED/CLOSED/WIDENED, RACE_UPCOMING — newest first, deduplicated per day, persisted across re-runs
- **Monitor auto-rerun**: sharp move on a top-2 pick triggers a full prediction re-run (not just dashboard regenerate)
- **20-feature ML model**: dropped `field_rel_speed` and `days_since_last_win` (correlation cleanup), retrained on 7,540 real race results
- **Isotonic calibrator**: replaces Platt scaling; mid-range [18-25%] calibration gap reduced from -18.6pp to -3.2pp after data backfill
- **Real HKJC dividend integration**: 6,005 dividend rows backfilled across 63 meetings; backtest uses real payouts where available, Harville fallback only when missing

### Previous Versions
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
| Horse Database (local) | 7,540+ race results, 6,005 dividend rows across 63 meetings (Sept 2025 – Apr 2026), jockey/trainer season stats |

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
    feature_schema.py        #   20-feature schema (Wave 2c)
    features.py              #   Feature extraction for prediction
    model.py                 #   XGBoost — isotonic-calibrated, integrity-checked
    training.py              #   Training data builders (DB + synthetic)
    integrity.py             #   F-05 model integrity check (≥1000 samples, <90 days)
    experiments.py           #   JSON-based experiment tracking
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

## Performance (124-race holdout, real HKJC dividends)

| Strategy | Hit Rate | Wilson 95% CI | ROI | Bootstrap 95% CI |
|----------|----------|---------------|-----|------------------|
| Recommender-gated | 40.9% | — | **+1843%** | — |
| Box-4 quinella | 54.2% | [42.7%, 65.2%] | +1768% | [+905%, +1977%] |
| Box-3 quinella | 41.7% | [31.0%, 53.2%] | +1713% | [+1161%, +2871%] |

### Calibration

- **Brier score (top-pick): 0.1945** (lower is better)
- **BSS vs base-rate: +0.0246** (positive = better than predicting the prior)
- **Mid-range [18-25%] gap: -3.2pp** (was -18.6pp before data backfill — structurally fixed)

### Strategy Decision

All three strategies (straight-Q, box-3, box-4) have overlapping ROI confidence intervals at the current 124-race sample size. **No strategy is statistically dominant.** The recommender's conviction-gated mode (BOX Q4 / BOX Q3 / QIN / SKIP per race) is the operational default.

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
