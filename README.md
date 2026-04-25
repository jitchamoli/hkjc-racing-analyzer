# HKJC Horse Racing Analyzer v6.4

Real-time horse racing analysis for the Hong Kong Jockey Club (HKJC). Combines a 24-factor rule-based engine with an XGBoost ML model (16 features, isotonic-calibrated) to surface bets the public may be underestimating. The dashboard uses an edge-first layout: a hero block of 0–5 high-conviction picks with a trust card showing per-tier historical hit-rates, a live alerts rail for sharp odds moves, and an honest "no bets today" empty state when the model and market agree too closely.

## What's New in v6.4

Data-scientist audit on v6.3 surfaced multiple data-quality bugs and architectural issues. v6.4 fixes them and ships a leaner model.

- **16-feature schema** (was 20): dropped `prior_pace`, `adjusted_speed`, `distance_pb_ratio`, `speed_consistency` — all had 0.0 importance because `last_start_margin` (16%) carried the form/speed signal
- **Field clipping fix**: training-side feature extraction was missing the [0,1] clip on `field_size` and `field_rel_draw` (inference clipped via F-16). Model trained on values up to 15.0 from oversize legacy fields
- **Class-type extractor**: new `extract_class_rank()` correctly handles HKJC's full class ladder. Old regex `r'(\d)'` read 'G1' as class 1; Group races sit ABOVE class 1. Now Group 1/2 → 0, Group 3 → 1, Class 1-5 → 1-5, GRIFFIN/MAIDEN → 4, 4YO → 3
- **Freshness fallback**: 87.7% of legacy rows had `days_since_last_run` = 0/null, making freshness near-constant 0.5 (0.0 importance in v6.3). Now falls back to `(race_date - prior race_date)` math like inference does
- **Exponential recency weighting**: was linear 0.3→1.0. Switched to `0.25 + 0.75·exp((i−n+1)/(n/3))` so newest third weighs ~2× the oldest third (was 1.5× under linear). With 50% jockey turnover and 94% horse turnover since 2021, older rows describe a different field
- **5-year training cutoff** (introduced v6.3): empirically training on 49y of data dropped ROI by ~30% vs the recent 5y window — different jockey rosters, rule sets, and pool dynamics in older eras

### v6.4 Performance (recent 500-race holdout, real HKJC dividends)

| Metric | v6.3 | v6.4 |
|---|---|---|
| Recommender-gated ROI | +1359% | **+1433%** |
| Hit rate | 38.3% | **39.5%** |
| Brier score (top-pick) | 0.176 | **0.108** |
| Dead features | 5 | **0** |

### Previous Versions
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
