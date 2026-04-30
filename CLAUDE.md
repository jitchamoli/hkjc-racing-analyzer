# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For deep history, version-by-version changelogs, full feature lists, and audit detail, see:
- `docs/ml-model.md` — feature inventory, removed-features changelog, training pipeline detail
- `docs/dashboard.md` — card-stack layout history, BET TICKET prose, escape rules
- `docs/betting-strategy.md` — backtest history, edge gates, recommender logic, exotic pricing
- `docs/renavon.md` — table inventory, backfill impact, rebuild instructions

## Commands

```bash
# Generate predictions + dashboard for a race date (fetches live data from HKJC)
python3 generate_predictions.py '2026/04/15'
# Output: results/race_data_YYYYMMDD.json + results/racing_dashboard.html (always-latest) + results/racing_dashboard_YYYYMMDD.html (archive)

# Scrape historical results into database (Playwright required)
python3 scrape_history.py                    # Last 10 meeting dates
python3 scrape_history.py --date 2026/04/09  # Specific date
python3 scrape_history.py --rankings         # Full-season jockey/trainer stats from HKJC
python3 scrape_history.py --days 90          # Look back 90 days (default: 60)

# Fetch results, score predictions, regenerate dashboard with results overlay
python3 results_fetcher.py '2026/04/15'
python3 results_fetcher.py '2026/04/15' --auto-learn  # Also ingest into DB + retrain ML

# Retrain ML on latest data
python3 train_ml_model.py

# Race day monitor (auto-polls odds 6min before each race)
python3 -m scripts.monitor '2026/04/16'

# CLI entry points (equivalent to root scripts)
python3 -m scripts.generate '2026/04/16'
python3 -m scripts.scrape --date 2026/04/09
python3 -m scripts.train
python3 -m scripts.fetch_results '2026/04/16'

# Out-of-sample backtest (ROI / CLV / calibration / quinella EV)
python3 -m scripts.backtest              # Held-out last 20% of DB (default)
python3 -m scripts.backtest --sweep      # Grid over softmax temp × model weight
python3 -m scripts.backtest --exclude-date 2026/04/19

# Backfill real HKJC dividends into race_dividends table
python3 -m scripts.backfill_dividends              # All missing dates
python3 -m scripts.backfill_dividends --last 20    # Last 20 meetings only
python3 -m scripts.backfill_dividends --force      # Re-scrape even if present

# Local dashboard server with in-browser refresh button
python3 -m scripts.serve                  # Port 8080 (refresh button only in served mode)
python3 -m scripts.serve --open           # Also open browser

# Rank-2 calibration audit
python3 -m scripts.analyze_rank2

# Accuracy report (Brier, BSS, calibration curve)
python3 -m scripts.accuracy --all
```

## Architecture

### Package Structure

```
core/       — fetcher, analyzer, scorer, recommender, ML predictor
ml/         — feature schema (19 features), feature extraction, model, training, integrity, experiments
analytics/  — 13 specialized analyzers (draw bias, pace, sectionals, etc.)
data/       — database, odds tracker, scraper, validation
monitor/    — race day monitor (auto-rerun on sharp moves)
dashboard/  — prediction generator, edge-first HTML template
betting/    — edge_calculator, conviction tiers, alerts, tier_stats, dividends, exotic_calculator, optimizer
scripts/    — CLI entry points
tests/      — pytest suite (170 tests)
```

Root-level shims exist for backwards compatibility (e.g., `hkjc_data_fetcher.py` re-exports from `core/fetcher.py`).

### Data Pipeline

```
core/fetcher.py  →  core/analyzer.py  →  dashboard/generator.py  →  results/
(fetch from HKJC)   (rule + ML)          (JSON + HTML dashboard)
        ↑                   ↑
        |                   |
data/database.py ←── dashboard/results_fetcher.py (post-race: ingest results, update stats)
(SQLite: tracking_data/horse_history.db)
        ↓
scripts/train.py → models/hkjc_racing_model.pkl
```

### Scoring System

**CombinedRaceAnalyzer** (`hkjc_racing_analyzer.py`) merges two systems:
- **Rule engine (60%)**: 24 dynamically-weighted factors. Missing-data factors are excluded and remaining weights renormalize to 1.0. The rule_score (0-100) is converted to a probability via a base-rate-anchored sigmoid: `sigmoid(0.06 × (score - 50) + logit(1/field_size))`. `odds` and `odds_movement` are NOT in the rule engine — market info enters via the dynamic market blend only (prevents triple-counting).
- **ML model (40%)**: XGBoost classifier with 19 features (`ml/model.py`, schema in `ml/feature_schema.py`). Includes a separate place model (`is_top3`). Both use IsotonicRegression recency-weighted, trained on 50,692 real race results from a 5-year window. After the market blend, a second isotonic (`models/blend_calibrator.pkl`, fit by `scripts/calibrate_blend.py` on 71,360 production-pipeline pairs) recalibrates the post-blend distribution per-horse with renormalization.

The 13 specialized analyzers live in `analytics/` (draw bias, pace, sectionals, barrier trials, C&D, class movement, gear, weight, going preference, rolling form, beaten margins, jockey change, speed figures).

→ Full feature list, importance table, removed-features history, training pipeline detail: `docs/ml-model.md`.

### Database

SQLite at `tracking_data/horse_history.db`. Five tables:
- **race_results** (primary data source): one row per horse per race, keyed by (race_date, race_number, horse_name). Contains finish_position, is_winner, is_top3, margin, odds, horse_weight, running_position, finish_time, days_since_last_run.
- **race_dividends** (pool payouts): one row per (race_date, race_number, pool, combination). Pools: WIN, PLACE, QUINELLA, QUINELLA PLACE, FORECAST, TIERCE, TRIO, FIRST 4, QUARTET. Backfilled via `python3 -m scripts.backfill_dividends`. Used by `scripts/backtest.py` for real-dividend ROI (replaces Harville estimate when data is available).
- **horses**: master record per horse. Note: `get_horse_record()` computes aggregates from race_results directly (not from this table's stale counters).
- **jockey_stats** / **trainer_stats**: season-level win/place rates. Populated two ways: `rebuild_stats()` computes from local race_results; `scrape_history.py --rankings` fetches full-season stats from HKJC (more accurate).

**Data integrity**: `ingest_race_day()` calls `rebuild_stats()` after every ingestion, which overwrites jockey/trainer stats with local-only data. If full-season HKJC rankings were previously imported, re-run `python3 scrape_history.py --rankings` after ingestion to restore them. The `horses` table upsert derives `total_starts`/`total_wins`/`total_places` from `race_results` via subquery — safe for re-ingestion (idempotent, no double-counting).

**Auto-computed fields**: `days_since_last_run` is backfilled from race_results on ingestion (uses `julianday(REPLACE(date,'/','-'))` since dates are YYYY/MM/DD). `rating_change` and `age` come from race cards (not results pages) so are only populated for races where predictions were generated.

### Data Fetching (`hkjc_data_fetcher.py`)

Three methods tried in order:
1. **requests + BeautifulSoup** (fast, reliable for race cards)
2. **Playwright** (slower, gets live odds but often times out)
3. Tips data fetched separately for expert ratings

Place odds are fetched via `_apply_odds_via_requests()` using `fetch_odds_html(date, race_no, 'PLA')` — but PLA odds are only available near race time (returns None pre-race).

**Venue detection is critical**: `HKJCApiClient.detect_venue(race_date)` auto-detects ST (Sha Tin) vs HV (Happy Valley). Using the wrong venue returns empty HTML with no tables. Wednesday meetings are typically Happy Valley; weekends are Sha Tin.

### Dashboard

`generate_predictions.py` generates a self-contained HTML dashboard with embedded race data as JSON. Venue is detected dynamically. `results/racing_dashboard.html` is always overwritten with the latest run (date-stamped archive also saved).

**Card-stack layout (v6.5)** — topbar with **📋 Race Order / ⭐ Top Bets** view tabs → hero stack of full bet cards → sticky alerts rail on ≥1024px. Cards are click-to-expand with per-race detail panels (per-horse table, history, speed map, reasoning).

**6-layer BET TICKET** per card:
1. **PRIMARY** (gold) — recommender's choice with jockey + 30-day strike rate
2. **+ BOX QIN-4** (green) — top-4 model picks, always-on protection (catches rank-3/4 placing-pair upsets)
3. **+ F4 SAVER** (purple) — top-4 in 1-2-3-4 exact order, single unit
4. **+ QIN PLACE BANKER** (blue) — only when top pick odds ≤ 1.8 (favorite + 3 bankers, 3 combos)
5. **+ BOX TRIO-5** (pink) — top-5 in 10-combo TRIO box (catches rank-5/6 spoilers)
6. **+ BOX TRIO-6** (rose, OPTIONAL) — top-6 in 20-combo TRIO box. 6-meeting evidence (Apr 12-29, 60 races): TRIO-6 lifts strike rate +10pp at both venues (HV +2/18, ST +4/42) but doubles stake. Economic break-even vs TRIO-5: avg TRIO dividend must exceed ~$111 to overcome 2× cost. Take on chaos/longshot-podium races; skip on heavy-favorite/short-field races. Marked OPTIONAL in the bet ticket.

**Top Bets sub-toggle**: `Next` / `Past` filter — `Next` is default, auto-flips to `Past` at end-of-meeting. Past races render outcome badges: 🟡 STRIKE·WIN, 🟢 STRIKE·QIN, 🟢 STRIKE·BOX-4, 🔴 STRIKE·TRIO-5, 🌹 STRIKE·TRIO-6, 🔴 MISS.

**Race-day live workflow** uses two persistent monitors:
- **Pre-race refresh + results merge** at **T-6 min** (chains `generate_predictions.py` → `results_fetcher.py` so dashboard has fresh win/place odds AND past STRIKE/MISS badges persist).
- **Results poll + auto-learn** every 15 min via `python3 results_fetcher.py 'YYYY/MM/DD' --auto-learn`. Note: the auto-learn scraper drops the LAST race of a card (end-of-card detection bug); recover via direct ingest from `race_data_*_with_results.json` if needed.

**Critical inline-handler note**: All click-handler functions (`showRaceDetail`, `closeRaceDetail`, `toggleCardDetail`, `setStripView`, `hardReload`, `refreshDashboard`, `copyWA`, `showInfoPop`, `_closeInfoPop`, `toggleExpandAll`) are explicitly assigned `window.X = X` at the bottom of the IIFE. Without this, inline `onclick=""` attributes throw "X is not defined". Add the same `window.X` line whenever introducing a new function called from inline HTML.

**Atomic writes**: `dashboard/results_fetcher.py:_regenerate_dashboard` writes BOTH `racing_dashboard_{date}.html` AND `racing_dashboard.html` via tmp + `os.replace` so a polling browser never sees a half-written file.

**XSS escapes**: All scraped data is HTML-escaped via `esc()` before `innerHTML` injection. JSON embedding sanitizes `</script>` tags. Attribute-context values use `escAttr` (NOT `esc()` — text-context doesn't escape `"`/`'`). WA Copy uses `data-payload` + `addEventListener` (not inline `onclick`).

→ Full layout history, BET TICKET rationale, escape rules, results-merging mechanism: `docs/dashboard.md`.

## Codebase Conventions

- Code is organized into packages (`core/`, `analytics/`, `data/`, `monitor/`, `dashboard/`, `betting/`, `scripts/`). Root-level shims provide backwards compatibility.
- Async/await used in data fetching pipeline (`HKJCDataFetcher.get_race_card` is async).
- Race dates use `YYYY/MM/DD` format in HKJC URLs and data, but `YYYYMMDD` in filenames. SQLite `julianday()` requires `YYYY-MM-DD` — always use `REPLACE(date,'/','-')`.
- The HTML parser (`HKJCParser`) expects specific HKJC table layouts (27 cells per row with documented column indices). HKJC site changes can break parsing — check `_parse_horse_table()` if horses come back empty.
- ML model saved as pickle dict with keys: `model`, `place_model`, `calibrator`, `version`, `n_features`, `feature_importance`, `trained_date`. A `.sha256` sidecar file is required for loading — model refuses to load without valid checksum.
- Dashboard probabilities are stored as decimals (0.178) in JSON but displayed as percentages (17.8%) — multiply by 100 in JavaScript display code.
- `HKJCParser._safe_int()` preserves minus signs (regex `[^\d\-]`) — important for `rating_change` and `weight_change` which can be negative.

## ML Critical Rules

These are timeless guidance — violations have caused multiple v6.x bugs.

1. **No data leakage**: Every feature must use ONLY data available BEFORE the race. Never use finish_position, finish_time, running_position, or margin from the CURRENT race. For horse history features (C&D form, pace, speed), use strictly `prior_date < current_date`. `extract_features()` accepts `race_date` via `horse_data` and filters DB queries with `AND race_date < ?`. Track records in training are also date-aware (cumulative best-time lookup, not global).

2. **Train/predict consistency**: Features in `build_training_data_from_db()` MUST match `extract_features()` exactly — same computation, same order, same fallback values. `generate_synthetic_training_data()` must also match the same feature schema. The DB (race_results) has different fields than race cards. Both training and inference must use the SAME helpers (e.g. `extract_class_rank` for class strings, dslr fallback for freshness) — divergence between them caused multiple v6.x bugs.

3. **No constant features**: If a feature has std < 0.01 in training data, it's wasted. Common cause: field is mostly null and falls back to 0.5. Check with `np.std()` after building features. Feature correlation matrix is logged after each training run — pairs with |r| > 0.7 are flagged.

4. **value_score must use ML blend**: `value_score` in HorseAnalysis feeds `combined_score` in the JSON. It must be `combined_prob * 100` (the 60/40 rule+ML blend), NOT raw `rule_score`.

5. **Probability sync after rank swaps**: When `_assign_bet_recommendations` swaps horse positions (market-favourite promotion or BANKER override), `combined_probability` and `expected_return` on ALL `HorseAnalysis` objects must be re-written from the swapped `calibrated_probs` list.

6. **ML fallback probability**: `predict()` accepts `field_size` param (default 12). When the model is untrained, returns `1/field_size` — not `1/len(features)` which is always 23.

7. **Feature count guard**: Model pickle stores `n_features`. On load, it's checked against the current `NUM_FEATURES` — mismatched models refuse to load with a clear error message.

8. **Model integrity**: `.sha256` sidecar is required for loading. Missing checksum = hard refusal. `train_ml_model.py` backs up the old model before training and restores on failure.

## Probability Pipeline

```
rule_score (0-100) → sigmoid(0.06×(score-50)+logit(1/field_size)) → ×0.60
ml_prob (isotonic-calibrated) → ×0.40
pure_prob = softmax(rule × 0.6 + ml × 0.4) — independent opinion, pre-market

→ dynamic blend: model × (30-60%) + market × (40-70%)
  (chosen 2026-04-23: base 45/55, sprint threshold 1200m, field threshold 10)
→ post-blend isotonic calibrator (v6.5, models/blend_calibrator.pkl, optional —
  loaded lazily by core/scorer.py if file present)
→ calibrated_prob (post-blend, normalized to sum=1.0)
→ expected_return = odds × calibrated_prob - 1

Edge math (computed from pure_prob NOT calibrated_prob):
edge_pp = (pure_prob − market_prob) × 100
ev_win  = pure_prob × win_odds − 1
overlay = pure_prob × win_odds
→ 6 EDGE gates → edge_fires bool → conviction_tier

Exotic pricing: conditional elimination for exacta/trifecta/quinella/first4.
```

Market odds enter ONLY through the dynamic market blend. Blend weights are heavier on market for large sprint fields (sharp pool), heavier on model for small/long fields and exotics (thinner pool). **Edge math uses `pure_prob` (pre-blend) NOT `calibrated_prob` (post-blend)** — otherwise edge would be mechanically tiny because we'd be comparing market-influenced-us against market.

→ Edge gates, conviction tiers, recommender logic, backtest history, exotic pricing math: `docs/betting-strategy.md`.

## DB Data Gaps (post-Renavon backfill)

- `rating`: 98.5% coverage (was 5%). `last_6`: 94.8%. `age`: 99.8%. `rating_change`: 62.6%. `gear`: 52.1%.
- `running_position` has two formats: space-separated (`"6 5 6"`) and concatenated (`"665"`). Always handle both when parsing.
- `horse_weight` is in lbs (900-1350 range), `weight` is carried weight assignment (113-133 lbs).
- `class_type`: 96% coverage; uses `extract_class_rank()` to handle G1/G2/G3/GRIFFIN/4YO. Missing/unparseable rows default to class 3 for class_change calculation.
- `days_since_last_run`: 87.7% missing in legacy 5y window — both training and inference fall back to `(race_date - prior race_date)` math.

## Renavon — frozen snapshot

`renavon_*` tables are a one-time historical backfill ending 2026/04/22. They will never grow. For new race data, dividends, and rankings, use the HKJC scraper. Do NOT design features that assume continuous renavon_* updates.

→ Table inventory, indexes, backfill impact, rebuild commands, untapped feature candidates: `docs/renavon.md`.
