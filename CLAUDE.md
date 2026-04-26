# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Generate predictions + dashboard for a race date (fetches live data from HKJC)
python3 generate_predictions.py '2026/04/15'

# Output: results/race_data_YYYYMMDD.json + results/racing_dashboard.html (always-latest) + results/racing_dashboard_YYYYMMDD.html (archive)

# Scrape historical results into database (Playwright required)
python3 scrape_history.py                    # Last 10 meeting dates
python3 scrape_history.py --date 2026/04/09  # Specific date
python3 scrape_history.py --rankings         # Scrape full-season jockey/trainer stats from HKJC
python3 scrape_history.py --days 90          # Look back 90 days (default: 60)

# Fetch results after races, score predictions, regenerate dashboard with results overlay
python3 results_fetcher.py '2026/04/15'
python3 results_fetcher.py '2026/04/15' --auto-learn  # Also ingest into DB + retrain ML

# Retrain ML model on latest data from horse_database
python3 train_ml_model.py

# Ingest training_data/historical_races.json into horse_database
python3 -c "from horse_database import ingest_from_training_data; ingest_from_training_data()"

# Race day monitor (auto-polls odds 6min before each race, regenerates dashboard)
python3 -m scripts.monitor '2026/04/16'

# New CLI entry points (equivalent to root scripts)
python3 -m scripts.generate '2026/04/16'
python3 -m scripts.scrape --date 2026/04/09
python3 -m scripts.train
python3 -m scripts.fetch_results '2026/04/16'

# Out-of-sample backtest (ROI / CLV / calibration / quinella EV)
python3 -m scripts.backtest              # Held-out last 20% of DB (default)
python3 -m scripts.backtest --sweep      # Grid over softmax temp × model weight
python3 -m scripts.backtest --exclude-date 2026/04/19  # Skip a specific date

# Backfill real HKJC quinella/trio/first4 dividends into race_dividends table
python3 -m scripts.backfill_dividends              # All missing dates
python3 -m scripts.backfill_dividends --last 20    # Last 20 meetings only
python3 -m scripts.backfill_dividends --force      # Re-scrape even if present

# Local dashboard server with in-browser refresh button
python3 -m scripts.serve                  # Port 8080
python3 -m scripts.serve --open           # Also open browser
python3 -m scripts.serve --port 9000      # Custom port
# Refresh button appears only when served this way; static-file mode hides it.
# POST /api/refresh triggers generate_predictions.py in background; GET /api/status polls.

# Rank-2 calibration audit (why the model misses 2nd-place finishers)
python3 -m scripts.analyze_rank2
```

## Architecture

### Package Structure

```
core/       — fetcher, analyzer, scorer, recommender, ML predictor
ml/         — feature schema (20 features), feature extraction, model, training, integrity, experiments
analytics/  — 12 specialized analyzers (draw bias, pace, sectionals, etc.)
data/       — database, odds tracker, scraper, validation
monitor/    — race day monitor (auto-rerun on sharp moves)
dashboard/  — prediction generator, edge-first HTML template
betting/    — edge_calculator (pure_prob + 6 gates), conviction tiers, alerts, tier_stats, dividends, exotic_calculator, optimizer
scripts/    — CLI entry points
tests/      — pytest suite (170 tests)
```

Root-level shims exist for backwards compatibility (e.g., `hkjc_data_fetcher.py` re-exports from `core/fetcher.py`).

### Data Pipeline

```
core/fetcher.py  →  core/analyzer.py  →  dashboard/generator.py  →  results/
(fetch from HKJC)   (26-factor + ML)     (JSON + HTML dashboard)
        ↑                   ↑
        |                   |
data/database.py ←── dashboard/results_fetcher.py (post-race: ingest results, update stats)
(SQLite: tracking_data/horse_history.db)
        ↓
scripts/train.py → models/hkjc_racing_model.pkl
```

### Scoring System

**CombinedRaceAnalyzer** (`hkjc_racing_analyzer.py`) merges two systems:
- **Rule engine (60%)**: 24 factors with dynamic weighting — factors with missing data are excluded and remaining weights renormalize to 1.0. The rule_score (0-100) is converted to a probability via a **base-rate-anchored sigmoid**: `sigmoid(0.06 × (score - 50) + logit(1/field_size))`, so an "average" horse (score=50) outputs ~1/field_size instead of 50%. Note: `odds` and `odds_movement` were removed from the rule engine to prevent triple-counting market signal (market info enters via the dynamic market blend in calibration). Approximate grouping: Core (rating, form, jockey, trainer, draw, tips), Expert (going, rolling form, margins, field size, career stats), Technical (pace, sectionals, C&D, class movement, gear, weight, freshness). Actual percentages vary per race due to renormalization.
- **ML model (40%)**: XGBoost classifier with **19 features** (`ml/model.py`, schema in `ml/feature_schema.py`). v6.5 added 3 features sourced from the Renavon backfill (`horse_age`, `rating_change`, `jockey_form_30d` — all non-zero importance). v6.4 had dropped 4 features with 0.0 importance after audit fixes (`prior_pace`, `adjusted_speed`, `distance_pb_ratio`, `speed_consistency`). Wave 2c had previously dropped `field_rel_speed` and `days_since_last_win` for r > 0.7 collinearity. Includes a dedicated **place model** (is_top3). Both models use **IsotonicRegression** (Wave 2c switch from Platt) recency-weighted. Trained on **50,692 real race results** from a 5-year window. The 60/40 rule+ML blend (pre-market) forms `pure_prob`. After the market blend a **second isotonic** (`models/blend_calibrator.pkl`, fit by `scripts/calibrate_blend.py` on 71,360 production-pipeline pairs) recalibrates the post-blend distribution per-horse with renormalization. v6.5 + calibrator backtest (5,966 races): recommender quinella ROI **+168.71%**, Brier 0.2145.

The 13 specialized analyzers live in `analytics/` (draw bias, pace, sectionals, barrier trials, C&D, class movement, gear, weight, going preference, rolling form, beaten margins, jockey change, **speed figures**).

### Supplementary Scripts

- `odds_tracker.py` / `odds_scraper.py` — SQLite odds history, movement tracking, multi-method odds fetching.
- `betting_optimizer.py` — Kelly Criterion value bet optimization.
- `interactive_analyzer.py` — Interactive mode with manual odds input.
- `racing_scheduler.py` — Automated daily race analysis scheduler.
- `results_tracker.py` / `update_results.py` — Track predictions vs actuals, manual result entry.
- `mark6_analyzer.py` — HKJC Mark Six lottery analyzer (separate from racing).

### Key Data Structures

- **HorseEntry / RaceCard** (`hkjc_data_fetcher.py`): Raw data from HKJC. HorseEntry has ~30 fields including tips index, running style, sectionals, weight changes, import category.
- **HorseAnalysis** (`hkjc_racing_analyzer.py`): Analysis output per horse — rule_score (0-100), combined_probability, bet_recommendation (BANKER/WIN/QIN/PLACE/SKIP), reasoning list.

### Database

SQLite at `tracking_data/horse_history.db`. Five tables:
- **race_results** (primary data source): one row per horse per race, keyed by (race_date, race_number, horse_name). Contains finish_position, is_winner, is_top3, margin, odds, horse_weight, running_position, finish_time, days_since_last_run.
- **race_dividends** (pool payouts, Apr 22 addition): one row per (race_date, race_number, pool, combination). Pools: WIN, PLACE, QUINELLA, QUINELLA PLACE, FORECAST, TIERCE, TRIO, FIRST 4, QUARTET. Backfilled via `python3 -m scripts.backfill_dividends` from HKJC results pages. Used by `scripts/backtest.py` for real-dividend ROI measurement (replaces Harville estimate when data is available).
- **horses**: master record per horse. Note: `get_horse_record()` computes aggregates from race_results directly (not from this table's stale counters).
- **jockey_stats** / **trainer_stats**: season-level win/place rates. Populated two ways: `rebuild_stats()` computes from local race_results; `scrape_history.py --rankings` fetches full-season stats from HKJC (more accurate, ~80 meets vs our ~10).

**Data integrity**: `ingest_race_day()` calls `rebuild_stats()` after every ingestion, which overwrites jockey/trainer stats with local-only data. If full-season HKJC rankings were previously imported, re-run `python3 scrape_history.py --rankings` after ingestion to restore them. The `horses` table upsert derives `total_starts`/`total_wins`/`total_places` from `race_results` via subquery — safe for re-ingestion (idempotent, no double-counting).

**Auto-computed fields**: `days_since_last_run` is backfilled from race_results on ingestion (uses `julianday(REPLACE(date,'/','-'))` since dates are YYYY/MM/DD format). `rating_change` and `age` come from race cards (not results pages) so are only populated for races where predictions were generated.

### Data Fetching (hkjc_data_fetcher.py)

Three methods tried in order:
1. **requests + BeautifulSoup** (fast, reliable for race cards)
2. **Playwright** (slower, gets live odds but often times out)
3. Tips data fetched separately for expert ratings

Place odds are fetched via `_apply_odds_via_requests()` using `fetch_odds_html(date, race_no, 'PLA')` — but PLA odds are only available near race time (returns None pre-race).

**Venue detection is critical**: `HKJCApiClient.detect_venue(race_date)` auto-detects ST (Sha Tin) vs HV (Happy Valley). Using wrong venue returns empty HTML with no tables. Wednesday meetings are typically Happy Valley; weekends are Sha Tin.

### Dashboard

`generate_predictions.py` generates a self-contained HTML dashboard with embedded race data as JSON. The venue is detected dynamically from race data. `results/racing_dashboard.html` is always overwritten with the latest run (date-stamped archive also saved).

**Card-stack layout** (v6.5): topbar with brand + **📋 Race Order / ⭐ Top Bets** view tabs → hero stack of full bet cards (one per race in Race Order, curated subset in Top Bets) → sticky alerts rail on ≥1024px.

Each card is **click-to-expand**: clicking anywhere on the card toggles a per-race detail panel inserted directly below it (per-horse table, **same-format history** of each horse's prior runs at this exact venue+distance, speed map, reasoning). Top-right chevron flips ▼ → ▲ and the card gets a gold ring when expanded. **Copy WA** button uses `event.stopPropagation` so it does not toggle the card. Each expanded panel also has a × close button in the top-LEFT (top-RIGHT was hidden behind the tier badge in earlier versions).

Each expanded race shows a **BET TICKET** block at the top — synthesizes per-horse `bet_recommendation` values into a single race-level instruction (e.g. "STRAIGHT QIN · #5 THE HEIR + #11 PRESTIGE HALL", "BOX QIN-4 · #1 + #9 + #13 + #2", "BANKER WIN · #1 KA YING RISING").

Plain-English Pure / Market / Edge labels with click-to-toggle ⓘ popovers (replaces native `title=""` tooltips so it works on touch). Topbar **⟳ Hard reload** button bypasses browser cache. Atomic dashboard writes (`os.replace`) so a polling browser never sees a half-written file.

The legacy 11-tile strip + per-tile detail panel are still in the DOM (hidden by default) for backwards compatibility; the unified card stack supersedes them. Legacy A+/B+ grade system, `is_best_ev`/`is_worst_ev` ring highlights, and `strategy_confidence` string have all been removed. Empty-state honest-day message renders in Top Bets view when no race clears HIGH/MED tier.

**Critical inline-handler note**: All click-handler functions (`showRaceDetail`, `closeRaceDetail`, `toggleCardDetail`, `setStripView`, `hardReload`, `refreshDashboard`, `copyWA`, `showInfoPop`, `_closeInfoPop`, `toggleExpandAll`) are explicitly assigned to `window.X = X` at the bottom of the IIFE. Without this, the IIFE wraps everything and inline `onclick=""` attributes throw "X is not defined". Add the same `window.X` line whenever introducing a new function called from inline HTML.

All scraped data (horse names, jockeys, trainers, reasoning) is HTML-escaped via a JS `esc()` helper before `innerHTML` injection (XSS protection). The JSON embedding sanitizes `</script>` tags to prevent script block injection. The WA Copy buttons use a `data-payload` attribute + `addEventListener` pattern (not inline `onclick`) to prevent attribute-injection on horse names containing apostrophes or backslashes.

### Race Day Monitor

`python3 -m scripts.monitor '2026/04/16'` — polls odds 6 minutes before each race (every 5 min), snapshots to odds_history.db, detects steam/drift/cold horses, regenerates dashboard with auto-refresh (60s). Dashboard at `results/racing_dashboard.html` reloads automatically in browser.

### Enhanced Dashboard Fields

JSON output now includes per-horse: `going_form`, `jockey_horse_stats`, `distance_pb`, `distance_record`, `odds_movement`, `steam_move`, `cold_horse`. Per-race: `speed_map`. Dashboard renders these as enhanced stats grid, market move badges (steam/drift/cold), and a visual speed map.

### Edge Math + Conviction Tiers (v6.0)

Per-race JSON fields produced by `betting/edge_calculator.py`, `betting/conviction.py`, `betting/alerts.py`, `betting/tier_stats.py`:

**Per-horse:**
- `pure_prob`: softmax of `0.6·sigmoid(rule_score) + 0.4·ml_prob` — our independent opinion BEFORE market blend. Feeds edge math.
- `market_prob`: normalized `1/win_odds`
- `edge_pp`: `(pure_prob − market_prob) × 100` — percentage-point gap
- `ev_win`: `pure_prob × win_odds − 1` — expected ROI per $1 WIN stake
- `overlay`: `pure_prob × win_odds` (>1.0 = edge exists)
- `edge_fires`: bool — True only when all 6 ★ EDGE gates pass
- `model_rank`: 1-indexed model rank within field

**Per-race:**
- `conviction_tier`: EDGE / HIGH / MED / LOW / SKIP (replaces deprecated `strategy_confidence`)
- `has_edge`: bool — any horse has `edge_fires=True`
- `odds_age_minutes`: derived from `MAX(odds_snapshots.timestamp)` for the race; feeds G5
- `top_2_share`, `top_3_share`: sum of `calibrated_prob` across top model picks (feeds tier math)
- `tier_stats`: per-tier `{hit, total, rate, ci_low, ci_high}` from `prediction_log × race_results` over a sliding 124-race window, Wilson 95% CI via `statsmodels.stats.proportion`. Cached to `tracking_data/tier_stats.json`, refreshed by `dashboard/results_fetcher.py` on every results ingestion.
- `alerts`: list of `{type, race, horse, ts, severity, detail}` — STEAM (-25%), COLD (+40% on top-pick), EDGE_OPENED/CLOSED/WIDENED, RACE_UPCOMING. Per-day deduplication persisted to `tracking_data/alerts_state_YYYYMMDD.json` so re-running `generate_predictions.py` doesn't double-fire alerts. EDGE_CLOSED suppresses subsequent EDGE_OPENED on the same race within the day.
- `qin_top_pair_dividend`, `qin_bet_ev`, `qin_dividend_source`: Harville-based quinella pricing (or real HKJC payout when results are ingested). HK 17.5% takeout factored in.
- **Removed in v6.0**: `is_best_ev`/`is_worst_ev` (superseded by EDGE gate), `combined_score` (redundant with `calibrated_prob × 100`), `strategy_confidence` (renamed to `conviction_tier`).

**6-gate EDGE filter** (all must pass for ★ EDGE):
| Gate | Threshold | Reason |
|------|-----------|--------|
| G1 EV | `ev_win ≥ +0.05` | 5% expected ROI per $1 WIN stake |
| G2 Prob floor | `pure_prob ≥ 0.10` | No edge-flagging on noise-floor longshots |
| G3 Edge gap | `edge_pp ≥ 5.0` | Raw gap backstop |
| G4 Rank alignment | `model_rank ≤ 2` | Don't flag rank-5 longshot as edge |
| G5 Odds freshness | `odds_age_min < 30` | Stale odds invalidate EV |
| G6 Integrity | F-05 check passes | Model trained on ≥1000 real samples, <90 days old |

**Sample-size note**: at the current 124-race holdout, `Corr(edge_pp, actual win) = -0.21` over the 25 model-vs-market disagreement points — i.e. our disagreements at ≥5pp are NEGATIVELY predictive. CI is wide (n=25). Watch for ~50 disagreements before deciding whether to tighten G3 from +5pp to +8pp.

### Historical Scraping (scrape_history.py)

Uses Playwright to scrape HKJC results pages (JS-rendered). Extracts from results table columns: position(0), horse_no(1), name(2), jockey(3), trainer(4), weight(5), horse_weight(6), draw(7), LBW/margin(8), running_position(9), finish_time(10), win_odds(11). The `--rankings` flag scrapes the HKJC Jockey/Trainer Ranking pages for full-season stats.

## Codebase Conventions

- Code is organized into packages (`core/`, `analytics/`, `data/`, `monitor/`, `dashboard/`, `betting/`, `scripts/`). Root-level shims provide backwards compatibility.
- Async/await used in data fetching pipeline (`HKJCDataFetcher.get_race_card` is async).
- Race dates use `YYYY/MM/DD` format in HKJC URLs and data, but `YYYYMMDD` in filenames. SQLite `julianday()` requires `YYYY-MM-DD` — always use `REPLACE(date,'/','-')`.
- The HTML parser (`HKJCParser`) expects specific HKJC table layouts (27 cells per row with documented column indices). HKJC site changes can break parsing — check `_parse_horse_table()` if horses come back empty.
- ML model saved as pickle dict with keys: `model`, `place_model`, `calibrator`, `version`, `n_features`, `feature_importance`, `trained_date`. A `.sha256` sidecar file is required for loading — model refuses to load without valid checksum.
- Dashboard probabilities are stored as decimals (0.178) in JSON but displayed as percentages (17.8%) — multiply by 100 in JavaScript display code.
- `HKJCParser._safe_int()` preserves minus signs (regex `[^\d\-]`) — important for `rating_change` and `weight_change` which can be negative.

## ML Model — Lessons Learned (2026-04-18 Audit)

### Critical Rules for ML Features

1. **No data leakage**: Every feature must use ONLY data available BEFORE the race. Never use finish_position, finish_time, running_position, or margin from the CURRENT race. For horse history features (C&D form, pace, speed), use strictly `prior_date < current_date`. `extract_features()` now accepts `race_date` via `horse_data` and filters DB queries with `AND race_date < ?`. Track records in training are also date-aware (cumulative best-time lookup, not global).

2. **Train/predict consistency**: Features in `build_training_data_from_db()` MUST match `extract_features()` exactly — same computation, same order, same fallback values. `generate_synthetic_training_data()` must also match the same 16-feature schema. The DB (race_results) has different fields than race cards: rating, last_6, gear were ~97% missing in DB before the Renavon backfill (now 95%+ populated). Both training and inference must use the SAME helpers (e.g. `extract_class_rank` for class strings, dslr fallback for freshness) — divergence between them caused multiple v6.x bugs.

3. **No constant features**: If a feature has std < 0.01 in training data, it's wasted. Common cause: field is mostly null and falls back to 0.5. Check with `np.std()` after building features. Feature correlation matrix is logged after each training run — pairs with |r| > 0.7 are flagged.

4. **value_score must use ML blend**: `value_score` in HorseAnalysis feeds `combined_score` in the JSON. It must be `combined_prob * 100` (the 60/40 rule+ML blend), NOT raw `rule_score`.

5. **Probability sync after rank swaps**: When `_assign_bet_recommendations` swaps horse positions (market-favourite promotion or BANKER override), `combined_probability` and `expected_return` on ALL `HorseAnalysis` objects must be re-written from the swapped `calibrated_probs` list.

6. **ML fallback probability**: `predict()` accepts `field_size` param (default 12). When model is untrained, returns `1/field_size` — not `1/len(features)` which is always 23.

7. **Feature count guard**: Model pickle stores `n_features`. On load, it's checked against the current `NUM_FEATURES` — mismatched models refuse to load with a clear error message.

8. **Model integrity**: `.sha256` sidecar is required for loading. Missing checksum = hard refusal. `train_ml_model.py` backs up the old model before training and restores on failure.

### Current 19 ML Features (v6.5 — all prior-race only)

Features 1–8: `prior_win_rate, recent_form, draw_advantage, jockey_win_rate, trainer_win_rate, field_size, field_rel_draw, cd_form_prior`

Features 9–16: `class_level` (Group=1.0, Class1=0.8, Class5=0.0), `freshness, going_performance, weight_vs_field, jockey_horse_coop, last_start_margin, class_change, horse_weight_change`

Features 17–19 (v6.5 additions): `horse_age` (3yo→1.0, 9yo+→0.0), `rating_change` (+10pts→1.0, 0→0.5, -10→0.0), `jockey_form_30d` (Bayesian-smoothed 30-day strike rate)

Feature importance on the v6.5 50K-row training (none zero):
- `last_start_margin` 15.8%, `jockey_win_rate` 8.4%, `recent_form` 8.3%, `field_rel_draw` 5.7%, `class_level` 5.0%, **`rating_change` 4.6%**, `prior_win_rate` 4.6%, **`jockey_form_30d` 4.5%**, `going_performance` 4.3%, `field_size` 4.2%, `trainer_win_rate` 4.2%, `horse_weight_change` 4.2%, `weight_vs_field` 4.1%, `jockey_horse_coop` 4.0%, `cd_form_prior` 3.9%, `freshness` 3.9%, `draw_advantage` 3.6%, `class_change` 3.5%, **`horse_age` 3.2%**

**Caveat (worth tracking)**: `jockey_win_rate × jockey_form_30d` correlation = 0.750 (just over Wave 2c's 0.72 drop threshold). Both jockey signals; XGBoost handles correlated trees fine. Worth revisiting in a v6.6 cleanup pass.

Features removed:
- v6.4 (2026-04-25): `prior_pace, adjusted_speed, distance_pb_ratio, speed_consistency` — all 0.0 importance after the v6.3 → v6.4 bug fixes; redundant with `last_start_margin`.
- Wave 2c: `field_rel_speed` (r=0.72 with `recent_form`), `days_since_last_win` (r=0.74 with `recent_form`).
- Earlier audits: `odds_implied_prob`, `venue_win_rate` (r=0.984), `weight_carried_norm` (r=0.877), `draw_field_interact` (r=0.934), `prior_avg_position`, `prior_speed` (replaced by adjusted_speed before adjusted_speed itself was dropped), `late_money_signal` (constant 0.5 — train/predict distribution mismatch), `jockey_trainer_combo`.

### Class extraction helper (`ml/features.py:extract_class_rank`)

`re.search(r'(\d)', class_type)` was buggy — turned `'G1'` into class 1, but Group races sit ABOVE the digit ladder. v6.4 introduces `extract_class_rank()`:
- Group 1 / Group 2 → 0 (highest)
- Group 3 → 1 (≈ top of class 1 in difficulty)
- Class 1-5 → 1-5
- GRIFFIN, MAIDEN → 4 (restricted, lower-tier)
- 4YO / age-restricted → 3
- Empty/unparseable → None (caller defaults to 3)

Both training and inference share this helper. The `class_level` feature uses it for current-race rank; `class_change` uses it for current-vs-prior rank comparison.

### Training Pipeline

- **5-year window** (default, configurable via `TRAINING_YEARS_BACK` env var; set 0 to disable). Empirically: training on 49y of legacy data dropped ROI ~30% vs the recent 5y window. Older eras have different jockey rosters, rule sets, and pool dynamics that confuse the model. Retraining: `TRAINING_YEARS_BACK=5 SKIP_FETCH=1 python3 train_ml_model.py` (SKIP_FETCH bypasses the slow Playwright meeting fetch when DB already has what we need).
- **60/20/20 temporal split**: Train on first 60%, validate (early stopping + isotonic calibration) on next 20%, evaluate on final 20%. Test metrics are computed through the calibrator but the calibrator was never fitted on test data — metrics are unbiased.
- **Exponential recency weighting** (v6.4 — was linear): `w(i) = 0.25 + 0.75·exp((i-(n-1))/(n/3))`. Newest row weight 1.0, oldest ~0.29. With 50% jockey turnover and 94% horse turnover since 2021, older rows describe a different field; exponential decay halves their influence twice as fast as the prior linear schedule.
- **Field clipping** (v6.4 fix): training-side `field_size` and `field_rel_draw` are now clipped to [0,1]. Inference (features.py) already clipped via F-16; training (training.py) was missing the clip, so the model trained on values up to 15.0 from oversize legacy fields.
- **Freshness fallback** (v6.4 fix): when DB column `days_since_last_run` is null/zero (87.7% of legacy rows), training now falls back to `(race_date - prior race_date)` math — matching the inference-side fallback. Without this, freshness was a near-constant 0.5 in training and got 0.0 importance in v6.3.
- **Isotonic calibration** (Wave 2c switch from Platt): `IsotonicRegression(out_of_bounds='clip')` fitted on validation-set predictions (logit-input) → actual outcomes. Recalibrates XGBoost's overconfident probabilities. Fitted on validation (not test) so test metrics remain unbiased. The val slice currently spans 2024-04 → 2025-04 (1y stale on field strength) — known structural trade-off, not addressed in v6.4.
- **Place model**: Separate XGBoost trained on `is_top3` labels alongside the win model. Used for place bet probability and quinella placer-boost pricing.
- **scale_pos_weight**: Computed as `n_neg / n_pos` (≈ 11 with 8.3% positive class) and passed to XGBClassifier. Without this, the model would bias toward predicting "loss" everywhere (92% accuracy from doing nothing).
- **experiment_log integrity** (v6.4 fix): `log_experiment` is only called when `save=True`. Tests passing `save=False` were polluting `models/experiment_log.json` with n_samples=300/2000 dummy entries (399 polluted rows cleaned out, 5 real training runs retained).

### Speed Figures (`analytics/speed_figures.py`) — DEPRECATED in v6.4

The Beyer/Timeform-style adjusted speed figures are still computed and exposed via `analytics.speed_figures` for analyst use, but no longer fed into the ML model. The `adjusted_speed` and `speed_consistency` features were dropped in v6.4 because XGBoost ignored them in favor of `last_start_margin`. The `compute_par_times` precompute was removed from training.

### Exotic Bet Calculator (`betting/exotic_calculator.py`)

Prices all exotic bet combinations using conditional probability elimination:
- **Exacta**: P(A wins) × P(B wins | A removed)
- **Quinella**: Exacta(A,B) + Exacta(B,A), enhanced with place-model boost (consistent placers +15%), pace pairing boost (leader+closer +10%), draw clustering boost (adjacent good draws +8%)
- **Trifecta**: extends to 3-horse permutations (top 8 candidates)
- **First4**: extends to 4-horse permutations (top 6 candidates)
- **Value detection**: compares model probability vs market-implied, flags combinations with 10%+ edge

### Dynamic Market/Model Blend

Market blend weights adjust by race characteristics (HK bettors are sharp — larger/shorter races have sharper pools):
- Large fields (10+) + sprints (≤1200m): 70% market / 30% model
- Small fields (<10) or routes (≥1800m): 40% market / 60% model
- Default: 55% market / 45% model
- Quinella pool: lean 55-65% ML (less sharp money, public ignores consistent placers)

Grid-sweep chosen 2026-04-23: see docs/audits/2026-04-23/raw/wave2-sweep-summary.json.
Selection rule: lowest Brier on 73-race holdout (15 grid points, field_thresh fixed=10,
shift=±15%), tie-break within ±0.005 Brier toward simpler schedule (base_model_w
closest to 0.5). All 15 points within ±0.002 Brier — effectively at the noise floor.
Winner: base_model_w=0.45, field_thresh=10, dist_thresh=1200, Brier=0.1678.
Prior heuristic (base=0.60, dist_thresh=1400): Brier=0.1684.

### Renavon Bulk Datasets (2026-04-25 import)

After purchasing Renavon's **one-time** $49 HKJC History Export, 11 gzipped CSVs
were ingested into dedicated `renavon_*` tables alongside our scraped data.
The DB grew from ~4MB to ~614MB; **`tracking_data/*.db` is now gitignored**,
so the database must be regenerated locally from the source CSVs in
`renavon_Hongkong_Jockey_Club_races_and_horses/` (also gitignored).

**IMPORTANT: Renavon is a frozen snapshot, not a subscription.** All
`renavon_*` tables end at 2026/04/22 and will never grow. For new race data,
dividends, and rankings, use the HKJC scraper (`scripts/scrape.py`,
`scripts/scrape.py --rankings`, `scripts/backfill_dividends.py`). Don't
design features that assume continuous renavon_* updates — they won't
arrive. Renavon's role is one-shot historical backfill; ongoing freshness
comes from the scraper.

To rebuild from scratch:
```bash
python3 -m scripts.scrape --days 240            # results-page scrape (basic results)
python3 -m scripts.load_renavon_csvs            # 11 renavon_* tables
python3 -m scripts.backfill_card_fields         # backfill from saved race_data JSONs
python3 -m scripts.backfill_from_renavon        # backfill rating/last_6/gear/age/rating_change
python3 -m scripts.backfill_dividends           # real HKJC dividends
python3 -m scripts.train                        # retrain on now-feature-complete data
```

| Table | Rows | Cols | What it has |
|-------|------|------|-------------|
| `renavon_race_results` | 368,330 | 84 | Per-horse-per-race: `entry_rating`, `gear`, `horse_age`, sectional times (s1-s6), splits 800m/600m/400m/final 200m, lbw per section, breeding (sire/dam/dam_sire), career stats, owner, declared horse weight |
| `renavon_horses` | 17,789 | 32 | Per horse: `current_rating`, `last_rating`, `start_of_season_rating`, `season_rating_change`, sex, colour, age, breeding, career win/place rates, total prize money, days_since_last_race, stable_location |
| `renavon_sectional_times` | 126,140 | 44 | Extended pace data: led_early, closed_strongly, consistent_position, wire_to_wire flags + per-section positions and lbw |
| `renavon_dividends` | 306,657 | 16 | Pool payouts 2001-present (50× more than our scraped `race_dividends`) |
| `renavon_race_catalog` | 30,959 | 27 | Race-level catalog with track conditions (penetrometer/hammer, track_going, distance_category, race_class, video_replay_url) |
| `renavon_pool_investments` | 1,646,369 | 19 | Pool investment snapshots over time (per pool per race per minute-to-post) |
| `renavon_jackpots` | 49,223 | 12 | Multi-race pool jackpots (Six Up, Triple Trio, etc.) with estimated dividends |
| `renavon_jockey_rankings` | 1,015 | 20 | Per-snapshot jockey rankings (career + season stats) |
| `renavon_trainer_rankings` | 1,091 | 20 | Per-snapshot trainer rankings |
| `renavon_racing_changes` | 443 | 18 | Late changes: scratches, jockey/gear/barrier swaps |
| `renavon_track_conditions` | 26 | 10 | Pure penetrometer/hammer readings per meeting day |

**Indexes:** `renavon_race_results` indexed on `(race_date, race_number)`,
`horse_name`, `horse_id`. `renavon_horses` indexed on `horse_name`,
`horse_id`. `renavon_sectional_times` indexed on `race_date`, `horse_name`.

**Backfill impact** — `scripts/backfill_from_renavon.py` lifted card-field coverage in our existing `race_results` table:

| Field | Before | After |
|-------|--------|-------|
| rating | 5.0% | **98.5%** |
| last_6 | 4.8% | **94.8%** |
| gear | 4.4% | **52.1%** |
| age | 3.3% | **99.8%** |
| rating_change | 2.4% | **62.6%** |

**Future feature engineering candidates** (untapped):
- `renavon_race_results.section_*_time` + per-section `position_*` for richer pace features (closer/leader/midfield over multiple sectionals)
- `renavon_horses.season_rating_change` for "in form vs out of form" signal independent of recent finishes
- `renavon_racing_changes` for late-breaking change alerts (gear changes, jockey swaps within 24h)
- `renavon_horses.career_stage` (Experienced / New / Returning) as a categorical feature
- `renavon_track_conditions.penetrometer_reading` + `hammer_reading` for objective ground hardness over the going string

### DB Data Gaps (post-Renavon backfill)

- `rating`: 98.5% coverage (was 5%). `last_6`: 94.8%. `age`: 99.8%. `rating_change`: 62.6%. `gear`: 52.1%.
- `running_position` has two formats: space-separated (`"6 5 6"`) and concatenated (`"665"`). Always handle both when parsing.
- `horse_weight` is in lbs (900-1350 range), `weight` is carried weight assignment (113-133 lbs).
- `class_type`: 96% coverage; uses `extract_class_rank()` to handle G1/G2/G3/GRIFFIN/4YO. Missing/unparseable rows default to class 3 for class_change calculation.
- `days_since_last_run`: 87.7% missing in legacy 5y window — both training (v6.4 fix) and inference now fall back to `(race_date - prior race_date)` math.

### Betting Strategy — v6.4 Post-Audit (2026-04-25)

**v6.4 backtest** (last 500 races, 41 with real HKJC dividends, 50K-sample retrain after audit fixes):
- Recommender-gated: 38 bets, 15 hits (39.5%), ROI **+1433%** (was +1359% on v6.3)
- Brier (top-pick): **0.108** (was 0.176 on v6.3 — major calibration improvement)
- Straight-Q: ROI +1813%, Box-3: +1918%, Box-4: +1283% (all overlapping CIs)
- All 16 features have 4-16% importance — no dead weight after schema cuts

The recommender continues to auto-downgrade through conviction tiers (BOX Q4 / BOX Q / QIN / SKIP). Wilson CIs on n≈40 still overlap across strategies — no statistically dominant strategy.

**Edge correlation (still negative)**: `Corr(edge_pp, actual win) = -0.13 to -0.34` on disagreement points (n still small). The model's high-edge picks remain anti-predictive against the market at the +5pp threshold. Wide CIs — don't act, but watch for ≥50 disagreements.

Historical baselines (kept for context):
- v6.3 (5y, 50K samples, 20 features): recommender ROI +1359%, Brier 0.176
- v6.2 (49y, 360K samples, 20 features): ROI +927% — regression, kept training too far back
- v6.1 / Wave 2c (7.5K samples, 20 features): ROI +1359%, Brier 0.179
- Wave 2a baseline (n=72): recommender ROI +1976%, hit rate 47.0%

**Recommendation logic** (`core/recommender.py`) — unchanged from Apr 23, box-4 default retained:
- Default: `BOX Q4` — all top-4 model picks flagged `QIN` (6 combinations). Requires rank-3 AND rank-4 each have strong-history signals + top_4_share ≥ 0.38.
- Fallback: `BOX Q` (box-3, 3 combos) when rank-3 has strong history + top_3_share ≥ 0.30
- Fallback: `QIN` (straight, 1 combo) when top_2_share ≥ 0.25
- `TRIO` when top 3 are tightly bunched (gap_1_2 < 0.02 AND gap_2_3 < 0.02)
- `WIN` rare, only when rank-1 has BANKER signal AND market consensus (≥35% implied) AND clear top-2 gap (>0.08)
- `SKIP` if no live odds, field < 2, or no conviction gate met

WIN pool context (73-race holdout): model top-pick = market favourite 84.9% of the time. Flat-bet ROI -20.8% vs market-favourite baseline -61.2%. No WIN edge identified.

**Backtest reproduction**:
```
python3 -m scripts.backtest              # Default
python3 -m scripts.backtest --sweep      # Grid over temperature × model_w
```

- Races without live odds (pre-race or post-race) will show all SKIP — this is correct behaviour.
- `results_fetcher.py --auto-learn` scrapes live results from HKJC via `scrape_meeting_results()` and ingests into DB (no longer reads from `training_data/historical_races.json`). Retraining timeout is caught gracefully; run `python3 train_ml_model.py` separately if it times out.

### Probability Pipeline

```
rule_score (0-100) → sigmoid(0.06×(score-50)+logit(1/field_size)) → ×0.60
ml_prob (isotonic-calibrated) → ×0.40
pure_prob = softmax(rule × 0.6 + ml × 0.4) — independent opinion, pre-market

→ dynamic blend: model × (30-60%) + market × (40-70%)
  (F-09 chosen: base 45/55, sprint threshold 1200m, field threshold 10)
→ post-blend isotonic calibrator (v6.5, models/blend_calibrator.pkl, optional —
  loaded lazily by core/scorer.py if file present; rebuilt by
  scripts/calibrate_blend.py on 71K production-pipeline pairs)
→ calibrated_prob (post-blend, normalized to sum=1.0)
→ expected_return = odds × calibrated_prob - 1

Edge math (additive, computed from pure_prob NOT calibrated_prob):
edge_pp = (pure_prob − market_prob) × 100
ev_win  = pure_prob × win_odds − 1
overlay = pure_prob × win_odds
→ 6 EDGE gates → edge_fires bool → conviction_tier

Exotic pricing: conditional elimination for exacta/trifecta/quinella/first4.
```

Market odds enter ONLY through the dynamic market blend — removed from rule engine to prevent triple-counting. Blend weights are heavier on market for large sprint fields (sharp pool), heavier on model for small/long fields and exotics (thinner pool). **Edge math uses `pure_prob` (pre-blend) NOT `calibrated_prob` (post-blend)** — otherwise edge would be mechanically tiny because we'd be comparing market-influenced-us against market.

### Accuracy Report

`python3 -m scripts.accuracy --all` includes Brier score, Brier Skill Score, and a 5-bin calibration curve (predicted vs actual win rates). Use this to monitor calibration quality after retraining.
