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

# --- Multi-jurisdiction pipelines (UK, FR via theracingapi.com; AU/JP reference-only) ---
python3 -m scripts.ingest_uk --since-last     # weekly pull of UK results
python3 -m scripts.ingest_fr --since-last     # weekly pull of FR results
python3 -m scripts.ingest_au --since-last     # AU stakes-only, reference data
python3 -m scripts.ingest_jp --since-last     # JP JRA-only, reference data
python3 -m scripts.train_uk                   # retrain UK MVP (15 features)
python3 -m scripts.train_fr                   # retrain FR MVP (15 features)
python3 -m scripts.backtest_uk                # UK kill-or-keep ROI on test fold
python3 -m scripts.backtest_fr                # FR kill-or-keep ROI on test fold
python3 -m scripts.validate_uk --since-cutoff # forward-validation (true OOS)
python3 -m scripts.validate_fr --since-cutoff # forward-validation (true OOS)
```

## Architecture

### Package Structure

```
core/       — fetcher, analyzer, scorer, recommender, ML predictor          (HK)
ml/         — feature schema (24 features, v6.8), feature extraction,       (HK)
              model, training, integrity, experiments
              + uk_features.py, uk_model.py                                   (UK MVP)
              + fr_features.py, fr_model.py                                   (FR MVP)
analytics/  — 14 specialized analyzers (draw bias, pace, sectionals, etc.)  (HK only)
data/       — database, odds tracker, scraper, validation, sectional_scraper (HK)
              + uk_racing_api.py     (generic theracingapi.com client —
                                      services UK/AU/JP/FR despite the name)
              + uk_database.py, au_database.py, jp_database.py, fr_database.py
monitor/    — race day monitor (auto-rerun on sharp moves)                  (HK only)
dashboard/  — prediction generator, edge-first HTML template                (HK only)
betting/    — edge_calculator, conviction tiers, alerts, tier_stats,        (HK only)
              dividends, exotic_calculator, optimizer
scripts/    — CLI entry points
              HK:  generate, scrape, train, monitor, backtest, accuracy, ...
              UK:  ingest_uk, train_uk, backtest_uk, validate_uk
              FR:  ingest_fr, train_fr, backtest_fr, validate_fr
              AU:  ingest_au       (reference data only — no model)
              JP:  ingest_jp       (reference data only — no model)
tests/      — pytest suite (322 total: 314 HK tests + 8 UK tests)
```

Root-level shims exist for backwards compatibility (e.g., `hkjc_data_fetcher.py` re-exports from `core/fetcher.py`).

**Strict isolation between jurisdictions.** Each non-HK jurisdiction has its own SQLite file (`uk_history.db`, `au_history.db`, `jp_history.db`, `fr_history.db`), its own model pickle (where applicable), and its own scripts. HK code paths (`core/`, `analytics/`, `monitor/`, `dashboard/`, `betting/`) are not touched by any non-HK pipeline. See "Multi-jurisdiction pipelines" section at the end of this doc.

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
- **Rule engine (60%)**: 25 dynamically-weighted factors (added `speed_fig` v6.6). Missing-data factors are excluded and remaining weights renormalize to 1.0. The rule_score (0-100) is converted to a probability via a base-rate-anchored sigmoid: `sigmoid(0.06 × (score - 50) + logit(1/field_size))`. `odds` and `odds_movement` are NOT in the rule engine — market info enters via the dynamic market blend only (prevents triple-counting). v6.6 (2026-05-09) audit purged 3 sites that violated this: `field_size` factor (used `1/win_odds`), no-form fallback (used `1/win_odds`), and BANKER signal #9 (used `1/win_odds`). All replaced with structural-only logic.
- **ML model (40%)**: XGBoost classifier with **24 features** (`ml/model.py`, schema in `ml/feature_schema.py`). Includes a separate place model (`is_top3`). Both use IsotonicRegression recency-weighted, trained on 50,733 real race results from a 5-year window. After the market blend, a second isotonic (`models/blend_calibrator.pkl`, fit by `scripts/calibrate_blend.py` on the 70-85% calibration fold ~8,271 production-pipeline pairs) recalibrates the post-blend distribution per-horse with renormalization. **Auto-learn refits the calibrator after every retrain** (gated on `retrain_ok` — won't refit against a stale model). 4-fold split: rows[0:60%] train, [60:70%] earlystop, [70:85%] calibrator, [85:100%] backtest holdout — strictly disjoint.

The 14 specialized analyzers live in `analytics/` (draw bias, pace, sectionals, barrier trials, C&D, class movement, gear, weight, going preference, rolling form, beaten margins, jockey change, speed figures, jockey form). `analytics/speed_figures.py` was wired into the rule engine in v6.6 (was previously orphaned). `analytics/gear_change.py` scores are now data-calibrated by `scripts/calibrate_gear.py` (auto-runs after each meeting via auto-learn) — `models/gear_scores.json` overrides hand-tuned defaults.

→ Full feature list, importance table, removed-features history, training pipeline detail: `docs/ml-model.md`.

### Database

SQLite at `tracking_data/horse_history.db`. Five tables:
- **race_results** (primary data source): one row per horse per race, keyed by (race_date, race_number, horse_name). Contains finish_position, is_winner, is_top3, margin, odds, horse_weight, running_position, finish_time, days_since_last_run, **sectional_times** (Path B, 99.4% backfilled).
- **race_dividends** (pool payouts): one row per (race_date, race_number, pool, combination). Pools: WIN, PLACE, QUINELLA, QUINELLA PLACE, FORECAST, TIERCE, TRIO, FIRST 4, QUARTET. Backfilled via `python3 -m scripts.backfill_dividends`. Used by `scripts/backtest.py` for real-dividend ROI (replaces Harville estimate when data is available).
- **horses**: master record per horse. Note: `get_horse_record()` computes aggregates from race_results directly (not from this table's stale counters).
- **jockey_stats** / **trainer_stats**: season-level win/place rates. Populated two ways: `rebuild_stats()` computes from local race_results; `scrape_history.py --rankings` fetches full-season stats from HKJC (more accurate).

**Data integrity**: `ingest_race_day()` calls `rebuild_stats()` after every ingestion, which overwrites jockey/trainer stats with local-only data. If full-season HKJC rankings were previously imported, re-run `python3 scrape_history.py --rankings` after ingestion to restore them. The `horses` table upsert derives `total_starts`/`total_wins`/`total_places` from `race_results` via subquery — safe for re-ingestion (idempotent, no double-counting).

**Auto-computed fields**: `days_since_last_run` is backfilled from race_results on ingestion (uses `julianday(REPLACE(date,'/','-'))` since dates are YYYY/MM/DD). `rating_change` and `age` come from race cards (not results pages) so are only populated for races where predictions were generated. **`rating_change` post-Renavon backfill caveat**: the Renavon CSV `renavon_race_results` lacks the column directly — it lives in `renavon_race_catalog`. After any DB rebuild from Renavon, re-run `python3 -m scripts.backfill_from_renavon` to restore non-zero `rating_change` (otherwise it defaults to 0 in 99.7% of rows and `test_no_constant_training_features` fails).

**Recovery workflow** (DB binary >100MB, can't be committed directly to GitHub):
- **Dump**: `python3 -m scripts.dump_db` → writes `tracking_data/backup/db_core_dump.sql.gz` (~3 MB, deterministic gzip with mtime=0 so byte-stable for git diff). Covers race_results, horses, jockey_stats, trainer_stats, race_dividends, prediction_log.
- **Restore**: `python3 -m scripts.restore_db` → moves existing DB aside, replays SQL dump, then re-loads Renavon CSVs via `scripts.load_renavon_csvs`. Idempotent.
- **Auto-dump**: launchd job `com.hkjc.dbdump` (`scripts/launchd/com.hkjc.dbdump.plist`) runs `scripts/auto_dump.sh` Mon + Thu 02:00 HKT (after Sun ST + Wed HV race days). Logs to `logs/auto_dump.log`. Commits + pushes only if dump bytes changed.

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

**7-layer BET TICKET** per card:
1. **PRIMARY** (gold) — recommender's choice with jockey + 30-day strike rate
2. **+ BOX QIN-4** (green) — top-4 model picks, always-on protection (catches rank-3/4 placing-pair upsets)
3. **+ F4 SAVER** (purple) — top-4 in 1-2-3-4 exact order, single unit. **Gated** to skip low-EV setups: heavy fav (top1 odds <$2.5) collapses the dividend pool; LOW conviction means top-4 ordering is unreliable; field <8 shrinks combos. Renders "F4 SAVER · SKIPPED · <reason>" badge when gated.
4. **+ QIN PLACE BANKER** (blue) — only when top pick odds ≤ 1.8 (favorite + 3 bankers, 3 combos)
5. **★ MAIN · F4 BOX-5** (purple→pink gradient) — top-5 model picks in a 5-combo F4 box (any order, top-4 finishers). Shipped 2026-05-06 after 60-race backtest: **+54% ROI vs TRIO-5's +8%** at half the stake ($50 vs $100). Median F4 dividend $1,227 (vs TRIO $605); per-hit return ~9× the stake. Same gates as F4 SAVER (skip heavy fav, LOW conv, field <8). Treated as MAIN bet alongside PRIMARY/TRIO-5 in WhatsApp.
6. **+ BOX TRIO-5** (pink) — top-5 in 10-combo TRIO box (catches rank-5/6 spoilers). Kept as parallel MAIN with F4 BOX-5: TRIO-5 still hits when 4th finisher is outside top-5 (different miss profile). Treated as MAIN bet in the WhatsApp message (per 2026-05-03 user feedback).
7. **+ BOX TRIO-6** (rose, OPTIONAL) — top-6 in 20-combo TRIO box. 6-meeting evidence (Apr 12-29, 60 races): TRIO-6 lifts strike rate +10pp at both venues (HV +2/18, ST +4/42) but doubles stake. Economic break-even vs TRIO-5: avg TRIO dividend must exceed ~$111. Per-race **TAKE/SKIP/YOUR CALL** recommendation rendered with a **CHAOS 0-4 score** (big field ≥11, competitive favorite $2.5-$5, ≥2 morning-steam horses, HIGH/MED conviction). Rules: SKIP if heavy favorite ($<2.5); TAKE if ≥2 steam OR (HIGH+field≥11+competitive fav); YOUR CALL otherwise. **Spoiler-swap (2026-05-03)**: when a horse ranked 7-12 has gear visible (B/TT/CP/etc) AND market odds ≥20% shorter than rank-6, it auto-swaps into rank-6 slot with a 🔁 SWAP badge. Validated retro on 2026-05-03 R4 (#14 SUNNY Q, 3rd) and R5 (#8 FLYING KNIGHT, won) — both would have hit TRIO-6 with the swap. Needs N≥6 meetings of forward validation per project rule.

**Strike-badge precedence**: STRIKE-WIN → STRIKE-QIN → STRIKE-BOX-4 → STRIKE-F4-5 → STRIKE-TRIO-5 → STRIKE-TRIO-6 → MISS. F4-5 is checked before TRIO-5 because an F4-5 hit always implies a TRIO-5 hit (top-3 are part of top-4); we display the stronger result.

**Top Bets sub-toggle**: `Next` / `Past` filter — `Next` is default, auto-flips to `Past` at end-of-meeting. Past races render outcome badges: 🟡 STRIKE·WIN, 🟢 STRIKE·QIN, 🟢 STRIKE·BOX-4, 💜 STRIKE·F4-5, 🔴 STRIKE·TRIO-5, 🌹 STRIKE·TRIO-6, 🔴 MISS.

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
- ML model saved as pickle dict with keys: `model`, `place_model`, `calibrator`, `place_calibrator`, `version`, `n_features`, `n_samples`, `feature_names`, `feature_importance`, `trained_date`. A `.sha256` sidecar file is required for loading — model refuses to load without valid checksum.
- Dashboard probabilities are stored as decimals (0.178) in JSON but displayed as percentages (17.8%) — multiply by 100 in JavaScript display code.
- `HKJCParser._safe_int()` preserves minus signs (regex `[^\d\-]`) — important for `rating_change` and `weight_change` which can be negative.

## ML Critical Rules

These are timeless guidance — violations have caused multiple v6.x bugs.

1. **No data leakage**: Every feature must use ONLY data available BEFORE the race. Never use finish_position, finish_time, running_position, or margin from the CURRENT race. For horse history features (C&D form, pace, speed), use strictly `prior_date < current_date`. `extract_features()` accepts `race_date` via `horse_data` and filters DB queries with `AND race_date < ?`. Track records in training are also date-aware (cumulative best-time lookup, not global).

2. **Train/predict consistency**: Features in `build_training_data_from_db()` MUST match `extract_features()` exactly — same computation, same order, same fallback values. `generate_synthetic_training_data()` must also match the same feature schema. The DB (race_results) has different fields than race cards. Both training and inference must use the SAME helpers (e.g. `extract_class_rank` for class strings, `parse_age_score` for ages, dslr fallback for freshness) — divergence between them caused multiple v6.x bugs. **Concrete pattern**: when adding a feature whose raw value can come in multiple formats (DB `'5.0'` vs race-card `'5'`), define a single `parse_X_score()` helper in `ml/features.py` and call it from both sites. Inline duplicates drift silently and degrade the feature.

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

## DB Data Gaps (post-Renavon backfill, audited 2026-05-16)

- `rating`: 100% coverage. `rating_change`: 100%. `last_6`: 95.5%. `gear`: 93%. `age`: **68.4%** (was claimed 99.8% — coverage actually degrades back in time: 2021 = 20%, 2022 = 45%, 2023 = 79%, 2024 = 86%, 2025 = 95%, 2026 = 100%). `sectional_times`: 99.4% (Path B backfill, permanent — auto-captured forward).
- `age` format gotcha: Renavon CSV writes `'5.0'`, live race-card scrape writes `'5'`. Both go through `parse_age_score()` (single source of truth in `ml/features.py`); never reimplement inline. Pre-fix the inline `int(str(age)[:2])` parser failed silently on `'5.0'` and 94% of training rows fell back to 0.5 — a real train/predict mismatch (ML rule #2 violation).
- `running_position` has two formats: space-separated (`"6 5 6"`) and concatenated (`"665"`). Always handle both when parsing.
- `horse_weight` is in lbs (900-1350 range), `weight` is carried weight assignment (113-133 lbs).
- `class_type`: 96% coverage; uses `extract_class_rank()` to handle G1/G2/G3/GRIFFIN/4YO. Missing/unparseable rows default to class 3 for class_change calculation.
- `days_since_last_run`: 87.7% missing in legacy 5y window — both training and inference fall back to `(race_date - prior race_date)` math.

## HKJC parser gotchas

- **Brand suffix on horse names**: `parse_results` returns names like `'PODIUM(H173)'` from the HKJC results page. The parser strips the trailing `\s*\([^)]+\)\s*$` so downstream code (especially `prediction_log.fill_prediction_results` which keys on `horse_name`) sees the clean `'PODIUM'`. Apply the same strip to any new HKJC parser that reads the horse-name cell, otherwise UPDATE-by-name silently matches 0 rows.
- **Multi-paren handling**: country-coded names like `'STAR (NZ)(H173)'` correctly become `'STAR (NZ)'` (only the trailing brand is stripped).

## Renavon — frozen snapshot

`renavon_*` tables are a one-time historical backfill ending 2026/04/22. They will never grow. For new race data, dividends, and rankings, use the HKJC scraper. Do NOT design features that assume continuous renavon_* updates.

→ Table inventory, indexes, backfill impact, rebuild commands, untapped feature candidates: `docs/renavon.md`.

## Multi-jurisdiction pipelines (UK / FR / AU / JP)

Added 2026-05-17. All non-HK racing data flows from **theracingapi.com** (HTTP Basic Auth via `.env` — `RACING_API_USERNAME` / `RACING_API_PASSWORD`). The single API client at `data/uk_racing_api.py` services every non-HK jurisdiction; only the `region` filter and DB destination differ.

### Jurisdiction status

| Jurisdiction | DB | Model | Status | 12mo volume |
|---|---|---|---|---|
| **HK** | `horse_history.db` | v6.8 production | Production. Untouched by this expansion. | 55,157 race results |
| **UK** | `uk_history.db` | `uk_racing_model.pkl` (15-feature XGB) | MVP — forward-validating | 81,975 runners |
| **FR** | `fr_history.db` | `fr_racing_model.pkl` (15-feature XGB) | MVP — forward-validating, **beats market favourite** in backtest | 21,267 runners |
| **AU** | `au_history.db` | none | Reference data only — too sparse on standard tier | 4,444 runners |
| **JP** | `jp_history.db` | none | Reference data only — no OR ratings on standard tier | 2,069 runners |

### Forward-validation loop (UK + FR weekly)

```
python3 -m scripts.ingest_uk --since-last     # pulls UK results since last DB date
python3 -m scripts.validate_uk --since-cutoff # scores model on all OOS races
python3 -m scripts.ingest_fr --since-last
python3 -m scripts.validate_fr --since-cutoff
```

`validate_*` reports flat WIN ROI for three strategies: **model top-1**, **market favourite**, and **model TOP1 vs market** (the disagreement edge probe). Decision rules:
- n < 200 disagreements → directional only
- n ≥ 200 AND ROI > 0 → edge confirmed, revisit Phase 2B build
- n ≥ 200 AND ROI ≤ 0 → kill that pipeline

### Critical design rules (apply to UK + FR; same as HK ML rules)

1. **No data leakage** — features use only `prior_date < current_date`. Verified by `groupby + shift(1)` patterns in `ml/*_features.py`.
2. **Feature schema is frozen at the MVP** — adding more features degraded ROI on both UK and FR (memory: `uk-feature-ceiling-2026-05-17`, `feature-experiments-brier-vs-roi-pattern`). Do NOT add features without empirical justification from forward-validation.
3. **Phase 2B (place model, calibration, market blend, recommender) is deferred** — post-processing layers can't fix a model with no underlying rank edge. Confirm the disagreement edge at n≥200 BEFORE building 2B.
4. **API client quirk**: `get_results_for_range` rejects `end_date >= today` with 422. The client clamps to (today − 1) silently — don't try to override.

### Adding a new jurisdiction (process if a 5th lands)

YAGNI says copy the AU pattern. The 4-way duplication of `*_database.py` files is intentional; refactor to a shared base class only when a 5th jurisdiction proves the abstraction is needed. Schema parity across all 4 stores means future cross-jurisdiction queries can `UNION ALL`.

### Credentials

`.env` contains `RACING_API_USERNAME` + `RACING_API_PASSWORD`. **Gitignored** (`.gitignore` line 16). `config.py` reads them via `os.environ.get(..., "")` — never hardcode. `.env.example` is a committed placeholder template. The full git history has been audited zero-leakage.
