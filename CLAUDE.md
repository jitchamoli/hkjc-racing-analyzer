# CLAUDE.md

Guidance for Claude Code working in this repo. Detail lives in `docs/`:
- `docs/ml-model.md` — feature inventory, removed-features changelog, training pipeline
- `docs/dashboard.md` — card-stack layout, BET TICKET prose, escape rules
- `docs/betting-strategy.md` — edge gates, recommender logic, exotic pricing
- `docs/renavon.md` — table inventory, backfill impact, rebuild

## Commands

```bash
# Race-day phase controller (preferred)
./raceday                       # interactive menu; choose a meeting date
./raceday start  YYYY/MM/DD     # predictions + shadows + optional API snapshot
./raceday live   YYYY/MM/DD     # refresh odds/predictions, then merge results
./raceday finish YYYY/MM/DD     # post-race ingestion, trackers and DB dump

# One-shots
python3 generate_predictions.py 'YYYY/MM/DD'
python3 results_fetcher.py     'YYYY/MM/DD' [--auto-learn]
python3 train_ml_model.py
python3 scrape_history.py [--date 2026/04/09 | --rankings | --days 90]

# Forward-validation + trackers (auto-run by post_race)
python3 -m scripts.validate_hk [--report]      # OOS-only ROI — the ONLY honest measure
python3 -m scripts.shadow_evaluate [--report] [--variant jockey_form|tips]
python3 -m scripts.stake_ab     [--report]     # flat-$10 vs ramp vs flat-1% vs invert
python3 -m scripts.tier_b_evaluate [--report]
python3 -m scripts.track_longshot_trio [--report]
python3 -m scripts.track_spoilers DATE | track_edge DATE   # or --backfill-all (no --report)
python3 -m scripts.gate_impact [--last 12 | --all]   # OLD vs NEW BOX TRIO-5 triple gate
python3 -m scripts.track_gate_skips [--report | --backfill-all]  # do the GATES earn their skips?
python3 -m scripts.shadow_pure_rank [--report | --backfill-all]  # model-only (pure_prob) ranking shadow
python3 -m scripts.clv_report                  # closing-line value on rank-1 picks
python3 -m scripts.clv_report --split-by-market-rank  # EDGE GAUGE (auto-runs in raceday finish):
                                               #   splits CLV by market agreement at OPEN — favourite
                                               #   late-money MIRAGE vs real sharpness. WATCH the rank-2-3
                                               #   (mild-disagreement) CLV + whether it converts; rank-4+
                                               #   longshot CLV% is inflated noise. Verdict (2026-07-04):
                                               #   rank-2-3 CROSSED POSITIVE +3.9% CLV, place 53%, n=36
                                               #   (was -12.7% n=9 on 05-30). First time positive AND
                                               #   converting — TENTATIVE signal, don't act; hold N.

# Backtest — reads race_results, whose win_odds are the FINAL odds. That is an
# UPPER BOUND, not a live estimate: the market's own top 5 caught the podium
# 30.9% on final odds vs 25.2% at 5-15 min before the off and 18.2% at 60+ min
# (measured 2026-09-14). We bet on the stale side of that gap. Treat every
# backtest ROI as optimistic; `validate_hk --replay` is the honest verdict.
python3 -m scripts.backtest [--sweep | --exclude-date YYYY/MM/DD | --production-stakes]
python3 -m scripts.accuracy --all              # Brier, BSS, calibration

# Feature candidate diagnostics (cheap → re-rank → direct A/B → ship)
python3 -m scripts.diagnose_<feature>          # correlation Gates A1/A2/B on full DB
python3 -m scripts.test_<feature>_rerank       # signal-additivity on burned holdout
python3 -m scripts.compare_v6_x_v6_y           # direct trained A/B (ML head only)
python3 -m scripts.eval_swap_rule --rank-out 6 --odds-ratio 0.5

# DB recovery (binary too large for git)
python3 -m scripts.dump_db     # → tracking_data/backup/db_core_dump.sql.gz (~3 MB)
python3 -m scripts.restore_db  # replay SQL + reload locally supplied Renavon CSVs; idempotent
python3 -m scripts.backfill_dividends [--last 20 | --force]

# Bankroll status (reads generated tickets; not confirmed purchases)
python3 -m scripts.bankroll_status DATE         # daily exposure vs caps (config.py pcts); DATE required

# Barrier trials (data infra only — feature KILLED; keep accruing for the record)
python3 -m scripts.scrape_trials --since-last  # → barrier_trials table (ST/HV/Conghua)

# Multi-jurisdiction (UK/FR via theracingapi.com; AU/JP reference-only)
python3 -m scripts.ingest_{uk,fr,au,jp} --since-last
python3 -m scripts.{train,backtest,validate}_{uk,fr} [--since-cutoff]

# Optional third-party RacingAPI integrations (not required by the HKJC path).
# The previous subscription is inactive; these commands need valid credentials
# and may fail until the service is re-enabled.
python3 -m scripts.hk_news YYYY/MM/DD [--picks]
```

## Architecture

```
core/       — fetcher, analyzer, scorer, recommender, ML predictor   (HK)
ml/         — feature_schema (23 features), features, model,          (HK)
              training, integrity; + uk_features/model, fr_features/model
analytics/  — 14 specialized analyzers                                (HK)
data/       — database, odds_tracker, scraper, sectional_scraper,    (HK)
              uk_racing_api (services UK/FR/AU/JP),
              {uk,au,jp,fr}_database
monitor/    — race day monitor (HK)
dashboard/  — generator + edge-first HTML template (HK)
betting/    — edge_calculator, conviction tiers, alerts, exotic_calculator (HK)
scripts/    — CLI entry points
tests/      — unit, integration, contract, Playwright and template tests
```

Root-level shims preserve backwards compatibility. Non-HK clients and models
are separate from HK, but their data ingestion requires an active third-party
RacingAPI subscription. Do not treat those pipelines as live-supported while
the subscription is inactive.

### Data pipeline
```
core/fetcher → core/analyzer → dashboard/generator → results/
                    ↑                ↓
            data/database  ← dashboard/results_fetcher (post-race)
            (tracking_data/horse_history.db)
                    ↓
            scripts/train → models/hkjc_racing_model.pkl
```

### Scoring (CombinedRaceAnalyzer)
- **Rule engine (60%)**: 25 dynamically-weighted factors, missing-data renormalized. `odds`/`odds_movement` NOT in rule engine — market enters via dynamic blend only.
- **ML model (40%)**: XGBoost win + place (`is_top3`) with 23 features (`ml/feature_schema.py`). IsotonicRegression recency-weighted; trained on ~50k real results. Post-blend isotonic recalibrator (`models/blend_calibrator.pkl`) refit after every auto-learn retrain (NOTE: manual `train_ml_model.py` does NOT refit it — deferred item). 4-fold disjoint split: 0-60 train / 60-70 earlystop / 70-85 cal / 85-100 holdout.

**v6.11 (2026-05-24)** Tier 1 PRUNE from 3-expert review (ML stats / HK domain / forward-validation pragmatist) dropped 3 features:
- `draw_advantage` — r=−0.0095 with `is_winner` on 50k rows (literal zero). `field_rel_draw` carries the draw signal.
- `speed_fig_norm` — r=+0.53 with `last_start_margin`. HK paceless; prior revival was NEUTRAL-for-betting.
- `running_style` — r=+0.914 with `early_speed` (coarsened duplicate).
Backup: `models/hkjc_racing_model.pkl.pre-prune-tier1`. Gate-tune attempt (+0.15) reverted after 300-meeting quantile-match showed real distribution shift was only +0.007. See memory: `project_v6_11_prune_and_gate_revert_2026-05-24.md`.

**v6.9/v6.10 (2026-05-24)** added two features via the cheap-diagnostic → re-rank → A/B pipeline:
- `mid_race_pace` (v6.9, Feature 22 in v6.11) — (800-400)-(last-400) split, last-3-runs mean. Source: `race_results.sectional_times`. Backup: `models/hkjc_racing_model.pkl.pre-mid-race-pace`.
- `sire_top3_rate_on_going` (v6.10, Feature 23 in v6.11) — sire's Beta(2,8) smoothed top-3 rate on TODAY's going (`_GOING_REVERSE` map: FIRM/GY/YIELDING/SOFT/WET-FAST). Source: `renavon_horses`. Backup: `models/hkjc_racing_model.pkl.pre-sire`.

→ Full layout, BET TICKET prose, strike-badge precedence: `docs/dashboard.md`.
→ Edge gates, conviction tiers, exotic pricing: `docs/betting-strategy.md`.
→ Full feature list, importance, removed-features: `docs/ml-model.md`.

## ML Critical Rules

1. **No data leakage**: features use ONLY data with `prior_date < current_date`. `extract_features()` filters DB queries with `AND race_date < ?`. Never use the current race's `finish_position`/`finish_time`/`running_position`/`margin`.

2. **Train/predict consistency**: `build_training_data_from_db()`, `extract_features()`, and `generate_synthetic_training_data()` must compute the SAME features in the SAME order with the SAME fallbacks. When raw values come in multiple formats (DB `'5.0'` vs card `'5'`), define ONE helper in `ml/features.py` (e.g., `parse_age_score`, `extract_class_rank`) and call from both sites — never inline duplicates. **PRIOR-history scoping**: training-time horse/jockey/trainer lookups must query the FULL DB (not the windowed training rows), because inference always queries the full DB. The cutoff applies only to training LABELS.

3. **No constant features**: std < 0.01 → wasted. Correlation matrix logged after each training run; pairs with `|r| > 0.7` flagged.

4. **value_score must use ML blend**: `value_score = combined_prob * 100`, NOT raw `rule_score`.

5. **No rank promotions after calibration** (GPT-5.5 R13 F3 2026-05-25): `assign_bet_recommendations` MUST emit horses strictly sorted by `calibrated_prob` desc. Market-favourite and BANKER rank swaps were REMOVED — calibrated_prob already includes the market blend, so a post-calibration swap is a second override. BANKER is reasoning/diagnostic only; BANKER WIN may fire only when the BANKER horse is already rank 1 by calibrated_prob. Snapshot pre-blend order as `pre_blend_rank`; the sort key is `(calibrated_prob, -pre_blend_rank)` desc, so ties (e.g. two co-favourites both clamped to iso(0.30)) break DETERMINISTICALLY by best pre-blend opinion — never by incidental input order (2026-06-27). Locked by `tests/test_banker_cal_prob_guard.py` + `test_race_day_contracts::test_recommender_does_not_swap_rank_for_market_fav_or_banker`.

6. **ML fail-closed** (F-17): untrained `predict()` (`is_trained False` / load or integrity failure) **raises `RuntimeError`** — it does NOT silently return `1/field_size`, so a bad/missing pickle surfaces loudly instead of being masked by plausible-looking uniform probabilities. Matches the `CombinedRaceAnalyzer(fail_closed_ml=True)` posture and the "ML Fail-Closed" contract-test group. (`predict_place()`'s no-place-model fallback delegates to `predict()`, so it fails closed too.)

7. **Feature count guard**: pickle stores `n_features`; mismatch with current `NUM_FEATURES` = hard refusal.

8. **Model integrity**: `.sha256` sidecar required for loading. `train_ml_model.py` backs up old model and restores on failure.

## Probability Pipeline

```
rule_score (0-100) → sigmoid(0.06×(score-50)+logit(1/field_size)) → ×0.60
ml_prob (isotonic) → ×0.40
pure_prob = normalise(rule×0.6 + ml_RAW×0.4) → pure_calibrator.pkl isotonic
            # independent opinion, pre-market. NOT softmax (that flattens the
            # model's own spread and drives every edge_pp negative). Uses the
            # RAW pre-isotonic XGBoost prob, not MLPrediction.win_probability.

→ dynamic blend: model × (30-60%) + market × (40-70%)
  (base 45/55, sprint threshold 1200m, field threshold 10)
  All weights live in `core/scorer.BlendConfig` (frozen dataclass, injectable via
  `blend_config=` on calibrate_probabilities / assign_bet_recommendations; default
  = `_DEFAULT_BLEND`). Calibrator likewise injectable (`USE_PRODUCTION_CALIBRATOR`
  sentinel loads the pickle; pass an explicit one for A/B, None to disable).
  60/40 vs 45/55 was A/B'd on a recalibrated 634-race holdout = wash (more-market
  loses slightly less). Don't re-tune without a fresh calibrator + ≥250-bet forward.
→ post-blend isotonic calibrator (blend_calibrator.pkl, lazy). Inputs CLAMPED at
  `_BLEND_CAL_INPUT_CLIP`=0.30 (last dense knot) before iso.predict — the fit tail
  had only ~80/7662 samples >0.30, so isotonic flattened 0.40..1.0 to one noisy
  value; clamp caps favourite over-statement (monotone, conservative, untouched
  <0.30). MUST be fit on its apply-time domain: `calibrate_blend.py` disables the
  cached calibrator during collection or re-fits compound the lift (memory:
  `project_calibration_fix_2026-06-27`).
→ calibrated_prob (normalized sum=1.0)
→ expected_return = odds × calibrated_prob - 1

Edge math uses pure_prob NOT calibrated_prob (else comparing market-influenced
us against market → mechanically tiny). Same for find_value_exotics in
dashboard/generator.py (fixed 2026-05-21):
  edge_pp = (pure_prob − market_prob) × 100
  ev_win  = pure_prob × win_odds − 1
  overlay = pure_prob × win_odds
→ 6 EDGE gates → edge_fires bool → conviction_tier
```

**ONE definition of the model's own opinion (2026-08-03).** `betting/edge_calculator`
owns it: `compute_pure_probs(dicts)` / `pure_probs_from_analyses(analyses)`.
`HorseAnalysis.model_probability` is now that same number (verified bit-identical
to the dashboard's `pure_prob` on 150 real races, max Δ 1.1e-16, per-race sum
exactly 1.0). It used to be a parallel re-derivation reading the CALIBRATED
ml_prob with no normalisation and no isotonic — per-race sums ran 0.70–2.32, so
subtracting a normalised market prob from it produced a number that was not an
edge. That mis-scaling generated a spurious "+44% edge / n=109" reading which is
now WITHDRAWN. Diagnostic only — nothing in the recommender, ticket, or
leg-selection path reads it, so ranking is unaffected (straight QIN −0.81%,
box-4 −15.74%, Brier 0.1920, place ECE 0.0079 — all bit-identical after).
**Do not add a third definition.**

pure_prob is calibrated by `models/pure_calibrator.pkl` (`scripts/calibrate_pure`,
fit on the disjoint 70-85% calib fold, signed, refuses unsigned). Refit it after
any retrain — an uncalibrated pure_prob reads ~2× under-confident and bakes a
structural −10pp into every edge_pp, which is what kept the EDGE tier dead.
Stored `edge_pp` in `results/race_data_*.json` from BEFORE 2026/08/03 is on the
old scale — `stake_ab.PURE_CALIBRATION_DATE` gates `edge_filtered` accordingly.

## Evaluation and Betting Posture

No production strategy has demonstrated a durable positive betting edge. Treat
all historical ROI and CLV values as dated measurements, not current estimates;
multiple audits corrected leakage, form parsing, dividend units and ticket
semantics over time. Read the specific dated reports in `docs/audits/`, and
rerun current forward validation before making profitability claims.

- Use `python3 -m scripts.validate_hk --report --replay` for frozen-ticket
  evaluation. A synthetic top-N backtest is not a replay of the user's ticket.
- Judge ranking experiments on grouped walk-forward top-four capture,
  calibration, timestamped price quality and frozen-ticket results.
- Keep model/objective/gate changes shadow-only until forward evidence supports
  them across multiple meetings. Do not retune on a repeatedly inspected
  holdout.
- `race_bet_ticket` records generated recommendations, not confirmed purchases.
  Exposure summaries are therefore recommended exposure unless actual stakes
  are separately captured.
- Current policy: realized daily loss is a warning; `BANKROLL_LOSS_BLOCKS_BETS`
  defaults to `False`. Card exposure above the cap is displayed for manual
  trimming; it does not automatically remove tickets.

## DB (`tracking_data/horse_history.db`)

Core tables include `race_results`, `race_dividends`, horse/jockey/trainer
statistics and `race_bet_ticket`. The frozen ticket stores pool, label, legs,
combos, amount and reason; it is the source for `validate_hk --replay`, ticket
display and result badges. See `data/database.py` for the current schema.

**Data integrity**: `ingest_race_day()` calls `rebuild_stats()`, which recomputes jockey/trainer stats from `race_results` for the CURRENT SEASON ONLY. This is deliberate and correct, NOT damage to be undone — the window comes from `MAX(race_date)` (not the wall clock) so the same DB rebuilt on different days aggregates identically, which is what makes backtests reproducible across a season boundary (fixed 2026-08-03). Early in a season the counts are legitimately small — six meetings in, the top jockey has ~50 rides, not the ~700 of a full season; that is a stub season, not corruption. `scrape_history.py --rankings` is OPTIONAL enrichment with authoritative HKJC season figures, not a repair step. Note the season window here is Jul 1-Jun 30 while HK racing actually resumes in September, so a September card's `jockey_stats` mixes the previous season's July tail with the new opener — deliberate (recency beats a table that zeroes on opening day), but the season LABEL is then a mild misnomer. `horses` upsert derives counters from race_results via subquery (idempotent). After any DB rebuild from Renavon, re-run `scripts.backfill_from_renavon` to restore `rating_change` (lives in `renavon_race_catalog`, not `renavon_race_results`).

**Auto-dump**: launchd `com.hkjc.dbdump` runs `scripts/auto_dump.sh` Mon+Thu 02:00 HKT (after Sun ST + Wed HV). Logs to `logs/auto_dump.log`; commits + pushes only if dump bytes changed.

### DB data gaps (audited 2026-05-16)
- `rating`/`rating_change`: 100%. `last_6`: 95.5%. `gear`: 93%. `sectional_times`: 99.4%.
- `age`: **68.4%** (degrades back in time: 2021=20%, 2022=45%, 2023=79%, 2024=86%, 2025=95%, 2026=100%). Format gotcha: Renavon writes `'5.0'`, live race-card writes `'5'`. ML uses `parse_age_score()`, rule engine uses `parse_age_int()` — both in `ml/features.py`, never reimplement inline.
- `running_position`: two formats — space-separated `"6 5 6"` and concatenated `"665"`. Handle both (concatenated is only ~8 rows).
- `last_6`: **three** formats — slash `'5/5/1/2/1/1'` (live HKJC card), space `'4 11 9 5 0 8'` (Renavon, 95% of rows), concatenated `'13425'` (legacy). Parse ONLY via `analytics.parse_form_positions`; never inline. Splitting on `/` alone reads a two-digit placing as two positions — 60% of stored values contain one (bug 2026-09-06, `80fc093`).
- `horse_weight` in lbs (900-1350); `weight` is carried (113-133 lbs).
- `class_type`: 96%; `extract_class_rank()` handles G1/G2/G3/GRIFFIN/4YO; default class 3.
- `days_since_last_run`: 87.7% missing in legacy 5y — fall back to date math.

## HKJC parser gotchas

- **Brand suffix on horse names**: `parse_results` returns `'PODIUM(H173)'`; parser strips trailing `\s*\([^)]+\)\s*$` so downstream (esp. `prediction_log.fill_prediction_results` UPDATE-by-name) sees clean `'PODIUM'`. Apply same strip to any new HKJC name parser.
- **Multi-paren**: `'STAR (NZ)(H173)'` → `'STAR (NZ)'` (only trailing brand stripped).
- **Venue detection** (`HKJCApiClient.detect_venue`): wrong venue returns empty HTML. Wed = HV, weekends = ST.
- **HTML parser** expects 27-cell HKJC table rows with documented column indices; check `_parse_horse_table()` if horses come back empty.
- **Two-stage result publish**: HKJC posts the first four placings within a minute of the all-clear with win odds still 0.00, then the full field + dividends several minutes later. Do NOT ingest inside that window — `data.scraper.truncated_races()` detects it (HK race with <5 finishers or no odds on any row; venue=NULL overseas rows exempt) and forces a re-scrape. Before running `raceday finish`, confirm every race shows its full field (observed on 4 races on 2026/09/06).
- **Odds pages are dead — pool odds come from GraphQL** (found 2026-09-14): both `DisplayOdds.aspx` URLs return 404 / a maintenance page, so `fetch_odds_html` is None on every call. Win odds survive only via the Tips Index backfill (`_apply_tips_data`), which has no place column — `place_odds` was 0 on all 48,031 snapshots since May. Place odds now come from `HKJCApiClient.fetch_pool_odds_json` (`info.cld.hkjc.com/graphql/base/`) → `HKJCParser.parse_pool_odds`. The endpoint rejects any query it has not whitelisted (`WHITELIST_ERROR`): `_RACING_ODDS_QUERY` is bet.hkjc.com's own text, verbatim — never reformat it. Pools read `INITIAL`/`STOP_SELL` with empty `oddsNodes` until betting opens. The Playwright fallback still targets the dead `DisplayOdds.aspx`.
- `_safe_int()` preserves minus signs (regex `[^\d\-]`) — required for `rating_change` and `weight_change`.

## Codebase conventions

- Dates: `YYYY/MM/DD` in HKJC URLs/data, `YYYYMMDD` in filenames. SQLite `julianday()` requires `YYYY-MM-DD` — always `REPLACE(date,'/','-')`.
- Async/await in fetcher pipeline (`HKJCDataFetcher.get_race_card`).
- Model pickle dict keys: `model, place_model, calibrator, place_calibrator, version, n_features, n_samples, feature_names, feature_importance, trained_date`. Sidecar `.sha256` required.
- Dashboard probabilities: stored as decimals (0.178), displayed as % (×100 in JS).

## Frozen bet_ticket is the single source of truth (R6–R15 hardening, 2026-05-23 → 2026-05-25)

Every betting-critical surface reads from `race_bet_ticket` instead of re-deriving from rank order. This closed a class of phantom-hit / WA-fabrication / leg-drift bugs across the dashboard, WA copy, replay scoring, and `stake_ab`.

- **Visual ticket** — every block (PRIMARY / QPL BANKER / BOX QIN-4 PROTECTION / F4 SAVER / BOX TRIO-5 / F4 BOX-5 / BOX TRIO-6) reads legs via `_ticketLegsFor(label)` (resolves leg.number → horse, rank as fallback). Never `race.horses.slice(...)`.
- **WA copy** — two buttons: **Copy WA Ticket** (compact, ticket-first, race-day entry — STAKE + BET TICKET with HKJC pool names derived via `_hkjcPoolLabel(bet)`; or NO SYSTEM BET + OPTIONAL FORCE PICK top-3 + "Manual only — not counted as model ROI ticket." footer when ticket empty) and **Copy WA Full** (rationale + edge + tier-trust). Both routed through `copyWA()` which appends the `⚠️ For entertainment only. Gamble responsibly.` disclaimer.
- **Status guards** in compact WA: `STATUS: LIVE` / `LOCKED` (race resolved) / `STALE ODDS` (odds_age ≥ 15m).
- **Replay scoring** — `validate_hk --replay` mode scores actual staked bets from `race_bet_ticket` (use it for post-race recaps; synthetic strategy mode is for ROI comparison only).
- **No-odds leg guard** — `_ticket_add_bet` refuses any bet whose legs have `win_odds <= 0`.
- **Monitor parity** — `monitor/race_day.py` calls shared `finalize_tickets_and_stakes()` + `CombinedRaceAnalyzer(fail_closed_ml=True)`.
- **Contract tests** — `tests/test_race_day_contracts.py` pins 6 invariant groups (Ticket Parity, Results Monotonicity, ML Fail-Closed, Ranking + Odds Coherence, Betting Product Semantics, Analytics Parity). Race-day pre-check: `python3 -m pytest tests/test_dashboard_playwright.py tests/test_race_day_contracts.py tests/test_template_js_syntax.py tests/test_dashboard_e2e.py tests/test_template_empty_state.py -q`.

→ When re-rendering a past meeting offline, ALWAYS load `results/race_data_{YYYYMMDD}_with_results.json` (dict with `races:` key) — NOT the predictions-only file. The predictions JSON lacks post-race fields and the dashboard will show finished races as "next race" in Top Bets. See memory: `feedback_rerender_past_dashboards_from_with_results.md`.

## Feature experiment kill record (cheap-diagnostic line)

10 candidates killed since 2026-05-22 (1–9 at Gate A1/A2/B; #10 passed the diagnostic but died at the price/trained-A/B gate):
1. Sectional closing figure (diagnostic-killed, +0.33 corr with place_prob)
2. `jockey_ae` from API (lifetime leak inflated +0.033)
3. Pace/trip flags ×5 (base rate <0.15% or wrong-sign-redundant with field_rel_draw)
4. `closing_position_gain` (r=+0.73 with `late_kick` — same signal smoothed)
5. `jockey_book_quality` (wrong sign + r=+0.59 with implied_prob — market prices it)
6. Race-shape feature (r=−0.014 wrong sign in HK — N.American 8f-dirt heuristic)
7. Draw × style interaction (Gate A2: cell rate is 90% running_style)
8. `jt_market_sharpness` (leak-free self-computed J×T a/e: r=−0.040, wrong sign — measures longshot fluke not skill)
9. **Jockey ELO (3 designs)** — V1 place-anchored r=−0.14 (wrong sign); V2 raw r=+0.0986 was a +K·0.43/ride drift artifact; V2 Harville-fixed r=−0.158 (Purton at ELO −665). Market-anchored skill ratings in HK invert through favorite-longshot bias. See `memory/project_jockey_elo_v1_killed_v2_passed_2026-05-25.md`. Archived scripts: `scripts/diagnose_jockey_elo.py`, `diagnose_jockey_elo_v2.py`, `jockey_elo_shadow.py`. **Do NOT extend** — market-anchored jockey skill rating is structurally broken for HK.
10. **Barrier trials (full 4-gate investigation, 2026-06-07)** — the ONLY candidate to PASS the cheap diagnostic (first-up trial winners place 35.6% vs 16.9%, r=+0.193 — strongest in project history) yet still die downstream. Beats-price: +30.4% WIN ROI was a 3-longshot mirage (odds-capped −3% to −13%); no win edge (13.8% ≈ implied 14%). Re-rank null. Trained A/B on the finishing-position signal = multi-seed noise. **Decisive follow-up after indep review (`scripts/ab_trials_time_trained.py`):** the untested trial CLOCK (time-vs-par, first-up ≥90d, ST/HV-only, multi-seed, scored on 344 affected races) is ALSO noise (capture flips sign, place-Brier Δ−0.00026). Both placing AND clock null → trials public → market prices them. Infra KEPT (`barrier_trials` table + `scripts/scrape_trials.py`, accrues via `--since-last`). See `memory/project_barrier_trials_pipeline_2026-06-07.md`. Predicts ≠ tradeable — textbook efficient-market wall on genuinely-new data.

Next architectural swing per user roadmap (`memory/project_roadmap_jockey_elo_then_top4_ranker.md`): **top-4-set-aware ranker** as shadow second-stage. Single-feature pace/trip/style/jockey adds are pre-killed; don't propose without new data evidence.

## Renavon — frozen snapshot

`renavon_*` tables are a one-time historical backfill ending 2026/04/22 — will never grow. New race data / dividends / rankings come from the HKJC scraper. Do NOT design features assuming continuous `renavon_*` updates. → `docs/renavon.md` for table inventory, indexes, rebuild commands.

## Optional Multi-jurisdiction (UK / FR / AU / JP)

Non-HK ingestion clients are present but depend on **theracingapi.com** (HTTP
Basic Auth via local `.env`: `RACING_API_USERNAME` /
`RACING_API_PASSWORD`). The subscription is currently inactive, so these paths
are not part of the supported HKJC race-day workflow and API calls may return
authorization errors. HK cards/results are fetched directly from HKJC; do not
make HK predictions depend on RacingAPI credentials.

| Jurisdiction | DB                | Model              | Status                                      |
|---|---|---|---|
| HK   | `horse_history.db`| current artifact in `models/` | Primary supported workflow       |
| UK   | `uk_history.db`   | separate model      | Ingestion requires inactive RacingAPI subscription |
| FR   | `fr_history.db`   | separate model      | Ingestion requires inactive RacingAPI subscription |
| AU   | `au_history.db`   | none                | Historical/reference code; API-dependent    |
| JP   | `jp_history.db`   | none                | Historical/reference code; API-dependent    |

**Forward-validation decisions** (UK + FR, on `validate_*` disagreement edge):
- n < 200 → directional only
- n ≥ 200 AND ROI > 0 → confirm, revisit Phase 2B
- n ≥ 200 AND ROI ≤ 0 → kill that pipeline

**Rules**: (1) No data leakage — `groupby + shift(1)` patterns in `ml/*_features.py`. (2) MVP feature schema is frozen — adding more degraded ROI on both UK + FR (memory: `uk-feature-ceiling-2026-05-17`). (3) Phase 2B (place model, calibration, blend, recommender) deferred until disagreement edge confirmed. (4) API quirk: `get_results_for_range` clamps `end_date` to (today-1) silently.

**Credentials**: `.env` is gitignored. `config.py` reads via
`os.environ.get(..., "")`; never hardcode credentials. `.env.example` contains
empty optional values only. Keep all credentials out of source, test fixtures,
logs and generated artifacts.
