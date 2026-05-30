# CLAUDE.md

Guidance for Claude Code working in this repo. Detail lives in `docs/`:
- `docs/ml-model.md` — feature inventory, removed-features changelog, training pipeline
- `docs/dashboard.md` — card-stack layout, BET TICKET prose, escape rules
- `docs/betting-strategy.md` — edge gates, recommender logic, exotic pricing
- `docs/renavon.md` — table inventory, backfill impact, rebuild

## Commands

```bash
# Race-day phase controller (PREFERRED — ./raceday is the friendly wrapper)
./raceday                       # interactive looping menu (pick date once, run phases)
./raceday start  2026/05/17     # AM: race_day (generate + shadow_generate + tier_b_shadow)
./raceday live   2026/05/17     # T-6 min before each race: generate → results_fetcher
./raceday finish 2026/05/17     # PM: post_race (10+1 steps: results, shadows, stake_ab,
                                #     tier_b, spoilers, longshot_trio, validate_hk,
                                #     track_edge, gate_impact, auto_dump)

# One-shots
python3 generate_predictions.py '2026/04/15'
python3 results_fetcher.py     '2026/04/15' [--auto-learn]
python3 train_ml_model.py
python3 scrape_history.py [--date 2026/04/09 | --rankings | --days 90]

# Forward-validation + trackers (auto-run by post_race)
python3 -m scripts.validate_hk [--report]      # OOS-only ROI — the ONLY honest measure
python3 -m scripts.shadow_evaluate [--report] [--variant jockey_form]
python3 -m scripts.stake_ab     [--report]     # flat-$10 vs ramp vs flat-1% vs invert
python3 -m scripts.tier_b_evaluate [--report]
python3 -m scripts.track_spoilers | track_edge | track_longshot_trio [--report]
python3 -m scripts.gate_impact [--last 12 | --all]   # OLD vs NEW BOX TRIO-5 triple gate
python3 -m scripts.clv_report                  # closing-line value on rank-1 picks
python3 -m scripts.clv_report --split-by-market-rank  # EDGE GAUGE (auto-runs in raceday finish):
                                               #   splits CLV by market agreement at OPEN — favourite
                                               #   late-money MIRAGE vs real sharpness. WATCH the rank-2-3
                                               #   (mild-disagreement) CLV + whether it converts; rank-4+
                                               #   longshot CLV% is inflated noise. Verdict: edge UNPROVEN
                                               #   until rank-2-3 CLV turns positive AND places over N>=10.

# Backtest
python3 -m scripts.backtest [--sweep | --exclude-date YYYY/MM/DD | --production-stakes]
python3 -m scripts.accuracy --all              # Brier, BSS, calibration

# Feature candidate diagnostics (cheap → re-rank → direct A/B → ship)
python3 -m scripts.diagnose_<feature>          # correlation Gates A1/A2/B on full DB
python3 -m scripts.test_<feature>_rerank       # signal-additivity on burned holdout
python3 -m scripts.compare_v6_x_v6_y           # direct trained A/B (ML head only)
python3 -m scripts.eval_swap_rule --rank-out 6 --odds-ratio 0.5

# DB recovery (binary too large for git)
python3 -m scripts.dump_db     # → tracking_data/backup/db_core_dump.sql.gz (~3 MB)
python3 -m scripts.restore_db  # replay SQL + reload Renavon CSVs; idempotent
python3 -m scripts.backfill_dividends [--last 20 | --force]

# Multi-jurisdiction (UK/FR via theracingapi.com; AU/JP reference-only)
python3 -m scripts.ingest_{uk,fr,au,jp} --since-last
python3 -m scripts.{train,backtest,validate}_{uk,fr} [--since-cutoff]

# HK editorial/news overlay (additive — HK pipeline still scrapes HKJC).
# theracingapi.com PRO tier carries HK racecards with comment/spotlight/medical/
# gear-first-time/trainer-14d. data/racing_api_hk_news.py normalises names to
# HKJC convention (UPPERCASE), distance to meters, time to HKT.
python3 -m scripts.hk_news 2026/05/24 [--picks]   # --picks = only MED/HIGH/EDGE
```

## Architecture

```
core/       — fetcher, analyzer, scorer, recommender, ML predictor   (HK)
ml/         — feature_schema (23 feat, v6.11), features, model,      (HK)
              training, integrity; + uk_features/model, fr_features/model
analytics/  — 14 specialized analyzers                                (HK)
data/       — database, odds_tracker, scraper, sectional_scraper,    (HK)
              uk_racing_api (services UK/FR/AU/JP),
              {uk,au,jp,fr}_database
monitor/    — race day monitor (HK)
dashboard/  — generator + edge-first HTML template (HK)
betting/    — edge_calculator, conviction tiers, alerts, exotic_calculator (HK)
scripts/    — CLI entry points
tests/      — pytest suite (694; +20 contract/Playwright tests in R11–R15 hardening)
```

Root-level shims (e.g., `hkjc_data_fetcher.py`) re-export from packages for backwards compat. Non-HK jurisdictions have isolated `*_history.db` + `*_racing_model.pkl` + scripts; HK code untouched.

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
- **ML model (40%)**: XGBoost win + place (`is_top3`) with 23 features (`ml/feature_schema.py`). IsotonicRegression recency-weighted; trained on ~50k real results. Post-blend isotonic recalibrator (`models/blend_calibrator.pkl`) refit after every retrain. 4-fold disjoint split: 0-60 train / 60-70 earlystop / 70-85 cal / 85-100 holdout.

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

5. **No rank promotions after calibration** (GPT-5.5 R13 F3 2026-05-25): `assign_bet_recommendations` MUST emit horses strictly sorted by `calibrated_prob` desc. Market-favourite and BANKER rank swaps were REMOVED — calibrated_prob already includes the market blend, so a post-calibration swap is a second override. BANKER is reasoning/diagnostic only; BANKER WIN may fire only when the BANKER horse is already rank 1 by calibrated_prob. Snapshot pre-blend order as `pre_blend_rank` for diagnostics. Locked by `tests/test_banker_cal_prob_guard.py` + `test_race_day_contracts::test_recommender_does_not_swap_rank_for_market_fav_or_banker`.

6. **ML fallback**: untrained `predict()` returns `1/field_size` (the `field_size` param, default 12) — NOT `1/len(features)`.

7. **Feature count guard**: pickle stores `n_features`; mismatch with current `NUM_FEATURES` = hard refusal.

8. **Model integrity**: `.sha256` sidecar required for loading. `train_ml_model.py` backs up old model and restores on failure.

## Probability Pipeline

```
rule_score (0-100) → sigmoid(0.06×(score-50)+logit(1/field_size)) → ×0.60
ml_prob (isotonic) → ×0.40
pure_prob = softmax(rule×0.6 + ml×0.4)         # independent opinion, pre-market

→ dynamic blend: model × (30-60%) + market × (40-70%)
  (base 45/55, sprint threshold 1200m, field threshold 10)
→ post-blend isotonic calibrator (blend_calibrator.pkl, lazy)
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

## 🚨 LEAK CORRECTION & HONEST POSTURE (2026-05-22, supersedes every ROI in this repo)

Three audit rounds found 10 date-leaking rule-engine DB lookups (no `before_date` filter — backtest read the scored race's own result via `get_rolling_stats`, `get_beaten_margins`, `get_horse_cd_history`, `get_jockey_stats`/`get_trainer_stats`, etc.). All 10 fixed (`before_date` threaded from `race.date` in `core/analyzer.py`). Live was unaffected.

**Fully leak-free 668-race holdout — every bet type NEGATIVE**: straight QIN −10.7%, gated QIN −27%, Box-4 −23%, TRIO-5 −45.4%, F4 BOX-5 −44.8%. **Earlier figures (incl. "+44% / +19%") were leak-inflated AND inflated 10× by a dividend-basis bug** (HKJC dividends are per HK$10, backtest scored vs $1 — fixed in `scripts/backtest.py`). **Treat every pre-2026-05-22 ROI as ~10× overstated AND leak-contaminated.**

**The PLACE model is unaffected (ML, not rule engine): BSS +0.10, ECE 0.0093** — model discriminates (rank-1 top-3 54.9% ≈ 2.2× random) but does NOT beat market + takeout on any bet type.

**HONEST POSTURE: no demonstrated betting edge. PAPER-ONLY until `scripts/validate_hk.py` shows positive lower-CI over N≥250 bets.** Judge changes on place/top-4 capture, NOT top-1 win Brier. Detail: `memory/project_7expert_panel_2026-05-22.md`.

**FAST EDGE GAUGE (the leading indicator while N accrues):** `scripts/clv_report --split-by-market-rank` (auto-runs in `raceday finish`). CLV converges far faster than ROI, but is read-with-care: the favourite bucket shortens structurally (late-money MIRAGE, not edge) and rank-4+ longshot CLV% is inflated noise. The ONLY monetizable signal is **rank-2-3 (mild-disagreement) CLV that also converts (places)**. As of 2026-05-30: rank-2-3 = −12.7% on n=9 → **edge UNPROVEN**. Don't touch the model; accrue to N≥10 meetings and watch that one line. See memory: `project_clv_edge_gauge_2026-05-30.md`.

## DB (`tracking_data/horse_history.db`)

Six tables: `race_results` (primary; sectional_times 99.4%), `race_dividends` (real HKJC payouts; backfill via `scripts.backfill_dividends`), `horses` (master; `get_horse_record()` aggregates from race_results), `jockey_stats` / `trainer_stats` (season; `rebuild_stats()` from local + optional `scrape_history.py --rankings` for full-season HKJC), **`race_bet_ticket`** (NEW 2026-05-24 — frozen per-meeting bet ticket: pool/label/legs/combos/amount/reason; ground truth for `validate_hk --replay`, dashboard STRIKE, WA copy, `stake_ab`; included in dump/restore).

**Data integrity**: `ingest_race_day()` calls `rebuild_stats()` which overwrites jockey/trainer stats with local-only data — re-run `--rankings` after ingest to restore. `horses` upsert derives counters from race_results via subquery (idempotent). After any DB rebuild from Renavon, re-run `scripts.backfill_from_renavon` to restore `rating_change` (lives in `renavon_race_catalog`, not `renavon_race_results`).

**Auto-dump**: launchd `com.hkjc.dbdump` runs `scripts/auto_dump.sh` Mon+Thu 02:00 HKT (after Sun ST + Wed HV). Logs to `logs/auto_dump.log`; commits + pushes only if dump bytes changed.

### DB data gaps (audited 2026-05-16)
- `rating`/`rating_change`: 100%. `last_6`: 95.5%. `gear`: 93%. `sectional_times`: 99.4%.
- `age`: **68.4%** (degrades back in time: 2021=20%, 2022=45%, 2023=79%, 2024=86%, 2025=95%, 2026=100%). Format gotcha: Renavon writes `'5.0'`, live race-card writes `'5'`. ML uses `parse_age_score()`, rule engine uses `parse_age_int()` — both in `ml/features.py`, never reimplement inline.
- `running_position`: two formats — space-separated `"6 5 6"` and concatenated `"665"`. Handle both.
- `horse_weight` in lbs (900-1350); `weight` is carried (113-133 lbs).
- `class_type`: 96%; `extract_class_rank()` handles G1/G2/G3/GRIFFIN/4YO; default class 3.
- `days_since_last_run`: 87.7% missing in legacy 5y — fall back to date math.

## HKJC parser gotchas

- **Brand suffix on horse names**: `parse_results` returns `'PODIUM(H173)'`; parser strips trailing `\s*\([^)]+\)\s*$` so downstream (esp. `prediction_log.fill_prediction_results` UPDATE-by-name) sees clean `'PODIUM'`. Apply same strip to any new HKJC name parser.
- **Multi-paren**: `'STAR (NZ)(H173)'` → `'STAR (NZ)'` (only trailing brand stripped).
- **Venue detection** (`HKJCApiClient.detect_venue`): wrong venue returns empty HTML. Wed = HV, weekends = ST.
- **HTML parser** expects 27-cell HKJC table rows with documented column indices; check `_parse_horse_table()` if horses come back empty.
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

9 candidates killed at Gate A1/A2/B since 2026-05-22:
1. Sectional closing figure (diagnostic-killed, +0.33 corr with place_prob)
2. `jockey_ae` from API (lifetime leak inflated +0.033)
3. Pace/trip flags ×5 (base rate <0.15% or wrong-sign-redundant with field_rel_draw)
4. `closing_position_gain` (r=+0.73 with `late_kick` — same signal smoothed)
5. `jockey_book_quality` (wrong sign + r=+0.59 with implied_prob — market prices it)
6. Race-shape feature (r=−0.014 wrong sign in HK — N.American 8f-dirt heuristic)
7. Draw × style interaction (Gate A2: cell rate is 90% running_style)
8. `jt_market_sharpness` (leak-free self-computed J×T a/e: r=−0.040, wrong sign — measures longshot fluke not skill)
9. **Jockey ELO (3 designs)** — V1 place-anchored r=−0.14 (wrong sign); V2 raw r=+0.0986 was a +K·0.43/ride drift artifact; V2 Harville-fixed r=−0.158 (Purton at ELO −665). Market-anchored skill ratings in HK invert through favorite-longshot bias. See `memory/project_jockey_elo_v1_killed_v2_passed_2026-05-25.md`. Archived scripts: `scripts/diagnose_jockey_elo.py`, `diagnose_jockey_elo_v2.py`, `jockey_elo_shadow.py`. **Do NOT extend** — market-anchored jockey skill rating is structurally broken for HK.

Next architectural swing per user roadmap (`memory/project_roadmap_jockey_elo_then_top4_ranker.md`): **top-4-set-aware ranker** as shadow second-stage. Single-feature pace/trip/style/jockey adds are pre-killed; don't propose without new data evidence.

## Renavon — frozen snapshot

`renavon_*` tables are a one-time historical backfill ending 2026/04/22 — will never grow. New race data / dividends / rankings come from the HKJC scraper. Do NOT design features assuming continuous `renavon_*` updates. → `docs/renavon.md` for table inventory, indexes, rebuild commands.

## Multi-jurisdiction (UK / FR / AU / JP)

All non-HK data flows from **theracingapi.com** (HTTP Basic Auth via `.env`: `RACING_API_USERNAME` / `RACING_API_PASSWORD`). One client at `data/uk_racing_api.py` services every jurisdiction; only region filter + DB destination differ.

| Jurisdiction | DB                | Model              | Status                                      |
|---|---|---|---|
| HK   | `horse_history.db`| v6.11 production    | Production                                  |
| UK   | `uk_history.db`   | 15-feat XGB         | MVP — forward-validating                    |
| FR   | `fr_history.db`   | 15-feat XGB         | MVP — beats market fav in backtest          |
| AU   | `au_history.db`   | none                | Reference only (sparse on standard tier)    |
| JP   | `jp_history.db`   | none                | Reference only (no OR ratings standard tier)|

**Forward-validation decisions** (UK + FR, on `validate_*` disagreement edge):
- n < 200 → directional only
- n ≥ 200 AND ROI > 0 → confirm, revisit Phase 2B
- n ≥ 200 AND ROI ≤ 0 → kill that pipeline

**Rules**: (1) No data leakage — `groupby + shift(1)` patterns in `ml/*_features.py`. (2) MVP feature schema is frozen — adding more degraded ROI on both UK + FR (memory: `uk-feature-ceiling-2026-05-17`). (3) Phase 2B (place model, calibration, blend, recommender) deferred until disagreement edge confirmed. (4) API quirk: `get_results_for_range` clamps `end_date` to (today-1) silently.

**Credentials**: `.env` gitignored. `config.py` reads via `os.environ.get(..., "")` — never hardcode. `.env.example` is the committed template. History audited zero-leakage.
