# HKJC Horse Racing Analyzer v6.11

Real-time horse racing analysis for the Hong Kong Jockey Club (HKJC). Combines a 25-factor rule-based engine with an XGBoost ML model (23 features, isotonic-calibrated, post-blend recalibrated, calibrator auto-refit on every retrain) to surface bets the public may be underestimating. The dashboard renders every race as a clickable card stack with two views — **Race Order** (all 11 chronological) and **Top Bets** (curated tier-sorted) — plus a sticky live alerts rail.

> **Multi-jurisdiction expansion (2026-05-17):** HK remains the primary, production pipeline. Four additional jurisdictions ship as **isolated, experimental pipelines** sharing only the API client — see [Multi-jurisdiction pipelines](#multi-jurisdiction-pipelines-uk--fr--au--jp) below.

## What's New (2026-05-22) — dividend-basis correction + place/top-4 reframe + expert-review fixes

> **🚨 LEAK CORRECTION (2026-05-22) — SUPERSEDES every ROI figure in this README and the docs.** A three-round expert audit found **10 date-leaking rule-engine DB lookups** with no `before_date` filter: in BACKTEST they read the scored race's own (and future) results. All 10 fixed (`before_date` + `race_date < ?`, threaded from `core/analyzer.py`); live was never affected (the race isn't in the DB at predict time). **Fully leak-free held-out figures (668 races) — EVERY bet type is NEGATIVE:** straight QIN **−10.7%**, gated QIN **−27.0%**, Box-4 QIN **−23.0%**, TRIO-5 **−45.4%**, F4 BOX-5 **−44.8%**. ~~Earlier "corrected" figures (straight QIN +44%, gated +19%, TRIO-5 +16%, TRIO-6 +29%, F4 BOX-5 +39%)~~ were themselves still leak-inflated. The 668-race holdout is also BURNED (dozens of prior comparisons), so even the negatives are noisy — but there is **ZERO evidence of a positive-EV bet**. The **PLACE MODEL is unaffected** (ML, not rule engine): BSS **+0.10**, ECE **0.0093** — it genuinely discriminates (rank-1 top-3 54.9% ≈ 2.2× random) but that skill does **not** beat the market + takeout on any bet type. **HONEST POSTURE: no demonstrated betting edge; treat the tool as a ranking/screening aid, PAPER-ONLY, until the forward-validation log (`scripts/validate_hk.py`) shows a positive lower-CI-bound over N≥250 bets.** Detail: `memory/project_7expert_panel_2026-05-22.md`.
>
> **⚠️ Separate ROI-basis bug (2026-05-22).** A backtest bug also scored HKJC dividends (declared **per HK$10**) against **$1** stakes — inflating every real-dividend ROI by **~10×** (`scripts/backtest.py` now `/10`). **Every ROI figure dated before 2026-05-22 in this README and the docs is ~10× overstated** — and, per the leak correction above, was measured on the leaking analyzer regardless.

**Evaluation reframe (the bigger change):** added a **place-model calibration** report (`print_place_calibration`) and made **top-4 / placing capture** the metric for judging model changes — *not* top-1 win Brier. This revealed our **place model is far better calibrated than our win model** (place ECE 0.0093 / BSS +0.100 vs win ECE 0.0527 / BSS +0.035). Since every production bet is a top-N set bet (QIN/TRIO/F4 boxes), the north star is now **top-4 ranking**, not winner prediction.

**At the top-4 ceiling (5 experiments, all tested, all null):** speed_fig revival, sectional closing figure, `is_top4` head, pace/race-shape feature, and place-ranked box membership were each tested cheaply (test-first) and **none improved top-4 capture** — confirming win-rank is the discriminator and the HKJC market already prices the rest. The lever is *not* more features.

**Fixes from a 10-expert review (all verified against source):** atomic write of the badge-restore JSON; blend-calibrator checksum hard-refusal; display-dividend unit consistency (was jumping 10× when real results landed); Kelly-branch case bug; QPL multi-combo parser; Playwright leak in ranking scrapers; monitor panel-wipe; post_race badge-wipe (step 5a → `fetch_results`); UK region lock; and **flattened the unjustified conviction stake-scaling** (`_confidence_scale` → no-op; no EV basis, N=3 origin). 434 tests pass (2 skip).

## What's New (2026-05-21) — multi-agent analyser audit + retrain

A multi-agent code audit of the race-analyser module surfaced three confirmed bugs (and dismissed several false positives after source verification):

1. **Rule-engine age factor dropped on `'5.0'`** — the age-vs-field factor was gated on `.isdigit()`, `False` for the DB/Renavon `'5.0'` format, so it was silently dropped for every DB-built race (backtest, history) while live (`'5'`) worked — a backtest≠live divergence. Added `parse_age_int` as the single source of truth for the rule engine (mirrors `parse_age_score`).
2. **Exotic value detection used `calibrated_prob`** — violated the "edge math uses `pure_prob` (pre-blend)" invariant, mechanically shrinking detected exotic edge. One-line fix in `dashboard/generator.py`.
3. **Training `horse_history` was recency-windowed** — horses racing early in the training window saw only within-window priors while inference queries the full DB (train/predict mismatch at the window edge). Now built from the full DB; labels stay cutoff-filtered. Mirrors the earlier Task #51 jockey/trainer fix.

**Model retrain (#3):** on the 668-race real-dividend holdout, calibration improved materially (Brier 0.2067→0.2030, BSS +0.0148→+0.0352, ECE 0.0935→0.0527), hit rate 29.3%→31.6%, ROI within noise (+1114%→+1069%, inside bootstrap CI [+889%, +1258%]). Still v6.8 schema (24 features).

**Race-day workflow:** `./raceday` friendly wrapper with a looping interactive menu (pick the date once, run `start`/`live`/`finish` repeatedly, no python invocation). **Measurement rigor:** re-baselined backtest on the real 668-race holdout, block-bootstrap CIs (`scripts/bootstrap_ci.py`), and a real closing-line-value report (`scripts/clv_report.py`). 433 tests pass (2 skip).

## What's New in v6.8 (2026-05-16) — health-check audit

A general health check on the locked v6.8 engine surfaced **4 production bugs** that had been silently degrading the pipeline. All shipped in PR #26.

**1. `horse_age` was secretly dead** (train/predict mismatch — ML rules #2 + #3 in one bug)
- The inline `int(str(age)[:2])` parser failed on the DB's `'5.0'` format (`int('5.')` raises `ValueError`) → 94% of training rows fell back to 0.5
- Live race-card scrapes returned `'5'` and parsed correctly → real train/predict mismatch hiding as a dead feature
- Extracted `parse_age_score()` in `ml/features.py` as single source of truth; both training and inference call it
- **Result:** training accuracy 69.7% → **71.1%**; `horse_age` importance 0.000 → **0.040 (#5 feature)**

**2. `prediction_log` results never filled** (silent UPDATE-zero-rows)
- `parse_results` returned `'PODIUM(H173)'` (HKJC collapses brand suffix); `prediction_log.horse_name` stores `'PODIUM'`
- 643 rows across 4 race dates had `results_filled = 0` while `race_results` filled normally (auto_learn uses a different scraper)
- Fix: strip `\s*\([^)]+\)\s*$` brand suffix at parse time. Backfilled 480 historical rows.

**3. Blend calibrator was fit BEFORE model retrain**
- Calibrator's isotonic targeted the prior model's distribution → residual under-confidence in backtests
- `auto_learn` now invokes `scripts.calibrate_blend` after retrain, gated on `retrain_ok` so a failed retrain doesn't poison the calibrator with a stale model

**4. `MIN_TOP4_CONVICTION` re-tuned 0.48 → 0.52**
- New model produces more confident top-4 distributions (top_4_share ≥ 0.45 in 96% of betted races)
- Both-windows sweep: full 666-race holdout +32pp ROI / -1 hit; last-6 meetings +63pp ROI / 0 hits lost
- ROI lift comes from stake reduction (24 races demoted from 6-combo BOX Q4 to 3-combo BOX Q), not hit-rate change

**Bonus fix from probability-expert audit:** `parse_age_score(float('inf'))` raised `OverflowError` past the `(ValueError, TypeError)` except tuple. Added `OverflowError` to the catch + 3 test cases.

### v6.8 backtest deltas (full 666-race holdout)

| Metric | Before | After | Δ |
|---|---|---|---|
| Brier (top-pick) | 0.2056 | **0.1976** | −0.008 |
| Calibration [25-100%] gap | +8.1pp | **+3.2pp** | tighter |
| Quinella straight ROI | +1006% | **+1340%** | +334pp |
| Recommender-gated ROI | +1481% | **+1426%** | within noise |
| `horse_age` importance | 0.000 | **0.040 (#5)** | feature alive |

### v6.8 forward validation (last 8 meetings — 80 races)

- Recommender: **26 hits (36.6%), +2117% ROI**, +$5,590 P&L on $264 stake
- Top-pick win rate: **25.0%** — exactly tied with market favourite
- Brier 0.168 — best calibration on any window measured

## What's New in v6.6 (2026-05-09)

**Rule-engine market-leak purge** — three sites that re-injected market odds into the rule score (creating a triple-count with the dynamic blend) were removed: the `field_size` factor, the no-form fallback, and BANKER signal #9. All replaced with structural-only logic.

**Critical speed-figures fix** — `analytics/speed_figures.py` was wired into the rule engine but had three layered bugs that silently disabled it:
1. `race.race_date` AttributeError (correct field is `race.date`) — caught by silent try/except, factor never fired
2. `par_times` cache wasn't keyed by date — leaked future data when the analyzer was reused across dates (caught by `test_backtest_production_parity`)
3. `parse_finish_time` only handled `M:SS.ss` format but 85% of DB rows use `M.SS.ss` — 85% of rows returned None, feature went neutral, ML pruned it

After all three fixes, the rule-engine `speed_fig` factor now fires for ~46% of horses with a real speed-vs-par signal.

**Two new ML features (19 → 21)**
- **`jt_pair_win_rate`** (Feature 20, **4.1% importance**) — Bayesian-smoothed jockey×trainer pairing strike rate. 305 pairs with ≥50 joint runs in DB; top pairs hit 22-28% wins vs 8.3% baseline.
- **`speed_fig_norm`** (Feature 21, 0% importance — DEAD as ML, useful as rule factor) — par-time-normalised speed figure. XGB found it redundant with `last_start_margin` (16.3%, top feature). Kept in schema but contributes only via the rule engine's 5%-weight `speed_fig` factor.

**ML inference leakage fix** — `_compute_jockey_win_rate` and `_compute_trainer_win_rate` had a fallback that read from `jockey_stats` / `trainer_stats` tables. These tables reflect current full-season state, so historical scoring (with `before_date` set) leaked future data. Fallback now skipped when date-filtering is active.

**Stats-table current-season filter** — `rebuild_stats()` no longer prefers the frozen Renavon snapshot (Renavon stops at 2026/04/22). It computes from `race_results` filtered to the current HK season (`race_date >= YYYY/07/01`) so visiting/retired jockeys from prior seasons don't leak into top-jockey leaderboards with stale lifetime totals.

**Self-calibrating gear scores** — `scripts/calibrate_gear.py` runs after each meeting in auto-learn. Computes per-gear win rate vs field baseline, applies shrinkage toward neutral (Bayesian smoothing), maps to score [20, 80]. Discovered most gear codes are essentially noise in HK racing (±1pp lift) — only Blinkers OFF (BO) shows real lift (+5.2pp).

**Monitor auto-merges results** — every pre-race window now also runs `results_fetcher.py` so STRIKE/MISS badges on past races stay current on the live dashboard automatically.

**Past-6-meeting smoke test (60 races)**: top-pick wins 13/60 (21.7%) → **17/60 (28.3%)** (+6.6pp), top-3-in-top-5 122/180 (67.8%) → **133/180 (73.9%)** (+6.1pp).

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
- **BET TICKET** with SEVEN layers per race:
  1. 🟡 **PRIMARY** — recommender's choice (QIN / TRIO / BOX QIN-4 / BANKER WIN), each pick tagged with jockey + 30-day strike rate (gold) + all-time strike rate.
  2. 🟢 **+ PROTECTION · BOX QIN-4 (6 combos)** — always-on top-4 quinella box. Catches placing-pair upsets where the winner sits in model ranks 3-4.
  3. 🟣 **+ F4 SAVER · 1 unit** — top-4 in exact 1-2-3-4 finishing order. Gated to skip heavy favs (<$2.5), LOW conviction, or fields <8 — cheap longshot saver only when the model has real conviction.
  4. 🔵 **+ QIN PLACE BANKER · 3 combos** — only when top pick odds ≤ 1.8. For short-favorite races a WIN bet pays barely above stake; QPL on (favorite + 3 banker picks) hits when fav places (very likely) AND any banker also places.
  5. 💜 **★ MAIN · F4 BOX-5 · 5 combos** — top-5 picks in a 5-combo F4 box (any order, top-4 finishers). Shipped 2026-05-06 after 60-race backtest: **+54% ROI vs TRIO-5's +8% at half the stake**. Same gates as F4 SAVER. Median F4 dividend $1,227 — per-hit return ~9× stake.
  6. 🩷 **+ BOX TRIO-5 · 10 combos** — top-5 model picks in a TRIO box. Hits when any 3 of 5 finish top-3 in any order. Catches rank-5/6 spoilers that QIN-4 misses. Kept alongside F4 BOX-5 because TRIO-5 still hits when the 4th finisher is outside top-5 (different miss profile).
  7. 🌹 **+ BOX TRIO-6 · 20 combos (OPTIONAL)** — top-6 in TRIO box with **CHAOS 0-4 score** + per-race TAKE/SKIP/YOUR CALL recommendation. Spoiler-swap detection: a horse ranked 7-12 with gear visible (B/TT/CP) AND market odds ≥20% shorter than rank-6 auto-swaps in with a 🔁 SWAP badge.
- **Top Bets · Next / Past sub-toggle** — Next surfaces upcoming bets, Past shows finished races with outcome badges (🟡 `★ STRIKE · WIN`, 🟢 `★ STRIKE · QIN`, 🟢 `★ STRIKE · BOX-4`, 💜 `★ STRIKE · F4-5`, 🩷 `★ STRIKE · TRIO-5`, 🌹 `★ STRIKE · TRIO-6`, 🔴 `✗ MISS`). Auto-flips to Past at end-of-meeting so a click after the last race lands on the day's record. Badges render in both views.
- **WhatsApp Copy** mirrors the BET TICKET — first line is a scannable summary like `BET: BOX QIN-4 → 1, 9, 2, 13` for terminal entry; followed by parallel MAIN blocks (`F4 BOX-5`, `BOX TRIO-5`), `F4 SAVER (exact)`, full per-horse breakdown with jockey strike rates, then "Why strong" bullets.
- **Race-day workflow** — two persistent monitors keep the dashboard fresh: pre-race refresh + results merge at **T-6 min** before each race, and a 15-min `--auto-learn` poll that ingests each new race result into `tracking_data/horse_history.db` so the model improves day-over-day.
- **Same-format history per horse** — for every horse with prior runs at this exact venue + distance, the expanded panel lists their last 5 attempts (date, going, class, finish position colour-coded, odds, margin, jockey).
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

## Multi-jurisdiction pipelines (UK / FR / AU / JP)

Shipped 2026-05-17 as completely isolated pipelines — separate DBs, separate models, separate scripts. **HK production code is untouched.** All non-HK data flows from [theracingapi.com](https://www.theracingapi.com) (HTTP Basic Auth via `.env`).

| Jurisdiction | Status | 12-month data | Model? |
|---|---|---|---|
| 🇭🇰 **HK** | Production (this README's primary subject) | 55,411 race results | v6.11 — 23 features, isotonic-calibrated |
| 🇬🇧 **UK** | MVP — forward-validating | 81,975 runners | 15-feature XGBoost (AUC tied with market) |
| 🇫🇷 **FR** | MVP — forward-validating | 21,267 runners | 15-feature XGBoost, **beats market favourite +4.64% ROI on test fold** |
| 🇦🇺 **AU** | Reference data only | 4,444 runners | none — stakes-only at this plan tier |
| 🇯🇵 **JP** | Reference data only | 2,069 runners | none — no OR ratings at this plan tier |

### Weekly forward-validation workflow (UK + FR)

```bash
# After UK weekend racing
python3 -m scripts.ingest_uk --since-last
python3 -m scripts.validate_uk --since-cutoff

# After FR weekend racing
python3 -m scripts.ingest_fr --since-last
python3 -m scripts.validate_fr --since-cutoff
```

`validate_*` reports flat WIN ROI for the model top-1, market favourite, and the **model-vs-market disagreement edge** (the strategy we're forward-validating). Decision rule: confirm at n≥200 disagreements before building further model layers; if ROI stays negative at n≥200, kill that pipeline.

### Setup (one-time)

```bash
cp .env.example .env                 # then paste API credentials
# .env is gitignored — credentials never enter git history
```

→ Full multi-jurisdiction architecture, design rules, and decision gates: `CLAUDE.md` § "Multi-jurisdiction pipelines"

## How It Works

### Data Sources (from racing.hkjc.com)

| Source | What It Provides |
|--------|-----------------|
| Race Card page | Horse entries, jockeys, trainers, draw, weight, gear, ratings, last 6 runs |
| Tips Index page | Live odds, Initial Tips Index, Race Day Tips Index, top picks per race |
| Horse Profile page | Running positions, sectional times, class history, C&D record |
| Results page | Finishing positions, margins, odds, real dividend payouts (WIN/PLACE/QIN/QPL/FORECAST/TRIO/TIERCE/FIRST 4/QUARTET) |
| Horse Database (local) | ~50,752 training rows (5y window, grows ~146/meeting), ~668-race real-dividend holdout (100% QIN/TRIO/First-4 coverage), jockey/trainer season stats sourced from Renavon rankings |

### Prediction Pipeline

```
rule_score (0-100)  → sigmoid → ×0.60
ml_prob (isotonic-calibrated)  → ×0.40
combined_prob = rule + ml (uncalibrated, pre-blend)
→ softmax (temp=1.5, mild normalization)
→ pure_prob (our independent opinion)
→ dynamic blend: model × (30-60%) + market × (40-70%)
→ post-blend isotonic calibrator (models/blend_calibrator.pkl, lazy-loaded; refit after every retrain)
→ normalize to sum = 1.0
→ calibrated_prob, expected_return = odds × prob - 1

Edge math (additive, pre-blend pure_prob):
edge_pp = (pure - market) × 100
ev_win  = pure × win_odds - 1
overlay = pure × win_odds
6 gates → edge_fires bool → conviction_tier
```

The 60/40 rule+ML blend forms the model's "independent opinion" (`pure_prob`); the market is then mixed in. Edge math compares `pure_prob` vs `market_prob` BEFORE the market blend so the gap is meaningful.

### Bet Recommendation Logic (`core/recommender.py`)

- **★ EDGE → WIN bet** when all 6 gates pass (rare — ~1-2 races per card on average)
- **BOX Q4** when `top_4_share ≥ 0.58` AND rank-3/4 have strong history
- **BOX Q3** fallback when `top_3_share ≥ 0.33`
- **straight QIN** when `top_2_share ≥ 0.30`
- **TRIO** when top 3 are tightly bunched (gap_1_2 < 0.02 AND gap_2_3 < 0.02)
- **PLACE** when top pick has strong place model probability
- **SKIP** when no live odds, field < 2, or no conviction gate met

### Conviction Tiers (replaces strategy_confidence)

| Tier | Condition | Treatment |
|------|-----------|-----------|
| **★ EDGE** | All 6 EDGE gates pass | Single-horse WIN, hero placement |
| **HIGH** | `top_2_share ≥ 0.40` OR `top_3_share ≥ 0.55` | QIN/BOX-3/BOX-4 hero |
| **MED** | `top_2_share ≥ 0.30` OR `top_3_share ≥ 0.42` | Hero, smaller stake |
| **LOW** | `top_2_share ≥ 0.22` | Strip only, not hero |
| **SKIP** | else | Strip greyed, no recommendation |

EDGE takes precedence over pair-conviction tiers.

## Project Structure

```
hkjc_racing_analyzer/
  core/                      # Core modules
    fetcher.py               #   Real-time data layer (HKJC API + parser)
    analyzer.py              #   Main analyzer — 25-factor engine + ML blend
    scorer.py                #   Score combination + probability calibration
    recommender.py           #   Bet recommendation engine
    predictor.py             #   ML predictor shim
  ml/                        # ML pipeline
    feature_schema.py        #   23-feature schema (v6.11 — pruned draw_advantage / speed_fig_norm / running_style)
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
  analytics/                 # 14 specialized analyzers (draw bias, pace, sectionals,
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
    race_day.py · post_race.py             # one-command race-day wrappers
    shadow_generate.py · shadow_evaluate.py # active rule_w=0.9 forward-validation
  tests/                     # pytest suite — 435 tests (433 pass + 2 skip)
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
# 435 tests total (433 pass + 2 skip):
#   - 9 accuracy regressions (race-level QIN/TRIO hit detection)
#   - 24 FR/AU/JP DB tests (8 each, mirror UK pattern)
#   - 11 backfill script tests (backfill_dividends + backfill_from_renavon)
# Coverage: edge math, gates, tier assignment, alerts dedup, Wilson CI,
# template rendering, JS-syntax validation, end-to-end integration,
# age parser contract (parse_age_score + parse_age_int, incl. inf/nan),
# HKJC results brand-strip, all 4 jurisdiction DB schemas + idempotency.
```

## Performance (668-race holdout, 100% real HKJC dividends)

> **🚨 The ROI table below is SUPERSEDED by the leak correction at the top of this README — it was measured on the leaking rule engine and is NOT valid. Kept only as an audit trail of what the leak inflated.** On the fully leak-free analyzer, EVERY bet type is NEGATIVE (straight QIN −10.7%, gated QIN −27.0%, Box-4 QIN −23.0%, TRIO-5 −45.4%, F4 BOX-5 −44.8%). There is no demonstrated positive-EV bet. The honest evidence path is the forward-validation log (`scripts/validate_hk.py --report`), not this burned holdout.

The 85-100% chronological holdout is ~668 races with real QIN/TRIO/First-4 dividends. The figures below were the pre-leak-correction estimates (leak-inflated AND ~10× overstated by the dividend-basis bug) — retained for the record only:

| Strategy | ROI (point) — SUPERSEDED, leaking | by-date 95% CI |
|----------|-------------|----------------|
| Recommender-gated | ~~+1069%~~ | ~~[+889%, +1258%]~~ |
| Straight quinella | ~~+1343%~~ | ~~[+960%, +1761%]~~ |
| Box-4 quinella | ~~+944%~~ | ~~[+795%, +1103%]~~ |

### Calibration

- **Brier score (top-pick): ~0.203** on the current 668-race real-dividend holdout (BSS +0.035, ECE 0.053 after the 2026-05-21 retrain)
- **24 features**; `last_start_margin` leads (~15%), all but `speed_fig_norm` carry non-zero importance (`speed_fig` contributes via the rule engine)

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
- ~~Calibrator validation slice spans 2024-04 → 2025-04 (1y stale on field strength)~~ — **fixed in v6.8**: 4-fold split (calib fold = rows[70:85%]) and `auto_learn` re-fits the calibrator after every retrain (gated on `retrain_ok`).
- Distance imbalance (1200m has 19,555 rows vs 14 at 1050m) — open
- `Corr(edge_pp, actual win)` is consistently mildly negative (~−0.30 across both holdout windows and both v6.7/v6.8 models). Structural property of rule engine + market interaction. Recommender works around it by using BOX bets, not WIN. Open.
- Calibrator's thin upper plateau (cal_prob ≥ 0.50 → pinned at 0.80, ~72 supporting samples). **Zero production blast radius** — across 643 logged predictions in last 5 meetings, no horse landed above 0.50, so the plateau is never exercised. Documented as known limitation; fix only when a horse actually trips it.

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
