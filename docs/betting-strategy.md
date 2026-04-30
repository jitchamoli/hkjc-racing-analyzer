# Betting Strategy — Detailed Reference

Companion to `CLAUDE.md`. Holds Edge Math + Conviction Tier field definitions, the 6-gate EDGE filter, recommender logic, dynamic blend grid-sweep selection, version-by-version backtest history, edge correlation note, exotic pricing math, and reproduction commands. CLAUDE.md keeps the Probability Pipeline diagram and the "edge math uses pure_prob NOT calibrated_prob" rule.

## Edge Math + Conviction Tiers (v6.0)

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

## 6-gate EDGE filter

All must pass for ★ EDGE:

| Gate | Threshold | Reason |
|------|-----------|--------|
| G1 EV | `ev_win ≥ +0.05` | 5% expected ROI per $1 WIN stake |
| G2 Prob floor | `pure_prob ≥ 0.10` | No edge-flagging on noise-floor longshots |
| G3 Edge gap | `edge_pp ≥ 5.0` | Raw gap backstop |
| G4 Rank alignment | `model_rank ≤ 2` | Don't flag rank-5 longshot as edge |
| G5 Odds freshness | `odds_age_min < 30` | Stale odds invalidate EV |
| G6 Integrity | F-05 check passes | Model trained on ≥1000 real samples, <90 days old |

**Sample-size note**: at the current 124-race holdout, `Corr(edge_pp, actual win) = -0.21` over the 25 model-vs-market disagreement points — i.e. our disagreements at ≥5pp are NEGATIVELY predictive. CI is wide (n=25). Watch for ~50 disagreements before deciding whether to tighten G3 from +5pp to +8pp.

## Recommendation logic (`core/recommender.py`)

Unchanged from Apr 23, box-4 default retained:

- Default: `BOX Q4` — all top-4 model picks flagged `QIN` (6 combinations). Requires rank-3 AND rank-4 each have strong-history signals + top_4_share ≥ 0.38.
- Fallback: `BOX Q` (box-3, 3 combos) when rank-3 has strong history + top_3_share ≥ 0.30
- Fallback: `QIN` (straight, 1 combo) when top_2_share ≥ 0.25
- `TRIO` when top 3 are tightly bunched (gap_1_2 < 0.02 AND gap_2_3 < 0.02)
- `WIN` rare, only when rank-1 has BANKER signal AND market consensus (≥35% implied) AND clear top-2 gap (>0.08)
- `SKIP` if no live odds, field < 2, or no conviction gate met

WIN pool context (73-race holdout): model top-pick = market favourite 84.9% of the time. Flat-bet ROI -20.8% vs market-favourite baseline -61.2%. No WIN edge identified.

## Dynamic Market/Model Blend

Market blend weights adjust by race characteristics (HK bettors are sharp — larger/shorter races have sharper pools):
- Large fields (10+) + sprints (≤1200m): 70% market / 30% model
- Small fields (<10) or routes (≥1800m): 40% market / 60% model
- Default: 55% market / 45% model
- Quinella pool: lean 55-65% ML (less sharp money, public ignores consistent placers)

Grid-sweep chosen 2026-04-23: see `docs/audits/2026-04-23/raw/wave2-sweep-summary.json`.
Selection rule: lowest Brier on 73-race holdout (15 grid points, field_thresh fixed=10, shift=±15%), tie-break within ±0.005 Brier toward simpler schedule (base_model_w closest to 0.5). All 15 points within ±0.002 Brier — effectively at the noise floor.
- Winner: `base_model_w=0.45, field_thresh=10, dist_thresh=1200, Brier=0.1678`.
- Prior heuristic (`base=0.60, dist_thresh=1400`): Brier=0.1684.

## Backtest history

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

v6.5 + calibrator backtest (5,966 races): recommender quinella ROI **+168.71%**, Brier 0.2145.

## Backtest reproduction

```bash
python3 -m scripts.backtest              # Default
python3 -m scripts.backtest --sweep      # Grid over temperature × model_w
```

Notes:
- Races without live odds (pre-race or post-race) will show all SKIP — this is correct behaviour.
- `results_fetcher.py --auto-learn` scrapes live results from HKJC via `scrape_meeting_results()` and ingests into DB (no longer reads from `training_data/historical_races.json`). Retraining timeout is caught gracefully; run `python3 train_ml_model.py` separately if it times out.

## Exotic Bet Calculator (`betting/exotic_calculator.py`)

Prices all exotic bet combinations using conditional probability elimination:
- **Exacta**: P(A wins) × P(B wins | A removed)
- **Quinella**: Exacta(A,B) + Exacta(B,A), enhanced with place-model boost (consistent placers +15%), pace pairing boost (leader+closer +10%), draw clustering boost (adjacent good draws +8%)
- **Trifecta**: extends to 3-horse permutations (top 8 candidates)
- **First4**: extends to 4-horse permutations (top 6 candidates)
- **Value detection**: compares model probability vs market-implied, flags combinations with 10%+ edge
