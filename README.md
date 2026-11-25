# HKJC Horse Racing Analyzer

An HKJC-focused racing analysis and race-day decision-support tool. It combines
historical form, race-card data, market odds, rule-based analysis and a
calibrated ML model. It generates a dashboard and an auditable, frozen bet
ticket for each race.

**This software has no demonstrated positive betting edge.** Treat its output as
analysis, not financial advice. Historical backtests are not proof of future
profitability; use forward validation and frozen-ticket replay when evaluating
changes. Gamble responsibly.

## Current Scope

- HKJC is the primary supported production workflow. Race cards, live odds and
  results are fetched from HKJC.
- The production feature schema is defined in `ml/feature_schema.py` (currently
  23 features). Model artifacts and checksums are under `models/`; the loader
  verifies integrity and refuses invalid artifacts.
- Post-calibration rank is ordered by `calibrated_prob`. Do not add market or
  BANKER rank promotions after calibration.
- `race_bet_ticket` is the audit record for generated recommendations. The
  dashboard, WhatsApp copy, replay scoring and result badges must agree with
  that ticket.
- User-configured ticket coverage currently includes top-three QIN PLACE boxes
  on unresolved races; F4 tickets are limited to EDGE/HIGH races, and the
  primary recommender may also add QIN, TRIO, WIN or PLACE products according
  to its gates. Inspect the frozen ticket rather than inferring stakes from
  labels or ranks.
- Realized daily losses currently generate a warning rather than blocking
  tickets (`BANKROLL_LOSS_BLOCKS_BETS=False`). Card exposure above the cap is
  surfaced for manual review; it does not automatically remove bets.
- UK/FR/AU/JP code remains in the repository, but data ingestion depends on a
  third-party RacingAPI account. RacingAPI credentials are optional and are not
  needed for the HKJC workflow. The prior subscription is inactive; API-backed
  commands may fail until a valid subscription is configured.

## Race-Day Workflow

Run commands from the repository root. `./raceday` opens an interactive menu;
one-shot commands default to the local system date, so pass the meeting date
explicitly when running for another day.

```bash
./raceday start 2026/11/25   # predictions and forward-shadow snapshots
./raceday live 2026/11/25    # refresh odds/predictions, then merge results
./raceday finish 2026/11/25  # post-race ingest, reports, trackers and DB dump
```

Run `live` again before the next race and after results become available. It
refreshes the pre-race ticket while the race has not started; started or
resulted races are protected from ticket mutation. `finish` should be run after
the meeting. Check its output for failed optional tasks as well as required
ingestion steps.

Generated dashboards and race JSON are written under `results/`. The current
dashboard is `results/racing_dashboard.html`; dated artifacts are also kept for
replay and audit.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
playwright install chromium
```

Some historical scraping and browser tests use Playwright/Chromium. The normal
HKJC race-day command also needs network access to HKJC. No `.env` file or
RacingAPI credential is required for the HKJC path. Optional RacingAPI
credentials, if used for non-HK experiments, belong only in an ignored local
`.env`; never commit them.

## Development Checks

```bash
python3 -m pytest -q
python3 -m pytest tests/test_dashboard_fail_closed.py tests/test_bet_ticket.py \
  tests/test_grouped_folds.py tests/test_validate_hk_exotic_div.py \
  tests/test_accuracy_exotic_hits.py -q
```

The full suite includes integration tests that read local database state. If a
database-backed test fails, report the exact test and determine whether the
failure is caused by code, local data, or unavailable external services; do not
silently skip it.

## Core Modules

| Path | Responsibility |
|---|---|
| `core/` | HKJC fetching, race analysis, probability calibration and recommendations |
| `ml/` | Feature schema, training, inference and model-integrity checks |
| `betting/` | Edge calculations, ticket sizing, pool rules and bankroll controls |
| `dashboard/` | Prediction generation, frozen ticket creation, HTML and result merging |
| `data/` | Persistence, scraping, odds snapshots and jurisdiction data clients |
| `scripts/` | Race-day phases, training, replay, validation and diagnostics |
| `tests/` | Unit, contract, integration and browser tests |
| `docs/` | Current model, dashboard, betting and data references; audit reports retain historical results |

## Trustworthy Evaluation

- Prefer `python3 -m scripts.validate_hk --report --replay` for forward
  validation and replay of frozen `race_bet_ticket` records.
- Replay scores what was recorded as recommended/staked; synthetic top-N
  diagnostics answer different questions and are not ticket ROI.
- Backtests using final odds are optimistic relative to the odds available at
  bet time. Use timestamped odds and CLV reports to assess price freshness.
- Require multiple forward meetings and sufficient sample size before changing
  production gates, blending or stakes. Keep candidate ranking objectives
  shadow-only until they improve walk-forward capture and frozen-ticket replay.

## Documentation

- [Model and training](docs/ml-model.md)
- [Dashboard and ticket contract](docs/dashboard.md)
- [Betting and validation policy](docs/betting-strategy.md)
- [Race-day wrapper](CLAUDE.md#commands)
- [Frozen Renavon dataset](docs/renavon.md)
- [Top-four shadow research](docs/top4-shadow-research.md)

Older dated audit/spec files are evidence of past investigations, not current
configuration. Check source code and the current model artifact before relying
on a dated metric or threshold.
