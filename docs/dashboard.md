# Dashboard and Ticket Contract

The current implementation is in `dashboard/generator.py`,
`dashboard/template.py` and `dashboard/results_fetcher.py`. This document
records the invariants that must stay aligned across the dashboard, WhatsApp
copy and replay.

## Frozen Ticket Is the Audit Record

`race_bet_ticket` stores each generated race ticket, including pool, label,
legs, combinations, unit stake, amount, paper status and reason. The HTML
dashboard, WhatsApp payloads, post-race badges and
`scripts.validate_hk --replay` must describe that same ticket. Do not rebuild
bet legs from rank slices in a consumer when a frozen ticket exists.

Before a race starts, a fresh generation may update its ticket. If the ticket
changes, the prior version is archived under `superseded_tickets`. Once the
scheduled off time is reached, the race's ticket is frozen; result fields also
provide a post-result guard. Do not run a prediction refresh after the race has
started expecting its ticket to change.

## Current Bet Surfaces

- **QPL BOX-3**: top three ranked horses, generated across unresolved races
  unless the bankroll guard halts betting.
- **F4**: `F4 SAVER` and `F4 BOX-5` are limited to EDGE/HIGH tier and other
  field/price gates. The saver is a single any-order First-4 combination;
  `F4 BOX-5` covers the five selected runners.
- **TRIO**: `BOX TRIO-5` is limited to EDGE/HIGH/MED with a sufficient field.
  `BOX TRIO-6` is marked paper-only and is not a payable system bet.
- Primary recommendations and QIN protection are generated from the server-side
  recommendation/ticket logic. An empty system ticket may still expose an
  optional force pick, clearly marked manual and outside system-ticket ROI.

The current predicates and exact labels live in `build_bet_ticket()`; update
this document when those product rules change. Ticket sizing, over-budget
disclosure and bankroll-halt behavior are also implemented there and covered by
`tests/test_bet_ticket.py` and `tests/test_dashboard_fail_closed.py`.

## Refresh and Results

`./raceday live DATE` refreshes predictions/odds and then merges results. Result
merging preserves previously saved actual results when a later fetch omits a
race, while fresh returned results may still correct prior values. The
post-off ticket freeze is distinct from this monotonic results merge.

The latest dashboard is `results/racing_dashboard.html`; dated dashboards and
JSON files under `results/` are generated artifacts used for audit and replay.
They are not documentation sources and may contain user betting strategy.

## Surface Parity Checks

When changing ticket generation or rendering, verify:

- Visual ticket legs, WhatsApp ticket copy and replay show identical payable
  bets and units.
- Paper tickets are visibly marked and excluded from payable exposure.
- A bankroll halt removes payable bets from the ticket, not just from a separate
  suggested-stake field.
- In the current configuration, a realized daily loss is a warning and does
  not block tickets (`BANKROLL_LOSS_BLOCKS_BETS=False`). Card exposure is
  disclosed with an over-cap banner for manual review; it does not silently
  delete recommendations.
- A started/resulted race cannot be mutated by refresh.
- `tests/test_bet_ticket.py`, `tests/test_race_day_contracts.py`,
  `tests/test_dashboard_fail_closed.py` and the dashboard rendering tests pass.
