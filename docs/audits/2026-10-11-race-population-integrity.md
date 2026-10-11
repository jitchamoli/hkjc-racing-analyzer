# Race Population Integrity Repair

## Confirmed Reproductions

- Seven date/race-number keys mix local races with overseas horse-history rows
  lacking venue and runner number. Ten such rows survived into the old five-year
  training population. On 2024/06/02 R1, ROMANTIC WARRIOR and VOYAGE BUBBLE's
  overseas results joined a complete twelve-runner Sha Tin Class 5 field.
- The old population included 220 groups with fewer rows than the highest
  finish position, and 19 groups without a winner. On 2025/09/10 HV R1, fourth
  and seventh place are absent from the database. Official HKJC results list
  WOODFIRE CHAMP fourth, independently confirming missing data:
  https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2025/09/10&Racecourse=HV&RaceNo=1
- Class-only grouping is NOT a reliable completeness diagnostic: a single race
  can store G1, Group 1 and Group One labels. Initial examples selected with
  LIKE '%Group%' understated some fields. The repair groups all class labels.

## Repair

Training and backtesting share drop_unusable_races. Non-HK rows are removed
before local race grouping or field statistics. Multiple local venues under
one date/race key, invalid or duplicate runner IDs, missing winners and gaps
in competition-rank finishing order are quarantined. Dead heats advance rank
by the number of finishers and are retained. Complete five/six-runner fields
and explicit non-finisher rows remain valid under the existing small-field
policy. Horse history is preserved in the database and in prior-form lookups.

No historical outcome, prediction, or frozen ticket was rewritten.

## Verification

- Reproducing regression tests failed before the fix and pass afterward.
- Five-year population: 48,416 rows / 3,998 races, previously 50,907 / 4,219.
- No missing runner IDs, mixed local venues or invalid finishing orders in the
  retained population, verified independently of the eligibility helper.
- Saved model population exactly matches backtest fold metadata; zero overlap
  between all four race folds. Model 20261011-205325; blend 20261011-205417;
  pure 20261011-205508. Both calibrators use 7,451 matching calibration rows.
- SQLite integrity passes; frozen-ticket SHA256 is unchanged:
  0dad721bfacd1fffae3e9bd69fce1c0a43a40519fa29647453bef56b36d98a05.
- Shadow diagnostic rerun on the corrected population: 869 test races;
  top-four-in-five capture 6.67% ranker versus 8.17% classifier. No promotion.
- Final full suite: 1,268 passed, 2 skipped. The previous pure-calibrator
  tail-clamp failure passes after refitting; the clamp itself was not changed.

## Remaining Limits

Quarantine is not a historical backfill. A completely missing trailing runner
cannot be identified from an otherwise contiguous finish sequence alone.
Authoritative declared-starter counts or full result-page reconciliation are
needed to certify every historical field. The existing four-or-fewer field
backstop is unchanged. Withdrawal refunds remain a separate replay issue.
