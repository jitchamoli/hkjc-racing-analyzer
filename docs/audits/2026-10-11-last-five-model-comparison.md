# Last Five HK Races: Model Comparison

Retrospective comparison of October 11 R7-R11. No production tickets regenerated.

| Race | Old top five | Corrected top five | Winner rank old/new |
|---|---|---|---|
| R7 | [3, 6, 14, 5, 10] | [3, 6, 14, 5, 10] | [4] / [4] |
| R8 | [11, 1, 7, 3, 13] | [11, 1, 7, 3, 12] | [2] / [2] |
| R9 | [3, 10, 4, 8, 1] | [3, 10, 4, 1, 2] | [4] / [6] |
| R10 | [2, 10, 3, 7, 9] | [2, 10, 3, 7, 5] | [5] / [6] |
| R11 | [11, 6, 2, 3, 1] | [6, 11, 2, 3, 1] | [4] / [4] |

## Capture

| Metric | Old | Corrected |
|---|---:|---:|
| top_pick_wins | 0/5 | 0/5 |
| winner_in_top5 | 5/5 | 3/5 |
| top3_in_top5 | 3/5 | 2/5 |
| top4_in_top5 | 1/5 | 1/5 |

## Top Four Set Accuracy

Order-independent: overlap of the predicted four horses with the actual first four finishers. Not exact finishing-order accuracy. These five races have no top-four dead-heat ambiguity.

| Race | Actual top four (unordered) | Old matches | Corrected matches |
|---|---|---:|---:|
| R7 | [3, 5, 10, 13] | 2/4 | 2/4 |
| R8 | [1, 7, 9, 11] | 3/4 | 3/4 |
| R9 | [1, 3, 8, 9] | 2/4 | 2/4 |
| R10 | [1, 4, 7, 9] | 1/4 | 1/4 |
| R11 | [2, 3, 6, 11] | 4/4 | 4/4 |

old: 12/20 matched horses (60.0%); all four captured in 1/5 races.

corrected: 12/20 matched horses (60.0%); all four captured in 1/5 races.

## Fixed HK$10 Boxes

| Pool | Old stake / payout / ROI | Corrected stake / payout / ROI |
|---|---|---|
| TRIO | HK$500 / HK$684 / 36.8% | HK$500 / HK$317 / -36.6% |
| FIRST 4 | HK$250 / HK$105 / -58.0% | HK$250 / HK$105 / -58.0% |

## Limitations

- Five individual races from one meeting, not five meetings; insufficient to establish superiority.
- Both are retrospective artifacts fitted after racing; fold metadata places these races after training/validation/calibration.
- Eligible population changed fold boundaries, so training sample selection and calibration also changed; not a pure venue-only ablation.
- Frozen odds/rule scores and as-of reconstructed ML inputs; old arm does not exactly reproduce original pre-race predictions.
- Fixed-box hypothetical settlement, not recommender gate or frozen-ticket ROI; no WIN price backtest.
- Prior research exposure to test races is not ruled out by fold metadata.

Full model versions, fold ranges, per-race probabilities and payouts are in the companion JSON.
