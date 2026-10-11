# Shadow Research Commands

Calibration reporting runs as a non-required stage of `raceday finish`.
Refresh manually with:

```bash
python3 -m scripts.calibration_report
```

Outputs: `tracking_data/calibration_report.json` and
`tracking_data/calibration_records.json`. Band/tier groups with sparse samples
are marked insufficient evidence. Race log loss is unavailable for a rank-one
view because that view is not a complete probability distribution.
Missing timestamp provenance, post-off timestamps, invalid probabilities and
incomplete/withdrawn fields are explicitly excluded. No probability correction
or production threshold adjustment occurs.

Run the expensive historical research separately:

```bash
python3 -m scripts.top4_diagnostic --years-back 2 --folds 3
```

Output: `tracking_data/top4_diagnostic.json`. It compares fixed-parameter
XGBoost win classification with binary top-four relevance using `rank:ndcg`.
Whole meeting dates separate training, reserved validation and testing.
Validation is reserved but unused: the diagnostic uses a fixed iteration count,
not early stopping or hyperparameter search. Neither arm uses full-dataset
recency weights. Models stay in memory and cannot overwrite production files.

The main benchmark uses actual starters and fixed HK$10 top-five TRIO/F4 boxes.
It is a historical counterfactual, not executable-price proof. The separate
paired-frozen section preserves the production ticket's pool, combination
count, unit and paper status, replacing applicable rank-indexed TRIO/F4 legs.
Unsupported bets, unverifiable fields and absent prices are not manufactured.
Ranker scores are not win probabilities and must not drive staking or value.

The first run covered 868 races across 88 test meetings. Top-four-in-five
capture was 7.49% for ranking versus 7.60% for classification, with a paired
interval crossing zero. Eligible historical frozen-system tickets showed a
better point estimate but a wide interval crossing zero. This candidate is
NOT approved for production or automatic prospective deployment. Historical
source corrections and actual-starter assumptions remain research limitations.
Future pre-off shadow logging is a separate step if evidence warrants it.
