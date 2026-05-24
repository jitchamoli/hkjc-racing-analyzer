"""Locks the contract that core/analyzer.py passes ALL handicap fields to
ml_analyzer.predict() — `horse_weight`, `age`, `rating_change` were
silently missing from the dict 2026-05-25 (GPT-5.5 audit Finding #1),
which neutralized features 13 (weight_change), 14 (age_score), and 15
(rating_change) at inference even though training used real values.

Same train/predict parity rule as field_weights (DS-03 2026-04-23):
features must read identical inputs at both sites; any missing key
silently falls back to a default (usually 0) → the model learns a signal
that's never present on race day.
"""
from __future__ import annotations
from unittest.mock import patch

from core.analyzer import CombinedRaceAnalyzer, Horse, Race


def _mk_horse(num: int) -> Horse:
    return Horse(
        number=num, name=f'H{num}', draw=num,
        jockey='J Doe', trainer='T Smith',
        weight='125', rating=60.0,
        win_odds=5.0, place_odds=1.8,
        last_6='1-2-3-4-5-6',
        rating_change=3, horse_weight=1100, age='5',
        gear='', tips_odds=4.5, opening_odds=6.0,
        days_since_last_run=14,
    )


def _mk_race(num=1) -> Race:
    return Race(
        race_number=num, venue='Sha Tin', date='2026/05/25',
        time='13:00', distance='1200', class_type='Class 4',
        going='GOOD', course='Turf',
        horses=[_mk_horse(i) for i in range(1, 9)],
    )


def test_horse_data_includes_all_handicap_features_for_ml_inference():
    """The horse_data dict passed to ml_analyzer.predict() must include
    `horse_weight`, `age`, and `rating_change` — features 13/14/15 in
    ml/features.py read from these keys. Missing keys → 0 default →
    train/predict drift. Pin against silent regressions."""
    race = _mk_race()
    captured: dict = {}

    analyzer = CombinedRaceAnalyzer(fail_closed_ml=False)
    with patch.object(analyzer.ml_analyzer, 'predict') as mock_predict:
        from ml.model import MLPrediction

        def _capture(horse_data, *a, **kw):
            captured[horse_data['number']] = horse_data
            return MLPrediction(
                horse_number=horse_data['number'],
                horse_name=horse_data.get('name', ''),
                win_probability=0.1,
                place_probability=0.3,
                ml_confidence='medium',
                feature_importance={},
                model_version='test',
            )
        mock_predict.side_effect = _capture

        analyzer.analyze_race(race)

    assert captured, 'analyze_race must call ml_analyzer.predict at least once'
    for num, hd in captured.items():
        assert 'horse_weight' in hd, (
            f"horse_data for #{num} missing horse_weight — "
            f"feature 13 (weight_change) will read default 0 at inference"
        )
        assert 'age' in hd, (
            f"horse_data for #{num} missing age — "
            f"feature 14 (age_score) will read default 0 at inference"
        )
        assert 'rating_change' in hd, (
            f"horse_data for #{num} missing rating_change — "
            f"feature 15 will read default 0 at inference"
        )
        # Spot-check values flowed through (not just keys present).
        assert hd['horse_weight'] == 1100
        assert hd['age'] == '5'  # Horse.age is a str field; ml/features.py parses
        assert hd['rating_change'] == 3
