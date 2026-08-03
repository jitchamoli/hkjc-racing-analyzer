"""`compute_pure_probs` must use the configured rule/ML blend, not literals.

Bug (code review 2026-08-03): `betting/edge_calculator.py` hardcoded the blend

    raw = [0.6 * _rule_score_to_prob(...) + 0.4 * h['ml_prob'] for h in horses]

while its own comment claimed to "mirror ml/model.py:create_combined_prediction",
which reads `config.RULE_WEIGHT` / `config.ML_WEIGHT`. This is exactly the
inlined-duplicate pattern CLAUDE.md ML Rule 2 forbids.

No divergence today (config is 0.60/0.40), but pure_prob is the input to all
six EDGE gates and to the exotic overlay math, so the moment either weight is
changed — `scripts/shadow_generate.py` already patches `config.RULE_WEIGHT` to
0.9 for its variant — edge/exotic math would silently be measuring a different
model than `value_score` does.
"""
import pytest

import config
from betting.edge_calculator import compute_pure_probs


def _field():
    return [
        {"rule_score": 80.0, "ml_prob": 0.05, "win_odds": 3.0},
        {"rule_score": 50.0, "ml_prob": 0.40, "win_odds": 5.0},
        {"rule_score": 30.0, "ml_prob": 0.10, "win_odds": 12.0},
    ]


class TestPureProbUsesConfiguredWeights:
    def test_changing_the_configured_weights_changes_pure_probs(self, monkeypatch):
        """A rule-heavy config must favour the high-rule_score horse."""
        baseline = compute_pure_probs(_field(), field_size=12)

        monkeypatch.setattr(config, "RULE_WEIGHT", 0.9)
        monkeypatch.setattr(config, "ML_WEIGHT", 0.1)
        rule_heavy = compute_pure_probs(_field(), field_size=12)

        assert rule_heavy != pytest.approx(baseline), (
            "pure_prob ignored the configured blend — weights are hardcoded"
        )
        # Horse 0 has the best rule_score and the worst ml_prob, so weighting
        # the rule engine up must raise its share.
        assert rule_heavy[0] > baseline[0]

    def test_default_config_behaviour_is_unchanged(self):
        """Regression guard on BEHAVIOUR, not on the config values.

        This used to also assert `config.RULE_WEIGHT == 0.60` — pinning a
        tunable inside a test whose whole point is that the weights are read
        from config rather than hardcoded. Any legitimate blend change would
        have broken it for no reason; the assertions below already prove the
        property.
        """
        probs = compute_pure_probs(_field(), field_size=12)
        assert sum(probs) == pytest.approx(1.0)
        assert probs[0] > probs[1] > probs[2]
