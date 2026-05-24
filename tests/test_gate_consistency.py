"""The BOX TRIO-5 F4-style triple gate (heavy fav / LOW or SKIP / small field)
is implemented in THREE places that must stay synchronized:

  1. dashboard/template.py:828, 1110, 2027     — the JS the user sees
  2. scripts/stake_ab.py:firing_bets()         — forward-validation log
  3. scripts/gate_impact.py:_triple_gate_skip  — OLD-vs-NEW diagnostic
  4. scripts/backtest.py:trio5_f4gate_skip     — burned-holdout A/B

A drift between any two would silently invalidate the forward-validation
comparisons (we'd measure a strategy that isn't what the user actually bets).
Test-engineer expert review 2026-05-23 priority #3.

These tests pin the gate definition by exercising the Python helpers on
the same synthetic race set and asserting they agree.
"""
import pytest

from scripts.stake_ab import firing_bets
from scripts.gate_impact import _triple_gate_skip


def _race(top1_odds: float, tier: str, n_horses: int):
    """Synthetic race with enough horses + bet recs to allow BOX TRIO-5 fire."""
    horses = [{'win_odds': top1_odds + i} for i in range(n_horses)]
    for h in horses[:4]:
        h['bet_recommendation'] = 'QIN'
    return {'horses': horses, 'conviction_tier': tier}


# 2026-05-24: production (stake_ab.firing_bets) UNGATED heavy-fav after user
# choice — F4 BOX-5 + BOX TRIO-5 now fire even when top1 < $2.50, as long as
# tier + field-size pass (see memory: heavy-fav-skip-ungated-2026-05-24).
# gate_impact._triple_gate_skip INTENTIONALLY retains the heavy-fav-gated
# semantics — its purpose is the A/B comparison between ungated production
# and the deprecated heavy-fav-gated alternative. So the two implementations
# legitimately DISAGREE on heavy-fav cases. This table documents both.
@pytest.mark.parametrize('top1,tier,n,gi_skip,sa_skip', [
    # Normal fires — both agree
    (3.0, 'MED', 10, False, False),
    (5.0, 'HIGH', 12, False, False),
    (8.0, 'EDGE', 9, False, False),
    # Heavy fav — gate_impact (deprecated) skips; production (ungated) fires
    (1.5, 'MED', 10, True, False),
    (2.4, 'HIGH', 10, True, False),
    # Low/SKIP conv — both skip on tier
    (5.0, 'LOW', 10, True, True),
    (5.0, 'SKIP', 10, True, True),
    # Small field — both skip on field
    (5.0, 'MED', 7, True, True),
    (5.0, 'MED', 5, True, True),
    # Boundary: top1 exactly 2.5 (gate is `< 2.5`) — both fire
    (2.5, 'MED', 10, False, False),
    # Boundary: field exactly 8 (gate is `< 8`) — both fire
    (5.0, 'MED', 8, False, False),
    # Multiple gate conditions — both skip on tier (heavy-fav clause moot)
    (1.8, 'LOW', 6, True, True),
])
def test_gate_decisions_match_documented_spec(top1, tier, n, gi_skip, sa_skip):
    """gate_impact stays on the deprecated heavy-fav-gated semantics by
    design (it's the A/B baseline). stake_ab matches production (ungated).
    """
    race = _race(top1, tier, n)

    actual_gi, _ = _triple_gate_skip(race)
    assert actual_gi is gi_skip, (
        f"gate_impact: top1={top1}, tier={tier}, n={n} → got skip={actual_gi}, "
        f"expected {gi_skip}"
    )

    fires = firing_bets(race)
    actual_sa = 'BOX TRIO-5' not in fires
    assert actual_sa is sa_skip, (
        f"stake_ab: top1={top1}, tier={tier}, n={n} → BOX TRIO-5 in fires="
        f"{('BOX TRIO-5' in fires)}, expected_skip={sa_skip}"
    )


def test_gate_uses_first_horse_for_top1_odds():
    """gate_impact must read horses[0].win_odds (rank-1 odds) for the
    heavy-fav clause, not the minimum or any other horse. stake_ab no
    longer cares about top1 odds after the 2026-05-24 ungate.
    """
    race = {
        'horses': [{'win_odds': 1.8, 'bet_recommendation': 'QIN'},
                   {'win_odds': 50.0, 'bet_recommendation': 'QIN'},
                   {'win_odds': 50.0, 'bet_recommendation': 'QIN'},
                   {'win_odds': 50.0, 'bet_recommendation': 'QIN'},
                   {'win_odds': 50.0},
                   {'win_odds': 50.0}, {'win_odds': 50.0}, {'win_odds': 50.0},
                   {'win_odds': 50.0}, {'win_odds': 50.0}],
        'conviction_tier': 'MED',
    }
    skip_gi, _ = _triple_gate_skip(race)
    assert skip_gi is True, "gate_impact must still skip on heavy-fav rank-1"
    # Production fires — tier + field pass; heavy-fav no longer matters.
    assert 'BOX TRIO-5' in firing_bets(race)
