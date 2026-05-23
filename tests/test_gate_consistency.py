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


@pytest.mark.parametrize('top1,tier,n,expected_skip', [
    # Normal fires
    (3.0, 'MED', 10, False),
    (5.0, 'HIGH', 12, False),
    (8.0, 'EDGE', 9, False),
    # Heavy fav
    (1.5, 'MED', 10, True),
    (2.4, 'HIGH', 10, True),
    # Low/SKIP conv
    (5.0, 'LOW', 10, True),
    (5.0, 'SKIP', 10, True),
    # Small field
    (5.0, 'MED', 7, True),
    (5.0, 'MED', 5, True),
    # Boundary: top1 exactly 2.5 is NOT skipped (gate is `< 2.5`)
    (2.5, 'MED', 10, False),
    # Boundary: field of exactly 8 is NOT skipped (gate is `< 8`)
    (5.0, 'MED', 8, False),
    # Multiple gate conditions simultaneously
    (1.8, 'LOW', 6, True),
])
def test_gate_decisions_agree_across_implementations(
    top1, tier, n, expected_skip
):
    """Both stake_ab.firing_bets and gate_impact._triple_gate_skip must
    agree on every gate decision. The expected_skip column is the
    ground-truth spec."""
    race = _race(top1, tier, n)

    # gate_impact returns (skip: bool, reason: str)
    gi_skip, _gi_reason = _triple_gate_skip(race)
    assert gi_skip is expected_skip, (
        f"gate_impact disagreed: top1={top1}, tier={tier}, n={n} → "
        f"got skip={gi_skip}, expected {expected_skip}"
    )

    # firing_bets includes BOX TRIO-5 iff gate does NOT skip
    fires = firing_bets(race)
    sa_skip = 'BOX TRIO-5' not in fires
    assert sa_skip is expected_skip, (
        f"stake_ab.firing_bets disagreed: top1={top1}, tier={tier}, n={n} → "
        f"BOX TRIO-5 in fires={('BOX TRIO-5' in fires)}, expected_skip={expected_skip}"
    )

    # Sanity: the two implementations agree with each other (transitivity
    # check — if both match expected, they must match each other, but
    # explicit assert documents the invariant).
    assert gi_skip == sa_skip, (
        f"stake_ab + gate_impact disagree on top1={top1}, tier={tier}, n={n}"
    )


def test_gate_uses_first_horse_for_top1_odds():
    """Both implementations must read horses[0].win_odds (rank-1 odds),
    not the minimum or any other horse."""
    # Build a race where horses[0] is the favorite (low odds) but later
    # horses have even-higher odds. Heavy fav gate should still fire on
    # horses[0]'s odds, not some other field.
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
    skip_sa = 'BOX TRIO-5' not in firing_bets(race)
    assert skip_gi is True
    assert skip_sa is True
