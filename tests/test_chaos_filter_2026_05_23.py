"""CHAOS no-bet filter — shipped 2026-05-23 from 15-expert audit +
cheap diagnostic (posterior 0.996, Δ −45.6pp on 10-meeting holdout).

Only soft-going trigger is active; the other two HK-pro-suggested triggers
(≥4 horses 60+d break, Class 5 + 10+ field + compact market) are TODO
comments pending forward evidence. Tests pin the active trigger AND the
intentional inactivity of the other two so we don't accidentally turn them
on without evidence.
"""
import pytest

from betting.conviction import detect_chaos
from scripts.stake_ab import firing_bets


# ---------------------------------------------------------------------------
# detect_chaos — soft-going trigger (ACTIVE)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('going', [
    'YIELDING', 'GOOD TO YIELDING', 'SOFT', 'HEAVY', 'WET-SLOW', 'WET SLOW',
    'yielding',  # case-insensitive
])
def test_detect_chaos_fires_on_soft_going(going):
    is_chaos, reason = detect_chaos(going=going)
    assert is_chaos is True
    assert reason == 'soft_going'


@pytest.mark.parametrize('going', [
    'GOOD', 'GOOD TO FIRM', 'FIRM', 'FAST', '', None,
])
def test_detect_chaos_does_not_fire_on_firm_or_missing(going):
    is_chaos, reason = detect_chaos(going=going)
    assert is_chaos is False
    assert reason == ''


# ---------------------------------------------------------------------------
# Other two triggers are INACTIVE (TODO) — pin until evidence accumulates
# ---------------------------------------------------------------------------

def test_detect_chaos_long_break_trigger_inactive():
    """The 4+ horses 60+d break trigger is TODO; fired 0× in diagnostic.
    Must NOT fire until forward evidence flips this expectation."""
    horses = [{'days_since_last_run': 90} for _ in range(6)]
    is_chaos, _ = detect_chaos(going='GOOD', horses=horses)
    assert is_chaos is False, (
        "Long-break trigger fired without forward evidence — see "
        "betting/conviction.py TODO comment"
    )


def test_detect_chaos_c5_compact_market_trigger_inactive():
    """Class 5 + 10-horse field + top-3 within $1 — should NOT fire (TODO)."""
    horses = [{'win_odds': 4.5, 'days_since_last_run': 14},
              {'win_odds': 5.0, 'days_since_last_run': 14},
              {'win_odds': 5.4, 'days_since_last_run': 14}] + \
             [{'win_odds': 20.0, 'days_since_last_run': 14}] * 7
    is_chaos, _ = detect_chaos(going='GOOD', horses=horses,
                               class_type='Class 5')
    assert is_chaos is False, (
        "Class-5-compact-market trigger fired without forward evidence"
    )


# ---------------------------------------------------------------------------
# Integration: firing_bets retrospectively suppresses bets on CHAOS races
# ---------------------------------------------------------------------------

def _race(going: str, class_type: str = 'Class 3', n_horses: int = 10,
          top1_odds: float = 3.0, bet_rec: str = 'QIN'):
    horses = [{'win_odds': top1_odds + i,
               'bet_recommendation': bet_rec if i < 4 else None}
              for i in range(n_horses)]
    return {'going': going, 'class_type': class_type,
            'horses': horses, 'conviction_tier': 'HIGH'}


def test_firing_bets_suppresses_all_bets_on_soft_going():
    """Soft-going race → firing_bets returns {} regardless of bet_recommendation
    or conviction_tier — CHAOS overrides everything."""
    race = _race(going='YIELDING')
    fires = firing_bets(race)
    assert fires == {}, f"Expected no bets on CHAOS race; got {fires}"


def test_firing_bets_does_not_suppress_on_firm_going():
    """Sanity: firm going with same bet_recommendation pattern should fire bets."""
    race = _race(going='GOOD')
    fires = firing_bets(race)
    assert len(fires) > 0, "Expected bets to fire on non-CHAOS race"


@pytest.mark.parametrize('going', ['GOOD TO YIELDING', 'SOFT', 'HEAVY'])
def test_firing_bets_suppresses_all_soft_going_variants(going):
    race = _race(going=going)
    assert firing_bets(race) == {}
