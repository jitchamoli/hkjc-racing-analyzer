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
    'YIELDING', 'SOFT', 'HEAVY', 'WET-SLOW', 'WET SLOW', 'YIELDING TO SOFT',
    'yielding',                     # case-insensitive
    'Y', 'YS', 'S', 'H', 'WS',      # HKJC short codes for genuine soft
])
def test_detect_chaos_fires_on_soft_going(going):
    is_chaos, reason = detect_chaos(going=going)
    assert is_chaos is True
    assert reason == 'soft_going'


@pytest.mark.parametrize('going', [
    'GOOD', 'GOOD TO FIRM', 'FIRM', 'FAST', '', None,
    # NARROWED 2026-07-15: Good-To-Yielding behaves like firm (fav strike ~31%
    # over 154 races) — it must NOT trigger CHAOS. Nor its short code GY.
    'GOOD TO YIELDING', 'Good To Yielding', 'GY',
    'WET FAST', 'WF',               # AWT fast surface — not soft
    'G', 'GF', 'GD',                # firm-ish short codes
])
def test_detect_chaos_does_not_fire_on_firm_gy_or_missing(going):
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


@pytest.mark.parametrize('going', ['YIELDING', 'SOFT', 'HEAVY', 'WET SLOW'])
def test_firing_bets_suppresses_all_soft_going_variants(going):
    race = _race(going=going)
    assert firing_bets(race) == {}


def test_firing_bets_does_not_suppress_good_to_yielding():
    """NARROWED 2026-07-15: Good-To-Yielding behaves like firm and must bet
    normally — the naive 'YIELDING' substring used to suppress it."""
    assert len(firing_bets(_race(going='GOOD TO YIELDING'))) > 0


def test_wet_awt_going_is_chaos_end_to_end():
    """AWT 'Wet Slow' must suppress: the fetcher now parses it (it was dropped
    to '' before) and detect_chaos catches WET SLOW."""
    assert detect_chaos(going='Wet Slow')[0] is True
    assert firing_bets(_race(going='Wet Slow')) == {}
    # 'Wet Fast' is a FAST surface — must NOT suppress.
    assert detect_chaos(going='Wet Fast')[0] is False


# ---------------------------------------------------------------------------
# Counterfactual A/B: apply_chaos=False bypasses the filter so we can
# forward-validate the CHAOS filter at N≥6 meetings via stake_ab.py.
# ---------------------------------------------------------------------------

def test_firing_bets_apply_chaos_false_does_not_suppress_soft_going():
    """The pre-CHAOS counterfactual — bets fire on soft going so we can
    measure what we WOULD have done without the filter, for forward A/B."""
    race = _race(going='YIELDING')
    fires_default = firing_bets(race)
    fires_pre_chaos = firing_bets(race, apply_chaos=False)
    assert fires_default == {}, "Default should still suppress (production)"
    assert len(fires_pre_chaos) > 0, (
        "apply_chaos=False should fire bets so the counterfactual is logged"
    )


def test_firing_bets_apply_chaos_false_matches_firm_going_behavior():
    """Sanity: with apply_chaos=False on a soft-going race, the fires should
    match what a firm-going version of the same race would fire."""
    soft = _race(going='YIELDING')
    firm = _race(going='GOOD')
    assert firing_bets(soft, apply_chaos=False) == firing_bets(firm)


def test_stake_ab_pre_chaos_variant_registered():
    """The flat_min_pre_chaos variant must exist with apply_chaos=False
    so forward-validation logs the counterfactual."""
    from scripts.stake_ab import VARIANTS
    assert 'flat_min_pre_chaos' in VARIANTS
    assert VARIANTS['flat_min_pre_chaos']['apply_chaos'] is False
    # All other variants stay at apply_chaos=True (production behavior)
    for name, cfg in VARIANTS.items():
        if name != 'flat_min_pre_chaos':
            assert cfg.get('apply_chaos', True) is True, (
                f"Variant {name} should keep apply_chaos=True"
            )
