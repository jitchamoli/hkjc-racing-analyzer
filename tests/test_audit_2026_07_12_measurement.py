"""Regression tests for the 2026-07-12 audit — ticket freeze + measurement.

These guard the numbers the PAPER-ONLY posture rests on. Each one corresponds to
a defect that made us score bets we never staked, or miss bets we did.
"""
import json

import pytest

from dashboard.generator import _post_time_passed, _ticket_add_bet, _pick_trio6_top6
from scripts.stake_ab import firing_bets
from scripts.validate_hk import _winning_combo_entries
from data.database import _qin_legs_hit


# --------------------------------------------------------------------------
# Post-off freeze — the ticket must not be rebuilt after the race has run
# --------------------------------------------------------------------------

def test_post_time_passed_only_fires_for_today(monkeypatch):
    """A past/future card must never read a time-of-day as 'off'd'."""
    assert _post_time_passed("00:01", race_date="1999/01/01") is False
    assert _post_time_passed("23:59", race_date="2099/01/01") is False


def test_post_time_passed_handles_missing_and_malformed():
    assert _post_time_passed(None) is False
    assert _post_time_passed("") is False
    assert _post_time_passed("not-a-time") is False


# --------------------------------------------------------------------------
# Duplicate legs — a repeated horse bills a small box at a big box's price
# --------------------------------------------------------------------------

def test_ticket_refuses_duplicate_legs():
    bets = []
    dup = [
        {"number": 1, "name": "A", "rank": 1, "win_odds": 3.0},
        {"number": 2, "name": "B", "rank": 2, "win_odds": 5.0},
        {"number": 1, "name": "A", "rank": 1, "win_odds": 3.0},  # repeat
    ]
    _ticket_add_bet(bets, "BOX TRIO-6", "TRIO", dup, 20, "recommender")
    assert bets == [], "a bet with a duplicate leg must be refused, not priced"


def test_trio6_manual_override_already_in_top5_does_not_duplicate():
    """Ranks move with odds between the AM override and the T-6min re-run. An
    override horse that has climbed into the top 5 must not be appended twice —
    that emitted 6 legs / 5 distinct horses and billed 20 combos for a 10-combo
    box."""
    horses = [{"number": n, "name": f"H{n}", "rank": n, "win_odds": 3.0 + n}
              for n in range(1, 9)]
    legs, swap = _pick_trio6_top6(horses, manual_trio6_horse=3,
                                  manual_reason="steam")
    nums = [h["number"] for h in legs]
    assert len(nums) == len(set(nums)), f"duplicate leg emitted: {nums}"
    assert swap is None, "no swap should occur — #3 is already in the top 5"


# --------------------------------------------------------------------------
# Dead heats — HKJC declares a dividend per tied combination
# --------------------------------------------------------------------------

def test_winning_combos_dead_heat_for_second():
    """2026/06/07 R1: #1 won, #3 and #8 dead-heated for 2nd. The DB really does
    carry TWO declared quinellas — '1,3' ($37) and '1,8' ($354)."""
    pos = {1: [{"number": 1}], 2: [{"number": 3}, {"number": 8}],
           3: [{"number": 2}]}
    quin = _winning_combo_entries(pos, 2)
    got = {tuple(sorted(e["number"] for e in c)) for c in quin}
    assert got == {(1, 3), (1, 8)}

    trio = _winning_combo_entries(pos, 3)
    assert {tuple(sorted(e["number"] for e in c)) for c in trio} == {(1, 3, 8)}

    f4 = _winning_combo_entries(pos, 4)
    assert {tuple(sorted(e["number"] for e in c)) for c in f4} == {(1, 2, 3, 8)}


def test_winning_combos_clean_race_is_single_combo():
    pos = {1: [{"number": 5}], 2: [{"number": 2}],
           3: [{"number": 9}], 4: [{"number": 4}]}
    for k, want in ((2, {(2, 5)}), (3, {(2, 5, 9)}), (4, {(2, 4, 5, 9)})):
        combos = _winning_combo_entries(pos, k)
        assert {tuple(sorted(e["number"] for e in c)) for c in combos} == want


def test_winning_combos_insufficient_finishers():
    assert _winning_combo_entries({1: [{"number": 1}]}, 2) == []


def test_qin_hit_requires_first_and_second():
    """Two horses DEAD-HEATING for 2nd is not a quinella — we don't hold the
    winner. The old `>=2 legs in positions 1-2` rule fabricated a hit."""
    assert _qin_legs_hit([1, 2]) is True            # normal 1st + 2nd
    assert _qin_legs_hit([2, 2]) is False           # both 2nd-equal, no winner
    assert _qin_legs_hit([1, 1]) is True            # dead heat FOR 1st = the quinella
    assert _qin_legs_hit([1, 5]) is False           # only the winner
    assert _qin_legs_hit([3, 4]) is False


# --------------------------------------------------------------------------
# stake_ab — frozen ticket is the truth; paper bets are not money
# --------------------------------------------------------------------------

def _race(bets, *, date, tier="HIGH", going="Good To Firm"):
    return {
        "race_number": 1, "conviction_tier": tier, "going": going,
        "class_type": "Class 4", "horses": [],
        "bet_ticket": {"race_date": date, "bets": bets},
    }


def test_stake_ab_keeps_frozen_bets_when_going_rewritten_after_freeze():
    """2026/07/04 R1+R2: tier frozen HIGH with real staked bets, but `going` was
    later rewritten to 'Good To Yielding'. Re-deriving CHAOS from the mutable
    going deleted HK$410 of bets that production actually staked — while
    validate_hk --replay still scored them."""
    race = _race([{"label": "STRAIGHT QIN", "combos": 1}],
                 date="2026/07/04", going="Good To Yielding")
    assert firing_bets(race, apply_chaos=True) == {"STRAIGHT QIN": 1}


def test_stake_ab_still_applies_chaos_to_pre_filter_tickets():
    """Tickets frozen BEFORE the CHAOS filter shipped still need the
    retrospective overlay, so old meetings stay comparable to production."""
    race = _race([{"label": "BOX QIN-3", "combos": 3}],
                 date="2026/05/01", going="SOFT")
    assert firing_bets(race, apply_chaos=True) == {}


def test_stake_ab_excludes_paper_bets():
    """EDGE PLACE is a $10 reference bet, never staked. Counting it let the
    `production` variant assign a whole EDGE race's stake to it."""
    race = _race(
        [{"label": "EDGE PLACE", "combos": 1, "paper": True},
         {"label": "STRAIGHT QIN", "combos": 1}],
        date="2026/07/04",
    )
    fires = firing_bets(race, apply_chaos=True)
    assert "EDGE PLACE" not in fires
    assert fires == {"STRAIGHT QIN": 1}


# --------------------------------------------------------------------------
# Round 2 — defects the adversarial review found in the FIRST round of fixes
# --------------------------------------------------------------------------

def test_real_exotic_pnl_pays_only_combos_we_held():
    """Dead heat for 2nd on 2026/06/07 R1: HKJC declared quinellas 1-3 ($37) and
    1-8 ($354). Matching declared combos against the PODIUM set instead of OUR
    LEGS paid a box holding only 1-3 for both — a phantom credit, the exact
    error class this audit exists to kill."""
    from data.database import HorseDatabase
    db = HorseDatabase()
    podium = [1, 3, 8]

    pnl_13, _ = db._real_exotic_pnl('2026/06/07', 1, 'QIN', 2, True,
                                    podium, podium, leg_nums=[1, 3])
    pnl_18, _ = db._real_exotic_pnl('2026/06/07', 1, 'QIN', 2, True,
                                    podium, podium, leg_nums=[1, 8])

    # 3.7u and 35.4u per HK$1; stake for a 2-leg QIN is 1 combo.
    assert round(pnl_13, 1) == 2.7, "credited a dividend we never held"
    assert round(pnl_18, 1) == 34.4
    assert pnl_13 < pnl_18, "the two tied combos must not collapse to one figure"


def test_real_exotic_pnl_box_holding_both_combos_collects_both():
    """A box that DOES hold both tied combinations legitimately collects both."""
    from data.database import HorseDatabase
    db = HorseDatabase()
    podium = [1, 3, 8]
    pnl, stake = db._real_exotic_pnl('2026/06/07', 1, 'QIN', 3, True,
                                     podium, podium, leg_nums=[1, 3, 8])
    assert stake == 3.0                      # C(3,2) combos
    assert round(pnl, 1) == 36.1             # (3.7 + 35.4) - 3
