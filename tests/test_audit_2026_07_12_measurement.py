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


# --------------------------------------------------------------------------
# Round 3 — defects the SECOND adversarial panel found in the merged audit
# --------------------------------------------------------------------------

def test_trio_hit_is_dead_heat_aware():
    """A tie for 3rd flags FOUR horses is_top3 (positions 1,2,3,3). The old
    inline `>= 3 legs in top-3` scored a box holding {1st, 3a, 3b} but NOT the
    2nd-place horse as a TRIO hit — it is not: a trio needs 1st AND 2nd AND a
    3rd. Mirrors the QIN dead-heat fix. Positions-only, full-field aware."""
    from data.database import _trio_legs_hit
    # field: #1=1st, #2=2nd, #5 & #7 dead-heat 3rd
    pos_by_num = {1: 1, 2: 2, 5: 3, 7: 3, 4: 5, 9: 6}
    assert _trio_legs_hit([1, 5, 7], pos_by_num) is False   # missing 2nd
    assert _trio_legs_hit([1, 2, 5], pos_by_num) is True    # 1st+2nd+one 3rd
    assert _trio_legs_hit([1, 2, 7], pos_by_num) is True    # 1st+2nd+other 3rd
    # clean race: exactly the top-3 required
    clean = {1: 1, 2: 2, 3: 3, 4: 4}
    assert _trio_legs_hit([1, 2, 3], clean) is True
    assert _trio_legs_hit([1, 2, 4], clean) is False


def test_trio_hit_dead_heat_for_first_requires_both_winners():
    """Dead heat for 1st (positions 1,1,3) → podium {1a,1b,3rd}. A 'cover
    positions 1,2,3' shortcut would wrongly reject this (there is no position
    2). The winning trio is both co-winners plus the 3rd."""
    from data.database import _trio_legs_hit
    pos_by_num = {1: 1, 8: 1, 5: 3, 2: 4}      # #1 & #8 dead-heat 1st, #5 3rd
    assert _trio_legs_hit([1, 8, 5], pos_by_num) is True
    assert _trio_legs_hit([1, 5, 2], pos_by_num) is False   # missing co-winner #8
    # triple dead heat for 1st → podium is the three co-winners
    triple = {1: 1, 2: 1, 3: 1, 9: 4}
    assert _trio_legs_hit([1, 2, 3], triple) is True
    assert _trio_legs_hit([1, 2, 9], triple) is False


def test_stake_ab_reconstructs_chaos_suppressed_counterfactual():
    """`flat_min_pre_chaos` (apply_chaos=False) must recover the bets a
    freeze-time CHAOS suppression removed — otherwise the CHAOS-filter forward
    validation silently reads as a no-op.

    This uses the REAL post-CHAOS state build_race_data produces: tier is
    overwritten to 'CHAOS' and every bet_recommendation is nulled, with the
    pre-chaos intent stashed in `pre_chaos_*`. The 5-expert review (2026-07-15)
    found the round-3 fix passed only because its fixture used an IMPOSSIBLE
    state (tier=HIGH + bet_recommendation set on a chaos-empty ticket); the real
    reconstruction reads the stashed fields."""
    race = {
        "race_number": 1, "conviction_tier": "CHAOS",
        "pre_chaos_conviction_tier": "HIGH",
        "going": "SOFT", "class_type": "Class 4",
        "horses": [{"number": n, "rank": n, "win_odds": 3.0 + n,
                    "bet_recommendation": None,
                    "pre_chaos_bet_recommendation": "TRIO"} for n in range(1, 11)],
        "bet_ticket": {"race_date": "2026/07/12", "bets": []},   # chaos-suppressed
    }
    # counterfactual: chaos filter removed — bets must reappear from pre_chaos_*
    assert firing_bets(race, apply_chaos=False) != {}
    # real variant: chaos ON keeps the suppression
    assert firing_bets(race, apply_chaos=True) == {}


def test_stake_ab_empty_ticket_non_chaos_stays_empty():
    """An empty ticket on a NON-soft card is genuinely empty (SKIP/no-edge),
    not chaos — the counterfactual must NOT fabricate bets for it."""
    race = {
        "race_number": 1, "conviction_tier": "SKIP", "going": "Good To Firm",
        "class_type": "Class 4",
        "horses": [{"number": n, "rank": n, "win_odds": 3.0 + n} for n in range(1, 11)],
        "bet_ticket": {"race_date": "2026/07/12", "bets": []},
    }
    assert firing_bets(race, apply_chaos=False) == {}
