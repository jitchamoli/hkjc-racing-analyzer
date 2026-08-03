"""`stake_ab` must score dead heats the way `validate_hk` does.

Bug (code review 2026-08-03): `scripts/validate_hk.py::_winning_combo_entries`
is dead-heat aware — HKJC declares a dividend for EACH tied combination, so a
dead heat for 2nd produces TWO winning quinellas and a box holding all three
horses collects both. `scripts/stake_ab.py` instead did:

    actual_top2 = _names(finishers[:2])      # slice a sorted list
    hit = actual_top2.issubset(box)

Slicing picks ONE arbitrary horse from the tied group, so a box genuinely
holding a winning combination is scored as a miss. `stake_ab.py` claimed to
"mirror validate_hk._score_meeting_replay branch-for-branch"; it no longer did.

43 of 4,759 races in the DB (0.90%) have a top-3 dead heat — small, but these
two tools produce the SHIP/KILL numbers that get compared against each other,
so a systematic disagreement between them is worse than the raw frequency
suggests. Real case: 2026/06/07 R1 declared quinellas '1,3' ($37) and '1,8'
($354).
"""
import pytest

from scripts.stake_ab import hit_and_dividend


def _horse(name, number, rank, pos):
    return {"name": name, "number": number, "rank": rank,
            "actual_position": pos, "win_odds": 4.0}


class _StubConn:
    """Minimal sqlite3-shaped stub: every dividend lookup declares HK$370."""
    def execute(self, *_a, **_k):
        return self

    def fetchone(self):
        return (370.0,)


def _race_dead_heat_for_2nd(our_ranks):
    """A beats B and C who dead-heat for 2nd; D is 4th.

    Winning quinellas: {A,B} and {A,C} — HKJC pays both.
    `our_ranks` maps horse name -> our model rank.
    """
    return {
        "race_number": 1,
        "qin_top_pair_dividend": 370.0,   # per HK$10
        "horses": [
            _horse("A", 1, our_ranks["A"], 1),
            _horse("B", 3, our_ranks["B"], 2),
            _horse("C", 8, our_ranks["C"], 2),   # dead heat for 2nd
            _horse("D", 5, our_ranks["D"], 4),
        ],
    }


def _race_dead_heat_for_3rd(our_ranks):
    """A 1st, B 2nd, C and D dead-heat for 3rd.

    Winning trios: {A,B,C} and {A,B,D}.
    """
    return {
        "race_number": 1,
        "horses": [
            _horse("A", 1, our_ranks["A"], 1),
            _horse("B", 3, our_ranks["B"], 2),
            _horse("C", 8, our_ranks["C"], 3),
            _horse("D", 5, our_ranks["D"], 3),   # dead heat for 3rd
        ],
    }


class TestDeadHeatQuinella:
    def test_box_holding_the_other_tied_quinella_is_a_hit(self):
        """Box holds A and C (the '1,8' winner) but not B — must still hit."""
        race = _race_dead_heat_for_2nd({"A": 1, "C": 2, "D": 3, "B": 4})
        hit, _ = hit_and_dividend("BOX QIN-3", race, conn=_StubConn(), race_date="2026/06/07")
        assert hit, "box holding a genuinely winning quinella scored as a miss"

    def test_box_holding_the_first_tied_quinella_is_a_hit(self):
        """Symmetry guard: the other tied pair must hit too."""
        race = _race_dead_heat_for_2nd({"A": 1, "B": 2, "D": 3, "C": 4})
        hit, _ = hit_and_dividend("BOX QIN-3", race, conn=_StubConn(), race_date="2026/06/07")
        assert hit

    def test_box_missing_the_winner_is_still_a_miss(self):
        """Guard against over-crediting: no winning pair present."""
        race = _race_dead_heat_for_2nd({"B": 1, "C": 2, "D": 3, "A": 4})
        hit, _ = hit_and_dividend("BOX QIN-3", race, conn=_StubConn(), race_date="2026/06/07")
        assert not hit, "scored a hit without holding the winner"

    def test_straight_qin_on_either_tied_pair_is_a_hit(self):
        race = _race_dead_heat_for_2nd({"A": 1, "C": 2, "B": 3, "D": 4})
        hit, _ = hit_and_dividend("STRAIGHT QIN", race, conn=_StubConn(), race_date="2026/06/07")
        assert hit


class TestDeadHeatTrio:
    def test_box_holding_the_other_tied_trio_is_a_hit(self):
        """Box holds A, B, D — a genuinely winning trio."""
        race = _race_dead_heat_for_3rd({"A": 1, "B": 2, "D": 3, "C": 4})
        hit, _ = hit_and_dividend("BOX TRIO-3", race, conn=_StubConn(), race_date="2026/06/07")
        assert hit, "box holding a genuinely winning trio scored as a miss"

    def test_box_missing_a_certain_finisher_is_a_miss(self):
        """A finished 1st and is not in the box — cannot be a winning trio."""
        race = _race_dead_heat_for_3rd({"B": 1, "C": 2, "D": 3, "A": 4})
        hit, _ = hit_and_dividend("BOX TRIO-3", race, conn=_StubConn(), race_date="2026/06/07")
        assert not hit


class TestNoDeadHeatUnchanged:
    def test_ordinary_race_scoring_is_unchanged(self):
        """Regression guard: the 99.1% no-tie path must behave identically."""
        race = {
            "race_number": 1,
            "qin_top_pair_dividend": 370.0,
            "horses": [
                _horse("A", 1, 1, 1),
                _horse("B", 3, 2, 2),
                _horse("C", 8, 3, 3),
                _horse("D", 5, 4, 4),
            ],
        }
        hit, div = hit_and_dividend("STRAIGHT QIN", race, conn=_StubConn(),
                                    race_date="2026/06/07")
        assert hit and div == pytest.approx(37.0)

        miss_race = {
            "race_number": 1,
            "qin_top_pair_dividend": 370.0,
            "horses": [
                _horse("A", 1, 3, 1),
                _horse("B", 3, 4, 2),
                _horse("C", 8, 1, 3),
                _horse("D", 5, 2, 4),
            ],
        }
        hit, _ = hit_and_dividend("STRAIGHT QIN", miss_race, conn=_StubConn(),
                                  race_date="2026/06/07")
        assert not hit
