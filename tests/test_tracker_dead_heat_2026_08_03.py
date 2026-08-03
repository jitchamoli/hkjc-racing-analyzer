"""gate_impact and track_qpl must score dead heats like validate_hk.

Both sliced a position-sorted finisher list:

    gate_impact.py   top3_actual = {h['name'] for h in finishers[:3]}
    track_qpl.py     top3_set    = {n for _, n in fin[:3]}

On a dead heat for 3rd that picks ONE arbitrary member of the tied group, so a
box genuinely holding a winning trio is booked as a miss. 43 of 4,759 races in
the DB (0.90%) have a top-3 dead heat.

These two feed decisions directly — gate_impact produces the OLD-vs-NEW BOX
TRIO-5 gate comparison, and track_qpl is the QPL tracker — so a systematic
scoring disagreement with validate_hk matters more than the raw frequency.
"""
import pytest

from betting.dead_heat import group_by_position, winning_combo_entries, winning_name_sets


def _h(name, num, pos):
    return {"name": name, "number": num, "actual_position": pos}


class TestSharedHelperCoversTheseCases:
    def test_dead_heat_for_third_yields_two_winning_trios(self):
        """A 1st, B 2nd, C and D dead-heat 3rd -> {A,B,C} and {A,B,D}."""
        finishers = [_h("A", 1, 1), _h("B", 3, 2), _h("C", 8, 3), _h("D", 5, 3)]
        combos = winning_name_sets(finishers, 3)
        assert len(combos) == 2
        assert {"A", "B", "C"} in combos and {"A", "B", "D"} in combos

    def test_box_holding_either_trio_is_a_hit(self):
        finishers = [_h("A", 1, 1), _h("B", 3, 2), _h("C", 8, 3), _h("D", 5, 3)]
        combos = winning_name_sets(finishers, 3)
        box = {"A", "B", "D", "E", "F"}
        assert any(w <= box for w in combos)

    def test_box_missing_a_certain_finisher_is_a_miss(self):
        finishers = [_h("A", 1, 1), _h("B", 3, 2), _h("C", 8, 3), _h("D", 5, 3)]
        combos = winning_name_sets(finishers, 3)
        box = {"B", "C", "D", "E", "F"}      # no A, who won
        assert not any(w <= box for w in combos)

    def test_no_dead_heat_is_a_single_trio(self):
        finishers = [_h("A", 1, 1), _h("B", 3, 2), _h("C", 8, 3), _h("D", 5, 4)]
        combos = winning_name_sets(finishers, 3)
        assert combos == [{"A", "B", "C"}]

    def test_dead_heat_for_second_gives_two_quinellas(self):
        finishers = [_h("A", 1, 1), _h("B", 3, 2), _h("C", 8, 2), _h("D", 5, 4)]
        combos = winning_name_sets(finishers, 2)
        assert len(combos) == 2


# REMOVED 2026-08-04 — two source-grep guards used to live here:
#
#   assert "top3_actual = {h['name'] for h in finishers[:3]}" not in src
#
# A test audit proved they were worse than nothing. Reintroducing the exact bug
# under a renamed variable (`winning_trios = [{h['name'] for h in
# finishers[:3]}]`) left the grep unmatched and all 1,056 tests green, while
# creating the appearance of coverage for gate_impact and track_qpl.
#
# Replaced by behavioural tests in tests/test_scoring_call_sites_2026_08_04.py,
# which feed a real dead-heat fixture through `track_qpl._rows_for_meeting` and
# `stake_ab._score_bet_from_ticket` and assert the hit is booked. Those were
# verified to FAIL when the position-slice is restored.
