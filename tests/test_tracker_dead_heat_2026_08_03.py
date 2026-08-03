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


class TestGateImpactUsesSharedHelper:
    def test_gate_impact_imports_the_canonical_helper(self):
        import scripts.gate_impact as gi
        src = open(gi.__file__).read()
        assert "dead_heat" in src, (
            "gate_impact still hand-rolls finisher slicing instead of using "
            "betting/dead_heat"
        )
        # The dividend LOOKUP may still index finishers[:3] (it builds the
        # combination string); what must not slice is the HIT test.
        assert "top3_actual = {h['name'] for h in finishers[:3]}" not in src, (
            "gate_impact still slices a position-sorted list to decide the hit"
        )


class TestTrackQplUsesSharedHelper:
    def test_track_qpl_imports_the_canonical_helper(self):
        import scripts.track_qpl as tq
        src = open(tq.__file__).read()
        assert "dead_heat" in src, (
            "track_qpl still hand-rolls finisher slicing instead of using "
            "betting/dead_heat"
        )
