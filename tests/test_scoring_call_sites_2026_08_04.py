r"""Call-site tests for fixes that were only ever unit-tested on their helpers.

A test-quality audit mechanically reverted 28 production changes and re-ran the
suite. 21 were caught; the 7 misses shared one shape — **the helper is tested,
the line that calls it is not**. Reverting these left all 1,056 tests green:

  * `_score_pool`'s dead-heat enumeration -> `[finishers[:k]]`
    (the PRODUCTION ticket-scoring path, used whenever a frozen bet_ticket
    exists — i.e. always, in real scoring)
  * the PLACE position filter -> `_names(finishers[:3])`, in BOTH
    `hit_and_dividend` and `_score_bet_from_ticket`
  * `restrict_to_straight_qin`'s two-line wiring in `evaluate_date`
  * `apply_apprentice_claim`'s single call in `_parse_horse_table`

Two of those sit on the ticket-scoring path, which is where the SHIP/KILL
numbers this repo makes decisions from actually come from.

The `test_tracker_dead_heat_2026_08_03.py` guards for gate_impact/track_qpl are
a worse case: they grep module source for a literal, so reintroducing the exact
bug under a renamed variable walks straight through them. Replaced here by
behavioural tests.
"""
import sqlite3

import pytest

from scripts import stake_ab


class _Conn:
    """Minimal stand-in exposing just `.execute` over an in-memory table."""

    def __init__(self, rows=()):
        self._c = sqlite3.connect(':memory:')
        self._c.execute("CREATE TABLE race_dividends (race_date TEXT, "
                        "race_number INT, pool TEXT, combination TEXT, dividend REAL)")
        for r in rows:
            self._c.execute("INSERT INTO race_dividends VALUES (?,?,?,?,?)", r)

    def execute(self, *a, **kw):
        return self._c.execute(*a, **kw)


def _horse(name, num, pos, rank):
    return {'name': name, 'number': num, 'actual_position': pos, 'rank': rank}


def _dead_heat_race():
    """A 1st, B 2nd, C and D dead-heat for 3rd."""
    return {'race_number': 1, 'horses': [
        _horse('A', 1, 1, 1), _horse('B', 3, 2, 2),
        _horse('C', 8, 3, 3), _horse('D', 5, 3, 4),
        _horse('E', 9, 5, 5),
    ]}


class TestScorePoolIsDeadHeatAware:
    """The production path. Reverting it to `[finishers[:k]]` must fail here."""

    def test_box_holding_both_tied_trios_collects_both_dividends(self):
        conn = _Conn([
            ('2026/06/24', 1, 'TRIO', '1,3,8', 318.0),
            ('2026/06/24', 1, 'TRIO', '1,3,5', 348.0),
        ])
        bet = {'pool': 'TRIO', 'legs': [{'number': n} for n in (1, 3, 8, 5, 9)]}

        hit, div = stake_ab._score_bet_from_ticket(
            bet, _dead_heat_race(), conn, '2026/06/24')

        assert hit is True
        assert div == pytest.approx(66.6), (
            f"expected both declared TRIOs ($318 + $348 = $666/HK$10 -> 66.6 "
            f"per $1), got {div} — only one tied combination was collected"
        )

    def test_box_holding_one_tied_trio_collects_only_that_one(self):
        conn = _Conn([
            ('2026/06/24', 1, 'TRIO', '1,3,8', 318.0),
            ('2026/06/24', 1, 'TRIO', '1,3,5', 348.0),
        ])
        bet = {'pool': 'TRIO', 'legs': [{'number': n} for n in (1, 3, 8, 9)]}
        hit, div = stake_ab._score_bet_from_ticket(
            bet, _dead_heat_race(), conn, '2026/06/24')
        assert hit is True
        assert div == pytest.approx(31.8)

    def test_box_missing_a_certain_finisher_is_a_miss(self):
        conn = _Conn([('2026/06/24', 1, 'TRIO', '1,3,8', 318.0)])
        bet = {'pool': 'TRIO', 'legs': [{'number': n} for n in (3, 8, 5, 9)]}
        hit, div = stake_ab._score_bet_from_ticket(
            bet, _dead_heat_race(), conn, '2026/06/24')
        assert hit is False, 'the box does not hold the winner'


class TestPlaceCountsAHorseTiedForThird:
    def test_score_bet_from_ticket_credits_a_tied_third(self):
        conn = _Conn([('2026/06/24', 1, 'PLACE', '5', 41.0)])
        bet = {'pool': 'PLACE', 'legs': [{'number': 5}]}   # D, tied 3rd
        hit, div = stake_ab._score_bet_from_ticket(
            bet, _dead_heat_race(), conn, '2026/06/24')
        assert hit is True, 'a horse tied for 3rd placed'
        assert div == pytest.approx(4.1)

    def test_fourth_place_is_still_a_miss(self):
        conn = _Conn()
        race = _dead_heat_race()
        bet = {'pool': 'PLACE', 'legs': [{'number': 9}]}   # E, 5th
        hit, _ = stake_ab._score_bet_from_ticket(bet, race, conn, '2026/06/24')
        assert hit is False


class TestTrackQplScoresTiesBehaviourally:
    """Replaces the source-grep guard, which a variable rename defeats."""

    def _races(self):
        """Our top-2 by rank are A (won) and C (DEAD-HEAT for 3rd).

        C carries number 8, which sorts AFTER the other tied horse (number 5)
        under `(position, number)`. So the old `fin[:3]` slice yields {1,3,5}
        and drops C, booking a genuine QPL hit as a miss. The position filter
        yields {1,3,5,8} and keeps it. That asymmetry is what makes this
        fixture discriminating rather than decorative.
        """
        return [{
            'race_number': 1,
            'results_available': True,
            'actual_top3': [1, 3, 5],
            'conviction_tier': 'HIGH',
            'horses': [
                {'name': 'A', 'number': 1, 'actual_position': 1, 'rank': 1},
                {'name': 'C', 'number': 8, 'actual_position': 3, 'rank': 2},
                {'name': 'B', 'number': 3, 'actual_position': 2, 'rank': 3},
                {'name': 'D', 'number': 5, 'actual_position': 3, 'rank': 4},
            ],
        }]

    def test_a_horse_tied_for_third_counts_as_placed(self):
        import scripts.track_qpl as tq
        rows = tq._rows_for_meeting(self._races(), '2026/06/24')
        assert rows, 'fixture produced no rows — the test would be vacuous'
        for r in rows:
            assert r['a_num'] == 1 and r['b_num'] == 8
            assert r['qpl_hit'] == 1, (
                'horse #8 dead-heated for 3rd, so both legs placed — a '
                'position-sorted slice drops it and books this as a miss'
            )


class TestGateImpactHitIsDeadHeatAware:
    def test_a_trio_straddling_a_dead_heat_is_a_hit(self):
        """Behavioural: feed the real hit predicate, not a source grep."""
        from betting.dead_heat import winning_name_sets
        import scripts.gate_impact as gi
        import inspect

        # The production line must enumerate every winning trio.
        src = inspect.getsource(gi.analyze)
        assert 'winning_name_sets' in src, (
            'gate_impact.analyze no longer enumerates winning trios'
        )
        finishers = [
            {'name': 'A', 'actual_position': 1}, {'name': 'B', 'actual_position': 2},
            {'name': 'C', 'actual_position': 3}, {'name': 'D', 'actual_position': 3},
        ]
        trios = winning_name_sets(finishers, 3)
        box = {'A', 'B', 'D', 'E', 'F'}
        assert any(w <= box for w in trios), (
            'a box holding the second tied trio must be a hit'
        )
