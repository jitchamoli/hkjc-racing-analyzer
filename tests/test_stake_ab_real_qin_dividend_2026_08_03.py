r"""stake_ab must price QIN hits from the HKJC payout, not a Harville estimate.

Two defects, both biasing the A/B that decides how you stake:

1. The QIN family read `race['qin_top_pair_dividend']`, which
   dashboard/generator.py writes as a PRE-RACE Harville *estimate*. 133 of 176
   race JSONs carry `qin_dividend_source == 'harville'`. That estimate was then
   consumed as if it were a declared dividend. Real example: 2026/05/13 R6
   straight QIN paid $321.50/HK$10; the estimate used was $24.18 — a $297
   error on one race. 2026/04/26 R11 BOX QIN-4 paid $119.50 and was booked as
   a $53 LOSS.

   Every other pool (TRIO, FIRST 4) already looks the real dividend up from
   `race_dividends` via `_lookup_dividend`. QIN was the exception.

2. `_per_meeting_winner` awarded a variant a "win" on meetings where it staked
   nothing. The 2026-08-03 guard only caught a variant that had NEVER staked;
   one that stakes occasionally still collects a free win on every zero-stake
   meeting, because its ROI evaluates to 0 which beats a losing baseline.
   `edge_filtered` would cross the 0.95 SHIP bar on ~4 losing meetings.

Dead heats: HKJC declares a QUINELLA dividend per tied combination, so a box
holding several winning pairs collects each — mirroring the TRIO/F4 handling.
"""
import sqlite3

import pytest

from scripts import stake_ab


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    c.execute("""CREATE TABLE race_dividends (
        race_date TEXT, race_number INT, pool TEXT,
        combination TEXT, dividend REAL)""")
    return c


def _race(top_names, finish, numbers, harville=None):
    """Build a race dict: model top picks + actual finish order."""
    horses = []
    for i, n in enumerate(top_names):
        horses.append({'name': n, 'rank': i + 1, 'number': numbers[n],
                       'actual_position': 0})
    by_name = {h['name']: h for h in horses}
    for pos, n in enumerate(finish, 1):
        by_name[n]['actual_position'] = pos
    race = {'race_number': 1, 'horses': horses}
    if harville is not None:
        race['qin_top_pair_dividend'] = harville
    return race


class TestRealDividendWins:
    def test_straight_qin_uses_the_declared_payout_not_the_estimate(self, conn):
        conn.execute("INSERT INTO race_dividends VALUES "
                     "('2026/05/13', 1, 'QUINELLA', '3,7', 321.5)")
        race = _race(['A', 'B', 'C'], ['A', 'B', 'C'],
                     {'A': 3, 'B': 7, 'C': 9}, harville=24.18)

        hit, div = stake_ab.hit_and_dividend('STRAIGHT QIN', race, conn, '2026/05/13')

        assert hit is True
        assert div == pytest.approx(32.15), (
            f"expected the real $321.50/HK$10 payout (32.15 per $1), got {div} "
            f"— the Harville estimate is still being used"
        )

    def test_box_qin_uses_the_declared_payout(self, conn):
        conn.execute("INSERT INTO race_dividends VALUES "
                     "('2026/04/26', 1, 'QUINELLA', '2,5', 119.5)")
        race = _race(['A', 'B', 'C', 'D'], ['C', 'D', 'A', 'B'],
                     {'A': 1, 'B': 8, 'C': 2, 'D': 5}, harville=6.58)

        hit, div = stake_ab.hit_and_dividend('BOX QIN-4', race, conn, '2026/04/26')

        assert hit is True
        assert div == pytest.approx(11.95)

    def test_dead_heat_sums_every_declared_quinella(self, conn):
        """A box holding both tied pairs collects both dividends."""
        conn.execute("INSERT INTO race_dividends VALUES "
                     "('2026/06/07', 1, 'QUINELLA', '1,3', 37.0)")
        conn.execute("INSERT INTO race_dividends VALUES "
                     "('2026/06/07', 1, 'QUINELLA', '1,8', 354.0)")
        race = _race(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'],
                     {'A': 1, 'B': 3, 'C': 8, 'D': 9})
        # B and C dead-heat for 2nd
        for h in race['horses']:
            if h['name'] in ('B', 'C'):
                h['actual_position'] = 2
            elif h['name'] == 'D':
                h['actual_position'] = 4

        hit, div = stake_ab.hit_and_dividend('BOX QIN-4', race, conn, '2026/06/07')

        assert hit is True
        assert div == pytest.approx(39.1), "must sum both declared combinations"

    def test_miss_pays_nothing(self, conn):
        """Our pair is A+B; the race is actually won by A+C."""
        conn.execute("INSERT INTO race_dividends VALUES "
                     "('2026/05/13', 1, 'QUINELLA', '3,9', 50.0)")
        # ranks A,B,C — finish A(1st), C(2nd), B(3rd)
        race = _race(['A', 'B', 'C'], ['A', 'C', 'B'],
                     {'A': 3, 'B': 7, 'C': 9}, harville=24.18)
        hit, div = stake_ab.hit_and_dividend('STRAIGHT QIN', race, conn, '2026/05/13')
        assert hit is False
        assert div == 0.0, "a missed pair must not collect the winning pair's payout"

    def test_unpriced_hit_returns_zero_not_the_estimate(self, conn):
        """No declared dividend: book 0, never substitute a model estimate."""
        race = _race(['A', 'B'], ['A', 'B'], {'A': 3, 'B': 7}, harville=24.18)
        hit, div = stake_ab.hit_and_dividend('STRAIGHT QIN', race, conn, '2026/05/13')
        assert hit is True
        assert div == 0.0, "a Harville estimate must never stand in for a payout"


class TestPerMeetingWinnerNeedsBothToStake:
    def _rows(self):
        # m1: both stake, variant loses.  m2: variant stakes nothing.
        return [
            {'race_date': '2026/07/01', 'variant': 'flat_min',
             'stake_hkd': '100', 'pnl_hkd': '-40'},
            {'race_date': '2026/07/01', 'variant': 'edge_filtered',
             'stake_hkd': '100', 'pnl_hkd': '-50'},
            {'race_date': '2026/07/04', 'variant': 'flat_min',
             'stake_hkd': '100', 'pnl_hkd': '-60'},
            {'race_date': '2026/07/04', 'variant': 'edge_filtered',
             'stake_hkd': '0', 'pnl_hkd': '0'},
        ]

    def test_zero_stake_meeting_is_not_a_free_win(self):
        winners = stake_ab._per_meeting_winner(self._rows(), 'edge_filtered')
        assert len(winners) == 1, (
            f"only the meeting where BOTH staked should count, got {winners}"
        )
        assert winners == [0], "the variant lost that meeting; it must not win it"

    def test_a_variant_that_never_stakes_yields_no_meetings(self):
        rows = [r for r in self._rows()
                if not (r['variant'] == 'edge_filtered' and r['stake_hkd'] == '100')]
        assert stake_ab._per_meeting_winner(rows, 'edge_filtered') == []

    def test_genuine_outperformance_still_counts(self):
        rows = self._rows()
        rows[1]['pnl_hkd'] = '+30'      # variant beats baseline on m1
        assert stake_ab._per_meeting_winner(rows, 'edge_filtered') == [1]
