r"""PLACE P&L must use the declared HKJC dividend, not a win-odds proxy.

`fill_prediction_results` scored a placed horse as:

    profit = (max(odds / 3, 1.2) * 10 - 10)

i.e. it GUESSED the place dividend from the win odds. The real figure sits in
`race_dividends` (pool 'PLACE', keyed by horse number) for the same race and
was never consulted.

Measured over 604 placed rows that have a real dividend, the proxy over-credits
by mean +$6.16 / median +$1.33 per HK$10 — total **+$3,721** of fabricated
profit. Worst case 2026/06/07 R3 NEXT FORTUNE: proxy $143.30 vs real $41.00, a
+$102.30 error on a $10 bet. This feeds `get_accuracy_summary`, so the
dashboard's PLACE ROI is systematically overstated.

(The WIN path is fine — median proxy minus real is $0.00 over 201 winners,
because a WIN dividend really is the win odds.)

The proxy is kept ONLY as a last resort for races with no declared dividend,
and the divisor-3 heuristic is documented as such rather than presented as a
payout.
"""
import sqlite3

import pytest

from data.database import place_profit_per_10


class TestRealDividendIsUsed:
    def _conn(self, rows=()):
        c = sqlite3.connect(':memory:')
        c.execute("CREATE TABLE race_dividends (race_date TEXT, race_number INT, "
                  "pool TEXT, combination TEXT, dividend REAL)")
        for r in rows:
            c.execute("INSERT INTO race_dividends VALUES (?,?,?,?,?)", r)
        return c

    def test_declared_place_dividend_wins_over_the_proxy(self):
        """2026/06/07 R3 NEXT FORTUNE: real $41.00, proxy would say $143.30."""
        conn = self._conn([('2026/06/07', 3, 'PLACE', '7', 41.0)])
        profit = place_profit_per_10(conn, '2026/06/07', 3, horse_number=7,
                                     win_odds=43.0, placed=True)
        assert profit == pytest.approx(31.0), (
            f"expected $41.00 declared - $10 stake = $31.00, got {profit} "
            f"— the win-odds proxy is still being used"
        )

    def test_a_short_priced_placer_is_not_inflated(self):
        conn = self._conn([('2026/06/07', 3, 'PLACE', '2', 12.5)])
        profit = place_profit_per_10(conn, '2026/06/07', 3, horse_number=2,
                                     win_odds=3.0, placed=True)
        assert profit == pytest.approx(2.5)

    def test_unplaced_horse_loses_the_stake(self):
        conn = self._conn([('2026/06/07', 3, 'PLACE', '7', 41.0)])
        profit = place_profit_per_10(conn, '2026/06/07', 3, horse_number=7,
                                     win_odds=43.0, placed=False)
        assert profit == pytest.approx(-10.0)

    def test_missing_dividend_falls_back_to_the_proxy(self):
        """No declared PLACE row: the estimate is the only option."""
        conn = self._conn()
        profit = place_profit_per_10(conn, '2026/06/07', 3, horse_number=7,
                                     win_odds=43.0, placed=True)
        assert profit == pytest.approx(max(43.0 / 3, 1.2) * 10 - 10)

    def test_missing_horse_number_falls_back_to_the_proxy(self):
        conn = self._conn([('2026/06/07', 3, 'PLACE', '7', 41.0)])
        profit = place_profit_per_10(conn, '2026/06/07', 3, horse_number=None,
                                     win_odds=6.0, placed=True)
        assert profit == pytest.approx(max(6.0 / 3, 1.2) * 10 - 10)


class TestAgainstTheLiveDatabase:
    def test_proxy_overstates_place_profit_on_real_data(self):
        """Documents the size of the error the fix removes."""
        import config
        if not config.DB_PATH.exists():
            pytest.skip('no local DB')
        conn = sqlite3.connect(str(config.DB_PATH))
        rows = conn.execute("""
            SELECT p.win_odds, d.dividend
              FROM prediction_log p
              JOIN race_results r ON r.race_date = p.race_date
                   AND r.race_number = p.race_number
                   AND r.horse_name = p.horse_name
              JOIN race_dividends d ON d.race_date = p.race_date
                   AND d.race_number = p.race_number AND d.pool = 'PLACE'
                   AND d.combination = CAST(r.horse_number AS TEXT)
             WHERE r.finish_position BETWEEN 1 AND 3 AND p.win_odds > 0
             LIMIT 2000
        """).fetchall()
        conn.close()
        if len(rows) < 50:
            pytest.skip('not enough resolved PLACE rows')

        err = [max(o / 3, 1.2) * 10 - d for o, d in rows]
        mean_err = sum(err) / len(err)
        assert mean_err > 0, (
            f'expected the proxy to OVERSTATE on average, got {mean_err:+.2f}'
        )
