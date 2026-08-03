r"""`validate_hk --replay` must not be a one-directional ledger.

Two defects made the project's stated "only honest measure" biased NEGATIVE:

1. A winning bet whose dividend row was missing at scoring time was dropped
   ENTIRELY — stake and return both. A losing bet was always scored (a miss
   yields 0.0, never None). Only hits can take the drop path, so the loss is
   one-directional: winners vanish, losers stay.

2. Meetings were scored once and never re-scored. Dividends arrive by a
   separate backfill (`scripts.backfill_dividends`), so a meeting scored before
   its dividends landed stayed permanently booked as a total loss.

Concretely, 2026/07/12 was logged as 0 hits and $0 returned on all seven pools
while the DB now holds 175 dividend rows for that date, including 11 QUINELLA,
11 TRIO and 11 FIRST 4. Re-scoring moves BOX TRIO-5 from -62.0% to -37.6% and
F4 BOX-5 from -83.3% to -46.1%.

Every "all bet types negative" conclusion in CLAUDE.md is derived from this
file, so the bias propagates straight into the betting posture.

Fix: re-score every meeting on each run (idempotent rewrite), and when a hit
still cannot be priced, keep its stake and its hit in the ledger and report the
count — never drop it silently.
"""
import csv

import pytest

from scripts import validate_hk


class TestUnpricedHitIsNotDropped:
    def test_unpriced_hit_keeps_its_stake_and_hit(self):
        """A win we cannot price is still a win that cost money."""
        acc = {}
        validate_hk._book_bet(acc, label='BOX TRIO-5', combos=10,
                              stake_per_combo=10.0, hit=True, div_per_unit=None)
        stake, ret, hits, unpriced = acc['BOX TRIO-5']
        assert stake == 100.0, "stake must be booked — the money was spent"
        assert hits == 1, "the hit must be counted — it genuinely won"
        assert unpriced == 1, "the bet must be flagged as unpriced, not silently zeroed"
        assert ret == 0.0, "with no dividend the return is a known lower bound"

    def test_priced_hit_books_the_dividend(self):
        acc = {}
        validate_hk._book_bet(acc, label='BOX TRIO-5', combos=10,
                              stake_per_combo=10.0, hit=True, div_per_unit=37.0)
        stake, ret, hits, unpriced = acc['BOX TRIO-5']
        assert (stake, ret, hits, unpriced) == (100.0, 370.0, 1, 0)

    def test_miss_books_stake_only(self):
        acc = {}
        validate_hk._book_bet(acc, label='BOX TRIO-5', combos=10,
                              stake_per_combo=10.0, hit=False, div_per_unit=None)
        assert acc['BOX TRIO-5'] == [100.0, 0.0, 0, 0]

    def test_dropping_a_win_makes_it_invisible(self):
        """What the drop actually cost: the win disappeared from the ledger.

        NOTE on ROI: booking an unpriced win at $0 return does NOT raise ROI —
        it lowers it slightly (stake added, no return). The ROI recovery
        (-62.0% -> -37.6% on BOX TRIO-5) comes from RE-SCORING once dividends
        land, not from this. What booking fixes is that a real win was being
        recorded as though the bet never happened: hit counts were understated,
        stake was understated, and nothing flagged that a payout was owed — so
        there was no signal to go run backfill_dividends.
        """
        acc = {}
        validate_hk._book_bet(acc, 'P', 10, 10.0, hit=True, div_per_unit=None)
        validate_hk._book_bet(acc, 'P', 10, 10.0, hit=False, div_per_unit=None)
        stake, ret, hits, unpriced = acc['P']
        assert stake == 200.0, "both bets cost money; both stakes are real"
        assert hits == 1, "the win must be visible in the hit count"
        assert unpriced == 1, "and flagged as owed a dividend"
        assert (ret - stake) / stake == pytest.approx(-1.0), (
            "with no dividend the ROI is a floor, not the truth"
        )


class TestReplayIsRescored:
    def test_rescore_replaces_a_stale_row_rather_than_skipping_it(self, tmp_path,
                                                                  monkeypatch):
        """A meeting already in the log must be re-scored, not skipped."""
        log = tmp_path / 'replay.csv'
        monkeypatch.setattr(validate_hk, 'REPLAY_LOG_FILE', log)

        stale = {'STRAIGHT QIN': [50.0, 0.0, 0, 0]}
        validate_hk._write_replay_row('2026/07/12', {'_races': 6, 'labels': stale})
        assert len(validate_hk._read_replay_csv()) == 1

        fresh = {'STRAIGHT QIN': [50.0, 321.5, 1, 0]}
        validate_hk._write_replay_row('2026/07/12', {'_races': 6, 'labels': fresh})

        rows = validate_hk._read_replay_csv()
        assert len(rows) == 1, "re-scoring must REPLACE the row, not append a duplicate"
        assert '321.5' in rows[0]['ticket_json'], "the refreshed dividend must win"

    def test_rows_stay_in_date_order(self, tmp_path, monkeypatch):
        log = tmp_path / 'replay.csv'
        monkeypatch.setattr(validate_hk, 'REPLAY_LOG_FILE', log)
        for d in ('2026/07/15', '2026/07/12', '2026/07/08'):
            validate_hk._write_replay_row(d, {'_races': 1, 'labels': {}})
        dates = [r['race_date'] for r in validate_hk._read_replay_csv()]
        assert dates == sorted(dates)

    def test_replay_loop_does_not_skip_already_logged_dates(self):
        """Source guard: the old `if d in replay_already: continue` is the bug.

        With it, a meeting scored before its dividends landed can never be
        corrected, no matter how many times validate_hk is re-run.
        """
        import inspect
        src = inspect.getsource(validate_hk.main)
        assert 'replay_already' not in src, (
            "main() still skips already-logged replay meetings — late-arriving "
            "dividends can never be picked up"
        )


class TestBackwardCompatibility:
    def test_legacy_three_field_rows_still_parse(self, tmp_path, monkeypatch):
        """Existing logs store [stake, ret, hits] with no unpriced counter."""
        log = tmp_path / 'replay.csv'
        log.write_text(
            'race_date,races,ticket_json\n'
            '2026/07/12,6,"{""STRAIGHT QIN"": [50.0, 0.0, 0]}"\n'
        )
        monkeypatch.setattr(validate_hk, 'REPLAY_LOG_FILE', log)
        rows = validate_hk._read_replay_csv()
        validate_hk._report_replay(rows, 'legacy')   # must not raise
