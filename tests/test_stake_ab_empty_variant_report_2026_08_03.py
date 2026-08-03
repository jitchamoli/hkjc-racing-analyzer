"""A variant with no logged rows must not be given a posterior.

Bug (found 2026-08-03 while adding `straight_qin_only`): the summary table
correctly skips a variant with zero stake, but the Bayesian block below it
looped over VARIANTS unconditionally. A brand-new variant with ZERO rows in
stake_ab_log.csv was reported as

    straight_qin_only meetings-won=14/23  posterior=0.846  -> WATCH

because `_per_meeting_winner` scores "variant cumulative ROI > baseline
cumulative ROI" and a variant with no rows has ROI 0, which beats a losing
baseline in most meetings. That is fabricated evidence: it reads as 23
meetings of support for a variant that has never placed a bet.

Any newly-added variant hits this, so the guard belongs in the report, not in
one variant's config.
"""
import csv

import pytest

from scripts.stake_ab import VARIANTS, _per_meeting_winner, report


def _rows():
    """Two meetings of flat_min rows only — no rows for any other variant."""
    return [
        {'race_date': '2026/05/03', 'variant': 'flat_min', 'bet_label': 'STRAIGHT QIN',
         'combos': '1', 'stake_hkd': '10', 'hit': '0', 'dividend_per_1': '0',
         'pnl_hkd': '-10'},
        {'race_date': '2026/05/06', 'variant': 'flat_min', 'bet_label': 'STRAIGHT QIN',
         'combos': '1', 'stake_hkd': '10', 'hit': '1', 'dividend_per_1': '7.0',
         'pnl_hkd': '+60'},
    ]


class TestEmptyVariantIsNotScored:
    def test_variant_with_no_rows_has_no_meeting_wins(self):
        """A variant that never bet cannot 'win' a meeting."""
        wins = _per_meeting_winner(_rows(), 'straight_qin_only')
        assert sum(wins) == 0, (
            f"a variant with zero logged rows was credited with "
            f"{sum(wins)} meeting win(s)"
        )

    def test_report_omits_empty_variants_from_the_posterior_block(self, tmp_path,
                                                                  monkeypatch, capsys):
        import scripts.stake_ab as sab
        log = tmp_path / 'stake_ab_log.csv'
        with open(log, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sab.LOG_HEADERS, lineterminator='\n')
            w.writeheader()
            w.writerows(_rows())
        monkeypatch.setattr(sab, 'LOG_PATH', log)

        report()
        out = capsys.readouterr().out

        # It may be NAMED (as awaiting data) — it must never carry a posterior.
        scored = [ln for ln in out.splitlines()
                  if 'straight_qin_only' in ln and 'posterior=' in ln]
        assert not scored, (
            f"a variant with no logged rows was given a posterior: {scored}"
        )
        assert 'Awaiting first data' in out and 'straight_qin_only' in out, (
            f"the pending variant should be listed as awaiting data:\n{out}"
        )

    def test_variant_with_rows_is_still_scored(self, tmp_path, monkeypatch, capsys):
        """Guard: the guard must not silence variants that DO have data."""
        import scripts.stake_ab as sab
        rows = _rows() + [
            {'race_date': '2026/05/03', 'variant': 'straight_qin_only',
             'bet_label': 'STRAIGHT QIN', 'combos': '1', 'stake_hkd': '10',
             'hit': '1', 'dividend_per_1': '7.0', 'pnl_hkd': '+60'},
        ]
        log = tmp_path / 'stake_ab_log.csv'
        with open(log, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sab.LOG_HEADERS, lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
        monkeypatch.setattr(sab, 'LOG_PATH', log)

        report()
        out = capsys.readouterr().out
        assert 'straight_qin_only' in out
