"""Regression tests for the 2026-06-08 audit follow-up (findings #5, #9, #1).

Three guards:
  #9 — static current-season jockey/trainer tier fallback must NOT leak into
       backtests (before_date set). core/analyzer.py.
  #5 — the bankroll guard (daily stop-loss / exposure cap) must be ENFORCED by
       the generator's finalize step, not merely advisory. dashboard/generator.py.
  #1 — dashboard generation already fail-closes on a bad blend calibrator
       (preflight). Lock that so it can't silently regress to fail-open.
"""
import types

import pytest

from core.analyzer import BettingStrategy


# ---------------------------------------------------------------------------
# #9 — no current-season tier leak into historical backtests
# ---------------------------------------------------------------------------

class TestTierFallbackNoBacktestLeak:
    """The DB stats path is date-aware (before_date). The hardcoded tier sets
    are 2026-season opinions; applying them to an old race leaks today's view
    backwards. With before_date set, a tier-1 name must score the SAME as an
    unknown name (i.e. the tier was bypassed). Live (before_date='') keeps the
    tier so production behaviour is unchanged."""

    def test_jockey_tier_bypassed_in_backtest(self):
        elite_bt = BettingStrategy.analyze_jockey('Z Purton', db=None,
                                                  before_date='2024/01/01')
        unknown_bt = BettingStrategy.analyze_jockey('Nobody Random', db=None,
                                                    before_date='2024/01/01')
        # Backtest: elite name gets no special treatment.
        assert elite_bt[0] == unknown_bt[0]

    def test_jockey_tier_applied_live(self):
        elite_live = BettingStrategy.analyze_jockey('Z Purton', db=None,
                                                    before_date='')
        unknown_live = BettingStrategy.analyze_jockey('Nobody Random', db=None,
                                                      before_date='')
        # Live: elite name still scores above an unknown one.
        assert elite_live[0] > unknown_live[0]

    def test_trainer_tier_bypassed_in_backtest(self):
        elite_bt = BettingStrategy.analyze_trainer('J Size', db=None,
                                                   before_date='2024/01/01')
        unknown_bt = BettingStrategy.analyze_trainer('Nobody Random', db=None,
                                                     before_date='2024/01/01')
        assert elite_bt[0] == unknown_bt[0]

    def test_trainer_tier_applied_live(self):
        elite_live = BettingStrategy.analyze_trainer('J Size', db=None,
                                                     before_date='')
        unknown_live = BettingStrategy.analyze_trainer('Nobody Random', db=None,
                                                       before_date='')
        assert elite_live[0] > unknown_live[0]


# ---------------------------------------------------------------------------
# #5 — bankroll guard is ENFORCED by the generator finalize step
# ---------------------------------------------------------------------------

def _unresolved_win_race():
    """A HIGH-conviction WIN race that earns a non-zero suggested stake."""
    horses = [
        {'number': 1, 'name': 'TOP', 'rank': 1, 'bet_recommendation': 'WIN',
         'win_odds': 4.0, 'calibrated_prob': 0.40},
        {'number': 2, 'name': 'SECOND', 'rank': 2, 'bet_recommendation': 'SKIP',
         'win_odds': 6.0, 'calibrated_prob': 0.20},
        {'number': 3, 'name': 'THIRD', 'rank': 3, 'bet_recommendation': 'SKIP',
         'win_odds': 9.0, 'calibrated_prob': 0.10},
    ]
    return {'race_number': 1, 'horses': horses, 'conviction_tier': 'HIGH',
            'top_2_share': 0.6, 'results_available': False}


def _resolved_race():
    return {'race_number': 2,
            'horses': [{'number': 1, 'name': 'DONE', 'rank': 1,
                        'bet_recommendation': 'WIN', 'win_odds': 3.0,
                        'calibrated_prob': 0.5, 'actual_position': 1}],
            'conviction_tier': 'HIGH', 'results_available': True}


def _patch_guard(monkeypatch, *, halted, reason=''):
    import betting.bankroll_guard as bg
    monkeypatch.setattr(
        bg, 'daily_status',
        lambda *a, **k: types.SimpleNamespace(halted=halted, halt_reason=reason),
    )


class TestBankrollGuardEnforced:
    def test_halted_day_zeroes_stake_and_flags(self, monkeypatch):
        from dashboard.generator import finalize_tickets_and_stakes
        _patch_guard(monkeypatch, halted=True,
                     reason='DAILY LOSS CAP (-8%)')
        races = [_unresolved_win_race(), _resolved_race()]
        finalize_tickets_and_stakes(races, '2026/06/07', object())

        # Unresolved race: stake zeroed, halt reason stamped -> NO SYSTEM BET.
        assert races[0]['bankroll_halt'] == 'DAILY LOSS CAP (-8%)'
        assert races[0]['suggested_stake']['amount'] == 0.0
        assert 'STOP' in races[0]['suggested_stake']['method']
        # The FROZEN TICKET ITSELF must carry no payable bets — the WA-copy +
        # visual ticket render from bet_ticket.bets, not suggested_stake, so a
        # stopped day must not leave a copyable live ticket (audit 2026-06-08 #1).
        tk = races[0]['bet_ticket']
        assert tk['bets'] == []
        assert tk['total_combos'] == 0
        assert tk['allocated_amount'] == 0.0
        assert tk['unit_stake'] == 0.0
        assert tk.get('bankroll_halt') == 'DAILY LOSS CAP (-8%)'
        # Resolved race must NOT be touched by the halt gate.
        assert 'bankroll_halt' not in races[1]

    def test_non_halted_day_keeps_payable_ticket(self, monkeypatch):
        """Control: a non-halted HIGH WIN race keeps a real payable ticket."""
        from dashboard.generator import finalize_tickets_and_stakes
        _patch_guard(monkeypatch, halted=False)
        races = [_unresolved_win_race()]
        finalize_tickets_and_stakes(races, '2026/06/07', object())
        tk = races[0]['bet_ticket']
        assert tk['bets'], "non-halted WIN race should produce a payable ticket"
        assert tk['allocated_amount'] > 0.0
        assert 'bankroll_halt' not in tk

    def test_non_halted_day_leaves_stake(self, monkeypatch):
        from dashboard.generator import finalize_tickets_and_stakes
        _patch_guard(monkeypatch, halted=False)
        races = [_unresolved_win_race()]
        finalize_tickets_and_stakes(races, '2026/06/07', object())

        assert 'bankroll_halt' not in races[0]
        # The HIGH WIN race should carry a real (non-zero) suggested stake.
        assert races[0]['suggested_stake']['amount'] > 0.0

    def test_guard_failure_never_blocks_generation(self, monkeypatch):
        """A guard exception must be swallowed — generation continues."""
        import betting.bankroll_guard as bg

        def _boom(*a, **k):
            raise RuntimeError("db gone")

        monkeypatch.setattr(bg, 'daily_status', _boom)
        from dashboard.generator import finalize_tickets_and_stakes
        races = [_unresolved_win_race()]
        finalize_tickets_and_stakes(races, '2026/06/07', object())  # must not raise
        assert 'bankroll_halt' not in races[0]


# ---------------------------------------------------------------------------
# #1 — dashboard generation already fail-closes on a bad blend calibrator
# ---------------------------------------------------------------------------

class TestBlendCalibratorFailClosed:
    def test_blend_calibrator_artifact_rejects_missing_file(self):
        from pathlib import Path
        from ml.integrity import check_blend_calibrator_artifact, IntegrityError
        with pytest.raises(IntegrityError, match='missing'):
            check_blend_calibrator_artifact(Path('/nonexistent/blend_calibrator.pkl'))

    def test_preflight_aborts_generation_on_integrity_failure(self):
        """Lock the wiring: generate_predictions runs the fail-closed preflight
        and refuses to generate (SystemExit) when it raises. Guards against a
        silent revert to fail-open (audit 2026-06-08 #1)."""
        import inspect
        import dashboard.generator as gen

        src = inspect.getsource(gen)
        assert 'preflight_dashboard_ml()' in src
        # The preflight call site must abort (SystemExit) on failure.
        assert 'raise SystemExit(2)' in src
        # And the preflight must route through the blend-calibrator validator.
        pf_src = inspect.getsource(gen.preflight_dashboard_ml)
        assert 'check_dashboard_ml_integrity' in pf_src
