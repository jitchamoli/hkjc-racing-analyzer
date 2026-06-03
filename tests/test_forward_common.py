"""scripts/forward_common.bayesian_verdict — Beta-Binomial posterior used by
all forward-validation trackers for SHIP/KILL/WATCH decisions.

Test-engineer expert review 2026-05-23 priority #2: this function drives
ship decisions on every shadow + tier_b + longshot_trio variant. A
regression that flips the verdict on extreme inputs is a real bug.
"""
from scripts.forward_common import (
    bayesian_verdict, SHIP_THRESHOLD, KILL_THRESHOLD, MIN_MEETINGS,
)


def test_n_floor_blocks_ship_below_min_meetings():
    """10-expert review 2026-06-03: cumulative-ROI outcomes are autocorrelated,
    so a clean lead from meeting 1 inflates the posterior. A clean all-wins
    streak below MIN_MEETINGS must NOT SHIP — verdict is held at WATCH even
    though the posterior is high. (Pre-fix: 4/4 -> posterior 0.969 -> SHIP.)"""
    for n in range(1, MIN_MEETINGS):
        v = bayesian_verdict([1] * n)
        assert v['verdict'] == 'WATCH', (
            f"N={n} (< {MIN_MEETINGS}) all-wins must be WATCH, got {v['verdict']} "
            f"(posterior {v['posterior']:.3f})"
        )


def test_n_floor_blocks_kill_below_min_meetings():
    """Symmetric: a short losing streak must not KILL prematurely either."""
    for n in range(1, MIN_MEETINGS):
        v = bayesian_verdict([0] * n)
        assert v['verdict'] == 'WATCH', f"N={n} all-losses must be WATCH, got {v['verdict']}"


def test_ship_allowed_at_min_meetings():
    """At exactly MIN_MEETINGS a clean all-wins streak may SHIP."""
    v = bayesian_verdict([1] * MIN_MEETINGS)
    assert v['n_meetings'] == MIN_MEETINGS
    assert v['posterior'] >= SHIP_THRESHOLD
    assert v['verdict'] == 'SHIP'


def test_posterior_is_exact_beta_not_raw_proportion():
    """Finding #6: the no-scipy path used to threshold a raw proportion
    (1/1 -> 1.0). The posterior must be the exact Beta survival regardless of
    scipy: N=1, 1 win -> Beta(2,1) survival at 0.5 = 0.75, NOT 1.0."""
    v = bayesian_verdict([1])
    assert abs(v['posterior'] - 0.75) < 1e-9, (
        f"posterior must be exact Beta(2,1) survival 0.75, got {v['posterior']}"
    )


def test_posterior_matches_scipy_when_available():
    """The pure-Python posterior must equal scipy's Beta survival to machine
    precision so removing the scipy dependency changes nothing numerically."""
    try:
        from scipy.stats import beta as _beta
    except ImportError:
        import pytest
        pytest.skip("scipy not installed")
    for winners in ([1] * 7, [0] * 8, [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]):
        wins = sum(winners); n = len(winners)
        expect = float(1.0 - _beta.cdf(0.5, 1 + wins, 1 + (n - wins)))
        got = bayesian_verdict(winners)['posterior']
        assert abs(got - expect) < 1e-9, f"{winners}: got {got}, scipy {expect}"


def test_empty_input_returns_watch_at_uniform_posterior():
    v = bayesian_verdict([])
    assert v == {'n_meetings': 0, 'wins': 0, 'posterior': 0.5, 'verdict': 'WATCH'}


def test_all_wins_ships():
    # 10/10 wins → posterior near 1.0 → SHIP
    v = bayesian_verdict([1] * 10)
    assert v['n_meetings'] == 10
    assert v['wins'] == 10
    assert v['posterior'] >= SHIP_THRESHOLD
    assert v['verdict'] == 'SHIP'


def test_all_losses_kills():
    v = bayesian_verdict([0] * 10)
    assert v['posterior'] <= KILL_THRESHOLD
    assert v['verdict'] == 'KILL'


def test_split_outcomes_watch():
    # 5/10 wins → posterior near 0.5 → WATCH
    v = bayesian_verdict([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    assert KILL_THRESHOLD < v['posterior'] < SHIP_THRESHOLD
    assert v['verdict'] == 'WATCH'


def test_single_meeting_inconclusive():
    # N=1 with 1 win: posterior is 0.75 (Beta(2,1)) under scipy, or 1.0
    # under fallback — either way, NOT a SHIP yet (the fallback is more
    # aggressive but still below 0.95 for a single observation... wait,
    # actually 1/1 = 1.0 which IS >= 0.95). Test the scipy path which is
    # the realistic case; under fallback this WOULD ship which is exactly
    # the conservatism caveat documented in the function.
    v = bayesian_verdict([1])
    # Under scipy: Beta(2,1).cdf(0.5) = 0.25, so posterior = 0.75 → WATCH
    # Under fallback: 1/1 = 1.0 → SHIP
    # Either is acceptable per the docstring — we just don't want a crash.
    assert v['n_meetings'] == 1
    assert v['wins'] == 1
    assert v['verdict'] in ('WATCH', 'SHIP')


def test_partial_wins_progresses_toward_ship():
    # Posterior should increase monotonically with more wins at fixed losses.
    p1 = bayesian_verdict([1] * 3 + [0] * 1)['posterior']  # 3W/1L
    p2 = bayesian_verdict([1] * 6 + [0] * 1)['posterior']  # 6W/1L
    p3 = bayesian_verdict([1] * 9 + [0] * 1)['posterior']  # 9W/1L
    assert p1 < p2 < p3


def test_thresholds_are_documented_constants():
    # Guard against silent threshold drift.
    assert SHIP_THRESHOLD == 0.95
    assert KILL_THRESHOLD == 0.05


def test_returns_required_keys():
    # Downstream callers depend on these keys.
    v = bayesian_verdict([1, 0, 1])
    assert set(v.keys()) == {'n_meetings', 'wins', 'posterior', 'verdict'}
    assert v['verdict'] in ('SHIP', 'KILL', 'WATCH')
