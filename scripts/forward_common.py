"""Shared helpers for forward-validation scripts.

Currently exports `bayesian_verdict()` — an exact Beta-Binomial posterior
with an N>=MIN_MEETINGS floor, used by shadow_evaluate, stake_ab,
track_longshot_trio, and tier_b_evaluate to make SHIP/KILL/WATCH
decisions on per-meeting binary outcomes. (The posterior is NOT
optional-stopping-safe here because the inputs are autocorrelated
cumulative-ROI outcomes, not i.i.d. draws — see bayesian_verdict's NOTE;
the MIN_MEETINGS floor is the small-N guard.)

Why this lives in its own module: extracted from inline copies in
shadow_evaluate + stake_ab on 2026-05-23 after the software-architect
review (4-expert dispatch) flagged the duplication + the divergence
with the older arbitrary "N>=6" heuristic in longshot_trio + tier_b
as a maintenance risk. One source of truth so all forward-validation
trackers ship/kill on the same statistical foundation.
"""
from __future__ import annotations

from math import comb

SHIP_THRESHOLD = 0.95
KILL_THRESHOLD = 0.05
# Minimum meetings before any SHIP/KILL fires. The inputs are CUMULATIVE-ROI
# through-meeting outcomes, which are positively autocorrelated (once cumulative
# ROI crosses it tends to stay crossed), so a clean lead from meeting 1 inflates
# the posterior — 4/4 all-wins already clears 0.95. Treating autocorrelated
# outcomes as i.i.d. overstates the effective sample size, so we hard-floor the
# verdict at N meetings to match the project's honest-posture gate. The posterior
# is still reported for transparency, but the VERDICT stays WATCH below the floor.
# (10-expert review 2026-06-03, finding: "can SHIP at N=4".)
MIN_MEETINGS = 6


def _beta_survival_at_half(wins: int, losses: int) -> float:
    """Exact P(Beta(1+wins, 1+losses) > 0.5) under a flat Beta(1,1) prior.

    For integer shape parameters the regularized incomplete beta at x=0.5 has a
    closed binomial form, so this is exact and dependency-free (no scipy):
        I_0.5(a, b) = 0.5^(a+b-1) * sum_{j=a}^{a+b-1} C(a+b-1, j),  a=1+wins, b=1+losses
        survival   = 1 - I_0.5(a, b)
    Matches scipy.stats.beta.sf(0.5, 1+wins, 1+losses) to machine precision.
    """
    n = wins + losses          # a + b - 1 == n + 1
    total = n + 1
    tail = sum(comb(total, j) for j in range(1 + wins, total + 1))
    return 1.0 - tail * (0.5 ** total)


def bayesian_verdict(meeting_winners: list[int]) -> dict:
    """Beta-Binomial posterior P(variant beats baseline) over per-meeting
    binary outcomes, with a hard N>=MIN_MEETINGS floor on the verdict.

    Args:
        meeting_winners: 1 if variant cumulative ROI > baseline cumulative
            ROI at that meeting, else 0. One entry per meeting in
            chronological order. (Cumulative-ROI through-meeting is
            preferred over per-meeting ROI so a single noisy meeting can't
            flip the verdict.)

    Returns:
        dict with keys:
          n_meetings: int — sample size
          wins:       int — meetings where variant beat baseline
          posterior:  float in [0, 1] — exact survival of Beta(1+wins,1+losses)
                      at 0.5 under a flat Beta(1,1) prior
          verdict:    'SHIP' (posterior >= 0.95 AND n >= MIN_MEETINGS) /
                      'KILL' (posterior <= 0.05 AND n >= MIN_MEETINGS) /
                      'WATCH' (otherwise, incl. all n < MIN_MEETINGS)

    NOTE on optional stopping: a Bayesian posterior is invariant to optional
    stopping ONLY when the modeled likelihood (independent Bernoulli draws)
    matches the data. Here the inputs are cumulative-through-meeting outcomes,
    which are autocorrelated, so that invariance does NOT transfer and the
    posterior overstates confidence at small N. The MIN_MEETINGS floor is the
    guard against shipping on that inflated small-N confidence; it is not a
    substitute for the production gate (validate_hk over N>=250 bets).
    """
    n = len(meeting_winners)
    if n == 0:
        return {'n_meetings': 0, 'wins': 0,
                'posterior': 0.5, 'verdict': 'WATCH'}
    wins = sum(meeting_winners)
    losses = n - wins
    posterior = _beta_survival_at_half(wins, losses)

    if n < MIN_MEETINGS:
        verdict = 'WATCH'          # not enough meetings to ship/kill
    elif posterior >= SHIP_THRESHOLD:
        verdict = 'SHIP'
    elif posterior <= KILL_THRESHOLD:
        verdict = 'KILL'
    else:
        verdict = 'WATCH'
    return {'n_meetings': n, 'wins': wins,
            'posterior': posterior, 'verdict': verdict}
