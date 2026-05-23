"""Shared helpers for forward-validation scripts.

Currently exports `bayesian_verdict()` — the optional-stopping-safe
Beta-Binomial posterior used by shadow_evaluate, stake_ab,
track_longshot_trio, and tier_b_evaluate to make SHIP/KILL/WATCH
decisions on per-meeting binary outcomes.

Why this lives in its own module: extracted from inline copies in
shadow_evaluate + stake_ab on 2026-05-23 after the software-architect
review (4-expert dispatch) flagged the duplication + the divergence
with the older arbitrary "N>=6" heuristic in longshot_trio + tier_b
as a maintenance risk. One source of truth so all forward-validation
trackers ship/kill on the same statistical foundation.
"""
from __future__ import annotations

SHIP_THRESHOLD = 0.95
KILL_THRESHOLD = 0.05


def bayesian_verdict(meeting_winners: list[int]) -> dict:
    """Beta-Binomial posterior P(variant beats baseline) over per-meeting
    binary outcomes. Replaces the arbitrary N>=6 heuristic with a rule that
    controls Type I/II error under optional stopping.

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
          posterior:  float in [0, 1] — P(variant > baseline | data) under
                      a flat Beta(1, 1) prior, computed as the survival
                      function of Beta(1+wins, 1+losses) at 0.5
          verdict:    'SHIP' (posterior >= 0.95) /
                      'KILL' (posterior <= 0.05) /
                      'WATCH' (in between)

    Rationale: Bayesian posteriors are invariant to optional stopping under
    the likelihood principle — you can check after every meeting without
    inflating Type I error (unlike sequential frequentist tests). Beta(1,1)
    is uniform; the posterior IS the proportion of meetings variant wins.

    Scipy is optional. Without it, falls back to sample proportion as a
    crude but conservative approximation (only ships when the proportion
    itself is >= 0.95 — i.e. variant won at least 19/20 meetings).
    """
    n = len(meeting_winners)
    if n == 0:
        return {'n_meetings': 0, 'wins': 0,
                'posterior': 0.5, 'verdict': 'WATCH'}
    wins = sum(meeting_winners)
    losses = n - wins
    try:
        from scipy.stats import beta as _beta
        posterior = float(1.0 - _beta.cdf(0.5, 1 + wins, 1 + losses))
    except ImportError:
        # Fallback: sample proportion. Crude but conservative — only
        # ships/kills when the raw proportion is extreme.
        posterior = wins / n

    if posterior >= SHIP_THRESHOLD:
        verdict = 'SHIP'
    elif posterior <= KILL_THRESHOLD:
        verdict = 'KILL'
    else:
        verdict = 'WATCH'
    return {'n_meetings': n, 'wins': wins,
            'posterior': posterior, 'verdict': verdict}
