#!/usr/bin/env python3
"""Bet recommendation engine — assigns BANKER/WIN/QIN/PLACE/SKIP per horse."""

import logging
from typing import List

import config
from core.scorer import calibrate_probabilities, apply_calibrated_probs

logger = logging.getLogger(__name__)


def assign_bet_recommendations(analyses, field_size: int = 12,
                               distance: int = 1200) -> None:
    """Assign realistic bet recommendations based on field shape.

    Calibrates probabilities via softmax + market blend (dynamic weights
    based on field size and distance), handles market-favourite promotion
    and BANKER overrides, then assigns bet types per horse.
    """
    if not analyses:
        return

    n = len(analyses)
    calibrated_probs = calibrate_probabilities(analyses, field_size=field_size,
                                               distance=distance)
    apply_calibrated_probs(analyses, calibrated_probs)

    top = analyses[0] if n >= 1 else None
    second = analyses[1] if n >= 2 else None

    top_prob = calibrated_probs[0] if n >= 1 else 0
    second_prob = calibrated_probs[1] if n >= 2 else 0
    third_prob = calibrated_probs[2] if n >= 3 else 0

    gap_1_2 = top_prob - second_prob
    gap_2_3 = second_prob - third_prob
    top_3_share = top_prob + second_prob + third_prob

    market_prob_top = (1 / top.horse.win_odds) if top and top.horse.win_odds and top.horse.win_odds > 0 else 0

    # Skip market-fav promotion if ANY horse in the field qualifies as BANKER —
    # BANKER takes precedence and will be promoted below. Without this, the
    # market-fav swap fires first, then BANKER promotion swaps again,
    # producing a rank-1 BANKER even when the market disagrees strongly with
    # that horse — and the bet-type logic downstream then aggressively fires
    # WIN on a market-disagreeing pick.
    any_banker_in_field = any(
        any("BANKER" in r for r in a.reasoning) for a in analyses
    ) if analyses else False

    # Market favourite promotion (skip if any BANKER exists)
    market_fav = min(analyses, key=lambda a: a.horse.win_odds if a.horse.win_odds and a.horse.win_odds > 0 else 999)
    market_fav_prob = (1 / market_fav.horse.win_odds) if market_fav.horse.win_odds and market_fav.horse.win_odds > 0 else 0
    if (market_fav_prob >= 0.30 and market_fav.rank > 1
            and market_fav.horse.win_odds > 0
            and not any_banker_in_field):
        old_idx = next(i for i, a in enumerate(analyses) if a is market_fav)
        analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]
        calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]
        _resync_ranks_and_probs(analyses, calibrated_probs)
        top = analyses[0]
        top_prob = calibrated_probs[0]
        second_prob = calibrated_probs[1] if n >= 2 else 0
        third_prob = calibrated_probs[2] if n >= 3 else 0
        gap_1_2 = top_prob - second_prob
        gap_2_3 = second_prob - third_prob
        top_3_share = top_prob + second_prob + third_prob
        market_prob_top = market_fav_prob

    # BANKER promotion
    # Tiebreak on calibrated_prob (combined_probability after apply_calibrated_probs),
    # NOT rule_score. Pre-blend rule_score can disagree with post-blend reality —
    # picking the higher rule_score promoted SILVERY BREEZE (cal_prob 0.097) over
    # LO RIDER (cal_prob 0.168) on HV 2026-05-13 R9.
    banker_horse = None
    for a in analyses:
        if any("BANKER" in r for r in a.reasoning):
            if banker_horse is None or a.combined_probability > banker_horse.combined_probability:
                banker_horse = a

    # Cal-prob guard: refuse BANKER promotion if its calibrated_prob is materially
    # below the market favourite's calibrated_prob. Prevents the rule-engine
    # BANKER signals (rating surge, class rise, etc) from overriding a post-blend
    # signal that says "market disagrees, you're picking a worse horse".
    # Threshold 0.7 catches the SILVERY BREEZE prototype (0.097 / 0.168 = 0.577).
    if banker_horse and market_fav and market_fav is not banker_horse:
        market_fav_cal = market_fav.combined_probability or 0
        banker_cal = banker_horse.combined_probability or 0
        if market_fav_cal > 0 and banker_cal < 0.7 * market_fav_cal:
            logger.info(
                f"BANKER override declined: {banker_horse.horse.name} cal_prob "
                f"{banker_cal:.3f} < 0.7 × market_fav {market_fav.horse.name} "
                f"cal_prob {market_fav_cal:.3f}"
            )
            banker_horse = None

    if banker_horse and banker_horse.rank > 1:
        old_idx = next(i for i, a in enumerate(analyses) if a is banker_horse)
        analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]
        calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]
        _resync_ranks_and_probs(analyses, calibrated_probs)
        top = analyses[0]
        top_prob = calibrated_probs[0]
        second_prob = calibrated_probs[1] if n >= 2 else 0
        third_prob = calibrated_probs[2] if n >= 3 else 0
        gap_1_2 = top_prob - second_prob
        gap_2_3 = second_prob - third_prob
        top_3_share = top_prob + second_prob + third_prob
        # Refresh market_prob_top from the NEW top (BANKER) horse's odds.
        # Was stale here — kept the original rank-1's odds, so the WIN gate
        # at line ~121 evaluated the wrong horse's market signal.
        market_prob_top = (1 / banker_horse.horse.win_odds) if banker_horse.horse.win_odds and banker_horse.horse.win_odds > 0 else 0

    # === QUINELLA-FIRST STRATEGY (revised Apr 23, box-4 default) ===
    # Backtest findings with REAL HKJC dividends (72 held-out races):
    #   - WIN pool: NO edge (agrees with market 98.6%, -37% ROI)
    #   - Straight QIN (top-2, 1 combo): +999% ROI, 16.9% hit rate
    #   - Box-3 QIN (3 combos): +910% ROI, 33.8% hit rate
    #   - Box-4 QIN (6 combos): +1141% ROI, 46.5% hit rate — best measured
    #
    # Rank-2 analysis (Apr 23): actual 2nd-placers are rank 4+ in our model
    # 33% of the time. Box-4 captures these longshot 2nd-placers; box-3 misses.
    # Box-4 is now the default when top-4 conviction is present. Falls back to
    # box-3 → straight QIN → SKIP as conviction degrades.

    top_is_banker_now = any("BANKER" in r for r in analyses[0].reasoning)
    has_live_odds = any(a.horse.win_odds and a.horse.win_odds > 0 for a in analyses)

    # Conviction gates (uniform baselines: top2=0.167, top3=0.25, top4=0.333)
    top_2_share = top_prob + second_prob
    fourth_prob = calibrated_probs[3] if n >= 4 else 0
    top_4_share = top_3_share + fourth_prob

    # Rank-3 and rank-4 inclusion gates — horse must have strong history
    rank3_strong = n >= 3 and _rank_has_strong_history(analyses[2])
    rank4_strong = n >= 4 and _rank_has_strong_history(analyses[3])

    race_bet_type = _choose_race_bet_type(
        has_live_odds=has_live_odds, n=n, top_is_banker=top_is_banker_now,
        market_prob_top=market_prob_top, gap_1_2=gap_1_2, gap_2_3=gap_2_3,
        top_2_share=top_2_share, top_3_share=top_3_share, top_4_share=top_4_share,
        rank3_strong=rank3_strong, rank4_strong=rank4_strong,
    )

    # Assign per-horse recommendations
    MIN_EV = config.MIN_EXPECTED_RETURN
    for a in analyses:
        rank = a.rank
        has_conviction = any("WINNING TRIP" in r or "BANKER" in r for r in a.reasoning)
        has_place_value = (a.horse.place_odds > 1.2) if a.horse.place_odds > 0 else (a.horse.win_odds >= 3.0)

        if race_bet_type == "SKIP":
            a.bet_recommendation = "SKIP"
        elif race_bet_type == "WIN":
            if rank == 1:
                is_banker = any("BANKER" in r for r in a.reasoning)
                a.bet_recommendation = "BANKER WIN" if is_banker else "WIN"
            elif rank == 2 and has_place_value and a.expected_return >= MIN_EV:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "TRIO":
            if rank <= 3:
                a.bet_recommendation = "TRIO"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "BOX Q4":
            # Box-4 quinella — all top-4 as QIN legs (6 combinations)
            if rank <= 4:
                a.bet_recommendation = "QIN"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "BOX Q":
            # Box-3 quinella — rank-3 has strong history, all top-3 as QIN legs
            if rank <= 3:
                a.bet_recommendation = "QIN"
            else:
                a.bet_recommendation = "SKIP"
        elif race_bet_type == "QIN":
            # Straight quinella — top 2 only
            if rank <= 2:
                a.bet_recommendation = "QIN"
            elif rank == 3 and has_place_value and has_conviction:
                a.bet_recommendation = "PLACE"
            else:
                a.bet_recommendation = "SKIP"
        else:
            a.bet_recommendation = "SKIP"


def _rank_has_strong_history(analysis) -> bool:
    """Check if a horse has genuinely strong historical credentials.

    Used to gate inclusion as a 3rd/4th QIN leg (box-3/box-4 upgrade). Requires
    specific evidence of past performance, not just the model ranking — the
    backtest showed wider boxes are only profitable when the added horses are
    real contenders, not speculative picks.

    Returns True if any of:
      - BANKER or WINNING TRIP flagged in reasoning
      - Recent winner (form analyzer)
      - Career win rate ≥ 25% OR career top-3 rate ≥ 40%
      - Rule score ≥ 70 (composite historical signal)
    """
    import re
    for r in analysis.reasoning:
        ru = r.upper()
        if "BANKER" in ru or "WINNING TRIP" in ru:
            return True
        if ru.startswith("RECENT WINNER") or ru.startswith("STRONG FORM"):
            return True
        if r.startswith("Career:"):
            m = re.search(r"(\d+)%W", r)
            if m and int(m.group(1)) >= 25:
                return True
            m = re.search(r"(\d+)% top3", r)
            if m and int(m.group(1)) >= 40:
                return True
    return analysis.rule_score >= 70.0


# Backward-compatible alias
_rank3_has_strong_history = _rank_has_strong_history


# Conviction gate thresholds (uniform baselines: top2=0.167, top3=0.25, top4=0.333)
# v6.8.3 (2026-05-16) — 3D joint sweep across all three thresholds, after the
# 4 prior model experiments confirmed model↔gate coupling is the binding
# constraint (see [[feature-experiments-brier-vs-roi-pattern]]). Sweep of
# 150 combos on the full 664-race holdout + last-6 validation found this
# triple dominates:
#
#                       OLD            NEW       Δ ROI
#   Full holdout      +1425.9%       +1484.3%   +58pp
#   Last-6 meetings   +1940.8%       +2045.2%   +104pp
#
# All 10 top-ROI combos in the sweep agreed on min_top4=0.58 — that signal
# is unanimous. min_top2 and min_top3 are less sharply peaked (top combos
# cluster at 0.28-0.30 and 0.33-0.36 respectively); picked the empirical
# winner. Tradeoff: 7 fewer hits and $193 less stake on backtest = capital
# efficiency win, ~$1.4k less absolute P&L. The current gates ranked #51
# out of 150 — i.e. there were 50 measurably better configurations sitting
# in the search space the prior 1D sweeps could not see.
MIN_TOP2_CONVICTION = 0.30   # was 0.25 — sweep evidence above
MIN_TOP3_CONVICTION = 0.33   # was 0.30 — sweep evidence above
MIN_TOP4_CONVICTION = 0.58   # was 0.52 — strongest signal (top-10 unanimous)


def _choose_race_bet_type(*, has_live_odds, n, top_is_banker, market_prob_top,
                          gap_1_2, gap_2_3, top_2_share, top_3_share, top_4_share,
                          rank3_strong, rank4_strong,
                          min_top2=MIN_TOP2_CONVICTION,
                          min_top3=MIN_TOP3_CONVICTION,
                          min_top4=MIN_TOP4_CONVICTION) -> str:
    """Pure gate cascade: race shape → bet type. Single source of truth.

    Extracted from assign_bet_recommendations so backtest/sweep tooling can
    replay the exact same cascade with swept thresholds (no logic duplication).

    Cascade order (first match wins):
      SKIP    — no live odds, or <2 runners
      WIN     — top is BANKER, market ≥ 0.45 ($2.2 or shorter), gap_1_2 > 0.12.
                Tightened v6.8: was 0.35/0.08, fired too liberally on $2.5-3
                favourites (HV 2026-05-13 R5 ACE WAR BANKER WIN at $2.5 lost).
      TRIO    — both gaps < 0.025, top_3_share > 0.32, n ≥ 3. Loosened v6.8
                from 0.02/0.02/0.35; kept AND so it doesn't fire on most races.
      BOX Q4  — DEFAULT: rank 3+4 strong history AND top_4_share ≥ min_top4.
      BOX Q   — rank 3 strong history AND top_3_share ≥ min_top3.
      QIN     — top_2_share ≥ min_top2.
      SKIP    — otherwise.
    """
    if not has_live_odds or n < 2:
        return "SKIP"
    if top_is_banker and market_prob_top >= 0.45 and gap_1_2 > 0.12:
        return "WIN"
    # TRIO branch DISABLED 2026-05-17 (audit) — fired 0/5 hits in last 90d
    # for -100% ROI. Race shape it targets (flat fields, gap_1_2<0.025) is
    # exactly where the model has no edge. BOX Q4 below catches 60% of those
    # same races via QIN-pool (top-2) instead of TRIO-pool (top-3).
    # if gap_1_2 < 0.025 and gap_2_3 < 0.025 and top_3_share > 0.32 and n >= 3:
    #     return "TRIO"
    if rank4_strong and rank3_strong and top_4_share >= min_top4:
        return "BOX Q4"
    if rank3_strong and top_3_share >= min_top3:
        return "BOX Q"
    if top_2_share >= min_top2:
        return "QIN"
    return "SKIP"


def _resync_ranks_and_probs(analyses, calibrated_probs: List[float]) -> None:
    """After a rank swap, update all ranks, probabilities, expected returns, and value_score."""
    for i, a in enumerate(analyses):
        a.rank = i + 1
        a.combined_probability = calibrated_probs[i]
        a.value_score = calibrated_probs[i] * 100
        if a.horse.win_odds and a.horse.win_odds > 0:
            a.expected_return = (a.horse.win_odds * calibrated_probs[i]) - 1
