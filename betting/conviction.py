"""Conviction tier assignment — replaces strategy_confidence.

Tiers: EDGE | HIGH | MED | LOW | SKIP
EDGE takes precedence over pair-conviction tiers. See spec §4.
"""

from dataclasses import dataclass


@dataclass
class ConvictionThresholds:
    """Conviction tier thresholds.

    Raised 2026-05-17 after backtest (40-race, 90d window) showed the original
    0.33/0.42 thresholds labeled 74% of races as HIGH — no tier discrimination.
    New 0.40/0.55: HIGH share drops to 46%, HIGH hit-rate +7pp (20.7→27.8%),
    and MED becomes informative (was 0/7 hits, now 1/13). LOW tightened
    proportionally so a real low-conviction race still surfaces as LOW.
    """
    high_top2: float = 0.40
    high_top3: float = 0.55
    med_top2: float = 0.30
    med_top3: float = 0.42
    low_top2: float = 0.22


def assign_conviction_tier(
    has_edge: bool, top_2_share: float, top_3_share: float,
    thresholds: ConvictionThresholds = None,
) -> str:
    """Return the conviction tier for a race.

    EDGE takes precedence. Otherwise tests are OR'd on (top_2, top_3).
    """
    t = thresholds or ConvictionThresholds()
    if has_edge:
        return 'EDGE'
    if top_2_share >= t.high_top2 or top_3_share >= t.high_top3:
        return 'HIGH'
    if top_2_share >= t.med_top2 or top_3_share >= t.med_top3:
        return 'MED'
    if top_2_share >= t.low_top2:
        return 'LOW'
    return 'SKIP'
