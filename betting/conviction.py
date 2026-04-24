"""Conviction tier assignment — replaces strategy_confidence.

Tiers: EDGE | HIGH | MED | LOW | SKIP
EDGE takes precedence over pair-conviction tiers. See spec §4.
"""

from dataclasses import dataclass


@dataclass
class ConvictionThresholds:
    """Spec §4 defaults."""
    high_top2: float = 0.33
    high_top3: float = 0.42
    med_top2: float = 0.27
    med_top3: float = 0.35
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
