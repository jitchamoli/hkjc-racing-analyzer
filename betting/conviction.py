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


# CHAOS tier sits ABOVE SKIP — suppresses ALL bets (not just exotic boxes).
# Shipped 2026-05-23 from the 15-expert audit + cheap diagnostic (posterior 0.996,
# Δ −45.6pp on 10-meeting burned holdout). Only the soft-going trigger has
# empirical evidence at our sample size; the other two HK-pro-suggested triggers
# (≥4 horses 60+d break, Class 5 + 10+ field + top-3 within $2) fired 0 times in
# the diagnostic — kept as TODO comments awaiting forward evidence.
SOFT_GOING_KEYWORDS = ('YIELDING', 'SOFT', 'HEAVY', 'WET-SLOW', 'WET SLOW')


def detect_chaos(going: str, horses: list = None,
                 class_type: str = '') -> tuple:
    """Return (is_chaos, trigger_label).
    Soft-going only; horses + class_type accepted for forward-compat with the
    two pending triggers (not yet active)."""
    g = (going or '').upper()
    if any(kw in g for kw in SOFT_GOING_KEYWORDS):
        return True, 'soft_going'
    # TODO (await N≥X meetings of evidence):
    #   - 4+ horses returning from 60+d break: fired 0× on 10-meeting backfill
    #   - Class 5 + 10+ field + top-3 odds within HK$2: fired 0× same
    return False, ''
