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
    New 0.40/0.55: HIGH share dropped to 46%, hit-rate +7pp.

    v6.11 (2026-05-24) prune-and-revert episode: I patched these to +0.15 after
    observing v6.10 → v6.11 shifted today's race-card top_2_share mean from
    0.30 → 0.49. Quantile-matched analysis over 300 historical races
    (scripts/quantile_match_gates.py) showed the TRUE distribution shift is
    only +0.007 — today's card was an outlier (clear-fav-heavy: R3, R4, R8,
    R11). Reverted back to 0.40/0.55/0.30/0.42/0.22 (quantile-matched values
    were within 0.013 of current — effectively identical). Memory:
    `feedback_small_sample_inflation_pattern` — tuning on n=11 is exactly
    the trap to avoid.
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
# HKJC short codes for genuinely soft going (Yielding / Yielding-to-Soft / Soft /
# Heavy / Wet-Slow). Carried separately because none contains a full-word
# keyword. Live cards use full strings; short codes appear in the frozen Renavon
# rows and any DB-sourced going, so normalising them here keeps the filter and
# its diagnostics consistent (5-expert review 2026-07-15).
_SOFT_GOING_SHORT_CODES = frozenset({'Y', 'YS', 'S', 'H', 'WS'})


def detect_chaos(going: str, horses: list = None,
                 class_type: str = '') -> tuple:
    """Return (is_chaos, trigger_label). Fires on genuinely soft going only.

    NARROWED 2026-07-15 (5-expert review) to EXCLUDE 'Good To Yielding' and its
    short code GY. On 154 full-DB races the favourite strikes ~31% on GY —
    indistinguishable from firm (31.6%) — and the model's own rank-1 place rate
    is intact; the genuine degradation lives in the Yielding/Soft/Heavy bucket.
    GY was only ever caught by the naive `'YIELDING' in 'GOOD TO YIELDING'`
    substring.

    This is a VARIANCE / DRAWDOWN guard, NOT a demonstrated EV edge: on the BOX
    products actually staked the soft-vs-firm ROI gap is ~0 (the shipped
    −45.6pp / posterior-0.996 was small-sample × box-variance; the honest
    population effect ~−7 to −10pp lives only in straight-favourite bets). Keep
    it to avoid blow-up meetings; do not count it as edge. See memory:
    project_chaos_filter_2026-05-23.

    horses + class_type accepted for forward-compat with two pending triggers.
    """
    g = (going or '').strip().upper()
    if not g:
        return False, ''
    # Good-To-Yielding behaves like firm — exclude it (and GY) before the scan.
    if 'GOOD TO YIELDING' in g or g == 'GY':
        return False, ''
    if any(kw in g for kw in SOFT_GOING_KEYWORDS) or g in _SOFT_GOING_SHORT_CODES:
        return True, 'soft_going'
    # TODO (await N≥X meetings of evidence):
    #   - 4+ horses returning from 60+d break: fired 0× on 10-meeting backfill
    #   - Class 5 + 10+ field + top-3 odds within HK$2: fired 0× same
    return False, ''
