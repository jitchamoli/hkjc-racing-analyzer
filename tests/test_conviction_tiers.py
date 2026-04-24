"""Tier assignment at spec boundaries."""
import pytest
from betting.conviction import assign_conviction_tier, ConvictionThresholds


def test_edge_takes_precedence_over_high():
    tier = assign_conviction_tier(has_edge=True, top_2_share=0.50, top_3_share=0.60)
    assert tier == 'EDGE'


def test_high_tier_on_top_2_share():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.35, top_3_share=0.40)
    assert tier == 'HIGH'


def test_high_tier_on_top_3_share():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.30, top_3_share=0.45)
    assert tier == 'HIGH'


def test_med_tier():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.28, top_3_share=0.36)
    assert tier == 'MED'


def test_low_tier():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.23, top_3_share=0.30)
    assert tier == 'LOW'


def test_skip_tier():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.15, top_3_share=0.22)
    assert tier == 'SKIP'


def test_boundary_high_at_exact_threshold():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.33, top_3_share=0.30)
    assert tier == 'HIGH'


def test_boundary_med_at_exact_threshold():
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.27, top_3_share=0.30)
    assert tier == 'MED'


def test_custom_thresholds():
    thresholds = ConvictionThresholds(high_top2=0.50)
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.40,
                                  top_3_share=0.50, thresholds=thresholds)
    # top_3_share=0.50 ≥ default high_top3=0.42 → still HIGH via top_3
    assert tier == 'HIGH'
