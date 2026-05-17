"""Tier assignment at spec boundaries.

Thresholds raised 2026-05-17 (audit) to 0.40/0.55/0.30/0.42/0.22 — see
ConvictionThresholds docstring. Tests below use values clearly above/below
each new boundary.
"""
import pytest
from betting.conviction import assign_conviction_tier, ConvictionThresholds


def test_edge_takes_precedence_over_high():
    tier = assign_conviction_tier(has_edge=True, top_2_share=0.50, top_3_share=0.60)
    assert tier == 'EDGE'


def test_high_tier_on_top_2_share():
    # top_2=0.42 ≥ high_top2=0.40 → HIGH
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.42, top_3_share=0.50)
    assert tier == 'HIGH'


def test_high_tier_on_top_3_share():
    # top_3=0.58 ≥ high_top3=0.55 → HIGH (via top_3 even when top_2 is low)
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.35, top_3_share=0.58)
    assert tier == 'HIGH'


def test_med_tier():
    # top_2=0.32 ≥ med_top2=0.30 but < high_top2=0.40, top_3 < high_top3 → MED
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.32, top_3_share=0.45)
    assert tier == 'MED'


def test_low_tier():
    # top_2=0.25 ≥ low_top2=0.22, all higher gates fail → LOW
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.25, top_3_share=0.32)
    assert tier == 'LOW'


def test_skip_tier():
    # top_2=0.15 < low_top2=0.22 → SKIP
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.15, top_3_share=0.22)
    assert tier == 'SKIP'


def test_boundary_high_at_exact_threshold():
    # Exactly at high_top2=0.40 → HIGH (≥ is inclusive)
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.40, top_3_share=0.30)
    assert tier == 'HIGH'


def test_boundary_med_at_exact_threshold():
    # Exactly at med_top2=0.30 (below high_top2) → MED
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.30, top_3_share=0.30)
    assert tier == 'MED'


def test_custom_thresholds():
    # Custom high_top2 raised to 0.50, default high_top3=0.55.
    # top_2=0.45 < 0.50 AND top_3=0.50 < 0.55 → MED (top_2 ≥ 0.30 default med_top2)
    thresholds = ConvictionThresholds(high_top2=0.50)
    tier = assign_conviction_tier(has_edge=False, top_2_share=0.45,
                                  top_3_share=0.50, thresholds=thresholds)
    assert tier == 'MED'
