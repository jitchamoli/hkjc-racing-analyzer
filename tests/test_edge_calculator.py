"""Edge math + 6-gate logic tests."""
import math
import pytest
from betting.edge_calculator import (
    compute_pure_probs, compute_market_probs, compute_edge_fields,
    apply_edge_gates, EdgeGates,
)


def test_pure_probs_softmax_sums_to_one():
    horses = [
        {'rule_score': 70, 'ml_prob': 0.30},
        {'rule_score': 50, 'ml_prob': 0.15},
        {'rule_score': 40, 'ml_prob': 0.10},
    ]
    pure = compute_pure_probs(horses)
    assert len(pure) == 3
    assert math.isclose(sum(pure), 1.0, abs_tol=1e-6)
    assert pure[0] > pure[1] > pure[2]


def test_market_probs_normalize_from_odds():
    horses = [{'win_odds': 2.0}, {'win_odds': 4.0}, {'win_odds': 8.0}]
    market = compute_market_probs(horses)
    assert math.isclose(sum(market), 1.0, abs_tol=1e-6)
    assert market[0] > market[1] > market[2]


def test_market_probs_handle_zero_odds():
    horses = [{'win_odds': 2.0}, {'win_odds': 0}, {'win_odds': 4.0}]
    market = compute_market_probs(horses)
    assert market[1] == 0.0
    assert math.isclose(market[0] + market[2], 1.0, abs_tol=1e-6)


def test_edge_fields_positive_gap():
    pure, market, odds = 0.22, 0.12, 8.3
    fields = compute_edge_fields(pure, market, odds)
    assert math.isclose(fields['edge_pp'], 10.0, abs_tol=0.01)
    assert math.isclose(fields['ev_win'], 0.22 * 8.3 - 1, abs_tol=0.01)
    assert math.isclose(fields['overlay'], 0.22 * 8.3, abs_tol=0.01)


def test_edge_fields_negative_gap():
    pure, market, odds = 0.10, 0.20, 5.0
    fields = compute_edge_fields(pure, market, odds)
    assert fields['edge_pp'] < 0
    assert fields['ev_win'] < 0
    assert fields['overlay'] < 1.0


def test_gates_all_pass_fires():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.10, pure_prob=0.22, edge_pp=10.0,
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is True
    assert failed == []


def test_gate_g1_ev_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.02, pure_prob=0.22, edge_pp=10.0,
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is False
    assert 'G1' in failed


def test_gate_g2_prob_floor_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.50, pure_prob=0.05, edge_pp=4.0,
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is False
    assert 'G2' in failed


def test_gate_g3_edge_gap_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.06, pure_prob=0.15, edge_pp=3.0,
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is False
    assert 'G3' in failed


def test_gate_g4_rank_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.10, pure_prob=0.15, edge_pp=8.0,
        model_rank=5, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is False
    assert 'G4' in failed


def test_gate_g5_odds_stale_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.10, pure_prob=0.22, edge_pp=10.0,
        model_rank=1, odds_age_min=45, integrity_ok=True, gates=gates,
    )
    assert ok is False
    assert 'G5' in failed


def test_gate_g6_integrity_fails():
    gates = EdgeGates()
    ok, failed = apply_edge_gates(
        ev_win=0.10, pure_prob=0.22, edge_pp=10.0,
        model_rank=1, odds_age_min=5, integrity_ok=False, gates=gates,
    )
    assert ok is False
    assert 'G6' in failed


def test_gates_with_custom_thresholds():
    gates = EdgeGates(min_ev=0.10)
    ok, _ = apply_edge_gates(
        ev_win=0.07, pure_prob=0.22, edge_pp=10.0,
        model_rank=1, odds_age_min=5, integrity_ok=True, gates=gates,
    )
    assert ok is False
