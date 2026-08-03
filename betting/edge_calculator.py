"""Pure-prob and edge math with 6-gate filtering.

pure_prob = softmax(0.6·sigmoid_anchored(rule_score, 1/N) + 0.4·ml_prob)
— our independent opinion BEFORE the market blend.

Feeds edge_pp, ev_win, overlay; gated by EV + prob floor + edge gap + rank
+ odds freshness + model integrity. See
docs/superpowers/specs/2026-04-24-dashboard-redesign-design.md §4.
"""

import logging
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import config

logger = logging.getLogger(__name__)


@dataclass
class EdgeGates:
    """Gate thresholds. Defaults per spec §4.

    G3 (min_edge_pp = 5.0) used to almost never fire, and the reason was a
    BUG, not design. pure_prob was uncalibrated and badly under-confident —
    over 15 production meetings the rank-1 horse read 13.3% while actually
    winning 26.5% (the market said 23.3%). Because edge_pp subtracts a
    well-scaled market number from that, every top pick carried a structural
    ~-10pp before any genuine disagreement, so G3 could only clear on noise.

    Fixed 2026-08-03 by fitting models/pure_calibrator.pkl on the disjoint
    calib fold (scripts/calibrate_pure). Measured on the same 15 meetings:

        mean pure_prob   13.3% -> 18.9%   (truth 26.5%)
        mean edge_pp    -10.0pp -> -4.4pp
        G3 fires          7.5% -> 15.6% of races

    Renormalising to a proper distribution projects part of the lift back out,
    so the correction is partial by construction — treat any remaining
    negative mean edge as partly real (the market is genuinely sharp) rather
    than as more scale error. The threshold itself is UNCHANGED; it now just
    measures what it always claimed to.
    """
    min_ev: float = 0.05
    min_prob: float = 0.10
    min_edge_pp: float = 5.0
    max_model_rank: int = 2
    max_odds_age_min: int = 30


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rule_score_to_prob(rule_score: float, field_size: int) -> float:
    # Base-rate-anchored sigmoid: a score=50 horse outputs ~1/field_size,
    # not 0.5. Mirrors ml/model.py:create_combined_prediction so the
    # edge math sees the same rule-prob distribution as the analyzer.
    base = 1.0 / max(field_size, 2)
    logit_base = math.log(base / (1.0 - base))
    return _sigmoid(0.06 * (rule_score - 50) + logit_base)


_PURE_CALIBRATOR = None
_PURE_CALIBRATOR_LOADED = False

# Sentinel so `calibrator=None` can explicitly DISABLE calibration while the
# default still loads the signed production artifact. Mirrors
# core.scorer.USE_PRODUCTION_CALIBRATOR.
USE_PRODUCTION_PURE_CALIBRATOR = object()

# Plateau tie-break weight — see compute_pure_probs. Mirrors
# core.scorer._BLEND_CAL_TIEBREAK_EPS.
_PURE_CAL_TIEBREAK_EPS = 1e-6


def reset_pure_calibrator_cache() -> None:
    """Drop the cached pure calibrator so the next call re-reads the artifact."""
    global _PURE_CALIBRATOR, _PURE_CALIBRATOR_LOADED
    _PURE_CALIBRATOR = None
    _PURE_CALIBRATOR_LOADED = False


def _load_pure_calibrator():
    """Lazy-load models/pure_calibrator.pkl. Returns None if absent.

    An IsotonicRegression fitted on (pure_prob -> is_winner) over the disjoint
    70-85% calib fold by `python3 -m scripts.calibrate_pure`. Without it
    pure_prob is badly under-confident (rank-1 read 13.3% for horses that win
    26.5%), which baked a structural ~-10pp into every edge_pp.

    Refuses to load unsigned, and caches only a SUCCESSFUL resolution so a
    repaired artifact is picked up on the next call.
    """
    global _PURE_CALIBRATOR, _PURE_CALIBRATOR_LOADED
    if _PURE_CALIBRATOR_LOADED:
        return _PURE_CALIBRATOR
    import hashlib
    import pickle
    from pathlib import Path
    pkl = Path(__file__).resolve().parent.parent / 'models' / 'pure_calibrator.pkl'
    sha = pkl.with_suffix('.sha256')
    if not pkl.exists():
        _PURE_CALIBRATOR_LOADED = True
        return None
    try:
        raw = pkl.read_bytes()
        if not sha.exists():
            logger.error("pure_calibrator .sha256 sidecar missing — refusing to "
                         "load unsigned calibrator; edge math runs uncalibrated.")
            return None
        if hashlib.sha256(raw).hexdigest() != sha.read_text().strip():
            logger.error("pure_calibrator checksum mismatch — edge math runs "
                         "uncalibrated.")
            return None
        _PURE_CALIBRATOR = pickle.loads(raw).get('calibrator')
        _PURE_CALIBRATOR_LOADED = True
    except Exception as e:
        logger.error(f"pure_calibrator load failed — edge math uncalibrated: {e}")
        _PURE_CALIBRATOR = None
    return _PURE_CALIBRATOR


def compute_pure_probs(horses: List[Dict], field_size: int = None,
                       calibrator=USE_PRODUCTION_PURE_CALIBRATOR) -> List[float]:
    # raw[i] = 0.6 * sigmoid_anchored(rule_score) + 0.4 * ml_prob is already
    # a probability in [0,1]. Softmax here would flatten the model's signal
    # (top horse 0.50 vs bottom 0.05 → 0.18 vs 0.07 after exp normalization),
    # making edge_pp = pure - market mechanically negative across the board
    # (mean -14pp on 124-race holdout, G3 gate fired 0 times by design but
    # actually because pure_prob was compressed). Just normalize.
    n = field_size if field_size is not None else len(horses)
    # Read the blend from config at CALL time so this cannot drift from
    # ml/model.py:create_combined_prediction, which does the same. These were
    # hardcoded 0.6/0.4 literals until 2026-08-03 — identical to config today,
    # but pure_prob feeds all six EDGE gates and the exotic overlay, so a
    # silent desync here would measure a different model than value_score.
    rule_w, ml_w = config.RULE_WEIGHT, config.ML_WEIGHT
    raw = [rule_w * _rule_score_to_prob(h['rule_score'], n) + ml_w * h['ml_prob']
           for h in horses]
    total = sum(raw) or 1.0
    probs = [r / total for r in raw]

    # Calibrate. Un-calibrated, this quantity is badly under-confident — over
    # 15 production meetings the rank-1 horse read 13.3% while actually winning
    # 26.5% (the market said 23.3%). Since edge_pp subtracts a well-scaled
    # market number from this one, every top pick carried a structural ~-10pp
    # and the EDGE gate (>= +5pp) could only fire on noise: 3/147 races.
    # Isotonic is monotone, so this never reorders the field — and pure_prob is
    # not in the recommender path, so calibrated_prob and leg selection are
    # untouched. Renormalise afterwards to keep a proper distribution.
    iso = (_load_pure_calibrator()
           if calibrator is USE_PRODUCTION_PURE_CALIBRATOR else calibrator)
    if iso is not None:
        try:
            import numpy as _np
            pre = _np.array(probs)
            out = _np.asarray(iso.predict(pre), dtype=float)
            # Same plateau tie-break as core/scorer: a fitted isotonic is a
            # step function (here 0.30 and 0.40 both map to 0.5625), so
            # distinct horses would otherwise collapse to identical pure_prob
            # and to exact 0.0 in the tail. A vanishing multiple of the
            # pre-calibration value restores a strict ordering without moving
            # the calibration.
            out = out + _PURE_CAL_TIEBREAK_EPS * pre
            t2 = float(out.sum()) or 1.0
            probs = [float(p) / t2 for p in out]
        except Exception as e:
            logger.warning("pure_calibrator.predict failed — edge math "
                           "uncalibrated this run: %s", e)
    return probs


def compute_market_probs(horses: List[Dict]) -> List[float]:
    inv = [(1.0 / h['win_odds']) if h.get('win_odds', 0) > 0 else 0.0
           for h in horses]
    total = sum(inv) or 1.0
    return [v / total for v in inv]


def compute_edge_fields(pure: float, market: float, win_odds: float) -> Dict[str, float]:
    return {
        'edge_pp': (pure - market) * 100.0,
        'ev_win': pure * win_odds - 1.0 if win_odds > 0 else -1.0,
        'overlay': pure * win_odds if win_odds > 0 else 0.0,
    }


def apply_edge_gates(
    ev_win: float, pure_prob: float, edge_pp: float,
    model_rank: int, odds_age_min, integrity_ok: bool,
    gates: EdgeGates = None,
) -> Tuple[bool, List[str]]:
    gates = gates or EdgeGates()
    failed = []
    if ev_win < gates.min_ev: failed.append('G1')
    if pure_prob < gates.min_prob: failed.append('G2')
    if edge_pp < gates.min_edge_pp: failed.append('G3')
    if model_rank > gates.max_model_rank: failed.append('G4')
    # G5: stale odds invalidate EV. odds_age_min=None means "no snapshot
    # exists at all" — strictly worse than stale, so fail the gate.
    if odds_age_min is None or odds_age_min > gates.max_odds_age_min:
        failed.append('G5')
    if not integrity_ok: failed.append('G6')
    return (len(failed) == 0, failed)
