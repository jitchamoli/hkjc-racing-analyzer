#!/usr/bin/env python3
"""Score combination, calibration, and probability normalization."""

import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple

import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlendConfig:
    """Tunables for the dynamic model/market blend in calibrate_probabilities.

    Defaults reproduce the F-09 production schedule EXACTLY (base 45/55
    model/market, ±0.15 ramps, sprint 1200m, field 10). Inject a variant to
    experiment with the blend without monkeypatching module globals — e.g.
    `BlendConfig(base_model_w=0.30)` for a more-market blend. This is the
    supported extension point for blend research (see scripts/backtest).
    """
    base_model_w: float = 0.45
    shift: float = 0.15                      # max ramp magnitude
    sprint_threshold: float = 1200.0
    sprint_tau: float = 100.0
    field_threshold: float = 10.0
    field_tau: float = 1.5
    long_distance_threshold: float = 1800.0
    distance_tau: float = 100.0
    model_w_floor: float = 0.30
    model_w_ceil: float = 0.70


_DEFAULT_BLEND = BlendConfig()

# Sentinel for calibrate_probabilities(calibrator=...): the default lazy-loads
# the signed production blend_calibrator.pkl. Pass an explicit IsotonicRegression
# to inject a variant, or None to disable calibration entirely. A plain None
# default would silently DISABLE calibration for every production caller, hence
# the sentinel.
USE_PRODUCTION_CALIBRATOR = object()


class BettingWeights:
    """Factor weight constants for the 24-factor rule engine.

    Note: odds and odds_movement REMOVED — market signal enters via the 30%
    market blend in calibrate_probabilities(). Keeping them here would
    triple-count market information (rule odds + rule odds_move + blend).
    """
    RATING = 0.08
    FORM = 0.10
    JOCKEY = 0.05
    TRAINER = 0.03
    DRAW = 0.05
    TIPS = 0.11
    PACE = 0.04
    SECTIONAL = 0.03
    CD_FORM = 0.08
    CLASS_MOVE = 0.03
    GEAR = 0.02
    WEIGHT = 0.04
    FRESHNESS = 0.05
    RATING_TREND = 0.03
    STAKES = 0.02
    GOING = 0.05
    ROLLING_FORM = 0.06
    MARGIN = 0.05
    JOCKEY_CHANGE = 0.04
    JOCKEY_FORM = 0.04
    TRAINER_FORM = 0.03
    FIELD_SIZE = 0.05
    CAREER_PLACE = 0.03
    AGE = 0.02
    SPEED_FIG = 0.05  # class+going-normalised time rating (Beyer-style)

    # Mapping factor keys to weights
    WEIGHT_MAP = {
        "rating": RATING, "form": FORM,
        "jockey": JOCKEY, "trainer": TRAINER, "draw": DRAW,
        "tips": TIPS, "pace": PACE, "sectional": SECTIONAL,
        "cd_form": CD_FORM,
        "class_movement": CLASS_MOVE, "gear": GEAR,
        "weight_carried": WEIGHT, "freshness": FRESHNESS,
        "rating_trend": RATING_TREND, "season_stakes": STAKES,
        "going_pref": GOING, "rolling_form": ROLLING_FORM,
        "margin": MARGIN, "jockey_change": JOCKEY_CHANGE,
        "jockey_form": JOCKEY_FORM, "trainer_form": TRAINER_FORM,
        "field_size": FIELD_SIZE, "career_place": CAREER_PLACE,
        "age": AGE, "speed_fig": SPEED_FIG,
    }


# Historical set — previously forced inclusion even at score=0.
# As of C1 (2026-04-23), no longer load-bearing; retained for documentation.
# Missing factors (score <= 0) are now uniformly excluded via renormalization.
ALWAYS_INCLUDED = {"rating", "form", "jockey", "trainer", "draw",
                   "class_movement", "gear", "weight_carried"}


def compute_weighted_score(factors: Dict[str, float]) -> float:
    """Compute rule-based score from factor dict with dynamic weight normalization.

    Factors with score <= 0 are excluded as "no data" and weights are
    renormalized across the remaining factors to sum to 1.0.

    Historical note (C1, 2026-04-23 audit): `ALWAYS_INCLUDED` previously
    forced inclusion of rating/gear/etc. even at score=0, dragging rule
    scores down ~8 points in 86% of backtest races where DB was missing
    these fields. The set is retained for documentation but no longer
    has load-bearing inclusion logic — exclusion is now uniform across
    all factors.
    """
    weight_map = {}
    for key, score in factors.items():
        if score > 0:
            if key in BettingWeights.WEIGHT_MAP:
                weight_map[key] = BettingWeights.WEIGHT_MAP[key]

    total_weight = sum(weight_map.values())
    if total_weight > 0:
        return sum(factors[k] * (w / total_weight) for k, w in weight_map.items())
    return 50.0


def _dynamic_blend_weights(field_size: int, distance: int,
                           cfg: BlendConfig = None) -> Tuple[float, float]:
    """Compute dynamic market/model blend weights based on race characteristics.

    HK bettors are sharp — larger fields and shorter races benefit from
    heavier market weighting. Smaller fields and longer distances have
    thinner pools where the model has more edge.

    F-09 grid sweep (2026-04-23, 15 points × 73-race holdout):
    best Brier at base_model_w=0.45, field_threshold=10, distance_threshold=1200.
    All 15 points fell within ±0.002 Brier (noise floor), so the tie-break rule
    (within ±0.005 Brier, prefer simpler schedule = base_model_w closest to 0.5)
    selected base_model_w=0.45 over the prior heuristic 0.60.
    See docs/audits/2026-04-23/raw/wave2-sweep-summary.json.

    `cfg` (BlendConfig) holds the tunables; defaults reproduce the F-09
    production schedule exactly. Pass a variant to experiment — no monkeypatch.

    Returns (model_weight, market_weight) summing to 1.0.
    """
    if cfg is None:
        cfg = _DEFAULT_BLEND
    # F-09 empirically chosen base 45/55 model/market, sprint threshold 1200m,
    # field threshold 10. Replaced hard cutoffs (40pp swing on field_size
    # differing by 1 horse, e.g. 10 vs 9 at 1200m) with logistic ramps so
    # the weight changes smoothly across boundaries — no noise amplification
    # at exactly threshold values.
    base_model_w = cfg.base_model_w

    # Sigmoid ramps: σ(x/τ) is 0 far below threshold, 1 far above, ~0.5 at threshold.
    # τ = scale (smoothness): larger = smoother transition.
    sprint_factor = 1.0 / (1.0 + math.exp(-(cfg.sprint_threshold - distance) / cfg.sprint_tau))
    big_field_factor = 1.0 / (1.0 + math.exp(-(field_size - cfg.field_threshold) / cfg.field_tau))
    long_distance_factor = 1.0 / (1.0 + math.exp(-(distance - cfg.long_distance_threshold) / cfg.distance_tau))
    small_field_factor = 1.0 / (1.0 + math.exp(-(cfg.field_threshold - field_size) / cfg.field_tau))

    # Large field + sprint → market sharper → shift toward market (max +shift)
    market_shift = cfg.shift * sprint_factor * big_field_factor
    # Small field OR long distance → thinner pool → shift toward model (max +shift)
    # Use OR via algebraic combination: 1 - (1-small)(1-long)
    pool_thinness = 1.0 - (1.0 - small_field_factor) * (1.0 - long_distance_factor)
    model_shift = cfg.shift * pool_thinness

    # Apply both shifts (they can partially offset on edge cases)
    model_w = base_model_w + model_shift - market_shift
    model_w = max(cfg.model_w_floor, min(cfg.model_w_ceil, model_w))
    market_w = 1.0 - model_w

    return model_w, market_w


_BLEND_CALIBRATOR = None  # cached on first load
_BLEND_CALIBRATOR_LOADED = False

# Last densely-populated isotonic knot. The blend calibrator's fit set has only
# ~80 of 7662 samples above 0.30 (n=3 above 0.60), so PAVA collapses the entire
# tail to one noise-driven value (blended 0.40 and 0.95 both map to ~0.737).
# We clamp the calibrator INPUT here so every strong favourite maps to iso(0.30)
# instead — monotone, conservative (caps over-statement, never inflates), and
# provably untouched below 0.30 (the well-calibrated 99%). Apply-time only: no
# refit, pickle unchanged. See 2-expert calibration-tail review 2026-06-27.
_BLEND_CAL_INPUT_CLIP = 0.30

# Weight of the plateau tie-break term (see calibrate_probabilities). Small
# enough that calibration is unchanged to ~1e-6, large enough to survive
# float renormalisation and order horses inside an isotonic step.
_BLEND_CAL_TIEBREAK_EPS = 1e-6


def reset_blend_calibrator_cache() -> None:
    """Drop the cached calibrator so the next call re-reads the artifact.

    Needed by tests, and by any caller that rewrites blend_calibrator.pkl
    mid-process (e.g. an auto-learn refit) and wants the new fit picked up.
    """
    global _BLEND_CALIBRATOR, _BLEND_CALIBRATOR_LOADED
    _BLEND_CALIBRATOR = None
    _BLEND_CALIBRATOR_LOADED = False


def _load_blend_calibrator():
    """Lazy-load models/blend_calibrator.pkl. Returns None if absent.

    The blend calibrator is an IsotonicRegression fitted on
    (production calibrated_prob → actual_won) over the held-out 20% of
    races. It corrects the residual under-prediction bias the post-C1
    pipeline exhibits on top picks. Built by `python3 -m scripts.calibrate_blend`.
    """
    global _BLEND_CALIBRATOR, _BLEND_CALIBRATOR_LOADED
    if _BLEND_CALIBRATOR_LOADED:
        return _BLEND_CALIBRATOR
    # NOTE: _BLEND_CALIBRATOR_LOADED is set only on a SUCCESSFUL resolution
    # (loaded, or the file legitimately absent). Setting it up-front — as this
    # did until 2026-08-03 — meant one bad read, missing sidecar or checksum
    # mismatch cached None for the life of the process, so every later race ran
    # uncalibrated and a repaired artifact was never re-read. Failures now
    # retry on the next call; the happy path still reads the file once.
    import hashlib, pickle
    from pathlib import Path
    repo_root = Path(__file__).resolve().parent.parent
    pkl = repo_root / 'models' / 'blend_calibrator.pkl'
    sha = pkl.with_suffix('.sha256')
    if not pkl.exists():
        _BLEND_CALIBRATOR_LOADED = True
        return None
    try:
        raw = pkl.read_bytes()
        # Hard refusal on a missing sidecar, matching the main model's policy
        # (ml/model.py): an unsigned calibrator is applied to every post-blend
        # probability, so loading one unverified would silently distort edges,
        # conviction tiers, and the bet ticket.
        if not sha.exists():
            logger.error(
                "blend_calibrator .sha256 sidecar missing — refusing to load "
                "unsigned calibrator; pipeline runs uncalibrated."
            )
            return None
        expected = sha.read_text().strip()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            logger.error(
                f"blend_calibrator checksum mismatch — pipeline running uncalibrated. "
                f"Expected {expected[:16]}..., got {actual[:16]}..."
            )
            return None
        payload = pickle.loads(raw)
        _BLEND_CALIBRATOR = payload.get('calibrator')
        _BLEND_CALIBRATOR_LOADED = True
    except Exception as e:
        logger.error(f"blend_calibrator load failed — pipeline running uncalibrated: {e}")
        _BLEND_CALIBRATOR = None
    return _BLEND_CALIBRATOR


def calibrate_probabilities(analyses, field_size: int = 12,
                            distance: int = 1200,
                            blend_config: BlendConfig = None,
                            calibrator=USE_PRODUCTION_CALIBRATOR) -> List[float]:
    """Convert combined_probability scores into calibrated race-level probabilities.

    Pipeline:
      1. Mild softmax over combined_probability → cal_probs
      2. Dynamic blend with market-implied probs → blended
      3. (optional) Apply post-blend isotonic calibrator if present
      4. Normalize so sum==1

    Injection points (defaults reproduce production exactly):
      blend_config — a BlendConfig to override the model/market schedule.
      calibrator   — USE_PRODUCTION_CALIBRATOR (default) lazy-loads the signed
                     blend_calibrator.pkl; pass an IsotonicRegression to inject
                     a variant, or None to disable calibration.

    The post-blend calibrator (step 3) corrects a residual under-prediction
    bias on the post-C1 anchored-sigmoid distribution — top picks were
    predicted ~32% but actual win rate was ~46% on a 5,966-race holdout.
    Calibrator is fit by `scripts/calibrate_blend.py`; absent file → step 3
    is skipped (backwards-compatible).

    Per-horse isotonic preserves rank ordering (monotonic), so renormalizing
    after applying maintains both the win-prob calibration and sum-to-one.
    The _BLEND_CAL_INPUT_CLIP clamp makes the isotonic non-strict on the
    >0.30 plateau, so co-favourites can tie in calibrated_prob; that tie is
    resolved deterministically by pre-blend rank in the recommender sort
    (assign_bet_recommendations), keeping calibrated_prob magnitudes clean.
    """
    if not analyses:
        return []

    # combined_probability is already a probability in [0,1]
    # (rule_prob × 0.6 + ml_prob × 0.4 from create_combined_prediction).
    # Just normalize so they sum to 1.0 — softmax with temp here would
    # collapse model separation: e.g. raw 0.50 vs 0.05 horses become
    # 0.13 vs 0.066 after softmax(temp=1.5), destroying signal before
    # the market blend.
    raw_scores = [a.combined_probability for a in analyses]
    raw_total = sum(raw_scores) or 1.0
    cal_probs = [s / raw_total for s in raw_scores]

    # Blend with market-implied probabilities
    market_probs = []
    for a in analyses:
        if a.horse.win_odds and a.horse.win_odds > 0:
            market_probs.append(1.0 / a.horse.win_odds)
        else:
            market_probs.append(0.0)
    market_total = sum(market_probs) or 1.0
    market_probs = [p / market_total for p in market_probs]

    # Dynamic blend weights based on race characteristics
    model_w, market_w = _dynamic_blend_weights(field_size, distance, blend_config)

    blended = []
    for i in range(len(cal_probs)):
        if market_probs[i] > 0:
            b = model_w * cal_probs[i] + market_w * market_probs[i]
        else:
            # No market signal (horse has no live odds). Multiply by model_w
            # alone (NOT cal_probs[i] directly) so this horse's blended value
            # is on the same scale as horses that DID get market blending.
            # Previous implementation `b = cal_probs[i]` over-inflated
            # no-odds horses in mixed fields (some with odds, some without)
            # because cal_probs[i] is bigger than `model_w * cal_probs[i] +
            # market_w * market_probs[i]` for non-favourites — after the
            # post-blend renormalization the asymmetry persisted.
            b = model_w * cal_probs[i]
        blended.append(b)

    # Pre-isotonic renorm: input to isotonic should sum to 1 within the race.
    # Post-isotonic renorm (Fix A): the calibrator was fit per-horse so its
    # output is monotonic but does NOT sum to 1 across a race — empirically
    # the post-iso sum landed at 1.16-1.28 on in-distribution inputs, which
    # then inflated:
    #   - expected_return = win_odds × prob - 1 (each horse looked +25% richer)
    #   - value_score = prob × 100 (top picks displayed 50%+ when truly ~40%)
    #   - recommender top_2/3/4_share thresholds (gates fired ~16-28% more)
    # Restore sum=1 after isotonic so downstream EV/threshold consumers see
    # a proper probability distribution. Per-horse calibration shape (rank
    # ordering, ratios over short ranges) is preserved by the renorm.
    blend_total = sum(blended) or 1.0
    blended = [p / blend_total for p in blended]

    iso = _load_blend_calibrator() if calibrator is USE_PRODUCTION_CALIBRATOR else calibrator
    if iso is not None:
        try:
            import numpy as _np
            # Clamp inputs at the last dense knot before the (sparse, overfit)
            # isotonic tail — see _BLEND_CAL_INPUT_CLIP above. NOTE: two horses
            # both >0.30 blended then map to iso(0.30) and TIE in calibrated_prob.
            # That tie is resolved deterministically by pre-blend rank in the
            # recommender sort (assign_bet_recommendations), NOT here — so
            # calibrated_prob magnitudes stay clean (a constant calibrator still
            # yields a uniform distribution).
            pre_iso = _np.array(blended)
            calibrated = iso.predict(_np.minimum(pre_iso, _BLEND_CAL_INPUT_CLIP))
            # Tie-break term uses the PRE-BLEND model probability, not the
            # blended one. Measured 2026-08-03: breaking plateau ties by the
            # blended value instead made things WORSE across the board
            # (top-pick Brier 0.1920 -> 0.1963, straight QIN -0.81% -> -2.05%,
            # TRIO-5 -15.6% -> -19.8%). Globally the blend outranks the pure
            # model, but INSIDE an isotonic plateau the blend has declared the
            # horses indistinguishable, and there the model's independent
            # opinion is the only signal left — which is precisely what the
            # pre_blend_rank tiebreak in the recommender was already doing.
            # This keeps that ordering and additionally removes the degenerate
            # exact-0.0 outputs (4.4% of live runners) the raw isotonic emits.
            tie_term = _np.array(cal_probs)
            # Restore a strict ordering inside each isotonic plateau. The
            # fitted isotonic is a coarse step function (48 distinct levels on
            # [0, 0.30]; everything below 0.0202 maps to exactly 0.0), so
            # distinct horses collapsed onto identical calibrated_prob — 4.4%
            # of runners hit exactly 0.0, and exact ties straddled the BOX
            # TRIO-5 leg cutoff in ~45% of races. Where that happened
            # calibrated_prob had no resolving power and the cutoff fell
            # through to the pre_blend_rank tiebreak, i.e. the PURE model
            # picked the legs — measurably the weakest ranking source
            # available (6.3% quinella capture vs 13.6% blended).
            #
            # The term is vanishing (1e-6), so calibration is preserved and the
            # isotonic's own ordering always dominates.
            blended = list(calibrated + _BLEND_CAL_TIEBREAK_EPS * tie_term)
        except Exception as e:
            # Fall through to raw blended values on any sklearn error, but
            # surface the failure — silent skip means every prediction is
            # uncalibrated with no observable signal (e.g. after a sklearn
            # version drift that invalidates the pickled isotonic).
            logger.warning(
                "blend_calibrator.predict failed — predictions uncalibrated "
                "this run: %s", e,
            )

    iso_total = sum(blended) or 1.0
    return [p / iso_total for p in blended]


def apply_calibrated_probs(analyses, calibrated_probs: List[float]) -> None:
    """Write calibrated probabilities and expected returns onto analyses.

    Does NOT write value_score — that field is pinned to the pre-blend
    combined_prob at analyzer construction time (CLAUDE.md ML Rule 4).

    F-19: On the first call (model_probability is still None), snapshot the
    PRE-blend pure model probability into model_probability so accuracy.py
    can measure the model itself, not the model+market blend. At call time
    a.combined_probability still holds the pre-blend value (set by analyzer);
    we capture it BEFORE overwriting with the post-blend calibrated value.
    Without this, accuracy reports systematically credit the market for the
    model's perceived performance.
    """
    for i, a in enumerate(analyses):
        if a.model_probability is None:
            a.model_probability = a.combined_probability  # pre-blend snapshot
        # value_score MUST stay pinned to the pre-blend combined_prob (set
        # by analyzer.py before calibration). Per CLAUDE.md ML Rule 4 it
        # reflects the model's independent opinion. Overwriting with the
        # calibrated (market-blended) prob was making value_score lie about
        # what the model actually thought. Locked by
        # tests/test_audit_2026_05_29.py::test_value_score_not_overwritten_in_recommender.
        a.combined_probability = calibrated_probs[i]
        if a.horse.win_odds and a.horse.win_odds > 0:
            a.expected_return = (a.horse.win_odds * calibrated_probs[i]) - 1
        else:
            a.expected_return = -1.0
