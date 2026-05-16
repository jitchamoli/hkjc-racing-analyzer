"""EXPERIMENT: train a top2-targeted model + compare to the current win model.

Hypothesis: the recommender mostly bets QIN/BOX-Q/BOX-Q4 — all "did the
top-2 finish inside our top-N picks" outcomes. The current model predicts
is_winner. A model trained on is_top2 should be more directly aligned with
what we actually bet.

Phase 1: train top2 model on same 24 features but with is_top2 label.
Phase 2: A/B compare top-N rankings on the backtest holdout.

NO production code change — outputs reports only. Save to /tmp so we
don't pollute models/. If results justify, the next step is wiring into
the recommender.
"""
import argparse
import logging
import pickle
import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb

import config
try:
    from data.database import apply_perf_pragmas
except ImportError:
    # PR #28 not merged yet — fall through to default SQLite settings.
    def apply_perf_pragmas(conn): pass
from ml.feature_schema import FEATURE_NAMES, NUM_FEATURES
from ml.training import build_training_data_from_db
from scripts.backtest import (
    load_held_out_races, group_by_race, predict_race_production,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TOP2_MODEL_PATH = Path("/tmp/top2_model_experiment.pkl")


# ─────────────────────────────────────────────────────────────────────
# Phase 1 — train top2 model + isotonic
# ─────────────────────────────────────────────────────────────────────

def derive_top2_labels(conn):
    """Build is_top2 labels in the same row order as build_training_data_from_db."""
    # Mirror the SELECT in build_training_data_from_db exactly so labels align.
    rows = conn.execute(
        "SELECT finish_position FROM race_results "
        "WHERE finish_position > 0 "
        "ORDER BY race_date, race_number"
    ).fetchall()
    return np.array([1 if r[0] in (1, 2) else 0 for r in rows], dtype=np.int8)


def train_top2_model():
    """Train + calibrate a top2 model. Returns (model, calibrator, metrics)."""
    out = build_training_data_from_db()
    X, y_win, sample_weight, y_place = out
    if X is None:
        raise RuntimeError("build_training_data_from_db returned no data")

    # is_top2: 1 if finish_position in (1, 2) else 0. Derive from y_win (winner)
    # + y_place (top3) — top2 is a superset of winner, subset of top3. We need
    # finish_position explicitly because top3 ⊅ top2.
    conn = sqlite3.connect(str(config.DB_PATH))
    apply_perf_pragmas(conn)
    y_top2 = derive_top2_labels(conn)
    conn.close()

    if len(y_top2) != len(X):
        raise RuntimeError(
            f"Label count mismatch: training data has {len(X)} rows but "
            f"derived {len(y_top2)} top2 labels. SELECT order may differ "
            f"between build_training_data_from_db and derive_top2_labels."
        )

    logger.info(f"Training top2 model on {len(X):,} samples ({int(y_top2.sum()):,} top2)")

    # 4-fold temporal split — same boundaries as ml/model.py:train (no
    # leakage, calibrator window stays disjoint from holdout).
    n = len(X)
    cut_train = int(n * 0.60)
    cut_val = int(n * 0.70)
    cut_calib = int(n * 0.85)

    X_train, X_val = X[:cut_train], X[cut_train:cut_val]
    X_calib, X_test = X[cut_val:cut_calib], X[cut_calib:]
    y_train, y_val = y_top2[:cut_train], y_top2[cut_train:cut_val]
    y_calib, y_test = y_top2[cut_val:cut_calib], y_top2[cut_calib:]
    sw_train = sample_weight[:cut_train] if sample_weight is not None else None

    pos = max(1, int(y_train.sum()))
    neg = max(1, len(y_train) - pos)

    params = dict(config.XGBOOST_PARAMS)
    params["scale_pos_weight"] = neg / pos
    params["early_stopping_rounds"] = 20

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sw_train,
              eval_set=[(X_val, y_val)], verbose=False)

    # Isotonic calibration on the dedicated calib fold (same pattern as win model)
    from scipy.special import logit as _logit
    calib_proba = model.predict_proba(X_calib)[:, 1]
    calib_logit = _logit(np.clip(calib_proba, 1e-6, 1 - 1e-6))
    sw_calib = sample_weight[cut_val:cut_calib] if sample_weight is not None else None

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(calib_logit.ravel(), y_calib, sample_weight=sw_calib)

    # Test metrics
    test_proba_raw = model.predict_proba(X_test)[:, 1]
    test_logit = _logit(np.clip(test_proba_raw, 1e-6, 1 - 1e-6))
    test_proba = calibrator.predict(test_logit.ravel())
    test_pred = (test_proba >= 0.5).astype(int)
    accuracy = float((test_pred == y_test).mean())
    brier = float(((test_proba - y_test) ** 2).mean())
    base_rate = float(y_test.mean())
    bss = 1.0 - brier / (base_rate * (1 - base_rate)) if base_rate > 0 else float("nan")

    metrics = {
        "n_samples": n,
        "n_top2_train": int(y_train.sum()),
        "n_top2_test": int(y_test.sum()),
        "test_accuracy": accuracy,
        "test_brier": brier,
        "test_base_rate": base_rate,
        "test_bss": bss,
    }
    logger.info(f"  test acc={accuracy:.3f}  brier={brier:.4f}  base_rate={base_rate:.3f}  bss={bss:+.4f}")
    return model, calibrator, metrics


def save_top2(model, calibrator, metrics):
    payload = {
        "model": model,
        "calibrator": calibrator,
        "metrics": metrics,
        "n_features": NUM_FEATURES,
        "feature_names": list(FEATURE_NAMES),
    }
    with open(TOP2_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    logger.info(f"Saved {TOP2_MODEL_PATH}")


def load_top2():
    with open(TOP2_MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────
# Phase 2 — A/B ranking comparison on backtest holdout
# ─────────────────────────────────────────────────────────────────────

def predict_top2_for_race(horse_rows, top2_payload, conn):
    """Return list of (h_row, top2_calibrated_prob) in the order horse_rows came in."""
    from ml.features import FeatureEngineering
    from scipy.special import logit as _logit

    model = top2_payload["model"]
    cal = top2_payload["calibrator"]
    fe = FeatureEngineering()
    field_size = len(horse_rows)
    out = []
    for h in horse_rows:
        from scripts.backtest import build_horse_data
        hd = build_horse_data(h, field_size)
        feats = fe.extract_features(hd, [], h.get("venue", ""), conn=conn)
        raw = float(model.predict_proba(feats.reshape(1, -1))[0, 1])
        lg = float(_logit(max(1e-6, min(1 - 1e-6, raw))))
        cal_p = float(cal.predict(np.array([lg]))[0])
        out.append((h, cal_p))
    return out


def evaluate_rankings(top2_payload, label: str):
    """For each holdout race, compute hit-rates under three ranking methods."""
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    apply_perf_pragmas(conn)
    rows = load_held_out_races(conn)
    races = group_by_race(rows)
    logger.info(f"\n=== {label} — {len(races)} races ===")

    # Hit counters under different rankings of top-2 / top-3 / top-4
    # Apples-to-apples: include a "top2 + market" variant so we compare a
    # market-blended top2-target to the market-blended win-target — instead
    # of penalising top2 for lacking the market signal the win path has.
    counts = {
        "win_rank_top2": 0,        # current: top-2 by win-blend (post-market)
        "top2_raw_top2": 0,        # raw top2_prob (NO market blend)
        "top2_market_top2": 0,     # 0.45 top2 + 0.55 market (matches prod blend)
        "blend_top2": 0,           # 0.5 win-blend + 0.5 top2-market
        "win_rank_top3": 0, "top2_raw_top3": 0, "top2_market_top3": 0, "blend_top3": 0,
        "win_rank_top4": 0, "top2_raw_top4": 0, "top2_market_top4": 0, "blend_top4": 0,
    }
    # Per-method P&L using REAL HKJC quinella dividends (look up combination
    # in race_dividends). Strategy: straight QIN (top-2 pair, 1 unit/race),
    # box-3 (3 units/race), box-4 (6 units/race). Loss = stake on miss.
    from itertools import combinations
    horse_no_lookup = {}  # (race_date, race_no) -> {horse_name: number}
    pnl = {m: {"straight": 0.0, "box3": 0.0, "box4": 0.0,
               "stake_straight": 0, "stake_box3": 0, "stake_box4": 0}
           for m in ("win_rank", "top2_raw", "top2_market", "blend")}

    def _lookup_qin_dividend(rd, rn, n1, n2):
        if n1 is None or n2 is None or n1 == n2:
            return None
        combo = f"{min(n1, n2)},{max(n1, n2)}"
        row = conn.execute(
            "SELECT dividend FROM race_dividends "
            "WHERE race_date=? AND race_number=? AND pool='QUINELLA' AND combination=?",
            (rd, rn, combo),
        ).fetchone()
        return row[0] if row else None

    n_races_used = 0

    for (rd, rn), horse_rows in races.items():
        if not any((h.get("win_odds") or 0) > 0 for h in horse_rows):
            continue

        # Current pipeline rankings (combined_probability post-blend)
        preds = predict_race_production(horse_rows, conn)
        if not preds:
            continue
        # preds is sorted by rank (rank-1 first), elements: (h_row, prob, market, analysis)
        win_ranked = [p[0]["horse_name"] for p in preds]

        # Top2 model rankings — raw
        top2_preds = predict_top2_for_race(horse_rows, top2_payload, conn)
        top2_raw_ranked = [h["horse_name"] for h, _ in
                           sorted(top2_preds, key=lambda x: -x[1])]

        # Top2 + market blend (apples-to-apples with win+market)
        win_prob_by_name = {p[0]["horse_name"]: p[1] for p in preds}
        market_prob_by_name = {p[0]["horse_name"]: p[2] for p in preds}
        top2_prob_by_name = {h["horse_name"]: pr for h, pr in top2_preds}
        names = list(win_prob_by_name.keys())
        # Production uses ~45/55 model/market — mirror it for fairness
        top2_market_score = {
            n: 0.45 * top2_prob_by_name.get(n, 0) + 0.55 * market_prob_by_name.get(n, 0)
            for n in names
        }
        top2_market_ranked = sorted(names, key=lambda n: -top2_market_score[n])

        # 50/50 ensemble of the two market-blended rankings
        blend_score = {
            n: 0.5 * win_prob_by_name.get(n, 0) + 0.5 * top2_market_score.get(n, 0)
            for n in names
        }
        blend_ranked = sorted(names, key=lambda n: -blend_score[n])

        actual_top2 = {h["horse_name"] for h in horse_rows
                       if h.get("finish_position") in (1, 2)}
        if len(actual_top2) < 2:
            continue  # incomplete result row

        n_races_used += 1
        name_to_num = {h["horse_name"]: h["horse_number"] for h in horse_rows}
        actual_top2_nums = sorted([h["horse_number"] for h in horse_rows
                                    if h.get("finish_position") in (1, 2)])
        actual_combo = (f"{min(actual_top2_nums)},{max(actual_top2_nums)}"
                        if len(actual_top2_nums) == 2 else None)

        for method, ranked in [("win_rank", win_ranked),
                                ("top2_raw", top2_raw_ranked),
                                ("top2_market", top2_market_ranked),
                                ("blend", blend_ranked)]:
            if set(ranked[:2]) == actual_top2:
                counts[f"{method}_top2"] += 1
            if actual_top2.issubset(set(ranked[:3])):
                counts[f"{method}_top3"] += 1
            if actual_top2.issubset(set(ranked[:4])):
                counts[f"{method}_top4"] += 1

            # Real-dividend P&L. Stake counts every race we bet on (always);
            # dividend lookup is on the ACTUAL winning combo when we hit.
            top2_nums = [name_to_num.get(n) for n in ranked[:2]]
            top3_nums = [name_to_num.get(n) for n in ranked[:3]]
            top4_nums = [name_to_num.get(n) for n in ranked[:4]]
            # Dividend for the ACTUAL winning pair — only looked up on hits
            actual_div = (_lookup_qin_dividend(rd, rn, *actual_top2_nums)
                          if actual_combo else None)

            # Straight QIN: 1 unit stake, hit if our top-2 == actual top-2
            if all(n is not None for n in top2_nums):
                pnl[method]["stake_straight"] += 1
                if set(ranked[:2]) == actual_top2 and actual_div is not None:
                    pnl[method]["straight"] += (actual_div - 1)
                else:
                    pnl[method]["straight"] -= 1
            # Box-3: 3 units stake (3 combos), hit if actual pair ⊂ our top-3
            if all(n is not None for n in top3_nums):
                pnl[method]["stake_box3"] += 3
                if actual_top2.issubset(set(ranked[:3])) and actual_div is not None:
                    pnl[method]["box3"] += (actual_div - 3)
                else:
                    pnl[method]["box3"] -= 3
            # Box-4: 6 units stake (6 combos)
            if all(n is not None for n in top4_nums):
                pnl[method]["stake_box4"] += 6
                if actual_top2.issubset(set(ranked[:4])) and actual_div is not None:
                    pnl[method]["box4"] += (actual_div - 6)
                else:
                    pnl[method]["box4"] -= 6

    print(f"\n{'method':<24s} {'top-2 hit':>14s} {'top-3 hit':>14s} {'top-4 hit':>14s}")
    print("-" * 70)
    label_map = {
        "win_rank":    "current (win + market)",
        "top2_raw":    "top2 raw (no market)",
        "top2_market": "top2 + market (45/55)",
        "blend":       "50/50 ensemble",
    }
    for method in ("win_rank", "top2_raw", "top2_market", "blend"):
        t2 = counts[f"{method}_top2"]
        t3 = counts[f"{method}_top3"]
        t4 = counts[f"{method}_top4"]
        print(f"  {label_map[method]:<22s} "
              f"{t2:>5d} ({100*t2/n_races_used:>4.1f}%)  "
              f"{t3:>5d} ({100*t3/n_races_used:>4.1f}%)  "
              f"{t4:>5d} ({100*t4/n_races_used:>4.1f}%)")
    print(f"\n  Races evaluated: {n_races_used}")

    # ROI table — real HKJC quinella dividends
    print(f"\n{'method':<24s} {'straight Q':>20s} {'box-3':>20s} {'box-4':>20s}")
    print("-" * 90)
    for method in ("win_rank", "top2_raw", "top2_market", "blend"):
        rows = pnl[method]
        def _fmt(pl, st):
            return f"${pl:+.0f} / ${st:>4} = {100*pl/max(st,1):+.0f}%"
        print(f"  {label_map[method]:<22s} "
              f"{_fmt(rows['straight'], rows['stake_straight']):>20s}  "
              f"{_fmt(rows['box3'], rows['stake_box3']):>20s}  "
              f"{_fmt(rows['box4'], rows['stake_box4']):>20s}")
    conn.close()


# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true",
                    help="Re-train the top2 model (default: load if cached)")
    args = ap.parse_args()

    if args.retrain or not TOP2_MODEL_PATH.exists():
        model, calibrator, metrics = train_top2_model()
        save_top2(model, calibrator, metrics)
    else:
        logger.info(f"Loading cached top2 model from {TOP2_MODEL_PATH}")

    payload = load_top2()
    evaluate_rankings(payload, "Backtest holdout (rows[85:100%])")


if __name__ == "__main__":
    sys.exit(main() or 0)
