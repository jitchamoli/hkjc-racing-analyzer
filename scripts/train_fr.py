"""Train the FR is_winner model. Mirror of scripts/train_uk.py.

Usage:
    python3 -m scripts.train_fr

4-fold temporal split: 60/10/15/15 train/earlystop/calibrate/test.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ml.fr_features import FEATURE_COLS, TARGET, build_training_table
from ml.fr_model import FRRacingModel
from scripts.oos_common import grouped_temporal_folds, cutoff_date_for


def split_temporal(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """4-fold temporal split with date-boundary snapping (no race split across
    folds). df must already be sorted by date."""
    return grouped_temporal_folds(df)


def main():
    t0 = time.time()
    print("Loading + computing features...")
    df = build_training_table()
    df = df.sort_values(["date", "race_id"], kind="stable").reset_index(drop=True)
    print(f"  Rows: {len(df):,}  date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Base rate: {df[TARGET].mean()*100:.2f}% winners")

    folds = split_temporal(df)
    for name, fold in folds.items():
        print(f"  {name:<10} n={len(fold):>6}  "
              f"dates {fold['date'].min().date()} → {fold['date'].max().date()}  "
              f"win_rate={fold[TARGET].mean()*100:.2f}%")

    X_tr, y_tr = folds["train"][FEATURE_COLS], folds["train"][TARGET]
    X_es, y_es = folds["earlystop"][FEATURE_COLS], folds["earlystop"][TARGET]
    X_te, y_te = folds["test"][FEATURE_COLS], folds["test"][TARGET]

    print("\nTraining XGBoost (early stopping on earlystop fold)...")
    model = FRRacingModel()
    model.fit(X_tr, y_tr, X_es, y_es)
    print(f"  Trained in {time.time()-t0:.1f}s on {len(X_tr):,} samples")

    print("\n=== Metrics ===")
    for name, X, y in [("train", X_tr, y_tr), ("earlystop", X_es, y_es), ("test", X_te, y_te)]:
        p = model.predict_proba(X)
        try:
            auc = roc_auc_score(y, p)
        except ValueError:
            auc = float("nan")
        brier = brier_score_loss(y, p)
        ll = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
        print(f"  {name:<10} AUC={auc:.4f}  Brier={brier:.4f}  LogLoss={ll:.4f}  n={len(y):,}")

    print("\n=== Baseline (SP-only) for comparison ===")
    test_df = folds["test"].copy()
    test_df["sp_imp"] = 1.0 / test_df["sp_dec"].replace(0, np.nan)
    test_df["sp_norm"] = test_df.groupby("race_id")["sp_imp"].transform(lambda x: x / x.sum())
    mask = test_df["sp_norm"].notna()
    sp_p = test_df.loc[mask, "sp_norm"].values
    y_te_sp = y_te.loc[mask].values
    print(f"  SP-only AUC={roc_auc_score(y_te_sp, sp_p):.4f}  "
          f"Brier={brier_score_loss(y_te_sp, sp_p):.4f}  "
          f"(n={len(sp_p):,} after dropping {(~mask).sum()} NaN rows)")

    print("\n=== Top features ===")
    print(model.feature_importance().to_string(index=False))

    p_te = model.predict_proba(X_te)
    model.training_metrics = {
        "test_brier": float(brier_score_loss(y_te, p_te)),
        "test_auc": float(roc_auc_score(y_te, p_te)),
        "sp_baseline_brier": float(brier_score_loss(y_te_sp, sp_p)),
        "sp_baseline_auc": float(roc_auc_score(y_te_sp, sp_p)),
        # Freeze the OOS cutoff so validate/backtest don't recompute it from the
        # (growing) live DB row count — see scripts/oos_common.py. cutoff_date_for
        # is robust to an empty calibrate fold (tiny datasets) → no NaT crash.
        "cutoff_date": cutoff_date_for(df).strftime("%Y-%m-%d"),
    }
    saved = model.save()
    print(f"\nSaved model to {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
