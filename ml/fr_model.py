"""FR XGBoost model wrapper — MVP.

Mirror of ml/uk_model.py. Same hyperparameters, same save/load format,
different feature import + model path.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from ml.fr_features import FEATURE_COLS

MODEL_PATH = config.MODEL_DIR / "fr_racing_model.pkl"


XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "eval_metric": "logloss",
}


class FRRacingModel:
    """Train + predict wrapper for the FR is_winner classifier."""

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = list(FEATURE_COLS)
        self.trained_date: Optional[str] = None
        self.n_train_samples: Optional[int] = None
        self.training_metrics: dict = {}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if list(X_train.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch. Expected {self.feature_names}, got "
                f"{list(X_train.columns)}"
            )
        params = dict(XGB_PARAMS)
        if X_val is not None:
            params["early_stopping_rounds"] = 25
        self.model = xgb.XGBClassifier(**params)
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self.trained_date = datetime.utcnow().isoformat(timespec="seconds")
        self.n_train_samples = len(X_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained — call fit() first.")
        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch at predict time. Expected {self.feature_names}, "
                f"got {list(X.columns)}"
            )
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        return (
            pd.DataFrame({
                "feature": self.feature_names,
                "importance": self.model.feature_importances_,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path or MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "trained_date": self.trained_date,
            "n_train_samples": self.n_train_samples,
            "training_metrics": self.training_metrics,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "FRRacingModel":
        path = Path(path or MODEL_PATH)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        inst = cls()
        inst.model = payload["model"]
        inst.feature_names = payload["feature_names"]
        inst.trained_date = payload.get("trained_date")
        inst.n_train_samples = payload.get("n_train_samples")
        inst.training_metrics = payload.get("training_metrics", {})
        return inst
