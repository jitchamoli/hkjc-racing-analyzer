#!/usr/bin/env python3
"""Lightweight experiment tracking — appends metrics per training run to JSON."""

import json
import logging
from datetime import datetime
from typing import Dict, List

import config

logger = logging.getLogger(__name__)


def log_experiment(metrics: Dict, features_used: List[str], n_samples: int,
                   feature_correlations: Dict = None) -> None:
    """Append one entry per training run to experiment log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model_version": metrics.get("model_version", "unknown"),
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "logloss": metrics.get("logloss"),
        "brier_score": metrics.get("brier_score"),
        "place_accuracy": metrics.get("place_accuracy"),
        "n_samples": n_samples,
        "n_features": len(features_used),
        "feature_importance": metrics.get("feature_importance"),
    }
    if feature_correlations:
        entry["high_correlations"] = feature_correlations

    log_path = config.EXPERIMENT_LOG_PATH
    log_path.parent.mkdir(exist_ok=True)

    existing = []
    if log_path.exists():
        try:
            with open(log_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    existing.append(entry)

    with open(log_path, 'w') as f:
        json.dump(existing, f, indent=2)

    logger.info(f"Logged experiment: accuracy={entry['accuracy']:.4f}, "
                f"n_samples={n_samples}, version={entry['model_version']}")


def get_experiments() -> List[Dict]:
    """Load all experiment entries."""
    log_path = config.EXPERIMENT_LOG_PATH
    if not log_path.exists():
        return []
    try:
        with open(log_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
