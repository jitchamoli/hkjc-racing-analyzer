"""On-disk feature cache for repeated backtest runs.

Backtest re-runs the same race history under different model/calibrator
versions, but the FEATURES for each (race_date, race_number, horse_name)
are deterministic given the schema and the historical DB state. Caching
them to disk shaves ~5-7s off every repeated backtest.

Cache key encodes:
  - NUM_FEATURES (schema width)         — schema change → new file
  - total race_results row count        — new data → new file
The cache file is written to:
  tracking_data/cache/features_v{NUM_FEATURES}_n{rows}.npz

Opt-in via `backtest_use_feature_cache(True)`. Default OFF so live
prediction paths (race-day) are never touched.

NOT a general-purpose cache. Only valid for backtest semantics where:
  - features depend only on prior-race data (no leakage)
  - DB is read-only for the duration of the backtest
  - the same race_results rows produce identical features
"""
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

import config
from ml.feature_schema import NUM_FEATURES

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "tracking_data" / "cache"

# Module-level state. Off by default. Backtest flips it on.
_ENABLED: bool = False
_CACHE: Dict[Tuple[str, int, str], np.ndarray] = {}
_CACHE_PATH: Optional[Path] = None
_DIRTY: bool = False  # True if any new entries since last save


def _db_row_count() -> int:
    """Count race_results rows. Used as a coarse data-freshness key."""
    try:
        conn = sqlite3.connect(str(config.DB_PATH))
        n = conn.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]
        conn.close()
        return int(n)
    except Exception:
        return 0


def _cache_path() -> Path:
    n_rows = _db_row_count()
    return CACHE_DIR / f"features_v{NUM_FEATURES}_n{n_rows}.npz"


def enable() -> None:
    """Turn the cache on for this process and load any existing file."""
    global _ENABLED, _CACHE, _CACHE_PATH, _DIRTY
    _ENABLED = True
    _CACHE = {}
    _DIRTY = False
    _CACHE_PATH = _cache_path()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _CACHE_PATH.exists():
        try:
            data = np.load(_CACHE_PATH, allow_pickle=False)
            keys = data["keys"]      # shape (N,) of fixed-width strings 'date|race|horse'
            features = data["features"]  # shape (N, NUM_FEATURES)
            if features.shape[1] != NUM_FEATURES:
                logger.warning(
                    f"Feature cache schema mismatch ({features.shape[1]} cols vs "
                    f"NUM_FEATURES={NUM_FEATURES}); ignoring."
                )
                return
            for k, v in zip(keys, features):
                parts = str(k).split("|", 2)
                if len(parts) == 3:
                    rd, rn, hn = parts[0], int(parts[1]), parts[2]
                    _CACHE[(rd, rn, hn)] = v
            logger.info(f"Feature cache loaded: {len(_CACHE):,} entries from {_CACHE_PATH.name}")
        except Exception as e:
            logger.warning(f"Failed to load feature cache: {e}; starting empty")
            _CACHE = {}


def disable() -> None:
    """Turn the cache off and clear state."""
    global _ENABLED, _CACHE, _CACHE_PATH, _DIRTY
    _ENABLED = False
    _CACHE = {}
    _CACHE_PATH = None
    _DIRTY = False


def is_enabled() -> bool:
    return _ENABLED


def get(race_date: str, race_number: int, horse_name: str) -> Optional[np.ndarray]:
    """Return cached feature vector or None on miss."""
    if not _ENABLED:
        return None
    return _CACHE.get((race_date, race_number, horse_name))


def put(race_date: str, race_number: int, horse_name: str, features: np.ndarray) -> None:
    """Store a feature vector. No-op when cache disabled."""
    global _DIRTY
    if not _ENABLED:
        return
    if features is None or features.shape != (NUM_FEATURES,):
        return
    key = (race_date, race_number, horse_name)
    if key in _CACHE:
        return
    _CACHE[key] = features.astype(np.float32, copy=False)
    _DIRTY = True


def save() -> None:
    """Write cache to disk if dirty. Safe no-op when nothing to write.

    Atomic write trick: numpy auto-appends `.npz` to whatever path you
    pass, so we write to `<stem>.tmp` (which numpy turns into
    `<stem>.tmp.npz`) then rename to `<stem>.npz`.
    """
    global _DIRTY
    if not _ENABLED or not _DIRTY or _CACHE_PATH is None or not _CACHE:
        return
    try:
        keys = np.array([f"{rd}|{rn}|{hn}" for (rd, rn, hn) in _CACHE.keys()])
        features = np.array(list(_CACHE.values()), dtype=np.float32)
        tmp_stem = _CACHE_PATH.parent / (_CACHE_PATH.stem + ".tmp")
        np.savez_compressed(str(tmp_stem), keys=keys, features=features)
        tmp_written = Path(str(tmp_stem) + ".npz")
        tmp_written.replace(_CACHE_PATH)
        logger.info(f"Feature cache saved: {len(_CACHE):,} entries → {_CACHE_PATH.name}")
        _DIRTY = False
    except Exception as e:
        logger.warning(f"Failed to save feature cache: {e}")
