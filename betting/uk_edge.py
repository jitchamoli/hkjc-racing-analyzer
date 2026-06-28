"""UK/GB value-betting core — edge + staking math.

Isolated from the HK betting modules (edge_calculator/conviction/exotic_*),
which are HK-pool-specific. This is fixed-odds (SP) value betting:

  market_implied = 1 / sp_dec
  overround      = sum(market_implied) over the race  (the book's > 1.0)
  fair_market_p  = market_implied / overround          (de-vigged market prob)
  model_p_norm   = model win prob, normalised to sum 1 within the race
  edge           = model_p_norm - fair_market_p        (our prob advantage)
  overlay        = model_p_norm * sp_dec               (>1 = positive EV)

Bet WIN only (the UK MVP model is win-only — no place model, so no honest
each-way). Staking: flat, or fractional Kelly capped for safety.

Shared by scripts.betting_engine_uk (OOS backtest) and the live simulcast
dashboard so the two never drift. See memory:
project_overseas_db_location_and_ireland_dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_edge_columns(df: pd.DataFrame, prob_col: str = "model_p",
                     sp_col: str = "sp_dec", race_col: str = "race_id") -> pd.DataFrame:
    """Return df with model_p_norm, mkt_implied, mkt_fair, edge, overlay.

    Rows with missing/invalid SP get NaN market columns (excluded downstream).
    model_p is normalised within each race over runners that HAVE a valid SP,
    so model_p_norm and mkt_fair live on the same support.
    """
    df = df.copy()
    valid = df[sp_col].notna() & (df[sp_col] > 1.0)
    df["mkt_implied"] = np.where(valid, 1.0 / df[sp_col], np.nan)

    # per-race overround over valid runners only
    over = df.assign(_mi=df["mkt_implied"]).groupby(race_col)["_mi"].transform("sum")
    df["mkt_fair"] = df["mkt_implied"] / over

    # normalise model prob within race over the SAME valid support
    mp = np.where(valid, df[prob_col], np.nan)
    df["_mp"] = mp
    psum = df.groupby(race_col)["_mp"].transform("sum")
    df["model_p_norm"] = df["_mp"] / psum

    df["edge"] = df["model_p_norm"] - df["mkt_fair"]
    df["overlay"] = df["model_p_norm"] * df[sp_col]
    return df.drop(columns=["_mi", "_mp"], errors="ignore")


def kelly_fraction(p: float, sp_dec: float, cap: float = 0.05) -> float:
    """Fractional-Kelly stake (as a fraction of bankroll), capped.

    f* = (p*b - q) / b,  b = sp_dec - 1,  q = 1 - p.  Negative → 0 (no bet).
    Capped at `cap` (default 5%) — full Kelly is too aggressive for an
    unvalidated MVP edge.
    """
    if not (sp_dec and sp_dec > 1.0) or not (0.0 < p < 1.0):
        return 0.0
    b = sp_dec - 1.0
    f = (p * b - (1.0 - p)) / b
    return float(min(max(f, 0.0), cap))


def select_value_bets(df: pd.DataFrame, min_edge: float = 0.0,
                      top1_only: bool = False, race_col: str = "race_id"):
    """Return the rows to bet WIN: positive edge >= min_edge (and, if
    top1_only, the model's highest-prob runner in the race). Expects the
    edge columns from add_edge_columns().
    """
    sel = df["edge"].notna() & (df["edge"] >= min_edge) & (df["overlay"] > 1.0)
    if top1_only:
        top = df.groupby(race_col)["model_p_norm"].transform("max")
        sel &= (df["model_p_norm"] == top)
    return df[sel]
