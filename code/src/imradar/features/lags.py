from __future__ import annotations

from typing import List, Sequence
import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
    lags: Sequence[int] = (1, 2, 3, 6, 12),
) -> pd.DataFrame:
    out = df.sort_values([group_col, time_col]).copy()
    for c in value_cols:
        if c not in out.columns:
            continue
        for l in lags:
            out[f"{c}__lag{l}"] = out.groupby(group_col)[c].shift(l)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
    windows: Sequence[int] = (3, 6),
) -> pd.DataFrame:
    out = df.sort_values([group_col, time_col]).copy()
    for c in value_cols:
        if c not in out.columns:
            continue
        g = out.groupby(group_col)[c]
        for w in windows:
            out[f"{c}__roll{w}_mean"] = g.shift(1).rolling(w).mean().reset_index(level=0, drop=True)
            out[f"{c}__roll{w}_std"] = g.shift(1).rolling(w).std().reset_index(level=0, drop=True)
    return out


def add_change_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
) -> pd.DataFrame:
    out = df.sort_values([group_col, time_col]).copy()
    for c in value_cols:
        if c not in out.columns:
            continue
        g = out.groupby(group_col)[c]
        out[f"{c}__mom"] = g.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out[f"{c}__yoy"] = g.pct_change(12, fill_method=None).replace([np.inf, -np.inf], np.nan)
    return out
