from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error

from imradar.config import RadarConfig


@dataclass
class ForecastModel:
    kpi: str
    horizon: int
    feature_cols: List[str]
    categorical_cols: List[str]
    model: lgb.LGBMRegressor


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.sum(np.abs(y_true))
    denom = max(float(denom), eps)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def train_lgbm_forecast(
    df: pd.DataFrame,
    kpi_col: str,
    horizon: int,
    time_col: str = "month",
    group_col: str = "segment_id",
    categorical_cols: Optional[List[str]] = None,
    train_end_month: Optional[pd.Timestamp] = None,
    cfg: Optional[RadarConfig] = None,
) -> ForecastModel:
    """
    Train a direct multi-horizon LightGBM model:
      features at time t -> target at time t+h
    """
    if cfg is None:
        cfg = RadarConfig()
    
    data = df.sort_values([group_col, time_col]).copy()

    # Create shifted target
    data[f"target__h{horizon}"] = data.groupby(group_col)[kpi_col].shift(-horizon)

    # Drop pre-birth rows and rows without target
    data = data[(data.get("pre_birth", 0) == 0)].copy()
    data = data.dropna(subset=[f"target__h{horizon}"])

    # Train cutoff
    if train_end_month is not None:
        train_mask = data[time_col] <= train_end_month
    else:
        train_mask = np.ones(len(data), dtype=bool)

    # Feature selection: use numeric engineered features + static cats + time features
    base_drop = {kpi_col, f"target__h{horizon}"}
    ignore_cols = {time_col, "first_month"}  # keep 't', year, month_num
    exclude = base_drop | ignore_cols

    # Candidate features: all columns excluding obvious non-features
    feature_cols = [c for c in data.columns if c not in exclude and c not in {"segment_id"}]
    # Remove raw string columns if not in categorical list
    if categorical_cols is None:
        categorical_cols = []
    feature_cols = [c for c in feature_cols if (c in categorical_cols) or (pd.api.types.is_numeric_dtype(data[c]))]

    X_train = data.loc[train_mask, feature_cols].copy()
    y_train = data.loc[train_mask, f"target__h{horizon}"].astype("float64").values

    # Ensure categorical dtype for LGBM
    cat_cols = [c for c in categorical_cols if c in X_train.columns]
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")

    model = lgb.LGBMRegressor(
        n_estimators=cfg.lgbm_n_estimators,
        learning_rate=cfg.lgbm_learning_rate,
        num_leaves=cfg.lgbm_num_leaves,
        subsample=cfg.lgbm_subsample,
        colsample_bytree=cfg.lgbm_colsample_bytree,
        random_state=cfg.lgbm_random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, categorical_feature=cat_cols)

    return ForecastModel(
        kpi=kpi_col,
        horizon=horizon,
        feature_cols=feature_cols,
        categorical_cols=cat_cols,
        model=model,
    )


def predict_with_model(fm: ForecastModel, df: pd.DataFrame) -> np.ndarray:
    X = df[fm.feature_cols].copy()
    for c in fm.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return fm.model.predict(X)


def evaluate_forecast(
    df: pd.DataFrame,
    fm: ForecastModel,
    time_col: str = "month",
    group_col: str = "segment_id",
    test_start_month: Optional[pd.Timestamp] = None,
    test_end_month: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    data = df.sort_values([group_col, time_col]).copy()
    data[f"target__h{fm.horizon}"] = data.groupby(group_col)[fm.kpi].shift(-fm.horizon)
    data = data[(data.get("pre_birth", 0) == 0)].copy()
    data = data.dropna(subset=[f"target__h{fm.horizon}"])

    mask = np.ones(len(data), dtype=bool)
    if test_start_month is not None:
        mask &= data[time_col] >= test_start_month
    if test_end_month is not None:
        mask &= data[time_col] <= test_end_month

    test = data.loc[mask].copy()
    if test.empty:
        return {"n": 0, "mae": np.nan, "smape": np.nan, "wmape": np.nan}

    y_true = test[f"target__h{fm.horizon}"].astype("float64").values
    y_pred = predict_with_model(fm, test)

    return {
        "n": float(len(test)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "smape": _smape(y_true, y_pred),
        "wmape": _wmape(y_true, y_pred),
    }
