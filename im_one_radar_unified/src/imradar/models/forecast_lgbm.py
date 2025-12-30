from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from imradar.config import RadarConfig


@dataclass
class ForecastModel:
    """예측 모델 컨테이너"""
    kpi: str
    horizon: int
    feature_cols: List[str]
    categorical_cols: List[str]
    model: lgb.LGBMRegressor
    feature_importance: Optional[pd.DataFrame] = None
    cv_scores: Optional[Dict[str, List[float]]] = None
    train_residual_std: float = 0.0  # For prediction intervals
    
    def save(self, path: Path) -> None:
        """모델을 파일로 저장"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: Path) -> "ForecastModel":
        """파일에서 모델 로드"""
        with open(path, 'rb') as f:
            return pickle.load(f)


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Weighted Mean Absolute Percentage Error"""
    denom = np.sum(np.abs(y_true))
    denom = max(float(denom), eps)
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    """Mean Absolute Percentage Error"""
    denom = np.abs(y_true)
    denom = np.maximum(denom, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_feature_importance(model: lgb.LGBMRegressor, feature_cols: List[str]) -> pd.DataFrame:
    """피처 중요도 추출 (gain, split 기준)"""
    imp_gain = model.booster_.feature_importance(importance_type='gain')
    imp_split = model.booster_.feature_importance(importance_type='split')
    
    df = pd.DataFrame({
        'feature': feature_cols,
        'importance_gain': imp_gain,
        'importance_split': imp_split,
    })
    df['importance_gain_pct'] = df['importance_gain'] / df['importance_gain'].sum() * 100
    df['importance_split_pct'] = df['importance_split'] / df['importance_split'].sum() * 100
    
    return df.sort_values('importance_gain', ascending=False).reset_index(drop=True)


def train_lgbm_forecast(
    df: pd.DataFrame,
    kpi_col: str,
    horizon: int,
    time_col: str = "month",
    group_col: str = "segment_id",
    categorical_cols: Optional[List[str]] = None,
    train_end_month: Optional[pd.Timestamp] = None,
    cfg: Optional[RadarConfig] = None,
    use_early_stopping: bool = True,
    use_cv: bool = False,
    cv_folds: int = 3,
    verbose: bool = False,
) -> ForecastModel:
    """
    Train a direct multi-horizon LightGBM model with enhanced features:
      - Early stopping to prevent overfitting
      - Optional time-series cross-validation
      - Feature importance extraction
      - Residual std for prediction intervals
    
    Args:
        df: Input DataFrame with features
        kpi_col: Target KPI column name
        horizon: Forecast horizon (months ahead)
        time_col: Time column name
        group_col: Group/segment column name
        categorical_cols: List of categorical feature columns
        train_end_month: Training cutoff date
        cfg: RadarConfig instance
        use_early_stopping: Enable early stopping
        use_cv: Perform time-series cross-validation
        cv_folds: Number of CV folds
        verbose: Print training progress
    
    Returns:
        ForecastModel with trained model and metadata
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
    ignore_cols = {time_col, "first_month"}
    exclude = base_drop | ignore_cols

    # Candidate features: all columns excluding obvious non-features
    feature_cols = [c for c in data.columns if c not in exclude and c not in {"segment_id"}]
    if categorical_cols is None:
        categorical_cols = []
    feature_cols = [c for c in feature_cols if (c in categorical_cols) or (pd.api.types.is_numeric_dtype(data[c]))]

    X_train = data.loc[train_mask, feature_cols].copy()
    y_train = data.loc[train_mask, f"target__h{horizon}"].astype("float64").values

    # Ensure categorical dtype for LGBM
    cat_cols = [c for c in categorical_cols if c in X_train.columns]
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")

    # Base model parameters
    base_params = {
        'n_estimators': cfg.lgbm_n_estimators,
        'learning_rate': cfg.lgbm_learning_rate,
        'num_leaves': cfg.lgbm_num_leaves,
        'subsample': cfg.lgbm_subsample,
        'colsample_bytree': cfg.lgbm_colsample_bytree,
        'random_state': cfg.lgbm_random_state,
        'n_jobs': -1,
        'verbosity': -1 if not verbose else 1,
    }
    
    # Enhanced regularization parameters
    base_params.update({
        'min_child_samples': cfg.lgbm_min_child_samples,
        'reg_alpha': cfg.lgbm_reg_alpha,
        'reg_lambda': cfg.lgbm_reg_lambda,
    })

    cv_scores = None
    
    # Time-series cross-validation
    if use_cv and len(X_train) >= cv_folds * 10:
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        cv_results = {'mae': [], 'rmse': [], 'smape': [], 'r2': []}
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            cv_model = lgb.LGBMRegressor(**base_params)
            
            if use_early_stopping:
                cv_model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=cfg.lgbm_early_stopping_rounds, verbose=verbose),
                        lgb.log_evaluation(period=0 if not verbose else 50),
                    ],
                    categorical_feature=cat_cols,
                )
            else:
                cv_model.fit(X_tr, y_tr, categorical_feature=cat_cols)
            
            y_pred = cv_model.predict(X_val)
            cv_results['mae'].append(float(mean_absolute_error(y_val, y_pred)))
            cv_results['rmse'].append(_rmse(y_val, y_pred))
            cv_results['smape'].append(_smape(y_val, y_pred))
            cv_results['r2'].append(float(r2_score(y_val, y_pred)))
        
        cv_scores = cv_results
        if verbose:
            print(f"CV MAE: {np.mean(cv_results['mae']):.4f} ± {np.std(cv_results['mae']):.4f}")
            print(f"CV R²: {np.mean(cv_results['r2']):.4f} ± {np.std(cv_results['r2']):.4f}")

    # Final model training
    model = lgb.LGBMRegressor(**base_params)
    
    if use_early_stopping:
        # Split for early stopping validation
        split_idx = int(len(X_train) * 0.85)
        X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=cfg.lgbm_early_stopping_rounds, verbose=verbose),
                lgb.log_evaluation(period=0 if not verbose else 50),
            ],
            categorical_feature=cat_cols,
        )
    else:
        model.fit(X_train, y_train, categorical_feature=cat_cols)

    # Calculate residual std for prediction intervals
    y_pred_train = model.predict(X_train)
    train_residual_std = float(np.std(y_train - y_pred_train))
    
    # Extract feature importance
    feature_importance = get_feature_importance(model, feature_cols)

    return ForecastModel(
        kpi=kpi_col,
        horizon=horizon,
        feature_cols=feature_cols,
        categorical_cols=cat_cols,
        model=model,
        feature_importance=feature_importance,
        cv_scores=cv_scores,
        train_residual_std=train_residual_std,
    )


def predict_with_model(
    fm: ForecastModel, 
    df: pd.DataFrame,
    return_interval: bool = False,
    confidence: float = 0.95,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    모델을 사용하여 예측 수행
    
    Args:
        fm: ForecastModel instance
        df: Input DataFrame
        return_interval: If True, return prediction intervals
        confidence: Confidence level for intervals (default 95%)
    
    Returns:
        predictions or (predictions, lower_bound, upper_bound)
    """
    X = df[fm.feature_cols].copy()
    for c in fm.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    
    predictions = fm.model.predict(X)
    
    if return_interval:
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * fm.train_residual_std
        lower = predictions - margin
        upper = predictions + margin
        return predictions, lower, upper
    
    return predictions


def predict_with_uncertainty(
    fm: ForecastModel,
    df: pd.DataFrame,
    n_iterations: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap 기반 불확실성 추정
    
    Args:
        fm: ForecastModel instance
        df: Input DataFrame
        n_iterations: Number of bootstrap iterations
    
    Returns:
        (mean_predictions, std_predictions)
    """
    X = df[fm.feature_cols].copy()
    for c in fm.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    
    base_pred = fm.model.predict(X)
    
    # Add random noise based on training residual std
    all_preds = np.zeros((n_iterations, len(X)))
    for i in range(n_iterations):
        noise = np.random.normal(0, fm.train_residual_std, len(X))
        all_preds[i] = base_pred + noise
    
    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)


def evaluate_forecast(
    df: pd.DataFrame,
    fm: ForecastModel,
    time_col: str = "month",
    group_col: str = "segment_id",
    test_start_month: Optional[pd.Timestamp] = None,
    test_end_month: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """
    예측 모델 성능 평가 (확장된 메트릭)
    
    Returns:
        Dict with n, mae, rmse, mape, smape, wmape, r2
    """
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
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, 
                "smape": np.nan, "wmape": np.nan, "r2": np.nan}

    y_true = test[f"target__h{fm.horizon}"].astype("float64").values
    y_pred = predict_with_model(fm, test)

    return {
        "n": float(len(test)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "wmape": _wmape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def get_top_features(fm: ForecastModel, top_n: int = 20) -> pd.DataFrame:
    """상위 N개 중요 피처 반환"""
    if fm.feature_importance is None:
        return pd.DataFrame()
    return fm.feature_importance.head(top_n)


def compare_models(models: List[ForecastModel]) -> pd.DataFrame:
    """여러 모델의 CV 성능 비교"""
    results = []
    for m in models:
        if m.cv_scores is not None:
            results.append({
                'kpi': m.kpi,
                'horizon': m.horizon,
                'cv_mae_mean': np.mean(m.cv_scores['mae']),
                'cv_mae_std': np.std(m.cv_scores['mae']),
                'cv_r2_mean': np.mean(m.cv_scores['r2']),
                'cv_r2_std': np.std(m.cv_scores['r2']),
            })
    return pd.DataFrame(results)
