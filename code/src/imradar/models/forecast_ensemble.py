"""
앙상블 예측 모델 + 하이퍼파라미터 튜닝

개선 사항:
1. Optuna 기반 하이퍼파라미터 자동 튜닝
2. LightGBM + XGBoost + CatBoost 앙상블
3. 피처 선택 (Feature Selection)
4. 가중 평균 앙상블
5. Quantile Regression for uncertainty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union, Any
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================================
# KPI별 최적 앙상블 가중치 (Stacking 실험 결과)
# ============================================================================
# 2026-01-12 advanced_model_improvements.py 실험 결과 반영
# - 예금: CatBoost(60.3%) > LGBM(35.2%) > XGB(9.3%)
# - 대출: LGBM(45.8%) > XGB(31.6%) > CatBoost(25.2%)
# - 순유입: CatBoost(80.5%) > XGB(9.8%) > LGBM(3.4%)  ★ 핵심 개선
KPI_OPTIMAL_WEIGHTS = {
    '예금총잔액': {'lgbm': 0.352, 'xgb': 0.093, 'catboost': 0.603},
    '대출총잔액': {'lgbm': 0.458, 'xgb': 0.316, 'catboost': 0.252},
    '순유입': {'lgbm': 0.034, 'xgb': 0.098, 'catboost': 0.805},
    '카드총사용': {'lgbm': 0.40, 'xgb': 0.30, 'catboost': 0.30},  # 기본값
    '디지털거래금액': {'lgbm': 0.40, 'xgb': 0.30, 'catboost': 0.30},  # 기본값
    'FX총액': {'lgbm': 0.40, 'xgb': 0.30, 'catboost': 0.30},  # 기본값
}

# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class EnsembleModel:
    """앙상블 예측 모델 컨테이너"""
    kpi: str
    horizon: int
    feature_cols: List[str]
    categorical_cols: List[str]
    models: Dict[str, Any]  # {'lgbm': model, 'xgb': model, 'catboost': model}
    weights: Dict[str, float]  # {'lgbm': 0.4, 'xgb': 0.3, 'catboost': 0.3}
    best_params: Dict[str, Dict]  # 각 모델의 최적 하이퍼파라미터
    feature_importance: Optional[pd.DataFrame] = None
    cv_scores: Optional[Dict[str, float]] = None
    train_residual_std: float = 0.0
    
    def save(self, path: Path) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: Path) -> "EnsembleModel":
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# 메트릭 함수
# ============================================================================

def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================================
# Optuna 하이퍼파라미터 튜닝
# ============================================================================

def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: List[str],
    n_trials: int = 30,
    timeout: int = 300,
) -> Dict:
    """LightGBM 하이퍼파라미터 튜닝"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 16, 128),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            categorical_feature=cat_cols if cat_cols else 'auto',
        )
        
        y_pred = model.predict(X_val)
        return _rmse(y_val, y_pred)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 30,
    timeout: int = 300,
) -> Dict:
    """XGBoost 하이퍼파라미터 튜닝"""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        y_pred = model.predict(X_val)
        return _rmse(y_val, y_pred)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: List[str],
    n_trials: int = 30,
    timeout: int = 300,
) -> Dict:
    """CatBoost 하이퍼파라미터 튜닝"""
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': 42,
            'verbose': False,
        }
        
        cat_indices = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            cat_features=cat_indices if cat_indices else None,
            early_stopping_rounds=50,
            verbose=False,
        )
        
        y_pred = model.predict(X_val)
        return _rmse(y_val, y_pred)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    
    return study.best_params


# ============================================================================
# 피처 선택
# ============================================================================

def select_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    cat_cols: List[str],
    threshold: str = "0.5*median",
) -> List[str]:
    """LightGBM 기반 피처 선택"""
    
    X_temp = X_train.copy()
    for c in cat_cols:
        if c in X_temp.columns:
            X_temp[c] = X_temp[c].astype('category')
    
    selector_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    selector_model.fit(X_temp, y_train, categorical_feature=cat_cols if cat_cols else 'auto')
    
    selector = SelectFromModel(selector_model, prefit=True, threshold=threshold)
    selected_mask = selector.get_support()
    
    return [col for col, selected in zip(X_train.columns, selected_mask) if selected]


# ============================================================================
# 앙상블 모델 학습
# ============================================================================

def train_ensemble_forecast(
    df: pd.DataFrame,
    kpi_col: str,
    horizon: int,
    time_col: str = "month",
    group_col: str = "segment_id",
    categorical_cols: Optional[List[str]] = None,
    train_end_month: Optional[pd.Timestamp] = None,
    tune_hyperparams: bool = True,
    n_trials: int = 30,
    tune_timeout: int = 300,
    use_feature_selection: bool = True,
    verbose: bool = True,
) -> EnsembleModel:
    """
    앙상블 예측 모델 학습 (LightGBM + XGBoost + CatBoost)
    
    Args:
        df: 입력 데이터프레임
        kpi_col: 타겟 KPI 컬럼
        horizon: 예측 호라이즌
        time_col: 시간 컬럼
        group_col: 그룹 컬럼
        categorical_cols: 범주형 컬럼 리스트
        train_end_month: 학습 종료 월
        tune_hyperparams: 하이퍼파라미터 튜닝 여부
        n_trials: Optuna trial 수
        tune_timeout: 튜닝 타임아웃 (초)
        use_feature_selection: 피처 선택 사용 여부
        verbose: 진행 상황 출력
    
    Returns:
        EnsembleModel
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"앙상블 모델 학습: {kpi_col} (H={horizon})")
        print(f"{'='*50}")
    
    data = df.sort_values([group_col, time_col]).copy()
    
    # 타겟 생성
    data[f"target__h{horizon}"] = data.groupby(group_col)[kpi_col].shift(-horizon)
    data = data[(data.get("pre_birth", 0) == 0)].copy()
    data = data.dropna(subset=[f"target__h{horizon}"])
    
    # 학습/검증 분할
    if train_end_month is not None:
        train_mask = data[time_col] <= train_end_month
    else:
        train_mask = np.ones(len(data), dtype=bool)
    
    # 피처 선택
    base_drop = {kpi_col, f"target__h{horizon}"}
    ignore_cols = {time_col, "first_month", group_col}
    exclude = base_drop | ignore_cols
    
    feature_cols = [c for c in data.columns if c not in exclude]
    if categorical_cols is None:
        categorical_cols = []
    feature_cols = [c for c in feature_cols if (c in categorical_cols) or (pd.api.types.is_numeric_dtype(data[c]))]
    
    X_all = data.loc[train_mask, feature_cols].copy()
    y_all = data.loc[train_mask, f"target__h{horizon}"].astype("float64").values
    
    cat_cols = [c for c in categorical_cols if c in X_all.columns]
    for c in cat_cols:
        X_all[c] = X_all[c].astype("category")
    
    # 피처 선택
    if use_feature_selection and len(feature_cols) > 30:
        if verbose:
            print(f"  피처 선택 중... (현재: {len(feature_cols)}개)")
        selected_features = select_features(X_all, y_all, cat_cols)
        if len(selected_features) >= 10:
            feature_cols = selected_features
            X_all = X_all[feature_cols]
            cat_cols = [c for c in cat_cols if c in feature_cols]
            if verbose:
                print(f"  → 선택된 피처: {len(feature_cols)}개")
    
    # Train/Val 분할
    split_idx = int(len(X_all) * 0.8)
    X_train, X_val = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]
    
    # XGBoost/CatBoost용 데이터 준비 (category → numeric)
    X_train_xgb = X_train.copy()
    X_val_xgb = X_val.copy()
    for c in cat_cols:
        if c in X_train_xgb.columns:
            X_train_xgb[c] = X_train_xgb[c].cat.codes.replace(-1, np.nan)
            X_val_xgb[c] = X_val_xgb[c].cat.codes.replace(-1, np.nan)
    
    best_params = {}
    models = {}
    val_scores = {}
    
    # ====== LightGBM ======
    if verbose:
        print("\n  [1/3] LightGBM 학습...")
    
    if tune_hyperparams:
        if verbose:
            print(f"    → 하이퍼파라미터 튜닝 중 (n_trials={n_trials})...")
        lgbm_params = tune_lgbm(X_train, y_train, X_val, y_val, cat_cols, n_trials, tune_timeout)
    else:
        lgbm_params = {
            'n_estimators': 800, 'learning_rate': 0.05, 'num_leaves': 63,
            'max_depth': 8, 'min_child_samples': 20, 'subsample': 0.9,
            'colsample_bytree': 0.9, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
        }
    
    lgbm_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
    best_params['lgbm'] = lgbm_params
    
    lgbm_model = lgb.LGBMRegressor(**lgbm_params)
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        categorical_feature=cat_cols if cat_cols else 'auto',
    )
    models['lgbm'] = lgbm_model
    val_scores['lgbm'] = _rmse(y_val, lgbm_model.predict(X_val))
    
    if verbose:
        print(f"    ✓ LightGBM RMSE: {val_scores['lgbm']:.4f}")
    
    # ====== XGBoost ======
    if verbose:
        print("\n  [2/3] XGBoost 학습...")
    
    if tune_hyperparams:
        if verbose:
            print(f"    → 하이퍼파라미터 튜닝 중 (n_trials={n_trials})...")
        xgb_params = tune_xgboost(X_train_xgb, y_train, X_val_xgb, y_val, n_trials, tune_timeout)
    else:
        xgb_params = {
            'n_estimators': 800, 'learning_rate': 0.05, 'max_depth': 8,
            'min_child_weight': 5, 'subsample': 0.9, 'colsample_bytree': 0.9,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'gamma': 0.1,
        }
    
    xgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': 0})
    best_params['xgb'] = xgb_params
    
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=50)
    xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_val_xgb, y_val)], verbose=False)
    models['xgb'] = xgb_model
    val_scores['xgb'] = _rmse(y_val, xgb_model.predict(X_val_xgb))
    
    if verbose:
        print(f"    ✓ XGBoost RMSE: {val_scores['xgb']:.4f}")
    
    # ====== CatBoost ======
    if verbose:
        print("\n  [3/3] CatBoost 학습...")
    
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    for c in cat_cols:
        if c in X_train_cat.columns:
            X_train_cat[c] = X_train_cat[c].astype(str)
            X_val_cat[c] = X_val_cat[c].astype(str)
    
    if tune_hyperparams:
        if verbose:
            print(f"    → 하이퍼파라미터 튜닝 중 (n_trials={n_trials})...")
        cat_params = tune_catboost(X_train_cat, y_train, X_val_cat, y_val, cat_cols, n_trials, tune_timeout)
    else:
        cat_params = {
            'iterations': 800, 'learning_rate': 0.05, 'depth': 8,
            'l2_leaf_reg': 3, 'bagging_temperature': 0.5, 'random_strength': 1,
        }
    
    cat_params.update({'random_seed': 42, 'verbose': False})
    best_params['catboost'] = cat_params
    
    cat_indices = [X_train_cat.columns.get_loc(c) for c in cat_cols if c in X_train_cat.columns]
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train_cat, y_train,
        eval_set=(X_val_cat, y_val),
        cat_features=cat_indices if cat_indices else None,
        early_stopping_rounds=50,
        verbose=False,
    )
    models['catboost'] = cat_model
    val_scores['catboost'] = _rmse(y_val, cat_model.predict(X_val_cat))
    
    if verbose:
        print(f"    ✓ CatBoost RMSE: {val_scores['catboost']:.4f}")
    
    # ====== 앙상블 가중치 계산 ======
    # KPI별 최적 가중치가 있으면 사용, 없으면 RMSE 기반 계산
    kpi_name = kpi_col.replace('_target_h1', '').replace('_target_h2', '').replace('_target_h3', '')
    
    if kpi_name in KPI_OPTIMAL_WEIGHTS:
        # Stacking 실험 결과 반영된 최적 가중치 사용
        weights = KPI_OPTIMAL_WEIGHTS[kpi_name].copy()
        if verbose:
            print(f"\n  📌 [{kpi_name}] 최적 가중치 적용:")
            print(f"      LightGBM={weights['lgbm']:.3f}, XGBoost={weights['xgb']:.3f}, CatBoost={weights['catboost']:.3f}")
    else:
        # RMSE 기반 가중치 계산 (기본값)
        inv_scores = {k: 1.0 / (v + 1e-9) for k, v in val_scores.items()}
        total_inv = sum(inv_scores.values())
        weights = {k: v / total_inv for k, v in inv_scores.items()}
        if verbose:
            print(f"\n  앙상블 가중치 (RMSE 기반): LightGBM={weights['lgbm']:.3f}, XGBoost={weights['xgb']:.3f}, CatBoost={weights['catboost']:.3f}")
    
    # 앙상블 예측 검증
    y_pred_lgbm = lgbm_model.predict(X_val)
    y_pred_xgb = xgb_model.predict(X_val_xgb)
    y_pred_cat = cat_model.predict(X_val_cat)
    
    y_pred_ensemble = (
        weights['lgbm'] * y_pred_lgbm +
        weights['xgb'] * y_pred_xgb +
        weights['catboost'] * y_pred_cat
    )
    
    ensemble_rmse = _rmse(y_val, y_pred_ensemble)
    ensemble_r2 = r2_score(y_val, y_pred_ensemble)
    
    if verbose:
        print(f"\n  ★ 앙상블 RMSE: {ensemble_rmse:.4f} | R²: {ensemble_r2:.4f}")
    
    # 잔차 표준편차 계산
    train_residual_std = float(np.std(y_val - y_pred_ensemble))
    
    # 피처 중요도 (LightGBM 기준)
    imp_gain = lgbm_model.booster_.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': imp_gain,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    cv_scores = {
        'ensemble_rmse': ensemble_rmse,
        'ensemble_r2': ensemble_r2,
        'lgbm_rmse': val_scores['lgbm'],
        'xgb_rmse': val_scores['xgb'],
        'catboost_rmse': val_scores['catboost'],
    }
    
    return EnsembleModel(
        kpi=kpi_col,
        horizon=horizon,
        feature_cols=feature_cols,
        categorical_cols=cat_cols,
        models=models,
        weights=weights,
        best_params=best_params,
        feature_importance=feature_importance,
        cv_scores=cv_scores,
        train_residual_std=train_residual_std,
    )


def predict_ensemble(
    model: EnsembleModel,
    df: pd.DataFrame,
    return_individual: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    앙상블 모델로 예측
    
    Args:
        model: EnsembleModel 인스턴스
        df: 입력 데이터
        return_individual: 개별 모델 예측값 반환 여부
    
    Returns:
        앙상블 예측값 또는 개별 모델 예측값 딕셔너리
    """
    X = df[model.feature_cols].copy()
    
    # LightGBM용
    X_lgbm = X.copy()
    for c in model.categorical_cols:
        if c in X_lgbm.columns:
            X_lgbm[c] = X_lgbm[c].astype("category")
    
    # XGBoost용
    X_xgb = X.copy()
    for c in model.categorical_cols:
        if c in X_xgb.columns:
            X_xgb[c] = pd.Categorical(X_xgb[c]).codes
            X_xgb[c] = X_xgb[c].replace(-1, np.nan)
    
    # CatBoost용
    X_cat = X.copy()
    for c in model.categorical_cols:
        if c in X_cat.columns:
            X_cat[c] = X_cat[c].astype(str)
    
    # 개별 예측
    preds = {
        'lgbm': model.models['lgbm'].predict(X_lgbm),
        'xgb': model.models['xgb'].predict(X_xgb),
        'catboost': model.models['catboost'].predict(X_cat),
    }
    
    # 앙상블 예측
    ensemble_pred = (
        model.weights['lgbm'] * preds['lgbm'] +
        model.weights['xgb'] * preds['xgb'] +
        model.weights['catboost'] * preds['catboost']
    )
    
    if return_individual:
        preds['ensemble'] = ensemble_pred
        return preds
    
    return ensemble_pred


def evaluate_ensemble(
    model: EnsembleModel,
    df: pd.DataFrame,
    kpi_col: str,
    horizon: int,
    group_col: str = "segment_id",
    test_start_month: Optional[pd.Timestamp] = None,
    test_end_month: Optional[pd.Timestamp] = None,
) -> Dict[str, float]:
    """
    앙상블 모델 평가
    """
    data = df.copy()
    data[f"target__h{horizon}"] = data.groupby(group_col)[kpi_col].shift(-horizon)
    data = data.dropna(subset=[f"target__h{horizon}"])
    
    if test_start_month is not None:
        data = data[data["month"] >= test_start_month]
    if test_end_month is not None:
        data = data[data["month"] <= test_end_month]
    
    if len(data) == 0:
        return {}
    
    y_true = data[f"target__h{horizon}"].values
    y_pred = predict_ensemble(model, data)
    
    return {
        'n': len(data),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': _rmse(y_true, y_pred),
        'smape': _smape(y_true, y_pred),
        'r2': float(r2_score(y_true, y_pred)),
    }
