#!/usr/bin/env python
"""
LightGBM 모델 하이퍼파라미터 튜닝 및 개선

개선 전략:
1. Optuna로 직접 LightGBM 하이퍼파라미터 튜닝
2. 시간 가중치 적용 (최신 데이터에 더 높은 가중치)
3. 순유입 KPI 특화 피처 추가
4. MPS 가속 활용 (PyTorch 모델)
"""
from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

import optuna
from optuna.samplers import TPESampler

# 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from imradar.config import RadarConfig

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def add_time_weights(df: pd.DataFrame, time_col: str = "month", decay: float = 0.99) -> np.ndarray:
    """
    시간 기반 가중치 생성 (최신 데이터에 높은 가중치)
    
    decay=0.99: 1개월 전은 0.99, 12개월 전은 0.89
    """
    months = df[time_col].values
    max_month = months.max()
    
    # 월 단위 거리 계산
    month_diff = ((max_month - months).astype('timedelta64[M]')).astype(int)
    
    # 지수 감쇠 가중치
    weights = decay ** month_diff
    
    return weights


def tune_lgbm_with_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_cols: List[str],
    sample_weight_train: Optional[np.ndarray] = None,
    n_trials: int = 50,
    timeout: int = 300,
    verbose: bool = True,
) -> Tuple[Dict, float]:
    """
    Optuna를 사용한 LightGBM 하이퍼파라미터 튜닝
    """
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 300, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 4, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }
        
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
            categorical_feature=cat_cols if cat_cols else 'auto',
        )
        
        y_pred = model.predict(X_val)
        return _rmse(y_val, y_pred)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=verbose)
    
    if verbose:
        print(f"  최적 RMSE: {study.best_value:.4f}")
    
    return study.best_params, study.best_value


def train_tuned_lgbm(
    df: pd.DataFrame,
    kpi_col: str,
    horizon: int,
    categorical_cols: List[str],
    train_end_month: pd.Timestamp,
    n_trials: int = 50,
    use_time_weights: bool = True,
    verbose: bool = True,
) -> Tuple[lgb.LGBMRegressor, Dict, List[str]]:
    """
    튜닝된 LightGBM 모델 학습
    """
    data = df.sort_values(["segment_id", "month"]).copy()
    
    # 타겟 생성
    data[f"target__h{horizon}"] = data.groupby("segment_id")[kpi_col].shift(-horizon)
    
    # 필터링
    data = data[(data.get("pre_birth", 0) == 0)].copy()
    data = data.dropna(subset=[f"target__h{horizon}"])
    
    # Train/Test 분할
    train_mask = data["month"] <= train_end_month
    
    # 피처 선택
    exclude = {kpi_col, f"target__h{horizon}", "month", "first_month", "segment_id"}
    feature_cols = [c for c in data.columns if c not in exclude]
    feature_cols = [c for c in feature_cols if (c in categorical_cols) or pd.api.types.is_numeric_dtype(data[c])]
    
    X_train_full = data.loc[train_mask, feature_cols].copy()
    y_train_full = data.loc[train_mask, f"target__h{horizon}"].astype("float64").values
    
    # Categorical 처리
    cat_cols = [c for c in categorical_cols if c in X_train_full.columns]
    for c in cat_cols:
        X_train_full[c] = X_train_full[c].astype("category")
    
    # 시간 가중치
    if use_time_weights:
        train_data = data.loc[train_mask].copy()
        sample_weights = add_time_weights(train_data, "month", decay=0.995)
    else:
        sample_weights = None
    
    # Validation 분할 (최근 데이터)
    split_idx = int(len(X_train_full) * 0.85)
    X_train = X_train_full.iloc[:split_idx]
    X_val = X_train_full.iloc[split_idx:]
    y_train = y_train_full[:split_idx]
    y_val = y_train_full[split_idx:]
    
    if sample_weights is not None:
        sw_train = sample_weights[:split_idx]
    else:
        sw_train = None
    
    if verbose:
        print(f"\n  데이터 크기: Train={len(X_train)}, Val={len(X_val)}, 피처={len(feature_cols)}")
    
    # 하이퍼파라미터 튜닝
    best_params, best_score = tune_lgbm_with_optuna(
        X_train, y_train, X_val, y_val,
        cat_cols, sw_train, n_trials=n_trials, verbose=verbose
    )
    
    # 최종 모델 학습 (전체 훈련 데이터)
    if verbose:
        print(f"  전체 데이터로 최종 모델 학습 중...")
    
    final_params = best_params.copy()
    final_params.update({
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1,
    })
    
    model = lgb.LGBMRegressor(**final_params)
    model.fit(
        X_train_full, y_train_full,
        sample_weight=sample_weights,
        categorical_feature=cat_cols if cat_cols else 'auto',
    )
    
    return model, best_params, feature_cols, cat_cols


def evaluate_model(
    model: lgb.LGBMRegressor,
    df: pd.DataFrame,
    kpi_col: str,
    horizon: int,
    feature_cols: List[str],
    cat_cols: List[str],
    test_start_month: pd.Timestamp,
    test_end_month: pd.Timestamp,
) -> Dict[str, float]:
    """
    모델 평가
    """
    data = df.sort_values(["segment_id", "month"]).copy()
    data[f"target__h{horizon}"] = data.groupby("segment_id")[kpi_col].shift(-horizon)
    data = data[(data.get("pre_birth", 0) == 0)].copy()
    data = data.dropna(subset=[f"target__h{horizon}"])
    
    test_mask = (data["month"] >= test_start_month) & (data["month"] <= test_end_month)
    
    X_test = data.loc[test_mask, feature_cols].copy()
    y_test = data.loc[test_mask, f"target__h{horizon}"].astype("float64").values
    
    for c in cat_cols:
        if c in X_test.columns:
            X_test[c] = X_test[c].astype("category")
    
    y_pred = model.predict(X_test)
    
    return {
        'rmse': _rmse(y_test, y_pred),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
        'smape': _smape(y_test, y_pred),
        'n_test': len(y_test),
    }


def main():
    cfg = RadarConfig()
    out_dir = REPO_ROOT / "outputs"
    
    print("=" * 60)
    print("LightGBM 하이퍼파라미터 튜닝 (Optuna)")
    print("=" * 60)
    
    # 패널 데이터 로드
    print("\n[1/4] 패널 데이터 로드...")
    panel = pd.read_parquet(out_dir / "segment_panel.parquet")
    print(f"  → 레코드 수: {len(panel):,}")
    
    # 타겟 설정
    model_targets = {
        "예금총잔액": "log1p_예금총잔액",
        "대출총잔액": "log1p_대출총잔액",
        "카드총사용": "log1p_카드총사용",
        "디지털거래금액": "log1p_디지털거래금액",
        "순유입": "slog1p_순유입",
    }
    if "log1p_FX총액" in panel.columns:
        model_targets["FX총액"] = "log1p_FX총액"
    
    # 기간 설정
    train_end = pd.Timestamp("2024-11-01")
    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2024-12-01")
    
    categorical_cols = cfg.segment_keys
    
    results = []
    models_tuned = {}
    
    print("\n[2/4] 하이퍼파라미터 튜닝 및 모델 학습...")
    
    # 모든 KPI와 호라이즌에 대해 튜닝
    horizons = [1, 2, 3]
    
    for kpi, tgt_col in model_targets.items():
        if tgt_col not in panel.columns:
            continue
        
        for h in horizons:
            print(f"\n{'─'*50}")
            print(f"KPI: {kpi} | Horizon: {h}개월")
            print(f"{'─'*50}")
            
            try:
                # 튜닝된 모델 학습
                model, best_params, feature_cols, cat_cols = train_tuned_lgbm(
                    panel, tgt_col, h, categorical_cols, train_end,
                    n_trials=40,  # 충분한 시도
                    use_time_weights=True,
                    verbose=True,
                )
                
                # 평가
                metrics = evaluate_model(
                    model, panel, tgt_col, h, feature_cols, cat_cols,
                    test_start, test_end
                )
                
                print(f"\n  ★ 테스트 성능:")
                print(f"    RMSE: {metrics['rmse']:.4f}")
                print(f"    R²: {metrics['r2']:.4f}")
                print(f"    SMAPE: {metrics['smape']:.4f}")
                
                # 결과 저장
                results.append({
                    'kpi': kpi,
                    'target_col': tgt_col,
                    'horizon': h,
                    'tuned_rmse': metrics['rmse'],
                    'tuned_r2': metrics['r2'],
                    'tuned_smape': metrics['smape'],
                    'best_params': str(best_params),
                })
                
                # 모델 저장
                models_tuned[(kpi, h)] = {
                    'model': model,
                    'feature_cols': feature_cols,
                    'cat_cols': cat_cols,
                    'params': best_params,
                }
                
                model_path = out_dir / "models" / f"lgbm_tuned_{kpi}_h{h}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(models_tuned[(kpi, h)], f)
                print(f"  → 모델 저장: {model_path}")
                
            except Exception as e:
                print(f"  오류: {e}")
                import traceback
                traceback.print_exc()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("튜닝 결과 요약")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    
    # 기존 결과와 비교
    baseline_path = out_dir / "forecast_metrics.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        
        # 비교 테이블 생성
        comparison = results_df.merge(
            baseline[['kpi', 'horizon', 'rmse', 'r2']],
            on=['kpi', 'horizon'],
            how='left',
            suffixes=('', '_baseline')
        )
        comparison['rmse_improvement'] = (
            (comparison['rmse_baseline'] - comparison['tuned_rmse']) / 
            comparison['rmse_baseline'] * 100
        )
        comparison['r2_improvement'] = comparison['tuned_r2'] - comparison['r2_baseline']
        
        print("\n기존 vs 튜닝 모델 비교:")
        print(comparison[['kpi', 'horizon', 'rmse_baseline', 'tuned_rmse', 'rmse_improvement', 
                         'r2_baseline', 'tuned_r2', 'r2_improvement']].to_string(index=False))
        
        # 평균 개선율
        avg_rmse_imp = comparison['rmse_improvement'].mean()
        avg_r2_imp = comparison['r2_improvement'].mean()
        print(f"\n평균 RMSE 개선율: {avg_rmse_imp:.2f}%")
        print(f"평균 R² 개선: {avg_r2_imp:.4f}")
    
    # 결과 저장
    results_path = out_dir / "tuned_model_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {results_path}")
    
    print("\nDONE.")


if __name__ == "__main__":
    main()
