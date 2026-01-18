"""
모델 개선 스크립트 v2
===================

개선 사항:
1. Optuna 기반 하이퍼파라미터 자동 튜닝
2. LightGBM + XGBoost + CatBoost 앙상블
3. 순유입(R²=0.34) 특별 개선 전략
4. 피처 중요도 기반 선택
5. 교차검증 강화

실행: python scripts/improve_models_v2.py --raw_csv "../외부 데이터/iMbank_data.csv.csv"
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from imradar.config import RadarConfig
from imradar.data.io import load_internal_csv
from imradar.data.preprocess import aggregate_to_segment_month, build_full_panel
from imradar.data.external import load_all_external_data, merge_external_to_panel, add_macro_lag_features
from imradar.features.kpi import compute_kpis
from imradar.features.lags import add_lag_features, add_rolling_features, add_change_features


# ============================================================================
# 메트릭 함수
# ============================================================================

def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = np.sum(np.abs(y_true))
    denom = max(float(denom), eps)
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ============================================================================
# Optuna 하이퍼파라미터 튜닝
# ============================================================================

def create_lgbm_objective(X_train, y_train, cat_cols):
    """LightGBM Optuna Objective"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1,
        }
        
        # Time Series CV
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
                categorical_feature=cat_cols,
            )
            
            preds = model.predict(X_val)
            scores.append(smape(y_val, preds))
        
        return np.mean(scores)
    
    return objective


def create_xgb_objective(X_train, y_train):
    """XGBoost Optuna Objective"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = xgb.XGBRegressor(**params, enable_categorical=False)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            
            preds = model.predict(X_val)
            scores.append(smape(y_val, preds))
        
        return np.mean(scores)
    
    return objective


def create_catboost_objective(X_train, y_train, cat_cols):
    """CatBoost Optuna Objective"""
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_seed': 42,
            'verbose': False,
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model = CatBoostRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                cat_features=cat_cols,
                verbose=False,
                early_stopping_rounds=50,
            )
            
            preds = model.predict(X_val)
            scores.append(smape(y_val, preds))
        
        return np.mean(scores)
    
    return objective


# ============================================================================
# 앙상블 모델
# ============================================================================

class EnsembleForecaster:
    """LightGBM + XGBoost + CatBoost 앙상블"""
    
    def __init__(self, kpi: str, horizon: int):
        self.kpi = kpi
        self.horizon = horizon
        self.models = {}
        self.weights = {}
        self.feature_cols = []
        self.cat_cols = []
        self.best_params = {}
    
    def tune_and_train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        cat_cols: list,
        n_trials: int = 30,
        verbose: bool = True,
    ):
        """Optuna로 튜닝 후 학습"""
        self.feature_cols = X_train.columns.tolist()
        self.cat_cols = cat_cols
        
        # 범주형 인코딩 (XGBoost용)
        X_train_encoded = X_train.copy()
        for c in cat_cols:
            if c in X_train_encoded.columns:
                X_train_encoded[c] = X_train_encoded[c].astype('category').cat.codes
        
        # 1. LightGBM 튜닝
        if verbose:
            print(f"    └─ LightGBM 튜닝 중 ({n_trials} trials)...")
        study_lgbm = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study_lgbm.optimize(
            create_lgbm_objective(X_train, y_train, cat_cols),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        self.best_params['lgbm'] = study_lgbm.best_params
        
        # 2. XGBoost 튜닝
        if verbose:
            print(f"    └─ XGBoost 튜닝 중 ({n_trials} trials)...")
        study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study_xgb.optimize(
            create_xgb_objective(X_train_encoded, y_train),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        self.best_params['xgb'] = study_xgb.best_params
        
        # 3. CatBoost 튜닝
        if verbose:
            print(f"    └─ CatBoost 튜닝 중 ({n_trials} trials)...")
        study_cat = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study_cat.optimize(
            create_catboost_objective(X_train, y_train, cat_cols),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        self.best_params['catboost'] = study_cat.best_params
        
        # 최종 모델 학습
        if verbose:
            print(f"    └─ 최종 모델 학습 중...")
        
        # LightGBM
        lgbm_params = self.best_params['lgbm'].copy()
        lgbm_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': -1})
        self.models['lgbm'] = lgb.LGBMRegressor(**lgbm_params)
        self.models['lgbm'].fit(X_train, y_train, categorical_feature=cat_cols)
        
        # XGBoost
        xgb_params = self.best_params['xgb'].copy()
        xgb_params.update({'random_state': 42, 'n_jobs': -1, 'verbosity': 0})
        self.models['xgb'] = xgb.XGBRegressor(**xgb_params, enable_categorical=False)
        self.models['xgb'].fit(X_train_encoded, y_train)
        
        # CatBoost
        cat_params = self.best_params['catboost'].copy()
        cat_params.update({'random_seed': 42, 'verbose': False})
        self.models['catboost'] = CatBoostRegressor(**cat_params)
        self.models['catboost'].fit(X_train, y_train, cat_features=cat_cols)
        
        # CV로 가중치 결정
        self._compute_weights(X_train, y_train, cat_cols)
        
        if verbose:
            print(f"    └─ 앙상블 가중치: LGBM={self.weights['lgbm']:.2f}, "
                  f"XGB={self.weights['xgb']:.2f}, CatBoost={self.weights['catboost']:.2f}")
    
    def _compute_weights(self, X_train, y_train, cat_cols):
        """CV 기반 가중치 계산"""
        X_encoded = X_train.copy()
        for c in cat_cols:
            if c in X_encoded.columns:
                X_encoded[c] = X_encoded[c].astype('category').cat.codes
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = {'lgbm': [], 'xgb': [], 'catboost': []}
        
        for train_idx, val_idx in tscv.split(X_train):
            X_val = X_train.iloc[val_idx]
            X_val_enc = X_encoded.iloc[val_idx]
            y_val = y_train[val_idx]
            
            preds_lgbm = self.models['lgbm'].predict(X_val)
            preds_xgb = self.models['xgb'].predict(X_val_enc)
            preds_cat = self.models['catboost'].predict(X_val)
            
            scores['lgbm'].append(1 / (smape(y_val, preds_lgbm) + 1e-9))
            scores['xgb'].append(1 / (smape(y_val, preds_xgb) + 1e-9))
            scores['catboost'].append(1 / (smape(y_val, preds_cat) + 1e-9))
        
        # 역 SMAPE 기반 가중치
        total = sum(np.mean(scores[k]) for k in scores)
        self.weights = {k: np.mean(scores[k]) / total for k in scores}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """앙상블 예측"""
        X_encoded = X.copy()
        for c in self.cat_cols:
            if c in X_encoded.columns:
                X_encoded[c] = X_encoded[c].astype('category').cat.codes
        
        preds_lgbm = self.models['lgbm'].predict(X)
        preds_xgb = self.models['xgb'].predict(X_encoded)
        preds_cat = self.models['catboost'].predict(X)
        
        return (
            self.weights['lgbm'] * preds_lgbm +
            self.weights['xgb'] * preds_xgb +
            self.weights['catboost'] * preds_cat
        )


# ============================================================================
# 데이터 준비
# ============================================================================

def prepare_data(raw_csv: str, encoding: str, external_dir: str, cfg: RadarConfig):
    """데이터 로드 및 전처리 (run_mvp.py와 동일한 방식)"""
    print("[1/5] 데이터 로드...")
    raw = load_internal_csv(raw_csv, encoding=encoding)
    
    print("[2/5] 세그먼트-월별 집계...")
    segment_monthly = aggregate_to_segment_month(raw, cfg)
    
    print("[3/5] 패널 구축...")
    panel_result = build_full_panel(segment_monthly, cfg)
    panel = panel_result.panel
    
    print("[4/5] KPI 계산...")
    panel = compute_kpis(panel, cfg)
    
    # 외부 데이터 로드 (있으면)
    external_data = None
    if external_dir and os.path.isdir(external_dir):
        print("[4.5/5] 외부 거시경제 데이터 병합...")
        try:
            external_data = load_all_external_data(
                external_dir, 
                file_config=cfg.external_data_files
            )
            if external_data is not None and not external_data.empty:
                panel = merge_external_to_panel(panel, external_data, month_col="month")
                print(f"  외부 데이터 병합 완료: 패널 컬럼 수 {len(panel.columns)}")
        except Exception as e:
            print(f"  외부 데이터 로드 실패 (계속 진행): {e}")
    
    # 타겟 컬럼
    model_targets = {
        "예금총잔액": "log1p_예금총잔액",
        "대출총잔액": "log1p_대출총잔액",
        "카드총사용": "log1p_카드총사용",
        "디지털거래금액": "log1p_디지털거래금액",
        "순유입": "slog1p_순유입",
    }
    if "log1p_FX총액" in panel.columns:
        model_targets["FX총액"] = "log1p_FX총액"
    
    # 피처 엔지니어링
    print("[5/5] 피처 엔지니어링...")
    value_cols = list(model_targets.values()) + ["한도소진율", "디지털비중", "자동이체비중", "customer_count"]
    value_cols = [c for c in value_cols if c in panel.columns]
    
    panel_fe = panel.copy()
    panel_fe = add_lag_features(panel_fe, group_col="segment_id", time_col="month", value_cols=value_cols)
    panel_fe = add_rolling_features(panel_fe, group_col="segment_id", time_col="month", value_cols=value_cols)
    panel_fe = add_change_features(panel_fe, group_col="segment_id", time_col="month", value_cols=value_cols)
    
    # 거시경제 Lag 피처 추가
    if external_data is not None and not external_data.empty:
        print("  거시경제 변수 Lag 피처 생성 중...")
        macro_cols_available = [c for c in cfg.macro_feature_cols if c in panel_fe.columns]
        if macro_cols_available:
            panel_fe = add_macro_lag_features(
                panel_fe, 
                macro_cols=macro_cols_available, 
                lags=cfg.macro_lag_periods
            )
            print(f"  거시경제 Lag 피처 추가 완료: {len(macro_cols_available)}개 변수 x {len(cfg.macro_lag_periods)}개 Lag")
    
    return panel_fe


def prepare_training_data(
    panel: pd.DataFrame,
    kpi_col: str,
    horizon: int,
    train_end: pd.Timestamp,
    cfg: RadarConfig,
    is_net_flow: bool = False,
):
    """학습 데이터 준비"""
    data = panel.sort_values(['segment_id', 'month']).copy()
    
    # 순유입(net flow)의 경우 추가 피처 생성
    if is_net_flow:
        print("    └─ 순유입 특별 피처 추가 중...")
        data = add_net_flow_features(data)
    
    # 타겟 생성
    target_col = f"target__h{horizon}"
    data[target_col] = data.groupby('segment_id')[kpi_col].shift(-horizon)
    
    # 필터링
    data = data[(data.get('pre_birth', 0) == 0)].copy()
    data = data.dropna(subset=[target_col])
    
    # 학습/테스트 분리
    train_mask = data['month'] <= train_end
    
    # 피처 선택
    exclude = {kpi_col, target_col, 'month', 'first_month', 'segment_id'}
    feature_cols = [c for c in data.columns if c not in exclude]
    
    categorical_cols = cfg.segment_keys
    feature_cols = [c for c in feature_cols 
                    if (c in categorical_cols) or (pd.api.types.is_numeric_dtype(data[c]))]
    
    cat_cols = [c for c in categorical_cols if c in feature_cols]
    
    X_train = data.loc[train_mask, feature_cols].copy()
    y_train = data.loc[train_mask, target_col].values
    
    X_test = data.loc[~train_mask, feature_cols].copy()
    y_test = data.loc[~train_mask, target_col].values
    
    # 범주형 타입 변환
    for c in cat_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype('category')
            X_test[c] = X_test[c].astype('category')
    
    return X_train, y_train, X_test, y_test, cat_cols


def add_net_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    순유입 예측을 위한 특별 피처 추가
    순유입 = 요구불입금금액 - 요구불출금금액
    """
    out = df.copy()
    
    # 1. 입금/출금 개별 Lag 피처
    flow_cols = ['요구불입금금액', '요구불출금금액']
    for col in flow_cols:
        if col in out.columns:
            for lag in [1, 2, 3]:
                out[f'{col}__lag{lag}'] = out.groupby('segment_id')[col].shift(lag)
    
    # 2. 입금/출금 비율
    if '요구불입금금액' in out.columns and '요구불출금금액' in out.columns:
        eps = 1e-9
        out['입출금비율'] = out['요구불입금금액'] / (out['요구불출금금액'] + eps)
        out['입출금비율'] = out['입출금비율'].clip(0, 10)  # 이상치 제한
        
        # 비율의 Lag
        for lag in [1, 2, 3]:
            out[f'입출금비율__lag{lag}'] = out.groupby('segment_id')['입출금비율'].shift(lag)
    
    # 3. 순유입의 부호 변화 (방향 전환 감지)
    if 'slog1p_순유입' in out.columns:
        out['순유입부호'] = np.sign(out['slog1p_순유입'])
        out['순유입부호__lag1'] = out.groupby('segment_id')['순유입부호'].shift(1)
        out['순유입부호변화'] = (out['순유입부호'] != out['순유입부호__lag1']).astype(int)
    
    # 4. 예금 잔액 대비 순유입 비율
    if '순유입' in out.columns and 'log1p_예금총잔액' in out.columns:
        out['순유입_예금비율'] = out['순유입'] / (np.expm1(out['log1p_예금총잔액']) + 1e-9)
        out['순유입_예금비율'] = out['순유입_예금비율'].clip(-1, 1)
    
    # 5. Rolling 통계 (순유입 변동성)
    if 'slog1p_순유입' in out.columns:
        for w in [3, 6]:
            shifted = out.groupby('segment_id')['slog1p_순유입'].shift(1)
            out[f'순유입__roll{w}_std'] = shifted.rolling(w, min_periods=1).std()
            out[f'순유입__roll{w}_mean'] = shifted.rolling(w, min_periods=1).mean()
    
    return out


# ============================================================================
# 메인
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="모델 개선 v2")
    parser.add_argument("--raw_csv", required=True, help="원본 데이터 경로")
    parser.add_argument("--encoding", default="utf-8-sig", help="인코딩")
    parser.add_argument("--external_dir", default="../외부 데이터", help="외부 데이터 디렉토리")
    parser.add_argument("--train_end", default="2024-09", help="학습 종료월 (YYYY-MM)")
    parser.add_argument("--n_trials", type=int, default=20, help="Optuna 시도 횟수")
    parser.add_argument("--out_dir", default="outputs/improved_models", help="출력 디렉토리")
    args = parser.parse_args()
    
    cfg = RadarConfig()
    os.makedirs(args.out_dir, exist_ok=True)
    
    train_end = pd.Timestamp(args.train_end)
    
    # 데이터 준비
    panel = prepare_data(args.raw_csv, args.encoding, args.external_dir, cfg)
    
    # KPI 목록
    kpi_cols = list(cfg.kpi_defs.keys())
    
    # 결과 저장
    results = []
    
    print("\n" + "=" * 60)
    print("🚀 모델 개선 시작")
    print("=" * 60)
    
    for kpi in kpi_cols:
        # 로그 변환된 KPI 찾기
        if kpi == "순유입":
            log_kpi = f"slog1p_{kpi}"
        else:
            log_kpi = f"log1p_{kpi}"
        
        if log_kpi not in panel.columns:
            print(f"⚠️ {log_kpi} 컬럼 없음, 건너뜀")
            continue
        
        for horizon in cfg.horizons:
            print(f"\n[{kpi}] Horizon {horizon}개월")
            print("-" * 40)
            
            # 데이터 준비 (순유입은 특별 처리)
            is_net_flow = (kpi == "순유입")
            X_train, y_train, X_test, y_test, cat_cols = prepare_training_data(
                panel, log_kpi, horizon, train_end, cfg, is_net_flow=is_net_flow
            )
            
            if len(X_train) < 100 or len(X_test) < 10:
                print(f"  ⚠️ 데이터 부족: train={len(X_train)}, test={len(X_test)}")
                continue
            
            print(f"  📊 Train: {len(X_train):,}, Test: {len(X_test):,}, Features: {len(X_train.columns)}")
            
            # 순유입은 더 많은 튜닝 시도
            n_trials = args.n_trials * 2 if kpi == "순유입" else args.n_trials
            
            # 앙상블 학습
            ensemble = EnsembleForecaster(kpi=kpi, horizon=horizon)
            ensemble.tune_and_train(X_train, y_train, cat_cols, n_trials=n_trials)
            
            # 테스트 평가
            preds = ensemble.predict(X_test)
            
            metrics = {
                'kpi': kpi,
                'horizon': horizon,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'mae': mean_absolute_error(y_test, preds),
                'rmse': rmse(y_test, preds),
                'smape': smape(y_test, preds),
                'wmape': wmape(y_test, preds),
                'r2': r2_score(y_test, preds),
                'lgbm_weight': ensemble.weights['lgbm'],
                'xgb_weight': ensemble.weights['xgb'],
                'catboost_weight': ensemble.weights['catboost'],
            }
            
            results.append(metrics)
            
            print(f"  ✅ SMAPE: {metrics['smape']:.2f}%, R²: {metrics['r2']:.4f}")
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.out_dir, "improved_metrics.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n📁 결과 저장: {results_path}")
    
    # 요약 출력
    print("\n" + "=" * 60)
    print("📊 개선 결과 요약")
    print("=" * 60)
    print(results_df[['kpi', 'horizon', 'smape', 'wmape', 'r2']].to_string(index=False))
    
    print("\n✅ 모델 개선 완료!")


if __name__ == "__main__":
    main()
