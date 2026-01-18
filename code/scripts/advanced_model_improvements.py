#!/usr/bin/env python3
"""
고급 모델 개선 스크립트

개선 사항:
1. 순유입 부호 분류 모델 (Two-Stage)
2. Stacking 앙상블
3. 시계열 교차검증 강화
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 설정 (run_mvp.py와 동일한 방식)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from imradar.config import RadarConfig
from imradar.data.io import load_internal_csv
from imradar.data.preprocess import aggregate_to_segment_month, build_full_panel
from imradar.features.kpi import compute_kpis
from imradar.features.lags import add_lag_features, add_rolling_features, add_change_features
from imradar.data.external import load_all_external_data, merge_external_to_panel, add_macro_lag_features


# ============================================================================
# 1. 순유입 부호 분류 모델 (Two-Stage)
# ============================================================================

class TwoStageNetFlowModel:
    """
    순유입 Two-Stage 모델:
    Stage 1: 부호 예측 (분류) - 순유입이 +인지 -인지
    Stage 2: 크기 예측 (회귀) - 절대값 크기 예측
    최종 예측 = 부호 * 크기
    """
    
    def __init__(self, horizon: int = 1):
        self.horizon = horizon
        self.sign_model = None  # 분류 모델
        self.magnitude_model_pos = None  # 양수 크기 모델
        self.magnitude_model_neg = None  # 음수 크기 모델
        self.feature_cols = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            categorical_cols: list = None, verbose: bool = True):
        """Two-Stage 모델 학습"""
        
        self.feature_cols = list(X_train.columns)
        
        # Stage 1: 부호 분류 모델
        y_sign = (y_train > 0).astype(int)  # 1: 양수, 0: 음수
        
        if verbose:
            print(f"    [Stage 1] 부호 분류 모델 학습...")
            print(f"      - 양수 비율: {y_sign.mean():.1%}")
        
        self.sign_model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            verbose=False,
            random_seed=42
        )
        
        cat_features = [X_train.columns.get_loc(c) for c in (categorical_cols or []) 
                       if c in X_train.columns]
        self.sign_model.fit(X_train, y_sign, cat_features=cat_features if cat_features else None)
        
        # Stage 1 성능
        sign_pred = self.sign_model.predict(X_train)
        sign_acc = accuracy_score(y_sign, sign_pred)
        if verbose:
            print(f"      - 부호 예측 정확도: {sign_acc:.1%}")
        
        # Stage 2: 크기 회귀 모델 (양수/음수 분리)
        if verbose:
            print(f"    [Stage 2] 크기 예측 모델 학습...")
        
        # 양수 데이터
        pos_mask = y_train > 0
        X_pos = X_train[pos_mask]
        y_pos = np.abs(y_train[pos_mask])
        
        # 음수 데이터
        neg_mask = y_train < 0
        X_neg = X_train[neg_mask]
        y_neg = np.abs(y_train[neg_mask])
        
        if verbose:
            print(f"      - 양수 샘플: {len(y_pos):,}, 음수 샘플: {len(y_neg):,}")
        
        # 양수 크기 모델
        if len(y_pos) > 100:
            self.magnitude_model_pos = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=64,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
                random_state=42
            )
            self.magnitude_model_pos.fit(X_pos, y_pos)
        
        # 음수 크기 모델
        if len(y_neg) > 100:
            self.magnitude_model_neg = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=64,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
                random_state=42
            )
            self.magnitude_model_neg.fit(X_neg, y_neg)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Two-Stage 예측"""
        # Stage 1: 부호 예측
        sign_prob = self.sign_model.predict_proba(X)[:, 1]  # 양수 확률
        sign_pred = (sign_prob > 0.5).astype(int)
        
        # Stage 2: 크기 예측
        magnitude = np.zeros(len(X))
        
        pos_mask = sign_pred == 1
        neg_mask = sign_pred == 0
        
        if self.magnitude_model_pos is not None and pos_mask.sum() > 0:
            magnitude[pos_mask] = self.magnitude_model_pos.predict(X[pos_mask])
        
        if self.magnitude_model_neg is not None and neg_mask.sum() > 0:
            magnitude[neg_mask] = self.magnitude_model_neg.predict(X[neg_mask])
        
        # 최종 예측: 부호 * 크기
        final_pred = np.where(sign_pred == 1, magnitude, -magnitude)
        
        # Soft prediction (확률 기반)
        # 부호 확률로 가중 평균
        soft_pred = sign_prob * magnitude + (1 - sign_prob) * (-magnitude)
        
        return soft_pred


# ============================================================================
# 2. Stacking 앙상블
# ============================================================================

class StackingEnsemble:
    """
    Stacking 앙상블:
    Level 1: LightGBM, XGBoost, CatBoost (기본 모델)
    Level 2: Ridge (메타 모델)
    """
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.feature_cols = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            categorical_cols: list = None, n_folds: int = 3,
            verbose: bool = True):
        """Stacking 학습"""
        
        self.feature_cols = list(X_train.columns)
        
        if verbose:
            print(f"    [Level 1] 기본 모델 학습 ({n_folds}-Fold CV)...")
        
        # Level 1: 기본 모델들의 OOF 예측 생성
        tscv = TimeSeriesSplit(n_splits=n_folds)
        
        oof_lgbm = np.zeros(len(X_train))
        oof_xgb = np.zeros(len(X_train))
        oof_cat = np.zeros(len(X_train))
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_vl = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_vl = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # LightGBM
            lgbm = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                num_leaves=64, subsample=0.8, colsample_bytree=0.8,
                verbose=-1, random_state=42
            )
            lgbm.fit(X_tr, y_tr)
            oof_lgbm[val_idx] = lgbm.predict(X_vl)
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                subsample=0.8, colsample_bytree=0.8,
                verbosity=0, random_state=42
            )
            xgb_model.fit(X_tr, y_tr)
            oof_xgb[val_idx] = xgb_model.predict(X_vl)
            
            # CatBoost
            cat = CatBoostRegressor(
                iterations=200, learning_rate=0.05, depth=8,
                verbose=False, random_seed=42
            )
            cat.fit(X_tr, y_tr)
            oof_cat[val_idx] = cat.predict(X_vl)
        
        # Level 2: 메타 모델 학습
        if verbose:
            print(f"    [Level 2] 메타 모델 학습...")
        
        meta_features = np.column_stack([oof_lgbm, oof_xgb, oof_cat])
        
        # 0이 아닌 부분만 사용 (첫 번째 fold 이전은 OOF가 없음)
        valid_mask = (oof_lgbm != 0) | (oof_xgb != 0) | (oof_cat != 0)
        
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features[valid_mask], y_train.values[valid_mask])
        
        if verbose:
            weights = self.meta_model.coef_
            print(f"      - 메타 가중치: LGBM={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}")
        
        # 전체 데이터로 기본 모델 재학습
        if verbose:
            print(f"    [Retrain] 전체 데이터로 기본 모델 재학습...")
        
        self.base_models['lgbm'] = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            num_leaves=64, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42
        )
        self.base_models['lgbm'].fit(X_train, y_train)
        
        self.base_models['xgb'] = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42
        )
        self.base_models['xgb'].fit(X_train, y_train)
        
        self.base_models['cat'] = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=8,
            verbose=False, random_seed=42
        )
        self.base_models['cat'].fit(X_train, y_train)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Stacking 예측"""
        pred_lgbm = self.base_models['lgbm'].predict(X)
        pred_xgb = self.base_models['xgb'].predict(X)
        pred_cat = self.base_models['cat'].predict(X)
        
        meta_features = np.column_stack([pred_lgbm, pred_xgb, pred_cat])
        
        return self.meta_model.predict(meta_features)


# ============================================================================
# 3. 시계열 교차검증 (Expanding Window)
# ============================================================================

class ExpandingWindowCV:
    """
    Expanding Window Cross-Validation:
    - 시간순으로 점점 커지는 학습 데이터
    - 미래 데이터 누출 방지
    """
    
    def __init__(self, initial_train_months: int = 36, test_months: int = 3, step: int = 3):
        self.initial_train_months = initial_train_months
        self.test_months = test_months
        self.step = step
        
    def split(self, df: pd.DataFrame, time_col: str = 'month'):
        """Expanding Window 분할 생성"""
        months = sorted(df[time_col].unique())
        n_months = len(months)
        
        folds = []
        start = self.initial_train_months
        
        while start + self.test_months <= n_months:
            train_months = months[:start]
            test_months = months[start:start + self.test_months]
            
            train_idx = df[df[time_col].isin(train_months)].index.tolist()
            test_idx = df[df[time_col].isin(test_months)].index.tolist()
            
            folds.append((train_idx, test_idx))
            start += self.step
        
        return folds
    
    def evaluate_model(self, model_class, df: pd.DataFrame, 
                      feature_cols: list, target_col: str,
                      time_col: str = 'month', verbose: bool = True):
        """Expanding Window CV로 모델 평가"""
        
        folds = self.split(df, time_col)
        
        all_metrics = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train = df.loc[train_idx, feature_cols]
            y_train = df.loc[train_idx, target_col]
            X_test = df.loc[test_idx, feature_cols]
            y_test = df.loc[test_idx, target_col]
            
            # 모델 학습
            model = model_class()
            model.fit(X_train, y_train)
            
            # 예측 및 평가
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # SMAPE
            denom = (np.abs(y_test) + np.abs(y_pred)) / 2.0
            denom = np.maximum(denom, 1e-9)
            smape = np.mean(np.abs(y_test - y_pred) / denom)
            
            all_metrics.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'smape': smape
            })
            
            if verbose:
                print(f"      Fold {fold_idx+1}: Train={len(train_idx):,}, Test={len(test_idx):,}, "
                      f"R²={r2:.4f}, SMAPE={smape:.4f}")
        
        return pd.DataFrame(all_metrics)


# ============================================================================
# 메트릭 함수
# ============================================================================

def compute_metrics(y_true, y_pred):
    """모든 메트릭 계산"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # SMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, 1e-9)
    smape = np.mean(np.abs(y_true - y_pred) / denom)
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'smape': smape}


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="고급 모델 개선")
    parser.add_argument("--raw_csv", type=str, default="../외부 데이터/iMbank_data.csv.csv")
    parser.add_argument("--external_dir", type=str, default="../외부 데이터")
    parser.add_argument("--output_dir", type=str, default="outputs/advanced_models")
    args = parser.parse_args()
    
    # 출력 디렉토리
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🚀 고급 모델 개선 시작")
    print("=" * 60)
    
    # ========================================================================
    # 데이터 로드
    # ========================================================================
    print("\n[1/4] 데이터 로드...")
    
    cfg = RadarConfig()
    raw_df = load_internal_csv(Path(args.raw_csv), cfg)
    print(f"  ✓ 원본 데이터: {len(raw_df):,}행")
    
    print("[2/4] 전처리...")
    monthly_df = aggregate_to_segment_month(raw_df, cfg)
    panel_result = build_full_panel(monthly_df, cfg)
    panel_df = panel_result.panel if hasattr(panel_result, 'panel') else panel_result
    print(f"  ✓ 패널 데이터: {len(panel_df):,}행")
    
    print("[3/4] KPI 및 피처 생성...")
    kpi_df = compute_kpis(panel_df, cfg)
    
    # 외부 데이터 병합
    external_dir = Path(args.external_dir)
    external_dict = load_all_external_data(external_dir)
    kpi_df = merge_external_to_panel(kpi_df, external_dict)
    kpi_df = add_macro_lag_features(kpi_df, lags=[1, 2, 3])
    
    # Lag 피처 - value_cols 정의
    target_cols_list = ['log1p_예금총잔액', 'log1p_대출총잔액', 'log1p_카드총사용', 
                       'log1p_디지털거래금액', 'slog1p_순유입', 'log1p_FX총액']
    value_cols = [c for c in target_cols_list if c in kpi_df.columns]
    value_cols += [c for c in ['한도소진율', '디지털비중', '자동이체비중', 'customer_count'] if c in kpi_df.columns]
    
    kpi_df = add_lag_features(kpi_df, group_col="segment_id", time_col="month", value_cols=value_cols)
    kpi_df = add_rolling_features(kpi_df, group_col="segment_id", time_col="month", value_cols=value_cols)
    kpi_df = add_change_features(kpi_df, group_col="segment_id", time_col="month", value_cols=value_cols)
    print(f"  ✓ 최종 데이터: {len(kpi_df):,}행, {len(kpi_df.columns)}개 컬럼")
    
    # ========================================================================
    # 피처 및 타겟 준비
    # ========================================================================
    print("[4/4] 피처 준비...")
    
    # 타겟 컬럼
    target_cols = {
        '예금총잔액': 'log1p_예금총잔액',
        '대출총잔액': 'log1p_대출총잔액', 
        '카드총사용': 'log1p_카드총사용',
        '디지털거래금액': 'log1p_디지털거래금액',
        '순유입': 'slog1p_순유입',
        'FX총액': 'log1p_FX총액'
    }
    
    # Horizon 타겟 생성
    for kpi, base_col in target_cols.items():
        for h in [1, 2, 3]:
            target_name = f"target_{kpi}_h{h}"
            kpi_df[target_name] = kpi_df.groupby('segment_id')[base_col].shift(-h)
    
    # 피처 컬럼 선택
    exclude_cols = ['segment_id', 'month', 'age_group', 'job', 'gender', 'region',
                   'segment_name', 'active_rate']
    exclude_cols += [c for c in kpi_df.columns if c.startswith('target_')]
    exclude_cols += [c for c in kpi_df.columns if 'log1p_' in c or 'slog1p_' in c]
    
    feature_cols = [c for c in kpi_df.columns if c not in exclude_cols 
                   and kpi_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    categorical_cols = ['age_group', 'job', 'gender', 'region']
    
    print(f"  ✓ 피처: {len(feature_cols)}개")
    
    # Train/Test 분할
    train_end = pd.Timestamp("2024-09-01")
    train_mask = kpi_df['month'] <= train_end
    
    # NaN 제거
    valid_mask = kpi_df[feature_cols].notna().all(axis=1)
    
    results = []
    
    # ========================================================================
    # 1. 순유입 Two-Stage 모델
    # ========================================================================
    print("\n" + "=" * 60)
    print("📊 1. 순유입 Two-Stage 모델")
    print("=" * 60)
    
    for horizon in [1, 2, 3]:
        print(f"\n[순유입] Horizon {horizon}개월")
        print("-" * 40)
        
        target_col = f"target_순유입_h{horizon}"
        
        # 데이터 준비 - 더 나은 분할
        mask = train_mask & valid_mask & kpi_df[target_col].notna()
        test_mask = (~train_mask) & valid_mask & kpi_df[target_col].notna()
        
        X_train = kpi_df.loc[mask, feature_cols].copy()
        y_train = kpi_df.loc[mask, target_col].copy()
        X_test = kpi_df.loc[test_mask, feature_cols].copy()
        y_test = kpi_df.loc[test_mask, target_col].copy()
        
        print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # 테스트 데이터가 없으면 스킵
        if len(X_test) < 10:
            print(f"  ⚠️ 테스트 데이터 부족, 스킵")
            continue
        
        # Two-Stage 모델 학습
        two_stage = TwoStageNetFlowModel(horizon=horizon)
        two_stage.fit(X_train, y_train, categorical_cols=categorical_cols)
        
        # 예측 및 평가
        y_pred_twostage = two_stage.predict(X_test)
        metrics_twostage = compute_metrics(y_test.values, y_pred_twostage)
        
        print(f"\n  ✅ Two-Stage 결과:")
        print(f"     R² = {metrics_twostage['r2']:.4f}")
        print(f"     SMAPE = {metrics_twostage['smape']:.4f}")
        
        # 기존 단일 모델 비교
        baseline = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            verbose=-1, random_state=42
        )
        baseline.fit(X_train, y_train)
        y_pred_baseline = baseline.predict(X_test)
        metrics_baseline = compute_metrics(y_test.values, y_pred_baseline)
        
        print(f"\n  📊 기존 모델:")
        print(f"     R² = {metrics_baseline['r2']:.4f}")
        print(f"     SMAPE = {metrics_baseline['smape']:.4f}")
        
        improvement = (metrics_twostage['r2'] - metrics_baseline['r2']) / max(abs(metrics_baseline['r2']), 0.001) * 100
        print(f"\n  🎯 R² 개선: {improvement:+.1f}%")
        
        results.append({
            'kpi': '순유입',
            'horizon': horizon,
            'model': 'Two-Stage',
            'r2': metrics_twostage['r2'],
            'smape': metrics_twostage['smape'],
            'baseline_r2': metrics_baseline['r2'],
            'improvement': improvement
        })
    
    # ========================================================================
    # 2. Stacking 앙상블 (주요 KPI)
    # ========================================================================
    print("\n" + "=" * 60)
    print("📊 2. Stacking 앙상블")
    print("=" * 60)
    
    for kpi in ['예금총잔액', '대출총잔액', '순유입']:
        for horizon in [1]:  # 1개월만 테스트
            print(f"\n[{kpi}] Horizon {horizon}개월")
            print("-" * 40)
            
            target_col = f"target_{kpi}_h{horizon}"
            
            # 데이터 준비
            mask = train_mask & valid_mask & kpi_df[target_col].notna()
            test_mask = (~train_mask) & valid_mask & kpi_df[target_col].notna()
            
            X_train = kpi_df.loc[mask, feature_cols].copy()
            y_train = kpi_df.loc[mask, target_col].copy()
            X_test = kpi_df.loc[test_mask, feature_cols].copy()
            y_test = kpi_df.loc[test_mask, target_col].copy()
            
            print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
            
            # 테스트 데이터가 없으면 스킵
            if len(X_test) < 10:
                print(f"  ⚠️ 테스트 데이터 부족, 스킵")
                continue
            
            # Stacking 학습
            stacking = StackingEnsemble()
            stacking.fit(X_train, y_train, categorical_cols=categorical_cols, n_folds=3)
            
            # 예측 및 평가
            y_pred_stack = stacking.predict(X_test)
            metrics_stack = compute_metrics(y_test.values, y_pred_stack)
            
            print(f"\n  ✅ Stacking 결과:")
            print(f"     R² = {metrics_stack['r2']:.4f}")
            print(f"     SMAPE = {metrics_stack['smape']:.4f}")
            
            # 기존 단일 모델 비교
            baseline = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=8,
                verbose=-1, random_state=42
            )
            baseline.fit(X_train, y_train)
            y_pred_baseline = baseline.predict(X_test)
            metrics_baseline = compute_metrics(y_test.values, y_pred_baseline)
            
            improvement = (metrics_stack['r2'] - metrics_baseline['r2']) / max(abs(metrics_baseline['r2']), 0.001) * 100
            print(f"\n  🎯 R² 개선: {improvement:+.1f}%")
            
            results.append({
                'kpi': kpi,
                'horizon': horizon,
                'model': 'Stacking',
                'r2': metrics_stack['r2'],
                'smape': metrics_stack['smape'],
                'baseline_r2': metrics_baseline['r2'],
                'improvement': improvement
            })
    
    # ========================================================================
    # 3. 시계열 CV 평가
    # ========================================================================
    print("\n" + "=" * 60)
    print("📊 3. 시계열 교차검증 (Expanding Window)")
    print("=" * 60)
    
    # 순유입에 대해 Expanding Window CV 수행
    target_col = "target_순유입_h1"
    mask = valid_mask & kpi_df[target_col].notna()
    eval_df = kpi_df.loc[mask].copy()
    
    print(f"\n[순유입] Expanding Window CV")
    print("-" * 40)
    
    cv = ExpandingWindowCV(initial_train_months=36, test_months=3, step=3)
    
    class LGBMWrapper:
        def __init__(self):
            self.model = None
        def fit(self, X, y):
            self.model = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=8,
                verbose=-1, random_state=42
            )
            self.model.fit(X, y)
        def predict(self, X):
            return self.model.predict(X)
    
    cv_results = cv.evaluate_model(
        LGBMWrapper, eval_df, feature_cols, target_col, 
        time_col='month', verbose=True
    )
    
    print(f"\n  📊 CV 평균 결과:")
    if len(cv_results) > 0 and 'r2' in cv_results.columns:
        print(f"     평균 R² = {cv_results['r2'].mean():.4f} (±{cv_results['r2'].std():.4f})")
        print(f"     평균 SMAPE = {cv_results['smape'].mean():.4f} (±{cv_results['smape'].std():.4f})")
    else:
        print(f"     ⚠️ CV 결과가 비어있습니다 (데이터 부족 또는 fold 생성 실패)")
        cv_results = pd.DataFrame(columns=['fold', 'train_size', 'test_size', 'mae', 'rmse', 'r2', 'smape'])
    
    # ========================================================================
    # 결과 저장
    # ========================================================================
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "advanced_model_results.csv", index=False)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    
    print("\n" + "=" * 60)
    print("✅ 고급 모델 개선 완료!")
    print("=" * 60)
    
    print("\n📊 최종 결과 요약:")
    print(results_df.to_string(index=False))
    
    print(f"\n💾 결과 저장: {output_dir}")


if __name__ == "__main__":
    main()
