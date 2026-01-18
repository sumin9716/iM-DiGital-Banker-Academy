#!/usr/bin/env python
"""
모델 개선 및 비교 스크립트

기능:
1. 기존 LightGBM 모델 vs 앙상블 모델 성능 비교
2. 하이퍼파라미터 튜닝 효과 검증
3. 결과 저장 및 시각화
"""
from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# 경로 설정
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from imradar.config import RadarConfig
from imradar.models.forecast_lgbm import train_lgbm_forecast, evaluate_forecast
from imradar.models.forecast_ensemble import train_ensemble_forecast, evaluate_ensemble


def load_panel(out_dir: Path) -> pd.DataFrame:
    """패널 데이터 로드"""
    parquet_path = out_dir / "segment_panel.parquet"
    csv_path = out_dir / "segment_panel.csv"
    
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["month"])
    else:
        raise FileNotFoundError("패널 데이터를 찾을 수 없습니다.")


def main():
    cfg = RadarConfig()
    out_dir = REPO_ROOT / "outputs"
    
    print("=" * 60)
    print("모델 개선 및 비교")
    print("=" * 60)
    
    # 패널 데이터 로드
    print("\n[1/4] 패널 데이터 로드...")
    panel = load_panel(out_dir)
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
    
    # 학습/테스트 기간 설정
    train_end = pd.Timestamp("2024-11-01")
    test_start = pd.Timestamp("2024-01-01")
    test_end = pd.Timestamp("2024-12-01")
    
    categorical_cols = cfg.segment_keys
    
    # 결과 저장
    results = []
    
    print("\n[2/4] 모델 학습 및 비교...")
    
    # 테스트할 KPI 선택 (순유입이 가장 성능이 낮으므로 우선 테스트)
    test_kpis = ["순유입", "예금총잔액", "카드총사용"]
    horizons = [1, 2, 3]
    
    for kpi in test_kpis:
        tgt_col = model_targets.get(kpi)
        if tgt_col is None or tgt_col not in panel.columns:
            continue
        
        for h in horizons:
            print(f"\n{'─'*50}")
            print(f"KPI: {kpi} | Horizon: {h}개월")
            print(f"{'─'*50}")
            
            # ====== 기존 LightGBM 모델 ======
            print("\n  [기존] LightGBM 단일 모델...")
            try:
                lgbm_model = train_lgbm_forecast(
                    panel,
                    kpi_col=tgt_col,
                    horizon=h,
                    categorical_cols=categorical_cols,
                    train_end_month=train_end,
                    verbose=False,
                )
                lgbm_metrics = evaluate_forecast(
                    panel, lgbm_model,
                    test_start_month=test_start,
                    test_end_month=test_end,
                )
                print(f"    RMSE: {lgbm_metrics['rmse']:.4f} | R²: {lgbm_metrics['r2']:.4f}")
            except Exception as e:
                print(f"    오류: {e}")
                lgbm_metrics = {'rmse': np.nan, 'r2': np.nan}
            
            # ====== 앙상블 모델 (튜닝 없이) ======
            print("\n  [개선1] 앙상블 모델 (기본 파라미터)...")
            try:
                ensemble_basic = train_ensemble_forecast(
                    panel,
                    kpi_col=tgt_col,
                    horizon=h,
                    categorical_cols=categorical_cols,
                    train_end_month=train_end,
                    tune_hyperparams=False,
                    use_feature_selection=True,
                    verbose=False,
                )
                ens_basic_metrics = evaluate_ensemble(
                    ensemble_basic, panel, tgt_col, h,
                    test_start_month=test_start,
                    test_end_month=test_end,
                )
                print(f"    RMSE: {ens_basic_metrics['rmse']:.4f} | R²: {ens_basic_metrics['r2']:.4f}")
            except Exception as e:
                print(f"    오류: {e}")
                ens_basic_metrics = {'rmse': np.nan, 'r2': np.nan}
            
            # ====== 앙상블 모델 (하이퍼파라미터 튜닝) ======
            print("\n  [개선2] 앙상블 모델 (하이퍼파라미터 튜닝)...")
            try:
                ensemble_tuned = train_ensemble_forecast(
                    panel,
                    kpi_col=tgt_col,
                    horizon=h,
                    categorical_cols=categorical_cols,
                    train_end_month=train_end,
                    tune_hyperparams=True,
                    n_trials=20,  # 빠른 테스트를 위해 줄임
                    tune_timeout=180,
                    use_feature_selection=True,
                    verbose=True,
                )
                ens_tuned_metrics = evaluate_ensemble(
                    ensemble_tuned, panel, tgt_col, h,
                    test_start_month=test_start,
                    test_end_month=test_end,
                )
                print(f"    ★ RMSE: {ens_tuned_metrics['rmse']:.4f} | R²: {ens_tuned_metrics['r2']:.4f}")
                
                # 모델 저장
                model_path = out_dir / "models" / f"ensemble_{kpi}_h{h}.pkl"
                ensemble_tuned.save(model_path)
                print(f"    → 모델 저장: {model_path}")
                
            except Exception as e:
                print(f"    오류: {e}")
                import traceback
                traceback.print_exc()
                ens_tuned_metrics = {'rmse': np.nan, 'r2': np.nan}
            
            # 결과 저장
            results.append({
                'kpi': kpi,
                'target_col': tgt_col,
                'horizon': h,
                'lgbm_rmse': lgbm_metrics.get('rmse', np.nan),
                'lgbm_r2': lgbm_metrics.get('r2', np.nan),
                'ensemble_basic_rmse': ens_basic_metrics.get('rmse', np.nan),
                'ensemble_basic_r2': ens_basic_metrics.get('r2', np.nan),
                'ensemble_tuned_rmse': ens_tuned_metrics.get('rmse', np.nan),
                'ensemble_tuned_r2': ens_tuned_metrics.get('r2', np.nan),
            })
    
    # ====== 결과 요약 ======
    print("\n" + "=" * 60)
    print("모델 성능 비교 요약")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    
    # 개선율 계산
    results_df['rmse_improvement'] = (
        (results_df['lgbm_rmse'] - results_df['ensemble_tuned_rmse']) / results_df['lgbm_rmse'] * 100
    )
    results_df['r2_improvement'] = (
        (results_df['ensemble_tuned_r2'] - results_df['lgbm_r2']) / (1 - results_df['lgbm_r2']).clip(lower=0.01) * 100
    )
    
    print("\n" + results_df.to_string(index=False))
    
    # CSV 저장
    results_path = out_dir / "model_comparison.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장: {results_path}")
    
    # 평균 개선율
    avg_rmse_imp = results_df['rmse_improvement'].mean()
    avg_r2_imp = results_df['r2_improvement'].mean()
    print(f"\n평균 RMSE 개선율: {avg_rmse_imp:.2f}%")
    print(f"평균 R² 개선율: {avg_r2_imp:.2f}%")
    
    print("\nDONE.")


if __name__ == "__main__":
    main()
