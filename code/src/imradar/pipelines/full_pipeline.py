"""
iM Digital Banker 통합 파이프라인

P0/P1 데이터를 기반으로 전체 분석 프로세스를 실행하는 메인 파이프라인
- P0: 세그먼트-월 집계 및 패널 구축
- P1: 외부 거시경제 데이터 로드 및 병합
- 시계열 피처 자동 생성
- 멀티호라이즌 예측 (1~3개월)
- 리스크 레이더 (이상탐지 + 경보)
- SHAP 기반 원인 설명
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 데이터 경로
    p0_data_path: Path = None
    external_data_dir: Path = None
    output_dir: Path = None
    
    # 세그먼트 설정
    segment_keys: List[str] = field(default_factory=lambda: [
        "업종_중분류", "사업장_시도", "법인_고객등급", "전담고객여부"
    ])
    
    # KPI 설정
    target_kpis: List[str] = field(default_factory=lambda: [
        "예금총잔액", "대출총잔액", "카드총사용", "FX총액"
    ])
    
    # 예측 호라이즌
    horizons: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # 피처 설정
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6])
    
    # 학습 설정
    train_end_month: Optional[pd.Timestamp] = None
    use_early_stopping: bool = True
    use_cv: bool = True
    cv_folds: int = 3
    
    # 알림 설정
    alert_down_threshold: float = -0.20
    alert_up_threshold: float = 0.20
    top_alerts: int = 20
    
    # Deep Learning 설정 (Optional)
    use_deep_learning: bool = False  # DL 모델 사용 여부
    dl_model_type: str = "tft"  # "tft", "lstm", "ensemble"
    use_dl_anomaly: bool = False  # DL 이상탐지 사용 여부
    dl_anomaly_type: str = "autoencoder"  # "autoencoder", "anomaly_transformer"
    
    # 출력 설정
    verbose: bool = True


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    panel: pd.DataFrame
    models: Dict[str, Dict[int, any]]  # {kpi: {horizon: model}}
    forecasts: pd.DataFrame
    risk_alerts: pd.DataFrame
    segment_risks: pd.DataFrame
    metrics: pd.DataFrame
    

def run_full_pipeline(config: PipelineConfig) -> PipelineResult:
    """
    전체 파이프라인 실행
    
    Args:
        config: 파이프라인 설정
    
    Returns:
        PipelineResult
    """
    from imradar.config import RadarConfig
    from imradar.data.preprocess import (
        load_p0_data, aggregate_to_segment_month, build_full_panel, compute_kpis
    )
    from imradar.data.external import load_all_external_data
    from imradar.features.lags import build_all_features
    from imradar.models.forecast_lgbm import train_lgbm_forecast, predict_with_model
    from imradar.models.risk_radar import run_risk_radar
    
    cfg = RadarConfig()
    
    def _log(msg):
        if config.verbose:
            print(msg)
    
    _log("=" * 60)
    _log("iM Digital Banker 통합 파이프라인 실행")
    _log("=" * 60)
    
    # ====== STEP 1: P0 데이터 로드 및 세그먼트-월 집계 ======
    _log("\n[STEP 1/7] P0 데이터 처리")
    
    raw_df = load_p0_data(config.p0_data_path, cfg)
    segment_month = aggregate_to_segment_month(raw_df, cfg)
    panel_result = build_full_panel(segment_month, cfg)
    panel = panel_result.panel
    
    _log(f"  → 세그먼트 수: {panel['segment_id'].nunique():,}")
    _log(f"  → 패널 레코드: {len(panel):,}")
    
    # ====== STEP 2: P1 외부 데이터 로드 및 병합 ======
    _log("\n[STEP 2/7] P1 외부 데이터 처리")
    
    if config.external_data_dir and config.external_data_dir.exists():
        external_data = load_all_external_data(
            config.external_data_dir, 
            verbose=config.verbose
        )
        
        if external_data is not None:
            panel["month"] = pd.to_datetime(panel["month"])
            external_data["month"] = pd.to_datetime(external_data["month"])
            
            panel = panel.merge(external_data, on="month", how="left")
            
            # Forward fill
            external_cols = [c for c in external_data.columns if c != "month"]
            for col in external_cols:
                if col in panel.columns:
                    panel[col] = panel.groupby("segment_id")[col].ffill()
            
            _log(f"  → {len(external_cols)}개 외부 변수 병합 완료")
    else:
        _log("  → 외부 데이터 디렉토리 없음, 스킵")
    
    # ====== STEP 3: KPI 계산 ======
    _log("\n[STEP 3/7] KPI 계산")
    panel = compute_kpis(panel, cfg)
    _log(f"  → KPI 컬럼: {list(cfg.kpi_defs.keys())}")
    
    # ====== STEP 4: 시계열 피처 생성 ======
    _log("\n[STEP 4/7] 시계열 피처 생성")
    
    # KPI 컬럼 리스트
    kpi_cols = [kpi for kpi in config.target_kpis if kpi in panel.columns]
    
    panel = build_all_features(
        panel,
        group_col="segment_id",
        time_col="month",
        value_cols=kpi_cols,
        lags=config.lag_periods,
        rolling_windows=config.rolling_windows,
        include_advanced=True,
    )
    
    # 거시경제 변수 피처도 추가
    macro_cols = [c for c in panel.columns if c in cfg.macro_feature_cols]
    if macro_cols:
        from imradar.data.external import add_macro_lag_features
        panel = add_macro_lag_features(panel, macro_cols, lags=[1, 2, 3])
    
    _log(f"  → 피처 컬럼 수: {len(panel.columns)}")
    
    # ====== STEP 5: 멀티호라이즌 예측 모델 학습 ======
    _log("\n[STEP 5/7] 멀티호라이즌 예측 모델 학습")
    
    models = {}
    all_forecasts = []
    all_metrics = []
    
    for kpi in kpi_cols:
        _log(f"\n  [{kpi}]")
        models[kpi] = {}
        
        for h in config.horizons:
            _log(f"    Horizon {h}개월...")
            
            try:
                fm = train_lgbm_forecast(
                    df=panel,
                    kpi_col=kpi,
                    horizon=h,
                    time_col="month",
                    group_col="segment_id",
                    train_end_month=config.train_end_month,
                    cfg=cfg,
                    use_early_stopping=config.use_early_stopping,
                    use_cv=config.use_cv,
                    cv_folds=config.cv_folds,
                    verbose=False,
                )
                
                models[kpi][h] = fm
                
                # CV 성능 기록
                if fm.cv_scores:
                    metrics = {
                        "kpi": kpi,
                        "horizon": h,
                        "cv_mae_mean": np.mean(fm.cv_scores['mae']),
                        "cv_r2_mean": np.mean(fm.cv_scores['r2']),
                    }
                    all_metrics.append(metrics)
                    _log(f"      MAE: {metrics['cv_mae_mean']:.2f}, R²: {metrics['cv_r2_mean']:.3f}")
                
                # 예측값 생성
                predictions = predict_with_model(fm, panel)
                forecast_df = panel[["segment_id", "month"]].copy()
                forecast_df[f"{kpi}_pred_h{h}"] = predictions
                all_forecasts.append(forecast_df)
                
            except Exception as e:
                _log(f"      ✗ 학습 실패: {e}")
    
    # 예측 결과 병합
    if all_forecasts:
        forecasts = all_forecasts[0]
        for f in all_forecasts[1:]:
            pred_cols = [c for c in f.columns if c.endswith(tuple([f"_h{h}" for h in config.horizons]))]
            forecasts = forecasts.merge(f[["segment_id", "month"] + pred_cols], 
                                        on=["segment_id", "month"], how="outer")
    else:
        forecasts = pd.DataFrame()
    
    metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    
    # ====== STEP 6: 리스크 레이더 분석 ======
    _log("\n[STEP 6/7] 리스크 레이더 분석")
    
    risk_alerts_list = []
    segment_risks_list = []
    
    for kpi in kpi_cols:
        if kpi not in models or not models[kpi]:
            continue
        
        # h=1 모델 사용
        if 1 not in models[kpi]:
            continue
        
        fm = models[kpi][1]
        
        try:
            result = run_risk_radar(
                panel_df=panel,
                fm=fm,
                actual_col=kpi,
                cfg=cfg,
                top_alerts=config.top_alerts,
                include_drivers=True,
            )
            
            if not result.summary_df.empty:
                result.summary_df["kpi"] = kpi
                risk_alerts_list.append(result.summary_df)
            
            result.segment_risk_scores["kpi"] = kpi
            segment_risks_list.append(result.segment_risk_scores)
            
            _log(f"  [{kpi}] {len(result.alerts)}개 알림 생성")
            
        except Exception as e:
            _log(f"  [{kpi}] 리스크 분석 실패: {e}")
    
    risk_alerts = pd.concat(risk_alerts_list, ignore_index=True) if risk_alerts_list else pd.DataFrame()
    segment_risks = pd.concat(segment_risks_list, ignore_index=True) if segment_risks_list else pd.DataFrame()
    
    # ====== STEP 7: 결과 저장 ======
    _log("\n[STEP 7/7] 결과 저장")
    
    if config.output_dir:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 예측 결과 저장
        forecasts.to_csv(config.output_dir / "segment_forecasts.csv", index=False, encoding="utf-8-sig")
        
        # 리스크 알림 저장
        risk_alerts.to_csv(config.output_dir / "risk_alerts.csv", index=False, encoding="utf-8-sig")
        
        # 세그먼트 리스크 점수 저장
        segment_risks.to_csv(config.output_dir / "segment_risks.csv", index=False, encoding="utf-8-sig")
        
        # 모델 메트릭 저장
        metrics_df.to_csv(config.output_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")
        
        # 모델 저장
        models_dir = config.output_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        for kpi, kpi_models in models.items():
            for h, fm in kpi_models.items():
                model_path = models_dir / f"{kpi}_h{h}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(fm, f)
        
        _log(f"  → 결과 저장 완료: {config.output_dir}")
    
    _log("\n" + "=" * 60)
    _log("파이프라인 실행 완료!")
    _log("=" * 60)
    
    return PipelineResult(
        panel=panel,
        models=models,
        forecasts=forecasts,
        risk_alerts=risk_alerts,
        segment_risks=segment_risks,
        metrics=metrics_df,
    )


def run_quick_analysis(
    p0_path: Union[str, Path],
    external_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    target_kpis: Optional[List[str]] = None,
    verbose: bool = True,
) -> PipelineResult:
    """
    빠른 분석 실행 (간편 래퍼)
    
    Args:
        p0_path: P0 데이터 파일 경로
        external_dir: 외부 데이터 디렉토리
        output_dir: 출력 디렉토리
        target_kpis: 분석할 KPI 리스트
        verbose: 출력 여부
    
    Returns:
        PipelineResult
    """
    config = PipelineConfig(
        p0_data_path=Path(p0_path),
        external_data_dir=Path(external_dir) if external_dir else None,
        output_dir=Path(output_dir) if output_dir else None,
        target_kpis=target_kpis or ["예금총잔액", "대출총잔액"],
        verbose=verbose,
    )
    
    return run_full_pipeline(config)


def generate_executive_summary(result: PipelineResult) -> str:
    """
    경영진용 요약 리포트 생성
    
    Args:
        result: PipelineResult
    
    Returns:
        요약 리포트 문자열
    """
    lines = [
        "=" * 60,
        "경영진 요약 리포트",
        "=" * 60,
        "",
    ]
    
    # 기본 통계
    n_segments = result.panel["segment_id"].nunique()
    n_months = result.panel["month"].nunique()
    latest_month = result.panel["month"].max()
    
    lines.append(f"■ 분석 개요")
    lines.append(f"  - 분석 세그먼트: {n_segments:,}개")
    lines.append(f"  - 분석 기간: {n_months}개월")
    lines.append(f"  - 최신 데이터: {latest_month.strftime('%Y년 %m월')}")
    lines.append("")
    
    # 모델 성능
    if not result.metrics.empty:
        lines.append(f"■ 예측 모델 성능")
        for _, row in result.metrics.iterrows():
            lines.append(f"  - {row['kpi']} (H{row['horizon']}): R²={row['cv_r2_mean']:.3f}")
        lines.append("")
    
    # 리스크 현황
    if not result.risk_alerts.empty:
        lines.append(f"■ 리스크 현황")
        
        critical = len(result.risk_alerts[result.risk_alerts["severity"] == "critical"])
        high = len(result.risk_alerts[result.risk_alerts["severity"] == "high"])
        drops = len(result.risk_alerts[result.risk_alerts["alert_type"] == "drop"])
        surges = len(result.risk_alerts[result.risk_alerts["alert_type"] == "surge"])
        
        lines.append(f"  - Critical 알림: {critical}건")
        lines.append(f"  - High 알림: {high}건")
        lines.append(f"  - 급감 세그먼트: {drops}건")
        lines.append(f"  - 급증 세그먼트: {surges}건")
        lines.append("")
        
        # 상위 위험 세그먼트
        if not result.segment_risks.empty:
            high_risk = result.segment_risks[result.segment_risks["risk_grade"] == "HIGH"]
            if len(high_risk) > 0:
                lines.append(f"■ 고위험 세그먼트 TOP 5")
                for _, row in high_risk.head(5).iterrows():
                    lines.append(f"  - {row['segment_id']}: 리스크점수 {row['risk_score']:.1f}")
                lines.append("")
    
    # 주요 권고사항
    lines.append(f"■ 주요 권고사항")
    lines.append(f"  1. 급감 세그먼트에 대한 긴급 고객 컨택 권고")
    lines.append(f"  2. 외부 거시경제 변수 변동에 따른 모니터링 강화")
    lines.append(f"  3. 고위험 세그먼트 월별 추적 관리 체계 구축")
    
    return "\n".join(lines)


# 테스트 코드
if __name__ == "__main__":
    print("통합 파이프라인 테스트")
    print("=" * 50)
    
    from pathlib import Path
    
    # 경로 설정
    base_dir = Path(r"C:\Users\campus4D052\Desktop\최종 프로젝트")
    p0_path = base_dir / "외부 데이터" / "iMbank_data.csv.csv"
    external_dir = base_dir / "외부 데이터"
    output_dir = base_dir / "코드 파일" / "outputs" / "pipeline_results"
    
    if p0_path.exists():
        config = PipelineConfig(
            p0_data_path=p0_path,
            external_data_dir=external_dir,
            output_dir=output_dir,
            target_kpis=["예금총잔액", "대출총잔액"],
            horizons=[1, 2, 3],
            verbose=True,
        )
        
        # 파이프라인 실행
        try:
            result = run_full_pipeline(config)
            
            # 경영진 요약
            print("\n")
            print(generate_executive_summary(result))
        except Exception as e:
            print(f"파이프라인 실행 오류: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"P0 데이터 파일을 찾을 수 없습니다: {p0_path}")
