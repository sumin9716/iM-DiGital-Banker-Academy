"""
리스크 레이더 모듈

예측 대비 실적(residual) 기반 알림 시스템
- residual_ratio = (actual - pred) / pred
- 임계치 기반 급증/급감 경보
- Rolling 표준편차 기반 z-score 이상 탐지
- SHAP 기반 원인 분석 통합
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

from imradar.config import RadarConfig
from imradar.models.forecast_lgbm import ForecastModel, predict_with_model


@dataclass
class RiskAlert:
    """리스크 알림 데이터 클래스"""
    segment_id: str
    month: pd.Timestamp
    kpi: str
    actual: float
    predicted: float
    residual: float
    residual_ratio: float
    zscore: float
    severity: str  # critical, high, medium, low
    alert_type: str  # surge, drop, anomaly
    drivers: List[str] = field(default_factory=list)
    driver_contributions: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class RiskRadarResult:
    """리스크 레이더 분석 결과"""
    alerts: List[RiskAlert]
    summary_df: pd.DataFrame
    segment_risk_scores: pd.DataFrame
    monthly_trend: pd.DataFrame


def compute_residuals(
    df: pd.DataFrame,
    fm: ForecastModel,
    actual_col: str,
    time_col: str = "month",
    group_col: str = "segment_id",
) -> pd.DataFrame:
    """
    예측 대비 잔차(residual) 계산
    
    Args:
        df: 패널 데이터
        fm: 학습된 예측 모델
        actual_col: 실제값 컬럼
        time_col: 시간 컬럼
        group_col: 세그먼트 컬럼
    
    Returns:
        잔차 정보가 추가된 DataFrame
    """
    data = df.copy()
    
    # 예측값 생성
    predictions = predict_with_model(fm, data)
    data["predicted"] = predictions
    data["actual"] = data[actual_col]
    
    # 잔차 계산
    eps = 1e-9
    data["residual"] = data["actual"] - data["predicted"]
    data["residual_ratio"] = data["residual"] / (np.abs(data["predicted"]) + eps)
    
    # Signed ratio (예측값이 음수일 수 있으므로)
    data["residual_pct"] = (data["residual"] / (np.abs(data["predicted"]) + eps)) * 100
    
    return data


def compute_rolling_zscore(
    df: pd.DataFrame,
    group_col: str = "segment_id",
    value_col: str = "residual_ratio",
    window: int = 6,
    min_periods: int = 3,
) -> pd.DataFrame:
    """
    Rolling Z-score 계산
    
    각 세그먼트별로 과거 residual 분포 대비 현재 값의 이상도 계산
    
    Args:
        df: residual이 포함된 DataFrame
        group_col: 세그먼트 컬럼
        value_col: z-score 계산할 값 컬럼
        window: rolling 윈도우 크기
        min_periods: 최소 관측 수
    
    Returns:
        z-score가 추가된 DataFrame
    """
    data = df.copy()
    
    # 세그먼트별 rolling 통계
    g = data.groupby(group_col)[value_col]
    
    # 이전 값들의 rolling 평균과 표준편차 (현재 값 제외)
    shifted = g.shift(1)
    rolling_mean = shifted.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = shifted.rolling(window=window, min_periods=min_periods).std()
    
    # Z-score 계산
    eps = 1e-9
    data[f"{value_col}_zscore"] = (data[value_col] - rolling_mean) / (rolling_std + eps)
    data[f"{value_col}_rolling_mean"] = rolling_mean
    data[f"{value_col}_rolling_std"] = rolling_std
    
    return data


def determine_severity(
    residual_ratio: float,
    zscore: float,
    cfg: RadarConfig,
) -> str:
    """
    알림 심각도 결정
    
    Args:
        residual_ratio: 잔차 비율
        zscore: z-score
        cfg: 설정
    
    Returns:
        severity 문자열 (critical, high, medium, low)
    """
    abs_ratio = abs(residual_ratio)
    abs_zscore = abs(zscore) if not pd.isna(zscore) else 0
    
    # 복합 점수 계산
    score = max(abs_ratio, abs_zscore / 3)  # z-score 3 = ratio 1.0 수준
    
    if score >= cfg.alert_severity_critical:
        return "critical"
    elif score >= cfg.alert_severity_high:
        return "high"
    elif score >= cfg.alert_severity_medium:
        return "medium"
    else:
        return "low"


def determine_alert_type(
    residual_ratio: float,
    zscore: float,
    down_threshold: float = -0.20,
    up_threshold: float = 0.20,
) -> str:
    """
    알림 유형 결정
    
    Args:
        residual_ratio: 잔차 비율
        zscore: z-score
        down_threshold: 급감 임계치
        up_threshold: 급증 임계치
    
    Returns:
        alert_type 문자열 (surge, drop, anomaly, normal)
    """
    if residual_ratio >= up_threshold:
        return "surge"
    elif residual_ratio <= down_threshold:
        return "drop"
    elif abs(zscore) > 2.5 if not pd.isna(zscore) else False:
        return "anomaly"
    else:
        return "normal"


def generate_alerts(
    df: pd.DataFrame,
    fm: ForecastModel,
    cfg: Optional[RadarConfig] = None,
    target_month: Optional[pd.Timestamp] = None,
    top_n: int = 20,
    include_normal: bool = False,
) -> List[RiskAlert]:
    """
    리스크 알림 생성
    
    Args:
        df: 잔차 정보가 포함된 DataFrame
        fm: 예측 모델 (SHAP 분석용)
        cfg: 설정
        target_month: 특정 월만 분석 (None이면 전체)
        top_n: 상위 N개 알림만 반환
        include_normal: 정상 알림 포함 여부
    
    Returns:
        RiskAlert 리스트
    """
    if cfg is None:
        cfg = RadarConfig()
    
    data = df.copy()
    
    # 특정 월 필터링
    if target_month is not None:
        data = data[data["month"] == target_month]
    
    alerts = []
    
    for idx, row in data.iterrows():
        residual_ratio = row.get("residual_ratio", 0)
        zscore = row.get("residual_ratio_zscore", np.nan)
        
        alert_type = determine_alert_type(
            residual_ratio, zscore,
            cfg.alert_down_threshold_pct, cfg.alert_up_threshold_pct
        )
        
        if alert_type == "normal" and not include_normal:
            continue
        
        severity = determine_severity(residual_ratio, zscore, cfg)
        
        alert = RiskAlert(
            segment_id=str(row.get("segment_id", "")),
            month=pd.to_datetime(row.get("month")),
            kpi=fm.kpi,
            actual=float(row.get("actual", 0)),
            predicted=float(row.get("predicted", 0)),
            residual=float(row.get("residual", 0)),
            residual_ratio=float(residual_ratio),
            zscore=float(zscore) if not pd.isna(zscore) else 0.0,
            severity=severity,
            alert_type=alert_type,
        )
        alerts.append(alert)
    
    # 심각도와 잔차비율 기준 정렬
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    alerts.sort(key=lambda x: (severity_order.get(x.severity, 4), -abs(x.residual_ratio)))
    
    return alerts[:top_n]


def add_driver_analysis(
    alerts: List[RiskAlert],
    df: pd.DataFrame,
    fm: ForecastModel,
    top_k: int = 5,
) -> List[RiskAlert]:
    """
    알림에 SHAP 기반 드라이버 분석 추가
    
    Args:
        alerts: RiskAlert 리스트
        df: 원본 데이터
        fm: 예측 모델
        top_k: 상위 드라이버 수
    
    Returns:
        드라이버 분석이 추가된 RiskAlert 리스트
    """
    # SHAP 계산을 위한 Feature 준비
    X = df[fm.feature_cols].copy()
    for c in fm.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    
    # LightGBM pred_contrib 사용
    contribs = fm.model.predict(X, pred_contrib=True)
    contribs = contribs[:, :-1]  # bias 제외
    
    # DataFrame에 인덱스 매핑
    df_indexed = df.reset_index(drop=True)
    
    for alert in alerts:
        # 해당 알림의 행 찾기
        mask = (df_indexed["segment_id"] == alert.segment_id) & \
               (df_indexed["month"] == alert.month)
        
        if mask.sum() == 0:
            continue
        
        row_idx = mask.idxmax()
        row_contrib = contribs[row_idx]
        
        # 상위 드라이버 추출
        contrib_series = pd.Series(row_contrib, index=fm.feature_cols)
        
        if alert.alert_type == "surge":
            # 상승 원인: 양의 기여도 상위
            top_drivers = contrib_series.nlargest(top_k)
        elif alert.alert_type == "drop":
            # 하락 원인: 음의 기여도 상위 (가장 작은 값)
            top_drivers = contrib_series.nsmallest(top_k)
        else:
            # 이상치: 절대값 기준
            top_drivers = contrib_series.abs().nlargest(top_k)
            top_drivers = contrib_series[top_drivers.index]
        
        alert.drivers = top_drivers.index.tolist()
        alert.driver_contributions = top_drivers.to_dict()
        alert.explanation = _generate_explanation(alert, top_drivers)
    
    return alerts


def _generate_explanation(alert: RiskAlert, top_drivers: pd.Series) -> str:
    """
    알림에 대한 설명 문장 생성 (개선된 버전)
    
    Args:
        alert: RiskAlert
        top_drivers: 상위 드라이버 Series
    
    Returns:
        설명 문장
    """
    # 피처명 포맷팅 (개선된 매핑)
    FEATURE_NAMES = {
        "예금총잔액": "예금잔액",
        "대출총잔액": "대출잔액",
        "순유입": "순유입",
        "카드총사용": "카드사용",
        "디지털거래금액": "디지털거래",
        "FX총액": "외환거래",
        "fed_rate": "미국금리",
        "usd_krw": "달러환율",
        "brent": "유가",
        "esi": "경기전망",
        "apt_price": "부동산",
    }
    
    def _format_feature(f: str) -> str:
        name = f.replace("log1p_", "").replace("__", " ")
        
        # 시점 정보 추출
        time_suffix = ""
        if "lag1" in name:
            time_suffix = " (전월)"
            name = name.replace("lag1", "").replace("_", " ")
        elif "lag3" in name:
            time_suffix = " (3개월전)"
            name = name.replace("lag3", "").replace("_", " ")
        elif "lag6" in name:
            time_suffix = " (6개월전)"
            name = name.replace("lag6", "").replace("_", " ")
        elif "roll3_mean" in name:
            time_suffix = " 3개월평균"
            name = name.replace("roll3_mean", "").replace("_", " ")
        elif "roll6_mean" in name:
            time_suffix = " 6개월평균"
            name = name.replace("roll6_mean", "").replace("_", " ")
        elif "mom" in name:
            time_suffix = " 전월비"
            name = name.replace("mom", "").replace("_", " ")
        elif "yoy" in name:
            time_suffix = " 전년비"
            name = name.replace("yoy", "").replace("_", " ")
        
        # 매핑된 이름 적용
        for key, display in FEATURE_NAMES.items():
            if key in name:
                name = name.replace(key, display)
        
        return (name.strip() + time_suffix).strip()
    
    # 알림 유형별 문장 생성
    ratio_pct = abs(alert.residual_ratio * 100)
    
    if alert.alert_type == "surge":
        direction = "급등" if ratio_pct > 20 else "상승"
        icon = "📈"
    elif alert.alert_type == "drop":
        direction = "급락" if ratio_pct > 20 else "하락"
        icon = "📉"
    else:
        direction = "이상 변동"
        icon = "⚠️"
    
    # 상위 드라이버 분류
    pos_drivers = [(f, v) for f, v in top_drivers.items() if v > 0]
    neg_drivers = [(f, v) for f, v in top_drivers.items() if v < 0]
    
    explanation_parts = []
    
    # 도입부
    explanation_parts.append(
        f"{icon} 예측 대비 {ratio_pct:.1f}% {direction}"
    )
    
    # 주요 원인 분석
    if alert.alert_type == "surge" and pos_drivers:
        # 상승의 경우 양수 기여 요인이 원인
        main_drivers = [_format_feature(f) for f, v in sorted(pos_drivers, key=lambda x: -x[1])[:2]]
        if main_drivers:
            explanation_parts.append(f"상승 요인: {', '.join(main_drivers)}")
    elif alert.alert_type == "drop" and neg_drivers:
        # 하락의 경우 음수 기여 요인이 원인
        main_drivers = [_format_feature(f) for f, v in sorted(neg_drivers, key=lambda x: x[1])[:2]]
        if main_drivers:
            explanation_parts.append(f"하락 요인: {', '.join(main_drivers)}")
    else:
        # 복합적 원인
        if pos_drivers:
            pos_names = [_format_feature(f) for f, _ in pos_drivers[:1]]
            explanation_parts.append(f"상승 압력: {', '.join(pos_names)}")
        if neg_drivers:
            neg_names = [_format_feature(f) for f, _ in neg_drivers[:1]]
            explanation_parts.append(f"하락 압력: {', '.join(neg_names)}")
    
    # 심각도에 따른 권고 추가
    if hasattr(alert, 'severity'):
        if alert.severity == "CRITICAL":
            explanation_parts.append("→ 즉시 점검 필요")
        elif alert.severity == "HIGH":
            explanation_parts.append("→ 조기 대응 권고")
    
    return " | ".join(explanation_parts)


def compute_segment_risk_score(
    df: pd.DataFrame,
    group_col: str = "segment_id",
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    세그먼트별 리스크 점수 계산
    
    최근 N개월간의 residual 패턴을 기반으로 종합 리스크 점수 산출
    
    Args:
        df: 잔차 정보가 포함된 DataFrame
        group_col: 세그먼트 컬럼
        lookback_months: 분석 기간 (월)
    
    Returns:
        세그먼트별 리스크 점수 DataFrame
    """
    # 최근 N개월 필터링
    max_month = df["month"].max()
    min_month = max_month - pd.DateOffset(months=lookback_months)
    recent = df[df["month"] >= min_month].copy()
    
    # 세그먼트별 집계
    risk_scores = recent.groupby(group_col).agg({
        "residual_ratio": ["mean", "std", "min", "max"],
        "residual_ratio_zscore": ["mean", "std", lambda x: (x.abs() > 2).sum()],
        "actual": "mean",
        "predicted": "mean",
    }).reset_index()
    
    # 컬럼명 정리
    risk_scores.columns = [
        group_col,
        "avg_residual_ratio", "std_residual_ratio", "min_residual_ratio", "max_residual_ratio",
        "avg_zscore", "std_zscore", "anomaly_count",
        "avg_actual", "avg_predicted",
    ]
    
    # 종합 리스크 점수 계산 (0-100)
    # 구성요소: 평균 잔차비율, 변동성, 이상치 빈도
    eps = 1e-9
    
    # 정규화
    risk_scores["ratio_score"] = risk_scores["avg_residual_ratio"].abs() * 100
    risk_scores["volatility_score"] = risk_scores["std_residual_ratio"] * 100
    risk_scores["anomaly_score"] = risk_scores["anomaly_count"] * 10
    
    # 종합 점수 (각 요소 가중 평균)
    risk_scores["risk_score"] = (
        0.4 * risk_scores["ratio_score"] +
        0.3 * risk_scores["volatility_score"] +
        0.3 * risk_scores["anomaly_score"]
    ).clip(0, 100)
    
    # 리스크 등급 부여
    conditions = [
        risk_scores["risk_score"] >= 70,
        risk_scores["risk_score"] >= 50,
        risk_scores["risk_score"] >= 30,
    ]
    choices = ["HIGH", "MEDIUM", "LOW"]
    risk_scores["risk_grade"] = np.select(conditions, choices, default="NORMAL")
    
    return risk_scores.sort_values("risk_score", ascending=False).reset_index(drop=True)


def compute_monthly_trend(
    df: pd.DataFrame,
    time_col: str = "month",
) -> pd.DataFrame:
    """
    월별 전체 리스크 트렌드 계산
    
    Args:
        df: 잔차 정보가 포함된 DataFrame
        time_col: 시간 컬럼
    
    Returns:
        월별 리스크 트렌드 DataFrame
    """
    monthly = df.groupby(time_col).agg({
        "residual_ratio": ["mean", "std"],
        "residual_ratio_zscore": lambda x: (x.abs() > 2).sum(),
        "segment_id": "nunique",
        "actual": "sum",
        "predicted": "sum",
    }).reset_index()
    
    monthly.columns = [
        time_col,
        "avg_residual_ratio", "std_residual_ratio",
        "anomaly_count", "segment_count",
        "total_actual", "total_predicted",
    ]
    
    # 전체 오차율
    eps = 1e-9
    monthly["overall_error_pct"] = (
        (monthly["total_actual"] - monthly["total_predicted"]) / 
        (monthly["total_predicted"].abs() + eps) * 100
    )
    
    # 이상 비율
    monthly["anomaly_pct"] = monthly["anomaly_count"] / monthly["segment_count"] * 100
    
    return monthly.sort_values(time_col).reset_index(drop=True)


def run_risk_radar(
    panel_df: pd.DataFrame,
    fm: ForecastModel,
    actual_col: str,
    cfg: Optional[RadarConfig] = None,
    target_month: Optional[pd.Timestamp] = None,
    top_alerts: int = 20,
    include_drivers: bool = True,
) -> RiskRadarResult:
    """
    리스크 레이더 전체 분석 실행
    
    Args:
        panel_df: 패널 데이터
        fm: 학습된 예측 모델
        actual_col: 실제값 컬럼
        cfg: 설정
        target_month: 특정 월만 분석
        top_alerts: 상위 알림 수
        include_drivers: 드라이버 분석 포함 여부
    
    Returns:
        RiskRadarResult
    """
    if cfg is None:
        cfg = RadarConfig()
    
    # 1. 잔차 계산
    data = compute_residuals(panel_df, fm, actual_col)
    
    # 2. Rolling Z-score 계산
    data = compute_rolling_zscore(data)
    
    # 3. 알림 생성
    alerts = generate_alerts(
        data, fm, cfg,
        target_month=target_month,
        top_n=top_alerts
    )
    
    # 4. 드라이버 분석 추가
    if include_drivers and alerts:
        alerts = add_driver_analysis(alerts, data, fm)
    
    # 5. 세그먼트별 리스크 점수
    segment_risk = compute_segment_risk_score(data)
    
    # 6. 월별 트렌드
    monthly_trend = compute_monthly_trend(data)
    
    # 7. 알림 요약 DataFrame
    if alerts:
        summary_df = pd.DataFrame([{
            "segment_id": a.segment_id,
            "month": a.month,
            "kpi": a.kpi,
            "actual": a.actual,
            "predicted": a.predicted,
            "residual_ratio": a.residual_ratio,
            "zscore": a.zscore,
            "severity": a.severity,
            "alert_type": a.alert_type,
            "top_driver_1": a.drivers[0] if len(a.drivers) > 0 else "",
            "top_driver_2": a.drivers[1] if len(a.drivers) > 1 else "",
            "top_driver_3": a.drivers[2] if len(a.drivers) > 2 else "",
            "explanation": a.explanation,
        } for a in alerts])
    else:
        summary_df = pd.DataFrame()
    
    return RiskRadarResult(
        alerts=alerts,
        summary_df=summary_df,
        segment_risk_scores=segment_risk,
        monthly_trend=monthly_trend,
    )


def export_alerts_to_csv(
    alerts: List[RiskAlert],
    output_path: Union[str, Path],
) -> None:
    """
    알림을 CSV로 내보내기
    
    Args:
        alerts: RiskAlert 리스트
        output_path: 출력 파일 경로
    """
    if not alerts:
        pd.DataFrame().to_csv(output_path, index=False)
        return
    
    rows = []
    for a in alerts:
        row = {
            "segment_id": a.segment_id,
            "month": a.month,
            "kpi": a.kpi,
            "actual": a.actual,
            "predicted": a.predicted,
            "residual": a.residual,
            "residual_ratio": a.residual_ratio,
            "zscore": a.zscore,
            "severity": a.severity,
            "alert_type": a.alert_type,
            "explanation": a.explanation,
        }
        
        # 드라이버 추가
        for i, driver in enumerate(a.drivers[:5]):
            row[f"driver_{i+1}"] = driver
            row[f"contrib_{i+1}"] = a.driver_contributions.get(driver, 0)
        
        rows.append(row)
    
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
