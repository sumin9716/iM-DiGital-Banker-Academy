from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from imradar.config import RadarConfig


@dataclass
class Alert:
    segment_id: str
    month: pd.Timestamp
    kpi: str
    actual: float
    predicted: float
    residual: float
    residual_pct: float
    severity: str
    alert_type: str


def compute_dynamic_threshold(
    residuals: np.ndarray,
    method: str = "mad",
    multiplier: float = 2.5,
) -> Tuple[float, float]:
    """
    동적 임계값 계산 (MAD 또는 IQR 기반)
    
    Args:
        residuals: 잔차 배열
        method: "mad" (Median Absolute Deviation) 또는 "iqr"
        multiplier: 임계값 배수
    
    Returns:
        (lower_threshold, upper_threshold)
    """
    residuals = residuals[~np.isnan(residuals)]
    
    if method == "mad":
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        # MAD를 표준편차로 변환 (정규분포 가정)
        mad_std = mad * 1.4826
        lower = median - multiplier * mad_std
        upper = median + multiplier * mad_std
    else:  # iqr
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
    
    return float(lower), float(upper)


def detect_trend_change(
    series: pd.Series,
    window: int = 3,
    threshold: float = 0.5,
) -> pd.Series:
    """
    트렌드 변곡점 탐지 (방향 전환 감지)
    
    Args:
        series: 시계열 데이터
        window: 이동 평균 윈도우
        threshold: 변화율 임계값
    
    Returns:
        변곡점 플래그 Series
    """
    # 이동 평균 계산
    ma = series.rolling(window=window, min_periods=1).mean()
    
    # 1차 차분 (방향)
    diff = ma.diff()
    
    # 방향 전환 감지: 이전 양수 → 현재 음수 또는 그 반대
    sign_change = (diff.shift(1) * diff) < 0
    
    # 변화 크기가 임계값 이상인 경우만
    pct_change = series.pct_change().abs()
    significant = pct_change >= threshold
    
    return (sign_change & significant).fillna(False)


def score_residual_alerts(
    df: pd.DataFrame,
    kpi: str,
    pred_col: str,
    actual_col: str,
    month_col: str = "month",
    segment_col: str = "segment_id",
    cfg: Optional[RadarConfig] = None,
    use_dynamic_threshold: bool = False,
) -> pd.DataFrame:
    """
    Residual-based alerting with enhanced detection:
      - Static or dynamic thresholds
      - Trend change detection
      - Consecutive alert grouping
    
    Args:
        df: Input DataFrame
        kpi: KPI name
        pred_col: Prediction column name
        actual_col: Actual value column name
        month_col: Month column name
        segment_col: Segment column name
        cfg: RadarConfig instance
        use_dynamic_threshold: Use MAD-based dynamic thresholds
    
    Returns:
        DataFrame with alerts
    """
    if cfg is None:
        cfg = RadarConfig()
    
    eps = 1e-9
    d = df[[segment_col, month_col, actual_col, pred_col]].copy()
    d = d.rename(columns={actual_col: "actual", pred_col: "predicted"})
    d["residual"] = d["actual"] - d["predicted"]
    d["residual_pct"] = d["residual"] / np.maximum(np.abs(d["predicted"]), eps)

    # 동적 임계값 또는 고정 임계값 사용
    if use_dynamic_threshold:
        lower_thr, upper_thr = compute_dynamic_threshold(d["residual_pct"].values)
        # 최소 임계값 보장
        down_threshold_pct = min(lower_thr, cfg.alert_down_threshold_pct)
        up_threshold_pct = max(upper_thr, cfg.alert_up_threshold_pct)
    else:
        down_threshold_pct = cfg.alert_down_threshold_pct
        up_threshold_pct = cfg.alert_up_threshold_pct

    def _sev(x: float) -> str:
        ax = abs(x)
        if ax >= cfg.alert_severity_critical:
            return "CRITICAL"
        if ax >= cfg.alert_severity_high:
            return "HIGH"
        if ax >= cfg.alert_severity_medium:
            return "MEDIUM"
        return "LOW"

    d["severity"] = d["residual_pct"].apply(_sev)
    
    # Alert type determination
    d["alert_type"] = np.where(
        d["residual_pct"] <= down_threshold_pct, "DROP",
        np.where(d["residual_pct"] >= up_threshold_pct, "SPIKE", "NORMAL")
    )
    
    # 트렌드 변곡점 탐지 (세그먼트별)
    d["trend_change"] = False
    for seg in d[segment_col].unique():
        mask = d[segment_col] == seg
        if mask.sum() >= 3:
            seg_series = d.loc[mask, "actual"]
            trend_flags = detect_trend_change(seg_series.reset_index(drop=True))
            d.loc[mask, "trend_change"] = trend_flags.values
    
    # 변곡점 알림 추가 (NORMAL이었던 경우)
    d.loc[(d["trend_change"]) & (d["alert_type"] == "NORMAL"), "alert_type"] = "INFLECTION"
    
    d["kpi"] = kpi
    
    # 연속 알림 그룹핑 (같은 세그먼트, 연속된 월)
    d = d.sort_values([segment_col, month_col])
    d["alert_group"] = 0
    current_group = 0
    prev_seg = None
    prev_type = None
    
    for idx in d.index:
        seg = d.loc[idx, segment_col]
        alert_type = d.loc[idx, "alert_type"]
        
        if alert_type != "NORMAL":
            if seg != prev_seg or alert_type != prev_type:
                current_group += 1
            d.loc[idx, "alert_group"] = current_group
        
        prev_seg = seg
        prev_type = alert_type if alert_type != "NORMAL" else None
    
    return d


def compute_alert_score(row: pd.Series, cfg: Optional[RadarConfig] = None) -> float:
    """
    알림 스코어 계산 (우선순위 결정용)
    
    Factors:
        - severity weight
        - absolute residual percentage
        - trend change bonus
    """
    if cfg is None:
        cfg = RadarConfig()
    
    severity_weights = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    base_score = severity_weights.get(row.get("severity", "LOW"), 1)
    
    # 잔차 비율에 따른 가중치
    residual_factor = min(abs(row.get("residual_pct", 0)) * 2, 2.0)
    
    # 변곡점 보너스
    trend_bonus = 1.5 if row.get("trend_change", False) else 1.0
    
    return base_score * residual_factor * trend_bonus


def build_watchlist(
    alerts: pd.DataFrame, 
    top_n: int = 50,
    include_scores: bool = True,
    cfg: Optional[RadarConfig] = None,
) -> pd.DataFrame:
    """
    Select top-N alerts by severity and score.
    
    Args:
        alerts: DataFrame with alert information
        top_n: Number of top alerts to return
        include_scores: Calculate and include alert scores
        cfg: RadarConfig instance
    
    Returns:
        Top N alerts DataFrame
    """
    sev_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    a = alerts.copy()
    a["sev_rank"] = a["severity"].map(sev_rank).fillna(0).astype(int)
    a = a[a["alert_type"] != "NORMAL"].copy()
    
    if include_scores:
        a["alert_score"] = a.apply(lambda row: compute_alert_score(row, cfg), axis=1)
        a = a.sort_values(["sev_rank", "alert_score"], ascending=[False, False])
    else:
        a = a.sort_values(["sev_rank", "residual_pct"], ascending=[False, True])
    
    result = a.head(top_n)
    if "sev_rank" in result.columns:
        result = result.drop(columns=["sev_rank"])
    
    return result


def get_alert_summary(alerts: pd.DataFrame) -> Dict[str, any]:
    """
    알림 요약 통계 생성
    
    Returns:
        Dict with summary statistics
    """
    non_normal = alerts[alerts["alert_type"] != "NORMAL"]
    
    summary = {
        "total_alerts": len(non_normal),
        "critical_count": len(non_normal[non_normal["severity"] == "CRITICAL"]),
        "high_count": len(non_normal[non_normal["severity"] == "HIGH"]),
        "medium_count": len(non_normal[non_normal["severity"] == "MEDIUM"]),
        "low_count": len(non_normal[non_normal["severity"] == "LOW"]),
        "drop_count": len(non_normal[non_normal["alert_type"] == "DROP"]),
        "spike_count": len(non_normal[non_normal["alert_type"] == "SPIKE"]),
        "inflection_count": len(non_normal[non_normal["alert_type"] == "INFLECTION"]),
        "unique_segments": non_normal["segment_id"].nunique() if "segment_id" in non_normal.columns else 0,
        "avg_residual_pct": float(non_normal["residual_pct"].abs().mean()) if len(non_normal) > 0 else 0,
    }
    
    return summary
