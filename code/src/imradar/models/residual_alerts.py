from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from imradar.config import RadarConfig


# ============================================================================
# KPI별 맞춤형 임계값 설정
# ============================================================================
# KPI 특성에 따라 서로 다른 임계값 적용:
# - 예금총잔액: 안정적, 작은 변동도 중요
# - 순유입: 변동성 높음, 더 넓은 임계값
# - 카드총사용: 월별 변동 있음, 중간 수준
# - FX총액: 환율 영향으로 변동성 높음
# NOTE: residual_pct는 실제 비율 (1.0 = 100%)
KPI_THRESHOLD_CONFIG = {
    "예금총잔액": {
        "down_threshold": -0.15,  # 15% 이상 감소 시 DROP
        "up_threshold": 0.30,     # 30% 이상 증가 시 SPIKE
        "severity_critical": 2.0,  # 200% 이상 CRITICAL
        "severity_high": 1.0,      # 100% 이상 HIGH
        "severity_medium": 0.5,    # 50% 이상 MEDIUM
    },
    "대출총잔액": {
        "down_threshold": -0.20,
        "up_threshold": 0.40,
        "severity_critical": 3.0,
        "severity_high": 1.5,
        "severity_medium": 0.7,
    },
    "순유입": {
        "down_threshold": -0.50,  # 순유입은 변동성 높음
        "up_threshold": 1.0,
        "severity_critical": 5.0,
        "severity_high": 2.5,
        "severity_medium": 1.0,
    },
    "카드총사용": {
        "down_threshold": -0.25,
        "up_threshold": 0.50,
        "severity_critical": 3.0,
        "severity_high": 1.5,
        "severity_medium": 0.7,
    },
    "디지털거래금액": {
        "down_threshold": -0.20,
        "up_threshold": 0.40,
        "severity_critical": 2.5,
        "severity_high": 1.2,
        "severity_medium": 0.6,
    },
    "FX총액": {
        "down_threshold": -0.30,  # FX는 환율 영향으로 변동성 높음
        "up_threshold": 0.60,
        "severity_critical": 4.0,
        "severity_high": 2.0,
        "severity_medium": 0.8,
    },
}

# 기본 임계값 (설정되지 않은 KPI용)
DEFAULT_THRESHOLD_CONFIG = {
    "down_threshold": -0.20,
    "up_threshold": 0.40,
    "severity_critical": 3.0,
    "severity_high": 1.5,
    "severity_medium": 0.7,
}


def get_kpi_thresholds(kpi: str) -> Dict[str, float]:
    """KPI별 맞춤 임계값 반환"""
    return KPI_THRESHOLD_CONFIG.get(kpi, DEFAULT_THRESHOLD_CONFIG)


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
    denom = np.maximum(np.abs(d["predicted"]), np.abs(d["actual"]) * 0.1)
    denom = np.maximum(denom, eps)
    d["residual_pct"] = d["residual"] / denom
    d["residual_pct"] = d["residual_pct"].clip(-5.0, 5.0)

    # ============== KPI별 맞춤 임계값 적용 ==============
    kpi_config = get_kpi_thresholds(kpi)
    
    # 동적 임계값 또는 KPI별 고정 임계값 사용
    if use_dynamic_threshold:
        lower_thr, upper_thr = compute_dynamic_threshold(d["residual_pct"].values)
        # KPI별 최소/최대 임계값 보장
        down_threshold_pct = min(lower_thr, kpi_config["down_threshold"])
        up_threshold_pct = max(upper_thr, kpi_config["up_threshold"])
    else:
        down_threshold_pct = kpi_config["down_threshold"]
        up_threshold_pct = kpi_config["up_threshold"]

    # KPI별 severity 임계값 적용
    def _sev(x: float) -> str:
        ax = abs(x)
        if ax >= kpi_config["severity_critical"]:
            return "CRITICAL"
        if ax >= kpi_config["severity_high"]:
            return "HIGH"
        if ax >= kpi_config["severity_medium"]:
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
    balanced_kpi: bool = True,
    balanced_industry: bool = True,
    max_per_industry: int = 3,
) -> pd.DataFrame:
    """
    Select top-N alerts by severity and score with KPI and industry diversity.
    
    Args:
        alerts: DataFrame with alert information
        top_n: Number of top alerts to return
        include_scores: Calculate and include alert scores
        cfg: RadarConfig instance
        balanced_kpi: KPI별 균형있게 선택 (True면 각 KPI에서 균등 추출)
        balanced_industry: 업종별 다양성 보장 (True면 업종별 최대 개수 제한)
        max_per_industry: 업종별 최대 알림 개수 (balanced_industry=True일 때)
    
    Returns:
        Top N alerts DataFrame
    """
    sev_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    a = alerts.copy()
    a["sev_rank"] = a["severity"].map(sev_rank).fillna(0).astype(int)
    a = a[a["alert_type"] != "NORMAL"].copy()
    
    # segment_id에서 업종 추출 (segment_id 형식: "업종|지역|등급|전담")
    if "segment_id" in a.columns:
        a["_industry"] = a["segment_id"].apply(lambda x: x.split("|")[0] if isinstance(x, str) and "|" in x else x)
    
    if include_scores:
        a["alert_score"] = a.apply(lambda row: compute_alert_score(row, cfg), axis=1)
        # severity 다양성을 위해 score 기반으로만 정렬 (severity는 결과에 포함만)
        a = a.sort_values(["alert_score"], ascending=[False])
    else:
        a = a.sort_values(["residual_pct"], ascending=[True])
    
    # ============== severity별 균형 선택 추가 ==============
    # severity 다양성 확보: 각 severity 레벨에서 일정 비율 선택
    severity_levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    available_severities = [s for s in severity_levels if s in a["severity"].values]
    
    if len(available_severities) > 1 and len(a) > 20:
        # severity별 할당량 계산 (CRITICAL 40%, HIGH 30%, MEDIUM 20%, LOW 10%)
        severity_quota = {
            "CRITICAL": int(top_n * 0.40),
            "HIGH": int(top_n * 0.30),
            "MEDIUM": int(top_n * 0.20),
            "LOW": int(top_n * 0.10),
        }
        
        # 각 severity에서 할당량만큼 선택
        balanced_by_severity = []
        for sev in available_severities:
            sev_alerts = a[a["severity"] == sev]
            quota = severity_quota.get(sev, 5)
            if len(sev_alerts) > 0:
                balanced_by_severity.append(sev_alerts.head(quota))
        
        if balanced_by_severity:
            a = pd.concat(balanced_by_severity, ignore_index=True)
            # 다시 score로 정렬
            if "alert_score" in a.columns:
                a = a.sort_values("alert_score", ascending=False)
    
    # ============== KPI별로 먼저 분배, 각 KPI 내에서 업종 다양성 보장 ==============
    if balanced_kpi and "kpi" in a.columns:
        kpis = a["kpi"].unique()
        n_kpis = len(kpis)
        per_kpi = max(top_n // n_kpis, 5)  # KPI당 할당량
        
        all_selected = []
        global_industry_counts = {}  # 전체 업종 카운트
        
        for kpi in kpis:
            kpi_alerts = a[a["kpi"] == kpi].copy()
            kpi_selected = []
            
            if balanced_industry and "_industry" in kpi_alerts.columns:
                # 업종별 다양성 보장하면서 선택
                for _, row in kpi_alerts.iterrows():
                    ind = row["_industry"]
                    global_cnt = global_industry_counts.get(ind, 0)
                    
                    # 전체적으로 업종당 max_per_industry 이하만 허용
                    if global_cnt < max_per_industry:
                        kpi_selected.append(row)
                        global_industry_counts[ind] = global_cnt + 1
                    
                    if len(kpi_selected) >= per_kpi:
                        break
            else:
                kpi_selected = kpi_alerts.head(per_kpi).to_dict('records')
            
            all_selected.extend(kpi_selected)
        
        result = pd.DataFrame(all_selected)
        
        # 부족한 경우 추가 채우기
        if len(result) < top_n:
            used_ids = set(result["segment_id"].tolist()) if not result.empty else set()
            remaining = a[~a["segment_id"].isin(used_ids)]
            
            for _, row in remaining.iterrows():
                ind = row["_industry"]
                cnt = global_industry_counts.get(ind, 0)
                if cnt < max_per_industry + 2:  # 약간 여유 허용
                    all_selected.append(row)
                    global_industry_counts[ind] = cnt + 1
                if len(all_selected) >= top_n:
                    break
            
            result = pd.DataFrame(all_selected)
        
        result = result.head(top_n)
    
    elif balanced_industry and "_industry" in a.columns:
        # KPI 균형 없이 업종 다양성만
        selected = []
        industry_counts = {}
        for _, row in a.iterrows():
            ind = row["_industry"]
            cnt = industry_counts.get(ind, 0)
            if cnt < max_per_industry:
                selected.append(row)
                industry_counts[ind] = cnt + 1
            if len(selected) >= top_n:
                break
        result = pd.DataFrame(selected)
    
    else:
        result = a.head(top_n)
    
    # 임시 컬럼 제거
    for col in ["sev_rank", "_industry"]:
        if col in result.columns:
            result = result.drop(columns=[col])
    
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


def compute_risk_impact_score(
    watchlist: pd.DataFrame,
    panel: pd.DataFrame,
    segment_col: str = "segment_id",
    month_col: str = "month",
    balance_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    리스크 임팩트 점수 계산 (잔액 규모 기반)
    
    임팩트 = |잔차비율| × 세그먼트 잔액 규모 × 심각도 가중치
    
    Args:
        watchlist: 워치리스트 DataFrame
        panel: 전체 패널 데이터
        segment_col: 세그먼트 ID 컬럼
        month_col: 월 컬럼
        balance_cols: 잔액 컬럼 목록 (기본: 예금총잔액, 대출총잔액)
    
    Returns:
        임팩트 점수가 추가된 워치리스트
    """
    if balance_cols is None:
        balance_cols = ["예금총잔액", "대출총잔액"]
    
    result = watchlist.copy()
    
    if result.empty:
        return result
    
    # 최신 월 데이터 추출
    latest_month = panel[month_col].max()
    latest_panel = panel[panel[month_col] == latest_month].copy()
    
    # 사용 가능한 잔액 컬럼 확인
    available_balance_cols = [c for c in balance_cols if c in latest_panel.columns]
    
    if not available_balance_cols:
        result["impact_score"] = result.get("alert_score", 1.0)
        return result
    
    # 총 잔액 계산
    latest_panel["total_balance"] = latest_panel[available_balance_cols].sum(axis=1)
    
    # 전체 평균 잔액 (정규화용)
    mean_balance = latest_panel["total_balance"].mean()
    if mean_balance == 0:
        mean_balance = 1
    
    # 워치리스트에 잔액 정보 조인
    balance_info = latest_panel[[segment_col, "total_balance"]].drop_duplicates()
    result = result.merge(balance_info, on=segment_col, how="left")
    result["total_balance"] = result["total_balance"].fillna(0)
    
    # 심각도 가중치
    severity_weights = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
    result["severity_weight"] = result["severity"].map(severity_weights).fillna(1)
    
    # 임팩트 점수 계산
    # Impact = |residual_pct| × (balance / mean_balance) × severity_weight
    result["balance_ratio"] = result["total_balance"] / mean_balance
    result["impact_score"] = (
        result["residual_pct"].abs() * 
        result["balance_ratio"].clip(lower=0.1, upper=10) * 
        result["severity_weight"]
    )
    
    # 정규화 (0-100 스케일)
    max_impact = result["impact_score"].max()
    if max_impact > 0:
        result["impact_score_normalized"] = result["impact_score"] / max_impact * 100
    else:
        result["impact_score_normalized"] = 0
    
    # 임시 컬럼 정리
    result = result.drop(columns=["severity_weight", "balance_ratio"], errors="ignore")
    
    # 임팩트 순으로 정렬
    result = result.sort_values("impact_score", ascending=False)
    
    return result
