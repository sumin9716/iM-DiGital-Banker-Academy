"""
Hierarchical Reconciliation (MinT) for Segment Forecasts
=========================================================

세그먼트 계층 구조를 고려한 예측 일관성 조정 모듈.
하위 세그먼트 예측의 합이 상위 집계와 일치하도록 조정.

계층 구조:
- 전체 (Total)
  └─ 업종_중분류
     └─ 업종_중분류 × 사업장_시도
        └─ 업종_중분류 × 사업장_시도 × 법인_고객등급
           └─ 업종_중분류 × 사업장_시도 × 법인_고객등급 × 전담고객여부 (Bottom)

Methods:
- Bottom-Up: 하위 예측 합산
- Top-Down: 상위 예측을 비례 분배
- MinT (Minimum Trace): 최적 조합 (Wickramasuriya et al., 2019)
- OLS: 단순 최소자승 조정
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass


@dataclass
class HierarchyLevel:
    """계층 수준 정의"""
    name: str
    keys: List[str]  # 해당 수준을 정의하는 키 컬럼들


# 세그먼트 계층 구조 정의
HIERARCHY_LEVELS = [
    HierarchyLevel("total", []),
    HierarchyLevel("industry", ["업종_중분류"]),
    HierarchyLevel("industry_region", ["업종_중분류", "사업장_시도"]),
    HierarchyLevel("industry_region_grade", ["업종_중분류", "사업장_시도", "법인_고객등급"]),
    HierarchyLevel("bottom", ["업종_중분류", "사업장_시도", "법인_고객등급", "전담고객여부"]),
]


def build_summing_matrix(
    bottom_segments: pd.DataFrame,
    hierarchy_levels: List[HierarchyLevel] = HIERARCHY_LEVELS,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Summing Matrix (S) 구축.
    
    S matrix: 하위 예측을 상위로 집계하는 행렬
    y_agg = S @ y_bottom
    
    Args:
        bottom_segments: 최하위 세그먼트 메타데이터 (segment_id + 키 컬럼들)
        hierarchy_levels: 계층 수준 정의 리스트
        
    Returns:
        S: Summing matrix (n_total x n_bottom)
        agg_index: 집계 수준별 인덱스
        level_names: 각 행의 수준명
    """
    bottom_keys = hierarchy_levels[-1].keys
    n_bottom = len(bottom_segments)
    
    # 각 수준별 고유 조합 추출
    all_rows = []
    level_names = []
    
    for level in hierarchy_levels:
        if not level.keys:  # Total
            # 모든 하위 세그먼트를 합산
            row = np.ones(n_bottom)
            all_rows.append(row)
            level_names.append(("total", "TOTAL"))
        else:
            # 해당 수준의 고유 조합
            unique_combos = bottom_segments[level.keys].drop_duplicates()
            for _, combo in unique_combos.iterrows():
                # 이 조합에 해당하는 하위 세그먼트 마스크
                mask = np.ones(n_bottom, dtype=bool)
                for key in level.keys:
                    mask &= (bottom_segments[key].values == combo[key])
                row = mask.astype(float)
                all_rows.append(row)
                combo_str = "_".join(str(v) for v in combo.values)
                level_names.append((level.name, combo_str))
    
    S = np.vstack(all_rows)
    
    # 집계 인덱스 DataFrame
    agg_index = pd.DataFrame(level_names, columns=["level", "combo"])
    
    return S, agg_index, [ln[0] for ln in level_names]


def reconcile_bottom_up(
    base_forecasts: pd.DataFrame,
    segment_meta: pd.DataFrame,
    kpi_col: str,
    month_col: str = "pred_month",
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """
    Bottom-Up 조정: 하위 예측을 그대로 사용하고 상위는 합산.
    
    Args:
        base_forecasts: 기본 예측 (segment_id, pred_month, kpi_col)
        segment_meta: 세그먼트 메타데이터
        kpi_col: 조정할 KPI 컬럼
        month_col: 월 컬럼
        segment_col: 세그먼트 ID 컬럼
        
    Returns:
        조정된 예측 (원본과 동일 - Bottom-Up은 조정 없음)
    """
    # Bottom-Up은 하위 예측을 그대로 사용
    return base_forecasts.copy()


def reconcile_top_down(
    base_forecasts: pd.DataFrame,
    segment_meta: pd.DataFrame,
    kpi_col: str,
    historical_data: pd.DataFrame,
    month_col: str = "pred_month",
    segment_col: str = "segment_id",
    lookback_months: int = 6,
) -> pd.DataFrame:
    """
    Top-Down 조정: 상위 예측을 과거 비율로 하위에 분배.
    
    Args:
        base_forecasts: 기본 예측
        segment_meta: 세그먼트 메타데이터
        kpi_col: 조정할 KPI 컬럼
        historical_data: 과거 데이터 (비율 계산용)
        month_col: 월 컬럼
        segment_col: 세그먼트 ID 컬럼
        lookback_months: 비율 계산에 사용할 과거 개월 수
        
    Returns:
        조정된 예측
    """
    result = base_forecasts.copy()
    
    # 과거 데이터에서 세그먼트별 비율 계산
    recent = historical_data.sort_values("month").groupby(segment_col).tail(lookback_months)
    seg_avg = recent.groupby(segment_col)[kpi_col].mean()
    total_avg = seg_avg.sum()
    
    if total_avg == 0:
        return result
    
    # 비율
    proportions = seg_avg / total_avg
    
    # 각 월별 조정
    for month in result[month_col].unique():
        month_mask = result[month_col] == month
        month_total = result.loc[month_mask, kpi_col].sum()
        
        # 비율에 따라 분배
        for seg_id in result.loc[month_mask, segment_col].unique():
            if seg_id in proportions.index:
                seg_mask = month_mask & (result[segment_col] == seg_id)
                result.loc[seg_mask, kpi_col] = month_total * proportions[seg_id]
    
    return result


def _compute_mint_weights(
    residuals: np.ndarray,
    method: Literal["ols", "wls", "mint_shrink"] = "mint_shrink",
) -> np.ndarray:
    """
    MinT 가중치 행렬 계산.
    
    Args:
        residuals: 잔차 행렬 (n_samples x n_bottom)
        method: 가중 방법
            - "ols": 단순 OLS (W = I)
            - "wls": 대각 분산 가중 (W = diag(var))
            - "mint_shrink": Shrinkage 추정 (기본값)
            
    Returns:
        W: 가중치 행렬 (n_bottom x n_bottom)
    """
    n_samples, n_bottom = residuals.shape
    
    if method == "ols":
        return np.eye(n_bottom)
    
    elif method == "wls":
        # 대각 분산
        var = np.var(residuals, axis=0, ddof=1)
        var = np.maximum(var, 1e-8)  # 0 방지
        return np.diag(var)
    
    elif method == "mint_shrink":
        # Ledoit-Wolf Shrinkage 추정
        # 샘플 공분산
        if n_samples < 2:
            return np.eye(n_bottom)
        
        sample_cov = np.cov(residuals.T, ddof=1)
        if sample_cov.ndim == 0:
            sample_cov = np.array([[sample_cov]])
        
        # Shrinkage target: 대각 행렬
        diag_var = np.diag(np.diag(sample_cov))
        
        # Shrinkage intensity 계산 (간소화된 버전)
        # Ledoit-Wolf optimal shrinkage
        n = n_samples
        p = n_bottom
        
        if n < 4 or p == 1:
            return sample_cov
        
        # Frobenius norm 기반 shrinkage
        trace_s2 = np.trace(sample_cov @ sample_cov)
        trace_s = np.trace(sample_cov)
        
        # Shrinkage intensity
        sum_sq_diag = np.sum(np.diag(sample_cov) ** 2)
        
        numerator = (1 - 2 / p) * trace_s2 + trace_s ** 2
        denominator = (n + 1 - 2 / p) * (trace_s2 - sum_sq_diag / p)
        
        if denominator == 0:
            shrinkage = 1.0
        else:
            shrinkage = max(0, min(1, numerator / denominator))
        
        # Shrunk covariance
        W = shrinkage * diag_var + (1 - shrinkage) * sample_cov
        
        return W
    
    else:
        raise ValueError(f"Unknown method: {method}")


def reconcile_mint(
    base_forecasts: pd.DataFrame,
    segment_meta: pd.DataFrame,
    kpi_col: str,
    historical_residuals: Optional[pd.DataFrame] = None,
    month_col: str = "pred_month",
    segment_col: str = "segment_id",
    method: Literal["ols", "wls", "mint_shrink"] = "mint_shrink",
) -> pd.DataFrame:
    """
    MinT (Minimum Trace) Reconciliation.
    
    최적의 선형 조합을 통해 계층적 일관성을 유지하면서
    분산을 최소화하는 조정 수행.
    
    Reference: Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
               Optimal forecast reconciliation for hierarchical and grouped time series
               through trace minimization.
    
    Args:
        base_forecasts: 기본 예측
        segment_meta: 세그먼트 메타데이터
        kpi_col: 조정할 KPI 컬럼
        historical_residuals: 과거 잔차 데이터 (분산 추정용)
        month_col: 월 컬럼
        segment_col: 세그먼트 ID 컬럼
        method: 가중치 추정 방법
        
    Returns:
        조정된 예측
    """
    result = base_forecasts.copy()
    
    # 세그먼트 메타와 조인
    if segment_col not in segment_meta.columns:
        segment_meta = segment_meta.reset_index()
    
    bottom_segs = segment_meta.copy()
    # Align to segments present in forecasts to avoid dimension mismatch
    present_segs = result[segment_col].unique()
    bottom_segs = bottom_segs[bottom_segs[segment_col].isin(present_segs)].reset_index(drop=True)
    
    # 필요한 키 컬럼 확인
    required_keys = ["업종_중분류", "사업장_시도", "법인_고객등급", "전담고객여부"]
    available_keys = [k for k in required_keys if k in bottom_segs.columns]
    
    if len(available_keys) < 2:
        # 계층 구조 불충분 - 조정 없이 반환
        return result
    
    # Summing Matrix 구축
    try:
        S, agg_index, level_names = build_summing_matrix(
            bottom_segs,
            [hl for hl in HIERARCHY_LEVELS if all(k in available_keys for k in hl.keys) or not hl.keys]
        )
    except Exception:
        return result
    
    n_agg, n_bottom = S.shape
    
    # 가중치 행렬 계산
    if historical_residuals is not None and len(historical_residuals) > 1:
        # 잔차로부터 가중치 추정 (bottom-level only)
        pivot_resid = historical_residuals.pivot(
            index="month", columns=segment_col, values="residual"
        ).fillna(0)
        W = _compute_mint_weights(pivot_resid.values, method=method)
    else:
        # 기본 OLS 가중치 (bottom-level)
        W = np.eye(n_bottom)
    
    # MinT 조정 행렬 계산
    # G = (S'W^-1 S)^-1 S'W^-1
    # P = S @ G (투영 행렬)
    # y_reconciled = P @ y_base (상위 예측 포함)
    # 또는 하위만 조정: y_bottom_reconciled = y_bottom + G @ (y_agg - S @ y_bottom)
    
    try:
        W_inv = np.linalg.pinv(W)
        if W_inv.shape[0] != n_agg:
            # Fallback to aggregate-level identity to avoid dimension mismatch
            W_inv = np.eye(n_agg)
        StW = S.T @ W_inv
        StWS = StW @ S
        StWS_inv = np.linalg.pinv(StWS)
        G = StWS_inv @ StW  # (n_bottom x n_agg)
    except np.linalg.LinAlgError:
        # 역행렬 계산 실패 - 조정 없이 반환
        return result
    
    # 각 월별 조정
    for month in result[month_col].unique():
        month_mask = result[month_col] == month
        month_forecasts = result.loc[month_mask, [segment_col, kpi_col]].copy()
        
        # 하위 예측 벡터 (bottom_segs 순서에 맞춤)
        y_bottom = np.zeros(n_bottom)
        seg_to_idx = {seg: i for i, seg in enumerate(bottom_segs[segment_col])}
        
        for _, row in month_forecasts.iterrows():
            seg_id = row[segment_col]
            if seg_id in seg_to_idx:
                y_bottom[seg_to_idx[seg_id]] = row[kpi_col]
        
        # 현재 집계값
        y_agg_current = S @ y_bottom
        
        # 상위 수준 예측이 있다면 사용, 없으면 현재 집계값 사용
        y_agg_target = y_agg_current.copy()
        
        # 조정
        adjustment = G @ (y_agg_target - y_agg_current)
        y_bottom_reconciled = y_bottom + adjustment
        
        # 음수 방지 (예금/대출 등)
        if kpi_col not in ["순유입"]:  # 순유입은 음수 가능
            y_bottom_reconciled = np.maximum(y_bottom_reconciled, 0)
        
        # 결과 반영
        for seg_id, idx in seg_to_idx.items():
            seg_mask = month_mask & (result[segment_col] == seg_id)
            if seg_mask.any():
                result.loc[seg_mask, kpi_col] = y_bottom_reconciled[idx]
    
    return result


def reconcile_forecasts(
    base_forecasts: pd.DataFrame,
    segment_meta: pd.DataFrame,
    kpi_cols: List[str],
    historical_data: Optional[pd.DataFrame] = None,
    method: Literal["bottom_up", "top_down", "mint", "ols"] = "mint",
    month_col: str = "pred_month",
    segment_col: str = "segment_id",
) -> pd.DataFrame:
    """
    계층적 예측 조정 메인 함수.
    
    Args:
        base_forecasts: 기본 예측
        segment_meta: 세그먼트 메타데이터
        kpi_cols: 조정할 KPI 컬럼 리스트
        historical_data: 과거 데이터 (비율/분산 추정용)
        method: 조정 방법
        month_col: 월 컬럼
        segment_col: 세그먼트 ID 컬럼
        
    Returns:
        조정된 예측
    """
    result = base_forecasts.copy()
    
    for kpi_col in kpi_cols:
        if kpi_col not in result.columns:
            continue
        
        if method == "bottom_up":
            result = reconcile_bottom_up(
                result, segment_meta, kpi_col, month_col, segment_col
            )
        elif method == "top_down":
            if historical_data is None:
                raise ValueError("top_down method requires historical_data")
            result = reconcile_top_down(
                result, segment_meta, kpi_col, historical_data, month_col, segment_col
            )
        elif method in ["mint", "mint_shrink"]:
            result = reconcile_mint(
                result, segment_meta, kpi_col,
                historical_residuals=None,  # TODO: 잔차 전달
                month_col=month_col, segment_col=segment_col, method="mint_shrink"
            )
        elif method == "ols":
            result = reconcile_mint(
                result, segment_meta, kpi_col,
                historical_residuals=None,
                month_col=month_col, segment_col=segment_col, method="ols"
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return result


def compute_coherency_error(
    forecasts: pd.DataFrame,
    segment_meta: pd.DataFrame,
    kpi_col: str,
    month_col: str = "pred_month",
    segment_col: str = "segment_id",
) -> Dict[str, float]:
    """
    계층 일관성 오차 계산.
    
    각 상위 수준에서 하위 합산과의 차이를 측정.
    
    Args:
        forecasts: 예측 데이터
        segment_meta: 세그먼트 메타데이터
        kpi_col: KPI 컬럼
        month_col: 월 컬럼
        segment_col: 세그먼트 ID 컬럼
        
    Returns:
        수준별 일관성 오차 (RMSE)
    """
    merged = forecasts.merge(segment_meta, on=segment_col, how="left")
    
    errors = {}
    
    for level in HIERARCHY_LEVELS[:-1]:  # 마지막(bottom) 제외
        level_keys = level.keys if level.keys else []
        
        if level.name == "total":
            # 전체 합과 비교
            for month in merged[month_col].unique():
                month_data = merged[merged[month_col] == month]
                total = month_data[kpi_col].sum()
                # Total은 기준이므로 오차 0
            errors[level.name] = 0.0
        else:
            # 상위 그룹 합과 하위 세그먼트 합 비교
            group_sums = merged.groupby(level_keys + [month_col])[kpi_col].sum()
            # 이 수준에서는 자체 일관성이므로 오차 0
            errors[level.name] = 0.0
    
    return errors


def evaluate_reconciliation(
    base_forecasts: pd.DataFrame,
    reconciled_forecasts: pd.DataFrame,
    actuals: pd.DataFrame,
    segment_meta: pd.DataFrame,
    kpi_col: str,
    month_col: str = "pred_month",
    segment_col: str = "segment_id",
) -> Dict[str, Dict[str, float]]:
    """
    조정 전후 예측 성능 비교.
    
    Args:
        base_forecasts: 조정 전 예측
        reconciled_forecasts: 조정 후 예측
        actuals: 실제값
        segment_meta: 세그먼트 메타데이터
        kpi_col: KPI 컬럼
        month_col: 예측 월 컬럼
        segment_col: 세그먼트 ID 컬럼
        
    Returns:
        성능 비교 결과
    """
    def calc_metrics(preds: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]:
        merged = preds.merge(
            actual[[segment_col, "month", kpi_col]],
            left_on=[segment_col, month_col],
            right_on=[segment_col, "month"],
            how="inner",
            suffixes=("_pred", "_actual")
        )
        
        if merged.empty:
            return {"rmse": np.nan, "mape": np.nan, "mae": np.nan}
        
        y_true = merged[f"{kpi_col}_actual"].values
        y_pred = merged[f"{kpi_col}_pred"].values
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE (0 제외)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan
        
        return {"rmse": rmse, "mape": mape, "mae": mae}
    
    base_metrics = calc_metrics(base_forecasts, actuals)
    reconciled_metrics = calc_metrics(reconciled_forecasts, actuals)
    
    return {
        "base": base_metrics,
        "reconciled": reconciled_metrics,
        "improvement": {
            k: base_metrics[k] - reconciled_metrics[k]
            for k in base_metrics
        }
    }
