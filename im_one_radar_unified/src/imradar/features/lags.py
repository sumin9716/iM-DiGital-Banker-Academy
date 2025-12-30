from __future__ import annotations

from typing import List, Sequence, Optional
import numpy as np
import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
    lags: Sequence[int] = (1, 2, 3, 6, 12),
) -> pd.DataFrame:
    """
    시계열 Lag 피처 생성
    
    Args:
        df: Input DataFrame
        group_col: 그룹 컬럼 (세그먼트 ID)
        time_col: 시간 컬럼
        value_cols: Lag 피처를 생성할 값 컬럼들
        lags: Lag 기간 리스트
    
    Returns:
        Lag 피처가 추가된 DataFrame
    """
    out = df.sort_values([group_col, time_col]).copy()
    for c in value_cols:
        if c not in out.columns:
            continue
        for l in lags:
            out[f"{c}__lag{l}"] = out.groupby(group_col)[c].shift(l)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
    windows: Sequence[int] = (3, 6),
    include_min_max: bool = False,
) -> pd.DataFrame:
    """
    Rolling 통계 피처 생성 (mean, std, 선택적으로 min/max)
    
    Args:
        df: Input DataFrame
        group_col: 그룹 컬럼
        time_col: 시간 컬럼
        value_cols: 피처 생성할 값 컬럼들
        windows: Rolling 윈도우 크기 리스트
        include_min_max: min/max 피처 포함 여부
    
    Returns:
        Rolling 피처가 추가된 DataFrame
    """
    out = df.sort_values([group_col, time_col]).copy()
    for c in value_cols:
        if c not in out.columns:
            continue
        g = out.groupby(group_col)[c]
        for w in windows:
            shifted = g.shift(1)
            rolling = shifted.rolling(w, min_periods=1)
            
            out[f"{c}__roll{w}_mean"] = rolling.mean().reset_index(level=0, drop=True)
            out[f"{c}__roll{w}_std"] = rolling.std().reset_index(level=0, drop=True)
            
            if include_min_max:
                out[f"{c}__roll{w}_min"] = rolling.min().reset_index(level=0, drop=True)
                out[f"{c}__roll{w}_max"] = rolling.max().reset_index(level=0, drop=True)
    return out


def add_ewm_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
    spans: Sequence[int] = (3, 6, 12),
) -> pd.DataFrame:
    """
    지수가중이동평균 (EWM) 피처 생성
    
    Args:
        df: Input DataFrame
        group_col: 그룹 컬럼
        time_col: 시간 컬럼
        value_cols: 피처 생성할 값 컬럼들
        spans: EWM span 리스트
    
    Returns:
        EWM 피처가 추가된 DataFrame
    """
    out = df.sort_values([group_col, time_col]).copy()
    for c in value_cols:
        if c not in out.columns:
            continue
        g = out.groupby(group_col)[c]
        for s in spans:
            shifted = g.shift(1)
            out[f"{c}__ewm{s}"] = shifted.ewm(span=s, min_periods=1).mean().reset_index(level=0, drop=True)
    return out


def add_change_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
    include_acceleration: bool = False,
) -> pd.DataFrame:
    """
    변화율 피처 생성 (MoM, YoY, 선택적으로 가속도)
    
    Args:
        df: Input DataFrame
        group_col: 그룹 컬럼
        time_col: 시간 컬럼
        value_cols: 피처 생성할 값 컬럼들
        include_acceleration: 가속도(2차 차분) 포함 여부
    
    Returns:
        변화율 피처가 추가된 DataFrame
    """
    out = df.sort_values([group_col, time_col]).copy()
    for c in value_cols:
        if c not in out.columns:
            continue
        g = out.groupby(group_col)[c]
        
        # Month-over-Month 변화율
        out[f"{c}__mom"] = g.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        
        # Year-over-Year 변화율
        out[f"{c}__yoy"] = g.pct_change(12, fill_method=None).replace([np.inf, -np.inf], np.nan)
        
        # 3개월 변화율
        out[f"{c}__3m_change"] = g.pct_change(3, fill_method=None).replace([np.inf, -np.inf], np.nan)
        
        if include_acceleration:
            # 가속도 (2차 차분): 변화율의 변화
            mom = out[f"{c}__mom"]
            out[f"{c}__accel"] = out.groupby(group_col)[f"{c}__mom"].diff()
    
    return out


def add_zscore_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
    window: int = 12,
) -> pd.DataFrame:
    """
    Rolling Z-score 피처 생성 (이상치 탐지에 유용)
    
    Args:
        df: Input DataFrame
        group_col: 그룹 컬럼
        time_col: 시간 컬럼
        value_cols: 피처 생성할 값 컬럼들
        window: Rolling 윈도우 크기
    
    Returns:
        Z-score 피처가 추가된 DataFrame
    """
    out = df.sort_values([group_col, time_col]).copy()
    eps = 1e-9
    
    for c in value_cols:
        if c not in out.columns:
            continue
        g = out.groupby(group_col)[c]
        shifted = g.shift(1)
        rolling_mean = shifted.rolling(window, min_periods=3).mean().reset_index(level=0, drop=True)
        rolling_std = shifted.rolling(window, min_periods=3).std().reset_index(level=0, drop=True)
        
        out[f"{c}__zscore{window}"] = (out[c] - rolling_mean) / (rolling_std + eps)
    
    return out


def add_ratio_features(
    df: pd.DataFrame,
    numerator_cols: Sequence[str],
    denominator_cols: Sequence[str],
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    비율 피처 생성
    
    Args:
        df: Input DataFrame
        numerator_cols: 분자 컬럼들
        denominator_cols: 분모 컬럼들
        feature_names: 생성될 피처 이름들 (None이면 자동 생성)
    
    Returns:
        비율 피처가 추가된 DataFrame
    """
    out = df.copy()
    eps = 1e-9
    
    if feature_names is None:
        feature_names = [f"{n}_to_{d}_ratio" for n, d in zip(numerator_cols, denominator_cols)]
    
    for n, d, name in zip(numerator_cols, denominator_cols, feature_names):
        if n in out.columns and d in out.columns:
            out[name] = out[n] / (out[d] + eps)
            # 무한대 처리
            out[name] = out[name].replace([np.inf, -np.inf], np.nan)
    
    return out


def add_momentum_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
    short_window: int = 3,
    long_window: int = 12,
) -> pd.DataFrame:
    """
    모멘텀 피처 생성 (단기 vs 장기 이동평균 비교)
    
    Args:
        df: Input DataFrame
        group_col: 그룹 컬럼
        time_col: 시간 컬럼
        value_cols: 피처 생성할 값 컬럼들
        short_window: 단기 이동평균 윈도우
        long_window: 장기 이동평균 윈도우
    
    Returns:
        모멘텀 피처가 추가된 DataFrame
    """
    out = df.sort_values([group_col, time_col]).copy()
    eps = 1e-9
    
    for c in value_cols:
        if c not in out.columns:
            continue
        g = out.groupby(group_col)[c]
        shifted = g.shift(1)
        
        short_ma = shifted.rolling(short_window, min_periods=1).mean().reset_index(level=0, drop=True)
        long_ma = shifted.rolling(long_window, min_periods=1).mean().reset_index(level=0, drop=True)
        
        # 모멘텀: 단기 이동평균 / 장기 이동평균 - 1
        out[f"{c}__momentum_{short_window}_{long_window}"] = (short_ma / (long_ma + eps)) - 1
        
        # 골든크로스/데드크로스 시그널
        out[f"{c}__ma_signal"] = np.where(short_ma > long_ma, 1, -1)
    
    return out


def add_seasonality_features(
    df: pd.DataFrame,
    time_col: str,
) -> pd.DataFrame:
    """
    계절성 피처 생성 (월, 분기 등)
    
    Args:
        df: Input DataFrame
        time_col: 시간 컬럼
    
    Returns:
        계절성 피처가 추가된 DataFrame
    """
    out = df.copy()
    
    if time_col in out.columns:
        dt = pd.to_datetime(out[time_col])
        
        # 기본 시간 피처
        out["month_num"] = dt.dt.month
        out["quarter"] = dt.dt.quarter
        out["year"] = dt.dt.year
        
        # 순환 인코딩 (사인/코사인)
        out["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
        out["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
        out["quarter_sin"] = np.sin(2 * np.pi * dt.dt.quarter / 4)
        out["quarter_cos"] = np.cos(2 * np.pi * dt.dt.quarter / 4)
        
        # 분기말/연말 플래그
        out["is_quarter_end"] = (dt.dt.month % 3 == 0).astype(int)
        out["is_year_end"] = (dt.dt.month == 12).astype(int)
    
    return out


def build_all_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_cols: Sequence[str],
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    rolling_windows: Sequence[int] = (3, 6),
    ewm_spans: Sequence[int] = (3, 6, 12),
    include_advanced: bool = True,
) -> pd.DataFrame:
    """
    모든 피처를 한번에 생성하는 통합 함수
    
    Args:
        df: Input DataFrame
        group_col: 그룹 컬럼
        time_col: 시간 컬럼
        value_cols: 피처 생성할 값 컬럼들
        lags: Lag 기간 리스트
        rolling_windows: Rolling 윈도우 리스트
        ewm_spans: EWM span 리스트
        include_advanced: 고급 피처 포함 여부
    
    Returns:
        모든 피처가 추가된 DataFrame
    """
    out = df.copy()
    
    # 기본 피처
    out = add_lag_features(out, group_col, time_col, value_cols, lags)
    out = add_rolling_features(out, group_col, time_col, value_cols, rolling_windows)
    out = add_change_features(out, group_col, time_col, value_cols)
    
    if include_advanced:
        # 고급 피처
        out = add_ewm_features(out, group_col, time_col, value_cols, ewm_spans)
        out = add_zscore_features(out, group_col, time_col, value_cols)
        out = add_momentum_features(out, group_col, time_col, value_cols)
        out = add_seasonality_features(out, time_col)
    
    return out
