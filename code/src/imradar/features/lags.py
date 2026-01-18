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


# ============== 거시경제 변수 관련 피처 ==============

def add_macro_interaction_features(
    df: pd.DataFrame,
    kpi_cols: Sequence[str],
    macro_cols: Sequence[str],
) -> pd.DataFrame:
    """
    KPI와 거시경제 변수 간의 상호작용 피처 생성
    
    Args:
        df: Input DataFrame
        kpi_cols: KPI 컬럼들 (예: 예금총잔액, 대출총잔액)
        macro_cols: 거시경제 변수 컬럼들 (예: ESI, fed_rate)
    
    Returns:
        상호작용 피처가 추가된 DataFrame
    """
    out = df.copy()
    eps = 1e-9
    
    # 주요 상호작용 정의
    interactions = [
        # (KPI, Macro, 설명)
        ("대출총잔액", "fed_rate", "loan_fedrate"),  # 대출과 금리
        ("예금총잔액", "ESI", "deposit_esi"),  # 예금과 경기심리
        ("FX총액", "usd_krw_mean", "fx_usdkrw"),  # 외환과 환율
        ("수출실적", "wti_price", "export_oil"),  # 수출과 유가
    ]
    
    for kpi, macro, name in interactions:
        if kpi in out.columns and macro in out.columns:
            # 표준화된 상호작용 (zscore 기반)
            kpi_norm = (out[kpi] - out[kpi].mean()) / (out[kpi].std() + eps)
            macro_norm = (out[macro] - out[macro].mean()) / (out[macro].std() + eps)
            out[f"interaction__{name}"] = kpi_norm * macro_norm
    
    return out


def add_macro_regime_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    거시경제 레짐(상태) 피처 생성
    
    금리, 환율, 유가 등의 현재 상태를 분류
    
    Args:
        df: Input DataFrame
    
    Returns:
        레짐 피처가 추가된 DataFrame
    """
    out = df.copy()
    
    # 금리 레짐: 고금리(>4%), 중금리(2-4%), 저금리(<2%)
    if "fed_rate" in out.columns:
        conditions = [
            out["fed_rate"] >= 4.0,
            out["fed_rate"] >= 2.0,
            out["fed_rate"] < 2.0,
        ]
        choices = [2, 1, 0]  # high, medium, low
        out["fed_rate_regime"] = np.select(conditions, choices, default=0)
    
    # 환율 레짐: 약세(>1300), 보통(1200-1300), 강세(<1200)
    if "usd_krw_mean" in out.columns:
        conditions = [
            out["usd_krw_mean"] >= 1300,
            out["usd_krw_mean"] >= 1200,
            out["usd_krw_mean"] < 1200,
        ]
        choices = [2, 1, 0]  # weak KRW, normal, strong KRW
        out["fx_regime"] = np.select(conditions, choices, default=1)
    
    # 유가 레짐: 고유가(>90), 중유가(70-90), 저유가(<70)
    if "wti_price" in out.columns:
        conditions = [
            out["wti_price"] >= 90,
            out["wti_price"] >= 70,
            out["wti_price"] < 70,
        ]
        choices = [2, 1, 0]  # high, medium, low
        out["oil_regime"] = np.select(conditions, choices, default=1)
    
    # ESI 레짐: 확장(>100), 중립(95-100), 위축(<95)
    if "ESI" in out.columns:
        conditions = [
            out["ESI"] >= 100,
            out["ESI"] >= 95,
            out["ESI"] < 95,
        ]
        choices = [2, 1, 0]  # expansion, neutral, contraction
        out["esi_regime"] = np.select(conditions, choices, default=1)
    
    return out


def add_macro_trend_features(
    df: pd.DataFrame,
    macro_cols: Optional[Sequence[str]] = None,
    window: int = 3,
) -> pd.DataFrame:
    """
    거시경제 변수의 추세 방향 피처 생성
    
    Args:
        df: Input DataFrame
        macro_cols: 거시경제 변수 컬럼들
        window: 추세 계산 윈도우
    
    Returns:
        추세 피처가 추가된 DataFrame
    """
    out = df.copy()
    
    if macro_cols is None:
        macro_cols = ["ESI", "fed_rate", "usd_krw_mean", "wti_price"]
    
    # 월별 유니크 데이터로 추세 계산
    for col in macro_cols:
        if col not in out.columns:
            continue
        
        # 월별 첫 번째 값 추출
        monthly_val = out.groupby("month")[col].first().reset_index()
        monthly_val = monthly_val.sort_values("month")
        
        # 추세 방향: 현재 값 vs window 이전 값
        prev_val = monthly_val[col].shift(window)
        trend = np.sign(monthly_val[col] - prev_val)
        monthly_val[f"{col}__trend"] = trend
        
        # 연속 상승/하락 개월 수
        def count_consecutive(series):
            result = []
            count = 0
            prev = 0
            for val in series:
                if pd.isna(val):
                    result.append(np.nan)
                    prev = 0
                    count = 0
                elif val == prev and val != 0:
                    count += 1
                    result.append(count * val)
                else:
                    count = 1
                    result.append(val)
                prev = val
            return result
        
        monthly_val[f"{col}__streak"] = count_consecutive(monthly_val[f"{col}__trend"].values)
        
        # 원본 데이터에 조인
        out = out.merge(
            monthly_val[["month", f"{col}__trend", f"{col}__streak"]],
            on="month", how="left"
        )
    
    return out


def add_external_shock_features(
    df: pd.DataFrame,
    threshold_std: float = 2.0,
) -> pd.DataFrame:
    """
    외부 충격 이벤트 피처 생성
    
    거시경제 변수가 평소 대비 크게 변동한 경우를 감지
    
    Args:
        df: Input DataFrame
        threshold_std: 충격 감지 임계값 (표준편차 배수)
    
    Returns:
        충격 피처가 추가된 DataFrame
    """
    out = df.copy()
    
    shock_cols = [
        ("usd_krw_mom_pct", "fx_shock"),
        ("wti_mom_pct", "oil_shock"),
        ("fed_rate_mom_diff", "rate_shock"),
    ]
    
    for col, shock_name in shock_cols:
        if col not in out.columns:
            continue
        
        # 월별 변동의 표준편차 계산
        monthly = out.groupby("month")[col].first().reset_index()
        std_val = monthly[col].std()
        mean_val = monthly[col].mean()
        
        # 충격 여부 판단
        monthly[shock_name] = np.where(
            np.abs(monthly[col] - mean_val) > threshold_std * std_val,
            np.sign(monthly[col] - mean_val),
            0
        )
        
        out = out.merge(monthly[["month", shock_name]], on="month", how="left")
    
    return out

