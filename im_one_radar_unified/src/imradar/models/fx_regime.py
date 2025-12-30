from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class FXRegimeResult:
    """FX 레짐 분석 결과"""
    fx_monthly: pd.DataFrame  # month-level FX features
    regime_labels: pd.Series  # regime label per month
    regime_stats: Optional[Dict[str, Dict]] = None  # 레짐별 통계
    transition_matrix: Optional[pd.DataFrame] = None  # 레짐 전이 확률


def compute_technical_indicators(
    df: pd.DataFrame,
    price_col: str,
    short_period: int = 5,
    long_period: int = 20,
) -> pd.DataFrame:
    """
    기술적 지표 계산 (RSI, MACD, Bollinger Bands 등)
    
    Args:
        df: Input DataFrame
        price_col: 가격 컬럼명
        short_period: 단기 이동평균 기간
        long_period: 장기 이동평균 기간
    
    Returns:
        기술적 지표가 추가된 DataFrame
    """
    out = df.copy()
    price = out[price_col].astype(float)
    
    # 이동평균
    out["fx_sma_short"] = price.rolling(window=short_period, min_periods=1).mean()
    out["fx_sma_long"] = price.rolling(window=long_period, min_periods=1).mean()
    out["fx_ema_short"] = price.ewm(span=short_period, min_periods=1).mean()
    out["fx_ema_long"] = price.ewm(span=long_period, min_periods=1).mean()
    
    # MACD
    ema12 = price.ewm(span=12, min_periods=1).mean()
    ema26 = price.ewm(span=26, min_periods=1).mean()
    out["fx_macd"] = ema12 - ema26
    out["fx_macd_signal"] = out["fx_macd"].ewm(span=9, min_periods=1).mean()
    out["fx_macd_hist"] = out["fx_macd"] - out["fx_macd_signal"]
    
    # RSI (Relative Strength Index)
    delta = price.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    out["fx_rsi"] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma20 = price.rolling(window=20, min_periods=1).mean()
    std20 = price.rolling(window=20, min_periods=1).std()
    out["fx_bb_upper"] = sma20 + (std20 * 2)
    out["fx_bb_lower"] = sma20 - (std20 * 2)
    out["fx_bb_pct"] = (price - out["fx_bb_lower"]) / (out["fx_bb_upper"] - out["fx_bb_lower"] + 1e-9)
    
    # 변동성 지표
    out["fx_volatility_20"] = price.pct_change().rolling(window=20, min_periods=1).std() * np.sqrt(252)
    out["fx_atr"] = (price.rolling(window=14, min_periods=1).max() - 
                     price.rolling(window=14, min_periods=1).min())
    
    return out


def compute_volatility_regime(
    series: pd.Series,
    window: int = 20,
    n_regimes: int = 3,
) -> pd.Series:
    """
    변동성 기반 레짐 분류
    
    Args:
        series: 가격 시계열
        window: 변동성 계산 윈도우
        n_regimes: 레짐 수 (3: Low/Medium/High)
    
    Returns:
        레짐 레이블 Series
    """
    returns = series.pct_change()
    volatility = returns.rolling(window=window, min_periods=3).std()
    
    # 분위수 기반 레짐 분류
    quantiles = volatility.quantile([0.33, 0.67])
    
    def classify(v):
        if pd.isna(v):
            return "Unknown"
        if v <= quantiles.iloc[0]:
            return "LowVol"
        elif v <= quantiles.iloc[1]:
            return "MedVol"
        else:
            return "HighVol"
    
    return volatility.apply(classify)


def compute_trend_regime(
    series: pd.Series,
    short_window: int = 5,
    long_window: int = 20,
    threshold: float = 0.02,
) -> pd.Series:
    """
    트렌드 기반 레짐 분류
    
    Args:
        series: 가격 시계열
        short_window: 단기 이동평균 윈도우
        long_window: 장기 이동평균 윈도우
        threshold: 트렌드 강도 임계값
    
    Returns:
        트렌드 레짐 레이블 Series
    """
    short_ma = series.rolling(window=short_window, min_periods=1).mean()
    long_ma = series.rolling(window=long_window, min_periods=1).mean()
    
    # 트렌드 강도: (단기 - 장기) / 장기
    trend_strength = (short_ma - long_ma) / (long_ma + 1e-9)
    
    def classify(t):
        if pd.isna(t):
            return "Unknown"
        if t > threshold:
            return "Uptrend"
        elif t < -threshold:
            return "Downtrend"
        else:
            return "Range"
    
    return trend_strength.apply(classify)


def compute_regime_transition_matrix(
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    레짐 전이 확률 행렬 계산
    
    Args:
        regimes: 레짐 레이블 시계열
    
    Returns:
        전이 확률 행렬 DataFrame
    """
    # 레짐 값 추출
    unique_regimes = regimes.dropna().unique()
    
    # 전이 카운트
    transitions = pd.DataFrame(0, index=unique_regimes, columns=unique_regimes)
    
    prev = None
    for curr in regimes.dropna():
        if prev is not None:
            transitions.loc[prev, curr] += 1
        prev = curr
    
    # 확률로 변환
    row_sums = transitions.sum(axis=1)
    transition_probs = transitions.div(row_sums.replace(0, 1), axis=0)
    
    return transition_probs


def compute_regime_stats(
    df: pd.DataFrame,
    regime_col: str,
    value_cols: List[str],
) -> Dict[str, Dict]:
    """
    레짐별 통계 계산
    
    Args:
        df: Input DataFrame
        regime_col: 레짐 컬럼명
        value_cols: 통계 계산할 값 컬럼들
    
    Returns:
        레짐별 통계 딕셔너리
    """
    stats = {}
    
    for regime in df[regime_col].unique():
        if pd.isna(regime):
            continue
        
        regime_data = df[df[regime_col] == regime]
        regime_stats = {
            "count": len(regime_data),
            "pct": len(regime_data) / len(df) * 100,
        }
        
        for col in value_cols:
            if col in regime_data.columns:
                regime_stats[f"{col}_mean"] = float(regime_data[col].mean())
                regime_stats[f"{col}_std"] = float(regime_data[col].std())
                regime_stats[f"{col}_median"] = float(regime_data[col].median())
        
        stats[str(regime)] = regime_stats
    
    return stats


def compute_fx_regime(
    fx_df: pd.DataFrame,
    month_col: str = "month",
    fx_level_col: str = "usdkrw_avg",
    fx_eom_col: Optional[str] = "usdkrw_eom",
    include_technical: bool = True,
    include_stats: bool = True,
) -> FXRegimeResult:
    """
    Enhanced FX regime analysis with multiple indicators.
    
    Features:
      - Trend regime (Uptrend/Downtrend/Range)
      - Volatility regime (Low/Medium/High)
      - Technical indicators (RSI, MACD, Bollinger Bands)
      - Regime transition probabilities
      - Regime statistics
    
    Args:
        fx_df: FX 데이터 DataFrame
        month_col: 월 컬럼명
        fx_level_col: 환율 레벨 컬럼명
        fx_eom_col: 월말 환율 컬럼명 (선택)
        include_technical: 기술적 지표 포함 여부
        include_stats: 레짐 통계 포함 여부
    
    Returns:
        FXRegimeResult with enhanced analysis
    """
    d = fx_df.sort_values(month_col).copy()
    d["fx_level"] = d[fx_level_col].astype(float)
    
    # 기본 피처
    d["fx_mom"] = d["fx_level"].diff(1)
    d["fx_mom_pct"] = d["fx_level"].pct_change(1)
    d["fx_vol3"] = d["fx_level"].diff(1).rolling(3, min_periods=1).std()
    d["fx_vol6"] = d["fx_level"].diff(1).rolling(6, min_periods=1).std()
    d["fx_vol12"] = d["fx_level"].diff(1).rolling(12, min_periods=1).std()
    
    # 기술적 지표
    if include_technical:
        d = compute_technical_indicators(d, "fx_level")
    
    # 트렌드 레짐
    d["fx_trend_regime"] = compute_trend_regime(d["fx_level"])
    
    # 변동성 레짐
    d["fx_vol_regime"] = compute_volatility_regime(d["fx_level"])
    
    # 복합 레짐 (트렌드 + 변동성)
    d["fx_regime"] = d["fx_trend_regime"] + "_" + d["fx_vol_regime"]
    
    # 레짐 전이 확률
    transition_matrix = compute_regime_transition_matrix(d["fx_regime"])
    
    # 레짐별 통계
    regime_stats = None
    if include_stats:
        value_cols = ["fx_level", "fx_mom_pct", "fx_vol3"]
        if "fx_rsi" in d.columns:
            value_cols.append("fx_rsi")
        regime_stats = compute_regime_stats(d, "fx_regime", value_cols)
    
    return FXRegimeResult(
        fx_monthly=d,
        regime_labels=d["fx_regime"],
        regime_stats=regime_stats,
        transition_matrix=transition_matrix,
    )


def get_current_regime_signal(result: FXRegimeResult) -> Dict:
    """
    현재 레짐 및 시그널 요약
    
    Args:
        result: FXRegimeResult instance
    
    Returns:
        현재 상태 요약 딕셔너리
    """
    df = result.fx_monthly
    latest = df.iloc[-1] if len(df) > 0 else None
    
    if latest is None:
        return {"status": "No data"}
    
    signal = {
        "current_regime": latest.get("fx_regime", "Unknown"),
        "trend": latest.get("fx_trend_regime", "Unknown"),
        "volatility": latest.get("fx_vol_regime", "Unknown"),
        "fx_level": float(latest.get("fx_level", 0)),
        "momentum_pct": float(latest.get("fx_mom_pct", 0)) * 100,
    }
    
    # 기술적 지표 시그널
    if "fx_rsi" in latest:
        rsi = float(latest["fx_rsi"])
        signal["rsi"] = rsi
        if rsi > 70:
            signal["rsi_signal"] = "Overbought"
        elif rsi < 30:
            signal["rsi_signal"] = "Oversold"
        else:
            signal["rsi_signal"] = "Neutral"
    
    if "fx_macd_hist" in latest:
        macd_hist = float(latest["fx_macd_hist"])
        signal["macd_signal"] = "Bullish" if macd_hist > 0 else "Bearish"
    
    if "fx_bb_pct" in latest:
        bb_pct = float(latest["fx_bb_pct"])
        signal["bollinger_position"] = f"{bb_pct:.1%}"
    
    return signal


def forecast_regime_probability(
    result: FXRegimeResult,
    current_regime: Optional[str] = None,
) -> Dict[str, float]:
    """
    다음 레짐 전이 확률 예측
    
    Args:
        result: FXRegimeResult instance
        current_regime: 현재 레짐 (None이면 최신 레짐 사용)
    
    Returns:
        다음 레짐별 확률 딕셔너리
    """
    if result.transition_matrix is None:
        return {}
    
    if current_regime is None:
        current_regime = result.regime_labels.iloc[-1]
    
    if current_regime not in result.transition_matrix.index:
        return {}
    
    probs = result.transition_matrix.loc[current_regime].to_dict()
    return {k: float(v) for k, v in probs.items()}
