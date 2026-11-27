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


# FX Opportunity & Risk Score

@dataclass
class FXSegmentScore:
    """FX 세그먼트별 기회/리스크 점수"""
    segment_id: str
    opportunity_score: float  # 0~100
    risk_score: float  # 0~100
    fx_size_score: float
    momentum_score: float
    cross_sell_score: float
    regime_score: float
    risk_factors: List[str]
    recommended_actions: List[str]


def compute_fx_opportunity_score(
    segment_row: pd.Series,
    fx_regime: str,
    fx_volatility: str,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    FX Opportunity Score 계산 (0~100)
    
    구성요소:
    1. FX 규모 (최근 3~6개월 평균)
    2. FX 성장 모멘텀 (MoM/YoY)
    3. 확장 여지 (교차판매 신호)
       - 한도소진율/대출 변화 (운영자금 수요)
       - 순유입 변화 (현금흐름)
       - 디지털 거래 비중 (셀프 FX 유도 가능성)
    4. 레짐 적합도 (변동성 확대 시 리스크관리 수요 가중)
    
    Args:
        segment_row: 세그먼트 데이터 (Series)
        fx_regime: 현재 환율 레짐
        fx_volatility: 변동성 레짐
        weights: 각 요소별 가중치 (기본값 제공)
    
    Returns:
        (opportunity_score, component_scores)
    """
    if weights is None:
        weights = {
            "fx_size": 0.30,
            "momentum": 0.25,
            "cross_sell": 0.25,
            "regime": 0.20,
        }
    
    scores = {}
    
    # 1. FX 규모 점수 (로그 스케일 정규화)
    fx_total = float(segment_row.get("FX총액", 0) or 0)
    # 대체 컬럼: log1p_FX총액__roll3_mean → 원래 스케일로 변환
    fx_roll3_log = float(segment_row.get("log1p_FX총액__roll3_mean", 0) or 0)
    if fx_roll3_log > 0:
        fx_ma3 = np.expm1(fx_roll3_log)  # log1p 역변환
    else:
        fx_ma3 = fx_total
    
    if fx_ma3 > 0:
        # 로그 스케일로 정규화 (억 단위: 0.1억 = 25점, 1억 = 50점, 10억 = 75점, 100억 = 100점)
        log_fx = np.log10(max(fx_ma3 * 1e8, 1))  # 억원 → 원 변환
        scores["fx_size"] = min(100, max(0, (log_fx - 6) * 25 + 50))  # 6 = log10(1백만)
    else:
        scores["fx_size"] = 0
    
    # 2. 모멘텀 점수 - 대체 컬럼 사용
    # log1p_FX총액__mom (MoM 변화량), log1p_FX총액__yoy (YoY 변화량)
    fx_mom_raw = float(segment_row.get("log1p_FX총액__mom", 0) or 0)
    fx_yoy_raw = float(segment_row.get("log1p_FX총액__yoy", 0) or 0)
    
    # log 스케일 변화량을 퍼센트로 변환 (대략 변환)
    fx_mom = fx_mom_raw * 100  # log 변화를 % 근사
    fx_yoy = fx_yoy_raw * 100
    
    # MoM: -50% ~ +50% → 0 ~ 100
    mom_score = min(100, max(0, (fx_mom + 50) * 1))
    # YoY: -100% ~ +100% → 0 ~ 100
    yoy_score = min(100, max(0, (fx_yoy + 100) * 0.5))
    scores["momentum"] = (mom_score * 0.6 + yoy_score * 0.4)
    
    # 3. 교차판매 기회 점수
    util_rate = float(segment_row.get("한도소진율", 0) or 0)
    net_inflow = float(segment_row.get("순유입", 0) or 0)
    digital_share = float(segment_row.get("디지털비중", 0) or 0)
    # 대체 컬럼: log1p_대출총잔액__mom
    loan_growth_raw = float(segment_row.get("log1p_대출총잔액__mom", 0) or 0)
    loan_growth = loan_growth_raw * 100  # log 변화를 % 근사
    
    # 한도소진율 높음 = 운영자금 필요 = 기회
    util_score = min(100, util_rate * 100)
    # 순유입 양호 = 자금력 있음
    inflow_score = 50 + min(50, max(-50, net_inflow / 1e8 * 10))
    # 디지털 비중 낮음 = 전환 유도 기회
    digital_opp = max(0, (1 - digital_share) * 100)
    # 대출 성장 = 사업 확장 신호
    loan_score = min(100, max(0, (loan_growth + 20) * 2.5))
    
    scores["cross_sell"] = (util_score * 0.3 + inflow_score * 0.2 + 
                            digital_opp * 0.25 + loan_score * 0.25)
    
    # 4. 레짐 적합도 점수
    regime_scores = {
        ("Uptrend", "HighVol"): 90,   # 변동성 확대 + 약세 = 헤지 수요 최고
        ("Uptrend", "MedVol"): 75,
        ("Uptrend", "LowVol"): 60,
        ("Downtrend", "HighVol"): 80,  # 강세 전환 + 변동성 = 환전 타이밍 기회
        ("Downtrend", "MedVol"): 65,
        ("Downtrend", "LowVol"): 50,
        ("Range", "HighVol"): 70,
        ("Range", "MedVol"): 55,
        ("Range", "LowVol"): 40,
    }
    
    # 트렌드와 변동성 추출
    trend = "Range"
    vol = "MedVol"
    if "Uptrend" in fx_regime:
        trend = "Uptrend"
    elif "Downtrend" in fx_regime:
        trend = "Downtrend"
    
    if "HighVol" in fx_volatility or "HighVol" in fx_regime:
        vol = "HighVol"
    elif "LowVol" in fx_volatility or "LowVol" in fx_regime:
        vol = "LowVol"
    
    scores["regime"] = regime_scores.get((trend, vol), 50)
    
    # 가중 합계
    total_score = sum(scores[k] * weights[k] for k in weights)
    
    return total_score, scores


def compute_fx_risk_score(
    segment_row: pd.Series,
    fx_regime: str,
    fx_volatility: str,
    alert_type: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float], List[str]]:
    """
    FX Risk Score 계산 (0~100)
    
    구성요소:
    1. FX 급감/변곡 탐지 (예측 대비 하회 + 변화점)
    2. 스트레스 신호 결합
       - 순유입 악화 + 한도소진율 급등 + 예금 감소
    3. 레짐 연동 (High Vol에서 FX 급감은 리스크 가중)
    
    Args:
        segment_row: 세그먼트 데이터 (Series)
        fx_regime: 현재 환율 레짐
        fx_volatility: 변동성 레짐
        alert_type: 알림 유형 (DROP, SPIKE, INFLECTION 등)
        weights: 각 요소별 가중치
    
    Returns:
        (risk_score, component_scores, risk_factors)
    """
    if weights is None:
        weights = {
            "fx_decline": 0.35,
            "stress_signal": 0.35,
            "regime_risk": 0.30,
        }
    
    scores = {}
    risk_factors = []
    
    # 1. FX 급감/변곡 점수 - 대체 컬럼 사용
    fx_mom_raw = float(segment_row.get("log1p_FX총액__mom", 0) or 0)
    fx_yoy_raw = float(segment_row.get("log1p_FX총액__yoy", 0) or 0)
    fx_mom = fx_mom_raw * 100  # log 변화를 % 근사
    fx_yoy = fx_yoy_raw * 100
    
    decline_score = 0
    if fx_mom < -20:
        decline_score += 50
        risk_factors.append(f"FX 월간 급감 ({fx_mom:.1f}%)")
    elif fx_mom < -10:
        decline_score += 30
        risk_factors.append(f"FX 월간 감소 ({fx_mom:.1f}%)")
    
    if fx_yoy < -30:
        decline_score += 50
        risk_factors.append(f"FX 연간 급감 ({fx_yoy:.1f}%)")
    elif fx_yoy < -15:
        decline_score += 25
    
    # 알림 유형 가중
    if alert_type == "DROP":
        decline_score = min(100, decline_score + 30)
        risk_factors.append("예측 대비 급락 경보")
    elif alert_type == "INFLECTION":
        decline_score = min(100, decline_score + 20)
        risk_factors.append("추세 변곡점 감지")
    
    scores["fx_decline"] = min(100, decline_score)
    
    # 2. 스트레스 신호 점수
    stress_score = 0
    
    # 순유입 악화
    net_inflow = float(segment_row.get("순유입", 0) or 0)
    # 대체: slog1p_순유입__mom 사용
    net_mom_raw = float(segment_row.get("slog1p_순유입__mom", 0) or 0)
    net_mom = net_mom_raw * 100  # log 변화를 % 근사
    if net_inflow < 0:
        stress_score += 25
        if net_mom < -20:
            stress_score += 15
            risk_factors.append("순유입 급락")
    
    # 한도소진율 급등
    util_rate = float(segment_row.get("한도소진율", 0) or 0)
    if util_rate > 0.8:
        stress_score += 30
        risk_factors.append(f"한도소진율 위험 ({util_rate:.0%})")
    elif util_rate > 0.6:
        stress_score += 15
    
    # 예금 감소 - 대체 컬럼 사용
    deposit_mom_raw = float(segment_row.get("log1p_예금총잔액__mom", 0) or 0)
    deposit_mom = deposit_mom_raw * 100  # log 변화를 % 근사
    if deposit_mom < -10:
        stress_score += 20
        risk_factors.append(f"예금 감소 ({deposit_mom:.1f}%)")
    
    # 복합 스트레스 (여러 신호 동시 발생 시 가중)
    if len([f for f in risk_factors if "급" in f or "위험" in f]) >= 2:
        stress_score = min(100, stress_score + 20)
        risk_factors.append("복합 스트레스 신호")
    
    scores["stress_signal"] = min(100, stress_score)
    
    # 3. 레짐 리스크 점수
    regime_risk_scores = {
        ("Uptrend", "HighVol"): 90,   # 약세 + 변동성 = 수입기업 리스크 최고
        ("Uptrend", "MedVol"): 70,
        ("Uptrend", "LowVol"): 50,
        ("Downtrend", "HighVol"): 60,  # 강세지만 변동성 높음
        ("Downtrend", "MedVol"): 40,
        ("Downtrend", "LowVol"): 25,
        ("Range", "HighVol"): 55,
        ("Range", "MedVol"): 35,
        ("Range", "LowVol"): 20,
    }
    
    trend = "Range"
    vol = "MedVol"
    if "Uptrend" in fx_regime:
        trend = "Uptrend"
    elif "Downtrend" in fx_regime:
        trend = "Downtrend"
    
    if "HighVol" in fx_volatility or "HighVol" in fx_regime:
        vol = "HighVol"
        if trend == "Uptrend":
            risk_factors.append("환율 약세 + 고변동성 국면")
    elif "LowVol" in fx_volatility or "LowVol" in fx_regime:
        vol = "LowVol"
    
    scores["regime_risk"] = regime_risk_scores.get((trend, vol), 35)
    
    # 가중 합계
    total_score = sum(scores[k] * weights[k] for k in weights)
    
    return total_score, scores, risk_factors


def compute_fx_segment_scores(
    panel: pd.DataFrame,
    fx_result: FXRegimeResult,
    segment_col: str = "segment_id",
    month_col: str = "month",
    alerts_df: Optional[pd.DataFrame] = None,
    min_fx_threshold: float = 0,
) -> pd.DataFrame:
    """
    모든 FX 활성 세그먼트에 대해 Opportunity/Risk Score 계산
    
    Args:
        panel: 패널 데이터
        fx_result: FX 레짐 분석 결과
        segment_col: 세그먼트 컬럼
        month_col: 월 컬럼
        alerts_df: 알림 데이터 (선택)
        min_fx_threshold: 최소 FX 금액 필터
    
    Returns:
        FX 세그먼트별 점수 DataFrame
    """
    # 최신 월 데이터
    latest_month = panel[month_col].max()
    latest_data = panel[panel[month_col] == latest_month].copy()
    
    # FX 활성 세그먼트 필터
    if "FX총액" in latest_data.columns:
        fx_active = latest_data[latest_data["FX총액"] > min_fx_threshold]
    else:
        fx_active = latest_data
    
    if fx_active.empty:
        return pd.DataFrame()
    
    # 현재 레짐 정보
    current_regime = fx_result.regime_labels.iloc[-1] if len(fx_result.regime_labels) > 0 else "Range_MedVol"
    current_vol = "MedVol"
    if "HighVol" in current_regime:
        current_vol = "HighVol"
    elif "LowVol" in current_regime:
        current_vol = "LowVol"
    
    results = []
    
    for _, row in fx_active.iterrows():
        seg_id = row[segment_col]
        
        # 알림 유형 확인
        alert_type = None
        if alerts_df is not None and not alerts_df.empty:
            seg_alerts = alerts_df[
                (alerts_df[segment_col] == seg_id) & 
                (alerts_df.get("kpi", "") == "FX총액")
            ]
            if not seg_alerts.empty:
                alert_type = seg_alerts.iloc[0].get("alert_type", None)
        
        # 점수 계산
        opp_score, opp_components = compute_fx_opportunity_score(
            row, current_regime, current_vol
        )
        risk_score, risk_components, risk_factors = compute_fx_risk_score(
            row, current_regime, current_vol, alert_type
        )
        
        # 추천 액션 생성
        actions = generate_fx_recommended_actions(
            opp_score, risk_score, current_regime, current_vol, 
            row, risk_factors
        )
        
        results.append({
            segment_col: seg_id,
            "month": latest_month,
            "FX총액": row.get("FX총액", 0),
            "opportunity_score": round(opp_score, 1),
            "risk_score": round(risk_score, 1),
            "fx_size_score": round(opp_components.get("fx_size", 0), 1),
            "momentum_score": round(opp_components.get("momentum", 0), 1),
            "cross_sell_score": round(opp_components.get("cross_sell", 0), 1),
            "regime_opp_score": round(opp_components.get("regime", 0), 1),
            "fx_decline_score": round(risk_components.get("fx_decline", 0), 1),
            "stress_score": round(risk_components.get("stress_signal", 0), 1),
            "regime_risk_score": round(risk_components.get("regime_risk", 0), 1),
            "current_regime": current_regime,
            "risk_factors": "; ".join(risk_factors[:3]) if risk_factors else "",
            "recommended_actions": "; ".join(actions[:3]) if actions else "",
            "priority_grade": get_fx_priority_grade(opp_score, risk_score),
        })
    
    result_df = pd.DataFrame(results)
    
    # 우선순위 정렬 (기회 높고 리스크 관리 필요한 순)
    result_df["combined_priority"] = (
        result_df["opportunity_score"] * 0.4 + 
        result_df["risk_score"] * 0.6
    )
    result_df = result_df.sort_values("combined_priority", ascending=False)
    
    return result_df.reset_index(drop=True)


def generate_fx_recommended_actions(
    opp_score: float,
    risk_score: float,
    regime: str,
    volatility: str,
    segment_row: pd.Series,
    risk_factors: List[str],
) -> List[str]:
    """
    레짐 + 세그먼트 상태에 따른 FX 추천 액션 생성 (개선된 버전)
    
    Args:
        opp_score: 기회 점수
        risk_score: 리스크 점수
        regime: 현재 레짐
        volatility: 변동성 레짐
        segment_row: 세그먼트 데이터
        risk_factors: 리스크 요인 리스트
    
    Returns:
        추천 액션 리스트 (최대 3개)
    """
    actions = []
    
    is_uptrend = "Uptrend" in regime
    is_downtrend = "Downtrend" in regime
    is_high_vol = "HighVol" in volatility or "HighVol" in regime
    
    digital_share = float(segment_row.get("디지털비중", 0) or 0)
    fx_total = float(segment_row.get("FX총액", 0) or 0)
    age_group = str(segment_row.get("연령대", ""))
    job_group = str(segment_row.get("직업군", ""))
    net_inflow = float(segment_row.get("순유입", 0) or 0)
    
    # 1. 긴급 리스크 대응
    if risk_score >= 80:
        actions.append("🔴 긴급: 담당 RM 즉시 컨택 - 환리스크 긴급 점검 필요")
    
    # 2. 변동성 확대 국면
    if is_high_vol:
        if fx_total > 10000:  # 대규모 FX 고객
            actions.append("🛡️ 환헤지 솔루션(선물환/옵션) 맞춤 설계 제안")
        else:
            actions.append("🛡️ 환리스크 점검 및 헤지 전략 기초 상담 제안")
        actions.append("📊 환율 변동성 모니터링 알림 서비스 등록 안내")
    
    # 3. 원화 약세 국면 (수입형 기업 부담↑)
    if is_uptrend:
        if risk_score > 60:
            if job_group in ["자영업", "소상공인", "제조업"]:
                actions.append("💰 수입대금 결제용 운영자금 대출 한도 점검 및 리프레시")
            else:
                actions.append("💰 운영자금 여유 점검 - 환율 부담 증가 대비")
        
        if fx_total > 5000:
            actions.append("🔄 환전 타이밍 분할 전략 수립 컨설팅 (평균환율 최적화)")
        else:
            actions.append("💱 중소기업 환율우대 프로그램 안내")
    
    # 4. 원화 강세 국면 (수출형 기업 기회)
    if is_downtrend:
        actions.append("📈 수출대금 조기 환전/정산 적기 타이밍 안내")
        
        if opp_score > 70:
            actions.append("� FX 거래 확대 고객 인센티브 프로그램 제안")
        elif fx_total > 0:
            actions.append("� 무역금융 상품(포페이팅/팩토링) 안내")
    
    # 5. 디지털 채널 유도
    if fx_total > 0 and digital_share < 0.4:
        if age_group in ["20대", "30대", "40대"]:
            actions.append("📱 기업뱅킹 셀프 FX 채널 온보딩 (환율우대+편의성)")
        else:
            actions.append("📱 FX 거래 디지털 채널 1:1 안내 서비스 제공")
    
    # 6. 순유입 상태 기반
    if net_inflow < -500 and "순유입" in str(risk_factors):
        actions.append("⚠️ 자금흐름 이상 감지 - RM 우선 컨택 및 원인 파악 필요")
    
    # 7. 한도소진율 위험
    if "한도소진율" in str(risk_factors):
        actions.append("🔴 여신한도 소진 임박 - 긴급 한도 점검 및 증액 검토")
    
    # 8. 기회 중심 제안 (리스크 낮고 기회 높을 때)
    if risk_score < 30 and opp_score > 60 and not actions:
        if fx_total > 10000:
            actions.append("🌟 VIP FX 고객 전용 프리미엄 서비스 안내")
        else:
            actions.append("📢 FX 거래 활성화 캠페인 타겟 등록")
    
    # 중복 제거 및 상위 3개 반환
    seen = set()
    unique_actions = []
    for a in actions:
        key = a[:20]  # 앞 20자로 중복 판단
        if key not in seen:
            seen.add(key)
            unique_actions.append(a)
    
    # 액션이 없으면 기본 메시지
    if not unique_actions:
        unique_actions.append("📋 정기 FX 거래 현황 리뷰 및 니즈 파악 상담")
    
    return unique_actions[:3]


def get_fx_priority_grade(opp_score: float, risk_score: float) -> str:
    """
    기회/리스크 점수 기반 우선순위 등급
    
    Returns:
        A (최우선), B (우선), C (일반), D (모니터링)
    """
    if risk_score >= 70:
        return "A"  # 리스크 높음 = 최우선 관리
    elif opp_score >= 70 and risk_score >= 40:
        return "A"  # 기회 높고 리스크도 있음
    elif opp_score >= 70:
        return "B"  # 기회 높음
    elif risk_score >= 50:
        return "B"  # 리스크 중간
    elif opp_score >= 50:
        return "C"  # 기회 중간
    else:
        return "D"  # 모니터링


def load_fx_data_from_external(external_dir: str) -> Optional[pd.DataFrame]:
    """
    외부 데이터 디렉토리에서 USD/KRW 환율 데이터 로드
    
    Args:
        external_dir: 외부 데이터 디렉토리 경로
    
    Returns:
        FX DataFrame with month, usdkrw_avg columns
    """
    import os
    
    # 가능한 파일명들
    possible_files = [
        "usd_krw_raw_daily.csv",
        "USD_KRW.csv",
        "usdkrw.csv",
    ]
    
    for fname in possible_files:
        fpath = os.path.join(external_dir, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, encoding="utf-8-sig")
                # 컬럼명 표준화
                if "날짜" in df.columns:
                    df["date"] = pd.to_datetime(df["날짜"])
                elif "Date" in df.columns:
                    df["date"] = pd.to_datetime(df["Date"])
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                else:
                    continue
                
                # 환율 컬럼 찾기
                fx_col = None
                for col in ["종가", "Close", "close", "usdkrw", "USDKRW", "usdkrw_avg", "rate", "Rate", "환율"]:
                    if col in df.columns:
                        fx_col = col
                        break
                
                if fx_col is None:
                    continue
                
                # 월별 집계
                df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
                monthly = df.groupby("month").agg({
                    fx_col: ["mean", "std", "min", "max", "last"]
                }).reset_index()
                monthly.columns = ["month", "usdkrw_avg", "usdkrw_std", "usdkrw_min", "usdkrw_max", "usdkrw_eom"]
                
                return monthly
            except Exception:
                continue
    
    return None


def compute_fx_regime_from_external(
    panel: pd.DataFrame,
    external_dir: str,
) -> Optional[FXRegimeResult]:
    """
    외부 데이터 디렉토리에서 FX 데이터를 로드하여 레짐 분석 수행
    
    Args:
        panel: 패널 데이터 (사용하지 않지만 호환성을 위해)
        external_dir: 외부 데이터 디렉토리 경로
    
    Returns:
        FXRegimeResult or None
    """
    fx_df = load_fx_data_from_external(external_dir)
    
    if fx_df is None or fx_df.empty:
        return None
    
    return compute_fx_regime(
        fx_df,
        month_col="month",
        fx_level_col="usdkrw_avg",
        fx_eom_col="usdkrw_eom",
        include_technical=True,
        include_stats=True,
    )
