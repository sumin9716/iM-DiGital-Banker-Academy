"""
원인 분석 엔진 - KPI 변동 원인을 자연어로 요약

Features:
- 피처 기여도 기반 원인 분석
- 거시경제 요인 연동
- 비즈니스 맥락 반영
- 자연어 설명 생성
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd


# 피처명 → 비즈니스 용어 매핑
FEATURE_DISPLAY_NAMES = {
    # KPI 관련
    "예금총잔액": "예금잔액",
    "대출총잔액": "대출잔액",
    "순유입": "순유입",
    "카드총사용": "카드사용",
    "디지털거래금액": "디지털거래",
    "FX총액": "외환거래",
    
    # Lag 피처
    "lag1": "전월",
    "lag2": "2개월전",
    "lag3": "3개월전",
    "lag6": "6개월전",
    "lag12": "전년동월",
    
    # Rolling 피처
    "roll3_mean": "3개월평균",
    "roll6_mean": "6개월평균",
    "roll12_mean": "연평균",
    
    # 변화율
    "pct_chg": "변화율",
    "mom": "전월비",
    "yoy": "전년비",
    
    # 거시경제
    "fed_rate": "미국기준금리",
    "fed_diff": "금리차",
    "usd_krw": "달러환율",
    "usd_krw_vol": "환율변동성",
    "brent": "유가",
    "esi": "경기전망지수",
    "kasi": "한국신뢰지수",
    "apt_price": "아파트가격",
    "apt_vol": "부동산거래량",
    "jeonse": "전세지수",
    "trade_balance": "무역수지",
    
    # 세그먼트
    "연령대": "연령대",
    "직업군": "직업군",
    "거래활성도": "활성도",
    "자산규모": "자산등급",
}


@dataclass
class CauseAnalysis:
    """원인 분석 결과"""
    kpi: str
    direction: str  # "증가", "감소", "급등", "급락"
    magnitude: float  # 변화 크기 (%)
    primary_causes: List[str]  # 주요 원인
    secondary_causes: List[str]  # 부가 원인
    macro_factors: List[str]  # 거시경제 요인
    summary: str  # 종합 요약
    confidence: float  # 분석 신뢰도


def format_feature_name(feature: str) -> str:
    """피처명을 비즈니스 용어로 변환"""
    result = feature
    
    # log1p 접두사 제거
    result = result.replace("log1p_", "")
    
    # 언더스코어 처리
    parts = result.split("_")
    formatted_parts = []
    
    for part in parts:
        # 매핑된 이름 사용
        if part in FEATURE_DISPLAY_NAMES:
            formatted_parts.append(FEATURE_DISPLAY_NAMES[part])
        elif part.isdigit():
            continue  # 숫자는 건너뜀
        else:
            formatted_parts.append(part)
    
    return " ".join(formatted_parts).strip()


def interpret_contribution(
    feature: str,
    contribution: float,
    feature_value: Optional[float] = None,
) -> str:
    """
    피처 기여도를 자연어로 해석
    
    Args:
        feature: 피처명
        contribution: 기여도 값
        feature_value: 피처 실제값 (선택)
        
    Returns:
        해석 문장
    """
    display_name = format_feature_name(feature)
    direction = "상승" if contribution > 0 else "하락"
    strength = abs(contribution)
    
    # 강도 표현
    if strength > 0.5:
        intensity = "크게"
    elif strength > 0.2:
        intensity = "상당히"
    elif strength > 0.1:
        intensity = ""
    else:
        intensity = "소폭"
    
    # Lag/Rolling 피처 해석
    if "lag" in feature.lower() or "roll" in feature.lower():
        if "lag1" in feature:
            time_ref = "전월 추세"
        elif "lag3" in feature or "roll3" in feature:
            time_ref = "최근 3개월 추세"
        elif "lag6" in feature or "roll6" in feature:
            time_ref = "중기 추세"
        elif "lag12" in feature:
            time_ref = "전년 동기 대비"
        else:
            time_ref = "과거 추세"
        
        return f"{display_name}의 {time_ref}가 {direction} 방향으로 {intensity} 기여"
    
    # 거시경제 피처 해석
    macro_keywords = ["fed", "usd", "krw", "brent", "esi", "kasi", "apt", "jeonse", "trade"]
    if any(kw in feature.lower() for kw in macro_keywords):
        return f"거시요인({display_name}) {direction} 영향"
    
    # 변화율 피처
    if "pct_chg" in feature or "mom" in feature or "yoy" in feature:
        return f"{display_name} 변화율이 {direction} 기여"
    
    # 일반 피처
    return f"{display_name}이(가) {intensity} {direction} 영향"


def analyze_cause(
    kpi: str,
    residual_pct: float,
    feature_contributions: Dict[str, float],
    segment_data: Optional[pd.Series] = None,
    macro_data: Optional[Dict] = None,
    top_k: int = 5,
) -> CauseAnalysis:
    """
    KPI 변동 원인 분석
    
    Args:
        kpi: KPI명
        residual_pct: 잔차 비율 (예측 대비 실제 차이)
        feature_contributions: 피처별 기여도
        segment_data: 세그먼트 데이터
        macro_data: 거시경제 데이터
        top_k: 상위 원인 개수
        
    Returns:
        CauseAnalysis 결과
    """
    # 방향 및 크기 결정
    magnitude = abs(residual_pct * 100)
    if residual_pct > 0.2:
        direction = "급등"
    elif residual_pct > 0.05:
        direction = "증가"
    elif residual_pct < -0.2:
        direction = "급락"
    elif residual_pct < -0.05:
        direction = "감소"
    else:
        direction = "변동"
    
    # 기여도 정렬
    sorted_contribs = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    # 주요 원인 (동일 방향 기여)
    if residual_pct > 0:
        primary_contribs = [(f, v) for f, v in sorted_contribs if v > 0][:top_k]
    else:
        primary_contribs = [(f, v) for f, v in sorted_contribs if v < 0][:top_k]
    
    primary_causes = [
        interpret_contribution(f, v) 
        for f, v in primary_contribs
    ]
    
    # 부가 원인 (반대 방향 기여)
    if residual_pct > 0:
        secondary_contribs = [(f, v) for f, v in sorted_contribs if v < 0][:3]
    else:
        secondary_contribs = [(f, v) for f, v in sorted_contribs if v > 0][:3]
    
    secondary_causes = [
        interpret_contribution(f, v)
        for f, v in secondary_contribs
    ]
    
    # 거시경제 요인 분리
    macro_keywords = ["fed", "usd", "krw", "brent", "esi", "kasi", "apt", "jeonse", "trade"]
    macro_factors = []
    for feature, contrib in sorted_contribs[:10]:
        if any(kw in feature.lower() for kw in macro_keywords):
            macro_factors.append(interpret_contribution(feature, contrib))
    macro_factors = macro_factors[:3]
    
    # 종합 요약 생성
    summary = generate_cause_summary(
        kpi=kpi,
        direction=direction,
        magnitude=magnitude,
        primary_causes=primary_causes,
        macro_factors=macro_factors,
        segment_data=segment_data,
    )
    
    # 신뢰도 (기여도 집중도 기반)
    if sorted_contribs:
        top_contrib_sum = sum(abs(v) for _, v in sorted_contribs[:3])
        total_contrib = sum(abs(v) for _, v in sorted_contribs)
        confidence = min(top_contrib_sum / (total_contrib + 1e-9), 1.0)
    else:
        confidence = 0.5
    
    return CauseAnalysis(
        kpi=kpi,
        direction=direction,
        magnitude=magnitude,
        primary_causes=primary_causes,
        secondary_causes=secondary_causes,
        macro_factors=macro_factors,
        summary=summary,
        confidence=confidence,
    )


def generate_cause_summary(
    kpi: str,
    direction: str,
    magnitude: float,
    primary_causes: List[str],
    macro_factors: List[str],
    segment_data: Optional[pd.Series] = None,
) -> str:
    """
    원인 분석 종합 요약 생성
    
    Args:
        kpi: KPI명
        direction: 변동 방향
        magnitude: 변동 크기
        primary_causes: 주요 원인
        macro_factors: 거시경제 요인
        segment_data: 세그먼트 데이터
        
    Returns:
        종합 요약 문장
    """
    parts = []
    
    # 도입부
    parts.append(f"{kpi}이(가) 예측 대비 {magnitude:.1f}% {direction}하였습니다.")
    
    # 주요 원인
    if primary_causes:
        if len(primary_causes) == 1:
            parts.append(f"주요 원인: {primary_causes[0]}.")
        else:
            top_causes = ", ".join(primary_causes[:2])
            parts.append(f"주요 원인: {top_causes}.")
    
    # 거시경제 요인
    if macro_factors:
        parts.append(f"외부 요인: {macro_factors[0]}.")
    
    # 세그먼트 맥락
    if segment_data is not None:
        age_group = segment_data.get("연령대", "")
        job_group = segment_data.get("직업군", "")
        activity = segment_data.get("거래활성도", "")
        
        context_parts = []
        if age_group:
            context_parts.append(age_group)
        if job_group:
            context_parts.append(job_group)
        if activity:
            context_parts.append(f"{activity} 고객")
        
        if context_parts:
            segment_desc = " ".join(context_parts)
            
            # 세그먼트 맥락 해석
            if direction in ["급락", "감소"]:
                if activity == "저활성":
                    parts.append(f"({segment_desc}: 관계 약화 징후 주의)")
                elif "60대" in age_group or "70대" in age_group:
                    parts.append(f"({segment_desc}: 시니어 자금이동 가능성)")
                elif "자영업" in job_group or "소상공인" in job_group:
                    parts.append(f"({segment_desc}: 사업자금 변동 가능성)")
            elif direction in ["급등", "증가"]:
                if "VIP" in str(segment_data.get("자산규모", "")):
                    parts.append(f"({segment_desc}: 자산이동 또는 거래확대)")
    
    return " ".join(parts)


def generate_watchlist_driver_summary(
    row: pd.Series,
    feature_contributions: Optional[Dict[str, float]] = None,
) -> str:
    """
    워치리스트 항목의 드라이버 요약
    
    Args:
        row: 워치리스트 행
        feature_contributions: 피처 기여도
        
    Returns:
        드라이버 요약 문자열
    """
    drivers = []
    
    # 알림 유형 기반
    alert_type = str(row.get("alert_type", ""))
    severity = str(row.get("severity", ""))
    residual_pct = float(row.get("residual_pct", 0) or 0)
    
    if alert_type == "DROP":
        drivers.append(f"예측 대비 {abs(residual_pct)*100:.1f}% 하락")
    elif alert_type == "SPIKE":
        drivers.append(f"예측 대비 {abs(residual_pct)*100:.1f}% 상승")
    elif alert_type == "INFLECTION":
        drivers.append("추세 전환점 감지")
    
    # 심각도 기반
    if severity == "CRITICAL":
        drivers.append("위험 수준 심각")
    elif severity == "HIGH":
        drivers.append("주의 필요")
    
    # 피처 기여도 기반
    if feature_contributions:
        sorted_contribs = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        for feature, contrib in sorted_contribs:
            drivers.append(interpret_contribution(feature, contrib))
    
    # 기타 컬럼 기반 정보
    risk_factors = str(row.get("risk_factors", ""))
    if risk_factors and risk_factors != "nan":
        factors = risk_factors.split(";")[:2]
        drivers.extend([f.strip() for f in factors if f.strip()])
    
    if not drivers:
        return "상세 분석 필요"
    
    return " | ".join(drivers[:4])


def generate_risk_factor_summary(row: pd.Series) -> str:
    """
    리스크 요인 요약 생성
    
    Args:
        row: 데이터 행
        
    Returns:
        리스크 요인 요약 문자열
    """
    factors = []
    
    # 리스크 점수
    risk_score = float(row.get("risk_score", 0) or 0)
    if risk_score >= 80:
        factors.append("🔴 고위험")
    elif risk_score >= 60:
        factors.append("🟠 중위험")
    elif risk_score >= 40:
        factors.append("🟡 관찰필요")
    
    # 순유입 상태
    net_inflow = float(row.get("순유입", 0) or 0)
    if net_inflow < -1000:
        factors.append("대규모 순유출")
    elif net_inflow < -500:
        factors.append("순유출 중")
    elif net_inflow < 0:
        factors.append("소폭 순유출")
    
    # 잔차 상태
    residual_pct = float(row.get("residual_pct", 0) or 0)
    if residual_pct < -0.3:
        factors.append("예측 대비 급락")
    elif residual_pct < -0.1:
        factors.append("예측 하회")
    
    # 변동성
    volatility = str(row.get("volatility_regime", ""))
    if "HighVol" in volatility:
        factors.append("환율 변동성↑")
    
    # 환율 레짐
    regime = str(row.get("regime", row.get("current_regime", "")))
    if "Uptrend" in regime:
        factors.append("원화약세")
    
    if not factors:
        return "특이사항 없음"
    
    return " | ".join(factors)


def enrich_watchlist_with_drivers(
    watchlist: pd.DataFrame,
    contributions_dict: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    워치리스트에 드라이버 정보 추가
    
    Args:
        watchlist: 워치리스트 DataFrame
        contributions_dict: segment_id → 피처 기여도 딕셔너리
        
    Returns:
        드라이버 정보가 추가된 DataFrame
    """
    result = watchlist.copy()
    
    # 드라이버 요약 컬럼 추가
    driver_summaries = []
    for _, row in result.iterrows():
        segment_id = str(row.get("segment_id", ""))
        contribs = contributions_dict.get(segment_id) if contributions_dict else None
        summary = generate_watchlist_driver_summary(row, contribs)
        driver_summaries.append(summary)
    
    result["driver_summary"] = driver_summaries
    
    # 리스크 요인 요약
    result["risk_summary"] = result.apply(generate_risk_factor_summary, axis=1)
    
    return result
