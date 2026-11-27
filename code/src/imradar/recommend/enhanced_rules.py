"""
개선된 추천 규칙 엔진 - 다양한 액션 유형과 완화된 조건

Changes from rules.py:
1. 조건 완화 (더 많은 세그먼트 커버)
2. 새로운 액션 유형 추가 (GROWTH_OPPORTUNITY, CHURN_PREVENTION, CROSS_SELL 등)
3. 트렌드 기반 추천 추가
4. 세그먼트 특성 기반 맞춤 추천
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from imradar.config import RadarConfig


@dataclass
class EnhancedAction:
    """개선된 액션 데이터 클래스"""
    segment_id: str
    month: pd.Timestamp
    action_type: str
    title: str
    rationale: str
    cause_summary: str  # 원인 요약
    score: float
    urgency: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # RETENTION, GROWTH, RISK_MGMT, CROSS_SELL, DIGITAL


# 액션 타입별 메타데이터
ACTION_METADATA = {
    # 성장 기회
    "GROWTH_OPPORTUNITY": {"title": "📈 성장 기회 포착", "category": "GROWTH", "urgency": "HIGH"},
    "DEPOSIT_GROWTH": {"title": "💰 예금 유치 강화", "category": "GROWTH", "urgency": "MEDIUM"},
    "LOAN_EXPANSION": {"title": "💳 여신 확대 기회", "category": "GROWTH", "urgency": "MEDIUM"},
    "FX_EXPANSION": {"title": "💱 FX 거래 확대", "category": "GROWTH", "urgency": "MEDIUM"},
    "CARD_UPSELL": {"title": "💳 카드 이용 확대", "category": "GROWTH", "urgency": "LOW"},
    
    # 리텐션/이탈 방지
    "CHURN_PREVENTION": {"title": "⚠️ 이탈 방지 필요", "category": "RETENTION", "urgency": "CRITICAL"},
    "DEPOSIT_DEFENSE": {"title": "🛡️ 예금 유출 방어", "category": "RETENTION", "urgency": "HIGH"},
    "ENGAGEMENT_DROP": {"title": "📉 거래 활성화 필요", "category": "RETENTION", "urgency": "MEDIUM"},
    
    # 리스크 관리
    "LIQUIDITY_STRESS": {"title": "🚨 유동성 점검 필요", "category": "RISK_MGMT", "urgency": "CRITICAL"},
    "FX_RISK": {"title": "⚡ 환리스크 관리", "category": "RISK_MGMT", "urgency": "HIGH"},
    "CREDIT_RISK": {"title": "⚠️ 신용 리스크 점검", "category": "RISK_MGMT", "urgency": "HIGH"},
    
    # 교차판매
    "CROSS_SELL_LOAN": {"title": "🔗 대출 교차판매", "category": "CROSS_SELL", "urgency": "LOW"},
    "CROSS_SELL_CARD": {"title": "🔗 카드 교차판매", "category": "CROSS_SELL", "urgency": "LOW"},
    "CROSS_SELL_FX": {"title": "🔗 FX 서비스 제안", "category": "CROSS_SELL", "urgency": "LOW"},
    "CROSS_SELL_INVEST": {"title": "🔗 투자상품 제안", "category": "CROSS_SELL", "urgency": "LOW"},
    
    # 디지털 전환
    "DIGITAL_ONBOARD": {"title": "📱 디지털 전환 유도", "category": "DIGITAL", "urgency": "LOW"},
    "DIGITAL_PREMIUM": {"title": "🌟 디지털 VIP 혜택", "category": "DIGITAL", "urgency": "LOW"},
}


def _safe_float(val, default=0.0) -> float:
    """안전한 float 변환"""
    try:
        if pd.isna(val):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_pct_change(cur: float, future: float, min_base: float = 1.0, cap: float = 5.0) -> float:
    denom = max(abs(cur), min_base)
    pct = (future - cur) / denom
    return float(np.clip(pct, -cap, cap))


def _classify_tier(deposit: float) -> str:
    """자산 규모별 등급 분류"""
    if deposit >= 1e9:  # 10억 이상
        return "VVIP"
    elif deposit >= 1e8:  # 1억 이상
        return "VIP"
    elif deposit >= 5e7:  # 5천만 이상
        return "우수"
    else:
        return "일반"


def _compute_trend_score(
    current: float,
    pred: Optional[float],
    lag1: Optional[float] = None,
    lag2: Optional[float] = None,
) -> Tuple[float, str]:
    """
    추세 점수 및 방향 계산
    
    Returns:
        (trend_score, trend_direction)
        - trend_score: -1 ~ 1 (음수: 하락, 양수: 상승)
        - trend_direction: "급상승", "상승", "보합", "하락", "급하락"
    """
    scores = []
    
    # 과거 추세
    if lag1 is not None:
        chg = _safe_pct_change(float(lag1), float(current), cap=2.0)
        scores.append(chg)
    
    if lag2 is not None and lag1 is not None:
        chg = _safe_pct_change(float(lag2), float(lag1), cap=2.0)
        scores.append(chg * 0.5)  # 가중치 낮춤
    
    # 예측 추세
    if pred is not None:
        chg = _safe_pct_change(float(current), float(pred), cap=2.0)
        scores.append(chg * 1.5)  # 예측에 더 높은 가중치
    
    if not scores:
        return 0.0, "보합"
    
    trend_score = np.clip(np.mean(scores), -1, 1)
    
    if trend_score >= 0.15:
        direction = "급상승"
    elif trend_score >= 0.05:
        direction = "상승"
    elif trend_score <= -0.15:
        direction = "급하락"
    elif trend_score <= -0.05:
        direction = "하락"
    else:
        direction = "보합"
    
    return trend_score, direction


def generate_enhanced_actions(
    row: pd.Series,
    preds: Dict[str, float],
    alerts: Optional[pd.DataFrame] = None,
    fx_regime: str = "",
    top_k: int = 5,
    cfg: Optional[RadarConfig] = None,
) -> List[EnhancedAction]:
    """
    개선된 추천 액션 생성 - 다양한 유형과 완화된 조건
    
    Args:
        row: 세그먼트 데이터 (현재 월)
        preds: KPI별 예측값 딕셔너리
        alerts: 알림 데이터프레임
        fx_regime: FX 레짐 문자열
        top_k: 상위 N개 액션 반환
        cfg: RadarConfig
        
    Returns:
        EnhancedAction 리스트 (점수순 정렬)
    """
    if cfg is None:
        cfg = RadarConfig()
    
    segment_id = str(row.get("segment_id", ""))
    month = row.get("month", pd.Timestamp.now())
    if isinstance(month, str):
        month = pd.Timestamp(month)
    
    # 세그먼트 기본 정보
    grade = str(row.get("법인_고객등급", ""))
    dedicated = str(row.get("전담고객여부", ""))
    industry = str(row.get("업종_중분류", "미분류"))
    region = str(row.get("사업장_시도", "미정"))
    
    # 현재 KPI 값
    cur_deposit = _safe_float(row.get("예금총잔액", 0))
    cur_loan = _safe_float(row.get("대출총잔액", 0))
    cur_net = _safe_float(row.get("순유입", 0))
    cur_card = _safe_float(row.get("카드총사용", 0))
    cur_fx = _safe_float(row.get("FX총액", 0))
    
    # 파생 지표
    util_rate = _safe_float(row.get("한도소진율", 0))
    util_rate = min(max(util_rate, 0.0), 5.0)
    digital_share = _safe_float(row.get("디지털비중", 0))
    
    # Lag 값들 (추세 계산용)
    dep_lag1 = _safe_float(row.get("예금총잔액_lag1"))
    dep_lag2 = _safe_float(row.get("예금총잔액_lag2"))
    net_lag1 = _safe_float(row.get("순유입_lag1"))
    fx_lag1 = _safe_float(row.get("FX총액_lag1"))
    
    # 예측값
    pred_deposit = preds.get("예금총잔액")
    pred_loan = preds.get("대출총잔액")
    pred_net = preds.get("순유입")
    pred_fx = preds.get("FX총액")
    
    # 자산 등급
    tier = _classify_tier(cur_deposit)
    
    # 가중치 계산
    grade_w = float(cfg.grade_weights.get(grade, 1.0)) if hasattr(cfg, 'grade_weights') else 1.0
    dedicated_w = float(cfg.dedicated_weights.get(dedicated, 1.0)) if hasattr(cfg, 'dedicated_weights') else 1.0
    base_weight = grade_w * dedicated_w
    
    # 추세 계산
    dep_trend, dep_dir = _compute_trend_score(cur_deposit, pred_deposit, dep_lag1, dep_lag2)
    net_trend, net_dir = _compute_trend_score(cur_net, pred_net, net_lag1)
    fx_trend, fx_dir = _compute_trend_score(cur_fx, pred_fx, fx_lag1)
    
    # 알림 정보
    alert_severity = 0.0
    if alerts is not None and not alerts.empty:
        seg_alerts = alerts[alerts["segment_id"] == segment_id]
        if not seg_alerts.empty:
            worst = seg_alerts.sort_values("residual_pct").iloc[0]
            alert_severity = min(2.0, max(0.0, abs(float(worst["residual_pct"]))))
    
    candidates: List[EnhancedAction] = []
    
    # 1. 성장 기회 액션 (완화된 조건)
    
    # 1-1. 예금 성장 기회 (예측값이 현재보다 2% 이상 증가)
    if pred_deposit and pred_deposit > cur_deposit * 1.02:
        growth_pct = _safe_pct_change(cur_deposit, pred_deposit)
        score = min(growth_pct * 50, 100) * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="DEPOSIT_GROWTH",
            title=f"💰 예금 유치 강화 ({tier} 우대)",
            rationale=f"예금잔액 {growth_pct:.1%} 증가 예측 → {industry} 특화 우대금리 상품 제안",
            cause_summary=f"예금 {dep_dir} 추세 | 예측 성장률 +{growth_pct:.1%}",
            score=score,
            urgency="MEDIUM" if growth_pct < 0.1 else "HIGH",
            category="GROWTH",
        ))
    
    # 1-2. 대출 확대 기회 (현재 대출이 어느 정도 있어야 함)
    if pred_loan and cur_loan > 1e6 and pred_loan > cur_loan * 1.03:
        growth_pct = _safe_pct_change(cur_loan, pred_loan, min_base=1e6)
        score = min(growth_pct * 40, 80) * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="LOAN_EXPANSION",
            title="💳 여신 확대 기회",
            rationale=f"여신 수요 {growth_pct:.1%} 증가 예측 → {industry} 업종 특화 대출상품 선제 안내",
            cause_summary=f"대출 수요 증가 예상 | 예측 성장률 +{growth_pct:.1%}",
            score=score,
            urgency="MEDIUM",
            category="GROWTH",
        ))
    
    # 1-3. 순유입 양호 → 투자 기회
    if cur_net > 0 and net_trend > 0:
        score = min(net_trend * 60, 80) * base_weight
        net_eok = cur_net / 1e8
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="GROWTH_OPPORTUNITY",
            title="📈 성장 기회 포착",
            rationale=f"순유입 {net_eok:,.1f}억원 양호, {net_dir} 추세 → 잉여자금 활용 투자상품 제안",
            cause_summary=f"순유입 양호 ({net_dir}) | 잉여자금 활용 기회",
            score=score,
            urgency="MEDIUM",
            category="GROWTH",
        ))
    
    # 1-4. FX 성장 기회
    if cur_fx > 0 and pred_fx and pred_fx > cur_fx * 1.03:
        growth_pct = _safe_pct_change(cur_fx, pred_fx)
        score = min(growth_pct * 45, 90) * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="FX_EXPANSION",
            title="💱 FX 거래 확대",
            rationale=f"외환거래 {growth_pct:.1%} 증가 예측 → 환전우대 및 무역금융 확대 제안",
            cause_summary=f"FX 거래 {fx_dir} | 예측 성장률 +{growth_pct:.1%}",
            score=score,
            urgency="MEDIUM",
            category="GROWTH",
        ))
    
    # 2. 이탈 방지 / 리텐션 액션
    
    # 2-1. 이탈 방지 (순유출 + 예금 감소)
    if cur_net < 0 and dep_trend < -0.05:
        score = (abs(dep_trend) * 50 + abs(cur_net) / 1e8 * 10) * base_weight
        score = min(score + alert_severity * 20, 100)
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="CHURN_PREVENTION",
            title="⚠️ 이탈 방지 필요",
            rationale=f"순유출 + 예금 {dep_dir} → {region} 지점 RM 긴급 접촉 및 리텐션 프로그램 적용",
            cause_summary=f"순유출 지속 | 예금 {dep_dir} | 이탈 위험 높음",
            score=score,
            urgency="CRITICAL" if alert_severity > 1 else "HIGH",
            category="RETENTION",
        ))
    
    # 2-2. 예금 유출 방어 (예금만 감소)
    if dep_trend < -0.03:
        score = abs(dep_trend) * 60 * base_weight
        dep_eok = abs(cur_deposit * dep_trend) / 1e8
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="DEPOSIT_DEFENSE",
            title="🛡️ 예금 유출 방어",
            rationale=f"예금 {abs(dep_trend):.1%} 감소 추세 (▼{dep_eok:,.1f}억원 예상) → 금리 우대 및 특별 혜택 제안",
            cause_summary=f"예금 {dep_dir} 추세 | 유출 규모 약 {dep_eok:,.1f}억원",
            score=score,
            urgency="HIGH",
            category="RETENTION",
        ))
    
    # 2-3. 거래 활성화 필요 (전반적 거래 감소)
    total_activity = cur_deposit + cur_loan + cur_card + cur_fx
    if total_activity > 0 and dep_trend < 0 and fx_trend < 0:
        score = 40 * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="ENGAGEMENT_DROP",
            title="📉 거래 활성화 필요",
            rationale="전반적 거래량 감소 추세 → 맞춤형 상품/서비스 재설계 및 관계 강화 필요",
            cause_summary="예금/FX 동반 감소 | 관계 약화 징후",
            score=score,
            urgency="MEDIUM",
            category="RETENTION",
        ))
    
    # 3. 리스크 관리 액션
    
    # 3-1. 유동성 스트레스 (조건 완화)
    if util_rate >= 0.6 or (util_rate >= 0.4 and cur_net < 0):
        score = (util_rate * 50 + max(-cur_net / 1e8 * 5, 0) + alert_severity * 20) * base_weight
        net_eok = abs(cur_net) / 1e8
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="LIQUIDITY_STRESS",
            title="🚨 유동성 점검 필요",
            rationale=f"한도소진율 {util_rate:.0%} + 순유출 {net_eok:,.1f}억원 → 긴급 한도 리프레시/운영자금 지원 검토",
            cause_summary=f"한도소진율 {util_rate:.0%} | 순유출 {net_eok:,.1f}억원",
            score=score,
            urgency="CRITICAL" if util_rate >= 0.8 else "HIGH",
            category="RISK_MGMT",
        ))
    
    # 3-2. FX 리스크 (환율 변동성 국면)
    if cur_fx > 0 and "HighVol" in fx_regime:
        score = 70 * base_weight
        fx_eok = cur_fx / 1e8
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="FX_RISK",
            title="⚡ 환리스크 관리",
            rationale=f"환율 고변동성 국면 (FX거래 {fx_eok:,.1f}억원) → 환헤지 상품 및 환전 타이밍 분산 전략 안내",
            cause_summary=f"환율 고변동성 | FX 노출 {fx_eok:,.1f}억원",
            score=score,
            urgency="HIGH",
            category="RISK_MGMT",
        ))
    
    # 3-3. 신용 리스크 (대출 증가 + 순유출)
    if cur_loan > 0 and pred_loan and pred_loan > cur_loan * 1.1 and cur_net < 0:
        score = 65 * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="CREDIT_RISK",
            title="⚠️ 신용 리스크 점검",
            rationale="대출 증가 예상 + 순유출 상태 → 상환 능력 점검 및 리스크 관리 강화 필요",
            cause_summary="대출 증가 예상 | 순유출 상태 | 상환 부담 우려",
            score=score,
            urgency="HIGH",
            category="RISK_MGMT",
        ))
    
    # 4. 교차판매 기회
    
    # 4-1. 대출 교차판매 (예금만 있고 대출 없음)
    if cur_deposit > 5e7 and cur_loan < 1e6:
        score = 35 * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="CROSS_SELL_LOAN",
            title="🔗 대출 교차판매",
            rationale=f"예금 거래 우수 ({tier}) + 여신 미보유 → 신용대출/담보대출 사전승인 한도 안내",
            cause_summary=f"예금 {tier} 고객 | 여신 미보유 | 교차판매 기회",
            score=score,
            urgency="LOW",
            category="CROSS_SELL",
        ))
    
    # 4-2. 카드 교차판매
    if cur_deposit > 3e7 and cur_card < 5e6:
        score = 30 * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="CROSS_SELL_CARD",
            title="🔗 카드 교차판매",
            rationale="예금 거래 활발 + 카드 활용도 낮음 → 맞춤형 카드 추천 및 발급 혜택 안내",
            cause_summary="카드 활용 저조 | 생활밀착 상품 교차판매 기회",
            score=score,
            urgency="LOW",
            category="CROSS_SELL",
        ))
    
    # 4-3. FX 교차판매 (예금 있지만 FX 없음)
    if cur_deposit > 1e8 and cur_fx < 1e6:
        score = 30 * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="CROSS_SELL_FX",
            title="🔗 FX 서비스 제안",
            rationale=f"{tier} 고객 + FX 미이용 → 외환 거래 니즈 파악 및 환전 우대 서비스 안내",
            cause_summary=f"{tier} 고객 | FX 미이용 | 신규 서비스 기회",
            score=score,
            urgency="LOW",
            category="CROSS_SELL",
        ))
    
    # 4-4. 투자상품 교차판매 (VIP + 잉여자금)
    if tier in ["VIP", "VVIP"] and cur_net > 5e7:
        score = 40 * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="CROSS_SELL_INVEST",
            title="🔗 투자상품 제안",
            rationale=f"{tier} 고객 + 순유입 양호 → PB 연결 및 자산관리 포트폴리오 제안",
            cause_summary=f"{tier} 고객 | 순유입 양호 | 자산관리 니즈",
            score=score,
            urgency="LOW",
            category="CROSS_SELL",
        ))
    
    # 5. 디지털 전환 액션
    
    # 5-1. 디지털 전환 유도 (디지털 비중 낮음)
    if digital_share < 0.3 and cur_deposit > 1e7:
        gap = 0.3 - digital_share
        score = gap * 30 * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="DIGITAL_ONBOARD",
            title="📱 디지털 전환 유도",
            rationale=f"디지털비중 {digital_share:.0%} (목표 30% 대비 {gap:.0%}p 부족) → 모바일뱅킹 온보딩 및 전환 인센티브 제공",
            cause_summary=f"디지털비중 {digital_share:.0%} | 목표 대비 부족 | 비용 절감 기회",
            score=score,
            urgency="LOW",
            category="DIGITAL",
        ))
    
    # 5-2. 디지털 VIP (디지털 활성 고객)
    if digital_share > 0.7 and tier in ["VIP", "VVIP"]:
        score = 30 * base_weight
        
        candidates.append(EnhancedAction(
            segment_id=segment_id,
            month=month,
            action_type="DIGITAL_PREMIUM",
            title="🌟 디지털 VIP 혜택",
            rationale=f"디지털 활성 {tier} 고객 → 디지털 VIP 프로그램 (수수료 면제/우대금리) 안내로 로열티 강화",
            cause_summary=f"디지털 활성 고객 | {tier} 등급 | 로열티 프로그램 대상",
            score=score,
            urgency="LOW",
            category="DIGITAL",
        ))
    
    # 6. 기본 액션 (다른 조건 미충족 시)
    
    if not candidates:
        # 어떤 조건도 충족하지 않으면 기본 액션 생성
        if cur_deposit > 0:
            score = 20 * base_weight
            
            candidates.append(EnhancedAction(
                segment_id=segment_id,
                month=month,
                action_type="ENGAGEMENT_DROP",
                title="📊 정기 점검 권장",
                rationale=f"{tier} 고객 정기 관리 → 맞춤형 상품 정보 제공 및 관계 유지 활동",
                cause_summary="정상 범위 | 정기 모니터링 대상",
                score=score,
                urgency="LOW",
                category="RETENTION",
            ))
    
    # 점수순 정렬 및 상위 K개 반환
    candidates.sort(key=lambda x: x.score, reverse=True)
    
    # 카테고리 다양성 확보 (같은 카테고리에서 최대 2개)
    result = []
    category_count: Dict[str, int] = {}
    
    for action in candidates:
        cat = action.category
        if category_count.get(cat, 0) < 2:
            result.append(action)
            category_count[cat] = category_count.get(cat, 0) + 1
        
        if len(result) >= top_k:
            break
    
    return result


def recommend_enhanced_actions(
    panel: pd.DataFrame,
    preds_df: pd.DataFrame,
    alerts: Optional[pd.DataFrame] = None,
    fx_regime_col: str = "regime",
    top_k: int = 5,
    cfg: Optional[RadarConfig] = None,
) -> pd.DataFrame:
    """
    전체 패널에 대해 개선된 추천 액션 생성
    
    Args:
        panel: 세그먼트 패널 데이터
        preds_df: 예측 데이터프레임 (segment_id, month, KPI별 예측)
        alerts: 알림 데이터프레임
        fx_regime_col: FX 레짐 컬럼명
        top_k: 세그먼트당 상위 N개 액션
        cfg: RadarConfig
        
    Returns:
        액션 데이터프레임
    """
    all_actions = []
    
    # 최신 월 데이터만 사용
    latest_month = panel["month"].max()
    latest_panel = panel[panel["month"] == latest_month].copy()
    
    # 예측 데이터 병합 준비
    pred_kpis = ["예금총잔액", "대출총잔액", "순유입", "카드총사용", "디지털거래금액", "FX총액"]
    
    for _, row in latest_panel.iterrows():
        seg_id = row["segment_id"]
        
        # 해당 세그먼트의 예측값 가져오기
        seg_preds = preds_df[preds_df["segment_id"] == seg_id]
        preds_dict = {}
        
        if not seg_preds.empty:
            seg_pred_row = seg_preds.iloc[0]
            for kpi in pred_kpis:
                # 다양한 컬럼명 패턴 시도 (_x, _y suffix 포함)
                for col in [kpi, f"{kpi}_x", f"{kpi}_y", f"{kpi}_pred", f"{kpi}_h1", f"pred_{kpi}", f"{kpi}__pred"]:
                    if col in seg_pred_row.index and pd.notna(seg_pred_row[col]):
                        preds_dict[kpi] = float(seg_pred_row[col])
                        break
        
        # FX 레짐
        fx_regime = str(row.get(fx_regime_col, row.get("current_regime", "")))
        
        # 액션 생성
        actions = generate_enhanced_actions(
            row=row,
            preds=preds_dict,
            alerts=alerts,
            fx_regime=fx_regime,
            top_k=top_k,
            cfg=cfg,
        )
        
        # 결과 수집
        for action in actions:
            all_actions.append({
                "segment_id": action.segment_id,
                "month": action.month,
                "action_type": action.action_type,
                "title": action.title,
                "rationale": action.rationale,
                "cause_summary": action.cause_summary,
                "score": action.score,
                "urgency": action.urgency,
                "category": action.category,
            })
    
    if not all_actions:
        return pd.DataFrame(columns=[
            "segment_id", "month", "action_type", "title", "rationale",
            "cause_summary", "score", "urgency", "category"
        ])
    
    result_df = pd.DataFrame(all_actions)
    result_df = result_df.sort_values(["segment_id", "score"], ascending=[True, False])
    
    return result_df
