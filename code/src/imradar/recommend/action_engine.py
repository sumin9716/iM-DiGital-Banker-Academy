"""
추천 액션 엔진 - 세그먼트 특성 기반 정교한 NBA(Next Best Action) 생성

Features:
- 세그먼트 프로파일 분석
- KPI 상태 기반 액션
- 외부 요인 연동 액션
- 우선순위 및 긴급도 산정
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class ActionCategory(Enum):
    """액션 카테고리"""
    RETENTION = "유지"        # 이탈 방지
    GROWTH = "성장"           # 거래 확대
    CROSS_SELL = "교차판매"   # 신규 상품
    RISK_MGMT = "리스크관리"  # 위험 관리
    ENGAGEMENT = "관계강화"   # 고객 관계
    DIGITAL = "디지털전환"    # 채널 유도


class Urgency(Enum):
    """긴급도"""
    CRITICAL = 1  # 즉시 조치
    HIGH = 2      # 1주 이내
    MEDIUM = 3    # 1개월 이내
    LOW = 4       # 분기 내


@dataclass
class RecommendedAction:
    """추천 액션"""
    action_text: str
    category: ActionCategory
    urgency: Urgency
    reason: str
    expected_impact: str
    score: float = 0.0
    
    def to_display(self) -> str:
        """디스플레이용 텍스트"""
        urgency_icons = {
            Urgency.CRITICAL: "🔴",
            Urgency.HIGH: "🟠",
            Urgency.MEDIUM: "🟡",
            Urgency.LOW: "🟢",
        }
        return f"{urgency_icons.get(self.urgency, '⚪')} {self.action_text}"


@dataclass
class SegmentProfile:
    """세그먼트 프로파일"""
    segment_id: str
    age_group: str = ""
    job_group: str = ""
    activity_level: str = ""
    asset_tier: str = ""
    
    # KPI 현황
    deposit_balance: float = 0.0
    loan_balance: float = 0.0
    net_inflow: float = 0.0
    card_usage: float = 0.0
    digital_amount: float = 0.0
    fx_amount: float = 0.0
    
    # 변화율
    deposit_change: float = 0.0
    loan_change: float = 0.0
    net_inflow_change: float = 0.0
    card_usage_change: float = 0.0
    digital_share: float = 0.0
    
    # 위험 지표
    risk_score: float = 0.0
    opportunity_score: float = 0.0
    alert_type: str = ""
    alert_severity: str = ""
    
    # 외부 요인
    current_regime: str = ""
    volatility_regime: str = ""
    macro_indicators: Dict[str, float] = field(default_factory=dict)


class ActionEngine:
    """추천 액션 엔진"""
    
    def __init__(self):
        self.action_rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict:
        """액션 규칙 초기화"""
        return {
            "deposit_rules": self._deposit_rules(),
            "loan_rules": self._loan_rules(),
            "net_inflow_rules": self._net_inflow_rules(),
            "digital_rules": self._digital_rules(),
            "fx_rules": self._fx_rules(),
            "cross_sell_rules": self._cross_sell_rules(),
        }
    
    def generate_actions(
        self,
        profile: SegmentProfile,
        max_actions: int = 5,
    ) -> List[RecommendedAction]:
        """
        세그먼트 프로파일 기반 추천 액션 생성
        
        Args:
            profile: 세그먼트 프로파일
            max_actions: 최대 액션 수
            
        Returns:
            우선순위순 추천 액션 리스트
        """
        all_actions = []
        
        # 1. KPI 기반 액션
        all_actions.extend(self._generate_deposit_actions(profile))
        all_actions.extend(self._generate_loan_actions(profile))
        all_actions.extend(self._generate_net_inflow_actions(profile))
        all_actions.extend(self._generate_digital_actions(profile))
        all_actions.extend(self._generate_fx_actions(profile))
        
        # 2. 교차판매 기회
        all_actions.extend(self._generate_cross_sell_actions(profile))
        
        # 3. 위험 관리 액션
        all_actions.extend(self._generate_risk_actions(profile))
        
        # 4. 점수화 및 정렬
        scored_actions = self._score_and_rank(all_actions, profile)
        
        # 5. 중복 제거 및 상위 N개 반환
        return self._deduplicate(scored_actions)[:max_actions]
    
    def _generate_deposit_actions(self, profile: SegmentProfile) -> List[RecommendedAction]:
        """예금 관련 액션"""
        actions = []
        
        # 예금 감소 시
        if profile.deposit_change < -0.05:
            severity = "급락" if profile.deposit_change < -0.15 else "감소"
            
            if profile.age_group in ["60대", "70대이상"]:
                actions.append(RecommendedAction(
                    action_text="시니어 맞춤 정기예금 금리 우대 안내",
                    category=ActionCategory.RETENTION,
                    urgency=Urgency.HIGH if profile.deposit_change < -0.15 else Urgency.MEDIUM,
                    reason=f"예금잔액 {abs(profile.deposit_change)*100:.1f}% {severity} - 시니어 고객 이탈 위험",
                    expected_impact="잔액 회복 및 장기 관계 유지",
                    score=85,
                ))
            elif profile.age_group in ["20대", "30대"]:
                actions.append(RecommendedAction(
                    action_text="급여이체 연동 자동저축 상품 제안",
                    category=ActionCategory.RETENTION,
                    urgency=Urgency.HIGH if profile.deposit_change < -0.15 else Urgency.MEDIUM,
                    reason=f"예금잔액 {abs(profile.deposit_change)*100:.1f}% {severity} - 청년층 자금 유출",
                    expected_impact="정기적 자금 유입 확보",
                    score=80,
                ))
            else:
                actions.append(RecommendedAction(
                    action_text="예금 금리 우대 프로그램 안내 및 RM 컨택",
                    category=ActionCategory.RETENTION,
                    urgency=Urgency.HIGH if profile.deposit_change < -0.15 else Urgency.MEDIUM,
                    reason=f"예금잔액 {abs(profile.deposit_change)*100:.1f}% {severity}",
                    expected_impact="고객 이탈 방지 및 잔액 안정화",
                    score=75,
                ))
        
        # 예금 증가 시 - 투자 제안
        if profile.deposit_change > 0.1 and profile.deposit_balance > 10000:
            if profile.asset_tier in ["VIP", "VVIP"]:
                actions.append(RecommendedAction(
                    action_text="PB 연결 및 자산관리 포트폴리오 제안",
                    category=ActionCategory.GROWTH,
                    urgency=Urgency.MEDIUM,
                    reason=f"예금잔액 {profile.deposit_change*100:.1f}% 증가 - 투자 니즈 파악 필요",
                    expected_impact="수익 다각화 및 AUM 확대",
                    score=70,
                ))
            else:
                actions.append(RecommendedAction(
                    action_text="정기예금 만기 연장 우대 및 적립식 펀드 안내",
                    category=ActionCategory.GROWTH,
                    urgency=Urgency.LOW,
                    reason=f"예금잔액 증가 - 잉여자금 활용 기회",
                    expected_impact="고객 수익 증대 및 충성도 향상",
                    score=60,
                ))
        
        return actions
    
    def _generate_loan_actions(self, profile: SegmentProfile) -> List[RecommendedAction]:
        """대출 관련 액션"""
        actions = []
        
        # 대출 증가 추세
        if profile.loan_change > 0.1:
            actions.append(RecommendedAction(
                action_text="대출 한도 여유 점검 및 금리 재협상 안내",
                category=ActionCategory.GROWTH,
                urgency=Urgency.MEDIUM,
                reason=f"대출잔액 {profile.loan_change*100:.1f}% 증가 - 자금 수요 활발",
                expected_impact="고객 금융비용 절감 및 만족도 향상",
                score=65,
            ))
        
        # 대출 급감 (상환) - 재유치
        if profile.loan_change < -0.2:
            if profile.job_group in ["자영업", "소상공인"]:
                actions.append(RecommendedAction(
                    action_text="사업자 운영자금 대출 신규 제안 (우대금리)",
                    category=ActionCategory.GROWTH,
                    urgency=Urgency.MEDIUM,
                    reason=f"대출잔액 {abs(profile.loan_change)*100:.1f}% 감소 - 상환 완료 추정",
                    expected_impact="신규 여신 유치",
                    score=70,
                ))
            else:
                actions.append(RecommendedAction(
                    action_text="주택담보대출 갈아타기 또는 신용대출 한도 안내",
                    category=ActionCategory.GROWTH,
                    urgency=Urgency.LOW,
                    reason=f"대출잔액 감소 - 재유치 기회",
                    expected_impact="여신 포트폴리오 유지",
                    score=55,
                ))
        
        return actions
    
    def _generate_net_inflow_actions(self, profile: SegmentProfile) -> List[RecommendedAction]:
        """순유입 관련 액션"""
        actions = []
        
        # 순유출 지속
        if profile.net_inflow < 0:
            outflow_severity = abs(profile.net_inflow)
            
            if profile.alert_severity in ["CRITICAL", "HIGH"]:
                actions.append(RecommendedAction(
                    action_text="긴급 RM 컨택 - 이탈 방지 상담 진행",
                    category=ActionCategory.RETENTION,
                    urgency=Urgency.CRITICAL,
                    reason="순유출 지속 + 위험 알림 발생 - 고객 이탈 임박",
                    expected_impact="고객 유지 및 자금 유출 중단",
                    score=95,
                ))
            elif profile.net_inflow_change < -0.5:
                actions.append(RecommendedAction(
                    action_text="자금흐름 점검 및 맞춤 금융상품 재설계 상담",
                    category=ActionCategory.RETENTION,
                    urgency=Urgency.HIGH,
                    reason=f"순유출 급증 ({abs(profile.net_inflow_change)*100:.0f}%↓) - 자금 이동 감지",
                    expected_impact="자금 이탈 원인 파악 및 대응",
                    score=85,
                ))
            else:
                actions.append(RecommendedAction(
                    action_text="급여이체/자동이체 혜택 안내로 정기 유입 확보",
                    category=ActionCategory.RETENTION,
                    urgency=Urgency.MEDIUM,
                    reason="순유출 상태 - 정기 유입원 확보 필요",
                    expected_impact="안정적 자금 유입 구조 구축",
                    score=70,
                ))
        
        # 순유입 급증 - Cross-sell 기회
        if profile.net_inflow > 0 and profile.net_inflow_change > 0.3:
            actions.append(RecommendedAction(
                action_text="유입 자금 활용 상품 제안 (예금/펀드/보험)",
                category=ActionCategory.CROSS_SELL,
                urgency=Urgency.MEDIUM,
                reason=f"순유입 급증 ({profile.net_inflow_change*100:.0f}%↑) - 잉여자금 발생",
                expected_impact="자산 확대 및 상품 다각화",
                score=75,
            ))
        
        return actions
    
    def _generate_digital_actions(self, profile: SegmentProfile) -> List[RecommendedAction]:
        """디지털 채널 관련 액션"""
        actions = []
        
        # 디지털 비중 낮음
        if profile.digital_share < 0.3:
            if profile.age_group in ["20대", "30대"]:
                actions.append(RecommendedAction(
                    action_text="모바일뱅킹 전환 인센티브 (포인트/쿠폰) 제공",
                    category=ActionCategory.DIGITAL,
                    urgency=Urgency.MEDIUM,
                    reason=f"디지털 비중 {profile.digital_share*100:.0f}%로 낮음 (청년층 평균 대비)",
                    expected_impact="운영비용 절감 및 고객 편의 향상",
                    score=65,
                ))
            elif profile.age_group in ["40대", "50대"]:
                actions.append(RecommendedAction(
                    action_text="디지털뱅킹 1:1 안내 서비스 제공",
                    category=ActionCategory.DIGITAL,
                    urgency=Urgency.LOW,
                    reason=f"디지털 비중 {profile.digital_share*100:.0f}%로 낮음",
                    expected_impact="점진적 디지털 전환 유도",
                    score=50,
                ))
        
        # 디지털 활성 고객에게 추가 서비스
        if profile.digital_share > 0.7 and profile.activity_level == "고활성":
            actions.append(RecommendedAction(
                action_text="디지털 VIP 프로그램 안내 (수수료 면제/우대금리)",
                category=ActionCategory.ENGAGEMENT,
                urgency=Urgency.LOW,
                reason="디지털 활성 우수 고객 - 로열티 프로그램 적용",
                expected_impact="고객 충성도 및 만족도 제고",
                score=55,
            ))
        
        return actions
    
    def _generate_fx_actions(self, profile: SegmentProfile) -> List[RecommendedAction]:
        """FX 관련 액션"""
        actions = []
        
        if profile.fx_amount <= 0:
            return actions
        
        is_high_vol = "HighVol" in profile.volatility_regime
        is_uptrend = "Uptrend" in profile.current_regime  # 원화 약세
        is_downtrend = "Downtrend" in profile.current_regime  # 원화 강세
        
        # 고변동성 시기
        if is_high_vol:
            actions.append(RecommendedAction(
                action_text="환리스크 헤지 컨설팅 - 선물환/옵션 상품 안내",
                category=ActionCategory.RISK_MGMT,
                urgency=Urgency.HIGH,
                reason="환율 고변동성 국면 - 환리스크 노출 증가",
                expected_impact="환손실 리스크 축소",
                score=85,
            ))
        
        # 원화 약세 (수입기업 부담)
        if is_uptrend:
            if profile.risk_score > 60:
                actions.append(RecommendedAction(
                    action_text="무역금융 한도 확대 및 결제 분할 전략 컨설팅",
                    category=ActionCategory.RISK_MGMT,
                    urgency=Urgency.HIGH,
                    reason="원화 약세 + 리스크 점수 높음 - 수입비용 부담 증가",
                    expected_impact="자금 부담 완화 및 환율 노출 분산",
                    score=80,
                ))
            actions.append(RecommendedAction(
                action_text="환전 타이밍 분산 전략 및 환율 알림 서비스 안내",
                category=ActionCategory.RISK_MGMT,
                urgency=Urgency.MEDIUM,
                reason="원화 약세 국면 - 환전 비용 최적화 필요",
                expected_impact="평균 환전 비용 절감",
                score=70,
            ))
        
        # 원화 강세 (수출기업 기회)
        if is_downtrend:
            actions.append(RecommendedAction(
                action_text="수출대금 조기 정산 및 FX 거래 확대 인센티브 제안",
                category=ActionCategory.GROWTH,
                urgency=Urgency.MEDIUM,
                reason="원화 강세 국면 - 수출기업 환차익 기회",
                expected_impact="FX 거래량 확대 및 고객 수익 증대",
                score=75,
            ))
        
        # 디지털 FX 유도
        if profile.digital_share < 0.4 and profile.fx_amount > 0:
            actions.append(RecommendedAction(
                action_text="기업뱅킹 셀프 FX 채널 온보딩 및 환율우대 안내",
                category=ActionCategory.DIGITAL,
                urgency=Urgency.LOW,
                reason="FX 거래 활발하나 디지털 비중 낮음",
                expected_impact="운영 효율화 및 비용 절감",
                score=60,
            ))
        
        return actions
    
    def _generate_cross_sell_actions(self, profile: SegmentProfile) -> List[RecommendedAction]:
        """교차판매 기회"""
        actions = []
        
        # 예금만 있고 대출 없음 → 대출 제안
        if profile.deposit_balance > 5000 and profile.loan_balance == 0:
            if profile.job_group in ["직장인", "공무원"]:
                actions.append(RecommendedAction(
                    action_text="신용대출 사전승인 한도 안내 (급여 연동)",
                    category=ActionCategory.CROSS_SELL,
                    urgency=Urgency.LOW,
                    reason="예금 거래만 보유 - 여신 상품 교차판매 기회",
                    expected_impact="여신 포트폴리오 확대",
                    score=55,
                ))
        
        # 카드 사용 낮음
        if profile.card_usage < 100 and profile.deposit_balance > 3000:
            actions.append(RecommendedAction(
                action_text="맞춤형 카드 추천 및 발급 혜택 안내",
                category=ActionCategory.CROSS_SELL,
                urgency=Urgency.LOW,
                reason="카드 활용도 낮음 - 생활밀착 상품 추가 기회",
                expected_impact="카드 수수료 수익 및 거래 활성화",
                score=45,
            ))
        
        # 활성 고객 보험/펀드
        if profile.activity_level == "고활성" and profile.asset_tier in ["VIP", "VVIP"]:
            actions.append(RecommendedAction(
                action_text="보험/펀드 맞춤 포트폴리오 제안 (PB 연결)",
                category=ActionCategory.CROSS_SELL,
                urgency=Urgency.LOW,
                reason="고활성 VIP 고객 - 종합 자산관리 니즈 예상",
                expected_impact="수익 다각화 및 고객 Lock-in",
                score=60,
            ))
        
        return actions
    
    def _generate_risk_actions(self, profile: SegmentProfile) -> List[RecommendedAction]:
        """위험 관리 액션"""
        actions = []
        
        # CRITICAL 알림
        if profile.alert_severity == "CRITICAL":
            actions.append(RecommendedAction(
                action_text="⚠️ 긴급 리스크 점검 - 담당 RM 즉시 컨택 필요",
                category=ActionCategory.RISK_MGMT,
                urgency=Urgency.CRITICAL,
                reason="CRITICAL 수준 이상 징후 감지 - 즉각 대응 필요",
                expected_impact="위험 조기 차단 및 손실 방지",
                score=100,
            ))
        
        # 높은 리스크 점수
        if profile.risk_score >= 70:
            actions.append(RecommendedAction(
                action_text="리스크 점검 미팅 스케줄링 및 대응 전략 수립",
                category=ActionCategory.RISK_MGMT,
                urgency=Urgency.HIGH,
                reason=f"리스크 점수 {profile.risk_score:.0f}점 - 관리 강화 필요",
                expected_impact="리스크 요인 해소 및 관계 안정화",
                score=85,
            ))
        
        # DROP 알림
        if profile.alert_type == "DROP":
            actions.append(RecommendedAction(
                action_text="이탈 징후 고객 집중 관리 - 맞춤 리텐션 프로그램",
                category=ActionCategory.RETENTION,
                urgency=Urgency.HIGH,
                reason="KPI 급락 알림 발생 - 이탈 가능성 높음",
                expected_impact="고객 이탈 방지",
                score=80,
            ))
        
        return actions
    
    def _score_and_rank(
        self,
        actions: List[RecommendedAction],
        profile: SegmentProfile,
    ) -> List[RecommendedAction]:
        """액션 점수화 및 정렬"""
        for action in actions:
            # 긴급도 가중치
            urgency_weights = {
                Urgency.CRITICAL: 2.0,
                Urgency.HIGH: 1.5,
                Urgency.MEDIUM: 1.0,
                Urgency.LOW: 0.7,
            }
            action.score *= urgency_weights.get(action.urgency, 1.0)
            
            # VIP 가중치
            if profile.asset_tier in ["VIP", "VVIP"]:
                action.score *= 1.2
        
        # 점수순 정렬
        return sorted(actions, key=lambda x: x.score, reverse=True)
    
    def _deduplicate(self, actions: List[RecommendedAction]) -> List[RecommendedAction]:
        """중복 제거"""
        seen_texts = set()
        unique = []
        for action in actions:
            key = action.action_text[:30]  # 앞 30자로 중복 판단
            if key not in seen_texts:
                seen_texts.add(key)
                unique.append(action)
        return unique
    
    # 규칙 정의 메서드들
    def _deposit_rules(self) -> Dict:
        return {}
    
    def _loan_rules(self) -> Dict:
        return {}
    
    def _net_inflow_rules(self) -> Dict:
        return {}
    
    def _digital_rules(self) -> Dict:
        return {}
    
    def _fx_rules(self) -> Dict:
        return {}
    
    def _cross_sell_rules(self) -> Dict:
        return {}


def build_segment_profile(
    row: pd.Series,
    macro_data: Optional[Dict] = None,
) -> SegmentProfile:
    """
    패널 데이터 행에서 세그먼트 프로파일 생성
    
    Args:
        row: 패널 데이터 행
        macro_data: 거시경제 지표 딕셔너리
        
    Returns:
        SegmentProfile
    """
    def safe_float(val, default=0.0):
        try:
            return float(val) if pd.notna(val) else default
        except:
            return default
    
    def safe_str(val, default=""):
        return str(val) if pd.notna(val) else default
    
    return SegmentProfile(
        segment_id=safe_str(row.get("segment_id", "")),
        age_group=safe_str(row.get("연령대", "")),
        job_group=safe_str(row.get("직업군", "")),
        activity_level=safe_str(row.get("거래활성도", "")),
        asset_tier=safe_str(row.get("자산규모", "")),
        
        deposit_balance=safe_float(row.get("예금총잔액", 0)),
        loan_balance=safe_float(row.get("대출총잔액", 0)),
        net_inflow=safe_float(row.get("순유입", 0)),
        card_usage=safe_float(row.get("카드총사용", 0)),
        digital_amount=safe_float(row.get("디지털거래금액", 0)),
        fx_amount=safe_float(row.get("FX총액", 0)),
        
        deposit_change=safe_float(row.get("deposit_pct_chg", row.get("예금총잔액_pct_chg", 0))),
        loan_change=safe_float(row.get("loan_pct_chg", row.get("대출총잔액_pct_chg", 0))),
        net_inflow_change=safe_float(row.get("net_inflow_pct_chg", row.get("순유입_pct_chg", 0))),
        card_usage_change=safe_float(row.get("card_pct_chg", row.get("카드총사용_pct_chg", 0))),
        digital_share=safe_float(row.get("디지털비중", 0)),
        
        risk_score=safe_float(row.get("risk_score", 0)),
        opportunity_score=safe_float(row.get("opportunity_score", 0)),
        alert_type=safe_str(row.get("alert_type", "")),
        alert_severity=safe_str(row.get("severity", "")),
        
        current_regime=safe_str(row.get("regime", row.get("current_regime", ""))),
        volatility_regime=safe_str(row.get("volatility_regime", "")),
        macro_indicators=macro_data or {},
    )


def generate_recommended_actions_for_segment(
    row: pd.Series,
    macro_data: Optional[Dict] = None,
    max_actions: int = 3,
) -> str:
    """
    세그먼트 행에 대한 추천 액션 문자열 생성
    
    Args:
        row: 패널 데이터 행
        macro_data: 거시경제 지표
        max_actions: 최대 액션 수
        
    Returns:
        세미콜론으로 구분된 추천 액션 문자열
    """
    profile = build_segment_profile(row, macro_data)
    engine = ActionEngine()
    actions = engine.generate_actions(profile, max_actions=max_actions)
    
    if not actions:
        return "현재 특별 권고사항 없음"
    
    return "; ".join([a.to_display() for a in actions])


def generate_action_reason(row: pd.Series) -> str:
    """
    추천 액션의 핵심 사유 생성
    
    Args:
        row: 패널 데이터 행
        
    Returns:
        핵심 사유 문자열
    """
    reasons = []
    
    # 리스크 관련
    risk_score = float(row.get("risk_score", 0) or 0)
    if risk_score >= 70:
        reasons.append(f"리스크점수 {risk_score:.0f}점 (경고)")
    elif risk_score >= 50:
        reasons.append(f"리스크점수 {risk_score:.0f}점 (주의)")
    
    # 순유입 관련
    net_inflow = float(row.get("순유입", 0) or 0)
    if net_inflow < -500:
        reasons.append("순유출 지속 중")
    elif net_inflow > 500:
        reasons.append("순유입 양호")
    
    # 알림 관련
    alert_type = str(row.get("alert_type", ""))
    severity = str(row.get("severity", ""))
    if alert_type == "DROP":
        reasons.append(f"급락 알림 ({severity})")
    elif alert_type == "SPIKE":
        reasons.append(f"급등 알림 ({severity})")
    elif alert_type == "INFLECTION":
        reasons.append("추세 변곡점 감지")
    
    # FX 관련
    regime = str(row.get("regime", row.get("current_regime", "")))
    if "HighVol" in regime:
        reasons.append("환율 고변동성")
    if "Uptrend" in regime:
        reasons.append("원화 약세 국면")
    elif "Downtrend" in regime:
        reasons.append("원화 강세 국면")
    
    # 기회 관련
    opp_score = float(row.get("opportunity_score", 0) or 0)
    if opp_score >= 70:
        reasons.append(f"기회점수 {opp_score:.0f}점 (높음)")
    
    if not reasons:
        return "정상 범위 내 - 정기 모니터링"
    
    return " | ".join(reasons)
