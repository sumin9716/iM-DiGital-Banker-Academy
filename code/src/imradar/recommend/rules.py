from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from imradar.config import RadarConfig


@dataclass
class Action:
    segment_id: str
    month: pd.Timestamp
    action_type: str
    title: str
    rationale: str
    score: float


def _w_grade(grade: str, cfg: RadarConfig) -> float:
    return float(cfg.grade_weights.get(str(grade), 1.0))


def _w_dedicated(flag: str, cfg: RadarConfig) -> float:
    return float(cfg.dedicated_weights.get(str(flag), 1.0))


def _safe_pct_change(cur: float, future: float, min_base: float = 1.0, cap: float = 5.0) -> float:
    denom = max(abs(cur), min_base)
    pct = (future - cur) / denom
    return float(np.clip(pct, -cap, cap))


def generate_actions_for_row(
    row: pd.Series,
    preds: Dict[str, float],
    alerts: Optional[pd.DataFrame] = None,
    top_k: int = 3,
    cfg: Optional[RadarConfig] = None,
) -> List[Action]:
    """
    Rule-based candidate generation + score ranking (label-free NBA).
    - row: current month segment metrics (KPIs + derived)
    - preds: dict of predicted next KPI values (e.g., {"예금총잔액":..., "대출총잔액":...})
    - alerts: residual alerts for the same month (optional)
    - cfg: RadarConfig for thresholds
    """
    if cfg is None:
        cfg = RadarConfig()
    
    segment_id = row["segment_id"]
    month = row["month"]
    grade = row.get("법인_고객등급", "")
    dedicated = row.get("전담고객여부", "")
    industry = row.get("업종_중분류", "미분류")
    region = row.get("사업장_시도", "미정")

    w = _w_grade(grade, cfg) * _w_dedicated(dedicated, cfg)

    cand: List[Action] = []

    # Helper: risk severity weight from alerts
    sev_w = 0.0
    alert_reason = ""
    if alerts is not None and not alerts.empty:
        a = alerts[alerts["segment_id"] == segment_id]
        if not a.empty:
            # Take worst residual_pct
            worst = a.sort_values("residual_pct").iloc[0]
            rp = float(worst["residual_pct"])
            # severity: negative drop is higher risk
            sev_w = min(2.0, max(0.0, -rp))  # 0~2
            alert_reason = f"(예측대비 {rp:.1%} 이탈)"

    # Growth deltas (pred - current)
    def delta(k: str) -> float:
        if k in preds and k in row.index and pd.notnull(row[k]) and pd.notnull(preds[k]):
            return float(preds[k] - float(row[k]))
        return 0.0
    
    def pct_delta(k: str) -> float:
        """변화율 계산"""
        if k in preds and k in row.index and pd.notnull(row[k]) and pd.notnull(preds[k]):
            cur = float(row[k])
            return _safe_pct_change(cur, float(preds[k]))
        return 0.0

    d_dep = delta("예금총잔액")
    d_loan = delta("대출총잔액")
    d_card = delta("카드총사용")
    d_dig = delta("디지털거래금액")
    d_fx = delta("FX총액")
    
    pct_dep = pct_delta("예금총잔액")
    pct_loan = pct_delta("대출총잔액")
    pct_card = pct_delta("카드총사용")

    # Basic derived metrics
    util = float(row.get("한도소진율", 0.0) or 0.0)
    util = min(max(util, 0.0), 5.0)
    net = float(row.get("순유입", 0.0) or 0.0)
    dig_share = float(row.get("디지털비중", 0.0) or 0.0)
    auto_share = float(row.get("자동이체비중", 0.0) or 0.0)
    
    # 자산 규모 정보
    cur_dep = float(row.get("예금총잔액", 0) or 0)
    tier = "VIP" if cur_dep >= 1e9 else ("우수" if cur_dep >= 1e8 else "일반")
    
    # Thresholds from config
    liquidity_threshold = cfg.liquidity_stress_util_threshold
    digital_threshold = cfg.digital_onboard_threshold
    auto_threshold = cfg.auto_transfer_threshold

    # --- Deposit actions (더 상세한 핵심사유) ---
    if d_dep >= 0 and pct_dep > 0.02:
        score = pct_dep * 50.0 * w
        delta_eok = d_dep / 1e8
        rationale = f"예금잔액 {pct_dep:.1%} 증가 예측 (▲{delta_eok:,.1f}억원) → {tier} 고객 대상 우대금리 {industry} 특화상품 제안"
        cand.append(Action(segment_id, month, "DEPOSIT_GROWTH", "💰 예금 유치 강화 (VIP 우대)", rationale, score))
        
    if d_dep < 0 or sev_w > 0.3:
        score = abs(pct_dep) * 50.0 * w + sev_w * 5.0
        delta_eok = abs(d_dep) / 1e8
        rationale = f"예금 유출 위험 감지 {alert_reason} (▼{delta_eok:,.1f}억원 예상) → {region} 지점 RM 긴급 접촉 필요"
        cand.append(Action(segment_id, month, "DEPOSIT_DEFENSE", "⚠️ 예금 유출 방어 (긴급)", rationale, score))

    # --- Loan / liquidity actions ---
    if util >= liquidity_threshold and net < 0:
        score = (util - liquidity_threshold) * 10.0 * w + sev_w * 3.0
        net_eok = abs(net) / 1e8
        rationale = f"한도소진율 {util:.0%} + 순유출 {net_eok:,.1f}억원 → 운영자금 압박 가능 → 긴급 한도 리프레시/자금 지원 검토"
        cand.append(Action(segment_id, month, "LIQUIDITY_STRESS", "🚨 운영자금/한도 긴급 점검", rationale, score))
        
    if d_loan > 0 and pct_loan > 0.05:
        score = pct_loan * 40.0 * w
        loan_eok = d_loan / 1e8
        rationale = f"여신 수요 {pct_loan:.1%} 증가 예측 (▲{loan_eok:,.1f}억원) → {industry} 업종 특화 대출상품 선제 안내"
        cand.append(Action(segment_id, month, "LOAN_OPP", "📈 여신 수요 선점", rationale, score))

    # --- Card usage actions ---
    if d_card < 0 and pct_card < -0.1:
        score = abs(pct_card) * 30.0 * w + sev_w * 2.0
        rationale = f"법인카드 사용 {pct_card:.1%} 감소 예측 → 결제처 확대/캐시백 혜택 제안으로 활성화 유도"
        cand.append(Action(segment_id, month, "CARD_DROP", "📉 법인카드 활성화 필요", rationale, score))

    # --- Digital/channel actions ---
    if dig_share < digital_threshold:
        score = (digital_threshold - dig_share) * 10.0 * w
        gap = digital_threshold - dig_share
        rationale = f"디지털비중 {dig_share:.0%} (목표 대비 {gap:.0%}p 부족) → 기업뱅킹 앱 온보딩/디지털 전환 교육 필요"
        cand.append(Action(segment_id, month, "DIGITAL_ONBOARD", "📱 디지털 전환 유도", rationale, score))
        
    if auto_share < auto_threshold:
        score = (auto_threshold - auto_share) * 10.0 * w
        gap = auto_threshold - auto_share
        rationale = f"자동이체비중 {auto_share:.0%} (목표 대비 {gap:.0%}p 부족) → CMS/정기결제 자동화 도입 권장"
        cand.append(Action(segment_id, month, "AUTO_TRANSFER", "🔄 자동이체/CMS 활성화", rationale, score))

    # --- FX (optional) ---
    if "FX총액" in row.index and row.get("FX총액", 0.0) > 0:
        fx_eok = float(row.get("FX총액", 0)) / 1e8
        pct_fx = pct_delta("FX총액")
        
        if d_fx >= 0 and pct_fx > 0.05:
            score = pct_fx * 40.0 * w
            rationale = f"외환거래 {pct_fx:.1%} 증가 예측 (현재 {fx_eok:,.1f}억원) → 환전우대/정산 편의 프로세스 개선 제안"
            cand.append(Action(segment_id, month, "FX_GROWTH", "💱 FX 거래 확대 기회", rationale, score))
            
        if d_fx < 0 and pct_fx < -0.1:
            score = abs(pct_fx) * 35.0 * w + sev_w * 2.0
            rationale = f"외환거래 {pct_fx:.1%} 감소 예측 (현재 {fx_eok:,.1f}억원) → 무역금융 니즈 점검/환헤지 상품 리텐션 필요"
            cand.append(Action(segment_id, month, "FX_RISK", "⚡ FX 거래 감소 경보", rationale, score))

    # Rank and return
    cand = sorted(cand, key=lambda a: a.score, reverse=True)
    return cand[:top_k]


# ============== 레짐 기반 FX 추천 확장 ==============

@dataclass
class FXAction:
    """FX 전용 액션 데이터 클래스"""
    segment_id: str
    month: pd.Timestamp
    action_type: str
    title: str
    rationale: str
    score: float
    regime: str
    volatility: str
    priority: str


def generate_fx_actions_for_regime(
    row: pd.Series,
    fx_regime: str,
    fx_volatility: str,
    opp_score: float = 50,
    risk_score: float = 50,
    cfg: Optional[RadarConfig] = None,
) -> List[FXAction]:
    """
    환율 레짐에 따른 FX 추천 액션 생성
    
    레짐별 추천 전략:
    - 변동성 확대: 환리스크 점검/헤지 컨설팅
    - 약세 국면 (수입형): 결제 부담↑ → 운영자금/한도 + 환율우대
    - 강세 전환 (수출형): 회수/환전 타이밍 → 환전우대/정산 편의 + 무역금융
    - FX 있는데 디지털 낮음: 셀프 채널 온보딩
    
    Args:
        row: 세그먼트 데이터
        fx_regime: 환율 레짐 (예: "Uptrend_HighVol")
        fx_volatility: 변동성 레짐
        opp_score: 기회 점수
        risk_score: 리스크 점수
        cfg: RadarConfig
    
    Returns:
        FXAction 리스트
    """
    if cfg is None:
        cfg = RadarConfig()
    
    segment_id = row.get("segment_id", "")
    month = row.get("month", pd.Timestamp.now())
    
    # 세그먼트 특성
    fx_total = float(row.get("FX총액", 0) or 0)
    digital_share = float(row.get("디지털비중", 0) or 0)
    util_rate = float(row.get("한도소진율", 0) or 0)
    net_inflow = float(row.get("순유입", 0) or 0)
    grade = row.get("법인_고객등급", "")
    dedicated = row.get("전담고객여부", "")
    
    # 가중치
    w = _w_grade(grade, cfg) * _w_dedicated(dedicated, cfg)
    
    # 레짐 파싱
    is_uptrend = "Uptrend" in fx_regime
    is_downtrend = "Downtrend" in fx_regime
    is_high_vol = "HighVol" in fx_volatility or "HighVol" in fx_regime
    is_low_vol = "LowVol" in fx_volatility or "LowVol" in fx_regime
    
    cand: List[FXAction] = []
    
    # 우선순위 결정
    if risk_score >= 70:
        priority = "CRITICAL"
    elif risk_score >= 50 or opp_score >= 70:
        priority = "HIGH"
    elif opp_score >= 50:
        priority = "MEDIUM"
    else:
        priority = "LOW"
    
    # === 레짐별 추천 로직 ===
    
    # 1. 고변동성 국면 - 모든 FX 세그먼트에 헤지 권고
    if is_high_vol:
        score = 80 * w + (risk_score * 0.3)
        cand.append(FXAction(
            segment_id, month, "FX_HEDGE_CONSULT",
            "🛡️ 환리스크 점검 및 헤지 컨설팅",
            f"환율 변동성 확대 국면 → 환헤지 상품/전략 점검 권고 (변동성: {fx_volatility})",
            score, fx_regime, fx_volatility, priority
        ))
        
        # 추가: 환율 모니터링 서비스
        score = 60 * w
        cand.append(FXAction(
            segment_id, month, "FX_MONITORING",
            "📊 환율 알림/모니터링 서비스 안내",
            "변동성 확대 시 실시간 환율 알림 서비스 제안",
            score, fx_regime, fx_volatility, "MEDIUM"
        ))
    
    # 2. 약세 국면 (Uptrend) - 수입 기업 부담 증가
    if is_uptrend:
        # 운영자금 필요성 체크
        if util_rate > 0.5 or net_inflow < 0:
            score = 75 * w + (risk_score * 0.4)
            cand.append(FXAction(
                segment_id, month, "FX_IMPORT_FINANCE",
                "💰 수입결제 자금 지원 (한도 리프레시/운영자금)",
                f"원화 약세로 수입 결제 부담 증가 예상 → 운영자금 점검 (한도소진율: {util_rate:.0%})",
                score, fx_regime, fx_volatility, priority
            ))
        
        # 환율우대 제안
        score = 65 * w + (opp_score * 0.2)
        cand.append(FXAction(
            segment_id, month, "FX_RATE_BENEFIT",
            "💱 환율우대 프로그램 안내",
            "원화 약세 국면 → 환전 시 우대환율 적용 프로그램 제안",
            score, fx_regime, fx_volatility, "MEDIUM"
        ))
        
        # 분할 환전 전략
        if fx_total > 1e8:  # 1억 이상
            score = 55 * w
            cand.append(FXAction(
                segment_id, month, "FX_SPLIT_STRATEGY",
                "🔄 분할 환전 타이밍 전략 안내",
                "환율 변동 리스크 분산을 위한 분할 환전 전략 컨설팅",
                score, fx_regime, fx_volatility, "MEDIUM"
            ))
    
    # 3. 강세 전환 (Downtrend) - 수출 기업 기회
    if is_downtrend:
        # 환전 적기 안내
        score = 70 * w + (opp_score * 0.3)
        cand.append(FXAction(
            segment_id, month, "FX_EXPORT_TIMING",
            "📈 수출대금 환전 적기 안내",
            "원화 강세 전환 → 수출대금 환전/회수 타이밍 최적화 컨설팅",
            score, fx_regime, fx_volatility, "HIGH" if opp_score > 60 else "MEDIUM"
        ))
        
        # 무역금융 옵션
        if fx_total > 5e7:  # 5천만 이상
            score = 60 * w + (opp_score * 0.2)
            cand.append(FXAction(
                segment_id, month, "FX_TRADE_FINANCE",
                "🏦 무역금융 옵션 상품 제안",
                "수출 활성화 지원을 위한 무역금융(포페이팅, 수출팩토링 등) 안내",
                score, fx_regime, fx_volatility, "MEDIUM"
            ))
        
        # FX 거래 확대 인센티브
        if opp_score > 65:
            score = 55 * w
            cand.append(FXAction(
                segment_id, month, "FX_INCENTIVE",
                "🎯 FX 거래 확대 인센티브 제안",
                "FX 거래량 증대 시 수수료 우대/캐시백 프로그램 안내",
                score, fx_regime, fx_volatility, "MEDIUM"
            ))
    
    # 4. 횡보/저변동성 국면 - 정기 서비스 안내
    if not is_high_vol and not is_uptrend and not is_downtrend:
        score = 40 * w
        cand.append(FXAction(
            segment_id, month, "FX_REGULAR_SERVICE",
            "📋 정기 FX 서비스 점검",
            "안정적 환율 국면 → 기존 FX 서비스 이용 현황 점검 및 개선 제안",
            score, fx_regime, fx_volatility, "LOW"
        ))
    
    # 5. 디지털 채널 온보딩 (공통)
    if digital_share < 0.4 and fx_total > 0:
        score = 50 * w + (opp_score * 0.1)
        cand.append(FXAction(
            segment_id, month, "FX_DIGITAL_ONBOARD",
            "📱 기업뱅킹/셀프 FX 채널 온보딩",
            f"디지털 비중 낮음({digital_share:.0%}) → 모바일/인터넷 FX 서비스 교육 및 온보딩",
            score, fx_regime, fx_volatility, "MEDIUM"
        ))
    
    # 6. 리스크 높은 세그먼트 긴급 컨택
    if risk_score >= 70:
        score = 90 * w
        cand.append(FXAction(
            segment_id, month, "FX_URGENT_CONTACT",
            "⚠️ 긴급 RM 컨택 필요",
            f"FX 리스크 점수 높음({risk_score:.0f}) → 즉시 담당 RM 연락 및 상황 점검",
            score, fx_regime, fx_volatility, "CRITICAL"
        ))
    
    # 정렬 및 반환
    cand = sorted(cand, key=lambda a: a.score, reverse=True)
    return cand[:5]  # 상위 5개


def recommend_fx_actions(
    panel_df: pd.DataFrame,
    fx_scores_df: pd.DataFrame,
    fx_regime: str,
    fx_volatility: str,
    month: pd.Timestamp,
    top_n_global: int = 100,
    cfg: Optional[RadarConfig] = None,
) -> pd.DataFrame:
    """
    전체 FX 활성 세그먼트에 대한 레짐 기반 추천 생성
    
    Args:
        panel_df: 패널 데이터
        fx_scores_df: FX 점수 DataFrame (segment_id, opportunity_score, risk_score)
        fx_regime: 현재 환율 레짐
        fx_volatility: 변동성 레짐
        month: 분석 월
        top_n_global: 전체 반환 액션 수
        cfg: RadarConfig
    
    Returns:
        FX 추천 액션 DataFrame
    """
    if cfg is None:
        cfg = RadarConfig()
    
    # 패널에서 해당 월 데이터
    month_data = panel_df[panel_df["month"] == month]
    
    all_actions = []
    
    for _, score_row in fx_scores_df.iterrows():
        seg_id = score_row["segment_id"]
        opp_score = score_row.get("opportunity_score", 50)
        risk_score = score_row.get("risk_score", 50)
        
        # 해당 세그먼트의 패널 데이터
        seg_data = month_data[month_data["segment_id"] == seg_id]
        if seg_data.empty:
            continue
        
        row = seg_data.iloc[0]
        
        # 레짐 기반 액션 생성
        actions = generate_fx_actions_for_regime(
            row, fx_regime, fx_volatility, opp_score, risk_score, cfg
        )
        
        for a in actions:
            all_actions.append({
                "month": a.month,
                "segment_id": a.segment_id,
                "action_type": a.action_type,
                "title": a.title,
                "rationale": a.rationale,
                "score": a.score,
                "regime": a.regime,
                "volatility": a.volatility,
                "priority": a.priority,
                "opportunity_score": opp_score,
                "risk_score": risk_score,
                "업종_중분류": row.get("업종_중분류", ""),
                "사업장_시도": row.get("사업장_시도", ""),
                "법인_고객등급": row.get("법인_고객등급", ""),
            })
    
    result = pd.DataFrame(all_actions)
    
    if not result.empty:
        # 우선순위 및 점수로 정렬
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        result["priority_order"] = result["priority"].map(priority_order)
        result = result.sort_values(["priority_order", "score"], ascending=[True, False])
        result = result.drop(columns=["priority_order"]).head(top_n_global)
    
    return result.reset_index(drop=True)


def recommend_actions(
    current_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    month: pd.Timestamp,
    kpi_cols: List[str],
    alerts: Optional[pd.DataFrame] = None,
    top_k_per_segment: int = 3,
    top_n_global: int = 200,
) -> pd.DataFrame:
    """
    Build NBA list for all segments for a given month.
    - current_df: panel with KPIs (for month)
    - pred_df: predictions for month+1 keyed by segment_id and KPI columns
    """
    cur = current_df[current_df["month"] == month].copy()
    pr = pred_df.copy()

    merged = cur.merge(pr, on="segment_id", how="left", suffixes=("", "__pred"))
    actions: List[Dict] = []

    for _, row in merged.iterrows():
        preds = {}
        for k in kpi_cols:
            pred_col = f"{k}__pred"
            if pred_col in merged.columns:
                preds[k] = float(row.get(pred_col, np.nan))
        acts = generate_actions_for_row(row, preds, alerts=alerts, top_k=top_k_per_segment)
        for a in acts:
            actions.append({
                "month": a.month,
                "segment_id": a.segment_id,
                "action_type": a.action_type,
                "title": a.title,
                "rationale": a.rationale,
                "score": a.score,
                "업종_중분류": row.get("업종_중분류", ""),
                "사업장_시도": row.get("사업장_시도", ""),
                "법인_고객등급": row.get("법인_고객등급", ""),
                "전담고객여부": row.get("전담고객여부", ""),
            })

    out = pd.DataFrame(actions)
    if out.empty:
        return out
    
    # action_type별 균형 선택 (다양성 확보)
    out = out.sort_values("score", ascending=False)
    
    if "action_type" in out.columns and len(out["action_type"].unique()) > 1:
        action_types = out["action_type"].unique()
        n_types = len(action_types)
        per_type = max(1, top_n_global // n_types)  # 타입당 최소 할당
        
        balanced_actions = []
        for at in action_types:
            type_actions = out[out["action_type"] == at].head(per_type)
            balanced_actions.append(type_actions)
        
        out = pd.concat(balanced_actions, ignore_index=True)
        # 남은 슬롯 채우기 (전체 score 상위에서)
        remaining = top_n_global - len(out)
        if remaining > 0:
            used_indices = set(out.index)
            original_sorted = pd.DataFrame(actions).sort_values("score", ascending=False)
            for _, row in original_sorted.iterrows():
                if len(out) >= top_n_global:
                    break
                # 중복 방지
                key = (row.get("segment_id"), row.get("action_type"))
                existing = out[(out["segment_id"] == row.get("segment_id")) & 
                               (out["action_type"] == row.get("action_type"))]
                if existing.empty:
                    out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
        
        out = out.sort_values("score", ascending=False).head(top_n_global).reset_index(drop=True)
    else:
        out = out.head(top_n_global).reset_index(drop=True)
    
    return out
