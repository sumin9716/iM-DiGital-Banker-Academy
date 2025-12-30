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

    w = _w_grade(grade, cfg) * _w_dedicated(dedicated, cfg)

    cand: List[Action] = []

    # Helper: risk severity weight from alerts
    sev_w = 0.0
    if alerts is not None and not alerts.empty:
        a = alerts[alerts["segment_id"] == segment_id]
        if not a.empty:
            # Take worst residual_pct
            worst = a.sort_values("residual_pct").iloc[0]
            rp = float(worst["residual_pct"])
            # severity: negative drop is higher risk
            sev_w = min(2.0, max(0.0, -rp))  # 0~2
            # KPI-specific flags available if needed

    # Growth deltas (pred - current)
    def delta(k: str) -> float:
        if k in preds and k in row.index and pd.notnull(row[k]) and pd.notnull(preds[k]):
            return float(preds[k] - float(row[k]))
        return 0.0

    d_dep = delta("예금총잔액")
    d_loan = delta("대출총잔액")
    d_card = delta("카드총사용")
    d_dig = delta("디지털거래금액")
    d_fx = delta("FX총액")

    # Basic derived metrics
    util = float(row.get("한도소진율", 0.0) or 0.0)
    net = float(row.get("순유입", 0.0) or 0.0)
    dig_share = float(row.get("디지털비중", 0.0) or 0.0)
    auto_share = float(row.get("자동이체비중", 0.0) or 0.0)
    
    # Thresholds from config
    liquidity_threshold = cfg.liquidity_stress_util_threshold
    digital_threshold = cfg.digital_onboard_threshold
    auto_threshold = cfg.auto_transfer_threshold

    # --- Deposit actions ---
    if d_dep >= 0:
        score = (d_dep / (abs(row.get("예금총잔액", 1.0)) + 1e-9)) * 10.0 * w
        cand.append(Action(segment_id, month, "DEPOSIT_GROWTH", "예금 유치 강화 타깃", "예금총잔액 상승 예측 → 선제 유치/우대 제안", score))
    if d_dep < 0 or sev_w > 0.3:
        score = (abs(d_dep) / (abs(row.get("예금총잔액", 1.0)) + 1e-9)) * 10.0 * w + sev_w * 5.0
        cand.append(Action(segment_id, month, "DEPOSIT_DEFENSE", "예금 유출 방어(리텐션)", "예금총잔액 하락/이탈 신호 → 조건 점검 및 우대/관계 강화", score))

    # --- Loan / liquidity actions ---
    if util >= liquidity_threshold and net < 0:
        score = (util - liquidity_threshold) * 10.0 * w + sev_w * 3.0
        cand.append(Action(segment_id, month, "LIQUIDITY_STRESS", "운영자금/한도 리프레시 제안", "한도소진율↑ + 순유입 악화 → 자금압박 가능성", score))
    if d_loan > 0:
        score = (d_loan / (abs(row.get("대출총잔액", 1.0)) + 1e-9)) * 10.0 * w
        cand.append(Action(segment_id, month, "LOAN_OPP", "여신 니즈 선점(금리/조건 안내)", "대출총잔액 증가 예측 → 선제 상담/조건 안내", score))

    # --- Card usage actions ---
    if d_card < 0:
        score = (abs(d_card) / (abs(row.get("카드총사용", 1.0)) + 1e-9)) * 10.0 * w + sev_w * 2.0
        cand.append(Action(segment_id, month, "CARD_DROP", "법인카드 활성/혜택 제안", "카드총사용 감소 예측/경보 → 혜택/결제처 확장 캠페인", score))

    # --- Digital/channel actions ---
    if dig_share < digital_threshold:
        score = (digital_threshold - dig_share) * 10.0 * w
        cand.append(Action(segment_id, month, "DIGITAL_ONBOARD", "디지털 전환 유도(기업뱅킹/자동화)", "디지털비중 낮음 → 셀프 채널 온보딩/교육", score))
    if auto_share < auto_threshold:
        score = (auto_threshold - auto_share) * 10.0 * w
        cand.append(Action(segment_id, month, "AUTO_TRANSFER", "자동이체/CMS 활성화", "자동이체비중 낮음 → 정기 결제/자동화 유도", score))

    # --- FX (optional) ---
    if "FX총액" in row.index and row.get("FX총액", 0.0) > 0:
        if d_fx >= 0:
            score = (d_fx / (abs(row.get("FX총액", 1.0)) + 1e-9)) * 10.0 * w
            cand.append(Action(segment_id, month, "FX_GROWTH", "FX 거래 확대(정산/환전 프로세스)", "FX총액 상승 예측 → 거래 편의/우대 제안", score))
        if d_fx < 0:
            score = (abs(d_fx) / (abs(row.get("FX총액", 1.0)) + 1e-9)) * 10.0 * w + sev_w * 2.0
            cand.append(Action(segment_id, month, "FX_RISK", "FX 급감 모니터링/리텐션", "FX총액 감소 예측/경보 → 리스크관리/관계 점검", score))

    # Rank and return
    cand = sorted(cand, key=lambda a: a.score, reverse=True)
    return cand[:top_k]


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
    out = out.sort_values("score", ascending=False).head(top_n_global).reset_index(drop=True)
    return out
