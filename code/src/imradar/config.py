from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass(frozen=True)
class RadarConfig:
    """Configuration for segment radar MVP."""

    # Column names (internal data)
    month_col: str = "기준년월"

    # Segment keys (recommended MVP)
    segment_keys: List[str] = field(default_factory=lambda: [
        "업종_중분류",
        "사업장_시도",
        "법인_고객등급",
        "전담고객여부",
    ])

    # Monetary amount columns (float)
    amount_cols: List[str] = field(default_factory=lambda: [
        "요구불예금잔액",
        "거치식예금잔액",
        "적립식예금잔액",
        "여신한도금액",
        "여신_운전자금대출잔액",
        "여신_시설자금대출잔액",
        "외환_수출실적금액",
        "외환_수입실적금액",
        "신용카드사용금액",
        "체크카드사용금액",
        "창구거래금액",
        "인터넷뱅킹거래금액",
        "스마트뱅킹거래금액",
        "폰뱅킹거래금액",
        "ATM거래금액",
        "자동이체금액",
        "요구불입금금액",
        "요구불출금금액",
        # Optional balances (can be used as additional features)
        "수익증권잔액",
        "신탁잔액",
        "퇴직연금잔액",
    ])

    # Bucketed count columns (categorical buckets such as "0건", "2건초과 5건이하", ...)
    bucket_count_cols: List[str] = field(default_factory=lambda: [
        "요구불예금좌수",
        "거치식예금좌수",
        "적립식예금좌수",
        "수익증권좌수",
        "신탁좌수",
        "퇴직연금좌수",
        "여신_운전자금대출좌수",
        "여신_시설자금대출좌수",
        "신용카드개수",
        "외환_수출실적거래건수",
        "외환_수입실적거래건수",
        "창구거래건수",
        "인터넷뱅킹거래건수",
        "스마트뱅킹거래건수",
        "폰뱅킹거래건수",
        "ATM거래건수",
        "자동이체거래건수",
    ])

    # KPI definitions
    kpi_defs: Dict[str, List[str]] = field(default_factory=lambda: {
        "예금총잔액": ["요구불예금잔액", "거치식예금잔액", "적립식예금잔액"],
        "대출총잔액": ["여신_운전자금대출잔액", "여신_시설자금대출잔액"],
        "카드총사용": ["신용카드사용금액", "체크카드사용금액"],
        "디지털거래금액": ["인터넷뱅킹거래금액", "스마트뱅킹거래금액", "폰뱅킹거래금액"],
        "순유입": ["요구불입금금액", "-요구불출금금액"],  # allow signed contribution
        "FX총액": ["외환_수출실적금액", "외환_수입실적금액"],
    })

    # Derived KPIs (for explainability/recommendation)
    derived_kpis: List[str] = field(default_factory=lambda: [
        "한도소진율",
        "디지털비중",
        "자동이체비중",
    ])

    # Forecast horizons (months ahead)
    horizons: List[int] = field(default_factory=lambda: [1, 2, 3])
