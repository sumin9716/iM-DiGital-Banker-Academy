"""
iM ONEderful - Segment Radar MVP package.

금융 세그먼트 기반 KPI 예측, 이상징후 탐지, 원인 분석, 액션 추천 시스템

Modules:
    - config: 설정 관리 (RadarConfig)
    - utils: 유틸리티 함수
    - data: 데이터 로딩 및 전처리 (P0/P1)
    - features: 피처 엔지니어링 (Lag/Rolling/변화율)
    - models: 예측 및 리스크 레이더
    - explain: 설명 가능성 (SHAP 기반 드라이버 분석)
    - recommend: Next Best Action 추천
    - pipelines: 통합 파이프라인 및 대시보드

주요 기능:
    - P0 세그먼트-월 집계 및 패널 구축
    - P1 외부 거시경제 데이터 로드 (CPI, PPI, 금리, 환율 등)
    - 멀티호라이즌 예측 (1~3개월)
    - 리스크 레이더 (residual 기반 경보)
    - SHAP 기반 원인 설명
"""

from imradar.config import RadarConfig
from imradar.utils import (
    setup_logging,
    to_month_start,
    month_add,
    yyyymm_to_ts,
    signed_log1p,
    inverse_signed_log1p,
    safe_log1p,
    inverse_log1p,
    parse_bucket_kor_to_number,
)

__version__ = "1.1.0"
__all__ = [
    # Config
    "RadarConfig",
    # Utils
    "setup_logging",
    "to_month_start",
    "month_add",
    "yyyymm_to_ts",
    "signed_log1p",
    "inverse_signed_log1p",
    "safe_log1p",
    "inverse_log1p",
    "parse_bucket_kor_to_number",
]
