"""
iM ONEderful - Segment Radar MVP package.

금융 세그먼트 기반 KPI 예측, 이상징후 탐지, 원인 분석, 액션 추천 시스템

Modules:
    - config: 설정 관리 (RadarConfig)
    - utils: 유틸리티 함수
    - data: 데이터 로딩 및 전처리
    - features: 피처 엔지니어링
    - models: 예측 및 알림 모델
    - explain: 설명 가능성 (드라이버 분석)
    - recommend: Next Best Action 추천
    - pipelines: 대시보드 및 리포트 생성
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

__version__ = "1.0.0"
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
