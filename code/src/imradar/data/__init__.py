"""
iM Radar Data 모듈

데이터 로딩, 전처리, 외부 데이터 통합을 위한 유틸리티
"""

from .io import load_internal_csv, read_csv_with_fallback
from .preprocess import (
    PanelBuildResult,
    add_time_columns,
    aggregate_to_segment_month,
    build_full_panel,
    convert_bucket_counts,
)
from .external import (
    load_esi,
    load_fed_rate,
    load_usd_krw_daily,
    load_oil_price,
    load_trade_data,
    load_kasi_calendar,
    load_kr_interest_rate,
    load_all_external_data,
    merge_external_to_panel,
    add_macro_lag_features,
    fill_missing_values,
    validate_external_data,
    get_external_feature_groups,
    create_composite_indicators,
)

__all__ = [
    "load_internal_csv",
    "read_csv_with_fallback",
    "PanelBuildResult",
    "add_time_columns",
    "aggregate_to_segment_month",
    "build_full_panel",
    "convert_bucket_counts",
    "load_esi",
    "load_fed_rate",
    "load_usd_krw_daily",
    "load_oil_price",
    "load_trade_data",
    "load_kasi_calendar",
    "load_kr_interest_rate",
    "load_all_external_data",
    "merge_external_to_panel",
    "add_macro_lag_features",
    "fill_missing_values",
    "validate_external_data",
    "get_external_feature_groups",
    "create_composite_indicators",
]
