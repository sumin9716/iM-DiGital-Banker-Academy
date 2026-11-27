"""합성 데이터 fixture. 실제 은행 데이터는 저장소에 없으므로 원본 CSV와 같은 컬럼 구조만 흉내 낸다."""

import numpy as np
import pandas as pd
import pytest

from imradar.config import RadarConfig


@pytest.fixture
def cfg() -> RadarConfig:
    return RadarConfig()


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """세그먼트 A는 202401~202403에 고객 2명, 세그먼트 B는 202402에만 고객 1명이 있다."""
    rows = []
    seg_a = {"업종_중분류": "음료 제조업", "사업장_시도": "대구광역시", "법인_고객등급": "우수", "전담고객여부": "Y"}
    seg_b = {"업종_중분류": "부동산업", "사업장_시도": np.nan, "법인_고객등급": "일반", "전담고객여부": "N"}
    for ym in (202401, 202402, 202403):
        for amount, bucket in ((100.0, "1건"), (300.0, "2건초과 5건이하")):
            rows.append({"기준년월": ym, **seg_a, "요구불예금잔액": amount, "요구불예금좌수": bucket})
    rows.append({"기준년월": 202402, **seg_b, "요구불예금잔액": 50.0, "요구불예금좌수": "0건"})
    return pd.DataFrame(rows)


@pytest.fixture
def kpi_panel() -> pd.DataFrame:
    """compute_kpis 입력 형태(집계 후 __sum 컬럼)의 2개 행."""
    return pd.DataFrame({
        "요구불예금잔액__sum": [100.0, 0.0],
        "거치식예금잔액__sum": [50.0, 0.0],
        "적립식예금잔액__sum": [25.0, 0.0],
        "여신_운전자금대출잔액__sum": [60.0, 10.0],
        "여신_시설자금대출잔액__sum": [40.0, 0.0],
        "여신한도금액__sum": [200.0, 0.0],
        "요구불입금금액__sum": [30.0, 5.0],
        "요구불출금금액__sum": [80.0, 1.0],
    })
