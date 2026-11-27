import numpy as np
import pandas as pd
import pytest

from imradar.data.io import read_csv_with_fallback
from imradar.data.preprocess import aggregate_to_segment_month, build_full_panel


@pytest.mark.parametrize("encoding", ["cp949", "utf-8-sig"])
def test_read_csv_with_fallback_reads_korean_headers(tmp_path, encoding):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"기준년월": [202401], "업종_중분류": ["음료 제조업"]}).to_csv(path, index=False, encoding=encoding)
    df = read_csv_with_fallback(path)
    assert list(df.columns) == ["기준년월", "업종_중분류"]
    assert df.loc[0, "업종_중분류"] == "음료 제조업"


def test_aggregate_to_segment_month_sums_amounts_and_counts_rows(raw_df, cfg):
    agg = aggregate_to_segment_month(raw_df, cfg)
    seg_a = agg[(agg["업종_중분류"] == "음료 제조업") & (agg["기준년월"] == 202401)].iloc[0]
    assert seg_a["요구불예금잔액__sum"] == 400.0
    assert seg_a["customer_count"] == 2
    # 구간 문자열은 숫자로 바뀐 뒤 합계와 평균이 모두 남는다: 1건 + (2+5)/2건
    assert seg_a["요구불예금좌수__sum"] == pytest.approx(4.5)
    assert seg_a["요구불예금좌수__mean"] == pytest.approx(2.25)
    # 결측 세그먼트 키는 '미상'으로 채운다
    assert "미상" in set(agg["사업장_시도"])


def test_build_full_panel_marks_pre_birth_and_fills_gaps(raw_df, cfg):
    agg = aggregate_to_segment_month(raw_df, cfg)
    agg = agg[~((agg["업종_중분류"] == "부동산업") & (agg["기준년월"] == 202403))]
    res = build_full_panel(agg, cfg)
    panel = res.panel

    assert len(res.segment_meta) == 2
    assert len(panel) == 2 * 3  # 세그먼트 2개 × 3개월
    seg_b = panel[panel["segment_id"] == "부동산업|미상|일반|N"].set_index("month")

    jan, feb, mar = (pd.Timestamp(f"2024-0{m}-01") for m in (1, 2, 3))
    assert seg_b.loc[jan, "pre_birth"] == 1
    assert np.isnan(seg_b.loc[jan, "요구불예금잔액__sum"])
    assert seg_b.loc[feb, "요구불예금잔액__sum"] == 50.0
    assert seg_b.loc[mar, "pre_birth"] == 0
    assert seg_b.loc[mar, "요구불예금잔액__sum"] == 0.0
    assert seg_b.loc[mar, "segment_age"] == 1
