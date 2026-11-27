import numpy as np
import pandas as pd
import pytest

from imradar.utils import (
    inverse_log1p,
    inverse_signed_log1p,
    month_add,
    parse_bucket_kor_to_number,
    safe_log1p,
    signed_log1p,
    to_month_start,
    yyyymm_to_ts,
)


def test_signed_log1p_round_trip_keeps_sign():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    y = signed_log1p(x)
    assert np.all(np.sign(y) == np.sign(x))
    np.testing.assert_allclose(inverse_signed_log1p(y), x)


def test_safe_log1p_clips_negative_values_to_zero():
    np.testing.assert_allclose(safe_log1p(np.array([-5.0, 0.0, 9.0])), [0.0, 0.0, np.log(10.0)])
    np.testing.assert_allclose(inverse_log1p(np.array([-1.0, np.log(10.0)])), [0.0, 9.0])


def test_month_helpers():
    assert to_month_start(202412) == pd.Timestamp("2024-12-01")
    assert yyyymm_to_ts("202401") == pd.Timestamp("2024-01-01")
    assert month_add(pd.Timestamp("2024-12-01"), 3) == pd.Timestamp("2025-03-01")
    assert month_add(pd.Timestamp("2024-01-01"), -1) == pd.Timestamp("2023-12-01")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0건", 0.0),
        ("10개", 10.0),
        ("2건초과 5건이하", 3.5),
        ("50건 초과", 51.0),
        ("3건 이하", 3.0),
        ("", None),
        ("nan", None),
        (None, None),
        ("없음", None),
    ],
)
def test_parse_bucket_kor_to_number(text, expected):
    assert parse_bucket_kor_to_number(text) == expected
