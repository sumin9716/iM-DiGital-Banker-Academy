import numpy as np
import pandas as pd
import pytest

from imradar.features.kpi import TARGET_COLS, compute_kpis, inverse_target_transform
from imradar.features.lags import add_lag_features


def test_compute_kpis_builds_totals_and_signed_net_flow(kpi_panel, cfg):
    out = compute_kpis(kpi_panel, cfg)
    assert out["예금총잔액"].tolist() == [175.0, 0.0]
    assert out["대출총잔액"].tolist() == [100.0, 10.0]
    assert out["순유입"].tolist() == [-50.0, 4.0]
    # 한도가 0이면 소진율은 0으로 둔다
    assert out["한도소진율"].tolist() == [0.5, 0.0]
    np.testing.assert_allclose(out["log1p_예금총잔액"], np.log1p([175.0, 0.0]))
    np.testing.assert_allclose(out["slog1p_순유입"], [-np.log1p(50.0), np.log1p(4.0)])


def test_target_cols_exist_after_compute_kpis(kpi_panel, cfg):
    out = compute_kpis(kpi_panel, cfg)
    for kpi in ("예금총잔액", "대출총잔액", "순유입"):
        assert TARGET_COLS[kpi] in out.columns


@pytest.mark.parametrize("kpi", list(TARGET_COLS))
def test_inverse_target_transform_restores_kpi_scale(kpi, kpi_panel, cfg):
    values = np.array([0.0, 12.0, 3456.0]) if kpi != "순유입" else np.array([-3456.0, 0.0, 12.0])
    transformed = np.sign(values) * np.log1p(np.abs(values))
    np.testing.assert_allclose(inverse_target_transform(kpi, transformed), values)


def test_add_lag_features_does_not_cross_segments():
    df = pd.DataFrame({
        "segment_id": ["b", "b", "a", "a"],
        "month": pd.to_datetime(["2024-02-01", "2024-01-01", "2024-02-01", "2024-01-01"]),
        "v": [20.0, 10.0, 2.0, 1.0],
    })
    out = add_lag_features(df, "segment_id", "month", ["v"], lags=(1,))
    lag = out.set_index(["segment_id", "month"])["v__lag1"]
    assert np.isnan(lag[("b", pd.Timestamp("2024-01-01"))])
    assert lag[("b", pd.Timestamp("2024-02-01"))] == 10.0
    assert lag[("a", pd.Timestamp("2024-02-01"))] == 1.0
