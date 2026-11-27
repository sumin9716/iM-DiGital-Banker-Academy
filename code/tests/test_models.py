import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

import imradar.models.forecast_ensemble as forecast_ensemble
from imradar.explain.lgbm_contrib import generate_narrative_report, generate_segment_explanation
from imradar.models.forecast_lgbm import ForecastModel
from imradar.models.residual_alerts import score_residual_alerts
from imradar.ops.monitoring import summarize_drift
from imradar.pipelines.dashboard import severity_level


def test_score_residual_alerts_uses_kpi_thresholds():
    month = pd.Timestamp("2024-12-01")
    df = pd.DataFrame({
        "segment_id": ["drop", "spike", "normal"],
        "month": [month] * 3,
        "예금총잔액": [80.0, 140.0, 105.0],
        "pred": [100.0, 100.0, 100.0],
    })
    out = score_residual_alerts(df, kpi="예금총잔액", pred_col="pred", actual_col="예금총잔액").set_index("segment_id")
    # 예금총잔액 임계값: -15% 이하 DROP, +30% 이상 SPIKE
    assert out.loc["drop", "alert_type"] == "DROP"
    assert out.loc["spike", "alert_type"] == "SPIKE"
    assert out.loc["normal", "alert_type"] == "NORMAL"
    assert out.loc["drop", "residual_pct"] == pytest.approx(-0.2)
    assert out.loc["spike", "severity"] == "LOW"
    assert out.loc["normal", "alert_group"] == 0


def test_summarize_drift_counts_thresholds():
    report = pd.DataFrame({"column": ["a", "b", "c"], "psi": [0.05, 0.15, 0.30]})
    summary = summarize_drift(report)
    assert summary["n_columns"] == 3
    assert summary["warn_count"] == 2
    assert summary["crit_count"] == 1


def test_ensemble_training_reports_missing_optional_deps(monkeypatch):
    monkeypatch.setattr(forecast_ensemble, "_OPTIONAL_IMPORT_ERROR", ImportError("no xgboost"))
    with pytest.raises(ImportError, match="requirements-optional.txt"):
        forecast_ensemble.train_ensemble_forecast(pd.DataFrame(), kpi_col="x", horizon=1)


@pytest.fixture
def tiny_model():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "segment_id": [f"s{i % 4}" for i in range(n)],
        "month": pd.Timestamp("2024-01-01"),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1).fit(df[["x1", "x2"]], 2 * df["x1"])
    fm = ForecastModel(kpi="log1p_예금총잔액", horizon=1, feature_cols=["x1", "x2"], categorical_cols=[], model=model)
    return fm, df


def test_generate_narrative_report_uses_contribution_detail(tiny_model):
    fm, df = tiny_model
    report = generate_narrative_report(fm, df, "s1", pd.Timestamp("2024-01-01"), actual=110.0, predicted=100.0)
    assert report.startswith("=== 세그먼트 분석 리포트 ===")
    assert "+10 (+10.0%)" in report
    assert "분석 오류" not in report


def test_generate_segment_explanation_returns_sentence(tiny_model):
    fm, df = tiny_model
    text = generate_segment_explanation(fm, df.iloc[[0]], "예금총잔액", residual_pct=25.0)
    assert isinstance(text, str)
    assert "급증" in text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("CRITICAL", "긴급"), ("HIGH", "주의"), ("MEDIUM", "참고"), ("LOW", "참고"), (0.9, "긴급"), (0.6, "주의"), (0.1, "참고")],
)
def test_watchlist_severity_level_accepts_labels_and_scores(raw, expected):
    assert severity_level(raw) == expected
