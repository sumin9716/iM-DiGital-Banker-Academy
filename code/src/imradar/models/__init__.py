"""
iM Radar 모델 모듈

예측 및 이상탐지 모델 컬렉션:
- forecast_lgbm: LightGBM 기반 멀티호라이즌 예측 (기본)
- forecast_ensemble: LightGBM + XGBoost + CatBoost 앙상블 (선택 의존성 필요)
- forecast_transformer: TFT/LSTM 기반 딥러닝 예측 (PyTorch 필요)
- residual_alerts: 잔차 기반 이상 경보
- fx_regime: 환율 레짐 탐지
- risk_radar: 리스크 레이더 통합
"""

from imradar.models.forecast_lgbm import (
    train_lgbm_forecast,
    predict_with_model,
    ForecastModel,
)

from imradar.models.residual_alerts import (
    score_residual_alerts,
    Alert,
)

from imradar.models.risk_radar import (
    run_risk_radar,
)

__all__ = [
    "train_lgbm_forecast",
    "predict_with_model",
    "ForecastModel",
    "score_residual_alerts",
    "Alert",
    "run_risk_radar",
]
