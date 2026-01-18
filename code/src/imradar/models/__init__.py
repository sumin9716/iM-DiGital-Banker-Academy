"""
iM Radar 모델 모듈

예측 및 이상탐지 모델 컬렉션:
- forecast_lgbm: LightGBM 기반 멀티호라이즌 예측 (기본)
- forecast_transformer: TFT/LSTM 기반 딥러닝 예측 (옵션)
- residual_alerts: 잔차 기반 이상 경보
- anomaly_transformer: Transformer 기반 이상 탐지 (옵션)
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

# Deep Learning 모델은 선택적 import (PyTorch 필요)
# PyTorch 미설치 시 오류 방지
TRANSFORMER_FORECAST_AVAILABLE = False
TRANSFORMER_ANOMALY_AVAILABLE = False

try:
    import torch
    TRANSFORMER_FORECAST_AVAILABLE = True
    TRANSFORMER_ANOMALY_AVAILABLE = True
except ImportError:
    pass

__all__ = [
    # LightGBM 예측
    "train_lgbm_forecast",
    "predict_with_model",
    "ForecastModel",
    # 이상 탐지
    "score_residual_alerts",
    "Alert",
    # 리스크 레이더
    "run_risk_radar",
    # 딥러닝 가용성 플래그
    "TRANSFORMER_FORECAST_AVAILABLE",
    "TRANSFORMER_ANOMALY_AVAILABLE",
]
