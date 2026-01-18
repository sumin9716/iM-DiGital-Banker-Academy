"""
Temporal Fusion Transformer (TFT) 기반 시계열 예측 모델

LightGBM 대비 장점:
1. 멀티호라이즌 예측을 단일 모델로 처리
2. Attention 기반 해석 가능성 (어떤 시점이 중요한지)
3. 정적/동적 공변량의 자연스러운 통합
4. 불확실성 정량화 (Quantile Regression)

사용 라이브러리: PyTorch + pytorch-forecasting
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# PyTorch 관련 import (설치 필요)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch가 설치되지 않았습니다. `pip install torch` 실행 필요")

# pytorch-forecasting (TFT 구현체)
try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss, SMAPE, MAE
    import lightning.pytorch as pl
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    print("⚠️ pytorch-forecasting이 설치되지 않았습니다.")
    print("   `pip install pytorch-forecasting lightning` 실행 필요")


# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class TFTConfig:
    """TFT 모델 설정"""
    # 시퀀스 길이
    max_encoder_length: int = 12  # 과거 12개월 참조
    max_prediction_length: int = 3  # 최대 3개월 예측
    
    # 모델 아키텍처
    hidden_size: int = 64
    attention_head_size: int = 4
    dropout: float = 0.1
    hidden_continuous_size: int = 32
    
    # 학습 설정
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 50
    gradient_clip_val: float = 0.1
    
    # Quantile 예측 (불확실성)
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    
    # 조기 종료
    early_stopping_patience: int = 5


@dataclass 
class TFTForecastModel:
    """TFT 예측 모델 컨테이너"""
    kpi: str
    model: object  # TemporalFusionTransformer
    training_dataset: object  # TimeSeriesDataSet
    config: TFTConfig
    feature_importance: Optional[pd.DataFrame] = None
    metrics: Optional[Dict[str, float]] = None
    
    def save(self, path: Path) -> None:
        """모델 저장"""
        path.parent.mkdir(parents=True, exist_ok=True)
        # PyTorch 모델은 state_dict로 저장
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'kpi': self.kpi,
            'config': self.config,
            'metrics': self.metrics,
        }, path)
    
    @staticmethod
    def load(path: Path, training_dataset) -> "TFTForecastModel":
        """모델 로드"""
        checkpoint = torch.load(path)
        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            hidden_size=checkpoint['config'].hidden_size,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return TFTForecastModel(
            kpi=checkpoint['kpi'],
            model=model,
            training_dataset=training_dataset,
            config=checkpoint['config'],
            metrics=checkpoint['metrics'],
        )


# ============================================================================
# 데이터 준비
# ============================================================================

def prepare_tft_dataset(
    df: pd.DataFrame,
    target_col: str,
    time_col: str = "month",
    group_col: str = "segment_id",
    static_categoricals: Optional[List[str]] = None,
    static_reals: Optional[List[str]] = None,
    time_varying_known_reals: Optional[List[str]] = None,
    time_varying_unknown_reals: Optional[List[str]] = None,
    config: Optional[TFTConfig] = None,
    training_cutoff: Optional[pd.Timestamp] = None,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    TFT용 TimeSeriesDataSet 준비
    
    Args:
        df: 패널 데이터
        target_col: 예측 대상 KPI
        time_col: 시간 컬럼
        group_col: 그룹(세그먼트) 컬럼
        static_categoricals: 정적 범주형 변수 (업종, 지역 등)
        static_reals: 정적 수치형 변수
        time_varying_known_reals: 미래에도 알 수 있는 변수 (캘린더 등)
        time_varying_unknown_reals: 미래에 모르는 변수 (KPI lag 등)
        config: TFT 설정
        training_cutoff: 학습 데이터 종료 시점
    
    Returns:
        (training_dataset, validation_dataset)
    """
    if not TFT_AVAILABLE:
        raise ImportError("pytorch-forecasting이 필요합니다")
    
    if config is None:
        config = TFTConfig()
    
    data = df.copy()
    
    # 시간 인덱스 생성 (정수형)
    data[time_col] = pd.to_datetime(data[time_col])
    min_date = data[time_col].min()
    data["time_idx"] = ((data[time_col] - min_date).dt.days / 30).astype(int)
    
    # 결측치 처리
    data[target_col] = data[target_col].fillna(0)
    
    # 기본 변수 설정
    if static_categoricals is None:
        static_categoricals = ["업종_중분류", "사업장_시도", "법인_고객등급", "전담고객여부"]
        static_categoricals = [c for c in static_categoricals if c in data.columns]
    
    if time_varying_known_reals is None:
        time_varying_known_reals = ["time_idx"]
        # 계절성 변수 추가
        data["month_num"] = data[time_col].dt.month
        data["quarter"] = data[time_col].dt.quarter
        time_varying_known_reals.extend(["month_num", "quarter"])
    
    if time_varying_unknown_reals is None:
        time_varying_unknown_reals = [target_col]
        # Lag 변수가 있으면 추가
        lag_cols = [c for c in data.columns if "__lag" in c and target_col.split("__")[0] in c]
        time_varying_unknown_reals.extend(lag_cols[:5])  # 상위 5개만
    
    # 범주형 변수 문자열 변환
    for col in static_categoricals:
        data[col] = data[col].astype(str).fillna("unknown")
    
    # 학습/검증 분할
    if training_cutoff is None:
        max_time_idx = data["time_idx"].max()
        training_cutoff_idx = max_time_idx - config.max_prediction_length
    else:
        training_cutoff_idx = int(((training_cutoff - min_date).days / 30))
    
    # TimeSeriesDataSet 생성
    training = TimeSeriesDataSet(
        data[data["time_idx"] <= training_cutoff_idx],
        time_idx="time_idx",
        target=target_col,
        group_ids=[group_col],
        max_encoder_length=config.max_encoder_length,
        max_prediction_length=config.max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals or [],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=[group_col],
            transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # 검증 데이터셋
    validation = TimeSeriesDataSet.from_dataset(
        training,
        data,
        predict=True,
        stop_randomization=True,
    )
    
    return training, validation


# ============================================================================
# 모델 학습
# ============================================================================

def train_tft_model(
    training_dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet,
    kpi: str,
    config: Optional[TFTConfig] = None,
    verbose: bool = True,
) -> TFTForecastModel:
    """
    TFT 모델 학습
    
    Args:
        training_dataset: 학습 데이터셋
        validation_dataset: 검증 데이터셋
        kpi: 예측 대상 KPI
        config: TFT 설정
        verbose: 진행 상황 출력
    
    Returns:
        TFTForecastModel
    """
    if not TFT_AVAILABLE:
        raise ImportError("pytorch-forecasting이 필요합니다")
    
    if config is None:
        config = TFTConfig()
    
    # DataLoader 생성
    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=config.batch_size,
        num_workers=0,
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=config.batch_size,
        num_workers=0,
    )
    
    # TFT 모델 생성
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_head_size,
        dropout=config.dropout,
        hidden_continuous_size=config.hidden_continuous_size,
        output_size=len(config.quantiles),  # Quantile 예측
        loss=QuantileLoss(quantiles=config.quantiles),
        reduce_on_plateau_patience=4,
    )
    
    if verbose:
        print(f"  TFT 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer 설정
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=config.early_stopping_patience,
        verbose=verbose,
        mode="min",
    )
    
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[early_stop_callback],
        enable_progress_bar=verbose,
        logger=False,
    )
    
    # 학습
    if verbose:
        print(f"  TFT 학습 시작 (max_epochs={config.max_epochs})...")
    
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # 검증 메트릭
    val_metrics = trainer.validate(model, dataloaders=val_dataloader, verbose=False)
    metrics = {"val_loss": val_metrics[0]["val_loss"]}
    
    if verbose:
        print(f"  ✓ 학습 완료 | Val Loss: {metrics['val_loss']:.4f}")
    
    return TFTForecastModel(
        kpi=kpi,
        model=model,
        training_dataset=training_dataset,
        config=config,
        metrics=metrics,
    )


# ============================================================================
# 예측 및 해석
# ============================================================================

def predict_with_tft(
    model: TFTForecastModel,
    data: pd.DataFrame,
    return_attention: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    TFT 모델로 예측 수행
    
    Args:
        model: TFTForecastModel
        data: 입력 데이터
        return_attention: Attention 가중치 반환 여부
    
    Returns:
        예측 DataFrame (quantile별) 또는 (예측, attention)
    """
    if not TFT_AVAILABLE:
        raise ImportError("pytorch-forecasting이 필요합니다")
    
    # 예측용 데이터셋 생성
    predict_dataset = TimeSeriesDataSet.from_dataset(
        model.training_dataset,
        data,
        predict=True,
        stop_randomization=True,
    )
    predict_dataloader = predict_dataset.to_dataloader(
        train=False,
        batch_size=model.config.batch_size,
        num_workers=0,
    )
    
    # 예측
    predictions = model.model.predict(
        predict_dataloader,
        return_x=True,
        return_index=True,
    )
    
    # 결과 정리
    pred_df = pd.DataFrame({
        "segment_id": predictions.index["segment_id"],
        "time_idx": predictions.index["time_idx"],
        "pred_q10": predictions.output[:, :, 0].numpy().flatten(),  # 10% quantile
        "pred_q50": predictions.output[:, :, 1].numpy().flatten(),  # 50% (median)
        "pred_q90": predictions.output[:, :, 2].numpy().flatten(),  # 90% quantile
    })
    
    if return_attention:
        # Attention 해석
        interpretation = model.model.interpret_output(
            predictions.output,
            reduction="sum",
        )
        attention_df = pd.DataFrame(interpretation["attention"])
        return pred_df, attention_df
    
    return pred_df


def get_tft_feature_importance(model: TFTForecastModel) -> pd.DataFrame:
    """
    TFT의 Variable Selection Network에서 피처 중요도 추출
    
    Returns:
        피처별 중요도 DataFrame
    """
    if not TFT_AVAILABLE:
        raise ImportError("pytorch-forecasting이 필요합니다")
    
    # Variable importance 추출
    interpretation = model.model.interpret_output(
        model.model.predict(
            model.training_dataset.to_dataloader(train=False, batch_size=32),
            return_x=True,
        ).output,
        reduction="sum",
    )
    
    importance_df = pd.DataFrame({
        "feature": list(interpretation["static_variables"].keys()) + 
                   list(interpretation["encoder_variables"].keys()),
        "importance": list(interpretation["static_variables"].values()) +
                      list(interpretation["encoder_variables"].values()),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    
    return importance_df


# ============================================================================
# 앙상블: LightGBM + TFT
# ============================================================================

def ensemble_lgbm_tft(
    lgbm_pred: np.ndarray,
    tft_pred: np.ndarray,
    weights: Tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """
    LightGBM과 TFT 예측 앙상블
    
    Args:
        lgbm_pred: LightGBM 예측값
        tft_pred: TFT 예측값 (q50 사용)
        weights: (lgbm_weight, tft_weight)
    
    Returns:
        앙상블 예측값
    """
    w_lgbm, w_tft = weights
    return w_lgbm * lgbm_pred + w_tft * tft_pred


# ============================================================================
# 간단한 LSTM 대안 (pytorch-forecasting 없이도 동작)
# ============================================================================

if TORCH_AVAILABLE:
    class SimpleLSTMForecaster(nn.Module):
        """
        간단한 LSTM 기반 예측 모델
        pytorch-forecasting 없이도 사용 가능
        """
        
        def __init__(
            self,
            input_size: int,
            hidden_size: int = 64,
            num_layers: int = 2,
            output_size: int = 3,  # 예측 호라이즌
            dropout: float = 0.1,
        ):
            super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, output_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 마지막 hidden state 사용
        last_hidden = h_n[-1]  # (batch, hidden_size)
        output = self.fc(last_hidden)
        return output


class SegmentTimeSeriesDataset(Dataset):
    """세그먼트 시계열 데이터셋"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        group_col: str = "segment_id",
        seq_length: int = 12,
        pred_length: int = 3,
    ):
        self.df = df
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.group_col = group_col
        self.seq_length = seq_length
        self.pred_length = pred_length
        
        self.samples = self._create_samples()
    
    def _create_samples(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        samples = []
        for seg_id, group in self.df.groupby(self.group_col):
            group = group.sort_values("month")
            values = group[self.feature_cols].values
            targets = group[self.target_col].values
            
            for i in range(len(group) - self.seq_length - self.pred_length + 1):
                x = values[i:i + self.seq_length]
                y = targets[i + self.seq_length:i + self.seq_length + self.pred_length]
                
                if not np.isnan(x).any() and not np.isnan(y).any():
                    samples.append((x.astype(np.float32), y.astype(np.float32)))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y)


def train_simple_lstm(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    group_col: str = "segment_id",
    seq_length: int = 12,
    pred_length: int = 3,
    hidden_size: int = 64,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    verbose: bool = True,
) -> Tuple[SimpleLSTMForecaster, Dict[str, float]]:
    """
    간단한 LSTM 모델 학습
    
    Args:
        df: 패널 데이터
        target_col: 예측 대상
        feature_cols: 피처 컬럼들
        seq_length: 입력 시퀀스 길이
        pred_length: 예측 호라이즌
        hidden_size: LSTM hidden size
        num_epochs: 학습 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        verbose: 진행 상황 출력
    
    Returns:
        (model, metrics)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch가 필요합니다")
    
    # 데이터셋 준비
    dataset = SegmentTimeSeriesDataset(
        df, target_col, feature_cols, group_col, seq_length, pred_length
    )
    
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 모델 생성
    model = SimpleLSTMForecaster(
        input_size=len(feature_cols),
        hidden_size=hidden_size,
        output_size=pred_length,
    )
    
    # MPS/CUDA/CPU 디바이스 선택
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    if verbose:
        print(f"  LSTM 학습 시작 (epochs={num_epochs}, device={device})")
        print(f"  Train samples: {train_size}, Val samples: {val_size}")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    metrics = {
        "train_loss": train_loss,
        "val_loss": best_val_loss,
        "rmse": np.sqrt(best_val_loss),
    }
    
    if verbose:
        print(f"  ✓ LSTM 학습 완료 | Best Val RMSE: {metrics['rmse']:.4f}")
    
    return model, metrics


# ============================================================================
# 메인 함수
# ============================================================================

def train_deep_forecast(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    model_type: str = "lstm",  # "lstm" or "tft"
    **kwargs,
) -> Tuple[object, Dict[str, float]]:
    """
    딥러닝 예측 모델 학습 통합 인터페이스
    
    Args:
        df: 패널 데이터
        target_col: 예측 대상
        feature_cols: 피처 컬럼들
        model_type: "lstm" 또는 "tft"
        **kwargs: 모델별 추가 인자
    
    Returns:
        (model, metrics)
    """
    if model_type == "lstm":
        return train_simple_lstm(df, target_col, feature_cols, **kwargs)
    elif model_type == "tft":
        if not TFT_AVAILABLE:
            print("⚠️ TFT 사용 불가, LSTM으로 대체")
            return train_simple_lstm(df, target_col, feature_cols, **kwargs)
        # TFT 학습 로직
        config = TFTConfig(**{k: v for k, v in kwargs.items() if hasattr(TFTConfig, k)})
        train_ds, val_ds = prepare_tft_dataset(df, target_col, config=config)
        return train_tft_model(train_ds, val_ds, target_col, config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
