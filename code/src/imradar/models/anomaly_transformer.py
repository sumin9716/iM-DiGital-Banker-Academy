"""
Transformer 기반 이상 탐지 모델

기존 잔차 기반 방식 대비 장점:
1. 시계열 패턴 학습을 통한 비정상 탐지
2. Self-Attention으로 장기 의존성 포착
3. 재구성 오차 기반 비지도 학습
4. 다변량 이상 탐지 가능

구현 모델:
- Transformer Autoencoder (재구성 오차 기반)
- Anomaly Transformer (Association Discrepancy 기반)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
import warnings
import math

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch가 설치되지 않았습니다. `pip install torch` 실행 필요")


# ============================================================================
# 설정
# ============================================================================

@dataclass
class AnomalyTransformerConfig:
    """이상 탐지 Transformer 설정"""
    # 시퀀스 설정
    seq_length: int = 12  # 입력 시퀀스 길이
    
    # 모델 아키텍처
    d_model: int = 64  # 임베딩 차원
    n_heads: int = 4  # Attention 헤드 수
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    d_ff: int = 128  # Feed-forward 차원
    dropout: float = 0.1
    
    # 학습 설정
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 5
    
    # 이상 탐지 임계값
    anomaly_threshold_percentile: float = 95.0  # 상위 5% = 이상


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """시계열 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================================
# Transformer Autoencoder
# ============================================================================

class TransformerAutoencoder(nn.Module):
    """
    Transformer 기반 Autoencoder
    정상 데이터로 학습 후 재구성 오차로 이상 탐지
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, input_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            reconstructed: (batch, seq_len, input_dim)
            latent: (batch, seq_len, d_model) - 인코더 출력
        """
        # Input projection + positional encoding
        x_proj = self.input_projection(x)
        x_enc = self.pos_encoder(x_proj)
        
        # Encode
        latent = self.encoder(x_enc)
        
        # Decode (teacher forcing: 입력을 타겟으로 사용)
        decoded = self.decoder(x_enc, latent)
        
        # Output projection
        reconstructed = self.output_projection(decoded)
        
        return reconstructed, latent
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        재구성 오차 계산 (이상 스코어)
        
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            errors: (batch, seq_len) - 각 시점의 재구성 오차
        """
        reconstructed, _ = self.forward(x)
        
        # MSE per timestep
        errors = torch.mean((x - reconstructed) ** 2, dim=-1)
        
        return errors


# ============================================================================
# Anomaly Transformer (Association Discrepancy 기반)
# ============================================================================

class AnomalyAttention(nn.Module):
    """
    Anomaly Transformer의 핵심: Association Discrepancy
    
    일반 attention과 prior-association의 차이를 학습하여
    이상 포인트에서 discrepancy가 커지도록 유도
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Prior-association을 위한 learnable sigma
        self.sigma = nn.Parameter(torch.ones(1))
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, d_model)
            return_attention: attention 가중치 반환 여부
        
        Returns:
            output: (batch, seq_len, d_model)
            series_attention: 학습된 self-attention
            prior_attention: Gaussian prior attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V projection
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Series attention (standard scaled dot-product)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        series_attention = F.softmax(scores, dim=-1)
        series_attention = self.dropout(series_attention)
        
        # Prior attention (Gaussian prior based on distance)
        # 시점 간 거리에 따른 Gaussian 분포
        positions = torch.arange(seq_len, device=x.device).float()
        distances = (positions.unsqueeze(0) - positions.unsqueeze(1)) ** 2
        prior_attention = F.softmax(-distances / (2 * self.sigma ** 2 + 1e-8), dim=-1)
        prior_attention = prior_attention.unsqueeze(0).unsqueeze(0).expand(
            batch_size, self.n_heads, -1, -1
        )
        
        # Output
        output = torch.matmul(series_attention, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        if return_attention:
            return output, series_attention, prior_attention
        return output, None, None


class AnomalyTransformerLayer(nn.Module):
    """Anomaly Transformer 레이어"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = AnomalyAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Self-attention with residual
        attn_out, series_attn, prior_attn = self.attention(x, return_attention)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        x = self.norm2(x + self.ff(x))
        
        return x, series_attn, prior_attn


class AnomalyTransformer(nn.Module):
    """
    Anomaly Transformer
    
    핵심 아이디어: 정상 데이터에서는 인접 시점 간 association이 강하고,
    이상 데이터에서는 그렇지 않음. 이 차이(Association Discrepancy)를
    이상 스코어로 활용.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_len: int = 100,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Anomaly Transformer layers
        self.layers = nn.ModuleList([
            AnomalyTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection (reconstruction)
        self.output_projection = nn.Linear(d_model, input_dim)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List], Optional[List]]:
        """
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            reconstructed: (batch, seq_len, input_dim)
            series_attentions: 각 레이어의 series attention (optional)
            prior_attentions: 각 레이어의 prior attention (optional)
        """
        # Input projection + positional encoding
        h = self.input_projection(x)
        h = self.pos_encoder(h)
        
        series_attentions = []
        prior_attentions = []
        
        # Pass through layers
        for layer in self.layers:
            h, series_attn, prior_attn = layer(h, return_attention)
            if return_attention:
                series_attentions.append(series_attn)
                prior_attentions.append(prior_attn)
        
        # Reconstruct
        reconstructed = self.output_projection(h)
        
        if return_attention:
            return reconstructed, series_attentions, prior_attentions
        return reconstructed, None, None
    
    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        이상 스코어 계산: 재구성 오차 + Association Discrepancy
        
        Args:
            x: (batch, seq_len, input_dim)
        
        Returns:
            scores: (batch, seq_len) - 각 시점의 이상 스코어
        """
        reconstructed, series_attns, prior_attns = self.forward(x, return_attention=True)
        
        # 1. 재구성 오차
        recon_error = torch.mean((x - reconstructed) ** 2, dim=-1)  # (batch, seq_len)
        
        # 2. Association Discrepancy (KL divergence between series and prior)
        assoc_discrepancy = torch.zeros_like(recon_error)
        
        for series_attn, prior_attn in zip(series_attns, prior_attns):
            # series_attn, prior_attn: (batch, n_heads, seq_len, seq_len)
            # KL(series || prior)
            kl = series_attn * (torch.log(series_attn + 1e-8) - torch.log(prior_attn + 1e-8))
            kl = kl.sum(dim=-1).mean(dim=1)  # (batch, seq_len)
            assoc_discrepancy += kl
        
        assoc_discrepancy /= len(series_attns)
        
        # 최종 이상 스코어: 재구성 오차 × Association Discrepancy
        # (discrepancy가 클수록 이상, 재구성 오차가 클수록 이상)
        scores = recon_error * (1 + assoc_discrepancy)
        
        return scores


# ============================================================================
# 데이터셋
# ============================================================================

class AnomalyDataset(Dataset):
    """이상 탐지용 시계열 데이터셋"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        group_col: str = "segment_id",
        seq_length: int = 12,
    ):
        self.feature_cols = feature_cols
        self.seq_length = seq_length
        self.samples = []
        self.metadata = []  # (segment_id, start_idx)
        
        for seg_id, group in df.groupby(group_col):
            group = group.sort_values("month")
            values = group[feature_cols].values
            months = group["month"].values
            
            for i in range(len(group) - seq_length + 1):
                x = values[i:i + seq_length]
                if not np.isnan(x).any():
                    self.samples.append(x.astype(np.float32))
                    self.metadata.append({
                        "segment_id": seg_id,
                        "start_month": months[i],
                        "end_month": months[i + seq_length - 1],
                    })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.samples[idx])


# ============================================================================
# 학습 및 탐지
# ============================================================================

def train_anomaly_transformer(
    df: pd.DataFrame,
    feature_cols: List[str],
    group_col: str = "segment_id",
    config: Optional[AnomalyTransformerConfig] = None,
    model_type: str = "autoencoder",  # "autoencoder" or "anomaly_transformer"
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, float], float]:
    """
    이상 탐지 Transformer 학습
    
    Args:
        df: 패널 데이터 (정상 데이터로 학습)
        feature_cols: 피처 컬럼들
        group_col: 그룹 컬럼
        config: 모델 설정
        model_type: "autoencoder" 또는 "anomaly_transformer"
        verbose: 진행 상황 출력
    
    Returns:
        (model, metrics, threshold)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch가 필요합니다")
    
    if config is None:
        config = AnomalyTransformerConfig()
    
    # 데이터셋 준비
    dataset = AnomalyDataset(df, feature_cols, group_col, config.seq_length)
    
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 모델 생성
    input_dim = len(feature_cols)
    
    if model_type == "autoencoder":
        model = TransformerAutoencoder(
            input_dim=input_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_encoder_layers=config.n_encoder_layers,
            n_decoder_layers=config.n_decoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
    else:
        model = AnomalyTransformer(
            input_dim=input_dim,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_encoder_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
    
    # MPS/CUDA/CPU 디바이스 선택
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    if verbose:
        print(f"  {model_type.upper()} 학습 시작")
        print(f"  Train samples: {train_size}, Val samples: {val_size}")
        print(f"  Device: {device}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.max_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if model_type == "autoencoder":
                reconstructed, _ = model(batch)
            else:
                reconstructed, _, _ = model(batch)
            
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if model_type == "autoencoder":
                    reconstructed, _ = model(batch)
                else:
                    reconstructed, _, _ = model(batch)
                val_loss += criterion(reconstructed, batch).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{config.max_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if patience_counter >= config.early_stopping_patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break
    
    # 임계값 계산 (학습 데이터의 재구성 오차 분포 기반)
    model.eval()
    all_scores = []
    
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            if model_type == "autoencoder":
                scores = model.get_reconstruction_error(batch)
            else:
                scores = model.compute_anomaly_score(batch)
            all_scores.append(scores.cpu().numpy())
    
    all_scores = np.concatenate([s.flatten() for s in all_scores])
    threshold = np.percentile(all_scores, config.anomaly_threshold_percentile)
    
    metrics = {
        "train_loss": train_loss,
        "val_loss": best_val_loss,
        "threshold": threshold,
        "mean_score": float(np.mean(all_scores)),
        "std_score": float(np.std(all_scores)),
    }
    
    if verbose:
        print(f"  ✓ 학습 완료 | Threshold: {threshold:.6f}")
    
    return model, metrics, threshold


def detect_anomalies(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
    group_col: str = "segment_id",
    seq_length: int = 12,
    model_type: str = "autoencoder",
) -> pd.DataFrame:
    """
    이상 탐지 수행
    
    Args:
        model: 학습된 모델
        df: 탐지할 데이터
        feature_cols: 피처 컬럼들
        threshold: 이상 판정 임계값
        group_col: 그룹 컬럼
        seq_length: 시퀀스 길이
        model_type: 모델 유형
    
    Returns:
        이상 탐지 결과 DataFrame
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch가 필요합니다")
    
    device = next(model.parameters()).device
    model.eval()
    
    results = []
    
    for seg_id, group in df.groupby(group_col):
        group = group.sort_values("month")
        values = group[feature_cols].values
        months = group["month"].values
        
        for i in range(len(group) - seq_length + 1):
            x = values[i:i + seq_length]
            
            if np.isnan(x).any():
                continue
            
            x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if model_type == "autoencoder":
                    scores = model.get_reconstruction_error(x_tensor)
                else:
                    scores = model.compute_anomaly_score(x_tensor)
            
            scores = scores.cpu().numpy().flatten()
            
            # 마지막 시점의 스코어 사용
            final_score = scores[-1]
            is_anomaly = final_score > threshold
            
            results.append({
                "segment_id": seg_id,
                "month": months[i + seq_length - 1],
                "anomaly_score": float(final_score),
                "threshold": threshold,
                "is_anomaly": is_anomaly,
                "severity": _get_severity(final_score, threshold),
            })
    
    return pd.DataFrame(results)


def _get_severity(score: float, threshold: float) -> str:
    """이상 심각도 결정"""
    ratio = score / threshold
    if ratio >= 2.0:
        return "critical"
    elif ratio >= 1.5:
        return "high"
    elif ratio >= 1.0:
        return "medium"
    else:
        return "low"


# ============================================================================
# 통합 인터페이스
# ============================================================================

def train_deep_anomaly_detector(
    df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str = "autoencoder",  # "autoencoder" or "anomaly_transformer"
    **kwargs,
) -> Tuple[nn.Module, Dict[str, float], float]:
    """
    딥러닝 이상 탐지 모델 학습 통합 인터페이스
    
    Args:
        df: 패널 데이터
        feature_cols: 피처 컬럼들
        model_type: "autoencoder" 또는 "anomaly_transformer"
        **kwargs: 추가 설정
    
    Returns:
        (model, metrics, threshold)
    """
    config = AnomalyTransformerConfig(
        **{k: v for k, v in kwargs.items() if hasattr(AnomalyTransformerConfig, k)}
    )
    return train_anomaly_transformer(df, feature_cols, config=config, model_type=model_type, **kwargs)
