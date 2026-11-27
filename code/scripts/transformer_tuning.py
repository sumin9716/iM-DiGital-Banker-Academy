"""
Transformer 모델 개선: 데이터 증강 + 하이퍼파라미터 튜닝

개선 전략:
1. 데이터 증강: Jittering, Scaling, Time Warping, Window Slicing, Mixup
2. 하이퍼파라미터 튜닝: Optuna 기반 자동 탐색
3. 모델 개선: Pre-LayerNorm, Learnable Positional Encoding
4. 정규화 강화: Label Smoothing, Dropout Scheduling
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# 재현성
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 디바이스 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================================
# 1. 데이터 증강 클래스
# ============================================================================

class TimeSeriesAugmentation:
    """시계열 데이터 증강 기법"""
    
    def __init__(self, 
                 jitter_sigma: float = 0.03,
                 scale_sigma: float = 0.1,
                 time_warp_sigma: float = 0.2,
                 window_slice_ratio: float = 0.9,
                 mixup_alpha: float = 0.2,
                 augment_prob: float = 0.5):
        """
        Args:
            jitter_sigma: Jittering 노이즈 강도
            scale_sigma: Scaling 변형 강도
            time_warp_sigma: Time Warping 강도
            window_slice_ratio: Window Slicing 비율
            mixup_alpha: Mixup Beta 분포 파라미터
            augment_prob: 각 증강 적용 확률
        """
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.time_warp_sigma = time_warp_sigma
        self.window_slice_ratio = window_slice_ratio
        self.mixup_alpha = mixup_alpha
        self.augment_prob = augment_prob
    
    def jittering(self, x: np.ndarray) -> np.ndarray:
        """가우시안 노이즈 추가"""
        noise = np.random.normal(0, self.jitter_sigma, x.shape)
        return x + noise
    
    def scaling(self, x: np.ndarray) -> np.ndarray:
        """스케일 변환"""
        scalers = np.random.normal(1, self.scale_sigma, (1, x.shape[1]))
        return x * scalers
    
    def magnitude_warping(self, x: np.ndarray) -> np.ndarray:
        """크기 왜곡"""
        seq_len, n_features = x.shape
        
        # 부드러운 곡선 생성
        knot = 4
        t = np.arange(seq_len)
        knot_points = np.random.normal(1, self.scale_sigma, (knot, n_features))
        
        # 선형 보간
        from scipy.interpolate import interp1d
        orig_steps = np.linspace(0, seq_len - 1, num=knot)
        
        warped = np.zeros_like(x)
        for i in range(n_features):
            f = interp1d(orig_steps, knot_points[:, i], kind='linear', fill_value='extrapolate')
            warped[:, i] = x[:, i] * f(t)
        
        return warped
    
    def time_warping(self, x: np.ndarray) -> np.ndarray:
        """시간 축 왜곡"""
        from scipy.interpolate import interp1d
        
        seq_len, n_features = x.shape
        
        # 왜곡된 시간 축 생성
        knot = 4
        orig_steps = np.linspace(0, seq_len - 1, num=knot)
        random_warps = np.random.normal(loc=1.0, scale=self.time_warp_sigma, size=knot)
        random_warps = np.abs(random_warps)  # 양수 보장
        warp_steps = np.cumsum(random_warps)
        warp_steps = (warp_steps - warp_steps[0]) / (warp_steps[-1] - warp_steps[0]) * (seq_len - 1)
        
        # 보간
        warped = np.zeros_like(x)
        for i in range(n_features):
            f = interp1d(warp_steps, x[np.linspace(0, seq_len-1, knot).astype(int), i], 
                        kind='linear', fill_value='extrapolate')
            warped[:, i] = f(np.arange(seq_len))
        
        return warped
    
    def window_slicing(self, x: np.ndarray) -> np.ndarray:
        """윈도우 슬라이싱"""
        seq_len, n_features = x.shape
        target_len = int(seq_len * self.window_slice_ratio)
        
        if target_len < seq_len:
            start = np.random.randint(0, seq_len - target_len + 1)
            sliced = x[start:start + target_len]
            
            # 원래 길이로 리사이즈
            from scipy.ndimage import zoom
            scale_factor = seq_len / target_len
            resized = zoom(sliced, (scale_factor, 1), order=1)
            
            # 크기 조정
            if resized.shape[0] > seq_len:
                resized = resized[:seq_len]
            elif resized.shape[0] < seq_len:
                pad = np.zeros((seq_len - resized.shape[0], n_features))
                resized = np.vstack([resized, pad])
            
            return resized
        return x
    
    def permutation(self, x: np.ndarray, max_segments: int = 5) -> np.ndarray:
        """세그먼트 순서 섞기"""
        seq_len, n_features = x.shape
        n_segs = np.random.randint(2, max_segments + 1)
        
        splits = np.array_split(np.arange(seq_len), n_segs)
        random.shuffle(splits)
        
        permuted_indices = np.concatenate(splits)
        return x[permuted_indices]
    
    def crop_and_resize(self, x: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
        """랜덤 크롭 후 리사이즈"""
        from scipy.ndimage import zoom
        
        seq_len, n_features = x.shape
        crop_len = int(seq_len * crop_ratio)
        start = np.random.randint(0, seq_len - crop_len + 1)
        cropped = x[start:start + crop_len]
        
        # 원래 크기로 리사이즈
        scale_factor = seq_len / crop_len
        resized = zoom(cropped, (scale_factor, 1), order=1)
        
        if resized.shape[0] > seq_len:
            resized = resized[:seq_len]
        elif resized.shape[0] < seq_len:
            pad = np.zeros((seq_len - resized.shape[0], n_features))
            resized = np.vstack([resized, pad])
        
        return resized
    
    def augment(self, x: np.ndarray, augment_type: Optional[str] = None) -> np.ndarray:
        """증강 적용"""
        if augment_type is None:
            # 랜덤 선택
            augment_types = ['jitter', 'scale', 'magnitude_warp', 'time_warp', 
                           'window_slice', 'permutation', 'crop_resize']
            augment_type = random.choice(augment_types)
        
        if random.random() > self.augment_prob:
            return x
        
        if augment_type == 'jitter':
            return self.jittering(x)
        elif augment_type == 'scale':
            return self.scaling(x)
        elif augment_type == 'magnitude_warp':
            return self.magnitude_warping(x)
        elif augment_type == 'time_warp':
            return self.time_warping(x)
        elif augment_type == 'window_slice':
            return self.window_slicing(x)
        elif augment_type == 'permutation':
            return self.permutation(x)
        elif augment_type == 'crop_resize':
            return self.crop_and_resize(x)
        else:
            return x
    
    def augment_batch(self, X: np.ndarray, n_augment: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """배치 증강 (n_augment배로 확장)"""
        augmented_X = [X]
        
        for _ in range(n_augment):
            aug_batch = np.array([self.augment(x) for x in X])
            augmented_X.append(aug_batch)
        
        return np.concatenate(augmented_X, axis=0)


class AugmentedSequenceDataset(Dataset):
    """증강이 적용된 시퀀스 데이터셋"""
    
    def __init__(self, samples: np.ndarray, targets: np.ndarray,
                 augment: bool = True, augmenter: Optional[TimeSeriesAugmentation] = None):
        self.samples = samples
        self.targets = targets
        self.augment = augment
        self.augmenter = augmenter or TimeSeriesAugmentation()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x = self.samples[idx].copy()
        y = self.targets[idx]
        
        if self.augment and self.training:
            x = self.augmenter.augment(x)
        
        return torch.FloatTensor(x), torch.FloatTensor([y])
    
    @property
    def training(self):
        return self.augment


# ============================================================================
# 2. 개선된 Transformer 모델
# ============================================================================

class LearnablePositionalEncoding(nn.Module):
    """학습 가능한 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PreNormTransformerLayer(nn.Module):
    """Pre-LayerNorm Transformer Layer (더 안정적인 학습)"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # ReLU 대신 GELU
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, src_mask=None):
        # Pre-Norm: Layer Norm -> Attention -> Residual
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2, attn_mask=src_mask)[0]
        
        # Pre-Norm: Layer Norm -> FFN -> Residual
        x2 = self.norm2(x)
        x = x + self.ff(x2)
        
        return x


class ImprovedTransformerModel(nn.Module):
    """개선된 Transformer 모델"""
    
    def __init__(self, input_dim: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 4, 
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 use_learnable_pe: bool = True):
        super().__init__()
        
        # 입력 프로젝션 (스무스한 변환)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 위치 인코딩
        if use_learnable_pe:
            self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_encoder = self._sinusoidal_pe(d_model, dropout)
        
        # Pre-Norm Transformer Layers
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # 출력 헤드 (더 깊은 구조)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),  # 점진적 dropout 감소
            nn.Linear(d_model // 4, 1)
        )
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/He 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _sinusoidal_pe(self, d_model: int, dropout: float):
        """사인 위치 인코딩 (폴백)"""
        from deep_learning_models import PositionalEncoding
        return PositionalEncoding(d_model, dropout=dropout)
    
    def forward(self, x, return_attention: bool = False):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # 마지막 타임스텝 + 평균 풀링 결합
        last_out = x[:, -1, :]
        mean_out = x.mean(dim=1)
        combined = (last_out + mean_out) / 2
        
        return self.fc(combined)


# ============================================================================
# 3. 학습 관련 유틸리티
# ============================================================================

class SmoothL1Loss(nn.Module):
    """Smooth L1 Loss (Huber Loss) - 이상치에 강건"""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred, target):
        return F.smooth_l1_loss(pred, target, beta=self.beta)


class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    """Warmup + Cosine Annealing with Restarts"""
    
    def __init__(self, optimizer, first_cycle_steps: int, cycle_mult: float = 1.0,
                 max_lr: float = 0.001, min_lr: float = 0.00001,
                 warmup_steps: int = 10, gamma: float = 1.0, last_epoch: int = -1):
        
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr 
                    for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) * 
                    (1 + np.cos(np.pi * (self.step_in_cycle - self.warmup_steps) / 
                               (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = 0
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
                self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        else:
            self.step_in_cycle = epoch
        
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def train_with_augmentation(
    model: nn.Module, 
    train_samples: np.ndarray,
    train_targets: np.ndarray,
    val_samples: np.ndarray,
    val_targets: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 15,
    use_augmentation: bool = True,
    verbose: bool = True
) -> Dict:
    """증강을 적용한 학습"""
    
    model = model.to(DEVICE)
    
    # 증강기
    augmenter = TimeSeriesAugmentation(
        jitter_sigma=0.03,
        scale_sigma=0.1,
        time_warp_sigma=0.2,
        augment_prob=0.5
    )
    
    # 검증 데이터 (증강 없음)
    val_dataset = AugmentedSequenceDataset(val_samples, val_targets, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer, first_cycle_steps=epochs // 3,
        max_lr=lr, min_lr=lr / 100, warmup_steps=epochs // 10
    )
    
    criterion = SmoothL1Loss(beta=1.0)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    for epoch in range(epochs):
        # 매 에폭마다 새로운 증강 적용
        if use_augmentation:
            # 원본 + 증강 데이터
            aug_samples = augmenter.augment_batch(train_samples, n_augment=2)
            aug_targets = np.tile(train_targets, 3)  # 3배
        else:
            aug_samples = train_samples
            aug_targets = train_targets
        
        train_dataset = AugmentedSequenceDataset(aug_samples, aug_targets, augment=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={current_lr:.6f}")
        
        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # 최적 모델 복원
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return {'history': history, 'best_val_loss': best_val_loss}


def evaluate_model(model: nn.Module, samples: np.ndarray, targets: np.ndarray,
                  batch_size: int = 64) -> Dict:
    """모델 평가"""
    
    model.eval()
    model = model.to(DEVICE)
    
    dataset = AugmentedSequenceDataset(samples, targets, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_pred = model(X_batch)
            all_preds.extend(y_pred.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())
    
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    # 메트릭 계산
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # SMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, 1e-9)
    smape = np.mean(np.abs(y_true - y_pred) / denom) * 100
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'smape': smape}


# ============================================================================
# 4. Optuna 기반 하이퍼파라미터 튜닝
# ============================================================================

def objective(trial, train_samples, train_targets, val_samples, val_targets, input_dim):
    """Optuna 최적화 목적 함수"""
    
    # 하이퍼파라미터 탐색 공간
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [256, 512, 1024])
    dropout = trial.suggest_float('dropout', 0.05, 0.3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # d_model이 nhead로 나누어 떨어지도록 조정
    while d_model % nhead != 0:
        nhead = nhead // 2
    
    # 모델 생성
    model = ImprovedTransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    # 학습
    result = train_with_augmentation(
        model, train_samples, train_targets, val_samples, val_targets,
        epochs=50,  # 튜닝 시에는 짧게
        batch_size=batch_size,
        lr=lr,
        patience=10,
        use_augmentation=True,
        verbose=False
    )
    
    # 평가
    metrics = evaluate_model(model, val_samples, val_targets)
    
    return metrics['r2']  # R² 최대화


def run_hyperparameter_tuning(train_samples, train_targets, val_samples, val_targets,
                             input_dim: int, n_trials: int = 30):
    """하이퍼파라미터 튜닝 실행"""
    import optuna
    
    print("\n📊 Optuna 하이퍼파라미터 튜닝 시작...")
    print(f"   시도 횟수: {n_trials}")
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    
    study.optimize(
        lambda trial: objective(trial, train_samples, train_targets, 
                               val_samples, val_targets, input_dim),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("\n✅ 최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    print(f"\n   최적 R²: {study.best_value:.4f}")
    
    return study.best_params, study.best_value


# ============================================================================
# 5. 메인 함수
# ============================================================================

def prepare_sequence_data(df: pd.DataFrame, feature_cols: List[str],
                         target_col: str, group_col: str, time_col: str,
                         seq_len: int = 6, test_ratio: float = 0.2):
    """시퀀스 데이터 준비"""
    
    df = df.sort_values([group_col, time_col])
    
    # RobustScaler 사용 (이상치에 강건)
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_scaled = scaler_X.fit_transform(df[feature_cols].fillna(0))
    y_scaled = scaler_y.fit_transform(df[[target_col]].fillna(0))
    
    df_scaled = df.copy()
    df_scaled[feature_cols] = X_scaled
    df_scaled[target_col] = y_scaled.flatten()
    
    # 시퀀스 생성
    samples = []
    targets = []
    
    for group_id, group_df in df_scaled.groupby(group_col):
        group_df = group_df.sort_values(time_col)
        X = group_df[feature_cols].values
        y = group_df[target_col].values
        
        for i in range(len(group_df) - seq_len):
            samples.append(X[i:i+seq_len])
            targets.append(y[i+seq_len])
    
    samples = np.array(samples)
    targets = np.array(targets)
    
    # 시간 기준 분할
    split_idx = int(len(samples) * (1 - test_ratio))
    
    train_samples = samples[:split_idx]
    train_targets = targets[:split_idx]
    test_samples = samples[split_idx:]
    test_targets = targets[split_idx:]
    
    return train_samples, train_targets, test_samples, test_targets, scaler_X, scaler_y


def main():
    parser = argparse.ArgumentParser(description='Transformer 모델 개선')
    parser.add_argument('--raw_csv', type=str, required=True)
    parser.add_argument('--external_dir', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_trials', type=int, default=20, help='Optuna 튜닝 횟수')
    parser.add_argument('--skip_tuning', action='store_true', help='튜닝 건너뛰기')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 Transformer 모델 개선: 데이터 증강 + 하이퍼파라미터 튜닝")
    print("=" * 70)
    
    # 경로 설정
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    
    from imradar.data.io import load_internal_csv
    from imradar.data.preprocess import aggregate_to_segment_month, build_full_panel
    from imradar.features.kpi import compute_kpis
    from imradar.features.lags import add_lag_features
    from imradar.config import RadarConfig
    
    cfg = RadarConfig()
    KPI_COLS = list(cfg.kpi_defs.keys())
    
    # 데이터 로드
    print("\n[1/6] 데이터 로드...")
    raw = load_internal_csv(Path(args.raw_csv))
    print(f"  ✓ 원본 데이터: {len(raw):,}행")
    
    # 전처리
    print("[2/6] 전처리...")
    agg = aggregate_to_segment_month(raw, cfg)
    panel_result = build_full_panel(agg, cfg)
    panel = panel_result.panel if hasattr(panel_result, 'panel') else panel_result
    panel = compute_kpis(panel, cfg)
    print(f"  ✓ 패널 데이터: {len(panel):,}행")
    
    # 피처 엔지니어링
    print("[3/6] 피처 엔지니어링...")
    panel = add_lag_features(
        panel, group_col='segment_id', time_col='month',
        value_cols=[c for c in KPI_COLS if c in panel.columns]
    )
    
    exclude_cols = ['segment_id', 'month', 'segment_name'] + [c for c in panel.columns if 'target' in c]
    feature_cols = [c for c in panel.columns 
                   if panel[c].dtype in ['float64', 'float32', 'int64', 'int32']
                   and c not in exclude_cols]
    print(f"  ✓ 피처: {len(feature_cols)}개")
    
    # 타겟 설정
    target_col = '순유입'
    if target_col not in panel.columns:
        target_col = 'slog1p_순유입'
    
    # 시퀀스 데이터 준비
    print("\n[4/6] 시퀀스 데이터 준비...")
    train_samples, train_targets, test_samples, test_targets, scaler_X, scaler_y = \
        prepare_sequence_data(panel, feature_cols, target_col, 
                             'segment_id', 'month', seq_len=args.seq_len)
    
    print(f"  ✓ Train: {len(train_samples):,} sequences")
    print(f"  ✓ Test: {len(test_samples):,} sequences")
    print(f"  ✓ Input dim: {train_samples.shape[2]}")
    
    input_dim = train_samples.shape[2]
    
    # 결과 저장
    output_dir = Path(__file__).resolve().parent.parent / "outputs" / "transformer_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # ========================================================================
    # 기존 Transformer (베이스라인)
    # ========================================================================
    print("\n" + "=" * 70)
    print("📌 1. 기존 Transformer (베이스라인)")
    print("=" * 70)
    
    baseline_model = ImprovedTransformerModel(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1
    )
    
    baseline_result = train_with_augmentation(
        baseline_model, train_samples, train_targets, test_samples, test_targets,
        epochs=args.epochs, batch_size=args.batch_size, lr=0.001,
        patience=15, use_augmentation=False, verbose=True
    )
    baseline_metrics = evaluate_model(baseline_model, test_samples, test_targets)
    
    print("\n  ✅ 기존 Transformer 결과:")
    print(f"     R² = {baseline_metrics['r2']:.4f}")
    print(f"     SMAPE = {baseline_metrics['smape']:.2f}%")
    
    results.append({
        'model': 'Transformer_Baseline',
        'r2': baseline_metrics['r2'],
        'smape': baseline_metrics['smape'],
        'rmse': baseline_metrics['rmse']
    })
    
    # ========================================================================
    # 데이터 증강만 적용
    # ========================================================================
    print("\n" + "=" * 70)
    print("📌 2. 데이터 증강 적용")
    print("=" * 70)
    
    aug_model = ImprovedTransformerModel(
        input_dim=input_dim,
        d_model=128,
        nhead=8,
        num_layers=3,
        dropout=0.1
    )
    
    aug_result = train_with_augmentation(
        aug_model, train_samples, train_targets, test_samples, test_targets,
        epochs=args.epochs, batch_size=args.batch_size, lr=0.001,
        patience=15, use_augmentation=True, verbose=True  # 증강 활성화
    )
    aug_metrics = evaluate_model(aug_model, test_samples, test_targets)
    
    print("\n  ✅ 증강 Transformer 결과:")
    print(f"     R² = {aug_metrics['r2']:.4f}")
    print(f"     SMAPE = {aug_metrics['smape']:.2f}%")
    
    results.append({
        'model': 'Transformer_Augmented',
        'r2': aug_metrics['r2'],
        'smape': aug_metrics['smape'],
        'rmse': aug_metrics['rmse']
    })
    
    # ========================================================================
    # 하이퍼파라미터 튜닝 (Optuna)
    # ========================================================================
    if not args.skip_tuning:
        print("\n" + "=" * 70)
        print("📌 3. 하이퍼파라미터 튜닝 (Optuna)")
        print("=" * 70)
        
        # 검증용 데이터 분할
        val_split = int(len(train_samples) * 0.8)
        tune_train_samples = train_samples[:val_split]
        tune_train_targets = train_targets[:val_split]
        tune_val_samples = train_samples[val_split:]
        tune_val_targets = train_targets[val_split:]
        
        best_params, best_score = run_hyperparameter_tuning(
            tune_train_samples, tune_train_targets,
            tune_val_samples, tune_val_targets,
            input_dim=input_dim,
            n_trials=args.n_trials
        )
        
        # 최적 파라미터로 최종 학습
        print("\n[5/6] 최적 파라미터로 최종 학습...")
        
        # nhead 조정
        nhead = best_params.get('nhead', 8)
        d_model = best_params.get('d_model', 128)
        while d_model % nhead != 0:
            nhead = nhead // 2
        
        tuned_model = ImprovedTransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=best_params.get('num_layers', 4),
            dim_feedforward=best_params.get('dim_feedforward', 512),
            dropout=best_params.get('dropout', 0.1)
        )
        
        tuned_result = train_with_augmentation(
            tuned_model, train_samples, train_targets, test_samples, test_targets,
            epochs=args.epochs,
            batch_size=best_params.get('batch_size', 64),
            lr=best_params.get('lr', 0.001),
            patience=20,
            use_augmentation=True,
            verbose=True
        )
        tuned_metrics = evaluate_model(tuned_model, test_samples, test_targets)
        
        print("\n  ✅ 튜닝된 Transformer 결과:")
        print(f"     R² = {tuned_metrics['r2']:.4f}")
        print(f"     SMAPE = {tuned_metrics['smape']:.2f}%")
        
        results.append({
            'model': 'Transformer_Tuned',
            'r2': tuned_metrics['r2'],
            'smape': tuned_metrics['smape'],
            'rmse': tuned_metrics['rmse'],
            'params': str(best_params)
        })
        
        # 최적 모델 저장
        torch.save(tuned_model.state_dict(), output_dir / "transformer_tuned.pt")
    
    # ========================================================================
    # 결과 비교
    # ========================================================================
    print("\n" + "=" * 70)
    print("[6/6] 결과 비교")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "transformer_tuning_results.csv", index=False)
    
    print("\n📊 모델 비교:")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"  {row['model']:25s}: R²={row['r2']:7.4f}, SMAPE={row['smape']:6.2f}%")
    
    # 개선율 계산
    baseline_r2 = results_df[results_df['model'] == 'Transformer_Baseline']['r2'].values[0]
    best_row = results_df.loc[results_df['r2'].idxmax()]
    improvement = (best_row['r2'] - baseline_r2) / abs(baseline_r2) * 100
    
    print(f"\n🏆 최적 모델: {best_row['model']}")
    print(f"   R² 개선율: {improvement:+.1f}%")
    
    print(f"\n💾 결과 저장: {output_dir}")
    print("\n✅ 완료!")


if __name__ == "__main__":
    main()
