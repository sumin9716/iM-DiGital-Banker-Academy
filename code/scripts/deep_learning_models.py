"""
딥러닝 기반 예측 모델 개선

전략:
1. 순유입 (R² 0.34) → LSTM + Transformer 집중 개선
2. 기타 KPI (R² > 0.85) → 기존 ML 앙상블 유지
3. 하이브리드 앙상블: ML + DL 결합

모델 구성:
- LSTM: 시계열 패턴 학습
- Transformer: Self-Attention 기반 장기 의존성
- TFT (Temporal Fusion Transformer): 해석 가능한 시계열 예측
"""

import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# 디바이스 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# 데이터셋 클래스

class TimeSeriesDataset(Dataset):
    """시계열 데이터셋"""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 6):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SequenceDataset(Dataset):
    """시퀀스 기반 시계열 데이터셋"""

    def __init__(self, data: pd.DataFrame, feature_cols: List[str],
                 target_col: str, group_col: str, time_col: str,
                 seq_len: int = 6):
        self.seq_len = seq_len
        self.samples = []
        self.targets = []

        # 그룹별로 시퀀스 생성
        for group_id, group_df in data.groupby(group_col):
            group_df = group_df.sort_values(time_col)
            X = group_df[feature_cols].values
            y = group_df[target_col].values

            # 시퀀스 슬라이딩 윈도우
            for i in range(len(group_df) - seq_len):
                self.samples.append(X[i:i+seq_len])
                self.targets.append(y[i+seq_len])

        self.samples = np.array(self.samples)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.samples[idx]),
                torch.FloatTensor([self.targets[idx]]))


# LSTM 모델

class LSTMModel(nn.Module):
    """LSTM 기반 시계열 예측 모델"""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 마지막 타임스텝의 출력 사용
        last_out = lstm_out[:, -1, :]

        return self.fc(last_out)


# Transformer 모델

class PositionalEncoding(nn.Module):
    """위치 인코딩"""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer 기반 시계열 예측 모델"""

    def __init__(self, input_dim: int, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # 마지막 타임스텝 출력
        x = x[:, -1, :]

        return self.fc(x)


# Temporal Fusion Transformer (Simplified)

class GatedResidualNetwork(nn.Module):
    """GRN: Gated Residual Network"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: Optional[int] = None):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)

        self.gate = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Skip connection
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x

        x = self.fc1(x)
        if context is not None and self.context_dim is not None:
            x = x + self.context_fc(context)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # Gating
        gate = torch.sigmoid(self.gate(x))
        x = gate * x

        # Add & Norm
        x = self.layer_norm(x + residual)

        return x


class TemporalFusionTransformer(nn.Module):
    """Simplified Temporal Fusion Transformer"""

    def __init__(self, num_features: int, num_static: int = 0,
                 hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Variable Selection Network (simplified)
        self.var_selection = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Softmax(dim=-1)
        )

        # Feature embedding
        self.feature_embedding = nn.Linear(num_features, hidden_dim)

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # GRN for post-attention
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Output
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch_size, seq_len, num_features = x.shape

        # Variable selection (apply to each timestep)
        var_weights = self.var_selection(x.mean(dim=1))  # (batch, features)
        x = x * var_weights.unsqueeze(1)  # (batch, seq_len, features)

        # Feature embedding
        x = self.feature_embedding(x)  # (batch, seq_len, hidden_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Self-attention
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)

        # Residual connection
        x = lstm_out + attn_out

        # GRN
        x = self.grn(x[:, -1, :])  # Use last timestep

        # Output
        return self.output_fc(x)


# 학습 및 평가 함수

def train_model(model: nn.Module, train_loader: DataLoader,
                val_loader: DataLoader, epochs: int = 100,
                lr: float = 0.001, patience: int = 10,
                verbose: bool = True) -> Dict:
    """모델 학습"""

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.view(-1, 1))
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
                loss = criterion(y_pred, y_batch.view(-1, 1))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # 최적 모델 복원
    if best_model_state:
        model.load_state_dict(best_model_state)

    return {'history': history, 'best_val_loss': best_val_loss}


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict:
    """모델 평가"""

    model.eval()
    model = model.to(DEVICE)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
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
    smape = np.mean(np.abs(y_true - y_pred) / denom)

    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'smape': smape}


# 하이브리드 앙상블

class HybridEnsemble:
    """ML + DL 하이브리드 앙상블"""

    def __init__(self, ml_model, dl_model: nn.Module,
                 ml_weight: float = 0.5, dl_weight: float = 0.5):
        self.ml_model = ml_model
        self.dl_model = dl_model
        self.ml_weight = ml_weight
        self.dl_weight = dl_weight

    def predict(self, X_ml: np.ndarray, X_dl: torch.Tensor) -> np.ndarray:
        """앙상블 예측"""

        # ML 예측
        ml_pred = self.ml_model.predict(X_ml)

        # DL 예측
        self.dl_model.eval()
        with torch.no_grad():
            X_dl = X_dl.to(DEVICE)
            dl_pred = self.dl_model(X_dl).cpu().numpy().flatten()

        # 가중 평균
        ensemble_pred = self.ml_weight * ml_pred + self.dl_weight * dl_pred

        return ensemble_pred

    def optimize_weights(self, X_ml: np.ndarray, X_dl: torch.Tensor,
                        y_true: np.ndarray) -> Tuple[float, float]:
        """최적 가중치 탐색"""

        # ML 예측
        ml_pred = self.ml_model.predict(X_ml)

        # DL 예측
        self.dl_model.eval()
        with torch.no_grad():
            X_dl = X_dl.to(DEVICE)
            dl_pred = self.dl_model(X_dl).cpu().numpy().flatten()

        # 그리드 서치로 최적 가중치 찾기
        best_r2 = -float('inf')
        best_weights = (0.5, 0.5)

        for ml_w in np.arange(0.0, 1.05, 0.1):
            dl_w = 1.0 - ml_w
            ensemble_pred = ml_w * ml_pred + dl_w * dl_pred
            r2 = r2_score(y_true, ensemble_pred)

            if r2 > best_r2:
                best_r2 = r2
                best_weights = (ml_w, dl_w)

        self.ml_weight, self.dl_weight = best_weights
        return best_weights


# 데이터 준비 함수

def prepare_sequence_data(df: pd.DataFrame, feature_cols: List[str],
                         target_col: str, group_col: str, time_col: str,
                         seq_len: int = 6, test_ratio: float = 0.2):
    """시퀀스 데이터 준비"""

    # 정렬
    df = df.sort_values([group_col, time_col])

    # 스케일링
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(df[feature_cols].fillna(0))
    y_scaled = scaler_y.fit_transform(df[[target_col]].fillna(0))

    df_scaled = df.copy()
    df_scaled[feature_cols] = X_scaled
    df_scaled[target_col] = y_scaled.flatten()

    # 시간 기준 분할
    unique_times = sorted(df[time_col].unique())
    split_idx = int(len(unique_times) * (1 - test_ratio))
    train_times = unique_times[:split_idx]
    test_times = unique_times[split_idx:]

    train_df = df_scaled[df_scaled[time_col].isin(train_times)]
    test_df = df_scaled[df_scaled[time_col].isin(test_times)]

    # 데이터셋 생성
    train_dataset = SequenceDataset(train_df, feature_cols, target_col,
                                    group_col, time_col, seq_len)
    test_dataset = SequenceDataset(test_df, feature_cols, target_col,
                                   group_col, time_col, seq_len)

    return train_dataset, test_dataset, scaler_X, scaler_y


# 메인 함수

def main():
    parser = argparse.ArgumentParser(description='딥러닝 모델 개선')
    parser.add_argument('--raw_csv', type=str, required=True)
    parser.add_argument('--external_dir', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--kpis', type=str, default='순유입', help='Comma-separated KPI names')
    args = parser.parse_args()

    print("=" * 60)
    print("딥러닝 모델 개선 시작")
    print("=" * 60)

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
    print("\n[1/5] 데이터 로드...")
    raw = load_internal_csv(Path(args.raw_csv))
    print(f"  원본 데이터: {len(raw):,}행")

    # 전처리
    print("[2/5] 전처리...")
    agg = aggregate_to_segment_month(raw, cfg)
    panel_result = build_full_panel(agg, cfg)
    panel = panel_result.panel if hasattr(panel_result, 'panel') else panel_result
    panel = compute_kpis(panel, cfg)
    print(f"  패널 데이터: {len(panel):,}행")

    # 피처 엔지니어링
    print("[3/5] 피처 엔지니어링...")
    panel = add_lag_features(
        panel,
        group_col='segment_id',
        time_col='month',
        value_cols=[c for c in KPI_COLS if c in panel.columns]
    )

    # 수치형 피처만 선택
    exclude_cols = ['segment_id', 'month', 'segment_name'] + [c for c in panel.columns if 'target' in c]
    feature_cols = [c for c in panel.columns
                   if panel[c].dtype in ['float64', 'float32', 'int64', 'int32']
                   and c not in exclude_cols]
    print(f"  피처: {len(feature_cols)}개")

    # 결과 저장
    results = []
    output_dir = Path(__file__).resolve().parent.parent / "outputs" / "deep_learning"
    output_dir.mkdir(parents=True, exist_ok=True)

    kpis = [k.strip() for k in args.kpis.split(",") if k.strip()]
    for kpi in kpis:
        print("\n" + "=" * 60)
        print(f"{kpi} 딥러닝 모델 학습")
        print("=" * 60)

        target_col = kpi
        if target_col not in panel.columns and kpi == "순유입":
            target_col = "slog1p_순유입"

        if target_col not in panel.columns:
            print(f"  [경고] KPI 컬럼 없음: {kpi} (스킵)")
            continue

        # 데이터 준비
        print("\n[4/5] 시퀀스 데이터 준비...")
        train_dataset, test_dataset, scaler_X, scaler_y = prepare_sequence_data(
            panel, feature_cols, target_col,
            'segment_id', 'month',
            seq_len=args.seq_len
        )

        print(f"  Train: {len(train_dataset):,} sequences")
        print(f"  Test: {len(test_dataset):,} sequences")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # LSTM 모델
        print("\n[5/5] 모델 학습...")
        print("\n  LSTM 모델 학습 중...")
        lstm_model = LSTMModel(
            input_dim=len(feature_cols),
            hidden_dim=128,
            num_layers=2,
            dropout=0.2
        )

        train_model(
            lstm_model, train_loader, test_loader,
            epochs=args.epochs, lr=0.001, patience=15, verbose=True
        )
        lstm_metrics = evaluate_model(lstm_model, test_loader)
        print("\n  LSTM 결과:")
        print(f"     R² = {lstm_metrics['r2']:.4f}")
        print(f"     SMAPE = {lstm_metrics['smape']:.4f}")

        results.append({
            'kpi': kpi, 'model': 'LSTM',
            'r2': lstm_metrics['r2'], 'smape': lstm_metrics['smape'],
            'rmse': lstm_metrics['rmse']
        })

        # Transformer 모델
        print("\n  Transformer 모델 학습 중...")
        transformer_model = TransformerModel(
            input_dim=len(feature_cols),
            d_model=128,
            nhead=8,
            num_layers=3,
            dropout=0.1
        )

        train_model(
            transformer_model, train_loader, test_loader,
            epochs=args.epochs, lr=0.0005, patience=15, verbose=True
        )
        transformer_metrics = evaluate_model(transformer_model, test_loader)
        print("\n  Transformer 결과:")
        print(f"     R² = {transformer_metrics['r2']:.4f}")
        print(f"     SMAPE = {transformer_metrics['smape']:.4f}")

        results.append({
            'kpi': kpi, 'model': 'Transformer',
            'r2': transformer_metrics['r2'], 'smape': transformer_metrics['smape'],
            'rmse': transformer_metrics['rmse']
        })

        # TFT 모델
        print("\n  Temporal Fusion Transformer 학습 중...")
        tft_model = TemporalFusionTransformer(
            num_features=len(feature_cols),
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )

        train_model(
            tft_model, train_loader, test_loader,
            epochs=args.epochs, lr=0.001, patience=15, verbose=True
        )
        tft_metrics = evaluate_model(tft_model, test_loader)
        print("\n  TFT 결과:")
        print(f"     R² = {tft_metrics['r2']:.4f}")
        print(f"     SMAPE = {tft_metrics['smape']:.4f}")

        results.append({
            'kpi': kpi, 'model': 'TFT',
            'r2': tft_metrics['r2'], 'smape': tft_metrics['smape'],
            'rmse': tft_metrics['rmse']
        })

        if kpi == "순유입":
            # 비교용 고정값: 이 스크립트를 만들 당시 ML 모델의 순유입 평가 결과를 옮겨 적은 것으로
            # 여기서 다시 계산하지 않는다. DL 지표는 원 단위 순유입을 표준화한 값으로, ML 지표는
            # slog1p 변환 값으로 계산하므로 두 R²는 직접 비교할 수 없다.
            print("\n  기존 ML 모델 (CatBoost 앙상블):")
            print("     R² = 0.3403")
            print("     SMAPE = 1.70%")

            results.append({
                'kpi': kpi, 'model': 'ML_Ensemble',
                'r2': 0.3403, 'smape': 1.70, 'rmse': 2.57
            })

        # 모델 저장
        kpi_tag = kpi
        torch.save(lstm_model.state_dict(), output_dir / f"lstm_model_{kpi_tag}.pt")
        torch.save(transformer_model.state_dict(), output_dir / f"transformer_model_{kpi_tag}.pt")
        torch.save(tft_model.state_dict(), output_dir / f"tft_model_{kpi_tag}.pt")

        model_params = {
            "LSTM": {
                "input_dim": len(feature_cols),
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.2,
            },
            "Transformer": {
                "input_dim": len(feature_cols),
                "d_model": 128,
                "nhead": 8,
                "num_layers": 3,
                "dim_feedforward": 256,
                "dropout": 0.1,
            },
            "TFT": {
                "num_features": len(feature_cols),
                "num_static": 0,
                "hidden_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
            },
        }
        best_model = max(
            [("LSTM", lstm_metrics), ("Transformer", transformer_metrics), ("TFT", tft_metrics)],
            key=lambda x: x[1]["r2"],
        )
        artifacts = {
            "feature_cols": feature_cols,
            "target_col": target_col,
            "seq_len": args.seq_len,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "best_model": best_model[0],
            "model_params": model_params,
            "kpi": kpi,
        }
        with open(output_dir / f"deep_learning_artifacts_{kpi_tag}.pkl", "wb") as f:
            pickle.dump(artifacts, f)

        if kpi == "순유입":
            # Backward-compatible artifacts
            with open(output_dir / "deep_learning_artifacts.pkl", "wb") as f:
                pickle.dump(artifacts, f)

    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "deep_learning_results.csv", index=False)

    print("\n" + "=" * 60)
    print("딥러닝 모델 개선 완료!")
    print("=" * 60)

    print("\n모델 비교 결과:")
    print(results_df.to_string(index=False))

    # 최적 모델 선택
    best_model = results_df.loc[results_df['r2'].idxmax()]
    print(f"\n최적 모델: {best_model['model']} (R² = {best_model['r2']:.4f})")
    print(f"\n결과 저장: {output_dir}")
    print("  모델/아티팩트 저장 완료")


if __name__ == "__main__":
    main()
