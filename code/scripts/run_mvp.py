from __future__ import annotations

import argparse
import os
import sys
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow "python scripts/run_mvp.py" from repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, REPO_ROOT)

from imradar.config import RadarConfig
from imradar.data.io import load_internal_csv
from imradar.data.preprocess import aggregate_to_segment_month, build_full_panel
from imradar.data.external import load_all_external_data, merge_external_to_panel, add_macro_lag_features
from imradar.features.kpi import compute_kpis
from imradar.features.lags import add_lag_features, add_rolling_features, add_change_features
from imradar.models.forecast_lgbm import (
    ForecastModel,
    train_lgbm_forecast,
    evaluate_forecast,
    predict_with_model,
)
from imradar.models.residual_alerts import score_residual_alerts, build_watchlist
from imradar.models.hierarchical import reconcile_forecasts
from imradar.explain.lgbm_contrib import local_feature_contrib, driver_sentence
from imradar.recommend.rules import recommend_actions
from imradar.recommend.enhanced_rules import recommend_enhanced_actions  # NEW: 개선된 액션 엔진
from imradar.pipelines.dashboard import save_overview_dashboard, save_watchlist_dashboard, save_growth_dashboard, save_fx_radar_dashboard
from imradar.pipelines.executive_summary import generate_executive_summary_html
from imradar.pipelines.report_pdf import generate_monthly_onepager
from imradar.models.fx_regime import compute_fx_regime_from_external, compute_fx_segment_scores

try:
    from imradar.models.forecast_ensemble import (
        train_ensemble_forecast,
        evaluate_ensemble,
        predict_ensemble,
    )
    ENSEMBLE_AVAILABLE = True
except Exception:
    ENSEMBLE_AVAILABLE = False


def _month_add(m: pd.Timestamp, k: int) -> pd.Timestamp:
    return (m.to_period("M") + k).to_timestamp()


def inverse_transform(kpi: str, y_hat: np.ndarray) -> np.ndarray:
    # y_hat is transformed prediction
    if kpi == "순유입":
        # signed expm1
        return np.sign(y_hat) * (np.expm1(np.abs(y_hat)))
    else:
        return np.expm1(np.maximum(y_hat, 0.0))


def _model_key(kpi: str, horizon: int, model_type: str) -> str:
    return f"{model_type}_{kpi}_h{horizon}.pkl"


def _predict_model(model_entry: dict, df: pd.DataFrame) -> np.ndarray:
    model_type = model_entry["type"]
    model = model_entry["model"]
    if model_type == "ensemble":
        return predict_ensemble(model, df)
    return predict_with_model(model, df)


def _load_best_model_map(path: str) -> Dict[Tuple[str, int], str]:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if not {"kpi", "horizon", "model_type", "r2"}.issubset(df.columns):
        return {}
    best = (
        df.sort_values("r2", ascending=False)
        .groupby(["kpi", "horizon"])
        .head(1)
    )
    return {(row["kpi"], int(row["horizon"])): str(row["model_type"]) for _, row in best.iterrows()}


def _load_ml_r2_map(path: str) -> Dict[Tuple[str, int], Dict[str, float]]:
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if not {"kpi", "horizon", "model_type", "r2"}.issubset(df.columns):
        return {}
    df = df[df["model_type"].isin(["lgbm", "ensemble"])]
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for _, row in df.iterrows():
        key = (row["kpi"], int(row["horizon"]))
        out.setdefault(key, {})[str(row["model_type"])] = float(row["r2"])
    return out


def _as_lgbm_forecast(model_entry: dict, kpi: str, horizon: int) -> ForecastModel | None:
    if model_entry["type"] == "lgbm":
        return model_entry["model"]
    if model_entry["type"] != "ensemble":
        return None
    em = model_entry["model"]
    return ForecastModel(
        kpi=kpi,
        horizon=horizon,
        feature_cols=em.feature_cols,
        categorical_cols=em.categorical_cols,
        model=em.models["lgbm"],
        feature_importance=em.feature_importance,
        cv_scores=None,
        train_residual_std=em.train_residual_std,
    )


def _load_dl_artifacts(artifact_path: str) -> dict | None:
    if not os.path.exists(artifact_path):
        return None
    try:
        with open(artifact_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _build_latest_sequences(
    panel: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    group_col: str = "segment_id",
    time_col: str = "month",
) -> Tuple[np.ndarray, List[str]]:
    panel_sorted = panel.sort_values([group_col, time_col])
    sequences = []
    seg_ids = []
    for seg_id, group in panel_sorted.groupby(group_col):
        group = group.sort_values(time_col)
        if len(group) < seq_len:
            continue
        window = group.tail(seq_len)
        x = window[feature_cols].fillna(0.0).values
        sequences.append(x)
        seg_ids.append(seg_id)
    if not sequences:
        return np.zeros((0, seq_len, len(feature_cols))), []
    return np.stack(sequences, axis=0), seg_ids


def main():
    parser = argparse.ArgumentParser(description="Segment Radar MVP pipeline")
    parser.add_argument("--raw_csv", required=True, help="Path to internal raw CSV")
    parser.add_argument("--encoding", default="cp949", help="CSV encoding (default: cp949)")
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "outputs"), help="Output directory")
    parser.add_argument("--external_dir", default=None, help="Path to external data directory (optional)")
    parser.add_argument("--train_end", default=None, help="Train end month (YYYYMM), e.g., 202411. If not set, use all.")
    parser.add_argument("--test_start", default="202401", help="Backtest start month (YYYYMM)")
    parser.add_argument("--test_end", default="202412", help="Backtest end month (YYYYMM)")
    parser.add_argument("--topn_watchlist", type=int, default=50, help="Top-N watchlist size")
    parser.add_argument("--use_external", action="store_true", help="Use external macro data")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and load saved models")
    parser.add_argument(
        "--model_strategy",
        choices=["lgbm", "ensemble", "auto"],
        default="lgbm",
        help="Forecast model strategy (default: lgbm)",
    )
    parser.add_argument(
        "--model_selection_csv",
        default=os.path.join(REPO_ROOT, "outputs", "model_evaluation.csv"),
        help="CSV with per-KPI best model selection (used when --model_strategy=auto)",
    )
    parser.add_argument("--ensemble_tune", action="store_true", help="Enable Optuna tuning for ensemble")
    parser.add_argument("--ensemble_trials", type=int, default=30, help="Optuna trials for ensemble")
    parser.add_argument("--ensemble_timeout", type=int, default=300, help="Optuna timeout (sec) for ensemble")
    parser.add_argument("--ensemble_fs", action="store_true", help="Enable feature selection for ensemble")
    parser.add_argument("--ml_diff_threshold", type=float, default=0.01, help="R2 diff threshold for ML tie")
    parser.add_argument("--prefer_dl_when_close", action="store_true", help="Use DL when ML models are close")
    args = parser.parse_args()

    cfg = RadarConfig()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "forecasts"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "dashboard"), exist_ok=True)

    print("[1/8] Load raw data")
    raw = load_internal_csv(args.raw_csv, encoding=args.encoding)

    print("[2/8] Aggregate to segment-month")
    segm = aggregate_to_segment_month(raw, cfg)

    print("[3/8] Build full panel")
    panel_res = build_full_panel(segm, cfg)
    panel = panel_res.panel

    print("[4/8] Compute KPIs")
    panel = compute_kpis(panel, cfg)

    # ============== 외부 데이터 로드 및 병합 ==============
    external_data = None
    if args.use_external or args.external_dir:
        print("[4.5/8] Load and merge external macro data")
        external_dir = args.external_dir
        if external_dir is None:
            # 기본 경로: 프로젝트 루트의 '외부 데이터' 폴더
            potential_paths = [
                os.path.join(os.path.dirname(REPO_ROOT), "외부 데이터"),
                os.path.join(REPO_ROOT, "..", "외부 데이터"),
                os.path.join(REPO_ROOT, "data", "external"),
            ]
            for p in potential_paths:
                if os.path.isdir(p):
                    external_dir = p
                    break
        
        if external_dir and os.path.isdir(external_dir):
            print(f"  외부 데이터 디렉토리: {external_dir}")
            try:
                external_data = load_all_external_data(
                    external_dir, 
                    file_config=cfg.external_data_files
                )
                if external_data is not None and not external_data.empty:
                    panel = merge_external_to_panel(panel, external_data, month_col="month")
                    print(f"  외부 데이터 병합 완료: 패널 컬럼 수 {len(panel.columns)}")
            except Exception as e:
                print(f"  외부 데이터 로드 실패 (계속 진행): {e}")
        else:
            print(f"  외부 데이터 디렉토리를 찾을 수 없음: {external_dir}")

    # Select model target columns (transformed)
    model_targets: Dict[str, str] = {
        "예금총잔액": "log1p_예금총잔액",
        "대출총잔액": "log1p_대출총잔액",
        "카드총사용": "log1p_카드총사용",
        "디지털거래금액": "log1p_디지털거래금액",
        "순유입": "slog1p_순유입",
    }
    if "log1p_FX총액" in panel.columns:
        model_targets["FX총액"] = "log1p_FX총액"

    # Feature engineering on transformed KPI columns (better behaved)
    value_cols = list(model_targets.values()) + ["한도소진율", "디지털비중", "자동이체비중", "customer_count"]
    value_cols = [c for c in value_cols if c in panel.columns]

    print("[5/8] Feature engineering (lags/rolling/change)")
    panel_fe = panel.copy()
    panel_fe = add_lag_features(panel_fe, group_col="segment_id", time_col="month", value_cols=value_cols)
    panel_fe = add_rolling_features(panel_fe, group_col="segment_id", time_col="month", value_cols=value_cols)
    panel_fe = add_change_features(panel_fe, group_col="segment_id", time_col="month", value_cols=value_cols)

    # 거시경제 변수 Lag 피처 추가
    if external_data is not None and not external_data.empty:
        print("  거시경제 변수 Lag 피처 생성 중...")
        macro_cols_available = [c for c in cfg.macro_feature_cols if c in panel_fe.columns]
        if macro_cols_available:
            panel_fe = add_macro_lag_features(
                panel_fe, 
                macro_cols=macro_cols_available, 
                lags=cfg.macro_lag_periods
            )
            print(f"  거시경제 Lag 피처 추가 완료: {len(macro_cols_available)}개 변수 x {len(cfg.macro_lag_periods)}개 Lag")

    # Save panel for reuse (parquet if available, else CSV)
    panel_parquet = os.path.join(args.out_dir, "segment_panel.parquet")
    panel_csv = os.path.join(args.out_dir, "segment_panel.csv")
    try:
        panel_fe.to_parquet(panel_parquet, index=False)
        print(f"Saved panel: {panel_parquet}")
    except Exception as e:
        panel_fe.to_csv(panel_csv, index=False, encoding="utf-8-sig")
        print(f"Parquet not available ({e}). Saved panel as CSV: {panel_csv}")

    # Time boundaries
    def _to_ts(yyyymm: str) -> pd.Timestamp:
        y = int(yyyymm) // 100
        m = int(yyyymm) % 100
        return pd.Timestamp(year=y, month=m, day=1)

    train_end = _to_ts(args.train_end) if args.train_end else None
    test_start = _to_ts(args.test_start) if args.test_start else None
    test_end = _to_ts(args.test_end) if args.test_end else None
    if train_end is None and test_start is not None:
        # Prevent train/test leakage by defaulting train_end before test_start
        train_end = test_start - pd.DateOffset(months=1)

    # Train models
    categorical_cols = cfg.segment_keys  # static cats

    model_registry: Dict[Tuple[str, int], Dict[str, object]] = {}
    metrics_rows = []

    strategy = args.model_strategy
    best_model_map = {}
    ml_r2_map = {}
    if strategy == "auto":
        best_model_map = _load_best_model_map(args.model_selection_csv)
        ml_r2_map = _load_ml_r2_map(args.model_selection_csv)
        if not best_model_map:
            print("  [warn] model_selection_csv not found or invalid; falling back to LGBM.")
            strategy = "lgbm"
    if strategy in {"ensemble", "auto"} and not ENSEMBLE_AVAILABLE:
        print("  [warn] Ensemble dependencies not available, falling back to LGBM.")
        strategy = "lgbm"

    print(f"[6/8] Train forecasting models ({strategy})")
    for kpi, tgt_col in model_targets.items():
        if tgt_col not in panel_fe.columns:
            continue
        for h in cfg.horizons:
            ensemble_path = os.path.join(args.out_dir, "models", _model_key(kpi, h, "ensemble"))
            if strategy == "auto":
                preferred = best_model_map.get((kpi, h), "lgbm")
                use_ensemble = preferred == "ensemble"
            else:
                use_ensemble = strategy == "ensemble"
            if use_ensemble:
                if args.skip_train and not os.path.exists(ensemble_path):
                    use_ensemble = False
                elif args.skip_train and os.path.exists(ensemble_path):
                    with open(ensemble_path, "rb") as f:
                        em = pickle.load(f)
                else:
                    em = train_ensemble_forecast(
                        panel_fe,
                        kpi_col=tgt_col,
                        horizon=h,
                        categorical_cols=categorical_cols,
                        train_end_month=train_end,
                        tune_hyperparams=args.ensemble_tune,
                        n_trials=args.ensemble_trials,
                        tune_timeout=args.ensemble_timeout,
                        use_feature_selection=args.ensemble_fs,
                        verbose=False,
                    )
                    with open(ensemble_path, "wb") as f:
                        pickle.dump(em, f)
                    met = evaluate_ensemble(em, panel_fe, kpi_col=tgt_col, horizon=h, test_start_month=test_start, test_end_month=test_end)
                    if met:
                        met.update({"kpi": kpi, "target_col": tgt_col, "horizon": h})
                        metrics_rows.append(met)
                model_registry[(kpi, h)] = {"type": "ensemble", "model": em}
            else:
                model_path = os.path.join(args.out_dir, "models", _model_key(kpi, h, "lgbm"))
                if args.skip_train and os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        fm = pickle.load(f)
                else:
                    fm = train_lgbm_forecast(
                        panel_fe,
                        kpi_col=tgt_col,
                        horizon=h,
                        categorical_cols=categorical_cols,
                        train_end_month=train_end,
                    )
                    with open(model_path, "wb") as f:
                        pickle.dump(fm, f)
                    met = evaluate_forecast(panel_fe, fm, test_start_month=test_start, test_end_month=test_end)
                    met.update({"kpi": kpi, "target_col": tgt_col, "horizon": h})
                    metrics_rows.append(met)
                model_registry[(kpi, h)] = {"type": "lgbm", "model": fm}
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).sort_values(["kpi", "horizon"])
        metrics_csv = os.path.join(args.out_dir, "forecast_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
        print(f"Saved metrics: {metrics_csv}")

    # ============== 세그먼트별 스케일 팩터 계산 ==============
    # 학습 데이터에서 세그먼트별 평균 규모를 계산하여 예측 시 보정에 활용
    # 원본 KPI 값 (로그 변환 전)을 기준으로 계산
    scale_factors = {}
    for kpi in model_targets.keys():
        if kpi not in panel_fe.columns:
            continue
        # 최근 6개월 평균 실제값 계산 (원본 값)
        recent_months = sorted(panel_fe["month"].unique())[-6:]
        recent_data = panel_fe[panel_fe["month"].isin(recent_months)]
        seg_means = recent_data.groupby("segment_id")[kpi].mean()
        scale_factors[kpi] = seg_means.to_dict()

    # Forecast next 1~3 months from last observed month
    last_month = panel_fe["month"].max()
    print(f"[7/8] Forecast next horizons from last observed month={last_month.date()}")

    # ============== 딥러닝 순유입 예측 로드 (h=1) ==============
    dl_pred_maps: Dict[str, Dict[str, float]] = {}
    dl_model_names: Dict[str, str] = {}
    for kpi in model_targets.keys():
        dl_artifacts_path = os.path.join(
            args.out_dir, "deep_learning", f"deep_learning_artifacts_{kpi}.pkl"
        )
        if not os.path.exists(dl_artifacts_path) and kpi == "순유입":
            dl_artifacts_path = os.path.join(args.out_dir, "deep_learning", "deep_learning_artifacts.pkl")
        dl_artifacts = _load_dl_artifacts(dl_artifacts_path)
        if dl_artifacts is None:
            continue
        try:
            import torch
            from scripts.deep_learning_models import LSTMModel, TransformerModel, TemporalFusionTransformer

            feature_cols = dl_artifacts["feature_cols"]
            seq_len = int(dl_artifacts["seq_len"])
            model_name = str(dl_artifacts.get("best_model", "TFT"))
            model_params = dl_artifacts.get("model_params", {}).get(model_name, {})

            panel_dl = panel.copy()
            kpi_cols = [k for k in cfg.kpi_defs.keys() if k in panel_dl.columns]
            panel_dl = add_lag_features(panel_dl, group_col="segment_id", time_col="month", value_cols=kpi_cols)
            for c in feature_cols:
                if c not in panel_dl.columns:
                    panel_dl[c] = 0.0

            X_scaled = dl_artifacts["scaler_X"].transform(panel_dl[feature_cols].fillna(0.0))
            panel_dl_scaled = panel_dl.copy()
            panel_dl_scaled[feature_cols] = X_scaled

            X_seq, seg_ids = _build_latest_sequences(panel_dl_scaled, feature_cols, seq_len)
            if len(seg_ids) > 0:
                model_name_lower = model_name.lower()
                model_path = os.path.join(args.out_dir, "deep_learning", f"{model_name_lower}_model_{kpi}.pt")
                if not os.path.exists(model_path) and kpi == "순유입":
                    model_path = os.path.join(args.out_dir, "deep_learning", f"{model_name_lower}_model.pt")
                if model_name == "LSTM":
                    model = LSTMModel(**model_params)
                elif model_name == "Transformer":
                    model = TransformerModel(**model_params)
                else:
                    model = TemporalFusionTransformer(**model_params)

                if os.path.exists(model_path):
                    state = torch.load(model_path, map_location="cpu")
                    model.load_state_dict(state)
                    model.eval()
                    with torch.no_grad():
                        preds_scaled = model(torch.FloatTensor(X_seq)).cpu().numpy().reshape(-1, 1)
                    preds = dl_artifacts["scaler_y"].inverse_transform(preds_scaled).flatten()
                    dl_pred_maps[kpi] = dict(zip(seg_ids, preds))
                    dl_model_names[kpi] = model_name
                    print(f"  └─ {kpi} DL 예측 준비: {model_name} ({len(seg_ids)} segments)")
        except Exception as e:
            print(f"  └─ {kpi} DL 예측 로드 실패 (ML 사용): {e}")

    latest_rows = panel_fe[panel_fe["month"] == last_month].copy()
    forecasts = []
    for kpi, tgt_col in model_targets.items():
        for h in cfg.horizons:
            model_entry = model_registry.get((kpi, h))
            if model_entry is None:
                continue
            y_hat = _predict_model(model_entry, latest_rows)
            y_raw = inverse_transform(kpi, y_hat)
            
            # ============== 순유입 예측 개선: DL 실제 예측값 반영 (h=1) ==============
            if h == 1 and kpi in dl_pred_maps:
                use_dl = False
                if args.prefer_dl_when_close and strategy == "auto":
                    scores = ml_r2_map.get((kpi, h), {})
                    if "lgbm" in scores and "ensemble" in scores:
                        if abs(scores["lgbm"] - scores["ensemble"]) <= args.ml_diff_threshold:
                            use_dl = True
                if use_dl or kpi == "순유입":
                    seg_ids = latest_rows["segment_id"].values
                    dl_map = dl_pred_maps.get(kpi, {})
                    y_raw = np.array([dl_map.get(seg_id, y_raw[i]) for i, seg_id in enumerate(seg_ids)])
            
            # 스케일 보정: 예측값이 히스토리 대비 너무 작거나 크면 보정
            if kpi in scale_factors:
                seg_means = scale_factors[kpi]
                for i, seg_id in enumerate(latest_rows["segment_id"].values):
                    if seg_id in seg_means:
                        hist_mean = seg_means[seg_id]
                        if hist_mean > 1 and y_raw[i] > 0:
                            # 예측값이 히스토리 평균의 30% 미만이면 보정
                            if y_raw[i] < hist_mean * 0.3:
                                y_raw[i] = hist_mean * 0.8  # 히스토리 평균의 80%로 보정
                            # 예측값이 히스토리 평균의 300% 초과이면 보정  
                            elif y_raw[i] > hist_mean * 3.0:
                                y_raw[i] = hist_mean * 1.3  # 히스토리 평균의 130%로 보정
            
            pred_month = _month_add(last_month, h)
            tmp = pd.DataFrame({
                "segment_id": latest_rows["segment_id"].values,
                "pred_month": pred_month,
                kpi: y_raw,
            })
            forecasts.append(tmp)
    # Merge forecasts across KPI columns
    if forecasts:
        fc = forecasts[0]
        for t in forecasts[1:]:
            fc = fc.merge(t, on=["segment_id", "pred_month"], how="outer")
        
        # ============== 전체 합계 레벨 스케일 보정 ==============
        # 예측 합계가 히스토리 합계와 유사하도록 비례 조정
        last_month_data = panel_fe[panel_fe["month"] == last_month]
        for kpi in model_targets.keys():
            # KPI 컬럼 찾기 (suffix 포함)
            kpi_cols_in_fc = [c for c in fc.columns if c.startswith(kpi)]
            if not kpi_cols_in_fc:
                continue
            
            # 실제 합계
            actual_sum = last_month_data[kpi].sum()
            
            # 각 컬럼별로 보정
            for kpi_col in kpi_cols_in_fc:
                pred_sum = fc[kpi_col].sum()
                
                if pred_sum > 0 and actual_sum > 0:
                    ratio = actual_sum / pred_sum
                    if ratio > 1.3:  # 예측이 너무 낮음
                        scale_factor = min(ratio, 2.5) * 0.95
                        fc[kpi_col] *= scale_factor
                        print(f"     {kpi_col}: 스케일 보정 적용 (x{scale_factor:.2f})")
                    elif ratio < 0.7:  # 예측이 너무 높음
                        scale_factor = max(ratio, 0.5) * 1.05
                        fc[kpi_col] *= scale_factor
                        print(f"     {kpi_col}: 스케일 보정 적용 (x{scale_factor:.2f})")
        
        # ============== Hierarchical Reconciliation (MinT) ==============
        print("  └─ Applying Hierarchical Reconciliation (MinT)...")
        kpi_cols_to_reconcile = [c for c in ["예금총잔액", "대출총잔액", "카드총사용", "디지털거래금액", "FX총액"] if c in fc.columns]
        if kpi_cols_to_reconcile and hasattr(panel_res, 'segment_meta'):
            try:
                fc = reconcile_forecasts(
                    base_forecasts=fc,
                    segment_meta=panel_res.segment_meta,
                    kpi_cols=kpi_cols_to_reconcile,
                    historical_data=panel_fe,
                    method="mint",
                    month_col="pred_month",
                    segment_col="segment_id",
                )
                print(f"     Reconciled KPIs: {kpi_cols_to_reconcile}")
            except Exception as e:
                print(f"     Reconciliation skipped: {e}")
        
        # ============== 중복 컬럼 정리 (_x, _y 병합) ==============
        # merge 과정에서 생성된 중복 컬럼들 정리
        for kpi in model_targets.keys():
            # _x, _y, 원본 컬럼 확인
            cols_x = [c for c in fc.columns if c == f"{kpi}_x"]
            cols_y = [c for c in fc.columns if c == f"{kpi}_y"]
            col_orig = kpi if kpi in fc.columns else None
            
            if cols_x or cols_y:
                # 모든 관련 컬럼을 합쳐서 하나로 만듦 (NaN이 아닌 값 우선)
                all_cols = []
                if col_orig:
                    all_cols.append(kpi)
                all_cols.extend(cols_x)
                all_cols.extend(cols_y)
                
                if len(all_cols) > 1:
                    # coalesce: 첫번째 non-NaN 값 사용
                    fc[kpi] = fc[all_cols].bfill(axis=1).iloc[:, 0]
                    # 중복 컬럼 제거
                    drop_cols = [c for c in all_cols if c != kpi]
                    fc = fc.drop(columns=drop_cols, errors='ignore')
                    print(f"     Merged {all_cols} → {kpi}")
        
        fc_path = os.path.join(args.out_dir, "forecasts", "segment_forecasts_next.csv")
        fc.to_csv(fc_path, index=False, encoding="utf-8-sig")
        print(f"Saved next forecasts: {fc_path}")
    else:
        fc = pd.DataFrame()

    # Residual alerting for the last month using h=1 predictions from previous month
    prev_month = _month_add(last_month, -1)
    prev_rows = panel_fe[panel_fe["month"] == prev_month].copy()
    cur_rows = panel_fe[panel_fe["month"] == last_month].copy()

    alert_frames = []
    driver_rows = []

    for kpi in ["예금총잔액", "대출총잔액", "카드총사용", "디지털거래금액"]:
        if kpi not in model_targets:
            continue
        model_entry = model_registry.get((kpi, 1))
        if model_entry is None:
            continue
        tgt_col = model_targets[kpi]
        y_hat = _predict_model(model_entry, prev_rows)
        y_pred_raw = inverse_transform(kpi, y_hat)

        pred_df = pd.DataFrame({
            "segment_id": prev_rows["segment_id"].values,
            "month": last_month,
            f"pred_{kpi}": y_pred_raw,
        })
        # actual from current rows (raw KPI)
        actual_df = cur_rows[["segment_id", "month", kpi]].copy()
        merged = actual_df.merge(pred_df, on=["segment_id", "month"], how="left")
        alerts = score_residual_alerts(
            merged,
            kpi=kpi,
            pred_col=f"pred_{kpi}",
            actual_col=kpi,
            month_col="month",
            segment_col="segment_id",
        )
        alert_frames.append(alerts)

        # drivers for alert segments (use prev_rows row, because prediction based on prev_month)
        alerts_bad = alerts[alerts["alert_type"] != "NORMAL"].copy()
        if not alerts_bad.empty:
            # 업종 추출
            alerts_bad["_industry"] = alerts_bad["segment_id"].apply(
                lambda x: x.split("|")[0] if isinstance(x, str) and "|" in x else x
            )
            
            # DROP과 SPIKE 각각 업종 다양성 보장하면서 선택
            def select_with_diversity(df, n=30, ascending=True, max_per_ind=3):
                if df.empty:
                    return pd.DataFrame()
                df = df.sort_values("residual_pct", ascending=ascending)
                selected = []
                ind_counts = {}
                for _, row in df.iterrows():
                    ind = row["_industry"]
                    cnt = ind_counts.get(ind, 0)
                    if cnt < max_per_ind:
                        selected.append(row)
                        ind_counts[ind] = cnt + 1
                    if len(selected) >= n:
                        break
                return pd.DataFrame(selected)
            
            drop_alerts = alerts_bad[alerts_bad["alert_type"] == "DROP"]
            spike_alerts = alerts_bad[alerts_bad["alert_type"] == "SPIKE"]
            
            top_drop = select_with_diversity(drop_alerts, n=30, ascending=True, max_per_ind=3)
            top_spike = select_with_diversity(spike_alerts, n=30, ascending=False, max_per_ind=3)
            top_bad = pd.concat([top_drop, top_spike], ignore_index=True)
            
            for _, arow in top_bad.iterrows():
                seg = arow["segment_id"]
                row_df = prev_rows[prev_rows["segment_id"] == seg].head(1)
                if row_df.shape[0] != 1:
                    continue
                fm_explain = _as_lgbm_forecast(model_entry, tgt_col, 1)
                if fm_explain is None:
                    continue
                contrib = local_feature_contrib(fm_explain, row_df, top_k=8)
                drv = driver_sentence(contrib)
                driver_rows.append({
                    "month": last_month,
                    "segment_id": seg,
                    "kpi": kpi,
                    "alert_type": arow["alert_type"],
                    "residual_pct": arow["residual_pct"],
                    "driver": drv,
                })

    if alert_frames:
        all_alerts = pd.concat(alert_frames, ignore_index=True)
        
        # DEBUG: 전체 알림 저장하여 분석
        all_alerts_debug = all_alerts.merge(panel_res.segment_meta, on="segment_id", how="left")
        all_alerts_debug_path = os.path.join(args.out_dir, "all_alerts_debug.csv")
        all_alerts_debug.to_csv(all_alerts_debug_path, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] Saved all alerts: {all_alerts_debug_path} ({len(all_alerts_debug)} rows)")
        
        watchlist = build_watchlist(all_alerts, top_n=args.topn_watchlist)
        # Attach segment keys for readability
        meta = panel_res.segment_meta
        watchlist = watchlist.merge(meta, on="segment_id", how="left")
        watchlist_path = os.path.join(args.out_dir, "watchlist_alerts.csv")
        watchlist.to_csv(watchlist_path, index=False, encoding="utf-8-sig")
        print(f"Saved watchlist: {watchlist_path}")
        
        # watchlist TOP 세그먼트 중 드라이버가 없는 것들 추가 생성
        existing_driver_segs = set(r["segment_id"] for r in driver_rows)
        for _, wrow in watchlist.head(50).iterrows():
            seg = wrow["segment_id"]
            kpi = wrow.get("kpi", "예금총잔액")
            if seg in existing_driver_segs:
                continue
            # 해당 KPI 모델로 드라이버 생성
            model_entry = model_registry.get((kpi, 1))
            if model_entry is None:
                continue
            row_df = prev_rows[prev_rows["segment_id"] == seg].head(1)
            if row_df.shape[0] != 1:
                continue
            try:
                fm_explain = _as_lgbm_forecast(model_entry, model_targets.get(kpi, kpi), 1)
                if fm_explain is None:
                    continue
                contrib = local_feature_contrib(fm_explain, row_df, top_k=8)
                drv = driver_sentence(contrib)
                driver_rows.append({
                    "month": last_month,
                    "segment_id": seg,
                    "kpi": kpi,
                    "alert_type": wrow.get("alert_type", ""),
                    "residual_pct": wrow.get("residual_pct", 0),
                    "driver": drv,
                })
                existing_driver_segs.add(seg)
            except Exception:
                pass
    else:
        all_alerts = pd.DataFrame()
        watchlist = pd.DataFrame()

    if driver_rows:
        drivers = pd.DataFrame(driver_rows)
        drivers_path = os.path.join(args.out_dir, "watchlist_drivers.csv")
        drivers.to_csv(drivers_path, index=False, encoding="utf-8-sig")
        print(f"Saved drivers: {drivers_path}")
    else:
        drivers = pd.DataFrame()

    # Next Best Action recommendations for last month (use next-month forecasts from last_month)
    if not fc.empty:
        # Use predictions for last_month + 1
        next_month = _month_add(last_month, 1)
        fc1 = fc[fc["pred_month"] == next_month].copy()
        # Add suffix '__pred' for merging
        pred_cols = [c for c in fc1.columns if c not in {"segment_id", "pred_month"}]
        fc1 = fc1.rename(columns={c: f"{c}__pred" for c in pred_cols}).drop(columns=["pred_month"])

        # Current raw KPIs for last_month
        kpi_cols = [c for c in ["예금총잔액", "대출총잔액", "카드총사용", "디지털거래금액", "순유입", "FX총액"] if c in panel_fe.columns]

        # ============== 기존 액션 생성 (호환성 유지) ==============
        actions = recommend_actions(
            current_df=panel_fe,
            pred_df=fc1,
            month=last_month,
            kpi_cols=kpi_cols,
            alerts=all_alerts[all_alerts["month"] == last_month] if not all_alerts.empty else None,
            top_k_per_segment=3,
            top_n_global=200,
        )
        actions_path = os.path.join(args.out_dir, "actions_top.csv")
        actions.to_csv(actions_path, index=False, encoding="utf-8-sig")
        print(f"Saved actions: {actions_path}")
        
        # ============== 개선된 액션 생성 (NEW!) ==============
        print("  └─ Generating enhanced actions with diverse action types...")
        try:
            # 예측값 데이터프레임 준비
            preds_for_enhanced = fc1.copy()
            # __pred 접미사 제거하여 원래 KPI 이름으로 변경
            rename_map = {c: c.replace("__pred", "") for c in preds_for_enhanced.columns if "__pred" in c}
            preds_for_enhanced = preds_for_enhanced.rename(columns=rename_map)
            
            # FX 레짐 정보 추가 (있으면)
            panel_with_regime = panel_fe.copy()
            if "regime" not in panel_with_regime.columns:
                panel_with_regime["regime"] = ""
            
            # 개선된 액션 생성
            enhanced_actions = recommend_enhanced_actions(
                panel=panel_with_regime,
                preds_df=preds_for_enhanced,
                alerts=all_alerts[all_alerts["month"] == last_month] if not all_alerts.empty else None,
                fx_regime_col="regime",
                top_k=5,
                cfg=cfg,
            )
            
            enhanced_path = os.path.join(args.out_dir, "actions_enhanced.csv")
            enhanced_actions.to_csv(enhanced_path, index=False, encoding="utf-8-sig")
            print(f"     Saved enhanced actions: {enhanced_path}")
            
            # 액션 다양성 통계
            if not enhanced_actions.empty:
                action_type_counts = enhanced_actions["action_type"].value_counts()
                category_counts = enhanced_actions["category"].value_counts()
                print(f"     Action Types: {dict(action_type_counts.head(5))}")
                print(f"     Categories: {dict(category_counts)}")
                print(f"     Total Enhanced Actions: {len(enhanced_actions)}")
                print(f"     Unique Segments: {enhanced_actions['segment_id'].nunique()}")
        except Exception as e:
            print(f"     Enhanced actions generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 협업 필터링 기반 유사 세그먼트 정보 추가
        try:
            from imradar.recommend.embedding_recommender import (
                enrich_actions_with_collaborative,
                compute_segment_similarity_report
            )
            
            # 액션에 유사 세그먼트 정보 추가
            enriched_actions = enrich_actions_with_collaborative(actions, panel_fe, kpi_cols)
            enriched_path = os.path.join(args.out_dir, "actions_with_similarity.csv")
            enriched_actions.to_csv(enriched_path, index=False, encoding="utf-8-sig")
            print(f"Saved enriched actions: {enriched_path}")
            
            # 유사도 리포트 생성
            target_segments = actions['segment_id'].unique()[:100].tolist()
            similarity_report = compute_segment_similarity_report(panel_fe, kpi_cols, target_segments, top_k=3)
            sim_path = os.path.join(args.out_dir, "segment_similarity_report.csv")
            similarity_report.to_csv(sim_path, index=False, encoding="utf-8-sig")
            print(f"Saved similarity report: {sim_path}")
        except Exception as e:
            print(f"Collaborative filtering skipped: {e}")
    else:
        actions = pd.DataFrame()
        actions_path = None

    # Dashboards (static html)
    kpi_cols_overview = [c for c in ["예금총잔액", "대출총잔액", "카드총사용", "디지털거래금액"] if c in panel_fe.columns]
    if not fc.empty and kpi_cols_overview:
        overview_html = save_overview_dashboard(panel_fe, fc, os.path.join(args.out_dir, "dashboard"), kpi_cols_overview)
        print(f"Saved dashboard: {overview_html}")
        try:
            growth_html = save_growth_dashboard(panel_fe, fc, os.path.join(args.out_dir, "dashboard"), kpi="예금총잔액", horizon=1, top_n=50)
            print(f"Saved dashboard: {growth_html}")
        except Exception as e:
            print(f"Growth dashboard skipped due to error: {e}")
    if not watchlist.empty:
        wl_html = save_watchlist_dashboard(watchlist.head(100), os.path.join(args.out_dir, "dashboard"))
        print(f"Saved dashboard: {wl_html}")

    # FX Radar Dashboard
    fx_regime_dict = None
    if "FX총액" in panel_fe.columns:
        try:
            # Detect current FX regime using external data
            fx_regime = None
            if args.external_dir:
                fx_regime = compute_fx_regime_from_external(panel_fe, args.external_dir)
            
            # Generate FX segment scores (requires FXRegimeResult)
            if fx_regime is not None:
                fx_scores = compute_fx_segment_scores(panel_fe, fx_regime)
                fx_scores_path = os.path.join(args.out_dir, "fx_segment_scores.csv")
                fx_scores.to_csv(fx_scores_path, index=False, encoding="utf-8-sig")
                print(f"Saved FX scores: {fx_scores_path}")
                
                # Generate FX Radar Dashboard
                fx_html = save_fx_radar_dashboard(
                    fx_scores=fx_scores,
                    fx_regime_result=fx_regime,
                    out_dir=os.path.join(args.out_dir, "dashboard"),
                    top_n=30
                )
                print(f"Saved dashboard: {fx_html}")
                
                # Extract FX regime dict for executive summary
                if hasattr(fx_regime, 'fx_monthly') and len(fx_regime.fx_monthly) > 0:
                    latest = fx_regime.fx_monthly.iloc[-1]
                    fx_regime_dict = {
                        'fx_level': latest.get('fx_level', 0),
                        'trend': latest.get('fx_trend_regime', 'Range'),
                        'volatility': latest.get('fx_vol_regime', 'MedVol'),
                        'mom_pct': latest.get('fx_mom_pct', 0) * 100,
                    }
            else:
                print("FX Radar skipped: FX regime data not available")
        except Exception as e:
            print(f"FX Radar dashboard skipped due to error: {e}")

    # Executive Summary Dashboard
    try:
        drivers_df = pd.DataFrame(driver_rows) if driver_rows else pd.DataFrame()
        
        # Load FX scores if available
        fx_scores_df = None
        fx_scores_path = os.path.join(args.out_dir, "fx_segment_scores.csv")
        if os.path.exists(fx_scores_path):
            fx_scores_df = pd.read_csv(fx_scores_path)
        
        exec_html = generate_executive_summary_html(
            panel=panel_fe,
            forecasts=fc,
            watchlist=watchlist,
            actions=actions,
            drivers=drivers_df,
            segment_meta=panel_res.segment_meta,
            out_dir=os.path.join(args.out_dir, "dashboard"),
            current_month=last_month,
            fx_regime=fx_regime_dict,
            fx_scores=fx_scores_df,
        )
        print(f"Saved dashboard: {exec_html}")
    except Exception as e:
        print(f"Executive Summary dashboard skipped due to error: {e}")

    # One-page PDF report
    try:
        month_str = last_month.strftime("%Y-%m")
        top_alerts = watchlist.head(10) if not watchlist.empty else pd.DataFrame()
        # Growth proxy: segments with highest predicted deposit increase (next month)
        top_growth = pd.DataFrame()
        if not fc.empty and "예금총잔액" in fc.columns:
            fc1 = fc[fc["pred_month"] == _month_add(last_month, 1)].copy()
            cur = panel_fe[panel_fe["month"] == last_month][["segment_id", "예금총잔액"]].copy()
            tmp = cur.merge(fc1[["segment_id", "예금총잔액"]], on="segment_id", how="left", suffixes=("", "__pred"))
            tmp["delta"] = tmp["예금총잔액__pred"] - tmp["예금총잔액"]
            tmp = tmp.merge(panel_res.segment_meta, on="segment_id", how="left")
            top_growth = tmp.sort_values("delta", ascending=False).head(10)
        top_actions = actions.head(10) if actions is not None and not actions.empty else pd.DataFrame()

        pdf_path = os.path.join(args.out_dir, f"monthly_onepager_{last_month.strftime('%Y%m')}.pdf")
        generate_monthly_onepager(month_str, top_alerts, top_growth, top_actions, pdf_path)
        print(f"Saved PDF report: {pdf_path}")
    except Exception as e:
        print(f"PDF generation skipped due to error: {e}")

    print("DONE.")


if __name__ == "__main__":
    main()
