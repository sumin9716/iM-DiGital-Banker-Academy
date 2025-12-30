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

from imradar.config import RadarConfig
from imradar.data.io import load_internal_csv
from imradar.data.preprocess import aggregate_to_segment_month, build_full_panel
from imradar.features.kpi import compute_kpis
from imradar.features.lags import add_lag_features, add_rolling_features, add_change_features
from imradar.models.forecast_lgbm import train_lgbm_forecast, evaluate_forecast, predict_with_model
from imradar.models.residual_alerts import score_residual_alerts, build_watchlist
from imradar.explain.lgbm_contrib import local_feature_contrib, driver_sentence
from imradar.recommend.rules import recommend_actions
from imradar.pipelines.dashboard import save_overview_dashboard, save_watchlist_dashboard, save_growth_dashboard
from imradar.pipelines.report_pdf import generate_monthly_onepager


def _month_add(m: pd.Timestamp, k: int) -> pd.Timestamp:
    return (m.to_period("M") + k).to_timestamp()


def inverse_transform(kpi: str, y_hat: np.ndarray) -> np.ndarray:
    # y_hat is transformed prediction
    if kpi == "순유입":
        # signed expm1
        return np.sign(y_hat) * (np.expm1(np.abs(y_hat)))
    else:
        return np.expm1(np.maximum(y_hat, 0.0))


def main():
    parser = argparse.ArgumentParser(description="Segment Radar MVP pipeline")
    parser.add_argument("--raw_csv", required=True, help="Path to internal raw CSV")
    parser.add_argument("--encoding", default="cp949", help="CSV encoding (default: cp949)")
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "outputs"), help="Output directory")
    parser.add_argument("--train_end", default=None, help="Train end month (YYYYMM), e.g., 202411. If not set, use all.")
    parser.add_argument("--test_start", default="202401", help="Backtest start month (YYYYMM)")
    parser.add_argument("--test_end", default="202412", help="Backtest end month (YYYYMM)")
    parser.add_argument("--topn_watchlist", type=int, default=50, help="Top-N watchlist size")
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

    # Train models
    categorical_cols = cfg.segment_keys  # static cats

    models = {}
    metrics_rows = []

    print("[6/8] Train forecasting models (LGBM)")
    for kpi, tgt_col in model_targets.items():
        if tgt_col not in panel_fe.columns:
            continue
        for h in cfg.horizons:
            fm = train_lgbm_forecast(
                panel_fe,
                kpi_col=tgt_col,
                horizon=h,
                categorical_cols=categorical_cols,
                train_end_month=train_end,
            )
            models[(kpi, h)] = fm
            # Backtest metrics (transformed space)
            met = evaluate_forecast(panel_fe, fm, test_start_month=test_start, test_end_month=test_end)
            met.update({"kpi": kpi, "target_col": tgt_col, "horizon": h})
            metrics_rows.append(met)
            # Save model
            model_path = os.path.join(args.out_dir, "models", f"lgbm_{kpi}_h{h}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(fm, f)
    metrics_df = pd.DataFrame(metrics_rows).sort_values(["kpi", "horizon"])
    metrics_csv = os.path.join(args.out_dir, "forecast_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"Saved metrics: {metrics_csv}")

    # Forecast next 1~3 months from last observed month
    last_month = panel_fe["month"].max()
    print(f"[7/8] Forecast next horizons from last observed month={last_month.date()}")

    latest_rows = panel_fe[panel_fe["month"] == last_month].copy()
    forecasts = []
    for kpi, tgt_col in model_targets.items():
        for h in cfg.horizons:
            fm = models.get((kpi, h))
            if fm is None:
                continue
            y_hat = predict_with_model(fm, latest_rows)
            y_raw = inverse_transform(kpi, y_hat)
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
        fm = models.get((kpi, 1))
        if fm is None:
            continue
        tgt_col = model_targets[kpi]
        y_hat = predict_with_model(fm, prev_rows)
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
            # take top 30 per KPI for driver summary
            top_bad = alerts_bad.sort_values("residual_pct").head(30)
            for _, arow in top_bad.iterrows():
                seg = arow["segment_id"]
                row_df = prev_rows[prev_rows["segment_id"] == seg].head(1)
                if row_df.shape[0] != 1:
                    continue
                contrib = local_feature_contrib(fm, row_df, top_k=8)
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
        watchlist = build_watchlist(all_alerts, top_n=args.topn_watchlist)
        # Attach segment keys for readability
        meta = panel_res.segment_meta
        watchlist = watchlist.merge(meta, on="segment_id", how="left")
        watchlist_path = os.path.join(args.out_dir, "watchlist_alerts.csv")
        watchlist.to_csv(watchlist_path, index=False, encoding="utf-8-sig")
        print(f"Saved watchlist: {watchlist_path}")
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
