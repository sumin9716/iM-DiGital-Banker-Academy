from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

import sys
sys.path.insert(0, str(SRC_ROOT))

from imradar.models.forecast_lgbm import ForecastModel, evaluate_forecast
from imradar.models.forecast_ensemble import EnsembleModel, evaluate_ensemble


MODEL_TARGETS = {
    "예금총잔액": "log1p_예금총잔액",
    "대출총잔액": "log1p_대출총잔액",
    "카드총사용": "log1p_카드총사용",
    "디지털거래금액": "log1p_디지털거래금액",
    "순유입": "slog1p_순유입",
    "FX총액": "log1p_FX총액",
}
TARGET_TO_KPI = {v: k for k, v in MODEL_TARGETS.items()}


def _parse_horizon(name: str) -> int:
    if "_h" in name:
        return int(name.split("_h")[-1].split(".")[0])
    return 1


def _to_ts(yyyymm: str | None) -> pd.Timestamp | None:
    if not yyyymm:
        return None
    y = int(yyyymm) // 100
    m = int(yyyymm) % 100
    return pd.Timestamp(year=y, month=m, day=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all saved models")
    parser.add_argument("--panel_path", default=str(REPO_ROOT / "outputs" / "segment_panel.parquet"))
    parser.add_argument("--model_dir", default=str(REPO_ROOT / "outputs" / "models"))
    parser.add_argument("--out_csv", default=str(REPO_ROOT / "outputs" / "model_evaluation.csv"))
    parser.add_argument("--test_start", default="202401")
    parser.add_argument("--test_end", default="202412")
    args = parser.parse_args()

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"panel_path not found: {panel_path}")
    df = pd.read_parquet(panel_path) if panel_path.suffix == ".parquet" else pd.read_csv(panel_path)

    test_start = _to_ts(args.test_start)
    test_end = _to_ts(args.test_end)

    rows = []
    for path in sorted(Path(args.model_dir).glob("*.pkl")):
        name = path.name
        horizon = _parse_horizon(name)
        model_type = "ensemble" if name.startswith("ensemble_") else "lgbm"

        with open(path, "rb") as f:
            model = pickle.load(f)

        if isinstance(model, EnsembleModel) or model_type == "ensemble":
            kpi_col = model.kpi
            metrics = evaluate_ensemble(
                model,
                df,
                kpi_col=kpi_col,
                horizon=horizon,
                test_start_month=test_start,
                test_end_month=test_end,
            )
            if not metrics:
                continue
            target_col = kpi_col
            kpi = TARGET_TO_KPI.get(target_col, target_col)
        elif isinstance(model, ForecastModel):
            fm: ForecastModel = model
            metrics = evaluate_forecast(
                df,
                fm,
                test_start_month=test_start,
                test_end_month=test_end,
            )
            target_col = fm.kpi
            kpi = TARGET_TO_KPI.get(target_col, target_col)
        else:
            # Skip unsupported model formats
            continue

        metrics.update({
            "model_type": model_type,
            "kpi": kpi,
            "target_col": target_col,
            "horizon": horizon,
            "model_file": name,
        })
        rows.append(metrics)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"Saved evaluation: {args.out_csv}")


if __name__ == "__main__":
    main()
