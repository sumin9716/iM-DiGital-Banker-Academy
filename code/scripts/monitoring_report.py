from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"

import sys
sys.path.insert(0, str(SRC_ROOT))

from imradar.ops.monitoring import compute_numeric_drift, summarize_drift, MonitoringConfig


def _load_panel(panel_path: Path) -> pd.DataFrame:
    if panel_path.suffix == ".parquet":
        return pd.read_parquet(panel_path)
    return pd.read_csv(panel_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate drift monitoring report")
    parser.add_argument("--panel_path", default=str(REPO_ROOT / "outputs" / "segment_panel.parquet"))
    parser.add_argument("--out_dir", default=str(REPO_ROOT / "outputs" / "monitoring"))
    parser.add_argument("--reference_months", type=int, default=6)
    parser.add_argument("--current_month", default=None, help="YYYY-MM (optional)")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    panel_path = Path(args.panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(f"panel_path not found: {panel_path}")

    df = _load_panel(panel_path)
    df["month"] = pd.to_datetime(df["month"])

    if args.current_month:
        current_month = pd.Timestamp(args.current_month)
    else:
        current_month = df["month"].max()

    ref_start = current_month - pd.DateOffset(months=args.reference_months)
    current_mask = df["month"] == current_month
    reference_mask = (df["month"] >= ref_start) & (df["month"] < current_month)

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in {"pre_birth"}
    ]

    report = compute_numeric_drift(
        df,
        current_mask=current_mask,
        reference_mask=reference_mask,
        cols=numeric_cols,
        bins=args.bins,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "psi_report.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    summary = summarize_drift(report, MonitoringConfig(bins=args.bins))
    summary.update({
        "current_month": current_month.strftime("%Y-%m-%d"),
        "reference_start": ref_start.strftime("%Y-%m-%d"),
    })
    with open(out_dir / "monitoring_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved drift report: {report_path}")
    print(f"Saved summary: {out_dir / 'monitoring_summary.json'}")


if __name__ == "__main__":
    main()
