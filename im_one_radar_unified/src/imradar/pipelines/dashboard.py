from __future__ import annotations

import os
from typing import Dict, Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def save_overview_dashboard(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    out_dir: str,
    kpi_cols: List[str],
) -> str:
    """
    Create an overview HTML showing:
    - historical total KPI trends
    - next 1~3 month forecasts (total across segments)
    """
    os.makedirs(out_dir, exist_ok=True)

    hist = panel[panel.get("pre_birth", 0) == 0].copy()
    hist_tot = hist.groupby("month")[kpi_cols].sum(numeric_only=True).reset_index()

    # forecast totals
    fc_tot = forecasts.groupby("pred_month")[kpi_cols].sum(numeric_only=True).reset_index().rename(columns={"pred_month": "month"})
    fc_tot["type"] = "forecast"
    hist_tot2 = hist_tot.copy()
    hist_tot2["type"] = "actual"

    combined = pd.concat([hist_tot2, fc_tot], axis=0, ignore_index=True)

    figs = []
    for k in kpi_cols:
        fig = px.line(combined, x="month", y=k, color="type", title=f"전체 {k} 추이 (actual vs forecast)")
        figs.append(fig)

    # Combine into single HTML
    html_path = os.path.join(out_dir, "01_overview.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>")
        f.write("<h1>Segment Radar — Overview</h1>")
        for fig in figs:
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</body></html>")
    return html_path


def save_watchlist_dashboard(
    watchlist: pd.DataFrame,
    out_dir: str,
    title: str = "Risk Radar — Watchlist",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    wl = watchlist.copy()
    wl["month"] = wl["month"].astype(str)

    table = go.Figure(
        data=[go.Table(
            header=dict(values=list(wl.columns)),
            cells=dict(values=[wl[c] for c in wl.columns])
        )]
    )
    table.update_layout(title=title)

    html_path = os.path.join(out_dir, "03_risk_watchlist.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>")
        f.write(f"<h1>{title}</h1>")
        f.write(table.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</body></html>")
    return html_path


def save_growth_dashboard(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    out_dir: str,
    kpi: str = "예금총잔액",
    horizon: int = 1,
    top_n: int = 50,
) -> str:
    """
    Growth Forecast tab (MVP):
    - Top segments by predicted delta for next month/horizon
    """
    os.makedirs(out_dir, exist_ok=True)

    last_month = panel["month"].max()
    cur = panel[panel["month"] == last_month][["segment_id", kpi]].copy()
    pred_month = (last_month.to_period("M") + horizon).to_timestamp()
    fc = forecasts[forecasts["pred_month"] == pred_month][["segment_id", kpi]].copy()

    merged = cur.merge(fc, on="segment_id", how="left", suffixes=("", "__pred"))
    merged["delta"] = merged[f"{kpi}__pred"] - merged[kpi]
    merged = merged.sort_values("delta", ascending=False).head(top_n)

    fig = px.bar(merged, x="segment_id", y="delta", title=f"{kpi} 예측 상승 TOP {top_n} (h={horizon})")
    fig.update_layout(xaxis_title="segment_id", yaxis_title="predicted delta")

    table = go.Figure(
        data=[go.Table(
            header=dict(values=list(merged.columns)),
            cells=dict(values=[merged[c] for c in merged.columns])
        )]
    )
    table.update_layout(title=f"{kpi} 예측 상승 리스트 (h={horizon})")

    html_path = os.path.join(out_dir, "02_growth_forecast.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'></head><body>")
        f.write("<h1>Growth Forecast</h1>")
        f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(table.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")
    return html_path
