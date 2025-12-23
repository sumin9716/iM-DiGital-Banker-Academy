from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class Alert:
    segment_id: str
    month: pd.Timestamp
    kpi: str
    actual: float
    predicted: float
    residual: float
    residual_pct: float
    severity: str
    alert_type: str


def score_residual_alerts(
    df: pd.DataFrame,
    kpi: str,
    pred_col: str,
    actual_col: str,
    month_col: str = "month",
    segment_col: str = "segment_id",
    down_threshold_pct: float = -0.20,
    up_threshold_pct: float = 0.20,
) -> pd.DataFrame:
    """
    Residual-based alerting:
      residual_pct = (actual - pred) / max(|pred|, eps)
    """
    eps = 1e-9
    d = df[[segment_col, month_col, actual_col, pred_col]].copy()
    d = d.rename(columns={actual_col: "actual", pred_col: "predicted"})
    d["residual"] = d["actual"] - d["predicted"]
    d["residual_pct"] = d["residual"] / np.maximum(np.abs(d["predicted"]), eps)

    def _sev(x: float) -> str:
        ax = abs(x)
        if ax >= 0.5:
            return "CRITICAL"
        if ax >= 0.3:
            return "HIGH"
        if ax >= 0.2:
            return "MEDIUM"
        return "LOW"

    d["severity"] = d["residual_pct"].apply(_sev)
    d["alert_type"] = np.where(d["residual_pct"] <= down_threshold_pct, "DROP", np.where(d["residual_pct"] >= up_threshold_pct, "SPIKE", "NORMAL"))
    d["kpi"] = kpi
    return d


def build_watchlist(alerts: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Select top-N alerts by severity then absolute residual_pct.
    """
    sev_rank = {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}
    a = alerts.copy()
    a["sev_rank"] = a["severity"].map(sev_rank).fillna(0).astype(int)
    a = a[a["alert_type"] != "NORMAL"].copy()
    a = a.sort_values(["sev_rank", "residual_pct"], ascending=[False, True])  # drops are negative => smaller is worse
    return a.head(top_n).drop(columns=["sev_rank"])
