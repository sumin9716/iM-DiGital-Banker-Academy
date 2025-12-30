from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class FXRegimeResult:
    fx_monthly: pd.DataFrame  # month-level FX features
    regime_labels: pd.Series  # regime label per month


def compute_fx_regime(
    fx_df: pd.DataFrame,
    month_col: str = "month",
    fx_level_col: str = "usdkrw_avg",
    fx_eom_col: Optional[str] = "usdkrw_eom",
) -> FXRegimeResult:
    """
    Simple regime labeling from monthly FX data.
    Expects fx_df to have:
      - month: datetime64[ns], month start
      - usdkrw_avg: monthly average
      - (optional) usdkrw_eom: end-of-month
    Produces:
      - momentum: delta over 1 month
      - vol proxy: rolling std(3)
      - regime: Uptrend/Downtrend × High/Low vol
    """
    d = fx_df.sort_values(month_col).copy()
    d["fx_level"] = d[fx_level_col].astype(float)
    d["fx_mom"] = d["fx_level"].diff(1)
    d["fx_vol3"] = d["fx_level"].diff(1).rolling(3).std()
    # Thresholds (median split)
    vol_thr = float(d["fx_vol3"].median(skipna=True)) if d["fx_vol3"].notna().any() else 0.0

    def _label(row) -> str:
        trend = "Range"
        if row["fx_mom"] > 0:
            trend = "Uptrend"
        elif row["fx_mom"] < 0:
            trend = "Downtrend"
        vol = "HighVol" if row["fx_vol3"] >= vol_thr else "LowVol"
        return f"{trend}_{vol}"

    d["fx_regime"] = d.apply(_label, axis=1)
    return FXRegimeResult(fx_monthly=d, regime_labels=d["fx_regime"])
