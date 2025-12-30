from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd

from imradar.config import RadarConfig


def _col_sum(col: str) -> str:
    """Return the aggregated sum column name."""
    return f"{col}__sum"


def compute_kpis(panel: pd.DataFrame, cfg: RadarConfig) -> pd.DataFrame:
    """
    Create KPI columns on the panel.
    Assumes panel contains aggregated sum columns like '요구불예금잔액__sum'.
    """
    df = panel.copy()

    # Main KPIs
    for kpi, components in cfg.kpi_defs.items():
        if kpi == "순유입":
            # allow signed: ["요구불입금금액", "-요구불출금금액"]
            vals = []
            for comp in components:
                sign = 1.0
                base = comp
                if comp.startswith("-"):
                    sign = -1.0
                    base = comp[1:]
                col = _col_sum(base)
                if col in df.columns:
                    vals.append(sign * df[col].astype("float64"))
            if vals:
                df[kpi] = np.sum(vals, axis=0)
        else:
            vals = []
            for base in components:
                col = _col_sum(base)
                if col in df.columns:
                    vals.append(df[col].astype("float64"))
            if vals:
                df[kpi] = np.sum(vals, axis=0)

    # Derived KPIs
    loan_total = df.get("대출총잔액")
    limit_sum = df.get(_col_sum("여신한도금액"))
    if loan_total is not None and limit_sum is not None:
        df["한도소진율"] = np.where(limit_sum > 0, loan_total / limit_sum, 0.0)

    digital = df.get("디지털거래금액")
    if digital is not None:
        total_channel = (
            df.get(_col_sum("창구거래금액"), 0.0)
            + df.get(_col_sum("ATM거래금액"), 0.0)
            + df.get(_col_sum("자동이체금액"), 0.0)
            + digital
        )
        df["디지털비중"] = np.where(total_channel > 0, digital / total_channel, 0.0)
        df["자동이체비중"] = np.where(total_channel > 0, df.get(_col_sum("자동이체금액"), 0.0) / total_channel, 0.0)

    # Convenience: log transforms
    for kpi in ["예금총잔액", "대출총잔액", "카드총사용", "디지털거래금액", "FX총액"]:
        if kpi in df.columns:
            df[f"log1p_{kpi}"] = np.log1p(np.maximum(df[kpi].astype("float64"), 0.0))

    # 순유입은 음수 가능 → signed log1p
    if "순유입" in df.columns:
        x = df["순유입"].astype("float64")
        df["slog1p_순유입"] = np.sign(x) * np.log1p(np.abs(x))

    return df
