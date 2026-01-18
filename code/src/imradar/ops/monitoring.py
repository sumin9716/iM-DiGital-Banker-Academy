from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


def _safe_histogram(values: np.ndarray, bins: int) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype("float64")
    if hist.sum() == 0:
        return np.zeros_like(hist)
    return hist / hist.sum()


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Population Stability Index (PSI) for drift."""
    ref = np.asarray(reference)
    cur = np.asarray(current)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan
    ref_bins = _safe_histogram(ref, bins=bins)
    cur_bins = _safe_histogram(cur, bins=bins)
    ref_bins = np.clip(ref_bins, eps, None)
    cur_bins = np.clip(cur_bins, eps, None)
    return float(np.sum((cur_bins - ref_bins) * np.log(cur_bins / ref_bins)))


def compute_numeric_drift(
    df: pd.DataFrame,
    current_mask: pd.Series,
    reference_mask: pd.Series,
    cols: List[str],
    bins: int = 10,
) -> pd.DataFrame:
    """Compute drift metrics for numeric columns."""
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        ref = df.loc[reference_mask, c].values
        cur = df.loc[current_mask, c].values
        psi = compute_psi(ref, cur, bins=bins)
        rows.append({
            "column": c,
            "psi": psi,
            "ref_mean": float(np.nanmean(ref)) if len(ref) else np.nan,
            "cur_mean": float(np.nanmean(cur)) if len(cur) else np.nan,
            "ref_std": float(np.nanstd(ref)) if len(ref) else np.nan,
            "cur_std": float(np.nanstd(cur)) if len(cur) else np.nan,
            "ref_count": int(np.sum(~np.isnan(ref))),
            "cur_count": int(np.sum(~np.isnan(cur))),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


@dataclass
class MonitoringConfig:
    bins: int = 10
    psi_warn: float = 0.1
    psi_crit: float = 0.25


def summarize_drift(report: pd.DataFrame, cfg: Optional[MonitoringConfig] = None) -> Dict[str, float]:
    if cfg is None:
        cfg = MonitoringConfig()
    n_cols = len(report)
    warn_cnt = int((report["psi"] >= cfg.psi_warn).sum())
    crit_cnt = int((report["psi"] >= cfg.psi_crit).sum())
    return {
        "n_columns": n_cols,
        "psi_warn": cfg.psi_warn,
        "psi_crit": cfg.psi_crit,
        "warn_count": warn_cnt,
        "crit_count": crit_cnt,
    }
