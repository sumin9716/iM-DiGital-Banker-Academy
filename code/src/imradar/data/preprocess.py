from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from imradar.config import RadarConfig
from imradar.utils import parse_bucket_kor_to_number


@dataclass
class PanelBuildResult:
    panel: pd.DataFrame
    segment_meta: pd.DataFrame


def _to_month_start(yyyymm: int) -> pd.Timestamp:
    y = int(yyyymm) // 100
    m = int(yyyymm) % 100
    return pd.Timestamp(year=y, month=m, day=1)


def add_time_columns(df: pd.DataFrame, month_col: str = "기준년월") -> pd.DataFrame:
    out = df.copy()
    out["month"] = out[month_col].apply(_to_month_start)
    out["year"] = out["month"].dt.year
    out["month_num"] = out["month"].dt.month
    # Month index from start
    min_month = out["month"].min()
    out["t"] = (out["month"].dt.to_period("M") - min_month.to_period("M")).apply(lambda p: p.n)
    return out


def convert_bucket_counts(df: pd.DataFrame, bucket_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in bucket_cols:
        if c in out.columns:
            out[c] = out[c].apply(parse_bucket_kor_to_number).astype("float32")
    return out


def aggregate_to_segment_month(
    raw_df: pd.DataFrame,
    cfg: RadarConfig,
    additional_group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Aggregate raw records to (segment_keys, month) level.

    - Amount columns: sum
    - Bucket count columns: sum (approximate) + mean (intensity); both are kept
    - Adds customer_count (number of raw rows per segment-month)
    """
    df = raw_df.copy()

    group_cols = [cfg.month_col] + cfg.segment_keys
    if additional_group_cols:
        group_cols += additional_group_cols

    # Convert bucket columns to numeric so we can aggregate
    df = convert_bucket_counts(df, cfg.bucket_count_cols)

    # Identify numeric columns (amounts + converted buckets)
    amount_cols = [c for c in cfg.amount_cols if c in df.columns]
    bucket_cols = [c for c in cfg.bucket_count_cols if c in df.columns]

    agg_dict: Dict[str, List[str]] = {}
    for c in amount_cols:
        agg_dict[c] = ["sum"]
    for c in bucket_cols:
        agg_dict[c] = ["sum", "mean"]

    # Customer count proxy
    df["_row"] = 1
    agg_dict["_row"] = ["sum"]

    g = df.groupby(group_cols, dropna=False).agg(agg_dict)
    # Flatten MultiIndex columns
    g.columns = ["__".join([c for c in col if c]) for col in g.columns.to_flat_index()]
    g = g.reset_index()
    g = g.rename(columns={"_row__sum": "customer_count"})

    return g


def build_full_panel(segment_month: pd.DataFrame, cfg: RadarConfig) -> PanelBuildResult:
    """
    Build a complete segment × month panel.

    Strategy:
    - Create full month range from global min to max.
    - For each segment, fill missing months with 0 after the segment's first appearance.
    - Months before first appearance are marked 'pre_birth'=1 and kept as NaN for KPI amounts (optional handling).
    """
    df = segment_month.copy()
    df = add_time_columns(df, month_col=cfg.month_col)

    # Segment id
    df["segment_id"] = df[cfg.segment_keys].astype(str).agg("|".join, axis=1)

    # Segment meta
    meta_cols = cfg.segment_keys + ["segment_id"]
    segment_meta = df[meta_cols].drop_duplicates().reset_index(drop=True)

    # Full month index
    all_months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    month_df = pd.DataFrame({"month": all_months})
    month_df["year"] = month_df["month"].dt.year
    month_df["month_num"] = month_df["month"].dt.month
    month_df["t"] = (month_df["month"].dt.to_period("M") - df["month"].min().to_period("M")).apply(lambda p: p.n)

    # Cross join segments × months
    segments = segment_meta[["segment_id"]]
    segments["key"] = 1
    month_df["key"] = 1
    panel = segments.merge(month_df, on="key", how="outer").drop(columns=["key"])

    # Join static keys back
    panel = panel.merge(segment_meta, on="segment_id", how="left")

    # Join observed data
    obs_cols = [c for c in df.columns if c not in ["year", "month_num", "t"]]
    panel = panel.merge(df[obs_cols], on=["segment_id", "month"] + cfg.segment_keys, how="left")

    # Determine first appearance per segment (customer_count not null)
    first_month = (
        panel.loc[panel["customer_count"].notna(), ["segment_id", "month"]]
        .groupby("segment_id")["month"]
        .min()
        .rename("first_month")
        .reset_index()
    )
    panel = panel.merge(first_month, on="segment_id", how="left")
    panel["pre_birth"] = (panel["month"] < panel["first_month"]).astype(int)

    # Fill missing numeric aggregates after birth with 0
    numeric_cols = [c for c in panel.columns if c.endswith("__sum") or c.endswith("__mean")] + ["customer_count"]
    for c in numeric_cols:
        if c in panel.columns:
            panel.loc[panel["pre_birth"] == 0, c] = panel.loc[panel["pre_birth"] == 0, c].fillna(0.0)

    # Segment age (months since first_month)
    panel["segment_age"] = (panel["month"].dt.to_period("M") - panel["first_month"].dt.to_period("M")).apply(
        lambda p: p.n if pd.notnull(p) else np.nan
    )
    panel.loc[panel["pre_birth"] == 1, "segment_age"] = np.nan

    return PanelBuildResult(panel=panel, segment_meta=segment_meta)
