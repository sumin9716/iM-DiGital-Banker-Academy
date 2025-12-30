from __future__ import annotations

import re
import logging
from typing import Optional

import numpy as np
import pandas as pd

# ============== Logging Setup ==============
logger = logging.getLogger("imradar")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the imradar package."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============== Date/Time Utilities ==============
def to_month_start(yyyymm: int) -> pd.Timestamp:
    """
    Convert YYYYMM integer to month-start Timestamp.
    
    Example:
        >>> to_month_start(202412)
        Timestamp('2024-12-01 00:00:00')
    """
    y = int(yyyymm) // 100
    m = int(yyyymm) % 100
    return pd.Timestamp(year=y, month=m, day=1)


def month_add(m: pd.Timestamp, k: int) -> pd.Timestamp:
    """
    Add k months to a timestamp.
    
    Example:
        >>> month_add(pd.Timestamp('2024-12-01'), 3)
        Timestamp('2025-03-01 00:00:00')
    """
    return (m.to_period("M") + k).to_timestamp()


def yyyymm_to_ts(yyyymm: str) -> pd.Timestamp:
    """
    Convert YYYYMM string to Timestamp.
    
    Example:
        >>> yyyymm_to_ts("202412")
        Timestamp('2024-12-01 00:00:00')
    """
    y = int(yyyymm) // 100
    m = int(yyyymm) % 100
    return pd.Timestamp(year=y, month=m, day=1)


# ============== Value Transformations ==============
def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Signed log1p transformation for values that can be negative."""
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_signed_log1p(y: np.ndarray) -> np.ndarray:
    """Inverse of signed log1p transformation."""
    return np.sign(y) * np.expm1(np.abs(y))


def safe_log1p(x: np.ndarray) -> np.ndarray:
    """Log1p with clipping for non-negative values."""
    return np.log1p(np.maximum(x, 0.0))


def inverse_log1p(y: np.ndarray) -> np.ndarray:
    """Inverse log1p transformation."""
    return np.expm1(np.maximum(y, 0.0))


# ============== Korean Bucket Parsing ==============


def parse_bucket_kor_to_number(s: str) -> Optional[float]:
    """
    Parse Korean bucket strings like:
      - "0건", "1건", "0개", "10개초과 20개이하", "50건 초과"
    to an approximate numeric value.
    Returns None if parsing fails.
    """
    if s is None:
        return None
    s = str(s).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None

    # Extract integers in the string
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return None

    if len(nums) >= 2:
        # e.g., "2건초과 5건이하" -> (2+5)/2
        return float(sum(nums[:2]) / 2.0)

    n = nums[0]

    # Single number cases
    # e.g., "1건", "0개"
    if "초과" not in s and "이하" not in s:
        return float(n)

    # Open-ended "N 초과"
    if "초과" in s and "이하" not in s:
        # Conservative: n + 1
        return float(n + 1)

    # "N 이하" (rare)
    if "이하" in s and "초과" not in s:
        # Conservative: n
        return float(n)

    return float(n)
