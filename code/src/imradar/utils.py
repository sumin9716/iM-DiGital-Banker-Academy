from __future__ import annotations

import re
from typing import Optional


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
