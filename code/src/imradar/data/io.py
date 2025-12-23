from __future__ import annotations

import pandas as pd
from typing import Optional, List


def load_internal_csv(path: str, encoding: Optional[str] = "cp949") -> pd.DataFrame:
    """
    Load internal CSV.
    - Handles common Korean encodings (cp949/euc-kr/utf-8-sig).
    """
    encodings_to_try: List[Optional[str]] = []
    if encoding:
        encodings_to_try.append(encoding)
    encodings_to_try += ["cp949", "euc-kr", "utf-8-sig", None]

    last_err: Optional[Exception] = None
    for enc in encodings_to_try:
        try:
            if enc is None:
                return pd.read_csv(path)
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to read CSV with tried encodings={encodings_to_try}. Last error={last_err}")
