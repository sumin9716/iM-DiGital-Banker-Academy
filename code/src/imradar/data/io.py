from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# 은행 내부 자료와 공공 데이터 CSV는 cp949로 저장된 경우가 많아 먼저 시도한다.
_FALLBACK_ENCODINGS = ("cp949", "euc-kr", "utf-8-sig", "utf-8")


def read_csv_with_fallback(path: Union[str, Path], encoding: Optional[str] = None) -> pd.DataFrame:
    """주어진 인코딩부터 시작해 한국어 CSV에 흔한 인코딩을 차례로 시도해 읽는다."""
    candidates: List[str] = [encoding] if encoding else []
    candidates += [e for e in _FALLBACK_ENCODINGS if e not in candidates]

    last_err: Optional[Exception] = None
    for enc in candidates:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeError, pd.errors.ParserError) as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV {path} with encodings={candidates}. Last error={last_err}")


def load_internal_csv(path: Union[str, Path], encoding: Optional[str] = "cp949") -> pd.DataFrame:
    """은행 내부 CSV(iMbank_data.csv.csv)를 읽는다."""
    return read_csv_with_fallback(path, encoding)
