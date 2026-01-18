"""
pytest fixtures
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_panel():
    """샘플 패널 데이터"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "segment_id": [f"seg_{i % 10}" for i in range(n)],
        "month": pd.date_range("2024-01-01", periods=n, freq="D"),
        "예금총잔액": np.random.randn(n) * 100 + 1000,
        "대출총잔액": np.random.randn(n) * 50 + 500,
        "순유입": np.random.randn(n) * 20,
    })


@pytest.fixture
def sample_actions():
    """샘플 액션 데이터"""
    return pd.DataFrame({
        "segment_id": ["seg_1", "seg_2", "seg_3"],
        "action_type": ["DEPOSIT_GROWTH", "CHURN_PREVENTION", "LIQUIDITY_STRESS"],
        "score": [80, 120, 95],
        "urgency": ["HIGH", "CRITICAL", "HIGH"],
    })
