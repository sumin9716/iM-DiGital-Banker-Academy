from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd

from imradar.models.forecast_lgbm import ForecastModel, predict_with_model


def local_feature_contrib(
    fm: ForecastModel,
    row_df: pd.DataFrame,
    top_k: int = 8,
) -> pd.DataFrame:
    """
    Compute local feature contributions using LightGBM's pred_contrib=True.
    Returns top positive/negative contributors for the given row (single-row dataframe).
    """
    if row_df.shape[0] != 1:
        raise ValueError("row_df must have exactly 1 row")
    X = row_df[fm.feature_cols].copy()
    for c in fm.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")

    contrib = fm.model.predict(X, pred_contrib=True)
    # contrib shape: (n, n_features+1) last column is bias
    contrib = contrib[0]
    feature_names = fm.feature_cols + ["bias"]
    s = pd.Series(contrib, index=feature_names).drop("bias")

    top_pos = s.sort_values(ascending=False).head(top_k // 2)
    top_neg = s.sort_values(ascending=True).head(top_k // 2)

    out = pd.concat([top_pos, top_neg]).reset_index()
    out.columns = ["feature", "contribution"]
    out["direction"] = np.where(out["contribution"] >= 0, "POS", "NEG")
    out = out.sort_values("contribution", ascending=False).reset_index(drop=True)
    return out


def driver_sentence(contrib_df: pd.DataFrame, max_terms: int = 3) -> str:
    """
    Convert contribution list to a concise Korean driver sentence.
    """
    pos = contrib_df[contrib_df["direction"] == "POS"].sort_values("contribution", ascending=False).head(max_terms)
    neg = contrib_df[contrib_df["direction"] == "NEG"].sort_values("contribution", ascending=True).head(max_terms)

    def _fmt(df: pd.DataFrame) -> str:
        feats = df["feature"].tolist()
        if not feats:
            return ""
        # shorten names
        feats = [f.replace("log1p_", "").replace("__", ":") for f in feats]
        return ", ".join(feats)

    pos_txt = _fmt(pos)
    neg_txt = _fmt(neg)

    if pos_txt and neg_txt:
        return f"상승 요인: {pos_txt} / 하락 요인: {neg_txt}"
    if pos_txt:
        return f"상승 요인: {pos_txt}"
    if neg_txt:
        return f"하락 요인: {neg_txt}"
    return "주요 드라이버가 뚜렷하지 않습니다."
