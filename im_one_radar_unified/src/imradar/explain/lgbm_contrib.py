from __future__ import annotations

from typing import List, Tuple, Optional, Dict
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
    
    Args:
        fm: ForecastModel instance
        row_df: Single-row DataFrame
        top_k: Number of top contributors to return
    
    Returns:
        DataFrame with feature contributions
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
    out["abs_contribution"] = out["contribution"].abs()
    out = out.sort_values("abs_contribution", ascending=False).reset_index(drop=True)
    return out


def batch_feature_contrib(
    fm: ForecastModel,
    df: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    배치로 모든 행에 대해 feature contribution 계산
    
    Args:
        fm: ForecastModel instance
        df: Input DataFrame
        top_k: 각 행당 반환할 top contributor 수
    
    Returns:
        DataFrame with segment, month, top features, contributions
    """
    X = df[fm.feature_cols].copy()
    for c in fm.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    
    contribs = fm.model.predict(X, pred_contrib=True)
    # Remove bias column
    contribs = contribs[:, :-1]
    
    results = []
    for i, row_contrib in enumerate(contribs):
        s = pd.Series(row_contrib, index=fm.feature_cols)
        top_feats = s.abs().nlargest(top_k).index.tolist()
        
        row_result = {
            "row_idx": i,
            "top_features": top_feats,
            "contributions": [float(s[f]) for f in top_feats],
        }
        
        # 개별 피처 컬럼 추가
        for j, feat in enumerate(top_feats):
            row_result[f"feature_{j+1}"] = feat
            row_result[f"contrib_{j+1}"] = float(s[feat])
        
        results.append(row_result)
    
    return pd.DataFrame(results)


def global_feature_importance(
    fm: ForecastModel,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    글로벌 피처 중요도 추출
    
    Args:
        fm: ForecastModel instance
        importance_type: "gain" 또는 "split"
    
    Returns:
        DataFrame with feature importance
    """
    if fm.feature_importance is not None:
        return fm.feature_importance.copy()
    
    # 직접 계산
    imp = fm.model.booster_.feature_importance(importance_type=importance_type)
    df = pd.DataFrame({
        'feature': fm.feature_cols,
        'importance': imp,
    })
    df['importance_pct'] = df['importance'] / df['importance'].sum() * 100
    return df.sort_values('importance', ascending=False).reset_index(drop=True)


def get_feature_groups(features: List[str]) -> Dict[str, List[str]]:
    """
    피처를 그룹별로 분류
    
    Args:
        features: 피처명 리스트
    
    Returns:
        그룹별 피처 딕셔너리
    """
    groups = {
        "lag_features": [],
        "rolling_features": [],
        "ewm_features": [],
        "momentum_features": [],
        "zscore_features": [],
        "time_features": [],
        "categorical_features": [],
        "base_features": [],
    }
    
    for f in features:
        if "__lag" in f:
            groups["lag_features"].append(f)
        elif "__roll" in f:
            groups["rolling_features"].append(f)
        elif "__ewm" in f:
            groups["ewm_features"].append(f)
        elif "__mom" in f or "__yoy" in f or "__momentum" in f or "__accel" in f:
            groups["momentum_features"].append(f)
        elif "__zscore" in f:
            groups["zscore_features"].append(f)
        elif f in ["month_num", "quarter", "year", "month_sin", "month_cos", 
                   "quarter_sin", "quarter_cos", "is_quarter_end", "is_year_end"]:
            groups["time_features"].append(f)
        elif f.endswith("_cat") or f.endswith("_enc"):
            groups["categorical_features"].append(f)
        else:
            groups["base_features"].append(f)
    
    return {k: v for k, v in groups.items() if v}  # 빈 그룹 제외


def group_importance_summary(
    fm: ForecastModel,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    피처 그룹별 중요도 요약
    
    Args:
        fm: ForecastModel instance
        importance_type: "gain" 또는 "split"
    
    Returns:
        그룹별 중요도 요약 DataFrame
    """
    imp_df = global_feature_importance(fm, importance_type)
    feature_groups = get_feature_groups(fm.feature_cols)
    
    results = []
    for group_name, group_features in feature_groups.items():
        group_imp = imp_df[imp_df['feature'].isin(group_features)]
        if len(group_imp) > 0:
            results.append({
                "group": group_name,
                "feature_count": len(group_features),
                "total_importance": float(group_imp['importance'].sum()),
                "avg_importance": float(group_imp['importance'].mean()),
                "top_feature": group_imp.iloc[0]['feature'] if len(group_imp) > 0 else "",
            })
    
    result_df = pd.DataFrame(results)
    result_df["importance_pct"] = result_df["total_importance"] / result_df["total_importance"].sum() * 100
    return result_df.sort_values("total_importance", ascending=False).reset_index(drop=True)


def driver_sentence(contrib_df: pd.DataFrame, max_terms: int = 3) -> str:
    """
    Convert contribution list to a concise Korean driver sentence.
    
    Args:
        contrib_df: Contribution DataFrame
        max_terms: Maximum number of terms to include
    
    Returns:
        Korean driver explanation sentence
    """
    pos = contrib_df[contrib_df["direction"] == "POS"].sort_values("contribution", ascending=False).head(max_terms)
    neg = contrib_df[contrib_df["direction"] == "NEG"].sort_values("contribution", ascending=True).head(max_terms)

    def _fmt(df: pd.DataFrame) -> str:
        feats = df["feature"].tolist()
        if not feats:
            return ""
        # shorten names
        feats = [_format_feature_name(f) for f in feats]
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


def _format_feature_name(feature: str) -> str:
    """
    피처명을 읽기 쉬운 형태로 변환
    
    Args:
        feature: 원본 피처명
    
    Returns:
        포맷된 피처명
    """
    # 접두사 제거
    name = feature.replace("log1p_", "").replace("__", ":")
    
    # 특수 변환
    replacements = {
        ":lag1": "(1개월전)",
        ":lag2": "(2개월전)",
        ":lag3": "(3개월전)",
        ":lag6": "(6개월전)",
        ":lag12": "(12개월전)",
        ":roll3_mean": "(3개월평균)",
        ":roll6_mean": "(6개월평균)",
        ":roll3_std": "(3개월변동)",
        ":roll6_std": "(6개월변동)",
        ":ewm3": "(단기추세)",
        ":ewm6": "(중기추세)",
        ":ewm12": "(장기추세)",
        ":mom": "(MoM)",
        ":yoy": "(YoY)",
        ":3m_change": "(3개월변화)",
        ":accel": "(가속도)",
        ":zscore12": "(이상도)",
        ":momentum_3_12": "(모멘텀)",
    }
    
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    return name


def generate_alert_explanation(
    fm: ForecastModel,
    alert_row: pd.DataFrame,
    include_recommendation: bool = True,
) -> Dict[str, any]:
    """
    알림에 대한 상세 설명 생성
    
    Args:
        fm: ForecastModel instance
        alert_row: 알림 데이터 (단일 행)
        include_recommendation: 권고사항 포함 여부
    
    Returns:
        설명 딕셔너리
    """
    contrib = local_feature_contrib(fm, alert_row, top_k=8)
    
    explanation = {
        "driver_summary": driver_sentence(contrib),
        "top_positive_drivers": contrib[contrib["direction"] == "POS"]["feature"].tolist()[:3],
        "top_negative_drivers": contrib[contrib["direction"] == "NEG"]["feature"].tolist()[:3],
        "contributions": contrib.to_dict('records'),
    }
    
    if include_recommendation:
        # 간단한 규칙 기반 권고사항
        pos_features = explanation["top_positive_drivers"]
        neg_features = explanation["top_negative_drivers"]
        
        recommendations = []
        
        for feat in neg_features:
            if "예금" in feat or "deposit" in feat.lower():
                recommendations.append("예금 이탈 방지를 위한 고객 컨택 필요")
            elif "대출" in feat or "loan" in feat.lower():
                recommendations.append("대출 상환 동향 모니터링 필요")
            elif "카드" in feat or "card" in feat.lower():
                recommendations.append("카드 활성화 캠페인 검토")
            elif "디지털" in feat or "digital" in feat.lower():
                recommendations.append("디지털 채널 이용 촉진 필요")
        
        explanation["recommendations"] = recommendations if recommendations else ["상세 분석 필요"]
    
    return explanation
