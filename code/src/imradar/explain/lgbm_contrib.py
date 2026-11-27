from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

from imradar.models.forecast_lgbm import ForecastModel, predict_with_model


# ============== SHAP 기반 피처 기여도 분석 ==============

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


def compute_shap_values(
    fm: ForecastModel,
    df: pd.DataFrame,
    use_tree_shap: bool = True,
) -> np.ndarray:
    """
    SHAP 값 계산 (LightGBM Tree SHAP 또는 pred_contrib)
    
    Args:
        fm: ForecastModel instance
        df: Input DataFrame
        use_tree_shap: True면 SHAP 라이브러리 사용, False면 pred_contrib 사용
    
    Returns:
        SHAP values array (n_samples, n_features)
    """
    X = df[fm.feature_cols].copy()
    for c in fm.categorical_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    
    if use_tree_shap:
        try:
            import shap
            explainer = shap.TreeExplainer(fm.model)
            shap_values = explainer.shap_values(X)
            return shap_values
        except ImportError:
            print("SHAP 라이브러리가 설치되지 않음. pred_contrib 사용.")
    
    # Fallback to pred_contrib
    contribs = fm.model.predict(X, pred_contrib=True)
    return contribs[:, :-1]  # bias 제외


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


# ============== 확장된 SHAP 분석 기능 ==============

def generate_segment_contrib_detail(
    fm: ForecastModel,
    segment_df: pd.DataFrame,
    segment_id: str,
    target_month: pd.Timestamp,
    top_k: int = 5,
) -> Dict[str, any]:
    """
    특정 세그먼트-월에 대한 상세 설명 생성
    
    Args:
        fm: ForecastModel
        segment_df: 해당 세그먼트 데이터
        segment_id: 세그먼트 ID
        target_month: 분석 대상 월
        top_k: 상위 드라이버 수
    
    Returns:
        설명 딕셔너리
    """
    # 해당 세그먼트-월 데이터 추출
    mask = (segment_df["segment_id"] == segment_id) & (segment_df["month"] == target_month)
    row = segment_df[mask]
    
    if row.empty:
        return {"error": "해당 세그먼트-월 데이터 없음"}
    
    row_df = row.iloc[[0]]
    
    # 기여도 계산
    contrib = local_feature_contrib(fm, row_df, top_k=top_k * 2)
    
    # 피처별 그룹화
    internal_drivers = []
    external_drivers = []
    
    external_keywords = ["ESI", "fed", "usd", "wti", "brent", "export", "import", 
                        "call_rate", "koribor", "cpi", "ppi", "retail", "employment"]
    
    for _, r in contrib.iterrows():
        feat = r["feature"]
        is_external = any(kw in feat.lower() for kw in external_keywords)
        
        driver_info = {
            "feature": feat,
            "display_name": _format_feature_name(feat),
            "contribution": float(r["contribution"]),
            "direction": r["direction"],
        }
        
        if is_external:
            external_drivers.append(driver_info)
        else:
            internal_drivers.append(driver_info)
    
    # 설명 문장 생성
    explanation_parts = []
    
    if internal_drivers:
        top_internal = sorted(internal_drivers, key=lambda x: abs(x["contribution"]), reverse=True)[:3]
        internal_names = [d["display_name"] for d in top_internal]
        explanation_parts.append(f"내부 요인: {', '.join(internal_names)}")
    
    if external_drivers:
        top_external = sorted(external_drivers, key=lambda x: abs(x["contribution"]), reverse=True)[:3]
        external_names = [d["display_name"] for d in top_external]
        explanation_parts.append(f"외부 요인: {', '.join(external_names)}")
    
    return {
        "segment_id": segment_id,
        "month": str(target_month),
        "internal_drivers": internal_drivers[:top_k],
        "external_drivers": external_drivers[:top_k],
        "explanation": " / ".join(explanation_parts) if explanation_parts else "주요 드라이버 미확인",
        "all_contributions": contrib.to_dict("records"),
    }


def generate_narrative_report(
    fm: ForecastModel,
    df: pd.DataFrame,
    segment_id: str,
    target_month: pd.Timestamp,
    actual: float,
    predicted: float,
) -> str:
    """
    사람이 읽기 쉬운 형태의 서술형 리포트 생성
    
    Args:
        fm: ForecastModel
        df: 분석 데이터
        segment_id: 세그먼트 ID
        target_month: 분석 월
        actual: 실제값
        predicted: 예측값
    
    Returns:
        서술형 리포트 문자열
    """
    # 기여도 분석
    explanation = generate_segment_contrib_detail(fm, df, segment_id, target_month)
    
    if "error" in explanation:
        return f"분석 오류: {explanation['error']}"
    
    # 잔차 계산
    residual = actual - predicted
    residual_ratio = residual / (abs(predicted) + 1e-9) * 100
    direction = "상승" if residual > 0 else "하락"
    
    # 리포트 작성
    report_lines = [
        "=== 세그먼트 분석 리포트 ===",
        "",
        f"■ 세그먼트: {segment_id}",
        f"■ 분석 월: {target_month.strftime('%Y년 %m월')}",
        "",
        "■ 성과 요약",
        f"  - 실적: {actual:,.0f}",
        f"  - 예측: {predicted:,.0f}",
        f"  - 차이: {residual:+,.0f} ({residual_ratio:+.1f}%)",
        f"  - 방향: 예측 대비 {direction}",
        "",
        "■ 주요 원인 분석",
    ]
    
    # 내부 요인
    if explanation.get("internal_drivers"):
        report_lines.append("  [내부 요인]")
        for i, d in enumerate(explanation["internal_drivers"][:3], 1):
            direction_txt = "+" if d["contribution"] > 0 else "-"
            report_lines.append(f"    {i}. {d['display_name']} ({direction_txt}{abs(d['contribution']):.2f})")
    
    # 외부 요인
    if explanation.get("external_drivers"):
        report_lines.append("  [외부 요인]")
        for i, d in enumerate(explanation["external_drivers"][:3], 1):
            direction_txt = "+" if d["contribution"] > 0 else "-"
            report_lines.append(f"    {i}. {d['display_name']} ({direction_txt}{abs(d['contribution']):.2f})")
    
    # 권고사항
    report_lines.append("")
    report_lines.append("■ 권고사항")
    
    recommendations = _generate_recommendations(explanation, residual_ratio)
    for i, rec in enumerate(recommendations[:3], 1):
        report_lines.append(f"  {i}. {rec}")
    
    return "\n".join(report_lines)


def _generate_recommendations(
    explanation: Dict[str, any],
    residual_ratio: float,
) -> List[str]:
    """
    분석 결과 기반 권고사항 생성
    
    Args:
        explanation: 설명 딕셔너리
        residual_ratio: 잔차 비율 (%)
    
    Returns:
        권고사항 리스트
    """
    recommendations = []
    
    # 잔차 방향에 따른 기본 권고
    if residual_ratio < -20:
        recommendations.append("실적 급감 원인 긴급 분석 및 대응 필요")
    elif residual_ratio < -10:
        recommendations.append("실적 하락 추세 모니터링 및 조기 대응 검토")
    elif residual_ratio > 20:
        recommendations.append("성과 상승 요인 분석 및 타 세그먼트 확산 검토")
    elif residual_ratio > 10:
        recommendations.append("긍정적 추세 지속 여부 모니터링")
    
    # 드라이버 기반 권고
    all_drivers = (
        explanation.get("internal_drivers", []) + 
        explanation.get("external_drivers", [])
    )
    
    for driver in all_drivers[:5]:
        feat = driver["feature"].lower()
        contrib = driver["contribution"]
        
        # 하락 원인 드라이버
        if contrib < 0:
            if "예금" in feat:
                recommendations.append("예금 이탈 방지를 위한 고금리 상품 프로모션 검토")
            elif "대출" in feat:
                recommendations.append("대출 고객 이탈 방지 및 리텐션 프로그램 강화")
            elif "카드" in feat:
                recommendations.append("카드 사용 활성화 캠페인 기획")
            elif "디지털" in feat:
                recommendations.append("디지털 채널 이용 촉진 및 온보딩 개선")
            elif "fx" in feat or "외환" in feat:
                recommendations.append("FX 거래 활성화 및 환율 변동 리스크 헤지 상품 제안")
            elif "usd" in feat or "환율" in feat:
                recommendations.append("환율 변동 영향 모니터링 및 고객 커뮤니케이션")
            elif "금리" in feat or "rate" in feat:
                recommendations.append("금리 민감도 분석 및 상품 포트폴리오 재검토")
        
        # 상승 원인 드라이버
        else:
            if "예금" in feat:
                recommendations.append("예금 유치 성공 요인 분석 및 타 세그먼트 적용")
            elif "대출" in feat:
                recommendations.append("대출 수요 증가 추세 활용, 교차판매 기회 탐색")
    
    # 중복 제거
    seen = set()
    unique_recs = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recs.append(rec)
    
    return unique_recs if unique_recs else ["상세 분석 필요"]


def generate_monthly_top_alerts_report(
    fm: ForecastModel,
    df: pd.DataFrame,
    target_month: pd.Timestamp,
    top_n: int = 10,
) -> str:
    """
    월간 TOP 경보 세그먼트 리포트 생성
    
    Args:
        fm: ForecastModel
        df: 분석 데이터 (예측값, 실제값 포함)
        target_month: 분석 월
        top_n: 상위 알림 수
    
    Returns:
        종합 리포트 문자열
    """
    # 해당 월 데이터 필터링
    month_data = df[df["month"] == target_month].copy()
    
    if month_data.empty:
        return f"데이터 없음: {target_month}"
    
    # 잔차 계산 (예측값, 실제값이 있다고 가정)
    if "predicted" not in month_data.columns:
        predictions = predict_with_model(fm, month_data)
        month_data["predicted"] = predictions
    
    if "actual" not in month_data.columns and fm.kpi in month_data.columns:
        month_data["actual"] = month_data[fm.kpi]
    
    eps = 1e-9
    month_data["residual"] = month_data["actual"] - month_data["predicted"]
    month_data["residual_ratio"] = month_data["residual"] / (month_data["predicted"].abs() + eps)
    
    # 상위/하위 알림 추출
    top_drop = month_data.nsmallest(top_n // 2, "residual_ratio")
    top_surge = month_data.nlargest(top_n // 2, "residual_ratio")
    
    report_lines = [
        f"{'='*60}",
        "월간 세그먼트 경보 리포트",
        f"분석 월: {target_month.strftime('%Y년 %m월')}",
        f"{'='*60}",
        "",
        f"■ 급감 세그먼트 TOP {len(top_drop)}",
        "-" * 40,
    ]
    
    for idx, (_, row) in enumerate(top_drop.iterrows(), 1):
        segment_id = row.get("segment_id", "N/A")
        residual_pct = row["residual_ratio"] * 100
        
        # 간단한 드라이버 분석
        try:
            row_df = pd.DataFrame([row])
            contrib = local_feature_contrib(fm, row_df[fm.feature_cols].copy(), top_k=4)
            top_drivers = [_format_feature_name(f) for f in contrib.head(3)["feature"]]
            driver_txt = ", ".join(top_drivers)
        except:
            driver_txt = "분석 불가"
        
        report_lines.append(f"{idx}. {segment_id}: {residual_pct:+.1f}%")
        report_lines.append(f"   드라이버: {driver_txt}")
        report_lines.append("")
    
    report_lines.append("")
    report_lines.append(f"■ 급증 세그먼트 TOP {len(top_surge)}")
    report_lines.append("-" * 40)
    
    for idx, (_, row) in enumerate(top_surge.iterrows(), 1):
        segment_id = row.get("segment_id", "N/A")
        residual_pct = row["residual_ratio"] * 100
        
        try:
            row_df = pd.DataFrame([row])
            contrib = local_feature_contrib(fm, row_df[fm.feature_cols].copy(), top_k=4)
            top_drivers = [_format_feature_name(f) for f in contrib.head(3)["feature"]]
            driver_txt = ", ".join(top_drivers)
        except:
            driver_txt = "분석 불가"
        
        report_lines.append(f"{idx}. {segment_id}: {residual_pct:+.1f}%")
        report_lines.append(f"   드라이버: {driver_txt}")
        report_lines.append("")
    
    return "\n".join(report_lines)


# ============== 자연어 문장형 설명 생성 ==============

# 피처-설명 매핑 사전
FEATURE_DESCRIPTIONS = {
    # 환율/FX 관련
    "usd_krw": "환율",
    "fx_level": "환율 수준",
    "fx_mom": "환율 변동",
    "fx_vol": "환율 변동성",
    "fx_regime": "환율 국면",
    
    # 금리 관련
    "fed_rate": "미국 금리",
    "call_rate": "콜금리",
    "koribor": "KORIBOR",
    "govt_bond": "국고채 금리",
    "corp_bond": "회사채 금리",
    
    # 유가/원자재
    "wti": "유가(WTI)",
    "brent": "유가(Brent)",
    "oil": "유가",
    
    # 경기 지표
    "esi": "경기심리지수(ESI)",
    "kasi": "기업경기심리지수",
    "cpi": "소비자물가",
    "ppi": "생산자물가",
    
    # 무역
    "export": "수출",
    "import": "수입",
    "trade_balance": "무역수지",
    
    # 부동산
    "housing_price": "주택가격",
    "jeonse_price": "전세가격",
    "apt_transaction": "아파트 거래량",
    "apt_vol": "부동산 거래량",
    
    # KPI 관련
    "예금총잔액": "예금잔액",
    "대출총잔액": "대출잔액",
    "카드총사용": "카드사용액",
    "디지털거래금액": "디지털거래",
    "순유입": "자금순유입",
    "FX총액": "외환거래",
    "한도소진율": "여신한도 사용률",
    "디지털비중": "디지털 채널 비중",
}

# 변화 유형별 설명
CHANGE_TYPE_DESCRIPTIONS = {
    "lag": "과거 {period}개월 전 값",
    "mom": "전월 대비 변화",
    "yoy": "전년 동월 대비 변화",
    "roll_mean": "{window}개월 이동평균",
    "roll_std": "{window}개월 변동성",
    "zscore": "이상치 수준",
    "ma": "이동평균",
    "vol": "변동성",
}


def parse_feature_name_v2(feature: str) -> Tuple[str, str, Optional[int]]:
    """
    피처명 파싱 (v2)
    
    Args:
        feature: 피처명 (예: "usd_krw_mean__lag3", "FX총액__mom_pct")
    
    Returns:
        (base_name, change_type, period)
    """
    import re
    
    parts = feature.split("__")
    base_name = parts[0]
    change_type = ""
    period = None
    
    if len(parts) > 1:
        suffix = parts[1]
        
        if suffix.startswith("lag"):
            change_type = "lag"
            try:
                period = int(suffix[3:])
            except:
                period = None
        elif "mom" in suffix:
            change_type = "mom"
        elif "yoy" in suffix:
            change_type = "yoy"
        elif "roll" in suffix and "mean" in suffix:
            change_type = "roll_mean"
            match = re.search(r'\d+', suffix)
            if match:
                period = int(match.group())
        elif "roll" in suffix and "std" in suffix:
            change_type = "roll_std"
            match = re.search(r'\d+', suffix)
            if match:
                period = int(match.group())
        elif "zscore" in suffix:
            change_type = "zscore"
        elif "ma" in suffix:
            change_type = "ma"
            match = re.search(r'\d+', suffix)
            if match:
                period = int(match.group())
    
    return base_name, change_type, period


def get_readable_feature_name(feature: str) -> str:
    """
    피처명을 읽기 쉬운 한국어로 변환
    
    Args:
        feature: 피처명
    
    Returns:
        한국어 설명
    """
    base_name, change_type, period = parse_feature_name_v2(feature)
    
    # 기본 이름 변환
    readable_base = base_name
    for key, desc in FEATURE_DESCRIPTIONS.items():
        if key in base_name.lower():
            readable_base = desc
            break
    
    # 변화 유형 추가
    if change_type == "lag" and period:
        return f"{readable_base} ({period}개월 전)"
    elif change_type == "mom":
        return f"{readable_base} (전월비)"
    elif change_type == "yoy":
        return f"{readable_base} (전년비)"
    elif change_type == "roll_mean" and period:
        return f"{readable_base} ({period}개월 평균)"
    elif change_type == "roll_std" and period:
        return f"{readable_base} ({period}개월 변동성)"
    elif change_type == "zscore":
        return f"{readable_base} (이상도)"
    elif change_type == "ma" and period:
        return f"{readable_base} ({period}일 이동평균)"
    else:
        return readable_base


def generate_explanation_sentence(
    segment_id: str,
    kpi: str,
    direction: str,
    top_contributors: pd.DataFrame,
    magnitude: float = 0,
) -> str:
    """
    단일 세그먼트에 대한 자연어 설명 문장 생성
    
    Args:
        segment_id: 세그먼트 ID
        kpi: KPI명
        direction: 변화 방향 ("increase", "decrease", "anomaly")
        top_contributors: 상위 기여 피처 DataFrame (feature, contribution 컬럼)
        magnitude: 변화 크기 (%)
    
    Returns:
        자연어 설명 문장
    """
    # 방향 텍스트
    direction_text = {
        "increase": "상승",
        "decrease": "하락",
        "anomaly": "이상 변동",
        "surge": "급증",
        "drop": "급감",
    }.get(direction, "변동")
    
    # 상위 기여 요인 추출
    positive_factors = []
    negative_factors = []
    
    for _, row in top_contributors.iterrows():
        feat_name = get_readable_feature_name(row["feature"])
        contrib = row["contribution"]
        
        if contrib > 0:
            positive_factors.append(feat_name)
        else:
            negative_factors.append(feat_name)
    
    # 문장 구성
    sentences = []
    
    # 메인 문장
    if magnitude != 0:
        main_sentence = f"해당 세그먼트의 {kpi}이(가) 예측 대비 {abs(magnitude):.1f}% {direction_text}하였습니다."
    else:
        main_sentence = f"해당 세그먼트의 {kpi}에서 {direction_text}이 감지되었습니다."
    sentences.append(main_sentence)
    
    # 원인 문장
    if positive_factors:
        pos_text = ", ".join(positive_factors[:3])
        sentences.append(f"주요 상승 요인: {pos_text}")
    
    if negative_factors:
        neg_text = ", ".join(negative_factors[:3])
        sentences.append(f"주요 하락 요인: {neg_text}")
    
    # 외부 요인 강조
    external_keywords = ["환율", "금리", "유가", "물가", "수출", "수입", "경기", "부동산"]
    external_factors = [f for f in (positive_factors + negative_factors) 
                        if any(kw in f for kw in external_keywords)]
    
    if external_factors:
        ext_text = ", ".join(external_factors[:2])
        sentences.append(f"※ 외부 거시경제 요인({ext_text})의 영향이 관측됨")
    
    return " ".join(sentences)


def generate_segment_explanation(
    fm: ForecastModel,
    row_df: pd.DataFrame,
    kpi: str,
    residual_pct: float,
    alert_type: str = "normal",
    top_k: int = 6,
) -> str:
    """
    세그먼트별 상세 설명 생성
    
    Args:
        fm: ForecastModel instance
        row_df: 단일 행 DataFrame
        kpi: KPI명
        residual_pct: 잔차 비율 (%)
        alert_type: 알림 유형
        top_k: 상위 기여 피처 수
    
    Returns:
        자연어 설명 문장
    """
    try:
        # 피처 기여도 계산
        contrib = local_feature_contrib(fm, row_df, top_k=top_k)
        
        # 방향 결정
        if residual_pct > 20:
            direction = "surge"
        elif residual_pct < -20:
            direction = "drop"
        elif residual_pct > 0:
            direction = "increase"
        elif residual_pct < 0:
            direction = "decrease"
        else:
            direction = "normal"
        
        segment_id = row_df.iloc[0].get("segment_id", "Unknown") if "segment_id" in row_df.columns else "Unknown"
        
        explanation = generate_explanation_sentence(
            segment_id=segment_id,
            kpi=kpi,
            direction=direction,
            top_contributors=contrib,
            magnitude=residual_pct,
        )
        
        return explanation
    
    except Exception as e:
        return f"설명 생성 실패: {e}"


def generate_batch_explanations(
    fm: ForecastModel,
    df: pd.DataFrame,
    kpi: str,
    residual_col: str = "residual_pct",
    alert_type_col: str = "alert_type",
    top_n: int = 20,
) -> pd.DataFrame:
    """
    배치로 여러 세그먼트의 설명 생성
    
    Args:
        fm: ForecastModel instance
        df: 데이터 DataFrame (segment_id, residual 포함)
        kpi: KPI명
        residual_col: 잔차 컬럼명
        alert_type_col: 알림 유형 컬럼명
        top_n: 설명 생성할 세그먼트 수
    
    Returns:
        DataFrame with explanations
    """
    # 상위/하위 세그먼트 선택
    sorted_df = df.sort_values(residual_col, key=abs, ascending=False).head(top_n)
    
    results = []
    
    for _, row in sorted_df.iterrows():
        segment_id = row.get("segment_id", "Unknown")
        residual_pct = row.get(residual_col, 0) * 100 if residual_col in row else 0
        alert_type = row.get(alert_type_col, "normal")
        
        try:
            row_df = pd.DataFrame([row])
            explanation = generate_segment_explanation(
                fm=fm,
                row_df=row_df[fm.feature_cols] if all(c in row_df.columns for c in fm.feature_cols) else row_df,
                kpi=kpi,
                residual_pct=residual_pct,
                alert_type=alert_type,
            )
        except Exception as e:
            explanation = f"설명 생성 실패: {e}"
        
        results.append({
            "segment_id": segment_id,
            "kpi": kpi,
            "residual_pct": residual_pct,
            "alert_type": alert_type,
            "explanation": explanation,
        })
    
    return pd.DataFrame(results)


def summarize_drivers_by_category(
    fm: ForecastModel,
    df: pd.DataFrame = None,
    top_k: int = 10,
) -> Dict[str, Dict]:
    """
    전체 데이터에서 카테고리별 주요 드라이버 요약
    
    Returns:
        {
            "internal": {"예금잔액 (1개월 전)": 15.2, ...},
            "external": {"환율 (전월비)": 8.5, ...},
            "segment": {"업종_중분류": 12.1, ...}
        }
    """
    # 글로벌 피처 중요도
    importance = global_feature_importance(fm)
    
    categories = {
        "internal": {},  # 내부 KPI 관련
        "external": {},  # 외부 거시경제
        "segment": {},   # 세그먼트 속성
        "time": {},      # 시간 관련
    }
    
    external_keywords = ["usd", "krw", "fed", "rate", "wti", "brent", "esi", "kasi", 
                         "cpi", "ppi", "export", "import", "trade", "housing", "jeonse", "apt"]
    segment_keywords = ["업종", "지역", "등급", "전담", "사업장", "시도"]
    time_keywords = ["month", "quarter", "year", "sin", "cos"]
    
    for _, row in importance.head(50).iterrows():
        feat = row["feature"]
        imp = row.get("importance_pct", row.get("importance_gain_pct", 0))
        readable = get_readable_feature_name(feat)
        
        feat_lower = feat.lower()
        
        if any(kw in feat_lower for kw in external_keywords):
            categories["external"][readable] = imp
        elif any(kw in feat for kw in segment_keywords):
            categories["segment"][readable] = imp
        elif any(kw in feat_lower for kw in time_keywords):
            categories["time"][readable] = imp
        else:
            categories["internal"][readable] = imp
    
    # 각 카테고리별 상위 N개만 유지
    for cat in categories:
        sorted_items = sorted(categories[cat].items(), key=lambda x: -x[1])[:top_k]
        categories[cat] = dict(sorted_items)
    
    return categories


def generate_executive_summary_report(
    fm: ForecastModel,
    alerts_df: pd.DataFrame,
    kpi: str,
    month: pd.Timestamp,
) -> str:
    """
    경영진용 한 페이지 요약 생성
    
    Args:
        fm: ForecastModel instance
        alerts_df: 경보 DataFrame
        kpi: KPI명
        month: 분석 월
    
    Returns:
        요약 문자열
    """
    lines = [
        "=" * 60,
        f"📊 {kpi} 월간 분석 요약",
        f"분석 기준: {month.strftime('%Y년 %m월')}",
        "=" * 60,
        "",
    ]
    
    # 드라이버 분석
    try:
        drivers = summarize_drivers_by_category(fm, None)
        
        if drivers.get("external"):
            lines.append("■ 주요 외부 영향 요인")
            for feat, imp in list(drivers["external"].items())[:3]:
                lines.append(f"  • {feat}: 중요도 {imp:.1f}%")
            lines.append("")
        
        if drivers.get("internal"):
            lines.append("■ 주요 내부 영향 요인")
            for feat, imp in list(drivers["internal"].items())[:3]:
                lines.append(f"  • {feat}: 중요도 {imp:.1f}%")
            lines.append("")
    except:
        pass
    
    # 경보 요약
    if not alerts_df.empty:
        n_drop = len(alerts_df[alerts_df.get("alert_type", "") == "DROP"])
        n_surge = len(alerts_df[alerts_df.get("alert_type", "") == "SPIKE"])
        n_critical = len(alerts_df[alerts_df.get("severity", "") == "CRITICAL"])
        
        lines.append("■ 경보 현황")
        lines.append(f"  • 급감 세그먼트: {n_drop}건")
        lines.append(f"  • 급증 세그먼트: {n_surge}건")
        lines.append(f"  • Critical 등급: {n_critical}건")
        lines.append("")
        
        # 대표 경보 1건 설명
        if n_critical > 0:
            critical = alerts_df[alerts_df.get("severity", "") == "CRITICAL"].iloc[0]
            seg_id = critical.get("segment_id", "N/A")
            lines.append(f"■ 대표 경보 세그먼트: {seg_id}")
            
            if "explanation" in critical:
                lines.append(f"  {critical['explanation']}")
            lines.append("")
    
    # 권고사항
    lines.append("■ 권고사항")
    lines.append("  1. 급감 세그먼트에 대한 긴급 RM 컨택 필요")
    lines.append("  2. 외부 거시경제 변동에 따른 리스크 모니터링 강화")
    lines.append("  3. Critical 등급 세그먼트 주간 추적 관리")
    
    return "\n".join(lines)
