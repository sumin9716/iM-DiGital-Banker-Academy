"""
롤링 백테스트 및 평가 지표 모듈

- 롤링 윈도우 백테스트: 월별 재학습 + 시계열 CV
- 예측 평가: SMAPE, MAPE, wMAPE, Hit Rate
- 경보 평가: Precision, Recall, Lead Time
- 추천 평가: Uplift, 성공률
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ============== 평가 지표 함수 ==============

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error
    
    SMAPE = (1/n) * Σ |y_pred - y_true| / ((|y_true| + |y_pred|) / 2) * 100
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_pred - y_true)
    
    # 0으로 나누기 방지
    mask = denominator > 0
    if mask.sum() == 0:
        return 0.0
    
    return float(np.mean(diff[mask] / denominator[mask]) * 100)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error
    
    MAPE = (1/n) * Σ |y_pred - y_true| / |y_true| * 100
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mask = np.abs(y_true) > 0
    if mask.sum() == 0:
        return 0.0
    
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Mean Absolute Percentage Error (금액 가중)
    
    wMAPE = Σ |y_pred - y_true| / Σ |y_true| * 100
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    total_true = np.sum(np.abs(y_true))
    if total_true == 0:
        return 0.0
    
    return float(np.sum(np.abs(y_pred - y_true)) / total_true * 100)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (Coefficient of Determination)"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return float(1 - ss_res / ss_tot)


def hit_rate_top_n(
    y_true_delta: np.ndarray,
    y_pred_delta: np.ndarray,
    top_n: int = 50,
) -> float:
    """
    Hit Rate: TOP N 예측 상승 세그먼트 중 실제 상승 비율
    
    "상승 예상 TOP 50 세그먼트 적중률"
    
    Args:
        y_true_delta: 실제 변화량 (t+1 - t)
        y_pred_delta: 예측 변화량
        top_n: TOP N 개수
    
    Returns:
        적중률 (0~1)
    """
    y_true_delta = np.asarray(y_true_delta).flatten()
    y_pred_delta = np.asarray(y_pred_delta).flatten()
    
    n = len(y_pred_delta)
    if n == 0:
        return 0.0
    
    top_n = min(top_n, n)
    
    # 예측 상승 TOP N 인덱스
    pred_top_idx = np.argsort(y_pred_delta)[-top_n:]
    
    # 실제 상승 여부
    actual_increased = y_true_delta[pred_top_idx] > 0
    
    return float(np.mean(actual_increased))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy: 방향 예측 정확도
    
    증가/감소 방향이 맞는 비율
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    if len(true_direction) == 0:
        return 0.0
    
    return float(np.mean(true_direction == pred_direction))


# ============== 경보 평가 함수 ==============

def compute_alert_precision_recall(
    alerts_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    segment_col: str = "segment_id",
    month_col: str = "month",
    kpi_col: str = "kpi",
    threshold_pct: float = -0.20,
) -> Dict[str, float]:
    """
    경보 Precision/Recall 계산
    
    정답 라벨: 실제 급변 (MoM -20% 이하 또는 +20% 이상)
    
    Args:
        alerts_df: 경보 DataFrame (segment_id, month, kpi, alert_type)
        actual_df: 실제 값 DataFrame (segment_id, month, KPI 값)
        segment_col: 세그먼트 컬럼
        month_col: 월 컬럼
        kpi_col: KPI 컬럼
        threshold_pct: 급변 임계치
    
    Returns:
        {precision, recall, f1, n_alerts, n_actual_events}
    """
    if alerts_df.empty:
        return {"precision": 0, "recall": 0, "f1": 0, "n_alerts": 0, "n_actual_events": 0}
    
    # 경보 세그먼트-월 집합
    alert_keys = set(
        zip(alerts_df[segment_col], alerts_df[month_col].astype(str))
    )
    
    # 실제 급변 세그먼트-월 집합 (정답 라벨)
    actual_events = set()
    
    if not actual_df.empty and kpi_col in actual_df.columns:
        # MoM 변화율 계산
        actual_sorted = actual_df.sort_values([segment_col, month_col])
        actual_sorted["pct_change"] = actual_sorted.groupby(segment_col)[kpi_col].pct_change()
        
        # 급변 이벤트
        drop_events = actual_sorted[actual_sorted["pct_change"] <= threshold_pct]
        surge_events = actual_sorted[actual_sorted["pct_change"] >= abs(threshold_pct)]
        
        for _, row in pd.concat([drop_events, surge_events]).iterrows():
            actual_events.add((row[segment_col], str(row[month_col])))
    
    n_alerts = len(alert_keys)
    n_actual = len(actual_events)
    
    if n_alerts == 0 and n_actual == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_alerts": 0, "n_actual_events": 0}
    
    # True Positives: 경보 AND 실제 급변
    tp = len(alert_keys & actual_events)
    
    precision = tp / n_alerts if n_alerts > 0 else 0
    recall = tp / n_actual if n_actual > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_alerts": n_alerts,
        "n_actual_events": n_actual,
        "true_positives": tp,
    }


def compute_alert_lead_time(
    alerts_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    segment_col: str = "segment_id",
    month_col: str = "month",
    kpi_col: str = "kpi",
    threshold_pct: float = -0.20,
) -> Dict[str, float]:
    """
    경보 리드타임 계산: 경보가 실제 급변보다 얼마나 빨리 발생했는지
    
    Args:
        alerts_df: 경보 DataFrame
        actual_df: 실제 값 DataFrame
        
    Returns:
        {avg_lead_time_months, max_lead_time, pct_early_warning}
    """
    if alerts_df.empty or actual_df.empty:
        return {"avg_lead_time_months": 0, "max_lead_time": 0, "pct_early_warning": 0}
    
    lead_times = []
    
    # 각 경보에 대해
    for _, alert in alerts_df.iterrows():
        seg_id = alert[segment_col]
        alert_month = pd.to_datetime(alert[month_col])
        
        # 해당 세그먼트의 이후 급변 이벤트 찾기
        seg_actual = actual_df[actual_df[segment_col] == seg_id].copy()
        if seg_actual.empty or kpi_col not in seg_actual.columns:
            continue
        
        seg_actual = seg_actual.sort_values(month_col)
        seg_actual["pct_change"] = seg_actual[kpi_col].pct_change()
        
        # 경보 이후 급변 이벤트
        future_events = seg_actual[
            (pd.to_datetime(seg_actual[month_col]) > alert_month) &
            ((seg_actual["pct_change"] <= threshold_pct) | (seg_actual["pct_change"] >= abs(threshold_pct)))
        ]
        
        if not future_events.empty:
            first_event = future_events.iloc[0]
            event_month = pd.to_datetime(first_event[month_col])
            lead_time = (event_month - alert_month).days / 30  # 월 단위
            lead_times.append(lead_time)
    
    if not lead_times:
        return {"avg_lead_time_months": 0, "max_lead_time": 0, "pct_early_warning": 0}
    
    return {
        "avg_lead_time_months": round(np.mean(lead_times), 2),
        "max_lead_time": round(max(lead_times), 2),
        "pct_early_warning": round(len(lead_times) / len(alerts_df), 4),
    }


# ============== 추천 평가 함수 ==============

def compute_recommendation_uplift(
    actions_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    segment_col: str = "segment_id",
    month_col: str = "month",
    kpi_col: str = "예금총잔액",
    top_n: int = 100,
) -> Dict[str, float]:
    """
    추천 Uplift 평가: 추천 타깃 세그먼트의 다음 달 KPI 개선률 vs 랜덤
    
    "추천 타깃 세그먼트의 다음 달 KPI 개선률이 랜덤 대비 높은가"
    
    Args:
        actions_df: 추천 액션 DataFrame (segment_id, month, score)
        panel_df: 패널 DataFrame (segment_id, month, KPI 값)
        segment_col: 세그먼트 컬럼
        month_col: 월 컬럼
        kpi_col: KPI 컬럼
        top_n: TOP N 추천 세그먼트
    
    Returns:
        {
            target_avg_improvement: 타깃 평균 개선률,
            random_avg_improvement: 랜덤 평균 개선률,
            uplift: 상대적 개선,
            success_rate: 개선 세그먼트 비중
        }
    """
    if actions_df.empty or panel_df.empty:
        return {"target_avg_improvement": 0, "random_avg_improvement": 0, "uplift": 0, "success_rate": 0}
    
    # 추천 월 (가장 최근)
    action_month = actions_df[month_col].max()
    next_month = (pd.to_datetime(action_month).to_period("M") + 1).to_timestamp()
    
    # TOP N 추천 세그먼트
    top_actions = actions_df.nlargest(top_n, "score") if "score" in actions_df.columns else actions_df.head(top_n)
    target_segments = set(top_actions[segment_col])
    
    # 현재 월과 다음 월 KPI
    current = panel_df[panel_df[month_col] == action_month][[segment_col, kpi_col]]
    future = panel_df[panel_df[month_col] == next_month][[segment_col, kpi_col]]
    
    if current.empty or future.empty:
        return {"target_avg_improvement": 0, "random_avg_improvement": 0, "uplift": 0, "success_rate": 0}
    
    merged = current.merge(future, on=segment_col, suffixes=("_now", "_next"))
    merged["improvement"] = (merged[f"{kpi_col}_next"] - merged[f"{kpi_col}_now"]) / (merged[f"{kpi_col}_now"].abs() + 1e-9)
    
    # 타깃 세그먼트 개선률
    target_mask = merged[segment_col].isin(target_segments)
    target_improvement = merged[target_mask]["improvement"]
    
    # 랜덤 (비타깃) 세그먼트 개선률
    random_improvement = merged[~target_mask]["improvement"]
    
    target_avg = float(target_improvement.mean()) if len(target_improvement) > 0 else 0
    random_avg = float(random_improvement.mean()) if len(random_improvement) > 0 else 0
    
    # 성공률: 개선된 세그먼트 비중
    success_rate = float((target_improvement > 0).mean()) if len(target_improvement) > 0 else 0
    
    return {
        "target_avg_improvement": round(target_avg * 100, 2),  # %
        "random_avg_improvement": round(random_avg * 100, 2),  # %
        "uplift": round((target_avg - random_avg) * 100, 2),  # %p
        "success_rate": round(success_rate * 100, 2),  # %
        "n_target": len(target_improvement),
        "n_random": len(random_improvement),
    }


# ============== 롤링 백테스트 ==============

@dataclass
class BacktestConfig:
    """백테스트 설정"""
    train_start: pd.Timestamp = None  # 학습 시작월
    train_end: pd.Timestamp = None    # 초기 학습 종료월
    test_start: pd.Timestamp = None   # 테스트 시작월
    test_end: pd.Timestamp = None     # 테스트 종료월
    horizons: List[int] = field(default_factory=lambda: [1, 2, 3])
    expanding_window: bool = True     # True=확장 윈도우, False=고정 윈도우
    window_size: int = 24             # 고정 윈도우 크기 (월)
    retrain_frequency: int = 1        # 재학습 주기 (월)


@dataclass
class BacktestResult:
    """백테스트 결과"""
    predictions: pd.DataFrame
    metrics_by_month: pd.DataFrame
    metrics_summary: Dict[str, float]
    alerts_evaluation: Optional[Dict] = None
    recommendation_evaluation: Optional[Dict] = None


def run_rolling_backtest(
    panel: pd.DataFrame,
    kpi_col: str,
    train_func: Callable,
    predict_func: Callable,
    config: BacktestConfig,
    segment_col: str = "segment_id",
    month_col: str = "month",
    verbose: bool = True,
) -> BacktestResult:
    """
    롤링 윈도우 백테스트 실행
    
    Args:
        panel: 패널 데이터
        kpi_col: 타겟 KPI 컬럼
        train_func: 학습 함수 (df, train_end) -> model
        predict_func: 예측 함수 (model, df) -> predictions
        config: 백테스트 설정
        segment_col: 세그먼트 컬럼
        month_col: 월 컬럼
        verbose: 출력 여부
    
    Returns:
        BacktestResult
    """
    panel = panel.copy()
    panel[month_col] = pd.to_datetime(panel[month_col])
    
    # 테스트 기간 월 리스트
    all_months = sorted(panel[month_col].unique())
    test_months = [m for m in all_months if config.test_start <= m <= config.test_end]
    
    all_predictions = []
    all_metrics = []
    
    for i, test_month in enumerate(test_months):
        if i % config.retrain_frequency != 0:
            continue
        
        if verbose:
            print(f"  Rolling backtest: {test_month.strftime('%Y-%m')}")
        
        # 학습 데이터 범위 결정
        if config.expanding_window:
            train_start = config.train_start if config.train_start else all_months[0]
        else:
            train_start = test_month - pd.DateOffset(months=config.window_size)
        
        train_end = test_month - pd.DateOffset(months=1)
        
        # 학습 데이터
        train_mask = (panel[month_col] >= train_start) & (panel[month_col] <= train_end)
        train_df = panel[train_mask]
        
        # 테스트 데이터
        test_df = panel[panel[month_col] == test_month]
        
        if train_df.empty or test_df.empty:
            continue
        
        try:
            # 모델 학습
            for h in config.horizons:
                model = train_func(train_df, kpi_col, h, train_end)
                
                # 예측
                preds = predict_func(model, test_df)
                
                # 실제값
                future_month = test_month + pd.DateOffset(months=h)
                future_df = panel[panel[month_col] == future_month]
                
                if future_df.empty:
                    continue
                
                # 결과 저장
                merged = test_df[[segment_col, month_col]].copy()
                merged["prediction"] = preds
                merged["horizon"] = h
                merged["pred_month"] = future_month
                
                # 실제값 병합
                actual = future_df[[segment_col, kpi_col]].rename(columns={kpi_col: "actual"})
                merged = merged.merge(actual, on=segment_col, how="left")
                
                all_predictions.append(merged)
                
                # 월별 메트릭 계산
                valid = merged.dropna(subset=["actual", "prediction"])
                if len(valid) > 0:
                    metrics = {
                        "test_month": test_month,
                        "horizon": h,
                        "rmse": rmse(valid["actual"].values, valid["prediction"].values),
                        "smape": smape(valid["actual"].values, valid["prediction"].values),
                        "r2": r2_score(valid["actual"].values, valid["prediction"].values),
                        "n_samples": len(valid),
                    }
                    all_metrics.append(metrics)
        
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            continue
    
    # 결과 정리
    predictions_df = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    metrics_df = pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()
    
    # 전체 요약 메트릭
    summary = {}
    if not metrics_df.empty:
        for h in config.horizons:
            h_metrics = metrics_df[metrics_df["horizon"] == h]
            if not h_metrics.empty:
                summary[f"h{h}_avg_rmse"] = round(h_metrics["rmse"].mean(), 4)
                summary[f"h{h}_avg_smape"] = round(h_metrics["smape"].mean(), 4)
                summary[f"h{h}_avg_r2"] = round(h_metrics["r2"].mean(), 4)
    
    return BacktestResult(
        predictions=predictions_df,
        metrics_by_month=metrics_df,
        metrics_summary=summary,
    )


def evaluate_all_metrics(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    alerts: pd.DataFrame,
    actions: pd.DataFrame,
    kpi_col: str,
    segment_col: str = "segment_id",
    month_col: str = "month",
) -> Dict[str, Dict]:
    """
    모든 평가 지표 종합 계산
    
    Returns:
        {
            forecast: {rmse, smape, r2, hit_rate, ...},
            alerts: {precision, recall, lead_time, ...},
            recommendations: {uplift, success_rate, ...}
        }
    """
    results = {
        "forecast": {},
        "alerts": {},
        "recommendations": {},
    }
    
    # 1. 예측 평가
    if not predictions.empty and "actual" in predictions.columns and "prediction" in predictions.columns:
        valid = predictions.dropna(subset=["actual", "prediction"])
        if len(valid) > 0:
            results["forecast"] = {
                "rmse": round(rmse(valid["actual"].values, valid["prediction"].values), 4),
                "smape": round(smape(valid["actual"].values, valid["prediction"].values), 4),
                "mape": round(mape(valid["actual"].values, valid["prediction"].values), 4),
                "wmape": round(wmape(valid["actual"].values, valid["prediction"].values), 4),
                "r2": round(r2_score(valid["actual"].values, valid["prediction"].values), 4),
                "n_samples": len(valid),
            }
            
            # Hit Rate (변화량 기준)
            if "delta_pred" in valid.columns and "delta_actual" in valid.columns:
                results["forecast"]["hit_rate_top50"] = round(
                    hit_rate_top_n(valid["delta_actual"].values, valid["delta_pred"].values, 50), 4
                )
    
    # 2. 경보 평가
    if not alerts.empty and not panel.empty:
        pr_result = compute_alert_precision_recall(
            alerts, panel, segment_col, month_col, kpi_col
        )
        results["alerts"] = pr_result
    
    # 3. 추천 평가
    if not actions.empty and not panel.empty:
        uplift_result = compute_recommendation_uplift(
            actions, panel, segment_col, month_col, kpi_col
        )
        results["recommendations"] = uplift_result
    
    return results


def generate_evaluation_report(
    results: Dict[str, Dict],
    output_path: Optional[str] = None,
) -> str:
    """
    평가 결과 리포트 생성
    
    Args:
        results: evaluate_all_metrics 결과
        output_path: 저장 경로 (선택)
    
    Returns:
        리포트 문자열
    """
    lines = [
        "=" * 60,
        "모델 평가 리포트",
        "=" * 60,
        "",
    ]
    
    # 예측 평가
    if results.get("forecast"):
        fc = results["forecast"]
        lines.append("■ 예측 성능")
        lines.append(f"  - RMSE: {fc.get('rmse', 'N/A')}")
        lines.append(f"  - SMAPE: {fc.get('smape', 'N/A')}%")
        lines.append(f"  - R²: {fc.get('r2', 'N/A')}")
        if "hit_rate_top50" in fc:
            lines.append(f"  - Hit Rate (TOP 50): {fc['hit_rate_top50']:.1%}")
        lines.append("")
    
    # 경보 평가
    if results.get("alerts"):
        al = results["alerts"]
        lines.append("■ 경보 성능")
        lines.append(f"  - Precision: {al.get('precision', 0):.1%}")
        lines.append(f"  - Recall: {al.get('recall', 0):.1%}")
        lines.append(f"  - F1 Score: {al.get('f1', 0):.3f}")
        lines.append(f"  - 경보 수: {al.get('n_alerts', 0)}")
        lines.append(f"  - 실제 이벤트 수: {al.get('n_actual_events', 0)}")
        lines.append("")
    
    # 추천 평가
    if results.get("recommendations"):
        rec = results["recommendations"]
        lines.append("■ 추천 성능")
        lines.append(f"  - 타깃 평균 개선률: {rec.get('target_avg_improvement', 0):.2f}%")
        lines.append(f"  - 랜덤 평균 개선률: {rec.get('random_avg_improvement', 0):.2f}%")
        lines.append(f"  - Uplift: {rec.get('uplift', 0):+.2f}%p")
        lines.append(f"  - 성공률: {rec.get('success_rate', 0):.1f}%")
        lines.append("")
    
    report = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
    
    return report
