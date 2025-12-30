"""
iM ONEderful Segment Radar - Backend API Server
FastAPI 기반 REST API 서버
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import date

# Add src to path
REPO_ROOT = Path(__file__).parent.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from imradar.config import RadarConfig

app = FastAPI(
    title="iM ONEderful Segment Radar API",
    description="세그먼트 기반 금융 KPI 예측 및 리스크 모니터링 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Pydantic Models ==============

class SegmentBase(BaseModel):
    """세그먼트 기본 정보"""
    segment_id: str = Field(..., description="세그먼트 고유 ID")
    업종_중분류: str = Field(..., description="업종 중분류")
    사업장_시도: str = Field(..., description="사업장 시도")
    법인_고객등급: str = Field(..., description="법인 고객등급")
    전담고객여부: str = Field(..., description="전담고객 여부")
    customer_count: int = Field(..., description="고객 수")


class SegmentKPI(SegmentBase):
    """세그먼트 KPI 정보"""
    month: str = Field(..., description="기준 월 (YYYY-MM)")
    예금총잔액: float = Field(..., description="예금 총잔액")
    대출총잔액: float = Field(..., description="대출 총잔액")
    카드총사용: float = Field(..., description="카드 총사용액")
    디지털거래금액: float = Field(..., description="디지털 거래금액")
    순유입: float = Field(..., description="순유입")
    한도소진율: Optional[float] = Field(None, description="한도 소진율")
    디지털비중: Optional[float] = Field(None, description="디지털 거래 비중")


class ForecastResult(BaseModel):
    """예측 결과"""
    segment_id: str
    target_kpi: str = Field(..., description="예측 대상 KPI")
    horizon: int = Field(..., description="예측 기간 (개월)")
    forecast_month: str = Field(..., description="예측 대상 월")
    predicted_value: float = Field(..., description="예측값")
    actual_value: Optional[float] = Field(None, description="실제값 (있는 경우)")
    lower_bound: Optional[float] = Field(None, description="예측 하한")
    upper_bound: Optional[float] = Field(None, description="예측 상한")


class WatchlistAlert(BaseModel):
    """워치리스트 알림"""
    segment_id: str
    segment_info: Dict[str, str]
    alert_type: str = Field(..., description="알림 유형: RISK_DOWN, RISK_UP, OPPORTUNITY")
    severity: str = Field(..., description="심각도: HIGH, MEDIUM, LOW")
    kpi: str = Field(..., description="관련 KPI")
    residual_zscore: float = Field(..., description="잔차 Z-스코어")
    message: str = Field(..., description="알림 메시지")
    drivers: List[str] = Field(default_factory=list, description="주요 드라이버")


class ActionRecommendation(BaseModel):
    """액션 추천"""
    segment_id: str
    action_type: str = Field(..., description="액션 유형")
    priority: int = Field(..., description="우선순위 (1-10)")
    expected_impact: str = Field(..., description="예상 효과")
    description: str = Field(..., description="액션 설명")
    target_kpi: str = Field(..., description="대상 KPI")


class OverviewStats(BaseModel):
    """대시보드 개요 통계"""
    total_segments: int
    total_customers: int
    total_deposit: float
    total_loan: float
    total_card_usage: float
    total_digital_amount: float
    month: str
    mom_deposit_growth: Optional[float] = None
    mom_loan_growth: Optional[float] = None


class SegmentListResponse(BaseModel):
    """세그먼트 목록 응답"""
    segments: List[SegmentKPI]
    total_count: int
    page: int
    page_size: int


class ForecastListResponse(BaseModel):
    """예측 목록 응답"""
    forecasts: List[ForecastResult]
    total_count: int


class WatchlistResponse(BaseModel):
    """워치리스트 응답"""
    alerts: List[WatchlistAlert]
    total_count: int


class ActionListResponse(BaseModel):
    """액션 추천 목록 응답"""
    actions: List[ActionRecommendation]
    total_count: int


# ============== Data Loading ==============

def get_latest_mvp_dir(outputs_dir: Path) -> Optional[Path]:
    """
    Find the latest mvp_* directory if it exists.
    Returns None if no mvp_* directory exists.
    """
    mvp_dirs = sorted(outputs_dir.glob("mvp_*"), reverse=True)
    if mvp_dirs:
        return mvp_dirs[0]
    return None


def find_file(outputs_dir: Path, *relative_paths: str) -> Optional[Path]:
    """
    Find a file in outputs directory.
    First checks directly under outputs_dir, then in mvp_* subdirectories.
    """
    # Check direct path first
    for rel_path in relative_paths:
        direct_path = outputs_dir / rel_path
        if direct_path.exists():
            return direct_path
    
    # Check in mvp_* directories
    mvp_dir = get_latest_mvp_dir(outputs_dir)
    if mvp_dir:
        for rel_path in relative_paths:
            mvp_path = mvp_dir / rel_path
            if mvp_path.exists():
                return mvp_path
    
    return None


def transform_wide_forecasts_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform wide format forecasts to long format expected by API.
    Wide format: segment_id, pred_month, 예금총잔액_x, 예금총잔액_y, 예금총잔액, ...
    Long format: segment_id, kpi, horizon, forecast_month, predicted
    """
    kpi_list = ['예금총잔액', '대출총잔액', '카드총사용', '디지털거래금액', '순유입', 'FX총액']
    horizon_suffixes = {'_x': 1, '_y': 2, '': 3}
    
    records = []
    for _, row in df.iterrows():
        segment_id = row.get('segment_id', '')
        forecast_month = row.get('pred_month', '')
        
        for kpi in kpi_list:
            for suffix, horizon in horizon_suffixes.items():
                col_name = f"{kpi}{suffix}" if suffix else kpi
                if col_name in row and pd.notna(row[col_name]):
                    records.append({
                        'segment_id': segment_id,
                        'kpi': kpi,
                        'horizon': horizon,
                        'forecast_month': forecast_month,
                        'predicted': float(row[col_name]),
                        'actual': None,
                        'lower': None,
                        'upper': None,
                    })
    
    return pd.DataFrame(records)


def load_outputs() -> Dict[str, pd.DataFrame]:
    """Load output files from outputs/ directory or mvp_* subdirectories"""
    outputs_dir = REPO_ROOT / "outputs"
    
    data = {}
    
    # Load forecast metrics
    path = find_file(outputs_dir, "forecast_metrics.csv")
    if path:
        data["forecast_metrics"] = pd.read_csv(path)
    
    # Load watchlist alerts
    path = find_file(outputs_dir, "watchlist_alerts.csv")
    if path:
        data["watchlist_alerts"] = pd.read_csv(path)
    
    # Load watchlist drivers
    path = find_file(outputs_dir, "watchlist_drivers.csv")
    if path:
        data["watchlist_drivers"] = pd.read_csv(path)
    
    # Load actions
    path = find_file(outputs_dir, "actions_top.csv")
    if path:
        data["actions"] = pd.read_csv(path)
    
    # Load segment panel (support both parquet and csv)
    panel_path = find_file(outputs_dir, "segment_panel.parquet")
    if panel_path:
        try:
            data["panel"] = pd.read_parquet(panel_path)
        except Exception:
            pass
    
    if "panel" not in data:
        panel_path = find_file(outputs_dir, "segment_panel.csv")
        if panel_path:
            data["panel"] = pd.read_csv(panel_path)
    
    # Load segment forecasts - check multiple possible locations
    forecasts_path = find_file(
        outputs_dir, 
        "forecasts/segment_forecasts_next.csv",
        "segment_forecasts_next.csv"
    )
    if forecasts_path:
        raw_forecasts = pd.read_csv(forecasts_path)
        # Check if the data is in wide format (contains pred_month and KPI_x columns)
        if 'pred_month' in raw_forecasts.columns:
            data["segment_forecasts"] = transform_wide_forecasts_to_long(raw_forecasts)
        else:
            # Already in expected format
            data["segment_forecasts"] = raw_forecasts
    
    return data


# Cache data
_cached_data: Dict[str, pd.DataFrame] = {}


def get_data() -> Dict[str, pd.DataFrame]:
    """Get cached data or load"""
    global _cached_data
    if not _cached_data:
        _cached_data = load_outputs()
    return _cached_data


def refresh_data():
    """Refresh cached data"""
    global _cached_data
    _cached_data = load_outputs()


# ============== API Endpoints ==============

@app.get("/", tags=["Root"])
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "iM ONEderful Segment Radar API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/overview", response_model=OverviewStats, tags=["Dashboard"])
async def get_overview(month: Optional[str] = Query(None, description="조회 월 (YYYY-MM)")):
    """
    대시보드 개요 통계 조회
    
    - 전체 세그먼트 수
    - 전체 고객 수
    - 주요 KPI 합계
    - 전월 대비 성장률
    """
    data = get_data()
    
    if "panel" not in data:
        raise HTTPException(status_code=404, detail="Panel data not found. Run MVP pipeline first.")
    
    panel = data["panel"]
    
    # Get latest month if not specified
    if month is None:
        if "month" in panel.columns:
            month = panel["month"].max()
        else:
            month = "2024-12"
    
    # Filter by month
    if "month" in panel.columns:
        month_data = panel[panel["month"] == month]
    else:
        month_data = panel
    
    if month_data.empty:
        raise HTTPException(status_code=404, detail=f"No data for month: {month}")
    
    # Calculate statistics
    total_segments = month_data["segment_id"].nunique() if "segment_id" in month_data.columns else len(month_data)
    total_customers = int(month_data["customer_count"].sum()) if "customer_count" in month_data.columns else 0
    
    total_deposit = float(month_data["예금총잔액"].sum()) if "예금총잔액" in month_data.columns else 0.0
    total_loan = float(month_data["대출총잔액"].sum()) if "대출총잔액" in month_data.columns else 0.0
    total_card = float(month_data["카드총사용"].sum()) if "카드총사용" in month_data.columns else 0.0
    total_digital = float(month_data["디지털거래금액"].sum()) if "디지털거래금액" in month_data.columns else 0.0
    
    return OverviewStats(
        total_segments=total_segments,
        total_customers=total_customers,
        total_deposit=total_deposit,
        total_loan=total_loan,
        total_card_usage=total_card,
        total_digital_amount=total_digital,
        month=str(month),
        mom_deposit_growth=None,
        mom_loan_growth=None,
    )


@app.get("/api/segments", response_model=SegmentListResponse, tags=["Segments"])
async def get_segments(
    page: int = Query(1, ge=1, description="페이지 번호"),
    page_size: int = Query(20, ge=1, le=100, description="페이지 크기"),
    month: Optional[str] = Query(None, description="조회 월 (YYYY-MM)"),
    업종_중분류: Optional[str] = Query(None, description="업종 중분류 필터"),
    사업장_시도: Optional[str] = Query(None, description="사업장 시도 필터"),
    법인_고객등급: Optional[str] = Query(None, description="법인 고객등급 필터"),
    전담고객여부: Optional[str] = Query(None, description="전담고객 여부 필터"),
    sort_by: Optional[str] = Query("예금총잔액", description="정렬 기준"),
    sort_order: str = Query("desc", description="정렬 순서 (asc/desc)"),
):
    """
    세그먼트 목록 조회
    
    - 페이지네이션 지원
    - 필터링 지원 (업종, 지역, 등급, 전담여부)
    - 정렬 지원
    """
    data = get_data()
    
    if "panel" not in data:
        raise HTTPException(status_code=404, detail="Panel data not found")
    
    panel = data["panel"].copy()
    
    # Filter by month
    if month and "month" in panel.columns:
        panel = panel[panel["month"] == month]
    elif "month" in panel.columns:
        # Use latest month
        panel = panel[panel["month"] == panel["month"].max()]
    
    # Apply filters
    if 업종_중분류 and "업종_중분류" in panel.columns:
        panel = panel[panel["업종_중분류"] == 업종_중분류]
    if 사업장_시도 and "사업장_시도" in panel.columns:
        panel = panel[panel["사업장_시도"] == 사업장_시도]
    if 법인_고객등급 and "법인_고객등급" in panel.columns:
        panel = panel[panel["법인_고객등급"] == 법인_고객등급]
    if 전담고객여부 and "전담고객여부" in panel.columns:
        panel = panel[panel["전담고객여부"] == 전담고객여부]
    
    # Sort
    if sort_by and sort_by in panel.columns:
        ascending = sort_order.lower() == "asc"
        panel = panel.sort_values(sort_by, ascending=ascending)
    
    total_count = len(panel)
    
    # Paginate
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    panel_page = panel.iloc[start_idx:end_idx]
    
    # Convert to response
    segments = []
    for _, row in panel_page.iterrows():
        seg = SegmentKPI(
            segment_id=str(row.get("segment_id", "")),
            업종_중분류=str(row.get("업종_중분류", "")),
            사업장_시도=str(row.get("사업장_시도", "")),
            법인_고객등급=str(row.get("법인_고객등급", "")),
            전담고객여부=str(row.get("전담고객여부", "")),
            customer_count=int(row.get("customer_count", 0)),
            month=str(row.get("month", "")),
            예금총잔액=float(row.get("예금총잔액", 0)),
            대출총잔액=float(row.get("대출총잔액", 0)),
            카드총사용=float(row.get("카드총사용", 0)),
            디지털거래금액=float(row.get("디지털거래금액", 0)),
            순유입=float(row.get("순유입", 0)),
            한도소진율=float(row.get("한도소진율", 0)) if "한도소진율" in row else None,
            디지털비중=float(row.get("디지털비중", 0)) if "디지털비중" in row else None,
        )
        segments.append(seg)
    
    return SegmentListResponse(
        segments=segments,
        total_count=total_count,
        page=page,
        page_size=page_size,
    )


@app.get("/api/segments/{segment_id}", response_model=SegmentKPI, tags=["Segments"])
async def get_segment_detail(segment_id: str, month: Optional[str] = None):
    """
    세그먼트 상세 조회
    """
    data = get_data()
    
    if "panel" not in data:
        raise HTTPException(status_code=404, detail="Panel data not found")
    
    panel = data["panel"]
    
    # Filter by segment_id
    seg_data = panel[panel["segment_id"] == segment_id]
    
    if seg_data.empty:
        raise HTTPException(status_code=404, detail=f"Segment not found: {segment_id}")
    
    # Get specific month or latest
    if month and "month" in seg_data.columns:
        seg_data = seg_data[seg_data["month"] == month]
    elif "month" in seg_data.columns:
        seg_data = seg_data[seg_data["month"] == seg_data["month"].max()]
    
    if seg_data.empty:
        raise HTTPException(status_code=404, detail=f"No data for segment {segment_id} in month {month}")
    
    row = seg_data.iloc[0]
    
    return SegmentKPI(
        segment_id=str(row.get("segment_id", "")),
        업종_중분류=str(row.get("업종_중분류", "")),
        사업장_시도=str(row.get("사업장_시도", "")),
        법인_고객등급=str(row.get("법인_고객등급", "")),
        전담고객여부=str(row.get("전담고객여부", "")),
        customer_count=int(row.get("customer_count", 0)),
        month=str(row.get("month", "")),
        예금총잔액=float(row.get("예금총잔액", 0)),
        대출총잔액=float(row.get("대출총잔액", 0)),
        카드총사용=float(row.get("카드총사용", 0)),
        디지털거래금액=float(row.get("디지털거래금액", 0)),
        순유입=float(row.get("순유입", 0)),
        한도소진율=float(row.get("한도소진율", 0)) if "한도소진율" in row else None,
        디지털비중=float(row.get("디지털비중", 0)) if "디지털비중" in row else None,
    )


@app.get("/api/segments/{segment_id}/history", tags=["Segments"])
async def get_segment_history(
    segment_id: str,
    start_month: Optional[str] = None,
    end_month: Optional[str] = None,
):
    """
    세그먼트 KPI 히스토리 조회
    """
    data = get_data()
    
    if "panel" not in data:
        raise HTTPException(status_code=404, detail="Panel data not found")
    
    panel = data["panel"]
    
    seg_data = panel[panel["segment_id"] == segment_id].copy()
    
    if seg_data.empty:
        raise HTTPException(status_code=404, detail=f"Segment not found: {segment_id}")
    
    # Filter by date range
    if "month" in seg_data.columns:
        seg_data = seg_data.sort_values("month")
        if start_month:
            seg_data = seg_data[seg_data["month"] >= start_month]
        if end_month:
            seg_data = seg_data[seg_data["month"] <= end_month]
    
    # Return history
    history = seg_data.to_dict(orient="records")
    
    return {
        "segment_id": segment_id,
        "history": history,
        "total_months": len(history),
    }


@app.get("/api/forecasts", response_model=ForecastListResponse, tags=["Forecasts"])
async def get_forecasts(
    segment_id: Optional[str] = None,
    kpi: Optional[str] = Query(None, description="KPI 필터 (예금총잔액, 대출총잔액 등)"),
    horizon: Optional[int] = Query(None, ge=1, le=3, description="예측 기간 (1-3개월)"),
):
    """
    세그먼트 예측 결과 조회
    """
    data = get_data()
    
    if "segment_forecasts" not in data:
        raise HTTPException(status_code=404, detail="Forecast data not found. Run MVP pipeline first.")
    
    forecasts_df = data["segment_forecasts"].copy()
    
    # Apply filters
    if segment_id and "segment_id" in forecasts_df.columns:
        forecasts_df = forecasts_df[forecasts_df["segment_id"] == segment_id]
    if kpi and "kpi" in forecasts_df.columns:
        forecasts_df = forecasts_df[forecasts_df["kpi"] == kpi]
    if horizon and "horizon" in forecasts_df.columns:
        forecasts_df = forecasts_df[forecasts_df["horizon"] == horizon]
    
    # Convert to response
    forecasts = []
    for _, row in forecasts_df.iterrows():
        fc = ForecastResult(
            segment_id=str(row.get("segment_id", "")),
            target_kpi=str(row.get("kpi", "")),
            horizon=int(row.get("horizon", 1)),
            forecast_month=str(row.get("forecast_month", "")),
            predicted_value=float(row.get("predicted", 0)),
            actual_value=float(row.get("actual")) if pd.notna(row.get("actual")) else None,
            lower_bound=float(row.get("lower")) if pd.notna(row.get("lower")) else None,
            upper_bound=float(row.get("upper")) if pd.notna(row.get("upper")) else None,
        )
        forecasts.append(fc)
    
    return ForecastListResponse(
        forecasts=forecasts,
        total_count=len(forecasts),
    )


@app.get("/api/watchlist", response_model=WatchlistResponse, tags=["Watchlist"])
async def get_watchlist(
    alert_type: Optional[str] = Query(None, description="알림 유형 필터"),
    severity: Optional[str] = Query(None, description="심각도 필터"),
    limit: int = Query(50, ge=1, le=200, description="조회 개수"),
):
    """
    워치리스트 알림 조회
    
    - RISK_DOWN: 급감 리스크
    - RISK_UP: 급증 (비용 증가)
    - OPPORTUNITY: 기회
    """
    data = get_data()
    
    if "watchlist_alerts" not in data:
        raise HTTPException(status_code=404, detail="Watchlist data not found. Run MVP pipeline first.")
    
    alerts_df = data["watchlist_alerts"].copy()
    drivers_df = data.get("watchlist_drivers", pd.DataFrame())
    
    # Apply filters
    if alert_type and "alert_type" in alerts_df.columns:
        alerts_df = alerts_df[alerts_df["alert_type"] == alert_type]
    if severity and "severity" in alerts_df.columns:
        alerts_df = alerts_df[alerts_df["severity"] == severity]
    
    # Sort by severity and limit
    if "residual_zscore" in alerts_df.columns:
        alerts_df = alerts_df.sort_values("residual_zscore", key=abs, ascending=False)
    
    alerts_df = alerts_df.head(limit)
    
    # Convert to response
    alerts = []
    for _, row in alerts_df.iterrows():
        seg_id = str(row.get("segment_id", ""))
        
        # Get drivers
        drivers = []
        if not drivers_df.empty and "segment_id" in drivers_df.columns:
            seg_drivers = drivers_df[drivers_df["segment_id"] == seg_id]
            if not seg_drivers.empty:
                drivers = seg_drivers["driver"].tolist() if "driver" in seg_drivers.columns else []
        
        alert = WatchlistAlert(
            segment_id=seg_id,
            segment_info={
                "업종_중분류": str(row.get("업종_중분류", "")),
                "사업장_시도": str(row.get("사업장_시도", "")),
                "법인_고객등급": str(row.get("법인_고객등급", "")),
                "전담고객여부": str(row.get("전담고객여부", "")),
            },
            alert_type=str(row.get("alert_type", "RISK_DOWN")),
            severity=str(row.get("severity", "MEDIUM")),
            kpi=str(row.get("kpi", "")),
            residual_zscore=float(row.get("residual_zscore", 0)),
            message=str(row.get("message", "")),
            drivers=drivers[:5],  # Top 5 drivers
        )
        alerts.append(alert)
    
    return WatchlistResponse(
        alerts=alerts,
        total_count=len(alerts),
    )


@app.get("/api/recommendations", response_model=ActionListResponse, tags=["Recommendations"])
async def get_recommendations(
    segment_id: Optional[str] = None,
    action_type: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100, description="조회 개수"),
):
    """
    액션 추천 조회
    """
    data = get_data()
    
    if "actions" not in data:
        raise HTTPException(status_code=404, detail="Actions data not found. Run MVP pipeline first.")
    
    actions_df = data["actions"].copy()
    
    # Apply filters
    if segment_id and "segment_id" in actions_df.columns:
        actions_df = actions_df[actions_df["segment_id"] == segment_id]
    if action_type and "action_type" in actions_df.columns:
        actions_df = actions_df[actions_df["action_type"] == action_type]
    
    # Sort by priority
    if "priority" in actions_df.columns:
        actions_df = actions_df.sort_values("priority", ascending=False)
    
    actions_df = actions_df.head(limit)
    
    # Convert to response
    actions = []
    for _, row in actions_df.iterrows():
        action = ActionRecommendation(
            segment_id=str(row.get("segment_id", "")),
            action_type=str(row.get("action_type", "")),
            priority=int(row.get("priority", 0)),
            expected_impact=str(row.get("expected_impact", "")),
            description=str(row.get("description", "")),
            target_kpi=str(row.get("target_kpi", "")),
        )
        actions.append(action)
    
    return ActionListResponse(
        actions=actions,
        total_count=len(actions),
    )


@app.get("/api/filters", tags=["Filters"])
async def get_filter_options():
    """
    필터 옵션 조회 (드롭다운용)
    """
    data = get_data()
    
    if "panel" not in data:
        return {
            "업종_중분류": [],
            "사업장_시도": [],
            "법인_고객등급": [],
            "전담고객여부": [],
            "months": [],
        }
    
    panel = data["panel"]
    
    options = {
        "업종_중분류": sorted(panel["업종_중분류"].dropna().unique().tolist()) if "업종_중분류" in panel.columns else [],
        "사업장_시도": sorted(panel["사업장_시도"].dropna().unique().tolist()) if "사업장_시도" in panel.columns else [],
        "법인_고객등급": sorted(panel["법인_고객등급"].dropna().unique().tolist()) if "법인_고객등급" in panel.columns else [],
        "전담고객여부": sorted(panel["전담고객여부"].dropna().unique().tolist()) if "전담고객여부" in panel.columns else [],
        "months": sorted(panel["month"].dropna().unique().tolist()) if "month" in panel.columns else [],
    }
    
    return options


@app.get("/api/kpi-trends", tags=["Dashboard"])
async def get_kpi_trends(
    kpi: str = Query("예금총잔액", description="KPI 이름"),
    group_by: Optional[str] = Query(None, description="그룹 기준 (업종_중분류, 사업장_시도 등)"),
    months: int = Query(12, ge=1, le=36, description="조회 개월 수"),
):
    """
    KPI 트렌드 조회 (차트용)
    """
    data = get_data()
    
    if "panel" not in data:
        raise HTTPException(status_code=404, detail="Panel data not found")
    
    panel = data["panel"].copy()
    
    if kpi not in panel.columns:
        raise HTTPException(status_code=400, detail=f"KPI not found: {kpi}")
    
    # Sort by month and get latest N months
    if "month" in panel.columns:
        panel = panel.sort_values("month")
        unique_months = panel["month"].unique()
        if len(unique_months) > months:
            start_month = unique_months[-months]
            panel = panel[panel["month"] >= start_month]
    
    # Aggregate
    if group_by and group_by in panel.columns:
        trend = panel.groupby(["month", group_by])[kpi].sum().reset_index()
        trend_dict = trend.to_dict(orient="records")
    else:
        trend = panel.groupby("month")[kpi].sum().reset_index()
        trend_dict = trend.to_dict(orient="records")
    
    return {
        "kpi": kpi,
        "group_by": group_by,
        "data": trend_dict,
    }


@app.post("/api/refresh", tags=["Admin"])
async def refresh_data_cache():
    """
    데이터 캐시 갱신
    """
    refresh_data()
    return {"message": "Data cache refreshed"}


# ============== Health Check ==============

@app.get("/health", tags=["Health"])
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "service": "iM ONEderful Segment Radar API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
