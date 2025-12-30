"""
API Views for iM ONEderful Segment Radar.
Django REST Framework 기반 API 뷰
"""
from __future__ import annotations

import pandas as pd
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from .data_loader import data_loader
from .serializers import (
    OverviewStatsSerializer,
    SegmentKPISerializer,
    SegmentListResponseSerializer,
    ForecastResultSerializer,
    ForecastListResponseSerializer,
    WatchlistAlertSerializer,
    WatchlistResponseSerializer,
    ActionRecommendationSerializer,
    ActionListResponseSerializer,
    FilterOptionsSerializer,
    KPITrendSerializer,
    SegmentHistorySerializer,
)


class RootView(APIView):
    """API 루트 엔드포인트"""
    
    def get(self, request):
        return Response({
            "message": "iM ONEderful Segment Radar API",
            "version": "1.0.0",
            "framework": "Django REST Framework",
            "docs": "/api/docs/",
        })


class HealthCheckView(APIView):
    """헬스 체크"""
    
    def get(self, request):
        return Response({
            "status": "healthy",
            "service": "iM ONEderful Segment Radar API (Django)"
        })


class OverviewView(APIView):
    """대시보드 개요 통계 조회"""
    
    def get(self, request):
        month = request.query_params.get('month')
        
        panel = data_loader.get_panel()
        if panel is None:
            return Response(
                {"error": "Panel data not found. Run MVP pipeline first."},
                status=status.HTTP_404_NOT_FOUND
            )
        
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
            return Response(
                {"error": f"No data for month: {month}"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Calculate statistics
        total_segments = month_data["segment_id"].nunique() if "segment_id" in month_data.columns else len(month_data)
        total_customers = int(month_data["customer_count"].sum()) if "customer_count" in month_data.columns else 0
        
        total_deposit = float(month_data["예금총잔액"].sum()) if "예금총잔액" in month_data.columns else 0.0
        total_loan = float(month_data["대출총잔액"].sum()) if "대출총잔액" in month_data.columns else 0.0
        total_card = float(month_data["카드총사용"].sum()) if "카드총사용" in month_data.columns else 0.0
        total_digital = float(month_data["디지털거래금액"].sum()) if "디지털거래금액" in month_data.columns else 0.0
        
        data = {
            "total_segments": total_segments,
            "total_customers": total_customers,
            "total_deposit": total_deposit,
            "total_loan": total_loan,
            "total_card_usage": total_card,
            "total_digital_amount": total_digital,
            "month": str(month),
            "mom_deposit_growth": None,
            "mom_loan_growth": None,
        }
        
        serializer = OverviewStatsSerializer(data)
        return Response(serializer.data)


class SegmentListView(APIView):
    """세그먼트 목록 조회"""
    
    def get(self, request):
        # Query parameters
        page = int(request.query_params.get('page', 1))
        page_size = int(request.query_params.get('page_size', 20))
        month = request.query_params.get('month')
        업종_중분류 = request.query_params.get('업종_중분류')
        사업장_시도 = request.query_params.get('사업장_시도')
        법인_고객등급 = request.query_params.get('법인_고객등급')
        전담고객여부 = request.query_params.get('전담고객여부')
        sort_by = request.query_params.get('sort_by', '예금총잔액')
        sort_order = request.query_params.get('sort_order', 'desc')
        
        panel = data_loader.get_panel()
        if panel is None:
            return Response(
                {"error": "Panel data not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        panel = panel.copy()
        
        # Filter by month
        if month and "month" in panel.columns:
            panel = panel[panel["month"] == month]
        elif "month" in panel.columns:
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
            seg = {
                "segment_id": str(row.get("segment_id", "")),
                "업종_중분류": str(row.get("업종_중분류", "")),
                "사업장_시도": str(row.get("사업장_시도", "")),
                "법인_고객등급": str(row.get("법인_고객등급", "")),
                "전담고객여부": str(row.get("전담고객여부", "")),
                "customer_count": int(row.get("customer_count", 0)),
                "month": str(row.get("month", "")),
                "예금총잔액": float(row.get("예금총잔액", 0)),
                "대출총잔액": float(row.get("대출총잔액", 0)),
                "카드총사용": float(row.get("카드총사용", 0)),
                "디지털거래금액": float(row.get("디지털거래금액", 0)),
                "순유입": float(row.get("순유입", 0)),
                "한도소진율": float(row.get("한도소진율", 0)) if "한도소진율" in row else None,
                "디지털비중": float(row.get("디지털비중", 0)) if "디지털비중" in row else None,
            }
            segments.append(seg)
        
        data = {
            "segments": segments,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
        }
        
        serializer = SegmentListResponseSerializer(data)
        return Response(serializer.data)


class SegmentDetailView(APIView):
    """세그먼트 상세 조회"""
    
    def get(self, request, segment_id):
        month = request.query_params.get('month')
        
        panel = data_loader.get_panel()
        if panel is None:
            return Response(
                {"error": "Panel data not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Filter by segment_id
        seg_data = panel[panel["segment_id"] == segment_id]
        
        if seg_data.empty:
            return Response(
                {"error": f"Segment not found: {segment_id}"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Get specific month or latest
        if month and "month" in seg_data.columns:
            seg_data = seg_data[seg_data["month"] == month]
        elif "month" in seg_data.columns:
            seg_data = seg_data[seg_data["month"] == seg_data["month"].max()]
        
        if seg_data.empty:
            return Response(
                {"error": f"No data for segment {segment_id} in month {month}"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        row = seg_data.iloc[0]
        
        data = {
            "segment_id": str(row.get("segment_id", "")),
            "업종_중분류": str(row.get("업종_중분류", "")),
            "사업장_시도": str(row.get("사업장_시도", "")),
            "법인_고객등급": str(row.get("법인_고객등급", "")),
            "전담고객여부": str(row.get("전담고객여부", "")),
            "customer_count": int(row.get("customer_count", 0)),
            "month": str(row.get("month", "")),
            "예금총잔액": float(row.get("예금총잔액", 0)),
            "대출총잔액": float(row.get("대출총잔액", 0)),
            "카드총사용": float(row.get("카드총사용", 0)),
            "디지털거래금액": float(row.get("디지털거래금액", 0)),
            "순유입": float(row.get("순유입", 0)),
            "한도소진율": float(row.get("한도소진율", 0)) if "한도소진율" in row else None,
            "디지털비중": float(row.get("디지털비중", 0)) if "디지털비중" in row else None,
        }
        
        serializer = SegmentKPISerializer(data)
        return Response(serializer.data)


class SegmentHistoryView(APIView):
    """세그먼트 KPI 히스토리 조회"""
    
    def get(self, request, segment_id):
        start_month = request.query_params.get('start_month')
        end_month = request.query_params.get('end_month')
        
        panel = data_loader.get_panel()
        if panel is None:
            return Response(
                {"error": "Panel data not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        seg_data = panel[panel["segment_id"] == segment_id].copy()
        
        if seg_data.empty:
            return Response(
                {"error": f"Segment not found: {segment_id}"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Filter by date range
        if "month" in seg_data.columns:
            seg_data = seg_data.sort_values("month")
            if start_month:
                seg_data = seg_data[seg_data["month"] >= start_month]
            if end_month:
                seg_data = seg_data[seg_data["month"] <= end_month]
        
        history = seg_data.to_dict(orient="records")
        
        data = {
            "segment_id": segment_id,
            "history": history,
            "total_months": len(history),
        }
        
        serializer = SegmentHistorySerializer(data)
        return Response(serializer.data)


class ForecastListView(APIView):
    """세그먼트 예측 결과 조회"""
    
    def get(self, request):
        segment_id = request.query_params.get('segment_id')
        kpi = request.query_params.get('kpi')
        horizon = request.query_params.get('horizon')
        
        forecasts_df = data_loader.get_forecasts()
        if forecasts_df is None:
            return Response(
                {"error": "Forecast data not found. Run MVP pipeline first."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        forecasts_df = forecasts_df.copy()
        
        # Apply filters
        if segment_id and "segment_id" in forecasts_df.columns:
            forecasts_df = forecasts_df[forecasts_df["segment_id"] == segment_id]
        if kpi and "kpi" in forecasts_df.columns:
            forecasts_df = forecasts_df[forecasts_df["kpi"] == kpi]
        if horizon and "horizon" in forecasts_df.columns:
            forecasts_df = forecasts_df[forecasts_df["horizon"] == int(horizon)]
        
        # Convert to response
        forecasts = []
        for _, row in forecasts_df.iterrows():
            fc = {
                "segment_id": str(row.get("segment_id", "")),
                "target_kpi": str(row.get("kpi", "")),
                "horizon": int(row.get("horizon", 1)),
                "forecast_month": str(row.get("forecast_month", "")),
                "predicted_value": float(row.get("predicted", 0)),
                "actual_value": float(row.get("actual")) if pd.notna(row.get("actual")) else None,
                "lower_bound": float(row.get("lower")) if pd.notna(row.get("lower")) else None,
                "upper_bound": float(row.get("upper")) if pd.notna(row.get("upper")) else None,
            }
            forecasts.append(fc)
        
        data = {
            "forecasts": forecasts,
            "total_count": len(forecasts),
        }
        
        serializer = ForecastListResponseSerializer(data)
        return Response(serializer.data)


class WatchlistView(APIView):
    """워치리스트 알림 조회"""
    
    def get(self, request):
        alert_type = request.query_params.get('alert_type')
        severity = request.query_params.get('severity')
        limit = int(request.query_params.get('limit', 50))
        
        alerts_df = data_loader.get_watchlist()
        if alerts_df is None:
            return Response(
                {"error": "Watchlist data not found. Run MVP pipeline first."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        alerts_df = alerts_df.copy()
        drivers_df = data_loader.get_drivers()
        if drivers_df is None:
            drivers_df = pd.DataFrame()
        
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
            
            alert = {
                "segment_id": seg_id,
                "segment_info": {
                    "업종_중분류": str(row.get("업종_중분류", "")),
                    "사업장_시도": str(row.get("사업장_시도", "")),
                    "법인_고객등급": str(row.get("법인_고객등급", "")),
                    "전담고객여부": str(row.get("전담고객여부", "")),
                },
                "alert_type": str(row.get("alert_type", "RISK_DOWN")),
                "severity": str(row.get("severity", "MEDIUM")),
                "kpi": str(row.get("kpi", "")),
                "residual_zscore": float(row.get("residual_zscore", 0)) if pd.notna(row.get("residual_zscore")) else 0.0,
                "message": str(row.get("message", "")),
                "drivers": drivers[:5],
            }
            alerts.append(alert)
        
        data = {
            "alerts": alerts,
            "total_count": len(alerts),
        }
        
        serializer = WatchlistResponseSerializer(data)
        return Response(serializer.data)


class RecommendationsView(APIView):
    """액션 추천 조회"""
    
    def get(self, request):
        segment_id = request.query_params.get('segment_id')
        action_type = request.query_params.get('action_type')
        limit = int(request.query_params.get('limit', 20))
        
        actions_df = data_loader.get_actions()
        if actions_df is None:
            return Response(
                {"error": "Actions data not found. Run MVP pipeline first."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        actions_df = actions_df.copy()
        
        # Apply filters
        if segment_id and "segment_id" in actions_df.columns:
            actions_df = actions_df[actions_df["segment_id"] == segment_id]
        if action_type and "action_type" in actions_df.columns:
            actions_df = actions_df[actions_df["action_type"] == action_type]
        
        # Sort by priority (score column if exists)
        if "score" in actions_df.columns:
            actions_df = actions_df.sort_values("score", ascending=False)
        
        actions_df = actions_df.head(limit)
        
        # Convert to response
        actions = []
        for _, row in actions_df.iterrows():
            action = {
                "segment_id": str(row.get("segment_id", "")),
                "action_type": str(row.get("action_type", "")),
                "priority": int(row.get("score", 0)) if "score" in row else int(row.get("priority", 0)),
                "expected_impact": str(row.get("rationale", "")),
                "description": str(row.get("title", "")),
                "target_kpi": str(row.get("target_kpi", "")),
            }
            actions.append(action)
        
        data = {
            "actions": actions,
            "total_count": len(actions),
        }
        
        serializer = ActionListResponseSerializer(data)
        return Response(serializer.data)


class FilterOptionsView(APIView):
    """필터 옵션 조회 (드롭다운용)"""
    
    def get(self, request):
        panel = data_loader.get_panel()
        
        if panel is None:
            data = {
                "업종_중분류": [],
                "사업장_시도": [],
                "법인_고객등급": [],
                "전담고객여부": [],
                "months": [],
            }
        else:
            data = {
                "업종_중분류": sorted(panel["업종_중분류"].dropna().unique().tolist()) if "업종_중분류" in panel.columns else [],
                "사업장_시도": sorted(panel["사업장_시도"].dropna().unique().tolist()) if "사업장_시도" in panel.columns else [],
                "법인_고객등급": sorted(panel["법인_고객등급"].dropna().unique().tolist()) if "법인_고객등급" in panel.columns else [],
                "전담고객여부": sorted(panel["전담고객여부"].dropna().unique().tolist()) if "전담고객여부" in panel.columns else [],
                "months": sorted([str(m) for m in panel["month"].dropna().unique().tolist()]) if "month" in panel.columns else [],
            }
        
        serializer = FilterOptionsSerializer(data)
        return Response(serializer.data)


class KPITrendsView(APIView):
    """KPI 트렌드 조회 (차트용)"""
    
    def get(self, request):
        kpi = request.query_params.get('kpi', '예금총잔액')
        group_by = request.query_params.get('group_by')
        months = int(request.query_params.get('months', 12))
        
        panel = data_loader.get_panel()
        if panel is None:
            return Response(
                {"error": "Panel data not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        panel = panel.copy()
        
        if kpi not in panel.columns:
            return Response(
                {"error": f"KPI not found: {kpi}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
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
        
        data = {
            "kpi": kpi,
            "group_by": group_by,
            "data": trend_dict,
        }
        
        serializer = KPITrendSerializer(data)
        return Response(serializer.data)


class RefreshDataView(APIView):
    """데이터 캐시 갱신"""
    
    def post(self, request):
        data_loader.refresh()
        return Response({"message": "Data cache refreshed"})
