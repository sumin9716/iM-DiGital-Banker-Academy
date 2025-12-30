"""
Serializers for the Segment Radar API.
"""
from rest_framework import serializers


class SegmentBaseSerializer(serializers.Serializer):
    """세그먼트 기본 정보"""
    segment_id = serializers.CharField(help_text="세그먼트 고유 ID")
    업종_중분류 = serializers.CharField(help_text="업종 중분류")
    사업장_시도 = serializers.CharField(help_text="사업장 시도")
    법인_고객등급 = serializers.CharField(help_text="법인 고객등급")
    전담고객여부 = serializers.CharField(help_text="전담고객 여부")
    customer_count = serializers.IntegerField(help_text="고객 수")


class SegmentKPISerializer(SegmentBaseSerializer):
    """세그먼트 KPI 정보"""
    month = serializers.CharField(help_text="기준 월 (YYYY-MM)")
    예금총잔액 = serializers.FloatField(help_text="예금 총잔액")
    대출총잔액 = serializers.FloatField(help_text="대출 총잔액")
    카드총사용 = serializers.FloatField(help_text="카드 총사용액")
    디지털거래금액 = serializers.FloatField(help_text="디지털 거래금액")
    순유입 = serializers.FloatField(help_text="순유입")
    한도소진율 = serializers.FloatField(required=False, allow_null=True, help_text="한도 소진율")
    디지털비중 = serializers.FloatField(required=False, allow_null=True, help_text="디지털 거래 비중")


class ForecastResultSerializer(serializers.Serializer):
    """예측 결과"""
    segment_id = serializers.CharField()
    target_kpi = serializers.CharField(help_text="예측 대상 KPI")
    horizon = serializers.IntegerField(help_text="예측 기간 (개월)")
    forecast_month = serializers.CharField(help_text="예측 대상 월")
    predicted_value = serializers.FloatField(help_text="예측값")
    actual_value = serializers.FloatField(required=False, allow_null=True, help_text="실제값")
    lower_bound = serializers.FloatField(required=False, allow_null=True, help_text="예측 하한")
    upper_bound = serializers.FloatField(required=False, allow_null=True, help_text="예측 상한")


class WatchlistAlertSerializer(serializers.Serializer):
    """워치리스트 알림"""
    segment_id = serializers.CharField()
    segment_info = serializers.DictField(child=serializers.CharField())
    alert_type = serializers.CharField(help_text="알림 유형: RISK_DOWN, RISK_UP, OPPORTUNITY")
    severity = serializers.CharField(help_text="심각도: HIGH, MEDIUM, LOW")
    kpi = serializers.CharField(help_text="관련 KPI")
    residual_zscore = serializers.FloatField(help_text="잔차 Z-스코어")
    message = serializers.CharField(help_text="알림 메시지")
    drivers = serializers.ListField(child=serializers.CharField(), help_text="주요 드라이버")


class ActionRecommendationSerializer(serializers.Serializer):
    """액션 추천"""
    segment_id = serializers.CharField()
    action_type = serializers.CharField(help_text="액션 유형")
    priority = serializers.IntegerField(help_text="우선순위 (1-10)")
    expected_impact = serializers.CharField(help_text="예상 효과")
    description = serializers.CharField(help_text="액션 설명")
    target_kpi = serializers.CharField(help_text="대상 KPI")


class OverviewStatsSerializer(serializers.Serializer):
    """대시보드 개요 통계"""
    total_segments = serializers.IntegerField()
    total_customers = serializers.IntegerField()
    total_deposit = serializers.FloatField()
    total_loan = serializers.FloatField()
    total_card_usage = serializers.FloatField()
    total_digital_amount = serializers.FloatField()
    month = serializers.CharField()
    mom_deposit_growth = serializers.FloatField(required=False, allow_null=True)
    mom_loan_growth = serializers.FloatField(required=False, allow_null=True)


class SegmentListResponseSerializer(serializers.Serializer):
    """세그먼트 목록 응답"""
    segments = SegmentKPISerializer(many=True)
    total_count = serializers.IntegerField()
    page = serializers.IntegerField()
    page_size = serializers.IntegerField()


class ForecastListResponseSerializer(serializers.Serializer):
    """예측 목록 응답"""
    forecasts = ForecastResultSerializer(many=True)
    total_count = serializers.IntegerField()


class WatchlistResponseSerializer(serializers.Serializer):
    """워치리스트 응답"""
    alerts = WatchlistAlertSerializer(many=True)
    total_count = serializers.IntegerField()


class ActionListResponseSerializer(serializers.Serializer):
    """액션 추천 목록 응답"""
    actions = ActionRecommendationSerializer(many=True)
    total_count = serializers.IntegerField()


class FilterOptionsSerializer(serializers.Serializer):
    """필터 옵션"""
    업종_중분류 = serializers.ListField(child=serializers.CharField())
    사업장_시도 = serializers.ListField(child=serializers.CharField())
    법인_고객등급 = serializers.ListField(child=serializers.CharField())
    전담고객여부 = serializers.ListField(child=serializers.CharField())
    months = serializers.ListField(child=serializers.CharField())


class KPITrendSerializer(serializers.Serializer):
    """KPI 트렌드"""
    kpi = serializers.CharField()
    group_by = serializers.CharField(required=False, allow_null=True)
    data = serializers.ListField(child=serializers.DictField())


class SegmentHistorySerializer(serializers.Serializer):
    """세그먼트 히스토리"""
    segment_id = serializers.CharField()
    history = serializers.ListField(child=serializers.DictField())
    total_months = serializers.IntegerField()
