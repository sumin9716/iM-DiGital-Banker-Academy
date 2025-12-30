// API Response Types

export interface OverviewStats {
  total_segments: number;
  total_customers: number;
  total_deposit: number;
  total_loan: number;
  total_card_usage: number;
  total_digital_amount: number;
  month: string;
  mom_deposit_growth?: number;
  mom_loan_growth?: number;
}

export interface SegmentKPI {
  segment_id: string;
  업종_중분류: string;
  사업장_시도: string;
  법인_고객등급: string;
  전담고객여부: string;
  customer_count: number;
  month: string;
  예금총잔액: number;
  대출총잔액: number;
  카드총사용: number;
  디지털거래금액: number;
  순유입: number;
  한도소진율?: number;
  디지털비중?: number;
}

export interface SegmentListResponse {
  segments: SegmentKPI[];
  total_count: number;
  page: number;
  page_size: number;
}

export interface ForecastResult {
  segment_id: string;
  target_kpi: string;
  horizon: number;
  forecast_month: string;
  predicted_value: number;
  actual_value?: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface ForecastListResponse {
  forecasts: ForecastResult[];
  total_count: number;
}

export interface WatchlistAlert {
  segment_id: string;
  segment_info: {
    업종_중분류: string;
    사업장_시도: string;
    법인_고객등급: string;
    전담고객여부: string;
  };
  alert_type: 'RISK_DOWN' | 'RISK_UP' | 'OPPORTUNITY';
  severity: 'HIGH' | 'MEDIUM' | 'LOW';
  kpi: string;
  residual_zscore: number;
  message: string;
  drivers: string[];
}

export interface WatchlistResponse {
  alerts: WatchlistAlert[];
  total_count: number;
}

export interface ActionRecommendation {
  segment_id: string;
  action_type: string;
  priority: number;
  expected_impact: string;
  description: string;
  target_kpi: string;
}

export interface ActionListResponse {
  actions: ActionRecommendation[];
  total_count: number;
}

export interface FilterOptions {
  업종_중분류: string[];
  사업장_시도: string[];
  법인_고객등급: string[];
  전담고객여부: string[];
  months: string[];
}

export interface KPITrendData {
  kpi: string;
  group_by?: string;
  data: Array<{
    month: string;
    [key: string]: string | number;
  }>;
}

export interface SegmentHistory {
  segment_id: string;
  history: Array<Record<string, unknown>>;
  total_months: number;
}
