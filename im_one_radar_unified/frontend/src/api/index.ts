import axios from 'axios';
import type {
  OverviewStats,
  SegmentListResponse,
  SegmentKPI,
  ForecastListResponse,
  WatchlistResponse,
  ActionListResponse,
  FilterOptions,
  KPITrendData,
  SegmentHistory,
} from '../types';

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Dashboard API
export const fetchOverview = async (month?: string): Promise<OverviewStats> => {
  const params = month ? { month } : {};
  const { data } = await api.get<OverviewStats>('/overview', { params });
  return data;
};

export const fetchKPITrends = async (
  kpi: string = '예금총잔액',
  groupBy?: string,
  months: number = 12
): Promise<KPITrendData> => {
  const params = { kpi, group_by: groupBy, months };
  const { data } = await api.get<KPITrendData>('/kpi-trends', { params });
  return data;
};

// Segments API
export interface SegmentFilters {
  page?: number;
  page_size?: number;
  month?: string;
  업종_중분류?: string;
  사업장_시도?: string;
  법인_고객등급?: string;
  전담고객여부?: string;
  sort_by?: string;
  sort_order?: 'asc' | 'desc';
}

export const fetchSegments = async (filters: SegmentFilters = {}): Promise<SegmentListResponse> => {
  const { data } = await api.get<SegmentListResponse>('/segments', { params: filters });
  return data;
};

export const fetchSegmentDetail = async (segmentId: string, month?: string): Promise<SegmentKPI> => {
  const params = month ? { month } : {};
  const { data } = await api.get<SegmentKPI>(`/segments/${segmentId}`, { params });
  return data;
};

export const fetchSegmentHistory = async (
  segmentId: string,
  startMonth?: string,
  endMonth?: string
): Promise<SegmentHistory> => {
  const params = { start_month: startMonth, end_month: endMonth };
  const { data } = await api.get<SegmentHistory>(`/segments/${segmentId}/history`, { params });
  return data;
};

// Forecasts API
export interface ForecastFilters {
  segment_id?: string;
  kpi?: string;
  horizon?: number;
}

export const fetchForecasts = async (filters: ForecastFilters = {}): Promise<ForecastListResponse> => {
  const { data } = await api.get<ForecastListResponse>('/forecasts', { params: filters });
  return data;
};

// Watchlist API
export interface WatchlistFilters {
  alert_type?: string;
  severity?: string;
  limit?: number;
}

export const fetchWatchlist = async (filters: WatchlistFilters = {}): Promise<WatchlistResponse> => {
  const { data } = await api.get<WatchlistResponse>('/watchlist', { params: filters });
  return data;
};

// Recommendations API
export interface RecommendationFilters {
  segment_id?: string;
  action_type?: string;
  limit?: number;
}

export const fetchRecommendations = async (filters: RecommendationFilters = {}): Promise<ActionListResponse> => {
  const { data } = await api.get<ActionListResponse>('/recommendations', { params: filters });
  return data;
};

// Filter Options API
export const fetchFilterOptions = async (): Promise<FilterOptions> => {
  const { data } = await api.get<FilterOptions>('/filters');
  return data;
};

// Refresh Data API
export const refreshData = async (): Promise<{ message: string }> => {
  const { data } = await api.post<{ message: string }>('/refresh');
  return data;
};

export default api;
