import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  BarChart,
  Bar,
} from 'recharts';
import { fetchForecasts, fetchFilterOptions } from '../api';
import { PageLoader } from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

function formatAmount(value: number): string {
  if (value >= 1e12) {
    return `${(value / 1e12).toFixed(1)}조`;
  } else if (value >= 1e8) {
    return `${(value / 1e8).toFixed(1)}억`;
  } else if (value >= 1e4) {
    return `${(value / 1e4).toFixed(1)}만`;
  }
  return value.toLocaleString();
}

const KPI_OPTIONS = ['예금총잔액', '대출총잔액', '카드총사용', '디지털거래금액', '순유입'];

export default function Forecasts() {
  const [selectedKPI, setSelectedKPI] = useState('예금총잔액');
  const [selectedHorizon, setSelectedHorizon] = useState<number | undefined>(undefined);

  // Fetch forecasts
  const {
    data: forecastData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['forecasts', { kpi: selectedKPI, horizon: selectedHorizon }],
    queryFn: () => fetchForecasts({ kpi: selectedKPI, horizon: selectedHorizon }),
  });

  if (isLoading) {
    return <PageLoader />;
  }

  if (error) {
    return (
      <ErrorMessage
        message="예측 데이터를 불러오는데 실패했습니다. MVP 파이프라인을 먼저 실행해주세요."
        onRetry={() => refetch()}
      />
    );
  }

  // Aggregate forecasts by month for chart
  const chartData = forecastData?.forecasts?.reduce((acc, item) => {
    const month = item.forecast_month;
    const existing = acc.find((d) => d.month === month);
    if (existing) {
      existing.predicted = (existing.predicted || 0) + item.predicted_value;
      existing.actual = item.actual_value !== undefined 
        ? (existing.actual || 0) + item.actual_value 
        : existing.actual;
      existing.count += 1;
    } else {
      acc.push({
        month,
        predicted: item.predicted_value,
        actual: item.actual_value,
        count: 1,
      });
    }
    return acc;
  }, [] as Array<{ month: string; predicted: number; actual?: number; count: number }>) || [];

  // Sort by month
  chartData.sort((a, b) => a.month.localeCompare(b.month));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">예측 결과</h1>
          <p className="text-gray-500">
            총 {forecastData?.total_count?.toLocaleString() || 0}개 예측
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">KPI</label>
            <select
              value={selectedKPI}
              onChange={(e) => setSelectedKPI(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
            >
              {KPI_OPTIONS.map((kpi) => (
                <option key={kpi} value={kpi}>
                  {kpi}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">예측 기간</label>
            <select
              value={selectedHorizon || ''}
              onChange={(e) => setSelectedHorizon(e.target.value ? Number(e.target.value) : undefined)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
            >
              <option value="">전체</option>
              <option value="1">1개월</option>
              <option value="2">2개월</option>
              <option value="3">3개월</option>
            </select>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card bg-gradient-to-r from-blue-50 to-blue-100 border border-blue-200">
          <p className="text-sm text-blue-600 font-medium">평균 예측값</p>
          <p className="text-2xl font-bold text-blue-800 mt-1">
            {formatAmount(
              (forecastData?.forecasts?.reduce((sum, f) => sum + f.predicted_value, 0) || 0) /
                (forecastData?.forecasts?.length || 1)
            )}
          </p>
        </div>
        <div className="card bg-gradient-to-r from-green-50 to-green-100 border border-green-200">
          <p className="text-sm text-green-600 font-medium">최대 예측값</p>
          <p className="text-2xl font-bold text-green-800 mt-1">
            {formatAmount(
              Math.max(...(forecastData?.forecasts?.map((f) => f.predicted_value) || [0]))
            )}
          </p>
        </div>
        <div className="card bg-gradient-to-r from-purple-50 to-purple-100 border border-purple-200">
          <p className="text-sm text-purple-600 font-medium">예측 세그먼트 수</p>
          <p className="text-2xl font-bold text-purple-800 mt-1">
            {new Set(forecastData?.forecasts?.map((f) => f.segment_id)).size.toLocaleString()}
          </p>
        </div>
      </div>

      {/* Chart */}
      <div className="card">
        <h3 className="card-header">예측 vs 실제 ({selectedKPI})</h3>
        <div className="h-80">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={formatAmount} />
                <Tooltip formatter={(value: number) => formatAmount(value)} />
                <Legend />
                <Bar dataKey="predicted" name="예측값" fill="#006CB8" />
                <Bar dataKey="actual" name="실제값" fill="#00B050" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              예측 데이터가 없습니다.
            </div>
          )}
        </div>
      </div>

      {/* Forecast Table */}
      <div className="card overflow-hidden">
        <h3 className="card-header">상세 예측 목록</h3>
        <div className="overflow-x-auto max-h-96">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200 sticky top-0">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600">
                  세그먼트 ID
                </th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-600">KPI</th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-600">
                  예측 기간
                </th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-600">
                  예측 월
                </th>
                <th className="px-4 py-3 text-right text-sm font-semibold text-gray-600">
                  예측값
                </th>
                <th className="px-4 py-3 text-right text-sm font-semibold text-gray-600">
                  실제값
                </th>
                <th className="px-4 py-3 text-right text-sm font-semibold text-gray-600">오차</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {forecastData?.forecasts?.slice(0, 100).map((forecast, index) => {
                const error =
                  forecast.actual_value !== undefined
                    ? ((forecast.predicted_value - forecast.actual_value) / forecast.actual_value) * 100
                    : null;

                return (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-mono text-gray-600">
                      {forecast.segment_id?.slice(0, 25)}...
                    </td>
                    <td className="px-4 py-3 text-sm text-center">{forecast.target_kpi}</td>
                    <td className="px-4 py-3 text-sm text-center">
                      <span className="badge badge-info">{forecast.horizon}개월</span>
                    </td>
                    <td className="px-4 py-3 text-sm text-center">{forecast.forecast_month}</td>
                    <td className="px-4 py-3 text-sm text-right font-medium text-im-primary">
                      {formatAmount(forecast.predicted_value)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right">
                      {forecast.actual_value !== undefined
                        ? formatAmount(forecast.actual_value)
                        : '-'}
                    </td>
                    <td className="px-4 py-3 text-sm text-right">
                      {error !== null ? (
                        <span className={error > 0 ? 'text-red-500' : 'text-green-500'}>
                          {error > 0 ? '+' : ''}
                          {error.toFixed(1)}%
                        </span>
                      ) : (
                        '-'
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {forecastData?.forecasts && forecastData.forecasts.length > 100 && (
          <div className="px-4 py-3 border-t border-gray-200 text-sm text-gray-500 text-center">
            상위 100개만 표시됩니다. (전체: {forecastData.total_count}개)
          </div>
        )}
      </div>
    </div>
  );
}
