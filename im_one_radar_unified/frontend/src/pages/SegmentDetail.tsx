import { useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { HiOutlineArrowLeft } from 'react-icons/hi';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { fetchSegmentDetail, fetchSegmentHistory, fetchForecasts } from '../api';
import { PageLoader } from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import StatCard from '../components/StatCard';

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

export default function SegmentDetail() {
  const { segmentId } = useParams<{ segmentId: string }>();

  // Fetch segment detail
  const {
    data: segment,
    isLoading: segmentLoading,
    error: segmentError,
    refetch: refetchSegment,
  } = useQuery({
    queryKey: ['segment', segmentId],
    queryFn: () => fetchSegmentDetail(segmentId!),
    enabled: !!segmentId,
  });

  // Fetch segment history
  const { data: history } = useQuery({
    queryKey: ['segment-history', segmentId],
    queryFn: () => fetchSegmentHistory(segmentId!),
    enabled: !!segmentId,
  });

  // Fetch forecasts for this segment
  const { data: forecasts } = useQuery({
    queryKey: ['forecasts', { segment_id: segmentId }],
    queryFn: () => fetchForecasts({ segment_id: segmentId }),
    enabled: !!segmentId,
  });

  if (segmentLoading) {
    return <PageLoader />;
  }

  if (segmentError || !segment) {
    return (
      <ErrorMessage
        message="세그먼트 정보를 불러오는데 실패했습니다."
        onRetry={() => refetchSegment()}
      />
    );
  }

  // Prepare chart data
  const chartData = history?.history?.map((item: Record<string, unknown>) => ({
    month: String(item.month || '').slice(-5),
    예금총잔액: Number(item['예금총잔액']) || 0,
    대출총잔액: Number(item['대출총잔액']) || 0,
    카드총사용: Number(item['카드총사용']) || 0,
  })) || [];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center gap-4">
        <Link
          to="/segments"
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <HiOutlineArrowLeft className="w-5 h-5" />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-gray-800">세그먼트 상세</h1>
          <p className="text-gray-500">{segment.segment_id}</p>
        </div>
      </div>

      {/* Segment Info */}
      <div className="card">
        <h3 className="card-header">세그먼트 정보</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-500">업종</p>
            <p className="font-semibold">{segment.업종_중분류}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">지역</p>
            <p className="font-semibold">{segment.사업장_시도}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">고객등급</p>
            <p className="font-semibold">{segment.법인_고객등급}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">전담여부</p>
            <p className="font-semibold">{segment.전담고객여부}</p>
          </div>
        </div>
      </div>

      {/* KPI Stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <StatCard
          title="고객 수"
          value={segment.customer_count?.toLocaleString() || '0'}
          color="purple"
        />
        <StatCard
          title="예금 총잔액"
          value={formatAmount(segment.예금총잔액)}
          color="green"
        />
        <StatCard
          title="대출 총잔액"
          value={formatAmount(segment.대출총잔액)}
          color="yellow"
        />
        <StatCard
          title="카드 총사용"
          value={formatAmount(segment.카드총사용)}
          color="red"
        />
        <StatCard
          title="디지털 거래액"
          value={formatAmount(segment.디지털거래금액)}
          color="blue"
        />
        <StatCard
          title="순유입"
          value={formatAmount(segment.순유입)}
          color={segment.순유입 >= 0 ? 'green' : 'red'}
        />
      </div>

      {/* History Chart */}
      <div className="card">
        <h3 className="card-header">월별 KPI 추이</h3>
        <div className="h-80">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={formatAmount} />
                <Tooltip formatter={(value: number) => formatAmount(value)} />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="예금총잔액"
                  stroke="#00B050"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="대출총잔액"
                  stroke="#FFC000"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="카드총사용"
                  stroke="#C00000"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-500">
              히스토리 데이터가 없습니다.
            </div>
          )}
        </div>
      </div>

      {/* Forecasts */}
      {forecasts?.forecasts && forecasts.forecasts.length > 0 && (
        <div className="card">
          <h3 className="card-header">예측 결과</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600">KPI</th>
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
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {forecasts.forecasts.map((forecast, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-medium">{forecast.target_kpi}</td>
                    <td className="px-4 py-3 text-sm text-center">{forecast.horizon}개월</td>
                    <td className="px-4 py-3 text-sm text-center">{forecast.forecast_month}</td>
                    <td className="px-4 py-3 text-sm text-right font-medium text-im-primary">
                      {formatAmount(forecast.predicted_value)}
                    </td>
                    <td className="px-4 py-3 text-sm text-right">
                      {forecast.actual_value !== undefined
                        ? formatAmount(forecast.actual_value)
                        : '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
