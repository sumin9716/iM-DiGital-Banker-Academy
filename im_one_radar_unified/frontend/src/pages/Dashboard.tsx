import { useQuery } from '@tanstack/react-query';
import {
  HiOutlineCube,
  HiOutlineUsers,
  HiOutlineCurrencyDollar,
  HiOutlineCreditCard,
  HiOutlineDeviceMobile,
  HiOutlineTrendingUp,
} from 'react-icons/hi';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend,
} from 'recharts';
import { fetchOverview, fetchKPITrends, fetchWatchlist } from '../api';
import StatCard from '../components/StatCard';
import { PageLoader } from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

// 금액 포맷
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

export default function Dashboard() {
  // Fetch overview stats
  const {
    data: overview,
    isLoading: overviewLoading,
    error: overviewError,
    refetch: refetchOverview,
  } = useQuery({
    queryKey: ['overview'],
    queryFn: () => fetchOverview(),
  });

  // Fetch KPI trends
  const { data: depositTrends } = useQuery({
    queryKey: ['kpi-trends', '예금총잔액'],
    queryFn: () => fetchKPITrends('예금총잔액', undefined, 12),
  });

  const { data: loanTrends } = useQuery({
    queryKey: ['kpi-trends', '대출총잔액'],
    queryFn: () => fetchKPITrends('대출총잔액', undefined, 12),
  });

  // Fetch watchlist
  const { data: watchlist } = useQuery({
    queryKey: ['watchlist', { limit: 5 }],
    queryFn: () => fetchWatchlist({ limit: 5 }),
  });

  if (overviewLoading) {
    return <PageLoader />;
  }

  if (overviewError) {
    return (
      <ErrorMessage
        message="데이터를 불러오는데 실패했습니다. MVP 파이프라인을 먼저 실행해주세요."
        onRetry={() => refetchOverview()}
      />
    );
  }

  // Prepare chart data
  const chartData = depositTrends?.data?.map((item, index) => ({
    month: item.month,
    예금총잔액: item['예금총잔액'] || 0,
    대출총잔액: loanTrends?.data?.[index]?.['대출총잔액'] || 0,
  })) || [];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">대시보드</h1>
          <p className="text-gray-500">기준월: {overview?.month || '-'}</p>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <StatCard
          title="총 세그먼트"
          value={overview?.total_segments?.toLocaleString() || '0'}
          icon={<HiOutlineCube className="w-6 h-6" />}
          color="blue"
        />
        <StatCard
          title="총 고객수"
          value={overview?.total_customers?.toLocaleString() || '0'}
          icon={<HiOutlineUsers className="w-6 h-6" />}
          color="purple"
        />
        <StatCard
          title="예금 총잔액"
          value={formatAmount(overview?.total_deposit || 0)}
          icon={<HiOutlineCurrencyDollar className="w-6 h-6" />}
          color="green"
          change={overview?.mom_deposit_growth}
          changeLabel="전월 대비"
        />
        <StatCard
          title="대출 총잔액"
          value={formatAmount(overview?.total_loan || 0)}
          icon={<HiOutlineTrendingUp className="w-6 h-6" />}
          color="yellow"
          change={overview?.mom_loan_growth}
          changeLabel="전월 대비"
        />
        <StatCard
          title="카드 사용액"
          value={formatAmount(overview?.total_card_usage || 0)}
          icon={<HiOutlineCreditCard className="w-6 h-6" />}
          color="red"
        />
        <StatCard
          title="디지털 거래액"
          value={formatAmount(overview?.total_digital_amount || 0)}
          icon={<HiOutlineDeviceMobile className="w-6 h-6" />}
          color="blue"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* KPI Trend Chart */}
        <div className="card">
          <h3 className="card-header">월별 KPI 추이</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="month"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => value.slice(-5)}
                />
                <YAxis
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => formatAmount(value)}
                />
                <Tooltip
                  formatter={(value: number) => formatAmount(value)}
                  labelStyle={{ color: '#333' }}
                />
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
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Watchlist Alerts */}
        <div className="card">
          <h3 className="card-header">최근 알림</h3>
          {watchlist?.alerts && watchlist.alerts.length > 0 ? (
            <div className="space-y-3">
              {watchlist.alerts.map((alert, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border ${
                    alert.severity === 'HIGH'
                      ? 'bg-red-50 border-red-200'
                      : alert.severity === 'MEDIUM'
                      ? 'bg-yellow-50 border-yellow-200'
                      : 'bg-blue-50 border-blue-200'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span
                          className={`badge ${
                            alert.alert_type === 'RISK_DOWN'
                              ? 'badge-danger'
                              : alert.alert_type === 'RISK_UP'
                              ? 'badge-warning'
                              : 'badge-success'
                          }`}
                        >
                          {alert.alert_type === 'RISK_DOWN'
                            ? '급감'
                            : alert.alert_type === 'RISK_UP'
                            ? '급증'
                            : '기회'}
                        </span>
                        <span
                          className={`badge ${
                            alert.severity === 'HIGH'
                              ? 'badge-danger'
                              : alert.severity === 'MEDIUM'
                              ? 'badge-warning'
                              : 'badge-info'
                          }`}
                        >
                          {alert.severity}
                        </span>
                      </div>
                      <p className="mt-2 text-sm font-medium text-gray-800">
                        {alert.segment_info?.업종_중분류} / {alert.segment_info?.사업장_시도}
                      </p>
                      <p className="text-sm text-gray-600">{alert.message}</p>
                    </div>
                    <span className="text-sm text-gray-500">{alert.kpi}</span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <p>알림이 없습니다.</p>
            </div>
          )}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 className="card-header">빠른 액션</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <a
            href="/segments"
            className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors text-center"
          >
            <HiOutlineCube className="w-8 h-8 mx-auto text-im-primary" />
            <p className="mt-2 font-medium">세그먼트 조회</p>
          </a>
          <a
            href="/forecasts"
            className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors text-center"
          >
            <HiOutlineTrendingUp className="w-8 h-8 mx-auto text-im-secondary" />
            <p className="mt-2 font-medium">예측 결과</p>
          </a>
          <a
            href="/watchlist"
            className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors text-center"
          >
            <HiOutlineCurrencyDollar className="w-8 h-8 mx-auto text-im-accent" />
            <p className="mt-2 font-medium">워치리스트</p>
          </a>
          <a
            href="/recommendations"
            className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors text-center"
          >
            <HiOutlineDeviceMobile className="w-8 h-8 mx-auto text-im-success" />
            <p className="mt-2 font-medium">액션 추천</p>
          </a>
        </div>
      </div>
    </div>
  );
}
