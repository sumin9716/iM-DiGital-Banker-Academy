import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  HiOutlineExclamation,
  HiOutlineTrendingUp,
  HiOutlineTrendingDown,
  HiOutlineLightBulb,
} from 'react-icons/hi';
import { fetchWatchlist } from '../api';
import { PageLoader } from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import type { WatchlistAlert } from '../types';

const ALERT_TYPE_OPTIONS = [
  { value: '', label: '전체' },
  { value: 'RISK_DOWN', label: '급감 리스크' },
  { value: 'RISK_UP', label: '급증 리스크' },
  { value: 'OPPORTUNITY', label: '기회' },
];

const SEVERITY_OPTIONS = [
  { value: '', label: '전체' },
  { value: 'HIGH', label: '높음' },
  { value: 'MEDIUM', label: '중간' },
  { value: 'LOW', label: '낮음' },
];

function AlertIcon({ type }: { type: string }) {
  switch (type) {
    case 'RISK_DOWN':
      return <HiOutlineTrendingDown className="w-6 h-6 text-red-500" />;
    case 'RISK_UP':
      return <HiOutlineTrendingUp className="w-6 h-6 text-yellow-500" />;
    case 'OPPORTUNITY':
      return <HiOutlineLightBulb className="w-6 h-6 text-green-500" />;
    default:
      return <HiOutlineExclamation className="w-6 h-6 text-gray-500" />;
  }
}

function getSeverityColor(severity: string) {
  switch (severity) {
    case 'HIGH':
      return 'bg-red-100 border-red-300 text-red-800';
    case 'MEDIUM':
      return 'bg-yellow-100 border-yellow-300 text-yellow-800';
    case 'LOW':
      return 'bg-blue-100 border-blue-300 text-blue-800';
    default:
      return 'bg-gray-100 border-gray-300 text-gray-800';
  }
}

function getAlertTypeLabel(type: string) {
  switch (type) {
    case 'RISK_DOWN':
      return '급감';
    case 'RISK_UP':
      return '급증';
    case 'OPPORTUNITY':
      return '기회';
    default:
      return type;
  }
}

export default function Watchlist() {
  const [alertType, setAlertType] = useState('');
  const [severity, setSeverity] = useState('');
  const [limit, setLimit] = useState(50);

  // Fetch watchlist
  const {
    data: watchlistData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['watchlist', { alert_type: alertType || undefined, severity: severity || undefined, limit }],
    queryFn: () =>
      fetchWatchlist({
        alert_type: alertType || undefined,
        severity: severity || undefined,
        limit,
      }),
  });

  if (isLoading) {
    return <PageLoader />;
  }

  if (error) {
    return (
      <ErrorMessage
        message="워치리스트 데이터를 불러오는데 실패했습니다. MVP 파이프라인을 먼저 실행해주세요."
        onRetry={() => refetch()}
      />
    );
  }

  // Group alerts by type for summary
  const alertSummary = watchlistData?.alerts?.reduce(
    (acc, alert) => {
      acc[alert.alert_type] = (acc[alert.alert_type] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  ) || {};

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">워치리스트</h1>
          <p className="text-gray-500">
            총 {watchlistData?.total_count || 0}개 알림
          </p>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card bg-gradient-to-r from-red-50 to-red-100 border border-red-200">
          <div className="flex items-center gap-3">
            <HiOutlineTrendingDown className="w-8 h-8 text-red-500" />
            <div>
              <p className="text-sm text-red-600 font-medium">급감 리스크</p>
              <p className="text-2xl font-bold text-red-800">
                {alertSummary['RISK_DOWN'] || 0}
              </p>
            </div>
          </div>
        </div>
        <div className="card bg-gradient-to-r from-yellow-50 to-yellow-100 border border-yellow-200">
          <div className="flex items-center gap-3">
            <HiOutlineTrendingUp className="w-8 h-8 text-yellow-500" />
            <div>
              <p className="text-sm text-yellow-600 font-medium">급증 리스크</p>
              <p className="text-2xl font-bold text-yellow-800">
                {alertSummary['RISK_UP'] || 0}
              </p>
            </div>
          </div>
        </div>
        <div className="card bg-gradient-to-r from-green-50 to-green-100 border border-green-200">
          <div className="flex items-center gap-3">
            <HiOutlineLightBulb className="w-8 h-8 text-green-500" />
            <div>
              <p className="text-sm text-green-600 font-medium">기회</p>
              <p className="text-2xl font-bold text-green-800">
                {alertSummary['OPPORTUNITY'] || 0}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">알림 유형</label>
            <select
              value={alertType}
              onChange={(e) => setAlertType(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
            >
              {ALERT_TYPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">심각도</label>
            <select
              value={severity}
              onChange={(e) => setSeverity(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
            >
              {SEVERITY_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">표시 개수</label>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
            >
              <option value="20">20개</option>
              <option value="50">50개</option>
              <option value="100">100개</option>
              <option value="200">200개</option>
            </select>
          </div>
        </div>
      </div>

      {/* Alert List */}
      <div className="space-y-4">
        {watchlistData?.alerts?.length === 0 ? (
          <div className="card text-center py-12">
            <HiOutlineExclamation className="w-12 h-12 mx-auto text-gray-400" />
            <p className="mt-4 text-gray-500">해당 조건의 알림이 없습니다.</p>
          </div>
        ) : (
          watchlistData?.alerts?.map((alert, index) => (
            <div
              key={index}
              className={`card border ${
                alert.severity === 'HIGH'
                  ? 'border-red-200 bg-red-50/50'
                  : alert.severity === 'MEDIUM'
                  ? 'border-yellow-200 bg-yellow-50/50'
                  : 'border-blue-200 bg-blue-50/50'
              }`}
            >
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 p-2 bg-white rounded-lg shadow-sm">
                  <AlertIcon type={alert.alert_type} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span
                      className={`badge ${
                        alert.alert_type === 'RISK_DOWN'
                          ? 'badge-danger'
                          : alert.alert_type === 'RISK_UP'
                          ? 'badge-warning'
                          : 'badge-success'
                      }`}
                    >
                      {getAlertTypeLabel(alert.alert_type)}
                    </span>
                    <span className={`badge ${getSeverityColor(alert.severity)}`}>
                      {alert.severity}
                    </span>
                    <span className="badge badge-info">{alert.kpi}</span>
                  </div>
                  <div className="mt-2">
                    <Link
                      to={`/segments/${alert.segment_id}`}
                      className="text-im-primary hover:underline font-medium"
                    >
                      {alert.segment_info?.업종_중분류} / {alert.segment_info?.사업장_시도} /
                      {alert.segment_info?.법인_고객등급}
                    </Link>
                  </div>
                  <p className="mt-1 text-gray-600">{alert.message}</p>
                  {alert.drivers && alert.drivers.length > 0 && (
                    <div className="mt-2">
                      <p className="text-sm text-gray-500">주요 드라이버:</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {alert.drivers.map((driver, i) => (
                          <span key={i} className="text-xs bg-gray-100 px-2 py-1 rounded">
                            {driver}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex-shrink-0 text-right">
                  <p className="text-sm text-gray-500">Z-Score</p>
                  <p
                    className={`text-lg font-bold ${
                      alert.residual_zscore > 0 ? 'text-red-600' : 'text-green-600'
                    }`}
                  >
                    {alert.residual_zscore > 0 ? '+' : ''}
                    {alert.residual_zscore?.toFixed(2)}
                  </p>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
