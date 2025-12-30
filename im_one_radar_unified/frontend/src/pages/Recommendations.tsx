import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  HiOutlineLightBulb,
  HiOutlineArrowRight,
  HiOutlineChevronUp,
  HiOutlineChevronDown,
} from 'react-icons/hi';
import { fetchRecommendations } from '../api';
import { PageLoader } from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';

const PRIORITY_COLORS = {
  high: 'bg-red-100 border-red-300 text-red-700',
  medium: 'bg-yellow-100 border-yellow-300 text-yellow-700',
  low: 'bg-green-100 border-green-300 text-green-700',
};

function getPriorityLevel(priority: number): 'high' | 'medium' | 'low' {
  if (priority >= 8) return 'high';
  if (priority >= 5) return 'medium';
  return 'low';
}

function getPriorityLabel(priority: number): string {
  if (priority >= 8) return '긴급';
  if (priority >= 5) return '중요';
  return '일반';
}

export default function Recommendations() {
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');
  const [limit, setLimit] = useState(20);
  const [expandedItems, setExpandedItems] = useState<Set<number>>(new Set());

  // Fetch recommendations
  const {
    data: recommendationsData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['recommendations', { limit }],
    queryFn: () => fetchRecommendations({ limit }),
  });

  const toggleExpand = (index: number) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedItems(newExpanded);
  };

  if (isLoading) {
    return <PageLoader />;
  }

  if (error) {
    return (
      <ErrorMessage
        message="추천 데이터를 불러오는데 실패했습니다. MVP 파이프라인을 먼저 실행해주세요."
        onRetry={() => refetch()}
      />
    );
  }

  // Sort actions
  const sortedActions = [...(recommendationsData?.actions || [])].sort((a, b) => {
    return sortOrder === 'desc' ? b.priority - a.priority : a.priority - b.priority;
  });

  // Group by action type
  const actionsByType = sortedActions.reduce(
    (acc, action) => {
      acc[action.action_type] = (acc[action.action_type] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">액션 추천</h1>
          <p className="text-gray-500">
            총 {recommendationsData?.total_count || 0}개 추천
          </p>
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(actionsByType).slice(0, 4).map(([type, count]) => (
          <div key={type} className="card">
            <p className="text-sm text-gray-500">{type}</p>
            <p className="text-2xl font-bold text-im-primary">{count}</p>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">정렬</label>
            <button
              onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
              className="flex items-center gap-2 px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              우선순위
              {sortOrder === 'desc' ? (
                <HiOutlineChevronDown className="w-4 h-4" />
              ) : (
                <HiOutlineChevronUp className="w-4 h-4" />
              )}
            </button>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">표시 개수</label>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
            >
              <option value="10">10개</option>
              <option value="20">20개</option>
              <option value="50">50개</option>
              <option value="100">100개</option>
            </select>
          </div>
        </div>
      </div>

      {/* Action List */}
      <div className="space-y-4">
        {sortedActions.length === 0 ? (
          <div className="card text-center py-12">
            <HiOutlineLightBulb className="w-12 h-12 mx-auto text-gray-400" />
            <p className="mt-4 text-gray-500">추천 액션이 없습니다.</p>
          </div>
        ) : (
          sortedActions.map((action, index) => {
            const priorityLevel = getPriorityLevel(action.priority);
            const isExpanded = expandedItems.has(index);

            return (
              <div
                key={index}
                className={`card border transition-all ${
                  priorityLevel === 'high'
                    ? 'border-red-200 hover:border-red-300'
                    : priorityLevel === 'medium'
                    ? 'border-yellow-200 hover:border-yellow-300'
                    : 'border-green-200 hover:border-green-300'
                }`}
              >
                <div
                  className="flex items-start gap-4 cursor-pointer"
                  onClick={() => toggleExpand(index)}
                >
                  {/* Priority Badge */}
                  <div
                    className={`flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center font-bold text-lg ${PRIORITY_COLORS[priorityLevel]}`}
                  >
                    {action.priority}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span
                        className={`badge ${
                          priorityLevel === 'high'
                            ? 'badge-danger'
                            : priorityLevel === 'medium'
                            ? 'badge-warning'
                            : 'badge-success'
                        }`}
                      >
                        {getPriorityLabel(action.priority)}
                      </span>
                      <span className="badge badge-info">{action.action_type}</span>
                      <span className="text-sm text-gray-500">대상 KPI: {action.target_kpi}</span>
                    </div>

                    <p className="mt-2 font-medium text-gray-800">{action.description}</p>

                    {isExpanded && (
                      <div className="mt-4 p-4 bg-gray-50 rounded-lg animate-fade-in">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div>
                            <p className="text-sm text-gray-500">세그먼트 ID</p>
                            <Link
                              to={`/segments/${action.segment_id}`}
                              className="text-im-primary hover:underline"
                              onClick={(e) => e.stopPropagation()}
                            >
                              {action.segment_id}
                            </Link>
                          </div>
                          <div>
                            <p className="text-sm text-gray-500">예상 효과</p>
                            <p className="font-medium text-green-600">{action.expected_impact}</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Expand Icon */}
                  <div className="flex-shrink-0">
                    {isExpanded ? (
                      <HiOutlineChevronUp className="w-5 h-5 text-gray-400" />
                    ) : (
                      <HiOutlineChevronDown className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Tip */}
      <div className="card bg-blue-50 border border-blue-200">
        <div className="flex items-start gap-3">
          <HiOutlineLightBulb className="w-6 h-6 text-blue-500 flex-shrink-0" />
          <div>
            <p className="font-medium text-blue-800">액션 우선순위 가이드</p>
            <ul className="mt-2 text-sm text-blue-700 space-y-1">
              <li>• <strong>긴급 (8-10):</strong> 즉시 조치가 필요한 항목</li>
              <li>• <strong>중요 (5-7):</strong> 1-2주 내 검토 권장</li>
              <li>• <strong>일반 (1-4):</strong> 월간 리뷰 시 검토</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
