import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { HiOutlineSearch, HiOutlineFilter, HiOutlineChevronDown } from 'react-icons/hi';
import { fetchSegments, fetchFilterOptions, type SegmentFilters } from '../api';
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

export default function Segments() {
  const [filters, setFilters] = useState<SegmentFilters>({
    page: 1,
    page_size: 20,
    sort_by: '예금총잔액',
    sort_order: 'desc',
  });
  const [showFilters, setShowFilters] = useState(false);

  // Fetch filter options
  const { data: filterOptions } = useQuery({
    queryKey: ['filter-options'],
    queryFn: fetchFilterOptions,
  });

  // Fetch segments
  const {
    data: segmentsData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['segments', filters],
    queryFn: () => fetchSegments(filters),
  });

  const handleFilterChange = (key: keyof SegmentFilters, value: string) => {
    setFilters((prev) => ({
      ...prev,
      [key]: value || undefined,
      page: 1, // Reset page when filter changes
    }));
  };

  const handlePageChange = (newPage: number) => {
    setFilters((prev) => ({ ...prev, page: newPage }));
  };

  if (isLoading) {
    return <PageLoader />;
  }

  if (error) {
    return (
      <ErrorMessage
        message="세그먼트 데이터를 불러오는데 실패했습니다."
        onRetry={() => refetch()}
      />
    );
  }

  const totalPages = Math.ceil((segmentsData?.total_count || 0) / (filters.page_size || 20));

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">세그먼트 목록</h1>
          <p className="text-gray-500">
            총 {segmentsData?.total_count?.toLocaleString() || 0}개 세그먼트
          </p>
        </div>
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="btn btn-outline flex items-center gap-2"
        >
          <HiOutlineFilter className="w-5 h-5" />
          필터
          <HiOutlineChevronDown
            className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`}
          />
        </button>
      </div>

      {/* Filters */}
      {showFilters && (
        <div className="card animate-fade-in">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {/* 업종 필터 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">업종</label>
              <select
                value={filters.업종_중분류 || ''}
                onChange={(e) => handleFilterChange('업종_중분류', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
              >
                <option value="">전체</option>
                {filterOptions?.업종_중분류?.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>

            {/* 지역 필터 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">지역</label>
              <select
                value={filters.사업장_시도 || ''}
                onChange={(e) => handleFilterChange('사업장_시도', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
              >
                <option value="">전체</option>
                {filterOptions?.사업장_시도?.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>

            {/* 등급 필터 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">고객등급</label>
              <select
                value={filters.법인_고객등급 || ''}
                onChange={(e) => handleFilterChange('법인_고객등급', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
              >
                <option value="">전체</option>
                {filterOptions?.법인_고객등급?.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>

            {/* 전담 필터 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">전담여부</label>
              <select
                value={filters.전담고객여부 || ''}
                onChange={(e) => handleFilterChange('전담고객여부', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
              >
                <option value="">전체</option>
                {filterOptions?.전담고객여부?.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>

            {/* 정렬 */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">정렬</label>
              <select
                value={filters.sort_by || '예금총잔액'}
                onChange={(e) => handleFilterChange('sort_by', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-im-primary focus:border-transparent"
              >
                <option value="예금총잔액">예금 총잔액</option>
                <option value="대출총잔액">대출 총잔액</option>
                <option value="카드총사용">카드 총사용</option>
                <option value="customer_count">고객수</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600">
                  세그먼트
                </th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600">업종</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600">지역</th>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-600">등급</th>
                <th className="px-4 py-3 text-right text-sm font-semibold text-gray-600">
                  고객수
                </th>
                <th className="px-4 py-3 text-right text-sm font-semibold text-gray-600">
                  예금총잔액
                </th>
                <th className="px-4 py-3 text-right text-sm font-semibold text-gray-600">
                  대출총잔액
                </th>
                <th className="px-4 py-3 text-right text-sm font-semibold text-gray-600">
                  카드총사용
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {segmentsData?.segments?.map((segment, index) => (
                <tr key={segment.segment_id || index} className="hover:bg-gray-50">
                  <td className="px-4 py-3">
                    <Link
                      to={`/segments/${segment.segment_id}`}
                      className="text-im-primary hover:underline font-medium"
                    >
                      {segment.segment_id?.slice(0, 20) || '-'}...
                    </Link>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">{segment.업종_중분류}</td>
                  <td className="px-4 py-3 text-sm text-gray-600">{segment.사업장_시도}</td>
                  <td className="px-4 py-3">
                    <span className="badge badge-info">{segment.법인_고객등급}</span>
                  </td>
                  <td className="px-4 py-3 text-right text-sm">
                    {segment.customer_count?.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 text-right text-sm font-medium text-green-600">
                    {formatAmount(segment.예금총잔액)}
                  </td>
                  <td className="px-4 py-3 text-right text-sm font-medium text-yellow-600">
                    {formatAmount(segment.대출총잔액)}
                  </td>
                  <td className="px-4 py-3 text-right text-sm font-medium text-red-600">
                    {formatAmount(segment.카드총사용)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200">
            <div className="text-sm text-gray-500">
              페이지 {filters.page} / {totalPages}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => handlePageChange((filters.page || 1) - 1)}
                disabled={(filters.page || 1) <= 1}
                className="btn btn-outline text-sm disabled:opacity-50"
              >
                이전
              </button>
              <button
                onClick={() => handlePageChange((filters.page || 1) + 1)}
                disabled={(filters.page || 1) >= totalPages}
                className="btn btn-outline text-sm disabled:opacity-50"
              >
                다음
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
