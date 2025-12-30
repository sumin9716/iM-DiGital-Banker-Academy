import { NavLink, Outlet } from 'react-router-dom';
import {
  HiOutlineChartPie,
  HiOutlineCube,
  HiOutlineChartBar,
  HiOutlineBell,
  HiOutlineLightBulb,
} from 'react-icons/hi';

const navigation = [
  { name: '대시보드', href: '/', icon: HiOutlineChartPie },
  { name: '세그먼트', href: '/segments', icon: HiOutlineCube },
  { name: '예측', href: '/forecasts', icon: HiOutlineChartBar },
  { name: '워치리스트', href: '/watchlist', icon: HiOutlineBell },
  { name: '추천', href: '/recommendations', icon: HiOutlineLightBulb },
];

export default function Layout() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-50 w-64 bg-im-dark text-white">
        {/* Logo */}
        <div className="flex items-center gap-3 px-6 py-5 border-b border-gray-700">
          <div className="w-10 h-10 bg-gradient-to-br from-im-primary to-im-secondary rounded-lg flex items-center justify-center">
            <span className="text-xl font-bold">iM</span>
          </div>
          <div>
            <h1 className="text-lg font-bold">ONEderful</h1>
            <p className="text-xs text-gray-400">Segment Radar</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="mt-6 px-3">
          <ul className="space-y-1">
            {navigation.map((item) => (
              <li key={item.name}>
                <NavLink
                  to={item.href}
                  end={item.href === '/'}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                      isActive
                        ? 'bg-im-primary text-white'
                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                    }`
                  }
                >
                  <item.icon className="w-5 h-5" />
                  <span className="font-medium">{item.name}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-700">
          <div className="text-xs text-gray-400 text-center">
            <p>© 2025 iM ONEderful</p>
            <p>Version 1.0.0</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="ml-64">
        {/* Header */}
        <header className="sticky top-0 z-40 bg-white border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-800">통합 세그먼트 레이더</h2>
              <p className="text-sm text-gray-500">실시간 금융 KPI 모니터링 및 예측</p>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-500">
                마지막 업데이트: {new Date().toLocaleDateString('ko-KR')}
              </span>
              <button className="btn btn-primary text-sm">
                데이터 새로고침
              </button>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <div className="p-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
