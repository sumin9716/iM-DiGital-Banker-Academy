import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Segments from './pages/Segments'
import SegmentDetail from './pages/SegmentDetail'
import Forecasts from './pages/Forecasts'
import Watchlist from './pages/Watchlist'
import Recommendations from './pages/Recommendations'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="segments" element={<Segments />} />
        <Route path="segments/:segmentId" element={<SegmentDetail />} />
        <Route path="forecasts" element={<Forecasts />} />
        <Route path="watchlist" element={<Watchlist />} />
        <Route path="recommendations" element={<Recommendations />} />
      </Route>
    </Routes>
  )
}

export default App
