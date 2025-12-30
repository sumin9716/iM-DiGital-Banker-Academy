# iM ONEderful Segment Radar - Backend API

FastAPI 기반 REST API 서버

## 실행 방법

### 1. 가상환경 설정 (권장)

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
# 개발 모드 (자동 리로드)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 모드
python main.py
```

### 3. API 문서 확인

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### Dashboard
- `GET /api/overview` - 대시보드 개요 통계

### Segments
- `GET /api/segments` - 세그먼트 목록 조회
- `GET /api/segments/{segment_id}` - 세그먼트 상세 조회
- `GET /api/segments/{segment_id}/history` - 세그먼트 히스토리 조회

### Forecasts
- `GET /api/forecasts` - 예측 결과 조회

### Watchlist
- `GET /api/watchlist` - 워치리스트 알림 조회

### Recommendations
- `GET /api/recommendations` - 액션 추천 조회

### Utilities
- `GET /api/filters` - 필터 옵션 조회
- `GET /api/kpi-trends` - KPI 트렌드 조회
- `POST /api/refresh` - 데이터 캐시 갱신
