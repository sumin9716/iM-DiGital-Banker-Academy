# iM ONEderful Segment Radar - Django Backend

Django REST Framework 기반의 세그먼트 레이더 API 서버입니다.

## 설치

```bash
cd backend
pip install -r requirements.txt
```

## 실행

```bash
# 마이그레이션 (최초 1회)
python manage.py migrate

# 개발 서버 실행
python manage.py runserver 8000
```

## API 엔드포인트

### Dashboard
- `GET /api/overview/` - 대시보드 개요 통계
- `GET /api/kpi-trends/` - KPI 트렌드 (차트용)

### Segments
- `GET /api/segments/` - 세그먼트 목록
- `GET /api/segments/{segment_id}/` - 세그먼트 상세
- `GET /api/segments/{segment_id}/history/` - 세그먼트 히스토리

### Forecasts
- `GET /api/forecasts/` - 예측 결과 조회

### Watchlist
- `GET /api/watchlist/` - 워치리스트 알림

### Recommendations
- `GET /api/recommendations/` - 액션 추천

### Filters
- `GET /api/filters/` - 필터 옵션 (드롭다운용)

### Admin
- `POST /api/refresh/` - 데이터 캐시 갱신
- `GET /health/` - 헬스 체크

## 쿼리 파라미터

### 세그먼트 목록
- `page` - 페이지 번호 (default: 1)
- `page_size` - 페이지 크기 (default: 20)
- `month` - 조회 월 (YYYY-MM)
- `업종_중분류` - 업종 필터
- `사업장_시도` - 지역 필터
- `법인_고객등급` - 등급 필터
- `전담고객여부` - 전담 필터
- `sort_by` - 정렬 기준
- `sort_order` - 정렬 순서 (asc/desc)

### 예측 결과
- `segment_id` - 세그먼트 ID
- `kpi` - KPI 필터
- `horizon` - 예측 기간 (1-3)

### 워치리스트
- `alert_type` - 알림 유형
- `severity` - 심각도
- `limit` - 조회 개수

## 기술 스택

- Django 5.0
- Django REST Framework 3.14
- pandas, numpy
- django-cors-headers
