# iM ONEderful Segment Radar - Frontend

React + TypeScript + Vite 기반 프론트엔드 애플리케이션

## 기술 스택

- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **Styling:** Tailwind CSS
- **State Management:** React Query (TanStack Query)
- **Routing:** React Router v6
- **Charts:** Recharts
- **HTTP Client:** Axios

## 프로젝트 구조

```
frontend/
├── src/
│   ├── api/              # API 호출 함수
│   ├── components/       # 재사용 가능한 컴포넌트
│   ├── pages/            # 페이지 컴포넌트
│   ├── types/            # TypeScript 타입 정의
│   ├── App.tsx           # 라우팅 설정
│   ├── main.tsx          # 엔트리 포인트
│   └── index.css         # 글로벌 스타일
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## 설치 및 실행

### 1. 의존성 설치

```bash
cd frontend
npm install
```

### 2. 개발 서버 실행

```bash
npm run dev
```

개발 서버가 http://localhost:3000 에서 시작됩니다.

### 3. 프로덕션 빌드

```bash
npm run build
```

## 주요 기능

### 대시보드
- 전체 세그먼트 통계
- 주요 KPI 월별 추이 차트
- 최근 알림 목록
- 빠른 액션 링크

### 세그먼트 조회
- 세그먼트 목록 조회
- 필터링 (업종, 지역, 등급, 전담여부)
- 정렬 및 페이지네이션
- 세그먼트 상세 정보 조회
- KPI 히스토리 차트

### 예측 결과
- KPI별 예측 결과 조회
- 예측 기간별 필터링
- 예측 vs 실제 비교 차트
- 오차율 분석

### 워치리스트
- 알림 유형별 필터링 (급감, 급증, 기회)
- 심각도별 필터링
- 주요 드라이버 표시
- Z-Score 기반 이상치 표시

### 액션 추천
- 우선순위 기반 정렬
- 액션 유형별 분류
- 예상 효과 표시
- 상세 정보 확장 보기

## API 프록시 설정

개발 환경에서 백엔드 API 호출을 위해 Vite 프록시가 설정되어 있습니다.

```typescript
// vite.config.ts
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

## 환경 변수

필요한 경우 `.env` 파일을 생성하여 환경 변수를 설정할 수 있습니다:

```
VITE_API_BASE_URL=http://localhost:8000
```
