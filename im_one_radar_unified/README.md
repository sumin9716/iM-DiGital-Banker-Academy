<div align="center">
  <img src="./img/iM_DiGital_Banker.png" width="600"/>
</div>

<br/>

# 📊 iM ONEderful Segment Radar - 세그먼트 기반 KPI 예측 및 조기경보 플랫폼

<p align="center">
  <b>"세그먼트 단위 예측 → 조기경보 → 원인설명 → 액션추천"</b>을 한 흐름으로 통합하는<br/>
  금융 세그먼트 운영 레이더 플랫폼입니다.
</p>

<br/>

---

## 📑 목차

- [프로젝트 소개](#-프로젝트-소개)
  - [💡 프로젝트를 왜 시작하게 되었나요?](#-프로젝트를-왜-시작하게-되었나요)
  - [🔑 프로젝트의 핵심은 무엇인가요?](#-프로젝트의-핵심은-무엇인가요)
  - [🎁 프로젝트가 가져올 수 있는 기대 효과는 무엇인가요?](#-프로젝트가-가져올-수-있는-기대-효과는-무엇인가요)
- [팀 소개](#-팀-소개)
- [개발 기간](#-개발-기간)
- [기술 스택](#-기술-스택)
- [시스템 아키텍처](#-시스템-아키텍처)
- [ERD / 데이터 구조](#-erd--데이터-구조)
- [API 명세](#-api-명세)
- [핵심 기능 소개](#-핵심-기능-소개)
- [실행 방법](#-실행-방법)
- [프로젝트 구조](#-프로젝트-구조)

---

## 🏦 프로젝트 소개

**iM ONEderful Segment Radar**는 **세그먼트(업종×지역×등급×전담)** 단위의 월별 금융지표를 분석하여,
다음 1~3개월 KPI 예측, 이상징후 조기 탐지, 변화 원인 설명, 그리고 Next Best Action 추천까지
원스톱으로 제공하는 **세그먼트 운영 레이더** 플랫폼입니다.

<br/>

### 💡 프로젝트를 왜 시작하게 되었나요?

<div align="center">
  <img src="./img/gantt.png" width="700"/>
</div>

<br/>

금융기관의 현업 담당자들은 월간 실적을 확인한 후 **사후 대응**하는 경우가 많습니다. 
이로 인해 급격한 자금 유출, 연체 증가, 거래 이탈 등의 이상 징후를 사전에 감지하기 어려워 
타이밍을 놓치기 쉽습니다.

또한, 수많은 세그먼트(업종×지역×등급) 조합을 일일이 분석하고 우선순위를 정하는 것은 
시간과 리소스 측면에서 매우 비효율적입니다.

<br/>

### 🔑 프로젝트의 핵심은 무엇인가요?

<table align="center">
  <tr>
    <td align="center"><b>🔮 예측</b></td>
    <td align="center"><b>🚨 조기경보</b></td>
    <td align="center"><b>🔍 원인설명</b></td>
    <td align="center"><b>🎯 액션추천</b></td>
  </tr>
  <tr>
    <td align="center">1~3개월 ahead<br/>KPI 예측</td>
    <td align="center">급감/급증/변곡<br/>이상징후 탐지</td>
    <td align="center">드라이버 분석으로<br/>변화 원인 파악</td>
    <td align="center">Next Best Action<br/>우선순위 추천</td>
  </tr>
</table>

<br/>

핵심은 **"이번 달 무엇을 누구에게 먼저 할지"**를 데이터 기반으로 제시하는 것입니다.
LightGBM 기반 예측 모델과 잔차 분석을 통해 사전 대응이 가능한 인사이트를 제공합니다.

<br/>

### 🎁 프로젝트가 가져올 수 있는 기대 효과는 무엇인가요?

- **현업 담당자 측면**
  - 월 단위 사후 분석 → **선제적 대응** 체계로 전환
  - 세그먼트별 자동화된 우선순위로 **업무 효율성** 향상
  - 변화 원인(드라이버)을 즉시 파악하여 **의사결정 속도** 개선

- **은행 측면**
  - 조기 경보를 통한 **리스크 관리** 강화
  - 데이터 기반 세그먼트 전략으로 **영업 효율** 극대화
  - 예측 결과를 활용한 **자원 배분 최적화**

---

## 👨🏻‍💻 팀 소개

### 👋 iM ONEderful 👋

> iM뱅크 디지털 금융 아카데미 프로젝트 팀

<table align="center">
    <tr>
        <td align="center"><b>김수민</b></td>
        <td align="center"><b>나효상</b></td>
        <td align="center"><b>배수원</b></td>
        <td align="center"><b>서범창</b></td>    
    </tr>
    <tr height="160px">
        <td align="center">
            <a href="https://github.com/sumin9716">
                <img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/221969214?v=4"/>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/TimePise">
                <img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/131345223?v=4"/>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/">
                <img height="120px" width="120px" src="https://via.placeholder.com/120"/>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/SeoBamm">
                <img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/70587328?v=4"/>
            </a>
        </td>
    </tr>
    <tr>
        <td align="center">ML 파이프라인<br/>Backend</td>
        <td align="center">데이터 전처리<br/>Feature Engineering</td>
        <td align="center">모델링<br/>분석</td>
        <td align="center">Frontend<br/>대시보드</td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/sumin9716">@sumin9716</a></td>
        <td align="center"><a href="https://github.com/TimePise">@TimePise</a></td>
        <td align="center"><a href="https://github.com/">@Github</a></td>
        <td align="center"><a href="https://github.com/SeoBamm">@SeoBamm</a></td>
    </tr>
</table>

---

## 📅 개발 기간

| 단계 | 기간 |
|------|------|
| 프로젝트 기획 및 설계 | 2024.12.23 ~ 2024.12.27 |
| 데이터 분석 및 모델링 | 2024.12.28 ~ 2025.01.10 |
| Backend/Frontend 개발 | 2025.01.11 ~ 2025.01.20 |
| 통합 테스트 및 배포 | 2025.01.21 ~ 2025.01.22 |
| **최종 발표** | **2025.01.22** |

---

## 🛠 기술 스택

<table align="center">
  <tr>
    <th>구분</th>
    <th>기술</th>
  </tr>
  <tr>
    <td><b>Language</b></td>
    <td>
      <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
      <img src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=TypeScript&logoColor=white"/>
    </td>
  </tr>
  <tr>
    <td><b>Backend</b></td>
    <td>
      <img src="https://img.shields.io/badge/Django-092E20?style=flat-square&logo=Django&logoColor=white"/>
      <img src="https://img.shields.io/badge/Django_REST_Framework-092E20?style=flat-square&logo=Django&logoColor=white"/>
    </td>
  </tr>
  <tr>
    <td><b>Frontend</b></td>
    <td>
      <img src="https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=React&logoColor=black"/>
      <img src="https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=Vite&logoColor=white"/>
      <img src="https://img.shields.io/badge/TailwindCSS-06B6D4?style=flat-square&logo=TailwindCSS&logoColor=white"/>
    </td>
  </tr>
  <tr>
    <td><b>ML/Data</b></td>
    <td>
      <img src="https://img.shields.io/badge/LightGBM-02569B?style=flat-square&logo=&logoColor=white"/>
      <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=Pandas&logoColor=white"/>
      <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
    </td>
  </tr>
  <tr>
    <td><b>Visualization</b></td>
    <td>
      <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=Plotly&logoColor=white"/>
      <img src="https://img.shields.io/badge/Recharts-FF6B6B?style=flat-square&logo=&logoColor=white"/>
    </td>
  </tr>
  <tr>
    <td><b>State Management</b></td>
    <td>
      <img src="https://img.shields.io/badge/TanStack_Query-FF4154?style=flat-square&logo=ReactQuery&logoColor=white"/>
    </td>
  </tr>
  <tr>
    <td><b>Tools</b></td>
    <td>
      <img src="https://img.shields.io/badge/VS_Code-007ACC?style=flat-square&logo=VisualStudioCode&logoColor=white"/>
      <img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=Git&logoColor=white"/>
      <img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/>
    </td>
  </tr>
</table>

---

## 🏗 시스템 아키텍처

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           iM ONEderful Segment Radar                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐     HTTP/REST     ┌─────────────────┐              │
│  │    Frontend     │ ◄───────────────► │     Backend     │              │
│  │  React + Vite   │                   │     Django      │              │
│  │  TailwindCSS    │                   │  REST Framework │              │
│  │  (Port 3000)    │                   │   (Port 8000)   │              │
│  └─────────────────┘                   └────────┬────────┘              │
│                                                 │                       │
│                                                 ▼                       │
│                                        ┌─────────────────┐              │
│                                        │   ML Pipeline   │              │
│                                        │    (imradar)    │              │
│                                        └────────┬────────┘              │
│                                                 │                       │
│         ┌───────────────────────────────────────┼───────────────┐       │
│         ▼                   ▼                   ▼               ▼       │
│  ┌────────────┐     ┌─────────────┐     ┌────────────┐   ┌──────────┐  │
│  │ Forecasting│     │ Risk Radar  │     │  Explain   │   │   NBA    │  │
│  │ (LightGBM) │     │ (Residual)  │     │ (Contrib)  │   │ (Rules)  │  │
│  └────────────┘     └─────────────┘     └────────────┘   └──────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### ML 파이프라인 아키텍처

```
[내부 데이터] + [외부 데이터(환율/달력)]
        │
        ▼
┌──────────────────────────────────┐
│   전처리 & 세그먼트 패널 구축    │
│  - 결측치 처리, 타입 변환        │
│  - 업종×지역×등급×전담 그룹핑    │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│   피처 엔지니어링                │
│  - Lag Features (1~3개월)        │
│  - Rolling Statistics            │
│  - Change Rate / Momentum        │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│   Forecasting Engine (LightGBM)  │
│  - 1/2/3개월 ahead 예측          │
│  - 글로벌 모델 + 세그먼트 범주   │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│   Risk Radar (Residual Alerts)   │
│  - 예측 vs 실적 잔차 기반 경보   │
│  - 급감/급증/변곡 탐지           │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│   Explain (Feature Contribution) │
│  - 로컬 드라이버 추출            │
│  - SHAP-like 기여도 분석         │
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│   NBA (Rule + Score)             │
│  - 라벨 없이 우선순위 산정       │
│  - 액션 유형별 추천              │
└──────────────────────────────────┘
        │
        ▼
[대시보드] + [월간 리포트]
```

---

## 📊 ERD / 데이터 구조

### 세그먼트 패널 데이터

| 필드 | 설명 | 타입 |
|------|------|------|
| `segment_id` | 세그먼트 고유 ID | VARCHAR |
| `ym` | 기준년월 (YYYY-MM) | DATE |
| `업종대` | 업종 대분류 | VARCHAR |
| `RM조직` | 지역 조직 | VARCHAR |
| `기업등급` | 신용등급 | VARCHAR |
| `전담여부` | 전담 고객 여부 | BOOLEAN |
| `예금총잔액` | 예금 합계 | NUMERIC |
| `대출총잔액` | 대출 합계 | NUMERIC |
| `카드총사용` | 카드 사용액 | NUMERIC |
| `디지털거래금액` | 디지털 채널 거래액 | NUMERIC |
| `순유입` | 순 자금 유입액 | NUMERIC |

### 핵심 KPI

| KPI | 구성 | 설명 |
|-----|------|------|
| 예금총잔액 | 요구불 + 거치식 + 적립식 | 총 예금 규모 |
| 대출총잔액 | 운전자금 + 시설자금 | 총 대출 규모 |
| 카드총사용 | 신용 + 체크 | 카드 사용 현황 |
| 디지털거래금액 | 인터넷 + 스마트 + 폰뱅킹 | 디지털 활성도 |
| 순유입 | 입금 – 출금 | 자금 흐름 |

---

## 📋 API 명세

### 주요 엔드포인트

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/overview/` | 대시보드 개요 통계 |
| `GET` | `/api/segments/` | 세그먼트 목록 조회 |
| `GET` | `/api/segments/{id}/` | 세그먼트 상세 정보 |
| `GET` | `/api/segments/{id}/history/` | 세그먼트 히스토리 |
| `GET` | `/api/forecasts/` | KPI 예측 결과 |
| `GET` | `/api/watchlist/` | 워치리스트 알림 |
| `GET` | `/api/recommendations/` | 액션 추천 목록 |
| `GET` | `/api/filters/` | 필터 옵션 목록 |
| `GET` | `/api/kpi-trends/` | KPI 트렌드 데이터 |

### Response 예시

```json
// GET /api/overview/
{
  "total_segments": 245,
  "total_deposits": 1234567890000,
  "total_loans": 987654321000,
  "alert_count": 12,
  "growth_segments": 45,
  "risk_segments": 8
}
```

---

## ✨ 핵심 기능 소개

### 📈 KPI 예측 (Forecasting)

- **세그먼트별 1~3개월 ahead 예측**
  - LightGBM 기반 글로벌 모델
  - Lag/Rolling/Change Rate 피처 활용
  - 예측 신뢰구간 제공

### 🚨 조기경보 (Risk Radar)

- **이상징후 자동 탐지**
  - 잔차(Residual) 기반 알림 생성
  - 급감/급증/변곡점 탐지
  - 심각도별 알림 등급 분류

### 🔍 드라이버 분석 (Explain)

- **변화 원인 자동 추출**
  - Feature Contribution 분석
  - Top-K 드라이버 리스트
  - 직관적인 설명 제공

### 🎯 액션 추천 (Next Best Action)

- **우선순위 기반 추천**
  - 룰 기반 후보 생성
  - 스코어링으로 우선순위 결정
  - 액션 유형별 분류

---

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/sumin9716/iM-DiGital-Banker-Academy.git
cd iM-DiGital-Banker-Academy/im_one_radar_unified

# Python 의존성 설치
pip install -r requirements.txt
```

### 2. MVP 파이프라인 실행

```bash
# 데이터 처리 및 모델 학습
python scripts/run_mvp.py --raw_csv iMbank_data.csv.csv --encoding cp949
```

### 3. Backend 서버 실행

```bash
cd backend

# Django 마이그레이션
python manage.py migrate

# 서버 실행
python manage.py runserver 8000
```

- API: http://localhost:8000/api/

### 4. Frontend 서버 실행

```bash
cd frontend

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

- 대시보드: http://localhost:3000

### 전체 실행 순서 요약

```bash
# 1. MVP 파이프라인 (데이터 생성)
python scripts/run_mvp.py --raw_csv iMbank_data.csv.csv --encoding cp949

# 2. Backend 서버 (새 터미널)
cd backend && python manage.py migrate && python manage.py runserver 8000

# 3. Frontend 서버 (새 터미널)
cd frontend && npm install && npm run dev
```

---

## 📂 프로젝트 구조

```
im_one_radar_unified/
├── 📄 README.md                 # 프로젝트 문서
├── 📄 requirements.txt          # Python 의존성
│
├── 📁 src/imradar/              # 🔧 핵심 ML 라이브러리
│   ├── config.py                # 설정 (RadarConfig)
│   ├── utils.py                 # 유틸리티 함수
│   ├── 📁 data/                 # 데이터 로딩 & 전처리
│   ├── 📁 features/             # 피처 엔지니어링
│   ├── 📁 models/               # 예측 & 경보 모델
│   ├── 📁 explain/              # 드라이버 설명
│   ├── 📁 recommend/            # Next Best Action
│   └── 📁 pipelines/            # 대시보드 & 리포트
│
├── 📁 backend/                  # 🖥️ Django Backend
│   ├── manage.py
│   ├── 📁 radar_api/            # Django 프로젝트 설정
│   └── 📁 api/                  # REST API 앱
│
├── 📁 frontend/                 # 🎨 React Frontend
│   ├── 📁 src/
│   │   ├── 📁 components/       # UI 컴포넌트
│   │   ├── 📁 pages/            # 페이지 컴포넌트
│   │   ├── 📁 api/              # API 클라이언트
│   │   └── 📁 types/            # TypeScript 타입
│   └── package.json
│
├── 📁 scripts/                  # 실행 스크립트
│   └── run_mvp.py
│
├── 📁 data/                     # 데이터
│   ├── 📁 raw/                  # 원천 데이터
│   └── 📁 external/             # 외부 데이터
│
├── 📁 outputs/                  # 산출물
│   ├── 📁 dashboard/            # HTML 대시보드
│   ├── 📁 forecasts/            # 예측 결과
│   └── 📁 models/               # 저장된 모델
│
├── 📁 docs/                     # 문서
│   ├── PROJECT_SPEC.md
│   ├── DEMO_GUIDE.md
│   └── PRESENTATION_OUTLINE.md
│
└── 📁 img/                      # 이미지
```

---

## 📜 라이선스

본 프로젝트는 **iM뱅크 디지털 금융 아카데미** 교육 과정의 일환으로 제작되었습니다.

교육/프로토타입 목적으로 제작되었으며, 실제 현업 적용 시 추가적인 검증이 필요합니다.

---

<div align="center">
  <b>Made with ❤️ by iM ONEderful Team</b>
</div>
