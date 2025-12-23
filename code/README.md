# iM ONEderful — 통합 세그먼트 레이더 (MVP)

본 저장소는 **세그먼트(업종×지역×등급×전담) 단위** 월별 금융지표를 기반으로,
1) 다음 1~3개월 KPI를 예측하고  
2) 이상징후(급감/급증/변곡)를 조기 탐지하며  
3) 변화 원인을 설명(드라이버)하고  
4) **“이번 달 무엇을 누구에게 먼저 할지”** Next Best Action을 추천하는  
**세그먼트 운영 레이더**를 구현하기 위한 MVP 코드 베이스입니다.

확장으로 **환율 레짐 기반 FX Opportunity & Risk Radar** 모듈을 추가할 수 있도록 구조를 분리했습니다.

---

## 1. 입력 데이터

### 내부 데이터 (필수)
- 월별(예: 2022-01 ~ 2024-12) 법인 고객 데이터 CSV  
- 컬럼 예시는 `기준년월, 업종_중분류, 사업장_시도, 법인_고객등급, 전담고객여부` 등

> 본 프로젝트 MVP 세그먼트 키(권장):  
> `업종_중분류 × 사업장_시도 × 법인_고객등급 × 전담고객여부`

### 외부 데이터 (선택/확장)
- 달력: 영업일/공휴일(월별)
- 거시: 기준금리(월별)
- FX 모듈(권장): USD/KRW 월평균/말일 + 월 변동성(표준편차 또는 고저폭)

외부 데이터는 `data/external/` 아래 CSV로 넣고 조인합니다.

---

## 2. 핵심 KPI (MVP 기본)
- 예금총잔액 = 요구불 + 거치식 + 적립식
- 대출총잔액 = 운전자금 + 시설자금
- 카드총사용 = 신용 + 체크
- 디지털거래금액 = 인터넷 + 스마트 + 폰
- 순유입 = 요구불입금 – 요구불출금

FX(확장)
- FX총액 = 수출 + 수입
- FX성장률 = MoM/YoY 또는 이동평균 대비 편차

---

## 3. 실행 방법 (Quickstart)

### 3.1 환경 설치
```bash
pip install -r requirements.txt
```

### 3.2 MVP 파이프라인 실행
```bash
python scripts/run_mvp.py --raw_csv /path/to/iMbank_data.csv --encoding cp949
```

실행 후 산출물은 `outputs/` 아래에 생성됩니다.
- `outputs/segment_panel.parquet` : 세그먼트×월 패널 데이터
- `outputs/forecasts/*.csv` : KPI별 1~3개월 예측 결과
- `outputs/watchlist_alerts.csv` : 이상징후 워치리스트
- `outputs/actions_top.csv` : Next Best Action TOP 리스트
- `outputs/dashboard/*.html` : 대시보드(정적 HTML)

---

## 4. 폴더 구조
```
im_one_radar/
  src/imradar/               # 재사용 가능한 라이브러리 코드
  scripts/                   # 실행 스크립트 (CLI)
  data/raw/                  # 원천 데이터(사용자 저장)
  data/external/             # 외부 데이터(선택)
  outputs/                   # 모델/예측/리포트 산출물
  docs/                      # 문서, 발표자료 보조
```

---

## 5. 구현 메모
- 예측은 **LightGBM 글로벌 모델**(세그먼트 범주 + lag/rolling/exogenous) 기반으로 구성합니다.
- 이상징후는 **예측 대비 잔차(residual)** 기반으로 스코어링합니다.
- 설명(Drivers)은 `LightGBM.predict(pred_contrib=True)`를 사용해 **로컬 기여도(유사 SHAP)** 를 산출합니다.
- 추천(NBA)은 라벨 부재를 전제로 **룰 기반 후보 생성 + 스코어(기대효과/방어필요/실행제약)** 로 우선순위를 산정합니다.

---

## 6. 라이선스 / 주의
- 본 코드는 교육/프로토타입 목적의 MVP입니다.
- 실제 현업 적용 시 데이터 정의(합산 기준), 정책/규제/보안 요건, 모델 검증/모니터링 체계를 추가해야 합니다.
