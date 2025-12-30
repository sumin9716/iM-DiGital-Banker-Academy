# 발표/시연 가이드 (Segment Radar MVP)

이 문서는 **통합 세그먼트 레이더**를 5~10분 내로 시연하기 위한 진행 스크립트입니다.

## 1) 핵심 메시지(30초)
- 현업은 월간 실적 확인 후 **사후 대응**이 많아 타이밍을 놓치기 쉽습니다.
- 본 MVP는 세그먼트(업종×지역×등급×전담) 단위로 **예측 → 조기경보 → 원인설명 → 액션추천**을 한 흐름으로 통합합니다.
- 확장으로 **환율 레짐 기반 FX Opportunity & Risk Radar**를 제공해 희소하지만 고가치(특히 최우수 등급/제조업 중심) 세그먼트를 선제 관리합니다.

## 2) 데모 준비(사전)
1. 내부 CSV 준비: `data/raw/iMbank_data.csv` (경로는 자유)
2. (선택) 외부 데이터 준비
   - 환율 템플릿: `data/external/usdkrw_monthly_template.csv`를 복사해 `usdkrw_monthly.csv`로 저장
   - 달력 템플릿: `data/external/calendar_monthly_template.csv`를 복사해 `calendar_monthly.csv`로 저장
3. 실행
```bash
python scripts/run_mvp.py --raw_csv data/raw/iMbank_data.csv --encoding cp949
```

## 3) 데모 흐름(추천)
### STEP A. Overview(전체 전망)
- `outputs/dashboard/01_overview.html`
- “전체 KPI가 다음 1~3개월에 어떻게 움직일지(실적 vs 예측)”를 보여주며, 자원배분 필요성을 제시합니다.

### STEP B. Growth Forecast(기회 타깃)
- `outputs/dashboard/02_growth_forecast.html`
- 예측 상승 TOP 세그먼트를 보여주고, “이번 달 먼저 공략할 세그먼트”를 제시합니다.

### STEP C. Risk Radar(워치리스트)
- `outputs/dashboard/03_risk_watchlist.html`
- 예측 대비 하회/상회(잔차)로 경보를 만들고, 급감/급증/변곡 유형을 보여줍니다.
- `outputs/watchlist_drivers.csv`로 “왜 흔들렸는지” 드라이버 상위 요인을 1~2줄로 설명합니다.

### STEP D. Actions(Next Best Action)
- `outputs/actions_top.csv`
- 라벨 부재 환경에서 **룰 기반 후보 생성 + 스코어링(기대효과/방어필요/실행제약)**로 우선순위를 만든다는 점을 강조합니다.

### STEP E. (옵션) FX Radar 스토리
- FX 활동은 데이터에서 **약 5.21%**로 희소하지만, 제조업/최우수 등급에 집중되어 “타깃 풀이 좁고 명확”합니다.
- 환율 레짐을 붙이면 “지금은 변동성 확대 국면이라 리스크관리 메시지가 타당” 같은 도메인 설명력이 생깁니다.

## 4) 시연 후 Q&A 대비 포인트
- 왜 세그먼트? (고객 단위보다 자원배분과 운영 표준화에 적합)
- 왜 residual 경보? (z-score보다 운영 친화적, 예측 대비 이탈을 바로 보여줌)
- 추천은 왜 룰+스코어? (라벨 부재 상황의 현실적 접근, 추후 캠페인 반응 데이터로 고도화 가능)
