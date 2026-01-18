# 발표/시연 가이드 (Segment Radar MVP)

> 최종 업데이트: 2025-01-19

이 문서는 **통합 세그먼트 레이더**를 5~10분 내로 시연하기 위한 진행 스크립트입니다.

---

## 🏆 주요 성과 요약 (시연 전 언급)

| 항목 | 수치 |
|------|------|
| 예측 정확도 | R² 81~86% (순유입 제외) |
| 세그먼트 수 | 2,038개 |
| 추천 액션 | 2,582개 (6개 유형) |
| Watchlist 알림 | 44개 (21개 업종, 4개 severity) |
| 대시보드 | 6개 페이지 |

---

## 1) 핵심 메시지(30초)
- 현업은 월간 실적 확인 후 **사후 대응**이 많아 타이밍을 놓치기 쉽습니다.
- 본 MVP는 세그먼트(업종×지역×등급×전담) 단위로 **예측 → 조기경보 → 원인설명 → 액션추천**을 한 흐름으로 통합합니다.
- 확장으로 **환율 레짐 기반 FX Opportunity & Risk Radar**를 제공해 희소하지만 고가치(특히 최우수 등급/제조업 중심) 세그먼트를 선제 관리합니다.

## 2) 데모 준비(사전)
1. 내부 CSV 준비: `../외부 데이터/iMbank_data.csv.csv`
2. 외부 데이터 준비: `../외부 데이터/` 폴더에 거시경제 데이터 배치
3. 실행
```bash
cd code
python scripts/run_mvp.py \
    --raw_csv "../외부 데이터/iMbank_data.csv.csv" \
    --external_dir "../외부 데이터"
```

## 3) 데모 흐름(추천)
### STEP A. Overview(전체 전망)
- `outputs/dashboard/01_overview.html`
- "전체 KPI가 다음 1~3개월에 어떻게 움직일지(실적 vs 예측)"를 보여주며, 자원배분 필요성을 제시합니다.

### STEP B. Growth Forecast(기회 타깃)
- `outputs/dashboard/02_growth_forecast.html`
- 예측 상승 TOP 세그먼트를 보여주고, "이번 달 먼저 공략할 세그먼트"를 제시합니다.

### STEP C. Risk Radar(워치리스트)
- `outputs/dashboard/03_risk_watchlist.html`
- **44개 알림, 21개 업종, 4개 severity 레벨**로 리스크 관리
- `outputs/watchlist_drivers.csv`로 "왜 흔들렸는지" 드라이버 상위 요인을 1~2줄로 설명합니다.

### STEP D. Actions(Next Best Action)
- `outputs/dashboard/04_actions.html`
- **2,582개 액션, 6개 유형**:
  - DEPOSIT_GROWTH (1,238개) - 예금 성장 기회
  - LIQUIDITY_STRESS (669개) - 유동성 리스크
  - ENGAGEMENT_DROP (290개) - 참여도 하락
  - GROWTH_OPPORTUNITY (221개) - 성장 기회
  - FX_EXPANSION (117개) - FX 확장 기회
  - CREDIT_RISK (47개) - 신용 리스크

### STEP E. (옵션) FX Radar 스토리
- `outputs/dashboard/05_fx_radar.html`
- FX 활동은 데이터에서 **약 5.21%**로 희소하지만, 제조업/최우수 등급에 집중되어 "타깃 풀이 좁고 명확"합니다.
- 환율 레짐을 붙이면 "지금은 변동성 확대 국면이라 리스크관리 메시지가 타당" 같은 도메인 설명력이 생깁니다.

## 4) 시연 후 Q&A 대비 포인트
- **왜 세그먼트?** (고객 단위보다 자원배분과 운영 표준화에 적합)
- **왜 residual 경보?** (z-score보다 운영 친화적, 예측 대비 이탈을 바로 보여줌)
- **추천은 왜 룰+스코어?** (라벨 부재 상황의 현실적 접근, 추후 캠페인 반응 데이터로 고도화 가능)
- **순유입 R²가 낮은 이유?** (양/음 값이 섞인 변동성 높은 지표 특성, 딥러닝 모델로 개선 가능)

---

*문서 작성: iM-ONEderful Team*
