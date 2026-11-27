# 세그먼트 레이더 파이프라인 (code/)

프로젝트 소개, 결과, 팀 구성은 [루트 README](../README.md)에 있습니다. 이 문서는 `code/` 아래의 코드를 설치하고 실행하는 방법, 산출물, 모듈 구성을 설명합니다. 아래 명령은 모두 `code/` 폴더에서 실행합니다.

## 설치

Python 3.10 이상이 필요합니다(3.11에서 확인).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt        # 기본 패키지 + pytest, ruff
pip install -r requirements-optional.txt   # 선택: 앙상블(XGBoost/CatBoost/Optuna), 딥러닝(PyTorch 등)
```

| 파일 | 용도 |
|------|------|
| `requirements.txt` | 기본 파이프라인(`run_mvp.py`, `evaluate_models.py`, `monitoring_report.py`, `advanced_features.py`) |
| `requirements-optional.txt` | `--model_strategy ensemble/auto`, `advanced_model_improvements.py`, `deep_learning_models.py`, `transformer_tuning.py` |
| `requirements-dev.txt` | 테스트(pytest)와 정적 검사(ruff) |

저장된 모델(`outputs/models/*.pkl`)은 numpy 2.x로 직렬화되어 있어 numpy 2.0 이상이 필요합니다.

## 데이터 준비

은행이 제공한 원본과 외부 거시 지표는 저장소에 없습니다.

- 내부 데이터: 과정에서 받은 `iMbank_data.csv.csv`(cp949). 고객 단위 월별 행이며, `기준년월`(YYYYMM), 세그먼트 키 4개(`업종_중분류`, `사업장_시도`, `법인_고객등급`, `전담고객여부`), 잔액·거래 금액 컬럼, `"2건초과 5건이하"` 같은 구간형 건수 컬럼이 있어야 합니다. 컬럼 목록은 `src/imradar/config.py`의 `RadarConfig`에 있습니다.
- 외부 데이터(선택): `RadarConfig.external_data_files`에 적힌 CSV(`ESI.csv`, `fed_금리_파생변수_202106_202506.csv`, `usd_krw_raw_daily.csv`, `원자재(유가)_파생변수.csv`, `무역_P1_파생변수.csv`, `KASI.csv`, `P1_금리.csv`)를 한 폴더에 둡니다. `--external_dir`를 주지 않고 `--use_external`만 주면 저장소 루트의 `외부 데이터/`, `code/data/external/` 순으로 찾습니다.

## 실행

### 전체 파이프라인

```bash
python scripts/run_mvp.py --raw_csv <iMbank_data.csv.csv 경로> --external_dir <외부 데이터 폴더>
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--raw_csv` | (필수) | 내부 원본 CSV |
| `--encoding` | `cp949` | 원본 인코딩. 실패하면 euc-kr, utf-8-sig, utf-8 순으로 다시 시도 |
| `--out_dir` | `outputs` | 산출물 폴더 |
| `--external_dir` / `--use_external` | 없음 | 거시 지표 사용 여부와 폴더 |
| `--test_start`, `--test_end` | `202401`, `202412` | 평가 구간(YYYYMM). `--train_end`를 안 주면 학습은 `test_start` 직전 달까지 |
| `--skip_train` | 꺼짐 | 학습하지 않고 `outputs/models/`의 모델을 불러옴 |
| `--model_strategy` | `lgbm` | `ensemble`, `auto`는 선택 의존성이 필요하며 없으면 `lgbm`으로 대체 |

단계는 원본 로드 → 세그먼트-월 집계 → 전체 패널 → KPI 계산 → (거시 지표 병합) → lag·rolling·변화율 피처 → KPI·예측 시점별 LightGBM 학습과 평가 → 다음 1~3개월 예측 → 잔차 경보와 워치리스트 → 피처 기여도 설명 → 액션 추천 → 대시보드·PDF 순입니다.

### 원본 없이 실행할 수 있는 스크립트

```bash
pytest                                   # 단위 테스트 (합성 데이터)
ruff check .                             # 정적 검사 (설정: ruff.toml)
python scripts/evaluate_models.py        # outputs/models/*.pkl을 커밋된 패널로 재평가 → outputs/model_evaluation.csv
python scripts/monitoring_report.py      # 마지막 달과 직전 6개월의 PSI 비교 → outputs/monitoring/
python scripts/advanced_features.py      # XAI 대상 피처 목록, CI·시나리오 설정, 세그먼트 클러스터링 → outputs/advanced_features/
```

### 그 밖의 스크립트

| 스크립트 | 필요 조건 | 내용 |
|----------|-----------|------|
| `scripts/auto_retrain.py` | 원본, 외부 데이터, PyTorch | 딥러닝 재학습 → `run_mvp.py` → `evaluate_models.py` → `monitoring_report.py`를 차례로 실행. 예시는 `docs/OPS_GUIDE.md` |
| `scripts/deep_learning_models.py` | 원본, 외부 데이터, PyTorch | 순유입 등 KPI에 대한 LSTM·Transformer·TFT 실험 → `outputs/deep_learning/` |
| `scripts/transformer_tuning.py` | 원본, 외부 데이터, PyTorch, Optuna | 데이터 증강과 Transformer 하이퍼파라미터 탐색 → `outputs/transformer_tuning/` |
| `scripts/advanced_model_improvements.py` | 원본, 외부 데이터, 앙상블 패키지 | 순유입 2단계 모델, 스태킹 앙상블 실험 → `outputs/advanced_models/` |

## 산출물

`*.csv`는 `.gitignore`에 걸려 있어 커밋되지 않습니다. 저장소에는 아래 파일만 들어 있습니다.

| 경로 | 만드는 코드 | 내용 |
|------|-------------|------|
| `outputs/segment_panel.parquet` | `run_mvp.py` | 세그먼트 2,038개 × 36개월 피처 패널 |
| `outputs/models/lgbm_{KPI}_h{1,2,3}.pkl` | `run_mvp.py` | KPI 6개 × 예측 시점 3개 LightGBM 모델 |
| `outputs/dashboard/00~03, 05` | `run_mvp.py` | 경영 요약, 전체 현황, 성장 예측, 리스크 워치리스트, FX 레이더 |
| `outputs/dashboard/06_performance_summary.html` | `imradar.pipelines.performance_dashboard` | 모델 성과 종합 |
| `outputs/dashboard/04_actions.html`, `index.html` | 저장소에 생성 코드 없음 | 액션 목록, 대시보드 목차 |
| `outputs/monthly_onepager_202412.pdf` | `run_mvp.py` | 월간 1페이지 리포트 |
| `outputs/monitoring/monitoring_summary.json` | `monitoring_report.py` | PSI 요약 |
| `outputs/advanced_features/*.json` | `advanced_features.py` | XAI 대상 피처, CI 설정, 시나리오 가정, 외부 지표 최신값 |
| `outputs/deep_learning/*.pkl`, `outputs/transformer_tuning/transformer_tuned.pt` | 딥러닝 스크립트 | 스케일러·메타데이터, 튜닝된 Transformer 가중치 |

`run_mvp.py`가 만들지만 커밋되지 않는 파일: `forecast_metrics.csv`, `forecasts/segment_forecasts_next.csv`, `watchlist_alerts.csv`, `watchlist_drivers.csv`, `actions_top.csv`, `actions_enhanced.csv`, `actions_with_similarity.csv`, `segment_similarity_report.csv`, `fx_segment_scores.csv`(외부 환율 자료가 있을 때).

## 모듈 구성

```
src/imradar/
├── config.py               # RadarConfig: 컬럼, 세그먼트 키, KPI 정의, 하이퍼파라미터, 외부 파일 이름
├── utils.py                # 날짜 변환, log1p 변환, 구간 문자열 파싱
├── data/
│   ├── io.py               # 인코딩을 바꿔 가며 CSV 읽기
│   ├── preprocess.py       # 세그먼트-월 집계, 전체 패널 구축
│   └── external.py         # 거시 지표 로더와 패널 병합, 거시 lag 피처
├── features/
│   ├── kpi.py              # KPI·파생 지표, 학습 대상 컬럼(TARGET_COLS)과 역변환
│   └── lags.py             # lag, rolling, 변화율 피처
├── models/
│   ├── forecast_lgbm.py    # LightGBM 학습·예측·평가 (ForecastModel)
│   ├── forecast_ensemble.py# LightGBM+XGBoost+CatBoost 앙상블 (선택 의존성)
│   ├── forecast_transformer.py # LSTM/TFT 예측 (PyTorch 필요)
│   ├── residual_alerts.py  # 잔차 경보, 워치리스트
│   ├── hierarchical.py     # 계층 예측 조정
│   ├── fx_regime.py        # 환율 국면, FX 기회·리스크 점수
│   ├── risk_radar.py       # z-score 기반 리스크 점수
│   └── backtest.py         # 롤링 백테스트 유틸
├── explain/lgbm_contrib.py # 피처 기여도와 설명 문장
├── recommend/              # 규칙 기반 액션(rules, enhanced_rules), 원인 요약, 유사 세그먼트
├── pipelines/              # HTML 대시보드, 경영 요약, 성과 종합, PDF 리포트
└── ops/monitoring.py       # PSI 드리프트
```

## 개발

- 커밋 전에 `pytest`와 `ruff check .`를 실행합니다.
- `outputs/models/*.pkl`은 `imradar.models.forecast_lgbm.ForecastModel`을 pickle로 저장한 것입니다. 이 클래스의 이름, 모듈 경로, 필드를 바꾸면 저장된 모델을 불러올 수 없습니다.
- 학습 대상 컬럼과 역변환은 `features/kpi.py`의 `TARGET_COLS`, `inverse_target_transform`만 고칩니다. 스크립트에 따로 복사하지 않습니다.
- XGBoost, CatBoost, Optuna, PyTorch는 선택 의존성입니다. 이 패키지를 쓰는 모듈은 import 시점에 실패하지 않고, 실제로 학습할 때 설치 안내와 함께 `ImportError`를 냅니다(`forecast_ensemble.py`의 `_require_ensemble_deps` 참고).
- 원본 데이터와 파생 CSV는 커밋하지 않습니다. `run_mvp.py`를 고쳤다면 원본과 같은 컬럼 구조의 합성 CSV로 수정 전후 산출물을 비교해 보는 것이 안전합니다.
- `code/.gitignore`는 `outputs/segment_panel.parquet`와 `outputs/models/`를 무시 대상으로 적고 있지만 두 경로는 이미 커밋되어 있어 계속 추적됩니다. 새로 학습한 모델을 커밋할지는 데이터 공개 범위에 따라 정합니다.

## 문서

| 문서 | 내용 |
|------|------|
| `docs/PROJECT_SPEC.md` | 문제 정의, KPI, 엔진 구성 요약 |
| `docs/MODEL_SUMMARY.md` | 발표 당시 모델 성능, 피처, 하이퍼파라미터 정리 |
| `docs/DEMO_GUIDE.md`, `docs/PRESENTATION_GUIDE.md`, `docs/PRESENTATION_OUTLINE.md` | 시연·발표 진행 자료 |
| `docs/OPS_GUIDE.md` | `auto_retrain.py`, `monitoring_report.py` 실행 예시 |
| `docs/월간_1페이지_리포트_템플릿_v3(1page).docx/.pdf` | 월간 운영 리포트 양식 |

발표 자료의 성능 수치는 발표 당시 기록이며, 커밋된 모델로 다시 계산한 값은 루트 README의 결과 표를 참고합니다.
