# Ops Guide (Automation)

## 1) Full retrain + pipeline + monitoring

```bash
python3 scripts/auto_retrain.py \
  --raw_csv "../외부 데이터/iMbank_data.csv.csv" \
  --external_dir "../외부 데이터" \
  --use_external \
  --epochs 50 \
  --model_strategy lgbm
```

## 2) Accuracy-max configuration (ensemble)

```bash
python3 scripts/auto_retrain.py \
  --raw_csv "../외부 데이터/iMbank_data.csv.csv" \
  --external_dir "../외부 데이터" \
  --use_external \
  --epochs 100 \
  --model_strategy ensemble \
  --ensemble_tune \
  --ensemble_trials 100 \
  --ensemble_timeout 900 \
  --ensemble_fs
```

## 3) Drift monitoring only

```bash
python3 scripts/monitoring_report.py
```
