# Architecture: Flat/Variance Transaction Detector

## 1. Problem

**What:** Detect "stuck" data feeds where prices stop changing.

**Why:** Stuck feeds corrupt downstream models and risk calculations.

**How:** Flag assets when rolling standard deviation falls below threshold.

**Success Metric:** Precision (minimize false alarms). Secondary: detection latency.

**Not Building:** ML models, real-time streaming, distributed systems, APIs.

## 2. Pipeline

```
CSV → Validate → Rolling Std → Threshold → Flags → Review → Iterate
```

1. **CSV:** Daily prices: date, ticker, close_price
2. **Validate:** Check schema, missing data, price ranges
3. **Rolling Std:** 5-day window per asset
4. **Threshold:** If std < 0.01 → flag = 1
5. **Flags:** Save to CSV with asset, date, flag
6. **Review:** Analyst marks false positives
7. **Iterate:** Tune window/threshold based on review

## 3. Data

**Assumptions:**
- CSV with columns: date, ticker, close_price
- 10 tickers, 2 years daily
- Prices in dollars (not pennies)

**Storage:**
- `data/raw/` - input CSVs
- `data/processed/` - output flags
- `experiments/run_N/` - per-run results

**Validation:**
- Check expected columns exist
- No negative/zero prices
- Missing data <5% per ticker

## 4. Features

**One feature:** rolling standard deviation of close_price.

```python
# Complete feature engineering
rolling_std = df.groupby('ticker')['close_price'].rolling(5).std()
flags = (rolling_std < 0.01).astype(int)
```

**Why:** Flat data has zero variance. Variance measured by std dev.

## 5. Model

**Not a model.** This is a rule:

```
if rolling_std < threshold:
    flag = 1
else:
    flag = 0
```

**Parameters to tune:** window size [3,5,7,10], threshold [0.005,0.01,0.02,0.05]

## 6. Validation

**Split:** First 18 months tune, last 6 months evaluate.

**Tuning:** Grid search parameters to maximize precision on validation set.

**Reproducibility:** Save config JSON per run.

```json
{
  "window": 5,
  "threshold": 0.01,
  "min_periods": 3,
  "validation_start": "2023-07-01"
}
```

## 7. Evaluation

**Metrics:**
- Precision = true flags / total flags
- Latency = days from flat start to first flag

**Error Analysis:**
- Review each flag: "Was this actually flat?"
- Log false positives
- Look for patterns (low-price assets, specific dates)

## 8. Experiment Tracking

```
experiments/
├── run_001/
│   ├── config.json
│   ├── flags.csv
│   └── metrics.txt
├── run_002/
└── experiment_log.csv
```

**metrics.txt example:**
```
precision: 0.92
latency_days: 2.3
false_positives: 3
notes: T-bills flagged, need adjusted threshold
```

## 9. Inference

**Batch only:** Run daily after market close.

```bash
python run.py --config configs/run_001.json
```

Output: `experiments/run_001/flags.csv`

Analyst reviews CSV directly. No dashboards, no APIs.

## 10. Iteration

**Loop:**
1. Run detector
2. Review false positives
3. Adjust threshold/window
4. New run, compare metrics

**Next:**
- Multi-window voting (3,5,7 day windows)
- Price normalization if needed
- That's it.

---

## Files

```
flat-detector/
├── run.py                 # orchestrator
├── src/
│   ├── simulate.py        # generate test data
│   ├── validate.py        # check input data
│   ├── detect.py          # rolling std + threshold
│   └── review.py          # false positive review
├── configs/               # JSON parameters
├── data/
│   ├── raw/               # input CSVs
│   └── processed/         # output flags
├── experiments/           # run results
└── requirements.txt       # pandas, numpy
```

## Non-Goals

- ❌ No ML models (autoencoders, isolation forests)
- ❌ No real-time processing
- ❌ No API servers
- ❌ No dashboards
- ❌ No distributed computing
- ❌ No MLOps platforms
