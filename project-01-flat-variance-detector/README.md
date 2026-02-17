# Flat/Variance Transaction Detector

A minimal monitoring tool that detects "stuck" data feeds in financial time series by identifying periods of abnormally low variance. Built for learning end-to-end ML workflow fundamentals.

## Project Purpose

Real-world data feeds from exchanges can get stuck, sending the same price repeatedly. This corrupts downstream models and risk calculations. This project detects flat periods using rolling statistics—no ML models needed, just disciplined implementation and evaluation.

**Core learning goals:**
- End-to-end ML workflow discipline
- Reproducible experimentation
- Evaluation methodology (precision/latency tradeoffs)
- Simple iteration based on error analysis

## Quick Start

```bash
# 1. Clone and setup
git clone <repo>
cd flat-detector

# 2. Generate sample data (or use your own CSV)
python src/simulate.py --config configs/sim_config.json

# 3. Run detection
python run.py --config configs/run_001.json

# 4. Review results
# Follow interactive prompts to mark false positives
# Metrics saved to experiments/run_001/metrics.txt
```

## Project Structure

```
flat-detector/
├── data/
│   ├── raw/               # Input CSVs (real or simulated)
│   └── processed/         # Latest detection output
├── src/
│   ├── simulate.py        # Synthetic data generator
│   ├── detect.py          # Rolling std + threshold logic
│   └── review.py          # False positive review workflow
├── configs/               # JSON parameter files
├── experiments/           # Run outputs by ID
└── run.py                 # Main workflow orchestrator
```

## How It Works

1. **Input:** Daily prices CSV with columns: `date, ticker, close_price`
2. **Detection:** For each asset, compute 5-day rolling standard deviation
3. **Flag:** If rolling std < 0.01 → mark as flat
4. **Review:** Analyst reviews flags, marks false positives
5. **Iterate:** Tune window/threshold based on precision

## Configuration Example

`configs/run_001.json`
```json
{
  "window": 5,
  "threshold": 0.01,
  "min_periods": 3,
  "data_path": "data/raw/daily_prices.csv",
  "output_path": "experiments/run_001/flags.csv"
}
```

## Evaluation

Two metrics matter:
- **Precision:** % of flags that were actually flat (minimize false alarms)
- **Latency:** Days from flat period start to first flag (window size tradeoff)

Review workflow captures false positives for threshold tuning.

## Simulated Data

`simulate.py` generates realistic financial data with:
- Price drift and volatility
- Multi-asset correlation
- Configurable flat periods (ground truth included)

Enables controlled experiments without waiting for real feed failures.

## Iteration Loop

1. Run detector → get flags
2. Review false positives → log patterns
3. Adjust threshold/window → new run
4. Compare precision across runs

## No Over-engineering

- ❌ No ML models (autoencoders, isolation forests)
- ❌ No real-time streaming
- ❌ No API servers
- ❌ No distributed processing
- ❌ No MLOps platforms
- ✅ Just rolling windows and CSVs

## License

MIT
