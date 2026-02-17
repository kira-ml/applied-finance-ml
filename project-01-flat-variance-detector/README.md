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
pip install -r requirements.txt

# 2. Generate sample data
python generate_data.py

# 3. Run detection
python detect.py

# 4. Review results
python evaluate.py
# Follow interactive prompts to mark false positives
# Precision printed to console, review saved to data/processed/
```

## Project Structure

```
flat-detector/
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── generate_data.py          # Synthetic data generator
├── detect.py                  # Rolling std + threshold logic
├── evaluate.py                # False positive review workflow
└── data/
    ├── raw/                   # Input CSVs (generated)
    │   └── prices.csv
    └── processed/             # Detection outputs and reviews
        ├── flags_20250218_123456.csv
        └── review_20250218_123456.csv
```

## How It Works

1. **Generate/Input:** Run `generate_data.py` to create realistic prices CSV with columns: `date, ticker, close_price` (or place your own CSV in `data/raw/prices.csv`)
2. **Detection:** For each asset, compute 5-day rolling standard deviation
3. **Flag:** If rolling std < 0.01 → mark as flat
4. **Review:** Analyst reviews flags interactively, marks false positives
5. **Iterate:** Tune window/threshold constants in `detect.py` based on precision

## Current Parameters (in detect.py)

```python
# Constants at top of file - edit and commit to change
WINDOW = 5
THRESHOLD = 0.01
MIN_PERIODS = 3
```

## Evaluation

Two metrics matter:
- **Precision:** % of flags that were actually flat (minimize false alarms)
- **Latency:** Days from flat period start to first flag (window size tradeoff)

Review workflow captures false positives for threshold tuning:
```bash
python evaluate.py
# Was this actually flat? (y/n): n
# Was this actually flat? (y/n): y
# Precision: 0.85
```

## Simulated Data

`generate_data.py` creates realistic financial data with:
- Price drift and volatility (random walk with drift)
- Multi-asset correlation (same drift, different volatility)
- Configurable flat periods (ground truth available for testing)

Enables controlled experiments without waiting for real feed failures.

## Iteration Loop

1. Run `python detect.py` → get flags in `data/processed/`
2. Run `python evaluate.py` → review false positives, see precision
3. Adjust threshold/window in `detect.py` (edit constants)
4. Commit changes: `git add detect.py && git commit -m "adjust threshold"`
5. Repeat

## No Over-engineering

- ❌ No ML models (autoencoders, isolation forests)
- ❌ No config files (parameters are code constants)
- ❌ No experiment tracking folders (just timestamps in filenames)
- ❌ No API servers or streaming
- ❌ No distributed processing
- ✅ Just rolling windows, CSVs, and interactive review

## Git Workflow

This project is Git-optimized for learning:
- Each parameter change is a visible commit
- Data files are ignored (in `.gitignore`)
- Only code and documentation are tracked
- Commit history shows iteration and tuning

## License

MIT
