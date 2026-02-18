# Architecture Decision Record

*This document outlines the design choices for my flat transaction detection project. I'm a machine learning student building this to understand data quality monitoring fundamentals. These decisions reflect what I've learned about keeping things simple while building something that works.*

## Why This Architecture?

I need to detect when price data goes flat (stops changing) in financial feeds. The core insight: this doesn't require machine learning - it's a rolling statistics check. My architecture prioritizes:

- **Running on my laptop** (Intel i5, no GPU)
- **Being understandable** when I revisit it in 6 months
- **Easy modification** as I learn what works
- **No infrastructure debt** - I want to learn detection, not DevOps

## Component Choices

| Stage | What I'm Using | Why I Chose It |
|-------|----------------|----------------|
| **Data Loading** | `pandas.read_csv()` | CSV is universal; pandas handles time series well and I'm already learning it |
| **Data Storage** | File system with dated files | No database setup needed; I can see my data directly |
| **Validation** | pandas assertions | A few lines check what matters (columns, nulls, types) |
| **Detection** | `rolling().std()` + threshold | 5 lines of code; leverages pandas speed; matches the math of "flatness" |
| **Alerting** | Print + log file | I'll be running this manually at first; log file keeps history |
| **Results** | CSV output | Easy to open in Excel, share with others, or feed into reports |
| **Orchestration** | Run script manually → later add cron | No tool learning curve; cron is built into my OS |
| **Version Control** | Git (local) | Tracks my changes; I can experiment with branches |
| **Experiment Tracking** | Comments + dated files | I'm not tuning hyperparameters; the "model" is a fixed rule |

## How Data Moves Through the System

**Start → End**

1. **Get data**: Look in `data/raw/` for today's CSV file
2. **Check it**: Make sure columns exist, dates parse, no missing prices
3. **Process**: Group by asset, calculate 5-day rolling standard deviation
4. **Flag**: If rolling std dev < 0.01, mark as flat
5. **Alert**: Print summary to screen, write flagged rows to `logs/alerts.log`
6. **Save**: Write full results to `outputs/` with timestamp
7. **Archive**: Move processed file to `data/processed/`

## What I'm Not Using (And Why)

- **Docker** - I'm running on my laptop directly; one more layer to learn/debug
- **Airflow** - Cron or manual execution is simpler for daily runs
- **Database** - CSV files work fine for 10 stocks × 2 years (~7k rows)
- **MLflow** - No model parameters to track; code comments suffice
- **REST API** - I'll read output files directly; no need for endpoints
- **Kafka** - Data arrives as daily files, not real-time streams
- **Fancy ML models** - Flat lines don't need autoencoders; a threshold works

## Project Structure

```
flat-detector/
├── data/
│   ├── raw/          # Drop new CSV files here
│   └── processed/    # Files move here after processing
├── logs/
│   └── alerts.log    # Flat detection history
├── outputs/          # Results with flags (dated + latest)
├── src/
│   └── detect_flat.py
└── ARCHITECTURE.md   # This file
```

## What Success Looks Like

1. **Precision** - Most flags are real flat periods, not stable assets
2. **Latency** - Flag appears when flat period reaches 5 days
3. **Understandability** - I can explain the logic to someone else in 2 minutes

## Things I Might Change Later

- Add command-line arguments for threshold/window size
- Generate test data with known flat periods for validation
- Add a simple summary report with charts
- Email alerts if this becomes part of daily workflow

*But I'll only add these if I actually need them.*

---

*Last updated: February 2026*