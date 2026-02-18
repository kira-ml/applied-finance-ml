# Flat/Zero-Variance Transaction Monitor

A rule-based data quality monitor that detects when a financial data feed gets "stuck" — repeatedly sending the same price with little or no variation. Built as a fundamentals ML systems project to practice end-to-end pipeline design, basic MLOps habits, and knowing when a simple rule beats a complex model.

---

## What This Does

Financial data feeds sometimes malfunction and send the same price repeatedly. This corrupts downstream models and risk calculations. This monitor catches that by computing a rolling standard deviation over a short window for each asset. If the stdev drops below a defined threshold, the asset gets flagged and logged.

No ML model is used here — because none is needed. A flat line is a broken feed, and a threshold check is the right tool for that.

---

## Project Structure

```
flat-variance-monitor/
│
├── data/
│   ├── prices.csv           # Daily closing prices: 10 assets × ~730 days
│   └── ground_truth.csv     # Manually labeled flat periods for evaluation
│
├── logs/
│   └── alerts.log           # Generated at runtime (gitignored)
│
├── src/
│   ├── config.py            # All parameters in one place
│   ├── data_loader.py       # Load and validate the CSV
│   ├── detector.py          # Rolling stdev + threshold flagging
│   ├── alerts.py            # Log flagged events
│   ├── evaluate.py          # Precision, recall, detection latency
│   └── main.py              # Runs the full pipeline
│
├── requirements.txt
└── README.md
```

---

## Pipeline Overview

```
prices.csv → data_loader.py → detector.py → alerts.py → alerts.log
                                                ↓
                                          evaluate.py → metrics printed to console
```

1. `config.py` sets the window size, threshold, and file paths
2. `data_loader.py` reads and validates the CSV
3. `detector.py` computes rolling stdev per asset and applies the threshold rule
4. `alerts.py` logs flagged (asset, date) pairs
5. `evaluate.py` compares flags against known injected flat windows and reports precision, recall, and detection latency

---

## Quickstart

**Install dependencies:**

```bash
pip install pandas numpy matplotlib
```

**Run the monitor:**

```bash
python src/main.py
```

Flagged events will be written to `logs/alerts.log` and evaluation metrics will print to the console.

---

## Configuration

All parameters live in `src/config.py`. Edit this file to change behavior:

```python
DATA_PATH = "data/prices.csv"
GROUND_TRUTH_PATH = "data/ground_truth.csv"
LOG_PATH = "logs/alerts.log"

WINDOW = 5        # Rolling window size (number of trading days)
THRESHOLD = 0.01  # StdDev below this → feed flagged as stuck
RANDOM_SEED = 42  # For any synthetic data generation
```

Changing the threshold or window only requires editing this one file, which makes it easy to re-run and compare results.

---

## Data Format

**prices.csv** — one row per trading day, one column per asset:

```
date,AAPL,MSFT,JPM,...
2022-01-03,182.01,336.32,168.22,...
2022-01-04,179.70,334.97,167.85,...
...
```

**ground_truth.csv** — flat periods manually injected for evaluation:

```
asset,start_date,end_date
AAPL,2022-06-01,2022-06-07
JPM,2023-02-14,2023-02-18
```

---

## Evaluation Metrics

| Metric | What It Measures |
|---|---|
| Precision | How often a flag is a real flat period (not a false alarm on a genuinely stable asset) |
| Recall | How many injected flat periods were actually caught |
| Detection Latency | How many rows after a flat period begins the first flag fires |

Detection latency matters because catching a stuck feed on day 2 is much better than catching it on day 5.

---

## Design Decisions

**Why no ML model?** A stuck feed has a near-zero standard deviation. That is a deterministic condition, not a distributional anomaly. Trying to train an isolation forest or autoencoder to detect it adds failure modes without adding any insight. A threshold is the correct tool.

**Why is the threshold hardcoded?** The threshold (0.01) is a domain rule, not a learned parameter. It should be set by someone who understands the price scale of the assets being monitored. In a real setting, this would be reviewed and signed off by a data operations analyst.

**Why no class abstractions?** The pipeline is five sequential steps. Sequential function calls are easier to read and debug than a `Pipeline` class wrapping the same logic. Abstraction here would only add confusion.

---

## What Was Left Out (and Why)

- **Unsupervised ML models** — not appropriate; the problem is deterministic
- **Docker / containerization** — unnecessary overhead for a local project
- **Experiment tracking (MLflow, etc.)** — there are no hyperparameters to tune
- **REST API** — no real-time consumer; a log file is sufficient for this scope
- **CI/CD** — out of scope for a single-engineer learning project

---

## Requirements

```
pandas
numpy
matplotlib
```

Python 3.9+ recommended. Runs entirely locally on a standard laptop.

---

## What I Learned

This project helped me think through a few things I want to keep in mind going forward:

- **Not every problem needs a model.** Recognizing when a rule is the right answer is a skill in itself.
- **Evaluation discipline applies to rule-based systems too.** Measuring precision, recall, and latency even for a threshold check gives you something concrete to reason about and improve.
- **Centralized config matters.** Changing one number in `config.py` and re-running is a lot cleaner than hunting through code.
- **Logging is a habit, not a feature.** Having a structured `alerts.log` makes the system's behavior observable in a way that `print()` statements don't.

---

## License

MIT