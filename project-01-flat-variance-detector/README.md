# Project 01: Flat/Zero-Variance Transaction Detector

A minimalist project to detect "stuck" data feeds in financial time series. Built for learning fundamentals, not production glory.

## 📖 Context

Data feeds from exchanges or counterparties sometimes get "stuck" - sending the exact same price or volume repeatedly. This can corrupt downstream models and risk calculations. Before building fancy anomaly detection models, sometimes you just need a simple monitor that catches flat lines.

## 🎯 Learning Goals

- Data quality monitoring fundamentals
- Rule-based anomaly detection
- Rolling window statistics
- Precision/latency trade-offs
- Synthetic data generation for controlled experiments

## 🧠 Core Concept

If a price series has zero (or near-zero) variance over a rolling window, it's likely broken. No neural networks needed - just a rolling standard deviation and a threshold.

## 📁 Project Structure

```
├── .gitignore                  # Python/ML artifacts ignored
├── README.md                   
├── requirements.txt            # Dependencies
├── config.yaml                 # Simulation & detection parameters
├── run.py                      # Main pipeline orchestrator
├── src/
│   ├── simulator.py            # Generate synthetic price paths with known flat periods
│   ├── data.py                  # Load/save CSVs
│   ├── features.py              # Rolling window calculations
│   ├── detector.py              # Threshold-based flagging
│   └── evaluate.py              # Precision/recall/latency metrics
├── tests/                       # Unit tests
│   ├── test_simulator.py
│   └── test_detector.py
├── data/                        # Generated datasets (gitignored)
└── logs/                        # Experiment logs (gitignored)
```

## 🚀 Getting Started

1. **Clone and setup**
```bash
git clone <repo-url>
cd project-01-flat-variance-detector
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure parameters** in `config.yaml`:
```yaml
# Simulation
n_assets: 10
n_days: 504  # ~2 trading years
volatility: 0.25  # 25% annual volatility
flat_probability: 0.05  # 5% chance of flat period

# Detection
rolling_window: 5
std_threshold: 0.01  # Flag if rolling std < 1%
```

3. **Run the pipeline**
```bash
python run.py
```

## 📊 How It Works

1. **Simulate** realistic price paths using Geometric Brownian Motion
2. **Inject** known flat periods at random intervals (ground truth)
3. **Calculate** rolling standard deviation for each asset
4. **Flag** assets where rolling std drops below threshold
5. **Evaluate** against ground truth to measure precision/recall/latency

## 📈 Success Metrics

- **Precision**: % of flags that are actual flat periods (minimize false alarms on stable assets)
- **Recall**: % of flat periods successfully detected
- **Detection Latency**: How many days after flat period starts before first flag

## ⚠️ What This Project Is NOT

- Not a production-ready monitoring system
- Not using autoencoders, isolation forests, or statistical tests
- Not trying to be fancy - just rolling windows and if statements

## 💡 Why So Simple?

In many real-world cases, over-engineering is the enemy. A flat line is mathematically simple - complex models just add complexity and potential failure points. This project focuses on getting the fundamentals right before adding complexity.

## 🐛 Known Limitations

- Assumes prices aren't trading at very small decimals (penny stocks might need threshold adjustment)
- Rolling window size affects detection latency vs sensitivity
- Currently batch processing, not streaming

## 📚 What's Next

After mastering this, consider:
- Adapting for streaming data
- Adding adaptive thresholds based on asset volatility
- Handling different data frequencies (intraday, tick data)
- Real data integration from actual exchanges

---

*Built by a student, for learning fundamentals. Questions? They're part of the process.*
