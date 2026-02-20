# 01-risk-probability-calibration

**Objective:** Demonstrate that raw machine learning outputs often misrepresent true default probabilities and show how post-hoc calibration corrects this for accurate risk pricing.

**Problem:** In lending, a model might correctly rank customers by risk (high AUC) but output probabilities that are too extreme or too conservative. This leads to mispriced loans and incorrect regulatory capital reserves.

**Solution:** 
1. Train a baseline Gradient Boosting Classifier on synthetic credit data.
2. Apply **Isotonic Regression** on a held-out validation set to calibrate probabilities.
3. Compare the uncalibrated vs. calibrated models using strict statistical metrics.

## 🛠 Tech Stack
- **Language:** Python 3.9
- **Libraries:** `scikit-learn`, `xgboost`, `pandas`, `numpy`
- **Infrastructure:** Docker (for reproducibility)

## 📊 Success Metrics
| Metric | Target | Result | Status |
| :--- | :--- | :--- | :--- |
| **Brier Score Reduction** | > 15% improvement | *See `reports/metrics.json`* | ✅ Pass |
| **Max Calibration Error** | < 0.05 (across deciles) | *See `reports/metrics.json`* | ✅ Pass |

## 🚀 Quick Start (Docker)
No local environment setup required. Runs entirely in a container.

```bash
# 1. Build the image
docker build -t credit-risk-calib .

# 2. Run the pipeline (outputs saved to local ./models and ./reports)
docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/reports:/app/reports credit-risk-calib
```

*(Note: For Windows PowerShell, replace `$(pwd)` with `${PWD}`)*

## 📂 Project Structure
```text
.
├── Dockerfile            # Reproducible environment
├── requirements.txt      # Dependencies
├── src/
│   ├── train.py        # Core logic: Training + Calibration
│   └── utils.py        # Metric calculations (Calibration Error)
├── models/             # Output: Saved .joblib model artifacts
└── reports/            # Output: JSON metrics and evaluation logs
```

## 🔑 Key Takeaways for Hiring Managers
- **Risk Awareness:** Understands that classification accuracy is insufficient for financial risk; probability calibration is mandatory.
- **Validation Hygiene:** Properly splits data into Train/Validation (for calibration fit) and Test (for final evaluation) to prevent leakage.
- **MLOps Basics:** Uses Docker for environment consistency and saves versioned artifacts with automated metric reporting.

## 📄 License
MIT