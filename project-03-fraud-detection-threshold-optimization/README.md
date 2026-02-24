# Fraud Detection Threshold Optimization

## Overview
This project implements a minimal end-to-end machine learning pipeline for binary fraud detection. The primary goal is to demonstrate how standard accuracy metrics fail on imbalanced datasets and how manually optimizing the classification threshold can improve business-relevant outcomes (specifically the F2-Score).

This codebase is designed to run entirely on a local CPU (e.g., Intel Core i5) without Docker, cloud services, or complex orchestration. It focuses strictly on the fundamentals of data handling, model training, and threshold tuning using Scikit-learn.

## Problem Context
In fraud detection, legitimate transactions often outnumber fraudulent ones by 99 to 1.
- **The Issue:** A model that predicts "Legitimate" for every transaction achieves 99% accuracy but catches 0% of fraud.
- **The Solution:** Instead of using the default 0.5 probability cutoff, we calculate Precision-Recall curves to find a specific threshold that maximizes the F2-Score (prioritizing Recall over Precision).

## Project Structure
The directory structure maps directly to the pipeline steps to avoid abstraction overhead.

```text
fraud_threshold_system/
├── data/
│   └── transactions.csv       # Synthetic dataset (generated)
├── models/
│   ├── model.joblib           # Trained Logistic Regression weights
│   ├── scaler.joblib          # Feature scaling parameters
│   └── threshold.txt          # Optimized float threshold
├── src/
│   ├── data.py                # Generates and loads data
│   ├── train.py               # Splits, scales, and trains model
│   ├── tune_threshold.py      # Calculates F2-Score and saves threshold
│   └── infer.py               # Runs single predictions
└── run.py                     # Executes the full pipeline sequentially
```

## Prerequisites
You need Python 3.8+ and the following standard libraries. No GPU drivers or special hardware are required.

```bash
pip install pandas numpy scikit-learn joblib
```

## How to Run

### 1. Execute the Full Pipeline
Run the main script to generate data, train the model, optimize the threshold, and save artifacts.

```bash
python run.py
```

**Expected Output:**
- `data/transactions.csv` is created with a 1% fraud rate.
- Console output showing the calculated Precision-Recall AUC and the selected optimal threshold.
- Model files saved in the `models/` directory.

### 2. Run Inference
To test the model on a new transaction, you can modify `run.py` or import the inference module directly. The system loads the saved threshold from `models/threshold.txt` rather than using a hardcoded value.

```python
from src.infer import predict_fraud

# Example transaction features
new_transaction = {
    "amount": 450.00,
    "time_delta": 120,
    "frequency": 3
}

result = predict_fraud(new_transaction)
print(f"Prediction: {'FRAUD' if result == 1 else 'LEGIT'}")
```

## Key Learning Points
This project intentionally avoids best-practice abstractions (like config files, logging frameworks, or APIs) to focus on core concepts:

1.  **Class Imbalance:** We use `class_weight='balanced'` in Logistic Regression to penalize misclassifying the minority class.
2.  **Metric Selection:** We ignore ROC-AUC and Accuracy, focusing exclusively on Precision-Recall AUC which is more informative for rare events.
3.  **Threshold Tuning:** The `tune_threshold.py` module explicitly iterates through probability cutoffs to maximize the F2-Score, demonstrating that the default 0.5 threshold is rarely optimal for real-world business costs.
4.  **Data Leakage Prevention:** The `StandardScaler` is fit *only* on the training set and applied to the validation/test sets, preventing information from the future leaking into the model.

## Notes on Constraints
- **Hardware:** Designed for limited RAM and CPU-only execution.
- **Data:** Uses synthetic data generation to ensure reproducibility without external downloads.
- **Scope:** This is a learning prototype. It does not include database connections, user authentication, or microservices.