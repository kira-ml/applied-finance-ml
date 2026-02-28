# Project 03 — Fraud Detection Threshold Optimization

## Overview

This is the third project in an applied finance ML learning journey. The goal here is not to build a production-grade fraud system — it is to build one correctly from the ground up, understand the pitfalls of imbalanced classification, and practise the discipline of defining measurable success criteria before looking at results.

The core focus is **threshold optimization**: training a binary fraud classifier (Random Forest) and then deliberately choosing the operating threshold based on a cost-aware objective rather than blindly accepting the default 0.5 cutoff. The F2-score is used to weight recall more heavily than precision, reflecting the real-world asymmetry where a missed fraud is far more damaging than a false alarm.

The numbers in this project are not impressive by any industry standard — and that is expected. What matters at this stage is that the pipeline is sound, the reasoning behind every decision is traceable, and the success criteria are defined and honestly evaluated.

---

## Problem Statement

A standard classifier trained on 1% fraud data will default to a 0.5 probability threshold and achieve 99% "accuracy" by predicting nothing is fraud. That is a failing classifier dressed up as a good one. This project confronts that problem head-on by shifting the evaluation to PR-AUC and F2-score, and by tuning the threshold post-training to minimise expected business cost under a configurable FN/FP cost ratio.

This is a foundational concept in applied finance ML: **accuracy is the wrong objective when the costs of different errors are not equal**. Building this understanding step by step is the entire point.

---

## Dataset

| Property | Value |
|---|---|
| Source | Synthetic (`data/raw/transactions.csv`) |
| Total rows | 200,000 |
| Fraud rate | 1.0% (2,000 fraud cases) |
| Features used | `TransactionAmt`, `card_type`, `merchant_category`, `hour_of_day`, `day_of_week`, `distance_from_home` |
| Train / Test split | 80 / 20 (stratified) |

---

## Pipeline Stages

```
ingest → validate → preprocess → split → train → threshold → error_analysis
```

| Stage | Script | Description |
|---|---|---|
| Ingest | `src/ingest.py` | Load CSV, enforce schema, compute fraud rate |
| Validate | `src/validate.py` | Drop nulls / zero-variance columns, flag log-transform need |
| Preprocess | `src/preprocess.py` | Encode categoricals, scale numerics |
| Split | `src/split.py` | Stratified 80/20 train-test split |
| Train | `src/train.py` | Fit Logistic Regression & Random Forest, select by PR-AUC |
| Threshold | `src/threshold.py` | Optimise F2 and cost thresholds over PR curve |
| Error Analysis | `src/error_analysis.py` | Analyse FN/FP by amount and hour-of-day |

---

## Model Selection

Both models are evaluated by **PR-AUC** (Precision-Recall AUC), which is the right metric for imbalanced binary classification — it is insensitive to the large number of true negatives.

| Model | PR-AUC |
|---|---|
| Logistic Regression | 0.035989 |
| **Random Forest** | **0.040035** ✓ selected |

---

## Threshold Optimization

Two thresholds are computed and compared:

| Strategy | Threshold | Metric Value |
|---|---|---|
| **F2-optimal** (selected) | **0.597613** | F2 = 0.148688 |
| Cost-optimal | 0.870694 | Expected Cost = 3,988.00 / 10k txns |

The two thresholds differ by more than 0.05, so the **F2-optimal threshold** is selected as it maximises recall-weighted performance without being purely cost-driven.

---

## Performance Metrics

All metrics are evaluated on the held-out test set (40,000 transactions, 400 fraud cases) at the **F2-optimal threshold of 0.597613**.

### Classification Metrics

| Metric | Value |
|---|---|
| PR-AUC | 0.0400 |
| Precision | 0.0432 (4.3%) |
| Recall | 0.3825 (38.3%) |
| F2-score | 0.1487 |

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN = 36,208 | FP = 3,392 |
| **Actual Positive** | FN = 247 | TP = 153 |

### Cost Analysis

| Parameter | Value |
|---|---|
| FN cost (missed fraud) | 10 |
| FP cost (false alarm) | 1 |
| Expected cost per 10,000 transactions | **1,465.50** |

### Error Analysis Highlights

- **False Negatives (247 missed fraud):** mean transaction amount = $127.28, median = $16.14 — the model struggles most with low-value fraud that blends with normal patterns.
- **True Positives (153 detected fraud):** mean transaction amount = $158.77 — higher-value fraud is more reliably caught.
- Missed fraud is distributed across all hours of the day with no strong hourly pattern.

---

## Success Metrics

These are the success criteria defined for this project before results were examined. Honest evaluation against them — including failures — is part of the learning process.

| Metric | Target | Achieved | Status |
|---|---|---|---|
| PR-AUC on 1% fraud rate dataset | > 0.40 | 0.040 | ❌ Not met |
| F2-optimal threshold documented | Yes | 0.597613 | ✅ Met |

### Honest Assessment

The PR-AUC target of > 0.40 was not met. The model achieved 0.040 — roughly one tenth of the goal. That is a significant gap and it is worth being clear about why.

The synthetic dataset was generated with only 6 features, and the fraud label does not have a strong enough signal in those features to push the Precision-Recall curve meaningfully above the baseline (which for a 1% fraud rate is 0.01). Both Logistic Regression (0.036) and Random Forest (0.040) landed in the same low range, which suggests the limitation is in the data and feature set, not in the choice of model or threshold strategy.

This is not a reason to lower the target. The target of PR-AUC > 0.40 is realistic on richer real-world fraud datasets (e.g. the IEEE-CIS Kaggle dataset regularly sees PR-AUC in the 0.5–0.8 range with engineered features). The gap here is a concrete, measurable signal about what needs to improve next: **richer features, more discriminative synthetic data generation, or moving to a real dataset.**

The threshold documentation target was fully met. The F2-optimal threshold of 0.597613 was computed, saved, and applied consistently across evaluation and inference. The pipeline mechanics are sound — it is the underlying signal that is weak.

> **What this project did build correctly:** end-to-end pipeline discipline, cost-aware threshold selection, PR-AUC as the right evaluation metric for imbalanced data, and F2-score as the right optimisation objective when recall matters more than precision. These are the fundamentals. The model quality is the next thing to improve.

---

## Artifacts

| File | Description |
|---|---|
| `artifacts/model.pkl` | Trained Random Forest model |
| `artifacts/scaler.pkl` | Fitted StandardScaler |
| `artifacts/threshold.txt` | Saved optimal threshold (0.597613) |
| `artifacts/evaluation_report.txt` | Full threshold optimization report |
| `artifacts/validation_report.txt` | Data validation summary |
| `artifacts/model_selection_log.txt` | Model selection winner log |
| `artifacts/error_analysis.txt` | FN/FP amount and hourly analysis |
| `artifacts/pr_curve.png` | Precision-Recall curve plot |
| `artifacts/deploy_ready/` | Deploy-ready artifacts directory |
| `data/output/predictions.csv` | Inference output with scores and labels |

---

## Quickstart

### Training

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate           # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run full training pipeline
python run_training.py data/raw/transactions.csv
```

### Inference

```bash
python run_inference.py --input data/inference/new_transactions.csv
```

---

## Project Structure

```
project-03-fraud-detection-threshold-optimization/
├── run_training.py          # End-to-end training pipeline entry point
├── run_inference.py         # Inference pipeline entry point
├── requirements.txt
├── src/
│   ├── ingest.py
│   ├── validate.py
│   ├── preprocess.py
│   ├── split.py
│   ├── train.py
│   ├── threshold.py
│   ├── error_analysis.py
│   ├── generate_synthetic_data.py
│   └── infer.py
├── data/
│   ├── raw/transactions.csv
│   ├── inference/new_transactions.csv
│   └── output/predictions.csv
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── threshold.txt
│   ├── evaluation_report.txt
│   └── ...
└── tests/
    ├── test_ingest.py
    └── test_validate.py
```
