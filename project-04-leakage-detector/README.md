# Project 4: Revenue Forecasting with Time-Series Split

---

## 1. Project Interpretation

### Problem Type

Supervised regression (univariate time-series forecasting).

### Target Variable

Future revenue at time **t** (e.g., daily revenue).

### Input Modality

Structured time-series tabular data:

* `date`
* `revenue`
* Optional known-at-time features (e.g., day-of-week)

### Assumptions About Dataset

* 2–5 years of daily data (≈ 700–1800 rows)
* Single time series
* Stored as a CSV file
* Moderate noise, possible seasonality
* No exogenous variables that require forecasting

### Key Risks

1. **Look-ahead bias (critical)**

   * Using future values in lag features
   * Global scaling before splitting
   * Random shuffling

2. **Temporal leakage via preprocessing**

   * Fitting scalers on entire dataset

3. **Autocorrelated residuals**

   * Indicates under-modeled temporal structure

4. **Variance instability across folds**

   * Indicates non-stationarity or insufficient window size

---

## 2. End-to-End System Architecture

### High-Level Flow

```
Raw CSV
  → Ingestion
  → Validation
  → Feature Engineering (Lag-only, time-safe)
  → TimeSeriesSplit (Expanding Window)
      → Per-fold:
          Fit scaler on train only
          Train model
          Validate
  → Aggregate Metrics
  → Residual Diagnostics
  → Final Model Training on Full Data
  → Persistence
  → Inference (recursive lag generation)
```

---

## Stage-by-Stage Design

---

### 2.1 Raw Data → Ingestion

**Purpose**
Load chronological revenue data without altering order.

**Input**
`data/raw/revenue.csv`

**Output**
`pandas.DataFrame` sorted by date ascending.

**Design Decisions**

* Explicit date parsing
* Enforce monotonic increasing date
* No resampling (keeps assumptions minimal)

**Leakage Prevention**

* No shuffling
* No transformation

---

### 2.2 Validation

**Purpose**
Ensure temporal integrity before modeling.

**Checks**

* No duplicate timestamps
* No missing dates (or explicitly logged)
* No negative revenue
* Strict chronological order

**Artifacts**

* Validation report (printed/logged)

**Justification**
Errors here silently corrupt CV.

---

### 2.3 Feature Engineering (Lag-Only)

**Purpose**
Create strictly past-dependent predictors.

**Input**
Validated DataFrame

**Transformations**
For each time t:

* `lag_1`
* `lag_7`
* `lag_14`
* `rolling_mean_7` (computed using `.shift(1)`)
* `rolling_std_7` (shifted)

Optional safe time features:

* `day_of_week`
* `month`

**Critical Rule**
All rolling operations use:

```python
df['revenue'].shift(1).rolling(window=7).mean()
```

**Output**
Feature matrix X
Target vector y

**Leakage Prevention**

* Every derived feature uses `.shift(1)`
* First max_lag rows dropped

---

### 2.4 Splitting: Expanding Window CV

**Strategy**
`sklearn.model_selection.TimeSeriesSplit`

Example:

* 5 folds
* Expanding training window
* Fixed validation size

```
Fold 1:
Train: [0 --- t1]
Valid: [t1+1 --- t2]

Fold 2:
Train: [0 --- t2]
Valid: [t2+1 --- t3]
```

**Justification**
Matches real deployment: future unseen data.

**No KFold**
Random KFold would leak.

---

### 2.5 Training (Per Fold)

**Model Choice**
`sklearn.linear_model.Ridge`

Why:

* Fast on CPU
* Stable with collinearity
* Low variance
* Interpretable
* No GPU required

**Scaling**
`StandardScaler` fit only on training fold.

Pipeline per fold:

```
Fit scaler on X_train
Transform X_train
Train Ridge
Transform X_valid
Predict
```

**Artifacts per Fold**

* RMSE
* Residuals
* Fold index ranges

---

### 2.6 Evaluation

#### Primary Metric

RMSE per fold.

#### Stability Requirement

Compute:

```
abs(RMSE_i - RMSE_(i-1)) / RMSE_(i-1)
```

Must be < 5%.

#### Residual Diagnostics

After each fold:

* Compute Pearson correlation between:

  * residuals
  * time index (0..n)

Must be ~0.

If significant correlation → temporal bias remains.

---

### 2.7 Error Analysis

Performed after CV:

* Plot residuals vs time
* Check:

  * Trend patterns
  * Increasing variance
  * Seasonal bias

Feedback loop:
If residuals show seasonality → add additional lag or seasonal features.

But baseline remains minimal.

---

### 2.8 Final Training

After validation:

Train model on **entire dataset** using:

* Same feature logic
* Same scaler logic

Persist:

* `model.pkl`
* `scaler.pkl`
* `feature_config.json`

---

### 2.9 Inference

Input:

* Latest available revenue history

Process:

1. Construct lag features from history
2. Scale using persisted scaler
3. Predict next step

For multi-step forecasting:

* Recursive forecasting
* Append prediction to history
* Recompute lag features

No retraining during inference.

---

## Realistic Failure Mode

Structural break:

* Sudden macroeconomic shift
* Model trained on past regime
* CV stable but future distribution shifts

---

## Known Modeling Limitation

Ridge regression assumes linear relationship.
Nonlinear patterns (e.g., saturation effects) will not be captured.

We accept this due to:

* Resource constraints
* Focus on validation hygiene

---

## 3. Resource Constraints Justification

All components:

* CPU-only
* sklearn only
* pandas only
* No deep learning
* No heavy libraries
* Dataset small (fits memory)
* Time complexity negligible

Rejected:

* XGBoost (heavier)
* Prophet (adds abstraction)
* LSTM (overkill, GPU-less)
* Hyperparameter search frameworks

---

## 4. Minimal Project Directory Structure

```
revenue_forecasting/
│
├── data/
│   └── raw/
│       └── revenue.csv
│
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_config.json
│
├── notebooks/
│   └── residual_analysis.ipynb
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── split.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── main.py
│
└── requirements.txt
```

No extra folders. Each has a concrete purpose.

---

## 5. Complete `src/` Module Specification

---

### 5.1 `config.py`

**Responsibilities**

* Define:

  * LAG_LIST
  * ROLLING_WINDOWS
  * N_SPLITS
  * VALIDATION_SIZE

**Produces**
Constants only.

**Must NOT**

* Import pandas
* Perform logic
* Read data

---

### 5.2 `data.py`

**Responsibilities**

* Load CSV
* Parse dates
* Sort
* Validate chronology

**Input**
Path to CSV

**Output**
Validated DataFrame

**Must NOT**

* Create features
* Split data
* Train model

---

### 5.3 `features.py`

**Responsibilities**

* Generate lag features
* Generate rolling features (shifted)
* Drop NA rows
* Return X, y

**Input**
Validated DataFrame

**Output**
X (DataFrame), y (Series)

**Artifacts**
Feature name list (for persistence)

**Must NOT**

* Split data
* Fit scalers
* Train model
* Perform CV

---

### 5.4 `split.py`

**Responsibilities**

* Implement TimeSeriesSplit
* Yield train_index, valid_index

**Input**
Length of dataset

**Output**
Generator of index tuples

**Must NOT**

* Access feature values
* Train models
* Compute metrics

---

### 5.5 `train.py`

**Responsibilities**

* Train Ridge model on provided X_train, y_train
* Fit scaler
* Return trained model + scaler

**Input**
Training arrays

**Output**
model, scaler

**Must NOT**

* Split data
* Compute CV
* Persist artifacts

---

### 5.6 `evaluate.py`

**Responsibilities**

* Compute RMSE
* Compute fold variance %
* Compute residual-time correlation
* Aggregate metrics

**Input**
Predictions, y_true, fold index

**Output**
Dictionary of metrics

**Must NOT**

* Train models
* Modify data
* Create plots

---

### 5.7 `inference.py`

**Responsibilities**

* Load artifacts
* Build lag features from new history
* Perform recursive forecasting

**Input**
Recent historical revenue

**Output**
Predicted next revenue

**Must NOT**

* Retrain model
* Modify artifacts

---

### 5.8 `main.py`

**Responsibilities**
Orchestrates:

1. Load data
2. Validate
3. Feature engineering
4. TimeSeriesSplit loop:

   * Train
   * Evaluate
5. Check stability criteria
6. Train final model
7. Persist artifacts

**Must NOT**

* Contain modeling math
* Duplicate feature logic
* Perform inference

---

## 6. Final Simplicity Review

Can anything be removed?

* `config.py` → Could inline constants, but separation improves clarity without overhead.
* `split.py` → Could embed inside main, but separation prevents leakage errors.
* `evaluate.py` → Necessary to isolate statistical checks.

No module is speculative.
No abstraction layer exists.
No cross-cutting utilities.
No premature extensibility.

The design is already minimal while preserving:

* Strict leakage prevention
* Statistically valid expanding CV
* Residual independence diagnostics
* CPU feasibility
* Clear pipeline boundaries

---

# Final Result

This architecture:

* Guarantees zero look-ahead bias
* Enforces expanding-window validation
* Quantifies fold RMSE stability (<5%)
* Verifies residual-time independence
* Runs efficiently on i5 CPU
* Remains simple, readable, and reproducible

It demonstrates disciplined time-series validation — the core differentiator of this project.
