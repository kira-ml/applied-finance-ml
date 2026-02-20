# Applied Finance ML

**Operational Machine Learning for Finance.**  
A collection of fundamental-focused ML solutions for internal finance operations. Prioritizing data integrity, risk-aware metrics, and interpretability over complexity.

---

## 🎯 Mission

As a Senior ML Hiring Manager, I've seen countless candidates sink their interviews by presenting a portfolio of flashy, complex projects that would be immediately rejected by our risk and model governance teams.

These 10 projects are designed to show you can do the opposite: solve the boring, high-stakes problems with boring, correct code. This is what we actually pay for.

Here are 10 underrated, high-ROI projects tailored for your transition into finance.

---

## 🛡️ Core Philosophy

This portfolio is built on the belief that **judgment outweighs tools** in finance ML. Every project follows these rules:

| Principle | Implementation |
| :--- | :--- |
| **Fundamentals Over Complexity** | No Deep Learning. No Reinforcement Learning. Standard models (Linear/Tree-based) only. |
| **Data Integrity** | Strict temporal splits. No look-ahead bias. Synthetic data used to protect privacy. |
| **Business Alignment** | Metrics are tied to cost/risk (e.g., "Hours Saved"), not vanity metrics (e.g., "Accuracy"). |
| **Interpretability** | Models must be explainable to auditors and stakeholders. No black boxes. |
| **Minimal Infrastructure** | Built to run on local machines. No cloud dependencies. No complex pipelines. |

---

## 📚 Project Catalog

### Project 1: Detecting Flat/Zero-Variance Transactions in Feed Data
**Real-World Context:** Data feeds from counterparties or exchanges often get "stuck," sending the same price or trade volume repeatedly. This corrupts downstream models and risk calculations. You need a simple monitor to catch it before it causes a million-dollar trading error.

**Core Fundamental(s):** Data Quality Monitoring; Anomaly Detection as a Rule.

**Minimal Solution Approach:** Load a time-series CSV of daily closing prices for 10 stocks over 2 years. Write a script that uses a rolling window (e.g., last 5 records) to calculate the rolling standard deviation. Flag any asset/security where the rolling stdev falls below an absolute, domain-defined threshold (e.g., < 0.01, assuming prices are not pennies). Use a simple `if` statement to trigger an alert log. **Forbidden:** Any clustering, isolation forests, or neural networks. If the data is flat, it's broken. You don't need an ML model to tell you that; you just need a window and a threshold.

**Success Metrics:** 1) Precision of flags (minimizing false alarms on genuinely stable assets like T-bills). 2) Detection latency (how quickly after the flat period begins is it flagged?). 3) The explicability of the rule to a data operations analyst.

**The "Over-Engineering" Trap:** Trying to build an "unsupervised anomaly detection model" with autoencoders or fancy statistical tests. This adds zero value. The problem is a flat line, which is the simplest mathematical concept. Over-engineering here makes the monitor itself a source of potential failure.

---

### Project 2: Credit Risk Probability Calibration
**Problem Statement:** Raw model outputs often fail to represent true default probabilities, leading to mispriced risk in lending portfolios.

**Core ML Solution:** Train a baseline Gradient Boosting Classifier and apply post-hoc calibration techniques (Platt Scaling or Isotonic Regression) on a held-out validation set.

**Success Metrics:** Achieve a Brier Score reduction of at least 15% compared to uncalibrated logits; demonstrate calibration error < 0.05 across decile bins.

**ROI Rationale:** Demonstrates understanding that risk management requires accurate probability estimates, not just class labels, which is critical for regulatory capital calculations.

---

### Project 3: Fraud Detection Threshold Optimization
**Problem Statement:** Standard accuracy metrics are misleading in fraud detection due to extreme class imbalance, causing high false negative rates.

**Core ML Solution:** Implement a Logistic Regression or Random Forest model optimized using Precision-Recall AUC rather than ROC AUC, with manual threshold tuning based on cost matrices.

**Success Metrics:** Achieve a Precision-Recall AUC > 0.40 on a 1% fraud rate dataset; document the specific classification threshold selected to maximize F2-Score.

**ROI Rationale:** Shows competency in handling imbalanced data and aligning model objectives with business costs rather than default library metrics.

---

### Project 4: Revenue Forecasting with Time-Series Split
**Problem Statement:** Random shuffling of financial time-series data introduces look-ahead bias, inflating performance metrics and causing production failure.

**Core ML Solution:** Build a regression model using only lagged features and enforce a strict expanding-window cross-validation strategy (TimeSeriesSplit) without future data leakage.

**Success Metrics:** Demonstrate < 5% variance in RMSE between consecutive validation folds; ensure zero correlation between residuals and time indices.

**ROI Rationale:** Proves rigorous validation hygiene and understanding of temporal dependencies, distinguishing the candidate from those who treat time-series as tabular data.

---

### Project 5: Feature Distribution Drift Monitoring (PSI)
**Problem Statement:** Model performance degrades over time as input data distributions shift away from training conditions (concept drift).

**Core ML Solution:** Develop a monitoring script that calculates Population Stability Index (PSI) for all input features between training and simulated production batches.

**Success Metrics:** Flag features with PSI > 0.1 as unstable; generate automated alerts when cumulative PSI exceeds 0.2 across the feature set.

**ROI Rationale:** Highlights MLOps readiness by focusing on model maintenance and observability rather than just initial model training.

---

### Project 6: Transaction Memo Categorization
**Problem Statement:** Unstructured text in transaction memos prevents automated expense tracking and financial reporting.

**Core ML Solution:** Construct a simple NLP pipeline using TF-IDF vectorization and Multinomial Naive Bayes to classify transaction descriptions into standard categories.

**Success Metrics:** Achieve macro-averaged F1-Score > 0.75 on a 10-class dataset; reduce vocabulary size by 30% through stop-word removal and lemmatization without losing accuracy.

**ROI Rationale:** Demonstrates ability to handle unstructured data commonly found in banking systems using lightweight, interpretable methods suitable for high-volume processing.

---

### Project 7: Adverse Action Code Generation
**Problem Statement:** Regulatory compliance requires providing specific reasons for loan denials based on model decisions.

**Core ML Solution:** Integrate SHAP (SHapley Additive exPlanations) values with a rule-based mapping system to translate top negative feature contributions into standard regulatory reason codes.

**Success Metrics:** Successfully map top 3 negative SHAP values to valid reason codes for 100% of denied applications in the test set; ensure explanation generation latency < 50ms per record.

**ROI Rationale:** Directly addresses Fair Lending compliance (ECOA), showing the candidate understands the legal constraints surrounding ML deployment in finance.

---

### Project 8: Target Leakage Detection Audit
**Problem Statement:** Accidental inclusion of future information in training features creates artificially high performance that fails in production.

**Core ML Solution:** Create an audit tool that calculates mutual information and correlation coefficients between each feature and the target variable across different time windows.

**Success Metrics:** Identify and remove at least one intentionally planted leakage feature with correlation > 0.8; document the reduction in validation score after removal.

**ROI Rationale:** Shows proactive risk management and data integrity skills, preventing costly errors before model deployment.

---

### Project 9: Unsupervised Transaction Anomaly Detection
**Problem Statement:** Labeled fraud data is scarce, requiring methods to detect novel suspicious patterns without supervised learning.

**Core ML Solution:** Implement an Isolation Forest algorithm on normalized transaction amounts and frequencies to flag outliers without using target labels.

**Success Metrics:** Capture 80% of known fraud cases (used only for final evaluation) within the top 5% of anomaly scores; maintain false positive rate < 10% on clean data.

**ROI Rationale:** Demonstrates ability to derive value from unlabeled data, a common scenario in financial crime compliance where new fraud patterns emerge constantly.

---

### Project 10: Cost-Sensitive Loan Collection Modeling
**Problem Statement:** Not all loan defaults carry equal financial loss; models should prioritize high-value accounts during collection efforts.

**Core ML Solution:** Train a classification model with custom class weights or a cost-sensitive loss function proportional to the outstanding loan balance.

**Success Metrics:** Achieve a 10% increase in total recovered value compared to a standard accuracy-optimized model on the same test set; document the cost matrix used.

**ROI Rationale:** Illustrates business acumen by optimizing for financial impact rather than abstract statistical metrics, aligning engineering work with bottom-line goals.

---

## 🛠️ Tech Stack

Designed for simplicity and reproducibility.

*   **Language:** Python 3.9+
*   **Core Libraries:** `pandas`, `numpy`, `scikit-learn`
*   **Visualization:** `matplotlib`, `seaborn`
*   **Environment:** Local Jupyter Notebooks / Python Scripts
*   **Data:** Synthetic generators included (No external APIs)

---

## 🚀 Getting Started

Each project is self-contained. To run a specific project:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kira-ml/applied-finance-ml.git
    cd applied-finance-ml
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Generate synthetic data:**
    ```bash
    python project-01-cash-forecast/data/generate_data.py
    ```

4.  **Run training pipeline:**
    ```bash
    python project-01-cash-forecast/src/train.py
    ```

*Refer to individual project READMEs for specific business logic and metric definitions.*

---

## ⚠️ Important Disclaimers

*   **Synthetic Data:** All data used in this repository is synthetically generated to mimic financial structures. It does not contain real PII or proprietary financial information.
*   **Educational Purpose:** These models are designed for learning and portfolio demonstration. They are not validated for live trading or regulatory compliance.
*   **No Financial Advice:** Nothing in this repository constitutes financial advice or investment recommendation.

---

## 📬 Contact

I am actively seeking opportunities as a **Machine Learning Engineer** in the finance industry.

*   **LinkedIn:** [https://www.linkedin.com/in/ken-ira-lacson-852026343/](https://www.linkedin.com/in/ken-ira-lacson-852026343/)
*   **GitHub:** [https://github.com/kira-ml](https://github.com/kira-ml)

---

<p align="center">
  <i>Built with a focus on fundamentals, integrity, and operational value.</i>
</p>
