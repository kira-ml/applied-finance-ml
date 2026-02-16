# Applied Finance ML

**Operational Machine Learning for Finance.**  
A collection of fundamental-focused ML solutions for internal finance operations. Prioritizing data integrity, risk-aware metrics, and interpretability over complexity.

---

## 🎯 Mission

This repository demonstrates competence in **Applied Machine Learning Engineering** within the finance industry. 

Unlike academic projects that focus on predictive accuracy on public datasets, these projects simulate **internal operational problems** faced by banks, funds, and corporate treasury teams. Each solution adheres to strict industry constraints: **zero over-engineering, maximum interpretability, and rigorous leakage prevention.**

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

This collection covers the **High-ROI Fundamentals** required for entry-level Finance ML roles.

| # | Project | Business Problem | Model | Key Fundamental |
| :--- | :--- | :--- | :--- | :--- |
| **01** | [Cash Position Forecast](./project-01-cash-forecast) | Treasury Liquidity Planning | Linear Regression | **Temporal Integrity** (No Leakage) |
| **02** | [AML Alert Triage](./project-02-aml-triage) | Compliance Risk Prioritization | Random Forest | **Cost-Sensitive Learning** |
| **03** | [Transaction Matcher](./project-03-transaction-match) | Operations Reconciliation | Logistic Regression | **Threshold Tuning & Confidence** |
| **04** | [Expense Policy Flag](./project-04-expense-policy) | Internal Audit Automation | Decision Tree | **Hybrid Rule-Based Systems** |
| **05** | [Settlement Failure Predictor](./project-05-settlement-failure) | Operational Risk Management | Weighted Logistic Reg | **Class Imbalance Handling** |
| **06** | [Fee Waiver Propensity](./project-06-fee-waiver) | Client Retention Strategy | Logistic Regression | **Expected Value Framing** |
| **07** | [Trade Ticket Correction](./project-07-trade-ticket) | Trade Operations Quality | Random Forest | **Inference Time Boundaries** |
| **08** | [Security Master Scorer](./project-08-security-master) | Data Quality Management | Isolation Forest | **Unsupervised Anomaly Detection** |
| **09** | [Vendor Payment Delay](./project-09-vendor-payment) | Accounts Payable Planning | Target Encoded Reg | **High Cardinality Encoding** |
| **10** | [Manual Journal Predictor](./project-10-journal-entry) | Accounting Workload Planning | Decision Tree | **Auditability & Explainability** |

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
