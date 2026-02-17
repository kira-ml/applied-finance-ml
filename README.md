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

### Project 2: Simple Corporate Bond Yield Interpolator
**Real-World Context:** A portfolio manager needs a daily "fair value" estimate for a bond that doesn't trade every day. They don't need a PhD thesis; they need a sensible, explainable number based on the trades that *did* happen that day for similar bonds.

**Core Fundamental(s):** Domain-Logic Feature Engineering; Interpretability.

**Minimal Solution Approach:** Use a small dataset of ~200 corporate bond trades (Maturity, Coupon, Rating, Last Price). To estimate the yield for a bond maturing in 4.5 years, filter the dataset to bonds with the same rating and maturities between 3 and 6 years. Perform a simple linear interpolation (or a piecewise linear fit) between the two closest maturities. **Forbidden:** Polynomial regression, Gaussian Processes, or any form of global spline that could produce non-monotonic or wildly swinging yield curves. The market expects a smooth, upward/downward sloping line.

**Success Metrics:** 1) Interpolation error on a hold-out set of bonds that did trade (i.e., how well did you guess the actual traded price?). 2) The time taken to explain the logic to a portfolio manager. 3) Monotonicity check (does your interpolated yield curve ever invert in a nonsensical way for the given rating?).

**The "Over-Engineering" Trap:** Jumping to a machine learning model like Random Forest to predict price. This is a problem of *interpolation*, not extrapolation. ML models can find spurious correlations (e.g., with the day of the month) that have no financial basis, creating an uninterpretable and risky "black box" price.

---

### Project 3: Alert Prioritization for Trade Settlement Failures
**Real-World Context:** The operations team gets 500 alerts a day about trades that *might* fail to settle. They only have time to investigate 50 manually. You need to build a simple prioritization system so they look at the 50 most likely to actually fail first.

**Core Fundamental(s):** Problem Framing as Ranking; Business-Aligned Metrics.

**Minimal Solution Approach:** Frame this as a binary classification problem, but evaluate it on ranking quality. Use 3-5 features: counterparty historical fail rate, currency (exotic vs. major), trade settlement lag (T+1, T+2), and notional value. Train a Logistic Regression on historical data of trades that settled vs. failed. Use the predicted probability to rank the day's 500 alerts. **Forbidden:** Do not build a complex dashboard or real-time system. A simple script that outputs a sorted CSV file is the deliverable. Do not use LightGBM.

**Success Metrics:** 1) Precision@K (e.g., of the top 50 ranked alerts, how many were actual failures?). 2) Coverage of total notional value at risk (are you surfacing the big, risky failures?). 3) Time saved by ops team.

**The "Over-Engineering" Trap:** Building a massive feature store and a real-time API. The ops team just needs a list at 8:00 AM. A perfect model that's too slow or complex to run is useless. A simple model that ranks the top 50 correctly 80% of the time is a game-changer.

---

### Project 4: Time-Series Leakage Detector in Backtests
**Real-World Context:** A quant on your team hands you a strategy's backtest results claiming a Sharpe ratio of 3.0. Your job is to sanity-check their data pipeline for the most common error: using future data to make past decisions.

**Core Fundamental(s):** Leakage Prevention; Critical Code Review.

**Minimal Solution Approach:** Take a simple momentum strategy script (provided by you or found online) that is intentionally written to leak data (e.g., it calculates the "close of day" signal using the day's closing price before the market close). Write a validation script that ingests the same time-series data and checks for two things: 1) That any calculated feature at time `t` does not contain any data from time `t` or later. 2) That when the strategy is re-run with a simple `shift(1)` applied to all features, the performance collapses to a realistic level. **Forbidden:** Do not attempt to fix the strategy. The project is to build the *detector*. No neural nets, just rigorous pandas index management.

**Success Metrics:** 1) Number of leakage bugs successfully identified in the target code. 2) Clarity of the report generated for the quant (e.g., "On day X, feature Y used future value Z"). 3) The detector's runtime overhead.

**The "Over-Engineering" Trap:** Building a generic data lineage tool. The problem is specific: "Did you use tomorrow's price to trade today?" It's a simple time-shift check. Trying to build a general-purpose solution for all possible leakages is a research project, not a practical tool.

---

### Project 5: Interpretable Credit Line Utilization Forecaster
**Real-World Context:** The bank needs to forecast how much of their committed credit lines will be drawn down next month to ensure they have enough cash reserves. A simple, explainable forecast is required for regulatory reporting.

**Core Fundamental(s):** Simplicity for Regulatory Approval; Baseline Models.

**Minimal Solution Approach:** Use a dataset of monthly credit line utilization rates for 1,000 corporate clients over 3 years. The primary feature is utilization from the previous month. Your entire "model" is `next_month_util = current_month_util`. Measure the error. Then, as a "complex" model, add a single rolling average feature (e.g., 3-month average) and compare. **Forbidden:** Any autoregressive model (ARIMA, SARIMA) or LSTM. The goal is to show that a "dumb" persistence forecast is often the most robust and defensible baseline for regulators. Present your results as a comparison showing the marginal, often negative, value of adding complexity.

**Success Metrics:** 1) Improvement (or lack thereof) over the naive persistence model. 2) The ability to explain the forecast driver in one sentence ("We expect them to draw down the same as last month"). 3) Worst-case error analysis.

**The "Over-Engineering" Trap:** Building an elaborate time-series model with external macro-economic variables. For a one-month-ahead forecast, last month's value is incredibly strong. Adding more variables creates an overfit model that fails during regime shifts and is impossible to explain to a regulator.

---

### Project 6: Systematic Data Type Validator for Feed Ingestion
**Real-World Context:** A new data feed arrives daily. Sometimes, a field that is supposed to be a float (like trade price) arrives as a string "N/A" or a negative number, breaking the entire risk system. You need a pre-ingestion check.

**Core Fundamental(s):** Data Hygiene; Defensive Coding.

**Minimal Solution Approach:** Write a Python script that reads a sample file (CSV/JSON) and a simple schema definition (e.g., a dictionary: `{'trade_price': float, 'volume': int, 'counterparty': str}`). The script checks each column: can all values be cast to the expected type? Are there nulls in a non-nullable field? Are there values outside a reasonable range (e.g., negative volume)? Output a simple pass/fail report with the first 3 offending rows. **Forbidden:** Do not build a data cleaning or imputation step. The goal is to *reject bad data*, not guess what it should be. No ML, just try/except blocks and type checking.

**Success Metrics:** 1) Zero production outages caused by data type mismatches. 2) Time from feed arrival to validation report (< 1 second). 3) False positive rate (rejecting good data due to a bad rule).

**The "Over-Engineering" Trap:** Using a schema validation library like Great Expectations or building a full data pipeline. The problem is a simple, critical check. Pulling in a heavy dependency or building a distributed system for a file that's <100MB introduces maintenance overhead and complexity that far outweighs the benefit.

---

### Project 7: "Is this a corporate action?" Binary Classifier
**Real-World Context:** Every day, thousands of unstructured news headlines hit the wire. A small fraction of them announce Corporate Actions (stock splits, dividend announcements, mergers). You need to filter the noise so analysts can investigate the relevant ones.

**Core Fundamental(s):** Text Classification with Extreme Simplicity; Signal vs. Noise.

**Minimal Solution Approach:** Create a dataset of 500 headlines (250 positive examples of corporate actions, 250 negative examples of general financial news). Use a simple Bag-of-Words (CountVectorizer) on the headlines. Train a Logistic Regression (with strong L2 regularization) to classify them. **Forbidden:** No BERT, no Transformers, no Word2Vec, no embeddings. The vocabulary of corporate actions is small and specific ("split," "dividend," "acquisition," "ex-date"). A linear model over word counts is perfectly sufficient and infinitely more interpretable.

**Success Metrics:** 1) Precision at the top of the ranked list (how many of the top 50 flagged headlines are actual corporate actions?). 2) Latency (time to classify 1000 headlines). 3) Ease of updating the model (adding new keywords like "spin-off").

**The "Over-Engineering" Trap:** Using the latest LLM or a deep learning model. This creates a huge, opaque, and computationally expensive pipeline for a problem solvable by keyword matching and a simple linear classifier. The LLM is overkill and will fail if the news vendor's language style differs from its training data.

---

### Project 8: Counterparty Risk Flagging via Simple Ratio Analysis
**Real-World Context:** The risk committee wants a monthly early-warning system for vendors or trading partners that might be heading toward bankruptcy, based on their latest financial filings.

**Core Fundamental(s):** Feature Engineering based on Financial Theory; Interpretability.

**Minimal Solution Approach:** Use a dataset of 1,000 companies with their last 3 years of financials and a label for "defaulted" vs. "healthy." Engineer 3-4 classic, domain-driven features from raw financial statements: the Altman Z-Score components (e.g., Working Capital/Total Assets, Retained Earnings/Total Assets, EBIT/Total Assets). Train a simple Decision Tree (max depth=3) to predict default. The entire value is in the feature engineering based on decades of accounting research. **Forbidden:** Do not let the model do the feature discovery. No PCA, no feature crossing, no automated feature selection. The features are the theory. The model is just a way to calibrate their weights.

**Success Metrics:** 1) The model's lift at identifying the 5% riskiest companies. 2) The auditability of the decision process (can you trace a "high risk" flag back to "low Working Capital/Total Assets"?). 3) Stability of feature importance over time.

**The "Over-Engineering" Trap:** Throwing the raw financial statement line items into a black-box model like XGBoost and letting it find interactions. This ignores 50 years of accounting research, creates an uninterpretable model, and is highly likely to overfit to spurious correlations in the small dataset.

---

### Project 9: Anomaly Detection in Monthly Expense Reports
**Real-World Context:** A mid-sized asset manager suspects employees are submitting personal expenses. They need a simple, non-accusatory way to flag statistically unusual submissions for audit.

**Core Fundamental(s):** Unsupervised Anomaly Detection via Statistics; Rule-Based Systems.

**Minimal Solution Approach:** Load a CSV of 10,000 past expense line items (Amount, Category, Employee Dept, Date). For a new expense report, define an anomaly as any line item where: 1) The amount is > 3 standard deviations above the mean for that category/dept. OR 2) The merchant category is rarely used by that department (e.g., a software engineer at a pet store). This is a pure rule-based system derived from simple descriptive statistics on the historical data. **Forbidden:** No isolation forests, no one-class SVMs. The goal is to create an *explainable* flag ("This is high for your dept"), not a mysterious score. An employee needs to be able to defend themselves.

**Success Metrics:** 1) Percentage of flagged expenses that are recovered or disallowed upon audit. 2) Low false positive rate (flags that are explained by a legitimate reason, like a one-time conference). 3) The time taken to add a new rule.

**The "Over-Engineering" Trap:** Building a "machine learning" anomaly detection model. This turns the process into a black box, making it legally and culturally problematic when an employee asks "Why was I flagged?" A simple statistical threshold is just math; it's defensible.

---

### Project 10: Cross-Sectional Pairs Trading Signal Generator
**Real-World Context:** A junior trader has a hunch that two highly correlated stocks (e.g., Coca-Cola and Pepsi) have temporarily diverged. They need a disciplined, mathematical signal to tell them when the spread is wide enough to bet on a convergence.

**Core Fundamental(s):** Statistical Arbitrage Fundamentals; Stationarity & Cointegration.

**Minimal Solution Approach:** Take the daily closing prices for two highly correlated stocks (e.g., KO and PEP) for 2 years. Calculate the historical spread: `log(price_KO) - log(price_PEP)`. Fit a simple linear regression to estimate the hedge ratio. Calculate the rolling mean and standard deviation of the spread (using a 60-day window). Generate a signal to "buy the spread" when it is 2 standard deviations below the mean and "sell the spread" when it is 2 standard deviations above. **Forbidden:** Do not build a full backtesting engine. Do not use Kalman filters for a dynamic hedge ratio. Do not use cointegration tests beyond a simple check. The signal is just a z-score of the spread. The complexity is in the interpretation, not the code.

**Success Metrics:** 1) The statistical significance of the mean-reversion (autocorrelation of the spread). 2) The number of round-trip trade signals generated per year. 3) The maximum drawdown of the strategy during the signal-holding period.

**The "Over-Engineering" Trap:** Building a high-frequency, multi-asset pairs trading system. This project is about understanding the core, static concept of statistical arbitrage. Adding complexity like dynamic hedging or multiple pairs introduces numerous points of failure and obscures the fundamental lesson: mean reversion is a statistical property, not a guarantee.

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
