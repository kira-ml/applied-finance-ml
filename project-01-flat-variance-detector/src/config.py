# config.py
# Central configuration for the flat/zero-variance transaction monitor.
# All parameters live here. Change values here only — never hardcode elsewhere.

import os

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR          = r"D:\applied-finance-ml\project-01-flat-variance-detector"

DATA_PATH         = os.path.join(BASE_DIR, "data", "prices.csv")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "data", "ground_truth.csv")
LOG_PATH          = os.path.join(BASE_DIR, "logs", "alerts.log")

# ── Reproducibility ──────────────────────────────────────────────────────────

RANDOM_SEED       = 42

# ── Data Generation ──────────────────────────────────────────────────────────

NUM_ASSETS        = 10
NUM_DAYS          = 730
START_DATE        = "2022-01-03"

# ── Detection ────────────────────────────────────────────────────────────────

WINDOW            = 5     # Rolling window size in trading days
THRESHOLD         = 0.01  # StdDev below this flags the asset as stuck