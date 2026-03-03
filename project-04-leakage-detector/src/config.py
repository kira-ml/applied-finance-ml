from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR       = Path(__file__).resolve().parents[1]
DATA_RAW_PATH  = ROOT_DIR / "data" / "raw" / "revenue.csv"
ARTIFACTS_DIR  = ROOT_DIR / "artifacts"

MODEL_PATH          = ARTIFACTS_DIR / "model.pkl"
SCALER_PATH         = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_CONFIG_PATH = ARTIFACTS_DIR / "feature_config.json"

# ---------------------------------------------------------------------------
# Data Schema
# ---------------------------------------------------------------------------
DATE_COL    = "date"
TARGET_COL  = "revenue"
DATE_FORMAT = "%Y-%m-%d"

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
# Lag offsets (days). Every lag uses .shift(n) — zero look-ahead.
LAG_LIST = [1, 7, 14]

# Rolling windows computed on shift(1) series — no look-ahead.
ROLLING_WINDOWS = [7]          # drives rolling_mean_<w> and rolling_std_<w>

# Safe calendar features derived solely from the date index.
CALENDAR_FEATURES = ["day_of_week", "month"]

# ---------------------------------------------------------------------------
# Cross-Validation  (TimeSeriesSplit — expanding window)
# ---------------------------------------------------------------------------
N_SPLITS        = 5     # number of CV folds
VALIDATION_SIZE = 90    # rows per validation fold (≈ 3 months of daily data)

# ---------------------------------------------------------------------------
# Model  (Ridge regression)
# ---------------------------------------------------------------------------
RIDGE_ALPHA  = 1.0   # L2 regularisation strength
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Evaluation Thresholds
# ---------------------------------------------------------------------------
MAX_FOLD_RMSE_VARIANCE_PCT  = 5.0   # max allowed % change in RMSE fold-over-fold
MAX_RESIDUAL_TIME_CORR      = 0.05  # max allowed |Pearson r| between residuals & time index
