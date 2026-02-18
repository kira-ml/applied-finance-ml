from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ==============================================================================
# 1. IMMUTABLE CONFIGURATION DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class DetectorConfig:
    window_size: int
    threshold: float
    version: str = field(default="1.0.0", init=False)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ConfigValidationError("window_size must be positive")
        if self.threshold < 0.0:
            raise ConfigValidationError("threshold must be non-negative")

DEFAULT_CONFIG: DetectorConfig = DetectorConfig(window_size=5, threshold=0.01)

# ==============================================================================
# 2. CUSTOM EXCEPTION DEFINITIONS
# ==============================================================================

class DetectorError(Exception):
    """Base exception for detector module."""
    pass

class ConfigValidationError(DetectorError):
    """Raised when configuration constraints are violated."""
    pass

class InputContractViolationError(DetectorError):
    """Raised when input DataFrame structure or types are invalid."""
    pass

class NumericalInstabilityError(DetectorError):
    """Raised when numerical guards detect NaN, Inf, or division by zero."""
    pass

# ==============================================================================
# 3. TYPED DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class FlaggedEvent:
    asset: str
    date: str
    rolling_stdev: float

@dataclass(frozen=True)
class DetectionState:
    events: List[FlaggedEvent]
    config: DetectorConfig

# Type alias matching data_loader.py context
PriceDataFrame = pd.DataFrame

# ==============================================================================
# 4. PURE CORE LOGIC
# ==============================================================================

def _validate_input_contract(df: PriceDataFrame, config: DetectorConfig) -> None:
    """Explicit structural validation of input DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise InputContractViolationError("Input data must be a pandas DataFrame")
    
    if df.empty:
        raise InputContractViolationError("Input DataFrame is empty")

    # Validate Index (must be datetime as per data_loader.py output)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise InputContractViolationError("DataFrame index must be DatetimeIndex")

    # Validate Columns (assets)
    if len(df.columns) == 0:
        raise InputContractViolationError("DataFrame must have at least one asset column")
    
    for col in df.columns:
        if not isinstance(col, str):
            raise InputContractViolationError(f"Column name '{col}' is not a string")
        
        series = df[col]
        
        # Type check
        if not pd.api.types.is_numeric_dtype(series):
            raise InputContractViolationError(f"Column '{col}' is not numeric")
        
        # Numerical Stability Guards on raw data
        if series.isnull().any():
            raise InputContractViolationError(f"Column '{col}' contains null values")
        
        # Check for Inf/NaN explicitly in values
        if math.isinf(series.abs().max()):
            raise NumericalInstabilityError(f"Column '{col}' contains infinite values")

def _compute_rolling_stdev_series(values: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling standard deviation for a single asset series.
    Uses population standard deviation (ddof=0) for determinism.
    Returns Series with same index, NaN for warm-up period.
    """
    # Pandas rolling is deterministic given fixed window and ddof
    result = values.rolling(window=window, min_periods=window).std(ddof=0)
    
    # Explicit stability check on result
    if result.isnull().all():
        # Only acceptable if input was too short, otherwise error
        if len(values) >= window:
             raise NumericalInstabilityError("Rolling stdev computation resulted in all NaNs unexpectedly")
    
    # Check for any remaining NaNs that aren't due to warmup (should be covered by min_periods)
    # But we ensure no NaNs exist in the valid range later during iteration
    
    return result

def run_detection(df: PriceDataFrame, config: DetectorConfig) -> DetectionState:
    """
    Pure functional core executing the detection logic.
    Validates inputs, computes stats, applies threshold, returns state.
    """
    # 1. Validate Input Contract
    _validate_input_contract(df, config)
    
    flagged_events: List[FlaggedEvent] = []
    
    # 2. Process each asset deterministically
    # Sort columns for deterministic iteration order
    sorted_assets = sorted(df.columns.tolist())
    
    for asset in sorted_assets:
        series = df[asset]
        
        # 3. Compute Rolling Stdev
        stdev_series = _compute_rolling_stdev_series(series, config.window_size)
        
        # 4. Apply Threshold Rule
        # Iterate over index and values explicitly to ensure control
        for i in range(len(stdev_series)):
            # Get date from index
            date_val = stdev_series.index[i]
            stdev_val = stdev_series.iloc[i]
            
            # Skip warm-up period (NaN)
            if pd.isna(stdev_val):
                continue
            
            # Defensive Numerical Stability Guards (Re-check)
            if math.isnan(stdev_val) or math.isinf(stdev_val):
                raise NumericalInstabilityError(f"Invalid stdev computed for asset {asset} at {date_val}")
            
            # Threshold Comparison
            if stdev_val < config.threshold:
                # Format date as ISO string for consistent output type
                date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, 'strftime') else str(date_val)
                
                event = FlaggedEvent(
                    asset=asset,
                    date=date_str,
                    rolling_stdev=float(stdev_val)
                )
                flagged_events.append(event)
    
    return DetectionState(events=flagged_events, config=config)

# ==============================================================================
# 5. MINIMAL SIDE-EFFECT BOUNDARY FUNCTIONS
# ==============================================================================

def _guard_initialization() -> bool:
    """Idempotent setup guard. Returns True if ready."""
    # No global state to initialize
    return True

def detect_low_volatility(
    df: PriceDataFrame, 
    config: Optional[DetectorConfig] = None
) -> pd.DataFrame:
    """
    Public API Entry Point.
    Accepts price DataFrame and optional config.
    Returns a DataFrame of flagged events with columns: asset, date, rolling_stdev.
    Enforces types and handles setup.
    """
    # Runtime Type Guard for arguments
    if not isinstance(df, pd.DataFrame):
        raise InputContractViolationError("Argument 'df' must be a pandas DataFrame")
    
    if config is None:
        effective_config = DEFAULT_CONFIG
    else:
        if not isinstance(config, DetectorConfig):
            raise InputContractViolationError("Argument 'config' must be DetectorConfig")
        effective_config = config
        
    # Guarded Initialization
    if not _guard_initialization():
        raise DetectorError("Initialization failed")
        
    # Execute Core Logic
    state = run_detection(df, effective_config)
    
    # Construct Output DataFrame
    if not state.events:
        # Return empty DataFrame with correct schema and version metadata in attrs if needed
        # but strictly returning columns: asset, date, rolling_stdev
        result_df = pd.DataFrame(columns=["asset", "date", "rolling_stdev"])
        result_df.attrs["version"] = effective_config.version
        return result_df
    
    # Convert list of dataclasses to list of dicts
    records = [
        {
            "asset": e.asset,
            "date": e.date,
            "rolling_stdev": e.rolling_stdev
        }
        for e in state.events
    ]
    
    result_df = pd.DataFrame(records)
    
    # Ensure column order is explicit
    result_df = result_df[["asset", "date", "rolling_stdev"]]
    
    # Attach version for serialization contract visibility
    result_df.attrs["version"] = effective_config.version
    
    return result_df

# ==============================================================================
# 6. EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Minimal self-verification using synthetic data mimicking data_loader.py output
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    
    # Asset A: Constant price (stdev should be 0)
    # Asset B: Volatile price
    data = {
        "Asset_A": [100.0] * 10,
        "Asset_B": [50.0, 55.0, 49.0, 52.0, 51.0, 80.0, 82.0, 79.0, 81.0, 80.0]
    }
    
    mock_df = pd.DataFrame(data, index=dates)
    mock_df.index.name = "date" # Mimic data_loader setting index
    
    try:
        results = detect_low_volatility(mock_df, DetectorConfig(window_size=5, threshold=0.01))
        
        if not results.empty:
            print(f"Detected {len(results)} low volatility events:")
            print(results.to_string(index=False))
        else:
            print("No low volatility events detected.")
            
    except DetectorError as e:
        # Single Failure Mode
        print(f"FATAL: {e}")
        exit(1)