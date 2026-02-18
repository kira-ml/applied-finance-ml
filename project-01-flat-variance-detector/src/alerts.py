from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Set

import pandas as pd

# ==============================================================================
# 1. IMMUTABLE CONFIGURATION DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class AlertConfig:
    log_file_path: str
    version: str = field(default="1.0.0", init=False)

    def __post_init__(self) -> None:
        if not self.log_file_path.endswith(".log"):
            raise ConfigValidationError("log_file_path must end with .log")
        if not self.log_file_path:
            raise ConfigValidationError("log_file_path cannot be empty")

DEFAULT_CONFIG: AlertConfig = AlertConfig(log_file_path="alerts.log")

# ==============================================================================
# 2. CUSTOM EXCEPTION DEFINITIONS
# ==============================================================================

class AlertError(Exception):
    """Base exception for alerting module."""
    pass

class ConfigValidationError(AlertError):
    """Raised when configuration constraints are violated."""
    pass

class InputContractViolationError(AlertError):
    """Raised when input DataFrame structure is invalid."""
    pass

class IoOperationError(AlertError):
    """Raised when file writing fails."""
    pass

# ==============================================================================
# 3. TYPED DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class AlertSummary:
    total_events: int
    unique_assets: Set[str]
    config_version: str

# Type alias matching detector.py output
EventsDataFrame = pd.DataFrame

# ==============================================================================
# 4. PURE CORE LOGIC
# ==============================================================================

def _validate_input_contract(df: EventsDataFrame) -> None:
    """Explicit structural validation of input DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise InputContractViolationError("Input must be a pandas DataFrame")
    
    required_columns = {"asset", "date", "rolling_stdev"}
    missing = required_columns - set(df.columns)
    if missing:
        raise InputContractViolationError(f"Missing required columns: {missing}")
    
    if df.empty:
        # Empty is valid, just no events to process, but structure must be correct
        return

    # Type checks on first row if not empty to save cost, assuming homogeneity
    # Strictly speaking, pandas dtypes should be checked
    if not pd.api.types.is_string_dtype(df["asset"]):
        # Allow object dtype which often holds strings in pandas
        if df["asset"].apply(lambda x: not isinstance(x, str)).any():
            raise InputContractViolationError("Column 'asset' must contain only strings")
            
    if not pd.api.types.is_string_dtype(df["date"]):
        if df["date"].apply(lambda x: not isinstance(x, str)).any():
            raise InputContractViolationError("Column 'date' must contain only strings")
            
    if not pd.api.types.is_numeric_dtype(df["rolling_stdev"]):
        raise InputContractViolationError("Column 'rolling_stdev' must be numeric")

def _generate_summary(df: EventsDataFrame, config_version: str) -> AlertSummary:
    """Pure function to compute summary statistics."""
    if df.empty:
        return AlertSummary(total_events=0, unique_assets=set(), config_version=config_version)
    
    count = len(df)
    assets = set(df["asset"].tolist())
    
    return AlertSummary(total_events=count, unique_assets=assets, config_version=config_version)

def _format_log_entry(row: dict) -> str:
    """Formats a single row into a structured log string."""
    # Explicit casting to ensure no silent truncation or repr issues
    asset = str(row["asset"])
    date = str(row["date"])
    stdev = float(row["rolling_stdev"])
    
    # Structured format: KEY=VALUE pairs for easy parsing
    return f"EVENT|asset={asset}|date={date}|rolling_stdev={stdev:.8f}"

# ==============================================================================
# 5. MINIMAL SIDE-EFFECT BOUNDARY FUNCTIONS
# ==============================================================================

_setup_complete = False

def _guard_initialization() -> bool:
    """Idempotent setup guard."""
    global _setup_complete
    if _setup_complete:
        return True
    
    # No complex setup needed, just state flag
    _setup_complete = True
    return True

def process_alerts(
    df: EventsDataFrame, 
    config: Optional[AlertConfig] = None
) -> AlertSummary:
    """
    Public API Entry Point.
    Accepts flagged events DataFrame, writes logs, prints summary.
    """
    # Runtime Type Guard
    if not isinstance(df, pd.DataFrame):
        raise InputContractViolationError("Argument 'df' must be a pandas DataFrame")
    
    if config is None:
        effective_config = DEFAULT_CONFIG
    else:
        if not isinstance(config, AlertConfig):
            raise InputContractViolationError("Argument 'config' must be AlertConfig")
        effective_config = config
        
    # Guarded Initialization
    if not _guard_initialization():
        raise AlertError("Initialization failed")
        
    # Validate Input Contract
    _validate_input_contract(df)
    
    # Generate Pure Summary Data (before side effects)
    summary = _generate_summary(df, effective_config.version)
    
    # I/O Boundary: Logging Setup and Execution
    # Configure logger strictly for this run
    logger = logging.getLogger("alert_processor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear() # Ensure clean state
    
    # File Handler
    try:
        file_handler = logging.FileHandler(effective_config.log_file_path, mode='w', encoding='utf-8')
        formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        raise IoOperationError(f"Failed to open log file: {e}")
    
    try:
        # Write Records
        if not df.empty:
            # Deterministic iteration: sort by asset then date to ensure reproducible log order
            sorted_df = df.sort_values(by=["asset", "date"]).reset_index(drop=True)
            
            for _, row in sorted_df.iterrows():
                log_line = _format_log_entry(row)
                logger.info(log_line)
        
        # Console Summary (Human Readable)
        asset_list = ", ".join(sorted(summary.unique_assets)) if summary.unique_assets else "None"
        print(f"ALERT SUMMARY: {summary.total_events} events flagged.")
        print(f"AFFECTED ASSETS: {asset_list}")
        
    finally:
        # Mechanical Resource Lifetime Management
        logger.handlers.clear()
        file_handler.close()
    
    return summary

# ==============================================================================
# 6. EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Mock Data mimicking detector.py output
    mock_data = {
        "asset": ["Asset_A", "Asset_A", "Asset_B"],
        "date": ["2023-01-05", "2023-01-06", "2023-01-05"],
        "rolling_stdev": [0.0, 0.0, 0.005]
    }
    mock_df = pd.DataFrame(mock_data)
    
    try:
        # Run with default config (writes to alerts.log in current dir)
        result = process_alerts(mock_df)
        
        # Verify version encapsulation
        assert result.config_version == "1.0.0", "Version mismatch"
        
    except AlertError as e:
        # Single Failure Mode
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)