from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import pandas as pd

# Import internal modules explicitly
try:
    import data_loader
    import detector
    import alerts
    import evaluate
except ImportError as e:
    print(f"FATAL: Missing required module: {e}", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# 1. IMMUTABLE CONFIGURATION DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class PipelineConfig:
    prices_path: str
    ground_truth_path: str
    log_dir: str
    alert_log_file: str
    detector_window_size: int
    detector_threshold: float
    version: str = field(default="1.0.0", init=False)

    def __post_init__(self) -> None:
        if not self.prices_path.endswith(".csv"):
            raise ConfigValidationError("prices_path must end with .csv")
        if not self.ground_truth_path.endswith(".csv"):
            raise ConfigValidationError("ground_truth_path must end with .csv")
        if not self.log_dir:
            raise ConfigValidationError("log_dir cannot be empty")
        if self.detector_window_size <= 0:
            raise ConfigValidationError("detector_window_size must be positive")
        if self.detector_threshold < 0.0:
            raise ConfigValidationError("detector_threshold must be non-negative")

DEFAULT_CONFIG: PipelineConfig = PipelineConfig(
    prices_path=r"D:\applied-finance-ml\project-01-flat-variance-detector\data\prices.csv",
    ground_truth_path=r"D:\applied-finance-ml\project-01-flat-variance-detector\data\ground_truth.csv",
    log_dir="logs",
    alert_log_file="alerts.log",
    detector_window_size=5,
    detector_threshold=0.01
)

# ==============================================================================
# 2. CUSTOM EXCEPTION DEFINITIONS
# ==============================================================================

class PipelineError(Exception):
    """Base exception for pipeline orchestration failures."""
    pass

class ConfigValidationError(PipelineError):
    """Raised when pipeline configuration is invalid."""
    pass

class DirectoryCreationError(PipelineError):
    """Raised when logs directory cannot be created."""
    pass

class StepExecutionError(PipelineError):
    """Raised when a specific pipeline step fails."""
    pass

class GroundTruthTransformationError(PipelineError):
    """Raised when ground truth cannot be mapped to dates."""
    pass

# ==============================================================================
# 3. TYPED DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class PipelineState:
    prices_df: Optional[pd.DataFrame]
    events_df: Optional[pd.DataFrame]
    metrics: Optional[evaluate.EvaluationMetrics]
    config: PipelineConfig

# ==============================================================================
# 4. PURE CORE LOGIC
# ==============================================================================

def _validate_config(config: PipelineConfig) -> None:
    if not isinstance(config, PipelineConfig):
        raise ConfigValidationError("Config must be an instance of PipelineConfig")

def _ensure_log_directory(path: str) -> None:
    if os.path.exists(path):
        if not os.path.isdir(path):
            os.remove(path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            raise DirectoryCreationError(f"Failed to create log directory: {e}")

def _transform_ground_truth_to_dates(
    source_path: str, 
    prices_df: pd.DataFrame
) -> str:
    """
    Reads index-based ground truth, maps indices to dates from prices_df,
    and writes a temporary CSV with the correct schema for evaluate.py.
    Returns the path to the temporary file.
    """
    try:
        gt_df = pd.read_csv(source_path)
    except Exception as e:
        raise GroundTruthTransformationError(f"Failed to read ground truth: {e}")

    # Validate expected input columns for transformation
    required_src_cols = {"asset_id", "window_start_idx", "window_end_idx"}
    if not required_src_cols.issubset(set(gt_df.columns)):
        # If it already has the right columns, no transform needed
        if {"asset", "start_date", "end_date"}.issubset(set(gt_df.columns)):
            return source_path
        raise GroundTruthTransformationError(
            f"Ground truth missing required index columns: {required_src_cols - set(gt_df.columns)}"
        )

    # Ensure prices index is reset to allow integer indexing if it's currently DatetimeIndex
    # The ground truth uses integer row indices (0, 1, 2...) relative to the asset time series?
    # OR relative to the whole DataFrame? 
    # Context implies per-asset windows or global? 
    # Given "window_start_idx", usually implies row position.
    # Assumption: Indices are relative to the sorted DataFrame of ALL assets? 
    # NO, standard practice for such datasets is usually global row index IF single column, 
    # BUT here we have multiple assets.
    # Re-reading context: "Load a time-series CSV of daily closing prices for 10 stocks".
    # If prices_df has assets as COLUMNS and dates as INDEX (standard wide format):
    # Then the index 6 refers to the 6th row (date) for THAT asset column.
    
    # Let's assume Wide Format: Index=Date, Columns=Assets.
    # gt_df 'asset_id' matches column name.
    # gt_df 'window_start_idx' matches row positional index (iloc).
    
    transformed_rows = []
    
    # Reset index of prices to ensure clean positional access if needed, 
    # but iloc works on DatetimeIndex too.
    
    for _, row in gt_df.iterrows():
        asset = str(row["asset_id"])
        start_idx = int(row["window_start_idx"])
        end_idx = int(row["window_end_idx"])
        
        if asset not in prices_df.columns:
            # Skip unknown assets or fail? Fail for safety.
            raise GroundTruthTransformationError(f"Asset '{asset}' in ground truth not found in price data.")
            
        try:
            # Map integer index to actual Date label
            start_date_val = prices_df.index[start_idx]
            end_date_val = prices_df.index[end_idx]
            
            # Format as string YYYY-MM-DD
            start_str = start_date_val.strftime("%Y-%m-%d") if hasattr(start_date_val, 'strftime') else str(start_date_val)
            end_str = end_date_val.strftime("%Y-%m-%d") if hasattr(end_date_val, 'strftime') else str(end_date_val)
            
            transformed_rows.append({
                "asset": asset,
                "start_date": start_str,
                "end_date": end_str
            })
        except IndexError:
            raise GroundTruthTransformationError(
                f"Index out of bounds for asset {asset}: range [{start_idx}, {end_idx}]"
            )

    if not transformed_rows:
        raise GroundTruthTransformationError("No valid rows generated from ground truth transformation.")

    temp_df = pd.DataFrame(transformed_rows)
    
    # Write to temporary file in the same directory to ensure permissions match
    temp_dir = os.path.dirname(source_path)
    if not temp_dir:
        temp_dir = "."
        
    temp_file_path = os.path.join(temp_dir, "ground_truth_transformed_temp.csv")
    
    try:
        temp_df.to_csv(temp_file_path, index=False)
    except Exception as e:
        raise GroundTruthTransformationError(f"Failed to write temporary ground truth: {e}")
        
    return temp_file_path

# ==============================================================================
# 5. MINIMAL SIDE-EFFECT BOUNDARY FUNCTIONS
# ==============================================================================

_setup_complete = False

def _guard_initialization() -> bool:
    global _setup_complete
    if _setup_complete:
        return True
    _setup_complete = True
    return True

def run_pipeline(config: Optional[PipelineConfig] = None) -> PipelineState:
    if config is None:
        effective_config = DEFAULT_CONFIG
    else:
        if not isinstance(config, PipelineConfig):
            raise ConfigValidationError("Argument 'config' must be PipelineConfig")
        effective_config = config

    if not _guard_initialization():
        raise PipelineError("Initialization failed")

    _validate_config(effective_config)

    # Step 0: Ensure Log Directory
    _ensure_log_directory(effective_config.log_dir)

    alert_log_path = os.path.join(effective_config.log_dir, effective_config.alert_log_file)
    
    loader_cfg = data_loader.Config(
        prices_path=effective_config.prices_path,
        date_column="date",
        serialization_version=1
    )
    
    detector_cfg = detector.DetectorConfig(
        window_size=effective_config.detector_window_size,
        threshold=effective_config.detector_threshold
    )
    
    alert_cfg = alerts.AlertConfig(
        log_file_path=alert_log_path
    )
    
    # Placeholder for eval config, will update path after transformation
    eval_cfg_base_path = effective_config.ground_truth_path

    # Step 1: Load Data
    try:
        prices_df = data_loader.load_prices(loader_cfg)
    except data_loader.DataLoaderError as e:
        raise StepExecutionError(f"Data Loader failed: {e}")

    # Step 2: Detect Anomalies
    try:
        events_df = detector.detect_low_volatility(prices_df, detector_cfg)
    except detector.DetectorError as e:
        raise StepExecutionError(f"Detector failed: {e}")

    # Step 3: Generate Alerts
    try:
        alerts.process_alerts(events_df, alert_cfg)
    except alerts.AlertError as e:
        raise StepExecutionError(f"Alerts failed: {e}")

    # Step 3.5: Transform Ground Truth (Index -> Date)
    try:
        transformed_gt_path = _transform_ground_truth_to_dates(eval_cfg_base_path, prices_df)
    except GroundTruthTransformationError as e:
        raise StepExecutionError(f"Ground Truth transformation failed: {e}")
    
    eval_cfg = evaluate.EvalConfig(ground_truth_path=transformed_gt_path)

    # Step 4: Evaluate Performance
    try:
        metrics = evaluate.evaluate_detection(events_df, eval_cfg)
    except evaluate.EvalError as e:
        raise StepExecutionError(f"Evaluation failed: {e}")
    finally:
        # Cleanup temporary file if it was created (i.e., if path changed)
        if transformed_gt_path != eval_cfg_base_path:
            try:
                os.remove(transformed_gt_path)
            except OSError:
                pass # Ignore cleanup errors in finally block

    return PipelineState(
        prices_df=prices_df,
        events_df=events_df,
        metrics=metrics,
        config=effective_config
    )

# ==============================================================================
# 6. EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        state = run_pipeline()
        
        print("-" * 40)
        print("PIPELINE EXECUTION SUCCESSFUL")
        print(f"Version: {state.config.version}")
        print(f"Events Detected: {len(state.events_df)}")
        print(f"Precision: {state.metrics.precision:.4f}")
        print(f"Recall: {state.metrics.recall:.4f}")
        print(f"Avg Latency: {state.metrics.avg_latency_days:.2f} days")
        print(f"Alerts written to: {os.path.join(state.config.log_dir, state.config.alert_log_file)}")
        
    except PipelineError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)