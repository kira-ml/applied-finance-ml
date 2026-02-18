from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# ==============================================================================
# 1. IMMUTABLE CONFIGURATION DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class EvalConfig:
    ground_truth_path: str
    version: str = field(default="1.0.0", init=False)

    def __post_init__(self) -> None:
        if not self.ground_truth_path.endswith(".csv"):
            raise ConfigValidationError("ground_truth_path must end with .csv")
        if not self.ground_truth_path:
            raise ConfigValidationError("ground_truth_path cannot be empty")

DEFAULT_CONFIG: EvalConfig = EvalConfig(ground_truth_path="ground_truth.csv")

# ==============================================================================
# 2. CUSTOM EXCEPTION DEFINITIONS
# ==============================================================================

class EvalError(Exception):
    """Base exception for evaluation module."""
    pass

class ConfigValidationError(EvalError):
    """Raised when configuration constraints are violated."""
    pass

class InputContractViolationError(EvalError):
    """Raised when input DataFrames violate structural expectations."""
    pass

class GroundTruthFormatError(EvalError):
    """Raised when ground truth file format is invalid."""
    pass

# ==============================================================================
# 3. TYPED DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class EvaluationMetrics:
    precision: float
    recall: float
    avg_latency_days: float
    version: str

@dataclass(frozen=True)
class GroundTruthWindow:
    asset: str
    start_date: str
    end_date: str

# Type aliases
EventsDataFrame = pd.DataFrame
GroundTruthDataFrame = pd.DataFrame

# ==============================================================================
# 4. PURE CORE LOGIC
# ==============================================================================

def _validate_events_contract(df: EventsDataFrame) -> None:
    """Validate detector output structure."""
    if not isinstance(df, pd.DataFrame):
        raise InputContractViolationError("Events input must be a pandas DataFrame")
    
    required = {"asset", "date", "rolling_stdev"}
    missing = required - set(df.columns)
    if missing:
        raise InputContractViolationError(f"Events missing columns: {missing}")
    
    if df.empty:
        return

    if not pd.api.types.is_string_dtype(df["asset"]) and not df["asset"].apply(lambda x: isinstance(x, str)).all():
        raise InputContractViolationError("Events 'asset' must be strings")
    if not pd.api.types.is_string_dtype(df["date"]) and not df["date"].apply(lambda x: isinstance(x, str)).all():
        raise InputContractViolationError("Events 'date' must be strings")

def _validate_ground_truth_contract(df: GroundTruthDataFrame) -> None:
    """Validate ground truth structure."""
    if not isinstance(df, pd.DataFrame):
        raise InputContractViolationError("Ground truth input must be a pandas DataFrame")
    
    required = {"asset", "start_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        raise GroundTruthFormatError(f"Ground truth missing columns: {missing}")
    
    if df.empty:
        return

    if not pd.api.types.is_string_dtype(df["asset"]) and not df["asset"].apply(lambda x: isinstance(x, str)).all():
        raise GroundTruthFormatError("Ground truth 'asset' must be strings")
    if not pd.api.types.is_string_dtype(df["start_date"]) or not pd.api.types.is_string_dtype(df["end_date"]):
         if not (df["start_date"].apply(lambda x: isinstance(x, str)).all() and df["end_date"].apply(lambda x: isinstance(x, str)).all()):
            raise GroundTruthFormatError("Ground truth dates must be strings")

def _parse_ground_truth_windows(df: GroundTruthDataFrame) -> List[GroundTruthWindow]:
    """Convert DF to list of immutable window objects."""
    windows: List[GroundTruthWindow] = []
    for _, row in df.iterrows():
        w = GroundTruthWindow(
            asset=str(row["asset"]),
            start_date=str(row["start_date"]),
            end_date=str(row["end_date"])
        )
        windows.append(w)
    return windows

def _calculate_metrics(
    true_windows: List[GroundTruthWindow],
    detected_events: List[Tuple[str, str]] # (asset, date)
) -> EvaluationMetrics:
    """
    Pure function to compute Precision, Recall, and Avg Latency.
    
    Logic:
    - True Positive (TP): A detected event falls within a true window.
    - False Positive (FP): A detected event falls outside any true window.
    - False Negative (FN): A true window has NO detected events within it.
    - Latency: For each true window, find the first detection date inside it. 
      Latency = (First Detection Date - Start Date) in days.
    """
    
    # Index true windows for fast lookup: Map[Asset, List[Windows]]
    true_windows_by_asset: Dict[str, List[GroundTruthWindow]] = {}
    for w in true_windows:
        if w.asset not in true_windows_by_asset:
            true_windows_by_asset[w.asset] = []
        true_windows_by_asset[w.asset].append(w)
    
    tp_count = 0
    fp_count = 0
    latencies: List[float] = []
    
    # Track which windows have been hit to calculate FN later
    # Using a set of indices relative to the original list isn't easy due to grouping.
    # Instead, we mark windows as 'hit' in a parallel structure or just re-scan.
    # Optimized: Map[Asset, Map[WindowStart, HitStatus]]
    window_hit_status: Dict[str, Dict[str, bool]] = {}
    
    for w in true_windows:
        if w.asset not in window_hit_status:
            window_hit_status[w.asset] = {}
        window_hit_status[w.asset][w.start_date] = False
    
    # Process detections
    for asset, date_str in detected_events:
        is_tp = False
        
        if asset in true_windows_by_asset:
            for w in true_windows_by_asset[asset]:
                # Simple string comparison for dates (ISO format YYYY-MM-DD sorts lexicographically)
                if w.start_date <= date_str <= w.end_date:
                    is_tp = True
                    window_hit_status[asset][w.start_date] = True
                    break # One event can only satisfy one window instance? 
                          # Actually, if multiple windows overlap, one event might count for multiple?
                          # Standard definition: An event is TP if it matches ANY truth.
                          # But for latency, we need specific mapping.
                          # Simplification: Event counts as 1 TP. It satisfies the first matching window found.
                          # To be rigorous: We need to assign events to windows optimally?
                          # Given "radical simplicity": An event is TP if inside ANY window.
                          # A window is Recalled if it has >= 1 event.
        
        if is_tp:
            tp_count += 1
        else:
            fp_count += 1
            
    # Calculate Recall and Latency
    # Iterate all true windows again to see if they were hit
    total_true_windows = len(true_windows)
    recalled_count = 0
    
    for w in true_windows:
        if window_hit_status[w.asset][w.start_date]:
            recalled_count += 1
            
            # Calculate Latency for this window
            # Find the earliest detection date for this asset that falls in this window
            min_lat = math.inf
            found_any = False
            
            for a, d in detected_events:
                if a == w.asset and w.start_date <= d <= w.end_date:
                    # Convert to ordinal for math
                    # Assuming ISO format YYYY-MM-DD
                    try:
                        d_ord = pd.to_datetime(d).toordinal()
                        s_ord = pd.to_datetime(w.start_date).toordinal()
                        lat = float(d_ord - s_ord)
                        if lat < min_lat:
                            min_lat = lat
                        found_any = True
                    except Exception:
                        # Should be caught by validation, but defensive guard
                        continue
            
            if found_any and min_lat != math.inf:
                latencies.append(min_lat)

    # Compute Metrics
    precision = 0.0
    if (tp_count + fp_count) > 0:
        precision = float(tp_count) / float(tp_count + fp_count)
    
    recall = 0.0
    if total_true_windows > 0:
        recall = float(recalled_count) / float(total_true_windows)
    
    avg_latency = 0.0
    if latencies:
        avg_latency = sum(latencies) / float(len(latencies))
    
    # Numerical Stability Guards
    if math.isnan(precision) or math.isinf(precision): precision = 0.0
    if math.isnan(recall) or math.isinf(recall): recall = 0.0
    if math.isnan(avg_latency) or math.isinf(avg_latency): avg_latency = 0.0
    
    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        avg_latency_days=avg_latency,
        version="1.0.0" # Hardcoded version for metrics contract
    )

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

def evaluate_detection(
    events_df: EventsDataFrame,
    config: Optional[EvalConfig] = None
) -> EvaluationMetrics:
    """
    Public API Entry Point.
    Loads ground truth, compares with events, prints metrics.
    """
    # Runtime Type Guard
    if not isinstance(events_df, pd.DataFrame):
        raise InputContractViolationError("Argument 'events_df' must be pandas DataFrame")
    
    if config is None:
        effective_config = DEFAULT_CONFIG
    else:
        if not isinstance(config, EvalConfig):
            raise InputContractViolationError("Argument 'config' must be EvalConfig")
        effective_config = config
        
    if not _guard_initialization():
        raise EvalError("Initialization failed")
        
    # Validate Events
    _validate_events_contract(events_df)
    
    # Load Ground Truth (I/O Boundary)
    try:
        gt_df = pd.read_csv(effective_config.ground_truth_path)
    except FileNotFoundError:
        raise GroundTruthFormatError(f"Ground truth file not found: {effective_config.ground_truth_path}")
    except Exception as e:
        raise GroundTruthFormatError(f"Failed to read ground truth: {e}")
        
    _validate_ground_truth_contract(gt_df)
    
    # Parse to Pure Structures
    true_windows = _parse_ground_truth_windows(gt_df)
    
    # Normalize Events to List of Tuples for pure processing
    # Sort to ensure deterministic processing order if needed (though logic is set-based mostly)
    detected_events: List[Tuple[str, str]] = []
    if not events_df.empty:
        sorted_events = events_df.sort_values(by=["asset", "date"])
        for _, row in sorted_events.iterrows():
            detected_events.append((str(row["asset"]), str(row["date"])))
    
    # Compute Metrics
    metrics = _calculate_metrics(true_windows, detected_events)
    
    # Print Results (Human Readable)
    print(f"EVALUATION METRICS (Version: {metrics.version})")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall:    {metrics.recall:.4f}")
    print(f"Avg Latency (days): {metrics.avg_latency_days:.2f}")
    
    return metrics

# ==============================================================================
# 6. EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Mock Data for Self-Verification
    # Simulating 1 True Window: Asset_A from 2023-01-05 to 2023-01-10
    # Simulating Detections: 
    #   - Asset_A on 2023-01-06 (TP, Latency=1)
    #   - Asset_A on 2023-01-07 (TP, redundant for recall, counts as TP? Yes, logic above counts all inside as TP)
    #   - Asset_B on 2023-01-05 (FP, no truth for B)
    
    mock_events = pd.DataFrame({
        "asset": ["Asset_A", "Asset_A", "Asset_B"],
        "date": ["2023-01-06", "2023-01-07", "2023-01-05"],
        "rolling_stdev": [0.0, 0.0, 0.0]
    })
    
    mock_gt = pd.DataFrame({
        "asset": ["Asset_A"],
        "start_date": ["2023-01-05"],
        "end_date": ["2023-01-10"]
    })
    
    # Write mock ground truth to disk for the function to load
    mock_gt.to_csv("ground_truth.csv", index=False)
    
    try:
        result = evaluate_detection(mock_events, EvalConfig(ground_truth_path="ground_truth.csv"))
        
        # Assertions for self-verification
        # TP=2 (both A events inside), FP=1 (B event). Precision = 2/3 = 0.6667
        # Recall: 1 window, hit? Yes. Recall = 1.0
        # Latency: Window A started 05. First detect 06. Latency = 1 day.
        
        assert abs(result.precision - (2.0/3.0)) < 0.0001, "Precision calculation error"
        assert abs(result.recall - 1.0) < 0.0001, "Recall calculation error"
        assert abs(result.avg_latency_days - 1.0) < 0.0001, "Latency calculation error"
        
    except EvalError as e:
        print(f"FATAL: {e}")
        exit(1)
    finally:
        # Cleanup mock file
        import os
        if os.path.exists("ground_truth.csv"):
            os.remove("ground_truth.csv")