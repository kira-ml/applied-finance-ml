"""
src/evaluate.py

Core Responsibilities:
- Calculate Brier Score for uncalibrated and calibrated predictions.
- Compute calibration error using decile binning.
- Verify Brier Score improvement meets threshold (>15%).
- Generate metrics dictionary and save to JSON.

Constraints:
- No visualization/plotting.
- No model training or data modification.
- Accept arrays/DataFrames as arguments (no file loading).
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Dict, Final, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

# Type Aliases
NumericArray = np.ndarray
Target = Union[pd.Series, np.ndarray]

_N_BINS: Final[int] = 10  # Deciles for calibration error


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class EvaluationError(Exception):
    """Base exception for evaluation failures."""
    pass


class MetricsComputationError(EvaluationError):
    """Raised when metric calculation fails."""
    pass


class ThresholdValidationError(EvaluationError):
    """Raised when success criteria are not met."""
    pass


# -----------------------------------------------------------------------------
# Core Metrics Functions
# -----------------------------------------------------------------------------

def calculate_brier_score(
    y_true: Target,
    y_proba: NumericArray
) -> float:
    """
    Calculates the Brier score for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_proba: Predicted probabilities for the positive class.
        
    Returns:
        Brier score (lower is better, range [0, 1]).
        
    Raises:
        MetricsComputationError: If calculation fails.
    """
    if len(y_true) == 0:
        raise MetricsComputationError("Cannot calculate Brier score on empty data")
    
    if len(y_true) != len(y_proba):
        raise MetricsComputationError(
            f"Length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}"
        )
    
    try:
        # Extract positive class probabilities if 2D array
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
        
        score = brier_score_loss(y_true, y_proba)
        return float(score)
    except Exception as e:
        raise MetricsComputationError(f"Brier score calculation failed: {e}") from e


def calculate_calibration_error(
    y_true: Target,
    y_proba: NumericArray,
    n_bins: int = _N_BINS
) -> float:
    """
    Calculates calibration error using binned predictions.
    
    Groups predictions into bins (default: deciles) and computes the mean
    absolute difference between predicted probabilities and actual outcomes.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_proba: Predicted probabilities for the positive class.
        n_bins: Number of bins for grouping predictions.
        
    Returns:
        Mean calibration error across bins.
        
    Raises:
        MetricsComputationError: If calculation fails.
    """
    if len(y_true) == 0:
        raise MetricsComputationError("Cannot calculate calibration error on empty data")
    
    if len(y_true) != len(y_proba):
        raise MetricsComputationError(
            f"Length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}"
        )
    
    try:
        # Extract positive class probabilities if 2D array
        if y_proba.ndim == 2:
            y_proba = y_proba[:, 1]
        
        # Create bins based on quantiles
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bin_edges[1:-1])
        
        calibration_errors = []
        
        for i in range(n_bins):
            bin_mask = (bin_indices == i)
            
            if np.sum(bin_mask) == 0:
                continue  # Skip empty bins
            
            bin_y_true = np.array(y_true)[bin_mask]
            bin_y_proba = y_proba[bin_mask]
            
            mean_predicted = np.mean(bin_y_proba)
            mean_actual = np.mean(bin_y_true)
            
            calibration_errors.append(abs(mean_predicted - mean_actual))
        
        if len(calibration_errors) == 0:
            raise MetricsComputationError("No valid bins for calibration error calculation")
        
        return float(np.mean(calibration_errors))
    except MetricsComputationError:
        raise
    except Exception as e:
        raise MetricsComputationError(f"Calibration error calculation failed: {e}") from e


def calculate_improvement_percentage(
    baseline_score: float,
    improved_score: float
) -> float:
    """
    Calculates percentage improvement from baseline to improved score.
    
    For Brier score (lower is better):
    improvement% = (baseline - improved) / baseline * 100
    
    Args:
        baseline_score: Original score.
        improved_score: New score after calibration.
        
    Returns:
        Percentage improvement (positive means better).
    """
    if baseline_score == 0:
        return 0.0
    
    improvement = ((baseline_score - improved_score) / baseline_score) * 100
    return float(improvement)


# -----------------------------------------------------------------------------
# Validation Functions
# -----------------------------------------------------------------------------

def validate_improvement_threshold(
    improvement_pct: float,
    min_threshold_pct: float = 15.0
) -> bool:
    """
    Validates if improvement meets minimum threshold.
    
    Args:
        improvement_pct: Achieved improvement percentage.
        min_threshold_pct: Minimum required improvement (default: 15%).
        
    Returns:
        True if threshold is met, False otherwise.
    """
    return improvement_pct >= min_threshold_pct


def validate_calibration_error(
    calibration_error: float,
    max_threshold: float = 0.05
) -> bool:
    """
    Validates if calibration error is below maximum threshold.
    
    Args:
        calibration_error: Computed calibration error.
        max_threshold: Maximum acceptable error (default: 0.05).
        
    Returns:
        True if below threshold, False otherwise.
    """
    return calibration_error <= max_threshold


# -----------------------------------------------------------------------------
# Metrics Report Generation
# -----------------------------------------------------------------------------

def generate_metrics_report(
    y_test: Target,
    y_proba_base: NumericArray,
    y_proba_calibrated: NumericArray,
    min_improvement_threshold: float = 15.0,
    max_calibration_error_threshold: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Generates a comprehensive metrics report comparing base and calibrated models.
    
    Args:
        y_test: True test labels.
        y_proba_base: Predicted probabilities from base model.
        y_proba_calibrated: Predicted probabilities from calibrated model.
        min_improvement_threshold: Minimum required Brier score improvement (%).
        max_calibration_error_threshold: Maximum acceptable calibration error.
        
    Returns:
        Dictionary containing all evaluation metrics and validation results.
    """
    # Calculate Brier scores
    brier_base = calculate_brier_score(y_test, y_proba_base)
    brier_calibrated = calculate_brier_score(y_test, y_proba_calibrated)
    
    # Calculate improvement
    improvement_pct = calculate_improvement_percentage(brier_base, brier_calibrated)
    
    # Calculate calibration errors
    cal_error_base = calculate_calibration_error(y_test, y_proba_base)
    cal_error_calibrated = calculate_calibration_error(y_test, y_proba_calibrated)
    
    # Validate thresholds
    meets_improvement = validate_improvement_threshold(
        improvement_pct, min_improvement_threshold
    )
    meets_calibration = validate_calibration_error(
        cal_error_calibrated, max_calibration_error_threshold
    )
    
    # Overall pass/fail
    overall_pass = meets_improvement and meets_calibration
    
    report = {
        "brier_score_base": round(brier_base, 6),
        "brier_score_calibrated": round(brier_calibrated, 6),
        "brier_improvement_pct": round(improvement_pct, 2),
        "calibration_error_base": round(cal_error_base, 6),
        "calibration_error_calibrated": round(cal_error_calibrated, 6),
        "threshold_improvement_min": min_improvement_threshold,
        "threshold_calibration_max": max_calibration_error_threshold,
        "meets_improvement_threshold": meets_improvement,
        "meets_calibration_threshold": meets_calibration,
        "overall_success": overall_pass,
        "status": "PASS" if overall_pass else "FAIL"
    }
    
    return report


def save_metrics_to_json(
    metrics: Dict[str, Union[float, bool, str]],
    output_path: Path
) -> None:
    """
    Saves metrics dictionary to JSON file.
    
    Args:
        metrics: Metrics dictionary to save.
        output_path: Path where JSON file will be saved.
        
    Raises:
        EvaluationError: If saving fails.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        raise EvaluationError(f"Failed to save metrics to {output_path}: {e}") from e


def evaluate_and_save(
    y_test: Target,
    y_proba_base: NumericArray,
    y_proba_calibrated: NumericArray,
    output_path: Path,
    min_improvement_threshold: float = 15.0,
    max_calibration_error_threshold: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    End-to-end evaluation: generate metrics and save to JSON.
    
    Args:
        y_test: True test labels.
        y_proba_base: Predicted probabilities from base model.
        y_proba_calibrated: Predicted probabilities from calibrated model.
        output_path: Path to save metrics JSON.
        min_improvement_threshold: Minimum Brier score improvement (%).
        max_calibration_error_threshold: Maximum calibration error.
        
    Returns:
        Metrics dictionary.
    """
    metrics = generate_metrics_report(
        y_test=y_test,
        y_proba_base=y_proba_base,
        y_proba_calibrated=y_proba_calibrated,
        min_improvement_threshold=min_improvement_threshold,
        max_calibration_error_threshold=max_calibration_error_threshold
    )
    
    save_metrics_to_json(metrics, output_path)
    
    return metrics
