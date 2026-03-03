"""
Time-series forecast evaluation metrics.

Purpose:
Compute accuracy, stability, and residual diagnostics for
time-series predictions without modifying data or training models.

This module intentionally avoids:
- Model training
- Data transformation
- Plotting
- Persistence
"""

import numpy as np
from typing import Dict, Union, Optional
from scipy import stats


def compute_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Root Mean Square Error.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE value.

    Raises
    ------
    ValueError
        If inputs have invalid shapes or contain invalid values.
    """
    # Input validation
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    
    if len(y_true) == 0:
        raise ValueError("Empty arrays provided")
    
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Arrays contain NaN or infinite values")
    
    # Compute RMSE
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    return float(rmse)


def compute_fold_variance_percent(
    fold_rmse_values: np.ndarray
) -> float:
    """
    Compute coefficient of variation (CV) of RMSE across folds.

    Measures stability of model performance across time-series splits.
    Lower values indicate more stable performance.

    Parameters
    ----------
    fold_rmse_values : np.ndarray
        Array of RMSE values from different validation folds.

    Returns
    -------
    float
        Coefficient of variation (std/mean) as percentage.
        Returns 0.0 if mean RMSE is zero.

    Raises
    ------
    ValueError
        If input is empty or contains invalid values.
    """
    if len(fold_rmse_values) == 0:
        raise ValueError("Empty fold_rmse_values array")
    
    if not np.isfinite(fold_rmse_values).all():
        raise ValueError("fold_rmse_values contains NaN or infinite values")
    
    mean_rmse = np.mean(fold_rmse_values)
    
    if mean_rmse == 0:
        return 0.0
    
    std_rmse = np.std(fold_rmse_values)
    cv_percent = (std_rmse / mean_rmse) * 100
    
    return float(cv_percent)


def compute_residual_time_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_indices: Optional[np.ndarray] = None
) -> float:
    """
    Compute correlation between residuals and time.

    Detects if errors have temporal patterns (autocorrelation).
    Values near zero suggest residuals are independent of time.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    time_indices : np.ndarray, optional
        Time indices for correlation. If None, uses 0..n-1.

    Returns
    -------
    float
        Pearson correlation between residuals and time.
        Returns NaN if correlation cannot be computed.

    Raises
    ------
    ValueError
        If inputs have invalid shapes or contain invalid values.
    """
    # Input validation
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    
    if len(y_true) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(y_true)}")
    
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Arrays contain NaN or infinite values")
    
    # Compute residuals
    residuals = y_true - y_pred
    
    # Create time indices if not provided
    if time_indices is None:
        time_indices = np.arange(len(y_true))
    else:
        if time_indices.shape != y_true.shape:
            raise ValueError(
                f"time_indices shape {time_indices.shape} does not match "
                f"y_true shape {y_true.shape}"
            )
        if not np.isfinite(time_indices).all():
            raise ValueError("time_indices contains NaN or infinite values")
    
    # Compute correlation
    with np.errstate(invalid='ignore'):
        correlation_matrix = np.corrcoef(residuals, time_indices)
        correlation = correlation_matrix[0, 1]
    
    # Handle NaN (e.g., when residuals are constant)
    if np.isnan(correlation):
        return float('nan')
    
    return float(correlation)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fold_index: Optional[int] = None,
    fold_rmse_history: Optional[np.ndarray] = None,
    time_indices: Optional[np.ndarray] = None
) -> Dict[str, Union[float, int, None]]:
    """
    Compute comprehensive evaluation metrics for time-series forecast.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values for current fold.
    y_pred : np.ndarray
        Predicted values for current fold.
    fold_index : int, optional
        Current fold number (0-based). If provided, included in output.
    fold_rmse_history : np.ndarray, optional
        RMSE values from previous folds for stability calculation.
    time_indices : np.ndarray, optional
        Time indices for residual correlation.

    Returns
    -------
    Dict[str, Union[float, int, None]]
        Dictionary containing:
        - rmse: Root Mean Square Error for current fold
        - fold_rmse_variance_percent: Stability across folds (if history provided)
        - residual_time_correlation: Temporal pattern in residuals
        - fold_index: Current fold number (if provided)
        - n_samples: Number of samples evaluated

    Raises
    ------
    ValueError
        If input validation fails.
    """
    # Validate required inputs
    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred are required")
    
    # Compute basic metrics
    rmse = compute_rmse(y_true, y_pred)
    n_samples = len(y_true)
    
    # Initialize metrics dictionary
    metrics = {
        "rmse": rmse,
        "n_samples": n_samples,
        "fold_index": fold_index if fold_index is not None else None,
        "fold_rmse_variance_percent": None,
        "residual_time_correlation": None
    }
    
    # Compute residual-time correlation
    try:
        rt_correlation = compute_residual_time_correlation(
            y_true, y_pred, time_indices
        )
        metrics["residual_time_correlation"] = rt_correlation
    except (ValueError, RuntimeError):
        # If correlation fails, leave as None
        pass
    
    # Compute fold variance if history provided
    if fold_rmse_history is not None:
        # Include current fold's RMSE in history
        all_rmse_values = np.append(fold_rmse_history, rmse)
        try:
            var_percent = compute_fold_variance_percent(all_rmse_values)
            metrics["fold_rmse_variance_percent"] = var_percent
        except ValueError:
            # If variance computation fails, leave as None
            pass
    
    return metrics


# Minimal usage example
if __name__ == "__main__":
    # Create sample predictions for 3 folds
    np.random.seed(42)
    
    # Simulate 3 folds of predictions
    fold_rmse_history = []
    
    for fold in range(3):
        # Generate synthetic true values and predictions
        n_samples = 50
        y_true = np.random.randn(n_samples) * 10 + 100  # Base + noise
        y_pred = y_true + np.random.randn(n_samples) * 2  # Predictions with error
        
        # Add slight time trend to residuals for fold 2
        if fold == 2:
            time_trend = np.linspace(0, 5, n_samples)
            y_pred = y_true - time_trend  # Negative trend in residuals
        
        # Compute metrics
        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            fold_index=fold,
            fold_rmse_history=np.array(fold_rmse_history) if fold_rmse_history else None,
            time_indices=np.arange(n_samples)
        )
        
        # Store RMSE for next fold's variance calculation
        fold_rmse_history.append(metrics["rmse"])
        
        # Print results
        print(f"\nFold {fold}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Residual-Time Correlation: {metrics['residual_time_correlation']:.4f}")
        if metrics['fold_rmse_variance_percent'] is not None:
            print(f"  RMSE Variance %: {metrics['fold_rmse_variance_percent']:.2f}%")