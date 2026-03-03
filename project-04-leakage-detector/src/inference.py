"""
Time-series inference module for revenue forecasting.

Purpose:
Load trained artifacts and generate recursive forecasts
using recent historical data without retraining.

This module intentionally avoids:
- Model training
- Artifact modification
- Data persistence
- Cross-validation
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Union, Optional, Tuple, List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def load_artifacts(
    model_path: Union[str, Path],
    scaler_path: Union[str, Path]
) -> Tuple[Ridge, StandardScaler]:
    """
    Load trained model and scaler from disk.

    Parameters
    ----------
    model_path : str or Path
        Path to saved Ridge model (joblib format).
    scaler_path : str or Path
        Path to saved StandardScaler (joblib format).

    Returns
    -------
    Tuple[Ridge, StandardScaler]
        Loaded model and scaler.

    Raises
    ------
    FileNotFoundError
        If artifact files don't exist.
    ValueError
        If loaded artifacts are of wrong type.
    """
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise ValueError(f"Failed to load artifacts: {e}")
    
    # Validate artifact types
    if not isinstance(model, Ridge):
        raise ValueError(f"Loaded model is not a Ridge instance, got {type(model)}")
    
    if not isinstance(scaler, StandardScaler):
        raise ValueError(f"Loaded scaler is not a StandardScaler instance, got {type(scaler)}")
    
    return model, scaler


def build_features_from_history(
    history: np.ndarray,
    lags: List[int],
    rolling_windows: Optional[List[int]] = None,
    include_time_features: bool = False,
    current_date: Optional[pd.Timestamp] = None
) -> np.ndarray:
    """
    Build feature vector from recent history for inference.

    Parameters
    ----------
    history : np.ndarray
        Recent historical revenue values (most recent last).
        Must have at least max(lags) + max(rolling_windows) values.
    lags : List[int]
        Lag periods used during training.
    rolling_windows : List[int], optional
        Rolling window sizes used during training.
    include_time_features : bool
        Whether time features were used during training.
    current_date : pd.Timestamp, optional
        Current date for time features. Required if include_time_features=True.

    Returns
    -------
    np.ndarray
        Feature vector of shape (1, n_features) ready for scaling.

    Raises
    ------
    ValueError
        If history is too short or feature construction fails.
    """
    # Input validation
    min_history = max(lags)
    if rolling_windows:
        min_history = max(min_history, max(rolling_windows))
    
    if len(history) < min_history:
        raise ValueError(
            f"History too short. Need at least {min_history} values, "
            f"got {len(history)}"
        )
    
    if include_time_features and current_date is None:
        raise ValueError("current_date required when include_time_features=True")
    
    features = []
    
    # Lag features
    for lag in lags:
        if lag > len(history):
            raise ValueError(f"Lag {lag} exceeds history length {len(history)}")
        features.append(history[-lag])
    
    # Rolling features (calculated on history before current point)
    if rolling_windows:
        for window in rolling_windows:
            if window > len(history):
                raise ValueError(f"Window {window} exceeds history length {len(history)}")
            
            window_data = history[-window:]
            features.append(np.mean(window_data))
            features.append(np.std(window_data))
    
    # Time features
    if include_time_features:
        # Extract date components
        dow = current_date.dayofweek
        month = current_date.month
        dom = current_date.day
        
        features.append(dow)
        features.append(month)
        features.append(dom)
        
        # Cyclical encoding
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))
    
    return np.array(features).reshape(1, -1)


def predict_next(
    model: Ridge,
    scaler: StandardScaler,
    history: Union[np.ndarray, pd.Series, List[float]],
    lags: List[int],
    rolling_windows: Optional[List[int]] = None,
    include_time_features: bool = False,
    current_date: Optional[pd.Timestamp] = None,
    n_steps: int = 1
) -> Union[float, np.ndarray]:
    """
    Generate recursive forecast for next revenue value(s).

    Parameters
    ----------
    model : Ridge
        Trained Ridge model.
    scaler : StandardScaler
        Fitted StandardScaler.
    history : array-like
        Recent historical revenue values (most recent last).
    lags : List[int]
        Lag periods used during training.
    rolling_windows : List[int], optional
        Rolling window sizes used during training.
    include_time_features : bool
        Whether time features were used during training.
    current_date : pd.Timestamp, optional
        Current date for time features. Required if include_time_features=True
        and n_steps=1. For multi-step, dates are auto-incremented.
    n_steps : int
        Number of future steps to forecast recursively.

    Returns
    -------
    float or np.ndarray
        Single predicted value if n_steps=1, else array of predictions.

    Raises
    ------
    ValueError
        If input validation fails or prediction fails.
    """
    # Convert history to numpy array
    if isinstance(history, pd.Series):
        history = history.values
    else:
        history = np.array(history, dtype=float)
    
    if history.ndim != 1:
        raise ValueError(f"History must be 1D, got shape {history.shape}")
    
    if len(history) == 0:
        raise ValueError("History cannot be empty")
    
    if not np.isfinite(history).all():
        raise ValueError("History contains NaN or infinite values")
    
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    
    # Validate current_date for single-step with time features
    if n_steps == 1 and include_time_features and current_date is None:
        raise ValueError("current_date required for single-step with time features")
    
    # Initialize predictions array
    predictions = []
    current_history = history.copy()
    current_date_iter = current_date
    
    for step in range(n_steps):
        try:
            # Build features
            features = build_features_from_history(
                history=current_history,
                lags=lags,
                rolling_windows=rolling_windows,
                include_time_features=include_time_features,
                current_date=current_date_iter
            )
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Predict
            pred = model.predict(features_scaled)[0]
            predictions.append(pred)
            
            # Update history for next recursive step
            current_history = np.append(current_history, pred)
            
            # Update date for next step if using time features
            if include_time_features and current_date_iter is not None:
                current_date_iter = current_date_iter + pd.Timedelta(days=1)
                
        except Exception as e:
            raise ValueError(f"Prediction failed at step {step + 1}: {e}")
    
    # Return single value or array
    if n_steps == 1:
        return float(predictions[0])
    else:
        return np.array(predictions)


# Minimal usage example
if __name__ == "__main__":
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    # Example: Train a simple model and use it for inference
    # (In production, you'd load artifacts instead of training)
    
    # Generate sample training data
    np.random.seed(42)
    n_samples = 100
    history_length = 30
    
    # Define feature configuration (must match training)
    lags = [1, 2, 3, 7]
    rolling_windows = [7, 14]
    include_time_features = True
    
    # Train a dummy model (in practice, load from disk)
    X_dummy = np.random.randn(n_samples, len(lags) + 2*len(rolling_windows) + 5)
    y_dummy = X_dummy @ np.random.randn(X_dummy.shape[1]) + np.random.randn(n_samples) * 0.1
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dummy)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_dummy)
    
    # Create recent history
    recent_history = np.random.randn(history_length) * 10 + 100
    current_date = pd.Timestamp("2024-01-01")
    
    # Single-step prediction
    next_value = predict_next(
        model=model,
        scaler=scaler,
        history=recent_history,
        lags=lags,
        rolling_windows=rolling_windows,
        include_time_features=include_time_features,
        current_date=current_date,
        n_steps=1
    )
    
    print(f"Single-step prediction:")
    print(f"  Current date: {current_date.strftime('%Y-%m-%d')}")
    print(f"  Next day revenue: {next_value:.2f}")
    
    # Multi-step prediction
    next_7_days = predict_next(
        model=model,
        scaler=scaler,
        history=recent_history,
        lags=lags,
        rolling_windows=rolling_windows,
        include_time_features=include_time_features,
        current_date=current_date,
        n_steps=7
    )
    
    print(f"\n7-day forecast:")
    for i, pred in enumerate(next_7_days, 1):
        pred_date = current_date + pd.Timedelta(days=i)
        print(f"  {pred_date.strftime('%Y-%m-%d')}: {pred:.2f}")