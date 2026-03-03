"""
Time-series feature engineering module.

Purpose:
Generate lag and rolling features from time-series data
while preventing data leakage through proper shifting.

This module intentionally avoids:
- Data splitting
- Scaling/normalization
- Model training
- Cross-validation
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def create_lag_features(
    series: pd.Series,
    lags: List[int],
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Create lag features for a time series.

    Parameters
    ----------
    series : pd.Series
        Time series data with datetime index.
    lags : List[int]
        List of lag periods to create (e.g., [1, 7, 14]).
    drop_na : bool
        Whether to drop rows with NaN values.

    Returns
    -------
    pd.DataFrame
        DataFrame with lag features, indexed by original dates.
    """
    lag_features = pd.DataFrame(index=series.index)
    
    for lag in lags:
        lag_features[f"lag_{lag}"] = series.shift(lag)
    
    if drop_na:
        lag_features = lag_features.dropna()
    
    return lag_features


def create_rolling_features(
    series: pd.Series,
    windows: List[int],
    min_periods: Optional[int] = None,
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Create rolling statistics features.

    Parameters
    ----------
    series : pd.Series
        Time series data with datetime index.
    windows : List[int]
        List of window sizes for rolling calculations.
    min_periods : int, optional
        Minimum number of observations in window required.
        Defaults to window size.
    drop_na : bool
        Whether to drop rows with NaN values.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling features, indexed by original dates.
    """
    rolling_features = pd.DataFrame(index=series.index)
    
    for window in windows:
        min_periods = min_periods or window
        
        # Rolling mean (shifted to prevent leakage)
        rolling_mean = series.rolling(
            window=window,
            min_periods=min_periods
        ).mean().shift(1)
        rolling_features[f"rolling_mean_{window}"] = rolling_mean
        
        # Rolling standard deviation (shifted to prevent leakage)
        rolling_std = series.rolling(
            window=window,
            min_periods=min_periods
        ).std().shift(1)
        rolling_features[f"rolling_std_{window}"] = rolling_std
    
    if drop_na:
        rolling_features = rolling_features.dropna()
    
    return rolling_features


def create_features(
    df: pd.DataFrame,
    target_column: str = "value",
    lags: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
    min_rolling_periods: Optional[int] = None,
    include_time_features: bool = False,
    drop_na: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Create feature matrix and target vector from time-series data.

    Parameters
    ----------
    df : pd.DataFrame
        Validated DataFrame with datetime index and 'value' column.
    target_column : str
        Name of the target column (default: 'value').
    lags : List[int], optional
        Lag periods to create as features.
    rolling_windows : List[int], optional
        Window sizes for rolling statistics.
    min_rolling_periods : int, optional
        Minimum periods for rolling calculations.
    include_time_features : bool
        Whether to include time-based features (day of week, month).
    drop_na : bool
        Whether to drop rows with NaN values after feature creation.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str]]
        - X: Feature matrix
        - y: Target vector (shifted -1 for next-step prediction)
        - feature_names: List of feature column names

    Raises
    ------
    ValueError
        If input validation fails or no features are generated.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    # Default feature configurations
    if lags is None:
        lags = [1, 2, 3, 7, 14]  # Common lags for daily data
    
    if rolling_windows is None:
        rolling_windows = [7, 14, 28]  # Weekly, bi-weekly, monthly
    
    # Create feature list
    all_features = []
    feature_names = []
    
    # 1. Lag features
    lag_df = create_lag_features(
        series=df[target_column],
        lags=lags,
        drop_na=False  # Handle NA removal globally
    )
    all_features.append(lag_df)
    feature_names.extend(lag_df.columns.tolist())
    
    # 2. Rolling features
    rolling_df = create_rolling_features(
        series=df[target_column],
        windows=rolling_windows,
        min_periods=min_rolling_periods,
        drop_na=False  # Handle NA removal globally
    )
    all_features.append(rolling_df)
    feature_names.extend(rolling_df.columns.tolist())
    
    # 3. Time-based features (optional)
    if include_time_features:
        time_features = pd.DataFrame(index=df.index)
        time_features["day_of_week"] = df.index.dayofweek
        time_features["month"] = df.index.month
        time_features["day_of_month"] = df.index.day
        
        # Cyclical encoding for day_of_week
        time_features["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        time_features["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        all_features.append(time_features)
        feature_names.extend([
            "day_of_week", "month", "day_of_month",
            "day_of_week_sin", "day_of_week_cos"
        ])
    
    # Combine all features
    X = pd.concat(all_features, axis=1)
    
    # Create target (next day's value)
    y = df[target_column].shift(-1)
    
    # Remove rows with NaN (from shifting and feature creation)
    if drop_na:
        # Create mask for rows with no NaN in features OR target
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
    
    # Final validation
    if len(X) == 0:
        raise ValueError("No valid samples after feature creation. "
                        "Try reducing lag/rolling windows or using more data.")
    
    if len(X) != len(y):
        raise ValueError("Feature matrix and target vector length mismatch")
    
    return X, y, feature_names


# Minimal usage example
if __name__ == "__main__":
    # Example: Create features for revenue data
    import numpy as np
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "value": 1000 + np.cumsum(np.random.randn(100) * 10)
    }, index=dates)
    
    # Create features
    X, y, feature_names = create_features(
        df=df,
        lags=[1, 2, 3, 7],
        rolling_windows=[7, 14],
        include_time_features=True,
        drop_na=True
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"\nFeature names: {feature_names}")
    print(f"\nFirst 3 rows of features:\n{X.head(3)}")
    print(f"\nFirst 3 target values:\n{y.head(3)}")