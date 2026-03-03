"""
Time-series data loader and validator.

Purpose:
Load CSV data, parse dates, enforce chronological order,
and perform basic validation.

This module intentionally avoids:
- Feature engineering
- Data splitting
- Model training
- Any transformation beyond loading and validation
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional


def load_and_validate_timeseries(
    filepath: Union[str, Path],
    date_column: str = "date",
    value_column: str = "value",
    date_format: Optional[str] = None,
    expected_frequency: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load CSV file and validate as chronological time-series.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file.
    date_column : str
        Name of column containing dates.
    value_column : str
        Name of column containing time-series values.
    date_format : str, optional
        Format string for date parsing (passed to pd.to_datetime).
        If None, pandas will attempt to infer format.
    expected_frequency : str, optional
        Expected pandas frequency string (e.g., 'D' for daily, 'H' for hourly).
        If provided, validates that data has this frequency.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
        - DatetimeIndex from parsed dates
        - Single column with values (named from value_column)

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist.
    ValueError
        If:
        - Required columns are missing
        - Dates cannot be parsed
        - Dates are not unique
        - Dates are not monotonically increasing
        - Expected frequency is violated
        - Values contain non-numeric data
    """
    # Validate file exists
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    # Read CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")

    # Validate required columns exist
    if date_column not in df.columns:
        raise ValueError(
            f"Date column '{date_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    if value_column not in df.columns:
        raise ValueError(
            f"Value column '{value_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse dates
    try:
        df[date_column] = pd.to_datetime(df[date_column], format=date_format)
    except Exception as e:
        raise ValueError(f"Failed to parse dates: {e}")

    # Check for missing dates
    if df[date_column].isnull().any():
        raise ValueError("Date column contains null values after parsing")

    # Sort by date
    df = df.sort_values(by=date_column).reset_index(drop=True)

    # Check for duplicate dates
    if df[date_column].duplicated().any():
        duplicate_dates = df[date_column][df[date_column].duplicated()].tolist()
        raise ValueError(f"Duplicate dates found: {duplicate_dates[:5]}")

    # Validate chronological order (already sorted, but double-check)
    if not df[date_column].is_monotonic_increasing:
        raise ValueError("Dates are not monotonically increasing after sorting")

    # Validate value column is numeric
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        try:
            df[value_column] = pd.to_numeric(df[value_column])
        except Exception as e:
            raise ValueError(f"Value column contains non-numeric data: {e}")

    # Check for infinite values
    if not np.isfinite(df[value_column]).all():
        raise ValueError("Value column contains infinite values")

    # Set date as index
    df = df.set_index(date_column)

    # Validate expected frequency if specified
    if expected_frequency is not None:
        try:
            inferred_freq = pd.infer_freq(df.index)
            if inferred_freq != expected_frequency:
                raise ValueError(
                    f"Expected frequency '{expected_frequency}', "
                    f"but inferred '{inferred_freq}'"
                )
        except Exception as e:
            raise ValueError(f"Failed to validate frequency: {e}")

    # Rename value column to standardized name
    df = df[[value_column]].rename(columns={value_column: "value"})

    return df


# Minimal usage example
if __name__ == "__main__":
    # Example: Load and validate daily time-series
    try:
        df = load_and_validate_timeseries(
            filepath="data/revenue.csv",
            date_column="date",
            value_column="revenue",
            expected_frequency="D"  # Expect daily data
        )
        print(f"Successfully loaded {len(df)} records")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data: {e}")