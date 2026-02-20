"""
Module: src/process.py
Contract: Preprocess credit risk data for Gradient Boosting Model training.
          Performs imputation, encoding, and stratified splitting.
Invariants:
  - Input DataFrame is not mutated; copies are returned.
  - Missing values are handled via median (numeric) or mode (categorical).
  - Categorical features are ordinal encoded to preserve order-free compatibility with GBM.
  - Split ratios are strictly 60% Train, 20% Validation, 20% Test.
  - Stratification is enforced on the 'default' target column.
Failure Modes:
  - Raises ProcessValidationError for missing columns, empty datasets, or invalid split params.
  - Raises ProcessRuntimeError for numerical instability (NaNs post-imputation).
Version: 1.0.0
Dependencies: python==3.11+, numpy==2.2.3, pandas==2.2.4, scikit-learn==1.6.0
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(message)s"
_LOG_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%SZ"
_ENCODING_UTF8: str = "utf-8"

_RATIO_TRAIN: float = 0.60
_RATIO_VAL: float = 0.20
_RATIO_TEST: float = 0.20
_TARGET_COLUMN: str = "default"

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def _configure_logging() -> None:
    """Initialize structured logging with deterministic format."""
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
        force=True
    )

# ==============================================================================
# EXCEPTIONS
# ==============================================================================

class ProcessValidationError(ValueError):
    """Raised when input data or parameters violate structural constraints."""
    pass

class ProcessRuntimeError(RuntimeError):
    """Raised when processing logic encounters an unrecoverable state."""
    pass

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class SplitData:
    """Container for stratified split results."""
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    version: str = "1.0"

# ==============================================================================
# PURE CORE LOGIC
# ==============================================================================

def _impute_numeric_column(series: pd.Series) -> pd.Series:
    """Impute missing values in a numeric series using the median."""
    if not pd.api.types.is_numeric_dtype(series):
        raise ProcessValidationError(f"Column {series.name} is not numeric.")
    
    median_val: float = series.median()
    
    # Arithmetic Overflow/NaN Guard
    if not np.isfinite(median_val):
        logging.warning(f"Median for {series.name} is non-finite ({median_val}); filling with 0.0")
        median_val = 0.0
        
    return series.fillna(median_val)

def _impute_categorical_column(series: pd.Series) -> pd.Series:
    """Impute missing values in a categorical series using the mode."""
    if pd.api.types.is_numeric_dtype(series):
        # Heuristic: if numeric but treated as cat (e.g. IDs), still use mode logic if needed
        # But per spec, this handles explicit categoricals.
        pass
        
    mode_result: pd.Series = series.mode()
    
    if len(mode_result) == 0:
        fill_val: Any = "UNKNOWN"
    else:
        fill_val: Any = mode_result.iloc[0]
        
    return series.fillna(fill_val)

def _encode_categorical_column(series: pd.Series) -> Tuple[pd.Series, Dict[Any, int]]:
    """
    Encode categorical series to integers (Ordinal Encoding).
    Returns encoded series and mapping dictionary for reproducibility.
    """
    unique_vals: List[Any] = series.astype(str).unique().tolist()
    unique_vals.sort() # Deterministic ordering
    
    mapping: Dict[Any, int] = {val: idx for idx, val in enumerate(unique_vals)}
    
    # Explicit mapping application
    encoded: pd.Series = series.astype(str).map(mapping)
    
    if encoded.isnull().any():
        # Should not happen if mapping covers all unique vals, but safety guard
        raise ProcessRuntimeError(f"Encoding failed for column {series.name}; unmapped values found.")
        
    return encoded, mapping

def _validate_input_dataframe(df: pd.DataFrame) -> None:
    """Validate structural integrity of input DataFrame."""
    if df.empty:
        raise ProcessValidationError("Input DataFrame is empty.")
    
    if _TARGET_COLUMN not in df.columns:
        raise ProcessValidationError(f"Target column '{_TARGET_COLUMN}' not found.")
    
    if df[_TARGET_COLUMN].isnull().any():
        raise ProcessValidationError(f"Target column '{_TARGET_COLUMN}' contains missing values.")

def _identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Separate columns into numeric and categorical lists excluding target."""
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    
    feature_cols: List[str] = [c for c in df.columns if c != _TARGET_COLUMN]
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    return numeric_cols, categorical_cols

# ==============================================================================
# EFFECTFUL BOUNDARY LOGIC
# ==============================================================================

def process_data(
    df: pd.DataFrame,
    random_seed: int
) -> SplitData:
    """
    Main processing pipeline: Impute -> Encode -> Split.
    
    Args:
        df: Raw input DataFrame.
        random_seed: Integer seed for deterministic splitting.
        
    Returns:
        SplitData container with train, validation, and test DataFrames.
    """
    # 1. Input Validation
    _validate_input_dataframe(df)
    
    # 2. Immutable Boundary: Work on a copy
    data_work: pd.DataFrame = df.copy(deep=True)
    
    # 3. Identify Types
    num_cols, cat_cols = _identify_column_types(data_work)
    
    logging.info(f"Identified {len(num_cols)} numeric and {len(cat_cols)} categorical features.")
    
    # 4. Imputation
    for col in num_cols:
        if data_work[col].isnull().any():
            data_work[col] = _impute_numeric_column(data_work[col])
            
    for col in cat_cols:
        if data_work[col].isnull().any():
            data_work[col] = _impute_categorical_column(data_work[col])
            
    # Post-imputation stability check
    if data_work.isnull().any().any():
        raise ProcessRuntimeError("Missing values remain after imputation phase.")
    
    # 5. Encoding
    encoding_maps: Dict[str, Dict[Any, int]] = {}
    for col in cat_cols:
        encoded_series, mapping = _encode_categorical_column(data_work[col])
        data_work[col] = encoded_series
        encoding_maps[col] = mapping
        
    logging.info(f"Categorical encoding complete. Maps generated for {len(encoding_maps)} columns.")
    
    # 6. Stratified Splitting
    # Calculate split weights
    # sklearn train_test_split uses 'test_size' relative to current input.
    # Step 1: Split off Test (20% of total)
    # Remaining is 80%. We need Val to be 20% of total, which is 25% of remaining (0.20/0.80).
    # Train will be 60% of total, which is 75% of remaining.
    
    X: pd.DataFrame = data_work.drop(columns=[_TARGET_COLUMN])
    y: pd.Series = data_work[_TARGET_COLUMN]
    
    # Ensure target is integer for stratification if it came in as float
    if not pd.api.types.is_integer_dtype(y):
        y = y.astype(int)
    
    # First Split: Separate Test Set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=_RATIO_TEST,
        random_state=random_seed,
        stratify=y
    )
    
    # Second Split: Separate Validation from Train
    # Ratio: 0.20 / (0.60 + 0.20) = 0.25
    ratio_val_adjusted: float = _RATIO_VAL / (_RATIO_TRAIN + _RATIO_VAL)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=ratio_val_adjusted,
        random_state=random_seed,
        stratify=y_temp
    )
    
    # Reconstruct DataFrames
    df_train: pd.DataFrame = pd.concat([X_train, y_train], axis=1)
    df_val: pd.DataFrame = pd.concat([X_val, y_val], axis=1)
    df_test: pd.DataFrame = pd.concat([X_test, y_test], axis=1)
    
    # Final Verification of Counts
    total_rows: int = len(df)
    if len(df_train) + len(df_val) + len(df_test) != total_rows:
        raise ProcessRuntimeError("Row count mismatch after splitting; data loss detected.")
        
    logging.info(f"Split complete: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}.")
    
    return SplitData(
        train=df_train,
        validation=df_val,
        test=df_test,
        version="1.0"
    )

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import sys
    import os

    _configure_logging()
    
    # Hardcoded paths for deployment context
    INPUT_PATH: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\data\raw\credit_data.csv"
    OUTPUT_DIR: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\data\processed"
    SEED: int = 42
    
    if not os.path.exists(INPUT_PATH):
        logging.error(f"Input file not found: {INPUT_PATH}")
        sys.exit(1)
        
    try:
        # Load Data (Minimal I/O for standalone execution)
        raw_df: pd.DataFrame = pd.read_csv(INPUT_PATH, encoding=_ENCODING_UTF8)
        
        # Execute Pipeline
        result: SplitData = process_data(raw_df, SEED)
        
        # Save Outputs
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
        train_path: str = os.path.join(OUTPUT_DIR, "train.csv")
        val_path: str = os.path.join(OUTPUT_DIR, "validation.csv")
        test_path: str = os.path.join(OUTPUT_DIR, "test.csv")
        
        result.train.to_csv(train_path, index=False, encoding=_ENCODING_UTF8)
        result.validation.to_csv(val_path, index=False, encoding=_ENCODING_UTF8)
        result.test.to_csv(test_path, index=False, encoding=_ENCODING_UTF8)
        
        logging.info(f"Successfully wrote splits to {OUTPUT_DIR}")
        
    except ProcessValidationError as e:
        logging.error(f"Validation Failed: {e}")
        sys.exit(1)
    except ProcessRuntimeError as e:
        logging.error(f"Processing Failed: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected System Error: {e}")
        sys.exit(1)