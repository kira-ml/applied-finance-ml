"""
src/preprocessing.py

Core Responsibilities:
- Construct a sklearn.pipeline.Pipeline containing imputers, one-hot encoders, and scalers.
- Fit the pipeline strictly on the X_train data.
- Transform X_train, X_cal, and X_test using the fitted pipeline.
- Serialize the fitted pipeline to models/preprocessor.pkl.

Constraints:
- Strict static typing.
- Deterministic execution (fixed random state where applicable).
- Immutable state enforcement (returns new DataFrames/Arrays).
- Explicit exception contracts.
- No target variable access.
- No feature engineering based on target.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Final, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

_RANDOM_STATE: Final[int] = 42
_IMPUTER_STRATEGY: Final[str] = "median"  # Robust to outliers compared to mean
_OHE_HANDLE_UNKNOWN: Final[str] = "ignore"  # Prevent errors on unseen categories in test set
_PIPELINE_FILENAME: Final[str] = "preprocessor.pkl"

# Type Aliases for clarity
DataFrame = pd.DataFrame
NumericArray = np.ndarray


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------

class PreprocessingError(Exception):
    """Base exception for preprocessing failures."""
    pass


class DataValidationError(PreprocessingError):
    """Raised when input data validation fails."""
    pass


class SerializationError(PreprocessingError):
    """Raised when serialization/deserialization fails."""
    pass


# -----------------------------------------------------------------------------
# Pure Logic: Schema Detection & Pipeline Construction
# -----------------------------------------------------------------------------

def _identify_column_types(df: DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numeric and categorical columns based on pandas dtypes.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple of (numeric_columns, categorical_columns).
        
    Time Complexity: O(C) where C is number of columns.
    Space Complexity: O(C).
    """
    numeric_cols: List[str] = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols: List[str] = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    # Exhaustive check: Ensure all columns are accounted for
    all_cols = set(df.columns)
    processed_cols = set(numeric_cols + categorical_cols)
    
    if all_cols != processed_cols:
        missing = all_cols - processed_cols
        raise DataValidationError(f"Columns with unsupported dtypes detected: {missing}")
        
    return numeric_cols, categorical_cols


def _build_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Pipeline:
    """
    Constructs the sklearn Pipeline with imputers, encoders, and scalers.
    
    Args:
        numeric_cols: List of numeric column names.
        categorical_cols: List of categorical column names.
        
    Returns:
        A fitted-ready sklearn Pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=_IMPUTER_STRATEGY)),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown=_OHE_HANDLE_UNKNOWN, sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
    
    return preprocessor


# -----------------------------------------------------------------------------
# Boundary Logic: I/O and Orchestration
# -----------------------------------------------------------------------------

def _save_pipeline(pipeline: Pipeline, output_path: Path) -> None:
    """
    Serializes the pipeline to disk using pickle.
    
    Args:
        pipeline: The fitted sklearn Pipeline.
        output_path: Destination file path.
        
    Raises:
        SerializationError: If writing fails.
    """
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise SerializationError("Failed to serialize preprocessing pipeline.") from e


def fit_and_transform_pipeline(
    X_train: DataFrame,
    X_cal: DataFrame,
    X_test: DataFrame,
    output_dir: Path
) -> Tuple[NumericArray, NumericArray, NumericArray, Path]:
    """
    Fits the preprocessing pipeline on X_train and transforms all three splits.
    Saves the fitted pipeline to disk.
    
    Args:
        X_train: Training features DataFrame.
        X_cal: Calibration features DataFrame.
        X_test: Test features DataFrame.
        output_dir: Directory to save the preprocessor.pkl.
        
    Returns:
        Tuple of (X_train_transformed, X_cal_transformed, X_test_transformed, pipeline_path).
        
    Raises:
        DataValidationError: If input schemas mismatch or are invalid.
        SerializationError: If saving fails.
        
    Time Complexity: O(N * C) for fitting and transforming.
    Space Complexity: O(N * C) for storing transformed arrays.
    """
    # 1. Validate Input Emptiness FIRST (Priority over schema check)
    if X_train.empty:
        raise DataValidationError("Training data cannot be empty.")
    
    # 2. Validate Input Schemas
    # Strict check: All sets must have identical columns
    if set(X_train.columns) != set(X_cal.columns) or set(X_train.columns) != set(X_test.columns):
        raise DataValidationError("Input DataFrames must have identical columns.")
        
    # 3. Identify Types
    numeric_cols, categorical_cols = _identify_column_types(X_train)
    
    if not numeric_cols and not categorical_cols:
        raise DataValidationError("No valid numeric or categorical columns found.")
    
    # 4. Build Pipeline
    pipeline = _build_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    # 5. Fit on Train ONLY
    # This ensures no data leakage from calibration or test sets
    pipeline.fit(X_train)
    
    # 6. Transform All Sets
    # Returns numpy arrays due to OneHotEncoder output
    X_train_np: NumericArray = pipeline.transform(X_train)
    X_cal_np: NumericArray = pipeline.transform(X_cal)
    X_test_np: NumericArray = pipeline.transform(X_test)
    
    # 7. Serialize
    output_path: Path = output_dir / _PIPELINE_FILENAME
    _save_pipeline(pipeline, output_path)
    
    return X_train_np, X_cal_np, X_test_np, output_path