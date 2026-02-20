# -*- coding: utf-8 -*-
"""
Module: features.py
Version: 1.0.1
Contract: Compute specific credit risk domain features based on explicit schema.
Dependencies: python==3.9+, numpy==1.24.3, pandas==2.0.3
Invariants:
    - Input DataFrames are never mutated; a copy is returned.
    - All numeric operations check for division by zero and overflow.
    - No implicit type coercion occurs; all conversions are explicit.
Failures:
    - FeatureValidationError: Raised when required columns are missing or data types are invalid.
    - NumericDomainError: Raised when arithmetic results in undefined states (Inf/NaN).
"""

from __future__ import annotations

import logging
import sys
from typing import List, Literal, Optional
from pathlib import Path

import numpy as np
import pandas as pd

# --- Configuration Constants ---
_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"
_ENCODING: Literal["utf-8"] = "utf-8"
_FLOAT_EPSILON: float = 1e-9
_MAX_RATIO_VALUE: float = 1e6  # Cap to prevent overflow representation issues

# Required input schema derived from provided CSV sample
_REQUIRED_COLUMNS: List[str] = [
    "income",
    "loan_amount",
    "revolving_balance",
    "num_delinquencies"
]

# Output column names
_COL_DTI: str = "feature_dti"
_COL_UTIL: str = "feature_utilization"
_COL_DELINQ_FLAG: str = "feature_delinq_flag"
_COL_REVOLVING_RATIO: str = "feature_revolving_ratio"


# --- Custom Exceptions (Closed Failure Domain) ---

class FeatureValidationError(ValueError):
    """Raised when input data violates structural preconditions."""
    pass

class NumericDomainError(ArithmeticError):
    """Raised when numeric operations yield undefined or unsafe results."""
    pass


# --- Logging Setup (Structured, Local) ---

def _configure_logger(name: str) -> logging.Logger:
    """Configure a structured logger with no external dependencies."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

logger = _configure_logger("features_engine")


# --- Pure Utility Functions ---

def _safe_divide(numerator: pd.Series, denominator: pd.Series, label: str) -> pd.Series:
    """
    Perform element-wise division with explicit zero-handling and overflow checks.
    Returns 0.0 where denominator is near zero.
    """
    if not isinstance(numerator, pd.Series) or not isinstance(denominator, pd.Series):
        raise TypeError(f"{label} requires pd.Series inputs")

    denom_abs = denominator.abs()
    mask_zero = denom_abs < _FLOAT_EPSILON
    
    # Initialize result with zeros (float64 for precision)
    result = pd.Series(np.zeros(len(numerator)), dtype=np.float64, index=numerator.index)
    
    # Compute only where safe
    safe_mask = ~mask_zero
    if safe_mask.any():
        raw_values = numerator.loc[safe_mask] / denominator.loc[safe_mask]
        
        # Check for Inf/Overflow
        if np.any(np.isinf(raw_values)):
            raise NumericDomainError(f"Infinite value generated in {label}")
        if np.any(np.isnan(raw_values)):
            raise NumericDomainError(f"NaN generated in {label}")
            
        # Cap values
        capped_values = np.clip(raw_values, -_MAX_RATIO_VALUE, _MAX_RATIO_VALUE)
        result.loc[safe_mask] = capped_values
        
    return result

def _validate_schema(df: pd.DataFrame) -> None:
    """Mechanically enforce input schema presence and types."""
    missing = set(_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise FeatureValidationError(f"Missing required columns: {missing}")
    
    for col in _REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                raise FeatureValidationError(f"Column '{col}' cannot be coerced to numeric")


# --- Core Feature Logic (Pure) ---

def _compute_dti(loan_amount: pd.Series, income: pd.Series) -> pd.Series:
    """Compute Debt-to-Income ratio (Loan Amount / Income)."""
    return _safe_divide(loan_amount, income, "DTI")

def _compute_utilization(revolving_balance: pd.Series, loan_amount: pd.Series) -> pd.Series:
    """
    Compute Credit Utilization Rate.
    Contextual Definition: Revolving Balance / Total Loan Amount.
    (Note: If 'credit_limit' existed, it would be used, but per schema, we use loan_amount as the debt baseline).
    """
    return _safe_divide(revolving_balance, loan_amount, "Utilization")

def _compute_delinq_flag(num_delinquencies: pd.Series) -> pd.Series:
    """Compute Total Delinquency Flag (1 if > 0, else 0)."""
    return (num_delinquencies > 0).astype(int)

def _compute_revolving_ratio(revolving_balance: pd.Series, income: pd.Series) -> pd.Series:
    """Compute Revolving Balance to Income Ratio."""
    return _safe_divide(revolving_balance, income, "RevolvingRatio")


# --- Main Effectful Function ---

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain features and return an augmented DataFrame.
    
    Args:
        df: Input pandas DataFrame containing raw credit data.
        
    Returns:
        A new DataFrame with engineered feature columns appended.
        
    Raises:
        FeatureValidationError: If input schema is invalid.
        NumericDomainError: If arithmetic yields unsafe values.
    """
    # 1. Immutable Data Boundary
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
        
    work_df = df.copy(deep=True)
    
    # 2. Structural Invariant Enforcement
    _validate_schema(work_df)
    
    logger.info("Starting feature computation", extra={"row_count": len(work_df)})
    
    # 3. Deterministic Execution & Linear Control Flow
    s_income = work_df["income"]
    s_loan = work_df["loan_amount"]
    s_revolving = work_df["revolving_balance"]
    s_delinq = work_df["num_delinquencies"]
    
    # Compute Features
    try:
        feat_dti = _compute_dti(s_loan, s_income)
        feat_util = _compute_utilization(s_revolving, s_loan)
        feat_delinq = _compute_delinq_flag(s_delinq)
        feat_rev_ratio = _compute_revolving_ratio(s_revolving, s_income)
    except NumericDomainError as e:
        logger.error("Numeric domain violation detected", exc_info=True)
        raise e

    # 4. Append Results
    work_df[_COL_DTI] = feat_dti
    work_df[_COL_UTIL] = feat_util
    work_df[_COL_DELINQ_FLAG] = feat_delinq
    work_df[_COL_REVOLVING_RATIO] = feat_rev_ratio
    
    # 5. Post-condition Check
    new_cols = [_COL_DTI, _COL_UTIL, _COL_DELINQ_FLAG, _COL_REVOLVING_RATIO]
    if work_df[new_cols].isnull().any().any():
        raise NumericDomainError("Final feature set contains unexpected NaN values")

    logger.info("Feature computation completed successfully")
    return work_df


# --- CLI Entry Point (Effectful) ---

def _process_file(input_path_str: str, output_path_str: Optional[str] = None) -> None:
    """Load, process, and save a single CSV file."""
    input_path = Path(input_path_str)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Processing file: {input_path.name}")
    
    try:
        # Explicit encoding declaration
        df = pd.read_csv(input_path, encoding=_ENCODING)
        
        df_augmented = compute_features(df)
        
        if output_path_str:
            out_path = Path(output_path_str)
        else:
            out_path = input_path.parent / f"processed_{input_path.name}"
            
        df_augmented.to_csv(out_path, index=False, encoding=_ENCODING)
        logger.info(f"Successfully wrote output to: {out_path}")
        
    except Exception as e:
        logger.critical(f"Pipeline failed for {input_path.name}: {str(e)}")
        raise


if __name__ == "__main__":
    paths_to_process: List[str] = [
        r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\data\processed\test.csv",
        r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\data\processed\train.csv",
        r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\data\processed\validation.csv"
    ]
    
    exit_code = 0
    for p in paths_to_process:
        try:
            _process_file(p)
        except Exception:
            exit_code = 1
            
    sys.exit(exit_code)