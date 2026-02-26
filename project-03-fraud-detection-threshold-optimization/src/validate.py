import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Set
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ValidationConfig:
    null_threshold: float = 0.40
    transaction_id_names: Tuple[str, ...] = ("TransactionID", "transaction_id")
    amount_column_names: Tuple[str, ...] = ("TransactionAmt", "amount")
    max_mean_ratio_threshold: float = 1000.0
    output_report_path: str = "artifacts/validation_report.txt"

class ValidationError(Exception):
    """Base exception for validation failures."""
    pass

class DataFrameEmptyError(ValidationError):
    """Raised when input DataFrame is empty."""
    pass

class NumericFillError(ValidationError):
    """Raised when numeric fill operation fails."""
    pass

class CategoricalFillError(ValidationError):
    """Raised when categorical fill operation fails."""
    pass

class ReportWriteError(ValidationError):
    """Raised when writing validation report fails."""
    pass

def _validate_input(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected pandas DataFrame, got {type(df)}")
    if df.empty:
        raise DataFrameEmptyError("Input DataFrame is empty")

def _drop_high_null_columns(df: pd.DataFrame, config: ValidationConfig) -> Tuple[pd.DataFrame, List[str]]:
    null_rates = df.isnull().mean()
    high_null_cols = null_rates[null_rates > config.null_threshold].index.tolist()
    
    if high_null_cols:
        df = df.drop(columns=high_null_cols)
    
    return df, high_null_cols

def _fill_numeric_nulls(df: pd.DataFrame, filled_cols: Dict[str, str]) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                raise NumericFillError(f"Cannot compute median for column {col} - all values are null")
            df[col] = df[col].fillna(median_val)
            filled_cols[col] = f"filled with median {median_val:.4f}"
    
    return df

def _fill_categorical_nulls(df: pd.DataFrame, filled_cols: Dict[str, str]) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_vals = df[col].mode()
            if len(mode_vals) == 0:
                raise CategoricalFillError(f"Cannot compute mode for column {col} - no valid values")
            mode_val = mode_vals[0]
            df[col] = df[col].fillna(mode_val)
            filled_cols[col] = f"filled with mode '{mode_val}'"
    
    return df

def _check_duplicate_transactions(df: pd.DataFrame, config: ValidationConfig) -> int:
    transaction_id_col = None
    for col_name in config.transaction_id_names:
        if col_name in df.columns:
            transaction_id_col = col_name
            break
    
    if transaction_id_col is None:
        return 0
    
    duplicate_count = df.duplicated(subset=[transaction_id_col]).sum()
    return duplicate_count

def _check_log_transform_flag(df: pd.DataFrame, config: ValidationConfig) -> bool:
    amount_col = None
    for col_name in config.amount_column_names:
        if col_name in df.columns:
            amount_col = col_name
            break
    
    if amount_col is None:
        return False
    
    if not pd.api.types.is_numeric_dtype(df[amount_col]):
        return False
    
    max_val = df[amount_col].max()
    median_val = df[amount_col].median()

    if median_val == 0:
        return False

    ratio = max_val / median_val
    return bool(ratio > config.max_mean_ratio_threshold)

def _find_zero_variance_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_var_cols = []
    
    for col in numeric_cols:
        if df[col].std() == 0:
            zero_var_cols.append(col)
    
    return zero_var_cols

def _write_validation_report(
    config: ValidationConfig,
    dropped_null_cols: List[str],
    dropped_zero_var_cols: List[str],
    filled_cols: Dict[str, str],
    duplicate_count: int,
    needs_log_transform: bool
) -> None:
    try:
        report_path = Path(config.output_report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("VALIDATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Columns dropped (>40% null):\n")
            if dropped_null_cols:
                for col in sorted(dropped_null_cols):
                    f.write(f"  - {col}\n")
            else:
                f.write("  None\n")
            f.write("\n")
            
            f.write("Columns dropped (zero variance):\n")
            if dropped_zero_var_cols:
                for col in sorted(dropped_zero_var_cols):
                    f.write(f"  - {col}\n")
            else:
                f.write("  None\n")
            f.write("\n")
            
            f.write("Null fill summary:\n")
            if filled_cols:
                for col in sorted(filled_cols.keys()):
                    f.write(f"  - {col}: {filled_cols[col]}\n")
            else:
                f.write("  No columns required filling\n")
            f.write("\n")
            
            f.write(f"Duplicate transaction count: {duplicate_count}\n\n")
            f.write(f"Log transform needed: {needs_log_transform}\n")
            
    except Exception as e:
        raise ReportWriteError(f"Failed to write validation report: {str(e)}") from e

def validate_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, List[str]]:
    _validate_input(df)
    
    config = ValidationConfig()
    
    working_df = df.copy()
    
    dropped_null_cols: List[str] = []
    dropped_zero_var_cols: List[str] = []
    filled_cols: Dict[str, str] = {}
    
    working_df, dropped_null_cols = _drop_high_null_columns(working_df, config)
    
    working_df = _fill_numeric_nulls(working_df, filled_cols)
    working_df = _fill_categorical_nulls(working_df, filled_cols)
    
    duplicate_count = _check_duplicate_transactions(working_df, config)
    needs_log_transform = _check_log_transform_flag(working_df, config)
    
    zero_var_cols = _find_zero_variance_columns(working_df)
    if zero_var_cols:
        dropped_zero_var_cols = zero_var_cols
        working_df = working_df.drop(columns=zero_var_cols)
    
    _write_validation_report(
        config,
        dropped_null_cols,
        dropped_zero_var_cols,
        filled_cols,
        duplicate_count,
        needs_log_transform
    )
    
    all_dropped = sorted(set(dropped_null_cols + dropped_zero_var_cols))
    
    return working_df, needs_log_transform, all_dropped

if __name__ == "__main__":
    import sys
    try:
        print("This module is designed to be imported, not run directly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)