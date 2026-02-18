#!/usr/bin/env python3
"""
validate.py
Enforces data quality gates by verifying schema and value constraints.
Deterministic, minimal, high-assurance implementation.
"""

import math
from dataclasses import dataclass
from typing import Any, List, Set

# Third-party imports (explicitly allowed stable dependencies)
import pandas as pd


# ==============================================================================
# 1. IMMUTABLE CONFIGURATION DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class ValidationConfig:
    """Frozen configuration for validation rules."""
    required_columns: Set[str]
    price_column: str
    date_column: str
    ticker_column: str
    schema_version: int

CONFIG = ValidationConfig(
    required_columns={"date", "ticker", "close_price"},
    price_column="close_price",
    date_column="date",
    ticker_column="ticker",
    schema_version=1
)


# ==============================================================================
# 2. CUSTOM EXCEPTION DEFINITIONS
# ==============================================================================

class ValidationError(Exception):
    """Base exception for all validation failures."""
    pass

class SchemaMismatchError(ValidationError):
    """Raised when DataFrame columns do not match the required schema."""
    pass

class NullValueError(ValidationError):
    """Raised when null values are detected in critical columns."""
    pass

class ValueConstraintError(ValidationError):
    """Raised when numeric values violate domain constraints (e.g., negative prices)."""
    pass

class TypeEnforcementError(ValidationError):
    """Raised when input types do not match strict contracts."""
    pass


# ==============================================================================
# 3. TYPED DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class ValidationResult:
    """Immutable container for validation outcome."""
    success: bool
    row_count: int
    version: int
    error_message: str | None


# ==============================================================================
# 4. PURE CORE LOGIC
# ==============================================================================

def _validate_type_strict(value: Any, expected_type: type, param_name: str) -> None:
    """
    Enforces strict type checking without coercion.
    Raises TypeEnforcementError on mismatch.
    """
    if not isinstance(value, expected_type):
        raise TypeEnforcementError(
            f"Parameter '{param_name}' must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

def _check_schema_columns(columns: List[str], required: Set[str]) -> None:
    """
    Verifies exact presence of required columns.
    Deterministic set operation.
    """
    present = set(columns)
    missing = required - present
    
    if missing:
        # Sort for deterministic error message
        missing_sorted = sorted(list(missing))
        raise SchemaMismatchError(f"Missing required columns: {missing_sorted}")

def _check_nulls(series: pd.Series, column_name: str) -> None:
    """
    Checks for null/NaN values in a specific series.
    Uses pandas native check. sum() on boolean series is guaranteed to be int >= 0.
    """
    # Direct count. No complex guards needed as pandas guarantees int return for sum().
    null_count: int = int(series.isna().sum())
    
    if null_count > 0:
        raise NullValueError(
            f"Column '{column_name}' contains {null_count} null values"
        )

def _check_numeric_constraints(series: pd.Series, column_name: str, min_val: float | None = None) -> None:
    """
    Validates numeric domain constraints.
    Specifically checks for negative prices if min_val=0.
    """
    # Ensure series is numeric before comparison
    if not pd.api.types.is_numeric_dtype(series):
        try:
            series = pd.to_numeric(series, errors='raise')
        except (ValueError, TypeError):
            raise ValueConstraintError(f"Column '{column_name}' contains non-numeric data")

    if min_val is not None:
        violation_mask = series < min_val
        violation_count: int = int(violation_mask.sum())
        
        if violation_count > 0:
            raise ValueConstraintError(
                f"Column '{column_name}' contains {violation_count} values below {min_val}"
            )
    
    # Explicit Infinity/NaN check using vectorized operations
    if series.isin([math.inf, -math.inf]).any():
        raise ValueConstraintError(f"Column '{column_name}' contains infinite values")
    
    # Redundant safety check for NaN specifically in numeric context
    if series.isna().any():
        raise NullValueError(f"Column '{column_name}' contains NaN values after numeric conversion")

def validate_dataframe_structure(df: pd.DataFrame, config: ValidationConfig) -> ValidationResult:
    """
    Pure function to validate DataFrame structure and content.
    Returns ValidationResult on success, raises specific ValidationError on failure.
    """
    # Pre-check: Empty DataFrame is a structural failure for this pipeline
    if len(df) == 0:
        raise SchemaMismatchError("Input DataFrame is empty")

    # 1. Schema Check
    _check_schema_columns(list(df.columns), config.required_columns)
    
    # 2. Null Checks on Critical Columns
    _check_nulls(df[config.price_column], config.price_column)
    _check_nulls(df[config.date_column], config.date_column)
    _check_nulls(df[config.ticker_column], config.ticker_column)
    
    # 3. Value Constraints (No negative prices)
    _check_numeric_constraints(df[config.price_column], config.price_column, min_val=0.0)
    
    # 4. Success State
    return ValidationResult(
        success=True,
        row_count=len(df),
        version=config.schema_version,
        error_message=None
    )


# ==============================================================================
# 5. MINIMAL SIDE-EFFECT BOUNDARY FUNCTIONS
# ==============================================================================

def load_and_validate(file_path: str) -> pd.DataFrame:
    """
    Boundary function: Handles I/O and orchestrates validation.
    Input: file_path (str)
    Output: pd.DataFrame (validated)
    Side Effects: Reads file from disk.
    """
    # Strict Input Contract Validation
    _validate_type_strict(file_path, str, "file_path")
    
    if not file_path:
        raise TypeEnforcementError("Parameter 'file_path' cannot be empty string")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise ValidationError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValidationError("Input file is empty")
    except Exception as e:
        raise ValidationError(f"Failed to load CSV: {str(e)}")

    # Validate Structure
    result = validate_dataframe_structure(df, CONFIG)
    
    # Assert post-condition
    assert result.success is True, "Validation logic failed to raise exception on error"
    assert result.row_count > 0, "Validated dataset must contain rows"
    
    return df


# ==============================================================================
# 6. EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    target_file = r"D:\applied-finance-ml\project-01-flat-variance-detector\data\raw\synthetic_prices.csv"
    
    try:
        validated_df = load_and_validate(target_file)
        print(f"VALIDATION_SUCCESS: Rows={len(validated_df)}, Version={CONFIG.schema_version}")
    except ValidationError as e:
        print(f"VALIDATION_FAILURE: {e}")
        raise SystemExit(1)
    except Exception as e:
        print(f"CRITICAL_SYSTEM_ERROR: {e}")
        raise SystemExit(2)