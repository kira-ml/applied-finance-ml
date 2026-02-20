import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

# ==============================================================================
# 1. IMMUTABLE CONFIGURATION DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class IngestConfig:
    source_path: str
    required_columns: tuple
    
    # Expected minimal dtype mapping for validation (strict subset check)
    # We only check that these columns exist and are not completely wrong types
    expected_dtypes: tuple = (
        ("income", "float"),
        ("loan_amount", "float"),
        ("revolving_balance", "float"),
        ("num_delinquencies", "int"),
        ("open_credit_lines", "int"),
        ("months_employed", "int"),
        ("prior_default", "int"),
        ("co_applicant", "int"),
        ("loan_purpose", "object"), # string/object
        ("default", "int")
    )

# ==============================================================================
# 2. CUSTOM EXCEPTION DEFINITIONS
# ==============================================================================

class IngestionError(Exception):
    """Base exception for ingestion failures."""
    pass

class FileNotFoundError(IngestionError):
    """Raised when the source CSV file does not exist."""
    pass

class MissingColumnError(IngestionError):
    """Raised when a required column is missing from the dataset."""
    pass

class DtypeValidationError(IngestionError):
    """Raised when a column has an unexpected data type."""
    pass

class EmptyDatasetError(IngestionError):
    """Raised when the loaded dataset contains no rows."""
    pass

# ==============================================================================
# 3. TYPED DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class IngestResult:
    dataframe: pd.DataFrame
    row_count: int
    column_count: int
    source_path: str
    version: str = "1.0"

# ==============================================================================
# 4. PURE CORE LOGIC
# ==============================================================================

def _validate_file_exists(path: str) -> None:
    """Explicit check for file existence before I/O."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Source file not found: {path}")

def _validate_columns(df: pd.DataFrame, required: tuple) -> None:
    """Exhaustive validation of required column presence."""
    existing_cols = set(df.columns)
    for col in required:
        if col not in existing_cols:
            raise MissingColumnError(f"Required column missing: {col}")

def _validate_dtypes_basic(df: pd.DataFrame, expectations: tuple) -> None:
    """Basic dtype expectation check without strict enforcement."""
    for col_name, expected_kind in expectations:
        if col_name not in df.columns:
            continue # Covered by column validation, but safe guard here
        
        actual_dtype = df[col_name].dtype
        
        # Map numpy/pandas kinds to simple strings for comparison
        kind_map = {
            'i': 'int',
            'f': 'float',
            'u': 'int',
            'O': 'object',
            'U': 'object',
            'b': 'bool'
        }
        
        actual_kind = kind_map.get(actual_dtype.kind, str(actual_dtype))
        
        # Simple containment check for flexibility (e.g., int8, int64 both match 'int')
        if expected_kind == "int" and actual_kind != "int":
             # Allow bools to pass as ints if strictly 0/1, but for now strict kind check
             if not (actual_kind == "bool" and expected_kind == "int"):
                 raise DtypeValidationError(f"Column '{col_name}' expected kind '{expected_kind}', got '{actual_kind}'")
        elif expected_kind == "float" and actual_kind != "float":
            raise DtypeValidationError(f"Column '{col_name}' expected kind '{expected_kind}', got '{actual_kind}'")
        elif expected_kind == "object" and actual_kind != "object":
            raise DtypeValidationError(f"Column '{col_name}' expected kind '{expected_kind}', got '{actual_kind}'")

def _validate_non_empty(df: pd.DataFrame) -> None:
    """Ensure dataset is not empty."""
    if len(df) == 0:
        raise EmptyDatasetError("Loaded dataset contains zero rows.")

# ==============================================================================
# 5. MINIMAL SIDE-EFFECT BOUNDARY FUNCTIONS
# ==============================================================================

def _load_csv_raw(path: str) -> pd.DataFrame:
    """Isolated I/O operation to load CSV."""
    try:
        # Explicit parameters to avoid implicit behavior
        df = pd.read_csv(filepath_or_buffer=path, index_col=False)
        return df
    except Exception as e:
        raise IngestionError(f"Failed to read CSV from {path}: {e}")

# ==============================================================================
# 6. EXECUTION ENTRY POINT
# ==============================================================================

def run_ingestion(source_path: str) -> IngestResult:
    """
    Main entry point for data ingestion.
    Loads CSV, validates structure, returns typed result.
    """
    # 1. Input Contract Validation
    if not isinstance(source_path, str):
        raise IngestionError("source_path must be a string")
    if not source_path:
        raise IngestionError("source_path cannot be empty")

    # 2. Configuration Construction
    config = IngestConfig(
        source_path=source_path,
        required_columns=(
            "income", "loan_amount", "revolving_balance",
            "num_delinquencies", "open_credit_lines", "months_employed",
            "prior_default", "co_applicant", "loan_purpose", "default"
        )
    )

    # 3. Guarded Initialization (File Existence)
    _validate_file_exists(config.source_path)

    # 4. I/O Boundary (Load)
    df = _load_csv_raw(config.source_path)

    # 5. Core Validation Logic
    _validate_columns(df, config.required_columns)
    _validate_dtypes_basic(df, config.expected_dtypes)
    _validate_non_empty(df)

    # 6. Return Result Container
    return IngestResult(
        dataframe=df,
        row_count=len(df),
        column_count=len(df.columns),
        source_path=config.source_path,
        version="1.0"
    )

if __name__ == "__main__":
    import sys

    # Direct instantiation for script execution
    SOURCE_FILE = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\data\raw\credit_data.csv"
    
    try:
        result = run_ingestion(source_path=SOURCE_FILE)
        print(f"Ingestion successful.")
        print(f"Rows: {result.row_count}, Columns: {result.column_count}")
        print(f"Source: {result.source_path}")
        print(f"Schema Version: {result.version}")
    except IngestionError as e:
        # Single, irreversible failure mode
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)