import pandas as pd
from dataclasses import dataclass
from typing import List
from pathlib import Path

# ==============================================================================
# 1. IMMUTABLE CONFIGURATION DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class Config:
    prices_path: str
    date_column: str
    serialization_version: int

CONFIG = Config(
    prices_path=r"D:\applied-finance-ml\project-01-flat-variance-detector\data\prices.csv",
    date_column="date",
    serialization_version=1
)

# ==============================================================================
# 2. CUSTOM EXCEPTION DEFINITIONS
# ==============================================================================

class DataLoaderError(Exception):
    """Base exception for data loading failures."""
    pass

class FileNotFoundError(DataLoaderError):
    """Raised when the specified CSV file does not exist."""
    pass

class SchemaValidationError(DataLoaderError):
    """Raised when the CSV schema violates expectations."""
    pass

class DataIntegrityError(DataLoaderError):
    """Raised when data content violates integrity constraints."""
    pass

# ==============================================================================
# 3. TYPED DATA STRUCTURES
# ==============================================================================

# No complex state containers needed; the DataFrame is the state container.
# Using a simple alias for clarity on return type.
PriceDataFrame = pd.DataFrame

# ==============================================================================
# 4. PURE CORE LOGIC
# ==============================================================================

def _validate_file_exists(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Data file not found at path: {path}")

def _validate_schema(df: pd.DataFrame, expected_date_col: str) -> None:
    if expected_date_col not in df.columns:
        raise SchemaValidationError(f"Missing required column: '{expected_date_col}'")
    
    if len(df.columns) <= 1:
        raise SchemaValidationError("DataFrame must contain date column and at least one asset column.")

def _validate_data_integrity(df: pd.DataFrame) -> None:
    null_counts = df.isnull().sum()
    for col in df.columns:
        if null_counts[col] == len(df):
            raise DataIntegrityError(f"Column '{col}' is entirely null.")

def _load_and_validate_core(path: str, date_col: str) -> PriceDataFrame:
    # Explicit read with strict parsing
    df = pd.read_csv(path)
    
    _validate_schema(df, date_col)
    
    # Explicit conversion with error raising
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='raise')
    except Exception as e:
        raise SchemaValidationError(f"Failed to parse date column '{date_col}': {str(e)}")
    
    _validate_data_integrity(df)
    
    # Set index explicitly
    df.set_index(date_col, inplace=True)
    
    # Verify index is datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise SchemaValidationError("Index failed to convert to datetime type.")
        
    return df

# ==============================================================================
# 5. MINIMAL SIDE-EFFECT BOUNDARY FUNCTIONS
# ==============================================================================

def load_prices(config: Config) -> PriceDataFrame:
    """
    Public interface to load prices.
    Validates inputs, checks file existence, and returns a clean DataFrame.
    """
    # Input contract validation
    if not isinstance(config, Config):
        raise DataLoaderError("Configuration must be an instance of Config dataclass.")
    if not isinstance(config.prices_path, str):
        raise DataLoaderError("Prices path must be a string.")
    if not isinstance(config.date_column, str):
        raise DataLoaderError("Date column name must be a string.")
        
    _validate_file_exists(config.prices_path)
    
    # Delegate to pure core logic
    df = _load_and_validate_core(config.prices_path, config.date_column)
    
    return df

# ==============================================================================
# 6. EXECUTION ENTRY POINT
# ==============================================================================

_setup_complete = False

def _guard_setup() -> None:
    global _setup_complete
    if _setup_complete:
        return
    # Minimal config validation
    if CONFIG.serialization_version != 1:
        raise DataLoaderError("Unsupported serialization version in config.")
    _setup_complete = True

def main() -> None:
    _guard_setup()
    
    assert _setup_complete, "Setup guard failed."
    
    df = load_prices(CONFIG)
    
    # Minimal visibility output
    print(f"Successfully loaded {len(df)} rows.")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

if __name__ == "__main__":
    main()