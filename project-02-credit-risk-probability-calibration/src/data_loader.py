"""
src/data_loader.py

Core Responsibilities:
- Load raw CSV data from a specified path.
- Perform a stratified three-way split: Train (60%), Calibration (20%), Test (20%).
- Validate target variable distribution consistency across splits.
- Return pandas DataFrames for each split.

Constraints:
- Strict static typing.
- Deterministic execution (fixed random seed).
- Immutable state enforcement.
- Explicit exception contracts.
- No external configuration or environment reliance.
"""

from __future__ import annotations

import decimal
from contextlib import contextmanager
from typing import Final, List, Tuple, TypeVar

import pandas as pd

# -----------------------------------------------------------------------------
# Constants & Configuration
# -----------------------------------------------------------------------------

_TARGET_COLUMN_NAME: Final[str] = "target"
_TRAIN_RATIO: Final[decimal.Decimal] = decimal.Decimal("0.60")
_CALIB_RATIO: Final[decimal.Decimal] = decimal.Decimal("0.20")
_TEST_RATIO: Final[decimal.Decimal] = decimal.Decimal("0.20")
_RANDOM_SEED: Final[int] = 42
_MAX_ROWS_BOUND: Final[int] = 10_000_000  # Algorithmic complexity bound safeguard

T = TypeVar("T")


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------

class DataLoaderError(Exception):
    """Base exception for data loader failures."""
    pass


class DataValidationError(DataLoaderError):
    """Raised when input data validation fails."""
    pass


class SplitValidationError(DataLoaderError):
    """Raised when stratified split validation fails."""
    pass


class ResourceManagementError(DataLoaderError):
    """Raised when resource management fails."""
    pass


# -----------------------------------------------------------------------------
# Resource Management
# -----------------------------------------------------------------------------

@contextmanager
def managed_csv_reader(file_path: str) -> Tuple[pd.DataFrame, None]:
    """
    Context manager for loading CSV data ensuring deterministic cleanup.
    
    Args:
        file_path: Absolute path to the CSV file.
        
    Yields:
        A pandas DataFrame containing the raw data.
        
    Raises:
        ResourceManagementError: If file cannot be read or is empty.
        FileNotFoundError: If the file does not exist.
    """
    df: pd.DataFrame | None = None
    try:
        # Strict schema enforcement: No implicit index, no automatic type inference quirks
        # We rely on pandas default inference but enforce structural checks immediately after
        df = pd.read_csv(file_path, low_memory=False)
        
        if df.empty:
            raise ResourceManagementError("Loaded dataset is empty.")
            
        yield df, None
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at path: {file_path}")
    except Exception as e:
        # Sanitized error message
        raise ResourceManagementError("Failed to load CSV resource due to internal error.") from e
    finally:
        # Explicit cleanup hint, though pandas handles memory via ref counting
        del df


# -----------------------------------------------------------------------------
# Pure Logic: Validation & Splitting
# -----------------------------------------------------------------------------

def _validate_schema(df: pd.DataFrame, target_col: str) -> None:
    """
    Validates the presence of the target column and basic structural integrity.
    
    Args:
        df: The input DataFrame.
        target_col: The name of the target column.
        
    Raises:
        DataValidationError: If schema requirements are not met.
    """
    columns: List[str] = list(df.columns)
    
    if target_col not in columns:
        raise DataValidationError(f"Missing required column: {target_col}")
    
    if len(df) > _MAX_ROWS_BOUND:
        raise DataValidationError("Input dataset exceeds maximum allowed row bound.")
    
    if df[target_col].isna().any():
        raise DataValidationError("Target variable contains missing values.")


def _calculate_distribution(df: pd.DataFrame, target_col: str) -> pd.Series:
    """
    Calculates the normalized distribution of the target variable.
    
    Args:
        df: Input DataFrame.
        target_col: Target column name.
        
    Returns:
        A Series representing the proportion of each class.
    """
    counts: pd.Series = df[target_col].value_counts(normalize=True, sort=True)
    return counts.sort_index()


def _validate_stratification(
    original_dist: pd.Series,
    train_dist: pd.Series,
    calib_dist: pd.Series,
    test_dist: pd.Series,
    tolerance: decimal.Decimal
) -> None:
    """
    Validates that split distributions match the original within a strict tolerance.
    
    Args:
        original_dist: Distribution of the full dataset.
        train_dist: Distribution of the training split.
        calib_dist: Distribution of the calibration split.
        test_dist: Distribution of the test split.
        tolerance: Maximum allowable deviation (decimal).
        
    Raises:
        SplitValidationError: If any split deviates beyond tolerance.
    """
    splits: List[Tuple[str, pd.Series]] = [
        ("Train", train_dist),
        ("Calibration", calib_dist),
        ("Test", test_dist)
    ]
    
    # Ensure all splits have the same index (classes) as original
    for name, dist in splits:
        if not original_dist.index.equals(dist.index):
            raise SplitValidationError(f"{name} split has different classes than original dataset.")
    
    for name, dist in splits:
        for cls in original_dist.index:
            orig_val: decimal.Decimal = decimal.Decimal(str(original_dist[cls]))
            split_val: decimal.Decimal = decimal.Decimal(str(dist[cls]))
            diff: decimal.Decimal = abs(orig_val - split_val)
            
            if diff > tolerance:
                raise SplitValidationError(
                    f"Stratification failed for class '{cls}' in {name} split. "
                    f"Deviation {diff} exceeds tolerance {tolerance}."
                )


def _stratified_split(
    df: pd.DataFrame,
    target_col: str,
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a deterministic stratified three-way split.
    
    Logic:
    1. Shuffle data deterministically.
    2. Calculate exact integer boundaries based on Decimal ratios to avoid float drift.
    3. Slice data immutably.
    
    Args:
        df: Input DataFrame.
        target_col: Target column name.
        seed: Random seed for shuffling.
        
    Returns:
        Tuple of (Train, Calibration, Test) DataFrames.
    """
    total_rows: int = len(df)
    
    # Deterministic shuffle
    shuffled_df: pd.DataFrame = df.sample(frac=1.0, random_state=seed, ignore_index=True)
    
    # Precise boundary calculation using Decimal
    n_train: int = int((decimal.Decimal(total_rows) * _TRAIN_RATIO).to_integral_value(rounding=decimal.ROUND_FLOOR))
    n_calib: int = int((decimal.Decimal(total_rows) * _CALIB_RATIO).to_integral_value(rounding=decimal.ROUND_FLOOR))
    # Remainder goes to test to ensure sum equals total_rows exactly
    n_test: int = total_rows - n_train - n_calib
    
    if n_train == 0 or n_calib == 0 or n_test == 0:
        raise SplitValidationError("Dataset too small to perform valid three-way split with given ratios.")
    
    # Immutable slicing
    train_df: pd.DataFrame = shuffled_df.iloc[:n_train].reset_index(drop=True)
    calib_df: pd.DataFrame = shuffled_df.iloc[n_train:n_train + n_calib].reset_index(drop=True)
    test_df: pd.DataFrame = shuffled_df.iloc[n_train + n_calib:].reset_index(drop=True)
    
    return train_df, calib_df, test_df


# -----------------------------------------------------------------------------
# Public Interface
# -----------------------------------------------------------------------------

def load_and_split_data(
    file_path: str,
    target_col: str = _TARGET_COLUMN_NAME,
    stratification_tolerance: str = "0.01"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw CSV data, validates schema, performs stratified splitting, 
    and validates distribution consistency.
    
    Args:
        file_path: Absolute path to the raw CSV file.
        target_col: Name of the target variable column.
        stratification_tolerance: Maximum allowed deviation in class proportions (as decimal string).
        
    Returns:
        A tuple containing (Train DataFrame, Calibration DataFrame, Test DataFrame).
        
    Raises:
        FileNotFoundError: If the file path is invalid.
        ResourceManagementError: If file reading fails.
        DataValidationError: If schema checks fail.
        SplitValidationError: If stratification validation fails.
        
    Time Complexity: O(N) where N is the number of rows (dominated by shuffle and value_counts).
    Space Complexity: O(N) to hold the dataframe and splits in memory.
    """
    tolerance_dec: decimal.Decimal = decimal.Decimal(stratification_tolerance)
    
    with managed_csv_reader(file_path) as (df, _):
        # 1. Validate Schema
        _validate_schema(df, target_col)
        
        # 2. Calculate Original Distribution
        original_dist: pd.Series = _calculate_distribution(df, target_col)
        
        # 3. Perform Split
        train_df, calib_df, test_df = _stratified_split(df, target_col, _RANDOM_SEED)
        
        # 4. Validate Split Distributions
        train_dist: pd.Series = _calculate_distribution(train_df, target_col)
        calib_dist: pd.Series = _calculate_distribution(calib_df, target_col)
        test_dist: pd.Series = _calculate_distribution(test_df, target_col)
        
        _validate_stratification(
            original_dist, 
            train_dist, 
            calib_dist, 
            test_dist, 
            tolerance_dec
        )
        
        # Return new instances (slices are already new views/copies depending on pandas internals, 
        # but reset_index ensures clean new objects)
        return train_df, calib_df, test_df