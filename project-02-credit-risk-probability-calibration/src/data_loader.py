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
_MAX_ROWS_BOUND: Final[int] = 10_000_000
_DEFAULT_TOLERANCE: Final[str] = "0.02"

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
        df = pd.read_csv(file_path, low_memory=False)
        
        if df.empty:
            raise ResourceManagementError("Loaded dataset is empty.")
            
        yield df, None
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at path: {file_path}")
    except pd.errors.EmptyDataError:
        raise ResourceManagementError("Loaded dataset is empty.")
    except ResourceManagementError:
        raise
    except Exception as e:
        if isinstance(e, (pd.errors.ParserError, OSError)):
            raise ResourceManagementError("Failed to load CSV resource due to internal error.") from e
        raise


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
    Performs a deterministic TRUE stratified three-way split.
    
    Logic:
    1. Group by target class.
    2. Shuffle and split EACH group independently to ensure perfect distribution.
    3. Concatenate results.
    
    Args:
        df: Input DataFrame.
        target_col: Target column name.
        seed: Random seed for shuffling.
        
    Returns:
        Tuple of (Train, Calibration, Test) DataFrames.
    """
    train_parts: List[pd.DataFrame] = []
    calib_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    
    # Group by target to ensure stratification
    grouped = df.groupby(target_col, sort=True)
    
    for _, group_df in grouped:
        n_total = len(group_df)
        
        # Deterministic shuffle within the group
        shuffled_group = group_df.sample(frac=1.0, random_state=seed, ignore_index=True)
        
        # Calculate boundaries
        n_train = int((decimal.Decimal(n_total) * _TRAIN_RATIO).to_integral_value(rounding=decimal.ROUND_FLOOR))
        n_calib = int((decimal.Decimal(n_total) * _CALIB_RATIO).to_integral_value(rounding=decimal.ROUND_FLOOR))
        n_test = n_total - n_train - n_calib
        
        if n_train == 0 or n_calib == 0 or n_test == 0:
            # If a specific class is too small to split, the whole dataset is effectively too small
            # for a valid 3-way stratified split.
            raise SplitValidationError("Dataset too small to perform valid three-way split with given ratios.")
        
        train_parts.append(shuffled_group.iloc[:n_train])
        calib_parts.append(shuffled_group.iloc[n_train:n_train + n_calib])
        test_parts.append(shuffled_group.iloc[n_train + n_calib:])
    
    # Concatenate and reset index to create clean, immutable new instances
    train_df = pd.concat(train_parts, ignore_index=True)
    calib_df = pd.concat(calib_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    
    # Final deterministic sort to ensure byte-level reproducibility regardless of group iteration order
    # Although groupby(sort=True) helps, explicit sorting guarantees stability.
    train_df = train_df.sort_values(by=list(train_df.columns)).reset_index(drop=True)
    calib_df = calib_df.sort_values(by=list(calib_df.columns)).reset_index(drop=True)
    test_df = test_df.sort_values(by=list(test_df.columns)).reset_index(drop=True)

    return train_df, calib_df, test_df


# -----------------------------------------------------------------------------
# Public Interface
# -----------------------------------------------------------------------------

def load_and_split_data(
    file_path: str,
    target_col: str = _TARGET_COLUMN_NAME,
    stratification_tolerance: str = _DEFAULT_TOLERANCE
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
        
    Time Complexity: O(N log N) due to sorting within groups and final concat sort.
    Space Complexity: O(N)
    """
    tolerance_dec: decimal.Decimal = decimal.Decimal(stratification_tolerance)
    
    with managed_csv_reader(file_path) as (df, _):
        _validate_schema(df, target_col)
        
        original_dist: pd.Series = _calculate_distribution(df, target_col)
        
        train_df, calib_df, test_df = _stratified_split(df, target_col, _RANDOM_SEED)
        
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
        
        return train_df, calib_df, test_df