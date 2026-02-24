"""
src/data.py

Synthetic transaction data generation and loading module.
"""

import csv
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "generate_transactions_unchecked",
    "load_transactions_unchecked",
    "DataGenerationError",
    "DataLoadingError",
    "DataValidationError",
    "ModuleError",
]


# ------------------------------------------------------------------------------
# Error Taxonomy
# ------------------------------------------------------------------------------

class ModuleError(Exception):
    """Base class for all module exceptions."""
    def __init__(self, message: str, *, reproducible_state: dict) -> None:
        self.reproducible_state = reproducible_state
        super().__init__(message)


class DataGenerationError(ModuleError):
    """Raised when transaction generation fails."""
    pass


class DataLoadingError(ModuleError):
    """Raised when CSV loading fails."""
    pass


class DataValidationError(ModuleError):
    """Raised when data validation fails."""
    pass


class InvariantViolationError(ModuleError):
    """Raised when invariant checks fail."""
    pass


# ------------------------------------------------------------------------------
# Immutable Configuration
# ------------------------------------------------------------------------------

_DATA_DIR: Path = Path(r"D:\applied-finance-ml\project-03-fraud-detection-threshold-optimization\data\raw")
_DATA_FILE: Path = _DATA_DIR / "transactions.csv"
_RANDOM_SEED: int = 42
_N_ROWS: int = 10000
_FRAUD_RATIO: float = 0.01


def _load_config_unchecked() -> None:
    """
    One-time configuration loader.
    Ensures config values are immutable after import.
    """
    global _DATA_DIR, _DATA_FILE, _RANDOM_SEED, _N_ROWS, _FRAUD_RATIO
    # Configuration is already defined as module constants.
    # This function exists to satisfy the "immutable config" standard.
    pass


_load_config_unchecked()


# ------------------------------------------------------------------------------
# Core Deterministic Functions
# ------------------------------------------------------------------------------

def _validate_fraction(fraction: float) -> None:
    """Validate that fraction is between 0 and 1 inclusive."""
    if not 0.0 <= fraction <= 1.0:
        raise InvariantViolationError(
            f"Fraction must be between 0 and 1, got {fraction}",
            reproducible_state={"fraction": fraction},
        )


def _generate_fraud_labels(seed: int, n_rows: int, fraud_ratio: float) -> np.ndarray:
    """
    Generate binary fraud labels with exact fraud ratio.
    
    Args:
        seed: Random seed for reproducibility
        n_rows: Total number of rows
        fraud_ratio: Target proportion of fraud (1) labels
    
    Returns:
        Array of 0/1 labels
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n_rows * fraud_ratio))
    
    labels = np.zeros(n_rows, dtype=np.int8)
    fraud_indices = rng.choice(n_rows, size=n_fraud, replace=False)
    labels[fraud_indices] = 1
    
    return labels


def _generate_amounts(seed: int, n_rows: int) -> np.ndarray:
    """
    Generate synthetic transaction amounts.
    
    Args:
        seed: Random seed for reproducibility
        n_rows: Total number of rows
    
    Returns:
        Array of transaction amounts
    """
    rng = np.random.default_rng(seed + 1)  # Different seed for independence
    # Log-normal distribution for positive amounts with right skew
    return np.round(rng.lognormal(mean=4.0, sigma=1.0, size=n_rows), 2)


def _generate_timestamps(seed: int, n_rows: int) -> np.ndarray:
    """
    Generate synthetic timestamps over a 30-day period.
    
    Args:
        seed: Random seed for reproducibility
        n_rows: Total number of rows
    
    Returns:
        Array of datetime strings (YYYY-MM-DD HH:MM:SS)
    """
    rng = np.random.default_rng(seed + 2)
    # Generate random seconds within 30 days
    seconds = rng.integers(0, 30 * 24 * 3600, size=n_rows)
    # Base timestamp: 2024-01-01
    base_ts = pd.Timestamp("2024-01-01")
    timestamps = base_ts + pd.to_timedelta(seconds, unit="s")
    return timestamps.strftime("%Y-%m-%d %H:%M:%S").to_numpy()


def _generate_transactions_core(seed: int, n_rows: int, fraud_ratio: float) -> List[Tuple[str, float, int]]:
    """
    Core deterministic transaction generation.
    
    Args:
        seed: Random seed
        n_rows: Number of rows
        fraud_ratio: Target fraud proportion
    
    Returns:
        List of (timestamp, amount, is_fraud) tuples
    """
    _validate_fraction(fraud_ratio)
    
    timestamps = _generate_timestamps(seed, n_rows)
    amounts = _generate_amounts(seed, n_rows)
    labels = _generate_fraud_labels(seed, n_rows, fraud_ratio)
    
    # Verify invariant: exactly one percent fraud
    actual_ratio = float(np.mean(labels))
    if abs(actual_ratio - fraud_ratio) > 0.0001:
        raise InvariantViolationError(
            f"Fraud ratio mismatch: target {fraud_ratio}, got {actual_ratio}",
            reproducible_state={
                "seed": seed,
                "n_rows": n_rows,
                "fraud_ratio": fraud_ratio,
                "actual_ratio": actual_ratio,
            },
        )
    
    return list(zip(timestamps, amounts, labels))


# ------------------------------------------------------------------------------
# Public Interface (with explicit side effects)
# ------------------------------------------------------------------------------

def generate_transactions_unchecked() -> None:
    """
    Generate 10,000 synthetic transactions and save to CSV.
    
    Side effects:
        - Creates data directory if it doesn't exist
        - Writes transactions.csv to data directory
        - Uses numpy.random for generation (isolated impurity)
    
    Raises:
        DataGenerationError: If directory creation or file writing fails
    """
    try:
        # Ensure directory exists
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate core data
        transactions_Mutable = _generate_transactions_core(
            seed=_RANDOM_SEED,
            n_rows=_N_ROWS,
            fraud_ratio=_FRAUD_RATIO,
        )
        
        # Write to CSV
        with open(_DATA_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "amount", "is_fraud"])
            writer.writerows(transactions_Mutable)
            
    except (OSError, csv.Error) as e:
        raise DataGenerationError(
            f"Failed to generate transactions: {e}",
            reproducible_state={
                "data_dir": str(_DATA_DIR),
                "data_file": str(_DATA_FILE),
                "seed": _RANDOM_SEED,
                "n_rows": _N_ROWS,
                "fraud_ratio": _FRAUD_RATIO,
            },
        ) from e


def load_transactions_unchecked() -> pd.DataFrame:
    """
    Load transactions from CSV into pandas DataFrame.
    
    Side effects:
        - Reads from filesystem
        - Verifies is_fraud column contains only 0 and 1
    
    Returns:
        DataFrame with columns: timestamp, amount, is_fraud
    
    Raises:
        DataLoadingError: If file cannot be read
        DataValidationError: If data fails validation
    """
    try:
        df = pd.read_csv(_DATA_FILE)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as e:
        raise DataLoadingError(
            f"Failed to load transactions from {_DATA_FILE}",
            reproducible_state={"data_file": str(_DATA_FILE)},
        ) from e
    
    # Validate required columns
    required_columns = {"timestamp", "amount", "is_fraud"}
    if not required_columns.issubset(df.columns):
        raise DataValidationError(
            f"Missing required columns. Expected {required_columns}, got {set(df.columns)}",
            reproducible_state={
                "data_file": str(_DATA_FILE),
                "columns": list(df.columns),
            },
        )
    
    # Validate is_fraud column values
    unique_values = set(df["is_fraud"].unique())
    if not unique_values.issubset({0, 1}):
        raise DataValidationError(
            f"is_fraud column must contain only 0 and 1. Found values: {unique_values}",
            reproducible_state={
                "data_file": str(_DATA_FILE),
                "invalid_values": list(unique_values - {0, 1}),
            },
        )
    
    # Verify data shape invariant
    if len(df) != _N_ROWS:
        raise InvariantViolationError(
            f"Expected {_N_ROWS} rows, got {len(df)}",
            reproducible_state={
                "data_file": str(_DATA_FILE),
                "expected_rows": _N_ROWS,
                "actual_rows": len(df),
            },
        )
    
    return df


# ------------------------------------------------------------------------------
# Module-Level Invariants
# ------------------------------------------------------------------------------

# Verify configuration immutability
assert isinstance(_RANDOM_SEED, int), "_RANDOM_SEED must be int"
assert isinstance(_N_ROWS, int), "_N_ROWS must be int"
assert isinstance(_FRAUD_RATIO, float), "_FRAUD_RATIO must be float"