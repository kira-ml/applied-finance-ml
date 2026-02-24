"""
test_data.py

Unit tests for src.data module.
Minimal, deterministic testing with no over-engineering.
"""

import csv
import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src import data

# ------------------------------------------------------------------------------
# Test Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def temp_data_dir():
    """Provide temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def patch_config_paths(temp_data_dir):
    """Patch module-level paths to use temp directory."""
    with mock.patch.object(data, "_DATA_DIR", temp_data_dir):
        with mock.patch.object(data, "_DATA_FILE", temp_data_dir / "transactions.csv"):
            yield


# ------------------------------------------------------------------------------
# Test Configuration and Invariants
# ------------------------------------------------------------------------------

def test_config_values_are_correct_types():
    """Verify configuration constants have correct types."""
    assert isinstance(data._RANDOM_SEED, int)
    assert isinstance(data._N_ROWS, int)
    assert isinstance(data._FRAUD_RATIO, float)


def test_config_values_are_immutable():
    """
    Verify config values cannot be mutated after import.
    Note: Python allows reassignment, but our module design treats them as immutable.
    We verify they're not accidentally mutated during module execution.
    """
    original_seed = data._RANDOM_SEED
    original_rows = data._N_ROWS
    original_ratio = data._FRAUD_RATIO
    
    # Run module functions that might accidentally modify config
    data._validate_fraction(0.5)
    data._generate_fraud_labels(seed=42, n_rows=100, fraud_ratio=0.01)
    data._generate_amounts(seed=42, n_rows=100)
    data._generate_timestamps(seed=42, n_rows=100)
    
    # Verify values haven't changed
    assert data._RANDOM_SEED == original_seed
    assert data._N_ROWS == original_rows
    assert data._FRAUD_RATIO == original_ratio


# ------------------------------------------------------------------------------
# Test Core Deterministic Functions
# ------------------------------------------------------------------------------

def test_validate_fraction_valid_values():
    """Test fraction validation with valid inputs."""
    # Should not raise
    data._validate_fraction(0.0)
    data._validate_fraction(0.5)
    data._validate_fraction(1.0)


def test_validate_fraction_invalid_values():
    """Test fraction validation rejects invalid inputs."""
    with pytest.raises(data.InvariantViolationError):
        data._validate_fraction(-0.1)
    with pytest.raises(data.InvariantViolationError):
        data._validate_fraction(1.1)


def test_generate_fraud_labels_exact_ratio():
    """Test fraud label generation hits target ratio exactly."""
    labels = data._generate_fraud_labels(seed=42, n_rows=10000, fraud_ratio=0.01)
    
    assert len(labels) == 10000
    assert labels.dtype == np.int8
    assert set(labels).issubset({0, 1})
    
    n_fraud = int(sum(labels))
    assert n_fraud == 100  # 1% of 10000


def test_generate_fraud_labels_reproducible():
    """Test same seed produces identical labels."""
    labels1 = data._generate_fraud_labels(seed=123, n_rows=100, fraud_ratio=0.1)
    labels2 = data._generate_fraud_labels(seed=123, n_rows=100, fraud_ratio=0.1)
    
    np.testing.assert_array_equal(labels1, labels2)


def test_generate_amounts_positive():
    """Test amounts are positive and rounded to 2 decimals."""
    amounts = data._generate_amounts(seed=42, n_rows=1000)
    
    assert len(amounts) == 1000
    assert (amounts > 0).all()
    # Check 2 decimal places (allowing for floating point)
    assert (np.abs(amounts * 100 - np.round(amounts * 100)) < 1e-10).all()


def test_generate_amounts_reproducible():
    """Test same seed produces identical amounts."""
    amounts1 = data._generate_amounts(seed=42, n_rows=100)
    amounts2 = data._generate_amounts(seed=42, n_rows=100)
    
    np.testing.assert_array_equal(amounts1, amounts2)


def test_generate_timestamps_format():
    """Test timestamps are valid datetime strings."""
    timestamps = data._generate_timestamps(seed=42, n_rows=100)
    
    assert len(timestamps) == 100
    # Verify format: YYYY-MM-DD HH:MM:SS
    for ts in timestamps:
        pd.to_datetime(ts)  # Will raise if invalid


def test_generate_timestamps_range():
    """Test timestamps are within 30-day window."""
    timestamps = data._generate_timestamps(seed=42, n_rows=1000)
    ts_series = pd.to_datetime(timestamps)
    
    min_ts = pd.Timestamp("2024-01-01")
    max_ts = pd.Timestamp("2024-01-31")
    
    assert (ts_series >= min_ts).all()
    assert (ts_series <= max_ts).all()


def test_generate_transactions_core_invariant_enforcement():
    """Test core generation enforces fraud ratio invariant."""
    # Should pass
    data._generate_transactions_core(seed=42, n_rows=1000, fraud_ratio=0.01)
    
    # Should fail with invalid fraction
    with pytest.raises(data.InvariantViolationError):
        data._generate_transactions_core(seed=42, n_rows=1000, fraud_ratio=1.5)


# ------------------------------------------------------------------------------
# Test Public Interface (Side Effects)
# ------------------------------------------------------------------------------

def test_generate_transactions_unchecked_creates_file(patch_config_paths, temp_data_dir):
    """Test generation creates CSV file with correct structure."""
    data.generate_transactions_unchecked()
    
    assert (temp_data_dir / "transactions.csv").exists()
    
    # Read and verify content
    df = pd.read_csv(temp_data_dir / "transactions.csv")
    assert list(df.columns) == ["timestamp", "amount", "is_fraud"]
    assert len(df) == 10000
    
    # Verify fraud ratio
    fraud_ratio = df["is_fraud"].mean()
    assert abs(fraud_ratio - 0.01) < 0.001


def test_generate_transactions_unchecked_directory_creation(temp_data_dir):
    """Test generation creates directory if it doesn't exist."""
    new_dir = temp_data_dir / "nested" / "path"
    
    with mock.patch.object(data, "_DATA_DIR", new_dir):
        with mock.patch.object(data, "_DATA_FILE", new_dir / "transactions.csv"):
            data.generate_transactions_unchecked()
    
    assert new_dir.exists()
    assert (new_dir / "transactions.csv").exists()


def test_generate_transactions_unchecked_file_write_error():
    """Test proper error wrapping for write failures."""
    with mock.patch("builtins.open", side_effect=OSError("Permission denied")):
        with pytest.raises(data.DataGenerationError) as excinfo:
            data.generate_transactions_unchecked()
        
        assert "Failed to generate transactions" in str(excinfo.value)
        assert "reproducible_state" in dir(excinfo.value)


def test_load_transactions_unchecked_success(patch_config_paths, temp_data_dir):
    """Test successful loading of valid CSV."""
    # First generate data
    data.generate_transactions_unchecked()
    
    # Then load it
    df = data.load_transactions_unchecked()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10000
    assert list(df.columns) == ["timestamp", "amount", "is_fraud"]


def test_load_transactions_unchecked_file_not_found():
    """Test error when file doesn't exist."""
    with mock.patch.object(data, "_DATA_FILE", Path("/nonexistent/path.csv")):
        with pytest.raises(data.DataLoadingError):
            data.load_transactions_unchecked()


def test_load_transactions_unchecked_missing_columns(patch_config_paths, temp_data_dir):
    """Test validation catches missing columns."""
    # Create invalid CSV
    with open(temp_data_dir / "transactions.csv", "w") as f:
        f.write("wrong_col1,wrong_col2\n1,2")
    
    with pytest.raises(data.DataValidationError) as excinfo:
        data.load_transactions_unchecked()
    
    assert "Missing required columns" in str(excinfo.value)


def test_load_transactions_unchecked_invalid_fraud_values(patch_config_paths, temp_data_dir):
    """Test validation catches invalid is_fraud values."""
    # Create CSV with invalid fraud values
    with open(temp_data_dir / "transactions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "amount", "is_fraud"])
        writer.writerow(["2024-01-01", 100.0, 2])  # 2 is invalid
    
    with pytest.raises(data.DataValidationError) as excinfo:
        data.load_transactions_unchecked()
    
    assert "is_fraud column must contain only 0 and 1" in str(excinfo.value)


def test_load_transactions_unchecked_row_count_invariant(patch_config_paths, temp_data_dir):
    """Test invariant check for row count."""
    # Create CSV with wrong number of rows
    with open(temp_data_dir / "transactions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "amount", "is_fraud"])
        for i in range(500):  # Only 500 rows instead of 10000
            writer.writerow(["2024-01-01", 100.0, 0])
    
    with pytest.raises(data.InvariantViolationError) as excinfo:
        data.load_transactions_unchecked()
    
    assert "Expected 10000 rows" in str(excinfo.value)


# ------------------------------------------------------------------------------
# Test Error Taxonomy and Exception Chains
# ------------------------------------------------------------------------------

def test_all_exceptions_inherit_from_module_error():
    """Verify all custom exceptions inherit from ModuleError."""
    assert issubclass(data.DataGenerationError, data.ModuleError)
    assert issubclass(data.DataLoadingError, data.ModuleError)
    assert issubclass(data.DataValidationError, data.ModuleError)
    assert issubclass(data.InvariantViolationError, data.ModuleError)


def test_exceptions_store_reproducible_state():
    """Test exceptions capture state for reproducibility."""
    try:
        raise data.DataGenerationError("test", reproducible_state={"key": "value"})
    except data.DataGenerationError as e:
        assert e.reproducible_state == {"key": "value"}


def test_generate_transactions_unchecked_preserves_cause():
    """Test exception chaining preserves original error."""
    with mock.patch("builtins.open", side_effect=OSError("Disk full")):
        with pytest.raises(data.DataGenerationError) as excinfo:
            data.generate_transactions_unchecked()
        
        assert isinstance(excinfo.value.__cause__, OSError)


# ------------------------------------------------------------------------------
# Test Module Interface
# ------------------------------------------------------------------------------

def test_module_exports_expected_names():
    """Test __all__ contains expected public symbols."""
    expected = {
        "generate_transactions_unchecked",
        "load_transactions_unchecked",
        "DataGenerationError",
        "DataLoadingError",
        "DataValidationError",
        "ModuleError",
    }
    assert set(data.__all__) == expected


def test_private_members_not_exported():
    """Test private members are not in __all__."""
    for name in data.__all__:
        assert not name.startswith("_")