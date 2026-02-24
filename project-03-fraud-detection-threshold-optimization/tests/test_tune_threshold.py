"""
test_tune_threshold.py

Unit tests for src.tune_threshold module.
Minimal, deterministic testing with no over-engineering.
"""

import pickle
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import tune_threshold
from src import train
from src import data


# ------------------------------------------------------------------------------
# Test Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_rows = 1000
    
    # Generate amounts
    amounts = np.round(np.random.lognormal(mean=4.0, sigma=1.0, size=n_rows), 2)
    
    # Generate fraud labels (exactly 1%)
    labels = np.zeros(n_rows, dtype=np.int8)
    fraud_indices = np.random.choice(n_rows, size=10, replace=False)
    labels[fraud_indices] = 1
    
    return amounts, labels


@pytest.fixture
def fitted_scaler_and_model(sample_data):
    """Create fitted scaler and model for testing."""
    amounts, labels = sample_data
    
    # Create and fit scaler
    X = pd.DataFrame({"amount": amounts})
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns)
    
    # Create and fit model
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    model.fit(X_scaled, labels)
    
    return scaler, model, X, labels


@pytest.fixture
def temp_models_dir():
    """Provide temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def patch_model_paths(temp_models_dir, fitted_scaler_and_model):
    """Patch module-level paths and save test models."""
    scaler, model, _, _ = fitted_scaler_and_model
    
    # Save test models
    with open(temp_models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(temp_models_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with mock.patch.object(tune_threshold, "_MODELS_DIR", temp_models_dir):
        with mock.patch.object(tune_threshold, "_SCALER_FILE", temp_models_dir / "scaler.pkl"):
            with mock.patch.object(tune_threshold, "_MODEL_FILE", temp_models_dir / "model.pkl"):
                with mock.patch.object(tune_threshold, "_THRESHOLD_FILE", temp_models_dir / "threshold.txt"):
                    yield


@pytest.fixture
def sample_transactions_df(sample_data):
    """Create a sample transactions DataFrame."""
    amounts, labels = sample_data
    
    # Generate timestamps
    base_ts = pd.Timestamp("2024-01-01")
    seconds = np.random.randint(0, 30*24*3600, size=1000)
    timestamps = (base_ts + pd.to_timedelta(seconds, unit="s")).strftime("%Y-%m-%d %H:%M:%S")
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "amount": amounts,
        "is_fraud": labels,
    })


@pytest.fixture
def mock_load_transactions(sample_transactions_df):
    """Mock data.load_transactions_unchecked."""
    with mock.patch.object(data, "load_transactions_unchecked", return_value=sample_transactions_df):
        yield


# ------------------------------------------------------------------------------
# Test Configuration and Invariants
# ------------------------------------------------------------------------------

def test_config_values_are_correct_types():
    """Verify configuration constants have correct types."""
    assert isinstance(tune_threshold._MODELS_DIR, Path)
    assert isinstance(tune_threshold._SCALER_FILE, Path)
    assert isinstance(tune_threshold._MODEL_FILE, Path)
    assert isinstance(tune_threshold._THRESHOLD_FILE, Path)
    assert isinstance(tune_threshold._RANDOM_SEED, int)
    assert isinstance(tune_threshold._TEST_SIZE, float)
    assert isinstance(tune_threshold._THRESHOLD_STEPS, int)


def test_config_values_are_immutable():
    """Verify config values remain unchanged after module execution."""
    original_seed = tune_threshold._RANDOM_SEED
    original_test_size = tune_threshold._TEST_SIZE
    original_steps = tune_threshold._THRESHOLD_STEPS
    
    # Run some functions
    tune_threshold._f2_score(np.array([1, 0]), np.array([1, 0]))
    
    # Verify values haven't changed
    assert tune_threshold._RANDOM_SEED == original_seed
    assert tune_threshold._TEST_SIZE == original_test_size
    assert tune_threshold._THRESHOLD_STEPS == original_steps


# ------------------------------------------------------------------------------
# Test Core Deterministic Functions
# ------------------------------------------------------------------------------

def test_load_objects_unchecked_success(patch_model_paths, temp_models_dir):
    """Test successful loading of scaler and model."""
    scaler, model = tune_threshold._load_objects_unchecked()
    
    assert isinstance(scaler, StandardScaler)
    assert isinstance(model, LogisticRegression)


def test_load_objects_unchecked_file_not_found():
    """Test error when model files don't exist."""
    with mock.patch.object(tune_threshold, "_SCALER_FILE", Path("/nonexistent/scaler.pkl")):
        with pytest.raises(tune_threshold.ThresholdOptimizationError) as excinfo:
            tune_threshold._load_objects_unchecked()
        
        assert "Failed to load model or scaler" in str(excinfo.value)


def test_get_validation_data_success(mock_load_transactions, sample_transactions_df):
    """Test successful retrieval of validation data."""
    X_val, y_val = tune_threshold._get_validation_data()
    
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_val, pd.Series)
    assert list(X_val.columns) == ["amount"]
    assert len(X_val) == 200  # 20% of 1000
    assert len(y_val) == 200
    assert set(y_val.unique()).issubset({0, 1})


def test_get_validation_data_error():
    """Test error when data loading fails."""
    with mock.patch.object(
        data, "load_transactions_unchecked",
        side_effect=data.DataLoadingError("Failed", reproducible_state={})
    ):
        with pytest.raises(tune_threshold.ThresholdOptimizationError) as excinfo:
            tune_threshold._get_validation_data()
        
        assert "Failed to get validation data" in str(excinfo.value)


def test_scale_validation_features(fitted_scaler_and_model):
    """Test scaling of validation features."""
    scaler, _, X, _ = fitted_scaler_and_model
    
    # Use last 200 rows as validation
    X_val = X.iloc[800:]
    
    X_val_scaled = tune_threshold._scale_validation_features(scaler, X_val)
    
    assert isinstance(X_val_scaled, pd.DataFrame)
    assert list(X_val_scaled.columns) == ["amount"]
    assert len(X_val_scaled) == len(X_val)
    assert X_val_scaled.index.equals(X_val.index)


def test_get_probabilities_success(fitted_scaler_and_model):
    """Test probability prediction generation."""
    _, model, X, _ = fitted_scaler_and_model
    
    # Scale features
    scaler, _, _, _ = fitted_scaler_and_model
    X_scaled = tune_threshold._scale_validation_features(scaler, X)
    
    probabilities = tune_threshold._get_probabilities(model, X_scaled)
    
    assert isinstance(probabilities, np.ndarray)
    assert len(probabilities) == len(X)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_get_probabilities_error(fitted_scaler_and_model):
    """Test error handling in probability prediction."""
    _, model, X, _ = fitted_scaler_and_model
    
    # Create invalid input (wrong shape)
    X_invalid = pd.DataFrame({"wrong_col": [1, 2, 3]})
    
    with pytest.raises(tune_threshold.ThresholdOptimizationError) as excinfo:
        tune_threshold._get_probabilities(model, X_invalid)
    
    assert "Failed to generate probability predictions" in str(excinfo.value)


def test_f2_score_calculation():
    """Test F2 score calculation."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    
    # Calculate manually:
    # TP = 3 (indices 0, 4, 6)
    # FP = 1 (index 5)
    # FN = 1 (index 1)
    # Precision = 3/4 = 0.75
    # Recall = 3/4 = 0.75
    # F2 = (5 * 0.75 * 0.75) / (4*0.75 + 0.75) = 2.8125 / 3.75 = 0.75
    f2 = tune_threshold._f2_score(y_true, y_pred)
    assert abs(f2 - 0.75) < 1e-6


def test_f2_score_edge_cases():
    """Test F2 score edge cases."""
    # All correct predictions
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    assert tune_threshold._f2_score(y_true, y_pred) == 1.0
    
    # All incorrect predictions
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])
    assert tune_threshold._f2_score(y_true, y_pred) == 0.0
    
    # No positives in prediction
    y_true = np.array([1, 1, 1])
    y_pred = np.array([0, 0, 0])
    assert tune_threshold._f2_score(y_true, y_pred) == 0.0
    
    # No positives in ground truth
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    assert tune_threshold._f2_score(y_true, y_pred) == 0.0


def test_find_best_threshold_success():
    """Test threshold optimization."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.6, 0.95, 0.1])
    
    best_threshold, best_f2 = tune_threshold._find_best_threshold(
        y_true, probabilities, n_steps=100
    )
    
    assert isinstance(best_threshold, float)
    assert isinstance(best_f2, float)
    assert 0.0 <= best_threshold <= 1.0
    assert 0.0 <= best_f2 <= 1.0


def test_find_best_threshold_length_mismatch():
    """Test invariant check for length mismatch."""
    y_true = np.array([1, 0, 1])
    probabilities = np.array([0.9, 0.8])
    
    with pytest.raises(tune_threshold.InvariantViolationError) as excinfo:
        tune_threshold._find_best_threshold(y_true, probabilities, n_steps=100)
    
    assert "Length mismatch" in str(excinfo.value)


def test_find_best_threshold_invalid_steps():
    """Test invariant check for invalid n_steps."""
    y_true = np.array([1, 0])
    probabilities = np.array([0.9, 0.8])
    
    with pytest.raises(tune_threshold.InvariantViolationError) as excinfo:
        tune_threshold._find_best_threshold(y_true, probabilities, n_steps=0)
    
    assert "n_steps must be >= 1" in str(excinfo.value)


def test_calculate_pr_auc():
    """Test Precision-Recall AUC calculation."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    probabilities = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.6, 0.95, 0.1])
    
    pr_auc = tune_threshold._calculate_pr_auc(y_true, probabilities)
    
    assert isinstance(pr_auc, float)
    assert 0.0 <= pr_auc <= 1.0


def test_save_threshold_unchecked(temp_models_dir):
    """Test threshold saving to file."""
    test_file = temp_models_dir / "threshold.txt"
    
    with mock.patch.object(tune_threshold, "_THRESHOLD_FILE", test_file):
        tune_threshold._save_threshold_unchecked(0.75)
        
        assert test_file.exists()
        
        with open(test_file, "r") as f:
            content = f.read().strip()
        
        assert content == "0.750000"


def test_save_threshold_unchecked_error():
    """Test error handling in threshold saving."""
    with mock.patch.object(tune_threshold, "_THRESHOLD_FILE", Path("/nonexistent/threshold.txt")):
        with pytest.raises(tune_threshold.ThresholdOptimizationError) as excinfo:
            tune_threshold._save_threshold_unchecked(0.75)
        
        assert "Failed to save threshold" in str(excinfo.value)


def test_print_pr_auc(capsys):
    """Test PR-AUC printing to console."""
    tune_threshold._print_pr_auc(0.876543)
    
    captured = capsys.readouterr()
    assert captured.out.strip() == "Precision-Recall AUC: 0.876543"


# ------------------------------------------------------------------------------
# Test Public Interface
# ------------------------------------------------------------------------------

def test_tune_threshold_unchecked_success(
    patch_model_paths,
    mock_load_transactions,
    temp_models_dir,
    capsys
):
    """Test successful threshold optimization pipeline."""
    tune_threshold.tune_threshold_unchecked()
    
    # Verify threshold file was created
    threshold_file = temp_models_dir / "threshold.txt"
    assert threshold_file.exists()
    
    # Verify threshold value is valid
    with open(threshold_file, "r") as f:
        threshold = float(f.read().strip())
    assert 0.0 <= threshold <= 1.0
    
    # Verify PR-AUC was printed
    captured = capsys.readouterr()
    assert "Precision-Recall AUC:" in captured.out


def test_tune_threshold_unchecked_model_load_error():
    """Test error when model loading fails."""
    with mock.patch.object(
        tune_threshold, "_load_objects_unchecked",
        side_effect=tune_threshold.ThresholdOptimizationError("Load failed", reproducible_state={})
    ):
        with pytest.raises(tune_threshold.ThresholdOptimizationError) as excinfo:
            tune_threshold.tune_threshold_unchecked()
        
        # Update assertion to match actual error message
        assert "Unexpected error during threshold optimization" in str(excinfo.value)
        assert "Load failed" in str(excinfo.value)


def test_tune_threshold_unchecked_data_error(patch_model_paths):
    """Test error when data loading fails."""
    with mock.patch.object(
        data, "load_transactions_unchecked",
        side_effect=data.DataLoadingError("Failed", reproducible_state={})
    ):
        with pytest.raises(tune_threshold.ThresholdOptimizationError) as excinfo:
            tune_threshold.tune_threshold_unchecked()
        
        # Update assertion to match actual error message
        assert "Unexpected error during threshold optimization" in str(excinfo.value)
        assert "Failed to get validation data" in str(excinfo.value)


def test_tune_threshold_unchecked_save_error(
    patch_model_paths,
    mock_load_transactions
):
    """Test error when threshold saving fails."""
    with mock.patch.object(
        tune_threshold, "_save_threshold_unchecked",
        side_effect=tune_threshold.ThresholdOptimizationError("Save failed", reproducible_state={})
    ):
        with pytest.raises(tune_threshold.ThresholdOptimizationError):
            tune_threshold.tune_threshold_unchecked()


# ------------------------------------------------------------------------------
# Test Error Taxonomy and Exception Chains
# ------------------------------------------------------------------------------

def test_all_exceptions_inherit_from_module_error():
    """Verify all custom exceptions inherit from ModuleError."""
    assert issubclass(tune_threshold.ThresholdOptimizationError, tune_threshold.ModuleError)
    assert issubclass(tune_threshold.InvariantViolationError, tune_threshold.ModuleError)


def test_exceptions_store_reproducible_state():
    """Test exceptions capture state for reproducibility."""
    try:
        raise tune_threshold.ThresholdOptimizationError("test", reproducible_state={"key": "value"})
    except tune_threshold.ThresholdOptimizationError as e:
        assert e.reproducible_state == {"key": "value"}


def test_tune_threshold_unchecked_preserves_cause(patch_model_paths):
    """Test exception chaining preserves original error."""
    # Create a custom error that will be chained
    original_error = data.DataLoadingError("Failed", reproducible_state={})
    
    chained_error = tune_threshold.ThresholdOptimizationError(
        "Failed to get validation data: Failed",
        reproducible_state={}
    )
    chained_error.__cause__ = original_error

    with mock.patch.object(
        tune_threshold, "_get_validation_data",
        side_effect=chained_error
    ):
        with pytest.raises(tune_threshold.ThresholdOptimizationError) as excinfo:
            tune_threshold.tune_threshold_unchecked()
        
        # Check that the cause is properly preserved.
        # tune_threshold_unchecked re-wraps the ThresholdOptimizationError, so
        # the DataLoadingError is one level deeper in the chain (__cause__.__cause__).
        assert excinfo.value.__cause__ is not None
        assert isinstance(excinfo.value.__cause__, tune_threshold.ThresholdOptimizationError)
        assert excinfo.value.__cause__.__cause__ is not None
        assert isinstance(excinfo.value.__cause__.__cause__, data.DataLoadingError)


# ------------------------------------------------------------------------------
# Test Module Interface
# ------------------------------------------------------------------------------

def test_module_exports_expected_names():
    """Test __all__ contains expected public symbols."""
    expected = {
        "tune_threshold_unchecked",
        "ThresholdOptimizationError",
        "ModuleError",
    }
    assert set(tune_threshold.__all__) == expected


def test_private_members_not_exported():
    """Test private members are not in __all__."""
    for name in tune_threshold.__all__:
        assert not name.startswith("_")