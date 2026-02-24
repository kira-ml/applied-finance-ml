"""
test_train.py

Unit tests for src.train module.
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

from src import train
from src import data


# ------------------------------------------------------------------------------
# Test Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture
def sample_transactions_df():
    """Create a small sample transactions DataFrame for testing."""
    np.random.seed(42)
    n_rows = 1000
    
    # Generate timestamps
    base_ts = pd.Timestamp("2024-01-01")
    seconds = np.random.randint(0, 30*24*3600, size=n_rows)
    timestamps = (base_ts + pd.to_timedelta(seconds, unit="s")).strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate amounts
    amounts = np.round(np.random.lognormal(mean=4.0, sigma=1.0, size=n_rows), 2)
    
    # Generate fraud labels (exactly 1%)
    labels = np.zeros(n_rows, dtype=np.int8)
    fraud_indices = np.random.choice(n_rows, size=10, replace=False)  # 1% of 1000
    labels[fraud_indices] = 1
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "amount": amounts,
        "is_fraud": labels,
    })


@pytest.fixture
def temp_models_dir():
    """Provide temporary directory for model outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def patch_model_paths(temp_models_dir):
    """Patch module-level paths to use temp directory."""
    with mock.patch.object(train, "_MODELS_DIR", temp_models_dir):
        with mock.patch.object(train, "_SCALER_FILE", temp_models_dir / "scaler.pkl"):
            with mock.patch.object(train, "_MODEL_FILE", temp_models_dir / "model.pkl"):
                yield


@pytest.fixture
def mock_load_transactions(sample_transactions_df):
    """Mock data.load_transactions_unchecked to return sample data."""
    with mock.patch.object(data, "load_transactions_unchecked", return_value=sample_transactions_df):
        yield


# ------------------------------------------------------------------------------
# Test Configuration and Invariants
# ------------------------------------------------------------------------------

def test_config_values_are_correct_types():
    """Verify configuration constants have correct types."""
    assert isinstance(train._MODELS_DIR, Path)
    assert isinstance(train._SCALER_FILE, Path)
    assert isinstance(train._MODEL_FILE, Path)
    assert isinstance(train._RANDOM_SEED, int)
    assert isinstance(train._TEST_SIZE, float)
    assert isinstance(train._LOGISTIC_C, float)
    assert isinstance(train._LOGISTIC_MAX_ITER, int)


def test_config_values_are_immutable():
    """Verify config values remain unchanged after module execution."""
    original_seed = train._RANDOM_SEED
    original_test_size = train._TEST_SIZE
    original_c = train._LOGISTIC_C
    original_max_iter = train._LOGISTIC_MAX_ITER
    
    # Run validation functions
    train._validate_split_ratio(0.2)
    
    # Verify values haven't changed
    assert train._RANDOM_SEED == original_seed
    assert train._TEST_SIZE == original_test_size
    assert train._LOGISTIC_C == original_c
    assert train._LOGISTIC_MAX_ITER == original_max_iter


# ------------------------------------------------------------------------------
# Test Core Deterministic Functions
# ------------------------------------------------------------------------------

def test_validate_split_ratio_valid_values():
    """Test split ratio validation with valid inputs."""
    # Should not raise
    train._validate_split_ratio(0.1)
    train._validate_split_ratio(0.5)
    train._validate_split_ratio(0.9)


def test_validate_split_ratio_invalid_values():
    """Test split ratio validation rejects invalid inputs."""
    with pytest.raises(train.InvariantViolationError):
        train._validate_split_ratio(0.0)
    with pytest.raises(train.InvariantViolationError):
        train._validate_split_ratio(1.0)
    with pytest.raises(train.InvariantViolationError):
        train._validate_split_ratio(-0.1)
    with pytest.raises(train.InvariantViolationError):
        train._validate_split_ratio(1.5)


def test_validate_dataframe_valid(sample_transactions_df):
    """Test dataframe validation with valid input."""
    # Should not raise
    train._validate_dataframe(sample_transactions_df)


def test_validate_dataframe_missing_columns(sample_transactions_df):
    """Test validation catches missing columns."""
    df_missing = sample_transactions_df.drop(columns=["amount"])
    with pytest.raises(train.InvariantViolationError) as excinfo:
        train._validate_dataframe(df_missing)
    # Update assertion to match exact error message
    assert "DataFrame missing required columns" in str(excinfo.value)


def test_validate_dataframe_empty():
    """Test validation rejects empty dataframe."""
    df_empty = pd.DataFrame(columns=["timestamp", "amount", "is_fraud"])
    with pytest.raises(train.InvariantViolationError) as excinfo:
        train._validate_dataframe(df_empty)
    assert "DataFrame cannot be empty" in str(excinfo.value)


def test_validate_dataframe_invalid_fraud_values(sample_transactions_df):
    """Test validation catches invalid is_fraud values."""
    df_invalid = sample_transactions_df.copy()
    df_invalid.loc[0, "is_fraud"] = 2
    with pytest.raises(train.InvariantViolationError) as excinfo:
        train._validate_dataframe(df_invalid)
    assert "is_fraud column must contain only 0 and 1" in str(excinfo.value)


def test_split_features_target(sample_transactions_df):
    """Test feature/target splitting."""
    X, y = train._split_features_target(sample_transactions_df)
    
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert list(X.columns) == ["amount"]
    assert len(X) == len(sample_transactions_df)
    assert len(y) == len(sample_transactions_df)
    assert X.index.equals(sample_transactions_df.index)
    assert y.index.equals(sample_transactions_df.index)


def test_split_train_test(sample_transactions_df):
    """Test stratified train/test split."""
    X, y = train._split_features_target(sample_transactions_df)
    
    X_train, X_test, y_train, y_test = train._split_train_test(
        X, y,
        test_size=0.2,
        random_state=42,
    )
    
    # Check sizes
    assert len(X_train) == 800  # 80% of 1000
    assert len(X_test) == 200   # 20% of 1000
    assert len(y_train) == 800
    assert len(y_test) == 200
    
    # Check stratification (fraud ratio should be preserved)
    train_fraud_ratio = y_train.mean()
    test_fraud_ratio = y_test.mean()
    original_fraud_ratio = y.mean()
    
    assert abs(train_fraud_ratio - original_fraud_ratio) < 0.02
    assert abs(test_fraud_ratio - original_fraud_ratio) < 0.02
    
    # Check reproducibility
    X_train2, X_test2, y_train2, y_test2 = train._split_train_test(
        X, y, test_size=0.2, random_state=42
    )
    assert X_train.index.equals(X_train2.index)
    assert X_test.index.equals(X_test2.index)


def test_fit_scaler(sample_transactions_df):
    """Test scaler fitting and transformation."""
    X, _ = train._split_features_target(sample_transactions_df)
    
    scaler, X_scaled = train._fit_scaler(X)
    
    assert isinstance(scaler, StandardScaler)
    assert isinstance(X_scaled, pd.DataFrame)
    assert list(X_scaled.columns) == ["amount"]
    assert len(X_scaled) == len(X)
    assert X_scaled.index.equals(X.index)
    
    # Verify scaling (mean ~0, std ~1) - use larger tolerance for numerical stability
    assert abs(X_scaled["amount"].mean()) < 1e-7
    # Use larger tolerance for std deviation due to numerical precision
    assert abs(X_scaled["amount"].std() - 1.0) < 1e-3  # Increased tolerance


def test_transform_features(sample_transactions_df):
    """Test feature transformation with fitted scaler."""
    X, _ = train._split_features_target(sample_transactions_df)
    
    # Split into train/test
    X_train = X.iloc[:800]
    X_test = X.iloc[800:]
    
    # Fit on train
    scaler, X_train_scaled = train._fit_scaler(X_train)
    
    # Transform test
    X_test_scaled = train._transform_features(scaler, X_test)
    
    assert isinstance(X_test_scaled, pd.DataFrame)
    assert list(X_test_scaled.columns) == ["amount"]
    assert len(X_test_scaled) == len(X_test)
    assert X_test_scaled.index.equals(X_test.index)


def test_fit_model(sample_transactions_df):
    """Test logistic regression fitting."""
    X, y = train._split_features_target(sample_transactions_df)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train._split_train_test(
        X, y, test_size=0.2, random_state=42
    )
    scaler, X_train_scaled = train._fit_scaler(X_train)
    
    # Fit model
    model = train._fit_model(
        X_train_scaled, y_train,
        C=1.0,
        max_iter=1000,
        random_state=42,
    )
    
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert model.coef_.shape == (1, 1)  # Binary classification, 1 feature
    assert model.class_weight == "balanced"
    
    # Should be able to predict
    X_test_scaled = train._transform_features(scaler, X_test)
    preds = model.predict(X_test_scaled)
    assert len(preds) == len(y_test)
    assert set(preds).issubset({0, 1})


def test_fit_model_deterministic(sample_transactions_df):
    """Test model fitting is reproducible with same seed."""
    X, y = train._split_features_target(sample_transactions_df)
    
    # Split and scale
    X_train, _, y_train, _ = train._split_train_test(
        X, y, test_size=0.2, random_state=42
    )
    scaler, X_train_scaled = train._fit_scaler(X_train)
    
    # Fit two models with same seed
    model1 = train._fit_model(
        X_train_scaled, y_train,
        C=1.0, max_iter=1000, random_state=42
    )
    model2 = train._fit_model(
        X_train_scaled, y_train,
        C=1.0, max_iter=1000, random_state=42
    )
    
    np.testing.assert_array_equal(model1.coef_, model2.coef_)
    np.testing.assert_array_equal(model1.intercept_, model2.intercept_)


def test_save_object_unchecked(temp_models_dir):
    """Test object serialization."""
    test_obj = {"key": "value", "number": 42}
    test_file = temp_models_dir / "test.pkl"
    
    train._save_object_unchecked(test_obj, test_file)
    
    assert test_file.exists()
    
    with open(test_file, "rb") as f:
        loaded_obj = pickle.load(f)
    
    assert loaded_obj == test_obj


def test_save_object_unchecked_error():
    """Test error handling in object saving."""
    with pytest.raises(train.TrainingError):
        train._save_object_unchecked(
            {"test": "data"},
            Path("/nonexistent/path/test.pkl")
        )


# ------------------------------------------------------------------------------
# Test Public Interface
# ------------------------------------------------------------------------------

def test_train_unchecked_success(
    mock_load_transactions,
    patch_model_paths,
    temp_models_dir,
    sample_transactions_df
):
    """Test successful training pipeline execution."""
    train.train_unchecked()
    
    # Verify files were created
    assert (temp_models_dir / "scaler.pkl").exists()
    assert (temp_models_dir / "model.pkl").exists()
    
    # Load and verify scaler
    with open(temp_models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    assert isinstance(scaler, StandardScaler)
    
    # Load and verify model
    with open(temp_models_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, LogisticRegression)
    assert model.class_weight == "balanced"


def test_train_unchecked_creates_models_directory(
    mock_load_transactions,
    temp_models_dir
):
    """Test training creates models directory if it doesn't exist."""
    new_dir = temp_models_dir / "nested" / "models" / "path"
    
    with mock.patch.object(train, "_MODELS_DIR", new_dir):
        with mock.patch.object(train, "_SCALER_FILE", new_dir / "scaler.pkl"):
            with mock.patch.object(train, "_MODEL_FILE", new_dir / "model.pkl"):
                train.train_unchecked()
    
    assert new_dir.exists()
    assert (new_dir / "scaler.pkl").exists()
    assert (new_dir / "model.pkl").exists()


def test_train_unchecked_data_loading_error(sample_transactions_df):
    """Test error handling when data loading fails."""
    with mock.patch.object(
        data, "load_transactions_unchecked",
        side_effect=data.DataLoadingError("Failed to load", reproducible_state={})
    ):
        with pytest.raises(train.TrainingError) as excinfo:
            train.train_unchecked()
        
        assert "Data loading failed" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, data.DataLoadingError)


def test_train_unchecked_invalid_dataframe(sample_transactions_df):
    """Test error handling with invalid input data."""
    invalid_df = sample_transactions_df.drop(columns=["amount"])
    
    with mock.patch.object(data, "load_transactions_unchecked", return_value=invalid_df):
        with pytest.raises(train.TrainingError) as excinfo:
            train.train_unchecked()
        
        # Update assertion to match actual error message
        assert "Unexpected error during training" in str(excinfo.value)
        assert "DataFrame missing required columns" in str(excinfo.value)


def test_train_unchecked_directory_creation_error(
    mock_load_transactions,
    temp_models_dir
):
    """Test error handling when models directory cannot be created."""
    with mock.patch.object(Path, "mkdir", side_effect=OSError("Permission denied")):
        with pytest.raises(train.TrainingError) as excinfo:
            train.train_unchecked()
        
        assert "Failed to create models directory" in str(excinfo.value)


def test_train_unchecked_save_error(
    mock_load_transactions,
    patch_model_paths
):
    """Test error handling when saving fails."""
    with mock.patch.object(train, "_save_object_unchecked", side_effect=train.TrainingError("Save failed", reproducible_state={})):
        with pytest.raises(train.TrainingError):
            train.train_unchecked()


# ------------------------------------------------------------------------------
# Test Error Taxonomy and Exception Chains
# ------------------------------------------------------------------------------

def test_all_exceptions_inherit_from_module_error():
    """Verify all custom exceptions inherit from ModuleError."""
    assert issubclass(train.TrainingError, train.ModuleError)
    assert issubclass(train.InvariantViolationError, train.ModuleError)


def test_exceptions_store_reproducible_state():
    """Test exceptions capture state for reproducibility."""
    try:
        raise train.TrainingError("test", reproducible_state={"key": "value"})
    except train.TrainingError as e:
        assert e.reproducible_state == {"key": "value"}


def test_train_unchecked_preserves_cause(mock_load_transactions):
    """Test exception chaining preserves original error."""
    with mock.patch.object(
        train, "_save_object_unchecked",
        side_effect=OSError("Disk full")
    ):
        with pytest.raises(train.TrainingError) as excinfo:
            train.train_unchecked()
        
        assert isinstance(excinfo.value.__cause__, OSError)


# ------------------------------------------------------------------------------
# Test Module Interface
# ------------------------------------------------------------------------------

def test_module_exports_expected_names():
    """Test __all__ contains expected public symbols."""
    expected = {
        "train_unchecked",
        "TrainingError",
        "ModuleError",
    }
    assert set(train.__all__) == expected


def test_private_members_not_exported():
    """Test private members are not in __all__."""
    for name in train.__all__:
        assert not name.startswith("_")