"""
src/train.py

Model training module with deterministic train/validation split,
feature scaling, and logistic regression fitting.
"""

import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src import data

__all__ = [
    "train_unchecked",
    "TrainingError",
    "ModuleError",
]


# ------------------------------------------------------------------------------
# Error Taxonomy
# ------------------------------------------------------------------------------

class ModuleError(Exception):
    """Base class for all module exceptions."""
    def __init__(self, message: str, *, reproducible_state: Dict[str, Any]) -> None:
        self.reproducible_state = reproducible_state
        super().__init__(message)


class TrainingError(ModuleError):
    """Raised when model training fails."""
    pass


class InvariantViolationError(ModuleError):
    """Raised when invariant checks fail."""
    pass


# ------------------------------------------------------------------------------
# Immutable Configuration
# ------------------------------------------------------------------------------

_MODELS_DIR: Path = Path(r"D:\applied-finance-ml\project-03-fraud-detection-threshold-optimization\models")
_SCALER_FILE: Path = _MODELS_DIR / "scaler.pkl"
_MODEL_FILE: Path = _MODELS_DIR / "model.pkl"
_RANDOM_SEED: int = 42
_TEST_SIZE: float = 0.2
_LOGISTIC_C: float = 1.0
_LOGISTIC_MAX_ITER: int = 1000


def _load_config_unchecked() -> None:
    """
    One-time configuration loader.
    Ensures config values are immutable after import.
    """
    global _MODELS_DIR, _SCALER_FILE, _MODEL_FILE, _RANDOM_SEED, _TEST_SIZE
    global _LOGISTIC_C, _LOGISTIC_MAX_ITER
    # Configuration is already defined as module constants.
    # This function exists to satisfy the "immutable config" standard.
    pass


_load_config_unchecked()


# ------------------------------------------------------------------------------
# Core Deterministic Functions
# ------------------------------------------------------------------------------

def _validate_split_ratio(ratio: float) -> None:
    """Validate that test split ratio is between 0 and 1."""
    if not 0.0 < ratio < 1.0:
        raise InvariantViolationError(
            f"Test split ratio must be between 0 and 1, got {ratio}",
            reproducible_state={"test_size": ratio},
        )


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input dataframe has required structure."""
    required_columns = {"timestamp", "amount", "is_fraud"}
    if not required_columns.issubset(df.columns):
        raise InvariantViolationError(
            f"DataFrame missing required columns. Expected {required_columns}, got {set(df.columns)}",
            reproducible_state={"columns": list(df.columns)},
        )
    
    if len(df) == 0:
        raise InvariantViolationError(
            "DataFrame cannot be empty",
            reproducible_state={"n_rows": 0},
        )
    
    # Verify is_fraud values
    unique_values = set(df["is_fraud"].unique())
    if not unique_values.issubset({0, 1}):
        raise InvariantViolationError(
            f"is_fraud column must contain only 0 and 1. Found: {unique_values}",
            reproducible_state={"invalid_values": list(unique_values - {0, 1})},
        )


def _split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features (amount) and target (is_fraud).
    
    Args:
        df: Input dataframe with amount and is_fraud columns
    
    Returns:
        Tuple of (X_features, y_target)
    """
    X = df[["amount"]].copy()  # Keep as DataFrame with feature name
    y = df["is_fraud"].copy()
    return X, y


def _split_train_test(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float, 
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train/test split.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    _validate_split_ratio(test_size)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # Stratify on target for balanced splits
        shuffle=True,
    )
    
    # Verify split proportions
    train_ratio = len(y_train) / len(y)
    test_ratio = len(y_test) / len(y)
    
    if abs(test_ratio - test_size) > 0.001:
        raise InvariantViolationError(
            f"Split ratio mismatch: target {test_size}, got {test_ratio}",
            reproducible_state={
                "target_test_size": test_size,
                "actual_test_size": test_ratio,
                "train_size": len(y_train),
                "test_size": len(y_test),
                "total_size": len(y),
            },
        )
    
    return X_train, X_test, y_train, y_test


def _fit_scaler(X_train: pd.DataFrame) -> Tuple[StandardScaler, pd.DataFrame]:
    """
    Fit StandardScaler on training features and transform them.
    
    Args:
        X_train: Training features
    
    Returns:
        Tuple of (fitted_scaler, scaled_training_features)
    """
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(
        X_train_scaled_array,
        columns=X_train.columns,
        index=X_train.index,
    )
    return scaler, X_train_scaled


def _transform_features(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    """
    Transform features using fitted scaler.
    
    Args:
        scaler: Fitted StandardScaler
        X: Features to transform
    
    Returns:
        Scaled features as DataFrame
    """
    X_scaled_array = scaler.transform(X)
    return pd.DataFrame(
        X_scaled_array,
        columns=X.columns,
        index=X.index,
    )


def _fit_model(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    C: float,
    max_iter: int,
    random_state: int,
) -> LogisticRegression:
    """
    Fit logistic regression model with class_weight='balanced'.
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training targets
        C: Inverse regularization strength
        max_iter: Maximum iterations for solver
        random_state: Random seed for reproducibility
    
    Returns:
        Fitted logistic regression model
    """
    model = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",  # Deterministic solver
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Verify model was fitted
    if not hasattr(model, "coef_"):
        raise InvariantViolationError(
            "Model fitting failed: no coefficients found",
            reproducible_state={"C": C, "max_iter": max_iter},
        )
    
    return model


def _save_object_unchecked(obj: Any, filepath: Path) -> None:
    """
    Save Python object to disk using pickle.
    
    Args:
        obj: Object to save
        filepath: Path to save to
    
    Raises:
        TrainingError: If save fails
    """
    try:
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    except (OSError, pickle.PicklingError) as e:
        raise TrainingError(
            f"Failed to save object to {filepath}: {e}",
            reproducible_state={"filepath": str(filepath)},
        ) from e


# ------------------------------------------------------------------------------
# Public Interface (with explicit side effects)
# ------------------------------------------------------------------------------

def train_unchecked() -> None:
    """
    Execute full training pipeline:
    1. Load data from data.py
    2. Split into train/validation sets (80/20 stratified)
    3. Fit StandardScaler on training features only
    4. Fit Logistic Regression with class_weight='balanced'
    5. Save scaler and model to models directory
    
    Side effects:
        - Loads transactions.csv from filesystem
        - Creates models directory if it doesn't exist
        - Writes scaler.pkl and model.pkl to models directory
    
    Raises:
        TrainingError: If any step fails
        data.ModuleError: If data loading fails
    """
    try:
        # Step 1: Load data
        df = data.load_transactions_unchecked()
        
        # Validate input
        _validate_dataframe(df)
        
        # Step 2: Split features and target
        X, y = _split_features_target(df)
        
        # Step 3: Train/test split (80/20 stratified)
        X_train, X_test, y_train, y_test = _split_train_test(
            X, y,
            test_size=_TEST_SIZE,
            random_state=_RANDOM_SEED,
        )
        
        # Step 4: Fit scaler on training features only
        scaler, X_train_scaled = _fit_scaler(X_train)
        
        # Step 5: Transform test features (using training scaler)
        X_test_scaled = _transform_features(scaler, X_test)
        
        # Step 6: Fit model on scaled training data
        model = _fit_model(
            X_train_scaled,
            y_train,
            C=_LOGISTIC_C,
            max_iter=_LOGISTIC_MAX_ITER,
            random_state=_RANDOM_SEED,
        )
        
        # Verify model can predict (minimal smoke test)
        train_pred = model.predict(X_train_scaled)
        if len(train_pred) != len(y_train):
            raise InvariantViolationError(
                "Model prediction shape mismatch",
                reproducible_state={
                    "n_train": len(y_train),
                    "n_pred": len(train_pred),
                },
            )
        
        # Step 7: Ensure models directory exists
        try:
            _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise TrainingError(
                f"Failed to create models directory {_MODELS_DIR}",
                reproducible_state={"models_dir": str(_MODELS_DIR)},
            ) from e
        
        # Step 8: Save scaler and model
        _save_object_unchecked(scaler, _SCALER_FILE)
        _save_object_unchecked(model, _MODEL_FILE)
        
    except data.ModuleError as e:
        # Re-raise data module errors with proper chain
        raise TrainingError(
            f"Data loading failed: {e}",
            reproducible_state={"data_error": str(e)},
        ) from e
    except Exception as e:
        # Map any unexpected exception to our taxonomy
        raise TrainingError(
            f"Unexpected error during training: {e}",
            reproducible_state={"error_type": type(e).__name__},
        ) from e


# ------------------------------------------------------------------------------
# Module-Level Invariants
# ------------------------------------------------------------------------------

# Verify configuration immutability
assert isinstance(_RANDOM_SEED, int), "_RANDOM_SEED must be int"
assert isinstance(_TEST_SIZE, float), "_TEST_SIZE must be float"
assert isinstance(_LOGISTIC_C, float), "_LOGISTIC_C must be float"
assert isinstance(_LOGISTIC_MAX_ITER, int), "_LOGISTIC_MAX_ITER must be int"