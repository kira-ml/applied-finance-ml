# -*- coding: utf-8 -*-
"""
Module: train.py
Version: 1.0.0
Contract: Train a fixed-hyperparameter GradientBoostingClassifier and serialize to disk.
Dependencies: python==3.9+, numpy==1.24.3, pandas==2.0.3, scikit-learn==1.3.0, joblib==1.3.0
Invariants:
    - Input data must contain specific feature and target columns.
    - Model hyperparameters are immutable constants defined at module level.
    - Serialization includes a version header for future compatibility checks.
Failures:
    - TrainingDataError: Raised if input data is missing required columns or has invalid shapes.
    - SerializationError: Raised if model writing fails or version mismatch occurs on load verification.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
import joblib

# --- Configuration Constants ---
_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"
_ENCODING: Literal["utf-8"] = "utf-8"
_MODEL_VERSION: int = 1
_MODEL_FILENAME: str = "gbm_model.pkl"
_ARTIFACTS_DIR: str = "artifacts"
_TRAIN_PATH: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\data\processed\processed_train.csv"

# Fixed Baseline Hyperparameters (Immutable)
_HYPERS: Dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 42, # Deterministic Execution
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "subsample": 0.8
}

# Schema Definitions
_FEATURE_COLS: List[str] = [
    "income",
    "loan_amount",
    "revolving_balance",
    "num_delinquencies",
    "open_credit_lines",
    "months_employed",
    "prior_default",
    "co_applicant",
    "loan_purpose",
    "feature_dti",
    "feature_utilization",
    "feature_delinq_flag",
    "feature_revolving_ratio"
]
_TARGET_COL: str = "default"

# --- Custom Exceptions (Closed Failure Domain) ---

class TrainingDataError(ValueError):
    """Raised when training data violates structural preconditions."""
    pass

class SerializationError(IOError):
    """Raised when model serialization or deserialization fails."""
    pass


# --- Logging Setup (Structured, Local) ---

def _configure_logger(name: str) -> logging.Logger:
    """Configure a structured logger with no external dependencies."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

logger = _configure_logger("train_engine")


# --- Pure Utility Functions ---

def _validate_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate schema and extract features/target.
    Returns pure numpy arrays to decouple from pandas mutability.
    """
    if not isinstance(df, pd.DataFrame):
        raise TrainingDataError("Input must be a pandas DataFrame")

    missing_features = set(_FEATURE_COLS) - set(df.columns)
    if missing_features:
        raise TrainingDataError(f"Missing feature columns: {missing_features}")
    
    if _TARGET_COL not in df.columns:
        raise TrainingDataError(f"Missing target column: {_TARGET_COL}")

    # Explicit type coercion and validation
    try:
        X = df[_FEATURE_COLS].astype(np.float64).values
        y = df[_TARGET_COL].astype(np.int32).values
    except (ValueError, TypeError) as e:
        raise TrainingDataError(f"Data type conversion failed: {e}")

    if X.shape[0] != y.shape[0]:
        raise TrainingDataError("Feature and target row counts mismatch")
    
    if X.shape[0] == 0:
        raise TrainingDataError("Input dataset is empty")

    # Check for NaNs explicitly (No Silent Numeric Degradation)
    if np.any(np.isnan(X)):
        raise TrainingDataError("Feature matrix contains NaN values")
    if np.any(np.isnan(y)):
        raise TrainingDataError("Target vector contains NaN values")

    return X, y

def _create_model() -> GradientBoostingClassifier:
    """Instantiate model with fixed hyperparameters."""
    return GradientBoostingClassifier(**_HYPERS)


# --- Effectful Core Logic ---

def train_model(data_path: str, output_dir: str) -> GradientBoostingClassifier:
    """
    Load data, train model, and serialize artifacts.
    
    Args:
        data_path: Path to the processed training CSV.
        output_dir: Directory to save the model artifact.
        
    Returns:
        The fitted sklearn estimator.
        
    Raises:
        TrainingDataError: If data validation fails.
        SerializationError: If saving fails.
    """
    path_obj = Path(data_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    logger.info(f"Loading training data from: {path_obj.name}")
    
    # Explicit Resource Lifecycle for reading
    try:
        df = pd.read_csv(path_obj, encoding=_ENCODING)
    except Exception as e:
        raise TrainingDataError(f"Failed to read CSV: {e}")

    logger.info("Validating data schema and types")
    X, y = _validate_training_data(df)
    
    logger.info(f"Data validation passed. Shape: X={X.shape}, y={y.shape}")
    
    # Instantiate and Fit
    logger.info("Instantiating GradientBoostingClassifier with fixed baselines")
    model = _create_model()
    
    logger.info("Fitting model...")
    model.fit(X, y)
    
    # Post-condition: Verify fitting
    try:
        _ = model.predict(X[:1]) # Test on single sample
    except NotFittedError:
        raise SerializationError("Model failed to fit correctly (NotFittedError)")

    logger.info("Model fitting completed successfully")
    
    # Serialization
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_file = out_path / _MODEL_FILENAME
    
    logger.info(f"Serializing model to: {model_file}")
    
    # Wrap payload with versioning (Versioned Serialization)
    payload = {
        "version": _MODEL_VERSION,
        "model": model,
        "features": _FEATURE_COLS,
        "hyperparameters": _HYPERS
    }
    
    try:
        # Explicit resource handling via joblib context (though joblib.dump handles file closing internally,
        # we wrap in try/except for specific error mapping as per Closed Failure Domain)
        joblib.dump(payload, model_file)
    except Exception as e:
        raise SerializationError(f"Failed to write model artifact: {e}")

    logger.info(f"Model successfully saved to {model_file}")
    return model

def verify_serialization(model_path: str) -> None:
    """
    Mechanically verify the serialized artifact can be loaded and matches version.
    """
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise SerializationError(f"Artifact not found: {model_path}")
        
    try:
        payload = joblib.load(path_obj)
    except Exception as e:
        raise SerializationError(f"Failed to load artifact: {e}")
        
    if not isinstance(payload, dict):
        raise SerializationError("Invalid artifact structure: expected dict payload")
        
    if "version" not in payload:
        raise SerializationError("Invalid artifact: missing version field")
        
    if payload["version"] != _MODEL_VERSION:
        raise SerializationError(
            f"Version mismatch: Expected {_MODEL_VERSION}, got {payload['version']}"
        )
        
    if "model" not in payload or not isinstance(payload["model"], GradientBoostingClassifier):
        raise SerializationError("Invalid artifact: missing or invalid model object")
        
    logger.info("Artifact verification successful")


# --- CLI Entry Point ---

if __name__ == "__main__":
    try:
        fitted_model = train_model(_TRAIN_PATH, _ARTIFACTS_DIR)
        
        # Immediate verification step (Self-check)
        artifact_path = str(Path(_ARTIFACTS_DIR) / _MODEL_FILENAME)
        verify_serialization(artifact_path)
        
        logger.info("Training pipeline completed successfully.")
        sys.exit(0)
        
    except TrainingDataError as e:
        logger.critical(f"Training Data Error: {e}")
        sys.exit(1)
    except SerializationError as e:
        logger.critical(f"Serialization Error: {e}")
        sys.exit(2)
    except Exception as e:
        logger.critical(f"Unexpected Pipeline Failure: {e}")
        sys.exit(99)