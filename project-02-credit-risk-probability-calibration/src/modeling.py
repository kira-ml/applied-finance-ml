"""
src/modeling.py

Core Responsibilities:
- Instantiate and train the baseline Gradient Boosting Classifier on (X_train, y_train).
- Extract raw predicted probabilities for the calibration set.
- Fit the calibration method using (X_cal, y_cal).
- Wrap the base estimator and calibrator into a single predict-proba interface.
- Save the base model and calibration parameters to the models/ directory.

Constraints:
- No evaluation metrics.
- No raw data loading.
- No preprocessing pipeline modification.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Final, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

# Type Aliases
NumericArray = np.ndarray
DataFrame = pd.DataFrame
Target = Union[pd.Series, np.ndarray]

_RANDOM_STATE: Final[int] = 42


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class ModelingError(Exception):
    """Base exception for modeling failures."""
    pass


class TrainingError(ModelingError):
    """Raised when model training fails."""
    pass


class CalibrationError(ModelingError):
    """Raised when calibration fails."""
    pass


class SerializationError(ModelingError):
    """Raised when model serialization fails."""
    pass


# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def train_base_model(
    X_train: Union[NumericArray, DataFrame],
    y_train: Target,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    subsample: float = 0.8,
    random_state: int = _RANDOM_STATE
) -> GradientBoostingClassifier:
    """
    Trains a Gradient Boosting Classifier on the training set.
    
    Args:
        X_train: Training features (preprocessed).
        y_train: Training targets.
        n_estimators: Number of boosting stages.
        max_depth: Maximum depth of individual trees.
        learning_rate: Learning rate shrinks contribution of each tree.
        min_samples_split: Minimum samples required to split node.
        min_samples_leaf: Minimum samples required at leaf node.
        subsample: Fraction of samples used for fitting trees.
        random_state: Random seed for reproducibility.
        
    Returns:
        Fitted GradientBoostingClassifier.
        
    Raises:
        TrainingError: If training fails.
    """
    if len(X_train) == 0:
        raise TrainingError("Training data is empty")
    
    if len(X_train) != len(y_train):
        raise TrainingError(
            f"X_train and y_train length mismatch: {len(X_train)} vs {len(y_train)}"
        )
    
    try:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
            verbose=0
        )
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise TrainingError(f"Base model training failed: {e}") from e


def calibrate_model(
    base_model: GradientBoostingClassifier,
    X_cal: Union[NumericArray, DataFrame],
    y_cal: Target,
    method: str = "sigmoid"
) -> CalibratedClassifierCV:
    """
    Calibrates the base model using the calibration set.
    
    Args:
        base_model: Trained base classifier.
        X_cal: Calibration features (preprocessed).
        y_cal: Calibration targets.
        method: Calibration method - "sigmoid" or "isotonic".
        
    Returns:
        Calibrated classifier with predict_proba interface.
        
    Raises:
        CalibrationError: If calibration fails.
    """
    if len(X_cal) == 0:
        raise CalibrationError("Calibration data is empty")
    
    if len(X_cal) != len(y_cal):
        raise CalibrationError(
            f"X_cal and y_cal length mismatch: {len(X_cal)} vs {len(y_cal)}"
        )
    
    if method not in ("sigmoid", "isotonic"):
        raise CalibrationError(
            f"Invalid calibration method: {method}. Must be 'sigmoid' or 'isotonic'"
        )
    
    try:
        # Use prefit=True since base model is already trained
        calibrated_model = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv="prefit"  # Use prefit base model
        )
        calibrated_model.fit(X_cal, y_cal)
        return calibrated_model
    except Exception as e:
        raise CalibrationError(f"Model calibration failed: {e}") from e


def get_raw_predictions(
    model: GradientBoostingClassifier,
    X: Union[NumericArray, DataFrame]
) -> NumericArray:
    """
    Extracts raw predicted probabilities from the base model.
    
    Args:
        model: Trained classifier with predict_proba method.
        X: Features to predict on.
        
    Returns:
        NumPy array of shape (n_samples, n_classes) with predicted probabilities.
    """
    if len(X) == 0:
        raise ValueError("Input data is empty")
    
    try:
        probabilities = model.predict_proba(X)
        return probabilities
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}") from e


def save_model(model: object, path: Path) -> None:
    """
    Serializes a model to disk using pickle.
    
    Args:
        model: Model object to serialize.
        path: Absolute path where model will be saved.
        
    Raises:
        SerializationError: If serialization fails.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        raise SerializationError(f"Failed to save model to {path}: {e}") from e


def load_model(path: Path) -> object:
    """
    Loads a serialized model from disk.
    
    Args:
        path: Absolute path to the saved model file.
        
    Returns:
        Deserialized model object.
        
    Raises:
        SerializationError: If loading fails.
    """
    if not path.exists():
        raise SerializationError(f"Model file not found: {path}")
    
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        raise SerializationError(f"Failed to load model from {path}: {e}") from e


def train_and_calibrate(
    X_train: Union[NumericArray, DataFrame],
    y_train: Target,
    X_cal: Union[NumericArray, DataFrame],
    y_cal: Target,
    base_model_path: Path,
    calibrated_model_path: Path,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    subsample: float = 0.8,
    calibration_method: str = "sigmoid",
    random_state: int = _RANDOM_STATE
) -> Tuple[GradientBoostingClassifier, CalibratedClassifierCV]:
    """
    End-to-end pipeline: train base model, calibrate, and save both.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_cal: Calibration features.
        y_cal: Calibration targets.
        base_model_path: Path to save base model.
        calibrated_model_path: Path to save calibrated model.
        n_estimators: Number of boosting stages.
        max_depth: Maximum tree depth.
        learning_rate: Learning rate.
        min_samples_split: Minimum samples to split.
        min_samples_leaf: Minimum samples at leaf.
        subsample: Subsample ratio.
        calibration_method: "sigmoid" or "isotonic".
        random_state: Random seed.
        
    Returns:
        Tuple of (base_model, calibrated_model).
    """
    # Train base model
    base_model = train_base_model(
        X_train=X_train,
        y_train=y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state
    )
    
    # Calibrate model using calibration set
    calibrated_model = calibrate_model(
        base_model=base_model,
        X_cal=X_cal,
        y_cal=y_cal,
        method=calibration_method
    )
    
    # Save both models
    save_model(base_model, base_model_path)
    save_model(calibrated_model, calibrated_model_path)
    
    return base_model, calibrated_model
