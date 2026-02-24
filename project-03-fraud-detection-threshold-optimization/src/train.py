"""
src/train.py

Training module for fraud detection logistic regression model.
Loads data, splits stratified 80/20, scales features, fits model, and saves artifacts.
Strictly deterministic, type-enforced, and side-effect isolated.
Uses pandas, sklearn, and joblib as required by module description.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Final, Tuple, Optional, Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# ==============================================================================
# 1. TOTAL STATIC TYPE PRECISION & RUNTIME ENFORCEMENT
# 2. SINGLE EXPLICIT ERROR TAXONOMY
# 4. SINGLE EXPLICIT ERROR TAXONOMY (Leaf Exceptions)
# 20. TOTAL LINEARITY OF ERROR HANDLING
# ==============================================================================

class ModuleError(Exception):
    """Base exception for all module-specific errors."""
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.context: Final[Optional[Dict[str, Any]]] = context if context is not None else {}

class InvariantViolationError(ModuleError):
    """Raised when a defined invariant is violated."""
    pass

class InvalidStateTransitionError(ModuleError):
    """Raised when an illegal state transition occurs."""
    pass

class DataLoadingError(ModuleError):
    """Raised when data loading from src.data fails."""
    pass

class ModelTrainingError(ModuleError):
    """Raised when model fitting fails."""
    pass

class FileSystemError(ModuleError):
    """Raised when file system operations fail."""
    pass

# ==============================================================================
# 8. IMMUTABLE CONFIGURATION AT LOAD TIME
# 14. NO MUTABLE GLOBAL STATE
# ==============================================================================

_CONFIG_TRAIN_RATIO: Final[float] = 0.8
_CONFIG_SEED: Final[int] = 42
_CONFIG_RANDOM_STATE: Final[int] = 42
_CONFIG_SOLVER: Final[str] = "lbfgs"
_CONFIG_MAX_ITER: Final[int] = 1000

# Updated path as per user requirement
_CONFIG_MODEL_DIR: Final[str] = r"D:\applied-finance-ml\project-03-fraud-detection-threshold-optimization\models"
_CONFIG_MODEL_FILE: Final[str] = "logistic_regression.joblib"
_CONFIG_SCALER_FILE: Final[str] = "standard_scaler.joblib"

# Lifecycle States
class TrainingLifecycleState(Enum):
    UNINITIALIZED = 0
    DATA_LOADED = 1
    MODEL_TRAINED = 2
    ARTIFACTS_SAVED = 3

# ==============================================================================
# 3. COMPREHENSIVE INVARIANT ENFORCEMENT
# 6. NO DYNAMIC TYPE INTROSPECTION FOR CONTROL FLOW
# 9. EXPLICIT DATA LIFECYCLE AND MUTATION CONTROL
# ==============================================================================

def _validate_type(value: object, expected_type: type, var_name: str) -> None:
    if type(value) is not expected_type:
        raise InvariantViolationError(
            f"Type mismatch for '{var_name}': expected {expected_type.__name__}, got {type(value).__name__}",
            context={"value": repr(value), "expected": expected_type.__name__}
        )

def _validate_in_range(value: float, min_val: float, max_val: float, var_name: str) -> None:
    if not (min_val <= value <= max_val):
        raise InvariantViolationError(
            f"Value out of range for '{var_name}': {value} not in [{min_val}, {max_val}]",
            context={"value": value, "min": min_val, "max": max_val}
        )

def _validate_dataframe(df: pd.DataFrame, var_name: str) -> None:
    if df.empty:
        raise InvariantViolationError(
            f"Empty DataFrame for '{var_name}'",
            context={"rows": len(df)}
        )
    if "is_fraud" not in df.columns:
        raise InvariantViolationError(
            f"Missing 'is_fraud' column in '{var_name}'",
            context={"columns": list(df.columns)}
        )
    unique_vals = df["is_fraud"].unique()
    if not all(v in [0, 1] for v in unique_vals):
        raise InvariantViolationError(
            f"Invalid values in 'is_fraud' column",
            context={"unique_values": [int(x) for x in unique_vals]}
        )

# ==============================================================================
# 2. PURE DETERMINISTIC CORE WITH EXPLICIT IMPURITY MARKERS
# 13. STRICT RESOURCE MANAGEMENT
# ==============================================================================

def _load_data_unchecked() -> pd.DataFrame:
    """
    Loads data from src.data.DataModule.
    SIDE EFFECT: Imports and executes logic from src.data (IO).
    """
    try:
        from src.data import DataModule
        dm = DataModule()
        data_list = dm.load_and_validate()
        df = pd.DataFrame(data_list)
        _validate_dataframe(df, "loaded_data")
        return df
    except ImportError as e:
        raise DataLoadingError("Failed to import src.data", context={"error": str(e)}) from e
    except ModuleError as e:
        raise DataLoadingError("Failed to load data from src.data", context={"original_error": str(e), "context": e.context}) from e
    except Exception as e:
        raise DataLoadingError("Unexpected error loading data", context={"error": str(e)}) from e

def _split_data_unchecked(
    df: pd.DataFrame, 
    train_ratio: float, 
    seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into train/val with stratification on 'is_fraud'.
    SIDE EFFECT: Uses sklearn (randomness isolated via seed).
    """
    try:
        train_df, val_df = train_test_split(
            df,
            train_size=train_ratio,
            stratify=df["is_fraud"],
            random_state=seed,
            shuffle=True
        )
        return train_df, val_df
    except ValueError as e:
        raise ModelTrainingError("Stratified split failed", context={"error": str(e)}) from e
    except Exception as e:
        raise ModelTrainingError("Unexpected error during split", context={"error": str(e)}) from e

def _fit_scaler_unchecked(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fits StandardScaler on training features.
    SIDE EFFECT: Sklearn fitting process.
    """
    try:
        scaler = StandardScaler()
        scaler.fit(X_train)
        return scaler
    except Exception as e:
        raise ModelTrainingError("Failed to fit StandardScaler", context={"error": str(e)}) from e

def _fit_model_unchecked(
    X_train_scaled: pd.DataFrame, 
    y_train: pd.Series
) -> LogisticRegression:
    """
    Fits Logistic Regression with balanced class weights.
    SIDE EFFECT: Sklearn fitting process.
    """
    try:
        model = LogisticRegression(
            solver=_CONFIG_SOLVER,
            class_weight="balanced",
            random_state=_CONFIG_RANDOM_STATE,
            max_iter=_CONFIG_MAX_ITER
        )
        model.fit(X_train_scaled, y_train)
        return model
    except Exception as e:
        raise ModelTrainingError("Failed to fit LogisticRegression", context={"error": str(e)}) from e

def _save_artifact_unchecked(obj: Any, file_path: str) -> None:
    """
    Saves object to disk using joblib.
    SIDE EFFECT: Filesystem write.
    """
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as e:
            raise FileSystemError(f"Failed to create directory {dir_path}", context={"path": dir_path}) from e
            
    try:
        joblib.dump(obj, file_path)
    except IOError as e:
        raise FileSystemError(f"Failed to write file {file_path}", context={"path": file_path}) from e
    except Exception as e:
        raise FileSystemError(f"Unexpected error saving artifact", context={"error": str(e)}) from e

# ==============================================================================
# MAIN LOGIC IMPLEMENTATION
# ==============================================================================

class TrainingModule:
    """
    Handles the full training lifecycle: Load -> Split -> Scale -> Fit -> Save.
    """
    
    def __init__(self) -> None:
        self._state: TrainingLifecycleState = TrainingLifecycleState.UNINITIALIZED
        self._scaler: Optional[StandardScaler] = None
        self._model: Optional[LogisticRegression] = None
        
    def run_training(self) -> None:
        """
        Executes the full training pipeline.
        """
        if self._state != TrainingLifecycleState.UNINITIALIZED:
            raise InvalidStateTransitionError(
                "Cannot start training: invalid state",
                context={"current_state": self._state.name}
            )
            
        # 1. Load Data
        try:
            df = _load_data_unchecked()
            self._state = TrainingLifecycleState.DATA_LOADED
        except ModuleError:
            raise
        except Exception as e:
            raise DataLoadingError("Unexpected failure during data load", context={"error": str(e)}) from e
            
        # 2. Split Data
        try:
            train_df, val_df = _split_data_unchecked(df, _CONFIG_TRAIN_RATIO, _CONFIG_SEED)
            _validate_dataframe(train_df, "train_set")
            _validate_dataframe(val_df, "val_set")
        except ModuleError:
            raise
        except Exception as e:
            raise ModelTrainingError("Failed to split data", context={"error": str(e)}) from e
            
        # 3. Prepare Features/Labels
        feature_cols = ["amount", "time_delta"]
        X_train = train_df[feature_cols]
        y_train = train_df["is_fraud"]
        
        # 4. Fit Scaler (Train only)
        try:
            self._scaler = _fit_scaler_unchecked(X_train)
        except ModuleError:
            raise
        except Exception as e:
            raise ModelTrainingError("Failed to fit scaler", context={"error": str(e)}) from e
            
        # 5. Scale Train Data
        try:
            X_train_scaled = pd.DataFrame(
                self._scaler.transform(X_train),
                columns=feature_cols,
                index=X_train.index
            )
        except Exception as e:
            raise ModelTrainingError("Failed to scale training data", context={"error": str(e)}) from e
            
        # 6. Fit Model
        try:
            self._model = _fit_model_unchecked(X_train_scaled, y_train)
            self._state = TrainingLifecycleState.MODEL_TRAINED
        except ModuleError:
            raise
        except Exception as e:
            raise ModelTrainingError("Failed to fit model", context={"error": str(e)}) from e
            
        # 7. Save Artifacts
        try:
            model_path = os.path.join(_CONFIG_MODEL_DIR, _CONFIG_MODEL_FILE)
            scaler_path = os.path.join(_CONFIG_MODEL_DIR, _CONFIG_SCALER_FILE)
            
            _save_artifact_unchecked(self._model, model_path)
            _save_artifact_unchecked(self._scaler, scaler_path)
            
            self._state = TrainingLifecycleState.ARTIFACTS_SAVED
        except ModuleError:
            raise
        except Exception as e:
            raise FileSystemError("Failed to save model artifacts", context={"error": str(e)}) from e

    def get_state(self) -> TrainingLifecycleState:
        return self._state

# ==============================================================================
# 19. STRICT PUBLIC INTERFACE CONTROL
# ==============================================================================

__all__ = [
    "TrainingModule", 
    "ModuleError", 
    "DataLoadingError", 
    "ModelTrainingError", 
    "FileSystemError", 
    "InvariantViolationError", 
    "InvalidStateTransitionError"
]