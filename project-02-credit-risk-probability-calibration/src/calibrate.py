# -*- coding: utf-8 -*-
"""
Module: calibrate.py
Version: 1.0.0
Contract: Load base GBM, fit Platt Scaling and Isotonic Regression calibrators on validation data, serialize artifacts.
Dependencies: python==3.9+, numpy==1.24.3, pandas==2.0.3, scikit-learn==1.3.0, joblib==1.3.0
Invariants:
    - Base model must be loaded from a versioned artifact matching current module version.
    - Calibration is performed strictly on validation data (no test set leakage).
    - Output artifacts include version headers and method identifiers.
Failures:
    - ArtifactLoadError: Raised if base model is missing, corrupted, or version mismatched.
    - CalibrationDataError: Raised if validation data schema is invalid or contains NaNs.
    - CalibrationFitError: Raised if the underlying sklearn calibrator fails to converge or fit.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.base import ClassifierMixin
import joblib

# --- Configuration Constants ---
_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"
_ENCODING: Literal["utf-8"] = "utf-8"
_MODEL_VERSION: int = 1

# Paths
_BASE_MODEL_PATH: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\src\artifacts\gbm_model.pkl"
_VALIDATION_PATH: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\data\processed\processed_validation.csv"
_OUTPUT_DIR: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\src\artifacts"

# Calibration Methods
_METHOD_PLATT: Literal["sigmoid"] = "sigmoid"
_METHOD_ISOTONIC: Literal["isotonic"] = "isotonic"

# Schema (Must match training features)
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

class ArtifactLoadError(IOError):
    """Raised when loading or verifying artifacts fails."""
    pass

class CalibrationDataError(ValueError):
    """Raised when validation data violates preconditions."""
    pass

class CalibrationFitError(RuntimeError):
    """Raised when the calibration fitting process fails."""
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

logger = _configure_logger("calibration_engine")


# --- Pure Utility Functions ---

def _validate_payload(payload: Any) -> Dict[str, Any]:
    """Mechanically enforce artifact structure and versioning."""
    if not isinstance(payload, dict):
        raise ArtifactLoadError("Artifact payload is not a dictionary")
    
    if "version" not in payload:
        raise ArtifactLoadError("Artifact missing 'version' field")
    
    if payload["version"] != _MODEL_VERSION:
        raise ArtifactLoadError(
            f"Version mismatch: Expected {_MODEL_VERSION}, got {payload['version']}"
        )
    
    if "model" not in payload or not isinstance(payload["model"], ClassifierMixin):
        raise ArtifactLoadError("Artifact missing valid 'model' instance")
        
    if "features" not in payload or not isinstance(payload["features"], list):
        raise ArtifactLoadError("Artifact missing 'features' list")
        
    return payload

def _validate_calibration_data(df: pd.DataFrame, expected_features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Validate validation split schema and purity."""
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise CalibrationDataError(f"Validation data missing features: {missing}")
    
    if _TARGET_COL not in df.columns:
        raise CalibrationDataError(f"Validation data missing target: {_TARGET_COL}")

    try:
        X = df[expected_features].astype(np.float64).values
        y = df[_TARGET_COL].astype(np.int32).values
    except (ValueError, TypeError) as e:
        raise CalibrationDataError(f"Type coercion failed: {e}")

    # No Silent Numeric Degradation
    if np.any(np.isnan(X)):
        raise CalibrationDataError("Validation features contain NaN")
    if np.any(np.isnan(y)):
        raise CalibrationDataError("Validation target contains NaN")
        
    if X.shape[0] == 0:
        raise CalibrationDataError("Validation dataset is empty")

    return X, y


# --- Effectful Core Logic ---

def load_base_model(path_str: str) -> ClassifierMixin:
    """Load and verify the base GBM model."""
    path = Path(path_str)
    if not path.exists():
        raise ArtifactLoadError(f"Base model not found: {path}")
    
    logger.info(f"Loading base model from: {path.name}")
    try:
        payload = joblib.load(path)
    except Exception as e:
        raise ArtifactLoadError(f"Failed to deserialize base model: {e}")
    
    validated = _validate_payload(payload)
    
    # Verify fitted state
    try:
        _ = validated["model"].predict_proba(np.zeros((1, len(validated["features"]))))
    except NotFittedError:
        raise ArtifactLoadError("Loaded base model is not fitted")
    except Exception as e:
        raise ArtifactLoadError(f"Base model prediction check failed: {e}")

    logger.info("Base model loaded and verified successfully")
    return validated["model"]

def fit_calibrator(
    base_model: ClassifierMixin, 
    X: np.ndarray, 
    y: np.ndarray, 
    method: Literal["sigmoid", "isotonic"]
) -> CalibratedClassifierCV:
    """
    Fit a specific calibration method on top of the prefit base model.
    """
    logger.info(f"Fitting calibration method: {method}")
    
    # cv='prefit' ensures we use the provided model without refitting via CV
    calibrator = CalibratedClassifierCV(
        estimator=base_model,
        method=method,
        cv="prefit"
    )
    
    try:
        calibrator.fit(X, y)
    except Exception as e:
        raise CalibrationFitError(f"Calibration fit failed for method {method}: {e}")
    
    # Post-condition: Check probability output validity
    try:
        probs = calibrator.predict_proba(X[:1])
        if not np.all((probs >= 0) & (probs <= 1)):
            raise CalibrationFitError("Calibrator produced probabilities outside [0, 1]")
        if not np.isclose(np.sum(probs, axis=1), 1.0).all():
            raise CalibrationFitError("Calibrator probabilities do not sum to 1")
    except Exception as e:
        if isinstance(e, CalibrationFitError):
            raise e
        raise CalibrationFitError(f"Post-fit validation failed: {e}")

    logger.info(f"Calibration method {method} fitted and validated")
    return calibrator

def save_calibrated_model(model: CalibratedClassifierCV, method: str, output_dir: str) -> None:
    """Serialize the calibrated model with versioning."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"calibrated_model_{method}.pkl"
    full_path = out_path / filename
    
    payload = {
        "version": _MODEL_VERSION,
        "method": method,
        "model": model,
        "features": _FEATURE_COLS
    }
    
    logger.info(f"Saving calibrated model ({method}) to: {filename}")
    try:
        joblib.dump(payload, full_path)
    except Exception as e:
        raise ArtifactLoadError(f"Failed to save calibrated model: {e}")

def run_calibration_pipeline(
    base_model_path: str,
    validation_data_path: str,
    output_dir: str
) -> None:
    """
    Orchestrate the full calibration workflow.
    """
    # 1. Load Base
    base_model = load_base_model(base_model_path)
    
    # 2. Load Validation Data
    logger.info(f"Loading validation data from: {Path(validation_data_path).name}")
    try:
        df_val = pd.read_csv(validation_data_path, encoding=_ENCODING)
    except Exception as e:
        raise CalibrationDataError(f"Failed to read validation CSV: {e}")
    
    # We need the feature list from the loaded model payload to ensure exact match
    # Re-load payload briefly to get features if not passed, but here we rely on constant sync 
    # or re-extract. To be rigorous, let's re-read the payload features to ensure alignment.
    # However, to avoid double IO, we assume contract consistency or re-validate inside load_base_model.
    # The load_base_model already validated the payload structure. Let's adjust load_base_model to return features too?
    # Refactoring for purity: load_base_model returns model. We need features.
    # Let's re-load payload specifically for features to ensure absolute safety against drift.
    
    payload_raw = joblib.load(Path(base_model_path))
    feature_list = payload_raw.get("features", _FEATURE_COLS) # Fallback to constant if missing (though validation should catch)
    
    X_val, y_val = _validate_calibration_data(df_val, feature_list)
    logger.info(f"Validation data ready: {X_val.shape}")
    
    # 3. Fit Calibrators
    methods = [_METHOD_PLATT, _METHOD_ISOTONIC]
    
    for method in methods:
        cal_model = fit_calibrator(base_model, X_val, y_val, method)
        save_calibrated_model(cal_model, method, output_dir)

    logger.info("Calibration pipeline completed successfully")


# --- CLI Entry Point ---

if __name__ == "__main__":
    try:
        run_calibration_pipeline(
            base_model_path=_BASE_MODEL_PATH,
            validation_data_path=_VALIDATION_PATH,
            output_dir=_OUTPUT_DIR
        )
        sys.exit(0)
    except ArtifactLoadError as e:
        logger.critical(f"Artifact Error: {e}")
        sys.exit(1)
    except CalibrationDataError as e:
        logger.critical(f"Data Error: {e}")
        sys.exit(2)
    except CalibrationFitError as e:
        logger.critical(f"Fit Error: {e}")
        sys.exit(3)
    except Exception as e:
        logger.critical(f"Unexpected Failure: {e}")
        sys.exit(99)