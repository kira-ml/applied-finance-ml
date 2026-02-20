# -*- coding: utf-8 -*-
"""
Module: evaluate.py
Version: 1.0.0
Contract: Evaluate calibration performance (Brier Score, ECE) of base and calibrated models on test set.
Dependencies: python==3.9+, numpy==1.24.3, pandas==2.0.3, scikit-learn==1.3.0, joblib==1.3.0, matplotlib==3.7.2
Invariants:
    - Evaluation is performed strictly on the test set (no data leakage).
    - All models must be loaded from versioned artifacts matching current module version.
    - Plots are generated deterministically with fixed DPI and layout.
Failures:
    - ArtifactLoadError: Raised if any model artifact is missing, corrupted, or version mismatched.
    - EvaluationDataError: Raised if test data schema is invalid or contains NaNs.
    - PlotGenerationError: Raised if figure rendering fails.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for safety in scripts
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.base import ClassifierMixin

# --- Configuration Constants ---
_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S%z"
_ENCODING: Literal["utf-8"] = "utf-8"
_MODEL_VERSION: int = 1
_DPI_VALUE: int = 300
_N_BINS: int = 10  # Decile bins for ECE

# Paths
_TEST_PATH: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\data\processed\processed_test.csv"
_BASE_MODEL_PATH: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\src\artifacts\gbm_model.pkl"
_PLATT_MODEL_PATH: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\src\artifacts\calibrated_model_sigmoid.pkl"
_ISOTONIC_MODEL_PATH: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\src\artifacts\calibrated_model_isotonic.pkl"
_OUTPUT_DIR: str = r"D:\applied-finance-ml\project-02-credit-risk-probability-calibration\outputs"
_PLOT_FILENAME: str = "calibration_report.png"

# Model Identifiers
_ID_BASE: str = "GBM_Uncalibrated"
_ID_PLATT: str = "GBM_Platt_Scaled"
_ID_ISOTONIC: str = "GBM_Isotonic"

# Schema
_FEATURE_COLS: List[str] = [
    "income", "loan_amount", "revolving_balance", "num_delinquencies",
    "open_credit_lines", "months_employed", "prior_default", "co_applicant",
    "loan_purpose", "feature_dti", "feature_utilization", "feature_delinq_flag",
    "feature_revolving_ratio"
]
_TARGET_COL: str = "default"


# --- Custom Exceptions (Closed Failure Domain) ---

class ArtifactLoadError(IOError):
    """Raised when loading or verifying artifacts fails."""
    pass

class EvaluationDataError(ValueError):
    """Raised when test data violates preconditions."""
    pass

class PlotGenerationError(RuntimeError):
    """Raised if plot rendering fails."""
    pass


# --- Logging Setup (Structured, Local) ---

def _configure_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

logger = _configure_logger("evaluate_engine")


# --- Pure Utility Functions ---

def _validate_payload(payload: Any, expected_type: type) -> Dict[str, Any]:
    """Mechanically enforce artifact structure and versioning."""
    if not isinstance(payload, dict):
        raise ArtifactLoadError("Artifact payload is not a dictionary")
    if "version" not in payload:
        raise ArtifactLoadError("Artifact missing 'version' field")
    if payload["version"] != _MODEL_VERSION:
        raise ArtifactLoadError(f"Version mismatch: Expected {_MODEL_VERSION}, got {payload['version']}")
    if "model" not in payload or not isinstance(payload["model"], expected_type):
        raise ArtifactLoadError(f"Artifact missing valid model instance of type {expected_type}")
    return payload

def _validate_test_data(df: pd.DataFrame, expected_features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Validate test split schema and purity."""
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise EvaluationDataError(f"Test data missing features: {missing}")
    if _TARGET_COL not in df.columns:
        raise EvaluationDataError(f"Test data missing target: {_TARGET_COL}")

    try:
        X = df[expected_features].astype(np.float64).values
        y = df[_TARGET_COL].astype(np.int32).values
    except (ValueError, TypeError) as e:
        raise EvaluationDataError(f"Type coercion failed: {e}")

    if np.any(np.isnan(X)):
        raise EvaluationDataError("Test features contain NaN")
    if np.any(np.isnan(y)):
        raise EvaluationDataError("Test target contains NaN")
    if X.shape[0] == 0:
        raise EvaluationDataError("Test dataset is empty")

    return X, y

def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = _N_BINS) -> float:
    """
    Compute Expected Calibration Error over decile bins.
    Pure function: No side effects.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece_value = 0.0
    total_samples = len(y_true)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
        # Handle edge case for first bin inclusive of 0
        if i == 0:
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
            
        prop_in_bin = np.sum(in_bin) / total_samples
        
        if np.sum(in_bin) > 0:
            avg_confidence = np.mean(y_prob[in_bin])
            avg_accuracy = np.mean(y_true[in_bin])
            ece_value += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return float(ece_value)


# --- Effectful Core Logic ---

def load_model(path_str: str, expected_type: type) -> ClassifierMixin:
    """Load and verify a single model artifact."""
    path = Path(path_str)
    if not path.exists():
        raise ArtifactLoadError(f"Model not found: {path}")
    
    logger.info(f"Loading model: {path.name}")
    try:
        payload = joblib.load(path)
    except Exception as e:
        raise ArtifactLoadError(f"Failed to deserialize {path.name}: {e}")
    
    validated = _validate_payload(payload, expected_type)
    return validated["model"]

def get_predictions(model: ClassifierMixin, X: np.ndarray, label: str) -> np.ndarray:
    """Generate probability predictions for positive class."""
    logger.info(f"Generating predictions for {label}")
    try:
        # Explicitly request probability of positive class (index 1)
        probs = model.predict_proba(X)
        if probs.shape[1] < 2:
            raise EvaluationDataError(f"Model {label} does not output binary probabilities")
        return probs[:, 1]
    except Exception as e:
        raise EvaluationDataError(f"Prediction failed for {label}: {e}")

def generate_reliability_plot(
    y_true: np.ndarray,
    results: Dict[str, np.ndarray],
    output_path: Path
) -> None:
    """
    Generate and save reliability diagram.
    Effectful: Interacts with matplotlib filesystem backend.
    """
    logger.info(f"Generating reliability plot: {output_path.name}")
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        colors = {"blue": _ID_BASE, "green": _ID_PLATT, "red": _ID_ISOTONIC}
        # Map internal IDs to plot labels and colors deterministically
        plot_config = [
            (_ID_BASE, results[_ID_BASE], "blue", "o"),
            (_ID_PLATT, results[_ID_PLATT], "green", "s"),
            (_ID_ISOTONIC, results[_ID_ISOTONIC], "red", "^")
        ]

        for name, y_prob, color, marker in plot_config:
            # Compute bin stats for plotting
            bin_boundaries = np.linspace(0.0, 1.0, _N_BINS + 1)
            bin_centers = []
            bin_accuracies = []
            
            for i in range(_N_BINS):
                in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
                if i == 0:
                    in_bin = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i+1])
                
                if np.sum(in_bin) > 0:
                    bin_centers.append(np.mean(y_prob[in_bin]))
                    bin_accuracies.append(np.mean(y_true[in_bin]))
            
            if bin_centers:
                ax.plot(bin_centers, bin_accuracies, marker=marker, linestyle='-', 
                        color=color, label=name, markersize=8, linewidth=2)

        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title("Reliability Diagram (Test Set)", fontsize=14)
        ax.legend(loc="upper left")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=_DPI_VALUE, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        raise PlotGenerationError(f"Failed to generate plot: {e}")

def run_evaluation_pipeline() -> None:
    """Orchestrate loading, prediction, metric computation, and reporting."""
    
    # 1. Load Test Data
    logger.info(f"Loading test data: {Path(_TEST_PATH).name}")
    try:
        df_test = pd.read_csv(_TEST_PATH, encoding=_ENCODING)
    except Exception as e:
        raise EvaluationDataError(f"Failed to read test CSV: {e}")

    # Extract features from one of the model payloads to ensure exact alignment
    # We load base model first to get features if needed, but constants are defined.
    # To be rigorous, we check the payload of the base model for the feature list used during training.
    base_payload = joblib.load(Path(_BASE_MODEL_PATH))
    feature_list = base_payload.get("features", _FEATURE_COLS)
    
    X_test, y_test = _validate_test_data(df_test, feature_list)
    logger.info(f"Test data validated: {X_test.shape}")

    # 2. Load Models
    models = {
        _ID_BASE: load_model(_BASE_MODEL_PATH, ClassifierMixin),
        _ID_PLATT: load_model(_PLATT_MODEL_PATH, ClassifierMixin),
        _ID_ISOTONIC: load_model(_ISOTONIC_MODEL_PATH, ClassifierMixin)
    }

    # 3. Generate Predictions & Compute Metrics
    results_probs: Dict[str, np.ndarray] = {}
    metrics: List[Dict[str, Union[str, float]]] = []

    for name, model in models.items():
        probs = get_predictions(model, X_test, name)
        results_probs[name] = probs
        
        brier = brier_score_loss(y_test, probs)
        ece = _compute_ece(y_test, probs)
        
        metrics.append({
            "Model": name,
            "Brier_Score": brier,
            "ECE": ece
        })
        logger.info(f"{name}: Brier={brier:.6f}, ECE={ece:.6f}")

    # 4. Generate Plot
    out_path = Path(_OUTPUT_DIR) / _PLOT_FILENAME
    generate_reliability_plot(y_test, results_probs, out_path)

    # 5. Print Summary Table
    print("\n--- Calibration Evaluation Summary (Test Set) ---")
    df_metrics = pd.DataFrame(metrics)
    # Format floats for display without mutating underlying data
    pd.options.display.float_format = "{:.6f}".format
    print(df_metrics.to_string(index=False))
    print("-----------------------------------------------\n")
    
    logger.info("Evaluation pipeline completed successfully")


# --- CLI Entry Point ---

if __name__ == "__main__":
    try:
        run_evaluation_pipeline()
        sys.exit(0)
    except ArtifactLoadError as e:
        logger.critical(f"Artifact Error: {e}")
        sys.exit(1)
    except EvaluationDataError as e:
        logger.critical(f"Data Error: {e}")
        sys.exit(2)
    except PlotGenerationError as e:
        logger.critical(f"Plot Error: {e}")
        sys.exit(3)
    except Exception as e:
        logger.critical(f"Unexpected Failure: {e}")
        sys.exit(99)