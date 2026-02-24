"""
tests/test_modeling.py

Unit tests for src/modeling.py.
Focus: Model training, calibration, prediction, and serialization.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.modeling import (
    ModelingError,
    TrainingError,
    CalibrationError,
    SerializationError,
    train_base_model,
    calibrate_model,
    get_raw_predictions,
    save_model,
    load_model,
    train_and_calibrate,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

def _create_sample_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Creates simple synthetic binary classification data."""
    np.random.seed(42)
    
    # Training data
    X_train = np.random.randn(100, 5)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    
    # Calibration data
    X_cal = np.random.randn(30, 5)
    y_cal = (X_cal[:, 0] + X_cal[:, 1] > 0).astype(int)
    
    return X_train, y_train, X_cal, y_cal


# -----------------------------------------------------------------------------
# Unit Tests: Base Model Training
# -----------------------------------------------------------------------------

class TestTrainBaseModel:
    def test_train_success(self) -> None:
        X_train, y_train, _, _ = _create_sample_data()
        
        model = train_base_model(X_train, y_train)
        
        assert isinstance(model, GradientBoostingClassifier)
        assert hasattr(model, "predict_proba")
        assert model.n_estimators == 100
    
    def test_empty_training_data(self) -> None:
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])
        
        with pytest.raises(TrainingError, match="Training data is empty"):
            train_base_model(X_empty, y_empty)
    
    def test_length_mismatch(self) -> None:
        X_train = np.random.randn(10, 5)
        y_train = np.array([0, 1, 0])  # Wrong length
        
        with pytest.raises(TrainingError, match="length mismatch"):
            train_base_model(X_train, y_train)
    
    def test_custom_hyperparameters(self) -> None:
        X_train, y_train, _, _ = _create_sample_data()
        
        model = train_base_model(
            X_train, y_train,
            n_estimators=50,
            max_depth=2,
            learning_rate=0.05
        )
        
        assert model.n_estimators == 50
        assert model.max_depth == 2
        assert model.learning_rate == 0.05


# -----------------------------------------------------------------------------
# Unit Tests: Model Calibration
# -----------------------------------------------------------------------------

class TestCalibrateModel:
    def test_calibration_sigmoid_success(self) -> None:
        X_train, y_train, X_cal, y_cal = _create_sample_data()
        
        base_model = train_base_model(X_train, y_train)
        calibrated = calibrate_model(base_model, X_cal, y_cal, method="sigmoid")
        
        assert isinstance(calibrated, CalibratedClassifierCV)
        assert hasattr(calibrated, "predict_proba")
    
    def test_calibration_isotonic_success(self) -> None:
        X_train, y_train, X_cal, y_cal = _create_sample_data()
        
        base_model = train_base_model(X_train, y_train)
        calibrated = calibrate_model(base_model, X_cal, y_cal, method="isotonic")
        
        assert isinstance(calibrated, CalibratedClassifierCV)
    
    def test_empty_calibration_data(self) -> None:
        X_train, y_train, _, _ = _create_sample_data()
        base_model = train_base_model(X_train, y_train)
        
        X_empty = np.array([]).reshape(0, 5)
        y_empty = np.array([])
        
        with pytest.raises(CalibrationError, match="Calibration data is empty"):
            calibrate_model(base_model, X_empty, y_empty)
    
    def test_invalid_method(self) -> None:
        X_train, y_train, X_cal, y_cal = _create_sample_data()
        base_model = train_base_model(X_train, y_train)
        
        with pytest.raises(CalibrationError, match="Invalid calibration method"):
            calibrate_model(base_model, X_cal, y_cal, method="invalid_method")
    
    def test_length_mismatch(self) -> None:
        X_train, y_train, X_cal, _ = _create_sample_data()
        base_model = train_base_model(X_train, y_train)
        
        y_cal_wrong = np.array([0, 1])  # Wrong length
        
        with pytest.raises(CalibrationError, match="length mismatch"):
            calibrate_model(base_model, X_cal, y_cal_wrong)


# -----------------------------------------------------------------------------
# Unit Tests: Predictions
# -----------------------------------------------------------------------------

class TestGetRawPredictions:
    def test_predictions_shape(self) -> None:
        X_train, y_train, _, _ = _create_sample_data()
        model = train_base_model(X_train, y_train)
        
        X_test = np.random.randn(20, 5)
        predictions = get_raw_predictions(model, X_test)
        
        assert predictions.shape == (20, 2)  # Binary classification
        assert np.all((predictions >= 0) & (predictions <= 1))
        assert np.allclose(predictions.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_empty_input(self) -> None:
        X_train, y_train, _, _ = _create_sample_data()
        model = train_base_model(X_train, y_train)
        
        X_empty = np.array([]).reshape(0, 5)
        
        with pytest.raises(ValueError, match="Input data is empty"):
            get_raw_predictions(model, X_empty)


# -----------------------------------------------------------------------------
# Unit Tests: Serialization
# -----------------------------------------------------------------------------

class TestModelSerialization:
    def test_save_and_load_model(self) -> None:
        X_train, y_train, _, _ = _create_sample_data()
        model = train_base_model(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            
            save_model(model, model_path)
            assert model_path.exists()
            
            loaded_model = load_model(model_path)
            assert isinstance(loaded_model, GradientBoostingClassifier)
            
            # Verify same predictions
            X_test = np.random.randn(10, 5)
            original_pred = model.predict_proba(X_test)
            loaded_pred = loaded_model.predict_proba(X_test)
            
            assert np.allclose(original_pred, loaded_pred)
    
    def test_load_nonexistent_model(self) -> None:
        fake_path = Path("/nonexistent/path/model.pkl")
        
        with pytest.raises(SerializationError, match="Model file not found"):
            load_model(fake_path)
    
    def test_save_creates_parent_directory(self) -> None:
        X_train, y_train, _, _ = _create_sample_data()
        model = train_base_model(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "model.pkl"
            
            save_model(model, nested_path)
            assert nested_path.exists()


# -----------------------------------------------------------------------------
# Unit Tests: End-to-End Pipeline
# -----------------------------------------------------------------------------

class TestTrainAndCalibrate:
    def test_end_to_end_pipeline(self) -> None:
        X_train, y_train, X_cal, y_cal = _create_sample_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base_model.pkl"
            cal_path = Path(tmpdir) / "calibrated_model.pkl"
            
            base_model, calibrated_model = train_and_calibrate(
                X_train, y_train,
                X_cal, y_cal,
                base_path, cal_path,
                n_estimators=50
            )
            
            # Check models are returned
            assert isinstance(base_model, GradientBoostingClassifier)
            assert isinstance(calibrated_model, CalibratedClassifierCV)
            
            # Check files are created
            assert base_path.exists()
            assert cal_path.exists()
            
            # Check models can make predictions
            X_test = np.random.randn(10, 5)
            base_pred = base_model.predict_proba(X_test)
            cal_pred = calibrated_model.predict_proba(X_test)
            
            assert base_pred.shape == (10, 2)
            assert cal_pred.shape == (10, 2)
    
    def test_custom_calibration_method(self) -> None:
        X_train, y_train, X_cal, y_cal = _create_sample_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.pkl"
            cal_path = Path(tmpdir) / "cal.pkl"
            
            _, calibrated = train_and_calibrate(
                X_train, y_train,
                X_cal, y_cal,
                base_path, cal_path,
                calibration_method="isotonic"
            )
            
            assert isinstance(calibrated, CalibratedClassifierCV)
