"""
tests/test_evaluate.py

Unit tests for src/evaluate.py.
Focus: Metrics calculation, threshold validation, and report generation.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluate import (
    EvaluationError,
    MetricsComputationError,
    calculate_brier_score,
    calculate_calibration_error,
    calculate_improvement_percentage,
    validate_improvement_threshold,
    validate_calibration_error,
    generate_metrics_report,
    save_metrics_to_json,
    evaluate_and_save,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

def _create_perfect_predictions() -> tuple:
    """Perfect predictions - Brier score = 0."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return y_true, y_proba


def _create_random_predictions() -> tuple:
    """Random predictions - Brier score ~0.25."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100)
    return y_true, y_proba


def _create_2d_probabilities() -> tuple:
    """2D probability array (n_samples, n_classes)."""
    y_true = np.array([0, 0, 1, 1])
    y_proba_2d = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.2, 0.8]
    ])
    return y_true, y_proba_2d


# -----------------------------------------------------------------------------
# Unit Tests: Brier Score
# -----------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_predictions(self) -> None:
        y_true, y_proba = _create_perfect_predictions()
        score = calculate_brier_score(y_true, y_proba)
        
        assert score == 0.0
    
    def test_worst_predictions(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # Completely wrong
        score = calculate_brier_score(y_true, y_proba)
        
        assert score == 1.0
    
    def test_2d_probability_array(self) -> None:
        y_true, y_proba_2d = _create_2d_probabilities()
        score = calculate_brier_score(y_true, y_proba_2d)
        
        assert 0.0 <= score <= 1.0
    
    def test_empty_data(self) -> None:
        y_true = np.array([])
        y_proba = np.array([])
        
        with pytest.raises(MetricsComputationError, match="empty data"):
            calculate_brier_score(y_true, y_proba)
    
    def test_length_mismatch(self) -> None:
        y_true = np.array([0, 1, 0])
        y_proba = np.array([0.1, 0.2])
        
        with pytest.raises(MetricsComputationError, match="Length mismatch"):
            calculate_brier_score(y_true, y_proba)


# -----------------------------------------------------------------------------
# Unit Tests: Calibration Error
# -----------------------------------------------------------------------------

class TestCalibrationError:
    def test_perfect_calibration(self) -> None:
        # Create perfectly calibrated predictions
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_proba = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        
        error = calculate_calibration_error(y_true, y_proba, n_bins=2)
        
        assert error == 0.0
    
    def test_calibration_with_bins(self) -> None:
        y_true, y_proba = _create_random_predictions()
        
        error = calculate_calibration_error(y_true, y_proba, n_bins=10)
        
        assert 0.0 <= error <= 1.0
    
    def test_2d_probability_array(self) -> None:
        y_true, y_proba_2d = _create_2d_probabilities()
        
        error = calculate_calibration_error(y_true, y_proba_2d)
        
        assert 0.0 <= error <= 1.0
    
    def test_empty_data(self) -> None:
        y_true = np.array([])
        y_proba = np.array([])
        
        with pytest.raises(MetricsComputationError, match="empty data"):
            calculate_calibration_error(y_true, y_proba)
    
    def test_length_mismatch(self) -> None:
        y_true = np.array([0, 1, 0])
        y_proba = np.array([0.1, 0.2])
        
        with pytest.raises(MetricsComputationError, match="Length mismatch"):
            calculate_calibration_error(y_true, y_proba)


# -----------------------------------------------------------------------------
# Unit Tests: Improvement Calculation
# -----------------------------------------------------------------------------

class TestImprovementPercentage:
    def test_improvement_calculation(self) -> None:
        baseline = 0.25
        improved = 0.20
        
        improvement = calculate_improvement_percentage(baseline, improved)
        
        assert improvement == pytest.approx(20.0)  # (0.25 - 0.20) / 0.25 * 100 = 20%
    
    def test_no_improvement(self) -> None:
        baseline = 0.25
        improved = 0.25
        
        improvement = calculate_improvement_percentage(baseline, improved)
        
        assert improvement == 0.0
    
    def test_negative_improvement(self) -> None:
        baseline = 0.20
        improved = 0.25
        
        improvement = calculate_improvement_percentage(baseline, improved)
        
        assert improvement == pytest.approx(-25.0)  # Got worse
    
    def test_zero_baseline(self) -> None:
        improvement = calculate_improvement_percentage(0.0, 0.0)
        
        assert improvement == 0.0


# -----------------------------------------------------------------------------
# Unit Tests: Threshold Validation
# -----------------------------------------------------------------------------

class TestThresholdValidation:
    def test_improvement_meets_threshold(self) -> None:
        assert validate_improvement_threshold(20.0, 15.0) is True
        assert validate_improvement_threshold(15.0, 15.0) is True
        assert validate_improvement_threshold(10.0, 15.0) is False
    
    def test_calibration_error_threshold(self) -> None:
        assert validate_calibration_error(0.03, 0.05) is True
        assert validate_calibration_error(0.05, 0.05) is True
        assert validate_calibration_error(0.07, 0.05) is False


# -----------------------------------------------------------------------------
# Unit Tests: Metrics Report Generation
# -----------------------------------------------------------------------------

class TestGenerateMetricsReport:
    def test_report_structure(self) -> None:
        y_true, y_proba = _create_random_predictions()
        
        report = generate_metrics_report(
            y_test=y_true,
            y_proba_base=y_proba,
            y_proba_calibrated=y_proba * 0.95  # Slight improvement
        )
        
        # Check all required keys
        required_keys = [
            "brier_score_base",
            "brier_score_calibrated",
            "brier_improvement_pct",
            "calibration_error_base",
            "calibration_error_calibrated",
            "threshold_improvement_min",
            "threshold_calibration_max",
            "meets_improvement_threshold",
            "meets_calibration_threshold",
            "overall_success",
            "status"
        ]
        
        for key in required_keys:
            assert key in report
    
    def test_passing_report(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1] * 20)
        y_proba_base = np.array([0.4, 0.4, 0.4, 0.6, 0.6, 0.6] * 20)
        y_proba_cal = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9] * 20)  # Much better
        
        report = generate_metrics_report(
            y_test=y_true,
            y_proba_base=y_proba_base,
            y_proba_calibrated=y_proba_cal,
            min_improvement_threshold=10.0,
            max_calibration_error_threshold=0.1
        )
        
        assert report["brier_score_calibrated"] < report["brier_score_base"]
        assert report["brier_improvement_pct"] > 0
        assert report["status"] == "PASS"
    
    def test_failing_report(self) -> None:
        y_true, y_proba = _create_random_predictions()
        
        report = generate_metrics_report(
            y_test=y_true,
            y_proba_base=y_proba,
            y_proba_calibrated=y_proba,  # No improvement
            min_improvement_threshold=15.0
        )
        
        assert report["meets_improvement_threshold"] is False
        assert report["status"] == "FAIL"
    
    def test_custom_thresholds(self) -> None:
        y_true, y_proba = _create_random_predictions()
        
        report = generate_metrics_report(
            y_test=y_true,
            y_proba_base=y_proba,
            y_proba_calibrated=y_proba * 0.9,
            min_improvement_threshold=5.0,
            max_calibration_error_threshold=0.2
        )
        
        assert report["threshold_improvement_min"] == 5.0
        assert report["threshold_calibration_max"] == 0.2


# -----------------------------------------------------------------------------
# Unit Tests: JSON Serialization
# -----------------------------------------------------------------------------

class TestSaveMetricsToJSON:
    def test_save_and_load(self) -> None:
        metrics = {
            "brier_score_base": 0.25,
            "brier_score_calibrated": 0.20,
            "status": "PASS"
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            
            save_metrics_to_json(metrics, output_path)
            
            assert output_path.exists()
            
            with open(output_path, "r") as f:
                loaded = json.load(f)
            
            assert loaded == metrics
    
    def test_creates_parent_directory(self) -> None:
        metrics = {"test": "value"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "metrics.json"
            
            save_metrics_to_json(metrics, nested_path)
            
            assert nested_path.exists()


# -----------------------------------------------------------------------------
# Unit Tests: End-to-End Evaluation
# -----------------------------------------------------------------------------

class TestEvaluateAndSave:
    def test_end_to_end_pipeline(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1] * 20)
        y_proba_base = np.array([0.4, 0.4, 0.4, 0.6, 0.6, 0.6] * 20)
        y_proba_cal = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9] * 20)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            
            metrics = evaluate_and_save(
                y_test=y_true,
                y_proba_base=y_proba_base,
                y_proba_calibrated=y_proba_cal,
                output_path=output_path,
                min_improvement_threshold=10.0
            )
            
            # Check return value
            assert isinstance(metrics, dict)
            assert "status" in metrics
            
            # Check file was created
            assert output_path.exists()
            
            # Verify file contents match return value
            with open(output_path, "r") as f:
                saved_metrics = json.load(f)
            
            assert saved_metrics == metrics
