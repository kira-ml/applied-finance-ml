import json
import pickle
import logging
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional


logger = logging.getLogger(__name__)


class EvaluatorError(Exception):
    """Base exception for evaluator module."""
    pass


class ModelLoadError(EvaluatorError):
    """Raised when model loading fails."""
    pass


class PreprocessorLoadError(EvaluatorError):
    """Raised when preprocessor loading fails."""
    pass


class ThresholdLoadError(EvaluatorError):
    """Raised when threshold loading fails."""
    pass


class TestDataError(EvaluatorError):
    """Raised when test data is invalid."""
    pass


class EvaluationError(EvaluatorError):
    """Raised when evaluation computation fails."""
    pass


class PrAucThresholdError(EvaluatorError):
    """Raised when PR-AUC falls below required threshold."""
    pass


@dataclass(frozen=True)
class EvaluationMetrics:
    """Immutable container for evaluation metrics."""
    precision: float
    recall: float
    f2_score: float
    pr_auc: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


class Evaluator:
    """
    Evaluates final model performance on held-out test set.
    
    Loads model, preprocessor, and threshold; computes metrics;
    asserts PR-AUC > 0.40; generates final report.
    """

    # Constants
    REQUIRED_PR_AUC = 0.40
    F2_BETA = 2.0
    
    def __init__(
        self,
        model_path: str = "model.pkl",
        preprocessor_path: str = "preprocessor.pkl",
        threshold_path: str = "threshold.txt",
        test_features_path: str = "X_test.npy",
        test_labels_path: str = "y_test.npy",
        report_path: str = "final_report.txt"
    ) -> None:
        """
        Initialize evaluator with file paths.
        
        Args:
            model_path: Path to trained model pickle
            preprocessor_path: Path to preprocessor pickle
            threshold_path: Path to threshold text file
            test_features_path: Path to test features numpy file
            test_labels_path: Path to test labels numpy file
            report_path: Path for output report
        """
        self._model_path = model_path
        self._preprocessor_path = preprocessor_path
        self._threshold_path = threshold_path
        self._test_features_path = test_features_path
        self._test_labels_path = test_labels_path
        self._report_path = report_path
        
        # State (initialized as None, populated during evaluate)
        self._model = None
        self._preprocessor = None
        self._threshold = None
        self._X_test = None
        self._y_test = None
    
    def _load_model(self) -> Any:
        """Load trained model from pickle file."""
        try:
            with open(self._model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded from {self._model_path}")
            return model
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {self._model_path}: {str(e)}") from e
    
    def _load_preprocessor(self) -> Any:
        """Load preprocessor from pickle file."""
        try:
            with open(self._preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            logger.info(f"Preprocessor loaded from {self._preprocessor_path}")
            return preprocessor
        except Exception as e:
            raise PreprocessorLoadError(f"Failed to load preprocessor from {self._preprocessor_path}: {str(e)}") from e
    
    def _load_threshold(self) -> float:
        """Load optimal threshold from text file."""
        try:
            with open(self._threshold_path, 'r') as f:
                content = f.read().strip()
                threshold = float(content)
            logger.info(f"Threshold loaded from {self._threshold_path}: {threshold}")
            return threshold
        except Exception as e:
            raise ThresholdLoadError(f"Failed to load threshold from {self._threshold_path}: {str(e)}") from e
    
    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test features and labels from numpy files."""
        try:
            X_test = np.load(self._test_features_path)
            y_test = np.load(self._test_labels_path)
            logger.info(f"Test data loaded: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
            return X_test, y_test
        except Exception as e:
            raise TestDataError(f"Failed to load test data: {str(e)}") from e
    
    def _validate_test_data(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Validate test data arrays."""
        if not isinstance(X_test, np.ndarray):
            raise TestDataError(f"X_test must be numpy array, got {type(X_test)}")
        if not isinstance(y_test, np.ndarray):
            raise TestDataError(f"y_test must be numpy array, got {type(y_test)}")
        
        if X_test.size == 0:
            raise TestDataError("X_test cannot be empty")
        if y_test.size == 0:
            raise TestDataError("y_test cannot be empty")
        
        if X_test.ndim != 2:
            raise TestDataError(f"X_test must be 2D, got {X_test.ndim}D")
        if y_test.ndim != 1:
            raise TestDataError(f"y_test must be 1D, got {y_test.ndim}D")
        
        if X_test.shape[0] != y_test.shape[0]:
            raise TestDataError(
                f"Sample mismatch: X_test has {X_test.shape[0]} samples, "
                f"y_test has {y_test.shape[0]} samples"
            )
        
        if np.any(np.isnan(X_test)):
            raise TestDataError("X_test contains NaN values")
        if np.any(np.isinf(X_test)):
            raise TestDataError("X_test contains infinite values")
        if np.any(np.isnan(y_test)):
            raise TestDataError("y_test contains NaN values")
        if np.any(np.isinf(y_test)):
            raise TestDataError("y_test contains infinite values")
        
        unique_labels = np.unique(y_test)
        if not np.array_equal(unique_labels, np.array([0, 1])) and \
           not np.array_equal(unique_labels, np.array([0])) and \
           not np.array_equal(unique_labels, np.array([1])):
            raise TestDataError(f"y_test must contain only 0 and 1, got {unique_labels}")
    
    def _validate_threshold(self, threshold: float) -> None:
        """Validate threshold value."""
        if not isinstance(threshold, (int, float)):
            raise ThresholdLoadError(f"Threshold must be numeric, got {type(threshold)}")
        if np.isnan(threshold) or np.isinf(threshold):
            raise ThresholdLoadError(f"Threshold must be finite, got {threshold}")
        if threshold < 0.0 or threshold > 1.0:
            raise ThresholdLoadError(f"Threshold must be in [0, 1], got {threshold}")
    
    def _compute_confusion_counts(
        self, predictions: np.ndarray, true_labels: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """Compute confusion matrix counts."""
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        return int(tp), int(fp), int(tn), int(fn)
    
    def _compute_pr_auc(self, probabilities: np.ndarray, true_labels: np.ndarray) -> float:
        """
        Compute Area Under Precision-Recall Curve.
        
        Uses all probability values as thresholds for precise AUC.
        """
        # Sort by probability descending
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probs = probabilities[sorted_indices]
        sorted_labels = true_labels[sorted_indices]
        
        # Compute precision-recall curve
        precisions = []
        recalls = []
        
        total_positives = np.sum(sorted_labels == 1)
        if total_positives == 0:
            return 0.0
        
        tp = 0
        fp = 0
        
        for i in range(len(sorted_probs)):
            if sorted_labels[i] == 1:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / total_positives if total_positives > 0 else 0.0
            
            # Add point when threshold changes or at end
            if i == len(sorted_probs) - 1 or sorted_probs[i] != sorted_probs[i + 1]:
                precisions.append(precision)
                recalls.append(recall)
        
        # Add endpoint if needed
        if recalls[-1] < 1.0:
            recalls.append(1.0)
            precisions.append(0.0)
        
        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(len(recalls) - 1):
            width = recalls[i + 1] - recalls[i]
            height = (precisions[i + 1] + precisions[i]) / 2.0
            auc += width * height
        
        return float(auc)
    
    def _compute_metrics(
        self, probabilities: np.ndarray, true_labels: np.ndarray, threshold: float
    ) -> EvaluationMetrics:
        """Compute all evaluation metrics."""
        # Apply threshold
        predictions = (probabilities >= threshold).astype(np.int64)
        
        # Compute confusion counts
        tp, fp, tn, fn = self._compute_confusion_counts(predictions, true_labels)
        
        # Compute metrics with safe division
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F2-score
        beta_squared = self.F2_BETA * self.F2_BETA
        denominator = (beta_squared * precision + recall)
        f2_score = ((1 + beta_squared) * precision * recall / denominator) if denominator > 0 else 0.0
        
        # PR-AUC
        pr_auc = self._compute_pr_auc(probabilities, true_labels)
        
        return EvaluationMetrics(
            precision=float(precision),
            recall=float(recall),
            f2_score=float(f2_score),
            pr_auc=float(pr_auc),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn
        )
    
    def _assert_pr_auc(self, pr_auc: float) -> None:
        """Assert PR-AUC meets minimum threshold."""
        if pr_auc <= self.REQUIRED_PR_AUC:
            raise PrAucThresholdError(
                f"PR-AUC {pr_auc:.4f} below required threshold {self.REQUIRED_PR_AUC:.4f}"
            )
        logger.info(f"PR-AUC {pr_auc:.4f} meets required threshold {self.REQUIRED_PR_AUC:.4f}")
    
    def _generate_report(self, metrics: EvaluationMetrics) -> str:
        """Generate formatted report string."""
        report_lines = [
            "FINAL MODEL EVALUATION REPORT",
            "=" * 40,
            "",
            f"Optimal Threshold: {self._threshold:.4f}",
            "",
            "Confusion Matrix:",
            f"  True Positives:  {metrics.true_positives:6d}",
            f"  False Positives: {metrics.false_positives:6d}",
            f"  True Negatives:  {metrics.true_negatives:6d}",
            f"  False Negatives: {metrics.false_negatives:6d}",
            "",
            "Performance Metrics:",
            f"  Precision: {metrics.precision:.4f}",
            f"  Recall:    {metrics.recall:.4f}",
            f"  F2-Score:  {metrics.f2_score:.4f}",
            f"  PR-AUC:    {metrics.pr_auc:.4f}",
            "",
            f"PR-AUC Requirement: > {self.REQUIRED_PR_AUC:.4f}",
            f"PR-AUC Status: {'✓ PASSED' if metrics.pr_auc > self.REQUIRED_PR_AUC else '✗ FAILED'}"
        ]
        
        return "\n".join(report_lines)
    
    def _save_report(self, report: str) -> None:
        """Save report to file."""
        try:
            with open(self._report_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {self._report_path}")
        except Exception as e:
            raise EvaluatorError(f"Failed to save report to {self._report_path}: {str(e)}") from e
    
    def evaluate(self) -> EvaluationMetrics:
        """
        Execute complete evaluation pipeline.
        
        Returns:
            EvaluationMetrics containing all computed metrics
            
        Raises:
            Various evaluation errors based on failure mode
            PrAucThresholdError if PR-AUC falls below required threshold
        """
        # Load all artifacts
        self._model = self._load_model()
        self._preprocessor = self._load_preprocessor()
        self._threshold = self._load_threshold()
        self._X_test, self._y_test = self._load_test_data()
        
        # Validate loaded data
        self._validate_test_data(self._X_test, self._y_test)
        self._validate_threshold(self._threshold)
        
        try:
            # Transform test features
            X_test_transformed = self._preprocessor.transform(self._X_test)
            
            # Generate probabilities
            probabilities = self._model.predict_proba(X_test_transformed)
            # Take positive class probabilities
            if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                positive_probs = probabilities[:, 1]
            else:
                positive_probs = probabilities
            
            # Compute metrics
            metrics = self._compute_metrics(positive_probs, self._y_test, self._threshold)
            
            # Assert PR-AUC requirement
            self._assert_pr_auc(metrics.pr_auc)
            
            # Generate and save report
            report = self._generate_report(metrics)
            self._save_report(report)
            
            return metrics
            
        except PrAucThresholdError:
            # Re-raise threshold errors
            raise
        except Exception as e:
            raise EvaluationError(f"Evaluation failed: {str(e)}") from e