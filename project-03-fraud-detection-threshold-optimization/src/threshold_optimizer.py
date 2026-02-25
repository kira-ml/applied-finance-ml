import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Union


class ThresholdOptimizerError(Exception):
    """Base exception for threshold optimizer module."""
    pass


class InvalidProbabilitiesError(ThresholdOptimizerError):
    """Raised when probability array is invalid."""
    pass


class InvalidLabelsError(ThresholdOptimizerError):
    """Raised when labels array is invalid."""
    pass


class InvalidCostMatrixError(ThresholdOptimizerError):
    """Raised when cost matrix values are invalid."""
    pass


class MetricComputationError(ThresholdOptimizerError):
    """Raised when metric computation fails."""
    pass


@dataclass(frozen=True)
class ThresholdMetrics:
    """Metrics computed for a single threshold."""
    threshold: float
    precision: float
    recall: float
    f2_score: float
    total_cost: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class OptimizationResult:
    """Complete result of threshold optimization."""
    optimal_threshold: float
    optimal_metrics: ThresholdMetrics
    pr_auc: float
    all_metrics: List[ThresholdMetrics]


class ThresholdOptimizer:
    """
    Optimizes classification threshold based on validation probabilities and true labels.
    
    Evaluates thresholds from 0.01 to 0.99, computes metrics, and identifies optimal
    threshold maximizing F2-score (or minimizing cost if cost matrix provided).
    """

    # Constants
    MIN_THRESHOLD = 0.01
    MAX_THRESHOLD = 0.99
    NUM_THRESHOLDS = 99  # (0.99 - 0.01) / 0.01 + 1
    F2_BETA = 2.0
    
    def __init__(self, cost_fn: float = 1.0, cost_fp: float = 1.0):
        """
        Initialize threshold optimizer with optional cost matrix.
        
        Args:
            cost_fn: Cost of false negative (must be non-negative)
            cost_fp: Cost of false positive (must be non-negative)
            
        Raises:
            InvalidCostMatrixError: If costs are negative or NaN
        """
        self._validate_costs(cost_fn, cost_fp)
        self._cost_fn = float(cost_fn)
        self._cost_fp = float(cost_fp)
    
    def _validate_costs(self, cost_fn: float, cost_fp: float) -> None:
        """Validate cost matrix values."""
        if not isinstance(cost_fn, (int, float)):
            raise InvalidCostMatrixError(f"cost_fn must be numeric, got {type(cost_fn)}")
        if not isinstance(cost_fp, (int, float)):
            raise InvalidCostMatrixError(f"cost_fp must be numeric, got {type(cost_fp)}")
        
        if np.isnan(cost_fn) or np.isinf(cost_fn):
            raise InvalidCostMatrixError(f"cost_fn must be finite, got {cost_fn}")
        if np.isnan(cost_fp) or np.isinf(cost_fp):
            raise InvalidCostMatrixError(f"cost_fp must be finite, got {cost_fp}")
        
        if cost_fn < 0.0:
            raise InvalidCostMatrixError(f"cost_fn must be non-negative, got {cost_fn}")
        if cost_fp < 0.0:
            raise InvalidCostMatrixError(f"cost_fp must be non-negative, got {cost_fp}")
    
    def _validate_inputs(self, probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Validate probability and label arrays.
        
        Raises:
            InvalidProbabilitiesError: If probabilities array is invalid
            InvalidLabelsError: If labels array is invalid
        """
        # Type checks
        if not isinstance(probabilities, np.ndarray):
            raise InvalidProbabilitiesError(f"probabilities must be numpy array, got {type(probabilities)}")
        if not isinstance(true_labels, np.ndarray):
            raise InvalidLabelsError(f"true_labels must be numpy array, got {type(true_labels)}")
        
        # Non-empty checks
        if probabilities.size == 0:
            raise InvalidProbabilitiesError("probabilities cannot be empty")
        if true_labels.size == 0:
            raise InvalidLabelsError("true_labels cannot be empty")
        
        # Dimension checks
        if probabilities.ndim != 1:
            raise InvalidProbabilitiesError(f"probabilities must be 1D, got {probabilities.ndim}D")
        if true_labels.ndim != 1:
            raise InvalidLabelsError(f"true_labels must be 1D, got {true_labels.ndim}D")
        
        # Length consistency
        if probabilities.shape[0] != true_labels.shape[0]:
            raise InvalidProbabilitiesError(
                f"Length mismatch: probabilities has {probabilities.shape[0]} samples, "
                f"true_labels has {true_labels.shape[0]} samples"
            )
        
        # Probability bounds
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise InvalidProbabilitiesError("probabilities must be in [0, 1] range")
        
        # NaN/Inf checks
        if np.any(np.isnan(probabilities)):
            raise InvalidProbabilitiesError("probabilities contains NaN values")
        if np.any(np.isinf(probabilities)):
            raise InvalidProbabilitiesError("probabilities contains infinite values")
        
        # Label validation (binary 0/1)
        unique_labels = np.unique(true_labels)
        if not np.array_equal(unique_labels, np.array([0, 1])) and \
           not np.array_equal(unique_labels, np.array([0])) and \
           not np.array_equal(unique_labels, np.array([1])):
            raise InvalidLabelsError(f"true_labels must contain only 0 and 1, got {unique_labels}")
        
        # Check for NaN/Inf in labels
        if np.any(np.isnan(true_labels)):
            raise InvalidLabelsError("true_labels contains NaN values")
        if np.any(np.isinf(true_labels)):
            raise InvalidLabelsError("true_labels contains infinite values")
    
    def _compute_confusion_counts(self, predictions: np.ndarray, true_labels: np.ndarray) -> Tuple[int, int, int]:
        """Compute TP, FP, FN counts."""
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        return int(tp), int(fp), int(fn)
    
    def _compute_metrics_for_threshold(self, threshold: float, 
                                       probabilities: np.ndarray, 
                                       true_labels: np.ndarray) -> ThresholdMetrics:
        """Compute all metrics for a single threshold."""
        # Apply threshold
        predictions = (probabilities >= threshold).astype(np.int64)
        
        # Compute confusion counts
        tp, fp, fn = self._compute_confusion_counts(predictions, true_labels)
        
        # Compute metrics with safe division
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F2-score
        beta_squared = self.F2_BETA * self.F2_BETA
        denominator = (beta_squared * precision + recall)
        f2_score = ((1 + beta_squared) * precision * recall / denominator) if denominator > 0 else 0.0
        
        # Total cost
        total_cost = (fn * self._cost_fn) + (fp * self._cost_fp)
        
        return ThresholdMetrics(
            threshold=threshold,
            precision=float(precision),
            recall=float(recall),
            f2_score=float(f2_score),
            total_cost=float(total_cost),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn
        )
    
    def _compute_pr_auc(self, metrics: List[ThresholdMetrics]) -> float:
        """
        Compute Area Under Precision-Recall Curve using trapezoidal rule.
        
        Sorts by recall (x-axis) and computes AUC.
        """
        # Sort by recall (x-axis for PR curve)
        sorted_metrics = sorted(metrics, key=lambda m: m.recall)
        
        # Extract precision and recall
        recalls = np.array([m.recall for m in sorted_metrics])
        precisions = np.array([m.precision for m in sorted_metrics])
        
        # Add endpoints if missing
        if recalls[0] > 0:
            recalls = np.concatenate([[0.0], recalls])
            precisions = np.concatenate([[precisions[0]], precisions])
        if recalls[-1] < 1:
            recalls = np.concatenate([recalls, [1.0]])
            precisions = np.concatenate([precisions, [0.0]])
        
        # Compute AUC using trapezoidal rule
        auc = 0.0
        for i in range(len(recalls) - 1):
            width = recalls[i + 1] - recalls[i]
            height = (precisions[i + 1] + precisions[i]) / 2.0
            auc += width * height
        
        return float(auc)
    
    def optimize(self, probabilities: np.ndarray, true_labels: np.ndarray) -> OptimizationResult:
        """
        Find optimal threshold by maximizing F2-score (or minimizing cost).
        
        Args:
            probabilities: Predicted probabilities for positive class (1D array)
            true_labels: True binary labels (1D array of 0/1)
            
        Returns:
            OptimizationResult containing optimal threshold, metrics, and PR-AUC
            
        Raises:
            InvalidProbabilitiesError: If probabilities array is invalid
            InvalidLabelsError: If labels array is invalid
            MetricComputationError: If metric computation fails
        """
        # Validate inputs
        self._validate_inputs(probabilities, true_labels)
        
        try:
            # Generate thresholds
            thresholds = np.linspace(
                self.MIN_THRESHOLD, 
                self.MAX_THRESHOLD, 
                self.NUM_THRESHOLDS
            )
            
            # Compute metrics for all thresholds
            all_metrics = []
            for threshold in thresholds:
                metrics = self._compute_metrics_for_threshold(
                    threshold, probabilities, true_labels
                )
                all_metrics.append(metrics)
            
            # Find optimal threshold (maximize F2-score)
            optimal_metrics = max(all_metrics, key=lambda m: m.f2_score)
            
            # Compute PR-AUC
            pr_auc = self._compute_pr_auc(all_metrics)
            
            return OptimizationResult(
                optimal_threshold=optimal_metrics.threshold,
                optimal_metrics=optimal_metrics,
                pr_auc=pr_auc,
                all_metrics=all_metrics
            )
            
        except Exception as e:
            raise MetricComputationError(f"Failed to compute metrics: {str(e)}") from e
    
    def save_analysis(self, result: OptimizationResult, json_path: str = "threshold_analysis.json") -> None:
        """
        Save threshold analysis to JSON file.
        
        Args:
            result: OptimizationResult from optimize() method
            json_path: Path to save JSON file
            
        Raises:
            ThresholdOptimizerError: If file writing fails
        """
        # Convert result to serializable dict
        analysis_dict = {
            "optimal_threshold": result.optimal_threshold,
            "pr_auc": result.pr_auc,
            "optimal_metrics": asdict(result.optimal_metrics),
            "all_metrics": [asdict(m) for m in result.all_metrics],
            "cost_matrix": {
                "cost_fn": self._cost_fn,
                "cost_fp": self._cost_fp
            }
        }
        
        try:
            with open(json_path, 'w') as f:
                json.dump(analysis_dict, f, indent=2)
        except Exception as e:
            raise ThresholdOptimizerError(f"Failed to save analysis to {json_path}: {str(e)}") from e
    
    def save_threshold(self, threshold: float, txt_path: str = "threshold.txt") -> None:
        """
        Save optimal threshold to text file.
        
        Args:
            threshold: Threshold value to save
            txt_path: Path to save text file
            
        Raises:
            ThresholdOptimizerError: If file writing fails
        """
        try:
            with open(txt_path, 'w') as f:
                f.write(f"{threshold:.4f}\n")
        except Exception as e:
            raise ThresholdOptimizerError(f"Failed to save threshold to {txt_path}: {str(e)}") from e