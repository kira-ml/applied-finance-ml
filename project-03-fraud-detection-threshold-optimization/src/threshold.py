import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, fbeta_score, confusion_matrix
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

@dataclass(frozen=True)
class CostConfig:
    fn_cost: int = 10
    fp_cost: int = 1
    comment: str = "FN (missed fraud) assumed 10× more costly than FP (false alarm). Adjust to reflect actual business costs."

@dataclass(frozen=True)
class ThresholdConfig:
    beta: float = 2.0
    agreement_tolerance: float = 0.05
    pr_curve_path: str = "artifacts/pr_curve.png"
    threshold_path: str = "artifacts/threshold.txt"
    report_path: str = "artifacts/evaluation_report.txt"
    scale_transactions: int = 10000

class ThresholdError(Exception):
    """Base exception for threshold optimization failures."""
    pass

class InputTypeError(ThresholdError):
    """Raised when input types are invalid."""
    pass

class EmptyInputError(ThresholdError):
    """Raised when input arrays are empty."""
    pass

class ShapeMismatchError(ThresholdError):
    """Raised when probabilities and labels have mismatched shapes."""
    pass

class InvalidValuesError(ThresholdError):
    """Raised when input contains invalid values (NaN, Inf)."""
    pass

class MetricComputationError(ThresholdError):
    """Raised when metric computation fails."""
    pass

class PlotSaveError(ThresholdError):
    """Raised when saving PR curve plot fails."""
    pass

class ReportWriteError(ThresholdError):
    """Raised when writing evaluation report fails."""
    pass

def _validate_inputs(best_proba: np.ndarray, y_test: pd.Series) -> None:
    if not isinstance(best_proba, np.ndarray):
        raise InputTypeError(f"best_proba must be numpy array, got {type(best_proba)}")
    
    if not isinstance(y_test, pd.Series):
        raise InputTypeError(f"y_test must be pandas Series, got {type(y_test)}")
    
    if best_proba.size == 0:
        raise EmptyInputError("best_proba array is empty")
    
    if y_test.empty:
        raise EmptyInputError("y_test Series is empty")
    
    if best_proba.shape[0] != y_test.shape[0]:
        raise ShapeMismatchError(
            f"best_proba ({best_proba.shape[0]}) and y_test ({y_test.shape[0]}) shape mismatch"
        )
    
    if not np.all(np.isfinite(best_proba)):
        raise InvalidValuesError("best_proba contains NaN or Inf values")
    
    if not np.all(np.isfinite(y_test.values)):
        raise InvalidValuesError("y_test contains NaN or Inf values")
    
    unique_labels = np.unique(y_test)
    if not set(unique_labels).issubset({0, 1}):
        raise InvalidValuesError(f"y_test must contain only 0 and 1, found {unique_labels}")

def _compute_metrics_for_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    threshold: float,
    cost_config: CostConfig
) -> Tuple[float, float, int, int]:
    preds = (proba >= threshold).astype(int)
    
    f2 = fbeta_score(y_true, preds, beta=2.0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    
    expected_cost = cost_config.fn_cost * fn + cost_config.fp_cost * fp
    
    return f2, expected_cost, fn, fp

def _find_optimal_thresholds(
    y_test: pd.Series,
    best_proba: np.ndarray,
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    cost_config: CostConfig
) -> Tuple[float, float, float, float, List[float], List[float], List[float], List[int], List[int]]:
    f2_scores = []
    costs = []
    fn_list = []
    fp_list = []
    
    for t in thresholds:
        f2, cost, fn, fp = _compute_metrics_for_threshold(y_test, best_proba, t, cost_config)
        f2_scores.append(f2)
        costs.append(cost)
        fn_list.append(fn)
        fp_list.append(fp)
    
    f2_scores = np.array(f2_scores)
    costs = np.array(costs)
    
    t_f2_idx = np.argmax(f2_scores)
    t_f2 = thresholds[t_f2_idx]
    f2_at_t_f2 = f2_scores[t_f2_idx]
    
    t_cost_idx = np.argmin(costs)
    t_cost = thresholds[t_cost_idx]
    cost_at_t_cost = costs[t_cost_idx]
    
    return t_f2, t_cost, f2_at_t_f2, cost_at_t_cost, f2_scores.tolist(), costs.tolist(), thresholds.tolist(), fn_list, fp_list

def _check_threshold_agreement(t_f2: float, t_cost: float, config: ThresholdConfig) -> bool:
    return abs(t_f2 - t_cost) <= config.agreement_tolerance

def _compute_final_metrics(
    y_test: pd.Series,
    best_proba: np.ndarray,
    t_f2: float,
    cost_config: CostConfig,
    config: ThresholdConfig
) -> Tuple[int, int, int, int, float, float, float, float]:
    preds = (best_proba >= t_f2).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2 = fbeta_score(y_test, preds, beta=2.0)
    
    expected_cost_per_10k = (cost_config.fn_cost * fn + cost_config.fp_cost * fp) * (config.scale_transactions / len(y_test))
    
    return tn, fp, fn, tp, precision, recall, f2, expected_cost_per_10k

def _generate_pr_curve(
    recalls: np.ndarray,
    precisions: np.ndarray,
    t_f2: float,
    t_cost: float,
    thresholds: np.ndarray,
    config: ThresholdConfig
) -> None:
    try:
        plt.figure(figsize=(10, 8))
        
        plt.plot(recalls, precisions, 'b-', linewidth=2, label='PR Curve')
        
        t_f2_idx = np.argmin(np.abs(thresholds - t_f2))
        t_cost_idx = np.argmin(np.abs(thresholds - t_cost))
        
        plt.axvline(x=recalls[t_f2_idx], color='g', linestyle='--', linewidth=2, label=f'F2-optimal (t={t_f2:.3f})')
        
        plt.axvline(x=recalls[t_cost_idx], color='r', linestyle='--', linewidth=2, label=f'Cost-optimal (t={t_cost:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve with Optimal Thresholds', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plot_path = Path(config.pr_curve_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        raise PlotSaveError(f"Failed to generate PR curve: {str(e)}") from e

def _write_evaluation_report(
    config: ThresholdConfig,
    cost_config: CostConfig,
    t_f2: float,
    t_cost: float,
    f2_at_t_f2: float,
    cost_at_t_cost: float,
    agreement: bool,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    precision: float,
    recall: float,
    f2: float,
    prauc: float,
    expected_cost_per_10k: float
) -> None:
    try:
        report_path = Path(config.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("THRESHOLD OPTIMIZATION EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("COST CONFIGURATION:\n")
            f.write(f"  FN Cost: {cost_config.fn_cost} (missed fraud)\n")
            f.write(f"  FP Cost: {cost_config.fp_cost} (false alarm)\n")
            f.write(f"  Note: {cost_config.comment}\n\n")
            
            f.write("OPTIMAL THRESHOLDS:\n")
            f.write(f"  F2-optimal threshold: {t_f2:.6f} (F2={f2_at_t_f2:.6f})\n")
            f.write(f"  Cost-optimal threshold: {t_cost:.6f} (Expected Cost={cost_at_t_cost:.2f})\n")
            f.write(f"  Thresholds within {config.agreement_tolerance}: {agreement}\n\n")
            
            f.write("FINAL METRICS (at F2-optimal threshold):\n")
            f.write(f"  Selected threshold: {t_f2:.6f}\n")
            f.write(f"  PR-AUC: {prauc:.6f}\n")
            f.write(f"  Precision: {precision:.6f}\n")
            f.write(f"  Recall: {recall:.6f}\n")
            f.write(f"  F2-score: {f2:.6f}\n\n")
            
            f.write("CONFUSION MATRIX:\n")
            f.write(f"  True Negatives: {tn}\n")
            f.write(f"  False Positives: {fp}\n")
            f.write(f"  False Negatives: {fn}\n")
            f.write(f"  True Positives: {tp}\n\n")
            
            f.write(f"Expected cost per {config.scale_transactions:,} transactions: {expected_cost_per_10k:.2f}\n")
            
    except Exception as e:
        raise ReportWriteError(f"Failed to write evaluation report: {str(e)}") from e

def _write_threshold_file(t_f2: float, config: ThresholdConfig) -> None:
    try:
        threshold_path = Path(config.threshold_path)
        threshold_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(threshold_path, 'w') as f:
            f.write(f"{t_f2:.6f}\n")
            
    except Exception as e:
        raise ReportWriteError(f"Failed to write threshold file: {str(e)}") from e

def optimize_threshold(
    best_proba: np.ndarray,
    y_test: pd.Series,
    prauc: float
) -> float:
    _validate_inputs(best_proba, y_test)
    
    cost_config = CostConfig()
    config = ThresholdConfig()
    
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)
        
        if len(thresholds) == 0:
            raise MetricComputationError("No thresholds generated from precision_recall_curve")
        
        global thresholds_var
        thresholds_var = thresholds
        
    except Exception as e:
        raise MetricComputationError(f"Failed to compute precision-recall curve: {str(e)}") from e
    
    t_f2, t_cost, f2_at_t_f2, cost_at_t_cost, f2_scores, costs, thresh_list, fn_list, fp_list = _find_optimal_thresholds(
        y_test, best_proba, precisions, recalls, thresholds, cost_config
    )
    
    agreement = _check_threshold_agreement(t_f2, t_cost, config)
    
    print(f"F2-optimal threshold: {t_f2:.6f} (F2={f2_at_t_f2:.6f})")
    print(f"Cost-optimal threshold: {t_cost:.6f} (Expected Cost={cost_at_t_cost:.2f})")
    print(f"Thresholds within {config.agreement_tolerance}: {agreement}")
    
    tn, fp, fn, tp, precision, recall, f2, expected_cost_per_10k = _compute_final_metrics(
        y_test, best_proba, t_f2, cost_config, config
    )
    
    _generate_pr_curve(recalls, precisions, t_f2, t_cost, thresholds, config)
    
    _write_evaluation_report(
        config, cost_config, t_f2, t_cost, f2_at_t_f2, cost_at_t_cost, agreement,
        tn, fp, fn, tp, precision, recall, f2, prauc, expected_cost_per_10k
    )
    
    _write_threshold_file(t_f2, config)
    
    return t_f2

if __name__ == "__main__":
    import sys
    try:
        print("This module is designed to be imported, not run directly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)