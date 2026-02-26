import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

@dataclass(frozen=True)
class ErrorAnalysisConfig:
    amount_column_names: Tuple[str, ...] = ("log_amount", "TransactionAmt", "amount")
    hour_column: str = "hour_of_day"
    top_n_features: int = 10
    output_path: str = "artifacts/error_analysis.txt"

class ErrorAnalysisError(Exception):
    """Base exception for error analysis failures."""
    pass

class InputTypeError(ErrorAnalysisError):
    """Raised when input types are invalid."""
    pass

class EmptyDataError(ErrorAnalysisError):
    """Raised when input data is empty."""
    pass

class ShapeMismatchError(ErrorAnalysisError):
    """Raised when inputs have mismatched shapes."""
    pass

class InvalidValuesError(ErrorAnalysisError):
    """Raised when input contains invalid values."""
    pass

class ModelTypeError(ErrorAnalysisError):
    """Raised when model type is unsupported."""
    pass

class FeatureImportanceError(ErrorAnalysisError):
    """Raised when feature importance extraction fails."""
    pass

class ReportWriteError(ErrorAnalysisError):
    """Raised when writing analysis report fails."""
    pass

def _validate_inputs(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_proba: np.ndarray,
    threshold: float
) -> None:
    if not isinstance(X_test, pd.DataFrame):
        raise InputTypeError(f"X_test must be pandas DataFrame, got {type(X_test)}")
    
    if not isinstance(y_test, pd.Series):
        raise InputTypeError(f"y_test must be pandas Series, got {type(y_test)}")
    
    if not isinstance(best_proba, np.ndarray):
        raise InputTypeError(f"best_proba must be numpy array, got {type(best_proba)}")
    
    if not isinstance(threshold, (int, float)):
        raise InputTypeError(f"threshold must be float, got {type(threshold)}")
    
    if X_test.empty:
        raise EmptyDataError("X_test DataFrame is empty")
    
    if y_test.empty:
        raise EmptyDataError("y_test Series is empty")
    
    if best_proba.size == 0:
        raise EmptyDataError("best_proba array is empty")
    
    if not (0.0 <= threshold <= 1.0):
        raise InvalidValuesError(f"threshold must be in [0,1], got {threshold}")
    
    if len(X_test) != len(y_test) or len(X_test) != len(best_proba):
        raise ShapeMismatchError(
            f"Shape mismatch: X_test {len(X_test)}, y_test {len(y_test)}, best_proba {len(best_proba)}"
        )
    
    if not np.all(np.isfinite(best_proba)):
        raise InvalidValuesError("best_proba contains NaN or Inf values")
    
    unique_labels = np.unique(y_test)
    if not set(unique_labels).issubset({0, 1}):
        raise InvalidValuesError(f"y_test must contain only 0 and 1, found {unique_labels}")

def _compute_predictions(best_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (best_proba >= threshold).astype(int)

def _get_error_masks(
    y_test: pd.Series,
    preds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fn_mask = (y_test.values == 1) & (preds == 0)
    fp_mask = (y_test.values == 0) & (preds == 1)
    tp_mask = (y_test.values == 1) & (preds == 1)
    return fn_mask, fp_mask, tp_mask

def _find_amount_column(X_test: pd.DataFrame, config: ErrorAnalysisConfig) -> str:
    for col_name in config.amount_column_names:
        if col_name in X_test.columns:
            return col_name
    return ""

def _analyze_amount_distribution(
    X_test: pd.DataFrame,
    fn_mask: np.ndarray,
    tp_mask: np.ndarray,
    config: ErrorAnalysisConfig
) -> List[str]:
    amount_col = _find_amount_column(X_test, config)
    if not amount_col:
        return ["No amount column found for analysis"]
    
    fn_amounts = X_test.loc[fn_mask, amount_col]
    tp_amounts = X_test.loc[tp_mask, amount_col]
    
    if len(fn_amounts) == 0:
        fn_stats = ["No false negatives to analyze"]
    else:
        fn_stats = [
            f"FN count: {len(fn_amounts)}",
            f"FN mean: {fn_amounts.mean():.4f}",
            f"FN median: {fn_amounts.median():.4f}",
            f"FN std: {fn_amounts.std():.4f}"
        ]
    
    if len(tp_amounts) == 0:
        tp_stats = ["No true positives to analyze"]
    else:
        tp_stats = [
            f"TP count: {len(tp_amounts)}",
            f"TP mean: {tp_amounts.mean():.4f}",
            f"TP median: {tp_amounts.median():.4f}",
            f"TP std: {tp_amounts.std():.4f}"
        ]
    
    return [
        f"Amount column analyzed: {amount_col}",
        "\nFalse Negatives (Missed Fraud):",
        *fn_stats,
        "\nTrue Positives (Detected Fraud):",
        *tp_stats
    ]

def _analyze_hour_distribution(
    X_test: pd.DataFrame,
    fn_mask: np.ndarray,
    tp_mask: np.ndarray,
    config: ErrorAnalysisConfig
) -> List[str]:
    if config.hour_column not in X_test.columns:
        return ["hour_of_day column not found for analysis"]
    
    fn_hours = X_test.loc[fn_mask, config.hour_column]
    tp_hours = X_test.loc[tp_mask, config.hour_column]
    
    lines = [f"\nHour of Day Distribution (counts):"]
    
    if len(fn_hours) > 0:
        fn_dist = fn_hours.value_counts().sort_index()
        lines.append("FN by hour:")
        for hour in range(24):
            count = fn_dist.get(hour, 0)
            lines.append(f"  Hour {hour:2d}: {count}")
    else:
        lines.append("FN by hour: No false negatives")
    
    if len(tp_hours) > 0:
        tp_dist = tp_hours.value_counts().sort_index()
        lines.append("TP by hour:")
        for hour in range(24):
            count = tp_dist.get(hour, 0)
            lines.append(f"  Hour {hour:2d}: {count}")
    else:
        lines.append("TP by hour: No true positives")
    
    return lines

def _extract_feature_importances(
    model: Union[RandomForestClassifier, LogisticRegression],
    X_test: pd.DataFrame,
    config: ErrorAnalysisConfig
) -> List[Tuple[str, float]]:
    feature_names = X_test.columns.tolist()
    
    if isinstance(model, RandomForestClassifier):
        if not hasattr(model, 'feature_importances_'):
            raise FeatureImportanceError("RandomForest model has no feature_importances_ attribute")
        importances = model.feature_importances_
        
    elif isinstance(model, LogisticRegression):
        if not hasattr(model, 'coef_'):
            raise FeatureImportanceError("LogisticRegression model has no coef_ attribute")
        importances = np.abs(model.coef_[0])
        
    else:
        raise ModelTypeError(f"Unsupported model type: {type(model)}")
    
    if len(importances) != len(feature_names):
        raise FeatureImportanceError(
            f"Feature importance length mismatch: {len(importances)} vs {len(feature_names)} features"
        )
    
    feature_importance_pairs = list(zip(feature_names, importances))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return feature_importance_pairs[:config.top_n_features]

def _write_analysis_report(
    config: ErrorAnalysisConfig,
    threshold: float,
    fn_count: int,
    fp_count: int,
    tp_count: int,
    tn_count: int,
    amount_lines: List[str],
    hour_lines: List[str],
    feature_lines: List[str]
) -> None:
    try:
        report_path = Path(config.output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Threshold used: {threshold:.6f}\n\n")
            
            f.write("CONFUSION MATRIX SUMMARY:\n")
            f.write(f"  True Negatives: {tn_count}\n")
            f.write(f"  False Positives: {fp_count}\n")
            f.write(f"  False Negatives: {fn_count}\n")
            f.write(f"  True Positives: {tp_count}\n\n")
            
            f.write("AMOUNT DISTRIBUTION ANALYSIS:\n")
            for line in amount_lines:
                f.write(f"  {line}\n")
            f.write("\n")
            
            f.write("HOURLY DISTRIBUTION ANALYSIS:\n")
            for line in hour_lines:
                f.write(f"  {line}\n")
            f.write("\n")
            
            f.write(f"TOP {config.top_n_features} FEATURE IMPORTANCES:\n")
            for line in feature_lines:
                f.write(f"  {line}\n")
            
    except Exception as e:
        raise ReportWriteError(f"Failed to write error analysis report: {str(e)}") from e

def analyze_errors(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_proba: np.ndarray,
    threshold: float,
    model: Union[RandomForestClassifier, LogisticRegression]
) -> None:
    _validate_inputs(X_test, y_test, best_proba, threshold)
    
    config = ErrorAnalysisConfig()
    
    preds = _compute_predictions(best_proba, threshold)
    fn_mask, fp_mask, tp_mask = _error_masks = _get_error_masks(y_test, preds)
    
    fn_count = np.sum(fn_mask)
    fp_count = np.sum(fp_mask)
    tp_count = np.sum(tp_mask)
    tn_count = len(y_test) - (fn_count + fp_count + tp_count)
    
    print(f"Error Analysis Summary:")
    print(f"  False Negatives (missed fraud): {fn_count}")
    print(f"  False Positives (false alarms): {fp_count}")
    print(f"  True Positives: {tp_count}")
    print(f"  True Negatives: {tn_count}")
    
    amount_lines = _analyze_amount_distribution(X_test, fn_mask, tp_mask, config)
    for line in amount_lines:
        if line.startswith("FN") or line.startswith("TP"):
            print(f"  {line}")
    
    hour_lines = _analyze_hour_distribution(X_test, fn_mask, tp_mask, config)
    
    try:
        top_features = _extract_feature_importances(model, X_test, config)
        feature_lines = [f"{name}: {importance:.6f}" for name, importance in top_features]
        
        print(f"\nTop {config.top_n_features} Feature Importances:")
        for name, importance in top_features:
            print(f"  {name}: {importance:.6f}")
            
    except (FeatureImportanceError, ModelTypeError) as e:
        feature_lines = [f"Could not extract feature importances: {str(e)}"]
        print(f"  Warning: {feature_lines[0]}")
    
    _write_analysis_report(
        config,
        threshold,
        fn_count,
        fp_count,
        tp_count,
        tn_count,
        amount_lines,
        hour_lines,
        feature_lines
    )

if __name__ == "__main__":
    import sys
    try:
        print("This module is designed to be imported, not run directly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)