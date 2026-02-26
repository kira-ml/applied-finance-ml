import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

@dataclass(frozen=True)
class TrainConfig:
    lr_c: float = 0.1
    lr_max_iter: int = 1000
    lr_solver: str = 'lbfgs'
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_random_state: int = 42
    rf_n_jobs: int = -1
    model_path: str = "artifacts/model.pkl"
    scaler_path: str = "artifacts/scaler.pkl"
    log_path: str = "artifacts/model_selection_log.txt"

class TrainError(Exception):
    """Base exception for training failures."""
    pass

class InputTypeError(TrainError):
    """Raised when input types are invalid."""
    pass

class EmptyDataError(TrainError):
    """Raised when input data is empty."""
    pass

class ShapeMismatchError(TrainError):
    """Raised when X and y have mismatched shapes."""
    pass

class ModelFitError(TrainError):
    """Raised when model fitting fails."""
    pass

class ScalerFitError(TrainError):
    """Raised when scaler fitting fails."""
    pass

class ModelSaveError(TrainError):
    """Raised when saving model artifacts fails."""
    pass

def _validate_inputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> None:
    if not isinstance(X_train, pd.DataFrame):
        raise InputTypeError(f"X_train must be pandas DataFrame, got {type(X_train)}")
    if not isinstance(X_test, pd.DataFrame):
        raise InputTypeError(f"X_test must be pandas DataFrame, got {type(X_test)}")
    if not isinstance(y_train, pd.Series):
        raise InputTypeError(f"y_train must be pandas Series, got {type(y_train)}")
    if not isinstance(y_test, pd.Series):
        raise InputTypeError(f"y_test must be pandas Series, got {type(y_test)}")
    
    if X_train.empty:
        raise EmptyDataError("X_train is empty")
    if X_test.empty:
        raise EmptyDataError("X_test is empty")
    if y_train.empty:
        raise EmptyDataError("y_train is empty")
    if y_test.empty:
        raise EmptyDataError("y_test is empty")
    
    if len(X_train) != len(y_train):
        raise ShapeMismatchError(
            f"X_train ({len(X_train)}) and y_train ({len(y_train)}) shape mismatch"
        )
    if len(X_test) != len(y_test):
        raise ShapeMismatchError(
            f"X_test ({len(X_test)}) and y_test ({len(y_test)}) shape mismatch"
        )

def _train_logistic_regression(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: TrainConfig
) -> Tuple[float, LogisticRegression, StandardScaler]:
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        raise ScalerFitError(f"Failed to fit/transform scaler: {str(e)}") from e
    
    try:
        lr = LogisticRegression(
            C=config.lr_c,
            class_weight='balanced',
            max_iter=config.lr_max_iter,
            solver=config.lr_solver,
            random_state=42
        )
        lr.fit(X_train_scaled, y_train)
    except Exception as e:
        raise ModelFitError(f"Logistic Regression failed to fit: {str(e)}") from e
    
    try:
        lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
        lr_prauc = average_precision_score(y_test, lr_proba)
    except Exception as e:
        raise TrainError(f"Failed to compute LR predictions/PR-AUC: {str(e)}") from e
    
    return lr_prauc, lr, scaler

def _train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: TrainConfig
) -> Tuple[float, RandomForestClassifier]:
    try:
        rf = RandomForestClassifier(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            class_weight='balanced',
            n_jobs=config.rf_n_jobs,
            random_state=config.rf_random_state
        )
        rf.fit(X_train, y_train)
    except Exception as e:
        raise ModelFitError(f"Random Forest failed to fit: {str(e)}") from e
    
    try:
        rf_proba = rf.predict_proba(X_test)[:, 1]
        rf_prauc = average_precision_score(y_test, rf_proba)
    except Exception as e:
        raise TrainError(f"Failed to compute RF predictions/PR-AUC: {str(e)}") from e
    
    return rf_prauc, rf

def _write_selection_log(
    config: TrainConfig,
    lr_prauc: float,
    rf_prauc: float,
    winner: str
) -> None:
    try:
        log_path = Path(config.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as f:
            if winner == "LR":
                f.write(f"LR selected, LR_PRAUC={lr_prauc:.6f}, RF_PRAUC={rf_prauc:.6f}\n")
            else:
                f.write(f"RF selected\n")
    except Exception as e:
        raise ModelSaveError(f"Failed to write selection log: {str(e)}") from e

def _save_artifacts(
    config: TrainConfig,
    winner: str,
    model: Union[LogisticRegression, RandomForestClassifier],
    scaler: StandardScaler = None
) -> None:
    try:
        model_path = Path(config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        
        if winner == "LR" and scaler is not None:
            scaler_path = Path(config.scaler_path)
            joblib.dump(scaler, scaler_path)
    except Exception as e:
        raise ModelSaveError(f"Failed to save model artifacts: {str(e)}") from e

def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[Union[LogisticRegression, RandomForestClassifier], np.ndarray, pd.Series]:
    _validate_inputs(X_train, X_test, y_train, y_test)
    
    config = TrainConfig()
    
    lr_prauc, lr, scaler = _train_logistic_regression(X_train, X_test, y_train, y_test, config)
    rf_prauc, rf = _train_random_forest(X_train, X_test, y_train, y_test, config)
    
    print(f"LR PR-AUC: {lr_prauc:.6f}")
    print(f"RF PR-AUC: {rf_prauc:.6f}")
    
    if lr_prauc >= rf_prauc:
        winner = "LR"
        winning_model = lr
        _write_selection_log(config, lr_prauc, rf_prauc, winner)
        _save_artifacts(config, winner, winning_model, scaler)
        
        X_test_scaled = scaler.transform(X_test)
        best_proba = winning_model.predict_proba(X_test_scaled)[:, 1]
    else:
        winner = "RF"
        winning_model = rf
        _write_selection_log(config, lr_prauc, rf_prauc, winner)
        _save_artifacts(config, winner, winning_model)
        
        best_proba = winning_model.predict_proba(X_test)[:, 1]
    
    return winning_model, best_proba, y_test

if __name__ == "__main__":
    import sys
    try:
        print("This module is designed to be imported, not run directly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)