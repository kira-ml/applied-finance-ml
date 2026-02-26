import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42
    min_test_fraud_count: int = 100

class SplitError(Exception):
    """Base exception for data splitting failures."""
    pass

class InputTypeError(SplitError):
    """Raised when input types are invalid."""
    pass

class EmptyDataError(SplitError):
    """Raised when input data is empty."""
    pass

class ShapeMismatchError(SplitError):
    """Raised when X and y have mismatched shapes."""
    pass

class StratificationError(SplitError):
    """Raised when stratification fails due to class imbalance."""
    pass

class InsufficientTestFraudError(SplitError):
    """Raised when test set contains insufficient fraud samples."""
    pass

def _validate_inputs(X: pd.DataFrame, y: pd.Series) -> None:
    if not isinstance(X, pd.DataFrame):
        raise InputTypeError(f"X must be pandas DataFrame, got {type(X)}")
    
    if not isinstance(y, pd.Series):
        raise InputTypeError(f"y must be pandas Series, got {type(y)}")
    
    if X.empty:
        raise EmptyDataError("X DataFrame is empty")
    
    if y.empty:
        raise EmptyDataError("y Series is empty")
    
    if len(X) != len(y):
        raise ShapeMismatchError(
            f"X and y have mismatched shapes: X {len(X)} rows, y {len(y)} rows"
        )

def _validate_stratification(y: pd.Series) -> None:
    class_counts = y.value_counts()
    min_class_count = class_counts.min()
    
    if min_class_count < 2:
        raise StratificationError(
            f"Cannot stratify: smallest class has only {min_class_count} sample(s)"
        )

def _print_split_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> None:
    train_fraud_count = y_train.sum()
    test_fraud_count = y_test.sum()
    train_fraud_rate = train_fraud_count / len(y_train)
    test_fraud_rate = test_fraud_count / len(y_test)
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Train fraud count: {train_fraud_count}")
    print(f"Test fraud count: {test_fraud_count}")
    print(f"Train fraud rate: {train_fraud_rate:.6f}")
    print(f"Test fraud rate: {test_fraud_rate:.6f}")

def split_data(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    _validate_inputs(X, y)
    _validate_stratification(y)
    
    config = SplitConfig()
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.test_size,
            stratify=y,
            random_state=config.random_state
        )
    except ValueError as e:
        raise SplitError(f"train_test_split failed: {str(e)}") from e
    
    test_fraud_count = y_test.sum()
    if test_fraud_count < config.min_test_fraud_count:
        raise InsufficientTestFraudError(
            f"Test set contains only {test_fraud_count} fraud samples, "
            f"minimum required is {config.min_test_fraud_count}"
        )
    
    _print_split_summary(X_train, X_test, y_train, y_test)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    import sys
    try:
        print("This module is designed to be imported, not run directly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)