"""
Ridge regression training module for time-series forecasting.

Purpose:
Train a Ridge regression model with feature scaling.
Returns fitted scaler and model for later use in prediction.

This module intentionally avoids:
- Data splitting
- Cross-validation
- Model persistence
- Evaluation metrics
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def train_ridge_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    random_state: Optional[int] = 42,
    copy_X: bool = True
) -> Tuple[StandardScaler, Ridge]:
    """
    Train a Ridge regression model with feature scaling.

    Parameters
    ----------
    X_train : np.ndarray
        Training features matrix of shape (n_samples, n_features).
    y_train : np.ndarray
        Training target vector of shape (n_samples,).
    alpha : float
        Regularization strength; must be positive.
    fit_intercept : bool
        Whether to fit intercept for Ridge model.
    random_state : int, optional
        Random seed for reproducibility.
    copy_X : bool
        Whether to copy X before scaling.

    Returns
    -------
    Tuple[StandardScaler, Ridge]
        - Fitted StandardScaler (for transforming future data)
        - Trained Ridge model

    Raises
    ------
    ValueError
        If input arrays have invalid shapes or contain invalid values.
    """
    # Input validation
    if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
        raise ValueError("X_train and y_train must be numpy arrays")
    
    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X_train.shape}")
    
    if y_train.ndim != 1:
        raise ValueError(f"y_train must be 1D, got shape {y_train.shape}")
    
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train and y_train have inconsistent samples: "
            f"{X_train.shape[0]} vs {y_train.shape[0]}"
        )
    
    if X_train.shape[0] == 0:
        raise ValueError("Training data is empty")
    
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    
    # Check for NaN or infinite values
    if not np.isfinite(X_train).all():
        raise ValueError("X_train contains NaN or infinite values")
    
    if not np.isfinite(y_train).all():
        raise ValueError("y_train contains NaN or infinite values")
    
    # Initialize scaler and model
    scaler = StandardScaler(copy=copy_X)
    model = Ridge(
        alpha=alpha,
        fit_intercept=fit_intercept,
        random_state=random_state,
        copy_X=copy_X
    )
    
    # Fit scaler and transform features
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    return scaler, model


# Minimal usage example
if __name__ == "__main__":
    # Create sample training data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X_train = np.random.randn(n_samples, n_features)
    # Create synthetic target with linear relationship + noise
    true_coef = np.random.randn(n_features)
    y_train = X_train @ true_coef + np.random.randn(n_samples) * 0.1
    
    print(f"Training data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    
    # Train model
    try:
        scaler, model = train_ridge_model(
            X_train=X_train,
            y_train=y_train,
            alpha=0.5,
            random_state=42
        )
        
        print(f"\nModel trained successfully:")
        print(f"Ridge coefficients: {model.coef_}")
        print(f"Ridge intercept: {model.intercept_:.4f}")
        print(f"Number of features: {model.n_features_in_}")
        
        # Quick validation on training set
        X_train_scaled = scaler.transform(X_train)
        y_pred = model.predict(X_train_scaled)
        mse = np.mean((y_train - y_pred) ** 2)
        print(f"Training MSE: {mse:.6f}")
        
    except ValueError as e:
        print(f"Error training model: {e}")