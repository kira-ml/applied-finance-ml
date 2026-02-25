import pickle
import logging
from dataclasses import dataclass
from typing import Tuple, Union, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin

# Configure module-level logger
logger = logging.getLogger(__name__)


class TrainerError(Exception):
    """Base exception for trainer module."""
    pass


class InvalidModelTypeError(TrainerError):
    """Raised when an invalid model type is specified."""
    pass


class InvalidInputDataError(TrainerError):
    """Raised when input data fails validation."""
    pass


class ModelTrainingError(TrainerError):
    """Raised when model training fails."""
    pass


class ModelPersistenceError(TrainerError):
    """Raised when model saving fails."""
    pass


@dataclass(frozen=True)
class TrainingResult:
    """Immutable container for training results."""
    validation_probabilities: np.ndarray
    validation_true_labels: np.ndarray


class Trainer:
    """
    A minimal trainer for binary classification models.
    
    Supports LogisticRegression (with class_weight='balanced') and RandomForestClassifier.
    Fits model on training data and returns validation probabilities.
    """

    VALID_MODEL_TYPES = frozenset({"logistic_regression", "random_forest"})
    
    def __init__(self, model_type: str, random_state: int = 42) -> None:
        """
        Initialize trainer with specified model type.
        
        Args:
            model_type: Type of model to use - 'logistic_regression' or 'random_forest'
            random_state: Random seed for reproducibility
            
        Raises:
            InvalidModelTypeError: If model_type is not in VALID_MODEL_TYPES
        """
        self._validate_model_type(model_type)
        self._model_type = model_type
        self._random_state = random_state
        self._model: Union[LogisticRegression, RandomForestClassifier, None] = None
        
    def _validate_model_type(self, model_type: str) -> None:
        """Validate model type against allowed set."""
        if model_type not in self.VALID_MODEL_TYPES:
            raise InvalidModelTypeError(
                f"Model type must be one of {sorted(self.VALID_MODEL_TYPES)}, got '{model_type}'"
            )
    
    def _validate_input_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Validate input data arrays.
        
        Raises:
            InvalidInputDataError: If any validation fails
        """
        # Check types
        if not isinstance(X_train, np.ndarray):
            raise InvalidInputDataError(f"X_train must be numpy array, got {type(X_train)}")
        if not isinstance(y_train, np.ndarray):
            raise InvalidInputDataError(f"y_train must be numpy array, got {type(y_train)}")
        if not isinstance(X_val, np.ndarray):
            raise InvalidInputDataError(f"X_val must be numpy array, got {type(X_val)}")
        if not isinstance(y_val, np.ndarray):
            raise InvalidInputDataError(f"y_val must be numpy array, got {type(y_val)}")
        
        # Check non-empty
        if X_train.size == 0:
            raise InvalidInputDataError("X_train cannot be empty")
        if y_train.size == 0:
            raise InvalidInputDataError("y_train cannot be empty")
        if X_val.size == 0:
            raise InvalidInputDataError("X_val cannot be empty")
        if y_val.size == 0:
            raise InvalidInputDataError("y_val cannot be empty")
        
        # Check dimensions
        if X_train.ndim != 2:
            raise InvalidInputDataError(f"X_train must be 2D, got {X_train.ndim}D")
        if y_train.ndim != 1:
            raise InvalidInputDataError(f"y_train must be 1D, got {y_train.ndim}D")
        if X_val.ndim != 2:
            raise InvalidInputDataError(f"X_val must be 2D, got {X_val.ndim}D")
        if y_val.ndim != 1:
            raise InvalidInputDataError(f"y_val must be 1D, got {y_val.ndim}D")
        
        # Check feature dimension consistency
        if X_train.shape[1] != X_val.shape[1]:
            raise InvalidInputDataError(
                f"Feature dimension mismatch: X_train has {X_train.shape[1]} features, "
                f"X_val has {X_val.shape[1]} features"
            )
        
        # Check sample count consistency
        if X_train.shape[0] != y_train.shape[0]:
            raise InvalidInputDataError(
                f"Training samples mismatch: X_train has {X_train.shape[0]} samples, "
                f"y_train has {y_train.shape[0]} samples"
            )
        if X_val.shape[0] != y_val.shape[0]:
            raise InvalidInputDataError(
                f"Validation samples mismatch: X_val has {X_val.shape[0]} samples, "
                f"y_val has {y_val.shape[0]} samples"
            )
        
        # Check for NaN or Inf in features
        if np.any(np.isnan(X_train)):
            raise InvalidInputDataError("X_train contains NaN values")
        if np.any(np.isinf(X_train)):
            raise InvalidInputDataError("X_train contains infinite values")
        if np.any(np.isnan(X_val)):
            raise InvalidInputDataError("X_val contains NaN values")
        if np.any(np.isinf(X_val)):
            raise InvalidInputDataError("X_val contains infinite values")
        
        # Check labels are binary (0/1)
        unique_train = np.unique(y_train)
        if not np.array_equal(unique_train, np.array([0, 1])) and not np.array_equal(unique_train, np.array([0])) and not np.array_equal(unique_train, np.array([1])):
            raise InvalidInputDataError(f"y_train must contain only 0 and 1, got {unique_train}")
        
        unique_val = np.unique(y_val)
        if not np.array_equal(unique_val, np.array([0, 1])) and not np.array_equal(unique_val, np.array([0])) and not np.array_equal(unique_val, np.array([1])):
            raise InvalidInputDataError(f"y_val must contain only 0 and 1, got {unique_val}")
    
    def _create_model(self) -> ClassifierMixin:
        """Create and return configured model instance."""
        if self._model_type == "logistic_regression":
            return LogisticRegression(
                class_weight='balanced',
                random_state=self._random_state,
                max_iter=1000  # Explicit bound to guarantee termination
            )
        else:  # random_forest
            return RandomForestClassifier(
                class_weight='balanced',
                random_state=self._random_state,
                n_estimators=100,  # Explicit bound
                max_depth=10  # Explicit bound to guarantee termination
            )
    
    def fit_and_validate(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> TrainingResult:
        """
        Fit model on training data and return validation probabilities.
        
        Args:
            X_train: Training features (2D numpy array)
            y_train: Training labels (1D numpy array of 0/1)
            X_val: Validation features (2D numpy array)
            y_val: Validation labels (1D numpy array of 0/1)
            
        Returns:
            TrainingResult containing validation probabilities and true labels
            
        Raises:
            InvalidInputDataError: If input validation fails
            ModelTrainingError: If model fitting fails
        """
        # Validate inputs
        self._validate_input_data(X_train, y_train, X_val, y_val)
        
        # Create and fit model
        try:
            self._model = self._create_model()
            self._model.fit(X_train, y_train)
            logger.info(f"Model training completed: {self._model_type}")
        except Exception as e:
            raise ModelTrainingError(f"Model fitting failed: {str(e)}") from e
        
        # Get validation probabilities
        try:
            probabilities = self._model.predict_proba(X_val)
            # For binary classification, take probability of positive class (class 1)
            positive_class_probs = probabilities[:, 1]
        except Exception as e:
            raise ModelTrainingError(f"Probability prediction failed: {str(e)}") from e
        
        # Ensure deterministic ordering
        # probabilities are already aligned with X_val order
        
        # Return immutable result
        return TrainingResult(
            validation_probabilities=positive_class_probs.copy(),  # Ensure we own the data
            validation_true_labels=y_val.copy()
        )
    
    def save_model(self, filepath: str = "model.pkl") -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path where model will be saved
            
        Raises:
            ModelPersistenceError: If model is not trained or saving fails
        """
        if self._model is None:
            raise ModelPersistenceError("No trained model to save. Call fit_and_validate first.")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self._model, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            raise ModelPersistenceError(f"Failed to save model to {filepath}: {str(e)}") from e
    
    @property
    def model(self) -> ClassifierMixin:
        """
        Get the trained model.
        
        Returns:
            The trained scikit-learn classifier
        
        Raises:
            ModelTrainingError: If model hasn't been trained yet
        """
        if self._model is None:
            raise ModelTrainingError("Model not trained yet. Call fit_and_validate first.")
        return self._model