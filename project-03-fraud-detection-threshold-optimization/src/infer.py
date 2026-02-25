import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, Tuple, List, Set

import numpy as np
from numpy.typing import NDArray


class FraudInferenceError(Exception):
    """Base exception for fraud inference errors."""
    pass


class ModelLoadError(FraudInferenceError):
    """Raised when model files cannot be loaded."""
    pass


class PreprocessingError(FraudInferenceError):
    """Raised when input preprocessing fails."""
    pass


class InvalidInputError(FraudInferenceError):
    """Raised when input dictionary fails validation."""
    pass


@dataclass(frozen=True)
class ModelArtifacts:
    """Container for all loaded model artifacts."""
    preprocessor: object
    model: object
    threshold: float


@dataclass(frozen=True)
class PredictionResult:
    """Immutable prediction result container."""
    prediction: int
    probability: float
    threshold: float


class FraudInferenceEngine:
    """Inference engine for fraud detection with loaded artifacts."""

    # Expected numeric features based on standard preprocessing
    _NUMERIC_FEATURES: Set[str] = frozenset({
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest'
    })

    # Expected categorical features
    _CATEGORICAL_FEATURES: Set[str] = frozenset({
        'type',
        'nameOrig',
        'nameDest'
    })

    # All required features
    _REQUIRED_FEATURES: Set[str] = _NUMERIC_FEATURES | _CATEGORICAL_FEATURES

    def __init__(self, artifacts_dir: Union[str, Path]) -> None:
        """
        Initialize inference engine by loading artifacts from directory.

        Args:
            artifacts_dir: Path to directory containing preprocessor.pkl,
                          model.pkl, and threshold.txt

        Raises:
            ModelLoadError: If any artifact file is missing or corrupt
            InvalidInputError: If threshold value is invalid
        """
        self._artifacts_dir = Path(artifacts_dir)
        self._artifacts = self._load_artifacts()

    def _load_artifacts(self) -> ModelArtifacts:
        """Load and validate all model artifacts from disk."""
        try:
            # Load preprocessor
            preprocessor_path = self._artifacts_dir / 'preprocessor.pkl'
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)

            # Load model
            model_path = self._artifacts_dir / 'model.pkl'
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            # Load and validate threshold
            threshold_path = self._artifacts_dir / 'threshold.txt'
            with open(threshold_path, 'r') as f:
                threshold_str = f.read().strip()
                try:
                    threshold = float(threshold_str)
                except ValueError as e:
                    raise InvalidInputError(
                        f"Invalid threshold value: {threshold_str}"
                    ) from e

            # Validate threshold range
            if not 0.0 <= threshold <= 1.0:
                raise InvalidInputError(
                    f"Threshold must be in [0, 1], got: {threshold}"
                )

            return ModelArtifacts(
                preprocessor=preprocessor,
                model=model,
                threshold=threshold
            )

        except FileNotFoundError as e:
            raise ModelLoadError(f"Missing artifact file: {e}") from e
        except pickle.UnpicklingError as e:
            raise ModelLoadError(f"Corrupted pickle file: {e}") from e
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading artifacts: {e}") from e

    def _validate_input_dict(self, raw_dict: Dict) -> None:
        """
        Validate input dictionary structure and types.

        Args:
            raw_dict: Input dictionary to validate

        Raises:
            InvalidInputError: For any validation failure
        """
        if not isinstance(raw_dict, dict):
            raise InvalidInputError(f"Input must be dict, got: {type(raw_dict)}")

        # Check for missing features
        provided_features = set(raw_dict.keys())
        missing_features = self._REQUIRED_FEATURES - provided_features
        if missing_features:
            raise InvalidInputError(
                f"Missing required features: {sorted(missing_features)}"
            )

        # Check for extra features (warning only, not error)
        extra_features = provided_features - self._REQUIRED_FEATURES
        if extra_features:
            # Log but continue - we'll ignore extras during preprocessing
            pass

        # Validate numeric features are numeric
        for feat in self._NUMERIC_FEATURES:
            value = raw_dict[feat]
            if not isinstance(value, (int, float)):
                raise InvalidInputError(
                    f"Feature {feat} must be numeric, got: {type(value)}"
                )

            # Check for NaN/Inf
            if isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    raise InvalidInputError(
                        f"Feature {feat} contains NaN or Inf: {value}"
                    )

        # Validate categorical features are strings
        for feat in self._CATEGORICAL_FEATURES:
            value = raw_dict[feat]
            if not isinstance(value, str):
                raise InvalidInputError(
                    f"Feature {feat} must be string, got: {type(value)}"
                )

            # Ensure non-empty strings
            if not value:
                raise InvalidInputError(f"Feature {feat} cannot be empty string")

    def _preprocess_input(self, raw_dict: Dict) -> NDArray:
        """
        Transform input dictionary to model-ready feature array.

        Args:
            raw_dict: Validated input dictionary

        Returns:
            Preprocessed feature array

        Raises:
            PreprocessingError: If preprocessing fails
        """
        try:
            # Create a copy with only required features in consistent order
            # This ensures deterministic preprocessing
            ordered_dict = {k: raw_dict[k] for k in sorted(self._REQUIRED_FEATURES)}

            # Convert to list in consistent order for preprocessing
            feature_values = [ordered_dict[k] for k in sorted(self._REQUIRED_FEATURES)]

            # Apply preprocessor (assumes it handles both numeric and categorical)
            processed = self._artifacts.preprocessor.transform([feature_values])

            # Ensure 2D array
            if processed.ndim != 2:
                raise PreprocessingError(
                    f"Preprocessor returned {processed.ndim}D array, expected 2D"
                )

            return processed

        except Exception as e:
            raise PreprocessingError(f"Preprocessing failed: {e}") from e

    def _predict_proba(self, features: NDArray) -> float:
        """
        Generate probability prediction from features.

        Args:
            features: Preprocessed feature array

        Returns:
            Probability of fraud (positive class)

        Raises:
            PreprocessingError: If prediction fails
        """
        try:
            # Get probability for positive class
            proba = self._artifacts.model.predict_proba(features)[0, 1]

            # Validate probability range
            if not 0.0 <= proba <= 1.0:
                raise PreprocessingError(
                    f"Model returned invalid probability: {proba}"
                )

            return float(proba)

        except Exception as e:
            raise PreprocessingError(f"Prediction failed: {e}") from e

    def predict(self, raw_transaction_dict: Dict) -> PredictionResult:
        """
        Generate fraud prediction for a single transaction.

        Args:
            raw_transaction_dict: Dictionary with transaction features

        Returns:
            PredictionResult containing binary prediction and probability

        Raises:
            InvalidInputError: For input validation failures
            PreprocessingError: For preprocessing or prediction failures
        """
        # Step 1: Validate input
        self._validate_input_dict(raw_transaction_dict)

        # Step 2: Preprocess
        features = self._preprocess_input(raw_transaction_dict)

        # Step 3: Generate probability
        probability = self._predict_proba(features)

        # Step 4: Apply threshold for binary prediction
        prediction = 1 if probability >= self._artifacts.threshold else 0

        # Step 5: Return immutable result
        return PredictionResult(
            prediction=prediction,
            probability=probability,
            threshold=self._artifacts.threshold
        )


def predict_fraud(raw_transaction_dict: Dict) -> Tuple[int, float]:
    """
    Standalone function for fraud prediction using latest artifacts.

    This is the primary public interface. Expects artifacts in './latest/'
    directory containing preprocessor.pkl, model.pkl, and threshold.txt.

    Args:
        raw_transaction_dict: Dictionary with transaction features

    Returns:
        Tuple of (binary_prediction, probability_score)

    Raises:
        FraudInferenceError: For any error during prediction
    """
    # Use './latest' as default artifacts directory
    engine = FraudInferenceEngine(Path('./latest'))
    result = engine.predict(raw_transaction_dict)
    return (result.prediction, result.probability)