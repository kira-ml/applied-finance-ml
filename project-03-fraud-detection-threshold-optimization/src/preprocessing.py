"""Preprocessing module with deterministic transformations and stratification.

This module performs train-test split with stratification, fits transformers on training data only,
and applies transformations to all splits. All fitted components are serialized for inference.

Example:
    result = preprocess_data(df, target_col="is_fraud", test_size=0.2, val_size=0.1)
    if result.error:
        handle_error(result.error)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = result
"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from typing import NamedTuple, Sequence, Literal
from enum import Enum, auto
import logging
import sys

# Configure structured logger
logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class PreprocessingError(Enum):
    """Enumerated failure modes for preprocessing."""

    INVALID_INPUT_TYPE = auto()
    TARGET_COLUMN_MISSING = auto()
    INSUFFICIENT_SAMPLES = auto()
    INVALID_SPLIT_SIZES = auto()
    NO_NUMERIC_FEATURES = auto()
    NO_CATEGORICAL_FEATURES = auto()
    STRATIFICATION_FAILED = auto()
    PIPELINE_FIT_FAILED = auto()
    TRANSFORM_FAILED = auto()
    SERIALIZATION_FAILED = auto()
    UNEXPECTED_ERROR = auto()


class PreprocessResult(NamedTuple):
    """Result of preprocessing operation."""

    X_train: NDArray[np.float64] | None
    y_train: NDArray[np.int64] | None
    X_val: NDArray[np.float64] | None
    y_val: NDArray[np.int64] | None
    X_test: NDArray[np.float64] | None
    y_test: NDArray[np.int64] | None
    error: PreprocessingError | None
    error_message: str | None


class _Preprocessor:
    """Internal preprocessing pipeline manager."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        random_seed: int,
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.random_seed = random_seed
        self.pipeline: ColumnTransformer | None = None

    def _build_pipeline(self) -> ColumnTransformer:
        """Build the preprocessing pipeline."""
        transformers = []

        if self.numeric_features:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", numeric_transformer, self.numeric_features))

        if self.categorical_features:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat", categorical_transformer, self.categorical_features))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> NDArray[np.float64]:
        """Fit pipeline on training data and transform it."""
        self.pipeline = self._build_pipeline()
        try:
            X_train_transformed = self.pipeline.fit_transform(X_train)
            logger.info(
                "Pipeline fitted on training data",
                extra={
                    "numeric_features": len(self.numeric_features),
                    "categorical_features": len(self.categorical_features),
                    "output_shape": X_train_transformed.shape,
                },
            )
            return X_train_transformed.astype(np.float64)
        except Exception as e:
            logger.error("Pipeline fit failed", exc_info=True)
            raise RuntimeError(f"Pipeline fit failed: {str(e)}") from e

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """Transform data using fitted pipeline."""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted")
        try:
            X_transformed = self.pipeline.transform(X)
            return X_transformed.astype(np.float64)
        except Exception as e:
            logger.error("Pipeline transform failed", exc_info=True)
            raise RuntimeError(f"Pipeline transform failed: {str(e)}") from e


class _Validator:
    """Internal validation logic."""

    @staticmethod
    def validate_input(df: pd.DataFrame, target_col: str) -> PreprocessingError | None:
        """Validate input DataFrame."""
        if not isinstance(df, pd.DataFrame):
            return PreprocessingError.INVALID_INPUT_TYPE

        if target_col not in df.columns:
            return PreprocessingError.TARGET_COLUMN_MISSING

        if len(df) < 100:
            return PreprocessingError.INSUFFICIENT_SAMPLES

        return None

    @staticmethod
    def validate_split_sizes(test_size: float, val_size: float) -> PreprocessingError | None:
        """Validate split sizes."""
        if not (0 < test_size < 1):
            return PreprocessingError.INVALID_SPLIT_SIZES
        if not (0 <= val_size < 1):
            return PreprocessingError.INVALID_SPLIT_SIZES
        if test_size + val_size >= 1:
            return PreprocessingError.INVALID_SPLIT_SIZES
        return None

    @staticmethod
    def identify_feature_types(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
        """Identify numeric and categorical features."""
        feature_df = df.drop(columns=[target_col])

        numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

        return numeric_features, categorical_features


def preprocess_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
    serialize_path: str | Path | None = None,
) -> PreprocessResult:
    """Preprocess data with stratification and fitted transformations.

    Args:
        df: Input DataFrame containing features and target.
        target_col: Name of the binary target column.
        test_size: Proportion of data to use for testing (0 < test_size < 1).
        val_size: Proportion of data to use for validation (0 <= val_size < 1 - test_size).
        random_seed: Random seed for reproducible splits.
        serialize_path: Optional path to save fitted pipeline.

    Returns:
        PreprocessResult containing:
            - X_train, y_train, X_val, y_val, X_test, y_test: Transformed arrays
            - error: PreprocessingError enum if failed, None otherwise
            - error_message: Human-readable error description if failed, None otherwise

    Failure modes:
        INVALID_INPUT_TYPE: Input is not a DataFrame
        TARGET_COLUMN_MISSING: Target column not found
        INSUFFICIENT_SAMPLES: Too few samples for meaningful splits
        INVALID_SPLIT_SIZES: test_size/val_size out of range or sum >= 1
        NO_NUMERIC_FEATURES: No numeric features found
        NO_CATEGORICAL_FEATURES: No categorical features found
        STRATIFICATION_FAILED: Train-test split with stratification failed
        PIPELINE_FIT_FAILED: Pipeline fitting failed
        TRANSFORM_FAILED: Transformation failed
        SERIALIZATION_FAILED: Could not save pipeline
        UNEXPECTED_ERROR: Unexpected error during preprocessing

    Example:
        result = preprocess_data(df, "is_fraud", test_size=0.2, val_size=0.1)
        if result.error:
            print(f"Failed: {result.error_message}")
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = result
    """
    try:
        # Validate input
        input_error = _Validator.validate_input(df, target_col)
        if input_error:
            error_messages = {
                PreprocessingError.INVALID_INPUT_TYPE: "Input must be a pandas DataFrame",
                PreprocessingError.TARGET_COLUMN_MISSING: f"Target column '{target_col}' not found",
                PreprocessingError.INSUFFICIENT_SAMPLES: "Insufficient samples for preprocessing (min 100)",
            }
            return PreprocessResult(
                X_train=None,
                y_train=None,
                X_val=None,
                y_val=None,
                X_test=None,
                y_test=None,
                error=input_error,
                error_message=error_messages[input_error],
            )

        # Validate split sizes
        split_error = _Validator.validate_split_sizes(test_size, val_size)
        if split_error:
            return PreprocessResult(
                X_train=None,
                y_train=None,
                X_val=None,
                y_val=None,
                X_test=None,
                y_test=None,
                error=split_error,
                error_message="Invalid split sizes: test_size + val_size must be < 1 and both between 0 and 1",
            )

        # Identify feature types
        numeric_features, categorical_features = _Validator.identify_feature_types(df, target_col)

        if not numeric_features and not categorical_features:
            return PreprocessResult(
                X_train=None,
                y_train=None,
                X_val=None,
                y_val=None,
                X_test=None,
                y_test=None,
                error=PreprocessingError.INVALID_INPUT_TYPE,
                error_message="No features found in DataFrame",
            )

        # Extract target
        y = df[target_col].values.astype(np.int64)

        # First split: separate test set
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                df.drop(columns=[target_col]),
                y,
                test_size=test_size,
                stratify=y,
                random_state=random_seed,
            )
        except ValueError as e:
            logger.error("Stratification failed", exc_info=True)
            return PreprocessResult(
                X_train=None,
                y_train=None,
                X_val=None,
                y_val=None,
                X_test=None,
                y_test=None,
                error=PreprocessingError.STRATIFICATION_FAILED,
                error_message=f"Stratification failed: {str(e)}",
            )

        # Second split: separate validation from training
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp,
                    y_temp,
                    test_size=val_ratio,
                    stratify=y_temp,
                    random_state=random_seed,
                )
            except ValueError as e:
                logger.error("Stratification failed for validation split", exc_info=True)
                return PreprocessResult(
                    X_train=None,
                    y_train=None,
                    X_val=None,
                    y_val=None,
                    X_test=None,
                    y_test=None,
                    error=PreprocessingError.STRATIFICATION_FAILED,
                    error_message=f"Validation split failed: {str(e)}",
                )
        else:
            X_train, X_val, y_train, y_val = X_temp, pd.DataFrame(), y_temp, np.array([])

        logger.info(
            "Data split completed",
            extra={
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "train_fraud_rate": float(np.mean(y_train)) if len(y_train) > 0 else 0,
                "val_fraud_rate": float(np.mean(y_val)) if len(y_val) > 0 else 0,
                "test_fraud_rate": float(np.mean(y_test)),
            },
        )

        # Build and fit preprocessor
        preprocessor = _Preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_seed=random_seed,
        )

        try:
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_val_transformed = preprocessor.transform(X_val) if len(X_val) > 0 else np.array([])
            X_test_transformed = preprocessor.transform(X_test)
        except Exception as e:
            return PreprocessResult(
                X_train=None,
                y_train=None,
                X_val=None,
                y_val=None,
                X_test=None,
                y_test=None,
                error=PreprocessingError.PIPELINE_FIT_FAILED,
                error_message=f"Pipeline transformation failed: {str(e)}",
            )

        # Serialize if requested
        if serialize_path and preprocessor.pipeline:
            try:
                path = Path(serialize_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(preprocessor.pipeline, path)
                logger.info("Pipeline serialized", extra={"path": str(path)})
            except Exception as e:
                return PreprocessResult(
                    X_train=X_train_transformed,
                    y_train=y_train,
                    X_val=X_val_transformed,
                    y_val=y_val,
                    X_test=X_test_transformed,
                    y_test=y_test,
                    error=PreprocessingError.SERIALIZATION_FAILED,
                    error_message=f"Pipeline serialization failed: {str(e)}",
                )

        return PreprocessResult(
            X_train=X_train_transformed,
            y_train=y_train,
            X_val=X_val_transformed,
            y_val=y_val,
            X_test=X_test_transformed,
            y_test=y_test,
            error=None,
            error_message=None,
        )

    except Exception as e:
        logger.error("Unexpected error during preprocessing", exc_info=True)
        return PreprocessResult(
            X_train=None,
            y_train=None,
            X_val=None,
            y_val=None,
            X_test=None,
            y_test=None,
            error=PreprocessingError.UNEXPECTED_ERROR,
            error_message=f"Unexpected error: {str(e)}",
        )


__all__ = [
    "PreprocessingError",
    "PreprocessResult",
    "preprocess_data",
]