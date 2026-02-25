"""Data ingestion module with schema validation and memory optimization.

This module reads CSV data, validates the target column, performs memory optimization
through downcasting, and saves a data profile JSON.

Example:
    result = ingest_data("data/transactions.csv", target_col="is_fraud")
    if result.error:
        handle_error(result.error)
    else:
        df = result.data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import NamedTuple, TypedDict, Any
from enum import Enum, auto
import logging
import sys
from datetime import datetime

# Configure structured logger
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class IngestionError(Enum):
    """Enumerated failure modes for data ingestion."""

    FILE_NOT_FOUND = auto()
    EMPTY_FILE = auto()
    TARGET_COLUMN_MISSING = auto()
    TARGET_NOT_BINARY = auto()
    TARGET_NULLS = auto()
    INVALID_DATA_TYPE = auto()
    PROFILE_SAVE_FAILED = auto()
    UNEXPECTED_ERROR = auto()


class IngestResult(NamedTuple):
    """Result of data ingestion operation."""

    data: pd.DataFrame | None
    error: IngestionError | None
    error_message: str | None


class DataProfile(TypedDict):
    """Data profile schema."""

    row_count: int
    column_count: int
    column_types: dict[str, str]
    null_counts: dict[str, int]
    null_percentages: dict[str, float]
    target_distribution: dict[str, int | float]
    memory_usage_mb: float
    ingestion_timestamp: str


class _Validator:
    """Internal validation logic."""

    EXPECTED_FRAUD_RATE_MIN = 0.005  # 0.5%
    EXPECTED_FRAUD_RATE_MAX = 0.05  # 5%

    @staticmethod
    def validate_file_exists(file_path: Path) -> IngestionError | None:
        """Validate that file exists."""
        if not file_path.exists():
            return IngestionError.FILE_NOT_FOUND
        return None

    @staticmethod
    def validate_target_column(df: pd.DataFrame, target_col: str) -> IngestionError | None:
        """Validate target column exists and is binary."""
        if target_col not in df.columns:
            return IngestionError.TARGET_COLUMN_MISSING

        if df[target_col].isnull().any():
            return IngestionError.TARGET_NULLS

        unique_values = df[target_col].dropna().unique()
        if not set(unique_values).issubset({0, 1}):
            return IngestionError.TARGET_NOT_BINARY

        return None

    @staticmethod
    def validate_fraud_rate(df: pd.DataFrame, target_col: str) -> float:
        """Calculate fraud rate and log warning if outside expected range."""
        fraud_rate = df[target_col].mean()

        if fraud_rate < _Validator.EXPECTED_FRAUD_RATE_MIN:
            logger.warning(
                "Fraud rate below expected minimum",
                extra={
                    "fraud_rate": float(fraud_rate),
                    "expected_min": _Validator.EXPECTED_FRAUD_RATE_MIN,
                },
            )
        elif fraud_rate > _Validator.EXPECTED_FRAUD_RATE_MAX:
            logger.warning(
                "Fraud rate above expected maximum",
                extra={
                    "fraud_rate": float(fraud_rate),
                    "expected_max": _Validator.EXPECTED_FRAUD_RATE_MAX,
                },
            )
        else:
            logger.info(
                "Fraud rate within expected range",
                extra={"fraud_rate": float(fraud_rate)},
            )

        return float(fraud_rate)


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to float32 and object columns to category."""
    df_optimized = df.copy()

    for col in df_optimized.columns:
        col_dtype = df_optimized[col].dtype

        # Downcast floats to float32
        if pd.api.types.is_float_dtype(col_dtype):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

        # Downcast integers to smallest possible
        elif pd.api.types.is_integer_dtype(col_dtype):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="integer")

        # Convert objects to category if low cardinality
        elif pd.api.types.is_object_dtype(col_dtype):
            # Only convert to category if cardinality is less than 50% of rows
            cardinality = df_optimized[col].nunique()
            if cardinality / len(df_optimized) < 0.5:
                df_optimized[col] = df_optimized[col].astype("category")

    return df_optimized


def _generate_profile(df: pd.DataFrame, target_col: str, file_path: Path) -> DataProfile:
    """Generate data profile for the ingested dataset."""
    target_dist = df[target_col].value_counts().to_dict()
    target_dist["fraud_rate"] = float(df[target_col].mean())

    null_counts = df.isnull().sum().to_dict()
    null_percentages = {col: float(df[col].isnull().mean() * 100) for col in df.columns}

    column_types = {col: str(df[col].dtype) for col in df.columns}

    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return DataProfile(
        row_count=len(df),
        column_count=len(df.columns),
        column_types=column_types,
        null_counts=null_counts,
        null_percentages=null_percentages,
        target_distribution={
            "class_0": int(target_dist.get(0, 0)),
            "class_1": int(target_dist.get(1, 0)),
            "fraud_rate": target_dist["fraud_rate"],
        },
        memory_usage_mb=float(memory_usage_mb),
        ingestion_timestamp=datetime.utcnow().isoformat() + "Z",
    )


def _save_profile(profile: DataProfile, profile_path: Path) -> IngestionError | None:
    """Save data profile to JSON file."""
    try:
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2, default=str)
        logger.info("Data profile saved", extra={"path": str(profile_path)})
        return None
    except Exception as e:
        logger.error("Failed to save profile", extra={"error": str(e)})
        return IngestionError.PROFILE_SAVE_FAILED


def ingest_data(file_path: str | Path, target_col: str, profile_path: str | Path | None = None) -> IngestResult:
    """Read, validate, and optimize CSV data.

    Args:
        file_path: Path to the input CSV file.
        target_col: Name of the binary target column.
        profile_path: Optional path to save data profile JSON. If None, profile is not saved.

    Returns:
        IngestResult containing:
            - data: Optimized DataFrame if successful, None otherwise
            - error: IngestionError enum if failed, None otherwise
            - error_message: Human-readable error description if failed, None otherwise

    Failure modes:
        FILE_NOT_FOUND: Input CSV file does not exist
        EMPTY_FILE: CSV file is empty
        TARGET_COLUMN_MISSING: Target column not found in data
        TARGET_NOT_BINARY: Target contains values other than 0/1
        TARGET_NULLS: Target column contains null values
        INVALID_DATA_TYPE: Data contains unsupported types
        PROFILE_SAVE_FAILED: Could not save profile JSON
        UNEXPECTED_ERROR: Unexpected error during ingestion

    Example:
        result = ingest_data("transactions.csv", "is_fraud", "profile.json")
        if result.error:
            print(f"Failed: {result.error_message}")
        else:
            df = result.data
    """
    try:
        # Validate file exists
        path = Path(file_path)
        file_error = _Validator.validate_file_exists(path)
        if file_error:
            return IngestResult(
                data=None,
                error=file_error,
                error_message=f"File not found: {path}",
            )

        # Read CSV
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.error("Failed to read CSV", extra={"error": str(e)})
            return IngestResult(
                data=None,
                error=IngestionError.INVALID_DATA_TYPE,
                error_message=f"Failed to read CSV: {str(e)}",
            )

        # Validate not empty
        if df.empty:
            return IngestResult(
                data=None,
                error=IngestionError.EMPTY_FILE,
                error_message="CSV file is empty",
            )

        # Validate target column
        target_error = _Validator.validate_target_column(df, target_col)
        if target_error:
            error_messages = {
                IngestionError.TARGET_COLUMN_MISSING: f"Target column '{target_col}' not found",
                IngestionError.TARGET_NULLS: f"Target column '{target_col}' contains nulls",
                IngestionError.TARGET_NOT_BINARY: f"Target column '{target_col}' must contain only 0 and 1",
            }
            return IngestResult(
                data=None,
                error=target_error,
                error_message=error_messages[target_error],
            )

        # Check fraud rate
        fraud_rate = _Validator.validate_fraud_rate(df, target_col)
        logger.info(
            "Data loaded successfully",
            extra={
                "rows": len(df),
                "columns": len(df.columns),
                "fraud_rate": fraud_rate,
            },
        )

        # Optimize dtypes
        df_optimized = _optimize_dtypes(df)
        memory_saved = (df.memory_usage(deep=True).sum() - df_optimized.memory_usage(deep=True).sum()) / (1024 * 1024)
        logger.info(
            "Memory optimization complete",
            extra={
                "original_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
                "optimized_mb": float(df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)),
                "saved_mb": float(memory_saved),
            },
        )

        # Save profile if requested
        if profile_path:
            profile = _generate_profile(df_optimized, target_col, path)
            profile_error = _save_profile(profile, Path(profile_path))
            if profile_error:
                return IngestResult(
                    data=df_optimized,
                    error=profile_error,
                    error_message="Data ingested but profile save failed",
                )

        return IngestResult(data=df_optimized, error=None, error_message=None)

    except Exception as e:
        logger.error("Unexpected error during ingestion", exc_info=True)
        return IngestResult(
            data=None,
            error=IngestionError.UNEXPECTED_ERROR,
            error_message=f"Unexpected error: {str(e)}",
        )


__all__ = [
    "IngestionError",
    "IngestResult",
    "DataProfile",
    "ingest_data",
]