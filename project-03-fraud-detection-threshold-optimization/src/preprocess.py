import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
from sklearn.preprocessing import OrdinalEncoder

@dataclass(frozen=True)
class PreprocessConfig:
    amount_column_names: Tuple[str, ...] = ("TransactionAmt", "amount")
    id_columns_to_drop: Tuple[str, ...] = ("TransactionID", "card1", "addr1", "addr2")
    timestamp_column: str = "TransactionDT"
    seconds_per_day: int = 86400
    encoder_path: str = "artifacts/preprocessor.pkl"

class PreprocessError(Exception):
    """Base exception for preprocessing failures."""
    pass

class DataFrameEmptyError(PreprocessError):
    """Raised when input DataFrame is empty."""
    pass

class MissingColumnError(PreprocessError):
    """Raised when required column is missing."""
    pass

class UnexpectedNullsError(PreprocessError):
    """Raised when null values are found after validation."""
    pass

class EncoderNotFoundError(PreprocessError):
    """Raised when encoder file not found during inference."""
    pass

class EncoderFitError(PreprocessError):
    """Raised when encoder fitting fails."""
    pass

def _validate_input(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataFrameEmptyError("Input DataFrame is empty")
    if not isinstance(df, pd.DataFrame):
        raise PreprocessError(f"Expected pandas DataFrame, got {type(df)}")

def _validate_no_nulls(df: pd.DataFrame) -> None:
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        raise UnexpectedNullsError(f"Unexpected null values found in columns: {null_cols}")

def _apply_log_transform(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    amount_col = None
    for col_name in config.amount_column_names:
        if col_name in df.columns:
            amount_col = col_name
            break
    
    if amount_col is None:
        raise MissingColumnError(f"No amount column found from {config.amount_column_names}")
    
    df = df.copy()
    df['log_amount'] = np.log1p(df[amount_col])
    df = df.drop(columns=[amount_col])
    return df

def _extract_temporal_features(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    if config.timestamp_column not in df.columns:
        return df
    
    df = df.copy()
    seconds = df[config.timestamp_column]
    
    hours = (seconds // 3600) % 24
    days = seconds // config.seconds_per_day
    weekdays = (days + 1) % 7
    
    df['hour_of_day'] = hours
    df['day_of_week'] = weekdays
    df = df.drop(columns=[config.timestamp_column])
    
    return df

def _drop_id_columns(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    cols_to_drop = [col for col in config.id_columns_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df

def _get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def _fit_and_save_encoder(
    df: pd.DataFrame,
    categorical_cols: List[str],
    config: PreprocessConfig
) -> OrdinalEncoder:
    if not categorical_cols:
        encoder = OrdinalEncoder(dtype=np.int64)
        return encoder
    
    try:
        encoder = OrdinalEncoder(
            dtype=np.int64,
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        encoder.fit(df[categorical_cols])
        
        encoder_path = Path(config.encoder_path)
        encoder_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoder, encoder_path)
        
        return encoder
    except Exception as e:
        raise EncoderFitError(f"Failed to fit and save encoder: {str(e)}") from e

def _load_and_transform_encoder(
    df: pd.DataFrame,
    categorical_cols: List[str],
    config: PreprocessConfig
) -> OrdinalEncoder:
    encoder_path = Path(config.encoder_path)
    if not encoder_path.exists():
        raise EncoderNotFoundError(f"Encoder not found at {config.encoder_path}")
    
    try:
        encoder = joblib.load(encoder_path)
        return encoder
    except Exception as e:
        raise EncoderNotFoundError(f"Failed to load encoder: {str(e)}") from e

def _apply_encoder(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoder: OrdinalEncoder
) -> pd.DataFrame:
    if not categorical_cols or len(encoder.categories_) == 0:
        return df
    
    df = df.copy()
    encoded_values = encoder.transform(df[categorical_cols])
    
    for i, col in enumerate(categorical_cols):
        df[col] = encoded_values[:, i]
    
    return df

def preprocess_dataframe(
    df: pd.DataFrame,
    needs_log_transform: bool,
    dropped_columns: List[str],
    fit: bool
) -> Tuple[pd.DataFrame, pd.Series]:
    _validate_input(df)
    _validate_no_nulls(df)
    
    config = PreprocessConfig()
    
    working_df = df.copy()
    
    if needs_log_transform:
        working_df = _apply_log_transform(working_df, config)
    
    working_df = _extract_temporal_features(working_df, config)
    working_df = _drop_id_columns(working_df, config)
    
    if 'is_fraud' not in working_df.columns:
        raise MissingColumnError("is_fraud column not found")
    
    y = working_df['is_fraud'].copy()
    X = working_df.drop(columns=['is_fraud'])
    
    categorical_cols = _get_categorical_columns(X)
    
    if fit:
        encoder = _fit_and_save_encoder(X, categorical_cols, config)
    else:
        encoder = _load_and_transform_encoder(X, categorical_cols, config)
    
    X = _apply_encoder(X, categorical_cols, encoder)
    
    return X, y

if __name__ == "__main__":
    import sys
    try:
        print("This module is designed to be imported, not run directly.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)