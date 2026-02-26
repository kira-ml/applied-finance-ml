import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import sys

@dataclass(frozen=True)
class InferenceConfig:
    null_threshold: float = 0.40
    transaction_id_names: Tuple[str, ...] = ("TransactionID", "transaction_id")
    amount_column_names: Tuple[str, ...] = ("TransactionAmt", "amount")
    id_columns_to_drop: Tuple[str, ...] = ("TransactionID", "card1", "addr1", "addr2")
    timestamp_column: str = "TransactionDT"
    seconds_per_day: int = 86400
    model_path: str = "artifacts/model.pkl"
    preprocessor_path: str = "artifacts/preprocessor.pkl"
    scaler_path: str = "artifacts/scaler.pkl"
    threshold_path: str = "artifacts/threshold.txt"
    output_path: str = "data/output/predictions.csv"

class InferenceError(Exception):
    """Base exception for inference failures."""
    pass

class FileNotFoundError(InferenceError):
    """Raised when required file does not exist."""
    pass

class EmptyFileError(InferenceError):
    """Raised when file is empty."""
    pass

class DataFrameEmptyError(InferenceError):
    """Raised when input DataFrame is empty."""
    pass

class MissingColumnsError(InferenceError):
    """Raised when required columns are missing."""
    pass

class NumericFillError(InferenceError):
    """Raised when numeric fill operation fails."""
    pass

class CategoricalFillError(InferenceError):
    """Raised when categorical fill operation fails."""
    pass

class PreprocessorLoadError(InferenceError):
    """Raised when preprocessor loading fails."""
    pass

class ScalerLoadError(InferenceError):
    """Raised when scaler loading fails."""
    pass

class ModelLoadError(InferenceError):
    """Raised when model loading fails."""
    pass

class ThresholdLoadError(InferenceError):
    """Raised when threshold loading fails."""
    pass

class TransformError(InferenceError):
    """Raised when transform operation fails."""
    pass

class OutputWriteError(InferenceError):
    """Raised when writing output fails."""
    pass

def _validate_file_path(file_path: str, description: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    if path.stat().st_size == 0:
        raise EmptyFileError(f"{description} is empty: {file_path}")
    return path

def _validate_input_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    if df.empty:
        raise DataFrameEmptyError("Input DataFrame is empty")
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise MissingColumnsError(f"Missing required columns: {missing}")

def _drop_high_null_columns(df: pd.DataFrame, config: InferenceConfig) -> Tuple[pd.DataFrame, List[str]]:
    null_rates = df.isnull().mean()
    high_null_cols = null_rates[null_rates > config.null_threshold].index.tolist()
    
    if high_null_cols:
        df = df.drop(columns=high_null_cols)
    
    return df, high_null_cols

def _fill_numeric_nulls(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                raise NumericFillError(f"Cannot compute median for column {col} - all values are null")
            df[col] = df[col].fillna(median_val)
    
    return df

def _fill_categorical_nulls(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_vals = df[col].mode()
            if len(mode_vals) == 0:
                raise CategoricalFillError(f"Cannot compute mode for column {col} - no valid values")
            mode_val = mode_vals[0]
            df[col] = df[col].fillna(mode_val)
    
    return df

def _apply_log_transform(df: pd.DataFrame, config: InferenceConfig) -> pd.DataFrame:
    amount_col = None
    for col_name in config.amount_column_names:
        if col_name in df.columns:
            amount_col = col_name
            break
    
    if amount_col is None:
        return df
    
    df = df.copy()
    df['log_amount'] = np.log1p(df[amount_col])
    df = df.drop(columns=[amount_col])
    return df

def _extract_temporal_features(df: pd.DataFrame, config: InferenceConfig) -> pd.DataFrame:
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

def _drop_id_columns(df: pd.DataFrame, config: InferenceConfig) -> pd.DataFrame:
    cols_to_drop = [col for col in config.id_columns_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df

def _validate_no_nulls(df: pd.DataFrame) -> None:
    null_cols = df.columns[df.isnull().any()].tolist()
    if null_cols:
        raise InferenceError(f"Unexpected null values found in columns: {null_cols}")

def _get_categorical_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def _apply_preprocessor(
    df: pd.DataFrame,
    preprocessor_path: str,
    categorical_cols: List[str]
) -> pd.DataFrame:
    try:
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        raise PreprocessorLoadError(f"Failed to load preprocessor: {str(e)}") from e
    
    if not categorical_cols or len(preprocessor.categories_) == 0:
        return df
    
    df = df.copy()
    
    try:
        encoded_values = preprocessor.transform(df[categorical_cols])
        
        for i, col in enumerate(categorical_cols):
            df[col] = encoded_values[:, i]
            
    except Exception as e:
        raise TransformError(f"Failed to apply preprocessor transform: {str(e)}") from e
    
    return df

def _load_scaler(scaler_path: str) -> Optional[object]:
    scaler_file = Path(scaler_path)
    if not scaler_file.exists():
        return None
    
    try:
        return joblib.load(scaler_path)
    except Exception as e:
        raise ScalerLoadError(f"Failed to load scaler: {str(e)}") from e

def _apply_scaler(df: pd.DataFrame, scaler: object) -> pd.DataFrame:
    try:
        scaled_array = scaler.transform(df)
        return pd.DataFrame(scaled_array, columns=df.columns, index=df.index)
    except Exception as e:
        raise TransformError(f"Failed to apply scaler transform: {str(e)}") from e

def _load_model(model_path: str) -> object:
    try:
        return joblib.load(model_path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {str(e)}") from e

def _load_threshold(threshold_path: str) -> float:
    try:
        with open(threshold_path, 'r') as f:
            threshold_str = f.read().strip()
            threshold = float(threshold_str)
            
        if not (0.0 <= threshold <= 1.0):
            raise ThresholdLoadError(f"Threshold must be in [0,1], got {threshold}")
        
        return threshold
    except Exception as e:
        raise ThresholdLoadError(f"Failed to load threshold: {str(e)}") from e

def _write_predictions(
    original_df: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    output_path: str
) -> None:
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        result_df = original_df.copy()
        result_df['fraud_probability'] = probabilities
        result_df['predicted_fraud'] = predictions
        
        result_df.to_csv(output_file, index=False)
        
    except Exception as e:
        raise OutputWriteError(f"Failed to write predictions: {str(e)}") from e

def run_inference(csv_path: str) -> None:
    config = InferenceConfig()
    
    input_file = _validate_file_path(csv_path, "Input CSV")
    model_file = _validate_file_path(config.model_path, "Model")
    preprocessor_file = _validate_file_path(config.preprocessor_path, "Preprocessor")
    threshold_file = _validate_file_path(config.threshold_path, "Threshold")
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        raise InferenceError(f"Failed to read input CSV: {str(e)}") from e
    
    required_columns = [
        "TransactionAmt", "TransactionDT", "hour_of_day", "day_of_week",
        "card_type", "merchant_category", "V1"
    ]
    _validate_input_dataframe(df, required_columns)
    
    original_df = df.copy()
    
    working_df, _ = _drop_high_null_columns(df, config)
    
    working_df = _fill_numeric_nulls(working_df)
    working_df = _fill_categorical_nulls(working_df)
    
    working_df = _apply_log_transform(working_df, config)
    working_df = _extract_temporal_features(working_df, config)
    working_df = _drop_id_columns(working_df, config)
    
    _validate_no_nulls(working_df)
    
    categorical_cols = _get_categorical_columns(working_df)
    working_df = _apply_preprocessor(working_df, str(preprocessor_file), categorical_cols)
    
    scaler = _load_scaler(config.scaler_path)
    if scaler is not None:
        working_df = _apply_scaler(working_df, scaler)
    
    model = _load_model(str(model_file))
    threshold = _load_threshold(str(threshold_file))
    
    try:
        probabilities = model.predict_proba(working_df)[:, 1]
    except Exception as e:
        raise InferenceError(f"Failed to generate predictions: {str(e)}") from e
    
    predictions = (probabilities >= threshold).astype(int)
    
    _write_predictions(original_df, probabilities, predictions, config.output_path)
    
    fraud_count = np.sum(predictions)
    fraud_rate = fraud_count / len(predictions)
    
    print(f"Total rows scored: {len(predictions)}")
    print(f"Predicted fraud count: {fraud_count}")
    print(f"Predicted fraud rate: {fraud_rate:.6f}")
    print(f"Predictions written to: {config.output_path}")

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python infer.py <path_to_new_transactions.csv>", file=sys.stderr)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        run_inference(csv_path)
        sys.exit(0)
    except InferenceError as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()