import pandas as pd
import sys
from pathlib import Path
from typing import List
from dataclasses import dataclass

@dataclass(frozen=True)
class IngestConfig:
    required_columns: List[str] = None
    min_rows: int = 50000
    max_rows: int = 600000
    min_fraud_rate: float = 0.005
    max_fraud_rate: float = 0.05
    
    def __post_init__(self) -> None:
        if self.required_columns is None:
            object.__setattr__(self, 'required_columns', [
                "TransactionAmt", "TransactionDT", "hour_of_day", "day_of_week",
                "card_type", "merchant_category", "V1", "is_fraud"
            ])

class IngestError(Exception):
    """Base exception for ingestion failures."""
    pass

class FileNotFoundError(IngestError):
    """Raised when CSV file does not exist."""
    pass

class EmptyFileError(IngestError):
    """Raised when CSV file is empty."""
    pass

class RowCountError(IngestError):
    """Raised when row count is outside acceptable range."""
    pass

class MissingColumnsError(IngestError):
    """Raised when required columns are missing."""
    pass

class FraudColumnError(IngestError):
    """Raised when is_fraud column has invalid dtype or values."""
    pass

class FraudRateError(IngestError):
    """Raised when fraud rate is outside acceptable range."""
    pass

def _validate_file_path(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.stat().st_size == 0:
        raise EmptyFileError(f"File is empty: {file_path}")
    return path

def _validate_row_count(df: pd.DataFrame, config: IngestConfig) -> None:
    row_count = len(df)
    if not (config.min_rows <= row_count <= config.max_rows):
        raise RowCountError(
            f"Row count {row_count} outside acceptable range "
            f"[{config.min_rows}, {config.max_rows}]"
        )

def _validate_columns(df: pd.DataFrame, config: IngestConfig) -> None:
    missing = [col for col in config.required_columns if col not in df.columns]
    if missing:
        raise MissingColumnsError(f"Missing required columns: {missing}")

def _validate_fraud_column(df: pd.DataFrame) -> None:
    if "is_fraud" not in df.columns:
        raise FraudColumnError("is_fraud column not found")
    
    if not pd.api.types.is_integer_dtype(df["is_fraud"]):
        raise FraudColumnError(
            f"is_fraud must be integer dtype, got {df['is_fraud'].dtype}"
        )
    
    unique_values = set(df["is_fraud"].unique())
    if not unique_values.issubset({0, 1}):
        raise FraudColumnError(
            f"is_fraud must contain only 0 and 1, found values: {unique_values - {0, 1}}"
        )

def _validate_fraud_rate(df: pd.DataFrame, config: IngestConfig) -> float:
    fraud_rate = df["is_fraud"].mean()
    if not (config.min_fraud_rate <= fraud_rate <= config.max_fraud_rate):
        raise FraudRateError(
            f"Fraud rate {fraud_rate:.6f} outside acceptable range "
            f"[{config.min_fraud_rate}, {config.max_fraud_rate}]"
        )
    return fraud_rate

def ingest_transactions(csv_path: str) -> pd.DataFrame:
    config = IngestConfig()
    
    file_path = _validate_file_path(csv_path)
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise IngestError(f"Failed to read CSV: {str(e)}") from e
    
    if df.empty:
        raise EmptyFileError(f"CSV file contains no data: {csv_path}")
    
    _validate_row_count(df, config)
    _validate_columns(df, config)
    _validate_fraud_column(df)
    fraud_rate = _validate_fraud_rate(df, config)
    
    print(f"Fraud rate: {fraud_rate:.6f}")
    
    return df

def main() -> None:
    csv_path = r"D:\applied-finance-ml\project-03-fraud-detection-threshold-optimization\data\raw\transactions.csv"
    
    try:
        df = ingest_transactions(csv_path)
        print(f"Successfully ingested {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except IngestError as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()