import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import math

# Immutable configuration
REQUIRED_COLUMNS = ('date', 'ticker', 'close_price')
EXPECTED_DTYPES = {'date': 'datetime64[ns]', 'ticker': 'object', 'close_price': 'float64'}
MAX_MISSING_PCT = 5.0
MAX_CONSECUTIVE_MISSING = 3
MIN_PRICE = 0.01
MAX_PRICE = 1000000.0
MIN_TICKERS = 1
MAX_TICKERS = 50
DATE_FREQ = 'D'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Custom exceptions
class ValidationError(Exception):
    """Base exception for validation errors."""
    pass

class FileValidationError(ValidationError):
    """Exception for file validation failures."""
    def __init__(self, filepath: str, message: str, code: str):
        self.filepath = filepath
        self.message = message
        self.code = code
        super().__init__(f"[{code}] File '{filepath}': {message}")

class SchemaValidationError(ValidationError):
    """Exception for schema validation failures."""
    def __init__(self, component: str, message: str, code: str):
        self.component = component
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {component}: {message}")

class DataQualityError(ValidationError):
    """Exception for data quality validation failures."""
    def __init__(self, check: str, message: str, code: str):
        self.check = check
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {check}: {message}")

# Pure core logic
def _validate_filepath(filepath: str) -> None:
    """Validate filepath string."""
    if not isinstance(filepath, str):
        raise FileValidationError(
            str(filepath),
            f"filepath must be string, got {type(filepath).__name__}",
            "INVALID_FILEPATH_TYPE"
        )
    if not filepath:
        raise FileValidationError(
            filepath,
            "filepath cannot be empty",
            "EMPTY_FILEPATH"
        )

def validate_file_exists(filepath: str) -> None:
    """
    Validate that a file exists and is readable.
    
    Args:
        filepath: Path to file
    
    Raises:
        FileValidationError: If file validation fails
    """
    _validate_filepath(filepath)
    
    path = Path(filepath)
    logger.info(f"Checking file: {filepath}")
    
    if not path.exists():
        raise FileValidationError(
            filepath,
            "file does not exist",
            "FILE_NOT_FOUND"
        )
    if not path.is_file():
        raise FileValidationError(
            filepath,
            "path exists but is not a file",
            "NOT_A_FILE"
        )
    if not path.stat().st_size > 0:
        raise FileValidationError(
            filepath,
            "file is empty",
            "EMPTY_FILE"
        )
    
    logger.info(f"File validation passed: {filepath} ({path.stat().st_size} bytes)")

def _validate_columns(df: pd.DataFrame, required: Tuple[str, ...]) -> None:
    """Validate DataFrame has required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SchemaValidationError(
            "columns",
            f"missing required columns: {missing}",
            "MISSING_COLUMNS"
        )
    logger.info(f"Column validation passed: {list(df.columns)}")

def _validate_dtypes(df: pd.DataFrame, dtypes: Dict[str, str]) -> None:
    """Validate DataFrame column types."""
    for col, expected_type in dtypes.items():
        if col not in df.columns:
            continue
        if expected_type == 'datetime64[ns]':
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                raise SchemaValidationError(
                    f"column '{col}'",
                    f"expected datetime type, got {df[col].dtype}",
                    "INVALID_DATETIME_TYPE"
                )
        elif expected_type == 'float64':
            if not pd.api.types.is_float_dtype(df[col]):
                raise SchemaValidationError(
                    f"column '{col}'",
                    f"expected float type, got {df[col].dtype}",
                    "INVALID_FLOAT_TYPE"
                )
        elif expected_type == 'object':
            if not pd.api.types.is_object_dtype(df[col]):
                raise SchemaValidationError(
                    f"column '{col}'",
                    f"expected object type, got {df[col].dtype}",
                    "INVALID_OBJECT_TYPE"
                )
    logger.info("Data type validation passed")

def _validate_non_empty(df: pd.DataFrame) -> None:
    """Validate DataFrame is not empty."""
    if df.empty:
        raise SchemaValidationError(
            "dataframe",
            "DataFrame is empty",
            "EMPTY_DATAFRAME"
        )
    logger.info(f"DataFrame has {len(df)} rows")

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Tuple[str, ...] = REQUIRED_COLUMNS
) -> pd.DataFrame:
    """
    Validate DataFrame schema and basic structure.
    
    Args:
        df: DataFrame to validate
        required_columns: Required column names
    
    Returns:
        Validated DataFrame (unchanged)
    
    Raises:
        SchemaValidationError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaValidationError(
            "input",
            f"expected pandas DataFrame, got {type(df).__name__}",
            "INVALID_DATAFRAME_TYPE"
        )
    
    _validate_non_empty(df)
    _validate_columns(df, required_columns)
    _validate_dtypes(df, EXPECTED_DTYPES)
    
    return df

def _calculate_missing_stats(
    df: pd.DataFrame,
    tickers: List[str],
    expected_dates: pd.DatetimeIndex
) -> Dict[str, float]:
    """Calculate missing percentage per ticker."""
    missing_stats = {}
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker]
        present_dates = set(ticker_data['date'].dt.date)
        expected_dates_set = set(expected_dates.date)
        missing_count = len(expected_dates_set - present_dates)
        missing_stats[ticker] = (missing_count / len(expected_dates)) * 100.0
    return missing_stats

def check_missing_dates(
    df: pd.DataFrame,
    freq: str = DATE_FREQ,
    max_missing_pct: float = MAX_MISSING_PCT
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Check for missing dates in time series data.
    
    Args:
        df: DataFrame with date column
        freq: Expected frequency of dates
        max_missing_pct: Maximum allowed missing percentage
    
    Returns:
        Tuple of (original DataFrame, missing stats dictionary)
    
    Raises:
        DataQualityError: If missing data exceeds threshold
    """
    logger.info("Checking for missing dates...")
    
    if 'date' not in df.columns:
        raise DataQualityError(
            "missing_dates",
            "date column not found",
            "DATE_COLUMN_MISSING"
        )
    
    # Get date range
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    if pd.isna(min_date) or pd.isna(max_date):
        raise DataQualityError(
            "missing_dates",
            "cannot determine date range due to null values",
            "INVALID_DATE_RANGE"
        )
    
    logger.info(f"Date range: {min_date} to {max_date}")
    
    # Create expected date range
    expected_dates = pd.date_range(start=min_date, end=max_date, freq=freq)
    logger.info(f"Expected {len(expected_dates)} trading days")
    
    # Get unique tickers
    tickers = df['ticker'].unique().tolist()
    if not tickers:
        raise DataQualityError(
            "missing_dates",
            "no tickers found in data",
            "NO_TICKERS"
        )
    
    logger.info(f"Found {len(tickers)} tickers")
    
    # Calculate missing stats
    missing_stats = _calculate_missing_stats(df, tickers, expected_dates)
    
    # Log missing stats
    for ticker, pct in missing_stats.items():
        if pct > 0:
            logger.warning(f"Ticker {ticker} missing {pct:.2f}% of dates")
    
    # Check against threshold
    for ticker, pct in missing_stats.items():
        if pct > max_missing_pct:
            raise DataQualityError(
                f"ticker '{ticker}'",
                f"missing {pct:.2f}% of dates (max allowed: {max_missing_pct}%)",
                "EXCESSIVE_MISSING_DATA"
            )
    
    logger.info("Missing date check passed")
    return df, missing_stats

def _validate_price_value(price: float, row_info: str) -> None:
    """Validate a single price value."""
    if pd.isna(price):
        raise DataQualityError(
            "price_quality",
            f"null price at {row_info}",
            "NULL_PRICE"
        )
    if price < MIN_PRICE:
        raise DataQualityError(
            "price_quality",
            f"price {price} below minimum {MIN_PRICE} at {row_info}",
            "PRICE_BELOW_MINIMUM"
        )
    if price > MAX_PRICE:
        raise DataQualityError(
            "price_quality",
            f"price {price} above maximum {MAX_PRICE} at {row_info}",
            "PRICE_ABOVE_MAXIMUM"
        )
    if math.isinf(price) or math.isnan(price):
        raise DataQualityError(
            "price_quality",
            f"invalid price value {price} at {row_info}",
            "INVALID_PRICE_VALUE"
        )

def check_price_quality(
    df: pd.DataFrame,
    min_price: float = MIN_PRICE,
    max_price: float = MAX_PRICE
) -> pd.DataFrame:
    """
    Validate price values are within acceptable ranges.
    
    Args:
        df: DataFrame with close_price column
        min_price: Minimum allowed price
        max_price: Maximum allowed price
    
    Returns:
        Validated DataFrame
    
    Raises:
        DataQualityError: If price validation fails
    """
    logger.info("Checking price quality...")
    
    if 'close_price' not in df.columns:
        raise DataQualityError(
            "price_quality",
            "close_price column not found",
            "PRICE_COLUMN_MISSING"
        )
    
    # Summary statistics
    logger.info(f"Price stats - Min: {df['close_price'].min():.4f}, "
                f"Max: {df['close_price'].max():.4f}, "
                f"Mean: {df['close_price'].mean():.4f}")
    
    # Check all prices are positive
    if (df['close_price'] <= 0).any():
        negative_count = (df['close_price'] <= 0).sum()
        raise DataQualityError(
            "price_quality",
            f"found {negative_count} non-positive prices",
            "NON_POSITIVE_PRICES"
        )
    
    # Sample check individual prices (first 1000 for performance)
    sample_size = min(1000, len(df))
    sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
    
    for idx in sample_indices:
        row = df.iloc[idx]
        _validate_price_value(
            row['close_price'],
            f"row {idx}, date {row.get('date', 'unknown')}, ticker {row.get('ticker', 'unknown')}"
        )
    
    logger.info("Price quality check passed")
    return df

def check_ticker_coverage(
    df: pd.DataFrame,
    min_tickers: int = MIN_TICKERS,
    max_tickers: int = MAX_TICKERS
) -> Tuple[pd.DataFrame, int]:
    """
    Validate ticker count is within acceptable range.
    
    Args:
        df: DataFrame with ticker column
        min_tickers: Minimum allowed number of tickers
        max_tickers: Maximum allowed number of tickers
    
    Returns:
        Tuple of (validated DataFrame, ticker count)
    
    Raises:
        DataQualityError: If ticker count validation fails
    """
    logger.info("Checking ticker coverage...")
    
    if 'ticker' not in df.columns:
        raise DataQualityError(
            "ticker_coverage",
            "ticker column not found",
            "TICKER_COLUMN_MISSING"
        )
    
    tickers = df['ticker'].unique()
    n_tickers = len(tickers)
    
    logger.info(f"Found {n_tickers} unique tickers: {sorted(tickers)[:5]}{'...' if n_tickers > 5 else ''}")
    
    if n_tickers < min_tickers:
        raise DataQualityError(
            "ticker_coverage",
            f"found {n_tickers} tickers, minimum required is {min_tickers}",
            "INSUFFICIENT_TICKERS"
        )
    
    if n_tickers > max_tickers:
        raise DataQualityError(
            "ticker_coverage",
            f"found {n_tickers} tickers, maximum allowed is {max_tickers}",
            "EXCESSIVE_TICKERS"
        )
    
    logger.info("Ticker coverage check passed")
    return df, n_tickers

def check_data_types(
    df: pd.DataFrame,
    dtypes: Dict[str, str] = EXPECTED_DTYPES
) -> pd.DataFrame:
    """
    Validate column data types.
    
    Args:
        df: DataFrame to validate
        dtypes: Dictionary of expected dtypes
    
    Returns:
        Validated DataFrame
    
    Raises:
        SchemaValidationError: If type validation fails
    """
    logger.info("Checking data types...")
    _validate_dtypes(df, dtypes)
    return df

def _forward_fill_series(
    series: pd.Series,
    max_gap: int,
    ticker: str,
    date_range: pd.DatetimeIndex
) -> pd.Series:
    """Forward fill with gap limit check."""
    # Check for gaps exceeding max_gap
    if series.isna().any():
        # Find consecutive null sequences
        is_null = series.isna()
        null_groups = (is_null != is_null.shift()).cumsum()
        null_lengths = null_groups[is_null].value_counts()
        
        if (null_lengths > max_gap).any():
            max_found = null_lengths.max()
            logger.warning(f"Ticker {ticker} has gap of {max_found} days")
            raise DataQualityError(
                f"ticker '{ticker}'",
                f"found gap of {max_found} days (max allowed: {max_gap})",
                "EXCESSIVE_MISSING_GAP"
            )
        
        logger.info(f"Ticker {ticker}: filling {int(null_lengths.sum())} missing values")
    
    return series.ffill(limit=max_gap)

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'ffill',
    max_gap: int = MAX_CONSECUTIVE_MISSING
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame with missing values
        strategy: Strategy for handling missing ('ffill' only)
        max_gap: Maximum consecutive missing values to fill
    
    Returns:
        DataFrame with handled missing values
    
    Raises:
        DataQualityError: If missing value handling fails
    """
    logger.info("Handling missing values...")
    
    if strategy != 'ffill':
        raise DataQualityError(
            "missing_values",
            f"unsupported strategy: {strategy}, only 'ffill' allowed",
            "UNSUPPORTED_STRATEGY"
        )
    
    total_missing = df.isna().sum().sum()
    if total_missing == 0:
        logger.info("No missing values found")
        return df
    
    logger.info(f"Found {total_missing} missing values")
    
    # Get full date range
    date_range = pd.date_range(
        start=df['date'].min(),
        end=df['date'].max(),
        freq=DATE_FREQ
    )
    
    # Process each ticker separately
    result_dfs = []
    tickers = df['ticker'].unique()
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Reindex to full date range
        ticker_data = ticker_data.set_index('date')
        ticker_data = ticker_data.reindex(date_range)
        ticker_data['ticker'] = ticker
        
        # Forward fill prices
        if 'close_price' in ticker_data.columns:
            ticker_data['close_price'] = _forward_fill_series(
                ticker_data['close_price'],
                max_gap,
                ticker,
                date_range
            )
        
        result_dfs.append(ticker_data.reset_index().rename(columns={'index': 'date'}))
    
    result = pd.concat(result_dfs, ignore_index=True)
    
    # Check if any nulls remain
    remaining_nulls = result['close_price'].isna().sum()
    if remaining_nulls > 0:
        raise DataQualityError(
            "missing_values",
            f"unable to fill {remaining_nulls} missing values",
            "UNFILLED_MISSING_VALUES"
        )
    
    logger.info(f"Successfully filled {total_missing} missing values")
    return result

def generate_validation_report(df: pd.DataFrame, report_file: Optional[str] = None) -> str:
    """
    Generate comprehensive validation report.
    
    Args:
        df: DataFrame to analyze
        report_file: Optional path to save report
    
    Returns:
        Report as string
    
    Raises:
        ValidationError: If report generation fails
    """
    logger.info("Generating validation report...")
    
    try:
        lines = []
        lines.append("=" * 80)
        lines.append("DATA VALIDATION REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        
        # Basic info
        lines.append("BASIC INFORMATION")
        lines.append("-" * 40)
        lines.append(f"Total rows: {len(df):,}")
        lines.append(f"Total columns: {len(df.columns)}")
        lines.append(f"Columns: {', '.join(df.columns)}")
        lines.append("")
        
        # Date range
        if 'date' in df.columns:
            lines.append("DATE RANGE")
            lines.append("-" * 40)
            lines.append(f"Start date: {df['date'].min()}")
            lines.append(f"End date: {df['date'].max()}")
            lines.append(f"Total days: {(df['date'].max() - df['date'].min()).days + 1}")
            lines.append("")
        
        # Ticker info
        if 'ticker' in df.columns:
            lines.append("TICKER INFORMATION")
            lines.append("-" * 40)
            tickers = df['ticker'].unique()
            lines.append(f"Unique tickers: {len(tickers)}")
            if len(tickers) <= 10:
                lines.append(f"Tickers: {', '.join(sorted(tickers))}")
            else:
                lines.append(f"First 10 tickers: {', '.join(sorted(tickers)[:10])}...")
            
            # Ticker counts
            ticker_counts = df['ticker'].value_counts()
            lines.append(f"Rows per ticker - Min: {ticker_counts.min()}, Max: {ticker_counts.max()}, Mean: {ticker_counts.mean():.1f}")
            lines.append("")
        
        # Price statistics
        if 'close_price' in df.columns:
            lines.append("PRICE STATISTICS")
            lines.append("-" * 40)
            lines.append(f"Min price: ${df['close_price'].min():.4f}")
            lines.append(f"Max price: ${df['close_price'].max():.4f}")
            lines.append(f"Mean price: ${df['close_price'].mean():.4f}")
            lines.append(f"Median price: ${df['close_price'].median():.4f}")
            lines.append(f"Std dev: ${df['close_price'].std():.4f}")
            
            # Percentiles
            percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
            lines.append("\nPrice percentiles:")
            for p in percentiles:
                val = df['close_price'].quantile(p)
                lines.append(f"  {p*100:2.0f}th: ${val:.4f}")
            lines.append("")
        
        # Missing data
        lines.append("MISSING DATA")
        lines.append("-" * 40)
        missing_total = df.isna().sum().sum()
        lines.append(f"Total missing values: {missing_total}")
        if missing_total > 0:
            missing_by_col = df.isna().sum()
            for col in missing_by_col[missing_by_col > 0].index:
                pct = (missing_by_col[col] / len(df)) * 100
                lines.append(f"  {col}: {missing_by_col[col]} ({pct:.2f}%)")
        
        # Duplicates
        lines.append("\nDUPLICATES")
        lines.append("-" * 40)
        duplicates = df.duplicated().sum()
        lines.append(f"Duplicate rows: {duplicates}")
        if duplicates > 0:
            dup_pct = (duplicates / len(df)) * 100
            lines.append(f"Duplicate percentage: {dup_pct:.2f}%")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        # Save to file if requested
        if report_file:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to: {report_file}")
        
        logger.info("Validation report generated successfully")
        return report
        
    except Exception as e:
        raise ValidationError(f"report generation failed: {str(e)}")

# Entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate price data CSV file')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--report', '-r', help='Path to save validation report')
    parser.add_argument('--output', '-o', help='Path to save cleaned data (optional)')
    
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting validation of {args.input_file}")
        
        # Validate file exists
        validate_file_exists(args.input_file)
        
        # Load data
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df)} rows from {args.input_file}")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Run validation pipeline
        df = validate_dataframe(df)
        df, missing_stats = check_missing_dates(df)
        df = check_price_quality(df)
        df, n_tickers = check_ticker_coverage(df)
        df = check_data_types(df)
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Generate report
        report = generate_validation_report(df, args.report)
        print("\n" + report)
        
        # Save cleaned data if requested
        if args.output:
            df.to_csv(args.output, index=False)
            logger.info(f"Cleaned data saved to: {args.output}")
        
        logger.info("Validation completed successfully")
        
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)