import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Union
import os
import sys

class ValidationError(Exception):
    pass

def validate(
    df: pd.DataFrame,
    required_columns: List[str],
    max_missing_pct: float,
    min_price: float,
    max_price: float
) -> Dict[str, Union[pd.DataFrame, bool, List[str], Dict]]:
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if not isinstance(required_columns, list) or not required_columns:
        raise ValidationError("required_columns must be a non-empty list")
    if not all(isinstance(col, str) for col in required_columns):
        raise ValidationError("All required_columns must be strings")
    if not isinstance(max_missing_pct, (int, float)):
        raise ValidationError(f"max_missing_pct must be a number, got {type(max_missing_pct).__name__}")
    if not (0.0 <= max_missing_pct <= 100.0):
        raise ValidationError(f"max_missing_pct must be between 0 and 100, got {max_missing_pct}")
    if not isinstance(min_price, (int, float)):
        raise ValidationError(f"min_price must be a number, got {type(min_price).__name__}")
    if not isinstance(max_price, (int, float)):
        raise ValidationError(f"max_price must be a number, got {type(max_price).__name__}")
    if min_price >= max_price:
        raise ValidationError(f"min_price ({min_price}) must be less than max_price ({max_price})")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValidationError(f"Required columns missing: {missing_cols}")
    
    total_rows = len(df)
    warnings = []
    stats = {
        'missing_analysis': {},
        'price_stats': {},
        'date_range': {},
        'ticker_count': 0,
        'duplicates': 0
    }
    
    for col in required_columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_rows) * 100.0 if total_rows > 0 else 0.0
        stats['missing_analysis'][col] = {
            'missing_count': missing_count,
            'missing_pct': round(missing_pct, 2)
        }
        if missing_pct > max_missing_pct:
            raise ValidationError(f"Column '{col}' has {missing_pct:.2f}% missing values, exceeds limit {max_missing_pct}%")
    
    cleaned_df = df.dropna(subset=required_columns)
    rows_dropped = total_rows - len(cleaned_df)
    if rows_dropped > 0:
        warnings.append(f"Dropped {rows_dropped} rows due to missing values in required columns")
    
    if len(cleaned_df) == 0:
        raise ValidationError("DataFrame is empty after dropping rows with missing values")
    
    price_col = 'close_price'
    if price_col not in required_columns:
        raise ValidationError(f"'{price_col}' must be in required_columns")
    
    stats['price_stats'] = {
        'min': float(cleaned_df[price_col].min()),
        'max': float(cleaned_df[price_col].max()),
        'mean': float(cleaned_df[price_col].mean()),
        'median': float(cleaned_df[price_col].median()),
        'std': float(cleaned_df[price_col].std()),
        'zeros': int((cleaned_df[price_col] == 0).sum()),
        'negatives': int((cleaned_df[price_col] < 0).sum())
    }
    
    out_of_range = cleaned_df[(cleaned_df[price_col] < min_price) | (cleaned_df[price_col] > max_price)]
    if not out_of_range.empty:
        raise ValidationError(f"Found {len(out_of_range)} rows with {price_col} outside [{min_price}, {max_price}]")
    
    if 'date' in cleaned_df.columns:
        try:
            dates = pd.to_datetime(cleaned_df['date'])
            stats['date_range'] = {
                'min_date': dates.min().strftime('%Y-%m-%d'),
                'max_date': dates.max().strftime('%Y-%m-%d'),
                'unique_days': dates.nunique()
            }
        except:
            warnings.append("Could not parse date column for date range analysis")
    
    if 'ticker' in cleaned_df.columns:
        stats['ticker_count'] = cleaned_df['ticker'].nunique()
        ticker_stats = cleaned_df.groupby('ticker').size().to_dict()
        stats['ticker_distribution'] = {k: v for k, v in list(ticker_stats.items())[:5]}
        if len(ticker_stats) > 5:
            stats['ticker_distribution']['others'] = sum(list(ticker_stats.values())[5:])
    
    duplicates = cleaned_df.duplicated().sum()
    stats['duplicates'] = duplicates
    if duplicates > 0:
        warnings.append(f"Found {duplicates} duplicate rows")
    
    return {
        'df': cleaned_df,
        'validation_passed': True,
        'warnings': warnings,
        'statistics': stats
    }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('validation.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        prices_file = os.path.join("data", "raw", "prices.csv")
        logger.info(f"Reading {prices_file}")
        
        if not os.path.exists(prices_file):
            logger.error(f"File not found: {prices_file}")
            sys.exit(1)
        
        df = pd.read_csv(prices_file)
        logger.info(f"Read {len(df)} rows, columns: {list(df.columns)}")
        
        result = validate(
            df=df,
            required_columns=['date', 'ticker', 'close_price'],
            max_missing_pct=5.0,
            min_price=0.01,
            max_price=1000000.0
        )
        
        stats = result['statistics']
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FINANCIAL DATA VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        report_lines.append("1. DATASET OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append(f"Original rows: {len(df):,}")
        report_lines.append(f"Valid rows: {len(result['df']):,}")
        report_lines.append(f"Rows removed: {len(df) - len(result['df']):,}")
        report_lines.append(f"Removal rate: {((len(df) - len(result['df']))/len(df)*100):.2f}%")
        report_lines.append(f"Total tickers: {stats['ticker_count']}")
        report_lines.append(f"Date range: {stats['date_range'].get('min_date', 'N/A')} to {stats['date_range'].get('max_date', 'N/A')}")
        report_lines.append(f"Trading days: {stats['date_range'].get('unique_days', 'N/A')}")
        report_lines.append("")
        
        report_lines.append("2. DATA QUALITY CHECKS")
        report_lines.append("-" * 40)
        report_lines.append("Missing values by column:")
        for col, missing_stats in stats['missing_analysis'].items():
            report_lines.append(f"  - {col}: {missing_stats['missing_count']:,} missing ({missing_stats['missing_pct']}%)")
        report_lines.append(f"Duplicate rows: {stats['duplicates']:,}")
        report_lines.append("")
        
        report_lines.append("3. PRICE STATISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Minimum price: ${stats['price_stats']['min']:.4f}")
        report_lines.append(f"Maximum price: ${stats['price_stats']['max']:.4f}")
        report_lines.append(f"Mean price: ${stats['price_stats']['mean']:.4f}")
        report_lines.append(f"Median price: ${stats['price_stats']['median']:.4f}")
        report_lines.append(f"Standard deviation: ${stats['price_stats']['std']:.4f}")
        report_lines.append(f"Zero prices: {stats['price_stats']['zeros']:,}")
        report_lines.append(f"Negative prices: {stats['price_stats']['negatives']:,}")
        report_lines.append("")
        
        report_lines.append("4. TICKER DISTRIBUTION (Top 5)")
        report_lines.append("-" * 40)
        if 'ticker_distribution' in stats:
            for ticker, count in stats['ticker_distribution'].items():
                if ticker != 'others':
                    pct = (count / len(result['df'])) * 100
                    report_lines.append(f"  - {ticker}: {count:,} rows ({pct:.1f}%)")
            if 'others' in stats['ticker_distribution']:
                report_lines.append(f"  - Other tickers: {stats['ticker_distribution']['others']:,} rows")
        report_lines.append("")
        
        report_lines.append("5. VALIDATION RESULTS")
        report_lines.append("-" * 40)
        report_lines.append(f"Status: {'PASSED' if result['validation_passed'] else 'FAILED'}")
        report_lines.append(f"Warnings: {len(result['warnings'])}")
        if result['warnings']:
            report_lines.append("")
            report_lines.append("Warnings:")
            for i, w in enumerate(result['warnings'], 1):
                report_lines.append(f"  {i}. {w}")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_file = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info("=" * 50)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Status: PASSED")
        logger.info(f"Valid rows: {len(result['df']):,}/{len(df):,}")
        logger.info(f"Date range: {stats['date_range'].get('min_date', 'N/A')} to {stats['date_range'].get('max_date', 'N/A')}")
        logger.info(f"Tickers: {stats['ticker_count']}")
        logger.info(f"Price range: ${stats['price_stats']['min']:.2f} - ${stats['price_stats']['max']:.2f}")
        logger.info(f"Warnings: {len(result['warnings'])}")
        logger.info("=" * 50)
        logger.info(f"Full report: {report_file}")
        
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)