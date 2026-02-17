import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union
from datetime import date
from enum import Enum

# -----------------------------------------------------------------------------
# Immutable Configuration Definitions
# -----------------------------------------------------------------------------

DEFAULT_EXPECTED_COLUMNS: Tuple[str, ...] = ("date", "ticker", "close_price")
DEFAULT_MIN_PRICE: float = 0.01
DEFAULT_MAX_PRICE: float = 10000.0
DEFAULT_MAX_MISSING_PCT: float = 0.05
DEFAULT_MIN_DAYS_PER_TICKER: int = 252


# -----------------------------------------------------------------------------
# Custom Exception Definitions
# -----------------------------------------------------------------------------

class ValidationErrorCode(Enum):
    INVALID_INPUT_TYPE = "E001"
    INVALID_CONFIG_STRUCTURE = "E002"
    MISSING_REQUIRED_CONFIG_KEY = "E003"
    INVALID_COLUMN_TYPE = "E004"
    INVALID_DATE_FORMAT = "E005"
    INVALID_NUMERIC_RANGE = "E006"


class DataValidationError(Exception):
    def __init__(self, code: ValidationErrorCode, message: str) -> None:
        self.code: str = code.value
        self.message: str = message
        super().__init__(f"[{self.code}] {message}")


# -----------------------------------------------------------------------------
# Typed Data Structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationStats:
    total_rows: int
    total_tickers: int
    date_range: Tuple[str, str]
    missing_by_ticker: Dict[str, float]


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    errors: Tuple[str, ...]
    warnings: Tuple[str, ...]
    stats: ValidationStats


@dataclass(frozen=True)
class PriceRangeConfig:
    min: float
    max: float


@dataclass(frozen=True)
class DateRangeConfig:
    expected_start: str
    expected_end: str
    allow_missing_days: bool
    max_missing_days_pct: float


@dataclass(frozen=True)
class TickerConfig:
    expected: Tuple[str, ...]
    min_days_per_ticker: int


@dataclass(frozen=True)
class MissingDataConfig:
    max_missing_per_ticker_pct: float
    max_total_missing_pct: float


@dataclass(frozen=True)
class BusinessRulesConfig:
    no_negative_prices: bool
    no_zero_prices: bool
    no_duplicate_dates_per_ticker: bool


@dataclass(frozen=True)
class FullValidationConfig:
    expected_columns: Tuple[str, ...]
    price_range: PriceRangeConfig
    date_range: DateRangeConfig
    tickers: TickerConfig
    missing_data: MissingDataConfig
    business_rules: BusinessRulesConfig


# -----------------------------------------------------------------------------
# Pure Core Logic
# -----------------------------------------------------------------------------

def _parse_config(raw_config: Dict[str, Any]) -> FullValidationConfig:
    try:
        cols = tuple(raw_config.get("expected_columns", DEFAULT_EXPECTED_COLUMNS))
        
        pr_raw = raw_config.get("price_range", {})
        price_range = PriceRangeConfig(
            min=float(pr_raw.get("min", DEFAULT_MIN_PRICE)),
            max=float(pr_raw.get("max", DEFAULT_MAX_PRICE))
        )

        dr_raw = raw_config.get("date_range", {})
        date_range = DateRangeConfig(
            expected_start=str(dr_raw.get("expected_start", "2000-01-01")),
            expected_end=str(dr_raw.get("expected_end", "2100-01-01")),
            allow_missing_days=bool(dr_raw.get("allow_missing_days", False)),
            max_missing_days_pct=float(dr_raw.get("max_missing_days_pct", 0.05))
        )

        tk_raw = raw_config.get("tickers", {})
        tickers = TickerConfig(
            expected=tuple(tk_raw.get("expected", [])),
            min_days_per_ticker=int(tk_raw.get("min_days_per_ticker", DEFAULT_MIN_DAYS_PER_TICKER))
        )

        md_raw = raw_config.get("missing_data", {})
        missing_data = MissingDataConfig(
            max_missing_per_ticker_pct=float(md_raw.get("max_missing_per_ticker_pct", 0.10)),
            max_total_missing_pct=float(md_raw.get("max_total_missing_pct", 0.05))
        )

        br_raw = raw_config.get("business_rules", {})
        business_rules = BusinessRulesConfig(
            no_negative_prices=bool(br_raw.get("no_negative_prices", True)),
            no_zero_prices=bool(br_raw.get("no_zero_prices", True)),
            no_duplicate_dates_per_ticker=bool(br_raw.get("no_duplicate_dates_per_ticker", True))
        )

        return FullValidationConfig(
            expected_columns=cols,
            price_range=price_range,
            date_range=date_range,
            tickers=tickers,
            missing_data=missing_data,
            business_rules=business_rules
        )
    except (TypeError, ValueError) as e:
        raise DataValidationError(ValidationErrorCode.INVALID_CONFIG_STRUCTURE, f"Config parsing failed: {e}")


def _validate_schema(df: pd.DataFrame, expected_columns: Tuple[str, ...]) -> Tuple[bool, List[str]]:
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        return False, [f"Missing required columns: {', '.join(missing_cols)}"]
    return True, []


def _validate_price_range(df: pd.DataFrame, config: PriceRangeConfig) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    
    if "close_price" not in df.columns:
        return True, [], [] # Schema check handles missing columns
    
    prices = df["close_price"]
    
    if config.no_negative_prices and (prices < 0).any():
        errors.append("Negative prices detected in 'close_price' column.")
        
    if config.no_zero_prices and (prices == 0).any():
        errors.append("Zero prices detected in 'close_price' column.")

    out_of_bounds = prices[(prices < config.min) | (prices > config.max)]
    if not out_of_bounds.empty:
        errors.append(f"{len(out_of_bounds)} records have prices outside range [{config.min}, {config.max}].")

    return len(errors) == 0, errors, warnings


def _validate_dates(df: pd.DataFrame, config: DateRangeConfig) -> Tuple[bool, List[str], List[str], Tuple[str, str]]:
    errors: List[str] = []
    warnings: List[str] = []
    
    if "date" not in df.columns:
        return True, [], [], ("", "")

    try:
        # Ensure date column is datetime for comparison, but do not mutate original df
        dates = pd.to_datetime(df["date"], errors='raise')
    except Exception as e:
        raise DataValidationError(ValidationErrorCode.INVALID_DATE_FORMAT, f"Date column invalid: {e}")

    if dates.empty:
        return True, [], [], ("", "")

    min_date = dates.min()
    max_date = dates.max()
    min_date_str = min_date.strftime("%Y-%m-%d")
    max_date_str = max_date.strftime("%Y-%m-%d")

    exp_start = pd.to_datetime(config.expected_start)
    exp_end = pd.to_datetime(config.expected_end)

    if min_date < exp_start:
        errors.append(f"Data starts before expected range: {min_date_str} < {config.expected_start}")
    elif min_date > exp_start:
        warnings.append(f"Date range shorter than expected start: got {min_date_str}")

    if max_date > exp_end:
        errors.append(f"Data ends after expected range: {max_date_str} > {config.expected_end}")
    elif max_date < exp_end:
        warnings.append(f"Date range shorter than expected end: got {max_date_str}")

    # Continuity check per ticker if required
    if not config.allow_missing_days:
        # Group by ticker and check for gaps
        # This is a simplified continuity check assuming sorted or using diff
        # For strictness, we check if any ticker has gaps > 1 business day roughly
        # Given constraints, we skip complex holiday logic and just check row count vs date span if needed
        # However, spec says "allow_missing_days": false implies strict continuity.
        # Implementing a simple gap check:
        grouped = dates.groupby(df["ticker"])
        for ticker, group_dates in grouped:
            sorted_dates = group_dates.sort_values()
            diffs = sorted_dates.diff().dt.days.dropna()
            # Allow weekends (2 or 3 days), flag anything significantly larger without specific calendar
            # Strict interpretation: any gap > 3 days is suspicious if allow_missing_days is False
            # But financial data often skips weekends. 
            # To remain deterministic and simple: if allow_missing_days is False, we expect no gaps > 3 days.
            if (diffs > 3).any():
                errors.append(f"Ticker '{ticker}' has unexpected gaps in dates.")
                break

    return len(errors) == 0, errors, warnings, (min_date_str, max_date_str)


def _validate_missing_data(df: pd.DataFrame, config: MissingDataConfig, ticker_config: TickerConfig) -> Tuple[Dict[str, float], List[str]]:
    if "ticker" not in df.columns or "close_price" not in df.columns:
        return {}, []

    errors: List[str] = []
    missing_stats: Dict[str, float] = {}
    
    total_rows = len(df)
    total_missing = df["close_price"].isna().sum()
    
    if total_rows > 0:
        total_missing_pct = total_missing / total_rows
        if total_missing_pct > config.max_total_missing_pct:
            errors.append(f"Total missing data ({total_missing_pct:.2%}) exceeds limit ({config.max_total_missing_pct:.2%}).")

    tickers = df["ticker"].unique()
    for ticker in tickers:
        subset = df[df["ticker"] == ticker]
        count = len(subset)
        missing = subset["close_price"].isna().sum()
        pct = (missing / count) if count > 0 else 0.0
        missing_stats[ticker] = pct

        if pct > config.max_missing_per_ticker_pct:
            errors.append(f"Ticker '{ticker}' has {pct:.2%} missing data (exceeds {config.max_missing_per_ticker_pct:.2%} limit).")
        
        # Check min days
        valid_days = count - missing
        if valid_days < ticker_config.min_days_per_ticker:
            errors.append(f"Ticker '{ticker}' has only {valid_days} valid days (requires {ticker_config.min_days_per_ticker}).")

    return missing_stats, errors


def _validate_ticker_presence(df: pd.DataFrame, expected_tickers: Tuple[str, ...]) -> List[str]:
    if "ticker" not in df.columns:
        return ["Cannot validate tickers: 'ticker' column missing."]
    
    present_tickers = set(df["ticker"].unique())
    expected_set = set(expected_tickers)
    
    missing = expected_set - present_tickers
    extra = present_tickers - expected_set
    
    errors = []
    for t in sorted(missing):
        errors.append(f"Missing expected ticker: '{t}' not found in data.")
    
    # Extra tickers are usually warnings, but spec says "returns invalid tickers" in function description
    # The run_all_validations returns errors/warnings. We treat missing as error.
    # Extra tickers might be acceptable depending on strictness, but let's add as warning if not in expected
    # Actually, spec interface says `validate_ticker_count` returns invalid tickers.
    # We will integrate this logic into the main flow.
    
    return errors


def _check_duplicates(df: pd.DataFrame, rule_enabled: bool) -> List[str]:
    if not rule_enabled:
        return []
    
    if "date" not in df.columns or "ticker" not in df.columns:
        return []
        
    duplicates = df.duplicated(subset=["date", "ticker"], keep=False)
    if duplicates.any():
        count = duplicates.sum()
        return [f"Found {count} duplicate entries for (date, ticker) pairs."]
    
    return []


def run_all_validations(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    # Input Validation
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(ValidationErrorCode.INVALID_INPUT_TYPE, "Input 'df' must be a pandas DataFrame.")
    
    parsed_config = _parse_config(config)
    
    all_errors: List[str] = []
    all_warnings: List[str] = []
    
    # 1. Schema
    schema_ok, schema_errs = _validate_schema(df, parsed_config.expected_columns)
    all_errors.extend(schema_errs)
    
    # If schema fails critically, we might not proceed, but spec says return dict with errors.
    # We continue to gather as many errors as possible unless columns are missing for specific checks.
    
    # 2. Price Range
    if "close_price" in df.columns:
        price_ok, price_errs, price_warns = _validate_price_range(df, parsed_config.price_range)
        all_errors.extend(price_errs)
        all_warnings.extend(price_warns)
    
    # 3. Dates
    date_ok, date_errs, date_warns, date_range_str = _validate_dates(df, parsed_config.date_range)
    all_errors.extend(date_errs)
    all_warnings.extend(date_warns)
    
    # 4. Missing Data & Ticker Counts
    missing_stats, missing_errs = _validate_missing_data(df, parsed_config.missing_data, parsed_config.tickers)
    all_errors.extend(missing_errs)
    
    # 5. Ticker Presence
    ticker_errs = _validate_ticker_presence(df, parsed_config.tickers.expected)
    all_errors.extend(ticker_errs)
    
    # 6. Business Rules (Duplicates)
    dup_errs = _check_duplicates(df, parsed_config.business_rules.no_duplicate_dates_per_ticker)
    all_errors.extend(dup_errs)
    
    # Compile Stats
    total_rows = len(df)
    total_tickers = df["ticker"].nunique() if "ticker" in df.columns else 0
    
    stats = ValidationStats(
        total_rows=total_rows,
        total_tickers=total_tickers,
        date_range=date_range_str if "date" in df.columns else ("", ""),
        missing_by_ticker=missing_stats
    )
    
    passed = len(all_errors) == 0
    
    return {
        "passed": passed,
        "errors": tuple(all_errors),
        "warnings": tuple(all_warnings),
        "stats": {
            "total_rows": stats.total_rows,
            "total_tickers": stats.total_tickers,
            "date_range": list(stats.date_range),
            "missing_by_ticker": stats.missing_by_ticker
        }
    }


# -----------------------------------------------------------------------------
# Public Interface Wrappers (as requested by specific function signatures)
# -----------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(ValidationErrorCode.INVALID_INPUT_TYPE, "Input 'df' must be a pandas DataFrame.")
    if not isinstance(expected_columns, list):
        raise DataValidationError(ValidationErrorCode.INVALID_INPUT_TYPE, "expected_columns must be a list.")
    
    missing = [c for c in expected_columns if c not in df.columns]
    return len(missing) == 0


def validate_price_range(df: pd.DataFrame, min_price: float, max_price: float) -> bool:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(ValidationErrorCode.INVALID_INPUT_TYPE, "Input 'df' must be a pandas DataFrame.")
    if "close_price" not in df.columns:
        return False
    
    prices = df["close_price"]
    if (prices < min_price).any() or (prices > max_price).any():
        return False
    return True


def validate_dates(df: pd.DataFrame, start_date: str, end_date: str) -> bool:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(ValidationErrorCode.INVALID_INPUT_TYPE, "Input 'df' must be a pandas DataFrame.")
    if "date" not in df.columns:
        return False
    
    try:
        dates = pd.to_datetime(df["date"])
    except Exception:
        return False
        
    min_d = dates.min()
    max_d = dates.max()
    
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    
    return min_d >= s and max_d <= e


def validate_missing_data(df: pd.DataFrame, max_missing_pct: float) -> Dict[str, float]:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(ValidationErrorCode.INVALID_INPUT_TYPE, "Input 'df' must be a pandas DataFrame.")
    if "ticker" not in df.columns or "close_price" not in df.columns:
        return {}
    
    stats: Dict[str, float] = {}
    tickers = df["ticker"].unique()
    
    for ticker in tickers:
        subset = df[df["ticker"] == ticker]
        total = len(subset)
        missing = subset["close_price"].isna().sum()
        if total > 0:
            stats[ticker] = missing / total
            
    return stats


def validate_ticker_count(df: pd.DataFrame, expected_tickers: List[str], min_days: int) -> List[str]:
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(ValidationErrorCode.INVALID_INPUT_TYPE, "Input 'df' must be a pandas DataFrame.")
    if "ticker" not in df.columns:
        return expected_tickers # All missing
    
    invalid: List[str] = []
    present = df["ticker"].unique()
    
    # Check missing expected
    for t in expected_tickers:
        if t not in present:
            invalid.append(t)
            
    # Check count per present ticker
    for t in present:
        count = len(df[df["ticker"] == t])
        if count < min_days:
            if t not in invalid: # Avoid double reporting if already missing
                invalid.append(t)
                
    return invalid


# -----------------------------------------------------------------------------
# Execution Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal smoke test with synthetic data to verify structure
    import sys
    
    data = {
        "date": ["2022-01-01", "2022-01-02", "2022-01-01", "2022-01-02"],
        "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "close_price": [150.0, 151.0, 280.0, 285.0]
    }
    df_test = pd.DataFrame(data)
    
    config_test = {
        "expected_columns": ["date", "ticker", "close_price"],
        "price_range": {"min": 1.0, "max": 1000.0},
        "date_range": {"expected_start": "2022-01-01", "expected_end": "2022-12-31", "allow_missing_days": True, "max_missing_days_pct": 0.1},
        "tickers": {"expected": ["AAPL", "MSFT"], "min_days_per_ticker": 2},
        "missing_data": {"max_missing_per_ticker_pct": 0.1, "max_total_missing_pct": 0.1},
        "business_rules": {"no_negative_prices": True, "no_zero_prices": True, "no_duplicate_dates_per_ticker": True}
    }
    
    result = run_all_validations(df_test, config_test)
    
    if result["passed"]:
        print("VALIDATION PASSED")
        sys.exit(0)
    else:
        print("VALIDATION FAILED")
        for err in result["errors"]:
            print(f"ERROR: {err}")
        sys.exit(1)