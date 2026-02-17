import csv
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Sequence, Union, Optional
from dataclasses import dataclass
import math

# Immutable configuration
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_N_STOCKS = 10
DEFAULT_DAYS = 504
DEFAULT_PRICE_START = 100.0
DEFAULT_VOLATILITY = 0.02
DEFAULT_DRIFT = 0.0005
DEFAULT_SEED = 42
DEFAULT_MIN_FLAT_DAYS = 3
DEFAULT_MAX_FLAT_DAYS = 10
DEFAULT_N_FLAT_PERIODS = 5

# Custom exceptions
class GeneratorError(Exception):
    """Base exception for generator errors."""
    pass

class InvalidParameterError(GeneratorError):
    """Exception for invalid input parameters."""
    def __init__(self, parameter: str, message: str, code: str):
        self.parameter = parameter
        self.message = message
        self.code = code
        super().__init__(f"[{code}] Parameter '{parameter}': {message}")

class DataGenerationError(GeneratorError):
    """Exception for data generation failures."""
    def __init__(self, message: str, code: str):
        self.message = message
        self.code = code
        super().__init__(f"[{code}] {message}")

# Typed data structures
@dataclass(frozen=True)
class PricePoint:
    date: str
    ticker: str
    close_price: float

@dataclass(frozen=True)
class FlatPeriod:
    ticker: str
    start_idx: int
    end_idx: int

# Pure core logic
def _validate_generation_params(
    n_stocks: int,
    days: int,
    start_date: str,
    seed: Optional[int]
) -> None:
    """Validate input parameters for data generation."""
    if n_stocks <= 0:
        raise InvalidParameterError(
            "n_stocks",
            f"must be positive, got {n_stocks}",
            "INVALID_N_STOCKS"
        )
    if days <= 0:
        raise InvalidParameterError(
            "days",
            f"must be positive, got {days}",
            "INVALID_DAYS"
        )
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        raise InvalidParameterError(
            "start_date",
            f"must be YYYY-MM-DD format, got {start_date}",
            "INVALID_DATE_FORMAT"
        )
    if seed is not None and seed < 0:
        raise InvalidParameterError(
            "seed",
            f"must be non-negative, got {seed}",
            "INVALID_SEED"
        )

def _validate_flat_period_params(
    n_flat_periods: int,
    min_flat_days: int,
    max_flat_days: int,
    total_days: int
) -> None:
    """Validate parameters for flat period injection."""
    if n_flat_periods < 0:
        raise InvalidParameterError(
            "n_flat_periods",
            f"must be non-negative, got {n_flat_periods}",
            "INVALID_N_FLAT_PERIODS"
        )
    if min_flat_days <= 0:
        raise InvalidParameterError(
            "min_flat_days",
            f"must be positive, got {min_flat_days}",
            "INVALID_MIN_FLAT_DAYS"
        )
    if max_flat_days < min_flat_days:
        raise InvalidParameterError(
            "max_flat_days",
            f"must be >= min_flat_days ({min_flat_days}), got {max_flat_days}",
            "INVALID_MAX_FLAT_DAYS"
        )
    if max_flat_days > total_days:
        raise InvalidParameterError(
            "max_flat_days",
            f"cannot exceed total days ({total_days}), got {max_flat_days}",
            "MAX_FLAT_DAYS_EXCEEDS_TOTAL"
        )

def _generate_price_series(
    days: int,
    start_price: float,
    volatility: float,
    drift: float,
    rng: random.Random
) -> Tuple[float, ...]:
    """Generate a geometric Brownian motion price series."""
    prices = [start_price]
    for _ in range(1, days):
        prev_price = prices[-1]
        daily_return = rng.gauss(drift, volatility)
        next_price = prev_price * math.exp(daily_return)
        if next_price <= 0.0 or math.isnan(next_price) or math.isinf(next_price):
            raise DataGenerationError(
                f"numerical instability detected: price={next_price}",
                "NUMERICAL_INSTABILITY"
            )
        prices.append(next_price)
    return tuple(prices)

def _generate_dates(start_date: str, days: int) -> Tuple[str, ...]:
    """Generate a sequence of dates."""
    current = datetime.strptime(start_date, "%Y-%m-%d")
    dates = []
    for _ in range(days):
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return tuple(dates)

def _select_flat_periods(
    ticker: str,
    total_days: int,
    n_periods: int,
    min_days: int,
    max_days: int,
    rng: random.Random
) -> Tuple[FlatPeriod, ...]:
    """Select random non-overlapping flat periods for a ticker."""
    if n_periods == 0:
        return ()
    
    periods = []
    max_attempts = 1000
    attempts = 0
    
    while len(periods) < n_periods and attempts < max_attempts:
        length = rng.randint(min_days, max_days)
        max_start = total_days - length
        if max_start < 0:
            attempts += 1
            continue
            
        start = rng.randint(0, max_start)
        end = start + length - 1
        
        # Check overlap with existing periods
        overlap = False
        for p in periods:
            if not (end < p.start_idx or start > p.end_idx):
                overlap = True
                break
        
        if not overlap:
            periods.append(FlatPeriod(ticker=ticker, start_idx=start, end_idx=end))
        attempts += 1
    
    if len(periods) < n_periods:
        raise DataGenerationError(
            f"could not place {n_periods} non-overlapping periods after {max_attempts} attempts",
            "PERIOD_PLACEMENT_FAILED"
        )
    
    return tuple(sorted(periods, key=lambda p: p.start_idx))

def _flatten_periods(
    prices: Sequence[float],
    periods: Tuple[FlatPeriod, ...]
) -> Tuple[float, ...]:
    """Set prices to constant values during flat periods."""
    result = list(prices)
    for period in periods:
        if period.start_idx >= len(prices) or period.end_idx >= len(prices):
            raise DataGenerationError(
                f"period indices out of bounds: [{period.start_idx}, {period.end_idx}] for length {len(prices)}",
                "PERIOD_INDEX_OUT_OF_BOUNDS"
            )
        flat_price = prices[period.start_idx]
        for i in range(period.start_idx, period.end_idx + 1):
            result[i] = flat_price
    return tuple(result)

def generate_portfolio_prices(
    n_stocks: int = DEFAULT_N_STOCKS,
    days: int = DEFAULT_DAYS,
    start_date: str = DEFAULT_START_DATE,
    seed: Optional[int] = DEFAULT_SEED
) -> Tuple[PricePoint, ...]:
    """
    Generate synthetic portfolio price data.
    
    Args:
        n_stocks: Number of stocks in portfolio
        days: Number of trading days
        start_date: Start date in YYYY-MM-DD format
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of PricePoint objects
    
    Raises:
        InvalidParameterError: If parameters are invalid
        DataGenerationError: If data generation fails
    """
    # Validate inputs
    _validate_generation_params(n_stocks, days, start_date, seed)
    
    # Initialize RNG
    rng = random.Random(seed)
    
    # Generate dates
    dates = _generate_dates(start_date, days)
    
    # Generate tickers
    tickers = tuple(f"STOCK_{i+1:03d}" for i in range(n_stocks))
    
    # Generate prices for each stock
    price_points = []
    for ticker in tickers:
        # Use deterministic but different seeds for each stock
        stock_seed = rng.randint(0, 2**32 - 1)
        stock_rng = random.Random(stock_seed)
        
        prices = _generate_price_series(
            days=days,
            start_price=DEFAULT_PRICE_START,
            volatility=DEFAULT_VOLATILITY,
            drift=DEFAULT_DRIFT,
            rng=stock_rng
        )
        
        for date_idx, date in enumerate(dates):
            price_points.append(PricePoint(
                date=date,
                ticker=ticker,
                close_price=round(prices[date_idx], 4)
            ))
    
    return tuple(price_points)

def inject_flat_periods(
    data: Sequence[PricePoint],
    n_flat_periods: int = DEFAULT_N_FLAT_PERIODS,
    min_flat_days: int = DEFAULT_MIN_FLAT_DAYS,
    max_flat_days: int = DEFAULT_MAX_FLAT_DAYS,
    seed: Optional[int] = DEFAULT_SEED
) -> Tuple[PricePoint, ...]:
    """
    Inject flat price periods into existing data.
    
    Args:
        data: Original price points
        n_flat_periods: Number of flat periods per stock
        min_flat_days: Minimum days per flat period
        max_flat_days: Maximum days per flat period
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of PricePoint objects with flat periods
    
    Raises:
        InvalidParameterError: If parameters are invalid
        DataGenerationError: If flat period injection fails
    """
    if not data:
        return data
    
    # Validate inputs
    tickers = tuple(sorted(set(p.ticker for p in data)))
    dates_per_ticker = len(data) // len(tickers)
    
    _validate_flat_period_params(
        n_flat_periods,
        min_flat_days,
        max_flat_days,
        dates_per_ticker
    )
    
    # Initialize RNG
    rng = random.Random(seed)
    
    # Organize data by ticker
    data_by_ticker = {}
    for point in data:
        if point.ticker not in data_by_ticker:
            data_by_ticker[point.ticker] = []
        data_by_ticker[point.ticker].append(point)
    
    # Validate data integrity
    for ticker, points in data_by_ticker.items():
        if len(points) != dates_per_ticker:
            raise DataGenerationError(
                f"inconsistent data for {ticker}: expected {dates_per_ticker} points, got {len(points)}",
                "INCONSISTENT_DATA"
            )
        # Verify dates are in order
        for i in range(1, len(points)):
            if points[i].date <= points[i-1].date:
                raise DataGenerationError(
                    f"dates not strictly increasing for {ticker}",
                    "INVALID_DATE_ORDER"
                )
    
    # Process each ticker
    result = []
    for ticker, points in data_by_ticker.items():
        ticker_rng = random.Random(rng.randint(0, 2**32 - 1))
        
        # Select flat periods
        periods = _select_flat_periods(
            ticker=ticker,
            total_days=len(points),
            n_periods=n_flat_periods,
            min_days=min_flat_days,
            max_days=max_flat_days,
            rng=ticker_rng
        )
        
        # Extract prices
        prices = tuple(p.close_price for p in points)
        
        # Flatten periods
        flattened_prices = _flatten_periods(prices, periods)
        
        # Create new price points
        for idx, point in enumerate(points):
            result.append(PricePoint(
                date=point.date,
                ticker=point.ticker,
                close_price=round(flattened_prices[idx], 4)
            ))
    
    return tuple(result)

def save_to_csv(
    data: Sequence[PricePoint],
    filepath: str
) -> None:
    """
    Save price points to CSV file.
    
    Args:
        data: Sequence of PricePoint objects
        filepath: Path to output CSV file
    
    Raises:
        InvalidParameterError: If parameters are invalid
        DataGenerationError: If file write fails
    """
    if not data:
        raise InvalidParameterError(
            "data",
            "cannot be empty",
            "EMPTY_DATA"
        )
    
    if not filepath:
        raise InvalidParameterError(
            "filepath",
            "cannot be empty",
            "EMPTY_FILEPATH"
        )
    
    if not filepath.endswith('.csv'):
        raise InvalidParameterError(
            "filepath",
            "must end with .csv",
            "INVALID_FILE_EXTENSION"
        )
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'ticker', 'close_price'])
            for point in data:
                writer.writerow([point.date, point.ticker, point.close_price])
    except IOError as e:
        raise DataGenerationError(
            f"failed to write to {filepath}: {str(e)}",
            "FILE_WRITE_FAILED"
        )

# Entry point
if __name__ == "__main__":
    # Generate base data
    data = generate_portfolio_prices(
        n_stocks=10,
        days=504,
        start_date="2022-01-01",
        seed=42
    )
    
    # Inject flat periods
    modified_data = inject_flat_periods(
        data=data,
        n_flat_periods=5,
        min_flat_days=3,
        max_flat_days=10,
        seed=42
    )
    
    # Save to CSV
    save_to_csv(modified_data, "data/raw/prices.csv")