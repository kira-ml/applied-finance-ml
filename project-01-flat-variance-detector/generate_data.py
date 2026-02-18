import csv
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
import sys
from typing import List, Dict, Tuple, Any

@dataclass(frozen=True)
class Config:
    n_stocks: int
    days: int
    start_date: str
    flat_periods: int
    random_seed: int
    output_dir: str = "data/raw"

    def __post_init__(self) -> None:
        if not isinstance(self.n_stocks, int) or self.n_stocks <= 0:
            raise ValueError("n_stocks must be positive integer")
        if not isinstance(self.days, int) or self.days <= 0:
            raise ValueError("days must be positive integer")
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
        except ValueError:
            raise ValueError("start_date must be YYYY-MM-DD")
        if not isinstance(self.flat_periods, int) or self.flat_periods < 0:
            raise ValueError("flat_periods must be non-negative integer")
        if not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be integer")
        if not isinstance(self.output_dir, str):
            raise ValueError("output_dir must be string")

class DataGenerationError(Exception):
    pass

class ConfigurationError(DataGenerationError):
    pass

class FileWriteError(DataGenerationError):
    pass

class InputValidationError(DataGenerationError):
    pass

def validate_inputs(config: Config) -> None:
    if not isinstance(config, Config):
        raise InputValidationError(f"config must be Config, got {type(config).__name__}")
    try:
        datetime.strptime(config.start_date, "%Y-%m-%d")
    except ValueError:
        raise InputValidationError("start_date must be valid YYYY-MM-DD")
    if config.n_stocks <= 0:
        raise InputValidationError("n_stocks must be positive")
    if config.days <= 0:
        raise InputValidationError("days must be positive")
    if config.flat_periods < 0:
        raise InputValidationError("flat_periods must be non-negative")
    if config.flat_periods > config.days // 5:
        raise InputValidationError("flat_periods too many for given days")

def generate_tickers(n: int) -> List[str]:
    if not isinstance(n, int):
        raise InputValidationError(f"n must be int, got {type(n).__name__}")
    if n <= 0:
        raise InputValidationError("n must be positive")
    base_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]
    tickers = []
    for i in range(n):
        if i < len(base_names):
            tickers.append(base_names[i])
        else:
            tickers.append(f"STOCK{i+1}")
    assert len(tickers) == n, f"Expected {n} tickers, got {len(tickers)}"
    return tickers

def generate_dates(start_date: str, days: int) -> List[str]:
    if not isinstance(start_date, str):
        raise InputValidationError(f"start_date must be str, got {type(start_date).__name__}")
    if not isinstance(days, int):
        raise InputValidationError(f"days must be int, got {type(days).__name__}")
    if days <= 0:
        raise InputValidationError("days must be positive")
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = []
    for i in range(days):
        current_date = start + timedelta(days=i)
        if current_date.weekday() < 5:
            dates.append(current_date.strftime("%Y-%m-%d"))
    assert len(dates) <= days, f"Expected <= {days} dates, got {len(dates)}"
    return dates

def assign_flat_periods(tickers: List[str], dates: List[str], num_flat_periods: int, rng: random.Random) -> Dict[str, List[Tuple[int, int]]]:
    if not isinstance(tickers, list):
        raise InputValidationError(f"tickers must be list, got {type(tickers).__name__}")
    if not isinstance(dates, list):
        raise InputValidationError(f"dates must be list, got {type(dates).__name__}")
    if not isinstance(num_flat_periods, int):
        raise InputValidationError(f"num_flat_periods must be int, got {type(num_flat_periods).__name__}")
    if num_flat_periods < 0:
        raise InputValidationError("num_flat_periods must be non-negative")
    if not isinstance(rng, random.Random):
        raise InputValidationError(f"rng must be random.Random, got {type(rng).__name__}")
    
    flat_assignments: Dict[str, List[Tuple[int, int]]] = {}
    if num_flat_periods == 0 or not dates:
        for ticker in tickers:
            flat_assignments[ticker] = []
        return flat_assignments
    
    max_start_idx = len(dates) - 5
    if max_start_idx < 0:
        for ticker in tickers:
            flat_assignments[ticker] = []
        return flat_assignments
    
    periods_per_ticker = num_flat_periods // len(tickers)
    remainder = num_flat_periods % len(tickers)
    
    for i, ticker in enumerate(tickers):
        ticker_periods = periods_per_ticker + (1 if i < remainder else 0)
        ticker_flats: List[Tuple[int, int]] = []
        
        for _ in range(ticker_periods):
            if max_start_idx < 0:
                break
            start_idx = rng.randint(0, max_start_idx)
            end_idx = min(start_idx + rng.randint(5, min(10, len(dates) - start_idx - 1)), len(dates) - 1)
            actual_start = start_idx
            actual_end = end_idx
            ticker_flats.append((actual_start, actual_end))
        
        flat_assignments[ticker] = ticker_flats
    
    return flat_assignments

def generate_price_series(dates: List[str], flat_periods: List[Tuple[int, int]], rng: random.Random) -> List[float]:
    if not isinstance(dates, list):
        raise InputValidationError(f"dates must be list, got {type(dates).__name__}")
    if not isinstance(flat_periods, list):
        raise InputValidationError(f"flat_periods must be list, got {type(flat_periods).__name__}")
    if not isinstance(rng, random.Random):
        raise InputValidationError(f"rng must be random.Random, got {type(rng).__name__}")
    
    prices: List[float] = []
    base_price = 100.0
    current_price = base_price
    
    is_flat = [False] * len(dates)
    for start_idx, end_idx in flat_periods:
        if start_idx < 0 or end_idx >= len(dates) or start_idx > end_idx:
            raise ValueError(f"Invalid flat period indices: {start_idx}-{end_idx}")
        for idx in range(start_idx, end_idx + 1):
            is_flat[idx] = True
    
    for i in range(len(dates)):
        if is_flat[i]:
            noise = 0.0
        else:
            change_pct = rng.uniform(-0.03, 0.03)
            max_change = current_price * 0.03
            if abs(change_pct * current_price) > max_change:
                change_pct = (max_change / current_price) * (1 if change_pct > 0 else -1)
            current_price = current_price * (1.0 + change_pct)
            if current_price <= 0.01:
                current_price = 0.01
        
        prices.append(round(current_price, 2))
        
        if not is_flat[i] and i < len(dates) - 1 and not is_flat[i + 1]:
            next_change = rng.uniform(-0.02, 0.02)
            next_price = current_price * (1.0 + next_change)
            if next_price <= 0.01:
                next_price = 0.01
            current_price = next_price
    
    assert len(prices) == len(dates), f"Expected {len(dates)} prices, got {len(prices)}"
    return prices

def ensure_directory_exists(directory: str) -> None:
    if not isinstance(directory, str):
        raise FileWriteError(f"directory must be string, got {type(directory).__name__}")
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            raise FileWriteError(f"Failed to create directory {directory}: {e}")
    elif not os.path.isdir(directory):
        raise FileWriteError(f"Path exists but is not a directory: {directory}")

def write_prices_csv(filepath: str, data: List[Dict[str, Any]]) -> None:
    if not isinstance(filepath, str):
        raise FileWriteError(f"filepath must be string, got {type(filepath).__name__}")
    if not isinstance(data, list):
        raise FileWriteError(f"data must be list, got {type(data).__name__}")
    
    try:
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['date', 'ticker', 'close_price']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    except (OSError, csv.Error) as e:
        raise FileWriteError(f"Failed to write prices CSV: {e}")

def write_ground_truth_csv(filepath: str, data: List[Dict[str, Any]]) -> None:
    if not isinstance(filepath, str):
        raise FileWriteError(f"filepath must be string, got {type(filepath).__name__}")
    if not isinstance(data, list):
        raise FileWriteError(f"data must be list, got {type(data).__name__}")
    
    try:
        with open(filepath, 'w', newline='') as f:
            fieldnames = ['ticker', 'start_date', 'end_date', 'actually_flat']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    except (OSError, csv.Error) as e:
        raise FileWriteError(f"Failed to write ground truth CSV: {e}")

def generate_data(config: Config) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    validate_inputs(config)
    
    rng = random.Random(config.random_seed)
    
    tickers = generate_tickers(config.n_stocks)
    dates = generate_dates(config.start_date, config.days)
    
    if not dates:
        return [], []
    
    flat_assignments = assign_flat_periods(tickers, dates, config.flat_periods, rng)
    
    prices_data: List[Dict[str, Any]] = []
    ground_truth_data: List[Dict[str, Any]] = []
    
    for ticker in tickers:
        ticker_flats = flat_assignments.get(ticker, [])
        
        for start_idx, end_idx in ticker_flats:
            ground_truth_data.append({
                'ticker': ticker,
                'start_date': dates[start_idx],
                'end_date': dates[end_idx],
                'actually_flat': 1
            })
        
        prices = generate_price_series(dates, ticker_flats, rng)
        
        for i, date in enumerate(dates):
            prices_data.append({
                'date': date,
                'ticker': ticker,
                'close_price': prices[i]
            })
    
    return prices_data, ground_truth_data

def main() -> None:
    config = Config(
        n_stocks=10,
        days=252,
        start_date="2023-01-01",
        flat_periods=15,
        random_seed=42
    )
    
    try:
        prices_data, ground_truth_data = generate_data(config)
        
        ensure_directory_exists(config.output_dir)
        
        prices_file = os.path.join(config.output_dir, "prices.csv")
        write_prices_csv(prices_file, prices_data)
        
        ground_truth_file = os.path.join(config.output_dir, "ground_truth.csv")
        write_ground_truth_csv(ground_truth_file, ground_truth_data)
        
        print(f"Generated {len(prices_data)} price records")
        print(f"Generated {len(ground_truth_data)} ground truth records")
        
    except (DataGenerationError, FileWriteError, InputValidationError, ValueError) as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    initialized = False
    if not initialized:
        main()
        initialized = True