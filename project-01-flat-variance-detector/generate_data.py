import csv
import random
import sys
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

RANDOM_SEED: int = 42

class InvalidParameterError(ValueError):
    pass

class ConfigurationError(Exception):
    pass

class FlatPeriod:
    __slots__ = ('stock', 'start_idx', 'end_idx', 'start_date', 'end_date')
    def __init__(self, stock: str, start_idx: int, end_idx: int, start_date: str, end_date: str):
        self.stock = stock
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.start_date = start_date
        self.end_date = end_date

def _validate_generation_params(n_stocks: int, days: int, start_date: str) -> None:
    if not isinstance(n_stocks, int):
        raise InvalidParameterError(f"n_stocks must be int, got {type(n_stocks)}")
    if n_stocks < 1 or n_stocks > 100:
        raise InvalidParameterError(f"n_stocks must be between 1 and 100, got {n_stocks}")
    if not isinstance(days, int):
        raise InvalidParameterError(f"days must be int, got {type(days)}")
    if days < 10 or days > 2000:
        raise InvalidParameterError(f"days must be between 10 and 2000, got {days}")
    if not isinstance(start_date, str):
        raise InvalidParameterError(f"start_date must be str, got {type(start_date)}")
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError as e:
        raise InvalidParameterError(f"start_date must be YYYY-MM-DD format: {e}")

def _validate_flat_params(n_flat_periods: int, min_flat_days: int, max_flat_days: int) -> None:
    if not isinstance(n_flat_periods, int):
        raise InvalidParameterError(f"n_flat_periods must be int, got {type(n_flat_periods)}")
    if n_flat_periods < 0 or n_flat_periods > 50:
        raise InvalidParameterError(f"n_flat_periods must be between 0 and 50, got {n_flat_periods}")
    if not isinstance(min_flat_days, int):
        raise InvalidParameterError(f"min_flat_days must be int, got {type(min_flat_days)}")
    if min_flat_days < 1 or min_flat_days > 30:
        raise InvalidParameterError(f"min_flat_days must be between 1 and 30, got {min_flat_days}")
    if not isinstance(max_flat_days, int):
        raise InvalidParameterError(f"max_flat_days must be int, got {type(max_flat_days)}")
    if max_flat_days < min_flat_days or max_flat_days > 60:
        raise InvalidParameterError(f"max_flat_days must be >= min_flat_days and <= 60, got {max_flat_days}")

def _validate_noise_params(volatility_range: Tuple[float, float]) -> None:
    if not isinstance(volatility_range, tuple):
        raise InvalidParameterError(f"volatility_range must be tuple, got {type(volatility_range)}")
    if len(volatility_range) != 2:
        raise InvalidParameterError(f"volatility_range must have 2 elements, got {len(volatility_range)}")
    low, high = volatility_range
    if not isinstance(low, (int, float)):
        raise InvalidParameterError(f"volatility_range[0] must be number, got {type(low)}")
    if not isinstance(high, (int, float)):
        raise InvalidParameterError(f"volatility_range[1] must be number, got {type(high)}")
    if low <= 0 or high <= 0 or low > high or high > 0.1:
        raise InvalidParameterError(f"volatility_range values invalid: {volatility_range}")

def generate_portfolio_prices(
    n_stocks: int = 10,
    days: int = 504,
    start_date: str = "2022-01-01"
) -> Tuple[List[Dict[str, Any]], List[str]]:
    _validate_generation_params(n_stocks, days, start_date)
    random.seed(RANDOM_SEED)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates: List[str] = []
    current = start
    for _ in range(days):
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    stock_names: List[str] = [f"STOCK_{i+1:03d}" for i in range(n_stocks)]
    base_prices: Dict[str, float] = {}
    for stock in stock_names:
        base_prices[stock] = random.uniform(50.0, 200.0)
    
    data: List[Dict[str, Any]] = []
    for i, date in enumerate(dates):
        row: Dict[str, Any] = {"date": date}
        for stock in stock_names:
            trend = 1.0 + random.uniform(-0.002, 0.002) * (i / 10.0)
            base_prices[stock] *= trend
            if base_prices[stock] < 10.0:
                base_prices[stock] = 10.0
            if base_prices[stock] > 500.0:
                base_prices[stock] = 500.0
            row[stock] = round(base_prices[stock], 4)
        data.append(row)
    
    return data, stock_names

def inject_flat_periods(
    data: List[Dict[str, Any]],
    stock_names: List[str],
    n_flat_periods: int = 5,
    min_flat_days: int = 3,
    max_flat_days: int = 10
) -> Tuple[List[Dict[str, Any]], List[FlatPeriod]]:
    _validate_flat_params(n_flat_periods, min_flat_days, max_flat_days)
    if not isinstance(data, list):
        raise InvalidParameterError(f"data must be list, got {type(data)}")
    if not data:
        raise InvalidParameterError("data cannot be empty")
    if not isinstance(stock_names, list):
        raise InvalidParameterError(f"stock_names must be list, got {type(stock_names)}")
    
    random.seed(RANDOM_SEED + 1)
    modified_data = [row.copy() for row in data]
    flat_periods: List[FlatPeriod] = []
    days = len(modified_data)
    
    for _ in range(n_flat_periods):
        stock = random.choice(stock_names)
        flat_length = random.randint(min_flat_days, max_flat_days)
        max_start = days - flat_length
        if max_start < 0:
            continue
        start_idx = random.randint(0, max_start)
        end_idx = start_idx + flat_length - 1
        
        if start_idx > 0:
            base_price = modified_data[start_idx - 1][stock]
        else:
            base_price = modified_data[start_idx][stock]
        
        for idx in range(start_idx, end_idx + 1):
            modified_data[idx][stock] = round(base_price, 4)
        
        flat_periods.append(FlatPeriod(
            stock=stock,
            start_idx=start_idx,
            end_idx=end_idx,
            start_date=modified_data[start_idx]["date"],
            end_date=modified_data[end_idx]["date"]
        ))
    
    return modified_data, flat_periods

def add_realistic_noise(
    data: List[Dict[str, Any]],
    stock_names: List[str],
    volatility_range: Tuple[float, float] = (0.005, 0.03)
) -> List[Dict[str, Any]]:
    _validate_noise_params(volatility_range)
    if not isinstance(data, list):
        raise InvalidParameterError(f"data must be list, got {type(data)}")
    if not data:
        raise InvalidParameterError("data cannot be empty")
    if not isinstance(stock_names, list):
        raise InvalidParameterError(f"stock_names must be list, got {type(stock_names)}")
    
    random.seed(RANDOM_SEED + 2)
    modified_data = [row.copy() for row in data]
    low_vol, high_vol = volatility_range
    
    for stock in stock_names:
        vol = random.uniform(low_vol, high_vol)
        for i in range(1, len(modified_data)):
            prev = modified_data[i-1][stock]
            if not isinstance(prev, (int, float)):
                raise InvalidParameterError(f"Price for {stock} at day {i-1} is not a number: {prev}")
            noise = random.gauss(0.0, vol)
            new_price = prev * (1.0 + noise)
            if new_price < 0.1:
                new_price = 0.1
            if new_price > 1000.0:
                new_price = 1000.0
            modified_data[i][stock] = round(new_price, 4)
    
    return modified_data

def save_ground_truth(flat_periods: List[FlatPeriod], filepath: str = "data/raw/ground_truth.csv") -> None:
    if not isinstance(flat_periods, list):
        raise InvalidParameterError(f"flat_periods must be list, got {type(flat_periods)}")
    for period in flat_periods:
        if not isinstance(period, FlatPeriod):
            raise InvalidParameterError(f"flat_periods contains non-FlatPeriod: {type(period)}")
    if not isinstance(filepath, str):
        raise InvalidParameterError(f"filepath must be str, got {type(filepath)}")
    
    try:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['stock', 'start_idx', 'end_idx', 'start_date', 'end_date'])
            for period in flat_periods:
                writer.writerow([
                    period.stock,
                    period.start_idx,
                    period.end_idx,
                    period.start_date,
                    period.end_date
                ])
    except OSError as e:
        raise RuntimeError(f"Failed to write ground truth file {filepath}: {e}")

def save_to_csv(data: List[Dict[str, Any]], filepath: str = "data/raw/prices.csv") -> None:
    if not isinstance(data, list):
        raise InvalidParameterError(f"data must be list, got {type(data)}")
    if not data:
        raise InvalidParameterError("data cannot be empty")
    if not isinstance(filepath, str):
        raise InvalidParameterError(f"filepath must be str, got {type(filepath)}")
    
    try:
        with open(filepath, 'w', newline='') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    except OSError as e:
        raise RuntimeError(f"Failed to write CSV file {filepath}: {e}")

def main() -> None:
    data, stock_names = generate_portfolio_prices(n_stocks=10, days=504, start_date="2022-01-01")
    data_with_flats, flat_periods = inject_flat_periods(data, stock_names, n_flat_periods=5, min_flat_days=3, max_flat_days=10)
    final_data = add_realistic_noise(data_with_flats, stock_names, volatility_range=(0.005, 0.03))
    
    import os
    os.makedirs("data/raw", exist_ok=True)
    
    save_to_csv(final_data, "data/raw/prices.csv")
    save_ground_truth(flat_periods, "data/raw/ground_truth.csv")

if __name__ == "__main__":
    main()