import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

NUM_TICKERS: int = 10
DAYS: int = 100
START_DATE: datetime = datetime(2023, 1, 1)
PRICE_MIN: float = 10.0
PRICE_MAX: float = 500.0
STUCK_PROBABILITY: float = 0.05
STUCK_DAYS_MAX: int = 5
PRICE_STEP: float = 2.0

class SimulationError(Exception):
    pass

class FileWriteError(SimulationError):
    pass

class ParameterValidationError(SimulationError):
    pass

def _validate_parameters(output_path: str, num_tickers: int, days: int) -> None:
    if not isinstance(output_path, str):
        raise ParameterValidationError(f"output_path must be str, got {type(output_path)}")
    if not isinstance(num_tickers, int):
        raise ParameterValidationError(f"num_tickers must be int, got {type(num_tickers)}")
    if not isinstance(days, int):
        raise ParameterValidationError(f"days must be int, got {type(days)}")
    if num_tickers <= 0:
        raise ParameterValidationError(f"num_tickers must be > 0, got {num_tickers}")
    if days <= 0:
        raise ParameterValidationError(f"days must be > 0, got {days}")
    if not output_path.strip():
        raise ParameterValidationError("output_path cannot be empty")

def _generate_random_walk(start_price: float, days: int, seed: int) -> List[float]:
    random.seed(seed)
    prices: List[float] = [start_price]
    i: int = 0
    stuck_remaining: int = 0
    
    while len(prices) < days:
        if stuck_remaining > 0:
            prices.append(prices[-1])
            stuck_remaining -= 1
            i += 1
            continue
        
        if random.random() < STUCK_PROBABILITY:
            stuck_remaining = random.randint(1, STUCK_DAYS_MAX)
            continue
        
        step: float = random.uniform(-PRICE_STEP, PRICE_STEP)
        new_price: float = prices[-1] + step
        if new_price < PRICE_MIN:
            new_price = PRICE_MIN
        if new_price > PRICE_MAX:
            new_price = PRICE_MAX
        prices.append(new_price)
        i += 1
    
    assert len(prices) == days, f"generated {len(prices)} prices, expected {days}"
    return prices

def _write_csv(output_path: str, ticker_prices: List[Tuple[str, int, List[float]]]) -> bool:
    try:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'ticker', 'close_price'])
            
            for ticker, seed, prices in ticker_prices:
                current_date: datetime = START_DATE
                for i, price in enumerate(prices):
                    if i > 0:
                        current_date += timedelta(days=1)
                    date_str: str = current_date.strftime('%Y-%m-%d')
                    writer.writerow([date_str, ticker, f"{price:.2f}"])
        return True
    except Exception as e:
        raise FileWriteError(f"failed to write {output_path}: {e}")

def simulate(output_path: str, num_tickers: int, days: int) -> bool:
    _validate_parameters(output_path, num_tickers, days)
    
    ticker_prices: List[Tuple[str, int, List[float]]] = []
    
    for i in range(num_tickers):
        ticker: str = f"TICKER_{i:04d}"
        seed: int = i * 1000 + 42
        start_price: float = random.uniform(PRICE_MIN, PRICE_MAX)
        prices: List[float] = _generate_random_walk(start_price, days, seed)
        ticker_prices.append((ticker, seed, prices))
    
    return _write_csv(output_path, ticker_prices)

def main() -> None:
    output_path: str = "synthetic_data.csv"
    num_tickers: int = NUM_TICKERS
    days: int = DAYS
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            num_tickers = int(sys.argv[2])
        except ValueError:
            raise ParameterValidationError(f"num_tickers must be integer, got {sys.argv[2]}")
    if len(sys.argv) > 3:
        try:
            days = int(sys.argv[3])
        except ValueError:
            raise ParameterValidationError(f"days must be integer, got {sys.argv[3]}")
    
    success: bool = simulate(output_path, num_tickers, days)
    if not success:
        raise SimulationError("simulation failed")

if __name__ == "__main__":
    main()