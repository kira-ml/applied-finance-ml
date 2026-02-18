import csv
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence, Tuple

NUM_TICKERS_RANGE = (1, 1000)
DAYS_RANGE = (1, 3650)
FILENAME_SUFFIX = ".csv"
REQUIRED_COLUMNS = ("date", "ticker", "close_price")
STUCK_PROBABILITY = 0.05
START_PRICE = 100.0
VOLATILITY = 1.0

class SimulationError(Exception):
    pass

class InvalidTickerCount(SimulationError):
    pass

class InvalidDays(SimulationError):
    pass

class UnwritablePath(SimulationError):
    pass

class SchemaMismatch(SimulationError):
    pass

def _validate_ticker_count(num_tickers: int) -> None:
    if not isinstance(num_tickers, int):
        raise InvalidTickerCount(f"num_tickers must be int, got {type(num_tickers)}")
    if num_tickers < NUM_TICKERS_RANGE[0] or num_tickers > NUM_TICKERS_RANGE[1]:
        raise InvalidTickerCount(f"num_tickers {num_tickers} outside [{NUM_TICKERS_RANGE[0]}, {NUM_TICKERS_RANGE[1]}]")

def _validate_days(days: int) -> None:
    if not isinstance(days, int):
        raise InvalidDays(f"days must be int, got {type(days)}")
    if days < DAYS_RANGE[0] or days > DAYS_RANGE[1]:
        raise InvalidDays(f"days {days} outside [{DAYS_RANGE[0]}, {DAYS_RANGE[1]}]")

def _validate_output_path(output_path: str) -> Path:
    if not isinstance(output_path, str):
        raise UnwritablePath(f"output_path must be str, got {type(output_path)}")
    path = Path(output_path)
    if not path.suffix == FILENAME_SUFFIX:
        path = path.with_suffix(FILENAME_SUFFIX)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            pass
    except Exception as e:
        raise UnwritablePath(f"cannot write to {path}: {e}")
    return path

def simulate_prices(
    output_path: str,
    num_tickers: int,
    days: int,
    seed: int
) -> bool:
    _validate_ticker_count(num_tickers)
    _validate_days(days)
    validated_path = _validate_output_path(output_path)

    random.seed(seed)

    start_date = date(2020, 1, 1)
    date_list = [start_date + timedelta(days=i) for i in range(days)]

    ticker_symbols = [f"TICKER_{i+1:04d}" for i in range(num_tickers)]

    rows = []
    for i, ticker in enumerate(ticker_symbols):
        price = START_PRICE
        ticker_seed = seed + i
        rng = random.Random(ticker_seed)

        for current_date in date_list:
            if rng.random() < STUCK_PROBABILITY:
                pass
            else:
                change = rng.gauss(0, VOLATILITY)
                price += change
                if price < 0.01:
                    price = 0.01

            rows.append([current_date.isoformat(), ticker, round(price, 4)])

    try:
        with open(validated_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(REQUIRED_COLUMNS)
            writer.writerows(rows)
    except Exception as e:
        raise UnwritablePath(f"write failed: {e}")

    try:
        with open(validated_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            if tuple(header) != REQUIRED_COLUMNS:
                raise SchemaMismatch(f"header {header} != {REQUIRED_COLUMNS}")
    except Exception as e:
        raise SchemaMismatch(f"verification failed: {e}")

    return True

def _get_positive_int_arg(args: Sequence[str], idx: int, name: str) -> int:
    if len(args) <= idx:
        raise SimulationError(f"missing {name} argument")
    try:
        value = int(args[idx])
    except ValueError:
        raise SimulationError(f"{name} must be integer")
    if value <= 0:
        raise SimulationError(f"{name} must be positive")
    return value

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: simulate.py <output_path> <num_tickers> <days> <seed>")
        sys.exit(1)

    try:
        output = sys.argv[1]
        tickers = _get_positive_int_arg(sys.argv, 2, "num_tickers")
        days = _get_positive_int_arg(sys.argv, 3, "days")
        seed = _get_positive_int_arg(sys.argv, 4, "seed")

        success = simulate_prices(output, tickers, days, seed)
        if success:
            print(f"Generated {tickers} tickers over {days} days to {output}")
            sys.exit(0)
        else:
            print("Simulation failed")
            sys.exit(1)
    except SimulationError as e:
        print(f"Error: {e}")
        sys.exit(1)