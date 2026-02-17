import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

WINDOW: int = 5
MIN_PERIODS: int = 3
THRESHOLD: float = 0.01


class Error(Exception):
    pass


class DataLoadError(Error):
    pass


class ColumnMissingError(DataLoadError):
    pass


class EmptyDataError(DataLoadError):
    pass


@dataclass(frozen=True)
class PriceRow:
    date: str
    ticker: str
    close_price: float


@dataclass(frozen=True)
class RollingStdRow:
    date: str
    ticker: str
    rolling_std: float


@dataclass(frozen=True)
class FlagRow:
    date: str
    ticker: str
    rolling_std: float
    is_flat: bool


def load_data(filepath: str = "data/raw/prices.csv") -> List[PriceRow]:
    path = Path(filepath)
    if not path.exists():
        raise DataLoadError(f"file not found: {filepath}")

    rows: List[PriceRow] = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise DataLoadError("empty CSV file")
        required = {"date", "ticker", "close_price"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ColumnMissingError(f"missing columns: {missing}")

        for line_num, row in enumerate(reader, start=2):
            date_val = row.get("date", "")
            if not isinstance(date_val, str) or not date_val:
                raise DataLoadError(f"line {line_num}: invalid or missing date")
            ticker_val = row.get("ticker", "")
            if not isinstance(ticker_val, str) or not ticker_val:
                raise DataLoadError(f"line {line_num}: invalid or missing ticker")
            try:
                price = float(row["close_price"])
            except (TypeError, ValueError):
                raise DataLoadError(f"line {line_num}: close_price not a number")

            rows.append(PriceRow(date=date_val, ticker=ticker_val, close_price=price))

    if not rows:
        raise EmptyDataError("no data rows found")

    return rows


def _pivot_to_wide(rows: List[PriceRow]) -> Tuple[List[str], List[str], List[List[float]]]:
    dates: List[str] = []
    tickers: List[str] = []
    matrix: List[List[Optional[float]]] = []

    ticker_to_idx: dict[str, int] = {}
    date_to_idx: dict[str, int] = {}

    for row in rows:
        if row.ticker not in ticker_to_idx:
            ticker_to_idx[row.ticker] = len(tickers)
            tickers.append(row.ticker)
        if row.date not in date_to_idx:
            date_to_idx[row.date] = len(dates)
            dates.append(row.date)

    for _ in range(len(dates)):
        matrix.append([None] * len(tickers))

    for row in rows:
        date_idx = date_to_idx[row.date]
        ticker_idx = ticker_to_idx[row.ticker]
        matrix[date_idx][ticker_idx] = row.close_price

    filled_matrix: List[List[float]] = []
    for date_idx in range(len(dates)):
        filled_row: List[float] = []
        for ticker_idx in range(len(tickers)):
            val = matrix[date_idx][ticker_idx]
            if val is None:
                raise DataLoadError(f"missing price for {tickers[ticker_idx]} on {dates[date_idx]}")
            filled_row.append(val)
        filled_matrix.append(filled_row)

    return dates, tickers, filled_matrix


def _compute_rolling_std_for_ticker(prices: List[float], window: int, min_periods: int) -> List[float]:
    result: List[float] = []
    for i in range(len(prices)):
        start = max(0, i - window + 1)
        window_values = prices[start:i+1]
        if len(window_values) < min_periods:
            result.append(float("nan"))
            continue

        mean = 0.0
        for v in window_values:
            mean += v
        mean /= len(window_values)

        sq_diff_sum = 0.0
        for v in window_values:
            diff = v - mean
            sq_diff_sum += diff * diff

        variance = sq_diff_sum / len(window_values)
        if variance < 0.0:
            variance = 0.0
        std = variance ** 0.5
        result.append(std)
    return result


def compute_rolling_std(rows: List[PriceRow], window: int = WINDOW, min_periods: int = MIN_PERIODS) -> List[RollingStdRow]:
    if not rows:
        return []
    if window <= 0:
        raise ValueError("window must be positive")
    if min_periods <= 0:
        raise ValueError("min_periods must be positive")
    if min_periods > window:
        raise ValueError("min_periods cannot exceed window")

    dates, tickers, price_matrix = _pivot_to_wide(rows)

    ticker_std_series: List[List[float]] = []
    for ticker_idx in range(len(tickers)):
        ticker_prices = [price_matrix[date_idx][ticker_idx] for date_idx in range(len(dates))]
        stds = _compute_rolling_std_for_ticker(ticker_prices, window, min_periods)
        ticker_std_series.append(stds)

    results: List[RollingStdRow] = []
    for date_idx, date in enumerate(dates):
        for ticker_idx, ticker in enumerate(tickers):
            std_val = ticker_std_series[ticker_idx][date_idx]
            results.append(RollingStdRow(date=date, ticker=ticker, rolling_std=std_val))

    return results


def flag_flat_periods(std_rows: List[RollingStdRow], threshold: float = THRESHOLD) -> List[FlagRow]:
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    results: List[FlagRow] = []
    for row in std_rows:
        std_val = row.rolling_std
        is_flat_val: bool
        if std_val != std_val:
            is_flat_val = False
        else:
            if std_val < 0.0:
                raise ValueError(f"negative standard deviation: {std_val}")
            is_flat_val = std_val < threshold
        results.append(FlagRow(
            date=row.date,
            ticker=row.ticker,
            rolling_std=row.rolling_std,
            is_flat=is_flat_val
        ))
    return results


def save_flags(flag_rows: List[FlagRow], output_dir: str = "data/processed/") -> str:
    if not flag_rows:
        raise ValueError("cannot save empty flag list")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"flags_{timestamp}.csv"
    full_path = out_path / filename

    with open(full_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "ticker", "rolling_std", "is_flat"])
        for row in flag_rows:
            writer.writerow([row.date, row.ticker, f"{row.rolling_std:.8f}", "1" if row.is_flat else "0"])

    return str(full_path)


def main() -> None:
    try:
        price_rows = load_data()
        if not price_rows:
            print("no data to process")
            return

        std_rows = compute_rolling_std(price_rows)
        flag_rows = flag_flat_periods(std_rows)
        output_path = save_flags(flag_rows)
        print(f"flags saved to {output_path}")

    except Error as e:
        print(f"error: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()