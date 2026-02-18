import csv
import dataclasses
import datetime
import math
import random
import sys
from typing import Dict, List, Sequence, Tuple, Union

import pandas as pd
import numpy as np

CONFIG_RANDOM_SEED: int = 42
CONFIG_N_DAYS: int = 1000
CONFIG_N_ASSETS: int = 5
CONFIG_FLAT_WINDOWS_PER_ASSET: Tuple[int, int] = (1, 3)
CONFIG_FLAT_WINDOW_MIN_LENGTH: int = 5
CONFIG_FLAT_WINDOW_MAX_LENGTH: int = 15
CONFIG_DRIFT_MEAN: float = 0.0005
CONFIG_DRIFT_STD: float = 0.01
CONFIG_START_PRICE: float = 100.0
CONFIG_PRICES_PATH: str = "prices.csv"
CONFIG_GROUND_TRUTH_PATH: str = "ground_truth.csv"
CONFIG_NEAR_FLAT_THRESHOLD: float = 0.0001
CONFIG_NEAR_FLAT_LENGTH: int = 10


class DataGenerationError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class AssetMetadata:
    asset_id: str
    start_price: float


@dataclasses.dataclass(frozen=True)
class FlatWindow:
    asset_id: str
    start_idx: int
    end_idx: int


def _validate_range_non_negative(value: Union[int, float], name: str) -> None:
    if value < 0:
        raise DataGenerationError(f"{name} must be non-negative, got {value}")


def _validate_positive(value: Union[int, float], name: str) -> None:
    if value <= 0:
        raise DataGenerationError(f"{name} must be positive, got {value}")


def _validate_asset_ids(asset_ids: List[str]) -> None:
    if not asset_ids:
        raise DataGenerationError("Asset IDs list cannot be empty")
    seen = set()
    for asset_id in asset_ids:
        if not isinstance(asset_id, str):
            raise DataGenerationError(f"Asset ID must be string, got {type(asset_id)}")
        if not asset_id:
            raise DataGenerationError("Asset ID cannot be empty string")
        if asset_id in seen:
            raise DataGenerationError(f"Duplicate asset ID: {asset_id}")
        seen.add(asset_id)


def _validate_config() -> None:
    _validate_positive(CONFIG_N_DAYS, "CONFIG_N_DAYS")
    _validate_positive(CONFIG_N_ASSETS, "CONFIG_N_ASSETS")
    if not (0 <= CONFIG_FLAT_WINDOWS_PER_ASSET[0] <= CONFIG_FLAT_WINDOWS_PER_ASSET[1]):
        raise DataGenerationError("Invalid flat window range")
    _validate_positive(CONFIG_FLAT_WINDOW_MIN_LENGTH, "CONFIG_FLAT_WINDOW_MIN_LENGTH")
    _validate_positive(CONFIG_FLAT_WINDOW_MAX_LENGTH, "CONFIG_FLAT_WINDOW_MAX_LENGTH")
    if CONFIG_FLAT_WINDOW_MIN_LENGTH > CONFIG_FLAT_WINDOW_MAX_LENGTH:
        raise DataGenerationError("Flat window min length > max length")
    _validate_positive(CONFIG_START_PRICE, "CONFIG_START_PRICE")
    if not CONFIG_PRICES_PATH:
        raise DataGenerationError("PRICES_PATH cannot be empty")
    if not CONFIG_GROUND_TRUTH_PATH:
        raise DataGenerationError("GROUND_TRUTH_PATH cannot be empty")
    _validate_positive(CONFIG_NEAR_FLAT_LENGTH, "CONFIG_NEAR_FLAT_LENGTH")


_validate_config()


def _setup_rng(seed: int) -> Tuple[random.Random, np.random.Generator]:
    _validate_range_non_negative(seed, "seed")
    py_rng = random.Random()
    py_rng.seed(seed)
    np_rng = np.random.default_rng(seed)
    return py_rng, np_rng


def _generate_asset_ids(n_assets: int, py_rng: random.Random) -> List[str]:
    if n_assets <= 0:
        raise DataGenerationError("Number of assets must be positive")
    asset_ids = [f"ASSET_{i+1:03d}" for i in range(n_assets)]
    py_rng.shuffle(asset_ids)
    return asset_ids


def _generate_random_walk(
    n_days: int,
    start_price: float,
    drift_mean: float,
    drift_std: float,
    np_rng: np.random.Generator,
) -> List[float]:
    if n_days <= 0:
        raise DataGenerationError("Number of days must be positive")
    _validate_positive(start_price, "start_price")
    log_returns = np_rng.normal(drift_mean, drift_std, n_days - 1)
    log_prices = np.zeros(n_days)
    log_prices[0] = math.log(start_price)
    for i in range(1, n_days):
        log_prices[i] = log_prices[i - 1] + log_returns[i - 1]
    prices = [math.exp(lp) for lp in log_prices]
    for price in prices:
        if not math.isfinite(price):
            raise DataGenerationError("Numerical instability: non-finite price generated")
    return prices


def _inject_flat_windows(
    prices: List[float],
    n_windows: int,
    min_len: int,
    max_len: int,
    py_rng: random.Random,
) -> List[FlatWindow]:
    if n_windows == 0:
        return []
    n_days = len(prices)
    if n_days <= 0:
        raise DataGenerationError("Cannot inject windows into empty price series")
    if min_len > n_days or max_len > n_days:
        raise DataGenerationError("Window length exceeds series length")
    windows = []
    occupied = [False] * n_days
    max_attempts = 1000
    for _ in range(n_windows):
        window_len = py_rng.randint(min_len, max_len)
        attempts = 0
        placed = False
        while not placed and attempts < max_attempts:
            start_idx = py_rng.randint(0, n_days - window_len)
            end_idx = start_idx + window_len - 1
            overlap = any(occupied[start_idx : end_idx + 1])
            if not overlap:
                base_price = prices[start_idx]
                for i in range(start_idx, end_idx + 1):
                    prices[i] = base_price
                    occupied[i] = True
                windows.append(FlatWindow(asset_id="", start_idx=start_idx, end_idx=end_idx))
                placed = True
            attempts += 1
        if not placed:
            raise DataGenerationError("Failed to place non-overlapping flat window")
    return windows


def _inject_near_flat_segment(
    prices: List[float],
    threshold: float,
    length: int,
    py_rng: random.Random,
) -> None:
    n_days = len(prices)
    if length > n_days:
        raise DataGenerationError("Near-flat segment length exceeds series length")
    start_idx = py_rng.randint(0, n_days - length)
    base_price = prices[start_idx]
    for i in range(start_idx, start_idx + length):
        noise = py_rng.uniform(-threshold, threshold)
        modified = base_price + noise
        if modified <= 0:
            raise DataGenerationError("Near-flat segment generated non-positive price")
        prices[i] = modified


def _generate_single_asset_prices(
    asset_id: str,
    n_days: int,
    start_price: float,
    drift_mean: float,
    drift_std: float,
    flat_windows_range: Tuple[int, int],
    flat_min_len: int,
    flat_max_len: int,
    near_flat_threshold: float,
    near_flat_length: int,
    py_rng: random.Random,
    np_rng: np.random.Generator,
    is_near_flat_asset: bool,
) -> Tuple[List[float], List[FlatWindow]]:
    prices = _generate_random_walk(n_days, start_price, drift_mean, drift_std, np_rng)
    n_flat = py_rng.randint(flat_windows_range[0], flat_windows_range[1])
    windows = _inject_flat_windows(prices, n_flat, flat_min_len, flat_max_len, py_rng)
    for w in windows:
        object.__setattr__(w, "asset_id", asset_id)
    if is_near_flat_asset:
        _inject_near_flat_segment(prices, near_flat_threshold, near_flat_length, py_rng)
    return prices, windows


def _generate_all_assets(
    asset_ids: List[str],
    n_days: int,
    start_price: float,
    drift_mean: float,
    drift_std: float,
    flat_windows_range: Tuple[int, int],
    flat_min_len: int,
    flat_max_len: int,
    near_flat_threshold: float,
    near_flat_length: int,
    py_rng: random.Random,
    np_rng: np.random.Generator,
) -> Tuple[Dict[str, List[float]], List[FlatWindow]]:
    all_prices: Dict[str, List[float]] = {}
    all_windows: List[FlatWindow] = []
    near_flat_asset = asset_ids[py_rng.randint(0, len(asset_ids) - 1)]
    for asset_id in asset_ids:
        asset_prices, asset_windows = _generate_single_asset_prices(
            asset_id,
            n_days,
            start_price,
            drift_mean,
            drift_std,
            flat_windows_range,
            flat_min_len,
            flat_max_len,
            near_flat_threshold,
            near_flat_length,
            py_rng,
            np_rng,
            asset_id == near_flat_asset,
        )
        all_prices[asset_id] = asset_prices
        all_windows.extend(asset_windows)
    return all_prices, all_windows


def _write_prices_csv(
    dates: Sequence[pd.Timestamp],
    prices_by_asset: Dict[str, List[float]],
    path: str,
) -> None:
    if not dates:
        raise DataGenerationError("No dates to write")
    if not prices_by_asset:
        raise DataGenerationError("No price data to write")
    asset_ids = list(prices_by_asset.keys())
    n_days = len(dates)
    for asset_id, prices in prices_by_asset.items():
        if len(prices) != n_days:
            raise DataGenerationError(f"Price length mismatch for {asset_id}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["date"] + asset_ids
        writer.writerow(header)
        for i in range(n_days):
            row = [dates[i].date().isoformat()] + [str(prices_by_asset[aid][i]) for aid in asset_ids]
            writer.writerow(row)


def _write_ground_truth_csv(windows: List[FlatWindow], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["asset_id", "start_idx", "end_idx"])
        for w in windows:
            if w.start_idx > w.end_idx:
                raise DataGenerationError(f"Invalid window indices for {w.asset_id}")
            writer.writerow([w.asset_id, str(w.start_idx), str(w.end_idx)])


def _print_summary(
    asset_ids: List[str],
    n_days: int,
    windows: List[FlatWindow],
    near_flat_asset: str,
) -> None:
    print(f"Generated data for {len(asset_ids)} assets over {n_days} business days")
    print(f"Ground truth flat windows: {len(windows)}")
    for w in windows:
        print(f"  {w.asset_id}: indices {w.start_idx}–{w.end_idx}")
    print(f"Near-flat ambiguous period injected into {near_flat_asset} (not in ground truth)")


def generate_data() -> None:
    _validate_config()
    py_rng, np_rng = _setup_rng(CONFIG_RANDOM_SEED)
    asset_ids = _generate_asset_ids(CONFIG_N_ASSETS, py_rng)
    _validate_asset_ids(asset_ids)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=CONFIG_N_DAYS).to_list()
    if len(dates) != CONFIG_N_DAYS:
        raise DataGenerationError("Date generation failed")
    prices_by_asset, all_windows = _generate_all_assets(
        asset_ids,
        CONFIG_N_DAYS,
        CONFIG_START_PRICE,
        CONFIG_DRIFT_MEAN,
        CONFIG_DRIFT_STD,
        CONFIG_FLAT_WINDOWS_PER_ASSET,
        CONFIG_FLAT_WINDOW_MIN_LENGTH,
        CONFIG_FLAT_WINDOW_MAX_LENGTH,
        CONFIG_NEAR_FLAT_THRESHOLD,
        CONFIG_NEAR_FLAT_LENGTH,
        py_rng,
        np_rng,
    )
    _write_prices_csv(dates, prices_by_asset, CONFIG_PRICES_PATH)
    _write_ground_truth_csv(all_windows, CONFIG_GROUND_TRUTH_PATH)
    _print_summary(asset_ids, CONFIG_N_DAYS, all_windows, asset_ids[-1])


if __name__ == "__main__":
    generate_data()