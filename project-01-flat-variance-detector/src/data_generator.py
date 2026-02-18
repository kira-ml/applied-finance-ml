import csv
import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Tuple

# ==============================================================================
# 1. IMMUTABLE CONFIGURATION DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class Config:
    random_seed: int
    start_date: date
    num_days: int
    num_assets: int
    initial_price: float
    drift: float
    volatility: float
    min_flat_window: int
    max_flat_window: int
    min_flat_windows_per_asset: int
    max_flat_windows_per_asset: int
    prices_path: str
    ground_truth_path: str
    serialization_version: int

CONFIG = Config(
    random_seed=42,
    start_date=date(2023, 1, 1),
    num_days=252,
    num_assets=5,
    initial_price=100.0,
    drift=0.0005,
    volatility=0.02,
    min_flat_window=3,
    max_flat_window=10,
    min_flat_windows_per_asset=1,
    max_flat_windows_per_asset=3,
    prices_path="prices.csv",
    ground_truth_path="ground_truth.csv",
    serialization_version=1
)

# ==============================================================================
# 2. CUSTOM EXCEPTION DEFINITIONS
# ==============================================================================

class DataGenerationError(Exception):
    """Base exception for data generation failures."""
    pass

class ConfigurationError(DataGenerationError):
    """Raised when configuration parameters are invalid."""
    pass

class NumericalStabilityError(DataGenerationError):
    """Raised when numerical guards fail."""
    pass

class StateConsistencyError(DataGenerationError):
    """Raised when internal state invariants are violated."""
    pass

# ==============================================================================
# 3. TYPED DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class FlatWindow:
    start_index: int
    end_index: int

@dataclass(frozen=True)
class AssetGroundTruth:
    asset_id: str
    flat_windows: List[FlatWindow]

@dataclass(frozen=True)
class GenerationState:
    prices: List[List[float]]
    ground_truth: List[AssetGroundTruth]
    dates: List[date]

# ==============================================================================
# 4. PURE CORE LOGIC
# ==============================================================================

def _validate_config(cfg: Config) -> None:
    if cfg.num_days <= 0:
        raise ConfigurationError("num_days must be positive")
    if cfg.num_assets <= 0:
        raise ConfigurationError("num_assets must be positive")
    if cfg.initial_price <= 0.0:
        raise ConfigurationError("initial_price must be positive")
    if cfg.volatility < 0.0:
        raise ConfigurationError("volatility cannot be negative")
    if cfg.min_flat_window > cfg.max_flat_window:
        raise ConfigurationError("min_flat_window cannot exceed max_flat_window")
    if cfg.serialization_version != 1:
        raise ConfigurationError("Unsupported serialization version")

def _generate_business_days(start: date, count: int) -> List[date]:
    dates: List[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates

def _select_non_overlapping_windows(
    rng: random.Random,
    length: int,
    min_count: int,
    max_count: int,
    min_len: int,
    max_len: int
) -> List[FlatWindow]:
    num_windows = rng.randint(min_count, max_count)
    windows: List[FlatWindow] = []
    attempts = 0
    max_attempts = 1000
    
    while len(windows) < num_windows and attempts < max_attempts:
        attempts += 1
        w_len = rng.randint(min_len, max_len)
        if w_len >= length:
            continue
        start_idx = rng.randint(0, length - w_len - 1)
        end_idx = start_idx + w_len - 1
        
        candidate = FlatWindow(start_index=start_idx, end_index=end_idx)
        
        overlap = False
        for existing in windows:
            if not (candidate.end_index < existing.start_index or candidate.start_index > existing.end_index):
                overlap = True
                break
        
        if not overlap:
            windows.append(candidate)
    
    if len(windows) < min_count:
        raise StateConsistencyError(f"Failed to generate {min_count} non-overlapping windows after {max_attempts} attempts")
    
    windows.sort(key=lambda w: w.start_index)
    return windows

def _simulate_asset_prices(
    rng: random.Random,
    num_days: int,
    initial_price: float,
    drift: float,
    volatility: float,
    flat_windows: List[FlatWindow]
) -> List[float]:
    prices: List[float] = [initial_price]
    
    for t in range(1, num_days):
        prev_price = prices[-1]
        
        if math.isnan(prev_price) or math.isinf(prev_price):
            raise NumericalStabilityError(f"Invalid price at step {t-1}")
        
        is_flat = False
        for fw in flat_windows:
            if fw.start_index <= t <= fw.end_index:
                is_flat = True
                break
        
        if is_flat:
            next_price = prev_price
        else:
            shock = rng.gauss(0.0, 1.0)
            next_price = prev_price * (1.0 + drift + volatility * shock)
        
        if next_price <= 0.0:
            next_price = 0.01
        
        if math.isnan(next_price) or math.isinf(next_price):
            raise NumericalStabilityError(f"Numerical overflow/underflow at step {t}")
            
        prices.append(next_price)
    
    return prices

def _inject_ambiguous_period(
    rng: random.Random,
    prices: List[float],
    flat_windows: List[FlatWindow]
) -> List[float]:
    modified_prices = list(prices)
    num_days = len(prices)
    
    valid_starts = []
    for i in range(num_days - 4):
        in_existing = False
        for fw in flat_windows:
            if fw.start_index <= i <= fw.end_index or fw.start_index <= i + 3 <= fw.end_index:
                in_existing = True
                break
        if not in_existing:
            valid_starts.append(i)
    
    if not valid_starts:
        return modified_prices
    
    start_idx = rng.choice(valid_starts)
    end_idx = start_idx + 3
    
    base_val = modified_prices[start_idx]
    for i in range(start_idx, end_idx + 1):
        noise = rng.gauss(0.0, 0.0001)
        modified_prices[i] = base_val + noise
    
    return modified_prices

def _run_generation_logic(cfg: Config, rng: random.Random) -> GenerationState:
    dates = _generate_business_days(cfg.start_date, cfg.num_days)
    
    all_prices: List[List[float]] = []
    all_ground_truth: List[AssetGroundTruth] = []
    
    for i in range(cfg.num_assets):
        asset_id = f"ASSET_{i:03d}"
        
        windows = _select_non_overlapping_windows(
            rng=rng,
            length=cfg.num_days,
            min_count=cfg.min_flat_windows_per_asset,
            max_count=cfg.max_flat_windows_per_asset,
            min_len=cfg.min_flat_window,
            max_len=cfg.max_flat_window
        )
        
        raw_prices = _simulate_asset_prices(
            rng=rng,
            num_days=cfg.num_days,
            initial_price=cfg.initial_price,
            drift=cfg.drift,
            volatility=cfg.volatility,
            flat_windows=windows
        )
        
        if i == 0:
            final_prices = _inject_ambiguous_period(rng, raw_prices, windows)
        else:
            final_prices = raw_prices
        
        gt = AssetGroundTruth(asset_id=asset_id, flat_windows=windows)
        
        all_prices.append(final_prices)
        all_ground_truth.append(gt)
    
    return GenerationState(prices=all_prices, ground_truth=all_ground_truth, dates=dates)

# ==============================================================================
# 5. MINIMAL SIDE-EFFECT BOUNDARY FUNCTIONS
# ==============================================================================

def _write_prices_csv(path: str, dates: List[date], prices: List[List[float]], version: int) -> None:
    num_assets = len(prices)
    if num_assets == 0:
        return
    
    headers = ["date"] + [f"ASSET_{i:03d}" for i in range(num_assets)]
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for t, dt in enumerate(dates):
            row = [dt.isoformat()]
            for i in range(num_assets):
                val = prices[i][t]
                if math.isnan(val) or math.isinf(val):
                    raise NumericalStabilityError("Cannot serialize invalid number")
                row.append(f"{val:.6f}")
            writer.writerow(row)

def _write_ground_truth_csv(path: str, ground_truth: List[AssetGroundTruth], version: int) -> None:
    headers = ["asset_id", "window_start_idx", "window_end_idx", "version"]
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for gt in ground_truth:
            for fw in gt.flat_windows:
                writer.writerow([gt.asset_id, fw.start_index, fw.end_index, version])

def _print_summary(state: GenerationState, cfg: Config) -> None:
    total_flat_windows = sum(len(gt.flat_windows) for gt in state.ground_truth)
    print(f"Generated data for {cfg.num_assets} assets over {cfg.num_days} days.")
    print(f"Total flat windows injected (ground truth): {total_flat_windows}")
    print(f"Prices written to: {cfg.prices_path}")
    print(f"Ground truth written to: {cfg.ground_truth_path}")
    print(f"Seed used: {cfg.random_seed}")

# ==============================================================================
# 6. EXECUTION ENTRY POINT
# ==============================================================================

_setup_complete = False

def _guard_setup() -> None:
    global _setup_complete
    if _setup_complete:
        return
    _validate_config(CONFIG)
    _setup_complete = True

def main() -> None:
    _guard_setup()
    
    assert _setup_complete, "Setup guard failed"
    
    rng = random.Random(CONFIG.random_seed)
    
    state = _run_generation_logic(CONFIG, rng)
    
    assert len(state.dates) == CONFIG.num_days, "Date count mismatch"
    assert len(state.prices) == CONFIG.num_assets, "Asset count mismatch"
    
    _write_prices_csv(CONFIG.prices_path, state.dates, state.prices, CONFIG.serialization_version)
    _write_ground_truth_csv(CONFIG.ground_truth_path, state.ground_truth, CONFIG.serialization_version)
    
    _print_summary(state, CONFIG)

if __name__ == "__main__":
    main()