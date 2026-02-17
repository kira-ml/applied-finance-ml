import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional, Tuple
import json
import os
import sys
from enum import Enum
import argparse


# Immutable configuration definitions
class DateRangeConfig(TypedDict):
    start: str
    end: str
    trading_days_only: bool


class FlatPeriodConfig(TypedDict):
    start: str
    duration_days: int
    noise_level: float


class AssetConfig(TypedDict):
    ticker: str
    base_price: float
    drift: float
    volatility: float
    flat_periods: List[FlatPeriodConfig]


class OutputConfig(TypedDict):
    price_file: str
    metadata_file: str
    ground_truth_file: str


class SimConfig(TypedDict):
    simulation_name: str
    random_seed: int
    date_range: DateRangeConfig
    assets: List[AssetConfig]
    correlation_matrix: List[List[float]]
    output: OutputConfig


# Custom exceptions
class SimulationError(Exception):
    """Base exception for simulation errors."""
    pass


class ConfigValidationError(SimulationError):
    """Configuration validation failed."""
    pass


class DataGenerationError(SimulationError):
    """Data generation process failed."""
    pass


class FileOperationError(SimulationError):
    """File read/write operation failed."""
    pass


# Enums for state representation
class ValidationStatus(Enum):
    VALID = 1
    INVALID = 2


# Typed data structures
class FlatPeriodGroundTruth(TypedDict):
    ticker: str
    start_date: str
    end_date: str


class SimulationStatistics(TypedDict):
    total_days: int
    total_assets: int
    flat_day_percentage: float


class SimulationMetadata(TypedDict):
    simulation_name: str
    generation_timestamp: str
    random_seed: int
    parameters: Dict[str, Any]
    flat_periods_ground_truth: List[FlatPeriodGroundTruth]
    statistics: SimulationStatistics


# Pure core logic functions
def _validate_config(config: Dict[str, Any]) -> Tuple[ValidationStatus, Optional[str]]:
    """Validate configuration structure and values."""
    try:
        # Check required top-level keys
        required_keys = {'simulation_name', 'random_seed', 'date_range', 
                        'assets', 'correlation_matrix', 'output'}
        if not all(key in config for key in required_keys):
            missing = required_keys - set(config.keys())
            return ValidationStatus.INVALID, f"Missing required keys: {missing}"
        
        # Validate random seed
        if not isinstance(config['random_seed'], int) or config['random_seed'] < 0:
            return ValidationStatus.INVALID, "random_seed must be non-negative integer"
        
        # Validate date_range
        dr = config['date_range']
        if not all(k in dr for k in ['start', 'end', 'trading_days_only']):
            return ValidationStatus.INVALID, "date_range missing required fields"
        
        try:
            start_date = datetime.strptime(dr['start'], '%Y-%m-%d')
            end_date = datetime.strptime(dr['end'], '%Y-%m-%d')
            if start_date >= end_date:
                return ValidationStatus.INVALID, "start_date must be before end_date"
        except ValueError:
            return ValidationStatus.INVALID, "date_range dates must be YYYY-MM-DD format"
        
        if not isinstance(dr['trading_days_only'], bool):
            return ValidationStatus.INVALID, "trading_days_only must be boolean"
        
        # Validate assets
        if not config['assets']:
            return ValidationStatus.INVALID, "assets list cannot be empty"
        
        tickers = set()
        for i, asset in enumerate(config['assets']):
            if not all(k in asset for k in ['ticker', 'base_price', 'drift', 'volatility', 'flat_periods']):
                return ValidationStatus.INVALID, f"Asset {i} missing required fields"
            
            if asset['ticker'] in tickers:
                return ValidationStatus.INVALID, f"Duplicate ticker: {asset['ticker']}"
            tickers.add(asset['ticker'])
            
            if asset['base_price'] <= 0:
                return ValidationStatus.INVALID, f"Asset {asset['ticker']} base_price must be positive"
            
            if asset['volatility'] < 0:
                return ValidationStatus.INVALID, f"Asset {asset['ticker']} volatility cannot be negative"
            
            for j, fp in enumerate(asset['flat_periods']):
                if not all(k in fp for k in ['start', 'duration_days', 'noise_level']):
                    return ValidationStatus.INVALID, f"Flat period {j} in asset {asset['ticker']} missing fields"
                
                if fp['duration_days'] <= 0:
                    return ValidationStatus.INVALID, f"Flat period {j} duration_days must be positive"
                
                if fp['noise_level'] < 0:
                    return ValidationStatus.INVALID, f"Flat period {j} noise_level cannot be negative"
        
        # Validate correlation matrix
        n_assets = len(config['assets'])
        corr = config['correlation_matrix']
        
        if len(corr) != n_assets:
            return ValidationStatus.INVALID, "correlation_matrix rows must match number of assets"
        
        for i, row in enumerate(corr):
            if len(row) != n_assets:
                return ValidationStatus.INVALID, f"correlation_matrix row {i} length mismatch"
            for j, val in enumerate(row):
                if i == j and abs(val - 1.0) > 1e-6:
                    return ValidationStatus.INVALID, f"correlation_matrix diagonal at [{i},{j}] must be 1.0"
                if val < -1.0 or val > 1.0:
                    return ValidationStatus.INVALID, f"correlation_matrix [{i},{j}] must be between -1 and 1"
        
        # Validate output config
        out = config['output']
        if not all(k in out for k in ['price_file', 'metadata_file', 'ground_truth_file']):
            return ValidationStatus.INVALID, "output missing required fields"
        
        return ValidationStatus.VALID, None
        
    except Exception as e:
        return ValidationStatus.INVALID, f"Validation error: {str(e)}"


def _generate_date_range(start_date: datetime, end_date: datetime, 
                        trading_days_only: bool) -> List[datetime]:
    """Generate list of dates between start and end."""
    dates = []
    current = start_date
    
    while current <= end_date:
        if not trading_days_only or current.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current)
        current += timedelta(days=1)
    
    return dates


def _cholesky_decomposition(matrix: List[List[float]]) -> List[List[float]]:
    """Compute Cholesky decomposition of correlation matrix."""
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            
            if i == j:
                val = matrix[i][i] - s
                if val <= 0:
                    raise DataGenerationError(f"Matrix not positive definite at index {i}")
                L[i][j] = max(val, 0) ** 0.5  # Numerical stability
            else:
                if abs(L[j][j]) < 1e-10:
                    raise DataGenerationError(f"Numerical instability in Cholesky at [{j},{j}]")
                L[i][j] = (1.0 / L[j][j] * (matrix[i][j] - s))
    
    return L


def _convert_date_to_idx(date: datetime, date_list: List[datetime]) -> int:
    """Convert date to index in date list."""
    for i, d in enumerate(date_list):
        if d.date() == date.date():
            return i
    return -1


# Public interface functions
def generate_price_series_from_config(config: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame, SimulationMetadata]:
    """Generate multi-asset price series based on configuration."""
    # Validate config
    status, error_msg = _validate_config(config)
    if status == ValidationStatus.INVALID:
        raise ConfigValidationError(error_msg)
    
    try:
        # Setup
        random_state = np.random.RandomState(config['random_seed'])
        n_assets = len(config['assets'])
        
        # Generate dates
        start_date = datetime.strptime(config['date_range']['start'], '%Y-%m-%d')
        end_date = datetime.strptime(config['date_range']['end'], '%Y-%m-%d')
        trading_days_only = config['date_range']['trading_days_only']
        
        dates = _generate_date_range(start_date, end_date, trading_days_only)
        n_days = len(dates)
        
        if n_days == 0:
            raise DataGenerationError("No dates generated from range")
        
        # Get correlation matrix
        corr_matrix = config['correlation_matrix']
        
        # Generate correlated random returns using Cholesky
        L = _cholesky_decomposition(corr_matrix)
        
        # Generate independent random normals
        independent_returns = random_state.normal(0, 1, (n_days, n_assets))
        
        # Apply correlation
        correlated_returns = np.zeros((n_days, n_assets))
        for i in range(n_assets):
            for j in range(i + 1):
                correlated_returns[:, i] += L[i][j] * independent_returns[:, j]
        
        # Generate price series for each asset
        all_prices = []
        all_is_flat = []
        ground_truth_list = []
        
        for asset_idx, asset in enumerate(config['assets']):
            # Extract asset params
            ticker = asset['ticker']
            base_price = asset['base_price']
            drift = asset['drift']
            volatility = asset['volatility']
            
            # Scale returns to match asset volatility and add drift
            scaled_returns = correlated_returns[:, asset_idx] * volatility + drift
            
            # Generate price series (geometric Brownian motion)
            log_prices = np.log(base_price) + np.cumsum(scaled_returns)
            prices = np.exp(log_prices)
            
            # Track flat periods
            is_flat = np.zeros(n_days, dtype=bool)
            
            # Process flat periods
            for flat_period in asset['flat_periods']:
                flat_start = datetime.strptime(flat_period['start'], '%Y-%m-%d')
                start_idx = _convert_date_to_idx(flat_start, dates)
                
                if start_idx == -1:
                    raise DataGenerationError(f"Flat period start {flat_start} not in date range")
                
                duration = flat_period['duration_days']
                noise_level = flat_period['noise_level']
                
                if start_idx + duration > n_days:
                    duration = n_days - start_idx  # Trim to available days
                
                end_idx = start_idx + duration
                is_flat[start_idx:end_idx] = True
                
                # Apply flat values (constant with tiny noise)
                flat_value = prices[start_idx]
                for i in range(start_idx, end_idx):
                    noise = random_state.normal(0, noise_level)
                    prices[i] = flat_value + noise
                
                # Add to ground truth
                flat_end = dates[end_idx - 1]
                ground_truth_list.append({
                    'ticker': ticker,
                    'start_date': flat_start.strftime('%Y-%m-%d'),
                    'end_date': flat_end.strftime('%Y-%m-%d')
                })
            
            # Store results
            all_prices.append(prices)
            all_is_flat.append(is_flat)
        
        # Create DataFrames
        date_strs = [d.strftime('%Y-%m-%d') for d in dates]
        
        # Price DataFrame - long format
        price_data = []
        for date_idx, date_str in enumerate(date_strs):
            for asset_idx, asset in enumerate(config['assets']):
                price_data.append({
                    'date': date_str,
                    'ticker': asset['ticker'],
                    'close_price': round(float(all_prices[asset_idx][date_idx]), 4)
                })
        price_df = pd.DataFrame(price_data)
        
        # Ground truth DataFrame - long format
        truth_data = []
        for date_idx, date_str in enumerate(date_strs):
            for asset_idx, asset in enumerate(config['assets']):
                truth_data.append({
                    'date': date_str,
                    'ticker': asset['ticker'],
                    'is_truly_flat': int(all_is_flat[asset_idx][date_idx])
                })
        truth_df = pd.DataFrame(truth_data)
        
        # Calculate statistics
        total_flat_days = sum(sum(flat) for flat in all_is_flat)
        total_possible_days = n_days * n_assets
        
        stats: SimulationStatistics = {
            'total_days': n_days,
            'total_assets': n_assets,
            'flat_day_percentage': round(total_flat_days / total_possible_days, 4) if total_possible_days > 0 else 0.0
        }
        
        # Create metadata
        metadata: SimulationMetadata = {
            'simulation_name': config['simulation_name'],
            'generation_timestamp': datetime.now().isoformat(),
            'random_seed': config['random_seed'],
            'parameters': dict(config),
            'flat_periods_ground_truth': ground_truth_list,
            'statistics': stats
        }
        
        return price_df, truth_df, metadata
        
    except Exception as e:
        raise DataGenerationError(f"Data generation failed: {str(e)}")


def save_simulated_data(price_df: pd.DataFrame, truth_df: pd.DataFrame, 
                       metadata: SimulationMetadata, output_config: OutputConfig) -> None:
    """Save simulated data to files."""
    try:
        # Create directories if needed
        for file_path in [output_config['price_file'], 
                         output_config['metadata_file'], 
                         output_config['ground_truth_file']]:
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
        
        # Save files
        price_df.to_csv(output_config['price_file'], index=False)
        truth_df.to_csv(output_config['ground_truth_file'], index=False)
        
        with open(output_config['metadata_file'], 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
            
    except Exception as e:
        raise FileOperationError(f"Failed to save data: {str(e)}")


def load_config(config_path: str) -> SimConfig:
    """Load and parse configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise ConfigValidationError(f"Invalid JSON in config file: {str(e)}")
    except FileNotFoundError as e:
        raise ConfigValidationError(f"Config file not found: {str(e)}")
    except Exception as e:
        raise ConfigValidationError(f"Failed to load config: {str(e)}")


# Side-effect boundary function
def run_simulation(config_path: str) -> None:
    """Main simulation entry point with side effects."""
    try:
        # Load and validate config
        config = load_config(config_path)
        
        # Generate data
        price_df, truth_df, metadata = generate_price_series_from_config(config)
        
        # Save data
        save_simulated_data(price_df, truth_df, metadata, config['output'])
        
        print(f"Simulation completed successfully. Files saved to:")
        print(f"  Prices: {config['output']['price_file']}")
        print(f"  Ground truth: {config['output']['ground_truth_file']}")
        print(f"  Metadata: {config['output']['metadata_file']}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"  Date range: {config['date_range']['start']} to {config['date_range']['end']}")
        print(f"  Total days: {metadata['statistics']['total_days']}")
        print(f"  Total assets: {metadata['statistics']['total_assets']}")
        print(f"  Flat day percentage: {metadata['statistics']['flat_day_percentage']:.2%}")
        print(f"  Flat periods: {len(metadata['flat_periods_ground_truth'])}")
        
    except SimulationError as e:
        print(f"Simulation failed: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}", file=sys.stderr)
        sys.exit(1)


# Execution entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic financial time series data')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to simulation configuration JSON file')
    args = parser.parse_args()
    
    run_simulation(args.config)