"""
Synthetic revenue data generator.

Purpose:
Generate a mathematically controlled univariate time-series
with configurable stability for pipeline testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict
import json


# Default paths
DEFAULT_RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
DEFAULT_FILENAME = "revenue.csv"
DEFAULT_FULL_PATH = DEFAULT_RAW_DIR / DEFAULT_FILENAME


def generate_synthetic_revenue(
    n_days: int,
    seed: int = 42,
    base_level: float = 1000.0,
    trend_slope: float = 0.5,
    seasonality_amplitude: float = 100.0,
    noise_std: float = 50.0,
    structural_break_day: Optional[int] = None,
    structural_break_magnitude: float = 0.0,
    stable_mode: bool = False,
) -> pd.DataFrame:
    """
    Generate synthetic daily revenue time-series.

    Parameters
    ----------
    n_days : int
        Number of days to generate.
    seed : int
        Random seed for reproducibility.
    base_level : float
        Initial revenue level.
    trend_slope : float
        Linear daily growth rate.
    seasonality_amplitude : float
        Weekly seasonality strength.
    noise_std : float
        Standard deviation of Gaussian noise.
    structural_break_day : int | None
        Day index where structural break begins.
    structural_break_magnitude : float
        Additive shift after structural break.
    stable_mode : bool
        If True, reduces non-stationarity for stable CV performance.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - date
        - revenue

    Raises
    ------
    ValueError
        If n_days is not positive or structural_break_day is out of range.
    """
    if n_days <= 0:
        raise ValueError(f"n_days must be positive, got {n_days}")

    t = np.arange(n_days)

    # Deterministic components
    if stable_mode:
        # Reduced trend and seasonality for stable CV performance
        trend = base_level + (trend_slope * 0.1) * t
        seasonality = (seasonality_amplitude * 0.3) * np.sin(2 * np.pi * t / 7)
    else:
        trend = base_level + trend_slope * t
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / 7)

    # Stochastic component
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, size=n_days)

    revenue = trend + seasonality + noise

    # Optional structural break (suppressed in stable_mode)
    if structural_break_day is not None and not stable_mode:
        if not (0 <= structural_break_day < n_days):
            raise ValueError(
                f"structural_break_day must be between 0 and {n_days-1}, "
                f"got {structural_break_day}"
            )
        revenue[structural_break_day:] += structural_break_magnitude

    # Ensure strictly positive revenue
    revenue = np.maximum(revenue, 1.0)

    dates = pd.date_range(
        start="2020-01-01",
        periods=n_days,
        freq="D"
    )

    return pd.DataFrame({
        "date": dates,
        "revenue": revenue
    })


def save_to_disk(
    df: pd.DataFrame,
    filename: str = DEFAULT_FILENAME,
    output_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    overwrite: bool = False,
    metadata: Optional[Dict] = None
) -> Path:
    """Save synthetic data to disk in the raw data directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename

    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"File already exists: {filepath}\n"
            f"Use overwrite=True to overwrite."
        )

    df.to_csv(filepath, index=False)
    print(f"✓ Data saved to: {filepath}")

    if metadata:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved to: {metadata_path}")

    return filepath


def generate_stable_dataset(
    overwrite: bool = False,
    n_days: int = 730,
    seed: int = 42,
    verbose: bool = True
) -> Path:
    """Generate a stable dataset that passes the 5% variance threshold."""
    if verbose:
        print("=" * 60)
        print("STABLE SYNTHETIC REVENUE DATASET GENERATOR")
        print("=" * 60)

    filename = "revenue_stable.csv"
    filepath = DEFAULT_RAW_DIR / filename

    if filepath.exists() and not overwrite:
        if verbose:
            print(f"\n✓ Stable dataset already exists at: {filepath}")
        return filepath

    df = generate_synthetic_revenue(
        n_days=n_days,
        seed=seed,
        base_level=1000.0,
        trend_slope=0.05,
        seasonality_amplitude=50.0,
        noise_std=30.0,
        structural_break_day=None,
        stable_mode=True
    )

    if verbose:
        print(f"  ✓ Generated {len(df)} records")
        print(f"  Revenue range: ${df['revenue'].min():.2f} to ${df['revenue'].max():.2f}")
        print(f"  Mean: ${df['revenue'].mean():.2f}  Std: ${df['revenue'].std():.2f}")

    metadata = {
        "type": "stable",
        "n_days": n_days,
        "seed": seed,
        "trend_slope": 0.05,
        "seasonality_amplitude": 50.0,
        "noise_std": 30.0,
        "structural_break": None,
        "stable_mode": True,
        "expected_variance": "<5%"
    }

    filepath = save_to_disk(df=df, filename=filename, output_dir=DEFAULT_RAW_DIR,
                            overwrite=overwrite, metadata=metadata)

    if verbose:
        print(f"\n✅ Stable dataset ready!")
        print(f"   Run: python src/main.py --data-path {filepath} --stability-threshold 5.0")

    return filepath


def generate_test_datasets(overwrite: bool = False):
    """Generate stable, moderate, and original datasets for comparison."""
    print("=" * 60)
    print("GENERATING TEST DATASETS")
    print("=" * 60)

    datasets = {}

    print("\n1. Generating STABLE dataset...")
    datasets['stable'] = generate_stable_dataset(overwrite=overwrite)

    print("\n2. Generating MODERATE dataset...")
    df_moderate = generate_synthetic_revenue(
        n_days=730, seed=43, base_level=1000.0,
        trend_slope=0.2, seasonality_amplitude=80.0,
        noise_std=50.0, structural_break_day=None
    )
    datasets['moderate'] = save_to_disk(
        df=df_moderate, filename="revenue_moderate.csv",
        overwrite=overwrite,
        metadata={"type": "moderate", "expected_variance": "5-10%"}
    )

    print("\n3. Generating ORIGINAL dataset (with structural break)...")
    df_original = generate_synthetic_revenue(
        n_days=730, seed=42, base_level=1000.0,
        trend_slope=0.3, seasonality_amplitude=150.0,
        noise_std=75.0, structural_break_day=365,
        structural_break_magnitude=200.0
    )
    datasets['original'] = save_to_disk(
        df=df_original, filename="revenue_original.csv",
        overwrite=overwrite,
        metadata={"type": "original_with_break", "expected_variance": ">10%"}
    )

    print("\n" + "=" * 60)
    print("DATASETS GENERATED:")
    for name, path in datasets.items():
        print(f"  {name}: {path}")
    print("=" * 60)

    return datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic revenue datasets")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['stable', 'moderate', 'original', 'all'],
        default='stable',
        help="Which dataset to generate (default: stable)"
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite existing files"
    )
    args = parser.parse_args()

    if args.mode == 'all':
        generate_test_datasets(overwrite=args.overwrite)
    elif args.mode == 'stable':
        generate_stable_dataset(overwrite=args.overwrite)
    elif args.mode == 'moderate':
        df = generate_synthetic_revenue(
            n_days=730, seed=43, base_level=1000.0,
            trend_slope=0.2, seasonality_amplitude=80.0, noise_std=50.0
        )
        save_to_disk(df=df, filename="revenue_moderate.csv", overwrite=args.overwrite)
    elif args.mode == 'original':
        df = generate_synthetic_revenue(
            n_days=730, seed=42, base_level=1000.0,
            trend_slope=0.3, seasonality_amplitude=150.0, noise_std=75.0,
            structural_break_day=365, structural_break_magnitude=200.0
        )
        save_to_disk(df=df, filename="revenue_original.csv", overwrite=args.overwrite)