"""
Synthetic revenue data generator.

Purpose:
Generate a mathematically controlled univariate time-series
with trend, weekly seasonality, and Gaussian noise.

This module intentionally avoids:
- AR processes
- Multivariate outputs
- File I/O
- Feature engineering
- Model evaluation

It produces raw chronological revenue only.
"""

import numpy as np
import pandas as pd


def generate_synthetic_revenue(
    n_days: int,
    seed: int = 42,
    base_level: float = 1000.0,
    trend_slope: float = 0.5,
    seasonality_amplitude: float = 100.0,
    noise_std: float = 50.0,
    structural_break_day: int | None = None,
    structural_break_magnitude: float = 0.0,
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
    trend = base_level + trend_slope * t
    seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / 7)

    # Stochastic component
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, size=n_days)

    revenue = trend + seasonality + noise

    # Optional structural break
    if structural_break_day is not None:
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


# Minimal usage example
if __name__ == "__main__":
    # Generate 90 days of synthetic revenue
    df = generate_synthetic_revenue(
        n_days=90,
        seed=42,
        base_level=1000.0,
        trend_slope=0.5,
        seasonality_amplitude=100.0,
        noise_std=50.0
    )
    
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Revenue range: {df['revenue'].min():.2f} to {df['revenue'].max():.2f}")