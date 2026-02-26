import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.random import Generator, PCG64


_FRAUD_RATE: float = 0.01
_TOTAL_ROWS: int = 200_000
_RANDOM_SEED: int = 42
_OUTPUT_PATH: Path = Path("data/raw/transactions.csv")

_LEGIT_AMT_MEAN: float = 3.5
_LEGIT_AMT_SIGMA: float = 1.1

_FRAUD_SMALL_WEIGHT: float = 0.7
_FRAUD_SMALL_MEAN: float = 2.0
_FRAUD_SMALL_SIGMA: float = 0.6
_FRAUD_LARGE_MEAN: float = 5.5
_FRAUD_LARGE_SIGMA: float = 0.8

_TIME_WINDOW_DAYS: int = 180
_SECONDS_PER_DAY: int = 86400
_HOURS_OVERNIGHT_START: int = 1
_HOURS_OVERNIGHT_END: int = 5
_FRAUD_OVERNIGHT_MULTIPLIER: float = 2.5

_CARD_TYPES: Tuple[str, ...] = ("Visa", "Mastercard", "Amex", "Discover")
_LEGIT_CARD_PROBS: Tuple[float, ...] = (0.52, 0.30, 0.12, 0.06)
_FRAUD_CARD_PROBS: Tuple[float, ...] = (0.45, 0.28, 0.20, 0.07)

_MERCHANT_CATEGORIES: Tuple[str, ...] = (
    "grocery", "restaurant", "online_retail", "travel",
    "entertainment", "health", "automotive", "utilities"
)
_LEGIT_MERCHANT_PROBS: Tuple[float, ...] = (0.20, 0.18, 0.15, 0.10, 0.12, 0.10, 0.08, 0.07)
_FRAUD_MERCHANT_MULTIPLIER: float = 1.8
_HIGH_RISK_CATEGORIES: Tuple[str, ...] = ("online_retail", "travel")

_NULL_MERCHANT_RATE: float = 0.03
_NULL_CARD_RATE: float = 0.015


class SyntheticDataError(Exception):
    """Base exception for synthetic data generation failures."""
    pass


class ProbabilityVectorError(SyntheticDataError):
    """Raised when probability vectors are invalid."""
    pass


class CountError(SyntheticDataError):
    """Raised when row counts are inconsistent."""
    pass


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable configuration for data generation."""
    total_rows: int
    fraud_rate: float
    seed: int
    output_path: Path


@dataclass(frozen=True)
class GeneratedData:
    """Container for generated data components."""
    transaction_dt: np.ndarray
    transaction_amt: np.ndarray
    card_type: np.ndarray
    merchant_category: np.ndarray
    is_fraud: np.ndarray
    v1: np.ndarray


def _validate_probabilities(probs: Tuple[float, ...], name: str) -> None:
    """Validate that a probability vector sums to 1.0 within tolerance."""
    if abs(sum(probs) - 1.0) > 1e-10:
        raise ProbabilityVectorError(f"{name} probabilities must sum to 1.0, got {sum(probs)}")
    if any(p < 0.0 for p in probs):
        raise ProbabilityVectorError(f"{name} probabilities cannot be negative")


def _compute_fraud_count(total_rows: int, fraud_rate: float) -> int:
    """Compute exact number of fraud rows using floor."""
    if not 0.0 <= fraud_rate <= 1.0:
        raise CountError(f"fraud_rate must be between 0 and 1, got {fraud_rate}")
    return int(total_rows * fraud_rate)


def _generate_timestamps(rng: Generator, n_legit: int, n_fraud: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate timestamps with overnight fraud bias via rejection sampling."""
    max_seconds = _TIME_WINDOW_DAYS * _SECONDS_PER_DAY
    
    legit_times = rng.integers(0, max_seconds, size=n_legit, dtype=np.int32)
    
    fraud_times = np.empty(n_fraud, dtype=np.int32)
    idx = 0
    while idx < n_fraud:
        candidates = rng.integers(0, max_seconds, size=n_fraud - idx, dtype=np.int32)
        hours = (candidates // 3600) % 24
        
        overnight_mask = (hours >= _HOURS_OVERNIGHT_START) & (hours <= _HOURS_OVERNIGHT_END)
        acceptance_probs = np.where(overnight_mask, _FRAUD_OVERNIGHT_MULTIPLIER, 1.0)
        acceptance_probs = acceptance_probs / acceptance_probs.max()
        
        accepted = candidates[rng.random(size=len(candidates)) < acceptance_probs]
        accepted_count = len(accepted)
        
        if accepted_count > 0:
            end_idx = min(idx + accepted_count, n_fraud)
            fraud_times[idx:end_idx] = accepted[:end_idx - idx]
            idx = end_idx
    
    return legit_times, fraud_times


def _sample_categorical(
    rng: Generator,
    categories: Tuple[str, ...],
    probs: Tuple[float, ...],
    size: int
) -> np.ndarray:
    """Sample from categorical distribution."""
    _validate_probabilities(probs, "categorical")
    indices = rng.choice(len(categories), size=size, p=probs)
    return np.array([categories[i] for i in indices], dtype=object)


def _generate_amounts(
    rng: Generator,
    n_legit: int,
    n_fraud: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate transaction amounts for legit and fraud."""
    legit_amts = np.exp(rng.normal(_LEGIT_AMT_MEAN, _LEGIT_AMT_SIGMA, size=n_legit))
    
    fraud_small_count = int(n_fraud * _FRAUD_SMALL_WEIGHT)
    fraud_large_count = n_fraud - fraud_small_count
    
    fraud_small = np.exp(rng.normal(_FRAUD_SMALL_MEAN, _FRAUD_SMALL_SIGMA, size=fraud_small_count))
    fraud_large = np.exp(rng.normal(_FRAUD_LARGE_MEAN, _FRAUD_LARGE_SIGMA, size=fraud_large_count))
    
    fraud_amts = np.concatenate([fraud_small, fraud_large])
    rng.shuffle(fraud_amts)
    
    return legit_amts, fraud_amts


def _generate_categorical_features(
    rng: Generator,
    n_legit: int,
    n_fraud: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate card type and merchant category features."""
    legit_card = _sample_categorical(rng, _CARD_TYPES, _LEGIT_CARD_PROBS, n_legit)
    fraud_card = _sample_categorical(rng, _CARD_TYPES, _FRAUD_CARD_PROBS, n_fraud)
    
    fraud_merchant_mult = np.ones(len(_MERCHANT_CATEGORIES))
    for cat in _HIGH_RISK_CATEGORIES:
        cat_idx = _MERCHANT_CATEGORIES.index(cat)
        fraud_merchant_mult[cat_idx] = _FRAUD_MERCHANT_MULTIPLIER
    
    fraud_merchant_probs = np.array(_LEGIT_MERCHANT_PROBS) * fraud_merchant_mult
    fraud_merchant_probs = fraud_merchant_probs / fraud_merchant_probs.sum()
    
    legit_merchant = _sample_categorical(rng, _MERCHANT_CATEGORIES, _LEGIT_MERCHANT_PROBS, n_legit)
    fraud_merchant = _sample_categorical(
        rng, _MERCHANT_CATEGORIES, tuple(fraud_merchant_probs), n_fraud
    )
    
    return legit_card, fraud_card, legit_merchant, fraud_merchant


def _introduce_nulls(
    rng: Generator,
    card_type: np.ndarray,
    merchant_category: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly set some values to None to simulate missing data."""
    card_null_mask = rng.random(size=len(card_type)) < _NULL_CARD_RATE
    card_type_out = card_type.copy()
    card_type_out[card_null_mask] = None
    
    merchant_null_mask = rng.random(size=len(merchant_category)) < _NULL_MERCHANT_RATE
    merchant_category_out = merchant_category.copy()
    merchant_category_out[merchant_null_mask] = None
    
    return card_type_out, merchant_category_out


def generate_synthetic_data(config: GenerationConfig) -> GeneratedData:
    """Generate synthetic transaction dataset with exact fraud count."""
    if config.total_rows <= 0:
        raise CountError(f"total_rows must be positive, got {config.total_rows}")
    
    fraud_count = _compute_fraud_count(config.total_rows, config.fraud_rate)
    legit_count = config.total_rows - fraud_count
    
    rng = Generator(PCG64(config.seed))
    
    legit_times, fraud_times = _generate_timestamps(rng, legit_count, fraud_count)
    
    legit_amts, fraud_amts = _generate_amounts(rng, legit_count, fraud_count)
    
    legit_card, fraud_card, legit_merchant, fraud_merchant = _generate_categorical_features(
        rng, legit_count, fraud_count
    )
    
    legit_v1 = rng.normal(0.0, 1.0, size=legit_count)
    fraud_v1 = rng.normal(0.0, 1.0, size=fraud_count)
    
    transaction_dt = np.concatenate([legit_times, fraud_times])
    transaction_amt = np.concatenate([legit_amts, fraud_amts])
    card_type = np.concatenate([legit_card, fraud_card])
    merchant_category = np.concatenate([legit_merchant, fraud_merchant])
    is_fraud = np.concatenate([np.zeros(legit_count, dtype=np.int8), np.ones(fraud_count, dtype=np.int8)])
    v1 = np.concatenate([legit_v1, fraud_v1])
    
    shuffle_idx = rng.permutation(config.total_rows)
    
    card_type, merchant_category = _introduce_nulls(
        rng, card_type[shuffle_idx], merchant_category[shuffle_idx]
    )
    
    return GeneratedData(
        transaction_dt=transaction_dt[shuffle_idx],
        transaction_amt=transaction_amt[shuffle_idx],
        card_type=card_type,
        merchant_category=merchant_category,
        is_fraud=is_fraud[shuffle_idx],
        v1=v1[shuffle_idx]
    )


def _compute_hour_of_day(seconds: int) -> int:
    """Compute hour of day from timestamp seconds."""
    return (seconds // 3600) % 24


def _compute_day_of_week(seconds: int) -> int:
    """Compute day of week from timestamp seconds."""
    return (seconds // _SECONDS_PER_DAY) % 7


def _write_csv(data: GeneratedData, output_path: Path) -> None:
    """Write generated data to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "TransactionDT", "TransactionAmt", "card_type", "merchant_category",
            "is_fraud", "V1", "hour_of_day", "day_of_week"
        ])
        
        for i in range(len(data.transaction_dt)):
            dt = int(data.transaction_dt[i])
            writer.writerow([
                dt,
                f"{data.transaction_amt[i]:.6f}",
                data.card_type[i] if data.card_type[i] is not None else "",
                data.merchant_category[i] if data.merchant_category[i] is not None else "",
                int(data.is_fraud[i]),
                f"{data.v1[i]:.6f}",
                _compute_hour_of_day(dt),
                _compute_day_of_week(dt)
            ])


def main() -> None:
    """Entry point for synthetic data generation."""
    config = GenerationConfig(
        total_rows=_TOTAL_ROWS,
        fraud_rate=_FRAUD_RATE,
        seed=_RANDOM_SEED,
        output_path=_OUTPUT_PATH
    )
    
    try:
        data = generate_synthetic_data(config)
        _write_csv(data, config.output_path)
        
        fraud_count = int(data.is_fraud.sum())
        print(f"Generated {config.total_rows} rows", file=sys.stdout)
        print(f"Fraud count: {fraud_count}", file=sys.stdout)
        print(f"Fraud rate: {fraud_count / config.total_rows:.6f}", file=sys.stdout)
        print(f"Columns: TransactionDT, TransactionAmt, card_type, merchant_category, is_fraud, V1, hour_of_day, day_of_week", file=sys.stdout)
        print(f"Random seed: {config.seed}", file=sys.stdout)
        
    except (SyntheticDataError, OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()