import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import math

@dataclass(frozen=True)
class GenerationConfig:
    n_rows: int = 200000
    fraud_rate: float = 0.01
    random_seed: int = 42
    output_path: str = "D:/applied-finance-ml/project-03-fraud-detection-threshold-optimization/data/raw/transactions.csv"

@dataclass(frozen=True)
class DistributionParams:
    legit_amount_mu: float = 3.5
    legit_amount_sigma: float = 1.1
    fraud_small_mu: float = 2.0
    fraud_small_sigma: float = 0.6
    fraud_large_mu: float = 5.5
    fraud_large_sigma: float = 0.8
    fraud_mixture_weight_small: float = 0.7
    fraud_mixture_weight_large: float = 0.3
    days_window: int = 180
    seconds_per_day: int = 86400
    overnight_hour_start: int = 1
    overnight_hour_end: int = 5
    overnight_fraud_multiplier: float = 2.5

@dataclass(frozen=True)
class CategoricalProbabilities:
    card_type_legit: Tuple[float, ...] = (0.52, 0.30, 0.12, 0.06)  # Visa, MC, Amex, Discover
    card_type_fraud: Tuple[float, ...] = (0.45, 0.28, 0.20, 0.07)
    merchant_categories: Tuple[str, ...] = (
        "grocery", "restaurant", "online_retail", "travel",
        "entertainment", "utilities", "healthcare", "automotive"
    )
    merchant_legit_prior: Tuple[float, ...] = (
        0.18, 0.15, 0.22, 0.10, 0.12, 0.08, 0.09, 0.06
    )
    fraud_overindex_factors: Tuple[float, ...] = (
        1.0, 1.0, 1.8, 1.8, 1.0, 1.0, 1.0, 1.0
    )
    card_type_labels: Tuple[str, ...] = ("Visa", "Mastercard", "Amex", "Discover")

@dataclass(frozen=True)
class NullRates:
    merchant_category_null_rate: float = 0.03
    card_type_null_rate: float = 0.015

class SyntheticDataGenerationError(Exception):
    """Base exception for synthetic data generation failures."""
    pass

class InvalidParameterError(SyntheticDataGenerationError):
    """Raised when generation parameters are invalid."""
    pass

class ProbabilityVectorError(SyntheticDataGenerationError):
    """Raised when probability vectors are invalid."""
    pass

class RandomNumberGenerationError(SyntheticDataGenerationError):
    """Raised when random number generation fails."""
    pass

def _validate_probability_vector(probs: Tuple[float, ...], name: str) -> None:
    if abs(sum(probs) - 1.0) > 1e-10:
        raise ProbabilityVectorError(f"{name} probabilities must sum to 1.0, got {sum(probs)}")
    for p in probs:
        if not (0.0 <= p <= 1.0):
            raise ProbabilityVectorError(f"{name} contains invalid probability: {p}")

def _validate_generation_params(config: GenerationConfig) -> None:
    if config.n_rows <= 0:
        raise InvalidParameterError(f"n_rows must be positive, got {config.n_rows}")
    if not (0.0 <= config.fraud_rate <= 1.0):
        raise InvalidParameterError(f"fraud_rate must be in [0,1], got {config.fraud_rate}")
    if not isinstance(config.random_seed, int) or config.random_seed < 0:
        raise InvalidParameterError(f"random_seed must be non-negative integer, got {config.random_seed}")

def _compute_overnight_adjustment(
    hours: np.ndarray, 
    legit_probs: np.ndarray,
    params: DistributionParams
) -> np.ndarray:
    fraud_mult = np.ones_like(hours)
    overnight_mask = (hours >= params.overnight_hour_start) & (hours <= params.overnight_hour_end)
    fraud_mult[overnight_mask] = params.overnight_fraud_multiplier
    
    adjusted = legit_probs * fraud_mult
    return adjusted / adjusted.sum()

def _sample_categorical(
    rng: np.random.Generator,
    size: int,
    categories: Tuple[str, ...],
    probabilities: Tuple[float, ...],
    name: str
) -> List[str]:
    _validate_probability_vector(probabilities, name)
    indices = rng.choice(len(categories), size=size, p=probabilities)
    return [categories[i] for i in indices]

def _introduce_nulls(
    series: List[str],
    null_rate: float,
    rng: np.random.Generator,
    name: str
) -> List[str]:
    if not (0.0 <= null_rate <= 1.0):
        raise InvalidParameterError(f"null_rate for {name} must be in [0,1], got {null_rate}")
    
    mask = rng.random(len(series)) < null_rate
    result = series.copy()
    for i in range(len(result)):
        if mask[i]:
            result[i] = np.nan
    return result

def generate_synthetic_data(config: GenerationConfig) -> pd.DataFrame:
    _validate_generation_params(config)
    
    params = DistributionParams()
    cat_probs = CategoricalProbabilities()
    null_rates = NullRates()
    
    _validate_probability_vector(cat_probs.card_type_legit, "card_type_legit")
    _validate_probability_vector(cat_probs.card_type_fraud, "card_type_fraud")
    _validate_probability_vector(cat_probs.merchant_legit_prior, "merchant_legit_prior")
    
    fraud_count = int(math.floor(config.n_rows * config.fraud_rate))
    legit_count = config.n_rows - fraud_count
    
    if legit_count < 0 or fraud_count < 0:
        raise InvalidParameterError(f"Invalid counts: legit={legit_count}, fraud={fraud_count}")
    
    rng = np.random.default_rng(config.random_seed)
    
    try:
        # Legitimate transactions
        legit_amounts = rng.lognormal(
            mean=params.legit_amount_mu,
            sigma=params.legit_amount_sigma,
            size=legit_count
        )
        
        # Fraudulent transactions (mixture distribution)
        fraud_mask = rng.random(fraud_count) < params.fraud_mixture_weight_small
        fraud_small_count = np.sum(fraud_mask)
        fraud_large_count = fraud_count - fraud_small_count
        
        fraud_small_amounts = rng.lognormal(
            mean=params.fraud_small_mu,
            sigma=params.fraud_small_sigma,
            size=fraud_small_count
        )
        fraud_large_amounts = rng.lognormal(
            mean=params.fraud_large_mu,
            sigma=params.fraud_large_sigma,
            size=fraud_large_count
        )
        fraud_amounts = np.concatenate([fraud_small_amounts, fraud_large_amounts])
        
        # Temporal features
        max_seconds = params.days_window * params.seconds_per_day
        legit_seconds = rng.integers(0, max_seconds, size=legit_count)
        fraud_seconds = rng.integers(0, max_seconds, size=fraud_count)
        
        legit_hours = (legit_seconds // 3600) % 24
        fraud_hours = (fraud_seconds // 3600) % 24
        
        legit_days = legit_seconds // params.seconds_per_day
        fraud_days = fraud_seconds // params.seconds_per_day
        
        legit_weekdays = (legit_days + 1) % 7  # Monday=0, offset by 1 for epoch
        fraud_weekdays = (fraud_days + 1) % 7
        
        # Apply overnight fraud adjustment via rejection sampling
        legit_hour_probs = np.ones(24) / 24.0
        adjusted_fraud_hour_probs = _compute_overnight_adjustment(
            np.arange(24), legit_hour_probs, params
        )
        
        fraud_hours_adjusted = rng.choice(24, size=fraud_count, p=adjusted_fraud_hour_probs)
        fraud_seconds_adjusted = (
            fraud_hours_adjusted * 3600 + 
            rng.integers(0, 3600, size=fraud_count)
        )
        fraud_days_adjusted = rng.integers(0, params.days_window, size=fraud_count)
        fraud_seconds_adjusted = fraud_days_adjusted * params.seconds_per_day + fraud_seconds_adjusted
        
        # Categorical features
        legit_card_types = _sample_categorical(
            rng, legit_count, cat_probs.card_type_labels, 
            cat_probs.card_type_legit, "card_type_legit"
        )
        fraud_card_types = _sample_categorical(
            rng, fraud_count, cat_probs.card_type_labels,
            cat_probs.card_type_fraud, "card_type_fraud"
        )
        
        fraud_merchant_probs = np.array(cat_probs.merchant_legit_prior) * np.array(cat_probs.fraud_overindex_factors)
        fraud_merchant_probs = fraud_merchant_probs / fraud_merchant_probs.sum()
        
        legit_merchants = _sample_categorical(
            rng, legit_count, cat_probs.merchant_categories,
            cat_probs.merchant_legit_prior, "merchant_legit_prior"
        )
        fraud_merchants = _sample_categorical(
            rng, fraud_count, cat_probs.merchant_categories,
            tuple(fraud_merchant_probs.tolist()), "merchant_fraud"
        )
        
        # Introduce nulls
        legit_merchants = _introduce_nulls(
            legit_merchants, null_rates.merchant_category_null_rate, 
            rng, "merchant_category"
        )
        fraud_merchants = _introduce_nulls(
            fraud_merchants, null_rates.merchant_category_null_rate,
            rng, "merchant_category"
        )
        legit_card_types = _introduce_nulls(
            legit_card_types, null_rates.card_type_null_rate,
            rng, "card_type"
        )
        fraud_card_types = _introduce_nulls(
            fraud_card_types, null_rates.card_type_null_rate,
            rng, "card_type"
        )
        
        # Noise feature V1
        legit_v1 = rng.normal(0, 1, legit_count)
        fraud_v1 = rng.normal(0, 1, fraud_count)
        
        # Assemble DataFrames
        legit_df = pd.DataFrame({
            "TransactionAmt": legit_amounts,
            "TransactionDT": legit_seconds,
            "hour_of_day": legit_hours,
            "day_of_week": legit_weekdays,
            "card_type": legit_card_types,
            "merchant_category": legit_merchants,
            "V1": legit_v1,
            "is_fraud": [0] * legit_count
        })
        
        fraud_df = pd.DataFrame({
            "TransactionAmt": fraud_amounts,
            "TransactionDT": fraud_seconds_adjusted,
            "hour_of_day": (fraud_seconds_adjusted // 3600) % 24,
            "day_of_week": ((fraud_seconds_adjusted // params.seconds_per_day) + 1) % 7,
            "card_type": fraud_card_types,
            "merchant_category": fraud_merchants,
            "V1": fraud_v1,
            "is_fraud": [1] * fraud_count
        })
        
        combined_df = pd.concat([legit_df, fraud_df], ignore_index=True)
        
        # Shuffle
        shuffle_indices = rng.permutation(len(combined_df))
        combined_df = combined_df.iloc[shuffle_indices].reset_index(drop=True)
        
        # Final validation
        if len(combined_df) != config.n_rows:
            raise RandomNumberGenerationError(
                f"Generated {len(combined_df)} rows, expected {config.n_rows}"
            )
        if combined_df["is_fraud"].sum() != fraud_count:
            raise RandomNumberGenerationError(
                f"Generated {combined_df['is_fraud'].sum()} fraud rows, expected {fraud_count}"
            )
        
        return combined_df
        
    except Exception as e:
        raise RandomNumberGenerationError(f"Failed to generate synthetic data: {str(e)}") from e

def main() -> None:
    config = GenerationConfig()
    
    try:
        df = generate_synthetic_data(config)
        
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        
        fraud_count = df["is_fraud"].sum()
        actual_fraud_rate = fraud_count / len(df)
        
        print(f"Total rows: {len(df)}")
        print(f"Fraud count: {fraud_count}")
        print(f"Fraud rate: {actual_fraud_rate:.6f}")
        print(f"Columns: {list(df.columns)}")
        print(f"Random seed: {config.random_seed}")
        print(f"Output written to: {output_path}")
        
    except SyntheticDataGenerationError as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()