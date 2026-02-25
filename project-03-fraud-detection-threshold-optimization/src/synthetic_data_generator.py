import logging
from typing import Final, TypedDict, assert_never
from enum import Enum
from dataclasses import dataclass
from math import exp, sqrt, pi, erf
import random
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import poisson, multivariate_normal, lognorm


# --- Structured Logging Setup ---
_logger = logging.getLogger(__name__)
_LOG_RECORD: Final = {"component": "synthetic_data_generator"}


# --- Failure Enumeration ---
class SyntheticDataGenerationError(Enum):
    INVALID_FRAUD_RATE = "invalid_fraud_rate"
    INVALID_EFFECT_SIZE = "invalid_effect_size"
    INVALID_NUM_FEATURES = "invalid_num_features"
    INVALID_CATEGORY_CONFIG = "invalid_category_config"
    INVALID_MISSING_RATE = "invalid_missing_rate"
    INVALID_PARAMETER_COMBINATION = "invalid_parameter_combination"
    COVARIANCE_MATRIX_SINGULAR = "covariance_matrix_singular"
    SEED_FAILURE = "seed_failure"
    UNEXPECTED_GENERATION_ERROR = "unexpected_generation_error"


# --- Immutable Configuration Types ---
@dataclass(frozen=True)
class GaussianParameters:
    """Parameters for class-conditional Gaussian distributions."""
    mean_legit: np.ndarray
    mean_fraud: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class CategoricalFeatureConfig:
    """Configuration for a categorical feature."""
    name: str
    categories: tuple[str, ...]
    conditional_probabilities: dict[str, dict[int, float]]  # category -> {class: probability}


@dataclass(frozen=True)
class GenerationConfig:
    """Complete immutable configuration for data generation."""
    num_samples: int
    fraud_rate: float
    effect_size: float
    num_signal_features: int
    num_noise_features: int
    amount_mu_legit: float
    amount_sigma_legit: float
    amount_mu_fraud: float
    amount_sigma_fraud: float
    poisson_lambda: float
    categorical_features: tuple[CategoricalFeatureConfig, ...]
    missing_rate: float
    random_seed: int
    dtype_numeric: np.dtype
    dtype_category: str


# --- Canonical Validation Logic ---
def _validate_fraud_rate(rate: float) -> float:
    """Validate fraud rate is between 0 and 1."""
    if not 0 < rate < 1:
        raise ValueError(f"fraud_rate must be between 0 and 1, got {rate}")
    return rate


def _validate_effect_size(size: float) -> float:
    """Validate effect size is positive."""
    if size <= 0:
        raise ValueError(f"effect_size must be positive, got {size}")
    return size


def _validate_feature_counts(num_signal: int, num_noise: int) -> tuple[int, int]:
    """Validate feature counts are non-negative."""
    if num_signal < 0 or num_noise < 0:
        raise ValueError("feature counts must be non-negative")
    if num_signal + num_noise == 0:
        raise ValueError("must have at least one feature")
    return num_signal, num_noise


def _validate_amount_params(
    mu_legit: float,
    sigma_legit: float,
    mu_fraud: float,
    sigma_fraud: float,
) -> tuple[float, float, float, float]:
    """Validate log-normal parameters are valid."""
    if sigma_legit <= 0 or sigma_fraud <= 0:
        raise ValueError("sigma parameters must be positive")
    return mu_legit, sigma_legit, mu_fraud, sigma_fraud


def _validate_poisson_lambda(lambda_val: float) -> float:
    """Validate Poisson lambda is positive."""
    if lambda_val <= 0:
        raise ValueError(f"poisson_lambda must be positive, got {lambda_val}")
    return lambda_val


def _validate_categorical_features(
    features: Sequence[CategoricalFeatureConfig],
) -> tuple[CategoricalFeatureConfig, ...]:
    """Validate categorical feature configurations."""
    for feat in features:
        if not feat.categories:
            raise ValueError(f"feature {feat.name} has no categories")
        if not feat.conditional_probabilities:
            raise ValueError(f"feature {feat.name} has no conditional probabilities")
        
        # Validate probability distributions sum to 1 for each class
        for class_label in (0, 1):
            total = 0.0
            for probs in feat.conditional_probabilities.values():
                if class_label in probs:
                    total += probs[class_label]
            if not 0.99 < total < 1.01:  # Allow small floating point tolerance
                raise ValueError(
                    f"feature {feat.name} probabilities for class {class_label} sum to {total}, must be 1"
                )
    return tuple(features)


def _validate_missing_rate(rate: float) -> float:
    """Validate missing rate is between 0 and 1."""
    if not 0 <= rate < 1:
        raise ValueError(f"missing_rate must be between 0 and 1, got {rate}")
    return rate


def _validate_generation_config(config: GenerationConfig) -> GenerationConfig:
    """Canonical validation of all generation parameters."""
    _validate_fraud_rate(config.fraud_rate)
    _validate_effect_size(config.effect_size)
    _validate_feature_counts(config.num_signal_features, config.num_noise_features)
    _validate_amount_params(
        config.amount_mu_legit,
        config.amount_sigma_legit,
        config.amount_mu_fraud,
        config.amount_sigma_fraud,
    )
    _validate_poisson_lambda(config.poisson_lambda)
    _validate_categorical_features(config.categorical_features)
    _validate_missing_rate(config.missing_rate)
    return config


# --- Pure Core Logic (Statistical Generation) ---
def _compute_theoretical_pr_auc(
    effect_size: float,
    fraud_rate: float,
) -> float:
    """
    Compute theoretical maximum PR-AUC based on Gaussian separation.
    Uses approximation based on Cohen's d and class imbalance.
    """
    # Simplified theoretical maximum based on separation and imbalance
    # More sophisticated calculation would use integral of precision-recall curve
    d_prime = effect_size / sqrt(2)  # Convert Cohen's d to discriminability
    theoretical_auc = 0.5 + 0.5 * erf(d_prime / 2)
    
    # Adjust for imbalance - PR-AUC is lower than ROC-AUC with imbalance
    # This is an approximation for the "Goldilocks" zone (0.70-0.90)
    imbalance_penalty = 0.15 * (1 - 4 * abs(fraud_rate - 0.5))  # Max penalty at 50/50
    adjusted_auc = theoretical_auc - imbalance_penalty
    
    # Clamp to reasonable range
    return max(0.5, min(0.95, adjusted_auc))


def _build_gaussian_parameters(
    num_signal: int,
    effect_size: float,
    random_gen: np.random.Generator,
) -> GaussianParameters:
    """
    Construct Gaussian parameters with controlled separation.
    Fraud class mean shifted by effect_size in random direction.
    """
    if num_signal == 0:
        # No signal features, return identical distributions
        return GaussianParameters(
            mean_legit=np.zeros(0),
            mean_fraud=np.zeros(0),
            covariance=np.zeros((0, 0)),
        )
    
    # Generate random orthonormal basis for signal features
    basis = random_gen.standard_normal((num_signal, num_signal))
    q, _ = np.linalg.qr(basis)  # Orthonormal basis
    
    # Create covariance matrix (identity for simplicity, but could be structured)
    covariance = np.eye(num_signal)
    
    # Shift fraud mean along first basis vector by effect_size
    shift_vector = q[:, 0] * effect_size
    mean_fraud = shift_vector
    
    return GaussianParameters(
        mean_legit=np.zeros(num_signal),
        mean_fraud=mean_fraud,
        covariance=covariance,
    )


def _generate_class_labels(
    num_samples: int,
    fraud_rate: float,
    random_gen: np.random.Generator,
) -> np.ndarray:
    """Generate binary class labels with exact fraud rate."""
    num_fraud = int(num_samples * fraud_rate)
    num_legit = num_samples - num_fraud
    
    labels = np.concatenate([
        np.zeros(num_legit, dtype=np.int8),
        np.ones(num_fraud, dtype=np.int8),
    ])
    random_gen.shuffle(labels)
    return labels


def _generate_signal_features(
    labels: np.ndarray,
    gaussian_params: GaussianParameters,
    random_gen: np.random.Generator,
) -> np.ndarray:
    """Generate signal features from class-conditional Gaussians."""
    if gaussian_params.mean_legit.size == 0:
        return np.empty((len(labels), 0))
    
    features = np.zeros((len(labels), gaussian_params.mean_legit.size))
    
    legit_mask = labels == 0
    fraud_mask = labels == 1
    
    if np.any(legit_mask):
        legit_samples = multivariate_normal.rvs(
            mean=gaussian_params.mean_legit,
            cov=gaussian_params.covariance,
            size=np.sum(legit_mask),
            random_state=random_gen,
        )
        features[legit_mask] = legit_samples
    
    if np.any(fraud_mask):
        fraud_samples = multivariate_normal.rvs(
            mean=gaussian_params.mean_fraud,
            cov=gaussian_params.covariance,
            size=np.sum(fraud_mask),
            random_state=random_gen,
        )
        features[fraud_mask] = fraud_samples
    
    return features


def _generate_noise_features(
    num_samples: int,
    num_noise: int,
    random_gen: np.random.Generator,
) -> np.ndarray:
    """Generate pure noise features (independent of labels)."""
    if num_noise == 0:
        return np.empty((num_samples, 0))
    return random_gen.standard_normal((num_samples, num_noise))


def _generate_amounts(
    labels: np.ndarray,
    mu_legit: float,
    sigma_legit: float,
    mu_fraud: float,
    sigma_fraud: float,
    random_gen: np.random.Generator,
) -> np.ndarray:
    """Generate transaction amounts from log-normal distributions."""
    amounts = np.zeros(len(labels))
    
    legit_mask = labels == 0
    fraud_mask = labels == 1
    
    if np.any(legit_mask):
        legit_amounts = lognorm.rvs(
            s=sigma_legit,
            scale=exp(mu_legit),
            size=np.sum(legit_mask),
            random_state=random_gen,
        )
        amounts[legit_mask] = legit_amounts
    
    if np.any(fraud_mask):
        fraud_amounts = lognorm.rvs(
            s=sigma_fraud,
            scale=exp(mu_fraud),
            size=np.sum(fraud_mask),
            random_state=random_gen,
        )
        amounts[fraud_mask] = fraud_amounts
    
    return amounts


def _generate_timestamps(
    num_samples: int,
    poisson_lambda: float,
    random_gen: np.random.Generator,
) -> pd.DatetimeIndex:
    """Generate timestamps using Poisson process."""
    # Generate inter-arrival times
    interarrival_times = poisson.rvs(
        mu=poisson_lambda,
        size=num_samples - 1,
        random_state=random_gen,
    )
    
    # Convert to cumulative seconds
    cumulative_seconds = np.concatenate([[0], np.cumsum(interarrival_times)])
    
    # Convert to datetime (starting from a fixed reference point)
    base_time = pd.Timestamp("2024-01-01")
    timestamps = base_time + pd.to_timedelta(cumulative_seconds, unit="s")
    
    return timestamps


def _generate_categorical_features(
    labels: np.ndarray,
    configs: tuple[CategoricalFeatureConfig, ...],
    random_gen: np.random.Generator,
) -> dict[str, pd.Series]:
    """Generate categorical features using conditional probability tables."""
    features = {}
    
    for config in configs:
        categories = config.categories
        probs = config.conditional_probabilities
        
        # For each sample, choose category based on class
        feature_values = []
        for label in labels:
            # Sample from categorical distribution for this class
            class_probs = np.array([probs[cat][label] for cat in categories])
            class_probs /= class_probs.sum()  # Normalize to ensure exact sum
            chosen = random_gen.choice(categories, p=class_probs)
            feature_values.append(chosen)
        
        features[config.name] = pd.Series(
            feature_values,
            dtype=config.dtype_category,
        )
    
    return features


def _inject_missingness(
    df: pd.DataFrame,
    missing_rate: float,
    random_gen: np.random.Generator,
) -> pd.DataFrame:
    """Inject MCAR missingness into numerical features."""
    if missing_rate == 0:
        return df
    
    df_missing = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        mask = random_gen.random(len(df)) < missing_rate
        df_missing.loc[mask, col] = np.nan
    
    return df_missing


# --- Public API ---
def generate_synthetic_data(
    config: GenerationConfig,
) -> tuple[pd.DataFrame, dict[str, float]] | SyntheticDataGenerationError:
    """
    Generate synthetic fraud detection dataset with controlled statistical properties.

    Args:
        config: GenerationConfig object containing all generation parameters.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]:
            - DataFrame with generated features and target
            - Metadata dict containing theoretical max PR-AUC and other statistics
        SyntheticDataGenerationError: Enumerated failure reason on error.

    Examples:
        >>> config = GenerationConfig(
        ...     num_samples=10000,
        ...     fraud_rate=0.01,
        ...     effect_size=0.8,
        ...     num_signal_features=5,
        ...     num_noise_features=10,
        ...     amount_mu_legit=2.0,
        ...     amount_sigma_legit=0.5,
        ...     amount_mu_fraud=3.5,
        ...     amount_sigma_fraud=0.8,
        ...     poisson_lambda=0.1,
        ...     categorical_features=(),
        ...     missing_rate=0.005,
        ...     random_seed=42,
        ...     dtype_numeric=np.float32,
        ...     dtype_category="category",
        ... )
        >>> result = generate_synthetic_data(config)
        >>> if isinstance(result, SyntheticDataGenerationError):
        ...     print(f"Generation failed: {result.value}")
        ... else:
        ...     df, metadata = result
        ...     print(f"Theoretical PR-AUC: {metadata['theoretical_max_pr_auc']:.3f}")
    """
    log_data = {**_LOG_RECORD}
    _logger.debug("Starting synthetic data generation", extra=log_data)

    # --- Input Validation Boundary ---
    try:
        validated_config = _validate_generation_config(config)
    except ValueError as e:
        log_data["error"] = str(e)
        error_msg = str(e)
        if "fraud_rate" in error_msg:
            _logger.error("Invalid fraud rate", extra=log_data)
            return SyntheticDataGenerationError.INVALID_FRAUD_RATE
        elif "effect_size" in error_msg:
            _logger.error("Invalid effect size", extra=log_data)
            return SyntheticDataGenerationError.INVALID_EFFECT_SIZE
        elif "feature" in error_msg:
            _logger.error("Invalid feature count", extra=log_data)
            return SyntheticDataGenerationError.INVALID_NUM_FEATURES
        elif "category" in error_msg.lower():
            _logger.error("Invalid category configuration", extra=log_data)
            return SyntheticDataGenerationError.INVALID_CATEGORY_CONFIG
        elif "missing_rate" in error_msg:
            _logger.error("Invalid missing rate", extra=log_data)
            return SyntheticDataGenerationError.INVALID_MISSING_RATE
        else:
            _logger.error("Invalid parameter combination", extra=log_data)
            return SyntheticDataGenerationError.INVALID_PARAMETER_COMBINATION

    # --- Seed Setup for Reproducibility ---
    try:
        random.seed(validated_config.random_seed)
        np.random.seed(validated_config.random_seed)
        rng = np.random.default_rng(validated_config.random_seed)
    except Exception as e:
        log_data["error"] = str(e)
        _logger.critical("Failed to set random seed", extra=log_data)
        return SyntheticDataGenerationError.SEED_FAILURE

    # --- Generation Pipeline ---
    try:
        # 1. Compute theoretical maximum PR-AUC
        theoretical_pr_auc = _compute_theoretical_pr_auc(
            validated_config.effect_size,
            validated_config.fraud_rate,
        )

        # 2. Build Gaussian parameters for signal features
        try:
            gaussian_params = _build_gaussian_parameters(
                validated_config.num_signal_features,
                validated_config.effect_size,
                rng,
            )
        except np.linalg.LinAlgError as e:
            log_data["error"] = str(e)
            _logger.error("Failed to build Gaussian parameters", extra=log_data)
            return SyntheticDataGenerationError.COVARIANCE_MATRIX_SINGULAR

        # 3. Generate class labels
        labels = _generate_class_labels(
            validated_config.num_samples,
            validated_config.fraud_rate,
            rng,
        )

        # 4. Generate features
        signal_features = _generate_signal_features(labels, gaussian_params, rng)
        noise_features = _generate_noise_features(
            validated_config.num_samples,
            validated_config.num_noise_features,
            rng,
        )

        # 5. Generate amounts
        amounts = _generate_amounts(
            labels,
            validated_config.amount_mu_legit,
            validated_config.amount_sigma_legit,
            validated_config.amount_mu_fraud,
            validated_config.amount_sigma_fraud,
            rng,
        )

        # 6. Generate timestamps
        timestamps = _generate_timestamps(
            validated_config.num_samples,
            validated_config.poisson_lambda,
            rng,
        )

        # 7. Generate categorical features
        categorical_features = _generate_categorical_features(
            labels,
            validated_config.categorical_features,
            rng,
        )

        # 8. Assemble DataFrame
        feature_names = (
            [f"signal_{i}" for i in range(signal_features.shape[1])] +
            [f"noise_{i}" for i in range(noise_features.shape[1])]
        )
        
        all_features = np.hstack([
            signal_features,
            noise_features,
            amounts.reshape(-1, 1),
        ])
        
        df = pd.DataFrame(
            all_features,
            columns=feature_names + ["amount"],
        )
        
        # Convert to specified dtype
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].astype(validated_config.dtype_numeric)
        
        # Add timestamps
        df["timestamp"] = timestamps
        
        # Add categorical features
        for name, series in categorical_features.items():
            df[name] = series
        
        # Add target
        df["is_fraud"] = labels

        # 9. Inject missingness
        df = _inject_missingness(df, validated_config.missing_rate, rng)

        # 10. Build metadata
        metadata = {
            "theoretical_max_pr_auc": theoretical_pr_auc,
            "actual_fraud_rate": float(labels.mean()),
            "num_samples": validated_config.num_samples,
            "num_features": len(feature_names) + len(categorical_features) + 2,  # + amount, timestamp
            "effect_size": validated_config.effect_size,
            "missing_rate_injected": validated_config.missing_rate,
            "random_seed_used": validated_config.random_seed,
        }

        _logger.info(
            "Synthetic data generated successfully",
            extra={**log_data, **metadata},
        )
        
        return df, metadata

    except Exception as e:
        log_data["error"] = str(e)
        _logger.critical(
            "Unexpected error during generation",
            extra=log_data,
            exc_info=True,
        )
        return SyntheticDataGenerationError.UNEXPECTED_GENERATION_ERROR