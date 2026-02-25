import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from dataclasses import dataclass
from typing import Literal, Tuple, TypedDict, List, Union
from enum import Enum
from collections.abc import Sequence


class GenerationError(Enum):
    INVALID_CONFIG = "invalid_config"
    INVALID_PARAMETER = "invalid_parameter"
    NUMERIC_OVERFLOW = "numeric_overflow"
    SHAPE_MISMATCH = "shape_mismatch"
    DISTRIBUTION_FAILURE = "distribution_failure"


@dataclass(frozen=True)
class GeneratorConfig:
    n_samples: int
    n_signal_features: int
    n_noise_features: int
    fraud_rate: float
    cohens_d: float
    fraud_amount_mu_shift: float
    fraud_amount_sigma_scale: float
    categorical_cpt: dict[str, dict[Union[int, str], float]]
    missing_rate: float
    random_seed: int


class GeneratedMetadata(TypedDict):
    theoretical_max_pr_auc: float
    class_means: dict[int, np.ndarray]
    class_covariances: dict[int, np.ndarray]
    categorical_conditional_probabilities: dict[str, pd.DataFrame]
    config: GeneratorConfig


class GenerationResult(TypedDict):
    data: pd.DataFrame
    metadata: GeneratedMetadata


class _Logger:
    def __init__(self) -> None:
        self.logger = logging.getLogger("synthetic_data_generator")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(handler)

    def log_failure(self, error: GenerationError, message: str) -> None:
        self.logger.error(f"{error.value}: {message}")


_logger = _Logger()


def _validate_config(config: GeneratorConfig) -> None:
    if config.n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if config.n_signal_features < 1:
        raise ValueError("n_signal_features must be at least 1")
    if config.n_noise_features < 0:
        raise ValueError("n_noise_features cannot be negative")
    if not 0.0 < config.fraud_rate < 1.0:
        raise ValueError("fraud_rate must be strictly between 0 and 1")
    if config.cohens_d <= 0.0:
        raise ValueError("cohens_d must be positive")
    if not 0.0 <= config.missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    if config.fraud_amount_mu_shift <= 0.0:
        raise ValueError("fraud_amount_mu_shift must be positive")
    if config.fraud_amount_sigma_scale <= 0.0:
        raise ValueError("fraud_amount_sigma_scale must be positive")
    for cat_name, cpt in config.categorical_cpt.items():
        total_fraud = 0.0
        total_legit = 0.0
        for prob in cpt.values():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"CPT probabilities must be in [0,1] for {cat_name}")
            total_fraud += prob
            total_legit += 1.0 - prob
        if abs(total_fraud - 1.0) > 1e-6 or abs(total_legit - 1.0) > 1e-6:
            raise ValueError(f"CPT probabilities must sum to 1 for each class in {cat_name}")


def _compute_theoretical_pr_auc(
    mu_0: np.ndarray, mu_1: np.ndarray, sigma_0: np.ndarray, sigma_1: np.ndarray, fraud_rate: float
) -> float:
    signal_dim = mu_0.shape[0]
    pooled_var = (sigma_0 + sigma_1) / 2.0
    inv_pooled = np.linalg.inv(pooled_var)
    delta = mu_1 - mu_0
    mahalanobis = np.sqrt(delta.T @ inv_pooled @ delta)
    overlap = 2.0 * stats.norm.cdf(-mahalanobis / 2.0)
    pr_auc = 1.0 - overlap * (1.0 - fraud_rate) / (fraud_rate + (1.0 - fraud_rate) * (1.0 - overlap))
    return float(np.clip(pr_auc, 0.0, 1.0))


def _generate_class_labels(rng: np.random.Generator, n_samples: int, fraud_rate: float) -> np.ndarray:
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    labels = np.concatenate([np.zeros(n_legit, dtype=np.int8), np.ones(n_fraud, dtype=np.int8)])
    rng.shuffle(labels)
    return labels


def _generate_signal_features(
    rng: np.random.Generator,
    labels: np.ndarray,
    n_signal: int,
    cohens_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = labels.shape[0]
    mu_0 = np.zeros(n_signal)
    mu_1 = np.full(n_signal, cohens_d)
    sigma = np.eye(n_signal)

    features = np.zeros((n_samples, n_signal), dtype=np.float32)
    fraud_idx = labels == 1
    legit_idx = labels == 0

    n_fraud = np.sum(fraud_idx)
    n_legit = np.sum(legit_idx)

    if n_fraud > 0:
        features[fraud_idx] = rng.multivariate_normal(mu_1, sigma, size=n_fraud).astype(np.float32)
    if n_legit > 0:
        features[legit_idx] = rng.multivariate_normal(mu_0, sigma, size=n_legit).astype(np.float32)

    return features, mu_0, mu_1


def _generate_noise_features(
    rng: np.random.Generator, n_samples: int, n_noise: int
) -> np.ndarray:
    if n_noise == 0:
        return np.empty((n_samples, 0), dtype=np.float32)
    return rng.normal(0, 1, size=(n_samples, n_noise)).astype(np.float32)


def _generate_amounts(
    rng: np.random.Generator,
    labels: np.ndarray,
    mu_shift: float,
    sigma_scale: float,
) -> np.ndarray:
    n_samples = labels.shape[0]
    amounts = np.zeros(n_samples, dtype=np.float32)
    fraud_idx = labels == 1
    legit_idx = labels == 0

    legit_mu, legit_sigma = 3.0, 0.5
    fraud_mu = legit_mu + mu_shift
    fraud_sigma = legit_sigma * sigma_scale

    n_fraud = np.sum(fraud_idx)
    n_legit = np.sum(legit_idx)

    if n_legit > 0:
        legit_log = rng.normal(legit_mu, legit_sigma, size=n_legit)
        amounts[legit_idx] = np.exp(legit_log).astype(np.float32)
    if n_fraud > 0:
        fraud_log = rng.normal(fraud_mu, fraud_sigma, size=n_fraud)
        amounts[fraud_idx] = np.exp(fraud_log).astype(np.float32)

    if not np.all(np.isfinite(amounts)):
        raise OverflowError("Amount generation produced non-finite values")

    return amounts


def _generate_timestamps(
    rng: np.random.Generator, n_samples: int, base_date: str = "2024-01-01"
) -> pd.Series:
    interarrival_times = rng.exponential(scale=60.0, size=n_samples)
    cum_seconds = np.cumsum(interarrival_times).astype(np.int64)
    base_ts = pd.Timestamp(base_date)
    timestamps = pd.Series([base_ts + pd.Timedelta(seconds=int(s)) for s in cum_seconds])
    return timestamps


def _generate_categoricals(
    rng: np.random.Generator,
    labels: np.ndarray,
    cpt: dict[str, dict[Union[int, str], float]],
) -> pd.DataFrame:
    n_samples = labels.shape[0]
    categorical_dfs = {}

    for cat_name, probs in cpt.items():
        categories = list(probs.keys())
        legit_probs = np.array([1.0 - probs[cat] for cat in categories])
        legit_probs = legit_probs / legit_probs.sum()
        fraud_probs = np.array([probs[cat] for cat in categories])
        fraud_probs = fraud_probs / fraud_probs.sum()

        cat_values = np.empty(n_samples, dtype=object)
        fraud_idx = labels == 1
        legit_idx = labels == 0

        if np.sum(legit_idx) > 0:
            cat_values[legit_idx] = rng.choice(categories, size=np.sum(legit_idx), p=legit_probs)
        if np.sum(fraud_idx) > 0:
            cat_values[fraud_idx] = rng.choice(categories, size=np.sum(fraud_idx), p=fraud_probs)

        categorical_dfs[cat_name] = pd.Series(cat_values, dtype="category")

    return pd.DataFrame(categorical_dfs)


def _inject_missingness(
    rng: np.random.Generator, df: pd.DataFrame, missing_rate: float
) -> pd.DataFrame:
    if missing_rate == 0.0:
        return df

    df_missing = df.copy()
    mask = rng.random(size=df.shape) < missing_rate
    df_missing[mask] = np.nan
    return df_missing


def generate_synthetic_data(config: GeneratorConfig) -> GenerationResult:
    try:
        _validate_config(config)

        rng = np.random.default_rng(config.random_seed)

        labels = _generate_class_labels(rng, config.n_samples, config.fraud_rate)

        signal_features, mu_0, mu_1 = _generate_signal_features(
            rng, labels, config.n_signal_features, config.cohens_d
        )
        noise_features = _generate_noise_features(rng, config.n_samples, config.n_noise_features)

        signal_cols = [f"signal_{i}" for i in range(config.n_signal_features)]
        noise_cols = [f"noise_{i}" for i in range(config.n_noise_features)]

        signal_df = pd.DataFrame(signal_features, columns=signal_cols, dtype=np.float32)
        noise_df = pd.DataFrame(noise_features, columns=noise_cols, dtype=np.float32)

        amounts = _generate_amounts(
            rng, labels, config.fraud_amount_mu_shift, config.fraud_amount_sigma_scale
        )
        timestamps = _generate_timestamps(rng, config.n_samples)

        categoricals = _generate_categoricals(rng, labels, config.categorical_cpt)

        df = pd.concat([signal_df, noise_df, categoricals], axis=1)
        df["transaction_amount"] = amounts
        df["timestamp"] = timestamps
        df["is_fraud"] = pd.Series(labels, dtype=np.int8)

        sigma_0 = np.eye(config.n_signal_features)
        sigma_1 = np.eye(config.n_signal_features)

        theoretical_pr_auc = _compute_theoretical_pr_auc(
            mu_0, mu_1, sigma_0, sigma_1, config.fraud_rate
        )

        df = _inject_missingness(rng, df, config.missing_rate)

        categorical_cpt_dfs = {}
        for cat_name, probs in config.categorical_cpt.items():
            categories = list(probs.keys())
            legit_probs = np.array([1.0 - probs[cat] for cat in categories])
            legit_probs = legit_probs / legit_probs.sum()
            fraud_probs = np.array([probs[cat] for cat in categories])
            fraud_probs = fraud_probs / fraud_probs.sum()
            cpt_df = pd.DataFrame({"legitimate": legit_probs, "fraud": fraud_probs}, index=categories)
            categorical_cpt_dfs[cat_name] = cpt_df

        metadata: GeneratedMetadata = {
            "theoretical_max_pr_auc": theoretical_pr_auc,
            "class_means": {0: mu_0, 1: mu_1},
            "class_covariances": {0: sigma_0, 1: sigma_1},
            "categorical_conditional_probabilities": categorical_cpt_dfs,
            "config": config,
        }

        return {"data": df, "metadata": metadata}

    except ValueError as e:
        _logger.log_failure(GenerationError.INVALID_CONFIG, str(e))
        raise
    except OverflowError as e:
        _logger.log_failure(GenerationError.NUMERIC_OVERFLOW, str(e))
        raise
    except Exception as e:
        _logger.log_failure(GenerationError.DISTRIBUTION_FAILURE, str(e))
        raise


__all__ = [
    "GeneratorConfig",
    "GenerationError",
    "GeneratedMetadata",
    "GenerationResult",
    "generate_synthetic_data",
]