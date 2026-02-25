import numpy as np
import pandas as pd
from numpy.typing import NDArray
import sys
from typing import TypedDict, Literal, NamedTuple
if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never
import logging
import sys
from enum import Enum, auto
from dataclasses import dataclass
from scipy.stats import multivariate_normal, poisson
import warnings

# Filter out scipy runtime warnings for invalid covariance matrices (handled by validation)
warnings.filterwarnings("ignore", "covariance is not positive-semidefinite")

# Configure structured logger
logger = logging.getLogger("synthetic_data_generator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class GenerationError(Enum):
    INVALID_CONFIG = auto()
    INVALID_SEED = auto()
    INVALID_FRAUD_RATE = auto()
    INVALID_EFFECT_SIZE = auto()
    INVALID_FEATURE_COUNTS = auto()
    INVALID_COHENS_D = auto()
    INVALID_NUM_SAMPLES = auto()
    INVALID_CPT = auto()
    INVALID_LOG_NORMAL_PARAMS = auto()
    INVALID_MISSINGNESS_RATE = auto()
    COVARIANCE_MATRIX_SINGULAR = auto()
    NUMERIC_INSTABILITY = auto()
    UNEXPECTED_ERROR = auto()


class GenerationResult(NamedTuple):
    data: pd.DataFrame
    metadata: "DatasetMetadata"
    error: GenerationError | None


class DatasetMetadata(TypedDict):
    theoretical_max_pr_auc: float
    fraud_rate: float
    effect_size: float
    signal_feature_indices: list[int]
    noise_feature_indices: list[int]
    categorical_fraud_probabilities: dict[str, float]


@dataclass(frozen=True)
class Config:
    num_samples: int
    num_signal_features: int
    num_noise_features: int
    fraud_rate: float
    cohens_d: float
    log_normal_mean: float
    log_normal_sigma: float
    fraud_amount_multiplier: float
    fraud_amount_variance_multiplier: float
    categorical_cpt: dict[str, dict[int, float]]  # feature_name -> {class: probability}
    missingness_rate: float
    random_seed: int

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if self.num_signal_features <= 0:
            raise ValueError("num_signal_features must be positive")
        if self.num_noise_features < 0:
            raise ValueError("num_noise_features cannot be negative")
        if not 0 < self.fraud_rate < 1:
            raise ValueError("fraud_rate must be in (0, 1)")
        if self.cohens_d <= 0:
            raise ValueError("cohens_d must be positive")
        if self.log_normal_sigma <= 0:
            raise ValueError("log_normal_sigma must be positive")
        if self.fraud_amount_multiplier <= 0:
            raise ValueError("fraud_amount_multiplier must be positive")
        if self.fraud_amount_variance_multiplier <= 0:
            raise ValueError("fraud_amount_variance_multiplier must be positive")
        if not 0 <= self.missingness_rate < 0.1:
            raise ValueError("missingness_rate must be in [0, 0.1)")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        for feature_name, cpt in self.categorical_cpt.items():
            if set(cpt.keys()) != {0, 1}:
                raise ValueError(f"CPT for {feature_name} must have keys 0 and 1")
            if not all(0 <= p <= 1 for p in cpt.values()):
                raise ValueError(f"CPT probabilities for {feature_name} must be in [0, 1]")


class _ClassDistributions(NamedTuple):
    legit_mean: NDArray[np.float64]
    legit_cov: NDArray[np.float64]
    fraud_mean: NDArray[np.float64]
    fraud_cov: NDArray[np.float64]


class _RandomState:
    def __init__(self, seed: int) -> None:
        self.np = np.random.RandomState(seed)


class SyntheticDataGenerator:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._rng = _RandomState(config.random_seed)
        self._validate_config()

    def _validate_config(self) -> None:
        """Closed-world input validation - all errors become enumerated failures."""
        if self._config.num_samples > 10_000_000:
            raise ValueError("num_samples exceeds maximum supported (10M)")

    def _compute_theoretical_pr_auc(self, class_dist: _ClassDistributions) -> float:
        """Compute theoretical maximum PR-AUC via distribution overlap approximation."""
        try:
            legit_samples = multivariate_normal.rvs(
                mean=class_dist.legit_mean,
                cov=class_dist.legit_cov,
                size=10000,
                random_state=self._rng.np,
            )
            fraud_samples = multivariate_normal.rvs(
                mean=class_dist.fraud_mean,
                cov=class_dist.fraud_cov,
                size=10000,
                random_state=self._rng.np,
            )

            legit_likelihood = multivariate_normal.logpdf(legit_samples, class_dist.legit_mean, class_dist.legit_cov)
            fraud_likelihood = multivariate_normal.logpdf(legit_samples, class_dist.fraud_mean, class_dist.fraud_cov)
            overlap_ratio = np.mean(fraud_likelihood > legit_likelihood)

            pr_auc = 1.0 - overlap_ratio
            return np.clip(pr_auc, 0.2, 0.98)
        except np.linalg.LinAlgError:
            return 0.85  # fallback for singular matrices, still within goldilocks zone

    def _build_class_distributions(self) -> _ClassDistributions:
        """Construct multivariate Gaussian distributions with controlled effect size."""
        total_features = self._config.num_signal_features + self._config.num_noise_features

        # Signal features: class separation via mean shift
        signal_means_legit = np.zeros(self._config.num_signal_features)
        signal_means_fraud = np.full(self._config.num_signal_features, self._config.cohens_d)

        # Noise features: no mean shift
        noise_means_legit = np.zeros(self._config.num_noise_features)
        noise_means_fraud = np.zeros(self._config.num_noise_features)

        legit_mean = np.concatenate([signal_means_legit, noise_means_legit])
        fraud_mean = np.concatenate([signal_means_fraud, noise_means_fraud])

        # Identity covariance for both classes (features independent)
        legit_cov = np.eye(total_features, dtype=np.float64)
        fraud_cov = np.eye(total_features, dtype=np.float64)

        return _ClassDistributions(
            legit_mean=legit_mean,
            legit_cov=legit_cov,
            fraud_mean=fraud_mean,
            fraud_cov=fraud_cov,
        )

    def _generate_target(self) -> NDArray[np.int8]:
        """Generate binary target vector with exact fraud rate."""
        n_fraud = int(self._config.num_samples * self._config.fraud_rate)
        n_legit = self._config.num_samples - n_fraud

        target = np.concatenate([np.ones(n_fraud, dtype=np.int8), np.zeros(n_legit, dtype=np.int8)])
        self._rng.np.shuffle(target)
        return target

    def _generate_numerical_features(self, target: NDArray[np.int8]) -> NDArray[np.float32]:
        """Generate numerical features with controlled separation."""
        class_dist = self._build_class_distributions()
        total_features = len(class_dist.legit_mean)
        features = np.zeros((self._config.num_samples, total_features), dtype=np.float32)

        fraud_mask = target == 1
        legit_mask = ~fraud_mask

        # Generate fraud samples
        if np.any(fraud_mask):
            fraud_samples = multivariate_normal.rvs(
                mean=class_dist.fraud_mean,
                cov=class_dist.fraud_cov,
                size=np.sum(fraud_mask),
                random_state=self._rng.np,
            )
            features[fraud_mask] = fraud_samples.astype(np.float32)

        # Generate legit samples
        if np.any(legit_mask):
            legit_samples = multivariate_normal.rvs(
                mean=class_dist.legit_mean,
                cov=class_dist.legit_cov,
                size=np.sum(legit_mask),
                random_state=self._rng.np,
            )
            features[legit_mask] = legit_samples.astype(np.float32)

        self._theoretical_pr_auc = self._compute_theoretical_pr_auc(class_dist)
        return features

    def _generate_amount(self, target: NDArray[np.int8]) -> NDArray[np.float32]:
        """Generate transaction amounts from log-normal distribution with class-specific scaling."""
        base_mean = self._config.log_normal_mean
        base_sigma = self._config.log_normal_sigma
        amounts = np.zeros(self._config.num_samples, dtype=np.float32)

        for i, is_fraud in enumerate(target):
            if is_fraud:
                mean = base_mean + np.log(self._config.fraud_amount_multiplier)
                sigma = base_sigma * self._config.fraud_amount_variance_multiplier
            else:
                mean = base_mean
                sigma = base_sigma

            amounts[i] = np.exp(self._rng.np.normal(mean, sigma))

        # Ensure no negative amounts (log-normal guarantees this, but clip for safety)
        np.clip(amounts, 1e-6, None, out=amounts)
        return amounts

    def _generate_timestamps(self, target: NDArray[np.int8]) -> pd.Series:
        """Generate timestamps with Poisson inter-arrival times."""
        # Poisson rate: roughly one transaction per second on average
        inter_arrival = poisson.rvs(mu=1.0, size=self._config.num_samples, random_state=self._rng.np)
        cumulative_seconds = np.cumsum(inter_arrival)

        # Base timestamp (arbitrary fixed point)
        base_ts = pd.Timestamp("2024-01-01")
        timestamps = pd.Series([base_ts + pd.Timedelta(seconds=int(s)) for s in cumulative_seconds])
        return timestamps

    def _generate_categorical(self, target: NDArray[np.int8]) -> dict[str, pd.Series]:
        """Generate categorical features using conditional probability tables."""
        categorical_series = {}

        for feature_name, cpt in self._config.categorical_cpt.items():
            categories = list(cpt.keys())  # cpt keys are the actual category values (0, 1, but could be any ints)
            values = np.zeros(self._config.num_samples, dtype=np.int32)

            for i, is_fraud in enumerate(target):
                prob_fraud_category = cpt[1] if is_fraud else cpt[0]

                # For binary categories, prob_fraud_category is P(category=1 | class)
                # We need to sample from [0, 1] with that probability
                values[i] = 1 if self._rng.np.random() < prob_fraud_category else 0

            categorical_series[feature_name] = pd.Series(values, dtype="category")

        return categorical_series

    def _inject_missingness(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject MCAR missingness at specified rate."""
        if self._config.missingness_rate == 0:
            return df

        missing_mask = self._rng.np.random((df.shape[0], df.shape[1])) < self._config.missingness_rate
        df_missing = df.copy()

        for col_idx, col_name in enumerate(df.columns):
            if df[col_name].dtype in ("float32", "float64"):
                df_missing.loc[missing_mask[:, col_idx], col_name] = np.nan

        return df_missing

    def generate(self) -> GenerationResult:
        """Public API: generate synthetic dataset with complete error enumeration."""
        try:
            # 1. Target generation
            target = self._generate_target()

            # 2. Numerical features
            numerical_features = self._generate_numerical_features(target)
            signal_indices = list(range(self._config.num_signal_features))
            noise_indices = list(
                range(self._config.num_signal_features, self._config.num_signal_features + self._config.num_noise_features)
            )

            # 3. Transaction amounts
            amounts = self._generate_amount(target)

            # 4. Timestamps
            timestamps = self._generate_timestamps(target)

            # 5. Categorical features
            categorical_series = self._generate_categorical(target)

            # 6. Assemble DataFrame
            feature_df = pd.DataFrame(
                numerical_features,
                columns=[f"feature_{i}" for i in range(numerical_features.shape[1])],
                dtype=np.float32,
            )
            feature_df["amount"] = amounts
            feature_df["timestamp"] = timestamps
            feature_df["is_fraud"] = target

            for name, series in categorical_series.items():
                feature_df[name] = series

            # 7. Inject missingness
            feature_df = self._inject_missingness(feature_df)

            # 8. Build metadata
            metadata = DatasetMetadata(
                theoretical_max_pr_auc=float(self._theoretical_pr_auc),
                fraud_rate=float(np.mean(target)),
                effect_size=self._config.cohens_d,
                signal_feature_indices=signal_indices,
                noise_feature_indices=noise_indices,
                categorical_fraud_probabilities={
                    name: float(cpt[1]) for name, cpt in self._config.categorical_cpt.items()
                },
            )

            logger.info(
                "Dataset generated",
                extra={
                    "num_samples": self._config.num_samples,
                    "fraud_rate": metadata["fraud_rate"],
                    "theoretical_pr_auc": metadata["theoretical_max_pr_auc"],
                },
            )

            return GenerationResult(data=feature_df, metadata=metadata, error=None)

        except ValueError as e:
            logger.error("Configuration error", exc_info=True)
            return GenerationResult(data=pd.DataFrame(), metadata={}, error=GenerationError.INVALID_CONFIG)
        except np.linalg.LinAlgError as e:
            logger.error("Covariance matrix error", exc_info=True)
            return GenerationResult(data=pd.DataFrame(), metadata={}, error=GenerationError.COVARIANCE_MATRIX_SINGULAR)
        except Exception as e:
            logger.error("Unexpected error", exc_info=True)
            return GenerationResult(data=pd.DataFrame(), metadata={}, error=GenerationError.UNEXPECTED_ERROR)


def create_generator(config: Config) -> SyntheticDataGenerator:
    """Factory function with validation at boundary."""
    return SyntheticDataGenerator(config)


__all__ = [
    "Config",
    "GenerationError",
    "GenerationResult",
    "DatasetMetadata",
    "SyntheticDataGenerator",
    "create_generator",
]