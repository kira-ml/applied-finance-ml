import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging
from typing import Any

# Ensure project root is on the path so 'src' is resolvable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Disable logging during tests
logging.disable(sys.maxsize)

# Import module under test
from src.synthetic_data_generator import (
    Config,
    GenerationError,
    GenerationResult,
    DatasetMetadata,
    SyntheticDataGenerator,
    create_generator,
)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation rules."""

    def test_valid_config(self) -> None:
        """Should accept valid configuration parameters."""
        config = Config(
            num_samples=10000,
            num_signal_features=5,
            num_noise_features=3,
            fraud_rate=0.01,
            cohens_d=0.8,
            log_normal_mean=4.0,
            log_normal_sigma=0.5,
            fraud_amount_multiplier=2.0,
            fraud_amount_variance_multiplier=1.5,
            categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
            missingness_rate=0.005,
            random_seed=42,
        )
        self.assertIsInstance(config, Config)

    def test_invalid_num_samples(self) -> None:
        """Should reject non-positive num_samples."""
        with self.assertRaises(ValueError):
            Config(
                num_samples=0,
                num_signal_features=5,
                num_noise_features=3,
                fraud_rate=0.01,
                cohens_d=0.8,
                log_normal_mean=4.0,
                log_normal_sigma=0.5,
                fraud_amount_multiplier=2.0,
                fraud_amount_variance_multiplier=1.5,
                categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
                missingness_rate=0.005,
                random_seed=42,
            )

    def test_invalid_fraud_rate(self) -> None:
        """Should reject fraud_rate outside (0,1)."""
        with self.assertRaises(ValueError):
            Config(
                num_samples=10000,
                num_signal_features=5,
                num_noise_features=3,
                fraud_rate=1.5,
                cohens_d=0.8,
                log_normal_mean=4.0,
                log_normal_sigma=0.5,
                fraud_amount_multiplier=2.0,
                fraud_amount_variance_multiplier=1.5,
                categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
                missingness_rate=0.005,
                random_seed=42,
            )

    def test_invalid_cpt_keys(self) -> None:
        """Should reject CPT missing class 0 or 1."""
        with self.assertRaises(ValueError):
            Config(
                num_samples=10000,
                num_signal_features=5,
                num_noise_features=3,
                fraud_rate=0.01,
                cohens_d=0.8,
                log_normal_mean=4.0,
                log_normal_sigma=0.5,
                fraud_amount_multiplier=2.0,
                fraud_amount_variance_multiplier=1.5,
                categorical_cpt={"merchant": {0: 0.1}},  # missing class 1
                missingness_rate=0.005,
                random_seed=42,
            )

    def test_invalid_missingness_rate(self) -> None:
        """Should reject missingness_rate >= 0.1."""
        with self.assertRaises(ValueError):
            Config(
                num_samples=10000,
                num_signal_features=5,
                num_noise_features=3,
                fraud_rate=0.01,
                cohens_d=0.8,
                log_normal_mean=4.0,
                log_normal_sigma=0.5,
                fraud_amount_multiplier=2.0,
                fraud_amount_variance_multiplier=1.5,
                categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
                missingness_rate=0.2,
                random_seed=42,
            )


class TestTargetGeneration(unittest.TestCase):
    """Test binary target generation."""

    def setUp(self) -> None:
        self.config = Config(
            num_samples=10000,
            num_signal_features=5,
            num_noise_features=3,
            fraud_rate=0.01,
            cohens_d=0.8,
            log_normal_mean=4.0,
            log_normal_sigma=0.5,
            fraud_amount_multiplier=2.0,
            fraud_amount_variance_multiplier=1.5,
            categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
            missingness_rate=0.005,
            random_seed=42,
        )
        self.generator = SyntheticDataGenerator(self.config)

    def test_exact_fraud_rate(self) -> None:
        """Should generate exactly the specified fraud rate."""
        target = self.generator._generate_target()
        fraud_rate = np.mean(target)
        expected_fraud = int(self.config.num_samples * self.config.fraud_rate)
        actual_fraud = np.sum(target)

        self.assertEqual(actual_fraud, expected_fraud)
        self.assertAlmostEqual(fraud_rate, self.config.fraud_rate, places=4)

    def test_binary_values(self) -> None:
        """Target should contain only 0 and 1."""
        target = self.generator._generate_target()
        unique_values = np.unique(target)
        np.testing.assert_array_equal(unique_values, [0, 1])

    def test_deterministic_reproducibility(self) -> None:
        """Same seed should produce identical target."""
        generator1 = SyntheticDataGenerator(self.config)
        generator2 = SyntheticDataGenerator(self.config)

        target1 = generator1._generate_target()
        target2 = generator2._generate_target()

        np.testing.assert_array_equal(target1, target2)


class TestNumericalFeatures(unittest.TestCase):
    """Test numerical feature generation."""

    def setUp(self) -> None:
        self.config = Config(
            num_samples=10000,
            num_signal_features=5,
            num_noise_features=3,
            fraud_rate=0.01,
            cohens_d=0.8,
            log_normal_mean=4.0,
            log_normal_sigma=0.5,
            fraud_amount_multiplier=2.0,
            fraud_amount_variance_multiplier=1.5,
            categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
            missingness_rate=0.005,
            random_seed=42,
        )
        self.generator = SyntheticDataGenerator(self.config)
        self.target = self.generator._generate_target()

    def test_feature_dimensions(self) -> None:
        """Should generate correct number of features."""
        features = self.generator._generate_numerical_features(self.target)
        expected_features = self.config.num_signal_features + self.config.num_noise_features
        self.assertEqual(features.shape[1], expected_features)

    def test_signal_noise_separation(self) -> None:
        """Signal features should have mean shift, noise features should not."""
        features = self.generator._generate_numerical_features(self.target)

        fraud_mask = self.target == 1
        legit_mask = ~fraud_mask

        # Signal features (first num_signal_features) should show separation
        for i in range(self.config.num_signal_features):
            fraud_mean = np.mean(features[fraud_mask, i])
            legit_mean = np.mean(features[legit_mask, i])
            self.assertGreater(fraud_mean, legit_mean)

        # Noise features should have similar means
        for i in range(self.config.num_signal_features, features.shape[1]):
            fraud_mean = np.mean(features[fraud_mask, i])
            legit_mean = np.mean(features[legit_mask, i])
            self.assertAlmostEqual(fraud_mean, legit_mean, delta=0.2)

    def test_float32_dtype(self) -> None:
        """Features should be float32."""
        features = self.generator._generate_numerical_features(self.target)
        self.assertEqual(features.dtype, np.float32)


class TestAmountGeneration(unittest.TestCase):
    """Test transaction amount generation."""

    def setUp(self) -> None:
        self.config = Config(
            num_samples=10000,
            num_signal_features=5,
            num_noise_features=3,
            fraud_rate=0.01,
            cohens_d=0.8,
            log_normal_mean=4.0,
            log_normal_sigma=0.5,
            fraud_amount_multiplier=2.0,
            fraud_amount_variance_multiplier=1.5,
            categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
            missingness_rate=0.005,
            random_seed=42,
        )
        self.generator = SyntheticDataGenerator(self.config)
        self.target = self.generator._generate_target()

    def test_amounts_positive(self) -> None:
        """All amounts should be positive."""
        amounts = self.generator._generate_amount(self.target)
        self.assertTrue(np.all(amounts > 0))

    def test_fraud_amounts_higher(self) -> None:
        """Fraudulent transactions should have higher mean amount."""
        amounts = self.generator._generate_amount(self.target)

        fraud_mask = self.target == 1
        legit_mask = ~fraud_mask

        fraud_mean = np.mean(amounts[fraud_mask])
        legit_mean = np.mean(amounts[legit_mask])

        self.assertGreater(fraud_mean, legit_mean)

    def test_amounts_float32(self) -> None:
        """Amounts should be float32."""
        amounts = self.generator._generate_amount(self.target)
        self.assertEqual(amounts.dtype, np.float32)


class TestCategoricalGeneration(unittest.TestCase):
    """Test categorical feature generation."""

    def setUp(self) -> None:
        self.config = Config(
            num_samples=50000,  # Increased sample size for better approximation
            num_signal_features=5,
            num_noise_features=3,
            fraud_rate=0.01,
            cohens_d=0.8,
            log_normal_mean=4.0,
            log_normal_sigma=0.5,
            fraud_amount_multiplier=2.0,
            fraud_amount_variance_multiplier=1.5,
            categorical_cpt={
                "merchant": {0: 0.01, 1: 0.5},  # Increased separation for better test
                "device": {0: 0.02, 1: 0.3},
            },
            missingness_rate=0.005,
            random_seed=42,
        )
        self.generator = SyntheticDataGenerator(self.config)
        self.target = self.generator._generate_target()

    def test_categorical_features_present(self) -> None:
        """Should generate all specified categorical features."""
        categorical = self.generator._generate_categorical(self.target)
        self.assertEqual(set(categorical.keys()), {"merchant", "device"})

    def test_conditional_probabilities(self) -> None:
        """Fraud rate per category should approximate CPT."""
        categorical = self.generator._generate_categorical(self.target)
        df = pd.DataFrame({name: cat for name, cat in categorical.items()})
        df["is_fraud"] = self.target

        # Check merchant category fraud rate approximation
        merchant_fraud_rate = df[df["merchant"] == 1]["is_fraud"].mean()
        # Should be close to P(fraud|merchant=1) which is derived from CPT
        # P(fraud|merchant=1) = P(merchant=1|fraud) * P(fraud) / P(merchant=1)
        # With our CPT: P(merchant=1|fraud)=0.5, P(fraud)=0.01, P(merchant=1) ≈ 0.5*0.01 + 0.01*0.99 ≈ 0.0149
        # So expected ≈ 0.5 * 0.01 / 0.0149 ≈ 0.335
        self.assertGreater(merchant_fraud_rate, 0.2)
        self.assertLess(merchant_fraud_rate, 0.5)

        # Check that fraud rate for category 1 is higher than category 0
        merchant_legit_rate = df[df["merchant"] == 0]["is_fraud"].mean()
        self.assertGreater(merchant_fraud_rate, merchant_legit_rate)

    def test_category_dtype(self) -> None:
        """Categorical series should have category dtype."""
        categorical = self.generator._generate_categorical(self.target)
        for series in categorical.values():
            self.assertEqual(series.dtype.name, "category")


class TestMissingnessInjection(unittest.TestCase):
    """Test MCAR missingness injection."""

    def setUp(self) -> None:
        self.config = Config(
            num_samples=10000,
            num_signal_features=5,
            num_noise_features=3,
            fraud_rate=0.01,
            cohens_d=0.8,
            log_normal_mean=4.0,
            log_normal_sigma=0.5,
            fraud_amount_multiplier=2.0,
            fraud_amount_variance_multiplier=1.5,
            categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
            missingness_rate=0.005,
            random_seed=42,
        )
        self.generator = SyntheticDataGenerator(self.config)
        self.target = self.generator._generate_target()
        self.features = self.generator._generate_numerical_features(self.target)
        self.amounts = self.generator._generate_amount(self.target)

    def test_missingness_rate(self) -> None:
        """Missingness should approximate specified rate."""
        # Create minimal DataFrame
        df = pd.DataFrame(self.features, columns=[f"f{i}" for i in range(self.features.shape[1])])
        df["amount"] = self.amounts

        df_missing = self.generator._inject_missingness(df)
        missing_ratio = df_missing.isna().sum().sum() / (df_missing.shape[0] * df_missing.shape[1])

        self.assertAlmostEqual(missing_ratio, self.config.missingness_rate, delta=0.002)

    def test_no_missingness_when_zero(self) -> None:
        """Should not inject missingness when rate is 0."""
        config_zero = Config(
            num_samples=10000,
            num_signal_features=5,
            num_noise_features=3,
            fraud_rate=0.01,
            cohens_d=0.8,
            log_normal_mean=4.0,
            log_normal_sigma=0.5,
            fraud_amount_multiplier=2.0,
            fraud_amount_variance_multiplier=1.5,
            categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
            missingness_rate=0.0,
            random_seed=42,
        )
        generator = SyntheticDataGenerator(config_zero)

        df = pd.DataFrame({"test": [1.0, 2.0, 3.0]})
        df_missing = generator._inject_missingness(df)

        self.assertEqual(df_missing.isna().sum().sum(), 0)


class TestEndToEndGeneration(unittest.TestCase):
    """Test complete generation pipeline."""

    def setUp(self) -> None:
        self.config = Config(
            num_samples=5000,
            num_signal_features=5,
            num_noise_features=3,
            fraud_rate=0.01,
            cohens_d=0.8,
            log_normal_mean=4.0,
            log_normal_sigma=0.5,
            fraud_amount_multiplier=2.0,
            fraud_amount_variance_multiplier=1.5,
            categorical_cpt={
                "merchant": {0: 0.1, 1: 0.3},
                "device": {0: 0.05, 1: 0.15},
            },
            missingness_rate=0.005,
            random_seed=42,
        )

    def test_successful_generation(self) -> None:
        """Should generate complete dataset without errors."""
        generator = SyntheticDataGenerator(self.config)
        result = generator.generate()

        self.assertIsNone(result.error)
        self.assertIsInstance(result.data, pd.DataFrame)
        self.assertIsInstance(result.metadata, dict)

    def test_metadata_content(self) -> None:
        """Metadata should contain all required fields."""
        generator = SyntheticDataGenerator(self.config)
        result = generator.generate()
        metadata = result.metadata

        required_fields = [
            "theoretical_max_pr_auc",
            "fraud_rate",
            "effect_size",
            "signal_feature_indices",
            "noise_feature_indices",
            "categorical_fraud_probabilities",
        ]

        for field in required_fields:
            self.assertIn(field, metadata)

    def test_data_columns(self) -> None:
        """DataFrame should contain all expected columns."""
        generator = SyntheticDataGenerator(self.config)
        result = generator.generate()
        df = result.data

        expected_columns = set()
        expected_columns.update([f"feature_{i}" for i in range(8)])  # 5 signal + 3 noise
        expected_columns.update(["amount", "timestamp", "is_fraud", "merchant", "device"])

        self.assertEqual(set(df.columns), expected_columns)

    def test_fraud_rate_in_data(self) -> None:
        """Actual fraud rate should match config."""
        generator = SyntheticDataGenerator(self.config)
        result = generator.generate()

        actual_fraud_rate = result.data["is_fraud"].mean()
        self.assertAlmostEqual(actual_fraud_rate, self.config.fraud_rate, places=2)

    def test_deterministic_output(self) -> None:
        """Same config should produce identical DataFrame."""
        generator1 = SyntheticDataGenerator(self.config)
        generator2 = SyntheticDataGenerator(self.config)

        result1 = generator1.generate()
        result2 = generator2.generate()

        pd.testing.assert_frame_equal(result1.data, result2.data)

    def test_error_handling_invalid_config(self) -> None:
        """Should return error enum for invalid config."""
        with self.assertRaises(ValueError):  # Config validation raises ValueError
            Config(
                num_samples=-100,  # Invalid
                num_signal_features=5,
                num_noise_features=3,
                fraud_rate=0.01,
                cohens_d=0.8,
                log_normal_mean=4.0,
                log_normal_sigma=0.5,
                fraud_amount_multiplier=2.0,
                fraud_amount_variance_multiplier=1.5,
                categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
                missingness_rate=0.005,
                random_seed=42,
            )


class TestCreateGenerator(unittest.TestCase):
    """Test factory function."""

    def test_create_generator(self) -> None:
        """Factory should create valid generator instance."""
        config = Config(
            num_samples=1000,
            num_signal_features=2,
            num_noise_features=1,
            fraud_rate=0.01,
            cohens_d=0.8,
            log_normal_mean=4.0,
            log_normal_sigma=0.5,
            fraud_amount_multiplier=2.0,
            fraud_amount_variance_multiplier=1.5,
            categorical_cpt={"merchant": {0: 0.1, 1: 0.3}},
            missingness_rate=0.005,
            random_seed=42,
        )

        generator = create_generator(config)
        self.assertIsInstance(generator, SyntheticDataGenerator)


if __name__ == "__main__":
    unittest.main()