"""Configuration loader with deterministic seed initialization.

This module loads and validates configuration from a YAML file, ensuring all
mandatory keys are present and properly typed. It initializes all random seeds
for deterministic behavior across numpy, random, and sklearn.

Example:
    config = load_config("config.yaml")
    if config.error:
        handle_error(config.error)
    else:
        use_config(config.data)
"""

import yaml
import random
import numpy as np
import sklearn.utils
from pathlib import Path
from typing import NamedTuple, TypedDict, Literal, Any
from enum import Enum, auto
import logging
import sys

# Configure structured logger
logger = logging.getLogger("config_loader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


class ConfigError(Enum):
    """Enumerated failure modes for configuration loading."""

    FILE_NOT_FOUND = auto()
    INVALID_YAML = auto()
    MISSING_MANDATORY_KEY = auto()
    INVALID_KEY_TYPE = auto()
    INVALID_PATH = auto()
    INVALID_SEED = auto()
    INVALID_MODEL_CONFIG = auto()
    INVALID_COST_MATRIX = auto()


class LoadResult(NamedTuple):
    """Result of configuration loading operation."""

    data: "ConfigDict | None"
    error: ConfigError | None
    error_message: str | None


class PathsConfig(TypedDict):
    """Paths configuration schema."""

    data: str
    output: str
    models: str


class ModelConfig(TypedDict):
    """Model configuration schema."""

    name: str
    params: dict[str, Any]


class CostMatrixConfig(TypedDict):
    """Cost matrix configuration schema."""

    false_positive_cost: float
    false_negative_cost: float


class SeedsConfig(TypedDict):
    """Seeds configuration schema."""

    python: int
    numpy: int
    sklearn: int


class ConfigDict(TypedDict):
    """Complete validated configuration dictionary."""

    paths: PathsConfig
    model: ModelConfig
    cost_matrix: CostMatrixConfig
    seeds: SeedsConfig


class _Validator:
    """Internal validation logic."""

    MANDATORY_KEYS = {"paths", "model", "cost_matrix", "seeds"}

    @staticmethod
    def validate_paths(paths: Any) -> PathsConfig | ConfigError:
        """Validate paths configuration."""
        if not isinstance(paths, dict):
            return ConfigError.INVALID_KEY_TYPE

        required = {"data", "output", "models"}
        if not all(key in paths for key in required):
            return ConfigError.MISSING_MANDATORY_KEY

        for key in required:
            if not isinstance(paths[key], str):
                return ConfigError.INVALID_KEY_TYPE

        return PathsConfig(
            data=str(Path(paths["data"])),
            output=str(Path(paths["output"])),
            models=str(Path(paths["models"])),
        )

    @staticmethod
    def validate_model(model: Any) -> ModelConfig | ConfigError:
        """Validate model configuration."""
        if not isinstance(model, dict):
            return ConfigError.INVALID_KEY_TYPE

        if "name" not in model or not isinstance(model["name"], str):
            return ConfigError.MISSING_MANDATORY_KEY

        if "params" not in model or not isinstance(model["params"], dict):
            model["params"] = {}

        return ModelConfig(name=model["name"], params=model["params"])

    @staticmethod
    def validate_cost_matrix(cost_matrix: Any) -> CostMatrixConfig | ConfigError:
        """Validate cost matrix configuration."""
        if not isinstance(cost_matrix, dict):
            return ConfigError.INVALID_KEY_TYPE

        required = {"false_positive_cost", "false_negative_cost"}
        if not all(key in cost_matrix for key in required):
            return ConfigError.MISSING_MANDATORY_KEY

        for key in required:
            if not isinstance(cost_matrix[key], (int, float)):
                return ConfigError.INVALID_KEY_TYPE
            if cost_matrix[key] < 0:
                return ConfigError.INVALID_COST_MATRIX

        return CostMatrixConfig(
            false_positive_cost=float(cost_matrix["false_positive_cost"]),
            false_negative_cost=float(cost_matrix["false_negative_cost"]),
        )

    @staticmethod
    def validate_seeds(seeds: Any) -> SeedsConfig | ConfigError:
        """Validate seeds configuration."""
        if not isinstance(seeds, dict):
            return ConfigError.INVALID_KEY_TYPE

        required = {"python", "numpy", "sklearn"}
        if not all(key in seeds for key in required):
            return ConfigError.MISSING_MANDATORY_KEY

        for key in required:
            if not isinstance(seeds[key], int):
                return ConfigError.INVALID_KEY_TYPE
            if seeds[key] < 0:
                return ConfigError.INVALID_SEED

        return SeedsConfig(
            python=int(seeds["python"]),
            numpy=int(seeds["numpy"]),
            sklearn=int(seeds["sklearn"]),
        )


def _set_seeds(seeds: SeedsConfig) -> None:
    """Set all random seeds for deterministic execution."""
    random.seed(seeds["python"])
    np.random.seed(seeds["numpy"])
    sklearn.utils.check_random_state(seeds["sklearn"])


def load_config(config_path: str | Path) -> LoadResult:
    """Load, validate, and apply configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        LoadResult containing:
            - data: Validated ConfigDict if successful, None otherwise
            - error: ConfigError enum if validation failed, None otherwise
            - error_message: Human-readable error description if failed, None otherwise

    Failure modes:
        FILE_NOT_FOUND: Config file does not exist
        INVALID_YAML: File contains invalid YAML
        MISSING_MANDATORY_KEY: Required top-level key missing
        INVALID_KEY_TYPE: Key value has wrong type
        INVALID_PATH: Path string is invalid
        INVALID_SEED: Seed value is invalid
        INVALID_MODEL_CONFIG: Model config is malformed
        INVALID_COST_MATRIX: Cost matrix contains negative values

    Example:
        result = load_config("config.yaml")
        if result.error:
            print(f"Failed: {result.error_message}")
        else:
            config = result.data
    """
    try:
        # Read and parse YAML
        path = Path(config_path)
        if not path.exists():
            logger.error("Config file not found", extra={"path": str(path)})
            return LoadResult(
                data=None,
                error=ConfigError.FILE_NOT_FOUND,
                error_message=f"Config file not found: {path}",
            )

        with open(path, "r") as f:
            try:
                raw_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error("Invalid YAML", extra={"error": str(e)})
                return LoadResult(
                    data=None,
                    error=ConfigError.INVALID_YAML,
                    error_message=f"Invalid YAML: {str(e)}",
                )

        if not isinstance(raw_config, dict):
            logger.error("Config root must be a dictionary")
            return LoadResult(
                data=None,
                error=ConfigError.INVALID_YAML,
                error_message="Config root must be a dictionary",
            )

        # Validate mandatory top-level keys
        missing = _Validator.MANDATORY_KEYS - set(raw_config.keys())
        if missing:
            logger.error("Missing mandatory keys", extra={"missing": list(missing)})
            return LoadResult(
                data=None,
                error=ConfigError.MISSING_MANDATORY_KEY,
                error_message=f"Missing mandatory keys: {missing}",
            )

        # Validate each section
        paths_result = _Validator.validate_paths(raw_config["paths"])
        if isinstance(paths_result, ConfigError):
            return LoadResult(data=None, error=paths_result, error_message="Invalid paths configuration")

        model_result = _Validator.validate_model(raw_config["model"])
        if isinstance(model_result, ConfigError):
            return LoadResult(data=None, error=model_result, error_message="Invalid model configuration")

        cost_matrix_result = _Validator.validate_cost_matrix(raw_config["cost_matrix"])
        if isinstance(cost_matrix_result, ConfigError):
            return LoadResult(data=None, error=cost_matrix_result, error_message="Invalid cost matrix configuration")

        seeds_result = _Validator.validate_seeds(raw_config["seeds"])
        if isinstance(seeds_result, ConfigError):
            return LoadResult(data=None, error=seeds_result, error_message="Invalid seeds configuration")

        # Build final config
        config = ConfigDict(
            paths=paths_result,
            model=model_result,
            cost_matrix=cost_matrix_result,
            seeds=seeds_result,
        )

        # Set seeds for deterministic execution
        _set_seeds(seeds_result)

        logger.info(
            "Configuration loaded successfully",
            extra={
                "model": model_result["name"],
                "seed_python": seeds_result["python"],
                "seed_numpy": seeds_result["numpy"],
                "seed_sklearn": seeds_result["sklearn"],
            },
        )

        return LoadResult(data=config, error=None, error_message=None)

    except Exception as e:
        logger.error("Unexpected error loading config", exc_info=True)
        return LoadResult(
            data=None,
            error=ConfigError.INVALID_YAML,
            error_message=f"Unexpected error: {str(e)}",
        )


__all__ = [
    "ConfigError",
    "LoadResult",
    "ConfigDict",
    "PathsConfig",
    "ModelConfig",
    "CostMatrixConfig",
    "SeedsConfig",
    "load_config",
]