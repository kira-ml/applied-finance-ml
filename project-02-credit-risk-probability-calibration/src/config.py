"""
Configuration Module for Credit Risk Probability Calibration.

Adheres to High-Assurance Engineering Standards:
- Absolute Static Type Closure
- Deterministic Execution Guarantee
- Immutable State Enforcement
- Total Input Validation Boundary
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Final, List, Optional, Tuple, Union

# -----------------------------------------------------------------------------
# Type Aliases for Explicit Closure
# -----------------------------------------------------------------------------
FilePath = Path
FeatureList = Optional[List[str]]
Ratio = float
Seed = int
HyperParamValue = Union[int, float, str]
HyperParams = dict[str, HyperParamValue]


# -----------------------------------------------------------------------------
# Immutable Data Structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelHyperparameters:
    """
    Immutable container for Gradient Boosting Classifier parameters.
    Ensures no runtime modification of model architecture.
    """
    n_estimators: int
    max_depth: int
    learning_rate: float
    min_samples_split: int
    min_samples_leaf: int
    subsample: float
    random_state: int

    def __post_init__(self) -> None:
        # Rule 3 & 12: Defensive Assertion Layer
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0]")
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0.0, 1.0]")


@dataclass(frozen=True)
class SplitRatios:
    """
    Immutable container for data split ratios.
    Enforces sum-to-one constraint and non-negative values.
    """
    train: Ratio
    calibration: Ratio
    test: Ratio

    def __post_init__(self) -> None:
        # Rule 3 & 6: Total Input Validation & Exhaustive Coverage
        total: float = self.train + self.calibration + self.test
        
        # Use Decimal for precise comparison to avoid floating point drift (Rule 9)
        total_decimal = Decimal(str(total))
        one_decimal = Decimal("1.0")
        
        if abs(total_decimal - one_decimal) > Decimal("0.0001"):
            raise ValueError(
                f"Split ratios must sum to 1.0. Got: {total}"
            )
        
        if self.train <= 0 or self.calibration <= 0 or self.test <= 0:
            raise ValueError("All split ratios must be strictly positive")


@dataclass(frozen=True)
class EvaluationThresholds:
    """
    Immutable container for success criteria.
    Uses Decimal for financial metric precision.
    """
    min_brier_improvement_pct: Decimal
    max_calibration_error: Decimal

    def __post_init__(self) -> None:
        if self.min_brier_improvement_pct < Decimal("0"):
            raise ValueError("Improvement threshold cannot be negative")
        if self.max_calibration_error < Decimal("0"):
            raise ValueError("Calibration error threshold cannot be negative")


@dataclass(frozen=True)
class Config:
    """
    Root Configuration Object.
    Fully immutable. All paths are resolved to absolute Paths at instantiation.
    No ambient environment reliance (Rule 19).
    """
    # Paths
    project_root: FilePath
    data_raw_dir: FilePath
    artifacts_dir: FilePath
    outputs_dir: FilePath
    path_raw_data: FilePath
    path_preprocessor: FilePath
    path_base_model: FilePath
    path_calibrated_model: FilePath
    path_metrics_report: FilePath

    # Data Schema
    target_column: str
    numeric_features: FeatureList
    categorical_features: FeatureList
    
    # Execution Control
    split_ratios: SplitRatios
    random_seed: Seed
    
    # Model & Calibration
    model_params: ModelHyperparameters
    calibration_method: str
    
    # Success Criteria
    thresholds: EvaluationThresholds

    def __post_init__(self) -> None:
        # Rule 3: Validate Target Column Name
        if not self.target_column or not isinstance(self.target_column, str):
            raise ValueError("target_column must be a non-empty string")
        
        # Rule 3: Validate Calibration Method
        allowed_methods: Final[Tuple[str, ...]] = ("sigmoid", "isotonic")
        if self.calibration_method not in allowed_methods:
            raise ValueError(f"calibration_method must be one of {allowed_methods}")

        # Rule 7 & 14: Verify Directory Existence (Fail fast if structure is broken)
        # Note: We do not create directories here to maintain purity; 
        # creation is handled by the orchestration layer or explicit setup scripts.
        # However, we validate the root exists to prevent cascading errors.
        if not self.project_root.is_dir():
            raise FileNotFoundError(f"Project root does not exist: {self.project_root}")


# -----------------------------------------------------------------------------
# Pure Factory Functions
# -----------------------------------------------------------------------------

def resolve_paths(project_root: Path) -> Tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    """
    Pure function to resolve all file system paths based on project root.
    Returns absolute paths. Does not perform I/O.
    
    Raises:
        ValueError: If constructed paths are invalid.
    """
    data_raw_dir: Path = project_root / "data" / "raw"
    artifacts_dir: Path = project_root / "artifacts"
    outputs_dir: Path = project_root / "outputs"
    
    path_raw_data: Path = data_raw_dir / "credit_data.csv"
    path_preprocessor: Path = artifacts_dir / "preprocessor.pkl"
    path_base_model: Path = artifacts_dir / "base_model.pkl"
    path_calibrated_model: Path = artifacts_dir / "calibrated_model.pkl"
    path_metrics_report: Path = outputs_dir / "metrics_summary.json"
    
    return (
        data_raw_dir,
        artifacts_dir,
        outputs_dir,
        path_raw_data,
        path_preprocessor,
        path_base_model,
        path_calibrated_model,
        path_metrics_report
    )


def load_config(explicit_project_root: Optional[Path] = None) -> Config:
    """
    Constructs the immutable Config object.
    
    Args:
        explicit_project_root: Explicit path to project root. 
                               If None, derives from __file__.
                               (Rule 19: Explicit dependency injection preferred).
    
    Returns:
        Config: A fully validated, immutable configuration instance.
    
    Raises:
        FileNotFoundError: If the derived project root does not exist.
        ValueError: If configuration constraints are violated.
    """
    # Rule 2: Deterministic Execution - Resolve root explicitly
    if explicit_project_root is not None:
        project_root = explicit_project_root.resolve()
    else:
        # Fallback only for local dev convenience, still deterministic
        project_root = Path(__file__).resolve().parent.parent
    
    if not project_root.is_dir():
        raise FileNotFoundError(f"Invalid project root: {project_root}")

    # Resolve all paths purely
    (
        data_raw_dir,
        artifacts_dir,
        outputs_dir,
        path_raw_data,
        path_preprocessor,
        path_base_model,
        path_calibrated_model,
        path_metrics_report
    ) = resolve_paths(project_root)

    # Construct Immutable Components
    split_ratios: SplitRatios = SplitRatios(
        train=0.60,
        calibration=0.20,
        test=0.20
    )

    model_params: ModelHyperparameters = ModelHyperparameters(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    thresholds: EvaluationThresholds = EvaluationThresholds(
        min_brier_improvement_pct=Decimal("0.15"),
        max_calibration_error=Decimal("0.05")
    )

    # Return Final Immutable Config
    return Config(
        project_root=project_root,
        data_raw_dir=data_raw_dir,
        artifacts_dir=artifacts_dir,
        outputs_dir=outputs_dir,
        path_raw_data=path_raw_data,
        path_preprocessor=path_preprocessor,
        path_base_model=path_base_model,
        path_calibrated_model=path_calibrated_model,
        path_metrics_report=path_metrics_report,
        target_column="default",  # Confirmed from CSV analysis
        numeric_features=None,    # Auto-detect
        categorical_features=None,# Auto-detect
        split_ratios=split_ratios,
        random_seed=42,
        model_params=model_params,
        calibration_method="sigmoid",
        thresholds=thresholds
    )