"""
Main orchestration pipeline for time-series revenue forecasting.

Purpose:
Coordinate the end-to-end ML pipeline:
1. Load and validate data
2. Create features
3. Run time-series CV with stability checks
4. Train final model
5. Persist artifacts

This module intentionally avoids:
- Modeling math implementation
- Feature engineering logic
- Inference logic
- Duplicating module responsibilities
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import sys
from datetime import datetime

# Import project modules
import config
from data import load_and_validate_timeseries
from features import create_features
from split import time_series_split
from train import train_ridge_model
from evaluate import compute_metrics, compute_fold_variance_percent


def run_pipeline(
    data_path: str,
    target_column: str = config.TARGET_COL,
    lags: Optional[list] = None,
    rolling_windows: Optional[list] = None,
    include_time_features: bool = True,
    n_splits: int = config.N_SPLITS,
    valid_size: int = config.VALIDATION_SIZE,
    gap: int = 0,
    expanding: bool = True,
    ridge_alpha: float = config.RIDGE_ALPHA,
    stability_threshold: float = config.MAX_FOLD_RMSE_VARIANCE_PCT,
    random_state: int = config.RANDOM_STATE,
    output_dir: str = str(config.ARTIFACTS_DIR)
) -> Dict[str, Any]:
    """
    Run complete time-series forecasting pipeline.

    Parameters
    ----------
    data_path : str
        Path to input CSV file.
    target_column : str
        Name of target column in CSV.
    lags : list, optional
        Lag periods for feature engineering.
    rolling_windows : list, optional
        Rolling window sizes for feature engineering.
    include_time_features : bool
        Whether to include time-based features.
    n_splits : int
        Number of CV splits.
    valid_size : int
        Validation set size for each split.
    gap : int
        Gap between train and validation sets.
    expanding : bool
        Whether to use expanding (True) or sliding (False) windows.
    ridge_alpha : float
        Regularization strength for Ridge.
    stability_threshold : float
        Maximum allowed RMSE variance percentage across folds.
    random_state : int
        Random seed for reproducibility.
    output_dir : str
        Directory to save artifacts.

    Returns
    -------
    Dict[str, Any]
        Pipeline results including:
        - cv_metrics: List of metrics per fold
        - final_rmse_variance: Stability metric
        - stable: Whether stability criteria met
        - artifact_paths: Paths to saved artifacts
        - feature_names: List of features used

    Raises
    ------
    ValueError
        If pipeline fails at any stage or stability criteria not met.
    """
    # Default feature config from config.py
    if lags is None:
        lags = config.LAG_LIST
    if rolling_windows is None:
        rolling_windows = config.ROLLING_WINDOWS

    np.random.seed(random_state)

    print("=" * 60)
    print("TIME-SERIES FORECASTING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and validate data
    print("\n[1/6] Loading and validating data...")
    try:
        df = load_and_validate_timeseries(
            filepath=data_path,
            value_column=target_column,
            expected_frequency="D"  # Assume daily data
        )
        print(f"✓ Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
    except Exception as e:
        raise ValueError(f"Data loading failed: {e}")
    
    # Step 2: Create features
    print("\n[2/6] Creating features...")
    try:
        X, y, feature_names = create_features(
            df=df,
            target_column="value",
            lags=lags,
            rolling_windows=rolling_windows,
            include_time_features=include_time_features,
            drop_na=True
        )
        print(f"✓ Created {X.shape[1]} features: {feature_names}")
        print(f"✓ Generated {X.shape[0]} samples")
    except Exception as e:
        raise ValueError(f"Feature creation failed: {e}")
    
    # Step 3: Time-series cross-validation
    print(f"\n[3/6] Running {n_splits}-fold time-series CV...")
    cv_metrics = []
    fold_rmse_values = []
    
    try:
        splits = list(time_series_split(
            n_samples=len(X),
            n_splits=n_splits,
            valid_size=valid_size,
            gap=gap,
            expanding=expanding
        ))
    except Exception as e:
        raise ValueError(f"Split generation failed: {e}")
    
    for fold_idx, (train_idx, valid_idx) in enumerate(splits):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}:")
        print(f"    Train: {train_idx[0]}-{train_idx[-1]} ({len(train_idx)} samples)")
        print(f"    Valid: {valid_idx[0]}-{valid_idx[-1]} ({len(valid_idx)} samples)")
        
        # Split data
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        # Train model
        try:
            scaler, model = train_ridge_model(
                X_train=X_train.values,
                y_train=y_train.values,
                alpha=ridge_alpha,
                random_state=random_state
            )
        except Exception as e:
            raise ValueError(f"Training failed on fold {fold_idx + 1}: {e}")
        
        # Predict on validation set
        X_valid_scaled = scaler.transform(X_valid.values)
        y_pred = model.predict(X_valid_scaled)
        
        # Evaluate
        try:
            fold_metrics = compute_metrics(
                y_true=y_valid.values,
                y_pred=y_pred,
                fold_index=fold_idx,
                fold_rmse_history=np.array(fold_rmse_values) if fold_rmse_values else None,
                time_indices=valid_idx
            )
            cv_metrics.append(fold_metrics)
            fold_rmse_values.append(fold_metrics["rmse"])
            
            print(f"    RMSE: {fold_metrics['rmse']:.4f}")
            if fold_metrics["residual_time_correlation"] is not None:
                print(f"    Resid-Time Corr: {fold_metrics['residual_time_correlation']:.4f}")
            
        except Exception as e:
            raise ValueError(f"Evaluation failed on fold {fold_idx + 1}: {e}")
    
    # Step 4: Check stability criteria
    print("\n[4/6] Checking stability criteria...")
    final_rmse_variance = compute_fold_variance_percent(np.array(fold_rmse_values))
    
    print(f"  RMSE variance across folds: {final_rmse_variance:.2f}%")
    print(f"  Stability threshold: {stability_threshold:.2f}%")
    
    is_stable = final_rmse_variance <= stability_threshold
    if not is_stable:
        print("\n" + "=" * 60)
        print("DIAGNOSTIC INFORMATION")
        print("=" * 60)
        print("\nThe model failed the stability threshold. Possible causes:")
        print("  1. Non-stationary data (trend/seasonality too strong)")
        print("  2. Structural break in the data")
        print("  3. Insufficient training data in early folds")
        print("\nSuggestions:")
        print("  • Generate stable data:  python src/synthetic_data.py --mode stable")
        print("  • Increase threshold:    --stability-threshold 12.0")
        print("  • Verify data quality:   check for structural breaks in revenue")
        print(f"\nExample with adjusted threshold:")
        print(f"  python src/main.py --data-path {data_path} --stability-threshold 12.0")
        print("=" * 60)
        raise ValueError(
            f"Model unstable: RMSE variance {final_rmse_variance:.2f}% "
            f"exceeds threshold {stability_threshold:.2f}%"
        )
    print("  ✓ Stability criteria met")
    
    # Step 5: Train final model on all data
    print("\n[5/6] Training final model on all data...")
    try:
        final_scaler, final_model = train_ridge_model(
            X_train=X.values,
            y_train=y.values,
            alpha=ridge_alpha,
            random_state=random_state
        )
        print("  ✓ Final model trained successfully")
    except Exception as e:
        raise ValueError(f"Final training failed: {e}")
    
    # Step 6: Persist artifacts
    print("\n[6/6] Persisting artifacts...")
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for artifact versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model and scaler
        model_path = output_path / f"ridge_model_{timestamp}.joblib"
        scaler_path = output_path / f"scaler_{timestamp}.joblib"
        
        joblib.dump(final_model, model_path)
        joblib.dump(final_scaler, scaler_path)
        
        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "n_samples": len(df),
            "date_range": [str(df.index.min()), str(df.index.max())],
            "features": feature_names,
            "n_features": len(feature_names),
            "cv_folds": n_splits,
            "cv_rmse_mean": float(np.mean(fold_rmse_values)),
            "cv_rmse_std": float(np.std(fold_rmse_values)),
            "cv_rmse_variance_percent": float(final_rmse_variance),
            "ridge_alpha": ridge_alpha,
            "lags": lags,
            "rolling_windows": rolling_windows,
            "include_time_features": include_time_features,
            "stable": is_stable,
            "random_state": random_state
        }
        
        metadata_path = output_path / f"metadata_{timestamp}.json"
        pd.Series(metadata).to_json(metadata_path)
        
        print(f"  ✓ Model saved to: {model_path}")
        print(f"  ✓ Scaler saved to: {scaler_path}")
        print(f"  ✓ Metadata saved to: {metadata_path}")
        
        artifact_paths = {
            "model": str(model_path),
            "scaler": str(scaler_path),
            "metadata": str(metadata_path)
        }
        
    except Exception as e:
        raise ValueError(f"Artifact persistence failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nFinal Model Performance:")
    print(f"  Mean CV RMSE: {np.mean(fold_rmse_values):.4f}")
    print(f"  CV RMSE Std: {np.std(fold_rmse_values):.4f}")
    print(f"  CV RMSE Variance: {final_rmse_variance:.2f}%")
    print(f"\nArtifacts saved to: {output_dir}/")
    
    return {
        "cv_metrics": cv_metrics,
        "final_rmse_variance": final_rmse_variance,
        "stable": is_stable,
        "artifact_paths": artifact_paths,
        "feature_names": feature_names
    }


def main():
    """Command-line entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Time-series revenue forecasting pipeline"
    )
    
    # Required arguments
    parser.add_argument(
        "--data-path", 
        type=str, 
        required=True,
        help="Path to input CSV file"
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        "--target-column",
        type=str,
        default=config.TARGET_COL,
        help=f"Name of target column (default: {config.TARGET_COL})"
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=config.N_SPLITS,
        help=f"Number of CV splits (default: {config.N_SPLITS})"
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=config.VALIDATION_SIZE,
        help=f"Validation set size in days (default: {config.VALIDATION_SIZE})"
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=config.RIDGE_ALPHA,
        help=f"Ridge regularization strength (default: {config.RIDGE_ALPHA})"
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=config.MAX_FOLD_RMSE_VARIANCE_PCT,
        help=f"Max allowed RMSE variance %% (default: {config.MAX_FOLD_RMSE_VARIANCE_PCT})"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=config.RANDOM_STATE,
        help=f"Random seed (default: {config.RANDOM_STATE})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.ARTIFACTS_DIR),
        help=f"Output directory for artifacts (default: {config.ARTIFACTS_DIR})"
    )
    parser.add_argument(
        "--no-time-features",
        action="store_true",
        help="Disable time-based features"
    )
    
    args = parser.parse_args()
    
    # Default feature config from config.py
    lags = config.LAG_LIST
    rolling_windows = config.ROLLING_WINDOWS
    
    try:
        results = run_pipeline(
            data_path=args.data_path,
            target_column=args.target_column,
            lags=lags,
            rolling_windows=rolling_windows,
            include_time_features=not args.no_time_features,
            n_splits=args.n_splits,
            valid_size=args.valid_size,
            ridge_alpha=args.ridge_alpha,
            stability_threshold=args.stability_threshold,
            random_state=args.random_state,
            output_dir=args.output_dir
        )
        
        # Exit with success
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        if "Model unstable" not in str(e):
            # Non-stability errors get a short hint
            print("\nTip: Run 'python src/synthetic_data.py --mode stable' to generate a stable dataset.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()