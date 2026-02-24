"""
src/run_pipeline.py

Entry point for the credit risk probability calibration pipeline.
Orchestrates: Load → Split → Preprocess → Train → Calibrate → Evaluate.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

from config import load_config
from data_loader import load_and_split_data
from preprocessing import fit_and_transform_pipeline
from modeling import train_and_calibrate, get_raw_predictions
from evaluate import evaluate_and_save


def main() -> int:
    """
    Executes the end-to-end pipeline.
    
    Returns:
        0 if successful, 1 if failed.
    """
    print("=" * 80)
    print("Credit Risk Probability Calibration Pipeline")
    print("=" * 80)
    
    try:
        # Step 1: Load configuration
        print("\n[1/6] Loading configuration...")
        config = load_config()
        print(f"  ✓ Project root: {config.project_root}")
        print(f"  ✓ Target column: {config.target_column}")
        print(f"  ✓ Calibration method: {config.calibration_method}")
        
        # Step 2: Load and split data
        print("\n[2/6] Loading and splitting data...")
        train_df, cal_df, test_df = load_and_split_data(
            file_path=str(config.path_raw_data),
            target_col=config.target_column
        )
        print(f"  ✓ Train: {len(train_df)} samples")
        print(f"  ✓ Calibration: {len(cal_df)} samples")
        print(f"  ✓ Test: {len(test_df)} samples")
        
        # Separate features and targets
        X_train = train_df.drop(columns=[config.target_column])
        y_train = train_df[config.target_column]
        
        X_cal = cal_df.drop(columns=[config.target_column])
        y_cal = cal_df[config.target_column]
        
        X_test = test_df.drop(columns=[config.target_column])
        y_test = test_df[config.target_column]
        
        # Step 3: Preprocess data
        print("\n[3/6] Preprocessing features...")
        X_train_processed, X_cal_processed, X_test_processed, pipeline_path = fit_and_transform_pipeline(
            X_train=X_train,
            X_cal=X_cal,
            X_test=X_test,
            output_dir=config.artifacts_dir
        )
        print(f"  ✓ Transformed shape: {X_train_processed.shape}")
        print(f"  ✓ Pipeline saved: {pipeline_path}")
        
        # Step 4: Train and calibrate models
        print("\n[4/6] Training and calibrating models...")
        base_model, calibrated_model = train_and_calibrate(
            X_train=X_train_processed,
            y_train=y_train,
            X_cal=X_cal_processed,
            y_cal=y_cal,
            base_model_path=config.path_base_model,
            calibrated_model_path=config.path_calibrated_model,
            n_estimators=config.model_params.n_estimators,
            max_depth=config.model_params.max_depth,
            learning_rate=config.model_params.learning_rate,
            min_samples_split=config.model_params.min_samples_split,
            min_samples_leaf=config.model_params.min_samples_leaf,
            subsample=config.model_params.subsample,
            calibration_method=config.calibration_method,
            random_state=config.model_params.random_state
        )
        print(f"  ✓ Base model saved: {config.path_base_model}")
        print(f"  ✓ Calibrated model saved: {config.path_calibrated_model}")
        
        # Step 5: Generate predictions
        print("\n[5/6] Generating predictions...")
        y_proba_base = get_raw_predictions(base_model, X_test_processed)
        y_proba_calibrated = get_raw_predictions(calibrated_model, X_test_processed)
        print(f"  ✓ Base predictions generated")
        print(f"  ✓ Calibrated predictions generated")
        
        # Step 6: Evaluate and save metrics
        print("\n[6/6] Evaluating performance...")
        min_improvement = float(config.thresholds.min_brier_improvement_pct) * 100  # Convert to percentage
        max_cal_error = float(config.thresholds.max_calibration_error)
        
        metrics = evaluate_and_save(
            y_test=y_test,
            y_proba_base=y_proba_base,
            y_proba_calibrated=y_proba_calibrated,
            output_path=config.path_metrics_report,
            min_improvement_threshold=min_improvement,
            max_calibration_error_threshold=max_cal_error
        )
        
        print(f"  ✓ Metrics saved: {config.path_metrics_report}")
        
        # Display results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Brier Score (Base):       {metrics['brier_score_base']:.6f}")
        print(f"Brier Score (Calibrated): {metrics['brier_score_calibrated']:.6f}")
        print(f"Improvement:              {metrics['brier_improvement_pct']:.2f}%")
        print(f"Calibration Error (Base): {metrics['calibration_error_base']:.6f}")
        print(f"Calibration Error (Cal):  {metrics['calibration_error_calibrated']:.6f}")
        print("-" * 80)
        print(f"Improvement Threshold:    ≥{min_improvement:.1f}% {'✓' if metrics['meets_improvement_threshold'] else '✗'}")
        print(f"Calibration Threshold:    ≤{max_cal_error:.2f} {'✓' if metrics['meets_calibration_threshold'] else '✗'}")
        print("-" * 80)
        
        if metrics['overall_success']:
            print("STATUS: ✓ PASS - All success criteria met")
            print("=" * 80)
            return 0
        else:
            print("STATUS: ✗ FAIL - Success criteria not met")
            print("=" * 80)
            return 1
            
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: File not found - {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: Pipeline failed - {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
