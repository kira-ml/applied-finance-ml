#!/usr/bin/env python
"""
run_training.py - End-to-end training pipeline for fraud detection threshold optimization.
Executes the complete training flow: ingest → validate → preprocess → split → train → threshold → error_analysis.
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent / "src"))

import ingest
import validate
import preprocess
import split as split_module
import train
import threshold
import error_analysis

class TrainingPipelineError(Exception):
    """Base exception for training pipeline failures."""
    pass

def run_training_pipeline(csv_path: str) -> None:
    """Execute the complete training pipeline."""
    start_time = time.time()
    
    try:
        print("=" * 60)
        print("FRAUD DETECTION THRESHOLD OPTIMIZATION - TRAINING PIPELINE")
        print("=" * 60)
        
        # Stage 1: Ingest
        print("\n[1/7] INGESTING DATA...")
        raw_df = ingest.ingest_transactions(csv_path)
        print(f"  ✓ Ingested {len(raw_df)} rows")
        
        # Stage 2: Validate
        print("\n[2/7] VALIDATING DATA...")
        validated_df, needs_log_transform, dropped_columns = validate.validate_dataframe(raw_df)
        print(f"  ✓ Validation complete - dropped {len(dropped_columns)} columns")
        
        # Stage 3: Preprocess (fit=True)
        print("\n[3/7] PREPROCESSING DATA (FITTING)...")
        X, y = preprocess.preprocess_dataframe(
            validated_df, 
            needs_log_transform, 
            dropped_columns,
            fit=True
        )
        print(f"  ✓ Preprocessing complete - X shape: {X.shape}, y shape: {y.shape}")
        
        # Stage 4: Split
        print("\n[4/7] SPLITTING DATA...")
        X_train, X_test, y_train, y_test = split_module.split_data(X, y)
        print(f"  ✓ Split complete - train: {len(X_train)}, test: {len(X_test)}")
        
        # Stage 5: Train
        print("\n[5/7] TRAINING MODELS...")
        winning_model, best_proba, y_test = train.train_models(X_train, X_test, y_train, y_test)
        model_type = "Logistic Regression" if hasattr(winning_model, 'coef_') else "Random Forest"
        print(f"  ✓ Training complete - winner: {model_type}")
        
        # Compute PR-AUC for threshold module
        from sklearn.metrics import average_precision_score
        prauc = average_precision_score(y_test, best_proba)
        
        # Stage 6: Threshold Optimization
        print("\n[6/7] OPTIMIZING THRESHOLD...")
        optimal_threshold = threshold.optimize_threshold(best_proba, y_test, prauc)
        print(f"  ✓ Threshold optimization complete - optimal: {optimal_threshold:.6f}")
        
        # Stage 7: Error Analysis
        print("\n[7/7] ANALYZING ERRORS...")
        error_analysis.analyze_errors(X_test, y_test, best_proba, optimal_threshold, winning_model)
        print(f"  ✓ Error analysis complete")
        
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total execution time: {elapsed:.2f} seconds")
        print("=" * 60)
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ PIPELINE FAILED after {elapsed:.2f} seconds", file=sys.stderr)
        raise TrainingPipelineError(f"Pipeline failed at stage: {str(e)}") from e

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python run_training.py <path_to_transactions.csv>", file=sys.stderr)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        run_training_pipeline(csv_path)
        sys.exit(0)
    except TrainingPipelineError as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()