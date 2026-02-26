#!/usr/bin/env python
"""
run_inference.py - Run inference on new transactions using trained artifacts.
Loads model, preprocessor, scaler (if exists), and threshold to score new data.
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

import infer

class InferencePipelineError(Exception):
    """Base exception for inference pipeline failures."""
    pass

def run_inference_pipeline(csv_path: str) -> None:
    """Execute inference on new transactions."""
    try:
        print("=" * 60)
        print("FRAUD DETECTION THRESHOLD OPTIMIZATION - INFERENCE PIPELINE")
        print("=" * 60)
        
        print(f"\nInput file: {csv_path}")
        infer.run_inference(csv_path)
        
        print("\n" + "=" * 60)
        print("INFERENCE PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        raise InferencePipelineError(f"Inference failed: {str(e)}") from e

def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on new transactions")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/inference/new_transactions.csv",
        help="Path to input CSV file (default: data/inference/new_transactions.csv)"
    )
    
    args = parser.parse_args()
    
    try:
        run_inference_pipeline(args.input)
        sys.exit(0)
    except InferencePipelineError as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()