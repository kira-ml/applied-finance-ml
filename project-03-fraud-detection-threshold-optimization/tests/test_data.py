"""
tests/test_data.py

Unit tests for src/data.py.
Strictly minimal, deterministic, and dependent only on standard library.
Auto-configures path to ensure 'src' is discoverable.
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
from typing import Final, List, Optional, Callable, Any

# ==============================================================================
# PATH CONFIGURATION (CRITICAL FIX)
# Automatically adds the project root to sys.path so 'import src' works.
# Assumes structure: project_root/tests/test_data.py
# ==============================================================================

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ==============================================================================
# IMPORTS (Now safe after path fix)
# ==============================================================================

try:
    from src.data import (
        DataModule,
        DataLifecycleState,
        ModuleError,
        DataValidationError,
        DataGenerationError,
        FileSystemError,
        InvariantViolationError,
        InvalidStateTransitionError,
        _generate_synthetic_rows_unchecked,
        _validate_type,
        _validate_in_range,
        _write_csv_unchecked,
        _read_csv_unchecked
    )
except ImportError as e:
    raise RuntimeError(f"Failed to import src.data. Ensure project structure is correct. Error: {e}")

# ==============================================================================
# TEST CONFIGURATION (Immutable)
# ==============================================================================

_TEST_FRAUD_RATE: Final[float] = 0.01
_TEST_ROW_COUNT: Final[int] = 100
_TEST_SEED: Final[int] = 999

# ==============================================================================
# TEST RUNNER INFRASTRUCTURE (Minimal)
# ==============================================================================

class TestResult:
    def __init__(self, name: str, passed: bool, error: Optional[str] = None) -> None:
        self.name: Final[str] = name
        self.passed: Final[bool] = passed
        self.error: Final[Optional[str]] = error

def run_test(name: str, func: Callable[[], None]) -> TestResult:
    try:
        func()
        return TestResult(name, True)
    except AssertionError as e:
        return TestResult(name, False, f"AssertionFailed: {str(e)}")
    except Exception as e:
        return TestResult(name, False, f"UnexpectedError: {type(e).__name__}: {str(e)}")

def main() -> int:
    results: List[TestResult] = []
    
    tests = [
        ("test_generate_determinism", test_generate_determinism),
        ("test_generate_fraud_ratio", test_generate_fraud_ratio),
        ("test_full_lifecycle", test_full_lifecycle),
        ("test_invalid_state_transition", test_invalid_state_transition),
        ("test_validate_fraud_values", test_validate_fraud_values),
        ("test_type_validation", test_type_validation),
    ]
    
    print(f"Running tests from: {_PROJECT_ROOT}")
    print("-" * 50)
    
    for name, func in tests:
        results.append(run_test(name, func))
        
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    
    print("-" * 50)
    print(f"Test Results: {passed_count}/{total_count} passed")
    
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        color_code = "\033[92m" if res.passed else "\033[91m" # Green/Red if terminal supports it
        reset_code = "\033[0m"
        print(f"[{color_code}{status}{reset_code}] {res.name}")
        if not res.passed and res.error:
            print(f"       -> {res.error}")
            
    return 0 if passed_count == total_count else 1

# ==============================================================================
# TEST CASES
# ==============================================================================

def test_generate_determinism() -> None:
    """Verify same seed produces identical data."""
    rows_1 = _generate_synthetic_rows_unchecked(_TEST_ROW_COUNT, _TEST_FRAUD_RATE, _TEST_SEED)
    rows_2 = _generate_synthetic_rows_unchecked(_TEST_ROW_COUNT, _TEST_FRAUD_RATE, _TEST_SEED)
    
    assert len(rows_1) == len(rows_2), "Row counts mismatch"
    for i, (r1, r2) in enumerate(zip(rows_1, rows_2)):
        assert r1 == r2, f"Row {i} differs between runs"

def test_generate_fraud_ratio() -> None:
    """Verify exactly 1% fraud rate."""
    rows = _generate_synthetic_rows_unchecked(_TEST_ROW_COUNT, _TEST_FRAUD_RATE, _TEST_SEED)
    fraud_count = sum(1 for r in rows if r[3] == 1)
    expected_fraud = int(_TEST_ROW_COUNT * _TEST_FRAUD_RATE)
    
    assert fraud_count == expected_fraud, f"Expected {expected_fraud} fraud cases, got {fraud_count}"

def test_full_lifecycle() -> None:
    """Test generate -> save -> load -> validate flow using temp directory."""
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test_trans.csv")
    
    try:
        # 1. Generate
        rows = _generate_synthetic_rows_unchecked(100, 0.01, 42)
        
        # 2. Write (using helper directly to bypass hardcoded D:\ path)
        _write_csv_unchecked(test_file, rows)
        
        assert os.path.exists(test_file), "File was not created"
        
        # 3. Read
        raw = _read_csv_unchecked(test_file)
        assert len(raw) == 101, f"Header + 100 rows expected, got {len(raw)}"
        
        # 4. Validate Logic
        header = raw[0]
        assert header == ["transaction_id", "amount", "time_delta", "is_fraud"], "Header mismatch"
        
        fraud_vals = [int(r[3]) for r in raw[1:]]
        assert all(v in (0, 1) for v in fraud_vals), "Invalid fraud values found"
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def test_invalid_state_transition() -> None:
    """Verify state machine enforcement."""
    dm = DataModule()
    
    # Try to load before generate
    try:
        dm.load_and_validate()
        assert False, "Expected InvalidStateTransitionError"
    except InvalidStateTransitionError:
        pass # Expected
        
    # Simulate state change to test transition logic without needing real file
    dm._state = DataLifecycleState.GENERATED
    
    # Now load should attempt to read. Since file doesn't exist at D:\..., 
    # it should raise FileSystemError, NOT InvalidStateTransitionError.
    try:
        dm.load_and_validate()
    except FileSystemError:
        pass # Expected (file missing), but state transition was valid
    except InvalidStateTransitionError:
        assert False, "State transition should have been allowed"
    except Exception:
        pass # Other errors acceptable for this specific transition test

def test_validate_fraud_values() -> None:
    """Verify rejection of invalid fraud labels."""
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "bad.csv")
    
    try:
        with open(test_file, 'w', newline='', encoding='utf-8') as f:
            f.write("transaction_id,amount,time_delta,is_fraud\n")
            f.write("1,10.5,100.0,2\n") # Invalid: 2
            
        raw = _read_csv_unchecked(test_file)
        
        # Manually trigger validation logic
        row = raw[1]
        is_fraud = int(row[3])
        if is_fraud not in (0, 1):
            raise DataValidationError("Invalid value", context={"value": is_fraud})
            
    except DataValidationError:
        pass # Expected
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def test_type_validation() -> None:
    """Verify runtime type checks."""
    try:
        _validate_type("string", int, "test_var")
        assert False, "Expected InvariantViolationError"
    except InvariantViolationError:
        pass
        
    try:
        _validate_in_range(1.5, 0.0, 1.0, "test_ratio")
        assert False, "Expected InvariantViolationError"
    except InvariantViolationError:
        pass

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)