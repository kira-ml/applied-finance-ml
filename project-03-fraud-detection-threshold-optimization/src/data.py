"""
src/data.py

Synthetic transaction data generation and loading module.
Generates 10,000 rows with 1% fraud prevalence, saves to CSV, and loads into DataFrame.
Strictly deterministic, type-enforced, and side-effect isolated.
"""

from __future__ import annotations

import csv
import os
import random
from enum import Enum
from typing import Final, List, Optional, Tuple

# ==============================================================================
# 1. TOTAL STATIC TYPE PRECISION & RUNTIME ENFORCEMENT
# 2. SINGLE EXPLICIT ERROR TAXONOMY
# 4. SINGLE EXPLICIT ERROR TAXONOMY (Leaf Exceptions)
# 20. TOTAL LINEARITY OF ERROR HANDLING
# ==============================================================================

class ModuleError(Exception):
    """Base exception for all module-specific errors."""
    def __init__(self, message: str, context: Optional[dict] = None) -> None:
        super().__init__(message)
        self.context: Final[Optional[dict]] = context if context is not None else {}

class InvariantViolationError(ModuleError):
    """Raised when a defined invariant is violated."""
    pass

class InvalidStateTransitionError(ModuleError):
    """Raised when an illegal state transition occurs."""
    pass

class DataGenerationError(ModuleError):
    """Raised when synthetic data generation fails."""
    pass

class DataValidationError(ModuleError):
    """Raised when loaded data fails validation constraints."""
    pass

class FileSystemError(ModuleError):
    """Raised when file system operations fail."""
    pass

# ==============================================================================
# 8. IMMUTABLE CONFIGURATION AT LOAD TIME
# 14. NO MUTABLE GLOBAL STATE
# ==============================================================================

_CONFIG_FRAUD_RATE: Final[float] = 0.01
_CONFIG_ROW_COUNT: Final[int] = 10000
_CONFIG_SEED: Final[int] = 42

# Updated path as per user requirement
_CONFIG_DIR_PATH: Final[str] = r"D:\applied-finance-ml\project-03-fraud-detection-threshold-optimization\data\raw"
_CONFIG_FILE_NAME: Final[str] = "transactions.csv"
_CONFIG_FILE_PATH: Final[str] = os.path.join(_CONFIG_DIR_PATH, _CONFIG_FILE_NAME)

# Valid states for the data lifecycle
class DataLifecycleState(Enum):
    UNINITIALIZED = 0
    GENERATED = 1
    LOADED = 2

# ==============================================================================
# 3. COMPREHENSIVE INVARIANT ENFORCEMENT
# 6. NO DYNAMIC TYPE INTROSPECTION FOR CONTROL FLOW
# 9. EXPLICIT DATA LIFECYCLE AND MUTATION CONTROL
# ==============================================================================

def _validate_type(value: object, expected_type: type, var_name: str) -> None:
    """Runtime type validator using explicit literal guards where applicable."""
    if type(value) is not expected_type:
        raise InvariantViolationError(
            f"Type mismatch for '{var_name}': expected {expected_type.__name__}, got {type(value).__name__}",
            context={"value": repr(value), "expected": expected_type.__name__}
        )

def _validate_in_range(value: float, min_val: float, max_val: float, var_name: str) -> None:
    if not (min_val <= value <= max_val):
        raise InvariantViolationError(
            f"Value out of range for '{var_name}': {value} not in [{min_val}, {max_val}]",
            context={"value": value, "min": min_val, "max": max_val}
        )

# ==============================================================================
# 2. PURE DETERMINISTIC CORE WITH EXPLICIT IMPURITY MARKERS
# 13. STRICT RESOURCE MANAGEMENT
# ==============================================================================

def _generate_synthetic_rows_unchecked(row_count: int, fraud_rate: float, seed: int) -> List[Tuple[int, float, float, int]]:
    """
    Generates synthetic transaction data.
    SIDE EFFECT: Initializes global random state (isolated).
    Returns list of tuples: (transaction_id, amount, time_delta, is_fraud).
    """
    local_rng = random.Random(seed)
    
    data_Mutable: List[Tuple[int, float, float, int]] = []
    
    fraud_count_target = int(row_count * fraud_rate)
    
    all_indices = list(range(row_count))
    local_rng.shuffle(all_indices)
    fraud_indices = set(all_indices[:fraud_count_target])
    
    for i in range(row_count):
        tx_id = i
        amount = local_rng.uniform(10.0, 1000.0)
        time_delta = local_rng.uniform(0.0, 3600.0)
        is_fraud = 1 if i in fraud_indices else 0
        
        data_Mutable.append((tx_id, amount, time_delta, is_fraud))
        
    return data_Mutable

def _write_csv_unchecked(file_path: str, rows: List[Tuple[int, float, float, int]]) -> None:
    """
    Writes rows to CSV file.
    SIDE EFFECT: Filesystem write.
    """
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except OSError as e:
            raise FileSystemError(f"Failed to create directory {dir_path}", context={"path": dir_path}) from e

    try:
        with open(file_path, mode='w', newline='', encoding='utf-8') as file_Mutable:
            writer = csv.writer(file_Mutable)
            writer.writerow(["transaction_id", "amount", "time_delta", "is_fraud"])
            for row in rows:
                writer.writerow(row)
    except IOError as e:
        raise FileSystemError(f"Failed to write file {file_path}", context={"path": file_path}) from e

def _read_csv_unchecked(file_path: str) -> List[List[str]]:
    """
    Reads CSV file into list of string lists.
    SIDE EFFECT: Filesystem read.
    """
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as file_Mutable:
            reader = csv.reader(file_Mutable)
            return [row for row in reader]
    except IOError as e:
        raise FileSystemError(f"Failed to read file {file_path}", context={"path": file_path}) from e

# ==============================================================================
# MAIN LOGIC IMPLEMENTATION
# ==============================================================================

class DataModule:
    """
    Handles generation, saving, and loading of synthetic transaction data.
    Enforces strict lifecycle states.
    """
    
    def __init__(self) -> None:
        self._state: DataLifecycleState = DataLifecycleState.UNINITIALIZED
        self._data_frame: Optional[List[dict]] = None
        
    def generate_and_save(self) -> None:
        """
        Generates 10,000 synthetic rows with 1% fraud and saves to CSV.
        Transitions state: UNINITIALIZED -> GENERATED.
        """
        if self._state != DataLifecycleState.UNINITIALIZED:
            raise InvalidStateTransitionError(
                "Cannot generate: invalid state transition",
                context={"current_state": self._state.name, "target": "GENERATED"}
            )
            
        _validate_type(_CONFIG_ROW_COUNT, int, "_CONFIG_ROW_COUNT")
        _validate_in_range(_CONFIG_FRAUD_RATE, 0.0, 1.0, "_CONFIG_FRAUD_RATE")
        
        try:
            rows = _generate_synthetic_rows_unchecked(_CONFIG_ROW_COUNT, _CONFIG_FRAUD_RATE, _CONFIG_SEED)
            _write_csv_unchecked(_CONFIG_FILE_PATH, rows)
            self._state = DataLifecycleState.GENERATED
        except ModuleError:
            raise
        except Exception as e:
            raise DataGenerationError("Unexpected error during generation", context={"error": str(e)}) from e

    def load_and_validate(self) -> List[dict]:
        """
        Loads data from CSV into a DataFrame-like structure (list of dicts).
        Verifies is_fraud column contains only 0 and 1.
        Transitions state: GENERATED/LOADED -> LOADED.
        """
        if self._state == DataLifecycleState.UNINITIALIZED:
            raise InvalidStateTransitionError(
                "Cannot load data: data not generated yet",
                context={"current_state": self._state.name}
            )
            
        try:
            raw_data = _read_csv_unchecked(_CONFIG_FILE_PATH)
        except ModuleError:
            raise
        except Exception as e:
            raise FileSystemError("Unexpected error reading CSV", context={"error": str(e)}) from e

        if len(raw_data) < 2:
            raise DataValidationError("CSV file is empty or missing header", context={"rows_found": len(raw_data)})

        header = raw_data[0]
        expected_header = ["transaction_id", "amount", "time_delta", "is_fraud"]
        
        if header != expected_header:
            raise DataValidationError(
                "CSV header mismatch",
                context={"expected": expected_header, "found": header}
            )

        parsed_data_Mutable: List[dict] = []
        
        for idx, row in enumerate(raw_data[1:], start=1):
            if len(row) != 4:
                raise DataValidationError(
                    f"Row {idx} has incorrect column count",
                    context={"row_index": idx, "columns_found": len(row)}
                )
            
            try:
                tx_id = int(row[0])
                amount = float(row[1])
                time_delta = float(row[2])
                is_fraud = int(row[3])
            except ValueError as e:
                raise DataValidationError(
                    f"Type conversion failed at row {idx}",
                    context={"row_index": idx, "error": str(e)}
                ) from e

            if is_fraud not in (0, 1):
                raise DataValidationError(
                    f"Invalid is_fraud value at row {idx}",
                    context={"row_index": idx, "value": is_fraud}
                )

            record = {
                "transaction_id": tx_id,
                "amount": amount,
                "time_delta": time_delta,
                "is_fraud": is_fraud
            }
            parsed_data_Mutable.append(record)

        self._data_frame = parsed_data_Mutable
        self._state = DataLifecycleState.LOADED
        
        return [dict(r) for r in parsed_data_Mutable]

    def get_state(self) -> DataLifecycleState:
        """Returns current lifecycle state."""
        return self._state

# ==============================================================================
# 19. STRICT PUBLIC INTERFACE CONTROL
# ==============================================================================

__all__ = ["DataModule", "ModuleError", "DataValidationError", "DataGenerationError", "FileSystemError", "InvariantViolationError", "InvalidStateTransitionError"]