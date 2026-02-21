from pathlib import Path
from typing import Dict, Tuple, Final
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit
import hashlib
import csv
from collections import Counter


class DataLoaderError(Exception):
    """Base exception for data loading errors."""
    pass


class FileNotFoundError(DataLoaderError):
    """Raised when data file does not exist."""
    pass


class InvalidFileFormatError(DataLoaderError):
    """Raised when file format is invalid."""
    pass


class TargetColumnNotFoundError(DataLoaderError):
    """Raised when target column is missing."""
    pass


class DistributionMismatchError(DataLoaderError):
    """Raised when target distribution differs across splits."""
    pass


class StratifiedDataLoader:
    """
    Loads CSV data and performs stratified train/calibration/test split.
    
    Time complexity: O(n) for reading and validation operations.
    Space complexity: O(n) for storing DataFrames.
    """
    
    def __init__(self, file_path: str | Path, target_column: str) -> None:
        """
        Initialize loader with file path and target column.
        
        Args:
            file_path: Path to CSV file
            target_column: Name of target variable column
            
        Raises:
            FileNotFoundError: If file does not exist
            InvalidFileFormatError: If file is not valid CSV
            TargetColumnNotFoundError: If target column missing from CSV
        """
        self._file_path: Final[Path] = Path(file_path)
        self._target_column: Final[str] = target_column
        self._validate_file()
        self._validate_target_column()
        
    def _validate_file(self) -> None:
        """Validate file existence and CSV format."""
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")
            
        if not self._file_path.is_file():
            raise FileNotFoundError(f"Path is not a file: {self._file_path}")
            
        if self._file_path.suffix.lower() != '.csv':
            raise InvalidFileFormatError(f"File must be CSV: {self._file_path}")
            
        try:
            with open(self._file_path, 'r', encoding='utf-8') as f:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
                reader = csv.reader(f, dialect)
                next(reader)  # Check header exists
        except (csv.Error, UnicodeDecodeError, StopIteration) as e:
            raise InvalidFileFormatError(f"Invalid CSV format: {e}")
            
    def _validate_target_column(self) -> None:
        """Validate target column exists in CSV."""
        try:
            df = pd.read_csv(self._file_path, nrows=0)
            if self._target_column not in df.columns:
                raise TargetColumnNotFoundError(
                    f"Target column '{self._target_column}' not found. "
                    f"Available columns: {list(df.columns)}"
                )
        except pd.errors.EmptyDataError:
            raise InvalidFileFormatError("CSV file is empty")
            
    def load_and_split(self) -> Dict[str, DataFrame]:
        """
        Load CSV and create stratified train/calibration/test splits.
        
        Returns:
            Dictionary with keys 'train', 'calibration', 'test' containing DataFrames
            
        Raises:
            DistributionMismatchError: If target distribution differs across splits
            DataLoaderError: For other data loading errors
        """
        raw_data: DataFrame = self._read_csv_with_validation()
        splits: Dict[str, DataFrame] = self._create_stratified_splits(raw_data)
        self._validate_distributions(splits)
        
        return {
            'train': splits['train'].copy(),
            'calibration': splits['calibration'].copy(),
            'test': splits['test'].copy()
        }
        
    def _read_csv_with_validation(self) -> DataFrame:
        """Read CSV with strict validation."""
        try:
            df = pd.read_csv(
                self._file_path,
                encoding='utf-8',
                dtype=str,  # Read all as strings initially to prevent type inference
                keep_default_na=False,  # Prevent automatic NA conversion
                na_values=[]  # No automatic NA handling
            )
            
            # Verify no missing values in target column
            if df[self._target_column].isna().any():
                raise DataLoaderError("Target column contains missing values")
                
            # Convert target column to appropriate type based on content
            try:
                df[self._target_column] = pd.to_numeric(df[self._target_column])
            except (ValueError, TypeError):
                # Keep as string if not numeric
                pass
                
            return df
            
        except pd.errors.ParserError as e:
            raise InvalidFileFormatError(f"CSV parsing error: {e}")
            
    def _create_stratified_splits(self, data: DataFrame) -> Dict[str, DataFrame]:
        """
        Create stratified splits with exact proportions: 60% train, 20% calibration, 20% test.
        """
        # First split: separate train (60%) from temp (40%)
        train_size: float = 0.6
        temp_size: float = 0.4
        
        splitter_1 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_size,
            test_size=temp_size,
            random_state=42  # Fixed seed for deterministic behavior
        )
        
        target: pd.Series = data[self._target_column]
        train_idx, temp_idx = next(splitter_1.split(data, target))
        
        train_data: DataFrame = data.iloc[train_idx].reset_index(drop=True)
        temp_data: DataFrame = data.iloc[temp_idx].reset_index(drop=True)
        
        # Second split: split temp into calibration (50%) and test (50%)
        # This yields final proportions: train 60%, calibration 20%, test 20%
        splitter_2 = StratifiedShuffleSplit(
            n_splits=1,
            train_size=0.5,  # 50% of temp = 20% of total
            test_size=0.5,   # 50% of temp = 20% of total
            random_state=42  # Fixed seed for deterministic behavior
        )
        
        temp_target: pd.Series = temp_data[self._target_column]
        cal_idx, test_idx = next(splitter_2.split(temp_data, temp_target))
        
        calibration_data: DataFrame = temp_data.iloc[cal_idx].reset_index(drop=True)
        test_data: DataFrame = temp_data.iloc[test_idx].reset_index(drop=True)
        
        # Verify exact counts
        total_rows: int = len(data)
        expected_train: int = int(total_rows * 0.6)
        expected_cal: int = int(total_rows * 0.2)
        expected_test: int = total_rows - expected_train - expected_cal
        
        assert len(train_data) == expected_train, "Train split size mismatch"
        assert len(calibration_data) == expected_cal, "Calibration split size mismatch"
        assert len(test_data) == expected_test, "Test split size mismatch"
        
        return {
            'train': train_data,
            'calibration': calibration_data,
            'test': test_data
        }
        
    def _validate_distributions(self, splits: Dict[str, DataFrame]) -> None:
        """
        Validate that target distribution is consistent across all splits.
        
        Raises:
            DistributionMismatchError: If distributions differ significantly
        """
        distributions: Dict[str, Counter] = {}
        
        for split_name, df in splits.items():
            target_series: pd.Series = df[self._target_column]
            
            # Calculate distribution as normalized frequencies
            value_counts = target_series.value_counts(normalize=True)
            distributions[split_name] = Counter(
                {str(k): round(v, 4) for k, v in value_counts.items()}
            )
            
        # Compare all pairs of splits
        split_names: list = list(splits.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                name_i: str = split_names[i]
                name_j: str = split_names[j]
                
                if distributions[name_i] != distributions[name_j]:
                    raise DistributionMismatchError(
                        f"Target distribution mismatch between {name_i} and {name_j}. "
                        f"{name_i}: {dict(distributions[name_i])}, "
                        f"{name_j}: {dict(distributions[name_j])}"
                    )
                    
    def __eq__(self, other: object) -> bool:
        """Equality based on immutable attributes."""
        if not isinstance(other, StratifiedDataLoader):
            return NotImplemented
        return (self._file_path == other._file_path and 
                self._target_column == other._target_column)
                
    def __hash__(self) -> int:
        """Hash based on immutable attributes."""
        return hash((self._file_path, self._target_column))


def load_credit_data() -> Dict[str, DataFrame]:
    """
    Load credit risk data with stratified split.
    
    Returns:
        Dictionary with train, calibration, and test DataFrames
        
    Raises:
        DataLoaderError: For any data loading or validation errors
    """
    data_path: Path = Path("D:/applied-finance-ml/project-02-credit-risk-probability-calibration/data/raw/credit_data.csv")
    target_column: str = "default"  # Assuming target column name from credit_data.csv
    
    loader: StratifiedDataLoader = StratifiedDataLoader(data_path, target_column)
    return loader.load_and_split()