import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
import tempfile

from src.validate import (
    ValidationConfig,
    ValidationError,
    DataFrameEmptyError,
    NumericFillError,
    CategoricalFillError,
    ReportWriteError,
    _validate_input,
    _drop_high_null_columns,
    _fill_numeric_nulls,
    _fill_categorical_nulls,
    _check_duplicate_transactions,
    _check_log_transform_flag,
    _find_zero_variance_columns,
    _write_validation_report,
    validate_dataframe,
)


class TestValidationConfig:
    def test_default_values(self):
        config = ValidationConfig()
        assert config.null_threshold == 0.40
        assert config.transaction_id_names == ("TransactionID", "transaction_id")
        assert config.amount_column_names == ("TransactionAmt", "amount")
        assert config.max_mean_ratio_threshold == 1000.0
        assert config.output_report_path == "artifacts/validation_report.txt"


class TestValidateInput:
    def test_valid_dataframe(self):
        df = pd.DataFrame({"a": [1, 2]})
        # Should not raise
        _validate_input(df)

    def test_non_dataframe_raises_error(self):
        with pytest.raises(ValidationError, match="Expected pandas DataFrame"):
            _validate_input([1, 2, 3])

    def test_empty_dataframe_raises_error(self):
        df = pd.DataFrame()
        with pytest.raises(DataFrameEmptyError, match="Input DataFrame is empty"):
            _validate_input(df)


class TestDropHighNullColumns:
    def test_no_columns_dropped_below_threshold(self):
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, 5, 6]})
        config = ValidationConfig(null_threshold=0.40)
        result_df, dropped = _drop_high_null_columns(df, config)
        
        assert len(dropped) == 0
        assert list(result_df.columns) == ["a", "b"]
        assert result_df.shape == df.shape

    def test_columns_dropped_above_threshold(self):
        # Column 'a' has 2/3 nulls (~66%), 'b' has 1/3 (~33%)
        df = pd.DataFrame({"a": [np.nan, np.nan, 3], "b": [1, np.nan, 3]})
        config = ValidationConfig(null_threshold=0.40)
        result_df, dropped = _drop_high_null_columns(df, config)
        
        assert dropped == ["a"]
        assert list(result_df.columns) == ["b"]
        assert result_df.shape[1] == 1

    def test_exact_threshold_behavior(self):
        # 2 nulls out of 5 = 0.4 exactly. Code uses > threshold, so 0.4 > 0.4 is False.
        df = pd.DataFrame({"a": [np.nan, np.nan, 1, 2, 3]})
        config = ValidationConfig(null_threshold=0.40)
        result_df, dropped = _drop_high_null_columns(df, config)
        
        assert len(dropped) == 0
        assert "a" in result_df.columns


class TestFillNumericNulls:
    def test_fill_with_median(self):
        df = pd.DataFrame({"num": [1.0, 2.0, np.nan, 4.0]})
        filled_cols = {}
        result_df = _fill_numeric_nulls(df, filled_cols)
        
        assert not result_df["num"].isnull().any()
        assert result_df["num"].iloc[2] == 2.0  # Median of 1, 2, 4
        assert "num" in filled_cols
        assert "median" in filled_cols["num"]

    def test_all_null_column_raises_error(self):
        df = pd.DataFrame({"num": [np.nan, np.nan, np.nan]})
        filled_cols = {}
        with pytest.raises(NumericFillError, match="Cannot compute median"):
            _fill_numeric_nulls(df, filled_cols)

    def test_no_nulls_no_change(self):
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
        filled_cols = {}
        result_df = _fill_numeric_nulls(df, filled_cols)
        
        assert filled_cols == {}
        pd.testing.assert_frame_equal(result_df, df)


class TestFillCategoricalNulls:
    def test_fill_with_mode(self):
        df = pd.DataFrame({"cat": ["A", "B", np.nan, "B", "C"]})
        filled_cols = {}
        result_df = _fill_categorical_nulls(df, filled_cols)
        
        assert not result_df["cat"].isnull().any()
        assert result_df["cat"].iloc[2] == "B"  # Mode
        assert "cat" in filled_cols

    def test_empty_column_raises_error(self):
        df = pd.DataFrame({"cat": pd.Series([np.nan, np.nan], dtype=object)})
        filled_cols = {}
        with pytest.raises(CategoricalFillError, match="Cannot compute mode"):
            _fill_categorical_nulls(df, filled_cols)

    def test_mixed_types_ignores_numeric(self):
        df = pd.DataFrame({"num": [1, 2, np.nan]})
        filled_cols = {}
        result_df = _fill_categorical_nulls(df, filled_cols)
        
        assert result_df["num"].isnull().any()
        assert filled_cols == {}


class TestCheckDuplicateTransactions:
    def test_finds_duplicates_default_name(self):
        df = pd.DataFrame({"TransactionID": [1, 2, 2, 3]})
        config = ValidationConfig()
        count = _check_duplicate_transactions(df, config)
        assert count == 1

    def test_finds_duplicates_alternate_name(self):
        df = pd.DataFrame({"transaction_id": ["A", "B", "A"]})
        config = ValidationConfig()
        count = _check_duplicate_transactions(df, config)
        assert count == 1

    def test_no_id_column_returns_zero(self):
        df = pd.DataFrame({"id": [1, 1]})
        config = ValidationConfig()
        count = _check_duplicate_transactions(df, config)
        assert count == 0

    def test_no_duplicates_returns_zero(self):
        df = pd.DataFrame({"TransactionID": [1, 2, 3]})
        config = ValidationConfig()
        count = _check_duplicate_transactions(df, config)
        assert count == 0


class TestCheckLogTransformFlag:
    def test_high_ratio_returns_true(self):
        df = pd.DataFrame({"TransactionAmt": [1, 1, 1, 10000]})
        config = ValidationConfig()
        assert _check_log_transform_flag(df, config) is True

    def test_low_ratio_returns_false(self):
        df = pd.DataFrame({"TransactionAmt": [2, 2, 2, 10]})
        config = ValidationConfig()
        assert _check_log_transform_flag(df, config) is False

    def test_missing_amount_column_returns_false(self):
        df = pd.DataFrame({"OtherCol": [1, 2, 3]})
        config = ValidationConfig()
        assert _check_log_transform_flag(df, config) is False

    def test_non_numeric_amount_returns_false(self):
        df = pd.DataFrame({"TransactionAmt": ["1", "2", "100"]})
        config = ValidationConfig()
        assert _check_log_transform_flag(df, config) is False

    def test_zero_median_returns_false(self):
        df = pd.DataFrame({"TransactionAmt": [0, 0, 0, 100]})
        config = ValidationConfig()
        assert _check_log_transform_flag(df, config) is False


class TestFindZeroVarianceColumns:
    def test_identifies_zero_variance(self):
        df = pd.DataFrame({"const": [5, 5, 5], "var": [1, 2, 3]})
        zero_var = _find_zero_variance_columns(df)
        assert zero_var == ["const"]

    def test_no_zero_variance(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        zero_var = _find_zero_variance_columns(df)
        assert zero_var == []

    def test_ignores_non_numeric(self):
        df = pd.DataFrame({"cat": ["A", "A", "A"], "num": [1, 2, 3]})
        zero_var = _find_zero_variance_columns(df)
        assert zero_var == []


class TestWriteValidationReport:
    def test_writes_report_successfully(self, tmp_path):
        report_path = tmp_path / "subdir" / "report.txt"
        config = ValidationConfig(output_report_path=str(report_path))
        
        _write_validation_report(
            config=config,
            dropped_null_cols=["col_a"],
            dropped_zero_var_cols=["col_b"],
            filled_cols={"col_c": "filled with median 1.0"},
            duplicate_count=5,
            needs_log_transform=True
        )
        
        assert report_path.exists()
        content = report_path.read_text()
        assert "VALIDATION REPORT" in content
        assert "col_a" in content
        assert "col_b" in content
        assert "col_c" in content
        assert "Duplicate transaction count: 5" in content
        assert "Log transform needed: True" in content

    def test_creates_parent_directories(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "report.txt"
        config = ValidationConfig(output_report_path=str(deep_path))
        
        _write_validation_report(
            config=config,
            dropped_null_cols=[],
            dropped_zero_var_cols=[],
            filled_cols={},
            duplicate_count=0,
            needs_log_transform=False
        )
        
        assert deep_path.exists()

    def test_write_failure_raises_error(self):
        config = ValidationConfig(output_report_path="/root/forbidden/report.txt")
        
        with patch("builtins.open", side_effect=PermissionError("Mocked permission error")):
            with pytest.raises(ReportWriteError, match="Failed to write validation report"):
                _write_validation_report(
                    config=config,
                    dropped_null_cols=[],
                    dropped_zero_var_cols=[],
                    filled_cols={},
                    duplicate_count=0,
                    needs_log_transform=False
                )


class TestValidateDataFrame:
    def test_full_pipeline_integration(self, tmp_path):
        data = {
            "TransactionID": [1, 2, 2, 3],
            "TransactionAmt": [1.0, 1.0, 1.0, 10000.0],
            "high_null": [np.nan, np.nan, np.nan, 1.0],
            "zero_var": [5, 5, 5, 5],
            "cat_fill": ["A", np.nan, "A", "B"],
            "num_fill": [1.0, np.nan, 3.0, 4.0]
        }
        df = pd.DataFrame(data)
        
        with patch("src.validate._write_validation_report") as mock_write:
            result_df, needs_log, dropped = validate_dataframe(df)
            
            mock_write.assert_called_once()
            args = mock_write.call_args.args
            
            assert args[4] == 1 # Duplicate count
            assert args[5] is True # Log transform flag
            
            assert "high_null" not in result_df.columns
            assert "zero_var" not in result_df.columns
            assert not result_df["cat_fill"].isnull().any()
            assert not result_df["num_fill"].isnull().any()
            
            assert "high_null" in dropped
            assert "zero_var" in dropped
            assert len(dropped) == 2

    def test_empty_input_raises(self):
        df = pd.DataFrame()
        with pytest.raises(DataFrameEmptyError):
            validate_dataframe(df)

    def test_returns_sorted_dropped_list(self):
        # Ensure the returned list is sorted and unique.
        # We create columns that will be dropped for different reasons to test the combined sorted list.
        data = {
            "z_col": [np.nan] * 10,      # Dropped: High Null
            "a_col": [np.nan] * 10,      # Dropped: High Null
            "m_col": [5] * 10,           # Dropped: Zero Variance (Constant)
            "kept_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Kept: Has variance and non-null
        }
        df = pd.DataFrame(data)
        
        with patch("src.validate._write_validation_report"):
            _, _, dropped = validate_dataframe(df)
        
        # Expected: a_col (null), m_col (var), z_col (null) -> Sorted alphabetically
        expected = ["a_col", "m_col", "z_col"]
        assert dropped == expected