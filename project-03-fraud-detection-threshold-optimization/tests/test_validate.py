import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch
import tempfile

# Adjusted import path based on user specification: src/validate.py
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


class TestValidateInput:
    def test_empty_dataframe_raises_error(self):
        df = pd.DataFrame()
        with pytest.raises(DataFrameEmptyError, match="Input DataFrame is empty"):
            _validate_input(df)

    def test_non_dataframe_raises_error(self):
        with pytest.raises(ValidationError, match="Expected pandas DataFrame"):
            _validate_input({"key": "value"})

    def test_valid_dataframe_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        # Should not raise
        _validate_input(df)


class TestDropHighNullColumns:
    def test_drops_columns_above_threshold(self):
        config = ValidationConfig(null_threshold=0.40)
        # 3 nulls out of 5 = 60% > 40%
        data = {
            "keep": [1, 2, 3, 4, 5],
            "drop": [np.nan, np.nan, np.nan, 4, 5],
        }
        df = pd.DataFrame(data)
        
        result_df, dropped = _drop_high_null_columns(df, config)
        
        assert "drop" in dropped
        assert "keep" not in dropped
        assert "drop" not in result_df.columns
        assert "keep" in result_df.columns

    def test_keeps_columns_below_threshold(self):
        config = ValidationConfig(null_threshold=0.40)
        # 1 null out of 5 = 20% < 40%
        data = {
            "keep": [1, np.nan, 3, 4, 5],
        }
        df = pd.DataFrame(data)
        
        result_df, dropped = _drop_high_null_columns(df, config)
        
        assert len(dropped) == 0
        assert list(result_df.columns) == ["keep"]

    def test_no_nulls_returns_original(self):
        config = ValidationConfig()
        df = pd.DataFrame({"a": [1, 2, 3]})
        result_df, dropped = _drop_high_null_columns(df, config)
        assert len(dropped) == 0
        assert result_df.equals(df)


class TestFillNumericNulls:
    def test_fills_with_median(self):
        filled_cols = {}
        df = pd.DataFrame({"num": [1.0, 2.0, np.nan, 4.0]})
        # Median of [1, 2, 4] is 2.0
        
        result_df = _fill_numeric_nulls(df, filled_cols)
        
        assert not result_df["num"].isnull().any()
        assert result_df["num"].iloc[2] == 2.0
        assert "num" in filled_cols
        assert "median" in filled_cols["num"]

    def test_all_null_column_raises_error(self):
        filled_cols = {}
        df = pd.DataFrame({"num": [np.nan, np.nan, np.nan]})
        
        with pytest.raises(NumericFillError, match="Cannot compute median"):
            _fill_numeric_nulls(df, filled_cols)

    def test_skips_columns_without_nulls(self):
        filled_cols = {}
        df = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
        
        result_df = _fill_numeric_nulls(df, filled_cols)
        
        assert len(filled_cols) == 0
        assert result_df["num"].equals(df["num"])


class TestFillCategoricalNulls:
    def test_fills_with_mode(self):
        filled_cols = {}
        df = pd.DataFrame({"cat": ["A", "B", "B", np.nan]}, dtype=object)
        
        result_df = _fill_categorical_nulls(df, filled_cols)
        
        assert not result_df["cat"].isnull().any()
        assert result_df["cat"].iloc[3] == "B"
        assert "cat" in filled_cols

    def test_empty_column_raises_error(self):
        filled_cols = {}
        # Create a column that is entirely null, resulting in empty mode
        df = pd.DataFrame({"cat": [np.nan, np.nan, np.nan]}, dtype=object)
        
        with pytest.raises(CategoricalFillError, match="Cannot compute mode"):
            _fill_categorical_nulls(df, filled_cols)

    def test_skips_columns_without_nulls(self):
        filled_cols = {}
        df = pd.DataFrame({"cat": ["A", "B", "C"]}, dtype=object)
        
        result_df = _fill_categorical_nulls(df, filled_cols)
        
        assert len(filled_cols) == 0
        assert result_df["cat"].equals(df["cat"])


class TestCheckDuplicateTransactions:
    def test_counts_duplicates_correctly(self):
        config = ValidationConfig(transaction_id_names=("TransactionID",))
        df = pd.DataFrame({"TransactionID": [1, 2, 2, 3, 3, 3]})
        
        count = _check_duplicate_transactions(df, config)
        
        # duplicated() marks subsequent occurrences as True.
        # Sequence: F, F, T, F, T, T -> Sum = 3
        assert count == 3

    def test_returns_zero_if_no_duplicates(self):
        config = ValidationConfig()
        df = pd.DataFrame({"TransactionID": [1, 2, 3]})
        assert _check_duplicate_transactions(df, config) == 0

    def test_returns_zero_if_id_column_missing(self):
        config = ValidationConfig(transaction_id_names=("MissingID",))
        df = pd.DataFrame({"OtherID": [1, 2, 2]})
        assert _check_duplicate_transactions(df, config) == 0

    def test_uses_alternative_id_name(self):
        config = ValidationConfig(transaction_id_names=("TransactionID", "transaction_id"))
        df = pd.DataFrame({"transaction_id": [1, 1, 2]})
        assert _check_duplicate_transactions(df, config) == 1


class TestCheckLogTransformFlag:
    def test_returns_true_when_ratio_exceeds_threshold(self):
        config = ValidationConfig(max_mean_ratio_threshold=1000.0)
        # Mean ~ 1009, Max 100000. Ratio > 1000
        df = pd.DataFrame({"TransactionAmt": [10, 10, 10, 10, 10, 10, 10, 10, 10, 100000]})
        
        assert _check_log_transform_flag(df, config) is True

    def test_returns_false_when_ratio_below_threshold(self):
        config = ValidationConfig(max_mean_ratio_threshold=1000.0)
        df = pd.DataFrame({"TransactionAmt": [1, 2, 3, 4, 5]})
        # Mean 3, Max 5. Ratio < 2
        assert _check_log_transform_flag(df, config) is False

    def test_returns_false_if_amount_column_missing(self):
        config = ValidationConfig()
        df = pd.DataFrame({"OtherAmt": [1, 10000]})
        assert _check_log_transform_flag(df, config) is False

    def test_returns_false_if_mean_is_zero(self):
        config = ValidationConfig()
        df = pd.DataFrame({"TransactionAmt": [0, 0, 0]})
        assert _check_log_transform_flag(df, config) is False

    def test_returns_false_if_column_not_numeric(self):
        config = ValidationConfig()
        df = pd.DataFrame({"TransactionAmt": ["1", "100", "1000"]})
        assert _check_log_transform_flag(df, config) is False


class TestFindZeroVarianceColumns:
    def test_identifies_zero_variance(self):
        df = pd.DataFrame({
            "static": [5, 5, 5, 5],
            "dynamic": [1, 2, 3, 4]
        })
        result = _find_zero_variance_columns(df)
        assert result == ["static"]

    def test_returns_empty_list_if_all_vary(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = _find_zero_variance_columns(df)
        assert result == []

    def test_ignores_non_numeric(self):
        df = pd.DataFrame({
            "text": ["A", "A", "A"],
            "num": [1, 2, 3]
        })
        result = _find_zero_variance_columns(df)
        assert result == []


class TestWriteValidationReport:
    def test_writes_report_successfully(self, tmp_path):
        report_path = tmp_path / "report.txt"
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
        nested_path = tmp_path / "sub" / "dir" / "report.txt"
        config = ValidationConfig(output_report_path=str(nested_path))
        
        _write_validation_report(
            config=config,
            dropped_null_cols=[],
            dropped_zero_var_cols=[],
            filled_cols={},
            duplicate_count=0,
            needs_log_transform=False
        )
        
        assert nested_path.exists()

    def test_raises_report_write_error_on_failure(self, tmp_path):
        # Mock open to simulate a permission failure cross-platform
        config = ValidationConfig(output_report_path=str(tmp_path / "report.txt"))

        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
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
    def test_full_pipeline_success(self, tmp_path):
        report_path = tmp_path / "test_report.txt"
        
        # Patch ValidationConfig to use our temporary report path
        with patch.object(ValidationConfig, '__init__', lambda self, **kwargs: 
            object.__setattr__(self, 'null_threshold', kwargs.get('null_threshold', 0.40)) or
            object.__setattr__(self, 'transaction_id_names', kwargs.get('transaction_id_names', ("TransactionID", "transaction_id"))) or
            object.__setattr__(self, 'amount_column_names', kwargs.get('amount_column_names', ("TransactionAmt", "amount"))) or
            object.__setattr__(self, 'max_mean_ratio_threshold', kwargs.get('max_mean_ratio_threshold', 1000.0)) or
            object.__setattr__(self, 'output_report_path', str(report_path))
        ):
            df = pd.DataFrame({
                "TransactionID": [1, 2, 2, 3, 4],
                "TransactionAmt": [10.0, 20.0, 30000.0, 15.0, 12.0],
                "high_null": [np.nan, np.nan, np.nan, 1.0, 1.0], # 60% null
                "zero_var": [5, 5, 5, 5, 5],
                "cat_fill": ["A", "A", np.nan, "A", "A"]
            })

            result_df, needs_log, dropped = validate_dataframe(df)

        # Assertions
        assert "high_null" not in result_df.columns
        assert "zero_var" not in result_df.columns
        assert "cat_fill" in result_df.columns
        assert not result_df["cat_fill"].isnull().any()
        
        assert needs_log is True # Due to high max/mean ratio
        
        assert "high_null" in dropped
        assert "zero_var" in dropped
        
        assert report_path.exists()

    def test_raises_on_empty_input(self):
        df = pd.DataFrame()
        with pytest.raises(DataFrameEmptyError):
            validate_dataframe(df)

    def test_handles_clean_data(self, tmp_path):
        report_path = tmp_path / "clean_report.txt"
        
        with patch.object(ValidationConfig, '__init__', lambda self, **kwargs: 
            object.__setattr__(self, 'null_threshold', 0.40) or
            object.__setattr__(self, 'transaction_id_names', ("TransactionID", "transaction_id")) or
            object.__setattr__(self, 'amount_column_names', ("TransactionAmt", "amount")) or
            object.__setattr__(self, 'max_mean_ratio_threshold', 1000.0) or
            object.__setattr__(self, 'output_report_path', str(report_path))
        ):
            df = pd.DataFrame({
                "a": [1, 2, 3],
                "b": [4.0, 5.0, 6.0]
            })
            result_df, needs_log, dropped = validate_dataframe(df)
        
        assert result_df.equals(df)
        assert needs_log is False
        assert len(dropped) == 0
        assert report_path.exists()