import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open

from src.ingest import (
    IngestConfig,
    IngestError,
    FileNotFoundError,
    EmptyFileError,
    RowCountError,
    MissingColumnsError,
    FraudColumnError,
    FraudRateError,
    _validate_file_path,
    _validate_row_count,
    _validate_columns,
    _validate_fraud_column,
    _validate_fraud_rate,
    ingest_transactions,
)


class TestIngestConfig:
    def test_default_required_columns(self):
        config = IngestConfig()
        expected_cols = [
            "TransactionAmt", "TransactionDT", "hour_of_day", "day_of_week",
            "card_type", "merchant_category", "V1", "is_fraud"
        ]
        assert config.required_columns == expected_cols

    def test_custom_required_columns(self):
        custom_cols = ["col_a", "col_b"]
        config = IngestConfig(required_columns=custom_cols)
        assert config.required_columns == custom_cols

    def test_default_limits(self):
        config = IngestConfig()
        assert config.min_rows == 50000
        assert config.max_rows == 600000
        assert config.min_fraud_rate == 0.005
        assert config.max_fraud_rate == 0.05


class TestValidateFilePath:
    def test_valid_file(self, tmp_path):
        file_path = tmp_path / "data.csv"
        file_path.write_text("a,b\n1,2")
        
        result = _validate_file_path(str(file_path))
        assert isinstance(result, Path)
        assert result.exists()

    def test_missing_file_raises_error(self, tmp_path):
        non_existent = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError, match="File not found"):
            _validate_file_path(str(non_existent))

    def test_empty_file_raises_error(self, tmp_path):
        file_path = tmp_path / "empty.csv"
        file_path.write_text("")
        
        with pytest.raises(EmptyFileError, match="File is empty"):
            _validate_file_path(str(file_path))


class TestValidateRowCount:
    def test_within_range(self):
        df = pd.DataFrame({"a": range(50000)})
        config = IngestConfig(min_rows=50000, max_rows=600000)
        # Should not raise
        _validate_row_count(df, config)

    def test_below_min_raises_error(self):
        df = pd.DataFrame({"a": range(49999)})
        config = IngestConfig(min_rows=50000, max_rows=600000)
        with pytest.raises(RowCountError, match="outside acceptable range"):
            _validate_row_count(df, config)

    def test_above_max_raises_error(self):
        df = pd.DataFrame({"a": range(600001)})
        config = IngestConfig(min_rows=50000, max_rows=600000)
        with pytest.raises(RowCountError, match="outside acceptable range"):
            _validate_row_count(df, config)


class TestValidateColumns:
    def test_all_columns_present(self):
        df = pd.DataFrame(columns=["TransactionAmt", "is_fraud"])
        config = IngestConfig(required_columns=["TransactionAmt", "is_fraud"])
        # Should not raise
        _validate_columns(df, config)

    def test_missing_columns_raises_error(self):
        df = pd.DataFrame(columns=["TransactionAmt"])
        config = IngestConfig(required_columns=["TransactionAmt", "is_fraud"])
        with pytest.raises(MissingColumnsError, match="Missing required columns"):
            _validate_columns(df, config)


class TestValidateFraudColumn:
    def test_valid_fraud_column(self):
        df = pd.DataFrame({"is_fraud": [0, 1, 0, 1]})
        # Should not raise
        _validate_fraud_column(df)

    def test_missing_fraud_column_raises_error(self):
        df = pd.DataFrame({"other_col": [1, 2]})
        with pytest.raises(FraudColumnError, match="is_fraud column not found"):
            _validate_fraud_column(df)

    def test_non_integer_dtype_raises_error(self):
        df = pd.DataFrame({"is_fraud": [0.0, 1.0]})
        with pytest.raises(FraudColumnError, match="must be integer dtype"):
            _validate_fraud_column(df)

    def test_invalid_values_raises_error(self):
        df = pd.DataFrame({"is_fraud": [0, 1, 2]})
        with pytest.raises(FraudColumnError, match="must contain only 0 and 1"):
            _validate_fraud_column(df)
        
        df_neg = pd.DataFrame({"is_fraud": [0, -1]})
        with pytest.raises(FraudColumnError, match="must contain only 0 and 1"):
            _validate_fraud_column(df_neg)


class TestValidateFraudRate:
    def test_within_range(self):
        # 1% fraud rate
        df = pd.DataFrame({"is_fraud": [0] * 99 + [1]})
        config = IngestConfig(min_fraud_rate=0.005, max_fraud_rate=0.05)
        rate = _validate_fraud_rate(df, config)
        assert 0.009 <= rate <= 0.011

    def test_below_min_raises_error(self):
        # 0.1% fraud rate
        df = pd.DataFrame({"is_fraud": [0] * 999 + [1]})
        config = IngestConfig(min_fraud_rate=0.005, max_fraud_rate=0.05)
        with pytest.raises(FraudRateError, match="outside acceptable range"):
            _validate_fraud_rate(df, config)

    def test_above_max_raises_error(self):
        # 10% fraud rate
        df = pd.DataFrame({"is_fraud": [0] * 90 + [1] * 10})
        config = IngestConfig(min_fraud_rate=0.005, max_fraud_rate=0.05)
        with pytest.raises(FraudRateError, match="outside acceptable range"):
            _validate_fraud_rate(df, config)


class TestIngestTransactions:
    def _create_valid_mock_df(self):
        """Helper to create a valid dataframe for ingestion tests."""
        data = {
            "TransactionAmt": [100.0] * 50000,
            "TransactionDT": [1] * 50000,
            "hour_of_day": [12] * 50000,
            "day_of_week": [1] * 50000,
            "card_type": ["visa"] * 50000,
            "merchant_category": ["retail"] * 50000,
            "V1": [0.5] * 50000,
            "is_fraud": [0] * 49000 + [1] * 1000  # ~2% fraud rate
        }
        return pd.DataFrame(data)

    @patch("src.ingest.pd.read_csv")
    @patch("src.ingest.Path.exists", return_value=True)
    @patch("src.ingest.Path.stat")
    def test_successful_ingestion(self, mock_stat, mock_exists, mock_read_csv):
        mock_stat.return_value.st_size = 1024
        mock_read_csv.return_value = self._create_valid_mock_df()
        
        df = ingest_transactions("dummy_path.csv")
        
        assert len(df) == 50000
        assert "is_fraud" in df.columns
        mock_read_csv.assert_called_once()

    @patch("src.ingest.pd.read_csv")
    @patch("src.ingest.Path.exists", return_value=True)
    @patch("src.ingest.Path.stat")
    def test_csv_read_failure_raises_ingest_error(self, mock_stat, mock_exists, mock_read_csv):
        mock_stat.return_value.st_size = 1024
        mock_read_csv.side_effect = Exception("Parse error")
        
        with pytest.raises(IngestError, match="Failed to read CSV"):
            ingest_transactions("dummy_path.csv")

    @patch("src.ingest.pd.read_csv")
    @patch("src.ingest.Path.exists", return_value=True)
    @patch("src.ingest.Path.stat")
    def test_empty_dataframe_after_read_raises_error(self, mock_stat, mock_exists, mock_read_csv):
        mock_stat.return_value.st_size = 1024
        mock_read_csv.return_value = pd.DataFrame()
        
        with pytest.raises(EmptyFileError, match="CSV file contains no data"):
            ingest_transactions("dummy_path.csv")

    @patch("src.ingest.pd.read_csv")
    @patch("src.ingest.Path.exists", return_value=True)
    @patch("src.ingest.Path.stat")
    def test_validation_failure_propagates(self, mock_stat, mock_exists, mock_read_csv):
        mock_stat.return_value.st_size = 1024
        
        # FIX: Create a DataFrame with 50,000 rows to pass row count validation,
        # but with wrong column names to trigger MissingColumnsError.
        bad_df = pd.DataFrame({
            "wrong_col": [1] * 50000,
            "another_wrong": [2] * 50000
        })
        
        mock_read_csv.return_value = bad_df
        
        with pytest.raises(MissingColumnsError):
            ingest_transactions("dummy_path.csv")