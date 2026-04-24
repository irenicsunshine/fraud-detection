# =============================================================================
# 🎓 TESTS FOR DATA PREPROCESSING
# =============================================================================
# These tests verify that our preprocessing code works correctly.
#
# TEST NAMING CONVENTION:
#   test_<what_we_test>_<expected_behavior>
#
# Example: test_clean_data_preserves_valid_rows
#   → "When we clean data, valid rows should NOT be removed"
#
# EACH TEST FOLLOWS THE PATTERN:
#   1. ARRANGE — set up the data
#   2. ACT    — run the function
#   3. ASSERT — check the result
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.config import Config


class TestDataPreprocessor:
    """Tests for the DataPreprocessor class."""
    
    def test_clean_data_preserves_valid_rows(self, sample_dataframe, config):
        """Valid rows should not be removed during cleaning."""
        preprocessor = DataPreprocessor(config)
        result = preprocessor.clean_data(sample_dataframe)
        
        # All 10 rows should survive (none have nulls or negative amounts)
        assert len(result) == len(sample_dataframe)
    
    def test_clean_data_removes_negative_amounts(self, sample_dataframe, config):
        """Transactions with negative amounts should be removed."""
        df = sample_dataframe.copy()
        df.loc[0, "amount"] = -100  # Inject a bad row
        
        preprocessor = DataPreprocessor(config)
        result = preprocessor.clean_data(df)
        
        assert len(result) == len(df) - 1
        assert (result["amount"] >= 0).all()
    
    def test_clean_data_adds_merchant_flags(self, sample_dataframe, config):
        """Merchant indicator columns should be created."""
        preprocessor = DataPreprocessor(config)
        result = preprocessor.clean_data(sample_dataframe)
        
        assert "isMerchantDest" in result.columns
        assert "isMerchantOrig" in result.columns
    
    def test_clean_data_handles_null_target(self, sample_dataframe, config):
        """Rows with null target should be dropped."""
        df = sample_dataframe.copy()
        df.loc[0, "isFraud"] = np.nan
        
        preprocessor = DataPreprocessor(config)
        result = preprocessor.clean_data(df)
        
        assert result["isFraud"].isnull().sum() == 0
        assert len(result) == len(df) - 1
    
    def test_clean_data_fills_numerical_nulls(self, sample_dataframe, config):
        """Null values in numerical columns should be filled with 0."""
        df = sample_dataframe.copy()
        df.loc[0, "amount"] = np.nan
        df.loc[0, "isFraud"] = 0  # Target must not be null
        
        preprocessor = DataPreprocessor(config)
        result = preprocessor.clean_data(df)
        
        # The row should survive (amount filled with 0, then filtered because 0 is not < 0)
        assert result["amount"].isnull().sum() == 0
    
    def test_validate_schema_passes_for_valid_data(self, sample_dataframe, config):
        """Schema validation should pass for correctly formatted data."""
        preprocessor = DataPreprocessor(config)
        assert preprocessor.validate_schema(sample_dataframe) is True
    
    def test_validate_schema_fails_for_missing_columns(self, config):
        """Schema validation should fail when required columns are missing."""
        bad_df = pd.DataFrame({"amount": [100], "wrong_column": [1]})
        
        preprocessor = DataPreprocessor(config)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocessor.validate_schema(bad_df)
    
    def test_stats_are_populated_after_cleaning(self, sample_dataframe, config):
        """Preprocessing stats should be available after cleaning."""
        preprocessor = DataPreprocessor(config)
        preprocessor.clean_data(sample_dataframe)
        
        stats = preprocessor.get_stats()
        assert "clean_shape" in stats
        assert "fraud_rate" in stats
        assert "fraud_count" in stats
        assert stats["fraud_count"] == 3  # Our sample has 3 fraud cases
