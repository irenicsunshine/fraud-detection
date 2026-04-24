# =============================================================================
# 🎓 TESTS FOR FEATURE ENGINEERING
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureEngineer
from src.config import Config


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""
    
    def test_create_features_adds_new_columns(self, sample_dataframe, config):
        """Feature engineering should add new columns."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        # Should have more columns than the original
        assert len(result.columns) > len(sample_dataframe.columns)
    
    def test_amount_log_feature(self, sample_dataframe, config):
        """Log-transformed amount should be created."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        assert "amount_log" in result.columns
        # log1p(100000) ≈ 11.51
        assert result["amount_log"].iloc[0] == pytest.approx(np.log1p(100000), rel=1e-4)
    
    def test_balance_error_detection(self, sample_dataframe, config):
        """Balance error features should detect inconsistencies."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        assert "balance_error_orig" in result.columns
        assert "balance_error_dest" in result.columns
        assert "has_balance_error_orig" in result.columns
    
    def test_temporal_features(self, sample_dataframe, config):
        """Time-based features should be created correctly."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        assert "hour_of_day" in result.columns
        assert "day_of_month" in result.columns
        assert "is_night_transaction" in result.columns
        
        # Hour should be 0-23
        assert result["hour_of_day"].between(0, 23).all()
    
    def test_suspicious_indicators(self, sample_dataframe, config):
        """Suspicious pattern indicators should be created."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        assert "full_balance_withdrawn" in result.columns
        assert "sender_zeroed_out" in result.columns
        assert "receiver_was_empty" in result.columns
    
    def test_account_features(self, sample_dataframe, config):
        """Account-level aggregation features should be created."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        assert "txn_count_orig" in result.columns
        assert "txn_mean_amount_orig" in result.columns
        assert "txn_count_dest" in result.columns
    
    def test_non_predictive_columns_dropped(self, sample_dataframe, config):
        """nameOrig and nameDest should be removed."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        assert "nameOrig" not in result.columns
        assert "nameDest" not in result.columns
    
    def test_target_column_preserved(self, sample_dataframe, config):
        """The isFraud target column should NOT be removed."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        assert "isFraud" in result.columns
    
    def test_encoder_fit_and_transform(self, sample_dataframe, config):
        """Categorical encoder should fit and transform correctly."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        # Fit encoder
        engineer.fit_encoder(result)
        
        # Transform
        encoded = engineer.encode_categoricals(result)
        
        # 'type' column should now be numeric
        assert encoded["type"].dtype in [np.int8, np.int16, np.int32, np.int64]
    
    def test_encoder_raises_before_fit(self, sample_dataframe, config):
        """Encoding without fitting should raise an error."""
        engineer = FeatureEngineer(config)
        result = engineer.create_features(sample_dataframe)
        
        with pytest.raises(RuntimeError, match="Encoder not fitted"):
            engineer.encode_categoricals(result)
    
    def test_no_data_leakage_in_features(self, sample_dataframe, config):
        """
        🎓 THE MOST IMPORTANT TEST
        Features should NOT depend on global statistics like mean/std.
        We verify by checking that features can be computed on a single row.
        """
        engineer = FeatureEngineer(config)
        
        # Take just ONE row
        single_row = sample_dataframe.iloc[[0]]
        
        # This should work without errors (no need for other rows' statistics)
        result = engineer.create_features(single_row)
        
        assert len(result) == 1
        assert "amount_log" in result.columns
