# =============================================================================
# 🎓 TESTS FOR THE INFERENCE ENGINE
# =============================================================================

import pytest
from src.inference import FraudPredictor
from src.config import Config


class TestFraudPredictorValidation:
    """Tests for input validation (don't need a trained model)."""
    
    def test_missing_required_field(self, sample_transaction):
        """Should raise error for missing required fields."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.REQUIRED_FIELDS = FraudPredictor.REQUIRED_FIELDS
        predictor.VALID_TYPES = FraudPredictor.VALID_TYPES
        
        # Remove a required field
        bad_txn = {k: v for k, v in sample_transaction.items() if k != "amount"}
        
        with pytest.raises(ValueError, match="Missing required fields"):
            predictor._validate_input(bad_txn)
    
    def test_negative_amount_rejected(self, sample_transaction):
        """Should raise error for negative amounts."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.REQUIRED_FIELDS = FraudPredictor.REQUIRED_FIELDS
        predictor.VALID_TYPES = FraudPredictor.VALID_TYPES
        
        bad_txn = sample_transaction.copy()
        bad_txn["amount"] = -100
        
        with pytest.raises(ValueError, match="non-negative"):
            predictor._validate_input(bad_txn)
    
    def test_invalid_type_rejected(self, sample_transaction):
        """Should raise error for invalid transaction types."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.REQUIRED_FIELDS = FraudPredictor.REQUIRED_FIELDS
        predictor.VALID_TYPES = FraudPredictor.VALID_TYPES
        
        bad_txn = sample_transaction.copy()
        bad_txn["type"] = "INVALID_TYPE"
        
        with pytest.raises(ValueError, match="must be one of"):
            predictor._validate_input(bad_txn)
    
    def test_valid_transaction_passes(self, sample_transaction):
        """Valid transaction should pass validation without errors."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.REQUIRED_FIELDS = FraudPredictor.REQUIRED_FIELDS
        predictor.VALID_TYPES = FraudPredictor.VALID_TYPES
        
        # Should not raise any exception
        predictor._validate_input(sample_transaction)
    
    def test_risk_tier_critical(self):
        """High probability should map to CRITICAL tier."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.RISK_TIERS = FraudPredictor.RISK_TIERS
        
        assert predictor._get_risk_tier(0.95) == "CRITICAL"
    
    def test_risk_tier_low(self):
        """Low probability should map to LOW tier."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.RISK_TIERS = FraudPredictor.RISK_TIERS
        
        assert predictor._get_risk_tier(0.05) == "LOW"
    
    def test_risk_tier_medium(self):
        """Medium probability should map to MEDIUM tier."""
        predictor = FraudPredictor.__new__(FraudPredictor)
        predictor.RISK_TIERS = FraudPredictor.RISK_TIERS
        
        assert predictor._get_risk_tier(0.35) == "MEDIUM"


class TestFraudPredictorIntegration:
    """
    Integration tests that require a trained model.
    These tests are skipped if no model file exists.
    """
    
    @pytest.fixture
    def predictor(self, config):
        """Try to load the pre-trained model."""
        try:
            return FraudPredictor(model_name="random_forest", config=config)
        except FileNotFoundError:
            pytest.skip("No trained model available — skipping integration tests")
    
    def test_predict_returns_expected_keys(self, predictor, sample_transaction):
        """Prediction result should contain all expected fields."""
        result = predictor.predict(sample_transaction)
        
        expected_keys = {
            "prediction", "fraud_probability", "risk_tier",
            "threshold_used", "model_used", "inference_time_ms",
        }
        assert expected_keys.issubset(set(result.keys()))
    
    def test_predict_fraud_probability_range(self, predictor, sample_transaction):
        """Fraud probability should be between 0 and 1."""
        result = predictor.predict(sample_transaction)
        
        assert 0 <= result["fraud_probability"] <= 1
    
    def test_predict_returns_valid_prediction(self, predictor, sample_transaction):
        """Prediction should be either 'fraud' or 'legitimate'."""
        result = predictor.predict(sample_transaction)
        
        assert result["prediction"] in {"fraud", "legitimate"}
    
    def test_inference_time_under_100ms(self, predictor, sample_transaction):
        """Single prediction should take less than 100ms."""
        result = predictor.predict(sample_transaction)
        
        assert result["inference_time_ms"] < 100
    
    def test_model_info(self, predictor):
        """Model info should return valid metadata."""
        info = predictor.get_model_info()
        
        assert "model_name" in info
        assert "threshold" in info
        assert info["n_features"] > 0
