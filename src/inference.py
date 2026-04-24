# =============================================================================
# 🎓 STEP 2.1: INFERENCE ENGINE — From Pickle File to Predictions
# =============================================================================
#
# WHAT IS AN INFERENCE ENGINE?
# ------------------------------
# Training builds the model. Inference USES the model.
#
# It's like the difference between a chef developing a recipe (training)
# and a cook following the recipe to make dishes (inference).
#
# In production:
# 1. A transaction happens (e.g., someone sends $50,000)
# 2. The inference engine receives the transaction data
# 3. It applies the SAME feature engineering as training
# 4. It loads the trained model and gets a prediction
# 5. It returns: "fraud" or "legitimate" + probability + risk tier
#
# WHY A SEPARATE MODULE?
# -----------------------
# You might think: "Can't I just call model.predict()?"
# Yes, but production needs more:
# - Input validation (what if "amount" is missing?)
# - Feature engineering (same transforms as training)
# - Error handling (what if the model file is corrupted?)
# - Risk tiering (probability 0.95 vs 0.60 need different responses)
# - Logging (for auditing — "why was this flagged?")
#
# This module wraps all of that into a clean interface.
# =============================================================================

import pandas as pd
import numpy as np
import joblib
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from src.config import Config
from src.feature_engineering import FeatureEngineer

logger = logging.getLogger("fraud_detection.inference")


class FraudPredictor:
    """
    Production-ready fraud prediction engine.
    
    🎓 USAGE:
    
        # Initialize once (loads model from disk)
        predictor = FraudPredictor()
        
        # Predict on a single transaction
        result = predictor.predict({
            "step": 1,
            "type": "TRANSFER",
            "amount": 500000,
            "nameOrig": "C123",
            "oldbalanceOrg": 500000,
            "newbalanceOrig": 0,
            "nameDest": "C456",
            "oldbalanceDest": 0,
            "newbalanceDest": 500000,
        })
        
        print(result)
        # {
        #     "prediction": "fraud",
        #     "fraud_probability": 0.94,
        #     "risk_tier": "CRITICAL",
        #     "model_used": "random_forest",
        #     "inference_time_ms": 12.3,
        # }
    """
    
    # 🎓 RISK TIERS
    # These thresholds determine the response level:
    # CRITICAL → block transaction + immediate alert
    # HIGH     → flag for manual review within 1 hour
    # MEDIUM   → flag for review within 24 hours
    # LOW      → allow transaction, log for batch analysis
    RISK_TIERS = {
        "CRITICAL": 0.80,    # Very likely fraud
        "HIGH": 0.50,        # Probably fraud
        "MEDIUM": 0.30,      # Suspicious
        "LOW": 0.0,          # Likely legitimate
    }
    
    # Required fields for a valid transaction
    REQUIRED_FIELDS = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    ]
    
    # Valid transaction types
    VALID_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
    
    def __init__(
        self,
        model_name: str = "random_forest",
        model_dir: Optional[Path] = None,
        threshold: Optional[float] = None,
        config: Optional[Config] = None,
    ):
        """
        Load a trained model for inference.
        
        🎓 WHY LAZY LOADING?
        ---------------------
        We load the model in __init__ (not in predict) because:
        1. Loading from disk takes ~100-500ms
        2. predict() should be fast (<50ms)
        3. You load once, predict thousands of times
        
        Args:
            model_name: Which model to load ("random_forest", "xgboost", etc.)
            model_dir: Directory containing model pickle files
            threshold: Decision threshold (overrides default 0.5)
            config: Configuration object
        """
        self.config = config or Config()
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else self.config.paths.models_dir
        self.threshold = threshold or 0.5  # Will be overridden if optimal threshold exists
        
        # Load model
        self.model = self._load_model()
        self.feature_names = self._load_feature_names()
        self.feature_engineer = FeatureEngineer(self.config)

        # Load encoder (fitted during training, needed to encode 'type' column)
        encoder_path = self.model_dir / "encoder.pkl"
        if encoder_path.exists():
            self.feature_engineer.encoder = joblib.load(encoder_path)
            self.feature_engineer._fitted = True
            logger.info("Encoder loaded successfully")
        else:
            logger.warning("encoder.pkl not found — categorical encoding will be skipped")

        logger.info(f"FraudPredictor initialized: model={model_name}, "
                     f"threshold={self.threshold:.3f}")
    
    def _load_model(self) -> Any:
        """Load the trained model from disk."""
        # Try individual model file first, then combined file
        individual_path = self.model_dir / f"{self.model_name}.pkl"
        combined_path = self.model_dir / "trained_models.pkl"
        
        if individual_path.exists():
            logger.info(f"Loading model from: {individual_path}")
            return joblib.load(individual_path)
        elif combined_path.exists():
            logger.info(f"Loading model from combined file: {combined_path}")
            models = joblib.load(combined_path)
            if self.model_name in models:
                return models[self.model_name]
            else:
                available = list(models.keys())
                raise ValueError(
                    f"Model '{self.model_name}' not found in {combined_path}. "
                    f"Available models: {available}"
                )
        else:
            raise FileNotFoundError(
                f"No model files found in {self.model_dir}. "
                f"Looked for: {individual_path} and {combined_path}"
            )
    
    def _load_feature_names(self) -> List[str]:
        """Load the feature names used during training."""
        path = self.model_dir / "feature_names.pkl"
        if path.exists():
            names = joblib.load(path)
            logger.info(f"Loaded {len(names)} feature names")
            return names
        else:
            logger.warning("feature_names.pkl not found — will infer from model")
            return []
    
    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict whether a single transaction is fraudulent.
        
        🎓 THE PREDICTION FLOW:
        1. Validate input  → catch bad data early
        2. Create DataFrame → models expect tabular input
        3. Engineer features → same transforms as training
        4. Select features  → match training feature order
        5. Get probability  → model.predict_proba()
        6. Apply threshold  → probability → decision
        7. Assign risk tier → probability → response level
        
        Args:
            transaction: Dict with transaction fields (see REQUIRED_FIELDS)
            
        Returns:
            Dict with prediction, probability, risk tier, and timing
        """
        start = time.time()
        
        # Step 1: Validate
        self._validate_input(transaction)
        
        # Step 2: Create DataFrame
        # 🎓 ML models expect tabular data (rows × columns), not dicts.
        # pd.DataFrame([transaction]) creates a 1-row table from a dict.
        df = pd.DataFrame([transaction])
        
        # Step 3: Add placeholder fields needed by feature engineering
        if "isFraud" not in df.columns:
            df["isFraud"] = 0  # Placeholder — we're predicting this!
        if "isFlaggedFraud" not in df.columns:
            df["isFlaggedFraud"] = 0
        
        # Add merchant indicators
        df["isMerchantDest"] = df["nameDest"].str.startswith("M").astype("int8")
        df["isMerchantOrig"] = df["nameOrig"].str.startswith("M").astype("int8")
        
        # Step 4: Engineer features (same as training)
        df_features = self.feature_engineer.create_features(df)

        # Step 4b: Encode categoricals (same as training)
        if self.feature_engineer._fitted:
            df_features = self.feature_engineer.encode_categoricals(df_features)

        # Remove target column (we're predicting it, not using it as input!)
        if "isFraud" in df_features.columns:
            df_features = df_features.drop(columns=["isFraud"])

        # Step 5: Align features with training
        # 🎓 The model expects features in the EXACT same order as training.
        # If training had columns [A, B, C] but we have [C, A, B], predictions
        # would be garbage. This alignment ensures consistency.
        df_aligned = self._align_features(df_features)
        
        # Step 6: Get prediction
        fraud_probability = float(self.model.predict_proba(df_aligned)[0, 1])
        is_fraud = fraud_probability >= self.threshold
        
        # Step 7: Assign risk tier
        risk_tier = self._get_risk_tier(fraud_probability)
        
        elapsed_ms = (time.time() - start) * 1000
        
        result = {
            "prediction": "fraud" if is_fraud else "legitimate",
            "fraud_probability": round(fraud_probability, 4),
            "risk_tier": risk_tier,
            "threshold_used": self.threshold,
            "model_used": self.model_name,
            "inference_time_ms": round(elapsed_ms, 2),
        }
        
        # Log for auditing
        level = logging.WARNING if is_fraud else logging.DEBUG
        logger.log(level, f"Prediction: {result['prediction']} | "
                   f"P(fraud)={fraud_probability:.4f} | "
                   f"Risk={risk_tier} | "
                   f"Amount=${transaction.get('amount', 0):,.0f} | "
                   f"Time={elapsed_ms:.1f}ms")
        
        return result
    
    def predict_batch(
        self, transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict fraud for multiple transactions at once.
        
        🎓 WHY BATCH PREDICTION?
        --------------------------
        Predicting one-by-one has overhead (DataFrame creation, feature alignment).
        Batch prediction processes all transactions in one model call, which is
        much faster for large volumes (e.g., nightly batch processing).
        
        Args:
            transactions: DataFrame with transaction columns
            
        Returns:
            Original DataFrame with added prediction columns
        """
        start = time.time()
        logger.info(f"Batch prediction on {len(transactions):,} transactions")
        
        # Add placeholders
        df = transactions.copy()
        if "isFraud" not in df.columns:
            df["isFraud"] = 0
        if "isFlaggedFraud" not in df.columns:
            df["isFlaggedFraud"] = 0
        if "isMerchantDest" not in df.columns:
            df["isMerchantDest"] = df["nameDest"].str.startswith("M").astype("int8")
        if "isMerchantOrig" not in df.columns:
            df["isMerchantOrig"] = df["nameOrig"].str.startswith("M").astype("int8")
        
        # Feature engineering
        df_features = self.feature_engineer.create_features(df)
        if "isFraud" in df_features.columns:
            df_features = df_features.drop(columns=["isFraud"])
        
        # Align and predict
        df_aligned = self._align_features(df_features)
        probabilities = self.model.predict_proba(df_aligned)[:, 1]
        
        # Add results to original DataFrame
        results = transactions.copy()
        results["fraud_probability"] = probabilities
        results["prediction"] = np.where(probabilities >= self.threshold, "fraud", "legitimate")
        results["risk_tier"] = [self._get_risk_tier(p) for p in probabilities]
        
        elapsed = time.time() - start
        n_fraud = (results["prediction"] == "fraud").sum()
        
        logger.info(f"Batch complete: {n_fraud:,} fraud detected in {len(transactions):,} "
                     f"transactions ({elapsed:.2f}s, "
                     f"{len(transactions)/elapsed:.0f} txn/sec)")
        
        return results
    
    def _validate_input(self, transaction: Dict[str, Any]) -> None:
        """
        Validate transaction input.
        
        🎓 WHY INPUT VALIDATION?
        -------------------------
        In production, ANYONE can call your API with ANY data.
        Without validation:
          - Missing "amount" → cryptic KeyError deep in the code
          - amount = "abc"  → confusing type error
          - type = "INVALID" → silent wrong prediction
        
        With validation:
          - Missing "amount" → clear error: "Missing required field: amount"
          - amount = "abc" → clear error: "amount must be numeric"
          - type = "INVALID" → clear error: "type must be one of [CASH_IN, ...]"
        """
        # Check required fields
        missing = [f for f in self.REQUIRED_FIELDS if f not in transaction]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        # Validate amount
        amount = transaction["amount"]
        if not isinstance(amount, (int, float)):
            raise ValueError(f"'amount' must be numeric, got {type(amount).__name__}")
        if amount < 0:
            raise ValueError(f"'amount' must be non-negative, got {amount}")
        
        # Validate type
        txn_type = transaction["type"]
        if txn_type not in self.VALID_TYPES:
            raise ValueError(
                f"'type' must be one of {self.VALID_TYPES}, got '{txn_type}'"
            )
    
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align DataFrame columns with what the model expects.
        
        🎓 FEATURE ALIGNMENT
        ----------------------
        Training might produce features: [A, B, C, D, E]
        Inference might produce:          [A, C, E, F, B]  (different order + extra F)
        
        We need to:
        1. Reorder to match training: [A, B, C, D, E]
        2. Add missing columns as 0:  D was missing → add D=0
        3. Drop extra columns:        F wasn't in training → drop it
        """
        if not self.feature_names:
            logger.warning("No feature names loaded — using DataFrame columns as-is")
            return df
        
        # Add missing columns as 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
                logger.debug(f"Added missing feature '{col}' with value 0")
        
        # Select and reorder to match training
        return df[self.feature_names]
    
    def _get_risk_tier(self, probability: float) -> str:
        """Map fraud probability to a risk tier."""
        for tier, threshold in self.RISK_TIERS.items():
            if probability >= threshold:
                return tier
        return "LOW"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "threshold": self.threshold,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "risk_tiers": self.RISK_TIERS,
        }


# ---------------------------------------------------------------------------
# Quick test with the pre-trained model
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.config import setup_logging
    
    config = Config()
    setup_logging("DEBUG")
    
    try:
        predictor = FraudPredictor(model_name="random_forest", config=config)
        
        # Test with a suspicious transaction
        result = predictor.predict({
            "step": 1,
            "type": "TRANSFER",
            "amount": 500000,
            "nameOrig": "C123456789",
            "oldbalanceOrg": 500000,
            "newbalanceOrig": 0,
            "nameDest": "C987654321",
            "oldbalanceDest": 0,
            "newbalanceDest": 500000,
        })
        
        print(f"\n🔍 Prediction Result:")
        for key, value in result.items():
            print(f"   {key}: {value}")
        
        # Test with a normal transaction
        result2 = predictor.predict({
            "step": 10,
            "type": "PAYMENT",
            "amount": 150,
            "nameOrig": "C111111111",
            "oldbalanceOrg": 5000,
            "newbalanceOrig": 4850,
            "nameDest": "M222222222",
            "oldbalanceDest": 100000,
            "newbalanceDest": 100150,
        })
        
        print(f"\n✅ Normal Transaction:")
        for key, value in result2.items():
            print(f"   {key}: {value}")
            
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.info("Train models first with: python main.py --mode train")
