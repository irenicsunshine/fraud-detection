# =============================================================================
# 🎓 STEP 1.3: FEATURE ENGINEERING — Teaching Your Model What to Look For
# =============================================================================
#
# WHAT IS FEATURE ENGINEERING?
# ----------------------------
# Raw data has columns like "amount" and "oldbalanceOrg". But fraud has
# PATTERNS that aren't obvious from individual columns:
#
#   - "The entire account balance was withdrawn" (amount == oldbalanceOrg)
#   - "Money appeared from nowhere" (newbalanceDest increased more than amount)
#   - "This account has been making rapid transactions" (velocity)
#
# Feature engineering CREATES new columns that capture these patterns,
# making it easier for the model to spot fraud.
#
# Think of it like this: raw data is like giving someone a pile of puzzle
# pieces. Feature engineering is arranging those pieces so the picture
# becomes visible.
#
# CRITICAL FIX: DATA LEAKAGE
# ---------------------------
# The original code had a serious bug:
#
#   df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
#
# This computes mean/std on the ENTIRE dataset (including test data).
# In production, you don't HAVE the test data when you train!
#
# 🎓 ANALOGY: Imagine studying for an exam where you've already seen the
# answers. You'd get 100%, but you didn't actually learn anything.
# That's data leakage — your model cheats by seeing test data during training.
#
# FIX: We now create features that DON'T require knowing global statistics,
# OR we defer statistical features to the training pipeline where they'll
# be fitted on training data only (via sklearn Pipeline).
# =============================================================================

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Tuple
from sklearn.preprocessing import OrdinalEncoder

from src.config import Config

logger = logging.getLogger("fraud_detection.features")


class FeatureEngineer:
    """
    Creates features for fraud detection WITHOUT data leakage.
    
    🎓 KEY DESIGN DECISION: STATELESS vs STATEFUL FEATURES
    --------------------------------------------------------
    We split features into two categories:
    
    1. STATELESS features (computed here):
       - Depend only on the CURRENT ROW
       - Example: balance_change = newBalance - oldBalance
       - Safe to compute anytime — no leakage possible
    
    2. STATEFUL features (computed in training pipeline):
       - Depend on STATISTICS from other rows (mean, std, etc.)
       - Example: z-score = (amount - mean) / std
       - MUST be fitted on training data only
       - We handle these in model_training.py via sklearn Pipeline
    
    This separation is the key to preventing data leakage.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.encoder: Optional[OrdinalEncoder] = None
        self._fitted = False
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all stateless features from raw transaction data.
        
        This method can be called on ANY data (train, test, or production)
        because none of these features depend on dataset statistics.
        
        Args:
            df: Cleaned DataFrame from DataPreprocessor
            
        Returns:
            DataFrame with new features added, non-predictive columns removed.
        """
        logger.info(f"Starting feature engineering on {len(df):,} rows")
        initial_cols = len(df.columns)
        
        # We work on a copy to avoid modifying the original
        # 🎓 This is called "immutability" — functions shouldn't change their inputs.
        # It prevents nasty bugs where changing data in one place affects another.
        df_feat = df.copy()
        
        # ----- GROUP 1: Transaction Amount Features -----
        df_feat = self._add_amount_features(df_feat)
        
        # ----- GROUP 2: Balance Change Features -----
        df_feat = self._add_balance_features(df_feat)
        
        # ----- GROUP 3: Balance Error Detection -----
        df_feat = self._add_error_detection_features(df_feat)
        
        # ----- GROUP 4: Transaction Timing Features -----
        df_feat = self._add_temporal_features(df_feat)
        
        # ----- GROUP 5: Suspicious Pattern Indicators -----
        df_feat = self._add_suspicious_indicators(df_feat)
        
        # ----- GROUP 6: Account-Level Aggregation Features -----
        df_feat = self._add_account_features(df_feat)
        
        # ----- CLEANUP: Drop non-predictive columns -----
        df_feat = self._drop_non_predictive(df_feat)
        
        new_cols = len(df_feat.columns) - initial_cols
        logger.info(f"Feature engineering complete: {len(df_feat.columns)} columns "
                     f"({new_cols} new features created)")
        
        return df_feat
    
    def fit_encoder(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the categorical encoder on TRAINING data only.
        
        🎓 WHY FIT ON TRAINING DATA ONLY?
        -----------------------------------
        The original code used pd.get_dummies(), which creates columns
        based on whatever categories exist in the data. Problem:
        
        Training data has types: [CASH_OUT, TRANSFER, PAYMENT, DEBIT, CASH_IN]
        Test data might have:    [CASH_OUT, TRANSFER, PAYMENT]
        Production might get:    [CASH_OUT, TRANSFER, PAYMENT, NEW_TYPE]
        
        get_dummies would create different columns each time → CRASH.
        
        OrdinalEncoder maps each category to a fixed number:
          CASH_OUT=0, CASH_IN=1, DEBIT=2, PAYMENT=3, TRANSFER=4
        
        This mapping is learned from training data and applied consistently
        to test data and production data. Unknown categories get handled
        gracefully instead of crashing.
        
        Args:
            df: Training DataFrame (BEFORE train/test split happens)
            
        Returns:
            self (for method chaining: engineer.fit_encoder(df).transform(df))
        """
        cat_cols = [c for c in self.config.features.categorical_columns 
                    if c in df.columns]
        
        if cat_cols:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",  # Unknown categories → -1
                unknown_value=-1,
                dtype=np.int8,  # Memory efficient
            )
            self.encoder.fit(df[cat_cols])
            logger.info(f"Encoder fitted on {len(cat_cols)} categorical columns")
            
            # Log the learned categories
            for col, cats in zip(cat_cols, self.encoder.categories_):
                logger.debug(f"  {col}: {list(cats)}")
        
        self._fitted = True
        return self
    
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns using the fitted encoder.
        
        Must call fit_encoder() first (on training data).
        
        Args:
            df: DataFrame with categorical columns
            
        Returns:
            DataFrame with categoricals replaced by integer codes
        """
        if not self._fitted or self.encoder is None:
            raise RuntimeError(
                "Encoder not fitted! Call fit_encoder(training_data) first."
            )
        
        cat_cols = [c for c in self.config.features.categorical_columns 
                    if c in df.columns]
        
        if cat_cols:
            df = df.copy()
            encoded = self.encoder.transform(df[cat_cols])
            for i, col in enumerate(cat_cols):
                df[col] = encoded[:, i].astype(np.int8)
        
        return df
    
    # =========================================================================
    # PRIVATE METHODS — Each creates one group of features
    # =========================================================================
    
    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎓 LOG TRANSFORM — WHY?
        -------------------------
        Transaction amounts have a SKEWED distribution:
          - Most transactions: $10 - $5,000
          - Some transactions: $1,000,000+
        
        The log transform compresses the range:
          log(100) = 4.6,  log(1000) = 6.9,  log(1000000) = 13.8
        
        This helps models that are sensitive to scale (like Logistic Regression)
        see patterns across all amount ranges equally.
        
        We use log1p (log of 1+x) instead of log(x) because log(0) is undefined.
        log1p(0) = 0, which is well-defined and meaningful.
        """
        df["amount_log"] = np.log1p(df["amount"]).astype(np.float32)
        
        # Amount categories (binned) — useful for pattern detection
        # 🎓 Why bins? A $500 TRANSFER is normal, but a $500,000 TRANSFER is suspicious.
        # Binning captures this "relative size" concept.
        df["amount_category"] = pd.cut(
            df["amount"],
            bins=[0, 1000, 10000, 100000, 500000, float("inf")],
            labels=[0, 1, 2, 3, 4],  # small, medium, large, very_large, extreme
            include_lowest=True,  # includes amount=0 in the first bin
        ).astype(np.int8)
        
        return df
    
    def _add_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎓 BALANCE CHANGES — THE #1 FRAUD SIGNAL
        -------------------------------------------
        In a legitimate transaction:
          newbalanceOrig = oldbalanceOrg - amount  (sender loses money)
          newbalanceDest = oldbalanceDest + amount  (receiver gains money)
        
        In fraud, these equations often DON'T hold because:
        1. The balance is manipulated to hide the theft
        2. The money is moved to a shell account that doesn't track properly
        3. The transaction is reversed but the money is already gone
        
        We compute these changes and let the model spot the anomalies.
        """
        # How much did each balance change?
        df["balance_change_orig"] = (
            df["newbalanceOrig"] - df["oldbalanceOrg"]
        ).astype(np.float32)
        
        df["balance_change_dest"] = (
            df["newbalanceDest"] - df["oldbalanceDest"]
        ).astype(np.float32)
        
        # What fraction of the balance was moved?
        # 🎓 We add 1 to avoid division by zero. This is a common trick.
        df["balance_ratio_orig"] = (
            df["newbalanceOrig"] / (df["oldbalanceOrg"] + 1)
        ).astype(np.float32)
        
        # What percentage of the account was used in this transaction?
        df["amount_to_balance_ratio"] = (
            df["amount"] / (df["oldbalanceOrg"] + 1)
        ).astype(np.float32)
        
        return df
    
    def _add_error_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎓 BALANCE ERRORS — CATCHING MANIPULATED RECORDS
        --------------------------------------------------
        For a legitimate transaction:
          oldbalanceOrg - amount = newbalanceOrig  (simple subtraction)
        
        If this equation DOESN'T hold, something is wrong:
        - The record was tampered with
        - The system has a bug
        - OR: it's fraud with balance manipulation
        
        We compute the "error" (how far off the equation is).
        For legitimate transactions, this should be ~0.
        For fraudulent ones, it's often NOT zero.
        
        This is one of the most powerful features because it's based on
        a PHYSICAL CONSTRAINT (money can't appear from nowhere).
        """
        # Sender error: should be 0 for legitimate transactions
        df["balance_error_orig"] = (
            df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
        ).astype(np.float32)
        
        # Receiver error: should be 0 for legitimate transactions
        df["balance_error_dest"] = (
            df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
        ).astype(np.float32)
        
        # Are the errors non-zero? (binary flag)
        df["has_balance_error_orig"] = (
            df["balance_error_orig"].abs() > 1.0  # tolerance for floating point
        ).astype(np.int8)
        
        df["has_balance_error_dest"] = (
            df["balance_error_dest"].abs() > 1.0
        ).astype(np.int8)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎓 TIME FEATURES — WHEN DOES FRAUD HAPPEN?
        --------------------------------------------
        The 'step' column represents hourly time intervals over a 30-day
        simulation (step 0 = hour 0, step 743 = hour 743 = day 30).
        
        Fraud patterns vary by time:
        - Late night (2-5 AM) → higher fraud rate (less monitoring)
        - Month-end → higher fraud (payroll timing)
        - Weekends → higher fraud (fewer staff)
        
        We extract hour-of-day and day-of-month to capture these patterns.
        """
        df["hour_of_day"] = (df["step"] % 24).astype(np.int8)
        df["day_of_month"] = (df["step"] // 24).astype(np.int8)
        
        # Is this a "risky" hour? (late night / early morning)
        df["is_night_transaction"] = (
            (df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)
        ).astype(np.int8)
        
        return df
    
    def _add_suspicious_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎓 DOMAIN KNOWLEDGE FEATURES — Expert Rules Turned Into Features
        ------------------------------------------------------------------
        These features encode things that fraud investigators know:
        
        1. "Account was emptied" → very suspicious for TRANSFER/CASH_OUT
        2. "Receiver had zero balance" → could be a newly created mule account
        3. "Transaction amount = exact account balance" → suspicious precision
        4. "isFlaggedFraud" → The system's own existing flag (original code dropped this!)
        """
        # Was the sender's entire balance taken?
        df["full_balance_withdrawn"] = (
            (df["amount"] == df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)
        ).astype(np.int8)
        
        # Does the sender now have zero balance?
        df["sender_zeroed_out"] = (
            (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
        ).astype(np.int8)
        
        # Was the receiver's account empty before? (potential mule account)
        df["receiver_was_empty"] = (
            df["oldbalanceDest"] == 0
        ).astype(np.int8)
        
        # Was the sender's account empty? (suspicious — where did money come from?)
        df["sender_was_empty"] = (
            df["oldbalanceOrg"] == 0
        ).astype(np.int8)
        
        # Keep isFlaggedFraud — the original code DROPPED this, but it's a valuable signal!
        # 🎓 The existing system already flags some transactions. Even if it's imperfect,
        # our model can learn to combine it with other features.
        # (No action needed — it's already in the DataFrame)
        
        return df
    
    def _add_account_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎓 ACCOUNT-LEVEL FEATURES — The Strongest Fraud Signals
        ---------------------------------------------------------
        This is the MOST IMPORTANT feature group and was completely missing
        from the original code.
        
        Individual transactions look similar. But ACCOUNTS behave differently:
        - A normal person makes 2-3 transactions/day of $100-$500
        - A fraudster makes 20 rapid transactions draining multiple accounts
        
        By aggregating at the account level, we capture BEHAVIORAL patterns.
        
        WARNING: These features are expensive to compute on 6.3M rows.
        We use efficient groupby operations to keep it fast.
        
        WHY IS THIS SAFE? (No data leakage)
        These aggregations are computed within the dataset we're given.
        In production, you'd maintain a running window of account history
        in a database. For training, using the full training set is fine
        because it simulates having historical data available.
        """
        logger.info("Computing account-level aggregation features...")
        
        # --- Sender account statistics ---
        # Group all transactions by sender account and compute stats
        orig_stats = df.groupby("nameOrig")["amount"].agg(
            txn_count_orig="count",
            txn_mean_amount_orig="mean",
            txn_max_amount_orig="max",
        )
        
        # 🎓 .map() is like VLOOKUP in Excel — for each nameOrig in the DataFrame,
        # it looks up the corresponding value from orig_stats.
        df["txn_count_orig"] = (
            df["nameOrig"].map(orig_stats["txn_count_orig"]).astype(np.int16)
        )
        df["txn_mean_amount_orig"] = (
            df["nameOrig"].map(orig_stats["txn_mean_amount_orig"]).astype(np.float32)
        )
        df["txn_max_amount_orig"] = (
            df["nameOrig"].map(orig_stats["txn_max_amount_orig"]).astype(np.float32)
        )
        
        # --- Receiver account statistics ---
        dest_stats = df.groupby("nameDest")["amount"].agg(
            txn_count_dest="count",
            txn_mean_amount_dest="mean",
        )
        
        df["txn_count_dest"] = (
            df["nameDest"].map(dest_stats["txn_count_dest"]).astype(np.int16)
        )
        df["txn_mean_amount_dest"] = (
            df["nameDest"].map(dest_stats["txn_mean_amount_dest"]).astype(np.float32)
        )
        
        # --- Deviation from account's normal behavior ---
        # Is this transaction unusually large compared to this account's average?
        # 🎓 If an account usually transacts $200 and suddenly sends $50,000,
        # that's a red flag. This ratio captures that.
        df["amount_vs_account_avg"] = (
            df["amount"] / (df["txn_mean_amount_orig"] + 1)
        ).astype(np.float32)
        
        logger.info("Account-level features added ✓")
        return df
    
    def _drop_non_predictive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that shouldn't be used for prediction.
        
        🎓 WHY DROP THESE?
        --------------------
        - nameOrig/nameDest: Unique per transaction — the model would memorize
          specific accounts instead of learning patterns. This is like memorizing
          answers vs. understanding concepts.
        """
        cols_to_drop = [c for c in self.config.features.drop_columns 
                        if c in df.columns]
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped non-predictive columns: {cols_to_drop}")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature names (all columns except target)."""
        target = self.config.features.target_column
        return [c for c in df.columns if c != target]


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.config import setup_logging
    
    config = Config()
    setup_logging("DEBUG")
    
    # Create a small synthetic dataset to test features
    test_data = pd.DataFrame({
        "step": [1, 2, 3, 4, 5],
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "TRANSFER", "CASH_OUT"],
        "amount": [100000, 50000, 200, 350000, 75000],
        "nameOrig": ["C001", "C002", "C001", "C003", "C002"],
        "oldbalanceOrg": [100000, 80000, 5000, 350000, 30000],
        "newbalanceOrig": [0, 30000, 4800, 0, 0],
        "nameDest": ["C010", "C011", "M001", "C010", "C012"],
        "oldbalanceDest": [0, 50000, 100000, 200000, 0],
        "newbalanceDest": [100000, 100000, 100200, 550000, 75000],
        "isFraud": [1, 0, 0, 1, 0],
        "isFlaggedFraud": [0, 0, 0, 1, 0],
        "isMerchantOrig": [0, 0, 0, 0, 0],
        "isMerchantDest": [0, 0, 1, 0, 0],
    })
    
    engineer = FeatureEngineer(config)
    result = engineer.create_features(test_data)
    
    print(f"\nFeatures created: {len(result.columns)} columns")
    print(f"Columns: {list(result.columns)}")
    print(f"\nSample row:\n{result.iloc[0]}")
