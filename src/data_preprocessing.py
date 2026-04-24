# =============================================================================
# 🎓 STEP 1.2: DATA PREPROCESSING — The First Line of Defense
# =============================================================================
#
# WHY IS PREPROCESSING SO IMPORTANT?
# -----------------------------------
# "Garbage in, garbage out" — the #1 rule in ML.
#
# If your data has:
#   - Missing values    → model learns wrong patterns
#   - Wrong types       → wastes 2× memory (float64 vs float32)
#   - No validation     → silent bugs that corrupt everything downstream
#   - print() logging   → no way to debug in production
#
# This module is the GATEKEEPER. It ensures only clean, validated, efficient
# data reaches the rest of the pipeline.
#
# WHAT CHANGED FROM THE ORIGINAL?
# --------------------------------
# Original:
#   ❌ df.dropna()           → nuclear: drops entire rows for ANY null
#   ❌ pd.read_csv(path)     → loads everything as float64 (wastes memory)
#   ❌ print("...")           → lost in production
#   ❌ hardcoded file path   → can't change without editing code
#
# New:
#   ✅ Column-specific null handling (impute vs drop)
#   ✅ Dtype optimization (50% memory savings)
#   ✅ Schema validation (catch data issues early)
#   ✅ Structured logging (timestamps, levels, filterable)
#   ✅ Config-driven (all paths and settings from config.py)
# =============================================================================

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from src.config import Config, FeatureConfig

logger = logging.getLogger("fraud_detection.preprocessing")


class DataPreprocessor:
    """
    Loads, validates, and cleans financial transaction data.
    
    🎓 WHY A CLASS?
    ----------------
    We use a class (not just functions) because:
    1. The preprocessor needs to REMEMBER things (like which columns had nulls)
    2. Multiple methods share the same config
    3. We can create different preprocessors for different datasets
    
    Usage:
        config = Config()
        preprocessor = DataPreprocessor(config)
        df = preprocessor.load_and_clean()
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: Configuration object. If None, uses defaults.
                    This pattern is called "dependency injection" — we pass
                    the config IN instead of creating it inside. This makes
                    testing much easier (you can pass a test config).
        """
        self.config = config or Config()
        self.stats: Dict[str, Any] = {}  # Store statistics about the data
    
    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load dataset from CSV with memory optimization.
        
        🎓 DTYPE OPTIMIZATION — WHY IT MATTERS
        ----------------------------------------
        By default, pandas reads numbers as float64 (8 bytes per number).
        But most of our numbers fit in float32 (4 bytes) just fine.
        
        For 6.3M rows × 11 columns:
          float64: 6.3M × 11 × 8 bytes = ~554 MB
          float32: 6.3M × 11 × 4 bytes = ~277 MB  ← 50% savings!
        
        We specify dtypes upfront so pandas doesn't waste memory.
        
        Args:
            file_path: Path to CSV. If None, uses config default.
            
        Returns:
            Raw DataFrame loaded with optimized dtypes.
            
        Raises:
            FileNotFoundError: If the CSV doesn't exist.
        """
        path = Path(file_path) if file_path else self.config.paths.raw_data_path
        
        if not path.exists():
            logger.error(f"Data file not found: {path}")
            logger.info("Run 'python download_data.py' to download the dataset")
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # 🎓 We specify dtypes to save memory. 'category' is like an enum —
        # it stores each unique value once and uses integer codes internally.
        # For a column with 5 unique values across 6.3M rows, this saves ~95% memory.
        dtype_map = {
            "step": "int32",           # Time step (0-743), int32 is plenty
            "type": "category",        # 5 unique values → category is perfect
            "amount": "float32",       # Money amounts — float32 gives 7 decimal digits
            "nameOrig": "object",      # Account IDs — keep as strings
            "oldbalanceOrg": "float32",
            "newbalanceOrig": "float32",
            "nameDest": "object",
            "oldbalanceDest": "float32",
            "newbalanceDest": "float32",
            "isFraud": "int8",         # 0 or 1 — int8 uses just 1 byte!
            "isFlaggedFraud": "int8",  # 0 or 1
        }
        
        logger.info(f"Loading data from: {path}")
        
        df = pd.read_csv(path, dtype=dtype_map)
        
        # Store some statistics for later reporting
        self.stats["raw_shape"] = df.shape
        self.stats["raw_memory_mb"] = df.memory_usage(deep=True).sum() / 1024**2
        
        logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        logger.info(f"Memory usage: {self.stats['raw_memory_mb']:.1f} MB")
        
        return df
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has the expected columns and types.
        
        🎓 WHY VALIDATE?
        ------------------
        Imagine you deploy this system and someone uploads a CSV with:
          - A misspelled column ("ammount" instead of "amount")
          - Missing the target column ("isFraud")
          - Extra columns you didn't expect
        
        Without validation, the pipeline would crash deep inside model training
        with a cryptic error. With validation, you get a clear message immediately.
        
        This is called "FAIL FAST" — it's better to fail early with a clear
        error than to fail late with a mysterious one.
        
        Returns:
            True if schema is valid.
            
        Raises:
            ValueError: If required columns are missing.
        """
        required_columns = {
            "step", "type", "amount", "nameOrig", "oldbalanceOrg",
            "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
            "isFraud", "isFlaggedFraud"
        }
        
        actual_columns = set(df.columns)
        missing = required_columns - actual_columns
        
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Got columns: {sorted(actual_columns)}"
            )
        
        # Check target variable has expected values
        unique_targets = set(df[self.config.features.target_column].unique())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(
                f"Target column '{self.config.features.target_column}' has unexpected "
                f"values: {unique_targets}. Expected only {{0, 1}}."
            )
        
        logger.info("Schema validation passed ✓")
        return True
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset with proper null handling and filtering.
        
        🎓 WHAT CHANGED FROM ORIGINAL
        ------------------------------
        Original: df.dropna()  → Drops ALL rows with ANY null value.
                  This is dangerous because:
                  - You might lose 10% of your data unnecessarily
                  - A null in a non-critical column kills the whole row
                  - You never even know which columns had nulls
        
        New approach:
        1. Log null counts per column (so you know what's happening)
        2. Drop rows only if the TARGET (isFraud) is null
        3. Fill numerical nulls with 0 (reasonable for balance fields)
        4. Filter invalid transactions with logging
        
        Args:
            df: Raw DataFrame from load_data()
            
        Returns:
            Cleaned DataFrame ready for feature engineering.
        """
        initial_rows = len(df)
        logger.info(f"Starting data cleaning on {initial_rows:,} rows")
        
        # --- Step 1: Report nulls per column ---
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found {null_counts.sum()} total null values:")
            for col, count in null_counts[null_counts > 0].items():
                logger.warning(f"  {col}: {count:,} nulls ({count/len(df)*100:.3f}%)")
        else:
            logger.info("No null values found ✓")
        
        # --- Step 2: Drop rows where target is null (can't train on these) ---
        target = self.config.features.target_column
        if df[target].isnull().any():
            before = len(df)
            df = df.dropna(subset=[target])
            logger.warning(f"Dropped {before - len(df)} rows with null target")
        
        # --- Step 3: Fill numerical nulls with 0 ---
        # 🎓 WHY 0 AND NOT MEDIAN?
        # For balance fields, 0 is a meaningful value (empty account).
        # Median might introduce bias because the median of all balances
        # is NOT the same as "no data available."
        num_cols = self.config.features.raw_numerical_columns
        for col in num_cols:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                df[col] = df[col].fillna(0)
                logger.info(f"Filled {null_count} nulls in '{col}' with 0")
        
        # --- Step 4: Filter invalid transactions ---
        # 🎓 Note: We keep amount=0 transactions (unlike the original).
        # Zero-amount transactions could be balance checks or account updates.
        # Only filter negative amounts which are clearly data errors.
        negative_mask = df["amount"] < 0
        if negative_mask.any():
            count = negative_mask.sum()
            df = df[~negative_mask]
            logger.warning(f"Removed {count} transactions with negative amounts")
        
        # --- Step 5: Create merchant indicator features ---
        # 🎓 In this dataset, account names starting with 'M' are merchants.
        # Merchants behave differently (they receive money, not send).
        # This is a binary feature: 1 = merchant, 0 = individual.
        df = df.copy()  # Avoid SettingWithCopyWarning
        df["isMerchantDest"] = df["nameDest"].str.startswith("M").astype("int8")
        df["isMerchantOrig"] = df["nameOrig"].str.startswith("M").astype("int8")
        
        # --- Log summary ---
        removed = initial_rows - len(df)
        fraud_rate = df[target].mean()
        
        self.stats["clean_shape"] = df.shape
        self.stats["rows_removed"] = removed
        self.stats["fraud_rate"] = fraud_rate
        self.stats["fraud_count"] = int(df[target].sum())
        self.stats["legitimate_count"] = int(len(df) - df[target].sum())
        
        logger.info(f"Cleaning complete: {len(df):,} rows remaining ({removed:,} removed)")
        logger.info(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.3f}%)")
        logger.info(f"Class distribution: {self.stats['legitimate_count']:,} legitimate, "
                     f"{self.stats['fraud_count']:,} fraud")
        
        return df
    
    def load_and_clean(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Convenience method: load + validate + clean in one call.
        
        This is the method you'll use 90% of the time.
        
        Args:
            file_path: Optional path override.
            
        Returns:
            Clean, validated DataFrame ready for feature engineering.
        """
        df = self.load_data(file_path)
        self.validate_schema(df)
        df = self.clean_data(df)
        return df
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics collected during preprocessing."""
        return self.stats


# ---------------------------------------------------------------------------
# Quick test: run this file directly to see it in action
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.config import setup_logging
    
    config = Config()
    setup_logging(config.log_level)
    
    preprocessor = DataPreprocessor(config)
    
    try:
        df = preprocessor.load_and_clean()
        print(f"\nPreprocessing stats: {preprocessor.get_stats()}")
    except FileNotFoundError:
        logger.warning("No data file found — this is expected if you haven't "
                       "downloaded the dataset yet.")
        logger.info("The preprocessing module is ready. Run download_data.py first.")
