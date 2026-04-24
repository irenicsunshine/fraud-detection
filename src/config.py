# =============================================================================
# 🎓 STEP 1.1: CENTRALIZED CONFIGURATION
# =============================================================================
#
# WHY DO WE NEED THIS?
# --------------------
# In the original code, you'll find things like:
#     pd.read_csv('data/PS_20174392719_1491204439457_log.csv')  # hardcoded path!
#     n_estimators=100  # magic number buried in code
#     test_size=0.3     # another magic number
#
# This is bad because:
# 1. If you want to change a value, you have to hunt through multiple files
# 2. If two files use different values for the same thing, bugs appear
# 3. When deploying, you can't easily switch between dev/staging/production
#
# SOLUTION: Put ALL configurable values in ONE place.
#
# WHAT IS A DATACLASS?
# --------------------
# A dataclass is Python's way of creating a class that's mainly for storing data.
# Instead of writing:
#
#     class Config:
#         def __init__(self):
#             self.test_size = 0.3
#             self.random_state = 42
#
# You write:
#
#     @dataclass
#     class Config:
#         test_size: float = 0.3
#         random_state: int = 42
#
# Same result, but cleaner, less code, and Python auto-generates __init__,
# __repr__, __eq__ for you. Think of it as a "smart dictionary."
# =============================================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import logging

# ---------------------------------------------------------------------------
# 🎓 LOGGING vs PRINT
# ---------------------------------------------------------------------------
# In production, you NEVER use print(). Why?
#
# print("Loading data...")           → Goes to stdout, lost forever
# logger.info("Loading data...")     → Goes to a file, has timestamps,
#                                      can be filtered by severity level
#
# Levels (from least to most severe):
#   DEBUG    → Detailed info for debugging ("Processing row 5000")
#   INFO     → General progress ("Model training started")
#   WARNING  → Something unexpected but not fatal ("Missing 3 rows")
#   ERROR    → Something failed ("Could not load file")
#   CRITICAL → System is unusable ("Out of memory")
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure structured logging for the entire application.
    
    Args:
        level: Minimum log level to display. "DEBUG" shows everything,
               "WARNING" shows only warnings and above.
    
    Returns:
        Logger instance configured with timestamp + level formatting.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("fraud_detection")


# ---------------------------------------------------------------------------
# 🎓 PATH CONFIGURATION
# ---------------------------------------------------------------------------
# We use pathlib.Path instead of string paths. Why?
#
# String:  "data" + "/" + "file.csv"          → breaks on Windows (uses \)
# Path:    Path("data") / "file.csv"          → works everywhere
#
# Path also gives you handy methods:
#   path.exists()    → does the file exist?
#   path.parent      → the folder containing this file
#   path.suffix      → the file extension (.csv, .pkl)
# ---------------------------------------------------------------------------

@dataclass
class PathConfig:
    """All file and directory paths used in the project."""
    
    # Project root — everything is relative to this
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    # Data paths
    raw_data_file: str = "transactions.csv"
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def raw_data_path(self) -> Path:
        return self.data_dir / self.raw_data_file
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"
    
    def ensure_dirs_exist(self) -> None:
        """Create output directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 🎓 TRAINING CONFIGURATION
# ---------------------------------------------------------------------------
# All the "knobs" you can turn when training models.
#
# WHAT IS random_state?
# Every ML algorithm has some randomness (shuffling data, picking features).
# Setting random_state=42 means: "use the SAME random sequence every time."
# This makes your results REPRODUCIBLE — you get the same numbers every run.
# 42 is just a convention (from Hitchhiker's Guide to the Galaxy 😄).
#
# WHAT IS test_size?
# If test_size=0.3, you use 70% of data for training and 30% for testing.
# Why 30%? It's a balance:
#   Too small (5%)  → not enough data to reliably evaluate
#   Too large (50%) → not enough data to train properly
#   30% is a safe default for large datasets (6.3M rows → 1.9M test rows)
#
# WHAT IS n_folds?
# Cross-validation splits your training data into N equal parts ("folds").
# It trains N separate models, each time using a different fold as validation.
# Then it averages the results. This gives you a much more reliable estimate
# of how your model will perform on new data.
#
# Example with 5 folds:
#   Fold 1: Train on [2,3,4,5], validate on [1]  → F1 = 96.2%
#   Fold 2: Train on [1,3,4,5], validate on [2]  → F1 = 97.1%
#   Fold 3: Train on [1,2,4,5], validate on [3]  → F1 = 95.8%
#   Fold 4: Train on [1,2,3,5], validate on [4]  → F1 = 96.5%
#   Fold 5: Train on [1,2,3,4], validate on [5]  → F1 = 96.9%
#   Average F1 = 96.5% ± 0.5%  ← much more trustworthy than a single number!
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Data splitting
    test_size: float = 0.3          # 30% for final testing
    n_folds: int = 5               # 5-fold cross-validation
    random_state: int = 42         # Reproducibility seed
    
    # Subsampling (for fast development — use 0.1 to iterate quickly)
    sample_fraction: float = 1.0   # 1.0 = use all data
    
    # ---------------------------------------------------------------------------
    # 🎓 MODEL HYPERPARAMETERS
    # ---------------------------------------------------------------------------
    # These are the "settings" for each model. Think of them like volume/bass/
    # treble knobs on a stereo — different settings produce different results.
    #
    # Random Forest:
    #   n_estimators: How many trees in the forest (more = better but slower)
    #   max_depth: How deep each tree can grow (deeper = more complex patterns
    #              but risk of overfitting — memorizing noise instead of patterns)
    #   class_weight: 'balanced_subsample' = automatically give more importance
    #                 to rare fraud cases. This REPLACES SMOTE.
    #
    # XGBoost:
    #   scale_pos_weight: Like class_weight but specific to XGBoost.
    #                     Set to (num_legitimate / num_fraud) ≈ 773
    #   early_stopping_rounds: Stop training if validation score doesn't improve
    #                          for N rounds. Saves time AND prevents overfitting.
    #   tree_method: 'hist' = fastest CPU method (uses histogram-based splits)
    #
    # Logistic Regression:
    #   max_iter: Maximum training iterations (1000 is usually enough)
    #   class_weight: Same concept as RF — balance the classes
    # ---------------------------------------------------------------------------
    
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,             # Use ALL CPU cores
        "random_state": 42,
    })
    
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1,
        "scale_pos_weight": 773,  # Will be recalculated from data
        "tree_method": "hist",    # Fast CPU method
        "eval_metric": "aucpr",   # Area Under PR Curve — best for imbalanced data
        "early_stopping_rounds": 15,
        "random_state": 42,
    })
    
    lr_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": 42,
    })


# ---------------------------------------------------------------------------
# 🎓 COST CONFIGURATION
# ---------------------------------------------------------------------------
# In fraud detection, different mistakes have VERY different costs:
#
# FALSE NEGATIVE (missed fraud):
#   A fraudster steals money and we didn't catch it.
#   Cost = average fraud amount = $1,470,832
#   This is CATASTROPHIC.
#
# FALSE POSITIVE (false alarm):
#   We flag a legitimate transaction as fraud. The customer gets a call
#   or their card is temporarily blocked.
#   Cost = investigation time + customer annoyance ≈ $500
#   This is annoying but manageable.
#
# RATIO: Missing one fraud costs 2,942× more than one false alarm.
# This means: we should STRONGLY prefer catching all fraud, even if it
# means a few extra false alarms.
# ---------------------------------------------------------------------------

@dataclass
class CostConfig:
    """Business cost assumptions for cost-sensitive evaluation."""
    
    false_negative_cost: float = 1_470_832.0  # Average fraud amount (from data)
    false_positive_cost: float = 500.0         # Investigation + customer friction
    monthly_operational_cost: float = 5_000.0  # Realistic for a startup
    
    @property
    def cost_ratio(self) -> float:
        """How many times worse is missing fraud vs. a false alarm?"""
        return self.false_negative_cost / self.false_positive_cost


# ---------------------------------------------------------------------------
# 🎓 FEATURE CONFIGURATION
# ---------------------------------------------------------------------------
# Lists of column names used throughout the pipeline.
# Having these in config means:
# 1. If a column name changes in the data, you update ONE place
# 2. Feature engineering, training, and inference all use the same lists
# 3. No typos from re-typing column names in 5 different files
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """Feature names and groups."""
    
    # Target variable — what we're trying to predict
    target_column: str = "isFraud"
    
    # Columns to drop (not useful for prediction)
    drop_columns: List[str] = field(default_factory=lambda: [
        "nameOrig",     # Account IDs — unique per row, not predictive
        "nameDest",     # Same reason
    ])
    
    # Categorical columns that need encoding
    categorical_columns: List[str] = field(default_factory=lambda: [
        "type",         # Transaction type: TRANSFER, CASH-OUT, etc.
    ])
    
    # Numerical columns from the raw data
    raw_numerical_columns: List[str] = field(default_factory=lambda: [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ])
    
    # Transaction types that are high-risk for fraud
    # (from domain knowledge: fraud mainly happens in TRANSFER and CASH_OUT)
    high_risk_types: List[str] = field(default_factory=lambda: [
        "TRANSFER", "CASH_OUT",
    ])


# ---------------------------------------------------------------------------
# 🎓 MASTER CONFIG
# ---------------------------------------------------------------------------
# One object to rule them all. Instead of importing 4 different configs,
# you import one `Config()` and access everything:
#
#   config = Config()
#   config.paths.raw_data_path    → Path to the CSV
#   config.training.n_folds       → Number of CV folds
#   config.costs.cost_ratio       → FN/FP cost ratio
#   config.features.target_column → "isFraud"
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Called automatically after __init__. Ensures directories exist."""
        self.paths.ensure_dirs_exist()


# ---------------------------------------------------------------------------
# Quick test: if you run this file directly, it prints the config
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = Config()
    logger = setup_logging(config.log_level)
    
    logger.info("Configuration loaded successfully")
    logger.info(f"Project root: {config.paths.project_root}")
    logger.info(f"Data file: {config.paths.raw_data_path}")
    logger.info(f"Test size: {config.training.test_size}")
    logger.info(f"CV folds: {config.training.n_folds}")
    logger.info(f"Cost ratio (FN/FP): {config.costs.cost_ratio:.0f}x")
    logger.info(f"Target column: {config.features.target_column}")
