# =============================================================================
# 🎓 WHAT IS __init__.py?
# =============================================================================
# This file tells Python: "Hey, this folder is a Python PACKAGE, not just a
# random folder with .py files in it."
#
# Without this file, `from src.config import Config` would fail because Python
# wouldn't recognize `src` as a package.
#
# It can be empty (like this one mostly is), but it's essential.
#
# 💡 ANALOGY: Think of it like a "OPEN" sign on a shop door. The shop (folder)
# exists either way, but Python only "walks in" if it sees this sign.
# =============================================================================

"""
Fraud Detection System — Source Package

Modules:
    config              - Centralized configuration
    data_preprocessing  - Data loading, cleaning, validation
    feature_engineering - Feature creation and transformation
    model_training      - Model training with cross-validation
    model_evaluation    - Metrics, threshold tuning, cost analysis
    business_insights   - Business impact and recommendations
    inference           - Production prediction engine
"""
