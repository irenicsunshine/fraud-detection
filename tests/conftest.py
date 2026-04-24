# =============================================================================
# 🎓 WHAT IS conftest.py?
# =============================================================================
# This file is automatically loaded by pytest BEFORE running any tests.
# It contains "fixtures" — reusable setup code that tests can share.
#
# Think of fixtures as pre-cooked ingredients:
# - Instead of each test creating its own DataFrame
# - They all share the same one from here
# - This keeps tests DRY (Don't Repeat Yourself)
#
# 🎓 HOW FIXTURES WORK
# ----------------------
# When a test function has a parameter like:
#
#   def test_something(sample_dataframe):
#       ...
#
# pytest sees "sample_dataframe" and looks for a fixture with that name.
# It runs the fixture, passes the result to the test function, and
# cleans up afterward.
# =============================================================================

import pytest
import pandas as pd
import numpy as np
from src.config import Config


@pytest.fixture
def config():
    """Provide a Config instance for tests."""
    return Config()


@pytest.fixture
def sample_dataframe():
    """
    Create a small synthetic dataset that mimics real transaction data.
    
    🎓 WHY SYNTHETIC DATA FOR TESTS?
    -----------------------------------
    - Tests should be FAST (< 1 second each)
    - Tests should be SELF-CONTAINED (no external files needed)
    - Tests should be DETERMINISTIC (same result every time)
    
    Real data is 6.3M rows. This fixture has 10 rows. Same structure,
    different scale. If code works on 10 rows, it works on 6.3M rows
    (for most logic — edge cases need separate tests).
    """
    np.random.seed(42)
    
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "TRANSFER", "CASH_OUT",
                 "CASH_IN", "DEBIT", "TRANSFER", "CASH_OUT", "PAYMENT"],
        "amount": [100000, 50000, 200, 350000, 75000, 
                   1000, 500, 900000, 25000, 150],
        "nameOrig": ["C001", "C002", "C003", "C004", "C005",
                     "C001", "C002", "C006", "C003", "C004"],
        "oldbalanceOrg": [100000, 80000, 5000, 350000, 75000,
                          0, 5000, 900000, 4800, 350000],
        "newbalanceOrig": [0, 30000, 4800, 0, 0,
                           1000, 4500, 0, 0, 349850],
        "nameDest": ["C010", "C011", "M001", "C012", "C013",
                     "C014", "M002", "C015", "C016", "M003"],
        "oldbalanceDest": [0, 50000, 100000, 0, 200000,
                           30000, 80000, 100000, 0, 500000],
        "newbalanceDest": [100000, 100000, 100200, 350000, 275000,
                           31000, 80500, 1000000, 25000, 500150],
        "isFraud": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        "isFlaggedFraud": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "isMerchantOrig": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "isMerchantDest": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    })


@pytest.fixture
def sample_transaction():
    """A single suspicious transaction for inference testing."""
    return {
        "step": 1,
        "type": "TRANSFER",
        "amount": 500000,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 500000,
        "newbalanceOrig": 0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0,
        "newbalanceDest": 500000,
    }


@pytest.fixture
def normal_transaction():
    """A normal-looking transaction."""
    return {
        "step": 10,
        "type": "PAYMENT",
        "amount": 150,
        "nameOrig": "C111111111",
        "oldbalanceOrg": 5000,
        "newbalanceOrig": 4850,
        "nameDest": "M222222222",
        "oldbalanceDest": 100000,
        "newbalanceDest": 100150,
    }
