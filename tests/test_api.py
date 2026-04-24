# =============================================================================
# TESTS FOR THE FASTAPI ENDPOINTS — Including API Key Authentication
# =============================================================================
#
# WHY TestClient AND NOT A REAL SERVER?
# ----------------------------------------
# Starting a real server with `python api.py &` has two problems:
# 1. TIMING — the server takes ~500ms to load the XGBoost model. If the
#    test script fires requests immediately, it gets connection-refused or
#    an empty response, which causes JSON decode errors.
# 2. CLEANUP — killing background processes reliably across platforms is messy.
#
# FastAPI's TestClient runs the app in-process, in a fake HTTP environment.
# It calls startup (loads the model) before the first request, and tears down
# cleanly after the last one. No ports, no timing, no OS processes.
# =============================================================================

import os
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# We import `app` but NOT at module level — we need to control the API_KEY
# env var BEFORE the app object is imported and validated. Instead, we
# import inside fixtures so each test class gets a fresh client.
# ---------------------------------------------------------------------------

FRAUD_TRANSACTION = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 500000.0,
    "nameOrig": "C123456789",
    "oldbalanceOrg": 500000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C987654321",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 500000.0,
}


@pytest.fixture(scope="module")
def client():
    """
    A TestClient with NO API_KEY set — auth is disabled.
    This is the default for most endpoint tests.
    """
    # Ensure the env var is absent for this test session
    os.environ.pop("API_KEY", None)

    from api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def client_with_key():
    """
    A TestClient with API_KEY=testsecret set — auth is enforced.
    Used specifically for authentication tests.
    """
    os.environ["API_KEY"] = "testsecret"

    # Re-import is not needed because verify_api_key reads os.environ at
    # request time (not at import time), so the same app object will enforce
    # the key once the env var is present.
    from api import app
    with TestClient(app) as c:
        yield c

    # Clean up so other test modules aren't affected
    os.environ.pop("API_KEY", None)


# =============================================================================
# HEALTH & INFO ENDPOINTS (no auth required)
# =============================================================================

class TestHealthAndInfo:

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_model_loaded(self, client):
        data = r = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_model_info_returns_200(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200

    def test_model_info_fields(self, client):
        data = client.get("/model/info").json()
        assert "model_name" in data
        assert "threshold" in data
        assert "n_features" in data
        assert data["n_features"] > 0


# =============================================================================
# PREDICT ENDPOINT — functional tests (auth disabled)
# =============================================================================

class TestPredictEndpoint:

    def test_predict_valid_transaction(self, client):
        r = client.post("/predict", json=FRAUD_TRANSACTION)
        assert r.status_code == 200

    def test_predict_response_shape(self, client):
        data = client.post("/predict", json=FRAUD_TRANSACTION).json()
        assert "prediction" in data
        assert "fraud_probability" in data
        assert "risk_tier" in data
        assert "model_used" in data
        assert "inference_time_ms" in data

    def test_predict_probability_in_range(self, client):
        data = client.post("/predict", json=FRAUD_TRANSACTION).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_prediction_valid_label(self, client):
        data = client.post("/predict", json=FRAUD_TRANSACTION).json()
        assert data["prediction"] in {"fraud", "legitimate"}

    def test_predict_missing_field_returns_422(self, client):
        bad = {k: v for k, v in FRAUD_TRANSACTION.items() if k != "amount"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_invalid_type_returns_422(self, client):
        bad = {**FRAUD_TRANSACTION, "type": "INVALID_TYPE"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_negative_amount_returns_422(self, client):
        bad = {**FRAUD_TRANSACTION, "amount": -100.0}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


# =============================================================================
# BATCH ENDPOINT — functional test (auth disabled)
# =============================================================================

class TestBatchEndpoint:

    def test_batch_single_transaction(self, client):
        r = client.post("/predict/batch", json={"transactions": [FRAUD_TRANSACTION]})
        assert r.status_code == 200

    def test_batch_response_shape(self, client):
        data = client.post(
            "/predict/batch", json={"transactions": [FRAUD_TRANSACTION]}
        ).json()
        assert data["total_transactions"] == 1
        assert "fraud_detected" in data
        assert "processing_time_ms" in data
        assert len(data["predictions"]) == 1

    def test_batch_empty_list_returns_422(self, client):
        r = client.post("/predict/batch", json={"transactions": []})
        assert r.status_code == 422


# =============================================================================
# API KEY AUTHENTICATION — the main focus of this test module
# =============================================================================

class TestApiKeyAuthentication:
    """
    Tests the three cases from the description:
      1. No key sent  → 403
      2. Wrong key    → 403
      3. Correct key  → 200
    """

    def test_no_key_returns_403(self, client_with_key):
        """Request with no X-API-Key header is rejected."""
        r = client_with_key.post("/predict", json=FRAUD_TRANSACTION)
        assert r.status_code == 403

    def test_wrong_key_returns_403(self, client_with_key):
        """Request with an incorrect key is rejected."""
        r = client_with_key.post(
            "/predict",
            json=FRAUD_TRANSACTION,
            headers={"X-API-Key": "wrongkey"},
        )
        assert r.status_code == 403

    def test_correct_key_returns_200(self, client_with_key):
        """Request with the correct key succeeds."""
        r = client_with_key.post(
            "/predict",
            json=FRAUD_TRANSACTION,
            headers={"X-API-Key": "testsecret"},
        )
        assert r.status_code == 200

    def test_error_message_is_helpful(self, client_with_key):
        """403 response includes a useful error message."""
        data = client_with_key.post("/predict", json=FRAUD_TRANSACTION).json()
        assert "detail" in data
        assert "API key" in data["detail"]

    def test_health_endpoint_needs_no_key(self, client_with_key):
        """Health check is unprotected — monitoring tools don't need a key."""
        r = client_with_key.get("/health")
        assert r.status_code == 200

    def test_auth_disabled_when_no_env_var(self, client, monkeypatch):
        """When API_KEY env var is absent, any request goes through."""
        # Both module-scoped fixtures may coexist. Explicitly unset the key
        # so verify_api_key sees no configured secret at request time.
        monkeypatch.delenv("API_KEY", raising=False)
        r = client.post("/predict", json=FRAUD_TRANSACTION)
        assert r.status_code == 200
