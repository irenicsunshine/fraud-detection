# =============================================================================
# 🎓 STEP 2.2: REST API — Making Your Model Available to the World
# =============================================================================
#
# WHAT IS AN API?
# ----------------
# An API (Application Programming Interface) lets OTHER applications talk to
# your model over the internet using HTTP requests.
#
# Without an API:
#   Someone wants to use your model → they need Python, your code, your model
#   file, all dependencies, and knowledge of your codebase.
#
# With an API:
#   Someone sends: POST /predict {"amount": 500000, "type": "TRANSFER", ...}
#   They get back:  {"prediction": "fraud", "probability": 0.94, "risk": "CRITICAL"}
#
# They don't need Python, they don't need your code. They just send HTTP
# requests — which every programming language can do.
#
# WHY FASTAPI?
# --------------
# FastAPI is a modern Python web framework. It's:
# 1. FAST — one of the fastest Python frameworks
# 2. AUTO-DOCUMENTED — generates interactive API docs automatically
# 3. TYPE-SAFE — uses Pydantic models for input validation
# 4. ASYNC — can handle many requests simultaneously
#
# After starting the server, visit http://localhost:8000/docs for
# interactive documentation where you can test every endpoint!
#
# 🎓 WHAT IS PYDANTIC?
# ----------------------
# Pydantic is like a bouncer at a club — it checks that incoming data
# meets your requirements BEFORE it reaches your code.
#
# Without Pydantic:
#   someone sends {"amount": "hello"} → crashes deep in your model
#
# With Pydantic:
#   someone sends {"amount": "hello"} → immediately returns:
#   "Error: amount must be a number" (model never sees the bad data)
# =============================================================================

import logging
import os
import time
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from src.config import Config, setup_logging
from src.inference import FraudPredictor

# ---------------------------------------------------------------------------
# 🎓 API KEY AUTHENTICATION
# ---------------------------------------------------------------------------
# We use an HTTP header called "X-API-Key". Callers must include it:
#
#   curl -H "X-API-Key: your-secret-key" http://localhost:8000/predict
#
# The key is read from the API_KEY environment variable so it never
# appears in source code. On Render.com, set it in the Environment tab.
# Locally, set it with: export API_KEY=your-secret-key
#
# If API_KEY is not set, auth is DISABLED (for easy local development).
# In production you MUST set it.

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> bool:
    """
    Verify the API key from the request header.

    Returns True if valid. Raises 403 if invalid.
    Skips check entirely if API_KEY env var is not set (dev mode).
    """
    expected_key = os.environ.get("API_KEY")

    if not expected_key:
        # No key configured → auth disabled (local dev mode)
        return True

    if api_key == expected_key:
        return True

    raise HTTPException(
        status_code=403,
        detail="Invalid or missing API key. Pass it as: X-API-Key: <your-key>",
    )

logger = logging.getLogger("fraud_detection.api")

# ---------------------------------------------------------------------------
# 🎓 PYDANTIC MODELS — Define what valid requests/responses look like
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    """
    A single financial transaction to evaluate for fraud.
    
    🎓 FIELD DEFINITIONS
    ----------------------
    Each field has:
    - A type (str, float, int) — Pydantic auto-converts if possible
    - A description — shows up in the API documentation
    - Optional constraints (ge=0 means "greater than or equal to 0")
    
    If someone sends a request missing "amount", Pydantic returns:
    {"detail": [{"msg": "field required", "loc": ["body", "amount"]}]}
    
    The 'example' in model_config shows up in the interactive docs,
    so users know what valid data looks like.
    """
    step: int = Field(..., ge=0, description="Time step (hourly intervals, 0-743)")
    type: str = Field(..., description="Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER")
    amount: float = Field(..., ge=0, description="Transaction amount in dollars")
    nameOrig: str = Field(..., description="Origin account ID")
    oldbalanceOrg: float = Field(..., ge=0, description="Origin account balance before transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Origin account balance after transaction")
    nameDest: str = Field(..., description="Destination account ID")
    oldbalanceDest: float = Field(..., ge=0, description="Destination balance before transaction")
    newbalanceDest: float = Field(..., ge=0, description="Destination balance after transaction")
    
    # 🎓 VALIDATOR — Custom validation logic
    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        valid_types = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
        if v not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got '{v}'")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response from the fraud prediction endpoint."""
    prediction: str = Field(..., description="'fraud' or 'legitimate'")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    risk_tier: str = Field(..., description="Risk level: CRITICAL, HIGH, MEDIUM, LOW")
    threshold_used: float = Field(..., description="Decision threshold used")
    model_used: str = Field(..., description="Name of the model used")
    inference_time_ms: float = Field(..., description="Prediction time in milliseconds")


class BatchRequest(BaseModel):
    """Batch of transactions to evaluate."""
    transactions: List[TransactionRequest] = Field(
        ..., description="List of transactions to evaluate",
        min_length=1, max_length=10000,
    )


class BatchResponse(BaseModel):
    """Response from batch prediction."""
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_detected: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    model_type: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    threshold: float
    n_features: int
    risk_tiers: dict


# ---------------------------------------------------------------------------
# 🎓 APPLICATION LIFECYCLE
# ---------------------------------------------------------------------------
# We use a "lifespan" to load the model ONCE when the server starts,
# not on every request. This is important because:
# 1. Loading a model takes ~100-500ms
# 2. API requests should respond in <100ms
# 3. The model is shared across all requests (read-only, thread-safe)

# Global variables for the app
predictor: Optional[FraudPredictor] = None
start_time: float = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    🎓 LIFESPAN EVENT
    ------------------
    This runs ONCE when the server starts. It loads the model into memory
    so all incoming requests can use it.
    
    The 'yield' separates startup code (above) from shutdown code (below).
    Startup: load model
    Shutdown: cleanup (nothing needed for our case)
    """
    global predictor, start_time
    
    config = Config()
    setup_logging(config.log_level)
    
    logger.info("Starting Fraud Detection API...")
    
    try:
        predictor = FraudPredictor(model_name="xgboost", config=config)
        start_time = time.time()
        logger.info("Model loaded successfully!")
    except FileNotFoundError:
        logger.error("Model file not found! Train a model first with: python main.py --mode train")
        logger.warning("API starting WITHOUT a model — predictions will fail")
        start_time = time.time()
    
    yield  # Server is running
    
    logger.info("Shutting down Fraud Detection API...")


# ---------------------------------------------------------------------------
# 🎓 CREATE THE FastAPI APP
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Production-ready API for detecting fraudulent financial transactions. "
        "Submit transaction data and receive real-time fraud predictions with "
        "risk tiering and probability scores."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# 🎓 CORS MIDDLEWARE
# CORS (Cross-Origin Resource Sharing) controls which websites can call your API.
# Without this, a frontend at example.com can't call your API at api.example.com.
# allow_origins=["*"] means "anyone can call" — fine for development, restrict in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 🎓 API ENDPOINTS
# ---------------------------------------------------------------------------
# Each endpoint is a function decorated with @app.get() or @app.post().
# GET = retrieve information (no side effects)
# POST = send data for processing (like scoring a transaction)

@app.get("/", tags=["System"])
async def root():
    """Welcome message and links to docs."""
    return {
        "name": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check if the API is running and the model is loaded.
    
    🎓 WHY A HEALTH ENDPOINT?
    ---------------------------
    Deployment platforms (Render, Railway, Docker) periodically call this
    endpoint to check if your service is alive. If it returns an error,
    they restart the container automatically.
    
    It's like a heartbeat monitor for your application.
    """
    model_loaded = predictor is not None
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_name=predictor.model_name if predictor else "none",
        model_type=type(predictor.model).__name__ if predictor else "none",
        uptime_seconds=round(time.time() - start_time, 2),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Get information about the loaded model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = predictor.get_model_info()
    return ModelInfoResponse(
        model_name=info["model_name"],
        model_type=info["model_type"],
        threshold=info["threshold"],
        n_features=info["n_features"],
        risk_tiers=info["risk_tiers"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(transaction: TransactionRequest, _: bool = Security(verify_api_key)):
    """
    Predict whether a single transaction is fraudulent.
    
    🎓 HOW IT WORKS:
    1. Pydantic validates the incoming JSON
    2. We convert it to a dict and pass to FraudPredictor
    3. Feature engineering + model prediction happens
    4. We return the result as JSON
    
    Try it in the interactive docs at /docs !
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first.",
        )
    
    try:
        # 🎓 .model_dump() converts the Pydantic model to a regular dict
        result = predictor.predict(transaction.model_dump())
        return PredictionResponse(**result)
    
    except ValueError as e:
        # Input validation error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected error
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Predictions"])
async def predict_batch(request: BatchRequest, _: bool = Security(verify_api_key)):
    """
    Predict fraud for a batch of transactions (up to 10,000).
    
    🎓 BATCH vs SINGLE PREDICTION
    --------------------------------
    Single: 1 request = 1 prediction. Simple but slow for bulk processing.
    Batch:  1 request = N predictions. Faster due to vectorized operations.
    
    Use batch when processing historical data or nightly sweeps.
    Use single for real-time transaction scoring.
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first.",
        )
    
    start = time.time()
    
    try:
        predictions = []
        for txn in request.transactions:
            result = predictor.predict(txn.model_dump())
            predictions.append(PredictionResponse(**result))
        
        elapsed_ms = (time.time() - start) * 1000
        n_fraud = sum(1 for p in predictions if p.prediction == "fraud")
        
        return BatchResponse(
            predictions=predictions,
            total_transactions=len(predictions),
            fraud_detected=n_fraud,
            processing_time_ms=round(elapsed_ms, 2),
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ---------------------------------------------------------------------------
# 🎓 RUN THE SERVER
# ---------------------------------------------------------------------------
# This block runs when you execute: python api.py
# In production, you'd use: uvicorn api:app --host 0.0.0.0 --port 8000
#
# uvicorn is a high-performance ASGI server that serves your FastAPI app.
# Think of FastAPI as the recipe and uvicorn as the kitchen that cooks it.

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting Fraud Detection API...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("❤️  Health Check:      http://localhost:8000/health")
    print("🔮 Predict:           POST http://localhost:8000/predict")
    print()
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",    # Listen on all interfaces
        port=8000,          # Port number
        reload=True,        # Auto-restart when code changes (dev mode)
    )
