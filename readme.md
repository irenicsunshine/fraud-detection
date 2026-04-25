# Fraud Detection — Production ML System

A production-ready machine learning system for detecting fraudulent financial transactions. Built on 6.3M synthetic transactions (PaySim dataset) with a FastAPI inference API and one-click deployment to Render.com.

---

## Results

| Model | F1-Score | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| **Random Forest** | **99.72%** | **99.76%** | **99.67%** | **99.98%** |
| XGBoost | 98.06% | 96.46% | 99.72% | 99.96% |
| Logistic Regression | 98.00% | 96.38% | 99.67% | 99.92% |

---

## Project Structure

```
fraud-detection/
├── api.py                  # FastAPI REST API (predict, batch, health)
├── main.py                 # CLI pipeline (train, evaluate, predict, serve)
├── download_data.py        # Kaggle dataset downloader
├── Dockerfile              # Container definition
├── render.yaml             # Render.com deployment config
├── requirements.txt        # Python dependencies
├── models/
│   ├── trained_models.pkl  # Pre-trained models (ready to use)
│   └── feature_names.pkl   # Feature schema
├── src/
│   ├── config.py           # Central configuration
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── business_insights.py
│   └── inference.py        # FraudPredictor class
├── data/                   # CSV goes here (gitignored)
└── tests/                  # pytest test suite
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API (pre-trained models included)

```bash
python api.py
```

Open `http://localhost:8000/docs` for interactive API documentation.

### 3. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 500000.0,
    "nameOrig": "C123456789",
    "oldbalanceOrg": 500000.0,
    "newbalanceOrig": 0.0,
    "nameDest": "C987654321",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 500000.0
  }'
```

Response:
```json
{
  "prediction": "fraud",
  "fraud_probability": 0.94,
  "risk_tier": "CRITICAL",
  "threshold_used": 0.5,
  "model_used": "random_forest",
  "inference_time_ms": 12.3
}
```

---

## Retrain the Model

### Step 1: Download the dataset

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
python download_data.py
```

### Step 2: Train

```bash
# Full training (~10-30 min on CPU)
python main.py --mode train

# Quick test with 10% of data (~2-3 min)
python main.py --mode train --sample-frac 0.1
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Check if API is running and model is loaded |
| GET | `/model/info` | Model name, threshold, feature count |
| POST | `/predict` | Score a single transaction |
| POST | `/predict/batch` | Score up to 10,000 transactions |

Full interactive docs at `/docs` when the server is running.

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Deploy to Render.com (Free)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Render detects `render.yaml` and auto-configures everything
5. Click **Deploy**

Every push to `main` triggers an automatic redeploy.

> **Note:** Render's free tier spins down after 15 minutes of inactivity. First request after idle has a ~30s cold start. Upgrade to the Starter plan ($7/mo) for always-on.

---

## Deploy with Docker

```bash
# Build the image
docker build -t fraud-detection .

# Run locally
docker run -p 8000:8000 fraud-detection
```

---

## Technical Approach

**Why class weights instead of SMOTE?**
SMOTE generates synthetic minority samples during training. The problem: synthetic samples can leak information about the test set distribution when used naively, inflating metrics. Class weights tell the model "mistakes on fraud cost more" without touching the data — cleaner and more honest.

**Why threshold optimization?**
The default 0.5 decision threshold isn't optimal for imbalanced data. We tune it to maximize F1 on the validation set, trading off precision vs. recall based on business cost assumptions.

**Feature engineering highlights:**
- Balance deltas (did the origin account drain to zero?)
- Error flags (did the reported balance change not match the transaction amount?)
- Transaction type one-hot encoding
- Amount-to-balance ratios

---

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a synthetic mobile money transaction simulator.
- 6,362,620 transactions over 30 days
- Fraud rate: ~0.13% (highly imbalanced)
- Only TRANSFER and CASH_OUT transactions can be fraudulent

---

## Roadmap

Things to add in future iterations:

- **Rate limiting** — cap requests per IP (e.g. 100/min) to prevent abuse
- **Request logging** — persist every prediction to a database or file for auditing and monitoring
- **Model versioning** — ability to hot-swap models via config without redeploying
- **Frontend** — a simple web form to test predictions without curl
