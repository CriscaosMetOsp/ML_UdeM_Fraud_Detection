"""
FastAPI REST API for fraud detection predictions.
Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Credit Card Fraud Detection - MLOps Project | Universidad de Medellín",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler at startup
MODEL = None
SCALER = None
CONFIG = None


@app.on_event("startup")
def load_artifacts():
    global MODEL, SCALER, CONFIG
    with open("configs/config.yaml") as f:
        CONFIG = yaml.safe_load(f)
    MODEL = joblib.load(CONFIG["api"]["model_path"])
    SCALER = joblib.load("models/scaler.pkl")
    logger.info("Model and scaler loaded successfully")


# ── Input schema ──────────────────────────────────────────────────────────────


class TransactionInput(BaseModel):
    """Single transaction to score."""

    amt: float = Field(..., gt=0, description="Transaction amount in USD")
    lat: float = Field(..., description="Customer latitude")
    long: float = Field(..., description="Customer longitude")
    city_pop: int = Field(..., gt=0, description="Population of the customer city")
    merch_lat: float = Field(..., description="Merchant latitude")
    merch_long: float = Field(..., description="Merchant longitude")
    hour: int = Field(..., ge=0, le=23, description="Hour of the transaction (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    age: int = Field(..., gt=0, description="Cardholder age in years")
    category: int = Field(..., description="Encoded merchant category")
    state: int = Field(..., description="Encoded US state")

    class Config:
        json_schema_extra = {
            "example": {
                "amt": 107.23,
                "lat": 48.8878,
                "long": -118.2105,
                "city_pop": 149,
                "merch_lat": 49.159,
                "merch_long": -118.186,
                "hour": 0,
                "day_of_week": 1,
                "age": 45,
                "category": 3,
                "state": 7,
            }
        }


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    message: str


class BatchInput(BaseModel):
    transactions: list[TransactionInput]


class BatchResponse(BaseModel):
    results: list[PredictionResponse]
    total: int
    fraud_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────


def _haversine(lat1, lon1, lat2, lon2) -> float:
    import math

    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _to_df(tx: TransactionInput) -> pd.DataFrame:
    d = tx.model_dump()
    d["distance_km"] = _haversine(d["lat"], d["long"], d["merch_lat"], d["merch_long"])
    return pd.DataFrame([d])


def _risk_label(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"


def _predict_one(tx: TransactionInput) -> PredictionResponse:
    df = _to_df(tx)
    num_cols = [c for c in CONFIG["features"]["numerical_cols"] if c in df.columns]
    df[num_cols] = SCALER.transform(df[num_cols])

    prob = float(MODEL.predict_proba(df)[0, 1])
    is_fraud = prob >= 0.5
    risk = _risk_label(prob)

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        risk_level=risk,
        message=(
            "⚠️ Suspicious transaction"
            if is_fraud
            else "✅ Transaction appears legitimate"
        ),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Fraud Detection API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "scaler_loaded": SCALER is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: TransactionInput):
    """Score a single transaction for fraud."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return _predict_one(transaction)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(payload: BatchInput):
    """Score multiple transactions at once."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not payload.transactions:
        raise HTTPException(status_code=400, detail="Empty transaction list")
    try:
        results = [_predict_one(tx) for tx in payload.transactions]
        fraud_count = sum(1 for r in results if r.is_fraud)
        return BatchResponse(
            results=results,
            total=len(results),
            fraud_count=fraud_count,
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return information about the loaded model."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(MODEL).__name__,
        "features": CONFIG["features"]["numerical_cols"]
        + CONFIG["features"]["categorical_cols"],
        "target": CONFIG["features"]["target"],
    }
