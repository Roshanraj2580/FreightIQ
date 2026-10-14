"""
FreightIQ - Stage 5: Enterprise FastAPI Backend
Exposes endpoints for ML cost prediction, margin-optimized price recommendation,
BigQuery analytical trends, and autonomous AI copilot chat.
"""

import os
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.modeling.predict import FreightPredictor
from src.pricing.pricing_engine import calculate_recommended_price, PricingRequest, PricingResponse
from src.agent.freight_agent import FreightAgent

app = FastAPI(
    title="FreightIQ API",
    description="AI-Powered Freight Price Recommendation Engine & Logistics Intelligence Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global instances
predictor = None
agent = None

@app.on_event("startup")
def startup_event():
    global predictor, agent
    try:
        predictor = FreightPredictor()
    except Exception as e:
        print(f"[API Startup] Predictor initialization note: {e}")
    agent = FreightAgent()

class CostPredictionInput(BaseModel):
    weight_g: float = Field(..., gt=0, example=2500, description="Package deadweight in grams")
    origin_state: str = Field("SP", min_length=2, max_length=2, example="SP")
    dest_state: str = Field("RJ", min_length=2, max_length=2, example="RJ")
    length_cm: float = Field(25.0, gt=0, example=25.0)
    height_cm: float = Field(15.0, gt=0, example=15.0)
    width_cm: float = Field(20.0, gt=0, example=20.0)
    delivery_speed_days: int = Field(7, ge=1, le=45, example=7)
    purchase_month: int = Field(10, ge=1, le=12, example=10)

class ChatRequest(BaseModel):
    query: str = Field(..., example="I need to ship a 5kg machine from SP to BA in 3 days with a 25% margin. What should I quote?")

@app.get("/")
def root():
    return {
        "service": "FreightIQ API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": [
            "/predict-cost",
            "/recommend-price",
            "/analytics/summary",
            "/models/metrics",
            "/agent/chat"
        ]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}

@app.post("/predict-cost")
def predict_cost(payload: CostPredictionInput):
    """Predicts baseline carrier shipping cost using LightGBM model."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded.")
    try:
        res = predictor.predict_freight(
            weight_g=payload.weight_g,
            origin_state=payload.origin_state,
            dest_state=payload.dest_state,
            length_cm=payload.length_cm,
            height_cm=payload.height_cm,
            width_cm=payload.width_cm,
            delivery_speed_days=payload.delivery_speed_days,
            purchase_month=payload.purchase_month
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-price", response_model=PricingResponse)
def recommend_price(payload: PricingRequest):
    """Applies dynamic margin, risk buffers, and SLA multipliers to calculate target customer price."""
    try:
        res = calculate_recommended_price(
            predicted_cost=payload.predicted_cost,
            target_margin_pct=payload.target_margin_pct,
            delivery_speed_days=payload.delivery_speed_days,
            is_same_state=payload.is_same_state,
            destination_state=payload.destination_state,
            billable_weight_kg=payload.billable_weight_kg,
            is_oversized=payload.is_oversized,
            is_peak_season=payload.is_peak_season
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analytics/summary")
def get_analytics_summary():
    """Returns BigQuery/DuckDB analytical aggregates (regional, seasonal, top lanes)."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    p = os.path.join(base, "data", "processed", "analytics_summary.json")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Analytics summary not found. Run run_analytics.py first.")
    with open(p, "r") as f:
        return json.load(f)

@app.get("/models/metrics")
def get_model_metrics():
    """Returns comparative metrics between Linear Regression baseline and LightGBM."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    p = os.path.join(base, "models", "model_metrics.json")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Model metrics not found. Run train.py first.")
    with open(p, "r") as f:
        return json.load(f)

@app.post("/agent/chat")
def agent_chat(payload: ChatRequest):
    """Conversational endpoint invoking the FreightIQ AI Copilot with tool calling."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent is not initialized.")
    try:
        response_text = agent.process_query(payload.query)
        return {"query": payload.query, "response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
