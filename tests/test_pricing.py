"""
FreightIQ - Unit Tests for Pricing Engine & Model Inference
"""

import pytest
from src.pricing.pricing_engine import calculate_recommended_price
from src.modeling.predict import FreightPredictor

def test_pricing_calculation_standard():
    predicted_cost = 50.0
    target_margin = 20.0
    
    result = calculate_recommended_price(
        predicted_cost=predicted_cost,
        target_margin_pct=target_margin,
        delivery_speed_days=7,
        is_same_state=True,
        destination_state="SP",
        billable_weight_kg=2.0
    )
    
    # Base price should be 50 / (1 - 0.20) = 62.50
    assert result["predicted_cost"] == 50.0
    assert result["recommended_price"] >= 62.50
    assert result["gross_profit"] > 0
    assert result["breakeven_price"] == 50.0
    assert result["effective_margin_pct"] >= 20.0

def test_pricing_express_surcharge():
    predicted_cost = 50.0
    
    standard = calculate_recommended_price(
        predicted_cost=predicted_cost,
        target_margin_pct=20.0,
        delivery_speed_days=7,
        destination_state="RJ"
    )
    
    express = calculate_recommended_price(
        predicted_cost=predicted_cost,
        target_margin_pct=20.0,
        delivery_speed_days=2,  # express SLA <= 3 days
        destination_state="RJ"
    )
    
    assert express["recommended_price"] > standard["recommended_price"]
    assert "express_delivery_premium" in express["adjustments"]

def test_pricing_remote_state_risk():
    predicted_cost = 40.0
    
    regular = calculate_recommended_price(
        predicted_cost=predicted_cost,
        destination_state="SP"
    )
    
    remote = calculate_recommended_price(
        predicted_cost=predicted_cost,
        destination_state="AM"  # Amazonas
    )
    
    assert remote["recommended_price"] > regular["recommended_price"]
    assert "remote_state_contingency" in remote["adjustments"]

def test_pricing_min_floor():
    predicted_cost = 100.0
    result = calculate_recommended_price(
        predicted_cost=predicted_cost,
        target_margin_pct=0.0  # 0% margin should still enforce 5% floor
    )
    assert result["recommended_price"] >= 105.0

def test_inference_predictor():
    predictor = FreightPredictor()
    pred = predictor.predict_freight(
        weight_g=1500,
        origin_state="SP",
        dest_state="MG",
        delivery_speed_days=5
    )
    assert pred["predicted_cost_brl"] > 0
    assert pred["distance_km"] > 0
    assert pred["billable_weight_kg"] >= 1.5
    assert pred["cost_lower_bound_brl"] <= pred["predicted_cost_brl"] <= pred["cost_upper_bound_brl"]
